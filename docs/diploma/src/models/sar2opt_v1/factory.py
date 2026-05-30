"""Factory for SAR2OPT-V1: models, criterions, optimizers, schedulers, EMA.

Conventions copied from ``llwt/factory.py``:
  * Criterion is instantiated only when its weight > 0 (saves VRAM + forward
    time during ablations).
  * Optimizers return ``(opt_d, opt_g)`` -- D first to match the Lightning
    module's consumer order.
  * EMA via the project's custom ``EMAWeightAveraging`` callback with epoch-
    gated start.
"""
from __future__ import annotations

from typing import Optional

import torch.optim as optim
from torch.optim.lr_scheduler import (
    ConstantLR, CosineAnnealingWarmRestarts, LinearLR, SequentialLR,
)

# Optional bnb 8-bit optimisers -- ~700 MB VRAM saved on G+D combo.
try:
    import bitsandbytes.optim as bnb_optim
    _HAS_BNB = True
except Exception:
    bnb_optim = None
    _HAS_BNB = False

from src.models.sar2opt_v1.dis import SAR2OPTDiscriminator
from src.models.sar2opt_v1.gen import SAR2OPTGenerator
from src.models.sar2opt_v1.losses import (
    FFLLoss,
    FeatureMatchingLoss,
    GANLoss,
    HingeGANLoss,
    L1Loss,
    LABChromaL1Loss,
    MSSSIMLoss,
    MultiLayerPatchNCE,
    WaveletDetailL1Loss,
)


__all__ = [
    "build_models",
    "build_criterions",
    "build_optimizers",
    "build_lr_schedulers",
    "build_ema_callback",
]


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def build_models(cfg):
    """Returns ``(netG, netD)``."""
    netG = SAR2OPTGenerator(cfg=cfg)
    netD = SAR2OPTDiscriminator(cfg=cfg)
    return netG, netD


# ---------------------------------------------------------------------------
# Criterions
# ---------------------------------------------------------------------------


def build_criterions(cfg, netG: Optional[SAR2OPTGenerator] = None) -> dict:
    """Instantiate active losses based on weights in ``cfg.loss``.

    ``netG`` is required if ``patchnce_weight > 0`` because the PatchNCE
    sampler needs the encoder's feature dimensions to size its MLPs.
    """
    loss_cfg = cfg.loss
    gan_type = str(getattr(loss_cfg, 'gan_type', 'lsgan')).lower()
    if gan_type == 'lsgan':
        gan_loss = GANLoss(
            real_smooth=float(getattr(loss_cfg, 'real_label_smooth', 0.9)),
            fake_smooth=float(getattr(loss_cfg, 'fake_label_smooth', 0.0)),
        )
    elif gan_type == 'hinge':
        gan_loss = HingeGANLoss()
    else:
        raise ValueError(f"build_criterions: unknown gan_type='{gan_type}' (expected 'lsgan' or 'hinge')")

    criterions = {
        'gan': gan_loss,
        'fm':  FeatureMatchingLoss(),
    }
    if float(getattr(loss_cfg, 'l1_weight', 0.0)) > 0:
        criterions['l1'] = L1Loss()
    if float(getattr(loss_cfg, 'msssim_weight', 0.0)) > 0:
        criterions['msssim'] = MSSSIMLoss()
    if float(getattr(loss_cfg, 'lab_chroma_weight', 0.0)) > 0:
        criterions['lab_chroma'] = LABChromaL1Loss()
    if float(getattr(loss_cfg, 'wavelet_weight', 0.0)) > 0:
        criterions['wavelet'] = WaveletDetailL1Loss()
    if float(getattr(loss_cfg, 'ffl_weight', 0.0)) > 0:
        criterions['ffl'] = FFLLoss(
            alpha=float(getattr(loss_cfg, 'ffl_alpha', 1.0)),
        )
    if float(getattr(loss_cfg, 'patchnce_weight', 0.0)) > 0:
        if netG is None:
            raise ValueError(
                "build_criterions: patchnce_weight>0 requires netG (need encoder_dims for MLP)."
            )
        layer_ids = tuple(int(i) for i in getattr(loss_cfg, 'patchnce_layers', (0, 1, 2, 3)))
        encoder_dims = netG.encoder_dims
        feature_dims = [encoder_dims[i] for i in layer_ids]
        criterions['patchnce'] = MultiLayerPatchNCE(
            feature_dims=feature_dims,
            layer_ids=layer_ids,
            nc=int(getattr(loss_cfg, 'patchnce_mlp_nc', 256)),
            num_patches=int(getattr(loss_cfg, 'patchnce_n_samples', 256)),
            temperature=float(getattr(loss_cfg, 'patchnce_temperature', 0.07)),
            batch_size=int(cfg.data.batch_size),
        )
    return criterions


# ---------------------------------------------------------------------------
# Optimizers
# ---------------------------------------------------------------------------


def build_optimizers(cfg, netG, netD, criterions: Optional[dict] = None):
    """AdamW for G (incl. PatchNCE MLP params); Adam for D.

    bitsandbytes 8-bit variants used when ``cfg.optimizer.use_bnb_8bit`` is
    True and the bnb wheel imports cleanly (falls back to torch fused otherwise).

    Returns ``(opt_d, opt_g)``.
    """
    g_params = list(netG.parameters())
    if criterions is not None and 'patchnce' in criterions:
        # PatchSampleF MLPs are part of the G optimisation graph (gradient flows
        # back through them to the encoder features that feed PatchNCE).
        g_params.extend(criterions['patchnce'].sampler.parameters())

    want_bnb = bool(getattr(cfg.optimizer, 'use_bnb_8bit', False))
    use_bnb = want_bnb and _HAS_BNB
    if want_bnb and not _HAS_BNB:
        print('[sar2opt_v1.factory] use_bnb_8bit=True but bitsandbytes import failed; falling back to torch fused fp32 Adam.')

    betas = (float(cfg.optimizer.beta1), float(cfg.optimizer.beta2))
    lr_g = float(cfg.optimizer.lr_g)
    lr_d = float(cfg.optimizer.lr_d)
    wd_g = float(cfg.optimizer.weight_decay_g)

    if use_bnb:
        opt_g = bnb_optim.AdamW8bit(g_params, lr=lr_g, betas=betas, weight_decay=wd_g)
        opt_d = bnb_optim.Adam8bit(netD.parameters(), lr=lr_d, betas=betas)
        print('[sar2opt_v1.factory] using bitsandbytes 8-bit Adam(W) for G + D')
    else:
        opt_g = optim.AdamW(g_params, lr=lr_g, betas=betas, weight_decay=wd_g, fused=True)
        opt_d = optim.Adam(netD.parameters(), lr=lr_d, betas=betas, fused=True)

    # Sanity: every parameter we intended to optimise is actually in opt_g.
    expected_ids = {id(p) for p in netG.parameters()}
    if criterions is not None and 'patchnce' in criterions:
        expected_ids.update(id(p) for p in criterions['patchnce'].sampler.parameters())
    opt_param_ids = {id(p) for pg in opt_g.param_groups for p in pg['params']}
    missing = expected_ids - opt_param_ids
    assert not missing, (
        f"build_optimizers: {len(missing)} G/criterion parameter(s) not in opt_g"
    )

    return opt_d, opt_g


# ---------------------------------------------------------------------------
# Schedulers
# ---------------------------------------------------------------------------


def build_lr_schedulers(cfg, opt_d, opt_g):
    """Cosine warm restarts (default) or linear decay."""
    sched_type = str(getattr(cfg.scheduler, 'type', 'cosine_warm_restarts')).lower()
    eta_min = float(cfg.scheduler.eta_min)

    def make_sched(opt, base_lr):
        if sched_type == 'cosine_warm_restarts':
            return CosineAnnealingWarmRestarts(
                opt,
                T_0=int(getattr(cfg.scheduler, 't_0', 20)),
                T_mult=int(getattr(cfg.scheduler, 't_mult', 2)),
                eta_min=eta_min,
            )
        if sched_type == 'linear_decay':
            decay = int(cfg.scheduler.linear_decay_epochs)
            max_epochs = int(cfg.system.max_epochs)
            warmup = max(max_epochs - decay, 0)
            end_factor = eta_min / max(base_lr, 1e-10)
            linear = LinearLR(opt, start_factor=1.0, end_factor=end_factor, total_iters=decay)
            if warmup == 0:
                return linear
            return SequentialLR(
                opt,
                schedulers=[ConstantLR(opt, factor=1.0, total_iters=warmup), linear],
                milestones=[warmup],
            )
        raise ValueError(f"build_lr_schedulers: unknown scheduler.type='{sched_type}'")

    sched_g = make_sched(opt_g, float(cfg.optimizer.lr_g))
    sched_d = make_sched(opt_d, float(cfg.optimizer.lr_d))
    return sched_d, sched_g


# ---------------------------------------------------------------------------
# EMA callback
# ---------------------------------------------------------------------------


def build_ema_callback(cfg):
    ema_cfg = getattr(cfg, 'ema', None)
    if ema_cfg is None or not bool(getattr(ema_cfg, 'use_ema', False)):
        return None
    from src.utils.callbacks import EMAWeightAveraging
    return EMAWeightAveraging(
        decay=float(ema_cfg.decay),
        update_starting_at_epoch=int(ema_cfg.start_epoch),
    )
