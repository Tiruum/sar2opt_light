"""Factory for LLW-Former: model + criterions + optimisers + schedulers + EMA.

Mirrors ``sarformer_wb/factory.py`` but adapted for the LLW-Former generator
(no pretrained backbone -> no encoder param group), the dual-head
discriminator (main + subband), and the two new regularisers
(``SpeckleDecoupleLoss``, ``PerfectReconLoss``).

Returns the same ``(opt_d, opt_g)`` order so the Lightning module's
``configure_optimizers`` contract is unchanged.
"""
from __future__ import annotations

import torch.optim as optim
from torch.optim.lr_scheduler import (
    ConstantLR, CosineAnnealingWarmRestarts, LinearLR, SequentialLR,
)

# Optional bitsandbytes 8-bit optimisers.  Halves Adam state VRAM (~700 MB
# saved on the LLW-Former G+D combo), bit-equivalent learning trajectory on
# GAN workloads per Dettmers 2022.  Falls back to torch fused on ImportError
# so a missing bnb wheel never blocks training.
try:
    import bitsandbytes.optim as bnb_optim
    _HAS_BNB = True
except Exception:  # bitsandbytes import can ImportError or CUDA-init-error
    bnb_optim = None
    _HAS_BNB = False

from src.models.llwt.dis import LLWFormerDiscriminator
from src.models.llwt.gen import LLWFormerGenerator
from src.models.llwt.losses import (
    AdaptiveLoss, FeatureMatchingLoss, FoundationPerceptualLoss, GANLoss,
    LABChromaL1Loss, LPIPSLoss, MSSSIMLoss, PerBandWaveletL1Loss,
    PerfectReconLoss, PlainL1Loss, SpeckleDecoupleLoss, WaveletDetailL1Loss,
)


__all__ = [
    "build_models",
    "build_criterions",
    "build_optimizers",
    "build_lr_schedulers",
    "build_ema_callback",
]


def build_models(cfg):
    """Returns ``(netG, netD)``."""
    netG = LLWFormerGenerator(cfg=cfg)
    netD = LLWFormerDiscriminator(cfg=cfg)
    return netG, netD


def build_criterions(cfg) -> dict:
    """Instantiate active losses based on weights in ``cfg.loss``.

    A criterion is created only when its weight is > 0 — saves both VRAM and
    forward time when running ablations.  The subband-D adversarial path uses
    the same ``GANLoss`` and ``FeatureMatchingLoss`` instances as the main D.
    """
    loss_cfg = cfg.loss
    # Mutual exclusion: per-band L1 on the predicted pre-IHaar subbands
    # (PerBandWaveletL1Loss) is mathematically near-identical to detail-band
    # L1 on the post-IHaar full-res output (WaveletDetailL1Loss) modulo the
    # tanh nonlinearity at the very end.  Allowing both would double-count
    # the detail-band gradient — fail loud at factory time rather than burn
    # GPU hours discovering it via training metrics.
    if loss_cfg.get('per_band_weight', 0.0) > 0 and loss_cfg.get('wavelet_weight', 0.0) > 0:
        raise ValueError(
            "per_band_weight and wavelet_weight both > 0 — these overlap on the "
            "Haar detail bands (PerBand on pre-IHaar sub ~= Wavelet on post-IHaar "
            "fake modulo tanh nonlinearity). Pick one."
        )
    criterions = {
        'gan': GANLoss(
            real_smooth=float(loss_cfg.get('real_smooth', 0.9)),
            fake_smooth=float(loss_cfg.get('fake_smooth', 0.0)),
            gan_type=str(loss_cfg.get('gan_type', 'lsgan')),
        ),
        'fm':  FeatureMatchingLoss(),
    }
    if loss_cfg.get('l1_weight', 0.0) > 0:
        criterions['l1'] = PlainL1Loss()
    if loss_cfg.get('msssim_weight', 0.0) > 0:
        criterions['msssim'] = MSSSIMLoss()
    if loss_cfg.get('lab_chroma_weight', 0.0) > 0:
        criterions['lab_chroma'] = LABChromaL1Loss()
    if loss_cfg.get('wavelet_weight', 0.0) > 0:
        criterions['wavelet'] = WaveletDetailL1Loss()
    if loss_cfg.get('per_band_weight', 0.0) > 0:
        # v0.4.1 R4: switchable Kendall uncertainty weighting on the per-band
        # L1 components.  When ``per_band_adaptive=True``, the four manual
        # per_band_{ll,lh,hl,hh} ratios are ignored — band weights are learned
        # via a (clamped) log_var Parameter that gets added to opt_g below.
        if bool(loss_cfg.get('per_band_adaptive', False)):
            criterions['per_band'] = PerBandWaveletL1Loss(
                adaptive=True,
                log_var_init=float(loss_cfg.get('per_band_log_var_init', 0.0)),
            )
        else:
            criterions['per_band'] = PerBandWaveletL1Loss(
                band_weights={
                    'll': float(loss_cfg.get('per_band_ll', 1.0)),
                    'lh': float(loss_cfg.get('per_band_lh', 1.0)),
                    'hl': float(loss_cfg.get('per_band_hl', 1.0)),
                    'hh': float(loss_cfg.get('per_band_hh', 1.0)),
                },
            )
    if loss_cfg.get('lpips_weight', 0.0) > 0:
        criterions['lpips'] = LPIPSLoss(net_type='alex')
    if loss_cfg.get('pepl_weight', 0.0) > 0:
        # v0.5.x: Foundation-model perceptual loss.  Frozen backbone, gradient
        # flows only through input activations to G (and through the 1x1
        # channel adapter when present).  Tiny extra opt_g param footprint.
        criterions['pepl'] = FoundationPerceptualLoss(
            backbone=str(loss_cfg.get('pepl_backbone', 'dinov2')),
            distance=str(loss_cfg.get('pepl_distance', 'l1')),
            channel_adapter=str(loss_cfg.get('pepl_channel_adapter', 'learnable')),
            layer_idxs=list(loss_cfg.get('pepl_layer_idxs', [-1])),
        )
    if loss_cfg.get('spkdec_weight', 0.0) > 0:
        criterions['spkdec'] = SpeckleDecoupleLoss(
            target_hh_energy=float(loss_cfg.get('spkdec_target_hh', 0.05)),
        )
    if loss_cfg.get('pr_weight', 0.0) > 0:
        criterions['pr'] = PerfectReconLoss()
    if loss_cfg.get('adaptive_balance', False):
        recon_keys = [
            k for k in ('l1', 'msssim', 'lab_chroma', 'wavelet', 'per_band')
            if k in criterions
        ]
        if len(recon_keys) >= 2:
            criterions['adaptive'] = AdaptiveLoss(
                n_losses=len(recon_keys),
                eta_max=float(loss_cfg.get('adaptive_eta_max', 5.0)),
            )
            criterions['adaptive']._loss_order = recon_keys
    return criterions


def build_optimizers(cfg, netG, netD, criterions: dict = None):
    """AdamW for G (single group — no pretrained encoder); Adam for D.

    Unlike ``sarformer_wb`` there is no separate ``encoder_lr_scale`` group
    because LLW-Former has no pretrained backbone — every G parameter is
    learned from scratch and shares the same LR.

    Returns ``(opt_d, opt_g)`` — D first to match the Lightning consumer.
    """
    g_params = list(netG.parameters())
    if criterions is not None:
        if 'adaptive' in criterions:
            g_params.extend(criterions['adaptive'].parameters())
        # v0.4.1 R4: per_band in adaptive mode owns a log_var Parameter that
        # must be optimised by opt_g.  Manual mode has no parameters so this
        # is a no-op.  Generic over both branches.
        if 'per_band' in criterions:
            g_params.extend(criterions['per_band'].parameters())
        # v0.5.0 PEPL: only the channel adapter is trainable; backbone is
        # frozen at construction.  ``trainable_parameters()`` skips the
        # frozen backbone explicitly.
        if 'pepl' in criterions:
            g_params.extend(criterions['pepl'].trainable_parameters())

    want_bnb = bool(getattr(cfg.optimizer, 'use_bnb_8bit', False))
    use_bnb = want_bnb and _HAS_BNB
    if want_bnb and not _HAS_BNB:
        print('[factory] use_bnb_8bit=True but bitsandbytes import failed; falling back to torch fused fp32 Adam.')

    betas = (float(cfg.optimizer.beta1), float(cfg.optimizer.beta2))
    lr_g = float(cfg.optimizer.lr_g)
    lr_d = float(cfg.optimizer.lr_d)
    wd_g = float(cfg.optimizer.weight_decay_g)

    if use_bnb:
        opt_g = bnb_optim.AdamW8bit(g_params, lr=lr_g, betas=betas, weight_decay=wd_g)
        opt_d = bnb_optim.Adam8bit(netD.parameters(), lr=lr_d, betas=betas)
        print('[factory] using bitsandbytes 8-bit Adam(W) for G + D')
    else:
        opt_g = optim.AdamW(g_params, lr=lr_g, betas=betas, weight_decay=wd_g, fused=True)
        opt_d = optim.Adam(netD.parameters(), lr=lr_d, betas=betas, fused=True)

    expected_ids = {id(p) for p in netG.parameters()}
    if criterions is not None:
        if 'adaptive' in criterions:
            expected_ids.update(id(p) for p in criterions['adaptive'].parameters())
        if 'per_band' in criterions:
            expected_ids.update(id(p) for p in criterions['per_band'].parameters())
        if 'pepl' in criterions:
            expected_ids.update(id(p) for p in criterions['pepl'].trainable_parameters())
    opt_param_ids = {id(p) for pg in opt_g.param_groups for p in pg['params']}
    missing = expected_ids - opt_param_ids
    assert not missing, (
        f"build_optimizers: {len(missing)} G/criterion parameter(s) not in any "
        f"opt_g group"
    )

    return opt_d, opt_g


def build_lr_schedulers(cfg, opt_d, opt_g):
    """Same scheduler menu as ``sarformer_wb.factory``."""
    sched_type = str(cfg.scheduler.get('type', 'cosine_warm_restarts')).lower()
    eta_min = float(cfg.scheduler.eta_min)

    def make_sched(opt, base_lr):
        if sched_type == 'constant':
            return ConstantLR(opt, factor=1.0,
                              total_iters=int(cfg.system.max_epochs))
        if sched_type == 'cosine_warm_restarts':
            return CosineAnnealingWarmRestarts(
                opt,
                T_0=int(cfg.scheduler.get('t_0', 20)),
                T_mult=int(cfg.scheduler.get('t_mult', 2)),
                eta_min=eta_min,
            )
        if sched_type == 'linear_decay':
            decay = int(cfg.scheduler.linear_decay_epochs)
            max_epochs = int(cfg.system.max_epochs)
            warmup = max(max_epochs - decay, 0)
            end_factor = eta_min / max(base_lr, 1e-10)
            linear = LinearLR(opt, start_factor=1.0, end_factor=end_factor,
                              total_iters=decay)
            if warmup == 0:
                return linear
            return SequentialLR(
                opt,
                schedulers=[ConstantLR(opt, factor=1.0, total_iters=warmup), linear],
                milestones=[warmup],
            )
        raise ValueError(
            f"Unknown scheduler.type='{sched_type}' "
            "(expected 'constant', 'cosine_warm_restarts', or 'linear_decay')"
        )

    sched_g = make_sched(opt_g, float(cfg.optimizer.lr_g))
    sched_d = make_sched(opt_d, float(cfg.optimizer.lr_d))
    return sched_d, sched_g


def build_ema_callback(cfg):
    """Returns an ``EMAWeightAveraging`` callback when ``cfg.ema.use_ema``."""
    ema_cfg = getattr(cfg, 'ema', None)
    if ema_cfg is None or not getattr(ema_cfg, 'use_ema', False):
        return None
    from src.utils.callbacks import EMAWeightAveraging
    return EMAWeightAveraging(
        decay=float(ema_cfg.decay),
        update_starting_at_epoch=int(ema_cfg.start_epoch),
    )
