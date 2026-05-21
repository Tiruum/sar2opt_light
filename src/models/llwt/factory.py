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

from src.models.llwt.dis import LLWFormerDiscriminator
from src.models.llwt.gen import LLWFormerGenerator
from src.models.llwt.losses import (
    AdaptiveLoss, FeatureMatchingLoss, GANLoss, LABChromaL1Loss, LPIPSLoss,
    MSSSIMLoss, PerfectReconLoss, PlainL1Loss, SpeckleDecoupleLoss,
    WaveletDetailL1Loss,
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
    criterions = {
        'gan': GANLoss(),
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
    if loss_cfg.get('lpips_weight', 0.0) > 0:
        criterions['lpips'] = LPIPSLoss(net_type='alex')
    if loss_cfg.get('spkdec_weight', 0.0) > 0:
        criterions['spkdec'] = SpeckleDecoupleLoss(
            target_hh_energy=float(loss_cfg.get('spkdec_target_hh', 0.05)),
        )
    if loss_cfg.get('pr_weight', 0.0) > 0:
        criterions['pr'] = PerfectReconLoss()
    if loss_cfg.get('adaptive_balance', False):
        recon_keys = [
            k for k in ('l1', 'msssim', 'lab_chroma', 'wavelet')
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
    if criterions is not None and 'adaptive' in criterions:
        g_params.extend(criterions['adaptive'].parameters())

    opt_g = optim.AdamW(
        g_params,
        lr=float(cfg.optimizer.lr_g),
        betas=(float(cfg.optimizer.beta1), float(cfg.optimizer.beta2)),
        weight_decay=float(cfg.optimizer.weight_decay_g),
        fused=True,                       # CUDA-only fused kernel; bit-identical math; ~1-3% step time win
    )

    expected_ids = {id(p) for p in netG.parameters()}
    if criterions is not None and 'adaptive' in criterions:
        expected_ids.update(id(p) for p in criterions['adaptive'].parameters())
    opt_param_ids = {id(p) for pg in opt_g.param_groups for p in pg['params']}
    missing = expected_ids - opt_param_ids
    assert not missing, (
        f"build_optimizers: {len(missing)} G/criterion parameter(s) not in any "
        f"opt_g group"
    )

    opt_d = optim.Adam(
        netD.parameters(),
        lr=float(cfg.optimizer.lr_d),
        betas=(float(cfg.optimizer.beta1), float(cfg.optimizer.beta2)),
        fused=True,                       # CUDA-only fused kernel; bit-identical math
    )

    return opt_d, opt_g


def build_lr_schedulers(cfg, opt_d, opt_g):
    """Same scheduler menu as ``sarformer_wb.factory``."""
    sched_type = str(cfg.scheduler.get('type', 'cosine_warm_restarts')).lower()
    eta_min = float(cfg.scheduler.eta_min)

    def make_sched(opt, base_lr):
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
            "(expected 'cosine_warm_restarts' or 'linear_decay')"
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
