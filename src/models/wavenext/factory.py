"""Factory for WaveNeXt v0.4.5 (self-contained).

Vendored copy of the llwt/llwt_v4 factory so wavenext owns its model, loss,
optimiser, and scheduler builders independently — editing them here never
touches other models.

  * ``build_models`` -> ``(WaveNeXtGenerator, WaveNeXtDiscriminator)`` from the
    local ``gen``/``dis`` modules.
  * ``build_criterions`` instantiates a loss only when its cfg weight > 0.
    The dead spkdec/pr paths (no learnable lifting in v0.4.x) are dropped.
  * ``build_optimizers`` — AdamW(G, single group) + Adam(D); bnb 8-bit with a
    torch-fused fallback. Extends opt_g with adaptive / per_band / pepl params.
  * ``build_lr_schedulers`` — constant / cosine_warm_restarts / linear_decay.

Returns ``(opt_d, opt_g)`` order — unchanged Lightning consumer contract.
"""
from __future__ import annotations

import torch.optim as optim
from torch.optim.lr_scheduler import (
    ConstantLR, CosineAnnealingWarmRestarts, LinearLR, SequentialLR,
)

try:
    import bitsandbytes.optim as bnb_optim
    _HAS_BNB = True
except Exception:
    bnb_optim = None
    _HAS_BNB = False

from src.models.wavenext.dis import WaveNeXtDiscriminator
from src.models.wavenext.gen import WaveNeXtGenerator
from src.models.wavenext.losses import (
    AdaptiveLoss, FeatureMatchingLoss, FFLLoss, FoundationPerceptualLoss,
    GANLoss, LABChromaL1Loss, LPIPSLoss, MSSSIMLoss, MultiLayerPatchNCE,
    PerBandWaveletL1Loss, PlainL1Loss, WaveletDetailL1Loss,
    WaveletDetailSWDLoss,
)


__all__ = [
    "build_models",
    "build_aligner",
    "build_criterions",
    "build_optimizers",
    "build_lr_schedulers",
    "build_ema_callback",
]


def build_models(cfg):
    """Returns ``(netG, netD)`` — v0.4.5 generator + LLW discriminator."""
    netG = WaveNeXtGenerator(cfg=cfg)
    netD = WaveNeXtDiscriminator(cfg=cfg)
    return netG, netD


def build_aligner(cfg):
    """Return (aligner, align_criterions_dict) or (None, {}) when align disabled."""
    align_cfg = getattr(cfg, 'align', None)
    if align_cfg is None or not bool(getattr(align_cfg, 'enabled', False)):
        return None, {}
    from src.models.wavenext.align import DeformationAligner
    from src.models.wavenext.losses import (
        DeformationRegLoss, PSCAnchorLoss, BackscatterStructureLoss,
    )
    aligner = DeformationAligner(
        ll_channels=3,
        max_disp_px=float(getattr(align_cfg, 'max_disp_px', 8.0)),
        hidden=int(getattr(align_cfg, 'phi_hidden', 64)),
    )
    crit = {
        'deform_reg': DeformationRegLoss(
            smooth_weight=float(getattr(align_cfg, 'reg_smooth_weight', 10.0)),
            mag_weight=float(getattr(align_cfg, 'reg_mag_weight', 1.0)),
        ),
        'psc_anchor': PSCAnchorLoss(
            lam=float(getattr(align_cfg, 'psc_anchor_lambda', 4.0)),
            corr_weight=float(getattr(align_cfg, 'psc_corr_weight', 1.0)),
        ),
        'bsc': BackscatterStructureLoss(
            struct_weight=float(getattr(align_cfg, 'bsc_struct_weight', 1.0)),
        ),
    }
    return aligner, crit


def build_criterions(cfg, netG=None) -> dict:
    """Instantiate active losses based on weights in ``cfg.loss`` (weight > 0).

    ``netG`` is needed only to build PatchNCE (reads ``netG.encoder_dims``).
    """
    loss_cfg = cfg.loss
    # Mutual exclusion: per-band L1 (pre-IHaar subbands) overlaps WaveletDetailL1
    # (post-IHaar detail bands) on the detail-band gradient.
    if loss_cfg.get('per_band_weight', 0.0) > 0 and loss_cfg.get('wavelet_weight', 0.0) > 0:
        raise ValueError(
            "per_band_weight and wavelet_weight both > 0 — these overlap on the "
            "Haar detail bands. Pick one."
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
    # FFS: sliced-Wasserstein distributional loss on the Haar DETAIL subbands
    # (LH/HL/HH) of the predicted_sub tensor. Independent of per_band (which can
    # supervise LL-only alongside, or coexist) — both share predicted_sub but
    # SWD touches only the 3 detail bands. Shift-invariant blur penalty.
    if loss_cfg.get('swd_weight', 0.0) > 0:
        criterions['wavelet_swd'] = WaveletDetailSWDLoss(
            n_proj=int(loss_cfg.get('swd_n_proj', 0)),
            in_channels=3,
        )
    if loss_cfg.get('lpips_weight', 0.0) > 0:
        criterions['lpips'] = LPIPSLoss(net_type='alex')
    if loss_cfg.get('ffl_weight', 0.0) > 0:
        criterions['ffl'] = FFLLoss(alpha=float(loss_cfg.get('ffl_alpha', 1.0)))
    if loss_cfg.get('patchnce_weight', 0.0) > 0:
        if netG is None or not hasattr(netG, 'encoder_dims'):
            print('[factory] patchnce_weight>0 but netG/encoder_dims unavailable; skipping PatchNCE.')
        else:
            layer_ids = list(loss_cfg.get('patchnce_layers', [0, 1, 2, 3]))
            feat_dims = [int(netG.encoder_dims[i]) for i in layer_ids]
            criterions['patchnce'] = MultiLayerPatchNCE(
                feature_dims=feat_dims,
                layer_ids=layer_ids,
                nc=int(loss_cfg.get('patchnce_nc', 256)),
                num_patches=int(loss_cfg.get('patchnce_num_patches', 256)),
                temperature=float(loss_cfg.get('patchnce_temperature', 0.07)),
                batch_size=int(cfg.data.batch_size),
            )
    if loss_cfg.get('pepl_weight', 0.0) > 0:
        criterions['pepl'] = FoundationPerceptualLoss(
            backbone=str(loss_cfg.get('pepl_backbone', 'dinov2')),
            distance=str(loss_cfg.get('pepl_distance', 'l1')),
            channel_adapter=str(loss_cfg.get('pepl_channel_adapter', 'learnable')),
            layer_idxs=list(loss_cfg.get('pepl_layer_idxs', [-1])),
        )
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
    """AdamW for G (single group); Adam for D. Returns ``(opt_d, opt_g)``."""
    g_params = list(netG.parameters())
    if criterions is not None:
        if 'adaptive' in criterions:
            g_params.extend(criterions['adaptive'].parameters())
        if 'per_band' in criterions:
            g_params.extend(criterions['per_band'].parameters())
        if 'pepl' in criterions:
            g_params.extend(criterions['pepl'].trainable_parameters())
        if 'patchnce' in criterions:
            g_params.extend(criterions['patchnce'].parameters())

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
        if 'patchnce' in criterions:
            expected_ids.update(id(p) for p in criterions['patchnce'].parameters())
    opt_param_ids = {id(p) for pg in opt_g.param_groups for p in pg['params']}
    missing = expected_ids - opt_param_ids
    assert not missing, (
        f"build_optimizers: {len(missing)} G/criterion parameter(s) not in any opt_g group"
    )

    return opt_d, opt_g


def build_lr_schedulers(cfg, opt_d, opt_g):
    """constant / cosine_warm_restarts / linear_decay. Returns ``(sched_d, sched_g)``."""
    sched_type = str(cfg.scheduler.get('type', 'cosine_warm_restarts')).lower()
    eta_min = float(cfg.scheduler.eta_min)

    def make_sched(opt, base_lr):
        if sched_type == 'constant':
            return ConstantLR(opt, factor=1.0, total_iters=int(cfg.system.max_epochs))
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
            linear = LinearLR(opt, start_factor=1.0, end_factor=end_factor, total_iters=decay)
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
