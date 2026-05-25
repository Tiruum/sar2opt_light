"""Factory for LLW-Former v0.5.x — adversarial residual refiner on frozen v4 G.

v0.5.2 (A3): the residual UNet refiner is trained adversarially against the
v4 discriminator stack (MainDis + SubbandDis) with GAN + feature-matching +
a light L1 anchor.  The v4 generator stays frozen — only the refiner and the
discriminator step.

Why adversarial (vs the v0.5.1 pure-L1 refiner): L1's minimiser is the
conditional mean E[opt|sar], which for the ill-posed SAR->opt mapping is
inherently blurry.  Pure-L1 refinement therefore slides along the PSNR<->FID
Pareto front toward the (blurry) PSNR corner — measured: PSNR +1.0 dB but FID
143->167, LPIPS 0.570->0.613 over 13 epochs.  Only a distributional
(adversarial) signal can ADVANCE the front past the frozen-G FID corner.  The
light L1 anchor (cfg.loss.l1_weight) is the leash that keeps the refiner near
the data and bounds adversarial hallucination.

Self-containment (copy-then-import rule): the discriminator and the GAN/FM
losses are COPIED into llwt_v5 (``dis.py``, ``losses.py``) because the A3
refiner stage may tune them independently of the v4 generator stage.  The LR
scheduler builder and EMA callback are IMPORTED from the shared ``llwt``
factory (stable utilities, not modified here).  The frozen ``LLWv4Generator``
is imported (frozen, never modified).

Builders:
  * ``build_models``             -> ``(netG, netD, refiner)`` when
                                     ``cfg.refiner.enabled``; else ``(netG, netD)``.
  * ``build_refiner_criterions``  -> ``{'gan': GANLoss, 'fm': FeatureMatchingLoss}``.
  * ``build_refiner_optimizers``  -> ``(opt_d, opt_g)`` — opt_d on netD,
                                     opt_g on the refiner only (G frozen).
  * ``build_lr_schedulers`` / ``build_ema_callback`` re-exported from llwt.
"""
from __future__ import annotations

import torch.optim as optim

from src.models.llwt.factory import (    # noqa: F401  (intentional re-export)
    build_ema_callback,
    build_lr_schedulers,
)

# Local (copied) discriminator + losses — self-contained per copy-then-import.
from src.models.llwt_v5.dis import LLWFormerDiscriminator
from src.models.llwt_v5.losses import GANLoss, FeatureMatchingLoss
# Frozen v4 generator — imported, never modified.
from src.models.llwt_v4.gen import LLWv4Generator
from src.models.llwt_v5.refiner import ResidualRefiner


__all__ = [
    "build_models",
    "build_refiner_criterions",
    "build_refiner_optimizers",
    "build_lr_schedulers",
    "build_ema_callback",
]


def _refiner_enabled(cfg) -> bool:
    refiner_cfg = getattr(cfg, 'refiner', None)
    if refiner_cfg is None:
        return False
    return bool(getattr(refiner_cfg, 'enabled', False))


def build_models(cfg):
    """Returns ``(netG, netD)`` or ``(netG, netD, refiner)`` if refiner enabled."""
    netG = LLWv4Generator(cfg=cfg)
    netD = LLWFormerDiscriminator(cfg=cfg)
    if not _refiner_enabled(cfg):
        return netG, netD

    refiner_cfg = cfg.refiner
    sar_channels = int(cfg.data.sar_channels)
    base_dim = int(refiner_cfg.get('base_dim', 48))
    ch_mult = tuple(int(m) for m in refiner_cfg.get('ch_mult', (1, 2, 4, 4, 6)))
    attn_levels = tuple(int(l) for l in refiner_cfg.get('attn_levels', (3, 4)))
    refiner = ResidualRefiner(
        sar_channels=sar_channels,
        base_dim=base_dim,
        ch_mult=ch_mult,
        attn_levels=attn_levels,
    )
    return netG, netD, refiner


def build_refiner_criterions(cfg) -> dict:
    """Adversarial criterions for the refiner stage: GAN + feature-matching.

    The light L1 pixel anchor is applied directly in ``main.py`` via
    ``F.l1_loss`` (no module needed).  GANLoss reads label smoothing and
    gan_type from ``cfg.loss`` so the refiner stage can tune them without
    touching the v4 generator-stage config.
    """
    loss_cfg = cfg.loss
    return {
        'gan': GANLoss(
            real_smooth=float(getattr(loss_cfg, 'real_smooth', 1.0)),
            fake_smooth=float(getattr(loss_cfg, 'fake_smooth', 0.0)),
            gan_type=str(getattr(loss_cfg, 'gan_type', 'lsgan')),
        ),
        'fm': FeatureMatchingLoss(),
    }


def _make_adamw(params, lr, betas, wd, want_bnb):
    """AdamW with bnb 8-bit when available + requested, else fused fp32 AdamW."""
    if want_bnb:
        try:
            import bitsandbytes.optim as bnb_optim
            return bnb_optim.AdamW8bit(params, lr=lr, betas=betas, weight_decay=wd)
        except Exception:
            pass
    return optim.AdamW(params, lr=lr, betas=betas, weight_decay=wd, fused=True)


def build_refiner_optimizers(cfg, netD, refiner):
    """Returns ``(opt_d, opt_g)``.

    opt_g steps the refiner only (v4 G frozen).  Both use ``cfg.optimizer``
    LRs/betas so the shared ``build_lr_schedulers`` (which reads
    ``cfg.optimizer.lr_g`` / ``lr_d`` for its linear-decay base) stays
    consistent — single source of truth for LR.  GAN-standard beta1=0.5.
    """
    opt_cfg = cfg.optimizer
    lr_g = float(opt_cfg.lr_g)
    lr_d = float(opt_cfg.lr_d)
    betas = (float(opt_cfg.beta1), float(opt_cfg.beta2))
    wd_g = float(getattr(opt_cfg, 'weight_decay_g', 0.0))
    want_bnb = bool(getattr(opt_cfg, 'use_bnb_8bit', False))

    opt_g = _make_adamw(refiner.parameters(), lr_g, betas, wd_g, want_bnb)
    opt_d = _make_adamw(netD.parameters(), lr_d, betas, 0.0, want_bnb)
    return opt_d, opt_g
