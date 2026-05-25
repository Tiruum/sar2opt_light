"""Loss suite for LLW-Former.

Re-exports the proven sarformer_wb losses so we don't fork a parallel
implementation (and so any bugfix in one place lands here automatically):

* ``GANLoss``                — LSGAN with label smoothing
* ``FeatureMatchingLoss``    — element-weighted L1 over D features
* ``MSSSIMLoss``             — 1 - MS-SSIM
* ``LABChromaL1Loss``        — L1 on CIE LAB a*/b* with ab_norm rescale
* ``WaveletDetailL1Loss``    — L1 on Haar detail subbands (full-res output)
* ``PerBandWaveletL1Loss``   — per-band L1 (LL+LH+HL+HH) on the pre-IHaar
  predicted subband tensor (v0.4.x).  Mutually exclusive with
  ``WaveletDetailL1Loss`` at the factory level.
* ``PlainL1Loss``            — F.l1_loss wrapper
* ``AdaptiveLoss``           — homoscedastic uncertainty weighting (opt-in)

LLW-Former-specific additions:

* ``SpeckleDecoupleLoss``    — anchors learned ``L1_LL`` to a despeckled-SAR
  proxy and constrains ``L1_HH`` magnitude (interpretable lifting).
* ``PerfectReconLoss``       — penalises ``|x - iLLW(LLW(x))|`` to keep the
  learnable wavelet's PR error close to zero under fp32/bf16.
* ``LPIPSLoss``              — AlexNet LPIPS as a perceptual term.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Re-export sarformer_wb losses.  Single source of truth keeps both modules
# in sync.
from src.models.sarformer_wb.losses import (  # noqa: F401  (re-export)
    AdaptiveLoss,
    FeatureMatchingLoss,
    FoundationPerceptualLoss,
    GANLoss,
    LABChromaL1Loss,
    MSSSIMLoss,
    PerBandWaveletL1Loss,
    PlainL1Loss,
    WaveletDetailL1Loss,
)


__all__ = [
    "AdaptiveLoss",
    "FeatureMatchingLoss",
    "FoundationPerceptualLoss",
    "GANLoss",
    "LABChromaL1Loss",
    "MSSSIMLoss",
    "PerBandWaveletL1Loss",
    "PlainL1Loss",
    "WaveletDetailL1Loss",
    "SpeckleDecoupleLoss",
    "PerfectReconLoss",
    "LPIPSLoss",
]


# ---------------------------------------------------------------------------
# Speckle-Decouple Regulariser (novelty twist)
# ---------------------------------------------------------------------------


class SpeckleDecoupleLoss(nn.Module):
    """Anchors the learned lifting so LL ~ despeckled-SAR, HH ~ speckle.

    Two components (both light-weight, ``weight=0.05`` default in the loss
    stack):

    1. ``l_ll = |LL1.mean(C) - despeckled(SAR)|`` — pulls the channel-averaged
       level-1 LL toward a 5x5 average-pool of ``|SAR|``.  Using ``avg_pool``
       (not NLM) keeps it GPU-cheap; the despeckle target is rough but
       directionally correct.
    2. ``l_hh = ||HH1|^2 - target_energy|`` — pins the spatial mean of the
       HH band's energy to a target (default ``0.05``).  Without this the
       network can collapse HH to zero (passing all information through LL).

    Both terms are differentiable through the lifting nets, so the wavelet
    learns to specialise toward despeckling without any external pretrained
    denoiser.
    """
    def __init__(self, target_hh_energy: float = 0.05, eps: float = 1e-6):
        super().__init__()
        self.target_hh_energy = float(target_hh_energy)
        self.eps = float(eps)

    def forward(self, sar: torch.Tensor, coeffs) -> torch.Tensor:
        """``sar``: (B,1,H,W).  ``coeffs``: dict from ``LLWTransform``."""
        ll1 = coeffs['L1_LL']                                       # (B,C,H/2,W/2)
        hh1 = coeffs['L1_HH']
        # ---- LL anchor ---------------------------------------------------
        despeckled = F.avg_pool2d(sar.abs(), 5, stride=2, padding=2)
        despeckled = despeckled.detach()
        # Shape sanity — LL may have larger spatial extent if SAR was odd:
        if despeckled.shape[-2:] != ll1.shape[-2:]:
            despeckled = F.interpolate(despeckled, size=ll1.shape[-2:],
                                       mode='bilinear', align_corners=False)
        ll_avg = ll1.mean(dim=1, keepdim=True)
        l_ll = F.l1_loss(ll_avg, despeckled)
        # ---- HH energy ---------------------------------------------------
        hh_energy = hh1.pow(2).mean()
        l_hh = (hh_energy - self.target_hh_energy).abs()
        return l_ll + l_hh


# ---------------------------------------------------------------------------
# Perfect-Reconstruction Regulariser
# ---------------------------------------------------------------------------


class PerfectReconLoss(nn.Module):
    """``|x - iLLW(LLW(x))|`` — keeps the learnable wavelet exactly invertible.

    Lifting math gives PR by construction at fp64; under bf16/fp32 mixed
    precision the round-trip picks up float-rounding noise that, with a
    deep-enough network and many epochs, can compound.  This tiny loss
    (default ``weight=0.01``) keeps that noise pinned to zero throughout
    training.  Costs one extra inverse pass per step, no extra forward
    backbone.
    """
    def forward(self, llw_module, x: torch.Tensor) -> torch.Tensor:
        coeffs = llw_module(x)
        x_hat = llw_module.inverse(coeffs)
        return F.l1_loss(x_hat, x)


# ---------------------------------------------------------------------------
# LPIPS (AlexNet) wrapped as nn.Module
# ---------------------------------------------------------------------------


class LPIPSLoss(nn.Module):
    """AlexNet LPIPS (Zhang et al. 2018) via ``torchmetrics``.

    Eager-built so the AlexNet parameters are registered as children of the
    LightningModule at ``__init__`` time.  A previous lazy-init version
    triggered an EMA crash at ``ema.start_epoch``: Lightning's
    ``WeightAveraging`` callback snapshots ``pl_module.parameters()`` during
    ``setup``, before the first training forward, so a lazy LPIPS module
    appeared in the *live* parameter list but not in the averaged shadow.
    On the first EMA update, ``zip(avg.parameters(), live.parameters())``
    drifted past the LPIPS insertion point and the AlexNet conv1 (kernel
    11x11) tried to copy into a downstream 3x3 conv.
    Inputs are expected in ``[-1, 1]``; LPIPS' ``normalize=False`` consumes
    that range directly.
    """
    def __init__(self, net_type: str = 'alex'):
        super().__init__()
        self.net_type = net_type
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
        self._lpips = LearnedPerceptualImagePatchSimilarity(
            net_type=net_type, normalize=False,
        )
        self._lpips.eval()
        for p in self._lpips.parameters():
            p.requires_grad_(False)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # LPIPS expects float32; cast if running under bf16 autocast.
        return self._lpips(pred.float(), target.float())
