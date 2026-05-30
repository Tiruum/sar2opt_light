"""Loss suite for SAR2OPT-V1.

Re-exports the proven huggingface_gan losses (single source of truth) and
adds three new wrappers:

  * :class:`HingeGANLoss`      -- Lim & Ye 2017 hinge loss as a drop-in
    alternative to LSGAN; switched on via ``cfg.loss.gan_type = 'hinge'``.
  * :class:`FFLLoss`           -- thin wrapper around the official
    ``focal_frequency_loss`` PyPI package (Jiang et al., ICCV 2021,
    arXiv:2012.12821). Sharp HF anchor in the frequency domain.
  * :class:`MultiLayerPatchNCE` -- owns the per-layer ``PatchSampleF`` MLP
    + a list of ``PatchNCELoss`` instances, one per encoder stage. Exposes
    a single ``forward(fake_feats, real_feats)`` -> scalar loss.
"""
from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

# Re-export the proven huggingface_gan losses -- single source of truth.
from src.models.huggingface_gan.losses import (  # noqa: F401  (intentional re-export)
    FeatureMatchingLoss,
    GANLoss,
    L1Loss,
    LABChromaL1Loss,
    MSSSIMLoss,
    WaveletDetailL1Loss,
)

from src.models.sar2opt_v1.patchnce import PatchNCELoss, PatchSampleF


__all__ = [
    "FeatureMatchingLoss",
    "GANLoss",
    "HingeGANLoss",
    "L1Loss",
    "LABChromaL1Loss",
    "MSSSIMLoss",
    "WaveletDetailL1Loss",
    "FFLLoss",
    "MultiLayerPatchNCE",
]


# ---------------------------------------------------------------------------
# Hinge GAN loss (Lim & Ye 2017 / SAGAN 2018 default).
# ---------------------------------------------------------------------------


class HingeGANLoss(nn.Module):
    """Hinge loss for GAN.

    For the discriminator:
        L_D_real = mean(relu(1 - D(real)))
        L_D_fake = mean(relu(1 + D(fake)))
    For the generator (when ``is_real=True`` is requested on fake logits):
        L_G = -mean(D(fake))

    The ``is_real`` flag is overloaded the same way as ``GANLoss``:
        * D-step real: ``criterion(real_logits, is_real=True)``
        * D-step fake: ``criterion(fake_logits, is_real=False)``
        * G-step    : ``criterion(fake_logits, is_real=True)`` -- D-fake hinge with sign flipped

    The Lightning module is responsible for calling this in the right places;
    no separate "for_d/for_g" branch is needed.
    """

    def _one(self, logit: torch.Tensor, is_real: bool, is_generator: bool) -> torch.Tensor:
        if is_generator:
            return -logit.mean()
        if is_real:
            return F.relu(1.0 - logit).mean()
        return F.relu(1.0 + logit).mean()

    def forward(self, logits, is_real: bool, is_generator: bool = False) -> torch.Tensor:
        if isinstance(logits, (list, tuple)):
            return sum(self._one(l, is_real, is_generator) for l in logits) / len(logits)
        return self._one(logits, is_real, is_generator)


# ---------------------------------------------------------------------------
# Focal Frequency Loss (Jiang ICCV 2021).
# ---------------------------------------------------------------------------


class FFLLoss(nn.Module):
    """Focal Frequency Loss wrapper (official PyPI ``focal_frequency_loss``).

    Defers the import so a missing package shows a clear error message at
    construction time (when the criterion is enabled by weight > 0), not at
    repo import.

    The underlying loss focuses the L1 in the FFT domain on hard-to-synthesise
    frequency components via a focal-style re-weighting (alpha=1.0 default).
    Inputs are auto-cast to float32 for bf16-mixed safety.
    """

    def __init__(self, alpha: float = 1.0, patch_factor: int = 1, ave_spectrum: bool = False):
        super().__init__()
        try:
            from focal_frequency_loss import FocalFrequencyLoss as FFL
        except ImportError as e:
            raise ImportError(
                "FFLLoss requires the `focal-frequency-loss` package. Install via "
                "`pip install focal-frequency-loss` (Jiang ICCV 2021 official)."
            ) from e
        self._ffl = FFL(
            loss_weight=1.0,
            alpha=float(alpha),
            patch_factor=int(patch_factor),
            ave_spectrum=bool(ave_spectrum),
            log_matrix=False,
            batch_matrix=False,
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self._ffl(pred.float(), target.float())


# ---------------------------------------------------------------------------
# Multi-layer PatchNCE wrapper.
# ---------------------------------------------------------------------------


class MultiLayerPatchNCE(nn.Module):
    """Owns the PatchSampleF MLP head + per-layer PatchNCE losses.

    Forward contract:
        loss = mlp_pnce(fake_feats, real_feats)
    where ``fake_feats`` / ``real_feats`` are lists of length L (one feature
    map per encoder stage in ``layer_ids``).  Returns the mean PatchNCE loss
    across the L layers.

    Patch IDs are sampled once on the query (fake) side and reused on the
    key (real) side so positive pairs align by spatial position -- this is
    the paired-translation adaptation of CUT.
    """

    def __init__(
        self,
        feature_dims: Sequence[int],
        layer_ids: Sequence[int],
        nc: int = 256,
        num_patches: int = 256,
        temperature: float = 0.07,
        batch_size: int = 1,
    ):
        super().__init__()
        assert len(feature_dims) == len(layer_ids), (
            f"feature_dims ({len(feature_dims)}) and layer_ids ({len(layer_ids)}) length mismatch"
        )
        self.layer_ids = tuple(int(i) for i in layer_ids)
        self.num_patches = int(num_patches)
        self.sampler = PatchSampleF(
            feature_dims=[int(d) for d in feature_dims],
            nc=int(nc),
        )
        self.nce_losses = nn.ModuleList([
            PatchNCELoss(temperature=temperature, batch_size=batch_size)
            for _ in feature_dims
        ])

    @staticmethod
    def _pick(feats: List[torch.Tensor], layer_ids: Sequence[int]) -> List[torch.Tensor]:
        return [feats[i] for i in layer_ids]

    def forward(
        self,
        fake_feats_full: List[torch.Tensor],
        real_feats_full: List[torch.Tensor],
    ) -> torch.Tensor:
        fake_feats = self._pick(fake_feats_full, self.layer_ids)
        real_feats = self._pick(real_feats_full, self.layer_ids)
        # Sample patches on the query side, then reuse the IDs on the key side
        # so positive pairs align by spatial position.
        fq_sampled, patch_ids = self.sampler(fake_feats, num_patches=self.num_patches)
        fk_sampled, _ = self.sampler(real_feats, num_patches=self.num_patches, patch_ids=patch_ids)
        total = 0.0
        for fq, fk, loss_fn in zip(fq_sampled, fk_sampled, self.nce_losses):
            total = total + loss_fn(fq, fk)
        return total / max(len(self.nce_losses), 1)
