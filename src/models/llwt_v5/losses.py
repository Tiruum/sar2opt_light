"""Loss primitives for LLW-Former v0.5.x (adversarial residual refiner).

Self-contained copy of the two adversarial-stage losses from
``src/models/sarformer_wb/losses.py`` (verbatim). Copied into llwt_v5 per the
copy-then-import rule: the A3 refiner stage may tune label smoothing / gan_type
independently of the v4 generator stage, so v5 owns its own copy.

* ``GANLoss``             — LSGAN-with-smoothing or hinge.
* ``FeatureMatchingLoss`` — element-weighted L1 across D feature layers.

Pixel anchor (light L1) uses ``torch.nn.functional.l1_loss`` directly in
``main.py`` — no wrapper needed.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["GANLoss", "FeatureMatchingLoss"]


class GANLoss(nn.Module):
    """LSGAN-with-smoothing or hinge.  Accepts a tensor or a tuple of tensors.

    ``gan_type='lsgan'``  (default): MSE against smoothed labels.
    ``gan_type='hinge'``  : D = relu(1 - D(real)) on real / relu(1 + D(fake)) on
                            fake; G = -D(fake).mean().  Pass ``for_d=False`` from
                            the G update to switch to the G branch — LSGAN
                            ignores the flag (its real/fake math is symmetric).
    """
    def __init__(self, real_smooth: float = 0.9, fake_smooth: float = 0.0,
                 gan_type: str = 'lsgan'):
        super().__init__()
        gan_type = str(gan_type).lower()
        if gan_type not in ('lsgan', 'hinge'):
            raise ValueError(f"GANLoss: gan_type must be 'lsgan' or 'hinge', got '{gan_type}'")
        self.gan_type = gan_type
        self.criterion = nn.MSELoss()
        self.real_smooth = float(real_smooth)
        self.fake_smooth = float(fake_smooth)

    def _lsgan_loss(self, logit: torch.Tensor, is_real: bool) -> torch.Tensor:
        val = self.real_smooth if is_real else self.fake_smooth
        return self.criterion(logit, torch.full_like(logit, val))

    def _hinge_loss(self, logit: torch.Tensor, is_real: bool, for_d: bool) -> torch.Tensor:
        if not for_d:
            return -logit.mean()
        if is_real:
            return F.relu(1.0 - logit).mean()
        return F.relu(1.0 + logit).mean()

    def _loss_one(self, logit: torch.Tensor, is_real: bool, for_d: bool) -> torch.Tensor:
        if self.gan_type == 'hinge':
            return self._hinge_loss(logit, is_real, for_d)
        return self._lsgan_loss(logit, is_real)

    def forward(self, logits, is_real: bool, for_d: bool = True) -> torch.Tensor:
        if isinstance(logits, (list, tuple)):
            return sum(self._loss_one(l, is_real, for_d) for l in logits) / len(logits)
        return self._loss_one(logits, is_real, for_d)


class FeatureMatchingLoss(nn.Module):
    """Element-weighted L1 across discriminator feature layers.

    ``sum(|fake_i - real_i|.sum()) / sum(numel_i)`` so each activation
    contributes equally — a 32×32 layer is not weighted the same as a 128×128.
    Real features are detached.
    """
    def forward(self, fake_feats: list, real_feats: list) -> torch.Tensor:
        total = sum(
            F.l1_loss(f, r.detach(), reduction='sum')
            for f, r in zip(fake_feats, real_feats)
        )
        n_elem = sum(f.numel() for f in fake_feats)
        return total / max(n_elem, 1)
