"""DiffAugment: differentiable augmentation for data-efficient GAN training.

Source: Zhao, Liu, Lin, Zhu, Han. "Differentiable Augmentation for
Data-Efficient GAN Training", NeurIPS 2020. arXiv:2006.10738.
Reference implementation: https://github.com/mit-han-lab/data-efficient-gans
License: BSD 2-clause. Copyright (c) 2020 Massachusetts Institute of Technology.

Adaptations for SAR2OPT-V1 (paired conditional GAN):
  * Geometric augmentations (translation, cutout) are applied JOINTLY to the
    (SAR, optical) pair so the paired condition stays aligned.
  * Color augmentations (brightness, saturation, contrast) are applied ONLY
    to the optical channels -- SAR is a backscatter measurement, perturbing
    its absolute value breaks the physics conditioning.
  * Apply to both real and fake D inputs in the same step (paper-mandated).
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


__all__ = ["DiffAugment", "diff_augment_pair"]


# ----------------------------- color augmentations -----------------------------


def _rand_brightness(x: torch.Tensor) -> torch.Tensor:
    return x + (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) - 0.5)


def _rand_saturation(x: torch.Tensor) -> torch.Tensor:
    x_mean = x.mean(dim=1, keepdim=True)
    return (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) * 2) + x_mean


def _rand_contrast(x: torch.Tensor) -> torch.Tensor:
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    return (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) + 0.5) + x_mean


# --------------------------- geometric augmentations ---------------------------


def _rand_translation(x: torch.Tensor, ratio: float = 0.125) -> torch.Tensor:
    shift_x = int(x.size(2) * ratio + 0.5)
    shift_y = int(x.size(3) * ratio + 0.5)
    tx = torch.randint(-shift_x, shift_x + 1, size=(x.size(0), 1, 1), device=x.device)
    ty = torch.randint(-shift_y, shift_y + 1, size=(x.size(0), 1, 1), device=x.device)
    grid_b, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
        indexing='ij',
    )
    grid_x = torch.clamp(grid_x + tx + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + ty + 1, 0, x.size(3) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    return x_pad.permute(0, 2, 3, 1).contiguous()[grid_b, grid_x, grid_y].permute(0, 3, 1, 2).contiguous()


def _rand_cutout(x: torch.Tensor, ratio: float = 0.5) -> torch.Tensor:
    cut_h = int(x.size(2) * ratio + 0.5)
    cut_w = int(x.size(3) * ratio + 0.5)
    off_x = torch.randint(0, x.size(2) + (1 - cut_h % 2), size=(x.size(0), 1, 1), device=x.device)
    off_y = torch.randint(0, x.size(3) + (1 - cut_w % 2), size=(x.size(0), 1, 1), device=x.device)
    grid_b, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(cut_h, dtype=torch.long, device=x.device),
        torch.arange(cut_w, dtype=torch.long, device=x.device),
        indexing='ij',
    )
    grid_x = torch.clamp(grid_x + off_x - cut_h // 2, 0, x.size(2) - 1)
    grid_y = torch.clamp(grid_y + off_y - cut_w // 2, 0, x.size(3) - 1)
    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mask[grid_b, grid_x, grid_y] = 0
    return x * mask.unsqueeze(1)


# ---------------------- public helpers --------------------------------------


_COLOR_FNS = [_rand_brightness, _rand_saturation, _rand_contrast]


def DiffAugment(x: torch.Tensor, policy: str = 'color,translation,cutout') -> torch.Tensor:
    """Single-tensor DiffAugment (NCHW). Used for the optical-only case."""
    if not policy:
        return x
    for p in policy.split(','):
        p = p.strip()
        if p == 'color':
            for f in _COLOR_FNS:
                x = f(x)
        elif p == 'translation':
            x = _rand_translation(x)
        elif p == 'cutout':
            x = _rand_cutout(x)
        else:
            raise ValueError(f"DiffAugment: unknown policy '{p}'")
    return x.contiguous()


def diff_augment_pair(
    sar: torch.Tensor,
    opt: torch.Tensor,
    policy: str = 'color,translation,cutout',
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply DiffAugment to a paired (SAR, optical) batch.

    Geometric augmentations are applied jointly so the SAR-optical pixel
    correspondence is preserved. Color augmentations are applied only to the
    optical channels (SAR pixel values encode physics; perturbing them would
    break the conditioning signal D learns to read).
    """
    if not policy:
        return sar, opt
    for p in policy.split(','):
        p = p.strip()
        if p == 'color':
            for f in _COLOR_FNS:
                opt = f(opt)
        elif p in ('translation', 'cutout'):
            stacked = torch.cat([sar, opt], dim=1)
            fn = _rand_translation if p == 'translation' else _rand_cutout
            stacked = fn(stacked)
            n_sar = sar.size(1)
            sar = stacked[:, :n_sar]
            opt = stacked[:, n_sar:]
        else:
            raise ValueError(f"diff_augment_pair: unknown policy '{p}'")
    return sar.contiguous(), opt.contiguous()
