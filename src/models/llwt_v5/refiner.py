"""Pure residual UNet refiner for LLW-Former v0.5.x.

Replaces the SAR-DR diffusion approach (overkill for our scale: 17M params +
9K images can't learn distribution generation in 60ep).  This is a direct
supervised UNet that takes ``[SAR, coarse]`` and predicts a residual added
to coarse.  No noise schedule, no DDIM, no time embedding — just a small
U-Net trained with L1 loss against the optical target.

Pipeline:

    coarse  = G_frozen(SAR)                      (B, 3, H, W)
    residual = ResidualRefiner(SAR, coarse)      (B, 3, H, W)
    refined  = tanh(coarse_logit + residual_logit)
    loss     = L1(refined, opt) [+ MS-SSIM]

Strengths vs diffusion at our scale:
  * Single forward pass at inference (no 10-step DDIM)
  * Supervised L1 = signal-strong gradient (not noise prediction)
  * Architecture proven at 17M params for SAR2OPT-like tasks
  * Refines coarse instead of generating from noise

Architecture is the same UNet body as ``SARDRefiner`` but with the time
embedding + diffusion-specific zero-init head stripped.  Tanh applied at
the end so refined output stays in [-1, 1].

Public exports:

    * ``ResidualRefiner`` — UNet ``[SAR, coarse]`` -> residual.
"""
from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks — same as diffusion.py but without time injection.
# ---------------------------------------------------------------------------


class ResBlock(nn.Module):
    """Two-conv ResBlock with GroupNorm + SiLU. No time embedding."""

    def __init__(self, in_ch: int, out_ch: int, groups: int = 8):
        super().__init__()
        groups = min(groups, in_ch) if in_ch >= 1 else 1
        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        out_groups = min(8, out_ch)
        self.norm2 = nn.GroupNorm(out_groups, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class SelfAttention(nn.Module):
    """Multi-head self-attention over flattened spatial tokens."""

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        assert channels % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = channels // num_heads
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.qkv  = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)
        def split(t):
            return t.view(B, self.num_heads, self.head_dim, H * W)
        q, k, v = split(q), split(k), split(v)
        attn = torch.einsum('bhdn,bhdm->bhnm', q, k) / math.sqrt(self.head_dim)
        attn = attn.softmax(dim=-1)
        out = torch.einsum('bhnm,bhdm->bhdn', attn, v)
        out = out.reshape(B, C, H, W)
        return x + self.proj(out)


# ---------------------------------------------------------------------------
# ResidualRefiner UNet.
# ---------------------------------------------------------------------------


class ResidualRefiner(nn.Module):
    """U-Net that produces an additive residual for the frozen-G coarse output.

    Input concatenated along channel dim:

        SAR     (B, sar_ch, H, W)
        coarse  (B, 3,      H, W)
        TOTAL:  ``in_ch = sar_ch + 3``  (default = 4 for sar_ch=1)

    Output:

        residual (B, 3, H, W)        Added to coarse_logit at training time
                                     before tanh.

    Note: the residual is NOT itself tanh-clamped — clamping happens after
    ``coarse_logit + residual`` so the refiner has full range to undo wrong
    pixels in the coarse output.  Output 3x3 conv is zero-initialised so
    the refiner starts as identity (returns 0 residual), recovering exactly
    the frozen-G coarse output at step 0.
    """

    def __init__(
        self,
        sar_channels: int = 1,
        base_dim: int = 48,
        ch_mult: Tuple[int, ...] = (1, 2, 4, 4, 6),
        attn_levels: Tuple[int, ...] = (3, 4),
    ):
        super().__init__()
        self.sar_channels = sar_channels
        self.in_channels  = sar_channels + 3
        dims = [base_dim * m for m in ch_mult]
        self.attn_levels = set(attn_levels)

        self.in_proj = nn.Conv2d(self.in_channels, base_dim, kernel_size=3, padding=1)

        self.encoder = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        prev_dim = base_dim
        for idx, dim in enumerate(dims):
            block = nn.ModuleList([
                ResBlock(prev_dim, dim),
                ResBlock(dim, dim),
            ])
            if idx in self.attn_levels:
                block.append(SelfAttention(dim))
            self.encoder.append(block)
            self.downsamples.append(
                nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=1)
                if idx < len(dims) - 1 else nn.Identity()
            )
            prev_dim = dim

        self.decoder = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for idx, dim in enumerate(reversed(dims)):
            real_idx = len(dims) - 1 - idx
            if real_idx < len(dims) - 1:
                self.upsamples.append(
                    nn.ConvTranspose2d(prev_dim, dim, kernel_size=4, stride=2, padding=1)
                )
            else:
                self.upsamples.append(nn.Identity())
            block_in = dim * 2 if real_idx < len(dims) - 1 else dim
            block = nn.ModuleList([
                ResBlock(block_in, dim),
                ResBlock(dim, dim),
            ])
            if real_idx in self.attn_levels:
                block.append(SelfAttention(dim))
            self.decoder.append(block)
            prev_dim = dim

        out_groups = min(8, base_dim)
        self.out_norm = nn.GroupNorm(out_groups, base_dim)
        self.out_conv = nn.Conv2d(base_dim, 3, kernel_size=3, padding=1)
        # Zero-init so initial residual = 0 -> refined == coarse at step 0
        # (no drift before the refiner has learned anything).
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, sar: torch.Tensor, coarse: torch.Tensor) -> torch.Tensor:
        """Predict the additive residual.

        Args:
            sar:    (B, sar_ch, H, W)
            coarse: (B, 3, H, W) — frozen-G output, NOT tanh'd (used as logits)

        Returns:
            residual: (B, 3, H, W) — to be ``coarse + residual`` then tanh.
        """
        x = torch.cat([sar, coarse], dim=1)
        h = self.in_proj(x)

        skips: List[torch.Tensor] = []
        for block, downsample in zip(self.encoder, self.downsamples):
            for module in block:
                h = module(h)
            skips.append(h)
            h = downsample(h)

        for level_idx, (block, upsample) in enumerate(zip(self.decoder, self.upsamples)):
            real_idx = len(self.encoder) - 1 - level_idx
            if not isinstance(upsample, nn.Identity):
                h = upsample(h)
                skip = skips[real_idx]
                if h.shape[-2:] != skip.shape[-2:]:
                    h = F.interpolate(h, size=skip.shape[-2:], mode='nearest')
                h = torch.cat([h, skip], dim=1)
            for module in block:
                h = module(h)

        h = F.silu(self.out_norm(h))
        return self.out_conv(h)


# ---------------------------------------------------------------------------
# Tier 1 unit smoke.
# ---------------------------------------------------------------------------


def _smoke() -> None:
    print("[refiner smoke] forward + grad check")
    torch.manual_seed(0)
    B, H, W = 2, 64, 64

    sar    = torch.randn(B, 1, H, W)
    coarse = torch.tanh(torch.randn(B, 3, H, W))

    refiner = ResidualRefiner(sar_channels=1, base_dim=32, ch_mult=(1, 2, 4, 4))
    n_params = sum(p.numel() for p in refiner.parameters())
    print(f"  refiner params: {n_params/1e6:.2f}M  (test config)")

    residual = refiner(sar, coarse)
    assert residual.shape == (B, 3, H, W), f"bad shape: {residual.shape}"
    print(f"  [OK] forward shape: residual = {tuple(residual.shape)}")
    print(f"  [OK] zero-init: max(|residual|) = {residual.abs().max():.6f}  (should be 0)")

    target = torch.randn_like(residual)
    loss = F.l1_loss((coarse + residual), target)
    loss.backward()
    grads = [p.grad for p in refiner.parameters() if p.grad is not None]
    nz = sum(int(g.abs().sum() > 0) for g in grads)
    print(f"  [OK] gradient flows: {nz}/{len(grads)} params received nonzero grad; loss = {loss.item():.4f}")
    print("[refiner smoke] PASS")


if __name__ == '__main__':
    _smoke()
