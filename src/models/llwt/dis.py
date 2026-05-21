"""Discriminators for LLW-Former.

Two heads, both LSGAN-trained, both contributing to the feature-matching loss:

* ``MainDis``        — 2-scale conditional PatchGAN on ``[InstanceNorm(SAR),
  optical]`` (mirrors the sarformer_wb MSPatchGAN pattern: coarse 5-layer
  70x70 RF + fine 3-layer ~46x46 RF, all spectral-norm, layer-0 features
  dropped from FM as per the hfgan-18 lesson).

* ``SubbandDis``     — *unconditional* PatchGAN on the **fixed Haar L=1**
  decomposition of the optical pair only.  Input shape ``(B, 12, H/2, W/2)``
  = concat of the 4 subbands (3 channels each).  Operates in coefficient
  space so the adversarial signal is forced to align frequency statistics
  (a known gap in the L1-anchored pix2pix lineage).

The wrapper ``LLWFormerDiscriminator`` calls both heads and concatenates their
features for FM.  Returns ``((logits_main_pair, logits_subband), features)``.

The Haar used by the subband head is **fixed** (not learnable) — separate
from the generator's *learnable* lifting wavelet — so the discriminator
cannot collude with the generator by co-adapting their wavelets.
"""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


__all__ = ["LLWFormerDiscriminator", "MainDis", "SubbandDis", "FixedHaarDWT"]


def _sn(ci: int, co: int, k: int, s: int, p: int) -> nn.Conv2d:
    return spectral_norm(nn.Conv2d(ci, co, kernel_size=k, stride=s, padding=p, bias=True))


# ---------------------------------------------------------------------------
# Fixed Haar L=1 DWT for the subband head.  Buffers, no trainable params.
# ---------------------------------------------------------------------------


class FixedHaarDWT(nn.Module):
    """One-level orthonormal Haar DWT, applied per channel.

    Input  ``(B, C, H, W)``  H,W even.
    Output ``(B, 4*C, H/2, W/2)``  channel order ``[LL, LH, HL, HH]`` per
    input channel: ``out[:, 0:C]`` = LL block, ``out[:, C:2C]`` = LH, etc.

    Implementation: 2x2 pixel-unshuffle + a fixed orthonormal mixing matrix.
    Parseval-preserving.  Identical filter coefficients to
    ``sarformer_wb.losses.WaveletDetailL1Loss``.
    """
    _HAAR = torch.tensor([
        [ 1.0,  1.0,  1.0,  1.0],   # LL
        [-1.0,  1.0, -1.0,  1.0],   # LH
        [-1.0, -1.0,  1.0,  1.0],   # HL
        [ 1.0, -1.0, -1.0,  1.0],   # HH
    ], dtype=torch.float32) * 0.5

    def __init__(self):
        super().__init__()
        self.register_buffer('haar_weights', self._HAAR.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        if H % 2 or W % 2:
            raise ValueError(f"FixedHaarDWT needs even H,W; got {(H, W)}")
        # 2x2 block-unshuffle: (B, 4*C, H/2, W/2).  Order: top-left,
        # top-right, bottom-left, bottom-right per 2x2 block.
        blocks = F.pixel_unshuffle(x, 2).view(B, C, 4, H // 2, W // 2)
        # Mix to (LL, LH, HL, HH) per channel.
        out = torch.einsum('bcihw, oi -> bcohw', blocks, self.haar_weights)
        # Flatten channel and band so subband index is contiguous per band:
        # band-major layout makes downstream "look at HH only" slicing easy.
        return out.permute(0, 2, 1, 3, 4).reshape(B, 4 * C, H // 2, W // 2)


# ---------------------------------------------------------------------------
# 2-scale conditional PatchGAN main head (sarformer_wb pattern)
# ---------------------------------------------------------------------------


class _CoarsePatchBranch(nn.Module):
    """5-layer 70x70 RF spectral-norm PatchGAN."""
    def __init__(self, in_ch: int, ndf: int = 64):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(_sn(in_ch,    ndf,     4, 2, 1), nn.LeakyReLU(0.2, inplace=True)),
            nn.Sequential(_sn(ndf,      ndf * 2, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True)),
            nn.Sequential(_sn(ndf * 2,  ndf * 4, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True)),
            nn.Sequential(_sn(ndf * 4,  ndf * 8, 4, 1, 1), nn.LeakyReLU(0.2, inplace=True)),
            _sn(ndf * 8, 1, 4, 1, 1),
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        feats: List[torch.Tensor] = []
        for layer in self.layers[:-1]:
            x = layer(x)
            feats.append(x)
        return self.layers[-1](x), feats


class _FinePatchBranch(nn.Module):
    """3-layer ~46x46 RF spectral-norm PatchGAN."""
    def __init__(self, in_ch: int, ndf: int = 64):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(_sn(in_ch,    ndf,     4, 2, 1), nn.LeakyReLU(0.2, inplace=True)),
            nn.Sequential(_sn(ndf,      ndf * 2, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True)),
            nn.Sequential(_sn(ndf * 2,  ndf * 4, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True)),
            _sn(ndf * 4, 1, 4, 1, 1),
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        feats: List[torch.Tensor] = []
        for layer in self.layers[:-1]:
            x = layer(x)
            feats.append(x)
        return self.layers[-1](x), feats


class MainDis(nn.Module):
    """2-scale conditional PatchGAN, asymmetric InstanceNorm on SAR only.

    Returns ``((logits_coarse, logits_fine), feats)``.  feats has layer-0
    dropped from each branch (5 features total at 256x256 input: 3 from
    coarse, 2 from fine).
    """
    def __init__(self, in_ch: int = 4, ndf: int = 64):
        super().__init__()
        self.coarse = _CoarsePatchBranch(in_ch, ndf)
        self.fine   = _FinePatchBranch(in_ch, ndf)

    def forward(self, sar: torch.Tensor, opt: torch.Tensor):
        sar_n = F.instance_norm(sar)
        x = torch.cat([sar_n, opt], dim=1)
        l1, f1 = self.coarse(x)
        l2, f2 = self.fine(x)
        return (l1, l2), f1[1:] + f2[1:]


# ---------------------------------------------------------------------------
# Subband PatchGAN head (unconditional, on Haar L=1 of optical)
# ---------------------------------------------------------------------------


class _SubbandPatchBranch(nn.Module):
    """4-layer spectral-norm PatchGAN on Haar-coefficient feature maps."""
    def __init__(self, in_ch: int = 12, ndf: int = 32):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(_sn(in_ch,    ndf,     4, 2, 1), nn.LeakyReLU(0.2, inplace=True)),
            nn.Sequential(_sn(ndf,      ndf * 2, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True)),
            nn.Sequential(_sn(ndf * 2,  ndf * 4, 4, 1, 1), nn.LeakyReLU(0.2, inplace=True)),
            _sn(ndf * 4, 1, 4, 1, 1),
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        feats: List[torch.Tensor] = []
        for layer in self.layers[:-1]:
            x = layer(x)
            feats.append(x)
        return self.layers[-1](x), feats


class SubbandDis(nn.Module):
    """Unconditional PatchGAN on fixed-Haar L=1 of the optical pair.

    Forward: ``opt -> ((B,1,h,w), feats)``.  The SAR is intentionally not
    consumed here — the subband head's job is to align *frequency
    statistics* of the optical distribution, not the conditional mapping.
    Conditional alignment is the main D's job.
    """
    def __init__(self, ndf: int = 32):
        super().__init__()
        self.dwt = FixedHaarDWT()
        self.branch = _SubbandPatchBranch(in_ch=12, ndf=ndf)  # 3ch * 4 subbands

    def forward(self, opt: torch.Tensor):
        coeffs = self.dwt(opt)                               # (B,12,H/2,W/2)
        return self.branch(coeffs)


# ---------------------------------------------------------------------------
# Wrapper: main + subband
# ---------------------------------------------------------------------------


class LLWFormerDiscriminator(nn.Module):
    """Wraps ``MainDis`` and ``SubbandDis`` with optional subband enable flag.

    Forward returns
        ``(logits_main_pair, logits_subband, features)``
    where ``logits_main_pair`` is the ``(coarse, fine)`` tuple from the main
    head, ``logits_subband`` is the subband-D logits (or ``None`` when the
    head is disabled), and ``features`` is the concatenated FM feature list.

    Subband-head can be disabled via ``cfg.model.dis.subband.enabled = false``;
    the codepath is then a no-op (and removed from the FM list) so a clean
    ablation can be run without architectural changes.
    """
    def __init__(self, cfg=None):
        super().__init__()
        dcfg = None if cfg is None else cfg.model.dis
        main_cfg     = _g(dcfg, 'main', None)
        in_ch        = int(_g(main_cfg, 'in_channels', 4))
        ndf_main     = int(_g(main_cfg, 'ndf', 64))
        sub_cfg      = _g(dcfg, 'subband', None)
        self.use_sub = bool(_g(sub_cfg, 'enabled', True))
        ndf_sub      = int(_g(sub_cfg, 'ndf', 32))

        self.main = MainDis(in_ch=in_ch, ndf=ndf_main)
        self.sub  = SubbandDis(ndf=ndf_sub) if self.use_sub else None

    def forward(self, sar: torch.Tensor, opt: torch.Tensor):
        """Returns ``((logits_coarse, logits_fine), logits_subband, feats_main,
        feats_sub)``.

        Keeping ``feats_main`` and ``feats_sub`` as separate lists lets the
        training loop weight their feature-matching contributions independently
        via ``loss.fm_main_weight`` and ``loss.fm_sub_weight``.  Concatenating
        them would force a single weight across the whole pool and silently
        drop ``fm_sub_weight``.
        """
        (lc, lf), feats_main = self.main(sar, opt)
        if self.sub is not None:
            ls, feats_sub = self.sub(opt)
        else:
            ls, feats_sub = None, []
        return (lc, lf), ls, feats_main, feats_sub


def _g(obj, key, default):
    if obj is None:
        return default
    if hasattr(obj, 'get') and callable(obj.get):
        try:
            return obj.get(key, default)
        except Exception:
            pass
    return getattr(obj, key, default)
