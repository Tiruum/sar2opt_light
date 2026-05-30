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


__all__ = ["LLWFormerDiscriminator", "MainDis", "SubbandDis", "FourierDis", "FixedHaarDWT"]


def _sn(ci: int, co: int, k: int, s: int, p: int) -> nn.Conv2d:
    return spectral_norm(nn.Conv2d(ci, co, kernel_size=k, stride=s, padding=p, bias=True))


def _conv(ci: int, co: int, k: int, s: int, p: int, use_sn: bool) -> nn.Conv2d:
    """Conv2d with optional spectral_norm wrapper.

    Main D drops SN (use_sn=False) — SN + heavy FM (fm_main_w=10) collapses
    the adversarial dynamic: FM target IS the D feature pool, so SN-bounded
    features are too easy for G to match via FM, leaving D no signal to
    distinguish real from fake.  pix2pixHD's original recipe (multi-D + FM)
    omits SN for the same reason; subband-D keeps SN since its target
    distribution (fixed-Haar of opt) needs the extra regularisation.
    """
    conv = nn.Conv2d(ci, co, kernel_size=k, stride=s, padding=p, bias=True)
    return spectral_norm(conv) if use_sn else conv


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
    """5-layer 70x70 RF PatchGAN; spectral-norm gated by ``use_sn``."""
    def __init__(self, in_ch: int, ndf: int = 64, use_sn: bool = True):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(_conv(in_ch,    ndf,     4, 2, 1, use_sn), nn.LeakyReLU(0.2, inplace=True)),
            nn.Sequential(_conv(ndf,      ndf * 2, 4, 2, 1, use_sn), nn.LeakyReLU(0.2, inplace=True)),
            nn.Sequential(_conv(ndf * 2,  ndf * 4, 4, 2, 1, use_sn), nn.LeakyReLU(0.2, inplace=True)),
            nn.Sequential(_conv(ndf * 4,  ndf * 8, 4, 1, 1, use_sn), nn.LeakyReLU(0.2, inplace=True)),
            _conv(ndf * 8, 1, 4, 1, 1, use_sn),
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        feats: List[torch.Tensor] = []
        for layer in self.layers[:-1]:
            x = layer(x)
            feats.append(x)
        return self.layers[-1](x), feats


class _FinePatchBranch(nn.Module):
    """3-layer ~46x46 RF PatchGAN; spectral-norm gated by ``use_sn``."""
    def __init__(self, in_ch: int, ndf: int = 64, use_sn: bool = True):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(_conv(in_ch,    ndf,     4, 2, 1, use_sn), nn.LeakyReLU(0.2, inplace=True)),
            nn.Sequential(_conv(ndf,      ndf * 2, 4, 2, 1, use_sn), nn.LeakyReLU(0.2, inplace=True)),
            nn.Sequential(_conv(ndf * 2,  ndf * 4, 4, 2, 1, use_sn), nn.LeakyReLU(0.2, inplace=True)),
            _conv(ndf * 4, 1, 4, 1, 1, use_sn),
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

    ``use_sn=False`` (default) drops spectral_norm — the pix2pixHD recipe.
    See ``_conv`` docstring for why SN + heavy FM collapses adversarial
    dynamics in this setup.
    """
    def __init__(self, in_ch: int = 4, ndf: int = 64, use_sn: bool = False):
        super().__init__()
        self.coarse = _CoarsePatchBranch(in_ch, ndf, use_sn=use_sn)
        self.fine   = _FinePatchBranch(in_ch, ndf, use_sn=use_sn)

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
# Fourier-D — unconditional PatchGAN on log|rfft2(opt)|
# ---------------------------------------------------------------------------


class _FourierPatchBranch(nn.Module):
    """4-layer spectral-norm PatchGAN on log-magnitude spectrum feature maps.

    Mirrors :class:`_SubbandPatchBranch` depth and width; the only difference
    is the channel count of the input (3 RGB log-magnitude maps instead of
    12 Haar subbands).
    """
    def __init__(self, in_ch: int = 3, ndf: int = 32, use_sn: bool = True):
        super().__init__()
        conv = _sn if use_sn else (lambda ci, co, k, s, p: nn.Conv2d(ci, co, k, s, p, bias=True))
        self.layers = nn.ModuleList([
            nn.Sequential(conv(in_ch,    ndf,     4, 2, 1), nn.LeakyReLU(0.2, inplace=True)),
            nn.Sequential(conv(ndf,      ndf * 2, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True)),
            nn.Sequential(conv(ndf * 2,  ndf * 4, 4, 1, 1), nn.LeakyReLU(0.2, inplace=True)),
            conv(ndf * 4, 1, 4, 1, 1),
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        feats: List[torch.Tensor] = []
        for layer in self.layers[:-1]:
            x = layer(x)
            feats.append(x)
        return self.layers[-1](x), feats


class FourierDis(nn.Module):
    """Unconditional PatchGAN on log-magnitude rfft2 of the optical pair.

    Forward: ``opt (B, 3, H, W) -> ((B, 1, h, w), feats)``.

    Rationale:
      * ``torch.fft.rfft2`` produces a complex tensor of shape
        ``(B, 3, H, W//2+1)``.  Taking ``log(|F| + eps)`` compresses the
        ~1/f power-law dynamic range so the first conv isn't dominated by
        the DC bin.
      * Unconditional (opt-only) — symmetric with :class:`SubbandDis`.
        SAR spectrum is speckle-dominated; conditional pairing adds noise
        without semantic value.
      * 4-conv depth + spectral_norm + ndf=32 — mirrors :class:`SubbandDis`.
        Keeps param + VRAM cost low for a third head added to an already
        bs=8-at-24GB training setup.

    Outputs feats list with the layer-0 activation *kept* (subband-D
    matches this), giving 3 FM features per forward pass.
    """
    _EPS = 1e-8

    def __init__(self, ndf: int = 32, use_sn: bool = True):
        super().__init__()
        self.branch = _FourierPatchBranch(in_ch=3, ndf=ndf, use_sn=use_sn)

    def _log_mag(self, x: torch.Tensor) -> torch.Tensor:
        # FFT under bf16 autocast: rfft2 is autocast-aware in torch>=2.0.
        # Force fp32 input so we get complex64 (not the under-supported
        # complex32) and keep magnitude/log numerics tight.
        F_x = torch.fft.rfft2(x.float(), dim=(-2, -1))
        mag = F_x.abs()
        return torch.log(mag + self._EPS)

    def forward(self, opt: torch.Tensor):
        spec = self._log_mag(opt)                            # (B, 3, H, W//2+1)
        return self.branch(spec)


# ---------------------------------------------------------------------------
# Wrapper: main + subband + fourier
# ---------------------------------------------------------------------------


class LLWFormerDiscriminator(nn.Module):
    """Wraps ``MainDis``, ``SubbandDis``, and ``FourierDis`` with per-head enable flags.

    Forward returns a 6-tuple:
        ``((logits_coarse, logits_fine), logits_subband, logits_fourier,
          feats_main, feats_sub, feats_fourier)``
    where ``logits_subband`` is ``None`` and ``feats_sub`` is ``[]`` when
    the subband head is disabled (``cfg.model.dis.subband.enabled = false``).
    Same convention for the fourier head
    (``cfg.model.dis.fourier.enabled = false``, default).

    Keeping ``feats_main``, ``feats_sub``, ``feats_fourier`` as separate
    lists lets the training loop weight each head's feature-matching
    contribution independently via ``loss.fm_main_weight``,
    ``loss.fm_sub_weight``, and ``loss.fm_fourier_weight``.  Concatenating
    them would force a single weight across the whole pool and silently
    drop the per-head weights.
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
        fourier_cfg     = _g(dcfg, 'fourier', None)
        self.use_fourier = bool(_g(fourier_cfg, 'enabled', False))
        ndf_fourier     = int(_g(fourier_cfg, 'ndf', 32))
        use_sn_fourier  = bool(_g(fourier_cfg, 'use_sn', True))

        self.main = MainDis(in_ch=in_ch, ndf=ndf_main)
        self.sub  = SubbandDis(ndf=ndf_sub) if self.use_sub else None
        self.fourier = (
            FourierDis(ndf=ndf_fourier, use_sn=use_sn_fourier)
            if self.use_fourier else None
        )

    def forward(self, sar: torch.Tensor, opt: torch.Tensor):
        """Returns 6-tuple: see class docstring for layout."""
        (lc, lf), feats_main = self.main(sar, opt)
        if self.sub is not None:
            ls, feats_sub = self.sub(opt)
        else:
            ls, feats_sub = None, []
        if self.fourier is not None:
            lfourier, feats_fourier = self.fourier(opt)
        else:
            lfourier, feats_fourier = None, []
        return (lc, lf), ls, lfourier, feats_main, feats_sub, feats_fourier


def _g(obj, key, default):
    if obj is None:
        return default
    if hasattr(obj, 'get') and callable(obj.get):
        try:
            return obj.get(key, default)
        except Exception:
            pass
    return getattr(obj, key, default)


# ---------------------------------------------------------------------------
# Standalone smokes (run with: python -m src.models.llwt.dis)
# ---------------------------------------------------------------------------


def _smoke_fourier_dis() -> None:
    """Tier 1 unit smoke for FourierDis."""
    print("[fourier-dis smoke] running FourierDis shape + sanity checks")
    torch.manual_seed(0)
    d = FourierDis(ndf=32, use_sn=True)
    for B, H, W in [(2, 256, 256), (1, 128, 128)]:
        opt = torch.randn(B, 3, H, W)
        logits, feats = d(opt)
        assert torch.isfinite(logits).all(), f"non-finite logits at {(B, 3, H, W)}"
        for i, f in enumerate(feats):
            assert torch.isfinite(f).all(), f"non-finite feat[{i}] at {(B, 3, H, W)}"
        assert len(feats) == 3, f"expected 3 FM feats, got {len(feats)}"
        print(f"  [OK] shape=({B},3,{H},{W}) -> logits {tuple(logits.shape)} + {len(feats)} feats")

    # Edge: saturated (all-ones) input — log(|F|) must not NaN.
    opt_sat = torch.ones(2, 3, 64, 64)
    logits, _ = d(opt_sat)
    assert torch.isfinite(logits).all(), "non-finite logits on saturated input"

    # Edge: zero input — log(0+eps) finite.
    opt_zero = torch.zeros(2, 3, 64, 64)
    logits, _ = d(opt_zero)
    assert torch.isfinite(logits).all(), "non-finite logits on zero input"

    print("[fourier-dis smoke] PASS - shapes, FM feat count, NaN guards all clean")


def _smoke_wrapper() -> None:
    """Tier 1 wrapper smoke - verify 6-tuple return with fourier on/off."""
    from omegaconf import OmegaConf
    print("[wrapper smoke] running LLWFormerDiscriminator with fourier off + on")

    # OFF path (default)
    cfg_off = OmegaConf.create({
        'model': {'dis': {
            'main':    {'in_channels': 4, 'ndf': 64},
            'subband': {'enabled': True, 'ndf': 32},
            'fourier': {'enabled': False, 'ndf': 32, 'use_sn': True},
        }},
    })
    d_off = LLWFormerDiscriminator(cfg=cfg_off)
    sar = torch.randn(2, 1, 64, 64)
    opt = torch.randn(2, 3, 64, 64)
    main_pair, sub_l, fourier_l, feats_m, feats_s, feats_f = d_off(sar, opt)
    assert isinstance(main_pair, tuple) and len(main_pair) == 2, "main pair"
    assert sub_l is not None, "subband enabled"
    assert fourier_l is None, "fourier disabled should be None"
    assert feats_f == [], "fourier feats disabled should be empty"
    print(f"  [OK] fourier OFF: 6-tuple unpacks; fourier_l=None feats_f=[]")

    # ON path
    cfg_on = OmegaConf.create({
        'model': {'dis': {
            'main':    {'in_channels': 4, 'ndf': 64},
            'subband': {'enabled': True, 'ndf': 32},
            'fourier': {'enabled': True, 'ndf': 32, 'use_sn': True},
        }},
    })
    d_on = LLWFormerDiscriminator(cfg=cfg_on)
    main_pair, sub_l, fourier_l, feats_m, feats_s, feats_f = d_on(sar, opt)
    assert fourier_l is not None, "fourier enabled should produce logits"
    assert len(feats_f) == 3, f"fourier feats should be 3, got {len(feats_f)}"
    print(f"  [OK] fourier ON: logits shape {tuple(fourier_l.shape)} + {len(feats_f)} FM feats")

    print("[wrapper smoke] PASS - 6-tuple contract holds on both gates")


if __name__ == '__main__':
    _smoke_fourier_dis()
    _smoke_wrapper()
