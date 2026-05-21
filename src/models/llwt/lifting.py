"""Learnable Lifting Wavelet (LLW): a wavelet-native PyTorch module.

A *lifting scheme* (Sweldens, 1996) factorises any FIR wavelet into alternating
*predict* and *update* steps:

    forward:   d = o - P(e),       s = e + U(d)
    inverse:   e = s - U(d),       o = d + P(e)

where ``e`` and ``o`` are the even / odd sub-samples of the signal.  Because the
inverse re-uses the same ``P`` / ``U`` (just in reversed order), **perfect
reconstruction (PR) is guaranteed by construction for arbitrary, learnable
``P`` and ``U``**.  No orthogonality constraint is needed.

This is the load-bearing primitive for LLW-Former (see design doc):
- forward (encoder side):  ``x -> {LL, LH, HL, HH}`` via two 1-D passes
- inverse (decoder side):  ``{LL, LH, HL, HH} -> x`` via the same modules

At initialisation, ``P`` and ``U`` output zero (zero-init on their final conv).
That makes the wavelet equal to the *lazy* wavelet (pure pixel-unshuffle); it
still has PR.  Training learns ``P``/``U`` away from zero towards problem-
appropriate filters (Haar, db2, or something dataset-specific).
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["LLWLevel", "LLWTransform"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _split_rows(x: torch.Tensor):
    """Split ``(B,C,H,W)`` into (even_rows, odd_rows), each ``(B,C,H/2,W)``."""
    return x[..., 0::2, :], x[..., 1::2, :]


def _split_cols(x: torch.Tensor):
    """Split ``(B,C,H,W)`` into (even_cols, odd_cols), each ``(B,C,H,W/2)``."""
    return x[..., :, 0::2], x[..., :, 1::2]


def _merge_rows(e: torch.Tensor, o: torch.Tensor) -> torch.Tensor:
    """Inverse of ``_split_rows`` — interleave even and odd rows back.

    Uses ``torch.stack(..., dim=3).reshape(...)`` so the merge happens as a
    single contiguous copy + view instead of an ``empty`` allocation followed
    by two strided assignment-copies.  Cuts ~30% of ``aten::copy_`` calls in
    the lifting forward + inverse hot path.  Math identical.
    """
    B, C, H2, W = e.shape
    return torch.stack((e, o), dim=3).reshape(B, C, H2 * 2, W)


def _merge_cols(e: torch.Tensor, o: torch.Tensor) -> torch.Tensor:
    """Inverse of ``_split_cols`` — interleave even and odd cols back.

    See ``_merge_rows`` for the kernel-launch rationale.  Math identical to
    the prior assignment-based implementation.
    """
    B, C, H, W2 = e.shape
    return torch.stack((e, o), dim=4).reshape(B, C, H, W2 * 2)


# ---------------------------------------------------------------------------
# LLWLevel
# ---------------------------------------------------------------------------


class _PUNet(nn.Sequential):
    """Predict / Update sub-net.

    Two 3x3 convs with reflect padding and a GELU; the *final* conv is
    zero-initialised so the net outputs zero at step 0 -> lifting reduces to the
    lazy wavelet (pixel-unshuffle).  Perfect reconstruction holds for any
    parameter values; the zero-init only sets the *starting point* of training.
    """
    def __init__(self, c: int, hidden: int):
        super().__init__(
            nn.Conv2d(c, hidden, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.GELU(),
            nn.Conv2d(hidden, c, kernel_size=3, padding=1, padding_mode='reflect'),
        )
        # Zero-init the final conv -> step-0 output is exactly zero.
        nn.init.zeros_(self[-1].weight)
        nn.init.zeros_(self[-1].bias)


class LLWLevel(nn.Module):
    """One level of separable 2-D learnable lifting.

    The level is *channel-preserving*: ``forward(x: (B,C,H,W))`` returns a dict
    of four subbands each shaped ``(B, C, H/2, W/2)``.  Channel changes are done
    *outside* the lifting (by the per-subband Swin encoder), so the lifting
    math stays exactly invertible.

    H and W must both be even.

    Parameters
    ----------
    c : int
        Number of channels (same on input and on every subband).
    hidden : int
        Hidden width of the predict / update sub-nets.

    Notes
    -----
    Two distinct ``P``/``U`` pairs are used: one for the row pass, one for the
    column pass.  Each pass operates on tensors of shape ``(B,C,H/2,W)`` (row
    pass) or ``(B,C,H',W/2)`` (column pass); inside the sub-net the convs see
    a half-resolution map.  Reflect padding avoids boundary artefacts.
    """
    def __init__(self, c: int, hidden: int = 32):
        super().__init__()
        self.c = int(c)
        self.predict_r = _PUNet(c, hidden)
        self.update_r  = _PUNet(c, hidden)
        self.predict_c = _PUNet(c, hidden)
        self.update_c  = _PUNet(c, hidden)

    # ---- forward ----------------------------------------------------------

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """``(B,C,H,W) -> {LL,LH,HL,HH} each (B,C,H/2,W/2)``."""
        if x.shape[-1] % 2 != 0 or x.shape[-2] % 2 != 0:
            raise ValueError(f"LLWLevel needs even H,W; got {tuple(x.shape)}")

        # ---- row pass: (B,C,H,W) -> (s_r, d_r) each (B,C,H/2,W) -----------
        e_r, o_r = _split_rows(x)
        d_r = o_r - self.predict_r(e_r)
        s_r = e_r + self.update_r(d_r)

        # ---- column pass on s_r -> (LL, LH) -------------------------------
        ll_e, ll_o = _split_cols(s_r)
        lh = ll_o - self.predict_c(ll_e)
        ll = ll_e + self.update_c(lh)

        # ---- column pass on d_r -> (HL, HH) -------------------------------
        hl_e, hl_o = _split_cols(d_r)
        hh = hl_o - self.predict_c(hl_e)
        hl = hl_e + self.update_c(hh)

        return {'LL': ll, 'LH': lh, 'HL': hl, 'HH': hh}

    # ---- inverse ----------------------------------------------------------

    def inverse(self, sub: Dict[str, torch.Tensor]) -> torch.Tensor:
        """``{LL,LH,HL,HH} -> (B,C,H,W)``.  Exact inverse of ``forward``."""
        ll, lh, hl, hh = sub['LL'], sub['LH'], sub['HL'], sub['HH']

        # ---- column inverse on (LL,LH) -> s_r -----------------------------
        ll_e = ll - self.update_c(lh)
        ll_o = lh + self.predict_c(ll_e)
        s_r  = _merge_cols(ll_e, ll_o)

        # ---- column inverse on (HL,HH) -> d_r -----------------------------
        hl_e = hl - self.update_c(hh)
        hl_o = hh + self.predict_c(hl_e)
        d_r  = _merge_cols(hl_e, hl_o)

        # ---- row inverse on (s_r, d_r) -> x -------------------------------
        e_r = s_r - self.update_r(d_r)
        o_r = d_r + self.predict_r(e_r)
        return _merge_rows(e_r, o_r)


# ---------------------------------------------------------------------------
# LLWTransform — multi-level
# ---------------------------------------------------------------------------


class LLWTransform(nn.Module):
    """L-level learnable lifting wavelet transform.

    ``forward(x) -> coeffs`` where ``coeffs`` is a dict keyed
    ``"L{l}_{band}"`` for ``l in [1..L]`` and ``band in {LL,LH,HL,HH}``.
    Only ``L{L}_LL`` is the "approximation" passed to deeper levels; the
    intermediate ``L{l<L}_LL`` is discarded by the standard wavelet pyramid
    but is *kept* in the dict for downstream consumers (e.g. the speckle-
    decouple loss needs level-1 LL/HH; the per-subband Swin needs every band
    at every level).

    ``inverse(coeffs)`` rebuilds the original input.  Round-trip error is
    floating-point only (< 1e-5 for fp32, < 1e-2 for bf16).
    """
    def __init__(self, levels: int = 3, channels: int = 3, hidden: int = 32):
        super().__init__()
        if levels < 1:
            raise ValueError(f"levels must be >= 1, got {levels}")
        self.levels = int(levels)
        self.channels = int(channels)
        self.stages = nn.ModuleList([
            LLWLevel(c=channels, hidden=hidden) for _ in range(levels)
        ])

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if x.shape[1] != self.channels:
            raise ValueError(
                f"LLWTransform({self.channels}ch) got x with {x.shape[1]}ch"
            )
        coeffs: Dict[str, torch.Tensor] = {}
        cur = x
        for l, stage in enumerate(self.stages, start=1):
            sub = stage(cur)
            coeffs[f'L{l}_LL'] = sub['LL']
            coeffs[f'L{l}_LH'] = sub['LH']
            coeffs[f'L{l}_HL'] = sub['HL']
            coeffs[f'L{l}_HH'] = sub['HH']
            cur = sub['LL']                     # next level operates on LL
        return coeffs

    def inverse(self, coeffs: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Reconstruct from deepest level upward.  At each level we replace the
        # current ``cur`` (the just-reconstructed coarser image) into the LL
        # slot of that level and apply the inverse.
        cur = coeffs[f'L{self.levels}_LL']
        for l in range(self.levels, 0, -1):
            sub = {
                'LL': cur,
                'LH': coeffs[f'L{l}_LH'],
                'HL': coeffs[f'L{l}_HL'],
                'HH': coeffs[f'L{l}_HH'],
            }
            cur = self.stages[l - 1].inverse(sub)
        return cur

    # ---- diagnostics -----------------------------------------------------

    def pu_param_norm(self) -> torch.Tensor:
        """Sum of L2 norms of all P/U final-conv weights.

        Useful in TB as a "is the wavelet actually learning?" probe.  Stays at
        exactly 0 if the lifting collapsed to lazy.
        """
        total = 0.0
        for stage in self.stages:
            for net in (stage.predict_r, stage.update_r, stage.predict_c, stage.update_c):
                total = total + net[-1].weight.detach().float().pow(2).sum()
        return total.sqrt()
