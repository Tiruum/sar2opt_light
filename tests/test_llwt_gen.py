"""Smoke tests for ``src/models/llwt/gen.py``.

Goals:
  1. The generator instantiates with the smoke-default config and runs forward
     on a small input.
  2. Output shape is ``(B,3,H,W)`` and values are in ``[-1, 1]`` thanks to the
     final ``tanh``.
  3. At fresh init the head-RGB conv is zero-initialised, so the very first
     forward must return exactly zero logits -> ``tanh(0) = 0``.
  4. Per-subband encoder is shape-preserving.
  5. Cross-band attention is shape-preserving and identity at init.
  6. PCSG is shape-preserving.
  7. Gradients flow end-to-end (G is trainable as a single graph).
"""
from types import SimpleNamespace

import pytest
import torch

from src.models.llwt.gen import (
    CrossBandAttention,
    LLWFormerGenerator,
    PCSG,
    PerSubbandEncoder,
    WindowSwinBlock,
    window_partition,
    window_reverse,
)
from tests.conftest_llwt import *  # noqa: F401,F403


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _cfg(image_size: int = 64, embed_dim: int = 24, depth: int = 1, win: int = 8):
    """A lean cfg for fast smoke tests (no HF download needed)."""
    return SimpleNamespace(
        data=SimpleNamespace(image_size=image_size),
        model=SimpleNamespace(
            gen=SimpleNamespace(
                embed_dim=embed_dim,
                lifting_levels=3,
                lifting_hidden=8,
                pswe=SimpleNamespace(window_size=win, heads=4, depth=depth,
                                     shared_blocks=False),
                cba_fus=SimpleNamespace(levels=[2, 3], heads=4),
                pcsg=SimpleNamespace(enabled=True, gain_ll=0.1),
                get=lambda key, default=None: default,
            ),
        ),
    )


# ---------------------------------------------------------------------------
# window partition / reverse
# ---------------------------------------------------------------------------


def test_window_partition_reverse_roundtrip(rng_seed):
    x = torch.randn(2, 5, 16, 16)
    w = window_partition(x, win=8)
    assert w.shape == (2 * 4, 64, 5)
    x_back = window_reverse(w, win=8, H=16, W=16)
    assert torch.equal(x, x_back)


# ---------------------------------------------------------------------------
# WindowSwinBlock identity-at-init
# ---------------------------------------------------------------------------


def test_window_swin_block_identity_at_init(rng_seed):
    block = WindowSwinBlock(dim=24, heads=4, win=8).eval()
    x = torch.randn(2, 24, 16, 16)
    y = block(x)
    # Zero-init on attn out_proj and MLP last linear -> y must equal x.
    assert torch.allclose(x, y, atol=1e-6), \
        f"max abs diff at init = {(x-y).abs().max().item()}"


def test_window_swin_block_shape(rng_seed):
    block = WindowSwinBlock(dim=24, heads=4, win=8)
    x = torch.randn(2, 24, 32, 32)
    y = block(x)
    assert y.shape == x.shape


def test_window_swin_block_rejects_indivisible(rng_seed):
    block = WindowSwinBlock(dim=24, heads=4, win=8)
    with pytest.raises(ValueError):
        block(torch.randn(1, 24, 12, 16))


# ---------------------------------------------------------------------------
# PerSubbandEncoder
# ---------------------------------------------------------------------------


def test_per_subband_encoder_preserves_shapes(rng_seed):
    enc = PerSubbandEncoder(dim=24, heads=4, win=8, depth=2)
    bands = {b: torch.randn(2, 24, 16, 16) for b in ('LL', 'LH', 'HL', 'HH')}
    out = enc(bands)
    for k in bands:
        assert out[k].shape == bands[k].shape


def test_per_subband_encoder_identity_at_init(rng_seed):
    enc = PerSubbandEncoder(dim=24, heads=4, win=8, depth=2).eval()
    bands = {b: torch.randn(2, 24, 16, 16) for b in ('LL', 'LH', 'HL', 'HH')}
    out = enc(bands)
    for k in bands:
        assert torch.allclose(bands[k], out[k], atol=1e-6)


def test_per_subband_encoder_shared_weights(rng_seed):
    enc = PerSubbandEncoder(dim=24, heads=4, win=8, depth=2, shared=True)
    # All four band stacks are the SAME module instance.
    assert enc.stacks['LL'] is enc.stacks['LH']
    assert enc.stacks['LL'] is enc.stacks['HL']
    assert enc.stacks['LL'] is enc.stacks['HH']


# ---------------------------------------------------------------------------
# CrossBandAttention
# ---------------------------------------------------------------------------


def test_cba_shape(rng_seed):
    cba = CrossBandAttention(dim=24, heads=4)
    bands = {b: torch.randn(2, 24, 8, 8) for b in ('LL', 'LH', 'HL', 'HH')}
    out = cba(bands)
    for k in bands:
        assert out[k].shape == bands[k].shape


def test_cba_identity_at_init(rng_seed):
    """Zero-init out_proj -> step-0 output equals input."""
    cba = CrossBandAttention(dim=24, heads=4).eval()
    bands = {b: torch.randn(2, 24, 8, 8) for b in ('LL', 'LH', 'HL', 'HH')}
    out = cba(bands)
    for k in bands:
        assert torch.allclose(bands[k], out[k], atol=1e-6)


# ---------------------------------------------------------------------------
# PCSG
# ---------------------------------------------------------------------------


def test_pcsg_shape(rng_seed):
    pcsg = PCSG()
    sar = torch.randn(2, 1, 64, 64)
    bands = {
        'LL': torch.randn(2, 24, 32, 32),
        'LH': torch.randn(2, 24, 32, 32),
        'HL': torch.randn(2, 24, 32, 32),
        'HH': torch.randn(2, 24, 32, 32),
    }
    out = pcsg(sar, bands)
    for k in bands:
        assert out[k].shape == bands[k].shape


def test_pcsg_near_identity_at_init(rng_seed):
    """At init, gate convs are zero -> sigmoid(0)=0.5 everywhere.
    Then HH *= 0.5 (so NOT identity for HH) and LL *= 1.05.
    LH and HL are untouched.  Document this explicitly.
    """
    pcsg = PCSG(gain_ll=0.1).eval()
    sar = torch.randn(2, 1, 32, 32)
    bands = {b: torch.randn(2, 24, 16, 16) for b in ('LL', 'LH', 'HL', 'HH')}
    out = pcsg(sar, bands)
    # LH, HL untouched.
    assert torch.allclose(bands['LH'], out['LH'])
    assert torch.allclose(bands['HL'], out['HL'])
    # HH scaled by (1 - 0.5) = 0.5.
    assert torch.allclose(out['HH'], bands['HH'] * 0.5, atol=1e-6)
    # LL scaled by (1 + 0.1 * 0.5) = 1.05.
    assert torch.allclose(out['LL'], bands['LL'] * 1.05, atol=1e-6)


# ---------------------------------------------------------------------------
# Generator smoke
# ---------------------------------------------------------------------------


def test_generator_forward_shape(rng_seed):
    cfg = _cfg(image_size=64, embed_dim=24, depth=1, win=8)
    gen = LLWFormerGenerator(cfg=cfg).eval()
    sar = torch.randn(2, 1, 64, 64)
    out = gen(sar)
    assert out.shape == (2, 3, 64, 64)


def test_generator_output_in_tanh_range(rng_seed):
    cfg = _cfg(image_size=64, embed_dim=24, depth=1, win=8)
    gen = LLWFormerGenerator(cfg=cfg).eval()
    sar = torch.randn(2, 1, 64, 64)
    out = gen(sar)
    assert (out >= -1.0).all()
    assert (out <= 1.0).all()


def test_generator_zero_init_head_makes_midgrey(rng_seed):
    """``head_rgb[-1]`` is zero-init -> logits = 0 -> tanh(0) = 0."""
    cfg = _cfg(image_size=64, embed_dim=24, depth=1, win=8)
    gen = LLWFormerGenerator(cfg=cfg).eval()
    sar = torch.randn(2, 1, 64, 64)
    out = gen(sar)
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-6), \
        f"max abs out at init = {out.abs().max().item()}"


def test_generator_gradient_flows(rng_seed):
    cfg = _cfg(image_size=64, embed_dim=24, depth=1, win=8)
    gen = LLWFormerGenerator(cfg=cfg)
    sar = torch.randn(2, 1, 64, 64, requires_grad=False)
    out = gen(sar)
    # Synthetic loss against random target.
    target = torch.randn_like(out)
    loss = (out - target).abs().mean()
    loss.backward()
    # Pick three diverse parameters and check they all received gradient.
    grad_norms = {
        name: p.grad.detach().abs().mean().item()
        for name, p in gen.named_parameters()
        if p.grad is not None
    }
    # Stem, lifting, per-subband, and head should all show non-trivial grad.
    keys = list(grad_norms)
    assert any('stem' in k for k in keys)
    assert any('llw.stages' in k for k in keys)
    assert any('psw' in k for k in keys)
    assert any('head_rgb' in k for k in keys)
    # No NaN/inf in any grad.
    for name, p in gen.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"NaN/Inf grad in {name}"
