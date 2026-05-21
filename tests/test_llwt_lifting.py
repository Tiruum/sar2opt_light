"""Tests for ``src/models/llwt/lifting.py``.

Goal of this file: prove the wavelet is *invertible* (perfect reconstruction)
for both the freshly-initialised lazy lifting and a learnable lifting whose
predict / update nets have non-zero weights.  PR is the foundational guarantee
the rest of LLW-Former relies on; if these tests are red the decoder will
hallucinate structure that wasn't in the encoder coefficients.
"""
import pytest
import torch

from src.models.llwt.lifting import (
    LLWLevel,
    LLWTransform,
    _merge_cols,
    _merge_rows,
    _split_cols,
    _split_rows,
)
from tests.conftest_llwt import *  # noqa: F401,F403  pull fixtures


# ---------------------------------------------------------------------------
# split / merge primitives
# ---------------------------------------------------------------------------


def test_split_merge_rows_roundtrip(rng_seed):
    x = torch.randn(2, 3, 8, 16)
    e, o = _split_rows(x)
    assert e.shape == (2, 3, 4, 16)
    assert o.shape == (2, 3, 4, 16)
    x_hat = _merge_rows(e, o)
    assert torch.equal(x, x_hat)


def test_split_merge_cols_roundtrip(rng_seed):
    x = torch.randn(2, 3, 8, 16)
    e, o = _split_cols(x)
    assert e.shape == (2, 3, 8, 8)
    assert o.shape == (2, 3, 8, 8)
    x_hat = _merge_cols(e, o)
    assert torch.equal(x, x_hat)


# ---------------------------------------------------------------------------
# LLWLevel single-level PR
# ---------------------------------------------------------------------------


def test_llwlevel_shapes(rng_seed):
    lvl = LLWLevel(c=3, hidden=8)
    x = torch.randn(2, 3, 16, 16)
    sub = lvl(x)
    for k in ('LL', 'LH', 'HL', 'HH'):
        assert sub[k].shape == (2, 3, 8, 8), f"{k} has shape {sub[k].shape}"


def test_llwlevel_pr_at_init(rng_seed):
    """At init (zero P/U) the wavelet is the lazy wavelet -> exact PR."""
    lvl = LLWLevel(c=3, hidden=8).eval()
    x = torch.randn(2, 3, 16, 16, dtype=torch.float64)
    lvl.double()
    sub = lvl(x)
    x_hat = lvl.inverse(sub)
    err = (x - x_hat).abs().max().item()
    assert err < 1e-10, f"lazy-init PR error {err} should be ~ machine zero"


def test_llwlevel_pr_after_random_p_u(rng_seed):
    """PR must hold for *any* P/U values, not only at init."""
    lvl = LLWLevel(c=3, hidden=8).eval().double()
    # Randomise all P/U weights so they are no longer zero.
    for net in (lvl.predict_r, lvl.update_r, lvl.predict_c, lvl.update_c):
        for p in net.parameters():
            torch.nn.init.normal_(p, mean=0.0, std=0.3)
    x = torch.randn(2, 3, 32, 32, dtype=torch.float64)
    sub = lvl(x)
    x_hat = lvl.inverse(sub)
    err = (x - x_hat).abs().max().item()
    assert err < 1e-8, f"learnable-lifting PR error {err}"


def test_llwlevel_pr_fp32(rng_seed):
    """fp32 PR error should still be < 1e-5."""
    lvl = LLWLevel(c=3, hidden=8).eval()
    for net in (lvl.predict_r, lvl.update_r, lvl.predict_c, lvl.update_c):
        for p in net.parameters():
            torch.nn.init.normal_(p, mean=0.0, std=0.2)
    x = torch.randn(2, 3, 32, 32)
    sub = lvl(x)
    x_hat = lvl.inverse(sub)
    err = (x - x_hat).abs().max().item()
    assert err < 1e-5, f"fp32 PR error {err}"


def test_llwlevel_rejects_odd(rng_seed):
    lvl = LLWLevel(c=3, hidden=8)
    with pytest.raises(ValueError):
        lvl(torch.randn(1, 3, 15, 16))


def test_llwlevel_lazy_init_subbands_are_disjoint_pixels(rng_seed):
    """At init the LL sub-band should be exactly the (even-row, even-col)
    sub-sample of the input.  This pins down the lazy wavelet semantics.
    """
    lvl = LLWLevel(c=1, hidden=4).eval().double()
    x = torch.arange(16, dtype=torch.float64).reshape(1, 1, 4, 4)
    sub = lvl(x)
    expected_ll = x[..., 0::2, 0::2]
    assert torch.allclose(sub['LL'], expected_ll)


# ---------------------------------------------------------------------------
# LLWTransform multi-level PR
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("levels,h", [(1, 16), (2, 16), (3, 32), (3, 256)])
def test_llwtransform_pr_at_init(levels, h, rng_seed):
    """Multi-level lazy PR — round-trip must be exact (fp64)."""
    t = LLWTransform(levels=levels, channels=3, hidden=8).eval().double()
    x = torch.randn(2, 3, h, h, dtype=torch.float64)
    coeffs = t(x)
    x_hat = t.inverse(coeffs)
    err = (x - x_hat).abs().max().item()
    assert err < 1e-10, f"L={levels} H={h} PR error {err}"


def test_llwtransform_pr_with_random_pu(rng_seed):
    t = LLWTransform(levels=3, channels=3, hidden=16).eval().double()
    for p in t.parameters():
        torch.nn.init.normal_(p, mean=0.0, std=0.05)
    x = torch.randn(2, 3, 64, 64, dtype=torch.float64)
    coeffs = t(x)
    x_hat = t.inverse(coeffs)
    err = (x - x_hat).abs().max().item()
    assert err < 1e-7, f"3-level learnable PR error {err}"


def test_llwtransform_coeff_keys_present(rng_seed):
    t = LLWTransform(levels=3, channels=3, hidden=8)
    x = torch.randn(1, 3, 32, 32)
    coeffs = t(x)
    expected = {
        f'L{l}_{b}' for l in (1, 2, 3) for b in ('LL', 'LH', 'HL', 'HH')
    }
    assert set(coeffs.keys()) == expected


def test_llwtransform_coeff_shapes_halve_each_level(rng_seed):
    t = LLWTransform(levels=3, channels=3, hidden=8)
    x = torch.randn(1, 3, 64, 64)
    c = t(x)
    assert c['L1_LL'].shape == (1, 3, 32, 32)
    assert c['L2_LL'].shape == (1, 3, 16, 16)
    assert c['L3_LL'].shape == (1, 3,  8,  8)
    assert c['L1_HH'].shape == c['L1_LL'].shape


def test_llwtransform_pu_param_norm_starts_zero(rng_seed):
    """At fresh init every final P/U conv is zero-initialised."""
    t = LLWTransform(levels=3, channels=3, hidden=8)
    norm = t.pu_param_norm().item()
    assert norm == 0.0, f"expected 0 at init, got {norm}"


def test_llwtransform_rejects_wrong_channels():
    t = LLWTransform(levels=2, channels=3, hidden=8)
    with pytest.raises(ValueError):
        t(torch.randn(1, 4, 32, 32))


def test_llwtransform_gradient_flows_through(rng_seed):
    """Optimisation must be able to push gradients back through forward+inverse."""
    t = LLWTransform(levels=2, channels=3, hidden=8)
    x = torch.randn(1, 3, 32, 32, requires_grad=True)
    coeffs = t(x)
    x_hat = t.inverse(coeffs)
    loss = (x_hat - x).abs().mean()
    loss.backward()
    # Every P/U conv weight gradient must exist (None means autograd never
    # touched the parameter — a sign that inverse() forgot to use it).
    for stage in t.stages:
        for net in (stage.predict_r, stage.update_r, stage.predict_c, stage.update_c):
            for p in net.parameters():
                assert p.grad is not None
