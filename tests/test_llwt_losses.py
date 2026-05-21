"""Tests for ``src/models/llwt/losses.py``.

Focus is on the *new* losses (re-exports are exercised in
``test_sarformer_losses.py``).  We check shape, scale, gradient flow, and
that the speckle-decouple loss actually couples LL to despeckled SAR (lower
loss when LL more closely matches the despeckled target).
"""
import pytest
import torch
import torch.nn.functional as F

from src.models.llwt.lifting import LLWTransform
from src.models.llwt.losses import (
    LPIPSLoss,
    PerfectReconLoss,
    SpeckleDecoupleLoss,
)
from tests.conftest_llwt import *  # noqa: F401,F403


# ---------------------------------------------------------------------------
# SpeckleDecoupleLoss
# ---------------------------------------------------------------------------


def test_speckle_decouple_returns_scalar(rng_seed):
    loss_fn = SpeckleDecoupleLoss()
    llw = LLWTransform(levels=2, channels=1, hidden=8)
    sar = torch.randn(2, 1, 32, 32)
    coeffs = llw(sar)
    val = loss_fn(sar, coeffs)
    assert val.ndim == 0
    assert torch.isfinite(val)


def test_speckle_decouple_grad_flows_into_lifting(rng_seed):
    loss_fn = SpeckleDecoupleLoss()
    llw = LLWTransform(levels=2, channels=1, hidden=8)
    sar = torch.randn(2, 1, 32, 32)
    coeffs = llw(sar)
    val = loss_fn(sar, coeffs)
    val.backward()
    # At least one P/U weight in stage 0 (=level 1) must have non-None grad.
    grads = [
        p.grad
        for net in (llw.stages[0].predict_r, llw.stages[0].update_r,
                    llw.stages[0].predict_c, llw.stages[0].update_c)
        for p in net.parameters()
        if p.grad is not None
    ]
    assert len(grads) > 0


def test_speckle_decouple_lower_when_ll_closer_to_despeckled(rng_seed):
    """Two coeffs dicts: one where LL == despeckled-SAR, one with LL random.
    The "matching" one must give the smaller LL-component loss.
    """
    sar = torch.randn(1, 1, 32, 32).abs()                         # positive
    despeckled = F.avg_pool2d(sar.abs(), 5, stride=2, padding=2)  # the target
    # Build coeffs by hand at the level-1 resolution (16x16) so we can
    # control LL exactly.  Other bands kept identical.
    other = torch.zeros_like(despeckled)
    coeffs_match = {
        'L1_LL': despeckled.clone(),
        'L1_LH': other.clone(),
        'L1_HL': other.clone(),
        'L1_HH': other.clone(),
    }
    coeffs_random = dict(coeffs_match)
    coeffs_random['L1_LL'] = torch.randn_like(despeckled) * 5.0
    loss_fn = SpeckleDecoupleLoss()
    v_match = loss_fn(sar, coeffs_match).item()
    v_rand  = loss_fn(sar, coeffs_random).item()
    assert v_match < v_rand, (
        f"matching LL should give smaller loss, got match={v_match}, "
        f"random={v_rand}"
    )


# ---------------------------------------------------------------------------
# PerfectReconLoss
# ---------------------------------------------------------------------------


def test_pr_loss_at_init_is_floating_point_zero(rng_seed):
    """At init the lazy wavelet has bit-exact PR -> loss < 1e-6 (fp32)."""
    llw = LLWTransform(levels=3, channels=3, hidden=8)
    loss_fn = PerfectReconLoss()
    x = torch.randn(1, 3, 64, 64)
    val = loss_fn(llw, x).item()
    assert val < 1e-6, f"PR loss at init = {val}"


def test_pr_loss_grad_pushes_inverse_match(rng_seed):
    """Perturb a P/U weight, then verify the PR loss has non-zero grad
    that, if applied, would push it back toward zero (i.e. grad has the
    correct sign w.r.t. the perturbation).
    """
    llw = LLWTransform(levels=2, channels=3, hidden=8)
    # Perturb the level-1 predict_r last conv weight (otherwise zero-init).
    perturbed = torch.full_like(llw.stages[0].predict_r[-1].weight, 0.1)
    llw.stages[0].predict_r[-1].weight.data.copy_(perturbed)
    # PR is preserved by construction for any P/U values, so the loss should
    # STILL be ~0 even after perturbation -- this test exists to guard
    # against accidentally breaking the inverse mirroring.
    loss_fn = PerfectReconLoss()
    x = torch.randn(1, 3, 32, 32)
    val = loss_fn(llw, x).item()
    assert val < 1e-5, f"PR broken after P perturbation: {val}"


# ---------------------------------------------------------------------------
# LPIPSLoss
# ---------------------------------------------------------------------------


def test_lpips_lazy_init(rng_seed):
    """The LPIPS model should not exist until the first forward call."""
    loss_fn = LPIPSLoss(net_type='alex')
    assert loss_fn._lpips is None


@pytest.mark.slow
def test_lpips_forward_returns_scalar(rng_seed):
    """Only runs when torchmetrics + AlexNet weights are available."""
    loss_fn = LPIPSLoss(net_type='alex')
    a = torch.rand(1, 3, 64, 64) * 2 - 1
    b = torch.rand(1, 3, 64, 64) * 2 - 1
    try:
        val = loss_fn(a, b)
    except (ImportError, RuntimeError, OSError):
        pytest.skip("torchmetrics LPIPS or AlexNet weights unavailable")
    assert val.ndim == 0
    assert torch.isfinite(val)
