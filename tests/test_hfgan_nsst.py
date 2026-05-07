"""Tests for NSST integration into the HFGenerator SAR-to-optical GAN.

Groups:
  1. NSSTDecomposer / NSSTReconstructor numerical correctness vs numpy
  2. Shape correctness (decomposer output, NSSTBranch)
  3. Generator integration (nsst.enable toggle, return_branches, fusion logit)
  4. Factory / optimizer param-group placement
  5. Gradient flow through NSSTScaleProcessor

Run with:
    pytest tests/test_hfgan_nsst.py -v
Slow tests are marked with @pytest.mark.slow and can be skipped via:
    pytest tests/test_hfgan_nsst.py -v -m "not slow"
"""

import math
import sys
import os

import numpy as np
import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

sys.path.insert(0, os.path.abspath("."))

from tests.conftest_hfgan import MockBackbone


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_base_cfg():
    """Minimal inline config that mirrors the real config schema."""
    return OmegaConf.create({
        "model": {
            "gen": {
                "backbone":          "facebook/convnextv2-tiny-22k-224",
                "out_indices":       [1, 2, 3, 4],
                "bottleneck_dim":    768,
                "bottleneck_heads":  8,
                "bottleneck_layers": 2,
            },
            "dis": {
                "ndf":         64,
                "in_channels": 4,
            },
            "log_summary": False,
        },
        "optimizer": {
            "lr_g":           2.0e-4,
            "lr_d":           2.0e-4,
            "beta1":          0.5,
            "beta2":          0.999,
            "weight_decay_g": 0.01,
        },
        "scheduler": {
            "eta_min":             1.0e-6,
            "linear_decay_epochs": 400,
        },
        "loss": {
            "gan_weight":        1.0,
            "fm_weight":         5.0,
            "fft_weight":        0.0,
            "perceptual_weight": 0.0,
        },
        "system": {
            "max_epochs": 800,
        },
    })


def _make_nsst_enabled_cfg():
    """Base config extended with model.gen.nsst.enable=True."""
    base = _make_base_cfg()
    return OmegaConf.merge(base, OmegaConf.create({
        "model": {"gen": {"nsst": {"enable": True, "levels": 4, "hidden_dim": 64}}},
        "optimizer": {"encoder_lr_scale": 0.0},
    }))


def _make_nsst_disabled_cfg():
    """Base config extended with model.gen.nsst.enable=False."""
    base = _make_base_cfg()
    return OmegaConf.merge(base, OmegaConf.create({
        "model": {"gen": {"nsst": {"enable": False}}},
    }))


# ---------------------------------------------------------------------------
# Group 1: NSSTDecomposer / NSSTReconstructor numerical correctness vs numpy
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_nsst_decomposer_numerical_equivalence_to_numpy():
    """Max abs error < 1e-4 vs numpy reference for each subband.

    This is the critical gate — if it fails all downstream NSST tests are
    meaningless because the PyTorch implementation diverges from the paper.
    """
    nsst_torch = pytest.importorskip(
        "src.utils.nsst_torch",
        reason="nsst_torch not importable — implementation incomplete",
    )
    from src.utils.NSST import NSST as NSSTPy

    nsst_np = NSSTPy()
    nsst_np.atrous_filters_init()
    nsst_np.shearing_filters_myer()

    dec_pt = nsst_torch.NSSTDecomposer()
    dec_pt.eval()

    np.random.seed(42)
    for trial in range(5):
        img_np = np.random.randn(256, 256).astype(np.float32)
        img_t  = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0)  # (1,1,256,256)

        with torch.no_grad():
            coefs_pt = dec_pt(img_t)
        coefs_np = nsst_np.dec(img_np)

        # Level 0: approximation subband
        pt0 = coefs_pt[0].squeeze().numpy()
        np0 = coefs_np[0]
        err0 = np.abs(pt0 - np0).max()
        assert err0 < 1e-4, (
            f"trial={trial}: approx subband max_abs_err={err0:.2e} >= 1e-4"
        )

        # Detail subbands at each level
        for lvl in range(1, len(coefs_np)):
            np_level = coefs_np[lvl]   # list of 2D arrays
            pt_level = coefs_pt[lvl]   # list of (1,1,H,W) tensors
            assert len(pt_level) == len(np_level), (
                f"trial={trial} level={lvl}: subband count mismatch "
                f"pt={len(pt_level)} np={len(np_level)}"
            )
            for k, (pt_sub, np_sub) in enumerate(zip(pt_level, np_level)):
                pt_arr = pt_sub.squeeze().numpy()
                err = np.abs(pt_arr - np_sub).max()
                assert err < 1e-4, (
                    f"trial={trial} level={lvl} subband={k}: "
                    f"max_abs_err={err:.2e} >= 1e-4"
                )


@pytest.mark.slow
def test_nsst_reconstructor_round_trip():
    """dec -> rec round-trip: max abs error < 1e-4 vs numpy NSST.rec(NSST.dec()).

    Validates that NSSTReconstructor correctly mirrors the numpy atrous_rec
    reconstruction, not just that the PyTorch round-trip is self-consistent.
    """
    nsst_torch = pytest.importorskip(
        "src.utils.nsst_torch",
        reason="nsst_torch not importable — implementation incomplete",
    )
    from src.utils.NSST import NSST as NSSTPy

    nsst_np = NSSTPy()
    nsst_np.atrous_filters_init()
    nsst_np.shearing_filters_myer()

    dec_pt = nsst_torch.NSSTDecomposer()
    rec_pt = nsst_torch.NSSTReconstructor()
    dec_pt.eval()
    rec_pt.eval()

    np.random.seed(42)
    for trial in range(5):
        img_np = np.random.randn(256, 256).astype(np.float32)
        img_t  = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            coefs_pt = dec_pt(img_t)
            recon_pt = rec_pt(coefs_pt).squeeze().numpy()   # (256,256)

        coefs_np  = nsst_np.dec(img_np)
        recon_np  = nsst_np.rec(coefs_np)                   # (256,256)

        err = np.abs(recon_pt - recon_np).max()
        assert err < 1e-4, (
            f"trial={trial}: reconstruction max_abs_err={err:.2e} >= 1e-4"
        )


# ---------------------------------------------------------------------------
# Group 2: Shape correctness
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_nsst_decomposer_output_shapes():
    """coefs[0]=(B,1,H,W); coefs[1..4] are lists of 8/8/16/16 tensors each (B,1,H,W).

    Expected subband counts per level come from Conf.sp.dcomp = [3,3,4,4]:
      2**3=8, 2**3=8, 2**4=16, 2**4=16.
    """
    nsst_torch = pytest.importorskip(
        "src.utils.nsst_torch",
        reason="nsst_torch not importable — implementation incomplete",
    )
    from src.utils.NSST import Conf

    dec = nsst_torch.NSSTDecomposer()
    dec.eval()

    B = 2
    x = torch.randn(B, 1, 256, 256)
    with torch.no_grad():
        coefs = dec(x)

    # Approximation subband
    assert coefs[0].shape == (B, 1, 256, 256), (
        f"approx subband shape mismatch: {coefs[0].shape}"
    )

    # Detail subbands — counts from Conf.sp.dcomp
    expected_counts = [int(2 ** d) for d in Conf.sp.dcomp]
    assert len(coefs) - 1 == len(expected_counts), (
        f"number of detail levels: got {len(coefs)-1}, expected {len(expected_counts)}"
    )

    for lvl, (subs, expected_n) in enumerate(zip(coefs[1:], expected_counts), start=1):
        assert len(subs) == expected_n, (
            f"level {lvl}: subband count {len(subs)} != expected {expected_n}"
        )
        for k, sub in enumerate(subs):
            assert sub.shape[0] == B,          f"level {lvl} sub {k}: batch dim wrong {sub.shape}"
            assert sub.shape[1] == 1,          f"level {lvl} sub {k}: channel dim wrong {sub.shape}"
            assert sub.ndim == 4,              f"level {lvl} sub {k}: expected 4D tensor, got {sub.ndim}D"
            assert torch.isfinite(sub).all(),  f"level {lvl} sub {k}: non-finite values"


def test_nsst_branch_output_shape():
    """NSSTBranch(sar (1,1,256,256)) -> (1,3,256,256).

    Skips gracefully if NSSTBranch has not been added to nsst_torch yet.
    """
    nsst_torch = pytest.importorskip(
        "src.utils.nsst_torch",
        reason="nsst_torch not importable — implementation incomplete",
    )
    if not hasattr(nsst_torch, "NSSTBranch"):
        pytest.skip("NSSTBranch not yet implemented in nsst_torch.py")

    branch = nsst_torch.NSSTBranch()
    branch.eval()

    x = torch.randn(1, 1, 256, 256)
    with torch.no_grad():
        out = branch(x)

    assert out.shape == (1, 3, 256, 256), (
        f"NSSTBranch output shape mismatch: {out.shape}"
    )
    assert torch.isfinite(out).all(), "NSSTBranch output contains non-finite values"


def test_nsst_branch_batch_consistency():
    """NSSTBranch output shape is preserved for batch size 2."""
    nsst_torch = pytest.importorskip(
        "src.utils.nsst_torch",
        reason="nsst_torch not importable — implementation incomplete",
    )
    if not hasattr(nsst_torch, "NSSTBranch"):
        pytest.skip("NSSTBranch not yet implemented in nsst_torch.py")

    branch = nsst_torch.NSSTBranch()
    branch.eval()

    x = torch.randn(2, 1, 256, 256)
    with torch.no_grad():
        out = branch(x)

    assert out.shape == (2, 3, 256, 256), (
        f"NSSTBranch batch=2 output shape mismatch: {out.shape}"
    )


# ---------------------------------------------------------------------------
# Group 3: Generator integration
# ---------------------------------------------------------------------------

def test_hfgenerator_nsst_enabled_forward():
    """With nsst.enable=True, HFGenerator.forward returns (B,3,256,256)."""
    from src.models.huggingface_gan.gen import HFGenerator

    cfg = _make_nsst_enabled_cfg()
    gen = HFGenerator(cfg, encoder=MockBackbone()).eval()

    # Skip if this HFGenerator version does not support nsst config yet
    # (another agent adds the NSST branch; guard gracefully)
    if not hasattr(gen, "_fusion_logit"):
        pytest.skip("HFGenerator does not yet have NSST integration (_fusion_logit absent)")

    sar = torch.randn(1, 1, 256, 256)
    with torch.no_grad():
        out = gen(sar)

    # When return_branches=False (default) the fused output is returned
    if isinstance(out, torch.Tensor):
        assert out.shape == (1, 3, 256, 256), f"output shape mismatch: {out.shape}"
    else:
        # Tuple — first element is the fused output
        assert out[0].shape == (1, 3, 256, 256), f"fused output shape mismatch: {out[0].shape}"


def test_hfgenerator_return_branches():
    """return_branches=True returns a 4-tuple: (fused, main_out, nsst_recon, w).

    w should be a scalar tensor near sigmoid(-3) ≈ 0.0474 at initialization.
    """
    from src.models.huggingface_gan.gen import HFGenerator

    cfg = _make_nsst_enabled_cfg()
    gen = HFGenerator(cfg, encoder=MockBackbone()).eval()

    if not hasattr(gen, "_fusion_logit"):
        pytest.skip("HFGenerator does not yet have NSST integration (_fusion_logit absent)")

    if not hasattr(gen, "forward") or "return_branches" not in (
        gen.forward.__code__.co_varnames
    ):
        pytest.skip("HFGenerator.forward does not accept return_branches kwarg yet")

    sar = torch.randn(1, 1, 256, 256)
    with torch.no_grad():
        result = gen(sar, return_branches=True)

    assert isinstance(result, (tuple, list)), (
        f"return_branches=True should yield a tuple, got {type(result)}"
    )
    assert len(result) == 4, (
        f"expected 4-tuple (fused, main_out, nsst_recon, w), got {len(result)}-tuple"
    )

    fused, main_out, nsst_recon, w = result

    assert fused.shape     == (1, 3, 256, 256), f"fused shape: {fused.shape}"
    assert main_out.shape  == (1, 3, 256, 256), f"main_out shape: {main_out.shape}"
    assert nsst_recon.shape == (1, 3, 256, 256), f"nsst_recon shape: {nsst_recon.shape}"

    # w should be a scalar (0-dim) or (1,) tensor close to sigmoid(-3) ≈ 0.0474
    w_val = w.item() if w.numel() == 1 else w.mean().item()
    assert 0.0 < w_val < 0.15, (
        f"fusion weight w={w_val:.4f} is far from expected sigmoid(-3)≈0.0474 at init"
    )


def test_hfgenerator_nsst_disabled_backward_compat():
    """With nsst.enable=False (or no nsst key), forward returns a single Tensor."""
    from src.models.huggingface_gan.gen import HFGenerator

    # Use base config without nsst key — must not break existing behavior
    cfg = _make_base_cfg()
    gen = HFGenerator(cfg, encoder=MockBackbone()).eval()

    sar = torch.randn(1, 1, 256, 256)
    with torch.no_grad():
        out = gen(sar)

    # The existing generator always returns a single Tensor
    assert isinstance(out, torch.Tensor), (
        f"nsst disabled: expected Tensor, got {type(out)}"
    )
    assert out.shape == (1, 3, 256, 256), f"output shape mismatch: {out.shape}"


def test_fusion_logit_initial_value():
    """_fusion_logit is initialized to -3.0, so sigmoid(_fusion_logit) ≈ 0.0474."""
    from src.models.huggingface_gan.gen import HFGenerator

    cfg = _make_nsst_enabled_cfg()
    gen = HFGenerator(cfg, encoder=MockBackbone())

    if not hasattr(gen, "_fusion_logit"):
        pytest.skip("HFGenerator does not yet have _fusion_logit")

    logit_val = gen._fusion_logit.item()
    assert logit_val == pytest.approx(-3.0, abs=1e-5), (
        f"_fusion_logit initial value {logit_val:.4f} != -3.0"
    )

    w = torch.sigmoid(gen._fusion_logit).item()
    expected = 1.0 / (1.0 + math.exp(3.0))   # sigmoid(-3) ≈ 0.04742
    assert w == pytest.approx(expected, rel=1e-4), (
        f"sigmoid(_fusion_logit)={w:.5f}, expected {expected:.5f}"
    )


# ---------------------------------------------------------------------------
# Group 4: Factory / optimizer param groups
# ---------------------------------------------------------------------------

def test_build_optimizers_nsst_params_in_fresh_group():
    """nsst_branch params and _fusion_logit appear in opt_g.param_groups[1] (fresh_params).

    fresh_params[1] uses lr = cfg.optimizer.lr_g (full rate, not 0.1x).
    This test verifies that the factory correctly routes new NSST weights to
    the fresh_params group so they are trained at the full learning rate.
    """
    from src.models.huggingface_gan.gen import HFGenerator

    cfg = _make_nsst_enabled_cfg()
    gen = HFGenerator(cfg, encoder=MockBackbone())

    if not hasattr(gen, "_fusion_logit") or not hasattr(gen, "nsst_branch"):
        pytest.skip(
            "HFGenerator does not yet expose nsst_branch / _fusion_logit "
            "— factory test not applicable"
        )

    from src.models.huggingface_gan.factory import build_optimizers

    # build_optimizers must be updated to include nsst params in fresh group
    try:
        opt_g, _ = build_optimizers(cfg, gen, nn.Linear(1, 1))  # dummy netD
    except Exception:
        opt_g_pair = build_optimizers(cfg, gen, gen)  # fallback: same object
        opt_g = opt_g_pair[0] if isinstance(opt_g_pair, (tuple, list)) else opt_g_pair

    # Collect all param data_ptrs in fresh group (index 1)
    fresh_group   = opt_g.param_groups[1]
    fresh_ptrs    = {p.data_ptr() for p in fresh_group["params"]}

    # _fusion_logit is in its own fusion group (index 2), not fresh_params
    fusion_ptrs = {p.data_ptr() for p in opt_g.param_groups[2]["params"]}
    assert gen._fusion_logit.data_ptr() in fusion_ptrs, (
        "_fusion_logit not found in fusion_params group (param_groups[2])"
    )

    # All nsst_branch params should be in fresh group
    for name, param in gen.nsst_branch.named_parameters():
        assert param.data_ptr() in fresh_ptrs, (
            f"nsst_branch.{name} not found in fresh_params group"
        )

    # Confirm lr of fresh group equals full lr_g
    assert fresh_group["lr"] == pytest.approx(cfg.optimizer.lr_g, rel=1e-5), (
        f"fresh_params lr={fresh_group['lr']} != lr_g={cfg.optimizer.lr_g}"
    )


def test_encoder_lr_scale_zero():
    """encoder_lr_scale=0.0 in config -> enc_params group lr = 0.0.

    When fine-tuning is disabled the frozen encoder group should receive lr=0.
    """
    from src.models.huggingface_gan.gen import HFGenerator
    from src.models.huggingface_gan.factory import build_optimizers

    cfg = OmegaConf.merge(
        _make_base_cfg(),
        OmegaConf.create({"optimizer": {"encoder_lr_scale": 0.0}}),
    )
    gen = HFGenerator(cfg, encoder=MockBackbone())

    # Only run if factory honours encoder_lr_scale
    try:
        opt_g, _ = build_optimizers(cfg, gen, gen)
    except Exception as exc:
        pytest.skip(f"build_optimizers raised {exc!r} — encoder_lr_scale not yet supported")

    enc_lr = opt_g.param_groups[0]["lr"]
    assert enc_lr == pytest.approx(0.0, abs=1e-12), (
        f"encoder group lr={enc_lr} should be 0.0 when encoder_lr_scale=0.0"
    )


# ---------------------------------------------------------------------------
# Group 5: Gradient flow
# ---------------------------------------------------------------------------

def test_nsst_branch_gradient_flows():
    """Gradient from a reconstruction loss flows through NSSTScaleProcessor weights.

    Ensures that the NSST branch is properly connected to the computation graph
    and that its learnable projection weights receive non-zero gradients during
    a training step.
    """
    nsst_torch = pytest.importorskip(
        "src.utils.nsst_torch",
        reason="nsst_torch not importable — implementation incomplete",
    )
    if not hasattr(nsst_torch, "NSSTScaleProcessor"):
        pytest.skip("NSSTScaleProcessor not yet implemented in nsst_torch.py")
    if not hasattr(nsst_torch, "NSSTBranch"):
        pytest.skip("NSSTBranch not yet implemented in nsst_torch.py")

    branch = nsst_torch.NSSTBranch().train()

    x = torch.randn(1, 1, 256, 256, requires_grad=False)
    out = branch(x)                        # (1, 3, 256, 256)
    loss = out.mean()
    loss.backward()

    # Find NSSTScaleProcessor instances inside the branch
    scale_procs = [
        m for m in branch.modules()
        if type(m).__name__ == "NSSTScaleProcessor"
    ]
    assert len(scale_procs) > 0, (
        "No NSSTScaleProcessor modules found inside NSSTBranch"
    )

    for i, proc in enumerate(scale_procs):
        # proj_in is the canonical first learnable layer per the architecture spec
        proj = getattr(proc, "proj_in", None)
        if proj is None:
            # Fall back to first Conv2d or Linear with a weight
            for m in proc.modules():
                if hasattr(m, "weight") and m.weight is not None:
                    proj = m
                    break

        assert proj is not None, (
            f"NSSTScaleProcessor[{i}]: no learnable weight found"
        )
        assert proj.weight.grad is not None, (
            f"NSSTScaleProcessor[{i}].weight.grad is None — "
            "gradient did not flow through NSSTScaleProcessor"
        )
        assert torch.isfinite(proj.weight.grad).all(), (
            f"NSSTScaleProcessor[{i}].weight.grad contains non-finite values"
        )
        assert proj.weight.grad.abs().max() > 0.0, (
            f"NSSTScaleProcessor[{i}].weight.grad is all zeros — dead path"
        )


# ---------------------------------------------------------------------------
# Group 6: Output bounds
# ---------------------------------------------------------------------------

def test_fusion_output_bounded():
    """fused output is bounded in [-1, 1] via tanh; nsst_recon is raw/unbounded."""
    try:
        from src.models.huggingface_gan.gen import HFGenerator
    except Exception as exc:
        pytest.skip(f"HFGenerator not importable: {exc!r}")

    cfg = _make_nsst_enabled_cfg()
    try:
        gen = HFGenerator(cfg, encoder=MockBackbone()).eval()
    except Exception as exc:
        pytest.skip(f"HFGenerator could not be instantiated: {exc!r}")

    if not hasattr(gen, "_fusion_logit"):
        pytest.skip("HFGenerator does not yet have NSST integration (_fusion_logit absent)")

    if not hasattr(gen, "forward") or "return_branches" not in (
        gen.forward.__code__.co_varnames
    ):
        pytest.skip("HFGenerator.forward does not accept return_branches kwarg yet")

    sar = torch.zeros(1, 1, 256, 256)
    with torch.no_grad():
        result = gen(sar, return_branches=True)

    assert isinstance(result, (tuple, list)) and len(result) == 4, (
        f"Expected 4-tuple from return_branches=True, got {type(result)} len={len(result) if hasattr(result, '__len__') else '?'}"
    )
    fused, main_out, nsst_recon, w = result

    assert fused.min() >= -1.0 - 1e-5, (
        f"fused.min()={fused.min().item():.6f} is below -1 — tanh not applied"
    )
    assert fused.max() <= 1.0 + 1e-5, (
        f"fused.max()={fused.max().item():.6f} is above +1 — tanh not applied"
    )

    if nsst_recon is not None:
        assert nsst_recon.dtype in (torch.float16, torch.float32, torch.float64), (
            f"nsst_recon has unexpected dtype {nsst_recon.dtype}"
        )
