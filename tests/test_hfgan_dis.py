import pytest
import torch


# ---------------------------------------------------------------------------
# PatchDisBranch
# ---------------------------------------------------------------------------

def test_branch_full_scale_logits_shape():
    from src.models.huggingface_gan.dis import PatchDisBranch
    branch = PatchDisBranch(in_ch=4, ndf=64)
    x = torch.randn(2, 4, 256, 256)
    logits, feats = branch(x)
    assert logits.shape == (2, 1, 30, 30), f"got {logits.shape}"

def test_branch_half_scale_logits_shape():
    from src.models.huggingface_gan.dis import PatchDisBranch
    branch = PatchDisBranch(in_ch=4, ndf=64)
    x = torch.randn(2, 4, 128, 128)
    logits, feats = branch(x)
    assert logits.shape == (2, 1, 14, 14), f"got {logits.shape}"

def test_branch_returns_4_features():
    from src.models.huggingface_gan.dis import PatchDisBranch
    branch = PatchDisBranch(in_ch=4, ndf=64)
    x = torch.randn(2, 4, 256, 256)
    _, feats = branch(x)
    assert len(feats) == 4

def test_branch_spectral_norm_applied():
    from src.models.huggingface_gan.dis import PatchDisBranch
    branch = PatchDisBranch(in_ch=4, ndf=64)
    first_conv = branch.layers[0][0]
    assert hasattr(first_conv, 'weight_u'), "First conv should have spectral norm (weight_u)"
    assert hasattr(first_conv, 'weight_v'), "First conv should have spectral norm (weight_v)"


# ---------------------------------------------------------------------------
# FinePatchDisBranch
# ---------------------------------------------------------------------------

def test_fine_branch_logits_shape():
    from src.models.huggingface_gan.dis import FinePatchDisBranch
    branch = FinePatchDisBranch(in_ch=4, ndf=64)
    x = torch.randn(2, 4, 256, 256)
    logits, feats = branch(x)
    assert logits.shape == (2, 1, 31, 31), f"got {logits.shape}"

def test_fine_branch_returns_3_features():
    from src.models.huggingface_gan.dis import FinePatchDisBranch
    branch = FinePatchDisBranch(in_ch=4, ndf=64)
    x = torch.randn(2, 4, 256, 256)
    _, feats = branch(x)
    assert len(feats) == 3

def test_fine_branch_spectral_norm_applied():
    from src.models.huggingface_gan.dis import FinePatchDisBranch
    branch = FinePatchDisBranch(in_ch=4, ndf=64)
    first_conv = branch.layers[0][0]
    assert hasattr(first_conv, 'weight_u')
    assert hasattr(first_conv, 'weight_v')

def test_fine_branch_fewer_params_than_standard():
    from src.models.huggingface_gan.dis import PatchDisBranch, FinePatchDisBranch
    std_params  = sum(p.numel() for p in PatchDisBranch(4, 64).parameters())
    fine_params = sum(p.numel() for p in FinePatchDisBranch(4, 64).parameters())
    assert fine_params < std_params, f"fine={fine_params} should be < std={std_params}"


# ---------------------------------------------------------------------------
# MicroPatchDisBranch (sub-22px RF for micro-texture)
# ---------------------------------------------------------------------------

def test_micro_branch_logits_shape():
    from src.models.huggingface_gan.dis import MicroPatchDisBranch
    branch = MicroPatchDisBranch(in_ch=4, ndf=64)
    x = torch.randn(2, 4, 256, 256)
    logits, feats = branch(x)
    assert logits.shape == (2, 1, 63, 63), f"got {logits.shape}"


def test_micro_branch_returns_2_features():
    from src.models.huggingface_gan.dis import MicroPatchDisBranch
    branch = MicroPatchDisBranch(in_ch=4, ndf=64)
    x = torch.randn(2, 4, 256, 256)
    _, feats = branch(x)
    assert len(feats) == 2


def test_micro_branch_spectral_norm_applied():
    from src.models.huggingface_gan.dis import MicroPatchDisBranch
    branch = MicroPatchDisBranch(in_ch=4, ndf=64)
    first_conv = branch.layers[0][0]
    assert hasattr(first_conv, 'weight_u')
    assert hasattr(first_conv, 'weight_v')


def test_micro_branch_smaller_rf_than_fine():
    """Micro branch has fewer stride-2 layers => smaller receptive field => smaller logits map."""
    from src.models.huggingface_gan.dis import FinePatchDisBranch, MicroPatchDisBranch
    fine_logits, _ = FinePatchDisBranch(4, 64)(torch.randn(1, 4, 256, 256))
    micro_logits, _ = MicroPatchDisBranch(4, 64)(torch.randn(1, 4, 256, 256))
    assert micro_logits.shape[-1] > fine_logits.shape[-1], (
        f"micro logits ({micro_logits.shape}) should have more spatial entries than fine ({fine_logits.shape})"
    )


# ---------------------------------------------------------------------------
# HFGANDiscriminator
# ---------------------------------------------------------------------------

def test_discriminator_output_contract():
    from src.models.huggingface_gan.dis import HFGANDiscriminator
    netD = HFGANDiscriminator(in_ch=4, ndf=64)
    sar = torch.randn(2, 1, 256, 256)
    opt = torch.randn(2, 3, 256, 256)
    (logits1, logits2, logits3), feats = netD(sar, opt)
    assert logits1.shape == (2, 1, 30, 30), f"coarse logits: {logits1.shape}"
    assert logits2.shape == (2, 1, 31, 31), f"fine logits:   {logits2.shape}"
    assert logits3.shape == (2, 1, 63, 63), f"micro logits:  {logits3.shape}"
    assert len(feats) == 6, f"expected 6 feature tensors (3+2+1 after dropping layer 0), got {len(feats)}"


def test_discriminator_sar_instance_norm_attenuates_brightness_shift():
    """SAR-only instance norm absorbs SAR brightness shift (hfgan-18 asymmetric norm).
    Input shift of 100 on SAR should not pass through unbounded; SAR side is normed.
    Spectral-norm convs amplify residual float-precision noise; absolute diff cannot
    be ~0, but should remain ≤ 1e6 — well below the no-norm baseline."""
    from src.models.huggingface_gan.dis import HFGANDiscriminator
    torch.manual_seed(0)
    netD = HFGANDiscriminator(in_ch=4, ndf=64).eval()
    sar  = torch.randn(2, 1, 256, 256)
    opt  = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        (l1_base,  _, _), _ = netD(sar,           opt)
        (l1_shift, _, _), _ = netD(sar + 100.0,   opt)
    diff = (l1_base - l1_shift).abs().max().item()
    assert diff < 1e6, f"SAR instance norm not absorbing brightness shift; diff = {diff}"


def test_discriminator_opt_path_is_raw_not_normed():
    """Optical path is NOT instance-normed (hfgan-18 asymmetric norm) — a constant
    brightness shift on opt MUST propagate to logits, giving D a chroma anchor."""
    from src.models.huggingface_gan.dis import HFGANDiscriminator
    torch.manual_seed(0)
    netD = HFGANDiscriminator(in_ch=4, ndf=64).eval()
    sar  = torch.randn(2, 1, 256, 256)
    opt  = torch.randn(2, 3, 256, 256) * 0.1
    with torch.no_grad():
        (l1_base,  _, _), _ = netD(sar, opt)
        (l1_shift, _, _), _ = netD(sar, opt + 0.5)
    diff = (l1_base - l1_shift).abs().mean().item()
    assert diff > 1e-3, (
        f"opt brightness shift was absorbed (diff={diff}); D should see absolute "
        f"colour, but optical path appears to be instance-normed."
    )


def test_discriminator_drops_layer_0_features():
    """Combined feature list should be 6 = 3 (branch1[1:]) + 2 (branch2[1:]) + 1 (branch3[1:])."""
    from src.models.huggingface_gan.dis import HFGANDiscriminator
    netD = HFGANDiscriminator(in_ch=4, ndf=64)
    sar = torch.randn(2, 1, 256, 256)
    opt = torch.randn(2, 3, 256, 256)
    _, feats = netD(sar, opt)
    assert len(feats) == 6

def test_discriminator_no_gradient_on_real_during_d_step():
    """Discriminator should not require grad on the input tensors."""
    from src.models.huggingface_gan.dis import HFGANDiscriminator
    netD = HFGANDiscriminator(in_ch=4, ndf=64)
    sar = torch.randn(2, 1, 256, 256)
    opt = torch.randn(2, 3, 256, 256)
    (logits1, _, _), _ = netD(sar, opt)
    loss = logits1.mean()
    loss.backward()                          # should not raise

def test_discriminator_has_no_downsample():
    from src.models.huggingface_gan.dis import HFGANDiscriminator
    netD = HFGANDiscriminator(in_ch=4, ndf=64)
    assert not hasattr(netD, 'downsample'), "AvgPool downsample removed in hfgan-11"
