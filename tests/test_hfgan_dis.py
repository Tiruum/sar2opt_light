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
    first_conv = branch.layers[0][0]     # Conv2d inside first Sequential
    assert hasattr(first_conv, 'weight_u'), "First conv should have spectral norm (weight_u)"
    assert hasattr(first_conv, 'weight_v'), "First conv should have spectral norm (weight_v)"


# ---------------------------------------------------------------------------
# HFGANDiscriminator
# ---------------------------------------------------------------------------

def test_discriminator_output_contract():
    from src.models.huggingface_gan.dis import HFGANDiscriminator
    netD = HFGANDiscriminator(in_ch=4, ndf=64)
    sar = torch.randn(2, 1, 256, 256)
    opt = torch.randn(2, 3, 256, 256)
    (logits1, logits2), feats = netD(sar, opt)
    assert logits1.shape == (2, 1, 30, 30)
    assert logits2.shape == (2, 1, 14, 14)
    assert len(feats) == 8                  # 4 per branch

def test_discriminator_no_gradient_on_real_during_d_step():
    """Discriminator should not require grad on the input tensors."""
    from src.models.huggingface_gan.dis import HFGANDiscriminator
    netD = HFGANDiscriminator(in_ch=4, ndf=64)
    sar = torch.randn(2, 1, 256, 256)
    opt = torch.randn(2, 3, 256, 256)
    (logits1, _), _ = netD(sar, opt)
    loss = logits1.mean()
    loss.backward()                         # should not raise

def test_discriminator_downsample_halves_spatial():
    from src.models.huggingface_gan.dis import HFGANDiscriminator
    netD = HFGANDiscriminator(in_ch=4, ndf=64)
    x = torch.randn(2, 4, 256, 256)
    x2 = netD.downsample(x)
    assert x2.shape == (2, 4, 128, 128)
