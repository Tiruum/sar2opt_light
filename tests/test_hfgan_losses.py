import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# GANLoss
# ---------------------------------------------------------------------------

def test_gan_loss_real_single():
    from src.models.huggingface_gan.losses import GANLoss
    loss = GANLoss(real_smooth=0.9)
    logit = torch.zeros(2, 1, 16, 16)       # pred=0, target=0.9 → MSE > 0
    l = loss(logit, is_real=True)
    assert l.shape == torch.Size([])
    assert l.item() > 0.0

def test_gan_loss_fake_single():
    from src.models.huggingface_gan.losses import GANLoss
    loss = GANLoss(fake_smooth=0.0)
    logit = torch.ones(2, 1, 16, 16)        # pred=1, target=0 → MSE > 0
    l = loss(logit, is_real=False)
    assert l.shape == torch.Size([])
    assert l.item() > 0.0

def test_gan_loss_perfect_real():
    from src.models.huggingface_gan.losses import GANLoss
    loss = GANLoss(real_smooth=0.9)
    logit = torch.full((2, 1, 16, 16), 0.9)  # pred == target → MSE = 0
    l = loss(logit, is_real=True)
    assert l.item() == pytest.approx(0.0, abs=1e-6)

def test_gan_loss_tuple_logits():
    from src.models.huggingface_gan.losses import GANLoss
    loss = GANLoss()
    logits = (torch.zeros(2, 1, 30, 30), torch.zeros(2, 1, 14, 14))
    l = loss(logits, is_real=True)
    assert l.shape == torch.Size([])
    assert l.item() > 0.0

def test_gan_loss_tuple_averaged():
    """Tuple loss should equal mean of individual losses."""
    from src.models.huggingface_gan.losses import GANLoss
    loss = GANLoss(real_smooth=0.9)
    l1 = torch.zeros(2, 1, 30, 30)
    l2 = torch.zeros(2, 1, 14, 14)
    combined = loss((l1, l2), is_real=True)
    individual = (loss(l1, is_real=True) + loss(l2, is_real=True)) / 2
    assert combined.item() == pytest.approx(individual.item(), rel=1e-5)


# ---------------------------------------------------------------------------
# FeatureMatchingLoss
# ---------------------------------------------------------------------------

def test_fm_loss_positive():
    from src.models.huggingface_gan.losses import FeatureMatchingLoss
    loss = FeatureMatchingLoss()
    fake  = [torch.randn(2, 64, 32, 32) for _ in range(8)]
    real  = [torch.randn(2, 64, 32, 32) for _ in range(8)]
    l = loss(fake, real)
    assert l.shape == torch.Size([])
    assert l.item() >= 0.0

def test_fm_loss_identical_inputs():
    from src.models.huggingface_gan.losses import FeatureMatchingLoss
    loss = FeatureMatchingLoss()
    feats = [torch.randn(2, 64, 32, 32) for _ in range(8)]
    l = loss(feats, feats)
    assert l.item() == pytest.approx(0.0, abs=1e-5)

def test_fm_loss_averaged_over_layers():
    from src.models.huggingface_gan.losses import FeatureMatchingLoss
    loss = FeatureMatchingLoss()
    import torch.nn.functional as F
    fake = [torch.ones(2, 8, 4, 4)]
    real = [torch.zeros(2, 8, 4, 4)]
    l = loss(fake, real)
    assert l.item() == pytest.approx(F.l1_loss(fake[0], real[0]).item(), rel=1e-5)


# ---------------------------------------------------------------------------
# FFTLoss
# ---------------------------------------------------------------------------

def test_fft_loss_shape():
    from src.models.huggingface_gan.losses import FFTLoss
    loss = FFTLoss()
    pred   = torch.randn(2, 3, 256, 256)
    target = torch.randn(2, 3, 256, 256)
    l = loss(pred, target)
    assert l.shape == torch.Size([])
    assert l.item() >= 0.0

def test_fft_loss_identical_inputs():
    from src.models.huggingface_gan.losses import FFTLoss
    loss = FFTLoss()
    x = torch.randn(2, 3, 256, 256)
    l = loss(x, x)
    assert l.item() == pytest.approx(0.0, abs=1e-5)


# ---------------------------------------------------------------------------
# PerceptualLoss — test _norm math only (no backbone download)
# ---------------------------------------------------------------------------

def test_perceptual_norm_maps_minus1_to_imagenet():
    """x=-1 (black) should map to (0 - mean) / std for each channel."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    x = torch.full((1, 3, 4, 4), -1.0)
    normed = ((x + 1) / 2 - mean) / std
    expected_ch0 = (0.0 - 0.485) / 0.229
    assert normed[0, 0, 0, 0].item() == pytest.approx(expected_ch0, rel=1e-4)

def test_perceptual_norm_maps_plus1_to_imagenet():
    """x=+1 (white) should map to (1 - mean) / std for each channel."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    x = torch.full((1, 3, 4, 4), 1.0)
    normed = ((x + 1) / 2 - mean) / std
    expected_ch0 = (1.0 - 0.485) / 0.229
    assert normed[0, 0, 0, 0].item() == pytest.approx(expected_ch0, rel=1e-4)
