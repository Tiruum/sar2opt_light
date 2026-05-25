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
# L1Loss
# ---------------------------------------------------------------------------

def test_l1_loss_identical_inputs():
    from src.models.huggingface_gan.losses import L1Loss
    loss = L1Loss()
    x = torch.randn(2, 3, 256, 256)
    l = loss(x, x)
    assert l.item() == pytest.approx(0.0, abs=1e-6)

def test_l1_loss_known_diff():
    """Constant 1-pixel offset → L1 = 1.0."""
    from src.models.huggingface_gan.losses import L1Loss
    loss = L1Loss()
    pred   = torch.zeros(2, 3, 16, 16)
    target = torch.ones(2, 3, 16, 16)
    l = loss(pred, target)
    assert l.item() == pytest.approx(1.0, rel=1e-5)

def test_l1_loss_grad_flows():
    from src.models.huggingface_gan.losses import L1Loss
    loss = L1Loss()
    pred   = torch.randn(2, 3, 16, 16, requires_grad=True)
    target = torch.randn(2, 3, 16, 16)
    l = loss(pred, target)
    l.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


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


def test_fft_loss_is_phase_sensitive():
    """Rolling the target by 1 px preserves magnitude spectrum but shifts phase.
    A magnitude-only loss would return ~0; the complex L1 loss must return >0.
    """
    from src.models.huggingface_gan.losses import FFTLoss
    loss = FFTLoss()
    x       = torch.randn(2, 3, 64, 64)
    rolled  = torch.roll(x, shifts=(1, 1), dims=(-2, -1))
    l = loss(x, rolled)
    assert l.item() > 1e-3, (
        f"FFTLoss({l.item():.6f}) should detect phase shift; "
        "magnitude-only loss would falsely report ~0"
    )


def test_fft_loss_grad_flows():
    from src.models.huggingface_gan.losses import FFTLoss
    loss = FFTLoss()
    pred   = torch.randn(2, 3, 64, 64, requires_grad=True)
    target = torch.randn(2, 3, 64, 64)
    l = loss(pred, target)
    l.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


# ---------------------------------------------------------------------------
# AdaptiveLoss (Kendall 2018 uncertainty weighting + eta_max clip)
# ---------------------------------------------------------------------------

def test_adaptive_loss_init_weights_are_one():
    """eta=0 at init -> exp(-0) = 1.0 weights for all components."""
    from src.models.huggingface_gan.losses import AdaptiveLoss
    adapt = AdaptiveLoss(n_losses=3)
    w = adapt.effective_weights()
    assert torch.allclose(w, torch.ones(3))

def test_adaptive_loss_forward_value():
    """At init (eta=0), total = sum(L_i * 1 + 0) = sum(L_i)."""
    from src.models.huggingface_gan.losses import AdaptiveLoss
    adapt = AdaptiveLoss(n_losses=3)
    losses = [torch.tensor(1.0), torch.tensor(2.0), torch.tensor(0.5)]
    out = adapt(losses)
    assert out.item() == pytest.approx(3.5, rel=1e-5)

def test_adaptive_loss_eta_gradients_flow():
    """eta must receive gradient when component losses are backpropped."""
    from src.models.huggingface_gan.losses import AdaptiveLoss
    adapt = AdaptiveLoss(n_losses=3)
    losses = [torch.tensor(1.0, requires_grad=True),
              torch.tensor(2.0, requires_grad=True),
              torch.tensor(0.5, requires_grad=True)]
    out = adapt(losses)
    out.backward()
    assert adapt.eta.grad is not None
    assert torch.isfinite(adapt.eta.grad).all()
    # gradient w.r.t. eta_i at eta=0 = -L_i + 1
    expected = torch.tensor([-1.0 + 1, -2.0 + 1, -0.5 + 1])
    assert torch.allclose(adapt.eta.grad, expected, atol=1e-5)

def test_adaptive_loss_eta_max_clip():
    """eta beyond eta_max should be clipped, preventing weight collapse."""
    from src.models.huggingface_gan.losses import AdaptiveLoss
    adapt = AdaptiveLoss(n_losses=2, eta_max=3.0)
    with torch.no_grad():
        adapt.eta.fill_(10.0)  # Way above clip
    w = adapt.effective_weights()
    # exp(-3.0) = 0.0498; without clip would be exp(-10)=4.5e-5
    assert torch.allclose(w, torch.full((2,), float(torch.exp(torch.tensor(-3.0)))), atol=1e-5)


def test_adaptive_loss_eta_min_clip():
    """eta below -eta_max should be clipped, preventing one loss from
    dominating via unbounded `exp(-eta)` growth (regression test for
    hfgan-15-cswa where w_fft ran to ~20 and suppressed GAN signal).
    """
    from src.models.huggingface_gan.losses import AdaptiveLoss
    adapt = AdaptiveLoss(n_losses=2, eta_max=3.0)
    with torch.no_grad():
        adapt.eta.fill_(-10.0)  # Way below -eta_max
    w = adapt.effective_weights()
    # Clamped to -3.0 -> exp(3) = 20.085; without clip would be exp(10)=22026
    expected = float(torch.exp(torch.tensor(3.0)))
    assert torch.allclose(w, torch.full((2,), expected), atol=1e-4)


def test_adaptive_loss_forward_with_negative_eta_clamped():
    """Forward pass with eta below clamp should use clamped value, not raw."""
    from src.models.huggingface_gan.losses import AdaptiveLoss
    adapt = AdaptiveLoss(n_losses=1, eta_max=2.0)
    with torch.no_grad():
        adapt.eta.fill_(-5.0)  # raw < -eta_max
    # Clamped eta = -2.0; L=0.5 -> 0.5 * exp(2) + (-2) = 0.5*7.389 - 2 = 1.694
    out = adapt([torch.tensor(0.5)])
    expected = 0.5 * float(torch.exp(torch.tensor(2.0))) - 2.0
    assert out.item() == pytest.approx(expected, rel=1e-4)

def test_adaptive_loss_wrong_length_raises():
    from src.models.huggingface_gan.losses import AdaptiveLoss
    adapt = AdaptiveLoss(n_losses=3)
    with pytest.raises(AssertionError):
        adapt([torch.tensor(1.0), torch.tensor(2.0)])


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


# ---------------------------------------------------------------------------
# MSSSIMLoss
# ---------------------------------------------------------------------------

def test_msssim_loss_zero_when_equal():
    from src.models.huggingface_gan.losses import MSSSIMLoss
    loss = MSSSIMLoss()
    x = torch.rand(2, 3, 256, 256) * 2 - 1
    out = loss(x, x.clone())
    assert out.item() < 1e-3, f"MS-SSIM(x,x) should give loss ~ 0, got {out.item()}"

def test_msssim_loss_positive_when_different():
    from src.models.huggingface_gan.losses import MSSSIMLoss
    loss = MSSSIMLoss()
    x = torch.rand(2, 3, 256, 256) * 2 - 1
    y = torch.rand(2, 3, 256, 256) * 2 - 1
    out = loss(x, y)
    assert out.item() > 0.1, f"MS-SSIM loss on random pair should be > 0.1, got {out.item()}"

def test_msssim_grad_flows():
    from src.models.huggingface_gan.losses import MSSSIMLoss
    loss = MSSSIMLoss()
    x = (torch.rand(2, 3, 256, 256) * 2 - 1).requires_grad_(True)
    y = torch.rand(2, 3, 256, 256) * 2 - 1
    loss(x, y).backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


# ---------------------------------------------------------------------------
# LABChromaL1Loss
# ---------------------------------------------------------------------------

def test_lab_chroma_loss_zero_when_equal():
    from src.models.huggingface_gan.losses import LABChromaL1Loss
    loss = LABChromaL1Loss()
    x = torch.rand(2, 3, 64, 64) * 2 - 1
    out = loss(x, x.clone())
    assert out.item() < 1e-3, f"LABChromaL1(x,x) should be ~0, got {out.item()}"

def test_lab_chroma_loss_responds_to_color_shift():
    """Shifting only red channel should produce a measurable chroma loss."""
    from src.models.huggingface_gan.losses import LABChromaL1Loss
    loss = LABChromaL1Loss()
    target = torch.zeros(2, 3, 64, 64)
    pred = target.clone()
    pred[:, 0] = 0.8           # red shift
    out = loss(pred, target)
    assert out.item() > 0.5, f"chroma loss should react to red shift, got {out.item()}"

def test_lab_chroma_grad_flows():
    from src.models.huggingface_gan.losses import LABChromaL1Loss
    loss = LABChromaL1Loss()
    x = (torch.rand(2, 3, 64, 64) * 2 - 1).requires_grad_(True)
    y = torch.rand(2, 3, 64, 64) * 2 - 1
    loss(x, y).backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


# ---------------------------------------------------------------------------
# WaveletDetailL1Loss
# ---------------------------------------------------------------------------

def test_wavelet_detail_loss_zero_when_equal():
    from src.models.huggingface_gan.losses import WaveletDetailL1Loss
    loss = WaveletDetailL1Loss()
    x = torch.rand(2, 3, 64, 64) * 2 - 1
    out = loss(x, x.clone())
    assert out.item() < 1e-6

def test_wavelet_detail_loss_picks_up_edges():
    """A sharp half-image edge that crosses 2x2 block boundaries must
    register on level-1 Haar detail subbands.

    Note: an edge aligned exactly with even pixel boundaries (e.g. col 32)
    would fall *between* 2x2 Haar blocks and produce zero detail
    coefficients — that is correct DWT behavior, not a bug. Using col 33
    forces the transition to cross a 2x2 block, exercising LH/HL/HH.
    """
    from src.models.huggingface_gan.losses import WaveletDetailL1Loss
    loss = WaveletDetailL1Loss()
    target = torch.zeros(2, 3, 64, 64)
    pred = target.clone()
    pred[:, :, :, 33:] = 1.0   # edge at odd column => crosses 2x2 blocks
    out = loss(pred, target)
    assert out.item() > 0.01, f"expected edge loss > 0.01, got {out.item()}"

def test_wavelet_detail_grad_flows():
    from src.models.huggingface_gan.losses import WaveletDetailL1Loss
    loss = WaveletDetailL1Loss()
    x = (torch.rand(2, 3, 64, 64) * 2 - 1).requires_grad_(True)
    y = torch.rand(2, 3, 64, 64) * 2 - 1
    loss(x, y).backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()

def test_wavelet_detail_loss_drops_ll():
    """Adding a DC (constant) offset should NOT change the detail-only loss."""
    from src.models.huggingface_gan.losses import WaveletDetailL1Loss
    loss = WaveletDetailL1Loss()
    target = torch.rand(2, 3, 64, 64)
    pred = target + 0.5            # constant DC shift = LL-only difference
    out = loss(pred, target)
    assert out.item() < 1e-4, f"DC offset should produce ~0 detail loss, got {out.item()}"
