"""Tests for SARFormer-WB losses.

``UncertaintyL1Loss`` and ``SpeckleConsistencyLoss`` were removed in
sarformer-wb-3-simple; their tests are xfailed below to keep prior contracts
visible.
"""
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import pytest
import torch

from src.models.sarformer_wb.losses import (
    GANLoss, FeatureMatchingLoss, MSSSIMLoss, LABChromaL1Loss,
    WaveletDetailL1Loss, PlainL1Loss, AdaptiveLoss,
)

_REMOVED = "removed in sarformer-wb-3-simple simplification"


# --------------------------------------------------------------------- GAN


def test_gan_loss_tensor():
    gan = GANLoss()
    logits = torch.zeros(2, 1, 30, 30)
    real = gan(logits, is_real=True)
    fake = gan(logits, is_real=False)
    assert real.item() > 0
    assert fake.item() >= 0
    assert torch.isfinite(real)
    assert torch.isfinite(fake)


def test_gan_loss_tuple():
    gan = GANLoss()
    logits = (torch.zeros(1, 1, 4, 4), torch.zeros(1, 1, 8, 8))
    val = gan(logits, is_real=True)
    assert torch.isfinite(val)


# --------------------------------------------------------------------- FM


def test_feature_matching_loss_zero_on_equal():
    fm = FeatureMatchingLoss()
    feats = [torch.randn(2, 16, 8, 8) for _ in range(3)]
    val = fm(feats, [f.clone() for f in feats])
    assert val.item() < 1e-6


# --------------------------------------------------------------------- MS-SSIM


def test_msssim_loss_zero_on_equal():
    ms = MSSSIMLoss()
    x = torch.rand(2, 3, 256, 256) * 2.0 - 1.0
    val = ms(x, x)
    # MS-SSIM(x,x) ~ 1.0 -> loss ~ 0
    assert val.abs().item() < 5e-3


# --------------------------------------------------------------------- LAB chroma


def test_lab_chroma_loss_zero_on_equal():
    lab = LABChromaL1Loss()
    x = torch.rand(1, 3, 32, 32) * 2.0 - 1.0
    val = lab(x, x)
    assert val.item() < 1e-3


# --------------------------------------------------------------------- wavelet


def test_wavelet_detail_zero_on_equal():
    wav = WaveletDetailL1Loss()
    x = torch.randn(1, 3, 32, 32)
    val = wav(x, x)
    assert val.item() < 1e-6


# --------------------------------------------------------------------- plain L1


def test_plain_l1_loss_zero_on_equal():
    l1 = PlainL1Loss()
    x = torch.randn(1, 3, 16, 16)
    assert l1(x, x).item() < 1e-6


def test_plain_l1_loss_matches_functional():
    import torch.nn.functional as F
    l1 = PlainL1Loss()
    x = torch.randn(1, 3, 16, 16)
    y = torch.randn(1, 3, 16, 16)
    assert abs(l1(x, y).item() - F.l1_loss(x, y).item()) < 1e-6


# --------------------------------------------------------------------- uncertainty (xfail)


@pytest.mark.xfail(reason=_REMOVED, run=False, strict=False)
def test_uncertainty_loss_zero_when_pred_eq_target_and_s_zero():
    from src.models.sarformer_wb.losses import UncertaintyL1Loss  # noqa: F401
    u = UncertaintyL1Loss(reg=0.01, clamp=2.0)
    x = torch.randn(1, 3, 16, 16)
    s = torch.zeros(1, 1, 16, 16)
    val = u(x, x, s)
    assert val.abs().item() < 1e-6


@pytest.mark.xfail(reason=_REMOVED, run=False, strict=False)
def test_uncertainty_loss_clamps_log_var():
    from src.models.sarformer_wb.losses import UncertaintyL1Loss  # noqa: F401
    u = UncertaintyL1Loss(reg=0.0, clamp=2.0)
    x = torch.zeros(1, 3, 4, 4)
    y = torch.ones(1, 3, 4, 4)
    big_s = torch.full((1, 1, 4, 4), 100.0)
    val = u(x, y, big_s)
    expected = torch.exp(torch.tensor(-2.0)).item() * 1.0 + 2.0
    assert abs(val.item() - expected) < 1e-3


@pytest.mark.xfail(reason=_REMOVED, run=False, strict=False)
def test_uncertainty_loss_reg_term_active():
    from src.models.sarformer_wb.losses import UncertaintyL1Loss  # noqa: F401
    u_no_reg = UncertaintyL1Loss(reg=0.0, clamp=2.0)
    u_with_reg = UncertaintyL1Loss(reg=1.0, clamp=2.0)
    x = torch.zeros(1, 3, 4, 4)
    y = torch.zeros(1, 3, 4, 4)
    s = torch.full((1, 1, 4, 4), 1.5)
    v1 = u_no_reg(x, y, s).item()
    v2 = u_with_reg(x, y, s).item()
    assert v2 > v1


# --------------------------------------------------------------------- speckle consistency (xfail)


@pytest.mark.xfail(reason=_REMOVED, run=False, strict=False)
def test_speckle_consistency_finite_and_scalar():
    from src.models.sarformer_wb.losses import SpeckleConsistencyLoss  # noqa: F401
    sc = SpeckleConsistencyLoss(in_ch=3, hidden=16, looks=4)
    sar = torch.randn(1, 1, 64, 64)
    fake = torch.randn(1, 3, 64, 64)
    s_spk = torch.randn(1, 1, 64, 64)
    val = sc(sar, fake, s_spk)
    assert val.dim() == 0
    assert torch.isfinite(val)


@pytest.mark.xfail(reason=_REMOVED, run=False, strict=False)
def test_speckle_consistency_has_trainable_params():
    from src.models.sarformer_wb.losses import SpeckleConsistencyLoss  # noqa: F401
    sc = SpeckleConsistencyLoss(in_ch=3, hidden=16, looks=4)
    n = sum(p.numel() for p in sc.parameters())
    assert n > 0


@pytest.mark.xfail(reason=_REMOVED, run=False, strict=False)
def test_speckle_consistency_grad_to_fake():
    from src.models.sarformer_wb.losses import SpeckleConsistencyLoss  # noqa: F401
    sc = SpeckleConsistencyLoss(in_ch=3, hidden=16, looks=4)
    sar = torch.randn(1, 1, 32, 32)
    fake = torch.randn(1, 3, 32, 32, requires_grad=True)
    s_spk = torch.randn(1, 1, 32, 32)
    sc(sar, fake, s_spk).backward()
    assert fake.grad is not None and torch.isfinite(fake.grad).all()


# --------------------------------------------------------------------- adaptive


def test_adaptive_loss_initial_weights_equal_one():
    a = AdaptiveLoss(n_losses=3, eta_max=5.0)
    w = a.effective_weights()
    assert torch.allclose(w, torch.ones(3), atol=1e-6)


def test_adaptive_loss_sums_correctly():
    a = AdaptiveLoss(n_losses=2, eta_max=5.0)
    losses = [torch.tensor(2.0), torch.tensor(3.0)]
    val = a(losses)
    # eta=0 -> 2*1 + 0 + 3*1 + 0 = 5
    assert abs(val.item() - 5.0) < 1e-6
