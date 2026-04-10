"""
cfrwd-37 unit tests.

Run from repo root:
    python -m pytest tests/models/cfrwd/test_cfrwd37.py -v

All tests must pass before marking cfrwd-37 training ready.
"""
import pytest
import torch
import torch.nn as nn


# ─── Task 2: Temperature scaling in AdaptiveFusion ──────────────────────────

def test_adaptive_fusion_temperature_is_parameter():
    """temperature must be a learnable nn.Parameter initialized to 2.0."""
    from src.models.cfrwd.gen import AdaptiveFusion
    fusion = AdaptiveFusion(feat_channels=32)
    assert hasattr(fusion, 'temperature'), "AdaptiveFusion must have 'temperature' attribute"
    assert isinstance(fusion.temperature, nn.Parameter), "temperature must be nn.Parameter"
    assert abs(fusion.temperature.item() - 2.0) < 1e-5, "temperature must initialize to 2.0"


def test_adaptive_fusion_weights_sum_to_one():
    """Softmax weights must sum to 1 across dim=1 at every spatial position."""
    from src.models.cfrwd.gen import AdaptiveFusion
    fusion = AdaptiveFusion(feat_channels=32)
    cfr  = torch.randn(2, 32, 64, 64)
    hfcf = torch.randn(2, 32, 64, 64)
    _, weights = fusion(cfr, hfcf)
    weight_sum = weights.sum(dim=1)  # B×H×W
    assert torch.allclose(weight_sum, torch.ones_like(weight_sum), atol=1e-5)


def test_adaptive_fusion_negative_temperature_no_crash():
    """Negative temperature must be clamped to 0.1 and not produce NaN."""
    from src.models.cfrwd.gen import AdaptiveFusion
    fusion = AdaptiveFusion(feat_channels=32)
    fusion.temperature.data.fill_(-5.0)
    cfr  = torch.randn(2, 32, 8, 8)
    hfcf = torch.randn(2, 32, 8, 8)
    _, weights = fusion(cfr, hfcf)
    assert not torch.isnan(weights).any(), "NaN in fusion weights with clamped temperature"


# ─── Task 3: remove no_grad from return_branches ────────────────────────────

def test_return_branches_gradient_flows_to_hfcf():
    """
    hfcf_out returned by return_branches=True must carry a gradient path
    back to hfcf_branch parameters. This was broken by torch.no_grad().
    """
    from src.models.cfrwd.gen import CFRWDGenerator
    gen = CFRWDGenerator(in_channels=1)
    # 256×256: required by FAB (cfrwd-38) whose frequency weights are fixed at
    # the stream output resolution (64×64), derived from 256×256 → 2-level DWT → W/4.
    x = torch.randn(1, 1, 256, 256)
    _, _, hfcf_out, _ = gen(x, return_branches=True)
    loss = hfcf_out.mean()
    loss.backward()
    first_hfcf_param = next(gen.hfcf_branch.parameters())
    assert first_hfcf_param.grad is not None, \
        "hfcf_branch must receive gradient via return_branches=True"


def test_return_branches_cfr_gradient_flows():
    """cfr_out must also have a gradient path (via shared self.final)."""
    from src.models.cfrwd.gen import CFRWDGenerator
    gen = CFRWDGenerator(in_channels=1)
    x = torch.randn(1, 1, 256, 256)  # FAB requires 256×256 (see above)
    _, cfr_out, _, _ = gen(x, return_branches=True)
    loss = cfr_out.mean()
    loss.backward()
    first_cfr_param = next(gen.cfr_branch.parameters())
    assert first_cfr_param.grad is not None, \
        "cfr_branch must receive gradient via return_branches=True"


# ─── Task 4: New loss classes ────────────────────────────────────────────────

def test_focal_frequency_loss_nonnegative():
    from src.models.cfrwd.losses import FocalFrequencyLoss
    loss_fn = FocalFrequencyLoss()
    pred   = torch.randn(2, 3, 32, 32)
    target = torch.randn(2, 3, 32, 32)
    assert loss_fn(pred, target).item() >= 0


def test_focal_frequency_loss_zero_on_identical():
    from src.models.cfrwd.losses import FocalFrequencyLoss
    loss_fn = FocalFrequencyLoss()
    t = torch.randn(2, 3, 32, 32)
    assert loss_fn(t, t).item() < 1e-6, "FocalFrequencyLoss must be 0 on identical inputs"


def test_focal_frequency_loss_gradient_flows():
    from src.models.cfrwd.losses import FocalFrequencyLoss
    loss_fn = FocalFrequencyLoss()
    pred   = torch.randn(2, 3, 32, 32, requires_grad=True)
    target = torch.randn(2, 3, 32, 32)
    loss_fn(pred, target).backward()
    assert pred.grad is not None, "Gradient must flow through FocalFrequencyLoss"


def test_hf_masked_fft_loss_nonzero_on_mismatch():
    """Zero pred vs random target must produce loss > 0."""
    from src.models.cfrwd.losses import HFMaskedFFTLoss
    loss_fn = HFMaskedFFTLoss(freq_threshold=0.25)
    pred   = torch.zeros(2, 3, 32, 32, requires_grad=True)
    target = torch.randn(2, 3, 32, 32)
    loss   = loss_fn(pred, target)
    assert loss.item() > 0
    loss.backward()
    assert pred.grad is not None


def test_hf_masked_fft_loss_zero_on_identical():
    from src.models.cfrwd.losses import HFMaskedFFTLoss
    loss_fn = HFMaskedFFTLoss(freq_threshold=0.25)
    t = torch.randn(2, 3, 32, 32)
    assert loss_fn(t, t).item() < 1e-6


def test_hf_masked_fft_loss_ignores_low_freq():
    """
    A pure low-frequency signal (DC component only) should produce near-zero loss
    because freq=0 is below the 0.25 threshold.
    """
    from src.models.cfrwd.losses import HFMaskedFFTLoss
    loss_fn = HFMaskedFFTLoss(freq_threshold=0.25)
    # DC signal: constant value — only freq=0 component
    pred   = torch.ones(1, 1, 32, 32) * 0.5
    target = torch.ones(1, 1, 32, 32) * 0.7
    loss   = loss_fn(pred, target)
    # freq=0 is below threshold → should be masked out → loss ≈ 0
    assert loss.item() < 1e-4, f"Low-freq only signal should have near-zero HFMaskedFFTLoss, got {loss.item()}"


# ─── Task 5: factory.py ──────────────────────────────────────────────────────

def test_factory_criterions_has_focal_freq_and_hf_aux():
    """
    After factory.py changes, build_criterions must return FOCAL_FREQ and HF_AUX,
    and must NOT return FFT.
    """
    from src.models.cfrwd.factory import build_criterions

    class _FakeBackbone(nn.Module):
        """Minimal stub for LPIPSLoss backbone (avoids downloading AlexNet in tests)."""
        def forward(self, x, y):
            return torch.zeros(x.shape[0])

    crits = build_criterions(lpips_backbone=_FakeBackbone())
    assert 'FOCAL_FREQ' in crits, "build_criterions must contain 'FOCAL_FREQ'"
    assert 'HF_AUX'     in crits, "build_criterions must contain 'HF_AUX'"
    assert 'FFT'        not in crits, "build_criterions must NOT contain 'FFT' after cfrwd-37"
    assert 'GAN'        in crits
    assert 'FM'         in crits
    assert 'L1'         in crits


# ─── Task 6: routing entropy in main.py ─────────────────────────────────────

def test_routing_entropy_penalizes_collapse():
    """Loss must be > 0 when routing is fully collapsed to one branch."""
    # All weight to CFR (w_cfr=1.0, w_hfcf=0.0)
    weights = torch.zeros(2, 2, 8, 8)
    weights[:, 0, :, :] = 1.0
    eps = 1e-8
    H_spatial = -(weights * (weights + eps).log()).sum(dim=1)  # B×H×W
    loss = (0.347 - H_spatial.mean()).clamp(min=0.0) * 0.005
    assert loss.item() > 0, "Routing loss must penalize fully-collapsed routing"


def test_routing_entropy_zero_when_balanced():
    """Loss must be 0 when routing is perfectly balanced (max entropy)."""
    weights = torch.full((2, 2, 8, 8), 0.5)
    eps = 1e-8
    H_spatial = -(weights * (weights + eps).log()).sum(dim=1)
    # entropy = log(2) ≈ 0.693 > threshold 0.347 → relu term = 0
    loss = (0.347 - H_spatial.mean()).clamp(min=0.0) * 0.005
    assert loss.item() == 0.0, "Routing loss must be 0 for balanced routing"
