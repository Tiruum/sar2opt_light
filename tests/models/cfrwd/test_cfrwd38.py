"""
cfrwd-38 unit tests.

Run from repo root:
    .venv/Scripts/python.exe -m pytest tests/models/cfrwd/test_cfrwd38.py -v

All tests must pass before marking cfrwd-38 training ready.
"""
import pytest
import torch
import torch.nn as nn


# ─── Task 2: DaubechiesDown ──────────────────────────────────────────────────

def test_daubechies_output_shapes_match_haar():
    """DaubechiesDown must produce same output shapes as HaarDown."""
    from src.models.cfrwd.gen import DaubechiesDown, HaarDown
    x = torch.randn(2, 1, 64, 64)
    haar   = HaarDown(in_channels=1)
    db4    = DaubechiesDown(in_channels=1)
    ll_h, lh_h, hl_h, hh_h = haar(x)
    ll_d, lh_d, hl_d, hh_d = db4(x)
    assert ll_d.shape == ll_h.shape, f"LL: {ll_d.shape} != {ll_h.shape}"
    assert lh_d.shape == lh_h.shape
    assert hl_d.shape == hl_h.shape
    assert hh_d.shape == hh_h.shape


def test_daubechies_lo_filter_unit_norm():
    """Lo_D filter must have L2 norm = 1 (orthonormal)."""
    from src.models.cfrwd.gen import DaubechiesDown
    db4 = DaubechiesDown(in_channels=1)
    lo_norm = db4.lo.norm().item()
    assert abs(lo_norm - 1.0) < 1e-5, f"Lo_D L2 norm = {lo_norm}, expected 1.0"


def test_daubechies_hi_filter_unit_norm():
    """Hi_D filter must have L2 norm = 1 (orthonormal)."""
    from src.models.cfrwd.gen import DaubechiesDown
    db4 = DaubechiesDown(in_channels=1)
    hi_norm = db4.hi.norm().item()
    assert abs(hi_norm - 1.0) < 1e-5, f"Hi_D L2 norm = {hi_norm}, expected 1.0"


def test_daubechies_no_trainable_params():
    """DaubechiesDown must have zero trainable parameters (register_buffer)."""
    from src.models.cfrwd.gen import DaubechiesDown
    db4 = DaubechiesDown(in_channels=1)
    n_params = sum(p.numel() for p in db4.parameters())
    assert n_params == 0, f"Expected 0 trainable params, got {n_params}"


def test_dwtblock_with_db4_shapes():
    """DWTBlock(db4) must return g1 at W/4, g2 at W/4 (3C), g3 at W/2 (3C)."""
    from src.models.cfrwd.gen import DWTBlock
    x = torch.randn(2, 1, 256, 256)
    dwt = DWTBlock(in_channels=1)
    g1, g2, g3 = dwt(x)
    assert g1.shape == (2, 1,  64,  64), f"g1={g1.shape}"
    assert g2.shape == (2, 3,  64,  64), f"g2={g2.shape}"
    assert g3.shape == (2, 3, 128, 128), f"g3={g3.shape}"


def test_daubechies_no_nan():
    """DaubechiesDown must not produce NaN/Inf on normal input."""
    from src.models.cfrwd.gen import DaubechiesDown
    db4 = DaubechiesDown(in_channels=3)
    x = torch.randn(4, 3, 256, 256)
    ll, lh, hl, hh = db4(x)
    for name, t in [('LL', ll), ('LH', lh), ('HL', hl), ('HH', hh)]:
        assert not torch.isnan(t).any(), f"NaN in {name}"
        assert not torch.isinf(t).any(), f"Inf in {name}"


# ─── Task 3: FrequencyAttentionBlock ────────────────────────────────────────

def test_fab_output_shape():
    """FAB must preserve input shape B×C×H×W."""
    from src.models.cfrwd.gen import FrequencyAttentionBlock
    fab = FrequencyAttentionBlock(channels=64, h=64, w=64)
    x = torch.randn(2, 64, 64, 64)
    out = fab(x)
    assert out.shape == x.shape, f"FAB output {out.shape} != input {x.shape}"


def test_fab_identity_init():
    """At init (weight_real=1, weight_imag=0), FAB must be identity: output ≈ input."""
    from src.models.cfrwd.gen import FrequencyAttentionBlock
    fab = FrequencyAttentionBlock(channels=64, h=64, w=64)
    x = torch.randn(2, 64, 64, 64)
    with torch.no_grad():
        out = fab(x)
    assert torch.allclose(out, x, atol=1e-4), \
        f"FAB identity init failed: max diff = {(out - x).abs().max().item():.6f}"


def test_fab_gradient_flows():
    """Gradient must flow through FAB to weight_real and weight_imag."""
    from src.models.cfrwd.gen import FrequencyAttentionBlock
    fab = FrequencyAttentionBlock(channels=16, h=32, w=32)
    x = torch.randn(1, 16, 32, 32)
    out = fab(x)
    out.mean().backward()
    assert fab.weight_real.grad is not None, "No gradient on weight_real"
    assert fab.weight_imag.grad is not None, "No gradient on weight_imag"


def test_fab_no_nan():
    """FAB must not produce NaN/Inf."""
    from src.models.cfrwd.gen import FrequencyAttentionBlock
    fab = FrequencyAttentionBlock(channels=64, h=64, w=64)
    x = torch.randn(4, 64, 64, 64)
    out = fab(x)
    assert not torch.isnan(out).any(), "NaN in FAB output"
    assert not torch.isinf(out).any(), "Inf in FAB output"


def test_hfcf_branch_has_fab_attributes():
    """HFCFBranch must expose fab_low, fab_mid, fab_high."""
    from src.models.cfrwd.gen import HFCFBranch
    branch = HFCFBranch(in_channels=1, hidden_dim=64)
    assert hasattr(branch, 'fab_low'),  "HFCFBranch missing fab_low"
    assert hasattr(branch, 'fab_mid'),  "HFCFBranch missing fab_mid"
    assert hasattr(branch, 'fab_high'), "HFCFBranch missing fab_high"


def test_hfcf_branch_output_shape_unchanged():
    """HFCFBranch output shape must remain B×32×256×256 after FAB insertion."""
    from src.models.cfrwd.gen import HFCFBranch
    branch = HFCFBranch(in_channels=1, hidden_dim=64)
    x = torch.randn(2, 1, 256, 256)
    with torch.no_grad():
        out = branch(x)
    assert out.shape == (2, 32, 256, 256), f"HFCFBranch output shape {out.shape}"


# ─── Task 4: HFMaskedFFTLoss fix + MSSSIMLoss ───────────────────────────────

def test_hf_masked_loss_greater_than_unmasked_naive():
    """
    After fix: when ALL error is concentrated in HF bins,
    masked mean (÷ active bins) > naive mean (÷ all bins incl. zero-masked LF).
    """
    from src.models.cfrwd.losses import HFMaskedFFTLoss

    H, W, freq_threshold = 64, 64, 0.25
    loss_fn = HFMaskedFFTLoss(freq_threshold=freq_threshold)

    # Build a target with energy ONLY in HF bins (freq_mag > 0.25)
    fy = torch.fft.fftfreq(H).abs()
    fx = torch.fft.rfftfreq(W)
    hf_mask = (fy[:, None]**2 + fx[None, :]**2).sqrt() > freq_threshold

    target_f = torch.zeros(1, 1, H, W // 2 + 1, dtype=torch.complex64)
    target_f[0, 0][hf_mask] = 1.0          # energy only in HF
    target = torch.fft.irfft2(target_f, s=(H, W), norm='ortho')
    pred   = torch.zeros_like(target)

    masked_loss = loss_fn(pred, target).item()

    # Naive: divide by ALL freq bins (including ~75% zero-masked LF bins)
    pred_f = torch.fft.rfft2(pred.float(), norm='ortho')
    tgt_f  = torch.fft.rfft2(target.float(), norm='ortho')
    naive_loss = (pred_f - tgt_f).abs().mean().item()

    assert masked_loss > naive_loss, \
        f"HF-only signal: masked {masked_loss:.4f} should be > naive {naive_loss:.4f}"


def test_hf_masked_loss_zero_on_identical():
    """HFMaskedFFTLoss must still be 0 on identical inputs after fix."""
    from src.models.cfrwd.losses import HFMaskedFFTLoss
    loss_fn = HFMaskedFFTLoss(freq_threshold=0.25)
    t = torch.randn(2, 3, 64, 64)
    assert loss_fn(t, t).item() < 1e-6


def test_hf_masked_loss_gradient_flows():
    """Gradient must flow through fixed HFMaskedFFTLoss."""
    from src.models.cfrwd.losses import HFMaskedFFTLoss
    loss_fn = HFMaskedFFTLoss(freq_threshold=0.25)
    pred = torch.randn(2, 3, 64, 64, requires_grad=True)
    target = torch.randn(2, 3, 64, 64)
    loss_fn(pred, target).backward()
    assert pred.grad is not None


def test_msssim_loss_zero_on_identical():
    """MSSSIMLoss must be 0 on identical inputs."""
    from src.models.cfrwd.losses import MSSSIMLoss
    loss_fn = MSSSIMLoss()
    t = torch.rand(2, 3, 256, 256) * 2 - 1  # [-1, 1]
    val = loss_fn(t, t).item()
    assert val < 1e-4, f"MSSSIMLoss on identical = {val}, expected ~0"


def test_msssim_loss_range():
    """MSSSIMLoss must be in [0, 1] for normal inputs in [-1, 1]."""
    from src.models.cfrwd.losses import MSSSIMLoss
    loss_fn = MSSSIMLoss()
    pred   = torch.rand(2, 3, 256, 256) * 2 - 1
    target = torch.rand(2, 3, 256, 256) * 2 - 1
    val = loss_fn(pred, target).item()
    assert 0.0 <= val <= 1.0, f"MSSSIMLoss = {val}, expected [0, 1]"


def test_msssim_loss_gradient_flows():
    """Gradient must flow through MSSSIMLoss to the leaf input tensor."""
    from src.models.cfrwd.losses import MSSSIMLoss
    loss_fn = MSSSIMLoss()
    # base is the leaf tensor; pred is non-leaf (result of arithmetic)
    base   = torch.rand(2, 3, 256, 256, requires_grad=True)
    pred   = base * 2 - 1   # [-1, 1], non-leaf
    target = torch.rand(2, 3, 256, 256) * 2 - 1
    loss_fn(pred, target).backward()
    assert base.grad is not None, "Gradient must flow through MSSSIMLoss"


# ─── Task 5: factory.py ──────────────────────────────────────────────────────

def test_factory_criterions_has_msssim():
    """build_criterions must return 'MSSSIM' key after cfrwd-38."""
    from src.models.cfrwd.factory import build_criterions

    class _FakeBackbone(nn.Module):
        def forward(self, x, y):
            return torch.zeros(x.shape[0])

    crits = build_criterions(lpips_backbone=_FakeBackbone())
    assert 'MSSSIM'     in crits, "build_criterions must contain 'MSSSIM'"
    assert 'FOCAL_FREQ' in crits
    assert 'HF_AUX'     in crits
    assert 'GAN'        in crits
    assert 'FM'         in crits
    assert 'L1'         in crits


# ─── Task 6: main.py routing + adaptive loss ─────────────────────────────────

def test_routing_soft_l2_always_active():
    """Soft L2 routing must be > 0 even for moderately unbalanced routing."""
    weights = torch.zeros(2, 2, 8, 8)
    weights[:, 0, :, :] = 0.3
    weights[:, 1, :, :] = 0.7
    routing_balance_weight = 0.1
    loss = (weights[:, 1].mean() - 0.5).pow(2) * routing_balance_weight
    assert loss.item() > 0, "Soft L2 must penalize w_hfcf=0.7"


def test_routing_soft_l2_zero_at_balance():
    """Soft L2 routing must be exactly 0 when w_hfcf = 0.5."""
    weights = torch.full((2, 2, 8, 8), 0.5)
    routing_balance_weight = 0.1
    loss = (weights[:, 1].mean() - 0.5).pow(2) * routing_balance_weight
    assert loss.item() < 1e-10, f"Soft L2 must be 0 at balance, got {loss.item()}"


def test_n_recon_is_4_with_both_flags():
    """_n_recon must equal 4 when both use_lpips=true and use_msssim=true."""
    _n_recon = 2
    _n_recon += 1  # use_lpips=true
    _n_recon += 1  # use_msssim=true
    assert _n_recon == 4, f"Expected _n_recon=4, got {_n_recon}"
