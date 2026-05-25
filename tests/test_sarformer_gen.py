"""Tests for SARFormer-WB generator and sub-modules.

In ``sarformer-wb-3-simple`` the wavelet bottleneck, SAR pyramid and SAR
cross-attn skip were removed.  Tests for those classes are kept (xfailed) so
the surface area stays visible — if anyone tries to re-introduce the
sub-modules, the xfailed tests document the prior contracts they had.
"""
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import pytest
import torch
from omegaconf import OmegaConf

from tests.conftest_sarformer import MockSwinBackbone

from src.models.sarformer_wb.gen import (
    SARPhysicsFrontEnd, GRN, DropPath,
    ConvNeXtV2GRNBlock, DecoderStage, SARFormerWBGenerator,
)

_REMOVED = "removed in sarformer-wb-3-simple simplification"


@pytest.fixture(scope='module')
def cfg():
    return OmegaConf.load('src/models/sarformer_wb/config.yaml')


# --------------------------------------------------------------------- physics


def test_physics_front_end_shapes_and_finite():
    front = SARPhysicsFrontEnd()
    sar = torch.randn(2, 1, 64, 64)
    three = front(sar)
    assert three.shape == (2, 3, 64, 64)
    assert torch.isfinite(three).all()


def test_physics_front_end_grad_flows():
    front = SARPhysicsFrontEnd()
    sar = torch.randn(2, 1, 32, 32, requires_grad=True)
    three = front(sar)
    three.mean().backward()
    assert sar.grad is not None
    assert torch.isfinite(sar.grad).all()


# --------------------------------------------------------------------- DWT


@pytest.mark.xfail(reason=_REMOVED, run=False, strict=False)
def test_haar_dwt_roundtrip():
    from src.models.sarformer_wb.gen import HaarDWT2d  # noqa: F401
    dwt = HaarDWT2d()
    x = torch.randn(2, 5, 16, 16)
    ll, lh, hl, hh = dwt(x)
    recon = dwt.inverse(ll, lh, hl, hh)
    assert recon.shape == x.shape
    assert torch.allclose(recon, x, atol=1e-5), (recon - x).abs().max()


@pytest.mark.xfail(reason=_REMOVED, run=False, strict=False)
def test_haar_dwt_subband_shapes():
    from src.models.sarformer_wb.gen import HaarDWT2d  # noqa: F401
    dwt = HaarDWT2d()
    x = torch.randn(1, 3, 8, 8)
    ll, lh, hl, hh = dwt(x)
    for sb in (ll, lh, hl, hh):
        assert sb.shape == (1, 3, 4, 4)


# --------------------------------------------------------------------- MiniSwinV2


@pytest.mark.xfail(reason=_REMOVED, run=False, strict=False)
def test_mini_swinv2_shape_preserved():
    from src.models.sarformer_wb.gen import MiniSwinV2Block  # noqa: F401
    block = MiniSwinV2Block(dim=64, window_size=4, num_heads=4)
    x = torch.randn(2, 64, 4, 4)
    y = block(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


@pytest.mark.xfail(reason=_REMOVED, run=False, strict=False)
def test_mini_swinv2_grad():
    from src.models.sarformer_wb.gen import MiniSwinV2Block  # noqa: F401
    block = MiniSwinV2Block(dim=32, window_size=4, num_heads=4)
    x = torch.randn(1, 32, 4, 4, requires_grad=True)
    block(x).mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


# --------------------------------------------------------------------- speckle gate


@pytest.mark.xfail(reason=_REMOVED, run=False, strict=False)
def test_speckle_gated_stack_shape():
    from src.models.sarformer_wb.gen import SpeckleGatedConvStack  # noqa: F401
    stack = SpeckleGatedConvStack(ch=96, depth=3)
    x = torch.randn(2, 96, 8, 8)
    s = torch.randn(2, 1, 8, 8)
    y = stack(x, s)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


# --------------------------------------------------------------------- bottleneck


@pytest.mark.xfail(reason=_REMOVED, run=False, strict=False)
def test_wavelet_bottleneck_shape():
    from src.models.sarformer_wb.gen import WaveletBottleneck  # noqa: F401
    bn = WaveletBottleneck(dim=768, swin_window=4, swin_heads=8)
    b3 = torch.randn(2, 768, 8, 8)
    s_spk = torch.randn(2, 1, 256, 256)
    y = bn(b3, s_spk)
    assert y.shape == b3.shape
    assert torch.isfinite(y).all()


@pytest.mark.xfail(reason=_REMOVED, run=False, strict=False)
def test_wavelet_bottleneck_near_identity_at_init():
    """Residual gate inits with sigmoid(-4) ~ 0.018, so b3' should be close to b3."""
    from src.models.sarformer_wb.gen import WaveletBottleneck  # noqa: F401
    bn = WaveletBottleneck(dim=64, swin_window=4, swin_heads=4)
    b3 = torch.randn(1, 64, 8, 8)
    s_spk = torch.zeros(1, 1, 32, 32)
    y = bn(b3, s_spk)
    diff = (y - b3).abs().mean().item()
    assert diff < 0.5, f"bottleneck should be near identity at init, got mean diff={diff}"


# --------------------------------------------------------------------- SAR pyramid


@pytest.mark.xfail(reason=_REMOVED, run=False, strict=False)
def test_sar_pyramid_keys_and_shapes():
    from src.models.sarformer_wb.gen import SARPyramid  # noqa: F401
    pyr = SARPyramid(base_ch=32, out_ch=64)
    sar = torch.randn(2, 1, 256, 256)
    out = pyr(sar)
    assert set(out.keys()) == {32, 64}
    assert out[32].shape == (2, 64, 32, 32)
    assert out[64].shape == (2, 64, 64, 64)


# --------------------------------------------------------------------- cross-attn skip


@pytest.mark.xfail(reason=_REMOVED, run=False, strict=False)
def test_sar_cross_attn_skip_shape():
    from src.models.sarformer_wb.gen import SARCrossAttnSkip  # noqa: F401
    skip = SARCrossAttnSkip(q_dim=128, kv_dim=64, inner_dim=64,
                            window_size=8, num_heads=1)
    q = torch.randn(2, 128, 32, 32)
    kv = torch.randn(2, 64, 32, 32)
    y = skip(q, kv)
    assert y.shape == q.shape
    assert torch.isfinite(y).all()


@pytest.mark.xfail(reason=_REMOVED, run=False, strict=False)
def test_sar_cross_attn_gate_init_near_zero():
    from src.models.sarformer_wb.gen import SARCrossAttnSkip  # noqa: F401
    skip = SARCrossAttnSkip(q_dim=64, kv_dim=64, inner_dim=64,
                            window_size=8, num_heads=1)
    assert torch.tanh(skip._gate).abs().item() < 0.01


@pytest.mark.xfail(reason=_REMOVED, run=False, strict=False)
def test_sar_cross_attn_kv_resize():
    from src.models.sarformer_wb.gen import SARCrossAttnSkip  # noqa: F401
    skip = SARCrossAttnSkip(q_dim=64, kv_dim=64, inner_dim=64,
                            window_size=8, num_heads=1)
    q = torch.randn(1, 64, 32, 32)
    kv = torch.randn(1, 64, 64, 64)
    y = skip(q, kv)
    assert y.shape == q.shape


# --------------------------------------------------------------------- GRN / DropPath


def test_grn_identity_at_init():
    grn = GRN(dim=16)
    x = torch.randn(2, 8, 8, 16)
    y = grn(x)
    assert torch.allclose(y, x, atol=1e-6)


def test_droppath_identity_in_eval():
    dp = DropPath(drop_prob=0.5)
    dp.eval()
    x = torch.randn(4, 16)
    y = dp(x)
    assert torch.allclose(y, x)


def test_convnextv2_grn_block_shape():
    blk = ConvNeXtV2GRNBlock(dim=64, drop_path=0.0)
    x = torch.randn(2, 64, 32, 32)
    y = blk(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


# --------------------------------------------------------------------- decoder stage


def test_decoder_stage_upsamples_with_skip():
    stage = DecoderStage(up_ch=128, skip_ch=64, out_ch=32, drop_path=0.0)
    x = torch.randn(1, 128, 16, 16)
    skip = torch.randn(1, 64, 32, 32)
    y = stage(x, skip=skip)
    assert y.shape == (1, 32, 32, 32)


def test_decoder_stage_no_skip():
    stage = DecoderStage(up_ch=32, skip_ch=0, out_ch=16, drop_path=0.0)
    x = torch.randn(1, 32, 64, 64)
    y = stage(x, skip=None)
    assert y.shape == (1, 16, 128, 128)


# --------------------------------------------------------------------- full generator


def test_generator_forward_train_returns_tensor(cfg):
    G = SARFormerWBGenerator(cfg, encoder=MockSwinBackbone()).train()
    sar = torch.randn(2, 1, 256, 256)
    out = G(sar)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (2, 3, 256, 256)
    assert torch.isfinite(out).all()
    assert (out >= -1).all() and (out <= 1).all()


def test_generator_forward_eval_returns_tensor(cfg):
    G = SARFormerWBGenerator(cfg, encoder=MockSwinBackbone()).eval()
    sar = torch.randn(1, 1, 256, 256)
    out = G(sar)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (1, 3, 256, 256)


def test_generator_backward_finite(cfg):
    G = SARFormerWBGenerator(cfg, encoder=MockSwinBackbone()).train()
    sar = torch.randn(1, 1, 256, 256, requires_grad=True)
    out = G(sar)
    out.mean().backward()
    assert sar.grad is not None
    assert torch.isfinite(sar.grad).all()
    # At least one decoder parameter should receive a gradient.
    found_param_grad = False
    for p in G.parameters():
        if p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0:
            found_param_grad = True
            break
    assert found_param_grad, "no generator parameter received a gradient"


def test_generator_param_count_under_budget(cfg):
    """Sanity: G should be ~48M params with mock backbone (which has fewer params)."""
    G = SARFormerWBGenerator(cfg, encoder=MockSwinBackbone())
    n_params = sum(p.numel() for p in G.parameters())
    # Mock backbone has very few params; total stays well below the 100M ceiling.
    assert n_params < 100_000_000, f"G has {n_params:,} params, exceeds 100M ceiling"
