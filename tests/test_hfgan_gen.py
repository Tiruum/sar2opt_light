import pytest
import torch
import sys
import os
sys.path.insert(0, os.path.abspath('.'))
from tests.conftest_hfgan import MockBackbone
from omegaconf import OmegaConf


@pytest.fixture(scope='module')
def test_cfg():
    return OmegaConf.load('src/models/huggingface_gan/config.yaml')

@pytest.fixture(scope='module')
def device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


# ---------------------------------------------------------------------------
# ChannelAdapter
# ---------------------------------------------------------------------------

def test_channel_adapter_maps_1_to_3ch():
    from src.models.huggingface_gan.gen import ChannelAdapter
    adapter = ChannelAdapter()
    x = torch.randn(2, 1, 256, 256)
    out = adapter(x)
    assert out.shape == (2, 3, 256, 256)


def test_sar_adapter_shape_and_dtype():
    from src.models.huggingface_gan.gen import SARAdapter
    adapter = SARAdapter()

    # shape
    x = torch.randn(2, 1, 256, 256)
    out = adapter(x)
    assert out.shape == (2, 3, 256, 256), f"expected (2,3,256,256), got {out.shape}"

    # finite
    assert torch.all(torch.isfinite(out)), "output contains NaN or Inf (float32)"

    # dtype preservation — float32
    assert out.dtype == torch.float32, f"dtype mismatch: {out.dtype}"

    # dtype preservation — float16 under autocast (matches Lightning bf16-mixed usage)
    if not torch.cuda.is_available():
        pytest.skip("float16 Sobel test requires CUDA")
    adapter_cuda = SARAdapter().cuda()
    x16 = torch.randn(2, 1, 256, 256, device='cuda', dtype=torch.float16)
    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
        out16 = adapter_cuda(x16)
    assert out16.dtype == torch.float16, f"fp16 dtype mismatch: {out16.dtype}"
    assert torch.all(torch.isfinite(out16)), "output contains NaN or Inf (float16)"


# ---------------------------------------------------------------------------
# CrossScaleWindowAttention (CSWA)
# ---------------------------------------------------------------------------

def test_cswa_same_scale_shape():
    """Neighbor resolution equals query resolution — no resize path."""
    from src.models.huggingface_gan.gen import CrossScaleWindowAttention
    cswa = CrossScaleWindowAttention(dim=192, window_size=8, num_heads=8)
    q = torch.randn(2, 192, 64, 64)
    out = cswa(q, q)
    assert out.shape == (2, 192, 64, 64)

def test_cswa_neighbor_resize_shape():
    """Neighbor at half resolution gets bilinearly upsampled to query resolution."""
    from src.models.huggingface_gan.gen import CrossScaleWindowAttention
    cswa = CrossScaleWindowAttention(dim=192, window_size=8, num_heads=8)
    q = torch.randn(2, 192, 64, 64)
    n = torch.randn(2, 192, 32, 32)
    out = cswa(q, n)
    assert out.shape == (2, 192, 64, 64)

def test_cswa_full_attention_at_bottleneck():
    """8x8 features with window_size=8 = single window = global attention."""
    from src.models.huggingface_gan.gen import CrossScaleWindowAttention
    cswa = CrossScaleWindowAttention(dim=192, window_size=8, num_heads=8)
    q = torch.randn(2, 192, 8, 8)
    n = torch.randn(2, 192, 16, 16)
    out = cswa(q, n)
    assert out.shape == (2, 192, 8, 8)

def test_cswa_finite_and_grad():
    """Output finite; gradients flow back to input + params."""
    from src.models.huggingface_gan.gen import CrossScaleWindowAttention
    cswa = CrossScaleWindowAttention(dim=128, window_size=4, num_heads=4)
    q = torch.randn(1, 128, 16, 16, requires_grad=True)
    n = torch.randn(1, 128, 8, 8)
    out = cswa(q, n)
    assert torch.isfinite(out).all()
    out.mean().backward()
    assert q.grad is not None and torch.isfinite(q.grad).all()
    assert cswa.rel_bias_table.grad is not None

def test_window_partition_reverse_roundtrip():
    """window_partition -> window_reverse should be identity."""
    from src.models.huggingface_gan.gen import window_partition, window_reverse
    x = torch.randn(2, 64, 32, 32)
    w = window_partition(x, 8)
    assert w.shape == (2 * 16, 64, 64)  # nW*B=32, ws^2=64, C=64
    y = window_reverse(w, B=2, H=32, W=32, window_size=8)
    assert torch.allclose(x, y)


# ---------------------------------------------------------------------------
# ConvNeXtBlock / CFRStage / CFRBlockT
# ---------------------------------------------------------------------------

def test_convnext_block_shape_preserved():
    from src.models.huggingface_gan.gen import ConvNeXtBlock
    block = ConvNeXtBlock(dim=192)
    x = torch.randn(2, 192, 32, 32)
    out = block(x)
    assert out.shape == (2, 192, 32, 32)

def test_convnext_block_nonidentity():
    from src.models.huggingface_gan.gen import ConvNeXtBlock
    block = ConvNeXtBlock(dim=192)
    x = torch.randn(2, 192, 32, 32)
    out = block(x)
    assert not torch.allclose(out, x)

def test_cfrcswa_stage_preserves_all_shapes():
    from src.models.huggingface_gan.gen import CFRCSWAStage
    stage = CFRCSWAStage(dim=192, num_scales=4, win_sizes=(8, 8, 4, 8), num_heads=8)
    feats = [
        torch.randn(2, 192, 64, 64),
        torch.randn(2, 192, 32, 32),
        torch.randn(2, 192, 16, 16),
        torch.randn(2, 192,  8,  8),
    ]
    out = stage(feats)
    assert len(out) == 4
    for o, f in zip(out, feats):
        assert o.shape == f.shape

def test_cfrcswa_stage_gradients_flow():
    from src.models.huggingface_gan.gen import CFRCSWAStage
    stage = CFRCSWAStage(dim=64, num_scales=4, win_sizes=(4, 4, 4, 4), num_heads=4)
    feats = [
        torch.randn(1, 64, 16, 16, requires_grad=True),
        torch.randn(1, 64,  8,  8, requires_grad=True),
        torch.randn(1, 64,  4,  4, requires_grad=True),
        torch.randn(1, 64,  4,  4, requires_grad=True),
    ]
    out = stage(feats)
    loss = sum(o.mean() for o in out)
    loss.backward()
    for f in feats:
        assert f.grad is not None and torch.isfinite(f.grad).all()


def test_cfr_stage_preserves_all_shapes():
    from src.models.huggingface_gan.gen import CFRStage
    stage = CFRStage(dim=192, num_scales=4)
    feats = [
        torch.randn(2, 192, 64, 64),
        torch.randn(2, 192, 32, 32),
        torch.randn(2, 192, 16, 16),
        torch.randn(2, 192,  8,  8),
    ]
    out = stage(feats)
    assert len(out) == 4
    for o, f in zip(out, feats):
        assert o.shape == f.shape

def test_cfrblock_t_shapes_roundtrip():
    from src.models.huggingface_gan.gen import CFRBlockT
    blk = CFRBlockT(dims=(96, 192, 384, 768), common_dim=192, num_stages=3)
    s0 = torch.randn(2,  96, 64, 64)
    s1 = torch.randn(2, 192, 32, 32)
    s2 = torch.randn(2, 384, 16, 16)
    s3 = torch.randn(2, 768,  8,  8)
    o0, o1, o2, o3 = blk(s0, s1, s2, s3)
    assert o0.shape == s0.shape
    assert o1.shape == s1.shape
    assert o2.shape == s2.shape
    assert o3.shape == s3.shape

def test_cfrblock_t_residual_finite():
    from src.models.huggingface_gan.gen import CFRBlockT
    blk = CFRBlockT(dims=(96, 192, 384, 768), common_dim=192, num_stages=2)
    s0 = torch.randn(1,  96, 64, 64)
    s1 = torch.randn(1, 192, 32, 32)
    s2 = torch.randn(1, 384, 16, 16)
    s3 = torch.randn(1, 768,  8,  8)
    outs = blk(s0, s1, s2, s3)
    for o in outs:
        assert torch.isfinite(o).all()


# ---------------------------------------------------------------------------
# ConvUpsampleBlock
# ---------------------------------------------------------------------------

def test_upsample_block_with_skip():
    from src.models.huggingface_gan.gen import ConvUpsampleBlock
    block = ConvUpsampleBlock(768, 384, 256)
    x    = torch.randn(2, 768, 8, 8)
    skip = torch.randn(2, 384, 16, 16)
    out  = block(x, skip)
    assert out.shape == (2, 256, 16, 16)

def test_upsample_block_no_skip():
    from src.models.huggingface_gan.gen import ConvUpsampleBlock
    block = ConvUpsampleBlock(64, 0, 32)
    x    = torch.randn(2, 64, 64, 64)
    out  = block(x)
    assert out.shape == (2, 32, 128, 128)

def test_upsample_block_residual_contributes():
    """shortcut(x) is added — output should be finite."""
    from src.models.huggingface_gan.gen import ConvUpsampleBlock
    block = ConvUpsampleBlock(64, 0, 32)
    x = torch.randn(1, 64, 8, 8)
    out = block(x)
    assert out.shape == (1, 32, 16, 16)
    assert torch.isfinite(out).all()


def test_upsample_block_uses_pixelshuffle():
    """Regression: ensure block uses PixelShuffle, not F.interpolate."""
    from src.models.huggingface_gan.gen import ConvUpsampleBlock
    block = ConvUpsampleBlock(64, 0, 32)
    assert isinstance(block.pixel_shuffle, torch.nn.PixelShuffle), (
        "ConvUpsampleBlock should use nn.PixelShuffle for upsampling (bilinear blurs)"
    )
    assert block.up_conv.weight.shape == (64 * 4, 64, 1, 1), (
        f"up_conv must inflate channels by scale^2=4 for PixelShuffle(2), got {block.up_conv.weight.shape}"
    )


# ---------------------------------------------------------------------------
# HFGenerator (MockBackbone — no HF download)
# ---------------------------------------------------------------------------

def test_generator_output_shape(test_cfg, device):
    from src.models.huggingface_gan.gen import HFGenerator
    gen = HFGenerator(test_cfg, encoder=MockBackbone()).to(device).eval()
    sar = torch.randn(2, 1, 256, 256, device=device)
    with torch.no_grad():
        out = gen(sar)
    assert out.shape == (2, 3, 256, 256), f"got {out.shape}"

def test_generator_output_tanh_range(test_cfg, device):
    from src.models.huggingface_gan.gen import HFGenerator
    gen = HFGenerator(test_cfg, encoder=MockBackbone()).to(device).eval()
    sar = torch.randn(2, 1, 256, 256, device=device)
    with torch.no_grad():
        out = gen(sar)
    assert out.min().item() >= -1.0 - 1e-5
    assert out.max().item() <=  1.0 + 1e-5

def test_generator_gradients_flow(test_cfg):
    from src.models.huggingface_gan.gen import HFGenerator
    gen = HFGenerator(test_cfg, encoder=MockBackbone()).train()
    sar = torch.randn(1, 1, 256, 256)
    out = gen(sar)
    out.mean().backward()
    grad_norms = [p.grad.norm().item() for p in gen.parameters() if p.grad is not None]
    assert len(grad_norms) > 0, "No gradients flowed"
    assert all(torch.isfinite(torch.tensor(g)) for g in grad_norms)


# ---------------------------------------------------------------------------
# HFCFBranch (hfgan-18 wavelet HF branch)
# ---------------------------------------------------------------------------

def test_hfcf_branch_output_shape():
    """HFCFBranch on (B,1,256,256) returns (B,32,256,256) — same HW, 32 channels."""
    from src.models.huggingface_gan.gen import HFCFBranch
    branch = HFCFBranch(in_channels=1, hidden_dim=64)
    x = torch.randn(2, 1, 256, 256)
    out = branch(x)
    assert out.shape == (2, 32, 256, 256), f"got {out.shape}"


def test_hfcf_branch_gradients_flow():
    """Gradients flow back to HFCFBranch parameters and input."""
    from src.models.huggingface_gan.gen import HFCFBranch
    branch = HFCFBranch(in_channels=1, hidden_dim=32)
    x = torch.randn(1, 1, 128, 128, requires_grad=True)
    out = branch(x)
    out.mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    grad_params = [p for p in branch.parameters() if p.grad is not None]
    assert len(grad_params) > 0
    assert all(torch.isfinite(p.grad).all() for p in grad_params)


def test_dwt_haar_orthonormal():
    """Haar DWT preserves L2 energy (Parseval) when matrix is orthonormal."""
    from src.models.huggingface_gan.gen import HaarDown
    dwt = HaarDown(in_channels=1)
    x = torch.randn(1, 1, 16, 16)
    ll, lh, hl, hh = dwt(x)
    energy_in  = (x ** 2).sum().item()
    energy_out = sum((b ** 2).sum().item() for b in (ll, lh, hl, hh))
    assert abs(energy_in - energy_out) < 1e-4, f"in={energy_in}, out={energy_out}"


def test_speckle_module_returns_gate():
    """SpeckleAwareModule returns (gated_x, gate); both have input shape."""
    from src.models.huggingface_gan.gen import SpeckleAwareModule
    sam = SpeckleAwareModule(in_channels=3, kernel_size=7)
    x = torch.randn(2, 3, 32, 32)
    y, gate = sam(x)
    assert y.shape == x.shape
    assert gate.shape == x.shape


# ---------------------------------------------------------------------------
# HFGenerator hybrid fusion (hfgan-18)
# ---------------------------------------------------------------------------

def test_generator_has_hybrid_components(test_cfg):
    """HFGenerator exposes cfr_final, hfcf_branch, hfcf_final, and _fusion_logit."""
    from src.models.huggingface_gan.gen import HFGenerator, HFCFBranch
    gen = HFGenerator(test_cfg, encoder=MockBackbone())
    assert hasattr(gen, 'cfr_final')
    assert hasattr(gen, 'hfcf_final')
    assert hasattr(gen, '_fusion_logit')
    assert isinstance(gen.hfcf_branch, HFCFBranch)


def test_generator_fusion_weight_init_half(test_cfg):
    """sigmoid(_fusion_logit) ≈ 0.5 at init (logit=0)."""
    from src.models.huggingface_gan.gen import HFGenerator
    gen = HFGenerator(test_cfg, encoder=MockBackbone())
    assert gen._fusion_logit.shape == (1,)
    w_hfcf = torch.sigmoid(gen._fusion_logit).item()
    assert abs(w_hfcf - 0.5) < 1e-6, f"expected 0.5, got {w_hfcf}"


def test_generator_speckle_module_bias_init(test_cfg):
    """SpeckleAwareModule gate[-2].bias initialised to 3.0 by HFGenerator."""
    from src.models.huggingface_gan.gen import HFGenerator, SpeckleAwareModule
    gen = HFGenerator(test_cfg, encoder=MockBackbone())
    sams = [m for m in gen.hfcf_branch.modules() if isinstance(m, SpeckleAwareModule)]
    assert len(sams) >= 1, "Expected at least one SpeckleAwareModule"
    for sam in sams:
        bias = sam.gate[-2].bias
        assert torch.allclose(bias, torch.full_like(bias, 3.0)), (
            f"SAM bias not init to 3.0, got {bias}"
        )


def test_generator_head_alias_points_to_cfr_final(test_cfg):
    """Backward-compat: gen.head should alias gen.cfr_final."""
    from src.models.huggingface_gan.gen import HFGenerator
    gen = HFGenerator(test_cfg, encoder=MockBackbone())
    assert gen.head is gen.cfr_final


def test_generator_fusion_logit_is_learnable(test_cfg):
    """_fusion_logit receives gradient through generator forward."""
    from src.models.huggingface_gan.gen import HFGenerator
    gen = HFGenerator(test_cfg, encoder=MockBackbone()).train()
    sar = torch.randn(1, 1, 256, 256)
    out = gen(sar)
    out.mean().backward()
    assert gen._fusion_logit.grad is not None
    assert torch.isfinite(gen._fusion_logit.grad).all()
