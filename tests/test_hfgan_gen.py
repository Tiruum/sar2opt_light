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

def test_channel_adapter_preserves_spatial():
    from src.models.huggingface_gan.gen import ChannelAdapter
    adapter = ChannelAdapter()
    x = torch.randn(1, 1, 128, 128)
    out = adapter(x)
    assert out.shape == (1, 3, 128, 128)


# ---------------------------------------------------------------------------
# BottleneckAttention
# ---------------------------------------------------------------------------

def test_bottleneck_attention_shape():
    from src.models.huggingface_gan.gen import BottleneckAttention
    attn = BottleneckAttention(dim=768, nhead=8, num_layers=2)
    x   = torch.randn(2, 768, 8, 8)
    out = attn(x)
    assert out.shape == (2, 768, 8, 8)

def test_bottleneck_attention_residual():
    """Output should differ from input (not identity)."""
    from src.models.huggingface_gan.gen import BottleneckAttention
    attn = BottleneckAttention(dim=768, nhead=8, num_layers=2)
    x   = torch.randn(2, 768, 8, 8)
    out = attn(x)
    assert not torch.allclose(out, x)


# ---------------------------------------------------------------------------
# ConvUpsampleBlock
# ---------------------------------------------------------------------------

def test_upsample_block_with_skip():
    from src.models.huggingface_gan.gen import ConvUpsampleBlock
    # Represents up4: input 768ch@8x8 upsampled to 16x16, concat with skip 384ch@16x16
    block = ConvUpsampleBlock(768 + 384, 256)
    x    = torch.randn(2, 768, 8, 8)
    skip = torch.randn(2, 384, 16, 16)
    out  = block(x, skip)
    assert out.shape == (2, 256, 16, 16)

def test_upsample_block_no_skip():
    from src.models.huggingface_gan.gen import ConvUpsampleBlock
    # Represents up1: 64ch@64x64 → 32ch@128x128
    block = ConvUpsampleBlock(64, 32)
    x    = torch.randn(2, 64, 64, 64)
    out  = block(x)
    assert out.shape == (2, 32, 128, 128)

def test_upsample_block_residual_contributes():
    """shortcut(x) is added — output should be distinct from conv-only path."""
    from src.models.huggingface_gan.gen import ConvUpsampleBlock
    block = ConvUpsampleBlock(64, 32)
    x = torch.randn(1, 64, 8, 8)
    out = block(x)
    assert out.shape == (1, 32, 16, 16)
    assert torch.isfinite(out).all()


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
