"""Shared fixtures for huggingface-gan tests. Import via: from tests.conftest_hfgan import ..."""
from types import SimpleNamespace
import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf


class MockBackbone(nn.Module):
    """Drop-in for AutoBackbone. Returns zeros at ConvNeXtV2-Tiny feature map sizes.
    No HF download, no internet required. Device-aware."""
    def forward(self, pixel_values):
        B = pixel_values.shape[0]
        dev = pixel_values.device
        return SimpleNamespace(feature_maps=(
            torch.zeros(B,  96, 64, 64, device=dev),
            torch.zeros(B, 192, 32, 32, device=dev),
            torch.zeros(B, 384, 16, 16, device=dev),
            torch.zeros(B, 768,  8,  8, device=dev),
        ))


@pytest.fixture(scope='module')
def device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


@pytest.fixture(scope='module')
def test_cfg():
    return OmegaConf.load('src/models/huggingface_gan/config.yaml')


@pytest.fixture(scope='module')
def mock_backbone():
    return MockBackbone()
