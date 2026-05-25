"""Shared fixtures for SARFormer-WB tests.

Provides a ``MockSwinBackbone`` that mimics ``transformers.AutoBackbone``'s
interface for Swin V2-Tiny at 256x256 input — four feature maps at
{(96,64,64), (192,32,32), (384,16,16), (768,8,8)}. Exposes a ``.channels``
attribute so ``SARFormerWBGenerator`` doesn't hit the fallback branch.

Use via: ``from tests.conftest_sarformer import MockSwinBackbone, sar_cfg``.
"""
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf


class MockSwinBackbone(nn.Module):
    """Drop-in for Swin V2 AutoBackbone (no HF download)."""

    channels = [96, 192, 384, 768]

    def __init__(self):
        super().__init__()
        # Tiny owning parameter so calls to ``.parameters()`` are not empty —
        # the factory still iterates these to build the encoder param group.
        self._stem = nn.Conv2d(3, 96, kernel_size=4, stride=4, bias=False)

    def forward(self, pixel_values):
        B = pixel_values.shape[0]
        dev = pixel_values.device
        # Feed the stem so the backbone has *some* dependence on the input
        # (lets gradient flow tests still pass when checking encoder grads).
        stem = self._stem(pixel_values)                              # (B, 96, 64, 64)
        return SimpleNamespace(feature_maps=(
            stem,
            torch.zeros(B, 192, 32, 32, device=dev),
            torch.zeros(B, 384, 16, 16, device=dev),
            torch.zeros(B, 768,  8,  8, device=dev),
        ))


@pytest.fixture(scope='module')
def device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


@pytest.fixture(scope='module')
def sar_cfg():
    return OmegaConf.load('src/models/sarformer_wb/config.yaml')


@pytest.fixture(scope='module')
def mock_swin():
    return MockSwinBackbone()
