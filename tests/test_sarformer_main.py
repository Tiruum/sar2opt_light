"""Tests for SARFormer-WB Lightning module (sarformer-wb-3-simple).

Smoke-level tests: construct module, run a single training_step + a single
validation_step, verify metric logging, verify optimizer ordering.
Uses ``MockSwinBackbone`` so no HF download is needed.

The previous (sarformer-wb-2-rebal) variants of these tests exercised the
Phi-D path and the speckle-consistency warm-up — both are removed in
sarformer-wb-3-simple.  Tests that referenced those features are xfailed
below to keep the prior surface area visible.
"""
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import pytest
import torch
from omegaconf import OmegaConf

from tests.conftest_sarformer import MockSwinBackbone

from src.models.sarformer_wb.main import SARFormerWBLightningModule

_REMOVED = "removed in sarformer-wb-3-simple simplification"


@pytest.fixture(scope='module')
def cfg():
    return OmegaConf.load('src/models/sarformer_wb/config.yaml')


def _module(cfg):
    cfg2 = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    return SARFormerWBLightningModule(cfg2, encoder=MockSwinBackbone())


def test_module_constructs(cfg):
    lm = _module(cfg)
    assert lm.netG is not None
    assert lm.netD is not None


def test_configure_optimizers_two(cfg):
    lm = _module(cfg)
    opts = lm.configure_optimizers()
    assert isinstance(opts, list)
    assert len(opts) == 2


@pytest.mark.xfail(reason=_REMOVED, run=False, strict=False)
def test_module_constructs_with_phi(cfg):
    lm = _module(cfg)
    assert lm.has_phi
    assert lm.netD_phi is not None


@pytest.mark.xfail(reason=_REMOVED, run=False, strict=False)
def test_module_constructs_without_phi(cfg):
    pass


@pytest.mark.xfail(reason=_REMOVED, run=False, strict=False)
def test_configure_optimizers_three_with_phi(cfg):
    lm = _module(cfg)
    opts = lm.configure_optimizers()
    assert len(opts) == 3


@pytest.mark.xfail(reason=_REMOVED, run=False, strict=False)
def test_speckle_cons_warmup_schedule(cfg):
    lm = _module(cfg)
    _ = lm.speckle_warmup_start  # attribute removed


def test_validation_step_updates_metrics(cfg):
    lm = _module(cfg).eval()
    sar = torch.randn(2, 1, 256, 256)
    opt = torch.randn(2, 3, 256, 256).clamp(-1, 1)
    lm.validation_step((sar, opt), batch_idx=0)
    psnr = lm.psnr.compute()
    ssim = lm.ssim.compute()
    assert torch.isfinite(psnr)
    assert torch.isfinite(ssim)
    lm.psnr.reset()
    lm.ssim.reset()


def test_training_step_runs_without_nan(cfg, tmp_path):
    """Single training-step smoke test.

    Bypasses ``self.optimizers()`` (which requires a full Trainer) by stubbing
    it with manual optimisers built from the factory.
    """
    lm = _module(cfg).train()
    opts = lm.configure_optimizers()
    # Patch optimizers() to return our manual list (LightningOptimizer wrapping
    # would normally happen — for this smoke we just bypass).
    lm.optimizers = lambda: opts                                  # type: ignore
    # Patch manual_backward to do a regular backward.
    lm.manual_backward = lambda loss: loss.backward()             # type: ignore
    # Patch log_dict + log to no-op so Trainer isn't required.
    lm.log = lambda *a, **k: None                                 # type: ignore
    lm.log_dict = lambda *a, **k: None                            # type: ignore
    # Patch current_epoch property.
    lm.__class__.current_epoch = property(lambda self: 0)
    sar = torch.randn(2, 1, 256, 256)
    opt = torch.randn(2, 3, 256, 256).clamp(-1, 1)
    # batch_idx=0 triggers R1 on the main D.
    lm.training_step((sar, opt), batch_idx=0)
    # Sanity: at least one G param should have a gradient after the step.
    n_with_grad = sum(
        1 for p in lm.netG.parameters()
        if p.grad is not None and torch.isfinite(p.grad).all()
    )
    assert n_with_grad > 0
