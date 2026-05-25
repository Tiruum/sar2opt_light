"""Tests for SARFormer-WB discriminators.

``PhiPhysicsDis`` was removed in sarformer-wb-3-simple; its tests below are
xfailed to keep the prior contract visible.
"""
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import pytest
import torch

from src.models.sarformer_wb.dis import MSPatchGANDis

_REMOVED = "removed in sarformer-wb-3-simple simplification"


# --------------------------------------------------------------------- main D


def test_mspatchgan_dis_shapes():
    D = MSPatchGANDis(in_ch=4, ndf=64)
    sar = torch.randn(2, 1, 256, 256)
    opt = torch.randn(2, 3, 256, 256)
    (l_coarse, l_fine), feats = D(sar, opt)
    assert l_coarse.shape == (2, 1, 30, 30)
    assert l_fine.shape == (2, 1, 31, 31)


def test_mspatchgan_dis_feature_count():
    """Coarse drops layer-0 (4 layers -> 3 feats); Fine drops layer-0 (3 layers -> 2 feats)."""
    D = MSPatchGANDis(in_ch=4, ndf=64)
    sar = torch.randn(1, 1, 256, 256)
    opt = torch.randn(1, 3, 256, 256)
    _, feats = D(sar, opt)
    assert len(feats) == 3 + 2


def test_mspatchgan_dis_grad_flow():
    D = MSPatchGANDis(in_ch=4, ndf=64)
    sar = torch.randn(1, 1, 256, 256)
    opt = torch.randn(1, 3, 256, 256, requires_grad=True)
    (l_coarse, l_fine), _ = D(sar, opt)
    (l_coarse.mean() + l_fine.mean()).backward()
    assert opt.grad is not None
    assert torch.isfinite(opt.grad).all()


# --------------------------------------------------------------------- Phi D


@pytest.mark.xfail(reason=_REMOVED, run=False, strict=False)
def test_phi_physics_dis_shape():
    from src.models.sarformer_wb.dis import PhiPhysicsDis  # noqa: F401
    D = PhiPhysicsDis(in_ch=5, ndf=32)
    sar = torch.randn(2, 1, 256, 256)
    opt = torch.randn(2, 3, 256, 256)
    s_spk = torch.randn(2, 1, 256, 256)
    logits, feats = D(sar, opt, s_spk)
    assert logits.dim() == 4
    assert logits.shape[0] == 2
    assert logits.shape[1] == 1


@pytest.mark.xfail(reason=_REMOVED, run=False, strict=False)
def test_phi_physics_dis_features_drop_layer0():
    from src.models.sarformer_wb.dis import PhiPhysicsDis  # noqa: F401
    D = PhiPhysicsDis(in_ch=5, ndf=32)
    sar = torch.randn(1, 1, 256, 256)
    opt = torch.randn(1, 3, 256, 256)
    s_spk = torch.randn(1, 1, 256, 256)
    _, feats = D(sar, opt, s_spk)
    assert len(feats) == 3


@pytest.mark.xfail(reason=_REMOVED, run=False, strict=False)
def test_phi_physics_dis_param_budget():
    from src.models.sarformer_wb.dis import PhiPhysicsDis  # noqa: F401
    D = PhiPhysicsDis(in_ch=5, ndf=32)
    n = sum(p.numel() for p in D.parameters())
    assert n < 3_000_000, f"Phi-D has {n:,} params, expected ~2M"


@pytest.mark.xfail(reason=_REMOVED, run=False, strict=False)
def test_phi_physics_dis_grad_flow():
    from src.models.sarformer_wb.dis import PhiPhysicsDis  # noqa: F401
    D = PhiPhysicsDis(in_ch=5, ndf=32)
    sar = torch.randn(1, 1, 256, 256)
    opt = torch.randn(1, 3, 256, 256, requires_grad=True)
    s_spk = torch.randn(1, 1, 256, 256)
    logits, _ = D(sar, opt, s_spk)
    logits.mean().backward()
    assert opt.grad is not None and torch.isfinite(opt.grad).all()
