"""Tests for ``src/models/llwt/dis.py``."""
from types import SimpleNamespace

import pytest
import torch

from src.models.llwt.dis import (
    FixedHaarDWT,
    LLWFormerDiscriminator,
    MainDis,
    SubbandDis,
)
from tests.conftest_llwt import *  # noqa: F401,F403


def _dcfg(use_sub: bool = True, ndf_main: int = 32, ndf_sub: int = 16):
    return SimpleNamespace(model=SimpleNamespace(dis=SimpleNamespace(
        main=SimpleNamespace(in_channels=4, ndf=ndf_main),
        subband=SimpleNamespace(enabled=use_sub, ndf=ndf_sub),
    )))


def test_fixed_haar_shape(rng_seed):
    dwt = FixedHaarDWT()
    x = torch.randn(2, 3, 32, 32)
    y = dwt(x)
    assert y.shape == (2, 12, 16, 16)


def test_fixed_haar_orthonormal_parseval(rng_seed):
    """Parseval: orthonormal Haar preserves L2 energy."""
    dwt = FixedHaarDWT().double()
    x = torch.randn(2, 3, 32, 32, dtype=torch.float64)
    y = dwt(x)
    e_in = x.pow(2).sum().item()
    e_out = y.pow(2).sum().item()
    assert abs(e_in - e_out) / e_in < 1e-10


def test_fixed_haar_rejects_odd(rng_seed):
    dwt = FixedHaarDWT()
    with pytest.raises(ValueError):
        dwt(torch.randn(1, 3, 31, 32))


def test_maindis_shape(rng_seed):
    d = MainDis(in_ch=4, ndf=32)
    sar = torch.randn(2, 1, 64, 64)
    opt = torch.randn(2, 3, 64, 64)
    (lc, lf), feats = d(sar, opt)
    assert lc.ndim == 4 and lc.shape[1] == 1
    assert lf.ndim == 4 and lf.shape[1] == 1
    # Layer-0 dropped from each branch: 3 from coarse + 2 from fine = 5 feats.
    assert len(feats) == 5


def test_subband_dis_shape(rng_seed):
    d = SubbandDis(ndf=16)
    opt = torch.randn(2, 3, 64, 64)
    logits, feats = d(opt)
    assert logits.ndim == 4 and logits.shape[1] == 1
    # 3 inner layers in the branch -> 3 features.
    assert len(feats) == 3


def test_full_dis_with_subband(rng_seed):
    cfg = _dcfg(use_sub=True)
    d = LLWFormerDiscriminator(cfg=cfg)
    sar = torch.randn(2, 1, 64, 64)
    opt = torch.randn(2, 3, 64, 64)
    (lc, lf), ls, feats_main, feats_sub = d(sar, opt)
    assert ls is not None
    assert len(feats_main) == 5
    assert len(feats_sub) == 3


def test_full_dis_subband_disabled(rng_seed):
    cfg = _dcfg(use_sub=False)
    d = LLWFormerDiscriminator(cfg=cfg)
    assert d.sub is None
    sar = torch.randn(2, 1, 64, 64)
    opt = torch.randn(2, 3, 64, 64)
    (lc, lf), ls, feats_main, feats_sub = d(sar, opt)
    assert ls is None
    assert len(feats_main) == 5
    assert feats_sub == []


def test_full_dis_grad_flow(rng_seed):
    cfg = _dcfg(use_sub=True)
    d = LLWFormerDiscriminator(cfg=cfg)
    sar = torch.randn(2, 1, 64, 64)
    opt = torch.randn(2, 3, 64, 64, requires_grad=True)
    (lc, lf), ls, feats_main, feats_sub = d(sar, opt)
    loss = lc.mean() + lf.mean() + ls.mean()
    loss.backward()
    for name, p in d.named_parameters():
        if p.grad is None:
            pytest.fail(f"{name} got no grad")
        assert torch.isfinite(p.grad).all(), f"NaN/Inf grad in {name}"
