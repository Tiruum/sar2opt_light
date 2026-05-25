"""Tests for SARFormer-WB factory: model + criterion + optimizer + scheduler builders.

In sarformer-wb-3-simple the Phi-D + speckle-consistency path was retired, so
``build_models`` returns ``(netG, netD)`` and ``build_optimizers`` returns
``(opt_d, opt_g)``.
"""
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import pytest
import torch
from omegaconf import OmegaConf

from tests.conftest_sarformer import MockSwinBackbone

from src.models.sarformer_wb import factory
from src.models.sarformer_wb.dis import MSPatchGANDis

_REMOVED = "removed in sarformer-wb-3-simple simplification"


@pytest.fixture(scope='module')
def cfg():
    return OmegaConf.load('src/models/sarformer_wb/config.yaml')


# --------------------------------------------------------------------- build_models


def test_build_models_returns_two_modules(cfg):
    netG, netD = factory.build_models(cfg, encoder=MockSwinBackbone())
    assert hasattr(netG, 'encoder')
    assert isinstance(netD, MSPatchGANDis)


@pytest.mark.xfail(reason=_REMOVED, run=False, strict=False)
def test_build_models_phi_disabled(cfg):
    cfg2 = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    cfg2.model.dis.phi.enable = False
    netG, netD_main, netD_phi = factory.build_models(cfg2, encoder=MockSwinBackbone())
    assert netD_phi is None


# --------------------------------------------------------------------- build_criterions


def test_build_criterions_contains_active_losses(cfg):
    crits = factory.build_criterions(cfg)
    assert 'gan' in crits
    assert 'fm' in crits
    assert 'l1' in crits
    assert 'msssim' in crits
    assert 'lab_chroma' in crits
    assert 'wavelet' in crits
    # adaptive disabled by default
    assert 'adaptive' not in crits


@pytest.mark.xfail(reason=_REMOVED, run=False, strict=False)
def test_build_criterions_drops_speckle_when_phi_disabled(cfg):
    cfg2 = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    cfg2.model.dis.phi.enable = False
    crits = factory.build_criterions(cfg2)
    assert 'speckle_cons' not in crits


# --------------------------------------------------------------------- build_optimizers


def test_build_optimizers_param_coverage(cfg):
    """Every netG parameter must appear in some opt_g param group."""
    netG, netD = factory.build_models(cfg, encoder=MockSwinBackbone())
    crits = factory.build_criterions(cfg)
    opt_d, opt_g = factory.build_optimizers(cfg, netG, netD, criterions=crits)
    g_ids = {id(p) for p in netG.parameters()}
    covered = {id(p) for pg in opt_g.param_groups for p in pg['params']}
    assert g_ids.issubset(covered)


def test_build_optimizers_encoder_lr_scaled(cfg):
    netG, netD = factory.build_models(cfg, encoder=MockSwinBackbone())
    crits = factory.build_criterions(cfg)
    opt_d, opt_g = factory.build_optimizers(cfg, netG, netD, criterions=crits)
    # group 0 = encoder, group 1 = fresh
    enc_lr = opt_g.param_groups[0]['lr']
    fresh_lr = opt_g.param_groups[1]['lr']
    scale = float(cfg.model.gen.encoder_lr_scale)
    assert abs(enc_lr / fresh_lr - scale) < 1e-9


@pytest.mark.xfail(reason=_REMOVED, run=False, strict=False)
def test_build_optimizers_speckle_cons_params_in_g(cfg):
    """The reflectivity head's params must be in opt_g's fresh group, not in opt_d."""
    netG, netD = factory.build_models(cfg, encoder=MockSwinBackbone())
    crits = factory.build_criterions(cfg)
    sc = crits['speckle_cons']
    sc_ids = {id(p) for p in sc.parameters()}
    opt_d, opt_g = factory.build_optimizers(cfg, netG, netD, criterions=crits)
    g_ids = {id(p) for pg in opt_g.param_groups for p in pg['params']}
    d_ids = {id(p) for pg in opt_d.param_groups for p in pg['params']}
    assert sc_ids.issubset(g_ids)
    assert not (sc_ids & d_ids)


@pytest.mark.xfail(reason=_REMOVED, run=False, strict=False)
def test_build_optimizers_phi_disabled_returns_none(cfg):
    cfg2 = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    cfg2.model.dis.phi.enable = False
    netG, netD_main, netD_phi = factory.build_models(cfg2, encoder=MockSwinBackbone())
    crits = factory.build_criterions(cfg2)
    opt_g, opt_d_main, opt_d_phi = factory.build_optimizers(
        cfg2, netG, netD_main, netD_phi, criterions=crits,
    )
    assert opt_d_phi is None


# --------------------------------------------------------------------- schedulers


def test_build_schedulers_two_returns(cfg):
    netG, netD = factory.build_models(cfg, encoder=MockSwinBackbone())
    crits = factory.build_criterions(cfg)
    opt_d, opt_g = factory.build_optimizers(cfg, netG, netD, criterions=crits)
    sched_d, sched_g = factory.build_lr_schedulers(cfg, opt_d, opt_g)
    assert sched_g is not None
    assert sched_d is not None


def test_build_ema_callback_present(cfg):
    cb = factory.build_ema_callback(cfg)
    assert cb is not None
