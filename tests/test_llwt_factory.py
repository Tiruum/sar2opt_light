"""Tests for ``src/models/llwt/factory.py``.

Goal: verify the factory builds a consistent model + criterion + optimiser
set with the parameter-coverage assertion, and that the criterion gating
behaves as documented (l1_weight=0 -> no l1 in criterions etc.).
"""
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from src.models.llwt import factory


def _full_cfg(image_size: int = 64, embed_dim: int = 24, depth: int = 1,
              win: int = 8, l1: float = 50.0, spkdec: float = 0.05,
              pr: float = 0.01, lpips: float = 1.0, use_sub: bool = True):
    """Build a synthetic ``OmegaConf`` cfg covering everything the factory
    reads.  Keeps numbers small for fast smoke."""
    cfg = OmegaConf.create({
        'data':   {'image_size': image_size},
        'model':  {
            'gen': {
                'embed_dim':       embed_dim,
                'lifting_levels':  3,
                'lifting_hidden':  8,
                'pswe':    {'window_size': win, 'heads': 4, 'depth': depth,
                            'shared_blocks': False},
                'cba_fus': {'levels': [2, 3], 'heads': 4},
                'pcsg':    {'enabled': True, 'gain_ll': 0.1},
            },
            'dis': {
                'main':    {'in_channels': 4, 'ndf': 32},
                'subband': {'enabled': use_sub, 'ndf': 16},
            },
        },
        'optimizer': {'lr_g': 2.0e-4, 'lr_d': 1.0e-4, 'beta1': 0.5,
                       'beta2': 0.999, 'weight_decay_g': 0.0},
        'scheduler': {'type': 'cosine_warm_restarts', 'eta_min': 1.0e-6,
                       't_0': 20, 't_mult': 2, 'linear_decay_epochs': 60},
        'ema':       {'use_ema': True, 'decay': 0.999, 'start_epoch': 10},
        'loss': {
            'gan_main_weight': 1.0, 'gan_sub_weight': 0.5,
            'fm_main_weight': 10.0, 'fm_sub_weight': 5.0,
            'l1_weight': l1, 'msssim_weight': 1.0,
            'lab_chroma_weight': 5.0, 'wavelet_weight': 2.0,
            'lpips_weight': lpips,
            'spkdec_weight': spkdec, 'pr_weight': pr,
            'adaptive_balance': False, 'adaptive_eta_max': 5.0,
            'r1_gamma': 0.5, 'r1_main_every': 16,
            'grad_clip_g': 1.0, 'grad_clip_d': 5.0,
        },
        'system': {'max_epochs': 120, 'tb_version': 'test'},
    })
    return cfg


def test_build_models(rng_seed=None):
    cfg = _full_cfg()
    netG, netD = factory.build_models(cfg)
    # Just check they are nn.Module instances and have parameters.
    assert sum(p.numel() for p in netG.parameters()) > 0
    assert sum(p.numel() for p in netD.parameters()) > 0
    assert netD.use_sub is True


def test_build_models_subband_disabled():
    cfg = _full_cfg(use_sub=False)
    netG, netD = factory.build_models(cfg)
    assert netD.use_sub is False
    assert netD.sub is None


def test_build_criterions_default_keys():
    cfg = _full_cfg()
    crits = factory.build_criterions(cfg)
    for k in ('gan', 'fm', 'l1', 'msssim', 'lab_chroma', 'wavelet', 'lpips',
              'spkdec', 'pr'):
        assert k in crits, f"missing criterion: {k}"


def test_build_criterions_gates_on_zero_weight():
    cfg = _full_cfg(l1=0.0, spkdec=0.0, pr=0.0, lpips=0.0)
    crits = factory.build_criterions(cfg)
    for k in ('l1', 'spkdec', 'pr', 'lpips'):
        assert k not in crits


def test_build_optimizers_param_coverage():
    cfg = _full_cfg()
    netG, netD = factory.build_models(cfg)
    crits = factory.build_criterions(cfg)
    opt_d, opt_g = factory.build_optimizers(cfg, netG, netD, criterions=crits)
    # Param-coverage: every G param must be reachable from opt_g.
    expected = {id(p) for p in netG.parameters()}
    got = {id(p) for pg in opt_g.param_groups for p in pg['params']}
    assert expected.issubset(got)


def test_build_lr_schedulers_cosine():
    cfg = _full_cfg()
    netG, netD = factory.build_models(cfg)
    opt_d, opt_g = factory.build_optimizers(cfg, netG, netD)
    sched_d, sched_g = factory.build_lr_schedulers(cfg, opt_d, opt_g)
    # Single .step() call should not crash.
    sched_g.step()
    sched_d.step()


def test_build_lr_schedulers_linear_decay():
    cfg = _full_cfg()
    cfg.scheduler.type = 'linear_decay'
    netG, netD = factory.build_models(cfg)
    opt_d, opt_g = factory.build_optimizers(cfg, netG, netD)
    sched_d, sched_g = factory.build_lr_schedulers(cfg, opt_d, opt_g)
    sched_g.step()
    sched_d.step()


def test_build_lr_schedulers_unknown_raises():
    cfg = _full_cfg()
    cfg.scheduler.type = 'snake_oil'
    netG, netD = factory.build_models(cfg)
    opt_d, opt_g = factory.build_optimizers(cfg, netG, netD)
    with pytest.raises(ValueError):
        factory.build_lr_schedulers(cfg, opt_d, opt_g)


def test_build_ema_callback_enabled():
    cfg = _full_cfg()
    cb = factory.build_ema_callback(cfg)
    assert cb is not None


def test_build_ema_callback_disabled():
    cfg = _full_cfg()
    cfg.ema.use_ema = False
    cb = factory.build_ema_callback(cfg)
    assert cb is None
