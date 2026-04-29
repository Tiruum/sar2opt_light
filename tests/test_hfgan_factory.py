import pytest
import torch
from omegaconf import OmegaConf
import sys, os
sys.path.insert(0, os.path.abspath('.'))
from tests.conftest_hfgan import MockBackbone


@pytest.fixture(scope='module')
def cfg():
    return OmegaConf.load('src/models/huggingface_gan/config.yaml')


def test_build_models_types(cfg):
    from src.models.huggingface_gan.factory import build_models
    from src.models.huggingface_gan.gen import HFGenerator
    from src.models.huggingface_gan.dis import HFGANDiscriminator
    netG, netD = build_models(cfg, encoder=MockBackbone())
    assert isinstance(netG, HFGenerator)
    assert isinstance(netD, HFGANDiscriminator)


def test_build_criterions_core_always_present(cfg):
    from src.models.huggingface_gan.factory import build_criterions
    cfg2 = OmegaConf.merge(cfg, {'loss': {'fft_weight': 0.0, 'perceptual_weight': 0.0}})
    c = build_criterions(cfg2)
    assert 'gan' in c
    assert 'fm'  in c
    assert 'fft' not in c
    assert 'perceptual' not in c


def test_build_criterions_fft_enabled(cfg):
    from src.models.huggingface_gan.factory import build_criterions
    cfg2 = OmegaConf.merge(cfg, {'loss': {'fft_weight': 1.0, 'perceptual_weight': 0.0}})
    c = build_criterions(cfg2)
    assert 'fft' in c
    assert 'perceptual' not in c


def test_build_criterions_no_perceptual_when_zero(cfg):
    from src.models.huggingface_gan.factory import build_criterions
    cfg2 = OmegaConf.merge(cfg, {'loss': {'fft_weight': 0.0, 'perceptual_weight': 0.0}})
    c = build_criterions(cfg2)
    assert 'perceptual' not in c


def test_build_optimizers_returns_two(cfg):
    from src.models.huggingface_gan.factory import build_models, build_optimizers
    netG, netD = build_models(cfg, encoder=MockBackbone())
    opt_g, opt_d = build_optimizers(cfg, netG, netD)
    assert opt_g is not None
    assert opt_d is not None


def test_build_optimizers_g_has_two_param_groups(cfg):
    from src.models.huggingface_gan.factory import build_models, build_optimizers
    netG, netD = build_models(cfg, encoder=MockBackbone())
    opt_g, _ = build_optimizers(cfg, netG, netD)
    assert len(opt_g.param_groups) == 2


def test_build_optimizers_encoder_lr_is_tenth(cfg):
    from src.models.huggingface_gan.factory import build_models, build_optimizers
    netG, netD = build_models(cfg, encoder=MockBackbone())
    opt_g, _ = build_optimizers(cfg, netG, netD)
    enc_lr  = opt_g.param_groups[0]['lr']
    full_lr = opt_g.param_groups[1]['lr']
    assert full_lr == pytest.approx(cfg.optimizer.lr_g, rel=1e-5)
    assert enc_lr  == pytest.approx(cfg.optimizer.lr_g * 0.1, rel=1e-5)


def test_build_lr_schedulers_returns_two(cfg):
    from src.models.huggingface_gan.factory import build_models, build_optimizers, build_lr_schedulers
    netG, netD = build_models(cfg, encoder=MockBackbone())
    opt_g, opt_d = build_optimizers(cfg, netG, netD)
    sched_g, sched_d = build_lr_schedulers(cfg, opt_g, opt_d)
    assert sched_g is not None
    assert sched_d is not None
