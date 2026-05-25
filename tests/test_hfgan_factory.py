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


def test_build_criterions_l1_enabled_when_weight_positive(cfg):
    from src.models.huggingface_gan.factory import build_criterions
    cfg2 = OmegaConf.merge(cfg, {'loss': {'l1_weight': 100.0, 'fft_weight': 0.0, 'perceptual_weight': 0.0}})
    c = build_criterions(cfg2)
    assert 'l1' in c


def test_build_criterions_no_l1_when_zero(cfg):
    from src.models.huggingface_gan.factory import build_criterions
    cfg2 = OmegaConf.merge(cfg, {'loss': {'l1_weight': 0.0, 'fft_weight': 0.0, 'perceptual_weight': 0.0}})
    c = build_criterions(cfg2)
    assert 'l1' not in c


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


# ---------------------------------------------------------------------------
# AdaptiveLoss wiring + EMA callback + param-coverage assertion
# ---------------------------------------------------------------------------

def test_build_criterions_adaptive_wraps_recon_losses(cfg):
    from src.models.huggingface_gan.factory import build_criterions
    cfg2 = OmegaConf.merge(cfg, {'loss': {
        'l1_weight': 10.0, 'fft_weight': 0.5, 'perceptual_weight': 0.0,
        'adaptive_balance': True, 'adaptive_eta_max': 5.0,
    }})
    c = build_criterions(cfg2)
    assert 'adaptive' in c
    # 2 recon losses present (l1, fft); perceptual disabled
    assert c['adaptive'].eta.numel() == 2
    assert c['adaptive']._loss_order == ['l1', 'fft']


def test_build_criterions_no_adaptive_when_flag_off(cfg):
    from src.models.huggingface_gan.factory import build_criterions
    cfg2 = OmegaConf.merge(cfg, {'loss': {
        'l1_weight': 10.0, 'fft_weight': 0.5, 'perceptual_weight': 0.0,
        'adaptive_balance': False,
    }})
    c = build_criterions(cfg2)
    assert 'adaptive' not in c


def test_build_criterions_no_adaptive_when_only_one_recon_loss(cfg):
    """AdaptiveLoss with n=1 is degenerate — should be skipped."""
    from src.models.huggingface_gan.factory import build_criterions
    cfg2 = OmegaConf.merge(cfg, {'loss': {
        'l1_weight': 10.0, 'fft_weight': 0.0, 'perceptual_weight': 0.0,
        'adaptive_balance': True,
    }})
    c = build_criterions(cfg2)
    assert 'adaptive' not in c


def test_build_optimizers_includes_adaptive_eta(cfg):
    """When adaptive_balance is enabled, eta params must be in opt_g fresh group."""
    from src.models.huggingface_gan.factory import (
        build_models, build_criterions, build_optimizers,
    )
    cfg2 = OmegaConf.merge(cfg, {'loss': {
        'l1_weight': 10.0, 'fft_weight': 0.5, 'perceptual_weight': 0.0,
        'adaptive_balance': True, 'adaptive_eta_max': 5.0,
    }})
    netG, netD = build_models(cfg2, encoder=MockBackbone())
    crits = build_criterions(cfg2)
    opt_g, _ = build_optimizers(cfg2, netG, netD, criterions=crits)
    fresh_group_ids = {id(p) for p in opt_g.param_groups[1]['params']}
    assert id(crits['adaptive'].eta) in fresh_group_ids


def test_build_optimizers_param_coverage_assertion(cfg):
    """Every netG parameter must live in some opt_g group."""
    from src.models.huggingface_gan.factory import build_models, build_optimizers
    netG, netD = build_models(cfg, encoder=MockBackbone())
    opt_g, _ = build_optimizers(cfg, netG, netD)
    g_ids = {id(p) for p in netG.parameters()}
    opt_ids = {id(p) for pg in opt_g.param_groups for p in pg['params']}
    assert g_ids <= opt_ids, f"netG params missing from optimizer: {len(g_ids - opt_ids)}"


def test_build_ema_callback_off_returns_none(cfg):
    from src.models.huggingface_gan.factory import build_ema_callback
    cfg2 = OmegaConf.merge(cfg, {'ema': {'use_ema': False}})
    assert build_ema_callback(cfg2) is None


def test_build_ema_callback_on_returns_instance(cfg):
    pytest.importorskip('lightning.pytorch')
    from src.models.huggingface_gan.factory import build_ema_callback
    from src.utils.callbacks import EMAWeightAveraging
    cfg2 = OmegaConf.merge(cfg, {'ema': {
        'use_ema': True, 'decay': 0.999, 'start_epoch': 10,
    }})
    cb = build_ema_callback(cfg2)
    assert isinstance(cb, EMAWeightAveraging)
    assert cb.update_starting_at_epoch == 10


# ---------------------------------------------------------------------------
# hfgan-18: hybrid generator param coverage + new criterions
# ---------------------------------------------------------------------------

def test_build_criterions_msssim_enabled(cfg):
    from src.models.huggingface_gan.factory import build_criterions
    from src.models.huggingface_gan.losses import MSSSIMLoss
    cfg2 = OmegaConf.merge(cfg, {'loss': {
        'msssim_weight': 1.0, 'lab_chroma_weight': 0.0, 'wavelet_weight': 0.0,
        'fft_weight': 0.0, 'perceptual_weight': 0.0,
    }})
    c = build_criterions(cfg2)
    assert 'msssim' in c
    assert isinstance(c['msssim'], MSSSIMLoss)
    assert 'lab_chroma' not in c
    assert 'wavelet' not in c


def test_build_criterions_lab_chroma_enabled(cfg):
    from src.models.huggingface_gan.factory import build_criterions
    from src.models.huggingface_gan.losses import LABChromaL1Loss
    cfg2 = OmegaConf.merge(cfg, {'loss': {
        'msssim_weight': 0.0, 'lab_chroma_weight': 5.0, 'wavelet_weight': 0.0,
        'fft_weight': 0.0, 'perceptual_weight': 0.0,
    }})
    c = build_criterions(cfg2)
    assert isinstance(c['lab_chroma'], LABChromaL1Loss)


def test_build_criterions_wavelet_enabled(cfg):
    from src.models.huggingface_gan.factory import build_criterions
    from src.models.huggingface_gan.losses import WaveletDetailL1Loss
    cfg2 = OmegaConf.merge(cfg, {'loss': {
        'msssim_weight': 0.0, 'lab_chroma_weight': 0.0, 'wavelet_weight': 1.0,
        'fft_weight': 0.0, 'perceptual_weight': 0.0,
    }})
    c = build_criterions(cfg2)
    assert isinstance(c['wavelet'], WaveletDetailL1Loss)


def test_build_criterions_no_new_losses_when_zero(cfg):
    """msssim/lab_chroma/wavelet absent when all weights are 0."""
    from src.models.huggingface_gan.factory import build_criterions
    cfg2 = OmegaConf.merge(cfg, {'loss': {
        'msssim_weight': 0.0, 'lab_chroma_weight': 0.0, 'wavelet_weight': 0.0,
        'fft_weight': 0.0, 'perceptual_weight': 0.0,
    }})
    c = build_criterions(cfg2)
    for k in ('msssim', 'lab_chroma', 'wavelet'):
        assert k not in c


def test_build_optimizers_covers_hfcf_branch_and_fusion(cfg):
    """HFCFBranch params, hfcf_final, cfr_final, and _fusion_logit all in optG."""
    from src.models.huggingface_gan.factory import build_models, build_optimizers
    netG, netD = build_models(cfg, encoder=MockBackbone())
    opt_g, _ = build_optimizers(cfg, netG, netD)
    opt_ids = {id(p) for pg in opt_g.param_groups for p in pg['params']}
    # _fusion_logit scalar
    assert id(netG._fusion_logit) in opt_ids
    # hfcf_branch (must have some params)
    hfcf_params = list(netG.hfcf_branch.parameters())
    assert len(hfcf_params) > 0
    for p in hfcf_params:
        assert id(p) in opt_ids
    # hfcf_final + cfr_final
    for p in netG.hfcf_final.parameters():
        assert id(p) in opt_ids
    for p in netG.cfr_final.parameters():
        assert id(p) in opt_ids
