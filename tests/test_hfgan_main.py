import pytest
import torch
from omegaconf import OmegaConf
import sys, os
sys.path.insert(0, os.path.abspath('.'))
from tests.conftest_hfgan import MockBackbone


@pytest.fixture(scope='module')
def cfg():
    c = OmegaConf.load('src/models/huggingface_gan/config.yaml')
    return OmegaConf.merge(c, {'loss': {'fft_weight': 0.0, 'perceptual_weight': 0.0}})

@pytest.fixture(scope='module')
def device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

@pytest.fixture(scope='module')
def module(cfg, device):
    from src.models.huggingface_gan.main import SAR2OPTLightningModule
    return SAR2OPTLightningModule(cfg, encoder=MockBackbone()).to(device)


def test_configure_optimizers_returns_two(module):
    opts = module.configure_optimizers()
    assert len(opts) == 2

def test_configure_optimizers_g_has_two_param_groups(module):
    opts = module.configure_optimizers()
    _, opt_g = opts
    assert len(opt_g.param_groups) == 2

def test_criterions_are_moduledict(module):
    import torch.nn as nn
    assert isinstance(module.criterions, nn.ModuleDict)
    assert 'gan' in module.criterions
    assert 'fm'  in module.criterions
    assert 'fft' not in module.criterions

def test_d_loss_is_finite(module, device):
    sar = torch.randn(2, 1, 256, 256, device=device)
    opt = torch.randn(2, 3, 256, 256, device=device)
    with torch.no_grad():
        fake_d = module.netG(sar)
    real_logits, _ = module.netD(sar, opt)
    fake_logits, _ = module.netD(sar, fake_d)
    d_loss = 0.5 * (
        module.criterions['gan'](real_logits, is_real=True) +
        module.criterions['gan'](fake_logits, is_real=False)
    )
    assert torch.isfinite(d_loss)
    assert d_loss.item() >= 0.0

def test_g_loss_is_finite(module, device):
    sar = torch.randn(2, 1, 256, 256, device=device)
    opt = torch.randn(2, 3, 256, 256, device=device)
    with torch.no_grad():
        fake_d = module.netG(sar)
    _, real_feats = module.netD(sar, opt)
    fake = module.netG(sar)
    fake_logits, fake_feats = module.netD(sar, fake)
    real_feats_d = [f.detach() for f in real_feats]
    cfg = module.cfg.loss
    g_loss = (
        module.criterions['gan'](fake_logits, is_real=True) * cfg.gan_weight +
        module.criterions['fm'](fake_feats, real_feats_d)   * cfg.fm_weight
    )
    assert torch.isfinite(g_loss)
    assert g_loss.item() >= 0.0

def test_validation_step_updates_metrics(module, device):
    sar = torch.randn(2, 1, 256, 256, device=device)
    opt = torch.randn(2, 3, 256, 256, device=device)
    batch = {'sar': sar, 'optical': opt}
    module.validation_step(batch, 0)
    psnr_val = module.psnr.compute()
    assert torch.isfinite(psnr_val)
    module.psnr.reset()
    module.ssim.reset()
