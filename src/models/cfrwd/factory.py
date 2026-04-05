from typing import Literal
import torch.optim as optim
from src.models.cfrwd.discriminator import CFRWDPatchDis
from src.models.cfrwd.gen import CFRWDGenerator
from src.models.cfrwd.losses import FeatureMatchingLoss, GANLoss, L1Loss, FFTLoss, LPIPSLoss
import torch.nn as nn
from functools import lru_cache
from omegaconf import OmegaConf

_CFG_PATH = 'src/models/cfrwd/config.yaml'

@lru_cache(maxsize=1)
def _load_cfg():
    return OmegaConf.load(_CFG_PATH)

def build_models():
    cfg = _load_cfg()
    netG = CFRWDGenerator(in_channels=cfg.model.gen.in_channels)
    netD = CFRWDPatchDis(in_channels=cfg.model.dis.in_channels, ndf=cfg.model.dis.ndf, return_features=True)
    return netG, netD

def build_optimizers(netG, netD,
                     extra_g_params=None,
                     lr_g: float = None,
                     lr_d: float = None,
                     beta1: float = None,
                     beta2: float = None):
    cfg = _load_cfg()
    if lr_g  is None: lr_g  = cfg.optimizer.lr_g
    if lr_d  is None: lr_d  = cfg.optimizer.lr_d
    if beta1 is None: beta1 = cfg.optimizer.beta1
    if beta2 is None: beta2 = cfg.optimizer.beta2
    # extra_g_params: обучаемые параметры не из netG (напр. AdaptiveLoss.eta)
    g_params = list(netG.parameters())
    if extra_g_params is not None:
        g_params += list(extra_g_params)
    optG = optim.Adam(g_params, lr=lr_g, betas=(beta1, beta2))
    optD = optim.Adam(netD.parameters(), lr=lr_d, betas=(beta1, beta2))
    return optG, optD

def build_criterions() -> dict[Literal['GAN', 'FM', 'L1', 'FFT', 'LPIPS'], nn.Module]:
    cfg = _load_cfg()
    crits = {
        'GAN': GANLoss(use_lsgan=True),
        'FM':  FeatureMatchingLoss(),
        'L1':  L1Loss(),
        'FFT': FFTLoss(),
    }
    if cfg.loss.get('use_lpips', False):
        crits['LPIPS'] = LPIPSLoss()
    return crits

def build_lr_schedulers(optG, optD,
                        eta_min: float = None,
                        decay_last_epochs: int = None):
    cfg = _load_cfg()
    if eta_min is None:
        eta_min = cfg.scheduler.eta_min
    if decay_last_epochs is None:
        decay_last_epochs = cfg.scheduler.linear_decay_epochs
    total_epochs = max(int(cfg.system.max_epochs), 1)
    decay_epochs = min(max(int(decay_last_epochs), 0), total_epochs)
    start_decay_epoch = total_epochs - decay_epochs

    def _linear_decay_lambda(initial_lr: float):
        if initial_lr <= 0 or decay_epochs == 0:
            return lambda epoch: 1.0

        min_factor = max(0.0, min(1.0, eta_min / initial_lr))

        def lr_lambda(epoch: int):
            clamped_epoch = min(max(int(epoch), 0), total_epochs - 1)

            if clamped_epoch < start_decay_epoch:
                return 1.0

            if decay_epochs == 1:
                return min_factor

            progress = (clamped_epoch - start_decay_epoch) / (decay_epochs - 1)
            return (1.0 - progress) * (1.0 - min_factor) + min_factor

        return lr_lambda

    scheduler_G = optim.lr_scheduler.LambdaLR(
        optG,
        lr_lambda=[_linear_decay_lambda(pg['lr']) for pg in optG.param_groups],
    )
    scheduler_D = optim.lr_scheduler.LambdaLR(
        optD,
        lr_lambda=[_linear_decay_lambda(pg['lr']) for pg in optD.param_groups],
    )
    return scheduler_G, scheduler_D