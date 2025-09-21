from typing import Literal
import torch.optim as optim
from src.models.cfrwd.dis import CFRWDPatchDis
from src.models.cfrwd.gen import CFRWDGenerator
from src.models.cfrwd.losses import FeatureMatchingLoss, GANLoss, L1Loss
import torch.nn as nn
from omegaconf import OmegaConf

cfg = OmegaConf.load('src/models/cfrwd/config.yaml')

def build_models():
    """Builds the generator and discriminator models for the GAN architecture.
        
    Returns:
        tuple: A tuple containing the generator and discriminator models.
    """

    netG = CFRWDGenerator(image_size=cfg.data.image_size, hfcf_concat_type='cat')
    netD = CFRWDPatchDis(input_channels=4, condition_channels=3, return_features=True)

    return netG, netD

def build_optimizers(netG, netD,
                     lr_g: float = cfg.optimizer.lr_g,
                     lr_d: float = cfg.optimizer.lr_d,
                     beta1: float = cfg.optimizer.beta1,
                     beta2: float = cfg.optimizer.beta2):
    optG = optim.Adam(netG.parameters(), lr=lr_g, betas=(beta1, beta2))
    optD = optim.Adam(netD.parameters(), lr=lr_d, betas=(beta1, beta2))
    return optG, optD

def build_criterions() -> dict[Literal['GAN', 'FM'], nn.Module]:
    crits = {}
    crits['GAN'] = GANLoss(use_lsgan=True)
    crits['FM'] = FeatureMatchingLoss()
    crits['L1'] = L1Loss()
    return crits

def build_lr_schedulers(optG, optD,
                        eta_min: float = cfg.scheduler.eta_min):
    scheduler_G = optim.lr_scheduler.CosineAnnealingLR(optG, T_max=cfg.system.max_epochs, eta_min=eta_min)
    scheduler_D = optim.lr_scheduler.CosineAnnealingLR(optD, T_max=cfg.system.max_epochs, eta_min=eta_min)
    return scheduler_G, scheduler_D