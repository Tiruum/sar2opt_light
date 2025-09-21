from typing import Literal
import torch.optim as optim
from src.models.pix2pix.generator import UNetGenerator
from src.models.pix2pix.multiscale_discriminator import MultiscaleDiscriminator
from src.models.pix2pix.losses import GANLoss, L1Loss
import torch.nn as nn
from omegaconf import OmegaConf
cfg = OmegaConf.load('src/models/pix2pix/config.yaml')


def build_models(
    input_nc: int = cfg.model.input_nc,
    output_nc: int = cfg.model.output_nc,
    ngf: int = cfg.model.ngf,
    ndf: int = cfg.model.ndf,
    n_blocks: int = cfg.model.n_blocks,
    n_layers: int = cfg.model.n_layers,
    num_D: int = cfg.model.num_D):
    """Builds the generator and discriminator models for the GAN architecture.
    Args:
        input_nc (int): Number of input channels (default: config.INPUT_NC).
        output_nc (int): Number of output channels (default: config.OUTPUT_NC).
        
    Returns:
        tuple: A tuple containing the generator and discriminator models.
    """
    netG = UNetGenerator(
        input_nc=input_nc,
        output_nc=output_nc,
        ngf=ngf,
        n_blocks=n_blocks
    )

    netD = MultiscaleDiscriminator(
        input_nc=input_nc + output_nc,
        ndf=ndf,
        n_layers=n_layers,
        num_D=num_D
    )
    return netG, netD

def build_optimizers(netG, netD,
                     lr_g: float = cfg.optimizer.lr_g,
                     lr_d: float = cfg.optimizer.lr_d,
                     beta1: float = cfg.optimizer.beta1,
                     beta2: float = cfg.optimizer.beta2):
    optG = optim.Adam(netG.parameters(), lr=lr_g, betas=(beta1, beta2))
    optD = optim.Adam(netD.parameters(), lr=lr_d, betas=(beta1, beta2))
    return optG, optD

def build_criterions() -> dict[Literal['GAN', 'L1'], nn.Module]:
    crits = {}
    crits['GAN'] = GANLoss(use_lsgan=True)
    crits['L1'] = L1Loss()
    return crits

def build_lr_schedulers(optG, optD,
                        eta_min: float = cfg.scheduler.eta_min):
    scheduler_G = optim.lr_scheduler.CosineAnnealingLR(optG, T_max=cfg.system.max_epochs, eta_min=eta_min)
    scheduler_D = optim.lr_scheduler.CosineAnnealingLR(optD, T_max=cfg.system.max_epochs, eta_min=eta_min)
    return scheduler_G, scheduler_D