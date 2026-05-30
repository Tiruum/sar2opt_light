import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR, SequentialLR, ConstantLR

from src.models.huggingface_gan.gen import HFGenerator
from src.models.huggingface_gan.dis import HFGANDiscriminator
from src.models.huggingface_gan.losses import GANLoss, FeatureMatchingLoss, FFTLoss


def build_models(cfg, encoder=None):
    netG = HFGenerator(cfg, encoder=encoder)
    netD = HFGANDiscriminator(
        in_ch=cfg.model.dis.in_channels,
        ndf=cfg.model.dis.ndf,
    )
    return netG, netD


def build_criterions(cfg) -> dict:
    """Optional losses instantiated only when weight > 0.
    PerceptualLoss loads a 28M frozen backbone — never instantiate at weight=0."""
    criterions = {
        'gan': GANLoss(),
        'fm':  FeatureMatchingLoss(),
    }
    if cfg.loss.fft_weight > 0:
        criterions['fft'] = FFTLoss()
    if cfg.loss.perceptual_weight > 0:
        from src.models.huggingface_gan.losses import PerceptualLoss
        criterions['perceptual'] = PerceptualLoss(cfg.model.gen.backbone)
    return criterions


def build_optimizers(cfg, netG, netD):
    """AdamW for G with two param groups (encoder at 0.1× LR). Adam for D."""
    enc_params = (
        list(netG.channel_adapter.parameters()) +
        list(netG.encoder.parameters())
    )
    fresh_params = (
        list(netG.bottleneck.parameters()) +
        list(netG.up4.parameters()) +
        list(netG.up3.parameters()) +
        list(netG.up2.parameters()) +
        list(netG.up1.parameters()) +
        list(netG.up0.parameters()) +
        list(netG.head.parameters())
    )
    opt_g = optim.AdamW(
        [
            {'params': enc_params,   'lr': cfg.optimizer.lr_g * 0.1},
            {'params': fresh_params, 'lr': cfg.optimizer.lr_g},
        ],
        betas=(cfg.optimizer.beta1, cfg.optimizer.beta2),
        weight_decay=cfg.optimizer.weight_decay_g,
    )
    opt_d = optim.Adam(
        netD.parameters(),
        lr=cfg.optimizer.lr_d,
        betas=(cfg.optimizer.beta1, cfg.optimizer.beta2),
    )
    return opt_g, opt_d


def build_lr_schedulers(cfg, opt_g, opt_d):
    """Flat LR then linear decay to eta_min."""
    decay  = cfg.scheduler.linear_decay_epochs
    warmup = max(cfg.system.max_epochs - decay, 0)

    def make_sched(opt, base_lr):
        end_factor = cfg.scheduler.eta_min / max(base_lr, 1e-10)
        linear = LinearLR(opt, start_factor=1.0, end_factor=end_factor, total_iters=decay)
        if warmup == 0:
            return linear
        return SequentialLR(
            opt,
            schedulers=[ConstantLR(opt, factor=1.0, total_iters=warmup), linear],
            milestones=[warmup],
        )

    return make_sched(opt_g, cfg.optimizer.lr_g), make_sched(opt_d, cfg.optimizer.lr_d)
