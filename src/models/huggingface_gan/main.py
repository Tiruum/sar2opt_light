import torch
import torch.nn as nn
import lightning.pytorch as pl
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.fid import FrechetInceptionDistance

from src.models.huggingface_gan import factory


class SAR2OPTLightningModule(pl.LightningModule):
    def __init__(self, cfg, encoder=None):
        super().__init__()
        self.cfg = cfg
        self.automatic_optimization = False

        self.netG, self.netD = factory.build_models(cfg, encoder=encoder)
        self.criterions = nn.ModuleDict(factory.build_criterions(cfg))

        self.psnr  = PeakSignalNoiseRatio(data_range=2.0)
        self.ssim  = StructuralSimilarityIndexMeasure(data_range=2.0)
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=False)
        self.fid   = FrechetInceptionDistance(feature=2048, reset_real_features=False, normalize=True)

    def configure_optimizers(self):
        opt_g, opt_d = factory.build_optimizers(self.cfg, self.netG, self.netD)
        self.sched_g, self.sched_d = factory.build_lr_schedulers(self.cfg, opt_g, opt_d)
        return [opt_d, opt_g]

    def training_step(self, batch, batch_idx):
        sar, opt     = batch
        opt_d, opt_g = self.optimizers()
        loss_cfg     = self.cfg.loss

        with torch.no_grad():
            fake_d = self.netG(sar)
        real_logits, real_feats = self.netD(sar, opt)
        fake_logits, _          = self.netD(sar, fake_d)
        d_loss = 0.5 * (
            self.criterions['gan'](real_logits, is_real=True) +
            self.criterions['gan'](fake_logits, is_real=False)
        )
        opt_d.zero_grad()
        self.manual_backward(d_loss)
        opt_d.step()

        fake = self.netG(sar)
        fake_logits_g, fake_feats = self.netD(sar, fake)
        real_feats_d = [f.detach() for f in real_feats]
        g_loss = (
            self.criterions['gan'](fake_logits_g, is_real=True) * loss_cfg.gan_weight +
            self.criterions['fm'](fake_feats, real_feats_d)      * loss_cfg.fm_weight
        )
        if 'fft' in self.criterions:
            g_loss = g_loss + self.criterions['fft'](fake, opt) * loss_cfg.fft_weight
        if 'perceptual' in self.criterions:
            g_loss = g_loss + self.criterions['perceptual'](fake, opt) * loss_cfg.perceptual_weight
        opt_g.zero_grad()
        self.manual_backward(g_loss)
        opt_g.step()

        self.log_dict({'train/d_loss': d_loss, 'train/g_loss': g_loss},
                      prog_bar=True, on_step=True, on_epoch=False)

    def validation_step(self, batch, batch_idx):
        sar, opt = batch
        with torch.no_grad():
            fake = self.netG(sar)
        self.psnr.update(fake, opt)
        self.ssim.update(fake, opt)
        self.lpips.update(fake.float(), opt.float())
        fake_01 = ((fake + 1) / 2).clamp(0, 1).float()
        opt_01  = ((opt  + 1) / 2).clamp(0, 1).float()
        self.fid.update(fake_01, real=False)
        if self.fid.real_features_num_samples == 0:
            self.fid.update(opt_01, real=True)

    def on_validation_epoch_end(self):
        self.log_dict({
            'val/psnr':  self.psnr.compute(),
            'val/ssim':  self.ssim.compute(),
            'val/lpips': self.lpips.compute(),
            'val/fid':   self.fid.compute(),
        }, prog_bar=True)
        self.psnr.reset()
        self.ssim.reset()
        self.lpips.reset()
        self.fid.reset()  # reset_real_features=False → only clears fake stats

    def on_train_epoch_end(self):
        self.sched_g.step()
        self.sched_d.step()
