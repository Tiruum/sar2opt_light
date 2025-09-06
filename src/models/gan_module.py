# src/models/lightning/gan_module.py

import gc
import torch
import pytorch_lightning as pl
from torch.amp import autocast
from src.models.spade_generator import UNetGenerator
from src.utils.factory import build_models, build_optimizers, build_criterions, build_lr_schedulers
from src.utils.visualize import visualize_batch
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

class SAR2OPTGANLightningModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.cfg = config
        self.save_hyperparameters()
        
        # Инициализация моделей и критерионов
        _, self.netD = build_models(
            input_nc = self.cfg.model.input_nc,
            output_nc = self.cfg.model.output_nc,
            ngf = self.cfg.model.ngf,
            ndf = self.cfg.model.ndf,
            n_blocks = self.cfg.model.n_blocks,
            n_layers = self.cfg.model.n_layers,
            num_D = self.cfg.model.num_D
        )
        self.netG = UNetGenerator()
        self.criterions = build_criterions()
        
        # Веса лоссов
        self.loss_weights = {
            'gan': self.cfg.loss.gan_weight,
            'l1': self.cfg.loss.l1_weight,
        }

        self.psnr = PeakSignalNoiseRatio(data_range=2.0).to(self.device)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=2.0).to(self.device)
        
        self.fixed_sar = None
        self.fixed_opt = None
        
        self.automatic_optimization = False

    def forward(self, x):
        return self.netG(x)
    
    def setup(self, stage=None):
        dm = self.trainer.datamodule
        train_loader = dm.train_dataloader()
        sar, opt = next(iter(train_loader))
        self.fixed_sar = sar.to(self.device)
        self.fixed_opt = opt.to(self.device)
    
    def training_step(self, batch, batch_idx):
        real_sar, real_optical = batch
        opt_d, opt_g = self.optimizers()
        
        # Discriminator step
        opt_d.zero_grad(set_to_none=True)
        with autocast(device_type=self.device.type, enabled=self.trainer.precision == 16):
            fake_opt = self.netG(real_sar).detach()
            fake_pair = torch.cat([real_sar, fake_opt], dim=1).contiguous()
            real_pair = torch.cat([real_sar, real_optical], dim=1).contiguous()

            pred_fake = self.netD(fake_pair)
            pred_real = self.netD(real_pair)

            d_loss = 0.5 * (
                sum([self.criterions['GAN'](fake, False) for fake in pred_fake]) +
                sum([self.criterions['GAN'](real, True, real_label_smooth=0.9) for real in pred_real])
            )

        self.manual_backward(d_loss)
        opt_d.step()

        # Generator step
        opt_g.zero_grad(set_to_none=True)
        with autocast(device_type=self.device.type, enabled=self.trainer.precision == 16):
            fake_opt = self.netG(real_sar)
            fake_pair = torch.cat([real_sar, fake_opt], dim=1)
            
            pred_fake = self.netD(fake_pair)
            loss_gan = sum(self.criterions['GAN'](pf, True) for pf in pred_fake)
            loss_l1  = self.criterions['L1'](fake_opt, real_optical)
            
            g_loss = (
                loss_gan * self.loss_weights['gan'] +
                loss_l1 * self.loss_weights['l1']
            )

        self.manual_backward(g_loss)
        opt_g.step()
        self.log('train_g_loss', g_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log_dict({
            'train_loss_l1': loss_l1,
            'train_loss_gan': loss_gan,
            'train_loss_d': d_loss
        }, prog_bar=False, on_step=False, on_epoch=True)
    
    def validation_step(self, batch, batch_idx):
        real_sar, real_opt = batch
        fake_opt = self(real_sar)

        # Losses
        loss_l1 = self.criterions['L1'](fake_opt, real_opt)
        self.log('val_l1', loss_l1, prog_bar=True, on_step=False, on_epoch=True)

        # Metrics
        psnr = self.psnr(fake_opt, real_opt)
        ssim = self.ssim(fake_opt, real_opt)

        self.log('val_psnr', psnr, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_ssim', ssim, prog_bar=True, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        pass
    
    def configure_optimizers(self):
        optG, optD = build_optimizers(self.netG, self.netD)
        schedG, schedD = build_lr_schedulers(optG, optD)
        return [optD, optG], [schedD, schedG]
    
    def on_train_epoch_end(self):
        if self.fixed_sar is None:
            return
        
        gc.collect()
        torch.cuda.empty_cache()


        if (self.current_epoch + 1) % 10 != 0:
            return
        
        fake_opt = self.netG(self.fixed_sar)
        os.makedirs(os.path.join(self.cfg.system.output_dir, 'images'), exist_ok=True)
        path = f"{self.cfg.system.output_dir}/images/epoch_{self.current_epoch+1}.png"
        visualize_batch(
            self.fixed_sar.cpu().detach(),
            fake_opt.cpu().detach(),
            self.fixed_opt.cpu().detach(),
            path, max_rows=6, mode='quality',
            title=f"Epoch {self.current_epoch+1}"
        )


from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from src.data.datamodule import SAR2OPTDataModule
import os
from src.utils.config_loader import load_config
cfg = load_config("configs/config.yaml")

if __name__ == "__main__":
    pl.seed_everything(42, workers=True)
    torch.set_float32_matmul_precision('high')

    os.makedirs(cfg.system.output_dir, exist_ok=True)
    os.makedirs(cfg.system.checkpoints_dir, exist_ok=True)

    dm = SAR2OPTDataModule(
        data_dir=cfg.data.data_dir,
        batch_size=cfg.data.batch_size,
        image_size=cfg.data.image_size,
        num_workers=cfg.data.num_workers,
        persistent_workers=cfg.data.persistent_workers,
        prefetch_factor=cfg.data.prefetch_factor,
        train_val_split_ratio=cfg.data.train_val_split_ratio,
        seed=cfg.data.seed,
    )

    model = SAR2OPTGANLightningModule(cfg)

    logger = TensorBoardLogger(save_dir=cfg.system.output_dir, name="sar2opt_gan")
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.system.checkpoints_dir,
        filename="epoch{epoch:03d}-{val_l1:.4f}",
        monitor="val_l1",
        mode="min",
        save_top_k=3,
        save_last=True
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    trainer = Trainer(
        max_epochs=10,
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor],
        accelerator=cfg.system.device,
        devices=1,
        limit_train_batches=0.3,
        limit_val_batches=0.3,
        precision=cfg.system.precision,
        deterministic=cfg.system.deterministic
    )

    trainer.fit(model, datamodule=dm)