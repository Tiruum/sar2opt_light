# src/models/lightning/gan_module.py

import gc
import os
import torch.nn as nn
import torch
import pytorch_lightning as pl
from torch.amp import autocast
from src.models.cfrwd.factory import build_models, build_optimizers, build_criterions, build_lr_schedulers
from src.utils.visualize import visualize_batch
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from omegaconf import OmegaConf

class SAR2OPTGANLightningModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.cfg = config
        self.save_hyperparameters()
        
        # Инициализация моделей и критерионов
        self.netG, self.netD = build_models()
        self.netG.to(self.cfg.system.device)
        self.criterions = build_criterions()
        
        # Веса лоссов
        self.loss_weights = {
            'gan': self.cfg.loss.gan_weight,
            'l1': self.cfg.loss.l1_weight,
            'fm': self.cfg.loss.fm_weight,
        }

        self.psnr = PeakSignalNoiseRatio(data_range=2.0)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=2.0)
        
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
        real_sar, real_opt = batch
        opt_d, opt_g = self.optimizers()

        # Discriminator step
        opt_d.zero_grad(set_to_none=True)
        with autocast(device_type=self.device.type, enabled=self.trainer.precision == 16):
            fake_opt = self.netG(real_sar).detach()

            d_fake, _ = self.netD(fake_opt=fake_opt, real_opt=real_opt)
            d_real, _ = self.netD(fake_opt=real_opt, real_opt=real_opt)

            num_scales = len(d_real)
            d_loss = 0.0
            for real_out, fake_out in zip(d_real, d_fake):
                real_loss = self.criterions['GAN'](real_out, target_is_real=True, real_label_smooth=0.9, fake_label_smooth=0.1)
                fake_loss = self.criterions['GAN'](fake_out, target_is_real=False, real_label_smooth=0.9, fake_label_smooth=0.1)
                d_loss += (real_loss + fake_loss) / 2
            d_loss /= num_scales

            real_means = torch.stack([real_out.detach().mean() for real_out in d_real])
            fake_means = torch.stack([fake_out.detach().mean() for fake_out in d_fake])

        self.manual_backward(d_loss)
        opt_d.step()

        # Generator step
        opt_g.zero_grad(set_to_none=True)
        with autocast(device_type=self.device.type, enabled=self.trainer.precision == 16):
            fake_opt = self.netG(real_sar)

            d_fake, fake_feats = self.netD(fake_opt=fake_opt, real_opt=real_opt)
            _, real_feats = self.netD(fake_opt=real_opt, real_opt=real_opt)
            loss_gan = sum(self.criterions['GAN'](pf, True) for pf in d_fake)
            loss_fm = self.criterions['FM'](
                [feat.detach() for feat in real_feats],
                [feat for feat in fake_feats]
            )
            loss_l1 = self.criterions['L1'](fake_opt, real_opt)

            g_loss = (
                loss_gan * self.loss_weights['gan'] +
                loss_fm * self.loss_weights['fm'] +
                loss_l1 * self.loss_weights['l1']
            )

        self.manual_backward(g_loss)
        opt_g.step()

        self.log('train_g_loss', g_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log_dict({
            'train_loss_fm': loss_fm,
            'train_loss_gan': loss_gan,
            'train_loss_d': d_loss,
            'd_real_mean': real_means.mean(),
            'd_fake_mean': fake_means.mean(),
        }, prog_bar=False, on_step=False, on_epoch=True)
        self.log('fusion_coeff', self.netG.fusion_coeff.detach(), prog_bar=True, on_step=False, on_epoch=True)
    
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
        
        if (self.cfg.system.image_freq != 0):
            if (self.current_epoch + 1) % self.cfg.system.image_freq == 0:
                fake_opt = self.netG(self.fixed_sar)
                os.makedirs(os.path.join(self.cfg.system.images_dir, self.cfg.system.tb_version), exist_ok=True)
                path = f"{self.cfg.system.images_dir}/{self.cfg.system.tb_version}/epoch_{self.current_epoch+1}.png"
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

from src.data.sen12.datamodule import SEN12Datamodule
from src.utils.logger import Logger
terminal_logger = Logger(name="CFRWD main.py", cfg_path='src/models/cfrwd/config.yaml')
cfg = OmegaConf.load("src/models/cfrwd/config.yaml")

def cleanup():
    """Функция для освобождения оперативной памяти и кэша."""
    global model, dm
    if 'model' in globals():
        del model
    if 'dm' in globals():
        del dm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    pl.seed_everything(42, workers=True)
    torch.set_float32_matmul_precision('high')

    os.makedirs(cfg.system.output_dir, exist_ok=True)
    os.makedirs(cfg.system.checkpoints_dir, exist_ok=True)

    dm = SEN12Datamodule(
        data_dir=cfg.data.data_dir.sen12,
        batch_size=cfg.data.batch_size,
        image_size=cfg.data.image_size,
        num_workers=cfg.data.num_workers,
        persistent_workers=cfg.data.persistent_workers,
        prefetch_factor=cfg.data.prefetch_factor,
        train_val_split_ratio=cfg.data.train_val_split_ratio,
        seed=cfg.data.seed,
        sar_channels=cfg.data.sar_channels
    )

    model = SAR2OPTGANLightningModule(cfg)

    logger = TensorBoardLogger(save_dir=cfg.system.output_dir, name="cfrwd")
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{cfg.system.checkpoints_dir}/{cfg.system.tb_version}",
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
        limit_train_batches=0.1,
        limit_val_batches=0.1,
        precision=cfg.system.precision,
        deterministic=cfg.system.deterministic
    )

    try:
        terminal_logger.info("Начинаем обучение...")
        trainer.fit(model, datamodule=dm)
    except KeyboardInterrupt:
        terminal_logger.warning("Обучение прервано пользователем. Выполняем очистку...")
        cleanup()
        raise  # Перевыбрасываем исключение
    except Exception as e:
        terminal_logger.error(f"Произошла ошибка: {e}. Выполняем очистку...")
        cleanup()
        raise  # Перевыбрасываем исключение
    finally:
        cleanup()