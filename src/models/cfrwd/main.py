# src/models/lightning/main.py

import gc
import os
import torch.nn as nn
import torch
import lightning.pytorch as pl
from src.models.cfrwd.factory import build_models, build_optimizers, build_criterions, build_lr_schedulers
from src.utils.visualize import visualize_batch
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure, SpectralAngleMapper #, LearnedPerceptualImagePatchSimilarity
from torchmetrics.regression import MeanSquaredError
from omegaconf import OmegaConf
from src.utils.notification import send_telegram, generate_tg_message
from src.utils.cleanup_memory import cleanup_memory, kill_dataloader_workers

class SAR2OPTGANLightningModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.cfg = config
        self.save_hyperparameters()
        
        # Инициализация моделей и критерионов
        self.netG, self.netD = build_models()
        self.criterions = build_criterions()
        
        # Веса лоссов
        self.loss_weights = {
            'gan': self.cfg.loss.gan_weight,
            'l1': self.cfg.loss.l1_weight,
            'fm': self.cfg.loss.fm_weight,
        }

        self.psnr = PeakSignalNoiseRatio(data_range=2.0)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=2.0)
        self.rmse = MeanSquaredError(squared=False)
        self.sam = SpectralAngleMapper()
        
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

        # ========== ОБУЧЕНИЕ ДИСКРИМИНАТОРА ==========
        opt_d.zero_grad(set_to_none=True)

        # Генерируем fake изображения (torch.no_grad() чтобы не обучать G)
        with torch.no_grad():
            fake_opt = self.netG(real_sar)

        # Дискриминатор смотрит на РЕАЛЬНЫЕ оптические изображения (условие + реал)
        d_real, real_feats = self.netD(torch.cat([real_sar, real_opt], dim=1))
        # Дискриминатор смотрит на FAKE оптические изображения (условие + фейк)
        d_fake, _ = self.netD(torch.cat([real_sar, fake_opt.detach()], dim=1))

        num_scales = len(d_real)
        d_loss = 0.0
        for real_out, fake_out in zip(d_real, d_fake):
            real_loss = self.criterions['GAN'](real_out, target_is_real=True)
            fake_loss = self.criterions['GAN'](fake_out, target_is_real=False)
            d_loss += 0.5 * (real_loss + fake_loss)
        d_loss /= num_scales

        # Статистика для логирования
        real_means = torch.stack([real_out.detach().mean() for real_out in d_real])
        fake_means = torch.stack([fake_out.detach().mean() for fake_out in d_fake])

        self.manual_backward(d_loss)
        opt_d.step()

        # ========== ОБУЧЕНИЕ ГЕНЕРАТОРА ==========
        opt_g.zero_grad(set_to_none=True)

        # Генерируем fake изображения заново (чтобы граф градиентов был привязан к G)
        fake_opt = self.netG(real_sar)

        d_fake, fake_feats = self.netD(torch.cat([real_sar, fake_opt], dim=1))  # Дискриминатор оценивает fake (для adversarial loss)

        # === GAN Loss с усреднением по масштабам ===
        num_scales = len(d_fake)
        loss_gan = sum(self.criterions['GAN'](pf, True) for pf in d_fake) / num_scales

        # === Feature Matching Loss с ВЗВЕШИВАНИЕМ ПО МАСШТАБАМ ===
        # Для SAR-to-Optical: large scale (256x256) важнее для глобальной структуры
        # small scale (128x128) — локальные текстуры (менее важны из-за спекл-шума)
        # Источник: Wang et al., "pix2pixHD", CVPR 2018
        real_feats_detached = [f.detach() for f in real_feats]
        
        # Веса для масштабов: [large_scale, small_scale]
        scale_weights = [1.0, 0.5]  # large scale в 2 раза важнее
        
        loss_fm = 0.0
        for i, (real_feat, fake_feat) in enumerate(zip(real_feats_detached, fake_feats)):
            weight = scale_weights[i] if i < len(scale_weights) else 1.0
            loss_fm += weight * self.criterions["FM"]([real_feat], [fake_feat])
        loss_fm = loss_fm / sum(scale_weights)  # Нормализация

        # === L1 Loss ===
        loss_l1 = self.criterions['L1'](fake_opt, real_opt)

        # === Total Generator Loss ===
        g_loss = (
            loss_gan * self.loss_weights['gan'] +
            loss_fm * self.loss_weights['fm'] +
            loss_l1 * self.loss_weights['l1']
        )

        self.manual_backward(g_loss)
        
        # === Gradient Clipping для G ===
        # Предотвращает "взрывы" градиентов в кросс-модальной задаче
        # Источник: Gulrajani et al., "WGAN-GP", NIPS 2017
        torch.nn.utils.clip_grad_norm_(
            self.netG.parameters(),
            max_norm=1.0,   # Было: 0.5 (слишком агрессивно)
            norm_type=2.0
        )
        
        opt_g.step()

        self.log('train/g_loss', g_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=real_sar.size(0))
        self.log_dict({
            'train/loss_fm': loss_fm,
            'train/loss_gan': loss_gan,
            'train/loss_d': d_loss,
            'train/loss_l1': loss_l1,
            'feats/d_real_mean': real_means.mean(),
            'feats/d_fake_mean': fake_means.mean(),
            'fusion/fusion_weight': self.netG.fusion_weight.item()
        }, prog_bar=False, on_step=False, on_epoch=True, batch_size=real_sar.size(0))
    
    def validation_step(self, batch, batch_idx):
        real_sar, real_opt = batch
        fake_opt = self(real_sar)

        # Кастим в fp32 для метрик (некоторые не поддерживают bf16)
        fake_opt_f32 = fake_opt.float()
        real_opt_f32 = real_opt.float()

        # Losses
        loss_l1 = self.criterions['L1'](fake_opt_f32, real_opt_f32)
        self.log('val/loss_l1', loss_l1, prog_bar=True, on_step=False, on_epoch=True, batch_size=real_sar.size(0))

        # Metrics
        psnr = self.psnr(fake_opt_f32, real_opt_f32)
        ssim = self.ssim(fake_opt_f32, real_opt_f32)
        rmse = self.rmse(fake_opt_f32, real_opt_f32)
        sam = self.sam(fake_opt_f32, real_opt_f32)

        self.log('val/psnr', psnr, prog_bar=True, on_step=False, on_epoch=True, batch_size=real_sar.size(0))
        self.log('val/ssim', ssim, prog_bar=True, on_step=False, on_epoch=True, batch_size=real_sar.size(0))
        self.log('val/rmse', rmse, prog_bar=False, on_step=False, on_epoch=True, batch_size=real_sar.size(0))
        self.log('val/sam', sam, prog_bar=False, on_step=False, on_epoch=True, batch_size=real_sar.size(0))

    def test_step(self, batch, batch_idx):
        pass
    
    def configure_optimizers(self):
        optG, optD = build_optimizers(self.netG, self.netD)
        schedG, schedD = build_lr_schedulers(optG, optD)
        return [optD, optG], [schedD, schedG]
    
    def on_train_epoch_end(self):
        schedD, schedG = self.lr_schedulers()
        schedD.step()
        schedG.step()

        opt_d, opt_g = self.optimizers()
        self.log("lr/d", opt_d.param_groups[0]["lr"], prog_bar=False, on_step=False, on_epoch=True)
        self.log("lr/g", opt_g.param_groups[0]["lr"], prog_bar=False, on_step=False, on_epoch=True)

        if self.fixed_sar is None:
            return
        
        if (self.cfg.system.image_freq != 0):
            if (self.current_epoch + 1) % self.cfg.system.image_freq == 0:
                with torch.no_grad():
                    fake_opt, cfr_out, hfcf_out = self.netG(self.fixed_sar, return_branches=True)
                os.makedirs(os.path.join(self.cfg.system.images_dir, self.cfg.system.tb_version), exist_ok=True)
                path = f"{self.cfg.system.images_dir}/{self.cfg.system.tb_version}/epoch_{self.current_epoch+1}.png"
                visualize_batch(
                    self.fixed_sar.cpu().detach(),
                    fake_opt.cpu().detach(),
                    self.fixed_opt.cpu().detach(),
                    path, max_rows=6, mode='quality',
                    title=f"Epoch {self.current_epoch+1}",
                    cfr_out=cfr_out.cpu().detach(),
                    hfcf_out=hfcf_out.cpu().detach(),
                    fusion_weight=self.netG.fusion_weight.item()
                )
                tg_message = generate_tg_message(self)
                send_telegram(image_path=path, message=tg_message)

    def on_train_start(self):
        send_telegram(message=f"[{str(self.cfg.system.tb_version).upper()}] Обучение началось")

    def on_train_end(self):
        # Освобождаем фиксированные тензоры
        self.fixed_sar = None
        self.fixed_opt = None
        cleanup_memory()
        send_telegram(message=f"[{str(self.cfg.system.tb_version).upper()}] Обучение завершено")


if __name__ == "__main__":
    from lightning.pytorch import Trainer
    from lightning.pytorch.callbacks import ModelCheckpoint
    from lightning.pytorch.loggers import TensorBoardLogger
    from src.data.sen12.datamodule import SEN12Datamodule
    from src.utils.logger import Logger
    from src.utils.cleanup_memory import full_cleanup

    cfg = OmegaConf.load("src/models/cfrwd/config.yaml")
    terminal_logger = Logger(cfg_path='src/models/cfrwd/config.yaml')
    pl.seed_everything(42, workers=True)
    torch.set_float32_matmul_precision('high')

    os.makedirs(cfg.system.output_dir, exist_ok=True)
    os.makedirs(cfg.system.checkpoints_dir, exist_ok=True)

    model = None
    dm = None
    trainer = None

    dm = SEN12Datamodule(
        data_dir=cfg.data.data_dir.sen12,
        batch_size=cfg.data.batch_size,
        image_size=cfg.data.image_size,
        num_workers=cfg.data.num_workers,
        persistent_workers=cfg.data.persistent_workers,
        prefetch_factor=cfg.data.prefetch_factor,
        train_val_split_ratio=cfg.data.train_val_split_ratio,
        seed=cfg.data.seed,
        sar_channels=cfg.data.sar_channels,
        use_augmentation=getattr(cfg.data, "use_train_common_transform", True)
    )

    model = SAR2OPTGANLightningModule(cfg)

    logger = TensorBoardLogger(save_dir=cfg.system.output_dir, name="cfrwd")
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{cfg.system.checkpoints_dir}/{cfg.system.tb_version}",
        filename="epoch{epoch:03d}-{val/psnr:.4f}",
        monitor="val/psnr",
        mode="max",
        save_top_k=3,
        save_last=True,
        auto_insert_metric_name=False
    )

    trainer = Trainer(
        max_epochs=10,
        logger=logger,
        callbacks=[checkpoint_callback],
        accelerator=cfg.system.device,
        devices=1,
        limit_train_batches=0.1,
        limit_val_batches=0.1,
        precision=cfg.system.precision,
        deterministic=cfg.system.deterministic
    )

    try:
        terminal_logger.info("Начинаем обучение...")
        trainer.fit(model, datamodule=dm, ckpt_path=cfg.system.resume_ckpt)
    except KeyboardInterrupt:
        terminal_logger.warning("Обучение прервано пользователем. Выполняем очистку...")
    except Exception as e:
        terminal_logger.error(f"Произошла ошибка: {e}. Выполняем очистку...")
        raise
    finally:
        full_cleanup(trainer=trainer, model=model, datamodule=dm, log=True)