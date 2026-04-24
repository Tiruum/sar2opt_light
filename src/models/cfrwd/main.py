# src/models/lightning/main.py

import gc
import os
import torch
import lightning.pytorch as pl
from src.models.cfrwd.factory import build_models, build_optimizers, build_criterions, build_lr_schedulers
from src.utils.visualize import visualize_batch
from torchmetrics.image import (PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure,
                                SpectralAngleMapper, LearnedPerceptualImagePatchSimilarity,
                                ErrorRelativeGlobalDimensionlessSynthesis)
from omegaconf import OmegaConf
from src.utils.notification import send_telegram, generate_tg_message
from src.utils.cleanup_memory import cleanup_memory

class SAR2OPTGANLightningModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.cfg = config
        self.save_hyperparameters()
        
        # Инициализация моделей и критерионов
        self.netG, self.netD = build_models()

        # lpips_metric is always used for val/lpips logging regardless of use_lpips.
        # Constructed first so its AlexNet backbone can be shared with LPIPSLoss
        # (training criterion) when use_lpips=True — avoids loading AlexNet twice.
        self.lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=False)
        # Важно: обычный dict не регистрирует loss-модули в nn.Module,
        # поэтому LPIPS может остаться на CPU. ModuleDict фиксирует это.
        self.criterions = torch.nn.ModuleDict(
            build_criterions(lpips_backbone=self.lpips_metric.net if self.cfg.loss.get('use_lpips', False) else None)
        )

        # Веса adversarial-компонент остаются фиксированными
        self.loss_weights = {
            'gan': self.cfg.loss.gan_weight,
            'fm': self.cfg.loss.fm_weight,
        }

        self.psnr  = PeakSignalNoiseRatio(data_range=2.0)
        self.ssim  = StructuralSimilarityIndexMeasure(data_range=2.0)
        self.sam   = SpectralAngleMapper()
        # ratio=1: SAR и OPT у нас одного пространственного разрешения (Sentinel-1/2, 10 м)
        self.ergas = ErrorRelativeGlobalDimensionlessSynthesis(ratio=1)
        
        self.fixed_sar = None
        self.fixed_opt = None

        self.automatic_optimization = False

    def forward(self, x):
        out, _ = self.netG(x)
        return out
    
    def setup(self, stage=None):
        if stage != 'fit':
            return  # fixed_sar/fixed_opt нужны только во время обучения
        from torch.utils.data import DataLoader
        dm = self.trainer.datamodule
        # num_workers=0: не порождаем лишние spawn-процессы на Windows ради одного батча
        one_shot = DataLoader(
            dm.train_dataset,
            batch_size=self.cfg.data.batch_size,
            shuffle=False,
            num_workers=0,
        )
        sar, opt = next(iter(one_shot))
        self.fixed_sar = sar.to(self.device)
        self.fixed_opt = opt.to(self.device)
    
    def training_step(self, batch, batch_idx):
        real_sar, real_opt = batch
        opt_d, opt_g = self.optimizers()

        # ========== ОБУЧЕНИЕ ДИСКРИМИНАТОРА ==========
        opt_d.zero_grad(set_to_none=True)

        # Генерируем fake изображения (torch.no_grad() чтобы не обучать G)
        with torch.no_grad():
            fake_opt, _ = self.netG(real_sar)

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
        self.clip_gradients(opt_d, gradient_clip_val=1.0, gradient_clip_algorithm='norm')
        opt_d.step()

        # ========== ОБУЧЕНИЕ ГЕНЕРАТОРА ==========
        opt_g.zero_grad(set_to_none=True)

        # Генерируем fake изображения заново (чтобы граф градиентов был привязан к G)
        fake_opt, cfr_out, hfcf_out, w_hfcf = self.netG(real_sar, return_branches=True)

        d_fake, fake_feats = self.netD(torch.cat([real_sar, fake_opt], dim=1))  # Дискриминатор оценивает fake (для adversarial loss)

        num_scales = len(d_fake)
        loss_gan = sum(self.criterions['GAN'](pf, True) for pf in d_fake) / num_scales

        # Feature matching loss (real_feats отвязаны от графа D)
        real_feats_detached = [f.detach() for f in real_feats]
        loss_fm = self.criterions["FM"](real_feats_detached, fake_feats)

        loss_l1  = self.criterions['L1'](fake_opt, real_opt)
        loss_fft = self.criterions['FFT'](fake_opt, real_opt)

        # CFR branch: full-image L1 is valid (CFR processes the complete spatial context)
        loss_cfr_aux = self.criterions['L1'](cfr_out, real_opt) * self.cfg.loss.get('cfr_aux_weight', 0.3)

        # HFCF branch: wavelet-domain L1 on 6 detail subbands — architecturally matched
        # (HFCF discards LL2; supervising LL would penalise content it cannot model)
        loss_wavelet = self.criterions['WAVELET'](hfcf_out, real_opt) * self.cfg.loss.get('wavelet_weight', 0.5)

        g_loss = (
            loss_gan     * self.loss_weights['gan'] +
            loss_fm      * self.loss_weights['fm'] +
            loss_cfr_aux +
            loss_wavelet
        )

        self.manual_backward(g_loss)
        self.clip_gradients(opt_g, gradient_clip_val=1.0, gradient_clip_algorithm='norm')
        opt_g.step()

        self.log('train/g_loss', g_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=real_sar.size(0))
        self.log_dict({
            'train/loss_fm':       loss_fm,
            'train/loss_gan':      loss_gan,
            'train/loss_d':        d_loss,
            'train/loss_l1':       loss_l1,
            'train/loss_fft':      loss_fft,
            'train/loss_cfr_aux':  loss_cfr_aux,
            'train/loss_wavelet':  loss_wavelet,
            'feats/d_real_mean':   real_means.mean(),
            'feats/d_fake_mean':   fake_means.mean(),
            'fusion/w_hfcf':       w_hfcf.squeeze(),
            # SpeckleAwareModule gate mean activations: 1.0 = pass-through, 0.0 = full attenuation
            'fusion/speckle_gate_mid':  self.netG.hfcf_branch._last_g2_gate_mean,
            'fusion/speckle_gate_high': self.netG.hfcf_branch._last_g3_gate_mean,
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

        # Только накапливаем состояние — compute() и logging в on_validation_epoch_end().
        # ВАЖНО: вызов metric(pred, target) без reset() заставляет SpectralAngleMapper
        # хранить все тензоры [B,C,H,W] на GPU бесконечно → 6 МБ × 640 × N_epochs → OOM.
        self.psnr.update(fake_opt_f32, real_opt_f32)
        self.ssim.update(fake_opt_f32, real_opt_f32)
        self.lpips_metric.update(fake_opt_f32, real_opt_f32)

        # ERGAS: divides by per-band mean → [-1,1] mean≈0 → ERGAS=∞
        # SAM: physically requires non-negative reflectance; zero-norm → NaN
        # Shift [-1,1] → [0,1] for these two metrics only.
        fake_01 = (fake_opt_f32 + 1.0) * 0.5
        real_01 = (real_opt_f32 + 1.0) * 0.5
        self.sam.update(fake_01, real_01)
        self.ergas.update(fake_01, real_01)

    def on_validation_epoch_end(self):
        # Правильный паттерн torchmetrics + Lightning:
        # compute() вызывается один раз за эпоху (а не 640 раз),
        # reset() освобождает GPU-тензоры SpectralAngleMapper и SSIM после каждой эпохи.
        self.log('val/psnr',  self.psnr.compute(),          prog_bar=True)
        self.log('val/ssim',  self.ssim.compute(),          prog_bar=True)
        self.log('val/lpips', self.lpips_metric.compute(),   prog_bar=True)
        ergas_val = self.ergas.compute()
        if torch.isfinite(ergas_val):
            self.log('val/ergas', ergas_val, prog_bar=False)
        sam_val = self.sam.compute()
        if not torch.isnan(sam_val):
            self.log('val/sam', sam_val, prog_bar=False)
        self.psnr.reset()
        self.ssim.reset()
        self.sam.reset()
        self.ergas.reset()
        self.lpips_metric.reset()
        # Возвращаем PyTorch-кэш CUDA обратно драйверу после каждой val-эпохи.
        # Без этого caching allocator накапливает freed-блоки → Windows видит 16 ГБ занято.
        gc.collect()
        torch.cuda.empty_cache()

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
                    fake_opt, cfr_out, hfcf_out, _ = self.netG(self.fixed_sar, return_branches=True)
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
                    fusion_weight=None
                )
                tg_message = generate_tg_message(self)
                send_telegram(image_path=path, message=tg_message)
                # Explicitly release large GPU tensors before flushing the
                # CUDA allocator cache; without del they stay alive until the
                # if-block exits anyway, but this makes the intent explicit.
                del fake_opt, cfr_out, hfcf_out
                torch.cuda.empty_cache()

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