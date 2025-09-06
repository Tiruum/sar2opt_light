# src/train.py

import os
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.profilers import SimpleProfiler
from pytorch_lightning.loggers import TensorBoardLogger


from src.models.gan_module import SAR2OPTGANLightningModule
from src.utils.config_loader import load_config
from src.data.datamodule import SAR2OPTDataModule

def main():
    # 1) Загрузить и валидировать конфиг
    cfg = load_config("configs/config.yaml")

    # 2) Фиксируем сиды и включаем оптимизацию матмуль
    seed_everything(cfg.data.seed, workers=True)
    torch.set_float32_matmul_precision('medium')

    # 3) Создаем папки для логов и чекпоинтов
    os.makedirs(cfg.system.output_dir, exist_ok=True)
    os.makedirs(cfg.system.checkpoints_dir, exist_ok=True)

    # 4) DataModule
    dm = SAR2OPTDataModule(
        data_dir=cfg.data.data_dir,
        batch_size=cfg.data.batch_size,
        image_size=cfg.data.image_size,
        num_workers=cfg.data.num_workers,
        persistent_workers=getattr(cfg.data, "persistent_workers", False),
        prefetch_factor=getattr(cfg.data, "prefetch_factor", 2),
        train_val_split_ratio=cfg.data.train_val_split_ratio,
        seed=cfg.data.seed,
    )

    # 5) LightningModule
    model = SAR2OPTGANLightningModule(cfg)

    # 6) Logger и Callbacks
    tb_logger = TensorBoardLogger(
        save_dir=cfg.system.output_dir,
        name='tb_logs'
    )
    checkpoints = ModelCheckpoint(
        dirpath=cfg.system.checkpoints_dir,
        filename="{epoch:03d}-{val_psnr:.4f}",
        monitor="val_psnr",
        mode="max",
        save_top_k=3,
        save_last=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # 7) Trainer
    trainer = Trainer(
        logger=tb_logger,
        profiler=SimpleProfiler(dirpath=cfg.system.output_dir, filename="profiler"),

        callbacks=[checkpoints, lr_monitor],
        accelerator=cfg.system.device,
        devices=1,

        precision=cfg.system.precision,
        max_epochs=cfg.system.max_epochs,
        num_sanity_val_steps=0,

        deterministic=cfg.system.deterministic,
        benchmark=cfg.system.benchmark,

        # limit_train_batches=cfg.system.limit_train_batches,
        # limit_val_batches=cfg.system.limit_val_batches,

        accumulate_grad_batches=cfg.system.accumulate_grad_batches,
        log_every_n_steps=1
    )

    # 8) Запуск обучения
    trainer.fit(model, datamodule=dm)

if __name__ == "__main__":
    main()
