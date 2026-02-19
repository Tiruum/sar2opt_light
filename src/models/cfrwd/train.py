# src/train.py

import os
from omegaconf import OmegaConf
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.profilers import SimpleProfiler
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.utilities.model_summary import ModelSummary


from src.models.cfrwd.main import SAR2OPTGANLightningModule
from src.data.sen12.datamodule import SEN12Datamodule
from src.utils.cleanup_memory import full_cleanup, cleanup_memory

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"  # disable albumentations update

if __name__ == "__main__":
    cfg = OmegaConf.load('src/models/cfrwd/config.yaml')
    from src.utils.logger import Logger
    terminal_logger = Logger(cfg_path='src/models/cfrwd/config.yaml')

    model = None
    dm = None
    trainer = None

    try:
        terminal_logger.info("Начинаем обучение...")
        print(OmegaConf.to_yaml(cfg))

        # 2) Фиксируем сиды и включаем оптимизацию матмуль
        seed_everything(cfg.data.seed, workers=True)
        torch.set_float32_matmul_precision('high')

        # 3) Создаем папки для логов и чекпоинтов
        os.makedirs(cfg.system.output_dir, exist_ok=True)
        os.makedirs(cfg.system.checkpoints_dir, exist_ok=True)
        os.makedirs(cfg.system.profiler_dir, exist_ok=True)

        # 4) DataModule
        dm = SEN12Datamodule(
            data_dir=cfg.data.data_dir.sen12,
            batch_size=cfg.data.batch_size,
            image_size=cfg.data.image_size,
            num_workers=cfg.data.num_workers,
            persistent_workers=getattr(cfg.data, "persistent_workers", False),
            prefetch_factor=getattr(cfg.data, "prefetch_factor", 2),
            train_val_split_ratio=cfg.data.train_val_split_ratio,
            seed=cfg.data.seed,
            sar_channels=cfg.data.sar_channels,
            use_augmentation=getattr(cfg.data, "use_train_common_transform", True)
        )

        # 5) LightningModule
        model = SAR2OPTGANLightningModule(cfg)
        if (cfg.model.log_summary):
            os.makedirs(os.path.join(cfg.system.summary_dir), exist_ok=True)
            with open(f"{cfg.system.summary_dir}/{cfg.system.tb_version}.txt", "w") as f:
                f.write(str(ModelSummary(model, max_depth=-1)))
        if getattr(cfg.system, 'compile', False):
            terminal_logger.info("torch.compile() включён — первая эпоха будет медленнее")
            model = torch.compile(model)

        # 6) Logger и Callbacks
        tb_logger = TensorBoardLogger(
            save_dir=cfg.system.output_dir,
            version=cfg.system.tb_version,
            name='tb_logs',
            default_hp_metric=False
        )
        csv_logger = CSVLogger(
            save_dir=cfg.system.output_dir,
            version=cfg.system.tb_version,
            name='csv_logs')
        
        checkpoints = ModelCheckpoint(
            dirpath=f"{cfg.system.checkpoints_dir}/{cfg.system.tb_version}",
            filename="epoch={epoch:03d}-psnr={val/psnr:.4f}",
            monitor="val/psnr",
            mode="max",
            save_top_k=3,
            save_last=True,
            auto_insert_metric_name=False
        )
        lr_monitor = LearningRateMonitor(logging_interval='epoch')

        # 7) Trainer
        trainer = Trainer(
            logger=[tb_logger, csv_logger],
            profiler=SimpleProfiler(dirpath=cfg.system.profiler_dir, filename=cfg.system.tb_version),

            callbacks=[checkpoints, lr_monitor],
            accelerator=cfg.system.device,
            devices=1,

            precision=cfg.system.precision,
            max_epochs=cfg.system.max_epochs,
            num_sanity_val_steps=2,

            deterministic=cfg.system.deterministic,
            benchmark=cfg.system.benchmark,

            limit_train_batches=cfg.system.limit_train_batches,
            limit_val_batches=cfg.system.limit_val_batches,

            log_every_n_steps=50,
            fast_dev_run=False  # This applies to fitting, validating, testing, and predicting. This flag is only recommended for debugging purposes and should not be used to limit the number of batches to run.
        )

        # 8) Запуск обучения
        ckpt_path = getattr(cfg.system, 'resume_ckpt', None)
        if ckpt_path:
            terminal_logger.info(f"Возобновление с чекпоинта: {ckpt_path}")
        trainer.fit(model, datamodule=dm, ckpt_path=ckpt_path)
    except KeyboardInterrupt:
        terminal_logger.warning("Обучение прервано (Ctrl+C). Выполняем очистку...")
    except Exception as e:
        terminal_logger.error(f"Произошла ошибка: {e}. Выполняем очистку...")
        raise
    finally:
        full_cleanup(trainer=trainer, model=model, datamodule=dm, log=True)
