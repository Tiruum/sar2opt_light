# src/train.py

import os
from omegaconf import OmegaConf
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.profilers import SimpleProfiler
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.model_summary import ModelSummary
import gc


from src.models.cfrwd.main import SAR2OPTGANLightningModule
from src.data.sen12.datamodule import SEN12Datamodule

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"  # disable albumentations update

def main():
    # 1) Загрузить и валидировать конфиг
    cfg = OmegaConf.load('src/models/cfrwd/config.yaml')
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
        sar_channels=cfg.data.sar_channels
    )

    # 5) LightningModule
    model = SAR2OPTGANLightningModule(cfg)
    if (cfg.model.log_summary):
        os.makedirs(os.path.join(cfg.system.summary_dir), exist_ok=True)
        with open(f"{cfg.system.summary_dir}/{cfg.system.tb_version}.txt", "w") as f:
            f.write(str(ModelSummary(model, max_depth=-1)))
    # model = torch.compile(model)

    # 6) Logger и Callbacks
    tb_logger = TensorBoardLogger(
        save_dir=cfg.system.output_dir,
        version=cfg.system.tb_version,
        name='tb_logs',
        default_hp_metric=False
    )
    checkpoints = ModelCheckpoint(
        dirpath=f"{cfg.system.checkpoints_dir}/{cfg.system.tb_version}",
        filename="{epoch:03d}-{val_psnr:.4f}",
        monitor="val/psnr",
        mode="max",
        save_top_k=3,
        save_last=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # 7) Trainer
    trainer = Trainer(
        logger=tb_logger,
        profiler=SimpleProfiler(dirpath=cfg.system.profiler_dir, filename=cfg.system.tb_version),

        callbacks=[checkpoints, lr_monitor],
        accelerator=cfg.system.device,
        devices=1,

        precision=cfg.system.precision,
        max_epochs=cfg.system.max_epochs,
        num_sanity_val_steps=0,

        deterministic=cfg.system.deterministic,
        benchmark=cfg.system.benchmark,

        limit_train_batches=cfg.system.limit_train_batches,
        limit_val_batches=cfg.system.limit_val_batches,

        log_every_n_steps=50
    )

    # 8) Запуск обучения
    trainer.fit(model, datamodule=dm)

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
    cfg = OmegaConf.load('src/models/cfrwd/config.yaml')
    from src.utils.logger import Logger
    terminal_logger = Logger(cfg_path='src/models/cfrwd/config.yaml')
    try:
        terminal_logger.info("Начинаем обучение...")
        main()
    except Exception as e:
        terminal_logger.error(f"Произошла ошибка: {e}. Выполняем очистку...")
        cleanup()
        raise  # Перевыбрасываем исключение
    finally:
        cleanup()
