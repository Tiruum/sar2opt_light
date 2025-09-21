import optuna
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from src.data.sen12.datamodule import SEN12Datamodule
from models.cfrwd.main import SAR2OPTGANLightningModule
from src.utils.config_loader import load_config
from omegaconf import OmegaConf
import torch
import gc

def objective(trial):
    # Гиперпараметры для поиска
    lr_g = trial.suggest_float('lr_g', 1e-4, 1e-3)
    lr_d = trial.suggest_float('lr_d', 1e-4, 1e-3)
    l1_weight = trial.suggest_float('l1_weight', 1.0, 200.0)
    gan_weight = trial.suggest_float('gan_weight', 0.1, 10.0)
    num_D = trial.suggest_int('num_D', 1, 4)

    cfg = load_config("configs/config.yaml")

    updates = {
        'data': {
            'image_size': 128,
            'batch_size': 16
        },
        'model': {
            'num_D': num_D,
        },
        'loss': {
            'l1_weight': l1_weight,
            'gan_weight': gan_weight,
        },
        'system': {
            'output_dir': 'optuna_outputs',
            'max_epochs': 50,
            'image_freq': 0
        },
        'optimizer': {
            'lr_g': lr_g,
            'lr_d': lr_d
        }
    }

    config = OmegaConf.merge(cfg, updates)

    torch.set_float32_matmul_precision('high')

    # Инициализация модели
    model = SAR2OPTGANLightningModule(config)
    
    # Logger
    logger = TensorBoardLogger("tb_logs", name=f"trial_{trial.number}")

    print(config)

    # DataModule
    dm = SEN12Datamodule(
        data_dir="./sen12-data/v_2",
        batch_size=cfg.data.batch_size * 4,
        image_size=cfg.data.image_size,
        num_workers=0,
        persistent_workers=False,
        prefetch_factor=None,
        train_val_split_ratio=cfg.data.train_val_split_ratio,
        seed=cfg.data.seed,
    )
    
    # Trainer с прунингом
    trainer = Trainer(
        logger=logger,
        accelerator=cfg.system.device,
        devices=1,
        precision=cfg.system.precision,
        max_epochs=cfg.system.max_epochs,
        num_sanity_val_steps=0,
        deterministic=cfg.system.deterministic,
        benchmark=cfg.system.benchmark,
        limit_train_batches=cfg.system.limit_train_batches,
        limit_val_batches=cfg.system.limit_val_batches,
        log_every_n_steps=1,
    )
    
    try:
        trainer.fit(model, datamodule=dm)
        
        # Получаем метрики из callback_metrics
        val_psnr = trainer.callback_metrics.get("val_psnr")
        val_ssim = trainer.callback_metrics.get("val_ssim")
        
        # Проверяем наличие метрик
        if val_psnr is None or val_ssim is None:
            score = float('-inf')
        else:
            # Комбинированная метрика
            score = val_psnr.item() + val_ssim.item() * 100 # PSNR \in [0, 100], SSIM \in [0, 1]
    except Exception as e:
        print(f"Trial failed with error: {e}")
        score = float('-inf')
    
    # Освобождаем память
    del trainer, model, dm
    torch.cuda.empty_cache()
    gc.collect()
    
    return score

if __name__ == '__main__':
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)

    # Анализ результатов
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value (PSNR+SSIM): {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # Визуализация
    try:
        optuna.visualization.plot_optimization_history(study).show()
        optuna.visualization.plot_parallel_coordinate(study).show()
    except Exception as e:
        print(f"Could not create visualizations: {e}")