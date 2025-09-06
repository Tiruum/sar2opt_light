import optuna
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from src.datamodule import YourDataModule
from src.models.lightning.gan_module import SAR2OPTGANLightningModule

def objective(trial):
    # Гиперпараметры для поиска
    lr_g = trial.suggest_loguniform('lr_g', 1e-5, 1e-3)
    lr_d = trial.suggest_loguniform('lr_d', 1e-5, 1e-3)
    l1_weight = trial.suggest_uniform('l1_weight', 10.0, 200.0)
    gan_weight = trial.suggest_uniform('gan_weight', 0.1, 1.0)
    
    config = {
        'model': {
            'input_nc': 1,
            'output_nc': 3,
            'ngf': 64,
            'ndf': 64,
            'n_blocks': 8,
            'n_layers': 3,
            'num_D': 4,
        },
        'loss': {
            'l1_weight': l1_weight,
            'gan_weight': gan_weight,
        },
        'system': {
            'output_dir': 'optuna_outputs',
        },
        'lr': {
            'g': lr_g,
            'd': lr_d,
        }
    }

    # Инициализация модели
    model = SAR2OPTGANLightningModule(config)
    
    # Logger
    logger = TensorBoardLogger("tb_logs", name=f"trial_{trial.number}")

    # DataModule (должен быть готов)
    dm = YourDataModule()  # Заменить на свой датамодуль
    
    # Trainer
    trainer = Trainer(
        max_epochs=10,
        logger=logger,
        precision=16,
        accelerator="gpu",
        devices=1,
    )
    
    trainer.fit(model, datamodule=dm)
    
    # Получаем метрики из callback_metrics
    val_psnr = trainer.callback_metrics.get("val_psnr")
    val_ssim = trainer.callback_metrics.get("val_ssim")
    
    # Проверяем наличие метрик
    if val_psnr is None or val_ssim is None:
        return float('-inf')
    
    # Комбинированная метрика: взвешенный суммарный балл (можно скорректировать веса)
    combined_score = val_psnr.item() + (val_ssim.item() * 100)  # Увеличиваем вес SSIM
    return combined_score
