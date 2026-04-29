import os
import functools
import torch
import lightning.pytorch as pl
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from omegaconf import OmegaConf

_orig_load = torch.load
@functools.wraps(_orig_load)
def _patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _orig_load(*args, **kwargs)
torch.load = _patched_load

from src.models.huggingface_gan.main import SAR2OPTLightningModule
from src.data.sen12_full.datamodule import SEN12FullDataModule
from src.utils.cleanup_memory import full_cleanup
from src.utils.callbacks import EMAWeightAveraging

os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

CONFIG_PATH = 'src/models/huggingface_gan/config.yaml'


def main():
    cfg = OmegaConf.load(CONFIG_PATH)

    dm    = SEN12FullDataModule(cfg)
    model = SAR2OPTLightningModule(cfg)

    checkpoints = ModelCheckpoint(
        dirpath=cfg.system.checkpoints_dir,
        filename='epoch={epoch:03d}-psnr={val/psnr:.4f}',
        monitor='val/psnr',
        mode='max',
        save_top_k=3,
        save_last=True,
        auto_insert_metric_name=False,
    )
    callbacks = [checkpoints]
    if cfg.ema.use_ema:
        callbacks.append(EMAWeightAveraging(
            decay=cfg.ema.decay,
            update_starting_at_epoch=cfg.ema.start_epoch,
        ))

    tb_logger  = TensorBoardLogger(cfg.system.output_dir + '/tb_logs',  name=cfg.system.tb_version)
    csv_logger = CSVLogger(cfg.system.output_dir + '/csv_logs', name=cfg.system.tb_version)

    trainer = Trainer(
        logger=[tb_logger, csv_logger],
        callbacks=callbacks,
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
    )

    try:
        ckpt_path = cfg.system.resume_ckpt or None
        trainer.fit(model, datamodule=dm, ckpt_path=ckpt_path)
    except KeyboardInterrupt:
        pass
    finally:
        full_cleanup(trainer=trainer, model=model, datamodule=dm, log=True)


if __name__ == '__main__':
    main()
