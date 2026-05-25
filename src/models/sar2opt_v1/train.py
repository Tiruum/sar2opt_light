"""SAR2OPT-V1 training entry point.

Run from repo root::

    python -m src.models.sar2opt_v1.train
"""
from __future__ import annotations

import functools
import os

import torch
import lightning.pytorch as pl
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from omegaconf import OmegaConf


torch.set_float32_matmul_precision('high')

import torch._dynamo
torch._dynamo.config.cache_size_limit = 16

# torch>=2.6 changed weights_only default; Lightning ckpts contain Python
# objects so opt out globally.
_orig_load = torch.load


@functools.wraps(_orig_load)
def _patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _orig_load(*args, **kwargs)


torch.load = _patched_load

from src.data.sen12_full.datamodule import SEN12FullDataModule
from src.models.sar2opt_v1 import factory
from src.models.sar2opt_v1.main import SAR2OPTV1LightningModule
from src.utils.cleanup_memory import full_cleanup


os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

CONFIG_PATH = 'src/models/sar2opt_v1/config.yaml'


def _load_weights_ckpt(model: SAR2OPTV1LightningModule, ckpt_path: str,
                       use_live_weights: bool = False) -> None:
    """Load generator + discriminator weights from a Lightning checkpoint.

    ``strict=False`` so a v0.x ckpt with a different generator family won't
    crash; mismatched keys are reported but ignored.
    """
    print(f'[weights_ckpt] loading {ckpt_path}')
    ckpt = torch.load(ckpt_path, map_location='cpu')
    if use_live_weights and 'current_model_state' in ckpt:
        raw_sd = ckpt['current_model_state']
        src_label = 'live (current_model_state)'
    else:
        raw_sd = ckpt.get('state_dict', ckpt)
        src_label = 'EMA-or-live (state_dict)'

    def _strip(prefix: str):
        return {k[len(prefix) + 1:]: v for k, v in raw_sd.items()
                if k.startswith(prefix + '.')}

    gen_sd = _strip('netG')
    dis_sd = _strip('netD')
    mg, ug = model.netG.load_state_dict(gen_sd, strict=False)
    md, ud = model.netD.load_state_dict(dis_sd, strict=False)
    print(f'[weights_ckpt] {src_label} | G missing={len(mg)} unexpected={len(ug)}')
    print(f'[weights_ckpt] {src_label} | D missing={len(md)} unexpected={len(ud)}')


def main():
    cfg = OmegaConf.load(CONFIG_PATH)

    dm = SEN12FullDataModule(
        data_dir=cfg.data.data_dir.sen12_full,
        batch_size=cfg.data.batch_size,
        image_size=cfg.data.image_size,
        num_workers=cfg.data.num_workers,
        persistent_workers=cfg.data.persistent_workers,
        prefetch_factor=cfg.data.prefetch_factor,
        train_val_split_ratio=cfg.data.train_val_split_ratio,
        seed=cfg.data.seed,
        sar_channels=cfg.data.sar_channels,
        use_augmentation=cfg.data.use_train_common_transform,
        scenes=list(cfg.data.scenes),
        train_crop_size=cfg.data.get('train_crop_size', None),
        val_batch_size=cfg.data.get('val_batch_size', None),
    )
    model = SAR2OPTV1LightningModule(cfg)

    checkpoints = ModelCheckpoint(
        dirpath=f"{cfg.system.checkpoints_dir}/{cfg.system.tb_version}",
        filename='epoch={epoch:03d}-psnr={val/psnr:.4f}',
        monitor='val/psnr',
        mode='max',
        save_top_k=3,
        save_last=True,
        auto_insert_metric_name=False,
    )
    callbacks = [checkpoints]
    ema_callback = factory.build_ema_callback(cfg)
    if ema_callback is not None:
        callbacks.append(ema_callback)

    tb_logger = TensorBoardLogger(cfg.system.output_dir + '/tb_logs',
                                  name=cfg.system.tb_version)
    csv_logger = CSVLogger(cfg.system.output_dir + '/csv_logs',
                           name=cfg.system.tb_version)

    trainer = Trainer(
        logger=[tb_logger, csv_logger],
        callbacks=callbacks,
        accelerator=cfg.system.device,
        devices=1,
        precision=cfg.system.precision,
        max_epochs=cfg.system.max_epochs,
        num_sanity_val_steps=0,
        deterministic=cfg.system.deterministic,
        benchmark=cfg.system.benchmark,
        limit_train_batches=cfg.system.limit_train_batches,
        limit_val_batches=cfg.system.limit_val_batches,
        check_val_every_n_epoch=int(cfg.system.get('check_val_every_n_epoch', 1)),
        log_every_n_steps=50,
    )

    try:
        weights_ckpt = cfg.system.get('weights_ckpt', None)
        if weights_ckpt:
            _load_weights_ckpt(model, weights_ckpt)
            ckpt_path = None
        else:
            ckpt_path = cfg.system.get('resume_ckpt', None) or None
            if ckpt_path:
                print(f'Resuming from {ckpt_path}')
        trainer.fit(model, datamodule=dm, ckpt_path=ckpt_path)
    except KeyboardInterrupt:
        pass
    finally:
        full_cleanup(trainer=trainer, model=model, datamodule=dm, log=True)


if __name__ == '__main__':
    main()
