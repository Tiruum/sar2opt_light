"""LLW-Former training entry point.

Mirrors ``src/models/sarformer_wb/train.py`` but wires the new module
``LLWFormerLightningModule``.  Run from repo root:

    python -m src.models.llwt.train
"""
import functools
import os

import torch
import lightning.pytorch as pl
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from omegaconf import OmegaConf

# Allow TF32 on Ada/Ampere Tensor Cores for fp32 matmuls. The bf16-mixed
# autocast handles most of the network, but R1 is force-promoted to fp32 to
# avoid second-order NaN; TF32 keeps that path on the Tensor Core path
# instead of falling back to vanilla fp32 CUDA. Math change is on the order
# of 1e-3 in matmul output, which is well below the noise floor for GAN
# training.
torch.set_float32_matmul_precision('high')

# Dynamo guards on bool kwargs, tensor shapes, branch flags. The default cache
# size of 8 was tripped earlier (148 graph breaks observed under the profiler);
# 16 leaves room for the train (return_internals=True) + val (False) + R1
# (autocast disabled) variants without thrashing recompiles.
import torch._dynamo
torch._dynamo.config.cache_size_limit = 16

# torch>=2.6 changed `weights_only` default; Lightning ckpts contain Python
# objects so we have to opt out globally.
_orig_load = torch.load


@functools.wraps(_orig_load)
def _patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _orig_load(*args, **kwargs)


torch.load = _patched_load

from src.data.sen12_full.datamodule import SEN12FullDataModule
from src.models.llwt import factory
from src.models.llwt.main import LLWFormerLightningModule
from src.utils.cleanup_memory import full_cleanup


os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

CONFIG_PATH = 'src/models/llwt/config.yaml'


def _load_weights_ckpt(model: LLWFormerLightningModule, ckpt_path: str,
                       use_live_weights: bool = False) -> None:
    """Load generator + discriminator weights from a Lightning checkpoint
    (strict=False to tolerate the warm-start scenario where a sub-module
    isn't present in the source checkpoint).
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
    dis_sd = _strip('netD') or _strip('netD_main')
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
    )
    model = LLWFormerLightningModule(cfg)

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
