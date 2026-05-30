"""LLW-Former v0.4.0 training entry point.

Clone of ``src/models/llwt_v3/train.py`` with:
  * ``CONFIG_PATH`` pointing at the v4 config.
  * Uses :class:`LLWv4LightningModule` (Haar-Stem ConvNeXt V2 + IHaar head).

Run from repo root::

    python -m src.models.llwt_v4.train
"""
import functools
import os
import sys

# Expandable-segments CUDA allocator: prevents caching-allocator fragmentation
# that causes a smooth per-epoch slowdown (reserved VRAM stays flat in
# nvidia-smi while allocations do progressively more synchronizing
# cudaMalloc/Free). Must be set before the CUDA allocator initializes.
# setdefault keeps a shell override (`$env:PYTORCH_CUDA_ALLOC_CONF=...`).
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import torch
import lightning.pytorch as pl
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from omegaconf import OmegaConf

torch.set_float32_matmul_precision('high')

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

# SEN12FullDataModule is imported conditionally inside main() — raw vs aligned
# tree is selected by cfg.data.dataset (both modules expose the same class API).
from src.models.llwt_v45 import factory
from src.models.llwt_v45.main import LLWv4LightningModule
from src.utils.cleanup_memory import full_cleanup


os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

CONFIG_PATH = sys.argv[1] if len(sys.argv) > 1 else 'src/models/llwt_v45/config.yaml'


def _load_weights_ckpt(model: LLWv4LightningModule, ckpt_path: str,
                       use_live_weights: bool = False) -> None:
    """Load generator + discriminator weights from a Lightning checkpoint.

    ``strict=False`` so a v0.1.x / v0.2.x / v0.3.x ckpt with a different
    generator (LLWFormerGenerator vs LLWv3Generator vs LLWv4Generator) won't
    crash; mismatched keys are reported but ignored.  Mainly useful for
    resuming a v4 run from a partially trained ckpt — not for cross-model
    warm-starting (the IHaar head means v3 -> v4 swaps have ~no overlap on
    the final decoder layers).
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

    # Dataset selection via cfg.data.dataset:
    #   sen12_full / sen12_full_align — SEN1-2 (raw or ECC-aligned mirror).
    #   qxs_saropt                    — QXSLAB SAR-OPT (Gaofen-3, transfer target).
    # SEN12FullDataModule and QXSSAROPTDataModule expose the same constructor
    # signature; QXS ignores `scenes` (flat layout, no scene partition).
    dataset_name = str(cfg.data.dataset)
    if dataset_name == 'sen12_full':
        from src.data.sen12_full.datamodule import SEN12FullDataModule as _DM
    elif dataset_name == 'sen12_full_align':
        from src.data.sen12_full_align.datamodule import SEN12FullDataModule as _DM
    elif dataset_name == 'qxs_saropt':
        from src.data.qxs_saropt.datamodule import QXSSAROPTDataModule as _DM
    else:
        raise ValueError(
            f"cfg.data.dataset='{dataset_name}' unsupported "
            "(expected 'sen12_full', 'sen12_full_align', or 'qxs_saropt')."
        )
    data_root = cfg.data.data_dir[dataset_name]
    print(f"[train] dataset={dataset_name} | data_root={data_root}")

    scenes_arg = None if dataset_name == 'qxs_saropt' else list(cfg.data.scenes)
    dm_kwargs = dict(
        data_dir=data_root,
        batch_size=cfg.data.batch_size,
        image_size=cfg.data.image_size,
        num_workers=cfg.data.num_workers,
        persistent_workers=cfg.data.persistent_workers,
        prefetch_factor=cfg.data.prefetch_factor,
        train_val_split_ratio=cfg.data.train_val_split_ratio,
        seed=cfg.data.seed,
        sar_channels=cfg.data.sar_channels,
        use_augmentation=cfg.data.use_train_common_transform,
        scenes=scenes_arg,
        train_crop_size=cfg.data.get('train_crop_size', None),
        val_batch_size=cfg.data.get('val_batch_size', None),
    )
    # sar_lognorm is QXS-only — SEN12 datamodules don't expose it.
    if dataset_name == 'qxs_saropt':
        dm_kwargs['sar_lognorm'] = bool(cfg.data.get('sar_lognorm', False))
    dm = _DM(**dm_kwargs)
    model = LLWv4LightningModule(cfg)

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
