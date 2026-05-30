"""One-shot profiler for LLW-Former training.

Runs 20 training steps with the Lightning PyTorchProfiler attached and dumps
a chrome trace + top-N table to ``cfg.system.profiler_dir``.  No validation,
no checkpoints, no loggers — purely a kernel-time measurement.

Run from repo root:

    python -m src.models.llwt.profile_run

Reads the same ``src/models/llwt/config.yaml`` as ``train.py`` so the profile
reflects the *actual* settings of the next training run.  Toggle
``system.compile`` in the config before running to compare eager vs compiled.

Schedule: wait=1, warmup=5, active=12, repeat=1.  Warmup must be large enough
for ``torch.compile`` recompiles to settle (LLW + per-subband Swin trigger
several traces on the first few steps).
"""
import functools
import os

import torch
import lightning.pytorch as pl
from lightning.pytorch import Trainer
from lightning.pytorch.profilers import PyTorchProfiler
from omegaconf import OmegaConf

# Mirror train.py: TF32 on R1 fp32 path so the profile measures the same
# code path that real training will hit.
torch.set_float32_matmul_precision('high')

_orig_load = torch.load


@functools.wraps(_orig_load)
def _patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _orig_load(*args, **kwargs)


torch.load = _patched_load

from src.data.sen12_full.datamodule import SEN12FullDataModule
from src.models.llwt.main import LLWFormerLightningModule
from src.utils.cleanup_memory import full_cleanup


os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
CONFIG_PATH = 'src/models/llwt/config.yaml'


def main():
    cfg = OmegaConf.load(CONFIG_PATH)
    out_dir = cfg.system.profiler_dir
    os.makedirs(out_dir, exist_ok=True)

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

    # Schedule: 1 idle batch (DDP/dataloader warmup), 5 warmup (dynamo compile
    # + first-step CUDA cache fills), 12 active (the actual measurement window
    # — enough to see R1 cycle once at every_16, but skip first cycle).
    schedule = torch.profiler.schedule(wait=1, warmup=5, active=12, repeat=1)
    profiler = PyTorchProfiler(
        dirpath=out_dir,
        filename=f"{cfg.system.tb_version}_profile",
        schedule=schedule,
        export_to_chrome=True,
        record_shapes=True,
        with_stack=False,
        with_flops=False,
        sort_by_key="cuda_time_total",
        row_limit=30,
        group_by_input_shapes=False,
    )

    trainer = Trainer(
        logger=False,
        callbacks=[],
        accelerator=cfg.system.device,
        devices=1,
        precision=cfg.system.precision,
        max_epochs=1,
        limit_train_batches=20,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        log_every_n_steps=1,
        profiler=profiler,
        enable_checkpointing=False,
        enable_progress_bar=True,
        deterministic=cfg.system.deterministic,
        benchmark=cfg.system.benchmark,
    )

    print(f"[profile] cfg.system.compile = {cfg.system.compile}")
    print(f"[profile] cfg.system.precision = {cfg.system.precision}")
    print(f"[profile] batch_size = {cfg.data.batch_size}, image_size = {cfg.data.image_size}")
    print(f"[profile] outputs -> {out_dir}/{cfg.system.tb_version}_profile.*")

    try:
        trainer.fit(model, datamodule=dm)
    except KeyboardInterrupt:
        pass
    finally:
        full_cleanup(trainer=trainer, model=model, datamodule=dm, log=True)

    print(f"\n[profile] done. Open chrome trace at chrome://tracing")
    print(f"[profile] table dumped in {out_dir}")


if __name__ == '__main__':
    main()
