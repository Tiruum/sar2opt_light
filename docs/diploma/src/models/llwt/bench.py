"""Compile vs eager benchmark for LLW-Former.

Runs N training steps WITHOUT the Lightning PyTorchProfiler so that
``torch.compile`` is not shredded by ``record_function`` graph breaks.  Pure
wall-clock measurement.

Reads ``src/models/llwt/config.yaml`` as-is.  Toggle ``system.compile`` in the
config between runs to compare.

Run from repo root:

    python -m src.models.llwt.bench
"""
import functools
import os
import time

import torch
import lightning.pytorch as pl
from lightning.pytorch import Trainer
from omegaconf import OmegaConf

_orig_load = torch.load


@functools.wraps(_orig_load)
def _patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _orig_load(*args, **kwargs)


torch.load = _patched_load
torch.set_float32_matmul_precision('high')

from src.data.sen12_full.datamodule import SEN12FullDataModule
from src.models.llwt.main import LLWFormerLightningModule
from src.utils.cleanup_memory import full_cleanup


os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
CONFIG_PATH = 'src/models/llwt/config.yaml'
WARMUP_STEPS = 8           # absorbs torch.compile cost
MEASURE_STEPS = 20


class _StepTimer(pl.Callback):
    """Records per-step wall-clock so we can drop warmup and report mean of measured."""
    def __init__(self):
        self.t0 = None
        self.times = []

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.t0 = time.perf_counter()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.times.append(time.perf_counter() - self.t0)


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

    timer = _StepTimer()
    total_steps = WARMUP_STEPS + MEASURE_STEPS
    trainer = Trainer(
        logger=False,
        callbacks=[timer],
        accelerator=cfg.system.device,
        devices=1,
        precision=cfg.system.precision,
        max_epochs=1,
        limit_train_batches=total_steps,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        log_every_n_steps=1,
        enable_checkpointing=False,
        enable_progress_bar=True,
        deterministic=cfg.system.deterministic,
        benchmark=cfg.system.benchmark,
        profiler=None,                 # critical: no Lightning profiler -> no record_function -> no dynamo breaks
    )

    print(f"[bench] cfg.system.compile = {cfg.system.compile}")
    print(f"[bench] cfg.system.precision = {cfg.system.precision}")
    print(f"[bench] batch_size = {cfg.data.batch_size}, image_size = {cfg.data.image_size}")
    print(f"[bench] warmup = {WARMUP_STEPS}, measure = {MEASURE_STEPS}")

    try:
        trainer.fit(model, datamodule=dm)
    except KeyboardInterrupt:
        pass
    finally:
        full_cleanup(trainer=trainer, model=model, datamodule=dm, log=False)

    warmup = timer.times[:WARMUP_STEPS]
    measured = timer.times[WARMUP_STEPS:]
    if not measured:
        print("[bench] not enough steps")
        return

    print()
    print(f"[bench] warmup steps ({len(warmup)}): min={min(warmup):.3f}s max={max(warmup):.3f}s")
    print(f"[bench] measured steps ({len(measured)}):")
    print(f"          min  = {min(measured):.3f}s")
    print(f"          mean = {sum(measured)/len(measured):.3f}s")
    print(f"          max  = {max(measured):.3f}s")
    n_per_epoch = int(21452 / cfg.data.batch_size)         # train=21452 patches in 9-scene subset
    print(f"[bench] estimated epoch time @ {n_per_epoch} steps/epoch = "
          f"{(sum(measured)/len(measured)) * n_per_epoch / 60:.1f} min")


if __name__ == '__main__':
    main()
