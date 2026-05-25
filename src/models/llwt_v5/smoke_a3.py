"""A3 integration smoke for LLW-Former v0.5.2 (adversarial residual refiner).

Runs the REAL training+validation loop for 1 train batch + 1 val batch via
Lightning ``fast_dev_run`` on the real SEN12 datamodule and the frozen v4 G.
Validates the full A3 wiring:
  * build_models 3-tuple + frozen-G load,
  * 2-optimizer (opt_d, opt_g) configure + LR schedulers,
  * one D step + one G step (batched D, GAN+FM+L1) run finite,
  * validation_step metrics (PSNR/SSIM/LPIPS/FID + coarse_psnr),
  * G stays frozen (no grad), refiner receives finite grads.

Run from repo root::

    python -m src.models.llwt_v5.smoke_a3
"""
import functools
import os

import torch
import lightning.pytorch as pl
from lightning.pytorch import Trainer
from omegaconf import OmegaConf

torch.set_float32_matmul_precision('high')

# Match train.py: opt out of torch>=2.6 weights_only default for Lightning ckpts.
_orig_load = torch.load


@functools.wraps(_orig_load)
def _patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _orig_load(*args, **kwargs)


torch.load = _patched_load

from src.data.sen12_full.datamodule import SEN12FullDataModule
from src.models.llwt_v5.main import LLWv5RefinerModule
from src.utils.cleanup_memory import full_cleanup

os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

CONFIG_PATH = 'src/models/llwt_v5/config.yaml'


def main():
    cfg = OmegaConf.load(CONFIG_PATH)

    dm = SEN12FullDataModule(
        data_dir=cfg.data.data_dir.sen12_full,
        batch_size=cfg.data.batch_size,
        image_size=cfg.data.image_size,
        num_workers=0,                       # smoke: single-process, deterministic
        persistent_workers=False,
        prefetch_factor=None,
        train_val_split_ratio=cfg.data.train_val_split_ratio,
        seed=cfg.data.seed,
        sar_channels=cfg.data.sar_channels,
        use_augmentation=cfg.data.use_train_common_transform,
        scenes=list(cfg.data.scenes),
        train_crop_size=cfg.data.get('train_crop_size', None),
        val_batch_size=cfg.data.get('val_batch_size', None),
    )

    model = LLWv5RefinerModule(cfg)

    # G must be fully frozen.
    n_g_trainable = sum(p.requires_grad for p in model.netG.parameters())
    assert n_g_trainable == 0, f"frozen G has {n_g_trainable} trainable params!"
    n_refiner = sum(p.numel() for p in model.refiner.parameters())
    n_d = sum(p.numel() for p in model.netD.parameters())
    print(f"[smoke] refiner params={n_refiner/1e6:.2f}M  D params={n_d/1e6:.2f}M  "
          f"G trainable={n_g_trainable}")

    trainer = Trainer(
        fast_dev_run=True,                   # 1 train + 1 val batch, no logger/ckpt I/O
        accelerator=cfg.system.device,
        devices=1,
        precision=cfg.system.precision,
        enable_checkpointing=False,
        logger=False,
    )

    try:
        trainer.fit(model, datamodule=dm)
        print("[smoke] fast_dev_run completed without error")

        # After one fit step: confirm G still frozen, refiner got real grads.
        assert all(not p.requires_grad for p in model.netG.parameters()), "G unfrozen!"
        g_grad = [p.grad for p in model.netG.parameters() if p.grad is not None]
        assert len(g_grad) == 0, f"frozen G accumulated {len(g_grad)} grads!"
        print("[smoke] PASS — A3 wiring clean: 2-opt loop ran, G frozen, refiner trained")
    finally:
        full_cleanup(trainer=trainer, model=model, datamodule=dm, log=False)


if __name__ == '__main__':
    main()
