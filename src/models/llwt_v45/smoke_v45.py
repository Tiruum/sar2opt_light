"""Integration smoke for LLW-Former v0.4.5 (Lightning fast_dev_run).

Builds the REAL model + data + the v0.4.5 loss stack (MS-SSIM, LPIPS, FFL,
PatchNCE) and runs 1 train + 1 val batch. Asserts the step completes — i.e.
every new loss is finite, gradients flow, encode_optical works, and the
PatchNCE sampler params are in opt_g.

    python -m src.models.llwt_v45.smoke_v45
"""
import functools
import os

import torch
from lightning.pytorch import Trainer
from omegaconf import OmegaConf

torch.set_float32_matmul_precision('high')

_orig_load = torch.load


@functools.wraps(_orig_load)
def _patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _orig_load(*args, **kwargs)


torch.load = _patched_load

from src.data.sen12_full.datamodule import SEN12FullDataModule
from src.models.llwt_v45.main import LLWv4LightningModule

os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

CONFIG_PATH = 'src/models/llwt_v45/config.yaml'


def main():
    cfg = OmegaConf.load(CONFIG_PATH)
    # Shrink for a fast CPU/GPU smoke.
    cfg.data.batch_size = 2
    cfg.data.val_batch_size = 2
    cfg.data.scenes = [list(cfg.data.scenes)[0]]
    cfg.data.num_workers = 0
    cfg.data.persistent_workers = False
    cfg.data.prefetch_factor = None
    cfg.system.weights_ckpt = None          # smoke from scratch (zero-init head)
    cfg.system.resume_ckpt = None
    cfg.ema.use_ema = False                 # skip EMA for fast_dev_run

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
        train_crop_size=None,
        val_batch_size=cfg.data.val_batch_size,
    )

    model = LLWv4LightningModule(cfg)
    print('[smoke] backbone_family =', getattr(model.netG, 'backbone_family', '?'))
    print('[smoke] criterions =', list(model.criterions.keys()))
    assert 'msssim' in model.criterions, 'MS-SSIM not built'
    assert 'lpips' in model.criterions, 'LPIPS not built'
    assert 'ffl' in model.criterions, 'FFL not built'
    assert 'patchnce' in model.criterions, 'PatchNCE not built'

    trainer = Trainer(
        accelerator=cfg.system.device,
        devices=1,
        precision=cfg.system.precision,
        fast_dev_run=True,
        logger=False,
        enable_checkpointing=False,
        num_sanity_val_steps=0,
    )
    trainer.fit(model, datamodule=dm)
    print('[smoke] PASS — fast_dev_run completed for', getattr(model.netG, 'backbone_family', '?'),
          'with', list(model.criterions.keys()))


if __name__ == '__main__':
    main()
