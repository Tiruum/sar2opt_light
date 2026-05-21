"""End-to-end smoke tests for ``src/models/llwt/main.py``.

Verifies the Lightning module can be constructed and that a single training
step (D update + G update) runs without exceptions on a 4-sample mini-batch.

These tests pin the contract between factory, gen, dis, losses, and the
Lightning module — if any of those drift apart this is the first thing that
fails.
"""
import lightning.pytorch as pl
import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset

from src.models.llwt.main import LLWFormerLightningModule
from tests.test_llwt_factory import _full_cfg


def _make_loader(n: int = 4, image_size: int = 64, batch_size: int = 2):
    sar = torch.randn(n, 1, image_size, image_size).clamp(-1, 1)
    opt = torch.randn(n, 3, image_size, image_size).clamp(-1, 1)
    return DataLoader(TensorDataset(sar, opt), batch_size=batch_size)


def test_lightning_module_constructs():
    cfg = _full_cfg(image_size=64, embed_dim=24, depth=1, win=8)
    mod = LLWFormerLightningModule(cfg)
    # Every advertised criterion is in the ModuleDict.
    for k in ('gan', 'fm', 'l1', 'msssim', 'lab_chroma', 'wavelet',
              'spkdec', 'pr'):
        assert k in mod.criterions


def test_lightning_module_training_smoke():
    """Single mini-epoch on synthetic data — just check no exceptions."""
    cfg = _full_cfg(image_size=32, embed_dim=24, depth=1, win=8,
                    l1=50.0, spkdec=0.05, pr=0.01, lpips=0.0)
    # Drop LPIPS (avoids AlexNet weight download) and MS-SSIM (needs >=160px).
    cfg.loss.lpips_weight = 0.0
    cfg.loss.msssim_weight = 0.0
    cfg.system.max_epochs = 1
    mod = LLWFormerLightningModule(cfg)

    train_loader = _make_loader(n=4, image_size=32, batch_size=2)
    val_loader = _make_loader(n=2, image_size=32, batch_size=2)

    trainer = pl.Trainer(
        max_epochs=1,
        accelerator='cpu',
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
    )
    # If this crashes, the contract between Lightning + manual-opt is broken.
    trainer.fit(mod, train_loader, val_loader)


def test_lightning_module_subband_disabled_smoke():
    """Same as above but with subband-D disabled — code-path coverage."""
    cfg = _full_cfg(image_size=32, embed_dim=24, depth=1, win=8,
                    use_sub=False, lpips=0.0)
    cfg.loss.lpips_weight = 0.0
    cfg.loss.msssim_weight = 0.0
    cfg.system.max_epochs = 1
    mod = LLWFormerLightningModule(cfg)
    assert mod.use_subband_d is False

    train_loader = _make_loader(n=4, image_size=32, batch_size=2)
    val_loader = _make_loader(n=2, image_size=32, batch_size=2)

    trainer = pl.Trainer(
        max_epochs=1,
        accelerator='cpu',
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
    )
    trainer.fit(mod, train_loader, val_loader)
