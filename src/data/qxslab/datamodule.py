# src/data/qxslab/datamodule.py

import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, random_split
from typing import Optional, List

from src.data.qxslab.dataset import QXSLABDataset
from src.data.transforms import (
    get_common_transform,
    get_input_specific,
    get_optical_specific,
    get_resize_transform,
    DATASET_NORM_STATS,
)


class QXSLABDataModule(LightningDataModule):
    """
    LightningDataModule for the QXSLAB SAR-OPT dataset.

    Args:
        data_dir:              Path to QXSLAB root (contains sar_*/opt_* dirs).
        batch_size:            Batch size for train/val loaders.
        image_size:            Resize all images to this square size.
        num_workers:           DataLoader workers per loader.
        persistent_workers:    Keep workers alive between epochs.
        prefetch_factor:       Batches to prefetch per worker.
        train_val_split_ratio: Fraction used for training.
        seed:                  RNG seed for reproducible split.
        sar_channels:          1 (grayscale) or 3.
        use_augmentation:      Apply geometric augmentations to train split.
        variants:              Variant suffixes to include. None = all found.
    """

    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        image_size: int,
        num_workers: int = 4,
        persistent_workers: bool = True,
        prefetch_factor: int = 2,
        train_val_split_ratio: float = 0.8,
        seed: int = 42,
        sar_channels: int = 1,
        use_augmentation: bool = True,
        variants: Optional[List[str]] = None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers if num_workers > 0 else False
        self.prefetch_factor = prefetch_factor if num_workers > 0 else None
        self.train_val_split_ratio = train_val_split_ratio
        self.seed = seed
        self.sar_channels = sar_channels
        self.variants = variants

        self.train_dataset: Optional[QXSLABDataset] = None
        self.val_dataset: Optional[QXSLABDataset] = None

        self.train_common_transform = get_common_transform() if use_augmentation else None
        self.val_common_transform = None
        self.resize_transform = get_resize_transform(self.image_size)
        _stats = DATASET_NORM_STATS['qxslab']
        sar_mean = _stats['sar_mean'] * self.sar_channels
        sar_std = _stats['sar_std'] * self.sar_channels
        self.input_specific = get_input_specific(
            sar_channels=self.sar_channels,
            sar_mean=sar_mean,
            sar_std=sar_std,
        )
        self.optical_specific = get_optical_specific(
            opt_mean=_stats['opt_mean'],
            opt_std=_stats['opt_std'],
        )

    def setup(self, stage: Optional[str] = None):
        full_dataset = QXSLABDataset(
            root_dir=self.data_dir,
            sar_channels=self.sar_channels,
            variants=self.variants,
        )
        n_total = len(full_dataset)
        if n_total == 0:
            raise RuntimeError(
                f"QXSLABDataset found 0 items in '{self.data_dir}' "
                f"(variants={self.variants}). Check data_dir and variant names."
            )

        n_train = int(n_total * self.train_val_split_ratio)
        n_val = n_total - n_train

        generator = torch.Generator().manual_seed(self.seed)
        train_idx, val_idx = random_split(range(n_total), [n_train, n_val], generator=generator)

        all_items = full_dataset.items
        variants_used = full_dataset.variants

        def make_dataset(indices, common_transform):
            return QXSLABDataset(
                root_dir=self.data_dir,
                common_transform=common_transform,
                input_specific=self.input_specific,
                optical_specific=self.optical_specific,
                resize_transform=self.resize_transform,
                sar_channels=self.sar_channels,
                variants=variants_used,
                items=[all_items[i] for i in indices],
            )

        self.train_dataset = make_dataset(train_idx.indices, self.train_common_transform)
        self.val_dataset = make_dataset(val_idx.indices, self.val_common_transform)

        print(
            f"[QXSLABDataModule] variants={variants_used} | "
            f"train={len(self.train_dataset)} val={len(self.val_dataset)} "
            f"(total={n_total})"
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
            pin_memory=True,
        )
