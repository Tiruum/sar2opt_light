# src/data/qxs_saropt/datamodule.py

import albumentations as A
import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, random_split
from typing import Optional, List

from src.data.qxs_saropt.dataset import QXSSAROPT
from src.data.transforms import (
    get_input_specific,
    get_optical_specific,
    get_resize_transform,
    get_sar_resize_transform,
    DATASET_NORM_STATS,
)


class QXSSAROPTDataModule(LightningDataModule):
    """LightningDataModule for the QXSLAB SAR-OPT dataset.

    API mirrors :class:`SEN12FullDataModule` for drop-in train.py swap.
    ``scenes`` / ``seasons`` args are accepted but ignored — QXS layout is flat
    (no scene partitioning).  Pass them as ``None`` from a shared config.

    Args:
        data_dir:              Path to QXSLAB_SAROPT root.
        batch_size:            Train batch size.
        image_size:            Resize all patches to this square size before augmentation.
        num_workers:           DataLoader workers per loader.
        persistent_workers:    Keep workers alive between epochs.
        prefetch_factor:       Batches to prefetch per worker.
        train_val_split_ratio: Fraction of data used for training (rest = val).
        seed:                  RNG seed for reproducible split.
        sar_channels:          1 (grayscale) or 3.
        use_augmentation:      Apply common geometric augmentations to train split.
        scenes:                Ignored (QXS has no scene folders).
        seasons:               Ignored (QXS has no season folders).
        train_crop_size:       If set, train pipeline appends ``A.RandomCrop``
                               after the deterministic resize to ``image_size``.
        val_batch_size:        Val batch size.  Defaults to ``batch_size`` when None.
        sar_subdir:            Override SAR subdirectory name.
        opt_subdir:            Override optical subdirectory name.
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
        scenes: Optional[List[str]] = None,
        seasons: Optional[List[str]] = None,
        train_crop_size: Optional[int] = None,
        val_batch_size: Optional[int] = None,
        sar_subdir: str = "sar_256_oc_0.2",
        opt_subdir: str = "opt_256_oc_0.2",
        sar_lognorm: bool = False,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size if val_batch_size is not None else batch_size
        self.image_size = image_size
        self.train_crop_size = train_crop_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers if num_workers > 0 else False
        self.prefetch_factor = prefetch_factor if num_workers > 0 else None
        self.train_val_split_ratio = train_val_split_ratio
        self.seed = seed
        self.sar_channels = sar_channels
        self.sar_subdir = sar_subdir
        self.opt_subdir = opt_subdir
        # scenes/seasons accepted for API parity, intentionally unused.
        self._scenes_ignored = scenes
        self._seasons_ignored = seasons

        self.train_dataset: Optional[QXSSAROPT] = None
        self.val_dataset: Optional[QXSSAROPT] = None

        common_ops = []
        if train_crop_size is not None:
            common_ops.append(A.RandomCrop(train_crop_size, train_crop_size, p=1.0))
        if use_augmentation:
            # Horizontal flip only — SAR-safe geometry preservation (same rule
            # as SEN12FullDataModule).
            common_ops.append(A.HorizontalFlip(p=0.5))
        self.train_common_transform = (
            A.Compose(common_ops, additional_targets={'optical': 'image'})
            if common_ops else None
        )
        self.val_common_transform = None
        self.resize_transform = get_resize_transform(self.image_size)
        self.sar_resize_transform = get_sar_resize_transform(self.image_size)
        _stats = DATASET_NORM_STATS['qxslab']
        sar_mean = _stats['sar_mean'] * self.sar_channels
        sar_std = _stats['sar_std'] * self.sar_channels
        self.input_specific = get_input_specific(
            sar_channels=self.sar_channels,
            sar_mean=sar_mean,
            sar_std=sar_std,
            use_lognorm=sar_lognorm,
        )
        self.optical_specific = get_optical_specific(
            opt_mean=_stats['opt_mean'],
            opt_std=_stats['opt_std'],
        )

    def setup(self, stage: Optional[str] = None):
        full_dataset = QXSSAROPT(
            root_dir=self.data_dir,
            sar_channels=self.sar_channels,
            sar_subdir=self.sar_subdir,
            opt_subdir=self.opt_subdir,
        )
        all_items = full_dataset.items
        n_total = len(all_items)
        if n_total == 0:
            raise RuntimeError(
                f"QXSSAROPT found 0 items in '{self.data_dir}' "
                f"(sar_subdir='{self.sar_subdir}', opt_subdir='{self.opt_subdir}'). "
                "Check data_dir and subdirectory names."
            )

        n_train = int(n_total * self.train_val_split_ratio)
        n_val = n_total - n_train

        generator = torch.Generator().manual_seed(self.seed)
        train_idx, val_idx = random_split(range(n_total), [n_train, n_val], generator=generator)

        def make_dataset(indices, common_transform):
            return QXSSAROPT(
                root_dir=self.data_dir,
                common_transform=common_transform,
                input_specific=self.input_specific,
                optical_specific=self.optical_specific,
                resize_transform=self.resize_transform,
                sar_resize_transform=self.sar_resize_transform,
                sar_channels=self.sar_channels,
                sar_subdir=self.sar_subdir,
                opt_subdir=self.opt_subdir,
                items=[all_items[i] for i in indices],
            )

        self.train_dataset = make_dataset(train_idx.indices, self.train_common_transform)
        self.val_dataset = make_dataset(val_idx.indices, self.val_common_transform)

        crop_str = f"crop={self.train_crop_size}" if self.train_crop_size else "no-crop"
        print(
            f"[QXSSAROPTDataModule] "
            f"train={len(self.train_dataset)}@{self.image_size}->{crop_str} bs={self.batch_size} | "
            f"val={len(self.val_dataset)}@{self.image_size} bs={self.val_batch_size} "
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
            batch_size=self.val_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
            pin_memory=True,
        )
