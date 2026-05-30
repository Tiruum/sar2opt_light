# src/data/sen12_full/datamodule.py

import albumentations as A
import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, random_split
from typing import Optional, List

from src.data.sen12_full.dataset import SEN12Full
from src.data.transforms import (
    get_common_transform,
    get_input_specific,
    get_optical_specific,
    get_resize_transform,
    get_sar_resize_transform,
    DATASET_NORM_STATS,
)


class SEN12FullDataModule(LightningDataModule):
    """
    LightningDataModule for the SEN1-2 dataset.

    Args:
        data_dir:              Path to SEN1-2 root (contains season subdirs).
        batch_size:            Train batch size.
        image_size:            Resize all patches to this square size before augmentation.
        num_workers:           DataLoader workers per loader.
        persistent_workers:    Keep workers alive between epochs.
        prefetch_factor:       Batches to prefetch per worker.
        train_val_split_ratio: Fraction of data used for training (rest = val).
        seed:                  RNG seed for reproducible split.
        sar_channels:          1 (grayscale) or 3.
        use_augmentation:      Apply common geometric augmentations to train split.
        scenes:                Scene IDs to include, e.g. ["5","45","52","84","100"].
                               Default = paper selection (5 landscape folders from SEN1-2).
                               Pass None to use ALL scenes in the dataset.
        seasons:               Season folder names to include. None = all found.
        train_crop_size:       If set, train pipeline appends ``A.RandomCrop(s, s, p=1)``
                               *after* the deterministic resize to ``image_size``.  Lets
                               the model train on N×N crops while val keeps the full
                               ``image_size`` for comparable PSNR/LPIPS/FID numbers.
                               None = no crop (legacy behaviour).
        val_batch_size:        Val batch size.  Defaults to ``batch_size`` when None,
                               but at higher val resolutions (e.g. train@128 / val@256)
                               you usually want a smaller batch to fit VRAM.
    """

    # Paper default: suburbs(S5), dense urban(S45), farmland(S52), rivers(S84), hills(S100)
    DEFAULT_SCENES: List[str] = ["5", "45", "52", "84", "100"]

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
        scenes: Optional[List[str]] = DEFAULT_SCENES,
        seasons: Optional[List[str]] = None,
        train_crop_size: Optional[int] = None,
        val_batch_size: Optional[int] = None,
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
        self.scenes = scenes
        self.seasons = seasons

        self.train_dataset: Optional[SEN12Full] = None
        self.val_dataset: Optional[SEN12Full] = None

        # ``train_common_transform`` is the per-sample augmentation pipeline applied
        # AFTER the deterministic resize to ``image_size``.  When ``train_crop_size``
        # is set we prepend ``A.RandomCrop`` so each step sees a fresh crop — this
        # is the speed lever for the v0.2 pass (train 128 / val 256).
        common_ops = []
        if train_crop_size is not None:
            common_ops.append(A.RandomCrop(train_crop_size, train_crop_size, p=1.0))
        if use_augmentation:
            # Mirror the geometry rationale from ``get_common_transform``: only
            # horizontal flip is SAR-safe (vertical / 90° rotations corrupt
            # azimuth-direction speckle vs paired optical).
            common_ops.append(A.HorizontalFlip(p=0.5))
        self.train_common_transform = (
            A.Compose(common_ops, additional_targets={'optical': 'image'})
            if common_ops else None
        )
        self.val_common_transform = None
        self.resize_transform = get_resize_transform(self.image_size)
        # SAR uses nearest-neighbour resize so speckle structure isn't blurred
        # by bilinear interpolation (which softens edges the wavelet branch
        # and discriminator depend on).
        self.sar_resize_transform = get_sar_resize_transform(self.image_size)
        _stats = DATASET_NORM_STATS['sen12_full']
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
        # Single filesystem scan for item collection
        full_dataset = SEN12Full(
            root_dir=self.data_dir,
            sar_channels=self.sar_channels,
            seasons=self.seasons,
            scenes=self.scenes,
        )
        all_items = full_dataset.items
        n_total = len(all_items)
        if n_total == 0:
            raise RuntimeError(
                f"SEN12Full found 0 items in '{self.data_dir}' "
                f"(scenes={self.scenes}, seasons={self.seasons}). "
                "Check data_dir and scene/season names."
            )

        n_train = int(n_total * self.train_val_split_ratio)
        n_val = n_total - n_train

        generator = torch.Generator().manual_seed(self.seed)
        train_idx, val_idx = random_split(range(n_total), [n_train, n_val], generator=generator)

        seasons_used = full_dataset.seasons

        def make_dataset(indices, common_transform):
            return SEN12Full(
                root_dir=self.data_dir,
                common_transform=common_transform,
                input_specific=self.input_specific,
                optical_specific=self.optical_specific,
                resize_transform=self.resize_transform,
                sar_resize_transform=self.sar_resize_transform,
                sar_channels=self.sar_channels,
                seasons=seasons_used,
                scenes=self.scenes,
                items=[all_items[i] for i in indices],
            )

        self.train_dataset = make_dataset(train_idx.indices, self.train_common_transform)
        self.val_dataset = make_dataset(val_idx.indices, self.val_common_transform)

        crop_str = f"crop={self.train_crop_size}" if self.train_crop_size else "no-crop"
        print(
            f"[SEN12FullDataModule] scenes={sorted(self.scenes) if self.scenes else 'all'} | "
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
