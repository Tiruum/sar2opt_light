# src/data/SEN1-2/datamodule.py

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from typing import Optional

from src.data.sen12_full.dataset import SEN12Full
from src.data.transforms import (
    get_common_transform,
    get_input_specific,
    get_optical_specific,
    get_resize_transform,
)

class SEN12FullDataModule(LightningDataModule):
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
        use_augmentation: bool = True
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

        self.train_dataset = None
        self.val_dataset = None

        self.train_common_transform = get_common_transform() if use_augmentation else None
        self.val_common_transform = None  # Нет аугментаций на валидации
        self.resize_transform = get_resize_transform(self.image_size)
        self.input_specific = get_input_specific(sar_channels=self.sar_channels)
        self.optical_specific = get_optical_specific()


    def setup(self, stage: Optional[str] = None):

        # Собираем полный датасет (только для сбора индексов!)
        full_dataset = SEN12Full(
            root_dir=self.data_dir,
            sar_channels=self.sar_channels
        )
        n_total = len(full_dataset)
        n_train = int(n_total * self.train_val_split_ratio)
        n_val = n_total - n_train

        generator = torch.Generator().manual_seed(self.seed)
        train_indices, val_indices = random_split(range(n_total), [n_train, n_val], generator=generator)

        all_items = full_dataset.items
        seasons = full_dataset.seasons

        def make_dataset(indices, common_transform):
            items = [all_items[i] for i in indices]
            return SEN12Full(
                root_dir=self.data_dir,
                common_transform=common_transform,
                input_specific=self.input_specific,
                optical_specific=self.optical_specific,
                resize_transform=self.resize_transform,
                sar_channels=self.sar_channels,
                seasons=seasons,
                items=items,
            )

        self.train_dataset = make_dataset(train_indices.indices, self.train_common_transform)
        self.val_dataset = make_dataset(val_indices.indices, self.val_common_transform)

    def train_dataloader(self):
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

    def val_dataloader(self):
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
