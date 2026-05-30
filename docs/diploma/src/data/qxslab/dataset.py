# src/data/qxslab/dataset.py

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Callable, Optional, Tuple, List


class QXSLABDataset(Dataset):
    """
    PyTorch Dataset for QXSLAB SAR-OPT image pairs.

    Directory layout:
        root_dir/
          sar_<variant>/   e.g. sar_256_oc_0.2/
            <N>.png
          opt_<variant>/
            <N>.png        (paired by identical filename)

    Args:
        root_dir:         Path to QXSLAB root (contains sar_*/opt_* dirs).
        common_transform: Albumentations transform applied equally to SAR+optical.
        input_specific:   Per-modality transform for SAR.
        optical_specific: Per-modality transform for optical.
        resize_transform: Resize applied before augmentation.
        sar_channels:     1 (grayscale) or 3.
        variants:         Variant suffixes to include, e.g. ["256_oc_0.2"].
                          None = all found.
        items:            Pre-built (variant, sar_fname, opt_fname) list to skip scan.
    """
    _ALLOWED_EXT = frozenset({'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'})

    def __init__(
        self,
        root_dir: str,
        common_transform: Optional[Callable] = None,
        input_specific: Optional[Callable] = None,
        optical_specific: Optional[Callable] = None,
        resize_transform: Optional[Callable] = None,
        sar_channels: int = 1,
        variants: Optional[List[str]] = None,
        items: Optional[List[Tuple[str, str, str]]] = None,
    ):
        if sar_channels not in (1, 3):
            raise ValueError(f"sar_channels must be 1 or 3, got {sar_channels}")
        if not os.path.isdir(root_dir):
            raise FileNotFoundError(f"root_dir does not exist: {root_dir}")

        self.root_dir = root_dir
        self.common_transform = common_transform
        self.input_specific = input_specific
        self.optical_specific = optical_specific
        self.resize_transform = resize_transform
        self.sar_channels = sar_channels

        if variants is None:
            self.variants = self._discover_variants()
        else:
            self.variants = list(variants)

        self.items: List[Tuple[str, str, str]] = (
            self._collect_items() if items is None else list(items)
        )

    def _discover_variants(self) -> List[str]:
        variants = []
        for d in sorted(os.listdir(self.root_dir)):
            if d.startswith('sar_') and os.path.isdir(os.path.join(self.root_dir, d)):
                suffix = d[4:]  # strip "sar_"
                opt_dir = os.path.join(self.root_dir, 'opt_' + suffix)
                if os.path.isdir(opt_dir):
                    variants.append(suffix)
        return variants

    def _collect_items(self) -> List[Tuple[str, str, str]]:
        items = []
        _isfile = os.path.isfile
        _join = os.path.join
        _ext = self._ALLOWED_EXT
        for variant in self.variants:
            sar_dir = _join(self.root_dir, 'sar_' + variant)
            opt_dir = _join(self.root_dir, 'opt_' + variant)
            if not os.path.isdir(sar_dir) or not os.path.isdir(opt_dir):
                continue
            for fname in sorted(
                f for f in os.listdir(sar_dir)
                if not f.startswith('.')
                and _isfile(_join(sar_dir, f))
                and os.path.splitext(f)[1].lower() in _ext
            ):
                if _isfile(_join(opt_dir, fname)):
                    items.append((variant, fname, fname))
        return items

    @staticmethod
    def _to_tensor(image: np.ndarray) -> torch.Tensor:
        if image.ndim == 2:
            image = image[..., None]
        image = image.astype(np.float32, copy=False)
        return torch.from_numpy(np.transpose(image, (2, 0, 1))).contiguous()

    def __len__(self) -> int:
        return len(self.items)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"root_dir='{self.root_dir}', "
            f"len={len(self)}, "
            f"variants={self.variants}, "
            f"sar_channels={self.sar_channels})"
        )

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        variant, sar_fname, opt_fname = self.items[idx]
        sar_path = os.path.join(self.root_dir, 'sar_' + variant, sar_fname)
        optical_path = os.path.join(self.root_dir, 'opt_' + variant, opt_fname)

        opt = cv2.imread(optical_path, cv2.IMREAD_COLOR)
        if opt is None:
            raise FileNotFoundError(f"Cannot read optical image: {optical_path}")
        opt = cv2.cvtColor(opt, cv2.COLOR_BGR2RGB)

        if self.sar_channels == 1:
            sar = cv2.imread(sar_path, cv2.IMREAD_GRAYSCALE)
        else:
            sar = cv2.imread(sar_path, cv2.IMREAD_COLOR)
        if sar is None:
            raise FileNotFoundError(f"Cannot read SAR image: {sar_path}")
        if self.sar_channels == 3 and sar.ndim == 3:
            sar = cv2.cvtColor(sar, cv2.COLOR_BGR2RGB)

        if self.resize_transform:
            sar = self.resize_transform(image=sar)['image']
            opt = self.resize_transform(image=opt)['image']

        if self.common_transform:
            aug = self.common_transform(image=sar, optical=opt)
            sar = aug['image']
            opt = aug['optical']

        if self.sar_channels == 1:
            inp_np = sar[..., np.newaxis] if sar.ndim == 2 else sar[..., :1]
        else:
            inp_np = np.stack([sar, sar, sar], axis=-1) if sar.ndim == 2 else sar[..., :3]

        if opt.ndim == 2:
            opt = np.stack([opt, opt, opt], axis=-1)
        elif opt.shape[2] > 3:
            opt = opt[..., :3]

        inp = self.input_specific(image=inp_np)['image'] if self.input_specific else self._to_tensor(inp_np)
        out = self.optical_specific(image=opt)['image'] if self.optical_specific else self._to_tensor(opt)

        return inp, out
