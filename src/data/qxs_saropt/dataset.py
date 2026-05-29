# src/data/qxs_saropt/dataset.py

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Callable, Optional, Tuple, List


class QXSSAROPT(Dataset):
    """PyTorch Dataset for QXSLAB SAR-OPT paired SAR-to-optical translation.

    Directory layout expected::

        root_dir/
          sar_256_oc_0.2/      SAR amplitude patches, <id>.png
          opt_256_oc_0.2/      Optical RGB patches, <id>.png (same filename)

    Pairing is by exact filename. 256x256 patches, ~20k pairs.

    Args:
        root_dir:               Path to QXSLAB_SAROPT root.
        common_transform:       Albumentations transform applied to SAR+optical jointly.
        input_specific:         Per-modality transform for SAR.
        optical_specific:       Per-modality transform for optical.
        resize_transform:       Resize applied to optical before augmentation.
        sar_resize_transform:   Resize applied to SAR (nearest-neighbour) before augmentation.
        sar_channels:           1 (grayscale) or 3.
        sar_subdir:             Override SAR subdirectory name.
        opt_subdir:             Override optical subdirectory name.
        items:                  Pre-built list of filenames to skip filesystem scan.
    """
    _ALLOWED_EXT = frozenset({'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'})

    def __init__(
        self,
        root_dir: str,
        common_transform: Optional[Callable] = None,
        input_specific: Optional[Callable] = None,
        optical_specific: Optional[Callable] = None,
        resize_transform: Optional[Callable] = None,
        sar_resize_transform: Optional[Callable] = None,
        sar_channels: int = 1,
        sar_subdir: str = "sar_256_oc_0.2",
        opt_subdir: str = "opt_256_oc_0.2",
        items: Optional[List[str]] = None,
    ):
        if sar_channels not in (1, 3):
            raise ValueError(f"sar_channels must be 1 or 3, got {sar_channels}")
        if not os.path.isdir(root_dir):
            raise FileNotFoundError(f"root_dir does not exist: {root_dir}")

        self.root_dir = root_dir
        self.sar_subdir = sar_subdir
        self.opt_subdir = opt_subdir
        self.sar_dir = os.path.join(root_dir, sar_subdir)
        self.opt_dir = os.path.join(root_dir, opt_subdir)
        if not os.path.isdir(self.sar_dir):
            raise FileNotFoundError(f"SAR subdir missing: {self.sar_dir}")
        if not os.path.isdir(self.opt_dir):
            raise FileNotFoundError(f"Optical subdir missing: {self.opt_dir}")

        self.common_transform = common_transform
        self.input_specific = input_specific
        self.optical_specific = optical_specific
        self.resize_transform = resize_transform
        self.sar_resize_transform = sar_resize_transform
        self.sar_channels = sar_channels

        self.items: List[str] = (
            self._collect_items() if items is None else list(items)
        )

    @staticmethod
    def _to_tensor(image: np.ndarray) -> torch.Tensor:
        if image.ndim == 2:
            image = image[..., None]
        image = image.astype(np.float32, copy=False)
        return torch.from_numpy(np.transpose(image, (2, 0, 1))).contiguous()

    def _collect_items(self) -> List[str]:
        _isfile = os.path.isfile
        _join = os.path.join
        _ext = self._ALLOWED_EXT

        sar_files = sorted(
            f for f in os.listdir(self.sar_dir)
            if not f.startswith('.')
            and _isfile(_join(self.sar_dir, f))
            and os.path.splitext(f)[1].lower() in _ext
        )

        items: List[str] = []
        for fname in sar_files:
            if _isfile(_join(self.opt_dir, fname)):
                items.append(fname)
        return items

    def __len__(self) -> int:
        return len(self.items)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"root_dir='{self.root_dir}', "
            f"len={len(self)}, "
            f"sar_subdir='{self.sar_subdir}', "
            f"opt_subdir='{self.opt_subdir}', "
            f"sar_channels={self.sar_channels})"
        )

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        fname = self.items[idx]
        sar_path = os.path.join(self.sar_dir, fname)
        optical_path = os.path.join(self.opt_dir, fname)

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
            sar_resize = self.sar_resize_transform or self.resize_transform
            sar = sar_resize(image=sar)['image']
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
