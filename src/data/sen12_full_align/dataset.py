# src/data/sen12_full_align/dataset.py
#
# Aligned mirror of ``src/data/sen12_full/dataset.py`` (copied verbatim).
# The class is intentionally layout-identical: ``data/sen12_full_align`` has the
# exact same ``<season>/s1_<scene>/`` + ``s2_<scene>/`` structure as the raw
# ``data/sen12_full``, so no code change is needed here — only the data root
# differs.  The aligned tree is produced by ``scripts/align_pairs.py`` (optical
# warped into the SAR frame; SAR copied unchanged).  Kept as its own module so
# the aligned pipeline owns its loader and edits never touch the raw one.

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Callable, Optional, Tuple, List

class SEN12Full(Dataset):
    """
    PyTorch Dataset for SAR-to-Optical image translation using the SEN1-2 dataset.

    Directory layout expected:
        root_dir/
          <season>/          e.g. ROIs1158_spring, ROIs1868_summer, ...
            s1_<scene>/      SAR patches, e.g. s1_5, s1_45, s1_52, s1_84, s1_100
            s2_<scene>/      Optical patches (paired by filename substitution s1→s2)

    Args:
        root_dir:         Path to SEN1-2 root (contains season subdirs).
        common_transform: Albumentations transform applied equally to SAR+optical.
        input_specific:   Per-modality transform for SAR.
        optical_specific: Per-modality transform for optical.
        resize_transform: Resize applied before augmentation.
        sar_channels:     1 (grayscale) or 3.
        seasons:          List of season folder names to include. None = all.
        scenes:           List of scene IDs (str) to include, e.g. ["5","45","52","84","100"].
                          None = all scenes found. Default paper selection: 5 landscape folders.
        items:            Pre-built item list to skip filesystem scan.
    """
    _ALLOWED_EXT = frozenset({'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'})

    PAPER_SCENES: List[str] = ["5", "45", "52", "84", "100"]

    def __init__(
        self,
        root_dir: str,
        common_transform: Optional[Callable] = None,
        input_specific: Optional[Callable] = None,
        optical_specific: Optional[Callable] = None,
        resize_transform: Optional[Callable] = None,
        sar_resize_transform: Optional[Callable] = None,
        sar_channels: int = 1,
        seasons: Optional[List[str]] = None,
        scenes: Optional[List[str]] = None,
        items: Optional[List[Tuple[str, str, str, str, str]]] = None,
    ):
        if sar_channels not in (1, 3):
            raise ValueError(f"sar_channels must be 1 or 3, got {sar_channels}")
        if not os.path.isdir(root_dir):
            raise FileNotFoundError(f"root_dir does not exist: {root_dir}")

        self.root_dir = root_dir

        if seasons is None:
            self.seasons = [
                d for d in sorted(os.listdir(self.root_dir))
                if os.path.isdir(os.path.join(self.root_dir, d))
                and not d.startswith('.')
            ]
        else:
            self.seasons = list(seasons)

        # scenes=None → no filter (all); scenes=[] → nothing
        self.scenes: Optional[set] = set(scenes) if scenes is not None else None

        self.common_transform = common_transform
        self.input_specific = input_specific
        self.optical_specific = optical_specific
        self.resize_transform = resize_transform
        # SAR-specific resize (nearest-neighbour by default if provided) so we
        # don't bilinear-blur speckle structure.  Falls back to `resize_transform`
        # when None for backward compatibility.
        self.sar_resize_transform = sar_resize_transform
        self.sar_channels = sar_channels

        self.items: List[Tuple[str, str, str, str, str]] = (
            self._collect_items() if items is None else list(items)
        )

    @staticmethod
    def _to_tensor(image: np.ndarray) -> torch.Tensor:
        if image.ndim == 2:
            image = image[..., None]
        image = image.astype(np.float32, copy=False)
        return torch.from_numpy(np.transpose(image, (2, 0, 1))).contiguous()

    @staticmethod
    def _build_s2_name(s1_name: str) -> str:
        if '_s1_' in s1_name:
            return s1_name.replace('_s1_', '_s2_')
        if s1_name.startswith('s1_'):
            return 's2_' + s1_name[3:]
        if 's1' in s1_name:
            return s1_name.replace('s1', 's2', 1)
        return s1_name

    def _collect_items(self) -> List[Tuple[str, str, str, str, str]]:
        items = []
        _isfile = os.path.isfile
        _join = os.path.join
        _build = self._build_s2_name
        _ext = self._ALLOWED_EXT

        for season in self.seasons:
            season_dir = _join(self.root_dir, season)
            if not os.path.isdir(season_dir):
                continue

            s1_dirs = sorted(
                d for d in os.listdir(season_dir)
                if os.path.isdir(_join(season_dir, d)) and d.startswith('s1_')
            )

            for s1_d in s1_dirs:
                # scene filter: s1_<id> → id must be in self.scenes
                scene_id = s1_d[3:]  # strip "s1_"
                if self.scenes is not None and scene_id not in self.scenes:
                    continue

                s2_d = 's2_' + scene_id
                s1_full_dir = _join(season_dir, s1_d)
                s2_full_dir = _join(season_dir, s2_d)

                if not os.path.isdir(s2_full_dir):
                    continue

                s1_files = sorted(
                    f for f in os.listdir(s1_full_dir)
                    if not f.startswith('.')
                    and _isfile(_join(s1_full_dir, f))
                    and os.path.splitext(f)[1].lower() in _ext
                )

                for fname in s1_files:
                    s2_fname = _build(fname)
                    if _isfile(_join(s2_full_dir, s2_fname)):
                        items.append((season, s1_d, s2_d, fname, s2_fname))

        return items

    def __len__(self) -> int:
        return len(self.items)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"root_dir='{self.root_dir}', "
            f"len={len(self)}, "
            f"seasons={self.seasons}, "
            f"scenes={sorted(self.scenes) if self.scenes else 'all'}, "
            f"sar_channels={self.sar_channels})"
        )

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        season, s1_d, s2_d, s1_fname, s2_fname = self.items[idx]

        sar_path = os.path.join(self.root_dir, season, s1_d, s1_fname)
        optical_path = os.path.join(self.root_dir, season, s2_d, s2_fname)

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
