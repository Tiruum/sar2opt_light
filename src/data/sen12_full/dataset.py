# src/data/SEN1-2/dataset.py

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Callable, Optional, Tuple, List

class SEN12Full(Dataset):
    """
    PyTorch Dataset for SAR-to-Optical image translation for SEN1-2 dataset.

    Args:
        root_dir: str, root directory containing season subdirs with 's1_X' (SAR) and 's2_X' (optical) subdirs
        common_transform: albumentations transform applied equally to SAR and optical
        input_specific: transform for model input (SAR, 3ch or 1ch)
        optical_specific: transform for model output (optical, 3ch)
        resize_transform: resize transform applied before augmentation
        sar_channels: int, number of channels for SAR images (1 or 3)
        seasons: optional list of season directories to use
        items: optional precomputed (season, s1_dir, s2_dir, s1_filename, s2_filename) tuples
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
        seasons: Optional[List[str]] = None,
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
            ]
        else:
            self.seasons = list(seasons)

        self.common_transform = common_transform
        self.input_specific = input_specific
        self.optical_specific = optical_specific
        self.resize_transform = resize_transform
        self.sar_channels = sar_channels

        self.items: List[Tuple[str, str, str, str, str]] = self._collect_items() if items is None else list(items)

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
                
            # Find all s1_* directories
            s1_dirs = [d for d in os.listdir(season_dir) if os.path.isdir(_join(season_dir, d)) and d.startswith('s1_')]
            
            for s1_d in s1_dirs:
                s2_d = s1_d.replace('s1_', 's2_')
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
            f"sar_channels={self.sar_channels})"
        )

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        season, s1_d, s2_d, s1_fname, s2_fname = self.items[idx]

        sar_path = os.path.join(self.root_dir, season, s1_d, s1_fname)
        optical_path = os.path.join(self.root_dir, season, s2_d, s2_fname)

        # --- ?????? ??????????? ??????????? (?????? RGB, uint8) ---
        opt = cv2.imread(optical_path, cv2.IMREAD_COLOR)
        if opt is None:
            raise FileNotFoundError(f"Cannot read optical image: {optical_path}")
        opt = cv2.cvtColor(opt, cv2.COLOR_BGR2RGB)

        # --- ?????? SAR: ????? ? ?????? ??????? ---
        if self.sar_channels == 1:
            sar = cv2.imread(sar_path, cv2.IMREAD_GRAYSCALE)
        else:
            sar = cv2.imread(sar_path, cv2.IMREAD_COLOR)
        if sar is None:
            raise FileNotFoundError(f"Cannot read SAR image: {sar_path}")
        if self.sar_channels == 3 and sar.ndim == 3:
            sar = cv2.cvtColor(sar, cv2.COLOR_BGR2RGB)

        # --- Resize ---
        if self.resize_transform:
            sar = self.resize_transform(image=sar)['image']
            opt = self.resize_transform(image=opt)['image']

        # --- ?????????? ?????????????? ??????????? ---
        if self.common_transform:
            aug = self.common_transform(image=sar, optical=opt)
            sar = aug['image']
            opt = aug['optical']

        # --- ?????????? SAR: ?????? ??????????? ??????? ---
        if self.sar_channels == 1:
            inp_np = sar[..., np.newaxis] if sar.ndim == 2 else sar[..., :1]
        else:
            if sar.ndim == 2:
                inp_np = np.stack([sar, sar, sar], axis=-1)
            else:
                inp_np = sar[..., :3]

        # --- ??????????? 3 ?????? ??? ?????? ---
        if opt.ndim == 2:
            opt = np.stack([opt, opt, opt], axis=-1)
        elif opt.shape[2] > 3:
            opt = opt[..., :3]

        # --- ???????????? ? ??????????? ? ?????? ---
        inp = self.input_specific(image=inp_np)['image'] if self.input_specific else self._to_tensor(inp_np)
        out = self.optical_specific(image=opt)['image'] if self.optical_specific else self._to_tensor(opt)

        return inp, out

if __name__ == '__main__':
    from src.data.transforms import get_common_transform, get_input_specific, get_optical_specific, get_resize_transform
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader

    dataset = SEN12Full(
        root_dir='data/SEN1-2',
        common_transform=get_common_transform(),
        input_specific=get_input_specific(sar_channels=1),
        optical_specific=get_optical_specific(),
        resize_transform=get_resize_transform(256),
        sar_channels=1
    )
    print(f'Dataset length: {len(dataset)}')
    if len(dataset) == 0:
        print('Dataset is empty. Check root_dir.')
        exit(1)

    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    for i, (inp, out) in enumerate(loader):
        print(f'[{i}] Input shape:  {inp.shape}')
        print(f'[{i}] Output shape: {out.shape}')

        sar = inp[0, 0].cpu().numpy()
        opt = out[0].cpu().numpy()

        sar = (sar * 0.5 + 0.5).clip(0, 1)
        opt = (opt * 0.5 + 0.5).clip(0, 1)

        opt = np.transpose(opt, (1, 2, 0))

        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        axs[0].imshow(sar, cmap='gray')
        axs[0].set_title('SAR (input)')
        axs[1].imshow(opt)
        axs[1].set_title('Optical (target)')
        for ax in axs:
            ax.axis('off')
        plt.tight_layout()
        plt.show()
        break
