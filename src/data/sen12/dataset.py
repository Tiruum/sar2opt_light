# src/data/dataset.py

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Callable, Optional, Tuple, List

class SEN12(Dataset):
    """
    PyTorch Dataset for SAR-to-Optical image translation.

    Args:
        root_dir: str, root directory containing class subdirs with 's1' (SAR) and 's2' (optical) subdirs
        common_transform: albumentations transform applied equally to SAR and optical
        input_specific: transform for model input (SAR, 3ch)
        optical_specific: transform for model output (optical, 3ch)
        resize_transform: resize transform applied before augmentation
        classes: optional precomputed class names with s1/s2 folders
        items: optional precomputed (class, s1_filename, s2_filename) pairs
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
        classes: Optional[List[str]] = None,
        items: Optional[List[Tuple[str, str, str]]] = None,
    ):
        if sar_channels not in (1, 3):
            raise ValueError(f"sar_channels must be 1 or 3, got {sar_channels}")
        if not os.path.isdir(root_dir):
            raise FileNotFoundError(f"root_dir does not exist: {root_dir}")

        self.root_dir = root_dir
        if classes is None:
            self.classes = [
                d for d in sorted(os.listdir(self.root_dir))
                if os.path.isdir(os.path.join(self.root_dir, d))
                and os.path.isdir(os.path.join(self.root_dir, d, 's1'))
                and os.path.isdir(os.path.join(self.root_dir, d, 's2'))
            ]
        else:
            self.classes = list(classes)

        self.common_transform = common_transform
        self.input_specific = input_specific
        self.optical_specific = optical_specific
        self.resize_transform = resize_transform
        self.sar_channels = sar_channels

        self.items: List[Tuple[str, str, str]] = self._collect_items() if items is None else list(items)

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

    def _collect_items(self) -> List[Tuple[str, str, str]]:
        items = []
        _isfile = os.path.isfile
        _join = os.path.join
        _build = self._build_s2_name
        _ext = self._ALLOWED_EXT
        for cls in self.classes:
            s1_dir = _join(self.root_dir, cls, 's1')
            s2_dir = _join(self.root_dir, cls, 's2')
            s1_files = sorted(
                f for f in os.listdir(s1_dir)
                if not f.startswith('.')
                and _isfile(_join(s1_dir, f))
                and os.path.splitext(f)[1].lower() in _ext
            )
            for fname in s1_files:
                s2_fname = _build(fname)
                if _isfile(_join(s2_dir, s2_fname)):
                    items.append((cls, fname, s2_fname))
        return items

    def __len__(self) -> int:
        return len(self.items)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"root_dir='{self.root_dir}', "
            f"len={len(self)}, "
            f"classes={self.classes}, "
            f"sar_channels={self.sar_channels})"
        )

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        cls, s1_fname, s2_fname = self.items[idx]

        sar_path = os.path.join(self.root_dir, cls, 's1', s1_fname)
        optical_path = os.path.join(self.root_dir, cls, 's2', s2_fname)

        # --- Чтение оптического изображения (всегда RGB, uint8) ---
        opt = cv2.imread(optical_path, cv2.IMREAD_COLOR)
        if opt is None:
            raise FileNotFoundError(f"Cannot read optical image: {optical_path}")
        opt = cv2.cvtColor(opt, cv2.COLOR_BGR2RGB)

        # --- Чтение SAR: сразу в нужном формате ---
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

        # --- Синхронные геометрические аугментации ---
        if self.common_transform:
            aug = self.common_transform(image=sar, optical=opt)
            sar = aug['image']
            opt = aug['optical']

        # --- Подготовка SAR: нужная размерность каналов ---
        if self.sar_channels == 1:
            inp_np = sar[..., np.newaxis] if sar.ndim == 2 else sar[..., :1]
        else:
            if sar.ndim == 2:
                inp_np = np.stack([sar, sar, sar], axis=-1)
            else:
                inp_np = sar[..., :3]

        # --- Гарантируем 3 канала для оптики ---
        if opt.ndim == 2:
            opt = np.stack([opt, opt, opt], axis=-1)
        elif opt.shape[2] > 3:
            opt = opt[..., :3]

        # --- Нормализация и конвертация в тензор ---
        inp = self.input_specific(image=inp_np)['image'] if self.input_specific else self._to_tensor(inp_np)
        out = self.optical_specific(image=opt)['image'] if self.optical_specific else self._to_tensor(opt)

        return inp, out

if __name__ == "__main__":
    from src.data.transforms import get_common_transform, get_input_specific, get_optical_specific, get_resize_transform
    import matplotlib.pyplot as plt

    dataset = SEN12(
        root_dir="./data/sen12",
        common_transform=get_common_transform(),
        input_specific=get_input_specific(),
        optical_specific=get_optical_specific(),
        resize_transform=get_resize_transform(256),
        sar_channels=1
    )
    print(f"Dataset length: {len(dataset)}")
    if len(dataset) == 0:
        print("Датасет пустой. Проверьте структуру папок и путь root_dir.")
        print("Пример ожидаемого пути: sen12-data/v_2")
        # Для тестирования используйте полный путь или относительный
        # dataset = SEN12(root_dir="sen12-data/v_2", ...)
        exit(1)

    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    for i, (inp, out) in enumerate(loader):
        print(f"[{i}] Input shape:  {inp.shape}")   # [B, 3, H, W]
        print(f"[{i}] Output shape: {out.shape}")   # [B, 3, H, W]

        # Для примера возьмём первый элемент батча
        sar = inp[0, 0].cpu().numpy()                # [H, W], float, [-1, 1] — первый канал SAR для grayscale
        opt = out[0].cpu().numpy()                   # [3, H, W], float, [-1, 1]

        # Вернуть значения из [-1,1] в [0,1]
        sar = (sar * 0.5 + 0.5).clip(0,1)
        opt = (opt * 0.5 + 0.5).clip(0,1)

        # Optical: [3, H, W] -> [H, W, 3] для matplotlib
        opt = np.transpose(opt, (1, 2, 0))

        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        axs[0].imshow(sar, cmap='gray')
        axs[0].set_title('SAR (input, 1ch from 3ch)')
        axs[1].imshow(opt)
        axs[1].set_title('Optical (target)')
        for ax in axs:
            ax.axis('off')
        plt.tight_layout()
        plt.show()
        
        break