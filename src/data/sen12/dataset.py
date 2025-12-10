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
    """
    def __init__(
        self,
        root_dir: str,
        common_transform: Optional[Callable] = None,
        input_specific: Optional[Callable] = None,
        optical_specific: Optional[Callable] = None,
        resize_transform: Optional[Callable] = None,
        sar_channels: int = 1
    ):
        self.root_dir = root_dir
        self.classes = ['agri', 'barrenland', 'grassland', 'urban']

        self.common_transform = common_transform
        self.input_specific = input_specific
        self.optical_specific = optical_specific
        self.resize_transform = resize_transform
        self.sar_channels = sar_channels

        self.items: List[Tuple[str, str]] = self._collect_items()

    def _collect_items(self) -> List[Tuple[str, str]]:
        items = []
        for cls in self.classes:
            s1_dir = os.path.join(self.root_dir, cls, 's1')
            s2_dir = os.path.join(self.root_dir, cls, 's2')
            if not os.path.isdir(s1_dir) or not os.path.isdir(s2_dir):
                continue
            s1_files = sorted([f for f in os.listdir(s1_dir) if not f.startswith('.') and os.path.isfile(os.path.join(s1_dir, f))])
            for fname in s1_files:
                # Заменяем s1 на s2 в имени файла
                s2_fname = fname.replace('_s1_', '_s2_')
                if os.path.isfile(os.path.join(s2_dir, s2_fname)):
                    items.append((cls, fname))
        return items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        cls, fname = self.items[idx]

        sar_path = os.path.join(self.root_dir, cls, 's1', fname)
        optical_path = os.path.join(self.root_dir, cls, 's2', fname.replace('_s1_', '_s2_'))

        sar = cv2.imread(sar_path, cv2.IMREAD_COLOR)
        sar = cv2.cvtColor(sar, cv2.COLOR_BGR2RGB)  # Предполагаем, что SAR хранится как RGB
        opt = cv2.imread(optical_path, cv2.IMREAD_COLOR)
        opt = cv2.cvtColor(opt, cv2.COLOR_BGR2RGB)

        if self.resize_transform:
            sar = self.resize_transform(image=sar)['image']
            opt = self.resize_transform(image=opt)['image']

        if self.common_transform:
            aug = self.common_transform(
                image=sar,
                optical=opt
            )
            sar_aug = aug['image']
            opt_aug = aug['optical']
        else:
            sar_aug, opt_aug = sar, opt

        if self.sar_channels == 1:
            sar_gray = cv2.cvtColor(sar_aug, cv2.COLOR_RGB2GRAY)  # (H, W)
            inp_np = sar_gray[..., None]
        else:
            inp_np = sar_aug  # (H, W, 3)

        inp = self.input_specific(image=inp_np)['image']     # Tensor [1, H, W]
        out = self.optical_specific(image=opt_aug)['image']  # Tensor [3, H, W]

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
        sar_channels=3
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