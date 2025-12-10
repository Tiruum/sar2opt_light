# src/data/dataset.py

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Callable, Optional, Tuple, List

class SARToOpticalDataset(Dataset):
    """
    PyTorch Dataset for SAR-to-Optical image translation.

    Args:
        root_dir: str, root directory containing 'sar', 'label', 'optical' subdirs
        common_transform: albumentations transform applied equally to SAR, label, optical
        input_specific: transform for model input (SAR+label)
        optical_specific: transform for model output (optical)
        resize_transform: resize transform applied before augmentation
    """
    def __init__(
        self,
        root_dir: str,
        common_transform: Optional[Callable] = None,
        input_specific: Optional[Callable] = None,
        optical_specific: Optional[Callable] = None,
        resize_transform: Optional[Callable] = None
    ):
        self.label_dir   = os.path.join(root_dir, 'label')
        self.optical_dir = os.path.join(root_dir, 'optical')
        self.sar_dir     = os.path.join(root_dir, 'sar')

        self.common_transform = common_transform
        self.input_specific = input_specific
        self.optical_specific = optical_specific
        self.resize_transform = resize_transform

        self.items: List[Tuple[str, str]] = self._collect_items()

    def _collect_items(self) -> List[Tuple[str, str]]:
        items = []
        for cls in sorted(os.listdir(self.label_dir)):
            label_cls_dir = os.path.join(self.label_dir, cls)
            sar_cls_dir   = os.path.join(self.sar_dir, cls)
            opt_cls_dir   = os.path.join(self.optical_dir, cls)
            if not os.path.isdir(label_cls_dir):
                continue
            for fname in sorted(os.listdir(label_cls_dir)):
                if fname.startswith('.'):
                    continue
                if (os.path.isfile(os.path.join(sar_cls_dir, fname)) and
                    os.path.isfile(os.path.join(opt_cls_dir, fname))):
                    items.append((cls, fname))
        return items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        cls, fname = self.items[idx]

        label_path   = os.path.join(self.label_dir, cls, fname)
        optical_path = os.path.join(self.optical_dir, cls, fname)
        sar_path     = os.path.join(self.sar_dir, cls, fname)

        label = cv2.imread(label_path, cv2.IMREAD_COLOR)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)

        sar = cv2.imread(sar_path, cv2.IMREAD_GRAYSCALE)
        opt = cv2.imread(optical_path, cv2.IMREAD_COLOR)
        opt = cv2.cvtColor(opt, cv2.COLOR_BGR2RGB)

        if self.resize_transform:
            label = self.resize_transform(image=label)['image']
            sar   = self.resize_transform(image=sar)['image']
            opt   = self.resize_transform(image=opt)['image']

        sar_3ch = np.stack([sar, sar, sar], axis=-1)  # (H, W, 3)

        if self.common_transform:
            aug = self.common_transform(
                image=sar_3ch,
                label=label,
                optical=opt
            )
            sar_aug   = aug['image']
            label_aug = aug['label']
            opt_aug   = aug['optical']
        else:
            sar_aug, label_aug, opt_aug = sar_3ch, label, opt

        sar_1ch = sar_aug[..., 0][..., None]  # (H, W, 1)
        inp_np = np.concatenate([sar_1ch, label_aug], axis=2)  # (H, W, 4)

        inp = self.input_specific(image=inp_np)['image']     # Tensor [4, H, W]
        out = self.optical_specific(image=opt_aug)['image']  # Tensor [3, H, W]

        return inp, out

if __name__ == "__main__":
    from data.transforms import get_common_transform, get_input_specific, get_optical_specific, get_resize_transform
    import matplotlib.pyplot as plt
    from src.utils.config_loader import load_config
    cfg = load_config('configs/config.yaml')

    dataset = SARToOpticalDataset(
        root_dir=cfg.data.data_dir,
        common_transform=get_common_transform(),
        input_specific=get_input_specific(),
        optical_specific=get_optical_specific(),
        resize_transform=get_resize_transform(cfg.data.image_size)
    )

    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    for i, (inp, out) in enumerate(loader):
        print(f"[{i}] Input shape:  {inp.shape}")   # [B, 4, H, W]
        print(f"[{i}] Output shape: {out.shape}")   # [B, 3, H, W]

        # Для примера возьмём первый элемент батча
        sar = inp[0, 0].cpu().numpy()                # [H, W], float, [-1, 1]
        label = inp[0, 1:4].cpu().numpy()            # [3, H, W], float, [-1, 1]
        opt = out[0].cpu().numpy()                   # [3, H, W], float, [-1, 1]

        # Вернуть значения из [-1,1] в [0,1]
        sar = (sar * 0.5 + 0.5).clip(0,1)
        label = (label * 0.5 + 0.5).clip(0,1)
        opt = (opt * 0.5 + 0.5).clip(0,1)

        # Label и Optical: [3, H, W] -> [H, W, 3] для matplotlib
        label = np.transpose(label, (1, 2, 0))
        opt = np.transpose(opt, (1, 2, 0))

        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].imshow(sar, cmap='gray')
        axs[0].set_title('SAR (input, 1ch)')
        axs[1].imshow(label)
        axs[1].set_title('Label (RGB)')
        axs[2].imshow(opt)
        axs[2].set_title('Optical (target)')
        for ax in axs:
            ax.axis('off')
        plt.tight_layout()
        plt.show()
        
        break
