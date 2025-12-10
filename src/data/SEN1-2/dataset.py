# src/data/sen12_dataset.py
import os
from glob import glob

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class SEN12Dataset(Dataset):
    def __init__(
        self,
        root_dir,
        season_dirs,          # список строк: ["ROIs1158_spring", ...]
        transform=None,       # синхронная геометрия (Albumentations)
        sar_norm=None,        # нормализация SAR
        opt_norm=None,        # нормализация Optical
    ):
        """
        root_dir: путь к папке SEN1-2 (где лежат ROIs1158_spring, ...).
        season_dirs: подкаталоги, которые используем (train/val разные).
        """
        self.root_dir = root_dir
        self.season_dirs = season_dirs
        self.transform = transform
        self.sar_norm = sar_norm
        self.opt_norm = opt_norm

        self.samples = self._collect_pairs()

    def _collect_pairs(self):
        samples = []
        for season in self.season_dirs:
            season_path = os.path.join(self.root_dir, season)
            # все SAR-файлы этого сезона
            sar_paths = sorted(glob(os.path.join(season_path, "s1_*.png")))
            for sar_path in sar_paths:
                fname = os.path.basename(sar_path)  # s1_12345.png
                idx_str = fname.split("_")[1].split(".")[0]  # "12345"
                opt_path = os.path.join(season_path, f"s2_{idx_str}.png")
                if os.path.exists(opt_path):
                    samples.append(
                        {
                            "sar": sar_path,
                            "opt": opt_path,
                            "season": season,
                        }
                    )
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        sar_path = sample["sar"]
        opt_path = sample["opt"]

        # 1. Чтение PNG
        sar = np.array(Image.open(sar_path).convert("L"), dtype=np.float32)  # (H, W)
        opt = np.array(Image.open(opt_path).convert("RGB"), dtype=np.float32)  # (H, W, 3)

        # 2. Приводим к [0,1]
        sar = sar / 255.0
        opt = opt / 255.0

        # 3. Добавляем канал SAR => (H, W, 1) для Albumentations
        sar = sar[..., None]

        # 4. Синхронные геометрические аугментации
        if self.transform is not None:
            augmented = self.transform(image=opt, sar=sar)
            opt = augmented["image"]
            sar = augmented["sar"]

        # 5. Раздельная нормализация в [-1,1] и ToTensorV2
        if self.sar_norm is not None:
            sar = self.sar_norm(image=sar)["image"]
        if self.opt_norm is not None:
            opt = self.opt_norm(image=opt)["image"]

        return {
            "sar": sar,        # tensor [1, H, W]
            "optical": opt,    # tensor [3, H, W]
            "season": sample["season"],
        }
