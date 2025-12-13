# src/data/dataset.py

import os
from glob import glob
from typing import List, Dict, Optional

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class SEN12Dataset(Dataset):
    """
    PyTorch Dataset для SEN1-2 в задаче SAR->Optical.

    - Здесь НЕ делаем никакого дополнительного лог-преобразования или перцентильного клиппинга.
      Всё это уже сделано на уровне исходных GeoTIFF авторским пайплайном SEN1-2.
    - В этом классе только:
        * чтение PNG,
        * перевод 0–255 -> [0,1],
        * синхронные геометрические аугментации (через Albumentations),
        * раздельная нормализация SAR/OPT (например, в [-1,1] + ToTensorV2).
    """

    def __init__(
        self,
        root_dir: str,
        season_dirs: List[str],
        transform: Optional[object] = None,
        sar_norm: Optional[object] = None,
        opt_norm: Optional[object] = None,
    ) -> None:
        """
        Parameters
        ----------
        root_dir : str
            Путь к корню SEN1-2 (где лежат папки ROIs1158_spring, ...).
        season_dirs : list of str
            Список подпапок-сезонов, которые нужно использовать.
            Например: ["ROIs1158_spring", "ROIs1868_summer",
                       "ROIs1970_fall", "ROIs2017_winter"].
        transform : albumentations.Compose or None
            Геометрические аугментации, применяемые СИНХРОННО
            к оптике и SAR. Должен быть создан с
            additional_targets={'sar': 'image'}.
        sar_norm : albumentations.Compose or None
            Пайплайн нормализации SAR (например, [0,1] -> [-1,1] + ToTensorV2()).
        opt_norm : albumentations.Compose or None
            Пайплайн нормализации оптики (аналогично).
        """
        self.root_dir = root_dir
        self.season_dirs = list(season_dirs)
        self.transform = transform
        self.sar_norm = sar_norm
        self.opt_norm = opt_norm

        self.samples: List[Dict[str, str]] = self._collect_pairs()

        if len(self.samples) == 0:
            raise ValueError(
                f"No SAR/OPT pairs found under {root_dir} "
                f"for seasons: {self.season_dirs}"
            )

    def _collect_pairs(self) -> List[Dict[str, str]]:
        """
        Собираем все пары (s1_..._pK.png, s2_..._pK.png) по указанным сезонам.

        Ожидаем структуру:
            root/ROIs1158_spring/
                s1_0/
                    ROIs1158_spring_s1_0_p1.png
                    ROIs1158_spring_s1_0_p2.png
                    ...
                s1_1/
                    ...
                s2_0/
                    ROIs1158_spring_s2_0_p1.png
                    ...
                s2_1/
                    ...

        Логика:
        - в каждом сезоне ищем все папки s1_*;
        - внутри каждой такой папки собираем все PNG;
        - для каждого SAR-файла строим путь к соответствующему OPT-файлу:
            * заменяем 's1_' на 's2_' в имени папки;
            * и 's1_' на 's2_' в имени файла;
        - если оптический файл существует — добавляем пару.
        """
        samples: List[Dict[str, str]] = []

        for season in self.season_dirs:
            season_path = os.path.join(self.root_dir, season)
            if not os.path.isdir(season_path):
                continue

            # все подкаталоги вида s1_*
            sar_dirs = sorted(
                d for d in glob(os.path.join(season_path, "s1_*"))
                if os.path.isdir(d)
            )

            for sar_dir in sar_dirs:
                # соответствующий каталог для оптики: s1_X -> s2_X
                opt_dir = sar_dir.replace(os.sep + "s1_", os.sep + "s2_")
                if not os.path.isdir(opt_dir):
                    # если нет пары каталога, пропускаем этот ROI
                    continue

                sar_paths = sorted(glob(os.path.join(sar_dir, "*.png")))
                for sar_path in sar_paths:
                    fname = os.path.basename(sar_path)
                    # имя оптического файла: меняем _s1_ -> _s2_
                    opt_fname = fname.replace("_s1_", "_s2_")
                    opt_path = os.path.join(opt_dir, opt_fname)

                    if os.path.exists(opt_path):
                        samples.append(
                            {
                                "sar": sar_path,
                                "opt": opt_path,
                                "season": season,
                            }
                        )

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _load_sar(self, path: str) -> np.ndarray:
        """
        Чтение SAR PNG:
        - конвертируем в grayscale (L),
        - приводим к float32,
        - нормализуем 0–255 -> [0,1],
        - добавляем ось канала: (H, W) -> (H, W, 1).
        """
        img = Image.open(path).convert("L")        # (H, W), uint8
        arr = np.asarray(img, dtype=np.float32)    # (H, W)
        arr /= 255.0                               # [0,1]
        arr = arr[..., None]                       # (H, W, 1)
        return arr

    def _load_opt(self, path: str) -> np.ndarray:
        """
        Чтение Optical PNG:
        - конвертируем в RGB,
        - приводим к float32,
        - нормализуем 0–255 -> [0,1].
        """
        img = Image.open(path).convert("RGB")      # (H, W, 3), uint8
        arr = np.asarray(img, dtype=np.float32)    # (H, W, 3)
        arr /= 255.0                               # [0,1]
        return arr

    def __getitem__(self, idx: int) -> Dict[str, object]:
        sample = self.samples[idx]
        sar_path = sample["sar"]
        opt_path = sample["opt"]

        # 1. Чтение PNG
        sar = self._load_sar(sar_path)  # (H, W, 1), float32 in [0,1]
        opt = self._load_opt(opt_path)  # (H, W, 3), float32 in [0,1]

        # 2. Синхронные геометрические аугментации (train-режим)
        if self.transform is not None:
            augmented = self.transform(image=opt, sar=sar)
            opt = augmented["image"]
            sar = augmented["sar"]

        # 3. Раздельная нормализация (в т.ч. ToTensorV2)
        if self.sar_norm is not None:
            sar = self.sar_norm(image=sar)["image"]       # tensor [C,H,W]
        if self.opt_norm is not None:
            opt = self.opt_norm(image=opt)["image"]       # tensor [C,H,W]

        return {
            "sar": sar,               # torch.Tensor или np.ndarray, в зависимости от norm
            "optical": opt,           # то же
            "season": sample["season"]
        }

if __name__ == "__main__":
    """
    Sanity-check для SEN12Dataset + augmentations.py.

    Проверяем:
    - что пары SAR/OPT находятся;
    - формы и диапазоны значений до/после нормализации;
    - что геометрические аугментации из augmentations.py
      применяются синхронно (SAR и OPT искажены одинаково).
    """

    import random
    from collections import Counter

    import matplotlib.pyplot as plt
    import torch

    from augmentations import (
        get_train_geo,
        get_val_geo,
        get_sar_norm,
        get_opt_norm,
    )

    DATA_ROOT = "/Users/timur/Desktop/SEN1-2"  # заменить на свой путь

    # 1. Датасет БЕЗ аугментаций и нормализации (raw)
    ds_raw = SEN12Dataset(
        root_dir=DATA_ROOT,
        season_dirs=[
            "ROIs1158_spring",
            "ROIs1868_summer",
            "ROIs1970_fall",
            "ROIs2017_winter"
        ],
        transform=None,
        sar_norm=None,
        opt_norm=None,
    )

    print(f"Total samples (raw): {len(ds_raw)}")
    season_counts = Counter(s["season"] for s in ds_raw.samples)
    print("Samples per season:")
    for s, c in season_counts.items():
        print(f"  {s}: {c}")

    # 2. Датасет С аугментациями train-режима + нормализациями
    train_geo = get_train_geo(image_size=256)
    sar_norm = get_sar_norm()
    opt_norm = get_opt_norm()

    ds_aug = SEN12Dataset(
        root_dir=DATA_ROOT,
        season_dirs=[
            "ROIs1158_spring",
            "ROIs1868_summer",
            "ROIs1970_fall",
            "ROIs2017_winter"
        ],
        transform=train_geo,
        sar_norm=sar_norm,
        opt_norm=opt_norm,
    )

    # Выберем несколько индексов
    indices = random.sample(range(len(ds_raw)), k=min(3, len(ds_raw)))

    for idx in indices:
        meta = ds_raw.samples[idx]

        # --- RAW версии (без аугментаций/нормализации) ---
        sar_raw = ds_raw._load_sar(meta["sar"])      # (H,W,1), [0,1]
        opt_raw = ds_raw._load_opt(meta["opt"])      # (H,W,3), [0,1]

        # --- AUG + NORM версии ---
        item_aug = ds_aug[idx]
        sar_t = item_aug["sar"]          # tensor [1,H,W] ~ [-1,1]
        opt_t = item_aug["optical"]      # tensor [3,H,W] ~ [-1,1]

        # Для отображения переведём тензоры обратно в [0,1]
        sar_aug = sar_t.clone().cpu().numpy()[0]     # [H,W]
        # SAR нормализован через mean=0.5,std=0.5 при входе [0,1]:
        # x_norm = (x - 0.5)/0.5 => x = 0.5 * x_norm + 0.5
        sar_aug = 0.5 * sar_aug + 0.5
        sar_aug = sar_aug.clip(0.0, 1.0)

        opt_aug = opt_t.clone().cpu().numpy()        # [3,H,W]
        # Аналогично: x = 0.5 * x_norm + 0.5
        opt_aug = 0.5 * opt_aug + 0.5
        opt_aug = opt_aug.clip(0.0, 1.0)
        opt_aug = opt_aug.transpose(1, 2, 0)         # [H,W,3]

        # --- Печатаем базовые статистики ---
        print(f"\nIndex {idx}, season={meta['season']}")
        print(f"  SAR raw shape: {sar_raw.shape}, min/max: {sar_raw.min():.3f}/{sar_raw.max():.3f}")
        print(f"  OPT raw shape: {opt_raw.shape}, min/max: {opt_raw.min():.3f}/{opt_raw.max():.3f}")
        print(f"  SAR aug tensor min/max: {sar_t.min().item():.3f}/{sar_t.max().item():.3f}")
        print(f"  OPT aug tensor min/max: {opt_t.min().item():.3f}/{opt_t.max().item():.3f}")

        # --- Рисуем 4 картинки: raw SAR/OPT и aug SAR/OPT ---
        fig, axes = plt.subplots(1, 4, figsize=(10, 3))
        fig.suptitle(f"idx={idx}, season={meta['season']}", fontsize=10)

        axes[0].imshow(sar_raw[..., 0], cmap="gray", vmin=0.0, vmax=1.0)
        axes[0].set_title("SAR raw")
        axes[0].axis("off")

        axes[1].imshow(opt_raw)
        axes[1].set_title("OPT raw")
        axes[1].axis("off")

        axes[2].imshow(sar_aug, cmap="gray", vmin=0.0, vmax=1.0)
        axes[2].set_title("SAR aug")
        axes[2].axis("off")

        axes[3].imshow(opt_aug)
        axes[3].set_title("OPT aug")
        axes[3].axis("off")

        plt.tight_layout()
        plt.show()
