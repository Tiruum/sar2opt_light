# src/data/dataset.py

import os
from glob import glob
from typing import List, Dict, Optional

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class SEN12Dataset(Dataset):
    """
    PyTorch Dataset РґР»СЏ SEN1-2 РІ Р·Р°РґР°С‡Рµ SAR->Optical.

    - Р—РґРµСЃСЊ РќР• РґРµР»Р°РµРј РЅРёРєР°РєРѕРіРѕ РґРѕРїРѕР»РЅРёС‚РµР»СЊРЅРѕРіРѕ Р»РѕРі-РїСЂРµРѕР±СЂР°Р·РѕРІР°РЅРёСЏ РёР»Рё РїРµСЂС†РµРЅС‚РёР»СЊРЅРѕРіРѕ РєР»РёРїРїРёРЅРіР°.
      Р’СЃС‘ СЌС‚Рѕ СѓР¶Рµ СЃРґРµР»Р°РЅРѕ РЅР° СѓСЂРѕРІРЅРµ РёСЃС…РѕРґРЅС‹С… GeoTIFF Р°РІС‚РѕСЂСЃРєРёРј РїР°Р№РїР»Р°Р№РЅРѕРј SEN1-2.
    - Р’ СЌС‚РѕРј РєР»Р°СЃСЃРµ С‚РѕР»СЊРєРѕ:
        * С‡С‚РµРЅРёРµ PNG,
        * РїРµСЂРµРІРѕРґ 0вЂ“255 -> [0,1],
        * СЃРёРЅС…СЂРѕРЅРЅС‹Рµ РіРµРѕРјРµС‚СЂРёС‡РµСЃРєРёРµ Р°СѓРіРјРµРЅС‚Р°С†РёРё (С‡РµСЂРµР· Albumentations),
        * СЂР°Р·РґРµР»СЊРЅР°СЏ РЅРѕСЂРјР°Р»РёР·Р°С†РёСЏ SAR/OPT (РЅР°РїСЂРёРјРµСЂ, РІ [-1,1] + ToTensorV2).
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
            РџСѓС‚СЊ Рє РєРѕСЂРЅСЋ SEN1-2 (РіРґРµ Р»РµР¶Р°С‚ РїР°РїРєРё ROIs1158_spring, ...).
        season_dirs : list of str
            РЎРїРёСЃРѕРє РїРѕРґРїР°РїРѕРє-СЃРµР·РѕРЅРѕРІ, РєРѕС‚РѕСЂС‹Рµ РЅСѓР¶РЅРѕ РёСЃРїРѕР»СЊР·РѕРІР°С‚СЊ.
            РќР°РїСЂРёРјРµСЂ: ["ROIs1158_spring", "ROIs1868_summer",
                       "ROIs1970_fall", "ROIs2017_winter"].
        transform : albumentations.Compose or None
            Р“РµРѕРјРµС‚СЂРёС‡РµСЃРєРёРµ Р°СѓРіРјРµРЅС‚Р°С†РёРё, РїСЂРёРјРµРЅСЏРµРјС‹Рµ РЎРРќРҐР РћРќРќРћ
            Рє РѕРїС‚РёРєРµ Рё SAR. Р”РѕР»Р¶РµРЅ Р±С‹С‚СЊ СЃРѕР·РґР°РЅ СЃ
            additional_targets={'sar': 'image'}.
        sar_norm : albumentations.Compose or None
            РџР°Р№РїР»Р°Р№РЅ РЅРѕСЂРјР°Р»РёР·Р°С†РёРё SAR (РЅР°РїСЂРёРјРµСЂ, [0,1] -> [-1,1] + ToTensorV2()).
        opt_norm : albumentations.Compose or None
            РџР°Р№РїР»Р°Р№РЅ РЅРѕСЂРјР°Р»РёР·Р°С†РёРё РѕРїС‚РёРєРё (Р°РЅР°Р»РѕРіРёС‡РЅРѕ).
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
        РЎРѕР±РёСЂР°РµРј РІСЃРµ РїР°СЂС‹ (s1_..._pK.png, s2_..._pK.png) РїРѕ СѓРєР°Р·Р°РЅРЅС‹Рј СЃРµР·РѕРЅР°Рј.

        РћР¶РёРґР°РµРј СЃС‚СЂСѓРєС‚СѓСЂСѓ:
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

        Р›РѕРіРёРєР°:
        - РІ РєР°Р¶РґРѕРј СЃРµР·РѕРЅРµ РёС‰РµРј РІСЃРµ РїР°РїРєРё s1_*;
        - РІРЅСѓС‚СЂРё РєР°Р¶РґРѕР№ С‚Р°РєРѕР№ РїР°РїРєРё СЃРѕР±РёСЂР°РµРј РІСЃРµ PNG;
        - РґР»СЏ РєР°Р¶РґРѕРіРѕ SAR-С„Р°Р№Р»Р° СЃС‚СЂРѕРёРј РїСѓС‚СЊ Рє СЃРѕРѕС‚РІРµС‚СЃС‚РІСѓСЋС‰РµРјСѓ OPT-С„Р°Р№Р»Сѓ:
            * Р·Р°РјРµРЅСЏРµРј 's1_' РЅР° 's2_' РІ РёРјРµРЅРё РїР°РїРєРё;
            * Рё 's1_' РЅР° 's2_' РІ РёРјРµРЅРё С„Р°Р№Р»Р°;
        - РµСЃР»Рё РѕРїС‚РёС‡РµСЃРєРёР№ С„Р°Р№Р» СЃСѓС‰РµСЃС‚РІСѓРµС‚ вЂ” РґРѕР±Р°РІР»СЏРµРј РїР°СЂСѓ.
        """
        samples: List[Dict[str, str]] = []

        for season in self.season_dirs:
            season_path = os.path.join(self.root_dir, season)
            if not os.path.isdir(season_path):
                continue

            # РІСЃРµ РїРѕРґРєР°С‚Р°Р»РѕРіРё РІРёРґР° s1_*
            sar_dirs = sorted(
                d for d in glob(os.path.join(season_path, "s1_*"))
                if os.path.isdir(d)
            )

            for sar_dir in sar_dirs:
                # СЃРѕРѕС‚РІРµС‚СЃС‚РІСѓСЋС‰РёР№ РєР°С‚Р°Р»РѕРі РґР»СЏ РѕРїС‚РёРєРё: s1_X -> s2_X
                opt_dir = sar_dir.replace(os.sep + "s1_", os.sep + "s2_")
                if not os.path.isdir(opt_dir):
                    # РµСЃР»Рё РЅРµС‚ РїР°СЂС‹ РєР°С‚Р°Р»РѕРіР°, РїСЂРѕРїСѓСЃРєР°РµРј СЌС‚РѕС‚ ROI
                    continue

                sar_paths = sorted(glob(os.path.join(sar_dir, "*.png")))
                for sar_path in sar_paths:
                    fname = os.path.basename(sar_path)
                    # РёРјСЏ РѕРїС‚РёС‡РµСЃРєРѕРіРѕ С„Р°Р№Р»Р°: РјРµРЅСЏРµРј _s1_ -> _s2_
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
        Р§С‚РµРЅРёРµ SAR PNG:
        - РєРѕРЅРІРµСЂС‚РёСЂСѓРµРј РІ grayscale (L),
        - РїСЂРёРІРѕРґРёРј Рє float32,
        - РЅРѕСЂРјР°Р»РёР·СѓРµРј 0вЂ“255 -> [0,1],
        - РґРѕР±Р°РІР»СЏРµРј РѕСЃСЊ РєР°РЅР°Р»Р°: (H, W) -> (H, W, 1).
        """
        img = Image.open(path).convert("L")        # (H, W), uint8
        arr = np.asarray(img, dtype=np.float32)    # (H, W)
        arr /= 255.0                               # [0,1]
        arr = arr[..., None]                       # (H, W, 1)
        return arr

    def _load_opt(self, path: str) -> np.ndarray:
        """
        Р§С‚РµРЅРёРµ Optical PNG:
        - РєРѕРЅРІРµСЂС‚РёСЂСѓРµРј РІ RGB,
        - РїСЂРёРІРѕРґРёРј Рє float32,
        - РЅРѕСЂРјР°Р»РёР·СѓРµРј 0вЂ“255 -> [0,1].
        """
        img = Image.open(path).convert("RGB")      # (H, W, 3), uint8
        arr = np.asarray(img, dtype=np.float32)    # (H, W, 3)
        arr /= 255.0                               # [0,1]
        return arr

    def __getitem__(self, idx: int) -> Dict[str, object]:
        sample = self.samples[idx]
        sar_path = sample["sar"]
        opt_path = sample["opt"]

        # 1. Р§С‚РµРЅРёРµ PNG
        sar = self._load_sar(sar_path)  # (H, W, 1), float32 in [0,1]
        opt = self._load_opt(opt_path)  # (H, W, 3), float32 in [0,1]

        # 2. РЎРёРЅС…СЂРѕРЅРЅС‹Рµ РіРµРѕРјРµС‚СЂРёС‡РµСЃРєРёРµ Р°СѓРіРјРµРЅС‚Р°С†РёРё (train-СЂРµР¶РёРј)
        if self.transform is not None:
            augmented = self.transform(image=opt, sar=sar)
            opt = augmented["image"]
            sar = augmented["sar"]

        # 3. Р Р°Р·РґРµР»СЊРЅР°СЏ РЅРѕСЂРјР°Р»РёР·Р°С†РёСЏ (РІ С‚.С‡. ToTensorV2)
        if self.sar_norm is not None:
            sar = self.sar_norm(image=sar)["image"]       # tensor [C,H,W]
        if self.opt_norm is not None:
            opt = self.opt_norm(image=opt)["image"]       # tensor [C,H,W]

        return {
            "sar": sar,               # torch.Tensor РёР»Рё np.ndarray, РІ Р·Р°РІРёСЃРёРјРѕСЃС‚Рё РѕС‚ norm
            "optical": opt,           # С‚Рѕ Р¶Рµ
            "season": sample["season"]
        }

if __name__ == "__main__":
    """
    Sanity-check РґР»СЏ SEN12Dataset + augmentations.py.

    РџСЂРѕРІРµСЂСЏРµРј:
    - С‡С‚Рѕ РїР°СЂС‹ SAR/OPT РЅР°С…РѕРґСЏС‚СЃСЏ;
    - С„РѕСЂРјС‹ Рё РґРёР°РїР°Р·РѕРЅС‹ Р·РЅР°С‡РµРЅРёР№ РґРѕ/РїРѕСЃР»Рµ РЅРѕСЂРјР°Р»РёР·Р°С†РёРё;
    - С‡С‚Рѕ РіРµРѕРјРµС‚СЂРёС‡РµСЃРєРёРµ Р°СѓРіРјРµРЅС‚Р°С†РёРё РёР· augmentations.py
      РїСЂРёРјРµРЅСЏСЋС‚СЃСЏ СЃРёРЅС…СЂРѕРЅРЅРѕ (SAR Рё OPT РёСЃРєР°Р¶РµРЅС‹ РѕРґРёРЅР°РєРѕРІРѕ).
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

    DATA_ROOT = "/Users/timur/Desktop/SEN1-2"  # Р·Р°РјРµРЅРёС‚СЊ РЅР° СЃРІРѕР№ РїСѓС‚СЊ

    # 1. Р”Р°С‚Р°СЃРµС‚ Р‘Р•Р— Р°СѓРіРјРµРЅС‚Р°С†РёР№ Рё РЅРѕСЂРјР°Р»РёР·Р°С†РёРё (raw)
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

    # 2. Р”Р°С‚Р°СЃРµС‚ РЎ Р°СѓРіРјРµРЅС‚Р°С†РёСЏРјРё train-СЂРµР¶РёРјР° + РЅРѕСЂРјР°Р»РёР·Р°С†РёСЏРјРё
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

    # Р’С‹Р±РµСЂРµРј РЅРµСЃРєРѕР»СЊРєРѕ РёРЅРґРµРєСЃРѕРІ
    indices = random.sample(range(len(ds_raw)), k=min(3, len(ds_raw)))

    for idx in indices:
        meta = ds_raw.samples[idx]

        # --- RAW РІРµСЂСЃРёРё (Р±РµР· Р°СѓРіРјРµРЅС‚Р°С†РёР№/РЅРѕСЂРјР°Р»РёР·Р°С†РёРё) ---
        sar_raw = ds_raw._load_sar(meta["sar"])      # (H,W,1), [0,1]
        opt_raw = ds_raw._load_opt(meta["opt"])      # (H,W,3), [0,1]

        # --- AUG + NORM РІРµСЂСЃРёРё ---
        item_aug = ds_aug[idx]
        sar_t = item_aug["sar"]          # tensor [1,H,W] ~ [-1,1]
        opt_t = item_aug["optical"]      # tensor [3,H,W] ~ [-1,1]

        # Р”Р»СЏ РѕС‚РѕР±СЂР°Р¶РµРЅРёСЏ РїРµСЂРµРІРµРґС‘Рј С‚РµРЅР·РѕСЂС‹ РѕР±СЂР°С‚РЅРѕ РІ [0,1]
        sar_aug = sar_t.clone().cpu().numpy()[0]     # [H,W]
        # SAR РЅРѕСЂРјР°Р»РёР·РѕРІР°РЅ С‡РµСЂРµР· mean=0.5,std=0.5 РїСЂРё РІС…РѕРґРµ [0,1]:
        # x_norm = (x - 0.5)/0.5 => x = 0.5 * x_norm + 0.5
        sar_aug = 0.5 * sar_aug + 0.5
        sar_aug = sar_aug.clip(0.0, 1.0)

        opt_aug = opt_t.clone().cpu().numpy()        # [3,H,W]
        # РђРЅР°Р»РѕРіРёС‡РЅРѕ: x = 0.5 * x_norm + 0.5
        opt_aug = 0.5 * opt_aug + 0.5
        opt_aug = opt_aug.clip(0.0, 1.0)
        opt_aug = opt_aug.transpose(1, 2, 0)         # [H,W,3]

        # --- РџРµС‡Р°С‚Р°РµРј Р±Р°Р·РѕРІС‹Рµ СЃС‚Р°С‚РёСЃС‚РёРєРё ---
        print(f"\nIndex {idx}, season={meta['season']}")
        print(f"  SAR raw shape: {sar_raw.shape}, min/max: {sar_raw.min():.3f}/{sar_raw.max():.3f}")
        print(f"  OPT raw shape: {opt_raw.shape}, min/max: {opt_raw.min():.3f}/{opt_raw.max():.3f}")
        print(f"  SAR aug tensor min/max: {sar_t.min().item():.3f}/{sar_t.max().item():.3f}")
        print(f"  OPT aug tensor min/max: {opt_t.min().item():.3f}/{opt_t.max().item():.3f}")

        # --- Р РёСЃСѓРµРј 4 РєР°СЂС‚РёРЅРєРё: raw SAR/OPT Рё aug SAR/OPT ---
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