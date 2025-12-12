# src/data/augmentations.py

"""
РђСѓРіРјРµРЅС‚Р°С†РёРё Рё РЅРѕСЂРјР°Р»РёР·Р°С†РёСЏ РґР»СЏ Р·Р°РґР°С‡Рё SAR->Optical РЅР° РґР°С‚Р°СЃРµС‚Рµ SEN1-2.

РџСЂРёРЅС†РёРїС‹:
- Р Р°Р±РѕС‚Р°РµРј СЃ PNG-РїР°С‚С‡Р°РјРё 256x256, СѓР¶Рµ РїСЂРёРІРµРґС‘РЅРЅС‹РјРё Р°РІС‚РѕСЂР°РјРё Рє [0,1] РїРµСЂРµРґ СЃРѕС…СЂР°РЅРµРЅРёРµРј.
- Р’ Р°СѓРіРјРµРЅС‚Р°С†РёСЏС… РЅРµ С‚СЂРѕРіР°РµРј СЂР°РґРёРѕРјРµС‚СЂРёСЋ (РЅРёРєР°РєРёС… Р»РѕРіРѕРІ Рё РєР»РёРїРїРёРЅРіРѕРІ).
- Р“РµРѕРјРµС‚СЂРёС‡РµСЃРєРёРµ Р°СѓРіРјРµРЅС‚Р°С†РёРё С‚РѕР»СЊРєРѕ РЅР° train, СЃРёРЅС…СЂРѕРЅРЅРѕ РґР»СЏ SAR Рё OPT.
- РќРѕСЂРјР°Р»РёР·Р°С†РёСЏ РїРѕРґ GAN: Р·РЅР°С‡РµРЅРёСЏ РІ РґРёР°РїР°Р·РѕРЅРµ РѕРєРѕР»Рѕ [-1,1], С„РѕСЂРјР°С‚ С‚РµРЅР·РѕСЂРѕРІ [C,H,W].
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


def get_train_geo(image_size: int = 256) -> A.Compose:
    """
    Р“РµРѕРјРµС‚СЂРёС‡РµСЃРєРёРµ Р°СѓРіРјРµРЅС‚Р°С†РёРё РґР»СЏ TRAIN.

    Р”РµР»Р°РµС‚:
    - Resize РґРѕ image_size;
    - СЃР»СѓС‡Р°Р№РЅС‹Рµ С„Р»РёРїС‹ Рё РїРѕРІРѕСЂРѕС‚С‹ РЅР° 0/90/180/270;
    - РЅРµР±РѕР»СЊС€РѕР№ Shift/Scale/Rotate.

    additional_targets={'sar': 'image'} РіР°СЂР°РЅС‚РёСЂСѓРµС‚
    РёРґРµРЅС‚РёС‡РЅС‹Рµ С‚СЂР°РЅСЃС„РѕСЂРјР°С†РёРё РґР»СЏ SAR Рё Optical.
    """
    return A.Compose(
        [
            A.Resize(image_size, image_size),

            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),

            A.Affine(
                scale=(0.9, 1.1),
                translate_percent=(-0.05, 0.05),
                rotate=(-10, 10),
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.9,
            )
        ],
        additional_targets={"sar": "image"},
    )


def get_val_geo(image_size: int = 256) -> A.Compose:
    """
    Р“РµРѕРјРµС‚СЂРёС‡РµСЃРєРёР№ РїСЂРµРїСЂРѕС†РµСЃСЃРёРЅРі РґР»СЏ VAL/TEST.

    РўРѕР»СЊРєРѕ РґРµС‚РµСЂРјРёРЅРёСЂРѕРІР°РЅРЅС‹Р№ Resize.
    """
    return A.Compose(
        [
            A.Resize(image_size, image_size),
        ],
        additional_targets={"sar": "image"},
    )


def get_sar_norm() -> A.Compose:
    """
    РќРѕСЂРјР°Р»РёР·Р°С†РёСЏ SAR-РїР°С‚С‡РµР№.

    РћР¶РёРґР°РµРј РЅР° РІС…РѕРґРµ:
    - numpy-РјР°СЃСЃРёРІ (H,W,1) float32 РІ РґРёР°РїР°Р·РѕРЅРµ [0,1].

    Р”РµР»Р°РµРј:
    - A.Normalize(mean=[0.5], std=[0.5], max_pixel_value=1.0)
      С‡С‚Рѕ РїСЂРёРІРѕРґРёС‚ [0,1] РїСЂРёРјРµСЂРЅРѕ Рє [-1,1];
    - ToTensorV2() -> torch.Tensor С„РѕСЂРјС‹ [C,H,W].
    """
    return A.Compose(
        [
            A.Normalize(
                mean=[0.5],
                std=[0.5],
                max_pixel_value=1.0,
            ),
            ToTensorV2(),
        ]
    )


def get_opt_norm() -> A.Compose:
    """
    РќРѕСЂРјР°Р»РёР·Р°С†РёСЏ Optical-РїР°С‚С‡РµР№.

    РћР¶РёРґР°РµРј РЅР° РІС…РѕРґРµ:
    - numpy-РјР°СЃСЃРёРІ (H,W,3) float32 РІ РґРёР°РїР°Р·РѕРЅРµ [0,1].
    """

    return A.Compose(
        [
            A.Normalize(
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5),
                max_pixel_value=1.0,
            ),
            ToTensorV2(),
        ]
    )