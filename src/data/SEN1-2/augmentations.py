# src/data/augmentations.py

"""
Аугментации и нормализация для задачи SAR->Optical на датасете SEN1-2.

Принципы:
- Работаем с PNG-патчами 256x256, уже приведёнными авторами к [0,1] перед сохранением.
- В аугментациях не трогаем радиометрию (никаких логов и клиппингов).
- Геометрические аугментации только на train, синхронно для SAR и OPT.
- Нормализация под GAN: значения в диапазоне около [-1,1], формат тензоров [C,H,W].
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


def get_train_geo(image_size: int = 256) -> A.Compose:
    """
    Геометрические аугментации для TRAIN.

    Делает:
    - Resize до image_size;
    - случайные флипы и повороты на 0/90/180/270;
    - небольшой Shift/Scale/Rotate.

    additional_targets={'sar': 'image'} гарантирует
    идентичные трансформации для SAR и Optical.
    """
    return A.Compose(
        [
            A.Resize(image_size, image_size),

            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),

            A.Affine(
                scale=(0.9, 1.1),
                translate_percent=(-0.03, 0.03),
                rotate=(-5, 5),
                p=0.7,
            )
        ],
        additional_targets={"sar": "image"},
    )


def get_val_geo(image_size: int = 256) -> A.Compose:
    """
    Геометрический препроцессинг для VAL/TEST.

    Только детерминированный Resize.
    """
    return A.Compose(
        [
            A.Resize(image_size, image_size),
        ],
        additional_targets={"sar": "image"},
    )


def get_sar_norm() -> A.Compose:
    """
    Нормализация SAR-патчей.

    Ожидаем на входе:
    - numpy-массив (H,W,1) float32 в диапазоне [0,1].

    Делаем:
    - A.Normalize(mean=[0.5], std=[0.5], max_pixel_value=1.0)
      что приводит [0,1] примерно к [-1,1];
    - ToTensorV2() -> torch.Tensor формы [C,H,W].
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
    Нормализация Optical-патчей.

    Ожидаем на входе:
    - numpy-массив (H,W,3) float32 в диапазоне [0,1].
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
