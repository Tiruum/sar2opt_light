# src/data/transforms.py
import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2
from albumentations.core.transforms_interface import ImageOnlyTransform
import numpy as np

class SARPreprocessing(ImageOnlyTransform):
    """Кастомное преобразование для SAR-препроцессинга"""
    def __init__(
        self,
        log_transform=True,
        percentile_range=(1, 99), # SOTA: отсекаем 1% самых ярких/темных пикселей
        normalization='minmax',   # 'minmax' для приведения к [-1, 1], 'zscore' для [0, 1]
        always_apply=False,
        p=1.0
    ):
        super().__init__(always_apply, p)
        self.log_transform = log_transform
        self.percentile_range = percentile_range
        self.normalization = normalization
        
    def apply(self, image, **params):
        img = image.copy().astype(np.float32)
        
        # 1. Логарифмическое преобразование
        if self.log_transform:
            img = np.log10(img + 1e-7)
        
        # 2. Отсечение выбросов
        if self.percentile_range:
            vmin, vmax = np.percentile(img, self.percentile_range)
            img = np.clip(img, vmin, vmax)
        
        # 3. Нормализация
        if self.normalization == 'zscore':
            mean = np.mean(img)
            std = np.std(img) + 1e-8
            img = (img - mean) / std
        elif self.normalization == 'minmax':
            # Масштабируем в [-1, 1] для GAN (Tanh output)
            # Сначала в [0, 1]
            img = (img - vmin) / (vmax - vmin + 1e-8)
            # Потом в [-1, 1]
            img = img * 2.0 - 1.0
        
        return img

def get_common_transform():
    """
    Геометрические аугментации для ОБУЧЕНИЯ.
    Применяются синхронно к SAR и Optical.
    """
    return A.Compose([
        A.Affine(
            scale=(0.9, 1.1),
            translate_percent=(-0.1, 0.1),
            rotate=(-5, 5),
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.9
        ),
        A.OneOf([
            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1.0),
        ], p=0.3),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
    ], additional_targets={
        'optical': 'image',
        'label': 'image'
    })

def get_input_specific(sar_channels=1):
    return A.Compose([
        SARPreprocessing(normalization='minmax'),
        A.Normalize(mean=[0.5] * sar_channels, std=[0.5] * sar_channels),
        ToTensorV2()
    ])

def get_optical_specific():
    return A.Compose([
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2()
    ])

def get_resize_transform(image_size):
    return A.Resize(image_size, image_size)
