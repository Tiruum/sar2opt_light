# src/data/transforms.py
import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2

def get_common_transform():
    """
    Геометрические аугментации для ОБУЧЕНИЯ.
    Применяются синхронно к SAR и Optical.
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(
            scale=(0.9, 1.1),
            translate_percent=(-0.1, 0.1),
            rotate=(-5, 5),
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.7
        ),
    ], additional_targets={
        'optical': 'image'
    })

def get_input_specific(sar_channels=1):
    return A.Compose([
        A.Normalize(mean=[0.5] * sar_channels, std=[0.5] * sar_channels, max_pixel_value=255.0),
        ToTensorV2()
    ])

def get_optical_specific():
    return A.Compose([
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0),
        ToTensorV2()
    ])

def get_resize_transform(image_size):
    return A.Resize(image_size, image_size)
