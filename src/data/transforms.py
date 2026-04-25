# src/data/transforms.py
import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2

DATASET_NORM_STATS = {
    'sen12_full': {
        'sar_mean': [0.5], 'sar_std': [0.5],
        'opt_mean': [0.5, 0.5, 0.5], 'opt_std': [0.5, 0.5, 0.5],
    },
    'qxslab': {
        'sar_mean': [0.5], 'sar_std': [0.5],
        'opt_mean': [0.5, 0.5, 0.5], 'opt_std': [0.5, 0.5, 0.5],
    },
}

def get_common_transform():
    """
    Geometric augmentations for training.
    Applied synchronously to SAR and optical images.

    SAR geometry note: Sentinel-1 GRD is geocoded North-up. Horizontal flip
    mirrors the scene left-right (valid: equivalent to descending vs ascending
    geometry). VerticalFlip and RandomRotate90 removed — they reverse azimuth
    direction or change look-direction orientation, producing shadow/layover
    patterns inconsistent with the paired optical image.
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
    ], additional_targets={
        'optical': 'image'
    })

def get_input_specific(sar_channels=1, sar_mean=None, sar_std=None):
    mean = sar_mean if sar_mean is not None else [0.5] * sar_channels
    std = sar_std if sar_std is not None else [0.5] * sar_channels
    return A.Compose([
        A.MultiplicativeNoise(multiplier=(0.8, 1.2), per_channel=False, p=0.4),
        A.GaussNoise(std_range=(0.009, 0.015), p=0.2),
        A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
        ToTensorV2()
    ])

def get_optical_specific(opt_mean=None, opt_std=None):
    mean = opt_mean if opt_mean is not None else [0.5, 0.5, 0.5]
    std = opt_std if opt_std is not None else [0.5, 0.5, 0.5]
    return A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=0, p=0.5),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
        ToTensorV2()
    ])

def get_resize_transform(image_size):
    return A.Resize(image_size, image_size)
