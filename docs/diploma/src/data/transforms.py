# src/data/transforms.py
import numpy as np
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

class LogNormSAR(A.ImageOnlyTransform):
    """
    Applies logarithmic normalization to SAR images (log1p).
    Scales the output back to [0, 255] or original max, so A.Normalize works properly.
    """
    def __init__(self, always_apply=True, p=1.0):
        super(LogNormSAR, self).__init__(always_apply, p)

    def apply(self, img, **params):
        img_float = img.astype(np.float32)
        # Assuming input is in [0, 255]
        img_log = np.log1p(img_float)
        img_log = (img_log / np.log1p(255.0)) * 255.0
        if img.dtype == np.uint8:
            return np.clip(img_log, 0, 255).astype(np.uint8)
        return img_log

def get_input_specific(sar_channels=1, sar_mean=None, sar_std=None, use_lognorm=False):
    mean = sar_mean if sar_mean is not None else [0.5] * sar_channels
    std = sar_std if sar_std is not None else [0.5] * sar_channels
    
    transforms = []
    if use_lognorm:
        transforms.append(LogNormSAR(always_apply=True))
        
    transforms.extend([
        A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
        ToTensorV2()
    ])
    
    return A.Compose(transforms)

def get_optical_specific(opt_mean=None, opt_std=None):
    mean = opt_mean if opt_mean is not None else [0.5, 0.5, 0.5]
    std = opt_std if opt_std is not None else [0.5, 0.5, 0.5]
    return A.Compose([
        # A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
        ToTensorV2()
    ])

def get_resize_transform(image_size, interpolation=cv2.INTER_LINEAR):
    """Bilinear by default — appropriate for the optical (continuous-tone) target."""
    return A.Resize(image_size, image_size, interpolation=interpolation)


def get_sar_resize_transform(image_size):
    """Nearest-neighbour interpolation preserves the discrete speckle structure
    of SAR amplitude.  Bilinear blurs speckle edges and removes high-frequency
    cues the wavelet branch and the discriminator depend on.
    """
    return A.Resize(image_size, image_size, interpolation=cv2.INTER_NEAREST)
