# models/losses.py

import torch
import torch.nn as nn
import numpy as np
from omegaconf import OmegaConf
cfg = OmegaConf.load('src/models/cfrwd/config.yaml')

class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True):
        super(GANLoss, self).__init__()
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCEWithLogitsLoss()

    def get_target_tensor(self, prediction, target_is_real, real_label_smooth=1.0, fake_label_smooth=0.0):
        if target_is_real:
            return torch.ones(prediction.shape)
        else:
            return torch.zeros(prediction.shape)

    def forward(self, prediction, target_is_real, real_label_smooth=1.0, fake_label_smooth=0.0):
        target_tensor = self.get_target_tensor(
            prediction, 
            target_is_real
        ).to(cfg.system.device)
        return self.loss(prediction, target_tensor)

class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()
        self.loss = nn.L1Loss()

    def forward(self, pred, target):
        return self.loss(pred, target)
    
class FeatureMatchingLoss(nn.Module):
    """
    High-dimensional Feature Matching Loss (LFM) as described in CFRWD-GAN paper.
    Assumes feat_real and feat_fake are lists of feature maps from discriminator layers.
    LFM = sum_{i=0}^M (1 / (C_i * W_i * H_i)) * || D^i(Y) - D^i(G(X)) ||_1
    Which simplifies to sum mean(|D^i(Y) - D^i(G(X))|) over layers i.
    """
    def __init__(self):  # M+1, adjust based on discriminator layers
        super(FeatureMatchingLoss, self).__init__()
        self.feat_layers = 8
        self.l1_loss = nn.L1Loss(reduction='mean')  # Mean L1 per layer

    def forward(self, feat_real, feat_fake):
        """
        feat_real: list of tensors [batch, C_i, W_i, H_i] from D on real
        feat_fake: list of tensors [batch, C_i, W_i, H_i] from D on fake
        """
        if len(feat_real) != len(feat_fake):
            raise ValueError(f"Длина feat_real ({len(feat_real)}) не совпадает с длиной feat_fake ({len(feat_fake)})")

        total_loss = 0.0
        for i in range(len(feat_real)):
            # Normalization: 1/(C W H) * sum |diff| == mean(|diff|) since L1 mean is sum/N
            diff_loss = self.l1_loss(feat_real[i], feat_fake[i])
            total_loss += diff_loss
        return total_loss / len(feat_real)  # Average over layers, or sum if preferred