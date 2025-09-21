# models/losses.py

import torch
import torch.nn as nn
from omegaconf import OmegaConf
cfg = OmegaConf.load('src/models/pix2pix/config.yaml')

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