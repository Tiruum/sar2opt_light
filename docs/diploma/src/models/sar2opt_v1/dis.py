"""SAR2OPT-V1 discriminator: re-export of proven HFGANDiscriminator.

The 3-scale (70/46/22 RF) conditional PatchGAN with asymmetric SAR-only
InstanceNorm is the strongest discriminator in this repo. Re-use as-is.
Subband-D is intentionally NOT included here (proven non-load-bearing on
this data scale and added needless complexity to the failed llwt_v3 stack).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.models.huggingface_gan.dis import HFGANDiscriminator


__all__ = ["SAR2OPTDiscriminator"]


class SAR2OPTDiscriminator(nn.Module):
    """Cfg-driven thin wrapper around HFGANDiscriminator.

    Reads ``cfg.model.dis.in_channels`` (SAR ch + 3 optical) and
    ``cfg.model.dis.ndf``. Forward returns ``(logits_list, features_list)``
    where ``logits_list = (l_coarse, l_fine, l_micro)`` -- the Lightning
    module iterates over the tuple for GAN + FM losses.
    """

    def __init__(self, cfg):
        super().__init__()
        dis_cfg = cfg.model.dis
        in_ch = int(getattr(dis_cfg, 'in_channels', 4))
        ndf = int(getattr(dis_cfg, 'ndf', 64))
        self.dis = HFGANDiscriminator(in_ch=in_ch, ndf=ndf)

    def forward(self, sar: torch.Tensor, opt: torch.Tensor):
        return self.dis(sar, opt)
