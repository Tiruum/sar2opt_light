"""Discriminators for SARFormer-WB.

  * ``MSPatchGANDis`` — 2-scale conditional PatchGAN on ``[SAR_instance_norm,
    optical_raw]``.  Coarse 70x70 RF, Fine ~46x46 RF, both spectral-norm.
    Layer-0 features are dropped from the FM list (hfgan-18 lesson: layer-0
    encodes SAR edge/speckle and would force G to copy speckle into optical).
    The 22x22 micro branch from hfgan-18 is intentionally dropped here.

The previous ``PhiPhysicsDis`` (a physics discriminator that consumed the
generator's ``s_spk`` channel) was retired in sarformer-wb-3-simple — its
adversarial signal was overwhelmed by the recon loss stack and the generator
never learned to differentiate the path.
"""
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


def _sn(ci: int, co: int, k: int, s: int, p: int) -> nn.Conv2d:
    return spectral_norm(nn.Conv2d(ci, co, kernel_size=k, stride=s, padding=p, bias=True))


class _CoarsePatchBranch(nn.Module):
    """5-layer 70x70 RF spectral-norm PatchGAN.  Logits (B,1,30,30) at 256x256."""
    def __init__(self, in_ch: int, ndf: int = 64):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(_sn(in_ch, ndf,     4, 2, 1), nn.LeakyReLU(0.2, inplace=True)),
            nn.Sequential(_sn(ndf,   ndf * 2, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True)),
            nn.Sequential(_sn(ndf*2, ndf * 4, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True)),
            nn.Sequential(_sn(ndf*4, ndf * 8, 4, 1, 1), nn.LeakyReLU(0.2, inplace=True)),
            _sn(ndf * 8, 1, 4, 1, 1),
        ])

    def forward(self, x: torch.Tensor):
        feats = []
        for layer in self.layers[:-1]:
            x = layer(x)
            feats.append(x)
        return self.layers[-1](x), feats


class _FinePatchBranch(nn.Module):
    """3-layer ~46x46 RF spectral-norm PatchGAN.  Logits (B,1,31,31) at 256x256."""
    def __init__(self, in_ch: int, ndf: int = 64):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(_sn(in_ch, ndf,     4, 2, 1), nn.LeakyReLU(0.2, inplace=True)),
            nn.Sequential(_sn(ndf,   ndf * 2, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True)),
            nn.Sequential(_sn(ndf*2, ndf * 4, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True)),
            _sn(ndf * 4, 1, 4, 1, 1),
        ])

    def forward(self, x: torch.Tensor):
        feats = []
        for layer in self.layers[:-1]:
            x = layer(x)
            feats.append(x)
        return self.layers[-1](x), feats


class MSPatchGANDis(nn.Module):
    """2-scale conditional PatchGAN with asymmetric input normalisation.

    SAR is instance-normed so brightness shifts (speckle baseline) cannot
    drive D; optical is passed through raw so D sees absolute colour.
    Layer-0 features of each branch are dropped from the FM list.
    Returns ``((logits_coarse, logits_fine), features)``.
    """
    def __init__(self, in_ch: int = 4, ndf: int = 64):
        super().__init__()
        self.coarse = _CoarsePatchBranch(in_ch, ndf)
        self.fine = _FinePatchBranch(in_ch, ndf)

    def forward(self, sar: torch.Tensor, opt: torch.Tensor) -> Tuple[Tuple[torch.Tensor, torch.Tensor], List[torch.Tensor]]:
        sar_n = F.instance_norm(sar)
        x = torch.cat([sar_n, opt], dim=1)
        l1, f1 = self.coarse(x)
        l2, f2 = self.fine(x)
        return (l1, l2), f1[1:] + f2[1:]
