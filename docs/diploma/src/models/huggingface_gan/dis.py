import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class PatchDisBranch(nn.Module):
    """5-layer 70×70 spectral-norm PatchGAN branch.

    At 256×256 input:  logits shape = (B, 1, 30, 30)
    At 128×128 input:  logits shape = (B, 1, 14, 14)
    Returns (logits, features) where features is a list of 4 intermediate tensors.
    """
    def __init__(self, in_ch: int, ndf: int = 64):
        super().__init__()

        def sn(ci, co, k, s, p):
            return spectral_norm(nn.Conv2d(ci, co, k, s, p, bias=True))

        self.layers = nn.ModuleList([
            nn.Sequential(sn(in_ch, ndf,     4, 2, 1), nn.LeakyReLU(0.2, inplace=True)),
            nn.Sequential(sn(ndf,   ndf * 2, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True)),
            nn.Sequential(sn(ndf*2, ndf * 4, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True)),
            nn.Sequential(sn(ndf*4, ndf * 8, 4, 1, 1), nn.LeakyReLU(0.2, inplace=True)),
            sn(ndf * 8, 1, 4, 1, 1),
        ])

    def forward(self, x: torch.Tensor):
        features = []
        for layer in self.layers[:-1]:
            x = layer(x)
            features.append(x)
        return self.layers[-1](x), features


class HFGANDiscriminator(nn.Module):
    """Two-scale conditional PatchGAN.

    Concatenates [SAR, OPT] and runs two branches: full resolution and 2× downsampled.
    Returns ((logits_large, logits_small), features) where features is a flat list of
    8 tensors (4 per branch) used by FeatureMatchingLoss.
    """
    def __init__(self, in_ch: int = 4, ndf: int = 64):
        super().__init__()
        self.branch1    = PatchDisBranch(in_ch, ndf)
        self.branch2    = PatchDisBranch(in_ch, ndf)
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)

    def forward(self, sar: torch.Tensor, opt: torch.Tensor):
        x  = torch.cat([sar, opt], dim=1)    # (B, 4, 256, 256)
        x2 = self.downsample(x)              # (B, 4, 128, 128)
        logits1, feats1 = self.branch1(x)
        logits2, feats2 = self.branch2(x2)
        return (logits1, logits2), feats1 + feats2
