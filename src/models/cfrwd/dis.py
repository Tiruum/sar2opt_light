import torch
import torch.nn as nn

class CFRWDPatchDisBranch(nn.Module):
    def __init__(self, in_channels: int = 3, ndf: int = 64, return_features: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.return_features = return_features

        layers = [
            nn.Conv2d(self.in_channels, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(ndf * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(ndf * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(ndf * 8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=1),
        ]

        self.main = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor):
        features = []
        out = x
        for layer in self.main:
            out = layer(out)
            if isinstance(layer, nn.LeakyReLU) and self.return_features:
                features.append(out)

        if self.return_features:
            return out, features
        return out

class CFRWDPatchDis(nn.Module):
    """Двухмасштабный PatchGAN-дискриминатор для условных пар (SAR, optical)."""

    def __init__(self, in_channels: int = 3, ndf: int = 64, return_features: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.return_features = return_features
        self.ndf = ndf

        self.large_scale_branch = CFRWDPatchDisBranch(self.in_channels, self.ndf, self.return_features)
        self.small_scale_branch = CFRWDPatchDisBranch(self.in_channels, self.ndf, self.return_features)
        self.downsample = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor):
        """
        x: конкатенированная пара [B, C_sar + C_opt, H, W]
        """
        if x.ndim != 4:
            raise ValueError(f"Ожидался 4D тензор [B, C, H, W], получено: {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Неверное число каналов: ожидалось {self.in_channels}, получено {x.shape[1]}. "
                f"Проверьте конкатенацию входной пары и cfg.model.dis.in_channels."
            )

        if self.return_features:
            large_output, large_features = self.large_scale_branch(x)
        else:
            large_output = self.large_scale_branch(x)

        x_small = self.downsample(x)

        if self.return_features:
            small_output, small_features = self.small_scale_branch(x_small)
        else:
            small_output = self.small_scale_branch(x_small)

        outputs = (large_output, small_output)

        if self.return_features:
            all_features = large_features + small_features
            return outputs, all_features

        return outputs


if __name__ == "__main__":
    batch_size = 4
    height, width = 256, 256
    sar = torch.randn(batch_size, 1, height, width)
    fake_optical = torch.randn(batch_size, 3, height, width)
    real_optical = torch.randn(batch_size, 3, height, width)
    fake_pair = torch.cat([sar, fake_optical], dim=1)
    real_pair = torch.cat([sar, real_optical], dim=1)

    discriminator = CFRWDPatchDis(in_channels=4, ndf=64, return_features=True)
    with torch.no_grad():
        (real_large, real_small), real_feats = discriminator(real_pair)
        (fake_large, fake_small), fake_feats = discriminator(fake_pair)

    print('Real:')
    print(f"\tLarge scale output shape: {real_large.shape}")
    print(f"\tSmall scale output shape: {real_small.shape}")
    print(f"\tNumber of feature maps extracted: {len(real_feats)}")

    print('Fake:')
    print(f"\tLarge scale output shape: {fake_large.shape}")
    print(f"\tSmall scale output shape: {fake_small.shape}")
    print(f"\tNumber of feature maps extracted: {len(fake_feats)}")