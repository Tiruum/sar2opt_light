import torch
import torch.nn as nn


def _default_norm(num_features: int) -> nn.Module:
    """Factory for discriminator normalization layers."""
    return nn.BatchNorm2d(num_features)


class CFRWDPatchDisBranch(nn.Module):
    def __init__(self, in_channels: int = 6, ndf: int = 64, return_features: bool = True):
        super().__init__()
        self.return_features = return_features

        layers = [
            nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
            _default_norm(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
            _default_norm(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1),
            _default_norm(ndf * 8),
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
    """Двухмасштабный PatchGAN-дискриминатор для пар (fake_opt, real_opt)."""

    def __init__(self, in_channels: int = 6, ndf: int = 64, return_features: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.return_features = return_features

        self.large_scale_branch = CFRWDPatchDisBranch(in_channels, ndf, return_features)
        self.small_scale_branch = CFRWDPatchDisBranch(in_channels, ndf, return_features)
        self.downsample = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

    def _build_input(self, fake_opt: torch.Tensor, real_opt: torch.Tensor) -> torch.Tensor:
        if fake_opt.shape != real_opt.shape:
            raise ValueError(
                f"Ожидаются совпадающие тензоры fake_opt и real_opt, получено {fake_opt.shape} vs {real_opt.shape}."
            )
        return torch.cat([fake_opt, real_opt], dim=1)

    def forward(self, fake_opt: torch.Tensor = None, real_opt: torch.Tensor = None):
        if fake_opt is None or real_opt is None:
            raise ValueError("Для дискриминатора требуется fake_opt и real_opt.")

        input_img = self._build_input(fake_opt, real_opt)
        if input_img.shape[1] != self.in_channels:
            raise ValueError(
                f"Ожидается {self.in_channels} каналов после конкатенации, получено {input_img.shape[1]}."
            )

        if self.return_features:
            large_output, large_features = self.large_scale_branch(input_img)
        else:
            large_output = self.large_scale_branch(input_img)

        small_fake = self.downsample(fake_opt)
        small_real = self.downsample(real_opt)
        small_input = self._build_input(small_fake, small_real)

        if self.return_features:
            small_output, small_features = self.small_scale_branch(small_input)
        else:
            small_output = self.small_scale_branch(small_input)

        outputs = (large_output, small_output)

        if self.return_features:
            all_features = large_features + small_features
            return outputs, all_features

        return outputs


if __name__ == "__main__":
    batch_size = 4
    height, width = 256, 256
    fake_optical = torch.randn(batch_size, 3, height, width)
    real_optical = torch.randn(batch_size, 3, height, width)

    discriminator = CFRWDPatchDis(in_channels=6, ndf=64, return_features=True)
    (large_out, small_out), feats = discriminator(fake_opt=fake_optical, real_opt=real_optical)

    print(f"Large scale output shape: {large_out.shape}")
    print(f"Small scale output shape: {small_out.shape}")
    print(f"Number of feature maps extracted: {len(feats)}")