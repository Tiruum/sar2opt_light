import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.attention import SelfAttention, CBAM

class UNetResidualBlock(nn.Module):
    """Residual block with optional CBAM and Self-Attention."""
    def __init__(self, channels, use_cbam=False, use_self_att=False):
        super(UNetResidualBlock, self).__init__()
        self.use_cbam = use_cbam
        self.use_sa = use_self_att
        layers = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, padding=0),
            nn.InstanceNorm2d(channels),
            nn.ReLU(True)
        ]
        if use_self_att:
            layers.append(SelfAttention(channels))
        layers += [
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, padding=0),
            nn.InstanceNorm2d(channels)
        ]
        self.block = nn.Sequential(*layers)
        if use_cbam:
            self.cbam = CBAM(channels)

    def forward(self, x):
        out = x + self.block(x)
        if self.use_cbam:
            out = self.cbam(out)
        return out

class UNetGenerator(nn.Module):
    """U-Net generator with skip connections, spectral norm and adaptive attention."""
    def __init__(self, input_nc=4, output_nc=3, ngf=64, n_blocks=8):
        super(UNetGenerator, self).__init__()

        # Initial block
        self.initial = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.utils.spectral_norm(nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0)),
            nn.InstanceNorm2d(ngf),
            nn.LeakyReLU(0.2, True)
        )

        # Encoder
        self.enc1 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(ngf, ngf*2, kernel_size=3, stride=2, padding=1)),
            nn.InstanceNorm2d(ngf*2),
            nn.LeakyReLU(0.2, True)
        )
        self.enc2 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(ngf*2, ngf*4, kernel_size=3, stride=2, padding=1)),
            nn.InstanceNorm2d(ngf*4),
            nn.LeakyReLU(0.2, True)
        )
        self.enc3 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(ngf*4, ngf*8, kernel_size=3, stride=2, padding=1)),
            nn.InstanceNorm2d(ngf*8),
            nn.LeakyReLU(0.2, True)
        )

        # Residual blocks
        self.resblocks = nn.ModuleList([
            UNetResidualBlock(ngf*8, use_cbam=True, use_self_att=(i >= n_blocks - 3))
            for i in range(n_blocks)
        ])

        self.dec3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.utils.spectral_norm(nn.Conv2d(ngf*8, ngf*4, kernel_size=3, padding=1)),
            nn.GroupNorm(16, ngf*4),
            nn.LeakyReLU(0.2, True)
        )

        # Decoder part 1 — сохраняем двойную свёртку для dec2
        self.dec2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.utils.spectral_norm(nn.Conv2d(ngf*4, ngf*4, kernel_size=3, padding=1)),
            nn.GroupNorm(8, ngf*4),
            nn.ReLU(True),
            nn.utils.spectral_norm(nn.Conv2d(ngf*4, ngf*2, kernel_size=3, padding=1)),
            nn.GroupNorm(8, ngf*2),
            nn.ReLU(True)
        )

        # Decoder part 2 — упрощаем dec1, меньше capacity
        self.dec1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.utils.spectral_norm(nn.Conv2d(ngf*2, ngf, kernel_size=3, padding=1)),
            nn.GroupNorm(8, ngf),
            nn.ReLU(True)
        )

        # Final conv
        self.final = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.utils.spectral_norm(nn.Conv2d(ngf, ngf, kernel_size=7, padding=0)),  # Увеличиваем каналы
            nn.InstanceNorm2d(ngf),
            nn.LeakyReLU(0.2, True),
            nn.utils.spectral_norm(nn.Conv2d(ngf, ngf//2, kernel_size=3, padding=1)),  # Добавляем свертку 3х3
            nn.InstanceNorm2d(ngf//2),
            nn.LeakyReLU(0.2, True),
            nn.ReflectionPad2d(3),
            nn.utils.spectral_norm(nn.Conv2d(ngf//2, output_nc, kernel_size=7, padding=0)),
            nn.Tanh()
        )

        # Инициализация весов
        for m in self.final.modules():
            if isinstance(m, nn.Conv2d) and m.out_channels == output_nc:
                nn.init.normal_(m.weight, mean=0.0, std=0.005)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # В метод __init__ класса UNetGenerator, после определения self.conv_after_cat2:
        self.conv_after_cat3 = nn.Sequential(
            nn.Conv2d(ngf*8, ngf*4, kernel_size=3, padding=1),
            nn.GroupNorm(16, ngf*4),  # Используем 16 групп для ngf*4 каналов
            nn.LeakyReLU(0.2, True),  # Для согласованности с dec3 используем LeakyReLU
            nn.Conv2d(ngf*4, ngf*4, kernel_size=3, padding=1),
            nn.GroupNorm(16, ngf*4),
            nn.LeakyReLU(0.2, True)
        )

        # Reduce conv_after_cat1 output channels to ngf to avoid overcapacity
        self.conv_after_cat2 = nn.Sequential(
            nn.Conv2d(ngf*4, ngf*2, kernel_size=3, padding=1),
            nn.GroupNorm(8, ngf*2),
            nn.ReLU(True),
            nn.Conv2d(ngf*2, ngf*2, kernel_size=3, padding=1),
            nn.GroupNorm(8, ngf*2),
            nn.ReLU(True)
        )

        self.conv_after_cat1 = nn.Sequential(
            nn.Conv2d(ngf*2, ngf, kernel_size=3, padding=1),
            nn.GroupNorm(8, ngf),
            nn.ReLU(True),
            nn.Conv2d(ngf, ngf, kernel_size=3, padding=1),
            nn.GroupNorm(8, ngf),
            nn.ReLU(True)
        )

    def forward(self, x):
        # Encoder path
        x1 = self.initial(x)
        x2 = self.enc1(x1)
        x3 = self.enc2(x2)
        x4 = self.enc3(x3)  # Используем enc3 для получения x4
        
        # Residual blocks
        for block in self.resblocks:
            x4 = block(x4)
        
        # Decoder path с полной конкатенацией
        y3 = self.dec3(x4)
        y3 = torch.cat([y3, x3], dim=1)  # Конкатенация с x3
        y3 = self.conv_after_cat3(y3)     # Обработка конкатенированных признаков
        
        y2 = self.dec2(y3)                # dec2 теперь принимает y3 вместо x4
        y2 = torch.cat([y2, x2], dim=1)
        y2 = self.conv_after_cat2(y2)
        
        y1 = self.dec1(y2)
        y1 = torch.cat([y1, x1], dim=1)
        y1 = self.conv_after_cat1(y1)
        
        with torch.amp.autocast('cuda', enabled=False):
            out = self.final(y1)
        return out

if __name__ == "__main__":
    batch, c, h, w = 1, 1, 600, 600
    inp = torch.randn(batch, c, h, w)
    net = UNetGenerator(input_nc=1, output_nc=3, ngf=64, n_blocks=8)
    out = net(inp)
    print(f"Input: {inp.shape} -> Output: {out.shape}")
