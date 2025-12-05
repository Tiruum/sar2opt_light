import torch
import torch.nn as nn
import math
from src.utils.logger import Logger

logger = Logger(name="CFRWD", cfg_path='src/models/cfrwd/config.yaml')

class ResBlock(nn.Module):
    """ Modern GAN Residual Block (No activation at output, LeakyReLU inside) """
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(channels, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(channels, affine=True)
        )
        
    def forward(self, x):
        return x + self.block(x)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2):
        logger.debug('Conv Block INIT')
        super(ConvBlock, self).__init__()

        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1), # Вообще по статье ReflectionPad используется один раз в начале, а затем, видимо, в Conv2d padding=1
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

    def forward(self, x):
        return self.conv_block(x)
    
class TConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2):
        super(TConvBlock, self).__init__()
        logger.debug('TConv Block INIT')
        self.t_conv_block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.t_conv_block(x)

class FinalTConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(FinalTConvBlock, self).__init__()
        logger.debug('Final TConv Block INIT')
        self.final_t_conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size, bias=False),
            nn.Tanh()
        )
        
    def forward(self, x):
        return self.final_t_conv_block(x)

class CFRBlock(nn.Module):
    def __init__(self, channels):
        super(CFRBlock, self).__init__()
        logger.debug('CFR Block INIT')

        # a1 = B C W H          p1 = B C/4 W H
        # a2 = B C W/2 H/2      p2 = B C/2 W/2 H/2

        # b1 = Conv(cat(p1, U(p2)))
        # b2 = Conv(cat(D(p1), p2))
        # b3 = Conv(cat(D(D(p1)), D(p2)))

        self.n11 = nn.Sequential(
            *[ResBlock(channels) for _ in range(3)],
            nn.Conv2d(channels, channels // 4, kernel_size=1, stride=1)
        )

        self.n12 = nn.Sequential(
            *[ResBlock(channels) for _ in range(3)],
            nn.Conv2d(channels, channels // 2, kernel_size=1, stride=1)
        )

        self.fuse1_to2_1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d((channels // 4 + channels // 2), channels // 8, kernel_size=3, stride=1)
        )
        self.fuse1_to2_2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d((channels // 4 + channels // 2), channels // 4, kernel_size=3, stride=1)
        )
        self.fuse1_to2_3 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d((channels // 4 + channels // 2), channels // 2, kernel_size=3, stride=1)
        )

        # b1 = B C/8 W H        q1 = B C/8 W H
        # b2 = B C/4 W/2 H/2    q2 = B C/4 W/2 H/2
        # b3 = B C/2 W/4 H/4    q3 = B C/2 W/4 H/4

        # c1 = Conv(cat(q1, U(q2), U(U(q3))))
        # c2 = Conv(cat(D(q1), q2, U(q3)))
        # c3 = Conv(cat(D(D(q1)), D(q2), q3))
        # c4 = Conv(cat(D(D(D(q1))), D(D(q2)), D(q3)))

        self.n21 = nn.Sequential(
            *[ResBlock(channels // 8) for _ in range(3)],
        )
        self.n22 = nn.Sequential(
            *[ResBlock(channels // 4) for _ in range(3)],
        )
        self.n23 = nn.Sequential(
            *[ResBlock(channels // 2) for _ in range(3)],
        )

        self.fuse2_to3_1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d((channels // 8 + channels // 4 + channels // 2), channels // 16, kernel_size=3, stride=1)
        )
        self.fuse2_to3_2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d((channels // 8 + channels // 4 + channels // 2), channels // 8, kernel_size=3, stride=1)
        )
        self.fuse2_to3_3 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d((channels // 8 + channels // 4 + channels // 2), channels // 4, kernel_size=3, stride=1)
        )
        self.fuse2_to3_4 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d((channels // 8 + channels // 4 + channels // 2), channels // 2, kernel_size=3, stride=1)
        )

        # c1 = B C/16 W H       k1 = B C/16 W H
        # c2 = B C/8 W/2 H/2    k2 = B C/8 W/2 H/2
        # c3 = B C/4 W/4 H/4    k3 = B C/4 W/4 H/4
        # c4 = B C/2 W/8 H/8    k4 = B C/2 W/8 H/8

        # d = Conv(cat(D(k1), k2, U(k3), U(U(k4))))

        self.n31 = nn.Sequential(
            *[ResBlock(channels // 16) for _ in range(3)],
        )
        self.n32 = nn.Sequential(
            *[ResBlock(channels // 8) for _ in range(3)],
        )
        self.n33 = nn.Sequential(
            *[ResBlock(channels // 4) for _ in range(3)],
        )
        self.n34 = nn.Sequential(
            *[ResBlock(channels // 2) for _ in range(3)],
        )

        self.fuse3_to4 = nn.Conv2d((channels // 16 + channels // 8 + channels // 4 + channels // 2), channels // 4, kernel_size=1, stride=1)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.down = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        logger.debug(f'Input shape: {x.shape}', once=True)

        a1 = x
        a2 = self.down(a1)

        # Stage 1
        p1 = self.n11(a1)
        p2 = self.n12(a2)

        logger.debug('Stage 1', once=True)
        logger.debug(f'a1 shape: {a1.shape}, p1 shape: {p1.shape}', once=True)
        logger.debug(f'a2 shape: {a2.shape}, p2 shape: {p2.shape}', once=True)

        # Cross-fusion 1
        b1 = self.fuse1_to2_1(torch.cat([p1, self.up(p2)], dim=1))
        b2 = self.fuse1_to2_2(torch.cat([self.down(p1), p2], dim=1))
        b3 = self.fuse1_to2_3(torch.cat([self.down(self.down(p1)), self.down(p2)], dim=1))

        # Stage 2
        q1 = self.n21(b1)
        q2 = self.n22(b2)
        q3 = self.n23(b3)

        logger.debug('Stage 2', once=True)
        logger.debug(f'b1 shape: {b1.shape}, q1 shape: {q1.shape}', once=True)
        logger.debug(f'b2 shape: {b2.shape}, q2 shape: {q2.shape}', once=True)
        logger.debug(f'b3 shape: {b3.shape}, q3 shape: {q3.shape}', once=True)

        # Cross-fusion 2
        c1 = self.fuse2_to3_1(torch.cat([q1, self.up(q2), self.up(self.up(q3))], dim=1))
        c2 = self.fuse2_to3_2(torch.cat([self.down(q1), q2, self.up(q3)], dim=1))
        c3 = self.fuse2_to3_3(torch.cat([self.down(self.down(q1)), self.down(q2), q3], dim=1))
        c4 = self.fuse2_to3_4(torch.cat([self.down(self.down(self.down(q1))), self.down(self.down(q2)), self.down(q3)], dim=1))

        # Stage 3
        k1 = self.n31(c1)
        k2 = self.n32(c2)
        k3 = self.n33(c3)
        k4 = self.n34(c4)

        logger.debug('Stage 3', once=True)
        logger.debug(f'c1 shape: {c1.shape}, k1 shape: {k1.shape}', once=True)
        logger.debug(f'c2 shape: {c2.shape}, k2 shape: {k2.shape}', once=True)
        logger.debug(f'c3 shape: {c3.shape}, k3 shape: {k3.shape}', once=True)
        logger.debug(f'c4 shape: {c4.shape}, k4 shape: {k4.shape}', once=True)

        # Final fusion
        d = self.fuse3_to4(torch.cat([self.down(k1), k2, self.up(k3), self.up(self.up(k4))], dim=1))
        logger.debug(f'Output shape: {d.shape}', once=True)

        return d
    
class CFRBranch(nn.Module):
    def __init__(self, in_channels=1):
        super(CFRBranch, self).__init__()
        logger.debug('CFR BRANCH INIT')
        logger.debug('CFR BRANCH INIT DEBUG')

        self.encoder = nn.Sequential(
            ConvBlock(in_channels, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
            ConvBlock(256, 512),
        )

        logger.debug(f"Conv:\t\t{' -> '.join(map(str, [in_channels, 64, 128, 256, 512]))}")

        self.CFR = CFRBlock(512)
        logger.debug(f'CFRBlock:\t{512} -> {512 // 4}')

        dec_in_ch = 512 // 4

        self.decoder = nn.Sequential(
            TConvBlock(dec_in_ch, 256),
            TConvBlock(256, 512),
            TConvBlock(512, 256),
            TConvBlock(256, 128),
            TConvBlock(128, 64),
            # TConvBlock(64, 3),
            FinalTConvBlock(64, 3),
        )

    def forward(self, x):
        logger.debug('CFR BRANCH FORWARD DEBUG', once=True)
        logger.debug(f'CFRBranch Input shape: {x.shape}', once=True)
        x = self.encoder(x)
        logger.debug(f'After Conv shape: {x.shape}', once=True)
        x = self.CFR(x)
        logger.debug(f'After CFR shape: {x.shape}', once=True)
        x = self.decoder(x)
        logger.debug(f'CFRBranch Output shape: {x.shape}', once=True)
        return x

class HaarDown(nn.Module):
    """
    Прямое вейвлет-преобразование (DWT, Haar)
    """
    def __init__(self, in_channels=1, normalize=True):
        super(HaarDown, self).__init__()
        self.scale = 0.5 if not normalize else 1 / math.sqrt(2)
        self.register_buffer('haar_weights', torch.tensor([
             [ 1.0,  1.0,  1.0,  1.0], # LL
             [-1.0,  1.0, -1.0,  1.0], # LH
             [-1.0, -1.0,  1.0,  1.0], # HL
             [ 1.0, -1.0, -1.0,  1.0]  # HH
        ], dtype=torch.float32) * self.scale)
        
    def forward(self, x):
        B, C, H, W = x.shape
        x_reshaped = torch.nn.functional.pixel_unshuffle(x, 2)
        x_reshaped = x_reshaped.view(B, C, 4, H // 2, W // 2)
        weights = self.haar_weights.to(x.device)
        out = torch.einsum('bcihw, oi -> bcohw', x_reshaped, weights)
        return out[:, :, 0], out[:, :, 1], out[:, :, 2], out[:, :, 3]
    
class HaarUp(nn.Module):
    def __init__(self, in_channels=1):
        super(HaarUp, self).__init__()
        # Обратная матрица Haar (transposed)
        self.register_buffer('inv_weights', torch.tensor([
             [ 1.0, -1.0, -1.0,  1.0],
             [ 1.0,  1.0, -1.0, -1.0],
             [ 1.0, -1.0,  1.0, -1.0],
             [ 1.0,  1.0,  1.0,  1.0]
        ], dtype=torch.float32))
        
    def forward(self, LL, LH, HL, HH):
        stack = torch.stack([LL, LH, HL, HH], dim=2)
        weights = self.inv_weights.to(LL.device)
        out_pixels = torch.einsum('bcihw, oi -> bcohw', stack, weights)
        B, C, _, H, W = out_pixels.shape
        out_pixels = out_pixels.view(B, C * 4, H, W)
        return torch.nn.functional.pixel_shuffle(out_pixels, 2)

class DWTBlock(nn.Module):
    def __init__(self, in_channels=1):
        super(DWTBlock, self).__init__()
        self.dwt = HaarDown(normalize=True, in_channels=in_channels)

    def forward(self, x):
        ll1, lh1, hl1, hh1 = self.dwt(x)
        ll2, lh2, hl2, hh2 = self.dwt(ll1)
        g1 = ll2
        g2 = torch.cat([lh2, hl2, hh2], dim=1)
        g3 = torch.cat([lh1, hl1, hh1], dim=1)
        return g1, g2, g3

class HFCFPreprocess(nn.Module):
    """
    High-frequency component preprocessing block.
    According to Figure 5 in CFRWD paper: Conv -> Norm -> ReLU -> MaxPool
    
    Note: InstanceNorm2d is used (rather than BatchNorm) because it performs better
    for image-to-image translation tasks where each sample should be processed 
    independently, particularly important for SAR-to-optical translation with 
    varying speckle noise patterns.
    
    This block prepares high-frequency wavelet components for HFCF processing.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(HFCFPreprocess, self).__init__()
        logger.debug('HFCF Preprocess INIT')
        self.block = nn.Sequential(
            nn.ReflectionPad2d(padding),
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
    
    def forward(self, x):
        return self.block(x)
    
class YellowBlock(nn.Module):
    """
    Block с уменьшением размерности (Stride=2).
    Структура соответствует схеме (3 свертки), но оптимизирована для GAN.
    """
    def __init__(self, in_channels, out_channels, negative_slope=0.2):
        super(YellowBlock, self).__init__()
        
        self.f_block = nn.Sequential(
            # Conv 1
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(in_channels, affine=True),
            nn.LeakyReLU(negative_slope, inplace=True),

            # Conv 2
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(in_channels, affine=True),
            nn.LeakyReLU(negative_slope, inplace=True),

            # Conv 3 (Downsample)
            # При stride=2 и kernel=3 нужен padding=1, чтобы выход был ровно H/2
            # ReflectionPad2d(1) работает корректно.
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=0, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True)
            # ВАЖНО: Нет активации в конце ветви
        )

        self.skip = nn.Sequential(
            # 1x1 свертка для изменения каналов и размера
            # bias=False, так как дальше Norm
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, padding=0, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True)
        )

    def forward(self, x):
        # Сложение без финальной активации
        return self.f_block(x) + self.skip(x)


class BlueBlock(nn.Module):
    """
    Block без изменения размерности.
    """
    def __init__(self, channels, negative_slope=0.2):
        super(BlueBlock, self).__init__()

        self.f_block = nn.Sequential(
            # Conv 1
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(channels, affine=True),
            nn.LeakyReLU(negative_slope, inplace=True),

            # Conv 2
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(channels, affine=True),
            nn.LeakyReLU(negative_slope, inplace=True),

            # Conv 3
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(channels, affine=True)
            # Без активации
        )

    def forward(self, x):
        return self.f_block(x) + x


class RedBlock(nn.Module):
    """
    Basic Block (ResNet18 style) для GAN.
    """
    def __init__(self, channels, negative_slope=0.2):
        super(RedBlock, self).__init__()

        self.f_block = nn.Sequential(
            # Conv 1
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(channels, affine=True),
            nn.LeakyReLU(negative_slope, inplace=True),

            # Conv 2
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(channels, affine=True)
            # Без активации
        )

    def forward(self, x):
        return self.f_block(x) + x

    
class UpperBranch(nn.Module):
    """
    Upper branch for HFCF structure using ResNet101-style blocks (Bottleneck).
    Processes high-frequency components from level 2 (LH2, HL2, HH2).
    
    Architecture from Figure 5: Yellow -> Blue -> Blue -> Yellow -> Blue -> Blue
    - Yellow blocks (Bottleneck): 3 convolutions with channel expansion and downsample
    - Blue blocks (Bottleneck): 3 convolutions maintaining channels
    
    Note: The blocks process features while changing channel dimensions but
    potentially changing spatial resolution. The final upsampling happens later.
    """
    def __init__(self, channels):
        super(UpperBranch, self).__init__()
        logger.debug('UpperBranch INIT')
        logger.debug(f'UpperBranch input channels: {channels}')
        
        # According to ResNet101 architecture (used in CFRWD paper):
        # Bottleneck blocks expand channels and may downsample
        # When channels=32 (typical): C -> 2C -> 4C progression
        self.yellow_block1 = YellowBlock(channels, channels*2)      # C -> 2C, spatial /2
        self.blue_block1 = BlueBlock(channels*2)                    # 2C, maintain
        self.blue_block2 = BlueBlock(channels*2)                    # 2C, maintain
        self.yellow_block2 = YellowBlock(channels*2, channels*4)    # 2C -> 4C, spatial /2
        self.blue_block3 = BlueBlock(channels*4)                    # 4C, maintain
        self.blue_block4 = BlueBlock(channels*4)                    # 4C, maintain
        
        logger.debug(f'UpperBranch output channels: {channels*4}')

    def forward(self, x):
        x1 = self.yellow_block1(x)  # B x 32 x H x W -> B x 64 x H/2 x W/2
        x2 = self.blue_block1(x1)   # B x 64 x H/2 x W/2 (maintain)
        x3 = self.blue_block2(x2)   # B x 64 x H/2 x W/2 (maintain)
        x4 = self.yellow_block2(x3) # B x 64 x H/2 x W/2 -> B x 128 x H/4 x W/4
        x5 = self.blue_block3(x4)   # B x 128 x H/4 x W/4 (maintain)
        out = self.blue_block4(x5)  # B x 128 x H/4 x W/4 (maintain)
        return out
    
class LowerBranch(nn.Module):
    """
    Lower branch for HFCF structure using ResNet18-style blocks (BasicBlock).
    Processes high-frequency components from level 1 (LH1, HL1, HH1).
    
    Architecture from Figure 5: Red -> Red
    - Red blocks (BasicBlock): 2 convolutions maintaining channels and resolution
    
    This branch preserves spatial information from higher resolution components.
    """
    def __init__(self, channels):
        super(LowerBranch, self).__init__()
        logger.debug('LowerBranch INIT')
        logger.debug(f'LowerBranch channels: {channels}')
        
        # BasicBlock from ResNet18 - maintains channels and resolution
        self.red_block1 = RedBlock(channels)  # Maintain channels
        self.red_block2 = RedBlock(channels)  # Maintain channels

    def forward(self, x):
        x1 = self.red_block1(x)   # B x 32 x H x W (maintain)
        out = self.red_block2(x1) # B x 32 x H x W (maintain)
        return out
    
class HFCFUpconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(HFCFUpconvBlock, self).__init__()
        logger.debug('HFCF Upconv INIT')
        logger.debug(f'HFCF Upconv: in_channels={in_channels}, out_channels={out_channels}')
        
        # Input: B x 160 x 8 x 8 (W/32 x H/32)
        # Need 5 upsampling steps (each ×2) to reach W x H (256 x 256)
        # Spatial dimensions: 8×8 → 16×16 → 32×32 → 64×64 → 128×128 → 256×256
        
        self.upconv_block = nn.Sequential(
            # First process features
            # B x 160 x 8 x 8 -> B x 128 x 8 x 8
            ConvBlock(in_channels, 128, stride=1),
            
            # Upsample 1: B x 128 x 8 x 8 -> B x 128 x 16 x 16
            TConvBlock(128, 128),
            
            # Upsample 2: B x 128 x 16 x 16 -> B x 64 x 32 x 32
            TConvBlock(128, 64),
            
            # Upsample 3: B x 64 x 32 x 32 -> B x 32 x 64 x 64
            TConvBlock(64, 32),
            
            # Upsample 4: B x 32 x 64 x 64 -> B x 16 x 128 x 128
            TConvBlock(32, 16),
            
            # Upsample 5: B x 16 x 128 x 128 -> B x 8 x 256 x 256
            TConvBlock(16, 8),
            
            # Final convolution to output 3 channels (RGB)
            # B x 8 x 256 x 256 -> B x 3 x 256 x 256
            FinalTConvBlock(8, 3)
        )
    
    def forward(self, x):
        return self.upconv_block(x)
    
class HFCFBranch(nn.Module):
    """
    High-Frequency Coding and Filtering (HFCF) Branch for CFRWD-GAN.
    
    This branch implements the wavelet decomposition pathway described in the CFRWD paper.
    It processes SAR images through discrete wavelet transform (DWT) to separate frequency
    components, then applies specialized filtering to reduce speckle noise while preserving
    high-frequency details.
    
    Architecture:
    1. DWT: Decomposes input into 7 wavelet components (2-level Haar decomposition)
       - g1 (LL2): Low frequency approximation
       - g2 (LH2, HL2, HH2): High frequency details from level 2
       - g3 (LH1, HL1, HH1): High frequency details from level 1
    
    2. HFCF Structure (as per Figure 5 in paper):
       - Upper Branch: Processes g2 with ResNet101-style blocks (Yellow + Blue)
       - Lower Branch: Processes g3 with ResNet18-style blocks (Red)
    
    3. Fusion & Reconstruction: Concatenates branch outputs and upsamples to original resolution
    
    Args:
        in_channels (int): Number of input channels (typically 1 for SAR images)
    
    Input shape: (B, in_channels, H, W)
    Output shape: (B, 3, H, W) - RGB optical image
    """
    def __init__(self, in_channels=1):
        super(HFCFBranch, self).__init__()
        logger.debug('HFCF BRANCH INIT')
        self.dwt = DWTBlock(in_channels=in_channels)

        # Preprocessing for g2 (high-frequency components from level 2)
        # g2 has 3 channels (LH2, HL2, HH2) at resolution W/4 x H/4 (64x64 for 256x256 input)
        self.HFCF_g2_prep = HFCFPreprocess(in_channels=in_channels*3, out_channels=32)
        
        # Preprocessing for g3 (high-frequency components from level 1)
        # g3 has 3 channels (LH1, HL1, HH1) at resolution W/2 x H/2 (128x128 for 256x256 input)
        self.HFCF_g3_prep = HFCFPreprocess(in_channels=in_channels*3, out_channels=32)

        p2_channels = 32
        p3_channels = 32

        # Upper branch processes g2 with ResNet101-style blocks
        self.upper_branch = UpperBranch(channels=p2_channels)
        
        # Lower branch processes g3 with ResNet18-style blocks
        self.lower_branch = LowerBranch(channels=p3_channels)

        # Calculate output channels after branches
        # Upper branch: 32 -> 64 -> 128 (final output is 128)
        # Lower branch: 32 -> 32 (final output is 32)
        upper_out_channels = p2_channels * 4  # 32 * 4 = 128
        lower_out_channels = p3_channels      # 32
        
        # After processing (dimensions shown are post-preprocessing):
        # - upper_out: g2 starts at 64x64 (from DWT) -> prep to 32x32 -> yellow1 to 16x16 -> yellow2 to 8x8
        # - lower_out: g3 starts at 128x128 (from DWT) -> prep to 64x64 -> red blocks maintain 64x64
        # We need to align them to the same spatial resolution (8x8) before concatenation
        
        # Downsample lower branch output to match upper branch resolution
        # Efficient downsampling from 64x64 to 8x8 (÷8) using adaptive pooling
        self.align_lower = nn.AdaptiveAvgPool2d((8, 8))
        
        total_in_channels = upper_out_channels + lower_out_channels  # 160
        
        self.hfcf_upconv = HFCFUpconvBlock(in_channels=total_in_channels, out_channels=3)

    def forward(self, x):
        # Apply 2-level wavelet decomposition
        # Input: B x 1 x 256 x 256
        # g1: B x 1 x 64 x 64 (LL2 - low frequency)
        # g2: B x 3 x 64 x 64 (LH2, HL2, HH2 - high frequency level 2)
        # g3: B x 3 x 128 x 128 (LH1, HL1, HH1 - high frequency level 1)
        g1, g2, g3 = self.dwt(x)

        logger.debug(f'DWT shapes - g1: {g1.shape}, g2: {g2.shape}, g3: {g3.shape}', once=True)

        # Preprocess g2 (high-frequency level 2) for upper branch
        # g2: B x 3 x 64 x 64 -> B x 32 x 32 x 32 (after conv and maxpool ÷2)
        hfcf_g2 = self.HFCF_g2_prep(g2)
        logger.debug(f'After g2 preprocess: {hfcf_g2.shape}', once=True)
        
        # Preprocess g3 (high-frequency level 1) for lower branch
        # g3: B x 3 x 128 x 128 -> B x 32 x 64 x 64 (after conv and maxpool ÷2)
        hfcf_g3 = self.HFCF_g3_prep(g3)
        logger.debug(f'After g3 preprocess: {hfcf_g3.shape}', once=True)

        # Process through upper branch (ResNet101-style with downsampling)
        # hfcf_g2: B x 32 x 32 x 32 
        #   -> Yellow1: B x 64 x 16 x 16 (stride=2)
        #   -> Yellow2: B x 128 x 8 x 8 (stride=2)
        # Final: B x 128 x 8 x 8
        upper_out = self.upper_branch(hfcf_g2)
        logger.debug(f'Upper branch output: {upper_out.shape}', once=True)
        
        # Process through lower branch (ResNet18-style, no downsampling)
        # hfcf_g3: B x 32 x 64 x 64 -> B x 32 x 64 x 64 (maintain resolution)
        lower_out = self.lower_branch(hfcf_g3)
        logger.debug(f'Lower branch output: {lower_out.shape}', once=True)
        
        # Align spatial dimensions: downsample lower_out to match upper_out
        # B x 32 x 64 x 64 -> B x 32 x 8 x 8 (AdaptiveAvgPool to target size)
        lower_out_aligned = self.align_lower(lower_out)
        logger.debug(f'Lower branch output aligned: {lower_out_aligned.shape}', once=True)

        # Concatenate outputs from both branches at the same spatial resolution
        # B x 128 x 8 x 8 + B x 32 x 8 x 8 -> B x 160 x 8 x 8
        out = torch.cat([upper_out, lower_out_aligned], dim=1)
        logger.debug(f'Concatenated output: {out.shape}', once=True)

        # Apply upconvolution to reconstruct image
        # B x 160 x 8 x 8 -> B x 3 x 256 x 256 (5 upsampling steps: ×2^5 = ×32)
        out = self.hfcf_upconv(out)
        logger.debug(f'HFCF final output: {out.shape}', once=True)
        
        return out
    
class CFRWDGenerator(nn.Module):
    def __init__(self, in_channels=1):
        super(CFRWDGenerator, self).__init__()
        self.cfr_branch = CFRBranch(in_channels=in_channels)
        self.hfcf_branch = HFCFBranch(in_channels=in_channels)
        # Learnable fusion coefficient initialized to 1.0 as described in the paper
        self.fusion_coeff = nn.Parameter(torch.tensor(1.0))
        self.fuse_cfr_hfcf = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=1, stride=1, padding=0),
            nn.Tanh()
        )

        self._initialize_weights()

    def _initialize_weights(self):
        hfcf_wavelet_convs = set()
        if hasattr(self.hfcf_branch, "dwt"):
            dwt_block = self.hfcf_branch.dwt
            if hasattr(dwt_block, "dwt"):
                haar_down = dwt_block.dwt
                for attr in ("low", "high_h", "high_v", "high_d"):
                    module = getattr(haar_down, attr, None)
                    if module is not None:
                        hfcf_wavelet_convs.add(module)
                        for param in module.parameters():
                            param.requires_grad = False

        for m in self.modules():
            if m in hfcf_wavelet_convs:
                continue
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        logger.info("Веса инициализированы (wavelet-конволюции сохранены, остальные Conv/ConvTranspose = fan_out, нормализации = 1/0).")


    def forward(self, x):
        cfr_out = self.cfr_branch(x)
        hfcf_out = self.hfcf_branch(x)
        fusion_weight = self.fusion_coeff
        print(f"CFR out shape: {cfr_out.shape}, HFCF out shape: {hfcf_out.shape}, Fusion weight: {fusion_weight.item()}")
        fused = fusion_weight * cfr_out + (1 - fusion_weight) * hfcf_out
        out = self.fuse_cfr_hfcf(fused)
        return out


import matplotlib.pyplot as plt
import cv2
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input_array = None #  cv2.imread('C:/Users/tiruu/Desktop/sar2opt_light/data/sen12/agri/s1/ROIs1868_summer_s1_59_p2.png', cv2.IMREAD_COLOR)
    if input_array is None:
        # Создаем случайный тензор с 3 каналами и размером 256x256
        input_tensor = torch.randn(1, 3, 256, 256).to(device)  # Форма: [1, 3, 256, 256]
    else:
        # Конвертируем BGR в RGB и изменяем размер до 256x256
        input_array = cv2.cvtColor(input_array, cv2.COLOR_BGR2RGB)
        input_array = cv2.resize(input_array, (256, 256))  # Приводим к размеру 256x256
        
        # Преобразуем в тензор и нормализуем
        input_tensor = torch.from_numpy(input_array).float().permute(2, 0, 1) / 255.0
        input_tensor = input_tensor.unsqueeze(0).to(device)  # Добавляем batch-размер -> [1, 3, 256, 256]

    gen = CFRWDGenerator(in_channels=3).to(device)
    out = gen(input_tensor)

    plt.subplot(1, 2, 1)
    plt.imshow(input_tensor.squeeze().permute(1, 2, 0).detach().cpu().numpy())
    plt.title('Input SAR Image')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(out.squeeze().permute(1, 2, 0).detach().cpu().numpy())
    plt.title('Generated Optical Image')
    plt.axis('off')
    plt.show()

    print('out:', out.shape)