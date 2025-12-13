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
            nn.BatchNorm2d(channels, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(channels, affine=True)
        )
        
    def forward(self, x):
        return x + self.block(x)

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        logger.debug('Conv Block INIT')
        super(EncoderBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1), # Вообще по статье ReflectionPad используется один раз в начале, а затем, видимо, в Conv2d padding=1
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0, bias=False),
            nn.BatchNorm2d(out_channels, affine=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

    def forward(self, x):
        return self.block(x)
    
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=False):
        super(DecoderBlock, self).__init__()
        logger.debug('DecoderBlock Block INIT')
        layers = []

        if upsample:
            layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))

        layers.extend([
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=False),
            # nn.BatchNorm2d(out_channels, affine=True),
            nn.BatchNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True)
        ])
        # В энкодере LeakyReLU нужен, чтобы градиенты не "умирали" при сжатии информации.
        # В декодере, когда мы восстанавливаем изображение,
        # обычный ReLU помогает "отсекать" лишний шум и делать выход более чистым (sparse activations),
        # что полезно для финальной картинки.

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class FinalDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7):
        super(FinalDecoderBlock, self).__init__()
        # Для kernel_size=7 паддинг должен быть 3, чтобы сохранить размер (H, W)
        pad = (kernel_size - 1) // 2  # для 7 это будет 3
        
        self.final_block = nn.Sequential(
            nn.ReflectionPad2d(pad),
            # Важно: bias=True, так как нет нормализации после этой свертки
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=0, bias=True),
            nn.Tanh() # Выход в диапазоне [-1, 1]
        )
        
    def forward(self, x):
        return self.final_block(x)



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
            # Дополнительно можно обернуть их в лёгкий Conv+Norm+LeakyReLU (InstanceNorm/BN + LeakyReLU),
            # чтобы смешивание масштабов было более выразительным.
            # Это уже инженерный апгрейд, а не строгое следование статье.
            # Аналогично для fuse1_to2_2, fuse1_to2_3, fuse2_to3_1..4
            # Если позже добавить BatchNorm2d после этих fusion‑свёрток,
            # тогда в них стоит поменять на bias=False,
            # чтобы не плодить бесполезные параметры.
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
        logger.debug(f'CFRBlock output shape: {d.shape}', once=True)

        return d
    
class CFRBranch(nn.Module):
    def __init__(self, in_channels=1):
        super(CFRBranch, self).__init__()
        logger.debug('CFR BRANCH INIT')
        logger.debug('CFR BRANCH INIT DEBUG')
        base_cfr_ch = 64

        self.encoder = nn.Sequential(
            EncoderBlock(in_channels, 16),
            EncoderBlock(16, 32),
            EncoderBlock(32, 64),
            EncoderBlock(base_cfr_ch, base_cfr_ch),
        )

        logger.debug(f"Conv:\t\t{' -> '.join(map(str, [in_channels, 16, 32, base_cfr_ch, base_cfr_ch]))}")

        self.CFR = CFRBlock(base_cfr_ch)
        logger.debug(f'CFRBlock:\t{base_cfr_ch} -> {base_cfr_ch // 4}')

        self.decoder = nn.Sequential(
            # Шаг 1: Восстанавливаем разрешение 128 -> 256.
            # Расширяем каналы 16 -> 64, чтобы дать модели пространство для генерации деталей.
            DecoderBlock(in_channels=base_cfr_ch // 4, out_channels=64, upsample=True),
            
            # Шаг 2: Дополнительный слой обработки на полном разрешении (256x256).
            # Сжимаем каналы 64 -> 32 перед финалом.
            DecoderBlock(in_channels=64, out_channels=32, upsample=False),
            
            # Шаг 3: Финальная проекция в RGB с большим ядром (7x7).
            # 32 -> 3 канала.
            FinalDecoderBlock(in_channels=32, out_channels=3, kernel_size=7)
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
    
# class HaarUp(nn.Module):
#     def __init__(self, in_channels=1):
#         super(HaarUp, self).__init__()
#         # Обратная матрица Haar (transposed)
#         self.register_buffer('inv_weights', torch.tensor([
#              [ 1.0, -1.0, -1.0,  1.0],
#              [ 1.0,  1.0, -1.0, -1.0],
#              [ 1.0, -1.0,  1.0, -1.0],
#              [ 1.0,  1.0,  1.0,  1.0]
#         ], dtype=torch.float32))
#     def forward(self, LL, LH, HL, HH):
#         stack = torch.stack([LL, LH, HL, HH], dim=2)
#         weights = self.inv_weights.to(LL.device)
#         out_pixels = torch.einsum('bcihw, oi -> bcohw', stack, weights)
#         B, C, _, H, W = out_pixels.shape
#         out_pixels = out_pixels.view(B, C * 4, H, W)
#         return torch.nn.functional.pixel_shuffle(out_pixels, 2)

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
    SOTA & SAR-Specific Preprocessing.
    1. Expand: Расширяем каналы сразу, чтобы "размазать" шум и выделить признаки.
    2. Filter/Downsample: Вторая свертка работает как обучаемый фильтр (вместо MaxPool).
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, downsample=True):
        super(HFCFPreprocess, self).__init__()
        logger.debug('HFCF Preprocess INIT')

        # 1. Этап "Conv2d -> BN -> ReLU"
        # Сразу расширяем каналы (in -> out). 
        # Это дает сети "место" для извлечения признаков.
        layers = [
            nn.ReflectionPad2d(padding),
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride),
            nn.BatchNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True)
        ]

        # 2. Этап "MaxPool" (заменен на Conv)
        # Каналы уже out_channels, поэтому здесь out -> out.
        if downsample:
            # Stride=2 свертка лучше MaxPool для SAR, так как учится игнорировать спеклы,
            # а не просто выбирать максимум (который может быть спеклом!).
            layers.extend([
                nn.ReflectionPad2d(1),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, bias=False),
                nn.BatchNorm2d(out_channels, affine=True),
                nn.ReLU(inplace=True)
            ])
        else:
            # Если сжимать не надо, добавляем еще слой обработки (глубина помогает денойзингу).
            layers.extend([
                nn.ReflectionPad2d(1),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, bias=False),
                nn.BatchNorm2d(out_channels, affine=True),
                nn.ReLU(inplace=True)
            ])
            
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.block(x)
    
class WDResBlock(nn.Module):
    """
    Универсальный класс для Yellow и Blue блоков.
    Тип блока (Yellow/Blue) определяется параметром `projection`.
    
    Структура Main: 1x1 -> 3x3 -> 1x1 (Bottleneck)
    Структура Skip:
       - Если projection=True (Yellow): Conv1x1 -> Norm
       - Если projection=False (Blue): Identity
    """
    def __init__(self, channels, projection=False):
        super(WDResBlock, self).__init__()
        
        # Внутренние каналы. В ResNet обычно channels // 4. 
        # Но у нас каналов мало (напр. 64), делить на 4 (16) может быть узко.
        # Давайте сделаем channels // 2 или оставим channels.
        # Для HFCF (детали) лучше сохранить поток: mid = channels.
        mid_channels = channels 
        
        self.main_branch = nn.Sequential(
            # 1x1
            nn.Conv2d(channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels, affine=True),
            nn.ReLU(inplace=True),
            
            # 3x3
            nn.ReflectionPad2d(1),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(mid_channels, affine=True),
            nn.ReLU(inplace=True),
            
            # 1x1
            nn.Conv2d(mid_channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels, affine=True)
        )
        
        # Skip connection
        if projection:
            self.skip_branch = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(channels, affine=True)
            )
        else:
            self.skip_branch = nn.Identity()
            
        self.final_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.final_relu(self.main_branch(x) + self.skip_branch(x))


class RedBlock(nn.Module):
    """
    SOTA implementation of the 'Red' block (Basic ResBlock).
    Conv3x3 -> IN -> ReLU -> Conv3x3 -> IN + Identity -> ReLU.
    """
    def __init__(self, channels):
        super(RedBlock, self).__init__()
        
        self.main_branch = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(channels, affine=True),
            nn.ReLU(inplace=True),
            
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(channels, affine=True)
        )
        self.final_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.final_relu(self.main_branch(x) + x)
    
class HFCFBranch(nn.Module):
    def __init__(self, in_channels=1, hidden_dim=64):
        super(HFCFBranch, self).__init__()
        logger.debug('HFCF BRANCH INIT')

        # --- DWT ---
        self.dwt = DWTBlock(in_channels=in_channels)
        freq_c = 3 * in_channels # 3 канала вейвлетов

        # --- 1. Preprocess (SOTA: Strided Conv) ---
        # Upper (G2, 64x64): Не сжимаем
        self.pre_top = HFCFPreprocess(freq_c, hidden_dim, downsample=False)
        # Lower (G3, 128x128): Сжимаем (Stride=2)
        self.pre_bot = HFCFPreprocess(freq_c, hidden_dim, downsample=True)

        # --- 2. Streams ---

        # Top Stream (Yellow/Blue -> Bottlenecks)
        # Y(Proj) -> B(Id) -> B(Id) -> Y(Proj) -> B(Id) -> B(Id)
        self.top_stream = nn.Sequential(
            WDResBlock(hidden_dim, projection=True),
            WDResBlock(hidden_dim, projection=False),
            WDResBlock(hidden_dim, projection=False),
            WDResBlock(hidden_dim, projection=True),
            WDResBlock(hidden_dim, projection=False),
            WDResBlock(hidden_dim, projection=False)
        )
        
        # Bottom Stream (Red -> Basic Blocks)
        # Red -> Red
        self.bot_stream = nn.Sequential(
            RedBlock(hidden_dim),
            RedBlock(hidden_dim)
        )
        
        # --- 3. Fusion & Transition ---
        # Оставим Conv+Norm+ReLU для стабильности,
        # 1x1 Conv для смешивания каналов после Concat (128 -> 64)
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim, affine=True),
            nn.ReLU(inplace=True)
        )

        # --- 4. Decoder (SOTA: Upsample + Conv) ---
        self.decoder = nn.Sequential(
            # 64 -> 128
            DecoderBlock(hidden_dim, 64, upsample=True),
            # Refine at 128
            DecoderBlock(64, 64, upsample=False),
            
            # 128 -> 256
            DecoderBlock(64, 32, upsample=True),
            # Refine at 256
            DecoderBlock(32, 32, upsample=False)
        )
        
        # --- 5. Final Output ---
        self.final = FinalDecoderBlock(32, 3, kernel_size=7)

    def forward(self, x):
        g1, g2, g3 = self.dwt(x)

        logger.debug('DWT shapes:', once=True)
        logger.debug(f'g1.shape: {g1.shape}, g2.shape: {g2.shape}, g3.shape: {g3.shape}', once=True)

        hfcf_g2_in = self.pre_top(g2).to(g2.device)
        hfcf_g3 = self.pre_bot(g3).to(g3.device)

        hfcf_g2 = hfcf_g2_in + hfcf_g3

        out_g2 = self.top_stream(hfcf_g2)
        out_g3 = self.bot_stream(hfcf_g3)

        merged = torch.cat([out_g2, out_g3], dim=1) # 64+64=128 ch
        merged = self.fusion_conv(merged) # -> 64 ch

        dec = self.decoder(merged)
        out = self.final(dec)

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
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                # Kaiming init
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm2d)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        logger.info("Веса инициализированы (wavelet-конволюции сохранены, остальные Conv/ConvTranspose = fan_out, нормализации = 1/0).")


    def forward(self, x):
        cfr_out = self.cfr_branch(x)
        hfcf_out = self.hfcf_branch(x)
        fusion_weight = self.fusion_coeff
        logger.debug(f"CFR out shape: {cfr_out.shape}, HFCF out shape: {hfcf_out.shape}, Fusion weight: {fusion_weight.item()}")
        alpha = torch.sigmoid(self.fusion_coeff)
        out = alpha * cfr_out + (1 - alpha) * hfcf_out
        return out


import matplotlib.pyplot as plt
import cv2
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input_array = None #  cv2.imread('C:/Users/tiruu/Desktop/sar2opt_light/data/sen12/agri/s1/ROIs1868_summer_s1_59_p2.png', cv2.IMREAD_COLOR)
    if input_array is None:
        # Создаем случайный тензор с 3 каналами и размером 256x256
        input_tensor = torch.randn(1, 1, 256, 256).to(device)  # Форма: [1, 3, 256, 256]
    else:
        # Конвертируем BGR в RGB и изменяем размер до 256x256
        input_array = cv2.cvtColor(input_array, cv2.COLOR_BGR2RGB)
        input_array = cv2.resize(input_array, (256, 256))  # Приводим к размеру 256x256
        
        # Преобразуем в тензор и нормализуем
        input_tensor = torch.from_numpy(input_array).float().permute(2, 0, 1) / 255.0
        input_tensor = input_tensor.unsqueeze(0).to(device)  # Добавляем batch-размер -> [1, 3, 256, 256]

    gen = CFRWDGenerator(in_channels=1).to(device)
    out = gen(input_tensor)

    plt.subplot(1, 2, 1)
    plt.imshow(input_tensor.squeeze().detach().cpu().numpy())
    plt.title('Input SAR Image')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(out.squeeze().permute(1, 2, 0).detach().cpu().numpy())
    plt.title('Generated Optical Image')
    plt.axis('off')
    plt.show()

    print('out:', out.shape)