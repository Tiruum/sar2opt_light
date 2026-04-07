import torch
import torch.nn as nn
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

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        logger.debug('Conv Block INIT')
        super(EncoderBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1), # Вообще по статье ReflectionPad используется один раз в начале, а затем, видимо, в Conv2d padding=1
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
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
            nn.InstanceNorm2d(out_channels, affine=True),
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
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=0, bias=True)
            # Tanh убран отсюда, чтобы делать fusion до нелинейности
        )
        
    def forward(self, x):
        return self.final_block(x)



class CFRBlock(nn.Module):
    def __init__(self, channels):
        super(CFRBlock, self).__init__()
        logger.debug('CFR Block INIT')

        # Базовые каналы для веток (как в HRNet: 16, 32, 64, 128)
        # Это дает сети емкость на глубоких слоях и экономит память на высоких разрешениях
        c1, c2, c3, c4 = channels // 4, channels // 2, channels, channels * 2

        # --- Stage 1 ---
        # a1 = B C W H          p1 = B c1 W H
        # a2 = B C W/2 H/2      p2 = B c2 W/2 H/2
        
        # Сначала понижаем каналы, чтобы ResBlock работал быстро
        self.proj11 = nn.Conv2d(channels, c1, kernel_size=1, bias=False)
        self.proj12 = nn.Conv2d(channels, c2, kernel_size=1, bias=False)

        self.n11 = nn.Sequential(*[ResBlock(c1) for _ in range(3)])
        self.n12 = nn.Sequential(*[ResBlock(c2) for _ in range(3)])

        # Cross-fusion 1
        self.fuse1_to2_1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(c1 + c2, c1, kernel_size=3, stride=1, bias=False),
            nn.InstanceNorm2d(c1, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.fuse1_to2_2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(c1 + c2, c2, kernel_size=3, stride=1, bias=False),
            nn.InstanceNorm2d(c2, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.fuse1_to2_3 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(c1 + c2, c3, kernel_size=3, stride=1, bias=False),
            nn.InstanceNorm2d(c3, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # --- Stage 2 ---
        self.n21 = nn.Sequential(*[ResBlock(c1) for _ in range(3)])
        self.n22 = nn.Sequential(*[ResBlock(c2) for _ in range(3)])
        self.n23 = nn.Sequential(*[ResBlock(c3) for _ in range(3)])

        # Cross-fusion 2
        self.fuse2_to3_1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(c1 + c2 + c3, c1, kernel_size=3, stride=1, bias=False),
            nn.InstanceNorm2d(c1, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.fuse2_to3_2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(c1 + c2 + c3, c2, kernel_size=3, stride=1, bias=False),
            nn.InstanceNorm2d(c2, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.fuse2_to3_3 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(c1 + c2 + c3, c3, kernel_size=3, stride=1, bias=False),
            nn.InstanceNorm2d(c3, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.fuse2_to3_4 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(c1 + c2 + c3, c4, kernel_size=3, stride=1, bias=False),
            nn.InstanceNorm2d(c4, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # --- Stage 3 ---
        self.n31 = nn.Sequential(*[ResBlock(c1) for _ in range(3)])
        self.n32 = nn.Sequential(*[ResBlock(c2) for _ in range(3)])
        self.n33 = nn.Sequential(*[ResBlock(c3) for _ in range(3)])
        self.n34 = nn.Sequential(*[ResBlock(c4) for _ in range(3)])

        # Final fusion (Сливаем все на максимальном разрешении W H)
        self.fuse3_to4 = nn.Sequential(
            nn.Conv2d(c1 + c2 + c3 + c4, channels, kernel_size=1, stride=1, bias=False),
            nn.InstanceNorm2d(channels, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.down = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        logger.debug(f'Input shape: {x.shape}', once=True)

        a1 = x
        a2 = self.down(a1)

        # Stage 1
        p1 = self.n11(self.proj11(a1))
        p2 = self.n12(self.proj12(a2))

        logger.debug('Stage 1', once=True)
        logger.debug(f'a1 shape: {a1.shape}, p1 shape: {p1.shape}', once=True)
        logger.debug(f'a2 shape: {a2.shape}, p2 shape: {p2.shape}', once=True)

        # Cross-fusion 1
        p1_d = self.down(p1)
        b1 = self.fuse1_to2_1(torch.cat([p1, self.up(p2)], dim=1))
        b2 = self.fuse1_to2_2(torch.cat([p1_d, p2], dim=1))
        b3 = self.fuse1_to2_3(torch.cat([self.down(p1_d), self.down(p2)], dim=1))

        # Stage 2
        q1 = self.n21(b1)
        q2 = self.n22(b2)
        q3 = self.n23(b3)

        logger.debug('Stage 2', once=True)
        logger.debug(f'b1 shape: {b1.shape}, q1 shape: {q1.shape}', once=True)
        logger.debug(f'b2 shape: {b2.shape}, q2 shape: {q2.shape}', once=True)
        logger.debug(f'b3 shape: {b3.shape}, q3 shape: {q3.shape}', once=True)

        # Cross-fusion 2
        q1_d  = self.down(q1)
        q2_d  = self.down(q2)
        q3_u  = self.up(q3)
        q1_dd = self.down(q1_d)   # cached: reused in c3 and c4
        q2_dd = self.down(q2_d)   # cached: reused in c4
        c1 = self.fuse2_to3_1(torch.cat([q1, self.up(q2), self.up(q3_u)], dim=1))
        c2 = self.fuse2_to3_2(torch.cat([q1_d, q2, q3_u], dim=1))
        c3 = self.fuse2_to3_3(torch.cat([q1_dd, q2_d, q3], dim=1))
        c4 = self.fuse2_to3_4(torch.cat([self.down(q1_dd), q2_dd, self.down(q3)], dim=1))

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

        # Final fusion (Сливаем все на максимальном разрешении k1)
        k2_u = self.up(k2)
        k3_u = self.up(self.up(k3))
        k4_u = self.up(self.up(self.up(k4)))
        
        d = self.fuse3_to4(torch.cat([k1, k2_u, k3_u, k4_u], dim=1))
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
        logger.debug(f'CFRBlock:\t{base_cfr_ch} -> {base_cfr_ch}')

        # Выводим 32ch feature map — финальная проекция в RGB выполняется
        # в CFRWDGenerator.final (общий для обеих веток).
        self.decoder = nn.Sequential(
            DecoderBlock(in_channels=base_cfr_ch, out_channels=32, upsample=False),
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
    Прямое 2D вейвлет-преобразование (Haar DWT), ортонормальная версия.

    pixel_unshuffle собирает 2×2-блок [TL, TR, BL, BR] в 4 канала,
    матрица Haar (масштаб 0.5) даёт ортонормальные коэффициенты:
        LL = (TL + TR + BL + BR) × 0.5
        LH = (–TL + TR – BL + BR) × 0.5
        HL = (–TL – TR + BL + BR) × 0.5
        HH = (TL – TR – BL + BR) × 0.5
    Сумма квадратов норм = сумма квадратов пикселей (Parseval). ✓
    """
    _HAAR = torch.tensor([
        [ 1.0,  1.0,  1.0,  1.0],  # LL
        [-1.0,  1.0, -1.0,  1.0],  # LH
        [-1.0, -1.0,  1.0,  1.0],  # HL
        [ 1.0, -1.0, -1.0,  1.0],  # HH
    ], dtype=torch.float32) * 0.5  # Orthonormal: each row has L2-norm 1.0

    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.register_buffer('haar_weights', self._HAAR.clone())

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        # pixel_unshuffle: B×C×H×W → B×(4C)×(H/2)×(W/2)
        # .view: B×C×4×(H/2)×(W/2)
        x_blocks = torch.nn.functional.pixel_unshuffle(x, 2).view(B, C, 4, H // 2, W // 2)
        out = torch.einsum('bcihw, oi -> bcohw', x_blocks, self.haar_weights)
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
        self.dwt = HaarDown(in_channels=in_channels)

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
            nn.InstanceNorm2d(out_channels, affine=True),
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
                nn.InstanceNorm2d(out_channels, affine=True),
                nn.ReLU(inplace=True)
            ])
        else:
            # Если сжимать не надо, добавляем еще слой обработки (глубина помогает денойзингу).
            layers.extend([
                nn.ReflectionPad2d(1),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, bias=False),
                nn.InstanceNorm2d(out_channels, affine=True),
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
        
        # ResNet101 bottleneck: expansion=4, mid = channels // 4
        # При channels=64 → mid=16 (стандартный bottleneck)
        mid_channels = max(channels // 4, 16)
        
        self.main_branch = nn.Sequential(
            # 1x1
            nn.Conv2d(channels, mid_channels, kernel_size=1, bias=False),
            nn.InstanceNorm2d(mid_channels, affine=True),
            nn.ReLU(inplace=True),
            
            # 3x3
            nn.ReflectionPad2d(1),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(mid_channels, affine=True),
            nn.ReLU(inplace=True),
            
            # 1x1
            nn.Conv2d(mid_channels, channels, kernel_size=1, bias=False),
            nn.InstanceNorm2d(channels, affine=True)
        )
        
        # Skip connection
        if projection:
            self.skip_branch = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=1, bias=False),
                nn.InstanceNorm2d(channels, affine=True)
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
            nn.InstanceNorm2d(channels, affine=True),
            nn.ReLU(inplace=True),
            
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, bias=False),
            nn.InstanceNorm2d(channels, affine=True)
        )
        self.final_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.final_relu(self.main_branch(x) + x)
    
class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (Woo et al., 2018).
    Последовательно применяет Channel Attention и Spatial Attention.

    В контексте SAR→OPT: подавляет спекл-шум в высокочастотных вейвлет-коэффициентах
    и усиливает коэффициенты, соответствующие реальным структурам сцены
    (края зданий, границы полей, русла рек).
    """
    def __init__(self, channels, reduction_ratio=4):
        super(CBAM, self).__init__()
        reduced = max(channels // reduction_ratio, 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # Общий MLP для avg и max ветвей канального внимания
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(channels, reduced, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, channels, kernel_size=1, bias=False),
        )
        # Пространственное внимание: 7×7 (большое рецептивное поле)
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Channel Attention: avg + max → shared MLP → sigmoid gate
        ca = self.channel_mlp(self.avg_pool(x)) + self.channel_mlp(self.max_pool(x))
        x = x * torch.sigmoid(ca)
        # Spatial Attention: channel-wise avg + max → conv → sigmoid gate
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = x * self.spatial_conv(torch.cat([avg_out, max_out], dim=1))
        return x


class HFCFBranch(nn.Module):
    """
    Wavelet Decomposition branch с тремя независимыми потоками.

    Исправления по аудиту архитектуры:
    - Ранняя сумма g2+g3 до потоков устранена: каждая группа обрабатывается независимо.
    - LL2 добавлен как отдельный Low-поток (структурный контекст), что помогает
      Mid/High потокам ориентироваться в пространстве и снижает вероятность
      "затухания" HFCF-ветки (fusion_weight → 0).
    - CBAM на входе каждого потока фильтрует спекл-шум SAR до ResBlocks.

    Архитектура потоков (все приводятся к W/4 перед слиянием):

      Low  (LL2 @ W/4)       : HFCFPreprocess → CBAM → WDResBlock×2      ──┐
      Mid  ([LH2,HL2,HH2] W/4): HFCFPreprocess → CBAM → WDResBlock×6 (Y/B)──┤ → merge(1×1) → decoder → 3ch
      High ([LH1,HL1,HH1] W/2): HFCFPreprocess(↓) → CBAM → RedBlock×2    ──┘
    """
    def __init__(self, in_channels=1, hidden_dim=64):
        super(HFCFBranch, self).__init__()
        logger.debug('HFCF BRANCH INIT')
        freq_c = 3 * in_channels  # каналы cat-группы высокочастотных подполос

        self.dwt = DWTBlock(in_channels=in_channels)

        # --- Low-freq stream (LL2 @ W/4): структурный контекст сцены ---
        self.ll_proj   = HFCFPreprocess(in_channels, hidden_dim, downsample=False)
        self.ll_cbam   = CBAM(hidden_dim)
        self.ll_stream = nn.Sequential(
            WDResBlock(hidden_dim, projection=True),
            WDResBlock(hidden_dim, projection=False),
        )

        # --- Mid-freq stream ([LH2,HL2,HH2] @ W/4): детали 2-го масштаба ---
        # Yellow/Blue блоки (ResNet101 bottleneck) — как в оригинальной статье
        self.mid_proj   = HFCFPreprocess(freq_c, hidden_dim, downsample=False)
        self.mid_cbam   = CBAM(hidden_dim)
        self.mid_stream = nn.Sequential(
            WDResBlock(hidden_dim, projection=True),
            WDResBlock(hidden_dim, projection=False),
            WDResBlock(hidden_dim, projection=False),
            WDResBlock(hidden_dim, projection=True),
            WDResBlock(hidden_dim, projection=False),
            WDResBlock(hidden_dim, projection=False),
        )

        # --- High-freq stream ([LH1,HL1,HH1] @ W/2 → W/4): тонкие края ---
        # HFCFPreprocess(downsample=True) выполняет W/2 → W/4 через stride=2 Conv
        # Red блоки (ResNet18 basic) — как в оригинальной статье
        self.high_proj   = HFCFPreprocess(freq_c, hidden_dim, downsample=True)
        self.high_cbam   = CBAM(hidden_dim)
        self.high_stream = nn.Sequential(
            RedBlock(hidden_dim),
            RedBlock(hidden_dim),
        )

        # Слияние трёх потоков: 3×hidden_dim → hidden_dim @ W/4
        self.merge = nn.Sequential(
            nn.Conv2d(3 * hidden_dim, hidden_dim, kernel_size=1, bias=False),
            nn.InstanceNorm2d(hidden_dim, affine=True),
            nn.ReLU(inplace=True),
        )

        # Декодер: W/4 → W. Выводим 32ch feature map —
        # финальная проекция в RGB выполняется в CFRWDGenerator.final.
        self.decoder = nn.Sequential(
            DecoderBlock(hidden_dim, 64, upsample=True),   # W/4 → W/2
            DecoderBlock(64, 64, upsample=False),          # W/2, refine
            DecoderBlock(64, 32, upsample=True),           # W/2 → W
            DecoderBlock(32, 32, upsample=False),          # W, refine
        )

    def forward(self, x):
        g1, g2, g3 = self.dwt(x)  # g1=LL2 @ W/4, g2=[LH2,HL2,HH2] @ W/4, g3=[LH1,HL1,HH1] @ W/2

        logger.debug(f'DWT shapes: g1={g1.shape}, g2={g2.shape}, g3={g3.shape}', once=True)

        # Три полностью независимых потока — никакого преждевременного смешения
        ll   = self.ll_stream(self.ll_cbam(self.ll_proj(g1)))
        mid  = self.mid_stream(self.mid_cbam(self.mid_proj(g2)))
        high = self.high_stream(self.high_cbam(self.high_proj(g3)))  # W/2 → W/4

        merged = self.merge(torch.cat([ll, mid, high], dim=1))
        return self.decoder(merged)
    
class AdaptiveFusion(nn.Module):
    """
    Пространственно-адаптивное слияние CFR и HFCF feature maps.

    Работает на уровне признаков (32ch), а не пиксельных логитов.
    Это гарантирует, что градиент лосса доходит до обеих веток через
    общий декодер независимо от весов fusion.

    Карта весов: weights = Softmax(Conv(Cat(cfr_feats, hfcf_feats))) → B×2×H×W
      w_hfcf ≈ 1 — на краях и текстурах (HFCF незаменима)
      w_hfcf ≈ 0 — на однородных областях (CFR достаточно)

    Возвращает (fused_feats, weights) — weights используются для логирования
    вклада веток (fusion/w_hfcf, fusion/spatial_std).
    """
    def __init__(self, feat_channels=32):
        super(AdaptiveFusion, self).__init__()
        mid = feat_channels * 2  # 64ch
        self.net = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(mid, mid, kernel_size=3, bias=False),
            nn.InstanceNorm2d(mid, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, 2, kernel_size=1),   # 2 ветки → 2 пространственных веса
        )
        # Temperature scaling: flattens softmax during early training when logits
        # are large → prevents premature routing collapse. Learned jointly with G.
        # clamp(min=0.1) in forward() prevents division by zero.
        self.temperature = nn.Parameter(torch.ones(1) * 2.0)

    def forward(self, cfr_feats, hfcf_feats):
        # weights: B×2×H×W, сумма по dim=1 = 1 в каждом пикселе
        logits = self.net(torch.cat([cfr_feats, hfcf_feats], dim=1))
        weights = torch.softmax(logits / self.temperature.clamp(min=0.1), dim=1)
        fused = cfr_feats * weights[:, 0:1] + hfcf_feats * weights[:, 1:2]
        return fused, weights


class CFRWDGenerator(nn.Module):
    def __init__(self, in_channels=1):
        super(CFRWDGenerator, self).__init__()
        self.cfr_branch = CFRBranch(in_channels=in_channels)
        self.hfcf_branch = HFCFBranch(in_channels=in_channels)
        # Слияние на уровне 32ch feature maps: градиент лосса всегда
        # проходит через обе ветки через общий self.final.
        self.adaptive_fusion = AdaptiveFusion(feat_channels=32)
        # Единственный финальный декодер 32ch → 3ch для всего генератора.
        self.final = FinalDecoderBlock(32, 3, kernel_size=7)

        self._initialize_weights()

    def _initialize_weights(self):
        # 1. Общая инициализация: Kaiming для Conv, единицы/нули для InstanceNorm
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.InstanceNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # 2. Xavier для финального Conv перед Tanh (gain подобран под tanh)
        for m in self.modules():
            if isinstance(m, FinalDecoderBlock):
                for sub in m.modules():
                    if isinstance(sub, nn.Conv2d):
                        nn.init.xavier_normal_(sub.weight, gain=nn.init.calculate_gain('tanh'))

        logger.info("Веса инициализированы (Kaiming fan_in + Xavier/Tanh).")

    def forward(self, x, return_branches=False):
        cfr_feats = self.cfr_branch(x)    # B×32×H×W
        hfcf_feats = self.hfcf_branch(x)  # B×32×H×W

        # Feature-level fusion: градиент идёт в обе ветки через self.final
        fused_feats, fusion_weights = self.adaptive_fusion(cfr_feats, hfcf_feats)
        out = torch.tanh(self.final(fused_feats))

        if return_branches:
            # Пропускаем ветки через общий финальный слой для визуализации.
            # torch.no_grad() — только для диагностики, не влияет на обучение.
            with torch.no_grad():
                cfr_out = torch.tanh(self.final(cfr_feats))
                hfcf_out = torch.tanh(self.final(hfcf_feats))
            return out, cfr_out, hfcf_out, fusion_weights
        return out, fusion_weights


if __name__ == "__main__":
    import cv2
    import matplotlib.pyplot as plt

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
    with torch.no_grad():
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