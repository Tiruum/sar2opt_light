import torch
import torch.nn as nn
import math

class ResBlock(nn.Module):
    """ Остаточный блок из ResNet """
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(channels)
        )
        
    def forward(self, x):
        return x + self.block(x) # Остаточное соединение: x + F(x)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2):
        super(ConvBlock, self).__init__()

        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1), # Вообще по статье ReflectionPad используется один раз в начале, а затем, видимо, в Conv2d padding=1
            nn.Conv2d(in_channels, out_channels, kernel_size, stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_block(x)
    
class TConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super(TConvBlock, self).__init__()
        self.t_conv_block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.t_conv_block(x)

class FinalTConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(FinalTConvBlock, self).__init__()
        self.final_t_conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=0),
            nn.Tanh()
        )
        
    def forward(self, x):
        return self.final_t_conv_block(x)

class CFRBlock(nn.Module):
    def __init__(self, channels, debug=False):
        super(CFRBlock, self).__init__()
        self.debug = debug

        # a1 = B C W H          p1 = B C/4 W H
        # a2 = B C W/2 H/2      p2 = B C/2 W/2 H/2

        # b1 = Conv(cat(p1, U(p2)))
        # b2 = Conv(cat(D(p1), p2))
        # b3 = Conv(cat(D(D(p1)), D(p2)))

        self.n11 = nn.Sequential(
            *[ResBlock(channels) for _ in range(3)],
            nn.Conv2d(channels, channels // 4, kernel_size=1, stride=1, padding=0)
        )

        self.n12 = nn.Sequential(
            *[ResBlock(channels) for _ in range(3)],
            nn.Conv2d(channels, channels // 2, kernel_size=1, stride=1, padding=0)
        )

        self.fuse1_to2_1 = nn.Conv2d((channels // 4 + channels // 2), channels // 8, kernel_size=3, stride=1, padding=1)
        self.fuse1_to2_2 = nn.Conv2d((channels // 4 + channels // 2), channels // 4, kernel_size=3, stride=1, padding=1)
        self.fuse1_to2_3 = nn.Conv2d((channels // 4 + channels // 2), channels // 2, kernel_size=3, stride=1, padding=1)

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

        self.fuse2_to3_1 = nn.Conv2d((channels // 8 + channels // 4 + channels // 2), channels // 16, kernel_size=3, stride=1, padding=1)
        self.fuse2_to3_2 = nn.Conv2d((channels // 8 + channels // 4 + channels // 2), channels // 8, kernel_size=3, stride=1, padding=1)
        self.fuse2_to3_3 = nn.Conv2d((channels // 8 + channels // 4 + channels // 2), channels // 4, kernel_size=3, stride=1, padding=1)
        self.fuse2_to3_4 = nn.Conv2d((channels // 8 + channels // 4 + channels // 2), channels // 2, kernel_size=3, stride=1, padding=1)

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

        self.fuse3_to4 = nn.Conv2d((channels // 16 + channels // 8 + channels // 4 + channels // 2), channels // 4, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        down = nn.AvgPool2d(kernel_size=2, stride=2)

        if self.debug: print('Input shape:', x.shape)

        a1 = x
        a2 = down(a1)

        # Stage 1
        p1 = self.n11(a1)
        p2 = self.n12(a2)

        if self.debug:
            print('\nStage 1')
            print('a1 shape:', a1.shape, '\t\tp1 shape:', p1.shape)
            print('a2 shape:', a2.shape, '\t\tp2 shape:', p2.shape)

        # Cross-fusion 1
        b1 = self.fuse1_to2_1(torch.cat([p1, up(p2)], dim=1))
        b2 = self.fuse1_to2_2(torch.cat([down(p1), p2], dim=1))
        b3 = self.fuse1_to2_3(torch.cat([down(down(p1)), down(p2)], dim=1))

        # Stage 2
        q1 = self.n21(b1)
        q2 = self.n22(b2)
        q3 = self.n23(b3)

        if self.debug:
            print('\nStage 2')
            print('b1 shape:', b1.shape, '\t\tq1 shape:', q1.shape)
            print('b2 shape:', b2.shape, '\t\tq2 shape:', q2.shape)
            print('b3 shape:', b3.shape, '\t\tq3 shape:', q3.shape)

        # Cross-fusion 2
        c1 = self.fuse2_to3_1(torch.cat([q1, up(q2), up(up(q3))], dim=1))
        c2 = self.fuse2_to3_2(torch.cat([down(q1), q2, up(q3)], dim=1))
        c3 = self.fuse2_to3_3(torch.cat([down(down(q1)), down(q2), q3], dim=1))
        c4 = self.fuse2_to3_4(torch.cat([down(down(down(q1))), down(down(q2)), down(q3)], dim=1))

        # Stage 3
        k1 = self.n31(c1)
        k2 = self.n32(c2)
        k3 = self.n33(c3)
        k4 = self.n34(c4)

        if self.debug:
            print('\nStage 3')
            print('c1 shape:', c1.shape, '\t\tk1 shape:', k1.shape)
            print('c2 shape:', c2.shape, '\t\tk2 shape:', k2.shape)
            print('c3 shape:', c3.shape, '\t\tk3 shape:', k3.shape)
            print('c4 shape:', c4.shape, '\t\tk4 shape:', k4.shape)

        # Final fusion
        d = self.fuse3_to4(torch.cat([down(k1), k2, up(k3), up(up(k4))], dim=1))
        if self.debug: print('Output shape:', d.shape)
        # d = up(d)

        return d
    
class CFRBranch(nn.Module):
    def __init__(self, in_channel=1, image_size=256, debug=False):
        super(CFRBranch, self).__init__()
        self.debug = debug
        self.image_size = image_size
        self.num_down = int(math.log2(self.image_size) - 4)
        self.base_channels = self.image_size * 2
        if self.debug:
            print('\nCFR BRANCH INIT DEBUG')
            print(f'Image size: {self.image_size}, num_down: {self.num_down}, base_channels: {self.base_channels}')

        in_ch = [in_channel] + [2** (i+6) for i in range(self.num_down)]  # 1→64→128→...→base_ch
        out_ch = [2** (i+6) for i in range(self.num_down)]  # 64→128→...→base_ch
        conv_layers = []
        for ic, oc in zip(in_ch, out_ch):
            conv_layers.append(ConvBlock(ic, oc))

        if self.debug: print('Conv:\t\t', ' -> '.join(map(str, in_ch)))
        self.conv = nn.Sequential(*conv_layers)

        self.CFR = CFRBlock(self.base_channels, debug)
        if self.debug: print(f'CFRBlock:\t{self.base_channels} -> {self.base_channels // 4}')

        # self.conv = nn.Sequential(
        #     ConvBlock(in_channel, 64),
        #     ConvBlock(64, 128),
        #     ConvBlock(128, 256),
        #     ConvBlock(256, 512)
        # )
        # self.CFR = CFRBlock(base_channels, debug)
        # self.upconv = nn.Sequential(
        #     TConvBlock(128, 256),  # 8x8 → 16x16
        #     TConvBlock(256, 128),  # 16x16 → 32x32
        #     TConvBlock(128, 64),   # 32x32 → 64x64
        #     TConvBlock(64, 32),    # 64x64 → 128x128
        #     TConvBlock(32, 16),    # 128x128 → 256x256
        #     FinalTConvBlock(16, 3) # 256x256 → 256x256, с Tanh на выходе
        # )

        dec_in_ch = self.base_channels // 4  # 128 for 512
        dec_out_ch_list = [dec_in_ch, dec_in_ch * 2] + out_ch[::-1][1:]  # Reverse encoder outs + extra
        up_layers = []
        for i in range(self.num_down):
            up_layers.append(TConvBlock(dec_out_ch_list[i], dec_out_ch_list[i+1]))

        up_layers.append(TConvBlock(dec_out_ch_list[-1], 16))  # Extra to 16, adjust if needed
        up_layers.append(FinalTConvBlock(16, 3))
        if self.debug: print('Upconv:\t\t', ' -> '.join(map(str, dec_out_ch_list + [16, 3])))
        self.upconv = nn.Sequential(*up_layers)

    def forward(self, x):
        if self.debug: print('\nCFR BRANCH FORWARD DEBUG')
        if self.debug: print('CFRBranch Input shape:', x.shape)
        x = self.conv(x)
        if self.debug: print('After Conv shape:', x.shape)
        x = self.CFR(x)
        if self.debug: print('After CFR shape:', x.shape)
        x = self.upconv(x)
        if self.debug: print('CFRBranch Output shape:', x.shape)
        return x

class HaarDown(nn.Module):
    def __init__(self, in_channels=1, normalize=True):
        super(HaarDown, self).__init__()
        base = 0.5 if not normalize else 1 / math.sqrt(2)
        
        # Low-pass filter (average)
        self.low = nn.Conv2d(in_channels, in_channels, kernel_size=2, stride=2, bias=False, groups=in_channels)
        self.low.weight.data.fill_(base)
        
        # High-pass horizontal (LH)
        self.high_h = nn.Conv2d(in_channels, in_channels, kernel_size=2, stride=2, bias=False, groups=in_channels)
        pattern_h = torch.tensor([[[base, base], [-base, -base]]])
        self.high_h.weight.data = pattern_h.repeat(in_channels, 1, 1, 1)
        
        # High-pass vertical (HL)
        self.high_v = nn.Conv2d(in_channels, in_channels, kernel_size=2, stride=2, bias=False, groups=in_channels)
        pattern_v = torch.tensor([[[base, -base], [base, -base]]])
        self.high_v.weight.data = pattern_v.repeat(in_channels, 1, 1, 1)
        
        # High-pass diagonal (HH)
        self.high_d = nn.Conv2d(in_channels, in_channels, kernel_size=2, stride=2, bias=False, groups=in_channels)
        pattern_d = torch.tensor([[[base, -base], [-base, base]]])
        self.high_d.weight.data = pattern_d.repeat(in_channels, 1, 1, 1)

    def forward(self, x):
        ll = self.low(x)
        lh = self.high_h(x)
        hl = self.high_v(x)
        hh = self.high_d(x)
        return ll, lh, hl, hh

class DWTBlock(nn.Module):
    def __init__(self):
        super(DWTBlock, self).__init__()
        self.dwt = HaarDown(normalize=True)

    def forward(self, x):
        ll1, lh1, hl1, hh1 = self.dwt(x)
        ll2, lh2, hl2, hh2 = self.dwt(ll1)
        # Group 1: LL2
        g1 = ll2
        # Group 2: cat high-freq level 2
        g2 = torch.cat([lh2, hl2, hh2], dim=1)
        # Group 3: cat high-freq level 1
        g3 = torch.cat([lh1, hl1, hh1], dim=1)

        return g1, g2, g3

class HFCFPreprocess(nn.Module):
    """Высокочастотный блок обработки и фильтрации"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(HFCFPreprocess, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
    
    def forward(self, x):
        return self.block(x)
    
class YellowBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(YellowBlock, self).__init__()
        self.yellow_block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.skip_connection = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, padding=0),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self, x):
        return self.yellow_block(x) + self.skip_connection(x)
    
class BlueBlock(nn.Module):
    def __init__(self, channels):
        super(BlueBlock, self).__init__()
        self.blue_block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.blue_block(x)

class RedBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RedBlock, self).__init__()
        self.red_block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
        )
    
    def forward(self, x):
        return self.red_block(x)
    
class UpperBranch(nn.Module):
    def __init__(self, channels):
        super(UpperBranch, self).__init__()
        self.yellow_block1 = YellowBlock(channels, channels*2)
        self.blue_block1 = BlueBlock(channels*2)
        self.blue_block2 = BlueBlock(channels*2)
        self.yellow_block2 = YellowBlock(channels*2, channels*4)
        self.blue_block3 = BlueBlock(channels*4)
        self.blue_block4 = BlueBlock(channels*4)

    def forward(self, x):
        x1 = self.yellow_block1(x)
        x2 = self.blue_block1(x1)
        x3 = self.blue_block2(x2)
        x4 = self.yellow_block2(x3)
        x5 = self.blue_block3(x4)
        out = self.blue_block4(x5)
        return out
    
class LowerBranch(nn.Module):
    def __init__(self, channels):
        super(LowerBranch, self).__init__()
        self.red_block1 = RedBlock(channels, channels*2)
        self.red_block2 = RedBlock(channels*2, channels*4)

    def forward(self, x):
        x1 = self.red_block1(x)
        out = self.red_block2(x1)
        return out
    
class HFCFUpconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(HFCFUpconvBlock, self).__init__()
        self.upconv_block = nn.Sequential(
            ConvBlock(in_channels, in_channels, stride=1),
            TConvBlock(in_channels, in_channels // 2),
            TConvBlock(in_channels // 2, in_channels // 4),
            TConvBlock(in_channels // 4, in_channels // 8),
            TConvBlock(in_channels // 8, in_channels // 16),
            TConvBlock(in_channels // 16, out_channels),
            FinalTConvBlock(out_channels, 3)
        )
    
    def forward(self, x):
        return self.upconv_block(x)
    
class HFCFBranch(nn.Module):
    def __init__(self, hfcf_concat_type='cat'):
        super(HFCFBranch, self).__init__()
        self.dwt = DWTBlock()

        self.HFCF_prep11 = HFCFPreprocess(in_channels=1, out_channels=32)
        self.HFCF_prep12 = HFCFPreprocess(in_channels=1, out_channels=32)
        self.HFCF_prep21 = HFCFPreprocess(in_channels=3, out_channels=32)
        self.HFCF_prep22 = HFCFPreprocess(in_channels=3, out_channels=32)
        self.HFCF_prep31 = HFCFPreprocess(in_channels=3, out_channels=32)
        self.HFCF_prep32 = HFCFPreprocess(in_channels=3, out_channels=32)

        self.hfcf_concat_type = hfcf_concat_type

    def forward(self, x):
        g1, g2, g3 = self.dwt(x)

        g1_p1_2_be_concat = self.HFCF_prep11(g1).to(g1.device)
        g1_p2 = self.HFCF_prep12(g1).to(g1.device)
        g2_p1_2_be_concat = self.HFCF_prep21(g2).to(g2.device)
        g2_p2 = self.HFCF_prep22(g2).to(g2.device)
        g3_p1_2_be_concat = self.HFCF_prep31(g3).to(g3.device)
        g3_p2 = self.HFCF_prep32(g3).to(g3.device)

        if (g3_p1_2_be_concat.shape[2] == 64 and g3_p2.shape[2] == 64):
            g3_p1_2_be_concat = nn.MaxPool2d(kernel_size=2, stride=2)(g3_p1_2_be_concat)
            g3_p2 = nn.MaxPool2d(kernel_size=2, stride=2)(g3_p2)

        if self.hfcf_concat_type == 'plus':
            g1_p1 = g1_p1_2_be_concat + g1_p2
            g2_p1 = g2_p1_2_be_concat + g2_p2
            g3_p1 = g3_p1_2_be_concat + g3_p2
        elif self.hfcf_concat_type == 'cat':
            g1_p1 = torch.cat([g1_p1_2_be_concat, g1_p2], dim=1)
            g2_p1 = torch.cat([g2_p1_2_be_concat, g2_p2], dim=1)
            g3_p1 = torch.cat([g3_p1_2_be_concat, g3_p2], dim=1)

        upper_branch1 = UpperBranch(channels=g1_p1.shape[1]).to(g1_p1.device)
        upper_branch2 = UpperBranch(channels=g2_p1.shape[1]).to(g2_p1.device)
        upper_branch3 = UpperBranch(channels=g3_p1.shape[1]).to(g3_p1.device)

        lower_branch1 = LowerBranch(channels=g1_p2.shape[1]).to(g1_p2.device)
        lower_branch2 = LowerBranch(channels=g2_p2.shape[1]).to(g2_p2.device)
        lower_branch3 = LowerBranch(channels=g3_p2.shape[1]).to(g3_p2.device)

        out1 = torch.cat([upper_branch1(g1_p1), lower_branch1(g1_p2)], dim=1)
        out2 = torch.cat([upper_branch2(g2_p1), lower_branch2(g2_p2)], dim=1)
        out3 = torch.cat([upper_branch3(g3_p1), lower_branch3(g3_p2)], dim=1)

        out = torch.cat([out1, out2, out3], dim=1)

        hfcf_upconv = HFCFUpconvBlock(in_channels=out.shape[1], out_channels=3).to(x.device)
        out = hfcf_upconv(out)
        return out
    
class CFRWDGenerator(nn.Module):
    def __init__(self, image_size=256, hfcf_concat_type='cat', debug=False):
        super(CFRWDGenerator, self).__init__()
        self.cfr_branch = CFRBranch(image_size=image_size, debug=debug)
        self.hfcf_branch = HFCFBranch(hfcf_concat_type=hfcf_concat_type)
        self.fuse_cfr_hfcf = nn.Conv2d(6, 3, kernel_size=1, stride=1, padding=0)
        self.alpha_logit = nn.Parameter(torch.tensor(0.5))

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        print('Weights initialized.')

    def forward(self, x):
        cfr_out = self.cfr_branch(x)
        hfcf_out = self.hfcf_branch(x)
        alpha_val = torch.sigmoid(self.alpha_logit)
        out = torch.cat([alpha_val * cfr_out, (1 - alpha_val) * hfcf_out], dim=1)
        out = self.fuse_cfr_hfcf(out)
        return out


import matplotlib.pyplot as plt
import cv2
import numpy as np
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input_array = cv2.imread('C:/Users/tiruu/Desktop/sar2opt_light/data/sen12/agri/s1/ROIs1868_summer_s1_59_p2.png', cv2.IMREAD_COLOR)  # Fix path
    if input_array is None:
        input_tensor = torch.randn(1, 1, 256, 256).to(device)  # Dummy if no image
    else:
        input_array = cv2.cvtColor(input_array, cv2.COLOR_BGR2RGB)
        input_tensor = torch.from_numpy(input_array).float().permute(2, 0, 1) / 255.0
        input_tensor = input_tensor.mean(dim=0, keepdim=True).unsqueeze(0).to(device)

    gen = CFRWDGenerator(image_size=256, hfcf_concat_type='cat').to(device)
    out = gen(input_tensor)

    import matplotlib.pyplot as plt
    plt.subplot(1, 2, 1)
    plt.imshow(input_tensor.squeeze().cpu().numpy(), cmap='gray')
    plt.title('Input SAR Image')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(out.squeeze().permute(1, 2, 0).detach().cpu().numpy())
    plt.title('Generated Optical Image')
    plt.axis('off')
    plt.show()

    print('out:', out.shape)