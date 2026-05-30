import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import SelfAttention, CBAM  # Используем относительный импорт
from .normalization import SPADE  # Используем относительный импорт
import gc
gc.collect()

class SPADEResBlock(nn.Module):
    """SPADE Residual block with optional CBAM and Self-Attention."""
    def __init__(self, fin, fout, label_nc, spade_norm_type='instance', spade_kernel_size=3, use_cbam=False, use_self_att=False, use_spectral_norm=True):
        super().__init__()
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout) # Промежуточное количество каналов

        self.norm_0 = SPADE(spade_norm_type, spade_kernel_size, fin, label_nc)
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        if use_spectral_norm:
            self.conv_0 = nn.utils.spectral_norm(self.conv_0)
        
        self.norm_1 = SPADE(spade_norm_type, spade_kernel_size, fmiddle, label_nc)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        if use_spectral_norm:
            self.conv_1 = nn.utils.spectral_norm(self.conv_1)

        if self.learned_shortcut:
            self.norm_s = SPADE(spade_norm_type, spade_kernel_size, fin, label_nc)
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)
            if use_spectral_norm:
                self.conv_s = nn.utils.spectral_norm(self.conv_s)

        self.activ = nn.LeakyReLU(0.2, True)

        self.use_cbam = use_cbam
        self.use_sa = use_self_att
        if use_cbam:
            self.cbam = CBAM(fout)
        if use_self_att:
            self.self_attention = SelfAttention(fout)

    def forward(self, x, segmap):
        x_s = self.shortcut(x, segmap)

        dx = self.activ(self.norm_0(x, segmap))
        dx = self.conv_0(dx)
        
        dx = self.activ(self.norm_1(dx, segmap))
        dx = self.conv_1(dx)

        out = x_s + dx

        if self.use_cbam:
            out = self.cbam(out)
        if self.use_sa:
            out = self.self_attention(out)
        return out

    def shortcut(self, x, segmap):
        if self.learned_shortcut:
            x_s = self.activ(self.norm_s(x, segmap))
            x_s = self.conv_s(x_s)
        else:
            x_s = x
        return x_s

class UNetGenerator(nn.Module):
    """U-Net generator with SPADE, skip connections, optional spectral norm, SelfAttention and CBAM.
    Accepts a concatenated input of image and segmentation map, and internally splits them.
    """
    def __init__(self, image_channels=1, segmap_channels=3, output_nc=3, ngf=64, n_blocks=8, 
                 spade_norm_type='instance', spade_kernel_size=3, use_spectral_norm=True):
        super().__init__()
        self.ngf = ngf
        self.image_channels = image_channels
        self.segmap_channels = segmap_channels

        # Initial block: Conv -> SPADE -> Activation
        # Conv operates on the image part
        self.initial_conv = nn.Conv2d(self.image_channels, ngf, kernel_size=7, padding=3)
        if use_spectral_norm:
            self.initial_conv = nn.utils.spectral_norm(self.initial_conv)
        # SPADE uses segmap_channels for label_nc
        self.norm_initial = SPADE(spade_norm_type, spade_kernel_size, ngf, self.segmap_channels)
        self.activ_initial = nn.LeakyReLU(0.2, True)

        # Encoder
        # enc1: ngf -> ngf*2
        self.conv_enc1 = nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1)
        if use_spectral_norm: self.conv_enc1 = nn.utils.spectral_norm(self.conv_enc1)
        self.norm_enc1 = SPADE(spade_norm_type, spade_kernel_size, ngf * 2, self.segmap_channels)
        self.activ_enc1 = nn.LeakyReLU(0.2, True)

        # enc2: ngf*2 -> ngf*4
        self.conv_enc2 = nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1)
        if use_spectral_norm: self.conv_enc2 = nn.utils.spectral_norm(self.conv_enc2)
        self.norm_enc2 = SPADE(spade_norm_type, spade_kernel_size, ngf * 4, self.segmap_channels)
        self.activ_enc2 = nn.LeakyReLU(0.2, True)

        # enc3: ngf*4 -> ngf*8
        self.conv_enc3 = nn.Conv2d(ngf * 4, ngf * 8, kernel_size=3, stride=2, padding=1)
        if use_spectral_norm: self.conv_enc3 = nn.utils.spectral_norm(self.conv_enc3)
        self.norm_enc3 = SPADE(spade_norm_type, spade_kernel_size, ngf * 8, self.segmap_channels)
        self.activ_enc3 = nn.LeakyReLU(0.2, True)

        # SPADE Residual blocks in the bottleneck
        self.resblocks = nn.ModuleList()
        for i in range(n_blocks):
            self.resblocks.append(SPADEResBlock(ngf * 8, ngf * 8, label_nc=self.segmap_channels, 
                                                spade_norm_type=spade_norm_type, spade_kernel_size=spade_kernel_size,
                                                use_cbam=(i < n_blocks // 2), 
                                                use_self_att=(i >= n_blocks // 2),
                                                use_spectral_norm=use_spectral_norm))

        # Decoder
        # Common structure for decoder blocks: Upsample -> Conv -> SPADE -> Activation
        # Common structure for post-concatenation blocks: Conv -> SPADE -> Activation -> Conv -> SPADE -> Activation

        # dec3: ngf*8 -> ngf*4. Skip connection from x3 (ngf*4)
        self.up_dec3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_dec3 = nn.Conv2d(ngf * 8, ngf * 4, kernel_size=3, padding=1)
        if use_spectral_norm: self.conv_dec3 = nn.utils.spectral_norm(self.conv_dec3)
        self.norm_dec3 = SPADE(spade_norm_type, spade_kernel_size, ngf * 4, self.segmap_channels)
        self.activ_dec3 = nn.LeakyReLU(0.2, True)
        
        self.conv_after_cat3_1 = nn.Conv2d(ngf * 8, ngf * 4, kernel_size=3, padding=1) # ngf*4 (skip) + ngf*4 (upsampled) = ngf*8
        if use_spectral_norm: self.conv_after_cat3_1 = nn.utils.spectral_norm(self.conv_after_cat3_1)
        self.norm_after_cat3_1 = SPADE(spade_norm_type, spade_kernel_size, ngf * 4, self.segmap_channels)
        self.activ_after_cat3_1 = nn.LeakyReLU(0.2, True)
        self.conv_after_cat3_2 = nn.Conv2d(ngf * 4, ngf * 4, kernel_size=3, padding=1)
        if use_spectral_norm: self.conv_after_cat3_2 = nn.utils.spectral_norm(self.conv_after_cat3_2)
        self.norm_after_cat3_2 = SPADE(spade_norm_type, spade_kernel_size, ngf * 4, self.segmap_channels)
        self.activ_after_cat3_2 = nn.LeakyReLU(0.2, True)

        # dec2: ngf*4 -> ngf*2. Skip connection from x2 (ngf*2)
        self.up_dec2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_dec2 = nn.Conv2d(ngf * 4, ngf * 2, kernel_size=3, padding=1)
        if use_spectral_norm: self.conv_dec2 = nn.utils.spectral_norm(self.conv_dec2)
        self.norm_dec2 = SPADE(spade_norm_type, spade_kernel_size, ngf * 2, self.segmap_channels)
        self.activ_dec2 = nn.LeakyReLU(0.2, True)

        self.conv_after_cat2_1 = nn.Conv2d(ngf * 4, ngf * 2, kernel_size=3, padding=1) # ngf*2 (skip) + ngf*2 (upsampled) = ngf*4
        if use_spectral_norm: self.conv_after_cat2_1 = nn.utils.spectral_norm(self.conv_after_cat2_1)
        self.norm_after_cat2_1 = SPADE(spade_norm_type, spade_kernel_size, ngf * 2, self.segmap_channels)
        self.activ_after_cat2_1 = nn.LeakyReLU(0.2, True)
        self.conv_after_cat2_2 = nn.Conv2d(ngf * 2, ngf * 2, kernel_size=3, padding=1)
        if use_spectral_norm: self.conv_after_cat2_2 = nn.utils.spectral_norm(self.conv_after_cat2_2)
        self.norm_after_cat2_2 = SPADE(spade_norm_type, spade_kernel_size, ngf * 2, self.segmap_channels)
        self.activ_after_cat2_2 = nn.LeakyReLU(0.2, True)

        # dec1: ngf*2 -> ngf. Skip connection from x1 (ngf)
        self.up_dec1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_dec1 = nn.Conv2d(ngf * 2, ngf, kernel_size=3, padding=1)
        if use_spectral_norm: self.conv_dec1 = nn.utils.spectral_norm(self.conv_dec1)
        self.norm_dec1 = SPADE(spade_norm_type, spade_kernel_size, ngf, self.segmap_channels)
        self.activ_dec1 = nn.LeakyReLU(0.2, True)

        self.conv_after_cat1_1 = nn.Conv2d(ngf * 2, ngf, kernel_size=3, padding=1) # ngf (skip) + ngf (upsampled) = ngf*2
        if use_spectral_norm: self.conv_after_cat1_1 = nn.utils.spectral_norm(self.conv_after_cat1_1)
        self.norm_after_cat1_1 = SPADE(spade_norm_type, spade_kernel_size, ngf, self.segmap_channels)
        self.activ_after_cat1_1 = nn.LeakyReLU(0.2, True)
        self.conv_after_cat1_2 = nn.Conv2d(ngf, ngf, kernel_size=3, padding=1)
        if use_spectral_norm: self.conv_after_cat1_2 = nn.utils.spectral_norm(self.conv_after_cat1_2)
        self.norm_after_cat1_2 = SPADE(spade_norm_type, spade_kernel_size, ngf, self.segmap_channels)
        self.activ_after_cat1_2 = nn.LeakyReLU(0.2, True)

        # Final convolution
        self.final_conv = nn.Conv2d(ngf, output_nc, kernel_size=7, padding=3) # ReflectionPad убран, т.к. padding в Conv2d
        self.final_activ = nn.Tanh()
        
        nn.init.normal_(self.final_conv.weight, mean=0.0, std=0.02)
        if self.final_conv.bias is not None:
            nn.init.zeros_(self.final_conv.bias)

    def forward(self, concatenated_input):
        # Split the concatenated input
        # image_part: SAR (e.g., 1 channel)
        # segmap_part: Segmentation map (e.g., 3 channels)
        image_part = concatenated_input[:, :self.image_channels, :, :]
        segmap_part = concatenated_input[:, self.image_channels : self.image_channels + self.segmap_channels, :, :]

        # Encoder path
        x1_conv = self.initial_conv(image_part) # Use image_part here
        x1 = self.activ_initial(self.norm_initial(x1_conv, segmap_part)) # Pass segmap_part to SPADE

        x2_conv = self.conv_enc1(x1)
        x2 = self.activ_enc1(self.norm_enc1(x2_conv, segmap_part))

        x3_conv = self.conv_enc2(x2)
        x3 = self.activ_enc2(self.norm_enc2(x3_conv, segmap_part))

        x4_conv = self.conv_enc3(x3)
        x4 = self.activ_enc3(self.norm_enc3(x4_conv, segmap_part))
        
        for block in self.resblocks:
            x4 = block(x4, segmap_part) # Pass segmap_part to SPADEResBlock
        
        # Decoder path
        y_dec3_upsampled = self.up_dec3(x4)
        y_dec3_conv = self.conv_dec3(y_dec3_upsampled)
        y3 = self.activ_dec3(self.norm_dec3(y_dec3_conv, segmap_part))
        
        y3_cat_input = torch.cat([y3, x3], dim=1)
        y3_cat_conv1 = self.conv_after_cat3_1(y3_cat_input)
        y3_cat_norm1 = self.norm_after_cat3_1(y3_cat_conv1, segmap_part)
        y3_cat_activ1 = self.activ_after_cat3_1(y3_cat_norm1)
        y3_cat_conv2 = self.conv_after_cat3_2(y3_cat_activ1)
        y3_processed = self.activ_after_cat3_2(self.norm_after_cat3_2(y3_cat_conv2, segmap_part))

        y_dec2_upsampled = self.up_dec2(y3_processed)
        y_dec2_conv = self.conv_dec2(y_dec2_upsampled)
        y2 = self.activ_dec2(self.norm_dec2(y_dec2_conv, segmap_part))

        y2_cat_input = torch.cat([y2, x2], dim=1)
        y2_cat_conv1 = self.conv_after_cat2_1(y2_cat_input)
        y2_cat_norm1 = self.norm_after_cat2_1(y2_cat_conv1, segmap_part)
        y2_cat_activ1 = self.activ_after_cat2_1(y2_cat_norm1)
        y2_cat_conv2 = self.conv_after_cat2_2(y2_cat_activ1)
        y2_processed = self.activ_after_cat2_2(self.norm_after_cat2_2(y2_cat_conv2, segmap_part))

        y_dec1_upsampled = self.up_dec1(y2_processed)
        y_dec1_conv = self.conv_dec1(y_dec1_upsampled)
        y1 = self.activ_dec1(self.norm_dec1(y_dec1_conv, segmap_part))

        y1_cat_input = torch.cat([y1, x1], dim=1)
        y1_cat_conv1 = self.conv_after_cat1_1(y1_cat_input)
        y1_cat_norm1 = self.norm_after_cat1_1(y1_cat_conv1, segmap_part)
        y1_cat_activ1 = self.activ_after_cat1_1(y1_cat_norm1)
        y1_cat_conv2 = self.conv_after_cat1_2(y1_cat_activ1)
        y1_processed = self.activ_after_cat1_2(self.norm_after_cat1_2(y1_cat_conv2, segmap_part))
        
        out_conv = self.final_conv(y1_processed)
        out = self.final_activ(out_conv)
        
        return out

if __name__ == "__main__":
    # Параметры для теста
    batch_size = 2
    sar_channels = 1  # SAR изображение
    segmap_input_channels = 3 # Карта сегментации
    total_input_chans = sar_channels + segmap_input_channels # Общее количество каналов на входе генератора
    
    height, width = 256, 256
    output_channels = 3 # Выходное оптическое изображение
    num_res_blocks = 8
    generator_features = 64

    # Создание случайных входных данных
    # concatenated_input тензор (SAR + segmap)
    concatenated_img_input = torch.randn(batch_size, total_input_chans, height, width)

    # Определение устройства
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Инициализация генератора
    # Указываем image_channels=sar_channels и segmap_channels=segmap_input_channels
    netG = UNetGenerator(image_channels=sar_channels, 
                         segmap_channels=segmap_input_channels, 
                         output_nc=output_channels, 
                         ngf=generator_features, 
                         n_blocks=num_res_blocks,
                         use_spectral_norm=True)
    netG.to(device)
    concatenated_img_input = concatenated_img_input.to(device)

    # Прогон через модель
    try:
        with torch.no_grad(): # Тестируем в режиме инференса
            output_img = netG(concatenated_img_input) # Передаем один тензор
        print(f"Concatenated input shape: {concatenated_img_input.shape}")
        print(f"Output image shape: {output_img.shape}")
        assert output_img.shape == (batch_size, output_channels, height, width)
        print("Shape assertion passed!")
    except Exception as e:
        print(f"An error occurred during model forward pass: {e}")
        import traceback
        traceback.print_exc()

    # Подсчет параметров
    num_params = sum(p.numel() for p in netG.parameters() if p.requires_grad)
    print(f"Number of trainable parameters in UNetGenerator: {num_params / 1e6:.2f} M")