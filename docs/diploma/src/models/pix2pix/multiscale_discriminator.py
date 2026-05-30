# models/multiscale_discriminator.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.pix2pix.discriminator import NLayerDiscriminator

class MultiscaleDiscriminator(nn.Module):
    """Многомасштабный дискриминатор"""
    def __init__(self, input_nc, ndf=64, n_layers=3, num_D=4):
        """
        input_nc: число каналов на входе (input + target)
        ndf: количество фильтров
        n_layers: число слоёв в каждом дискриминаторе
        num_D: сколько дискриминаторов использовать (масштабов)
        """
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D

        # Создаем несколько дискриминаторов
        self.discriminators = nn.ModuleList()
        for i in range(num_D):
            disc = NLayerDiscriminator(input_nc, ndf, n_layers)
            self.discriminators.append(disc)

    def downsample(self, input):
        """Билинейный даунсемплинг изображения в 2 раза"""
        return F.interpolate(input, scale_factor=0.5, mode='bilinear', align_corners=True)

    def forward(self, input):
        """
        input: конкатенированное SAR + Optical изображение
        returns: список выходов дискриминаторов на разных масштабах
        """
        results = []
        input_downsampled = input

        for D in self.discriminators:
            out = D(input_downsampled)
            results.append(out)
            input_downsampled = self.downsample(input_downsampled)

        return results
    
if __name__ == "__main__":
    batch_size = 8
    input_nc = 1
    output_nc = 3
    image_size = 256

    # Создаем случайный input (конкатенированное изображение)
    x = torch.randn((batch_size, input_nc + output_nc, image_size, image_size))

    # Создаем многомасштабный дискриминатор
    model = MultiscaleDiscriminator(
        input_nc=input_nc + output_nc,
        ndf=64, n_layers=3, num_D=4)

    # Прогоняем
    outputs = model(x)

    print(f"Input shape: {x.shape}")
    for i, out in enumerate(outputs):
        print(f"Output {i} shape: {out.shape}")