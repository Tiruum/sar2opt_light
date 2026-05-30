# models/discriminator.py

import torch
import torch.nn as nn
from src.models.pix2pix.attention import SelfAttention

class NLayerDiscriminator(nn.Module):
    """N-слойный дискриминатор с использованием спектральной нормализации и самовнимания."""
    def __init__(self, input_nc, ndf=64, n_layers=3):
        """
        input_nc: число каналов на входе (input + target)
        ndf: число фильтров в первом слое
        n_layers: глубина дискриминатора
        """

        super(NLayerDiscriminator, self).__init__()

        kw = 4  # размер ядра свертки
        padw = 1  # паддинг для сохранения размера
        sequence = [
            nn.utils.spectral_norm(
                nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw)
            ),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1

        # Строим слои глубже
        self.attention_layer_idx = 2
        self.attention_dim = None

        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.utils.spectral_norm(
                    nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                              kernel_size=kw, stride=2, padding=padw)
                ),
                nn.InstanceNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
            if n == self.attention_layer_idx:
                self.attention_dim = ndf * nf_mult

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.utils.spectral_norm(
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=1, padding=padw)
            ),
            nn.InstanceNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.utils.spectral_norm(
                nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
            )
        ]

        self.model = nn.Sequential(*sequence)

        if self.attention_dim is not None:
            self.attention = SelfAttention(self.attention_dim)
        else:
            self.attention = None

    def forward(self, input):
        x = input
        for idx, layer in enumerate(self.model):
            x = layer(x)
            if self.attention is not None and idx == (self.attention_layer_idx * 3 - 1):
                x = self.attention(x)
        return x