# models/attention.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    """Self-Attention механизм для карт признаков"""
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.in_dim = in_dim
        
        # Query, Key, Value свёртки
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        
        self.gamma = nn.Parameter(torch.zeros(1))  # обучаемый вес
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        x: вход размером (batch_size, channels, height, width)
        """
        batch_size, C, width, height = x.size()
        
        # Формируем Query, Key, Value
        proj_query = self.query_conv(x).contiguous().reshape(batch_size, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).contiguous().reshape(batch_size, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)  # батчевое матричное произведение
        attention = self.softmax(energy).contiguous()  # нормализация
        proj_value = self.value_conv(x).contiguous().reshape(batch_size, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.contiguous().reshape(batch_size, C, width, height)

        out = self.gamma * out + x  # skip connection с весом gamma
        return out

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    """CBAM Attention блок"""
    def __init__(self, in_planes, ratio=8, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_planes, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.channel_attention(x)
        out = out * self.spatial_attention(out)
        return out