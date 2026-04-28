import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAdapter(nn.Module):
    """Projects 1-channel SAR to 3-channel space matching ConvNeXtV2 stem input."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class BottleneckAttention(nn.Module):
    """Global attention at the 8×8 encoder bottleneck (64 tokens).

    Uses nn.TransformerEncoderLayer with Pre-LN (norm_first=True) for
    training stability. No dropout — dataset is small.
    """
    def __init__(self, dim: int = 768, nhead: int = 8, num_layers: int = 2):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=nhead,
            dim_feedforward=dim * 2,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.pos = nn.Parameter(torch.zeros(1, 64, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape                         # (B, 768, 8, 8)
        t = x.flatten(2).transpose(1, 2)             # (B, 64, 768)
        t = self.transformer(t + self.pos)
        return t.transpose(1, 2).reshape(B, C, H, W)


class ConvUpsampleBlock(nn.Module):
    """Bilinear upsample + optional skip concat + two-conv block with residual.

    in_ch must be the channel count AFTER concatenation with the skip tensor.
    Example: ConvUpsampleBlock(768 + 384, 256) — input 768ch is upsampled 2×
    then concatenated with a 384ch skip before the convolutions.
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_ch,  out_ch, 3, bias=False),
            nn.GroupNorm(8, out_ch),
            nn.GELU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(out_ch, out_ch, 3, bias=False),
            nn.GroupNorm(8, out_ch),
            nn.GELU(),
        )
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1, bias=False)

    def forward(self, x: torch.Tensor, skip: torch.Tensor = None) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.conv(x) + self.shortcut(x)


class HFGenerator(nn.Module):
    """ConvNeXtV2 U-Net generator with bottleneck attention.

    Args:
        cfg: OmegaConf config with model.gen.{backbone, out_indices, bottleneck_*}
        encoder: optional pre-built backbone (used in tests to avoid HF download)
    """
    def __init__(self, cfg, encoder=None):
        super().__init__()
        self.channel_adapter = ChannelAdapter()

        if encoder is not None:
            self.encoder = encoder
        else:
            from transformers import AutoBackbone
            self.encoder = AutoBackbone.from_pretrained(
                cfg.model.gen.backbone,
                out_indices=tuple(cfg.model.gen.out_indices),
            )

        dim = cfg.model.gen.bottleneck_dim          # 768
        self.bottleneck = BottleneckAttention(
            dim=dim,
            nhead=cfg.model.gen.bottleneck_heads,
            num_layers=cfg.model.gen.bottleneck_layers,
        )

        # Decoder: channel counts are (post-concat, out)
        self.up4 = ConvUpsampleBlock(dim + 384, 256)   # 8→16,  concat s2
        self.up3 = ConvUpsampleBlock(256 + 192, 128)   # 16→32, concat s1
        self.up2 = ConvUpsampleBlock(128 +  96,  64)   # 32→64, concat s0
        self.up1 = ConvUpsampleBlock( 64,        32)   # 64→128
        self.up0 = ConvUpsampleBlock( 32,        16)   # 128→256

        self.head = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(16, 3, kernel_size=7),
            nn.Tanh(),
        )

    def forward(self, sar: torch.Tensor) -> torch.Tensor:
        x               = self.channel_adapter(sar)               # (B, 3, 256, 256)
        s0, s1, s2, s3  = self.encoder(pixel_values=x).feature_maps
        s3              = self.bottleneck(s3)                      # (B, 768, 8, 8)
        x               = self.up4(s3, s2)                        # (B, 256, 16, 16)
        x               = self.up3(x,  s1)                        # (B, 128, 32, 32)
        x               = self.up2(x,  s0)                        # (B,  64, 64, 64)
        x               = self.up1(x)                             # (B,  32, 128, 128)
        x               = self.up0(x)                             # (B,  16, 256, 256)
        return self.head(x)                                        # (B,   3, 256, 256)
