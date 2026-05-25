import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Reusable building blocks needed by downstream models (llwt_v3 / v4 / v5).
# Restored here so the v4 generator imports don't break when this file is
# reverted to a pre-wavelet revision.  Drop-in compatible with the weights
# saved in v0.4.x checkpoints (sar_adapter.proj.weight, sar_adapter.sx/sy,
# haar.haar_weights).
# ---------------------------------------------------------------------------


class SARAdapter(nn.Module):
    """SAR-physics-aware adapter: cat(raw, log-amplitude, sobel-gradient) -> Conv 3->3 + GELU.

    Replaces naive Conv(1->3) with three physically meaningful channels so the
    ImageNet-pretrained ConvNeXtV2 stem sees a SAR-derived 3-channel input.

    Restored exactly from user's desktop hfgan/gen.py backup.  Two details that
    matter for ckpt parity:
      * ``proj`` uses ``padding_mode='reflect'`` (NOT zero-pad).
      * Final activation is ``nn.GELU()`` after the proj (NOT bare conv).
    """
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1., 0., 1.],
                                [-2., 0., 2.],
                                [-1., 0., 1.]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = sobel_x.transpose(-1, -2).contiguous()
        self.register_buffer('sx', sobel_x)
        self.register_buffer('sy', sobel_y)
        self.proj = nn.Conv2d(3, 3, kernel_size=3, padding=1, padding_mode='reflect', bias=False)
        self.act  = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        eps = 1e-6
        log_amp = torch.log(x.abs() + eps)
        # bf16-safe Sobel: fp32 then cast back.  Reflect-pad before Sobel so
        # corner pixels don't see a zero border — otherwise the 1-px boundary
        # bias accumulates and shows as a ring artifact (hfgan-17 -> hfgan-18 fix).
        with torch.amp.autocast(device_type='cuda', enabled=False):
            xf = x.float()
            xf_pad = F.pad(xf, (1, 1, 1, 1), mode='reflect')
            gx = F.conv2d(xf_pad, self.sx)
            gy = F.conv2d(xf_pad, self.sy)
            grad = torch.sqrt(gx * gx + gy * gy + eps).to(x.dtype)
        feat = torch.cat([x, log_amp, grad], dim=1)         # (B, 3, H, W)
        return self.act(self.proj(feat))


class HaarDown(nn.Module):
    """Orthonormal 2D Haar DWT — produces 4 subbands [LL, LH, HL, HH] each at H/2.

    Matches the v4 ckpt key ``haar_weights`` (4x4 orthonormal Haar matrix).
    No learnable params — ``haar_weights`` is registered as a buffer so it
    moves with the module via ``.to(device)`` but stays fixed during training.
    """
    _HAAR = torch.tensor([
        [ 1.0,  1.0,  1.0,  1.0],   # LL
        [-1.0,  1.0, -1.0,  1.0],   # LH
        [-1.0, -1.0,  1.0,  1.0],   # HL
        [ 1.0, -1.0, -1.0,  1.0],   # HH
    ], dtype=torch.float32) * 0.5

    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.register_buffer('haar_weights', self._HAAR.clone())

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        x_blocks = F.pixel_unshuffle(x, 2).view(B, C, 4, H // 2, W // 2)
        w = self.haar_weights.to(dtype=x.dtype)
        out = torch.einsum('bcihw, oi -> bcohw', x_blocks, w)
        return out[:, :, 0], out[:, :, 1], out[:, :, 2], out[:, :, 3]


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
    """PixelShuffle 2x upsample + optional skip concat + two-conv block with residual.

    Three-arg signature ``(in_ch, skip_ch, out_ch)``, matching the v0.4.x
    checkpoint state-dict layout (``up_conv`` 1x1 to ``in_ch*4`` -> PixelShuffle
    2x -> optional concat with ``skip_ch`` -> 3x3 conv stack to ``out_ch``).

    Set ``skip_ch=0`` to disable the skip concat (used by the terminal up1
    block in LLWv4Generator, which has no encoder skip at H/2).
    """
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        # PixelShuffle 2x: 1x1 to in_ch*4, then PixelShuffle/2 -> in_ch at 2x spatial.
        self.up_conv = nn.Conv2d(in_ch, in_ch * 4, kernel_size=1, bias=False)
        concat_ch = in_ch + int(skip_ch)
        groups = min(out_ch // 4, 8) if out_ch >= 4 else 1
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(1),                            # conv.0
            nn.Conv2d(concat_ch, out_ch, 3, bias=False),      # conv.1
            nn.GroupNorm(groups, out_ch),                     # conv.2
            nn.GELU(),                                         # conv.3
            nn.ReflectionPad2d(1),                            # conv.4
            nn.Conv2d(out_ch, out_ch, 3, bias=False),         # conv.5
            nn.GroupNorm(groups, out_ch),                     # conv.6
            nn.GELU(),                                         # conv.7
        )
        self.shortcut = nn.Conv2d(concat_ch, out_ch, 1, bias=False)

    def forward(self, x: torch.Tensor, skip: torch.Tensor = None) -> torch.Tensor:
        x = self.up_conv(x)
        x = F.pixel_shuffle(x, 2)
        if skip is not None and skip.numel() > 0 and skip.size(1) > 0:
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
