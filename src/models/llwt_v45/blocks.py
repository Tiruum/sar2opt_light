"""Vendored building blocks for LLW-Former v0.4.5 (self-contained).

Copied verbatim from ``src/models/huggingface_gan/gen.py`` (SARAdapter,
HaarDown, ConvUpsampleBlock) so llwt_v45 owns its dependencies and edits here
never touch other models.  Drop-in compatible with v0.4.x checkpoint keys
(``sar_adapter.proj.weight``, ``sar_adapter.sx/sy``, ``haar.haar_weights``,
``up*.up_conv/conv/shortcut``).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["SARAdapter", "HaarDown", "ConvUpsampleBlock", "lee_despeckle"]


def lee_despeckle(x: torch.Tensor, win: int = 5, q: float = 0.05) -> torch.Tensor:
    """Adaptive Lee despeckle in the log/dB domain (additive-noise form).

    SEN1-2 SAR is already dB/log-scaled and normalized to [-1, 1], so the
    multiplicative speckle is ~additive constant-variance noise in this domain.
    Lee then reduces to an edge-preserving adaptive local filter:

        out = mean + w * (x - mean),   w = clamp(1 - sigma_n2 / var_local, 0, 1)

    ``sigma_n2`` (log-domain speckle variance) is estimated PER IMAGE as the low
    quantile of the local-variance map (homogeneous patches ~= pure speckle), so
    there is no dataset-specific tuning. w -> 1 at high-variance edges (preserve),
    w -> 0 in flat speckle (smooth). O(N), non-iterative, no params/weights.
    Call inside an autocast-disabled (fp32) scope.
    """
    pad = win // 2
    xp = F.pad(x, (pad, pad, pad, pad), mode='reflect')
    mean = F.avg_pool2d(xp, win, stride=1)
    mean_sq = F.avg_pool2d(xp * xp, win, stride=1)
    var = (mean_sq - mean * mean).clamp_min(1e-6)
    b = var.shape[0]
    sigma_n2 = torch.quantile(var.reshape(b, -1), q, dim=1).view(b, 1, 1, 1)
    w = (1.0 - sigma_n2 / var).clamp(0.0, 1.0)
    return mean + w * (x - mean)


class SARAdapter(nn.Module):
    """SAR-physics-aware adapter: cat(raw, log-amplitude, sobel-gradient) -> Conv 3->3 + GELU.

    Replaces naive Conv(1->3) with three physically meaningful channels so the
    ImageNet-pretrained ConvNeXtV2 stem sees a SAR-derived 3-channel input.

    Two details that matter for ckpt parity:
      * ``proj`` uses ``padding_mode='reflect'`` (NOT zero-pad).
      * Final activation is ``nn.GELU()`` after the proj (NOT bare conv).
    """
    def __init__(self, channel2: str = 'log'):
        super().__init__()
        self.channel2 = str(channel2)   # 'log' = log|x| (v4); 'despeckle' = log-domain Lee
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
        # Channel 2: log|x| (v4 default) OR log-domain Lee despeckle (v0.4.5).
        # 'log' kept byte-identical to v4 for warm-start parity.
        if self.channel2 == 'despeckle':
            with torch.amp.autocast(device_type='cuda', enabled=False):
                c2 = lee_despeckle(x.float()).to(x.dtype)
        else:
            c2 = torch.log(x.abs() + eps)
        # bf16-safe Sobel: fp32 then cast back.  Reflect-pad before Sobel so
        # corner pixels don't see a zero border — otherwise the 1-px boundary
        # bias accumulates and shows as a ring artifact (hfgan-17 -> hfgan-18 fix).
        with torch.amp.autocast(device_type='cuda', enabled=False):
            xf = x.float()
            xf_pad = F.pad(xf, (1, 1, 1, 1), mode='reflect')
            gx = F.conv2d(xf_pad, self.sx)
            gy = F.conv2d(xf_pad, self.sy)
            grad = torch.sqrt(gx * gx + gy * gy + eps).to(x.dtype)
        feat = torch.cat([x, c2, grad], dim=1)              # (B, 3, H, W)
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
