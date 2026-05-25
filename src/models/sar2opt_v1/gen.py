"""SAR2OPT-V1 generator: SARAdapter -> pretrained backbone -> PGCA decoder.

Architecture summary:

    sar (B, 1, H, W) ----[SARAdapter]----> sar3 (B, 3, H, W)
                                              |
                                              v
                              [HF AutoBackbone (SwinV2 / ConvNeXt V2)]
                                              |
                                              v
                       (s0, s1, s2, s3)  # 4-scale feature pyramid

    decoder (PixelShuffle U-Net with skip connections):
        s3  -> up4 -> PGCA(physics) -> up3 -> PGCA(physics) -> up2 -> PGCA(physics)
                                                                       -> up1 -> up0
                                                                              -> Conv 7x7 -> tanh
                                                                              -> out (B, 3, H, W) in [-1, 1]

The PGCA (Physics-Guided Cross-Attention) blocks at the 3 coarsest decoder
resolutions take Q = decoder feature, K/V = SAR-physics (raw + log|SAR| + Sobel
gradient) downsampled to the decoder stage's resolution and projected to the
decoder channel dim. Window-partitioned cross-attention (ws=8) keeps the
complexity linear in pixel count. This conditions optical reconstruction on
local SAR physics at every scale; it is an inline, end-to-end-trained,
physics-keyed analogue of ControlNet (Zhang ICCV 2023).

`encode_optical(opt_img)` runs the backbone on a 3-channel optical tensor
(ImageNet-normalised) and returns the 4-scale feature pyramid. Used by
PatchNCE (CUT, Park ECCV 2020) which compares fake-vs-real encoder features
during the G step. This re-uses the generator's own encoder rather than
introducing a separate VGG/ResNet -- keeps the contrastive signal in the
same representational space the generator is learning.

The constructor reads `hidden_sizes` from the loaded HF backbone config, so
swapping `cfg.model.gen.backbone` (SwinV2-Base / SwinV2-Tiny / ConvNeXtV2-Base)
does not require any decoder edits.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.huggingface_gan.gen import (
    ConvUpsampleBlock,
    SARAdapter,
    window_partition,
    window_reverse,
)


__all__ = ["SAR2OPTGenerator", "SARPhysicsPyramid", "PGCABlock"]


# ---------------------------------------------------------------------------
# SAR physics pyramid -- raw + log|SAR| + Sobel-gradient, un-mixed (no
# learnable projection). Reused at every PGCA block as K/V.
# ---------------------------------------------------------------------------


class SARPhysicsPyramid(nn.Module):
    """Compute the 3-channel SAR physics representation once per forward.

    Output channels (in order):
      0. raw SAR ``sar``
      1. log-amplitude ``log(|sar| + eps)``
      2. Sobel-gradient magnitude ``sqrt(gx^2 + gy^2)``

    Sobel is reflect-padded (matches SARAdapter's hfgan-17 -> hfgan-18 fix that
    eliminated the top-left corner ring artifact, observation 1015). All math
    runs in fp32 under autocast-disabled scope for bf16-mixed safety.
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)
        sx = torch.tensor([[-1., 0., 1.],
                           [-2., 0., 2.],
                           [-1., 0., 1.]], dtype=torch.float32).view(1, 1, 3, 3)
        sy = sx.transpose(-1, -2).contiguous()
        self.register_buffer('sx', sx)
        self.register_buffer('sy', sy)

    def forward(self, sar: torch.Tensor) -> torch.Tensor:
        log_amp = torch.log(sar.abs() + self.eps)
        with torch.amp.autocast(device_type='cuda', enabled=False):
            sf = sar.float()
            sf_pad = F.pad(sf, (1, 1, 1, 1), mode='reflect')
            gx = F.conv2d(sf_pad, self.sx)
            gy = F.conv2d(sf_pad, self.sy)
            grad = torch.sqrt(gx * gx + gy * gy + self.eps).to(sar.dtype)
        return torch.cat([sar, log_amp, grad], dim=1)


# ---------------------------------------------------------------------------
# Physics-Guided Cross-Attention block (PGCA) -- novelty axis.
# ---------------------------------------------------------------------------


class PGCABlock(nn.Module):
    """Window-partitioned cross-attention: Q = decoder feat, K/V = SAR physics.

    Decoder feature dim `dim`, physics dim 3 (raw / log / sobel). Physics is
    1x1-projected to `dim` channels at every block (separate projection so
    each decoder scale learns its own physics weighting). Both Q and KV are
    window-partitioned (window size `ws`, default 8) so attention complexity
    is linear in the number of windows. Cross-attention -> residual; then a
    2-layer GLU-ish FFN (Linear -> GELU -> Linear) with a second residual.

    LayerNorms are channel-last (NHWC -> reshape) to match standard transformer
    convention. Bf16-safe (no fp64 ops).
    """

    def __init__(self, dim: int, physics_dim: int = 3, ws: int = 8, num_heads: int = 8):
        super().__init__()
        assert dim % num_heads == 0, f"PGCA dim={dim} not divisible by num_heads={num_heads}"
        self.dim = int(dim)
        self.ws = int(ws)
        self.proj_kv = nn.Conv2d(physics_dim, dim, kernel_size=1, bias=True)
        nn.init.kaiming_normal_(self.proj_kv.weight, mode='fan_in', nonlinearity='linear')
        nn.init.zeros_(self.proj_kv.bias)
        self.norm_q = nn.LayerNorm(dim, eps=1e-6)
        self.norm_kv = nn.LayerNorm(dim, eps=1e-6)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm_ffn = nn.LayerNorm(dim, eps=1e-6)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )
        # Output gate -- starts near-zero so the block is near-identity at init
        # (the decoder learns to lean on PGCA only when it helps).
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, physics: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # Downsample physics to the decoder scale.
        if physics.shape[-1] != W or physics.shape[-2] != H:
            physics_s = F.adaptive_avg_pool2d(physics, (H, W))
        else:
            physics_s = physics
        kv_map = self.proj_kv(physics_s)                        # (B, C, H, W)

        # Window-partition both Q and KV: (B, C, H, W) -> (nW*B, ws^2, C).
        # `window_partition` from hfgan/gen.py reshapes to (nW*B, ws*ws, C) and
        # asserts H, W divisible by ws.  At our PGCA scales (16, 32, 64) with
        # ws=8 this is satisfied (2x2, 4x4, 8x8 windows respectively).
        ws = self.ws
        if H % ws != 0 or W % ws != 0:
            # Defensive: pad-replicate to next multiple of ws then crop after.
            pad_h = (ws - H % ws) % ws
            pad_w = (ws - W % ws) % ws
            x_p = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')
            kv_p = F.pad(kv_map, (0, pad_w, 0, pad_h), mode='replicate')
            Hp, Wp = H + pad_h, W + pad_w
        else:
            x_p, kv_p = x, kv_map
            Hp, Wp = H, W

        q_win = window_partition(x_p, ws)                       # (nW*B, ws^2, C)
        kv_win = window_partition(kv_p, ws)
        q_n = self.norm_q(q_win)
        kv_n = self.norm_kv(kv_win)
        attn_out, _ = self.attn(q_n, kv_n, kv_n, need_weights=False)
        attn_2d = window_reverse(attn_out, B, Hp, Wp, ws)       # (B, C, Hp, Wp)
        if Hp != H or Wp != W:
            attn_2d = attn_2d[:, :, :H, :W].contiguous()

        # Gated residual + FFN.
        x = x + self.gate * attn_2d
        x_cl = x.permute(0, 2, 3, 1).contiguous()               # NHWC for LN+Linear
        ffn_out = self.ffn(self.norm_ffn(x_cl))
        x = x + self.gate * ffn_out.permute(0, 3, 1, 2).contiguous()
        return x


# ---------------------------------------------------------------------------
# SAR2OPT-V1 generator.
# ---------------------------------------------------------------------------


class SAR2OPTGenerator(nn.Module):
    """SAR-physics adapter -> pretrained backbone -> PGCA U-Net decoder.

    The decoder channel taper is computed from the loaded backbone's
    ``hidden_sizes`` so swapping the backbone name in ``cfg`` works without
    touching this file.
    """

    # ImageNet mean/std for ``encode_optical`` (used by PatchNCE).
    _IM_MEAN = (0.485, 0.456, 0.406)
    _IM_STD = (0.229, 0.224, 0.225)

    def __init__(self, cfg, encoder: Optional[nn.Module] = None):
        super().__init__()
        gen_cfg = cfg.model.gen
        backbone_name = str(getattr(gen_cfg, 'backbone', 'microsoft/swinv2-base-patch4-window8-256'))
        out_indices = tuple(getattr(gen_cfg, 'out_indices', (1, 2, 3, 4)))
        self.use_sar_physics = bool(getattr(gen_cfg, 'use_sar_physics', True))
        self.pgca_enabled = bool(getattr(gen_cfg, 'pgca_enabled', True))
        self.pgca_window = int(getattr(gen_cfg, 'pgca_window', 8))
        self.pgca_heads = int(getattr(gen_cfg, 'pgca_heads', 8))

        # SAR -> 3-ch input expected by the pretrained backbone.
        if self.use_sar_physics:
            self.sar_adapter = SARAdapter()
        else:
            self.sar_adapter = None
            self.naive_proj = nn.Conv2d(1, 3, kernel_size=1, bias=True)
            nn.init.kaiming_normal_(self.naive_proj.weight, mode='fan_in', nonlinearity='linear')
            nn.init.zeros_(self.naive_proj.bias)

        # SAR physics pyramid (raw + log + sobel) for PGCA K/V.
        self.physics = SARPhysicsPyramid()

        # Pretrained encoder.
        if encoder is not None:
            self.encoder = encoder
        else:
            from transformers import AutoBackbone
            self.encoder = AutoBackbone.from_pretrained(
                backbone_name,
                out_indices=out_indices,
            )

        # Discover encoder feature dims dynamically.
        hidden_sizes = getattr(self.encoder.config, 'hidden_sizes', None)
        if hidden_sizes is None:
            # Sensible default for SwinV2-Base.
            hidden_sizes = (128, 256, 512, 1024)
        self.encoder_dims = tuple(int(d) for d in hidden_sizes[:4])

        # ImageNet normalisation buffers for ``encode_optical``.
        self.register_buffer('im_mean', torch.tensor(self._IM_MEAN).view(1, 3, 1, 1))
        self.register_buffer('im_std', torch.tensor(self._IM_STD).view(1, 3, 1, 1))

        # ---- PixelShuffle U-Net decoder ----
        d0, d1, d2, d3 = self.encoder_dims                                # e.g. 128/256/512/1024
        self.up4 = ConvUpsampleBlock(d3, d2, 256)                         # H/32 -> H/16, skip s2
        self.up3 = ConvUpsampleBlock(256, d1, 128)                        # H/16 -> H/8,  skip s1
        self.up2 = ConvUpsampleBlock(128, d0,  64)                        # H/8  -> H/4,  skip s0
        self.up1 = ConvUpsampleBlock( 64,  0,  32)                        # H/4  -> H/2
        self.up0 = ConvUpsampleBlock( 32,  0,  16)                        # H/2  -> H

        # PGCA blocks at the 3 coarse-mid decoder scales (16, 32, 64 at 256 in).
        if self.pgca_enabled:
            self.pgca4 = PGCABlock(256, physics_dim=3, ws=self.pgca_window, num_heads=self.pgca_heads)
            self.pgca3 = PGCABlock(128, physics_dim=3, ws=self.pgca_window, num_heads=self.pgca_heads)
            self.pgca2 = PGCABlock( 64, physics_dim=3, ws=self.pgca_window, num_heads=self.pgca_heads)
        else:
            self.pgca4 = self.pgca3 = self.pgca2 = nn.Identity()

        # Pre-tanh RGB logits head.
        # Zero-init the conv so output starts at 0 (gray, in [-1,1]). At init
        # tanh(0)=0 has gradient 1; if we initted with default Kaiming and a
        # big upstream gradient (LAB-Chroma weight 5 at init contributed 43 of
        # g_loss=51 in the first smoke), logits got pushed past +/-5 in the
        # first step and tanh saturated -- output went near-black and stayed.
        # Zero-init -> safe start; downstream losses still drive the weights.
        self.final = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(16, 3, kernel_size=7),
        )
        nn.init.zeros_(self.final[1].weight)
        nn.init.zeros_(self.final[1].bias)

    # ------------------------------------------------------------------ helpers

    def _sar_to_3ch(self, sar: torch.Tensor) -> torch.Tensor:
        if self.sar_adapter is not None:
            return self.sar_adapter(sar)
        return self.naive_proj(sar)

    def _encoder_features(self, x3ch: torch.Tensor) -> List[torch.Tensor]:
        feats = self.encoder(pixel_values=x3ch).feature_maps
        if len(feats) < 4:
            raise RuntimeError(
                f"SAR2OPTGenerator: expected 4 backbone feature maps, got {len(feats)}. "
                "Check cfg.model.gen.out_indices."
            )
        return list(feats[:4])

    def _apply_pgca(self, block: nn.Module, x: torch.Tensor, physics: torch.Tensor) -> torch.Tensor:
        if isinstance(block, nn.Identity):
            return x
        return block(x, physics)

    # ------------------------------------------------------------------ forward

    def forward(
        self,
        sar: torch.Tensor,
        return_internals: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        physics = self.physics(sar)                            # (B, 3, H, W)
        sar3 = self._sar_to_3ch(sar)
        feats = self._encoder_features(sar3)                   # [s0, s1, s2, s3]
        s0, s1, s2, s3 = feats

        x = self.up4(s3, s2)
        x = self._apply_pgca(self.pgca4, x, physics)
        x = self.up3(x, s1)
        x = self._apply_pgca(self.pgca3, x, physics)
        x = self.up2(x, s0)
        x = self._apply_pgca(self.pgca2, x, physics)
        x = self.up1(x)
        x = self.up0(x)
        # Clamp pre-tanh logits to keep gradient alive (avoids the near-black
        # saturation seen in the first smoke). |logits|<=6 -> |tanh|<=0.99998,
        # gradient ~7e-6 minimum -- enough to recover if a bad update lands.
        out = torch.tanh(self.final(x).clamp(-6.0, 6.0))

        if return_internals:
            # `feats` is the SAR-conditioned encoder pyramid (B, C, H, W) per stage.
            return out, feats, sar3
        return out

    # ------------------------------------------------------------------ encode_optical (for PatchNCE)

    def encode_optical(self, opt_img: torch.Tensor) -> List[torch.Tensor]:
        """Run the encoder on a 3-channel optical input.

        ``opt_img`` is in [-1, 1] (matches Lightning module convention).
        Output is the 4-scale ImageNet-normalised feature pyramid; PatchNCE
        consumes this list of (B, C_s, H_s, W_s) tensors.
        """
        if opt_img.shape[1] != 3:
            raise ValueError(f"encode_optical expects 3 channels, got {opt_img.shape[1]}")
        opt01 = (opt_img + 1.0) / 2.0
        x = (opt01 - self.im_mean) / self.im_std
        return self._encoder_features(x)
