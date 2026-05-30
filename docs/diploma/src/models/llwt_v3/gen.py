"""LLW-Former v0.3.0 generator: Haar-Stem ConvNeXt V2.

Architecture (defense story, §16.4 with v0.3.1 simplification):

    sar (B,1,H,W)
      │
      ├── SARAdapter (sar -> 3-ch physics)              [hfgan/gen.py:6]
      │
      ▼
    HaarStemProjection (replaces ConvNeXt's 4x4 stride-4 stem)
      │   2-level fixed orthonormal Haar DWT on the 3-ch input
      │   subbands stacked as channels at H/4 (LL2 + 6 details) -> 21ch
      │   1x1 Conv 21 -> 96 ch + channel-wise LayerNorm
      ▼
    ConvNeXt V2-Tiny stages 1-4 (pretrained, stem bypassed)
      │   feature_maps: s0=(96, H/4), s1=(192, H/8), s2=(384, H/16), s3=(768, H/32)
      ▼
    PixelShuffle decoder chain (ConvUpsampleBlock reused from hfgan)
      │   skip connections from s2, s1, s0
      ▼
    cfr_logits (B,3,H,W)   # pre-tanh
      │
      ▼
    out = tanh(cfr_logits)  ∈ [-1, 1]

The hfgan-style parallel HFCFBranch + softmax fusion gate was dropped in
v0.3.1 — Haar-Stem already exposes Haar subbands directly to the ConvNeXt
encoder, so a second Haar-DWT branch is redundant. Past pattern (hfgan-9):
the fusion gate stayed dormant when one branch dominated; removing it gives
a single, defendable mechanism and saves ~3M params + ~10% step time.

`return_internals=True` returns ``(out, None, raw_feat)`` where ``raw_feat``
is the post-Haar-stem 96-ch tensor. The first slot stays None for
compatibility with the v0.1.x ``(out, raw_coeffs, raw_feat)`` triple-return
contract — main.py's training step ignores both when spkdec/pr criterions
are absent (which they are for v0.3.x).
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.huggingface_gan.gen import (
    ConvUpsampleBlock,
    HaarDown,
    SARAdapter,
)


# ---------------------------------------------------------------------------
# Haar-Stem replacement for ConvNeXt V2's 4x4 stride-4 patch embedding.
# ---------------------------------------------------------------------------


class HaarStemProjection(nn.Module):
    """Fixed 2-level orthonormal Haar DWT + learnable 1x1 projection + LayerNorm.

    Drops in where ConvNeXt V2's ``patch_embeddings`` (Conv2d(3, 96, k=4, s=4))
    normally sits.  Input contract is identical: ``(B, 3, H, W) -> (B, 96, H/4, W/4)``.

    Sub-band layout at output resolution H/4:
      - LL2  (3-ch): low-frequency content at H/4
      - LH2, HL2, HH2  (3-ch each): level-2 details at H/4
      - LH1, HL1, HH1  (3-ch each): level-1 details at H/2, average-pooled 2x to H/4

    Stacked channel order: ``[LL2, LH2, HL2, HH2, LH1↓, HL1↓, HH1↓]`` = 21 channels.
    Projected to ``out_channels`` (default 96) via a single 1x1 conv, then
    channel-wise LayerNorm to match the activation statistics the pretrained
    ConvNeXt V2 stages were trained on (post their original ImageNet-normalised
    4x4 stem).

    Why use a 1x1 conv rather than the standard 4x4 stride-4 conv:
      * the spatial mixing is already done by the Haar — using a 4x4 here would
        re-mix what we just decomposed,
      * a 1x1 lets the network learn per-subband amplitude weights independently
        (interpretable, ablatable),
      * keeps the param budget under 3k for the stem replacement (vs ~4.6k for
        the original).
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 96):
        super().__init__()
        self.haar = HaarDown(in_channels=in_channels)
        # Stacked subband count: LL2 + (LH2, HL2, HH2) + (LH1, HL1, HH1) = 7
        self.in_subbands = 7 * in_channels
        self.proj = nn.Conv2d(self.in_subbands, out_channels, kernel_size=1, bias=True)
        # Channel-wise LayerNorm AFTER projection — normalises the distribution
        # of the Haar-derived stem output to zero-mean unit-variance per channel,
        # matching what the pretrained ConvNeXt V2 stages expect from the
        # original ImageNet-normalised 4x4 stride-4 stem.
        self.norm = nn.LayerNorm(out_channels, eps=1e-6)
        # Kaiming-init the projection so each subband contributes meaningfully at step 0.
        nn.init.kaiming_normal_(self.proj.weight, mode='fan_in', nonlinearity='linear')
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W) — assumed even spatial extents (enforced upstream).
        ll1, lh1, hl1, hh1 = self.haar(x)              # each (B, 3, H/2, W/2)
        ll2, lh2, hl2, hh2 = self.haar(ll1)            # each (B, 3, H/4, W/4)
        # Down-sample level-1 details to H/4 via 2x2 average pool — Haar's LL row
        # equivalent (preserves energy interpretation; nearest would alias).
        lh1d = F.avg_pool2d(lh1, kernel_size=2, stride=2)
        hl1d = F.avg_pool2d(hl1, kernel_size=2, stride=2)
        hh1d = F.avg_pool2d(hh1, kernel_size=2, stride=2)
        # Stack along channel dim. Order matters for any per-subband ablation.
        feat = torch.cat([ll2, lh2, hl2, hh2, lh1d, hl1d, hh1d], dim=1)
        out = self.proj(feat)                          # (B, out_channels, H/4, W/4)
        # Channel-last LayerNorm to match the activation statistics expected by
        # the pretrained ConvNeXt V2 stages.
        out = self.norm(out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()
        return out


# ---------------------------------------------------------------------------
# v0.3.1 generator: Haar-Stem ConvNeXt V2 (single-branch).
# ---------------------------------------------------------------------------


class LLWv3Generator(nn.Module):
    """Single-branch Haar-Stem ConvNeXt V2-Tiny generator.

    Loads a HuggingFace ``AutoBackbone`` (ConvNeXt V2-Tiny by default), replaces
    its ``embeddings.patch_embeddings`` with :class:`HaarStemProjection`, and
    decodes via a PixelShuffle chain with skip connections from the 3 lower
    encoder stages.  No parallel wavelet branch — Haar-Stem already injects
    Haar subbands at the encoder input.

    Config (under ``cfg.model.gen``):
      * ``backbone``: HF backbone name (default ``"facebook/convnextv2-tiny-22k-224"``).
      * ``out_indices``: tuple of stage indices to extract (default ``(1,2,3,4)``).
      * ``use_sar_physics``: whether to insert SARAdapter before the Haar-stem
        (default True; the adapter learns log-amplitude + Sobel-gradient mixing
        before the otherwise-deterministic Haar transform).
    """

    def __init__(self, cfg, encoder=None):
        super().__init__()
        gen_cfg = cfg.model.gen
        backbone_name = str(getattr(gen_cfg, 'backbone', 'facebook/convnextv2-tiny-22k-224'))
        # HF ConvNeXt V2 stage_names = ['stem', 'stage1', 'stage2', 'stage3', 'stage4'].
        # We want the 4 actual stages (indices 1..4), not the stem (index 0).
        out_indices = tuple(getattr(gen_cfg, 'out_indices', (1, 2, 3, 4)))
        self.use_sar_physics = bool(getattr(gen_cfg, 'use_sar_physics', True))

        # SAR -> 3-ch physics-aware input (raw / log|sar| / sobel-grad + GELU proj).
        if self.use_sar_physics:
            self.sar_adapter = SARAdapter()
        else:
            self.sar_adapter = None
            # Fallback: 1x1 conv 1->3 channels if physics is disabled.
            self.naive_proj = nn.Conv2d(1, 3, kernel_size=1, bias=True)
            nn.init.kaiming_normal_(self.naive_proj.weight, mode='fan_in', nonlinearity='linear')
            nn.init.zeros_(self.naive_proj.bias)

        # Pretrained ConvNeXt V2-Tiny backbone.  ``out_indices`` returns the
        # 4-scale feature pyramid via ``.feature_maps`` (HF AutoBackbone API).
        if encoder is not None:
            self.encoder = encoder
        else:
            from transformers import AutoBackbone
            self.encoder = AutoBackbone.from_pretrained(
                backbone_name,
                out_indices=out_indices,
            )

        # Discover the hidden size of stage 0 from the loaded model.
        hidden_sizes = getattr(self.encoder.config, 'hidden_sizes', None)
        if hidden_sizes is None:
            hidden_sizes = (96, 192, 384, 768)
        stem_out = int(hidden_sizes[0])
        self.stem_out_channels = stem_out
        self.encoder_dims = tuple(int(d) for d in hidden_sizes[:4])

        # Replace ConvNeXt's 4x4 stride-4 stem with our Haar-Stem.  The
        # ``embeddings`` module also has a LayerNorm that normalises the
        # channel axis after the stem — we keep that untouched so the
        # downstream stages see the same statistics they were pretrained on.
        haar_stem = HaarStemProjection(in_channels=3, out_channels=stem_out)
        try:
            self.encoder.embeddings.patch_embeddings = haar_stem
        except AttributeError:
            raise RuntimeError(
                "LLWv3Generator: could not locate ``embeddings.patch_embeddings`` on "
                f"the loaded encoder (got type {type(self.encoder).__name__}).  "
                "Only HF ConvNeXt V2 backbones are supported."
            )

        # ---- decoder: PixelShuffle 2x per stage ----
        d0, d1, d2, d3 = self.encoder_dims
        self.up4 = ConvUpsampleBlock(d3, d2, 256)   # H/32 -> H/16, skip s2
        self.up3 = ConvUpsampleBlock(256, d1, 128)  # H/16 -> H/8,  skip s1
        self.up2 = ConvUpsampleBlock(128, d0,  64)  # H/8  -> H/4,  skip s0
        self.up1 = ConvUpsampleBlock( 64,  0,  32)  # H/4  -> H/2
        self.up0 = ConvUpsampleBlock( 32,  0,  16)  # H/2  -> H

        # Pre-tanh RGB logits head.
        self.final = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(16, 3, kernel_size=7),
        )
        # Zero-init the final conv so G outputs tanh(0)=0 (mid-gray) at step 0.
        # Same trick as BigGAN/StyleGAN's output-skip: avoids the random-noise
        # start where D trivially distinguishes for the first ~50 steps and
        # the adversarial signal is dominated by "G output is garbage" rather
        # than useful gradient.  Costs 0 compute, only changes initialization.
        nn.init.zeros_(self.final[1].weight)
        nn.init.zeros_(self.final[1].bias)

    # ----------------------------------------------------------------- forward

    def _decoder(self, sar3: torch.Tensor) -> torch.Tensor:
        """SAR (3-ch adapted) -> ConvNeXt V2 features -> PixelShuffle decoder -> pre-tanh RGB."""
        feats = self.encoder(pixel_values=sar3).feature_maps
        if len(feats) >= 4:
            s0, s1, s2, s3 = feats[0], feats[1], feats[2], feats[3]
        else:
            raise RuntimeError(
                f"LLWv3Generator: expected 4 feature maps from encoder, got {len(feats)}. "
                "Check cfg.model.gen.out_indices."
            )
        x = self.up4(s3, s2)
        x = self.up3(x, s1)
        x = self.up2(x, s0)
        x = self.up1(x)
        x = self.up0(x)
        return self.final(x)            # (B, 3, H, W) pre-tanh

    def forward(self, sar: torch.Tensor, return_internals: bool = False):
        # 3-channel SAR-physics representation.
        if self.sar_adapter is not None:
            sar3 = self.sar_adapter(sar)
        else:
            sar3 = self.naive_proj(sar)

        logits = self._decoder(sar3)
        out = torch.tanh(logits)

        if return_internals:
            # First slot stays None (no learnable lifting); ``raw_feat`` is the
            # post-Haar-stem 96-ch tensor for any future regulariser.
            with torch.no_grad():
                raw_feat = self.encoder.embeddings.patch_embeddings(sar3)
            return out, None, raw_feat
        return out
