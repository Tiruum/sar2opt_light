"""LLW-Former v0.4.0 generator: Haar-Stem ConvNeXt V2 + Inverse-Haar output head.

Architecture (Stage 1 of the v0.4.x roadmap — Inverse-Haar output only;
per-band loss weighting and Fourier-D discriminator land in v0.4.1 / v0.4.2):

    sar (B,1,H,W)
      │
      ├── SARAdapter (sar -> 3-ch physics)            [hfgan/gen.py:6]
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
      │   up4: H/32 -> H/16 (skip s2)
      │   up3: H/16 -> H/8  (skip s1)
      │   up2: H/8  -> H/4  (skip s0)
      │   up1: H/4  -> H/2  (32-ch features at H/2)
      ▼
    subband_head: 7x7 Conv 32 -> 12 (= 3 RGB x 4 Haar subbands), zero-init
      │
      ▼
    split -> (LL, LH, HL, HH) each (B, 3, H/2, W/2)
      │
      ▼
    InverseHaarUp (W^T @ subbands; pixel_shuffle 2x) -> (B, 3, H, W) pre-tanh
      │
      ▼
    out = tanh(logits)  in [-1, 1]

Rationale for the Inverse-Haar head (vs the v0.3.x up0 + 7x7 RGB conv):
  * v0.3.x produced RGB directly at full-H via a learned PixelShuffle up0
    (H/2 -> H) + 7x7 conv. The final 7x7 conv learns an *implicit*
    spatial-frequency basis with no inductive bias toward separating
    low-frequency colour from edge / texture detail.
  * The IHaar head forces G to predict each Haar subband (LL, LH, HL, HH)
    explicitly at H/2. LL carries DC + low-frequency colour; the three
    detail subbands carry horizontal / vertical / diagonal edges. Because
    the inverse is fixed (orthonormal Haar W^T, no learnable mixing),
    losses applied downstream act on a known frequency-decomposed basis.
  * Stage 2 (v0.4.1) will hook per-band losses onto the *predicted* subband
    tensor `sub` returned by `return_internals=True` — this stage just
    plumbs the head so future losses have somewhere to anchor.
  * Drops the up0 PixelShuffle block (32->16ch @ H/2->H). Net effect:
    -1 conv block in the decoder, +1 trivial einsum + pixel_shuffle, +12
    output channels on the head vs 3. Param delta is essentially zero
    (~ -1.5k params vs v0.3.x's up0+final).
  * Zero-init on subband_head keeps the v0.3.x trick: at step 0, all
    predicted subbands = 0 -> IDWT(0) = 0 -> tanh(0) = mid-gray. No random
    "garbage" start that lets D win the first ~50 steps trivially.

`return_internals=True` returns ``(out, sub, raw_feat)`` where:
  * ``sub``: (B, 3, 4, H/2, W/2) — the predicted subbands BEFORE IHaar
    (order [LL, LH, HL, HH] on dim=2). Returned for v0.4.1 per-band losses.
  * ``raw_feat``: post-Haar-stem 96-ch tensor at H/4 (unchanged from v3).
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.llwt_v45.blocks import (
    ConvUpsampleBlock,
    HaarDown,
    SARAdapter,
)


# ---------------------------------------------------------------------------
# Haar-Stem replacement for ConvNeXt V2's 4x4 stride-4 patch embedding.
# (Reused verbatim from v0.3.x — only the decoder tail changes in v0.4.0.)
# ---------------------------------------------------------------------------


class HaarStemProjection(nn.Module):
    """Fixed 2-level orthonormal Haar DWT + learnable 1x1 projection + LayerNorm.

    See ``src/models/llwt_v3/gen.py:HaarStemProjection`` for the full design
    rationale. Drop-in replacement for ConvNeXt V2's ``patch_embeddings``:
    ``(B, 3, H, W) -> (B, 96, H/4, W/4)``.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 96, token_output: bool = False):
        super().__init__()
        self.haar = HaarDown(in_channels=in_channels)
        self.in_subbands = 7 * in_channels
        self.proj = nn.Conv2d(self.in_subbands, out_channels, kernel_size=1, bias=True)
        self.norm = nn.LayerNorm(out_channels, eps=1e-6)
        # SwinV2's patch_embeddings returns ``(tokens, (H/4, W/4))``; ConvNeXtV2's
        # returns a spatial ``(B, C, H/4, W/4)`` map.  token_output=True emits the
        # Swin contract so the same Haar-stem drops into either backbone family.
        self.token_output = bool(token_output)
        nn.init.kaiming_normal_(self.proj.weight, mode='fan_in', nonlinearity='linear')
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor):
        ll1, lh1, hl1, hh1 = self.haar(x)
        ll2, lh2, hl2, hh2 = self.haar(ll1)
        lh1d = F.avg_pool2d(lh1, kernel_size=2, stride=2)
        hl1d = F.avg_pool2d(hl1, kernel_size=2, stride=2)
        hh1d = F.avg_pool2d(hh1, kernel_size=2, stride=2)
        feat = torch.cat([ll2, lh2, hl2, hh2, lh1d, hl1d, hh1d], dim=1)
        out = self.proj(feat)
        out = self.norm(out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()
        if self.token_output:
            B, C, Hh, Ww = out.shape
            tokens = out.flatten(2).transpose(1, 2)            # (B, Hh*Ww, C)
            return tokens, (Hh, Ww)
        return out


# ---------------------------------------------------------------------------
# Inverse Haar DWT — fixed, no learnable params. Exact inverse of HaarDown.
# ---------------------------------------------------------------------------


class InverseHaarUp(nn.Module):
    """Orthonormal 2D inverse Haar DWT.

    Mirror of :class:`HaarDown`. Takes the 4 subbands (LL, LH, HL, HH) each of
    shape (B, C, H/2, W/2) and reconstructs (B, C, H, W) by applying W^T to
    each 2x2 block then ``pixel_shuffle``.

    The Haar matrix W is orthonormal (each row has L2-norm 1 and rows are
    mutually orthogonal), so W^T = W^-1 and ``InverseHaarUp(*HaarDown(x))``
    reproduces ``x`` up to floating-point round-off. The ``__main__`` smoke
    at the bottom of this file verifies this identity at fp32.

    Channel-layout invariant (must match HaarDown):
      * ``pixel_unshuffle(x, 2)`` arranges each 2x2 input block as 4
        consecutive channels: [TL, TR, BL, BR].
      * Reshape ``(B, C, 4, H/2, W/2)`` so the 4-dim indexes [TL, TR, BL, BR].
      * HaarDown's einsum ``'bcihw, oi -> bcohw'`` with W gives output o in
        [LL, LH, HL, HH].
      * IHaar inverts: einsum ``'bcohw, oi -> bcihw'`` with the same W
        produces blocks[i] = sum_o W[o, i] * subband[o] = (W^T @ subbands)[i].
      * Reshape (B, C, 4, H/2, W/2) -> (B, 4C, H/2, W/2) keeps channel c's
        4 subpixels in slots [4c..4c+3], which is exactly what
        ``pixel_shuffle`` consumes to map back to (B, C, H, W).
    """

    _HAAR = torch.tensor([
        [ 1.0,  1.0,  1.0,  1.0],   # LL
        [-1.0,  1.0, -1.0,  1.0],   # LH
        [-1.0, -1.0,  1.0,  1.0],   # HL
        [ 1.0, -1.0, -1.0,  1.0],   # HH
    ], dtype=torch.float32) * 0.5

    def __init__(self, channels: int = 3):
        super().__init__()
        self.register_buffer('haar_weights', self._HAAR.clone())
        self.channels = channels

    def forward(
        self,
        ll: torch.Tensor,
        lh: torch.Tensor,
        hl: torch.Tensor,
        hh: torch.Tensor,
    ) -> torch.Tensor:
        # Stack on a new dim=2 so the order [LL, LH, HL, HH] matches the row
        # order of W used in HaarDown.
        sub = torch.stack([ll, lh, hl, hh], dim=2)            # (B, C, 4, H/2, W/2)
        B, C, _, Hh, Ww = sub.shape
        # Cast weights to match sub's dtype (handles bf16/fp16 autocast cleanly).
        w = self.haar_weights.to(dtype=sub.dtype)
        # W^T @ sub: blocks[..., i, ...] = sum_o W[o, i] * sub[..., o, ...]
        blocks = torch.einsum('bcohw, oi -> bcihw', sub, w)
        # (B, C, 4, H/2, W/2) -> (B, 4C, H/2, W/2): channel c's [TL,TR,BL,BR]
        # occupy slots [4c..4c+3], which is what pixel_shuffle expects.
        blocks = blocks.reshape(B, C * 4, Hh, Ww)
        return F.pixel_shuffle(blocks, 2)                     # (B, C, H, W)


# ---------------------------------------------------------------------------
# v0.4.0 generator: Haar-Stem ConvNeXt V2 + Inverse-Haar output head.
# ---------------------------------------------------------------------------


class LLWv4Generator(nn.Module):
    """Single-branch Haar-Stem ConvNeXt V2-Tiny generator with IHaar output head.

    Identical encoder to v0.3.x (HaarStemProjection + pretrained ConvNeXt V2
    backbone + ConvUpsampleBlock chain). The only architectural change vs v3
    is the decoder tail:

        v0.3.x:  up1 (H/4 -> H/2) -> up0 (H/2 -> H) -> 7x7 Conv 16->3 RGB
        v0.4.0:  up1 (H/4 -> H/2) -> 7x7 Conv 32->12 subbands -> InverseHaarUp

    See module docstring for the rationale.

    Config (under ``cfg.model.gen``):
      * ``backbone``: HF backbone name (default ``"facebook/convnextv2-tiny-22k-224"``).
      * ``out_indices``: tuple of stage indices to extract (default ``(1,2,3,4)``).
      * ``use_sar_physics``: whether to insert SARAdapter before the Haar-stem
        (default True; matches v3 behaviour).
    """

    def __init__(self, cfg, encoder=None):
        super().__init__()
        gen_cfg = cfg.model.gen
        backbone_name = str(getattr(gen_cfg, 'backbone', 'facebook/convnextv2-tiny-22k-224'))
        out_indices = tuple(getattr(gen_cfg, 'out_indices', (1, 2, 3, 4)))
        self.use_sar_physics = bool(getattr(gen_cfg, 'use_sar_physics', True))

        # SAR -> 3-ch physics-aware input.
        if self.use_sar_physics:
            self.sar_adapter = SARAdapter(channel2=str(getattr(gen_cfg, 'sar_channel2', 'log')))
        else:
            self.sar_adapter = None
            self.naive_proj = nn.Conv2d(1, 3, kernel_size=1, bias=True)
            nn.init.kaiming_normal_(self.naive_proj.weight, mode='fan_in', nonlinearity='linear')
            nn.init.zeros_(self.naive_proj.bias)

        # Pretrained ConvNeXt V2-Tiny backbone.
        if encoder is not None:
            self.encoder = encoder
        else:
            from transformers import AutoBackbone
            try:
                self.encoder = AutoBackbone.from_pretrained(
                    backbone_name,
                    out_indices=out_indices,
                )
            except Exception as e:
                # HF Hub network flakiness (SSL EOF / timeout in the repo_exists
                # precheck) crashes boot even when the backbone is already cached
                # — and HF_HUB_OFFLINE=1 doesn't help because repo_exists raises
                # under offline mode. Retry forcing the local cache so a transient
                # Hub outage can't kill a training run on startup.
                print(f"[gen] AutoBackbone Hub load failed "
                      f"({type(e).__name__}: {e}); retrying from local cache "
                      f"(local_files_only=True).")
                self.encoder = AutoBackbone.from_pretrained(
                    backbone_name,
                    out_indices=out_indices,
                    local_files_only=True,
                )

        hidden_sizes = getattr(self.encoder.config, 'hidden_sizes', None)
        if hidden_sizes is None:
            hidden_sizes = (96, 192, 384, 768)
        stem_out = int(hidden_sizes[0])
        self.stem_out_channels = stem_out
        self.encoder_dims = tuple(int(d) for d in hidden_sizes[:4])

        # Replace the backbone's patch-embed stem with our Haar-Stem so BOTH
        # backbone families consume the same wavelet front-end (clean one-variable
        # ablation).  ConvNeXtV2's patch_embeddings emits a spatial (B,C,H/4,W/4)
        # map; SwinV2's emits (tokens, (H/4,W/4)) -> token_output handles that.
        model_type = str(getattr(self.encoder.config, 'model_type', '')).lower()
        self.backbone_family = 'swinv2' if 'swin' in model_type else 'convnextv2'
        haar_stem = HaarStemProjection(
            in_channels=3, out_channels=stem_out,
            token_output=(self.backbone_family == 'swinv2'),
        )
        try:
            self.encoder.embeddings.patch_embeddings = haar_stem
        except AttributeError:
            raise RuntimeError(
                "LLWv4Generator: could not locate ``embeddings.patch_embeddings`` on "
                f"the loaded encoder (got type {type(self.encoder).__name__}).  "
                "Only HF ConvNeXt V2 / Swin V2 backbones are supported."
            )

        # ---- decoder: PixelShuffle 2x per stage; ends at H/2 (IHaar covers H/2->H) ----
        d0, d1, d2, d3 = self.encoder_dims
        self.up4 = ConvUpsampleBlock(d3, d2, 256)   # H/32 -> H/16, skip s2
        self.up3 = ConvUpsampleBlock(256, d1, 128)  # H/16 -> H/8,  skip s1
        self.up2 = ConvUpsampleBlock(128, d0,  64)  # H/8  -> H/4,  skip s0
        self.up1 = ConvUpsampleBlock( 64,  0,  32)  # H/4  -> H/2  (terminal feature map)

        # Subband head: 32 -> (3 RGB x 4 subbands) = 12 channels at H/2.
        # 7x7 reflection-padded conv matches v0.3.x's RGB head receptive field.
        # Zero-init: predicted subbands = 0 at step 0 -> IDWT(0) = 0 -> tanh(0)
        # = mid-gray, same warm-start trick as v0.3.x.
        self.subband_head = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(32, 12, kernel_size=7),
        )
        nn.init.zeros_(self.subband_head[1].weight)
        nn.init.zeros_(self.subband_head[1].bias)

        # Fixed inverse Haar — no learnable params, channels=3 RGB.
        self.ihaar = InverseHaarUp(channels=3)

        # v0.4.5 novelty (optional): full-res SAR-conditioned detail residual.
        # The IHaar head reconstructs H from predicted H/2 subbands; this adds a
        # DIRECT full-resolution high-frequency path conditioned on the SAR-physics
        # input (sar3) + upsampled terminal decoder features, summed into the
        # pre-tanh logits. Attacks the H/2 reconstruction bottleneck (raises the
        # representational ceiling = sharper, more photorealistic detail).
        # Zero-init last conv -> detail=0 at step 0, so the mid-gray warm-start
        # contract AND v4-checkpoint warm-loading both still hold.
        self.use_detail_residual = bool(getattr(gen_cfg, 'detail_residual', False))
        if self.use_detail_residual:
            self.detail_head = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(3 + 32, 32, kernel_size=3, bias=False),
                nn.GroupNorm(8, 32),
                nn.GELU(),
                nn.ReflectionPad2d(1),
                nn.Conv2d(32, 3, kernel_size=3),
            )
            nn.init.zeros_(self.detail_head[-1].weight)
            nn.init.zeros_(self.detail_head[-1].bias)

    # ----------------------------------------------------------------- forward

    def _decoder(self, sar3: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """SAR (3-ch adapted) -> ConvNeXt features -> decoder -> (logits, subbands).

        Returns:
          logits: (B, 3, H, W) pre-tanh RGB.
          sub:    (B, 3, 4, H/2, W/2) predicted subbands BEFORE IHaar
                  (order [LL, LH, HL, HH] on dim=2). Useful for per-band losses.
        """
        feats = self.encoder(pixel_values=sar3).feature_maps
        if len(feats) >= 4:
            s0, s1, s2, s3 = feats[0], feats[1], feats[2], feats[3]
        else:
            raise RuntimeError(
                f"LLWv4Generator: expected 4 feature maps from encoder, got {len(feats)}. "
                "Check cfg.model.gen.out_indices."
            )
        x = self.up4(s3, s2)
        x = self.up3(x, s1)
        x = self.up2(x, s0)
        x = self.up1(x)                                       # (B, 32, H/2, W/2)
        sub_flat = self.subband_head(x)                       # (B, 12, H/2, W/2)
        B, _, Hh, Ww = sub_flat.shape
        # (B, 12, H/2, W/2) -> (B, 3, 4, H/2, W/2): channel c's 4 subbands
        # occupy slots [4c..4c+3]. dim=2 order is [LL, LH, HL, HH].
        sub = sub_flat.view(B, 3, 4, Hh, Ww)
        ll, lh, hl, hh = sub[:, :, 0], sub[:, :, 1], sub[:, :, 2], sub[:, :, 3]
        logits = self.ihaar(ll, lh, hl, hh)                   # (B, 3, H, W) pre-tanh
        if self.use_detail_residual:
            # Upsample terminal features H/2 -> H, condition on SAR-physics input,
            # predict a zero-init high-freq residual added to the IHaar logits.
            x_full = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            detail = self.detail_head(torch.cat([sar3, x_full], dim=1))   # (B, 3, H, W)
            logits = logits + detail
        return logits, sub

    def forward(self, sar: torch.Tensor, return_internals: bool = False):
        if self.sar_adapter is not None:
            sar3 = self.sar_adapter(sar)
        else:
            sar3 = self.naive_proj(sar)

        logits, sub = self._decoder(sar3)
        out = torch.tanh(logits)

        if return_internals:
            with torch.no_grad():
                raw_feat = self.encoder.embeddings.patch_embeddings(sar3)
            # Triple-return contract: (out, sub, raw_feat). v0.4.1 per-band
            # losses will consume ``sub`` directly; v3-only diagnostics that
            # ignored the middle slot continue to work.
            return out, sub, raw_feat
        return out

    def encode_optical(self, img3: torch.Tensor):
        """Run the (Haar-stem) encoder on a 3-ch image in [-1, 1] -> 4-scale pyramid.

        Used by PatchNCE to compare generated-vs-real optical features in the
        generator's own representation space (no external VGG).  Backbone-agnostic:
        works for both the ConvNeXtV2 and SwinV2 arms.
        """
        feats = self.encoder(pixel_values=img3).feature_maps
        return list(feats[:4])


# ---------------------------------------------------------------------------
# Standalone smoke: verify InverseHaarUp(HaarDown(x)) == x (identity).
# Run with::  python -m src.models.llwt_v4.gen
# ---------------------------------------------------------------------------


def _smoke_identity() -> None:
    """Round-trip identity check for the orthonormal Haar pair."""
    torch.manual_seed(0)
    print("[ihaar smoke] running InverseHaarUp(HaarDown(x)) == x identity test")
    for B, C, H, W in [(1, 1, 8, 8), (2, 3, 64, 64), (4, 3, 256, 256)]:
        x = torch.randn(B, C, H, W)
        haar = HaarDown(in_channels=C)
        ihaar = InverseHaarUp(channels=C)
        ll, lh, hl, hh = haar(x)
        x_rec = ihaar(ll, lh, hl, hh)
        err = (x - x_rec).abs().max().item()
        rel = err / (x.abs().max().item() + 1e-12)
        status = 'OK' if err < 1e-5 else 'FAIL'
        print(f"  [{status}] shape=({B},{C},{H},{W})  max abs err = {err:.2e}  (rel = {rel:.2e})")
        assert err < 1e-5, f"InverseHaarUp not exact inverse of HaarDown at ({B},{C},{H},{W}): max err {err}"
    print("[ihaar smoke] PASS — orthonormal Haar inverse confirmed at fp32")


def _smoke_generator_shape() -> None:
    """Build a v4 generator with a tiny mock encoder and check output shape.

    Skips if ``transformers`` / network access is unavailable — relies on a
    dummy nn.Module that mimics the HF AutoBackbone API (feature_maps list +
    embeddings.patch_embeddings attribute).
    """
    print("[gen shape] running LLWv4Generator forward shape check with mock encoder")

    class _MockEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = type('Cfg', (), {'hidden_sizes': (96, 192, 384, 768)})()
            self.embeddings = nn.Module()
            self.embeddings.patch_embeddings = nn.Identity()   # replaced by HaarStem

        def forward(self, pixel_values: torch.Tensor):
            # In the real path HF AutoBackbone runs pixel_values (B, 3, H, W)
            # through embeddings (= our HaarStem, 4x downsample) then stages.
            # The mock skips embeddings, so we just compute the 4-scale pyramid
            # off the SAR input shape directly.
            B, _, H, W = pixel_values.shape
            feats = [
                torch.zeros(B,  96, H //  4, W //  4, device=pixel_values.device, dtype=pixel_values.dtype),
                torch.zeros(B, 192, H //  8, W //  8, device=pixel_values.device, dtype=pixel_values.dtype),
                torch.zeros(B, 384, H // 16, W // 16, device=pixel_values.device, dtype=pixel_values.dtype),
                torch.zeros(B, 768, H // 32, W // 32, device=pixel_values.device, dtype=pixel_values.dtype),
            ]
            return type('Out', (), {'feature_maps': feats})()

    from omegaconf import OmegaConf
    cfg = OmegaConf.create({
        'model': {'gen': {
            'backbone': 'mock',
            'out_indices': [1, 2, 3, 4],
            'use_sar_physics': False,
        }},
    })
    gen = LLWv4Generator(cfg=cfg, encoder=_MockEncoder())
    sar = torch.randn(2, 1, 256, 256)
    out = gen(sar)
    assert out.shape == (2, 3, 256, 256), f"expected (2,3,256,256), got {tuple(out.shape)}"
    assert out.min() >= -1.0 and out.max() <= 1.0, f"tanh out of bounds: [{out.min()},{out.max()}]"
    print(f"  [OK] forward: sar (2,1,256,256) -> out {tuple(out.shape)}  range [{out.min():.4f}, {out.max():.4f}]")

    out2, sub, raw = gen(sar, return_internals=True)
    assert sub.shape == (2, 3, 4, 128, 128), f"subbands shape mismatch: {tuple(sub.shape)}"
    print(f"  [OK] return_internals: sub {tuple(sub.shape)}  raw_feat {tuple(raw.shape)}")

    # Zero-init contract: at step 0, subband_head zero -> sub = 0 -> tanh(0) = 0.
    err0 = out.abs().max().item()
    print(f"  [info] step-0 output magnitude: max|out| = {err0:.2e}  (should be ~0 from zero-init head)")
    assert err0 < 1e-5, f"expected mid-gray (tanh(0)=0) at init, got max|out|={err0}"
    print("[gen shape] PASS — IHaar head plumbed end-to-end, zero-init contract holds")


if __name__ == '__main__':
    _smoke_identity()
    _smoke_generator_shape()
