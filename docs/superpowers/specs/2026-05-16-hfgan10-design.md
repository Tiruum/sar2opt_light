# hfgan-10 Design Spec

**Date:** 2026-05-16  
**Branch:** physics-aware  
**Predecessor:** hfgan-9 (PSNR 18.22 dB, LPIPS 0.303, FID 342 @ ep83 — plateau)

## Problem

FID/LPIPS plateau despite good PSNR. Root cause: no cross-modal SAR↔optical reasoning anywhere in the architecture. ChannelAdapterV2 translates SAR→pseudo-RGB, losing SAR-specific information before ConvNeXtV2 ever sees it. SPADE conditioning is local (3×3) and has failed in multiple experiments.

## Goal

Break FID/LPIPS plateau via:
1. Cross-modal SAR↔optical attention at the bottleneck
2. Global-receptive-field cross-attention at semantic decoder stages (up4/up3)

Discriminator upgrade (ProjectedDiscriminator) deferred to hfgan-11.

## Architecture Changes

### Generator (`gen.py`)

#### 1. SARBottleneckEncoder (new)
Lightweight 5-block strided conv encoder: SAR (1×256×256) → (768×8×8).
Matches ConvNeXtV2 bottleneck resolution and channel dim.
Produces K/V for cross-modal bottleneck attention.

```
Conv(1→64, s=2) → GN → GELU  # 128×128
Conv(64→128, s=2) → GN → GELU  # 64×64
Conv(128→256, s=2) → GN → GELU  # 32×32
Conv(256→512, s=2) → GN → GELU  # 16×16
Conv(512→768, s=2) → GN → GELU  # 8×8
```

#### 2. BottleneckAttention (modified)
Add cross-attention stream alongside existing self-attention.
- Self-attn: optical tokens (Q=K=V=s3)  — unchanged
- Cross-attn: Q=optical tokens, K/V=SAR bottleneck tokens + 2D pos embed
- Outputs summed with residual connection
- Zero-init on cross-attn output projection → identity at step 0

#### 3. CrossAttentionBlock (new)
Replaces SPADE at up4 and up3.
- Q: decoder features flattened → (B, HW, C_dec)
- K/V: encoder skip flattened + 2D sin-cos pos embed → (B, H'W', C_skip)
- `F.scaled_dot_product_attention` (PyTorch 2.0 flash attn)
- Linear projections: to_q(C_dec→inner), to_k(C_skip→inner), to_v(C_skip→inner), to_out(inner→C_dec)
- Residual: `out = x + to_out(attn(to_q(x), to_k(ctx), to_v(ctx)))`
- Zero-init on `to_out` weight → identity at step 0

Applied at:
- up4: decoder 256ch, context s2 384ch@16² (256 tokens) ✅
- up3: decoder 128ch, context s1 192ch@32² (1024 tokens) ✅
- up2/up1/up0: plain GroupNorm (4096+ tokens too expensive)

#### 4. ConvUpsampleBlock (modified)
Remove SPADE entirely. Replace with CrossAttentionBlock when `cond_ch > 0`, else GroupNorm.
Signature unchanged — backward-compatible via `cond_ch=0` default.

#### 5. HFGenerator (modified)
- Instantiate SARBottleneckEncoder
- Pass SAR to SARBottleneckEncoder in forward, inject into BottleneckAttention
- up4/up3: CrossAttentionBlock (cond_ch=384/192)
- up2/up1/up0: GroupNorm (cond_ch=0)
- Config flag: `model.gen.use_cross_attn: true`, `model.gen.cross_attn_heads: 4`

### Config (`config.yaml`)
```yaml
system:
  tb_version: "hfgan-10"
  weights_ckpt: null
  resume_ckpt: null

model:
  gen:
    use_spade: false
    use_cross_attn: true
    cross_attn_heads: 4

optimizer:
  fm_weight: 10.0  # match cfrwd-36 (was 5.0)
```

### Losses
No changes. GAN×1.0, FM×10.0, L1×3.0.

## Deferred

- `dis.py`: ProjectedDiscriminator (frozen DINOv2-small + linear heads) → hfgan-11

## Files Modified

| File | Change |
|------|--------|
| `gen.py` | SARBottleneckEncoder, CrossAttentionBlock, modified BottleneckAttention, modified ConvUpsampleBlock, updated HFGenerator |
| `config.yaml` | tb_version, use_cross_attn, fm_weight=10.0 |
| `changelog.md` | hfgan-10 entry |
| `tests/test_hfgan_gen.py` | Tests for new modules |

## Training Stability

- CrossAttentionBlock `to_out` zero-init → identity at step 0
- SARBottleneckEncoder cross-attn `to_out` zero-init → identity at step 0
- GroupNorm at up2/up1/up0 → no conditioning risk
- Fresh start (no warm ckpt)

## Success Criteria

- ep10: d_loss stable >0.05, val/psnr >14 dB
- ep30: visual green chroma in agricultural fields
- ep80: LPIPS <0.28, FID <280, PSNR ≥18 dB
