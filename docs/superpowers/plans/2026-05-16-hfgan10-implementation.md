# hfgan-10 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace SPADE with cross-attention in hfgan decoder (up4/up3) and add SAR↔optical cross-modal attention at the bottleneck.

**Architecture:** SARBottleneckEncoder compresses raw SAR to 768ch@8² for cross-modal bottleneck attention. CrossAttentionBlock (Q=decoder, K/V=encoder skip + 2D pos embed) replaces SPADENorm at up4/up3. up2/up1/up0 use plain GroupNorm. Discriminator changes deferred.

**Tech Stack:** PyTorch 2.0+ (`F.scaled_dot_product_attention`), OmegaConf, Lightning, existing `build_2d_sincos_pos_embed` in gen.py

---

## File Map

| File | Action | What changes |
|------|--------|-------------|
| `src/models/huggingface_gan/gen.py` | Modify | Remove SPADENorm; add SARBottleneckEncoder, CrossAttentionBlock; modify BottleneckAttention, ConvUpsampleBlock, HFGenerator |
| `src/models/huggingface_gan/config.yaml` | Modify | tb_version=hfgan-10, use_cross_attn=true, fm_weight=10.0, use_spade=false |
| `changelog.md` | Modify | Add hfgan-10 entry |
| `tests/test_hfgan_gen.py` | Modify | Add tests for new modules; update existing for new signatures |

---

## Task 1: SARBottleneckEncoder

**Files:**
- Modify: `src/models/huggingface_gan/gen.py`
- Test: `tests/test_hfgan_gen.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_hfgan_gen.py`:

```python
# ---------------------------------------------------------------------------
# SARBottleneckEncoder
# ---------------------------------------------------------------------------

def test_sar_bottleneck_encoder_shape():
    from src.models.huggingface_gan.gen import SARBottleneckEncoder
    enc = SARBottleneckEncoder(out_dim=768)
    x = torch.randn(2, 1, 256, 256)
    out = enc(x)
    assert out.shape == (2, 768, 8, 8), f"got {out.shape}"

def test_sar_bottleneck_encoder_gradients():
    from src.models.huggingface_gan.gen import SARBottleneckEncoder
    enc = SARBottleneckEncoder(out_dim=768)
    x = torch.randn(2, 1, 256, 256)
    out = enc(x)
    out.mean().backward()
    assert all(p.grad is not None for p in enc.parameters() if p.requires_grad)
```

- [ ] **Step 2: Run tests to verify they fail**

```
.venv/Scripts/python -m pytest tests/test_hfgan_gen.py::test_sar_bottleneck_encoder_shape tests/test_hfgan_gen.py::test_sar_bottleneck_encoder_gradients -v
```
Expected: ImportError or AttributeError — `SARBottleneckEncoder` not defined.

- [ ] **Step 3: Add SARBottleneckEncoder to gen.py**

Insert after the `SARSkipPyramid` class (before `HFGenerator`):

```python
class SARBottleneckEncoder(nn.Module):
    """Compresses raw SAR (1×256²) to bottleneck resolution (out_dim×8²) for
    cross-modal attention at the ConvNeXtV2 bottleneck."""
    def __init__(self, out_dim: int = 768):
        super().__init__()
        def _block(ic, oc):
            return nn.Sequential(
                nn.Conv2d(ic, oc, 3, stride=2, padding=1, bias=False),
                nn.GroupNorm(min(oc // 4, 8), oc),
                nn.GELU(),
            )
        self.encoder = nn.Sequential(
            _block(1,   64),       # 256→128
            _block(64,  128),      # 128→64
            _block(128, 256),      # 64→32
            _block(256, 512),      # 32→16
            _block(512, out_dim),  # 16→8
        )

    def forward(self, sar: torch.Tensor) -> torch.Tensor:
        return self.encoder(sar)   # (B, out_dim, 8, 8)
```

- [ ] **Step 4: Run tests to verify they pass**

```
.venv/Scripts/python -m pytest tests/test_hfgan_gen.py::test_sar_bottleneck_encoder_shape tests/test_hfgan_gen.py::test_sar_bottleneck_encoder_gradients -v
```
Expected: PASS

---

## Task 2: CrossAttentionBlock

**Files:**
- Modify: `src/models/huggingface_gan/gen.py`
- Test: `tests/test_hfgan_gen.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_hfgan_gen.py`:

```python
# ---------------------------------------------------------------------------
# CrossAttentionBlock
# ---------------------------------------------------------------------------

def test_cross_attn_output_shape():
    from src.models.huggingface_gan.gen import CrossAttentionBlock
    blk = CrossAttentionBlock(query_dim=256, context_dim=384, nhead=4)
    x   = torch.randn(2, 256, 16, 16)   # decoder at up4
    ctx = torch.randn(2, 384, 16, 16)   # s2 encoder skip
    out = blk(x, ctx)
    assert out.shape == (2, 256, 16, 16)

def test_cross_attn_identity_at_init():
    """to_out zero-init → output == input at step 0."""
    from src.models.huggingface_gan.gen import CrossAttentionBlock
    blk = CrossAttentionBlock(query_dim=256, context_dim=384, nhead=4)
    blk.eval()
    x   = torch.randn(2, 256, 16, 16)
    ctx = torch.randn(2, 384, 16, 16)
    with torch.no_grad():
        out = blk(x, ctx)
    assert torch.allclose(out, x, atol=1e-5), "Expected identity at init"

def test_cross_attn_gradients_flow():
    from src.models.huggingface_gan.gen import CrossAttentionBlock
    blk = CrossAttentionBlock(query_dim=128, context_dim=192, nhead=4)
    x   = torch.randn(2, 128, 32, 32, requires_grad=True)
    ctx = torch.randn(2, 192, 32, 32)
    out = blk(x, ctx)
    out.mean().backward()
    assert x.grad is not None
```

- [ ] **Step 2: Run tests to verify they fail**

```
.venv/Scripts/python -m pytest tests/test_hfgan_gen.py::test_cross_attn_output_shape tests/test_hfgan_gen.py::test_cross_attn_identity_at_init tests/test_hfgan_gen.py::test_cross_attn_gradients_flow -v
```
Expected: ImportError — `CrossAttentionBlock` not defined.

- [ ] **Step 3: Add CrossAttentionBlock to gen.py**

Insert after `SARBottleneckEncoder` (before `HFGenerator`). Note: `build_2d_sincos_pos_embed` is already defined in gen.py.

```python
class CrossAttentionBlock(nn.Module):
    """Cross-attention: Q=decoder features, K/V=encoder/SAR context.

    Residual: out = x + to_out(attention(Q, K, V)).
    to_out zero-initialized → pure identity at step 0; learns from there.
    K/V receive 2D sin-cos positional encoding for spatial awareness.
    Uses F.scaled_dot_product_attention (flash attention when available).
    """
    def __init__(self, query_dim: int, context_dim: int, nhead: int = 4):
        super().__init__()
        assert query_dim % nhead == 0, f"query_dim {query_dim} must be divisible by nhead {nhead}"
        self.nhead    = nhead
        self.head_dim = query_dim // nhead
        inner_dim     = query_dim          # inner == query_dim for residual convenience
        self.to_q   = nn.Linear(query_dim,   inner_dim, bias=False)
        self.to_k   = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v   = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim,   query_dim, bias=False)
        self.norm_q = nn.LayerNorm(query_dim)
        nn.init.zeros_(self.to_out.weight)   # identity at init

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B, C, H, W   = x.shape
        Bc, Cc, Hc, Wc = context.shape

        q   = x.flatten(2).transpose(1, 2)          # (B, HW, C)
        ctx = context.flatten(2).transpose(1, 2)     # (B, HcWc, Cc)

        # 2D sin-cos pos embed on K/V (spatial awareness)
        if Hc == Wc and Cc % 4 == 0:
            pos = build_2d_sincos_pos_embed(Hc, Cc).to(x.device)
            ctx = ctx + pos                          # (B, HcWc, Cc)

        q   = self.norm_q(q)
        Q   = self.to_q(q).reshape(B, H*W,   self.nhead, self.head_dim).transpose(1, 2)
        K   = self.to_k(ctx).reshape(B, Hc*Wc, self.nhead, self.head_dim).transpose(1, 2)
        V   = self.to_v(ctx).reshape(B, Hc*Wc, self.nhead, self.head_dim).transpose(1, 2)

        attn = F.scaled_dot_product_attention(Q, K, V)     # flash attn if available
        attn = attn.transpose(1, 2).reshape(B, H*W, self.nhead * self.head_dim)

        out = self.to_out(attn).transpose(1, 2).reshape(B, C, H, W)
        return x + out                               # residual
```

- [ ] **Step 4: Run tests to verify they pass**

```
.venv/Scripts/python -m pytest tests/test_hfgan_gen.py::test_cross_attn_output_shape tests/test_hfgan_gen.py::test_cross_attn_identity_at_init tests/test_hfgan_gen.py::test_cross_attn_gradients_flow -v
```
Expected: PASS

---

## Task 3: Modified BottleneckAttention

**Files:**
- Modify: `src/models/huggingface_gan/gen.py` (BottleneckAttention class)
- Test: `tests/test_hfgan_gen.py`

- [ ] **Step 1: Write failing test**

Add to `tests/test_hfgan_gen.py`:

```python
def test_bottleneck_with_sar_cross_attn_shape():
    from src.models.huggingface_gan.gen import BottleneckAttention
    attn = BottleneckAttention(dim=768, nhead=8, num_layers=2,
                               sar_cross_attn=True, cross_attn_heads=4)
    x       = torch.randn(2, 768, 8, 8)
    sar_bot = torch.randn(2, 768, 8, 8)
    out     = attn(x, sar_feat=sar_bot)
    assert out.shape == (2, 768, 8, 8)

def test_bottleneck_sar_cross_attn_identity_at_init():
    from src.models.huggingface_gan.gen import BottleneckAttention
    attn = BottleneckAttention(dim=768, nhead=8, num_layers=2,
                               sar_cross_attn=True, cross_attn_heads=4)
    attn.eval()
    x       = torch.randn(2, 768, 8, 8)
    sar_bot = torch.randn(2, 768, 8, 8)
    with torch.no_grad():
        out_with    = attn(x, sar_feat=sar_bot)
        out_without = attn(x)            # no SAR feat
    # cross-attn identity at init → both should match
    assert torch.allclose(out_with, out_without, atol=1e-5)
```

- [ ] **Step 2: Run tests to verify they fail**

```
.venv/Scripts/python -m pytest tests/test_hfgan_gen.py::test_bottleneck_with_sar_cross_attn_shape tests/test_hfgan_gen.py::test_bottleneck_sar_cross_attn_identity_at_init -v
```
Expected: FAIL — BottleneckAttention does not accept `sar_cross_attn` kwarg.

- [ ] **Step 3: Replace BottleneckAttention in gen.py**

Replace the entire `BottleneckAttention` class:

```python
class BottleneckAttention(nn.Module):
    """Global self-attention at the 8×8 encoder bottleneck (64 tokens).
    Optionally adds a cross-attention stream where optical tokens (Q) attend
    to SAR bottleneck features (K/V) from SARBottleneckEncoder.
    Cross-attn to_out is zero-initialized → identity at step 0."""
    def __init__(self, dim: int = 768, nhead: int = 8, num_layers: int = 2,
                 mlp_ratio: int = 4, grid_size: int = 8,
                 sar_cross_attn: bool = False, cross_attn_heads: int = 4):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=nhead, dim_feedforward=dim * mlp_ratio,
            dropout=0.0, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.register_buffer('pos', build_2d_sincos_pos_embed(grid_size, dim))
        self.sar_cross_attn = sar_cross_attn
        if sar_cross_attn:
            self.cross_attn = CrossAttentionBlock(
                query_dim=dim, context_dim=dim, nhead=cross_attn_heads,
            )

    def forward(self, x: torch.Tensor,
                sar_feat: torch.Tensor = None) -> torch.Tensor:
        B, C, H, W = x.shape
        t = x.flatten(2).transpose(1, 2) + self.pos   # (B, 64, 768)
        t = self.transformer(t)
        t = t.transpose(1, 2).reshape(B, C, H, W)
        if self.sar_cross_attn and sar_feat is not None:
            t = self.cross_attn(t, sar_feat)
        return t
```

- [ ] **Step 4: Run all bottleneck tests**

```
.venv/Scripts/python -m pytest tests/test_hfgan_gen.py -k "bottleneck" -v
```
Expected: all PASS (existing + new).

---

## Task 4: Rewrite ConvUpsampleBlock (remove SPADE, add CrossAttentionBlock)

**Files:**
- Modify: `src/models/huggingface_gan/gen.py`
- Test: `tests/test_hfgan_gen.py`

- [ ] **Step 1: Write failing test for cross-attn path**

Add to `tests/test_hfgan_gen.py`:

```python
def test_upsample_block_with_cross_attn():
    from src.models.huggingface_gan.gen import ConvUpsampleBlock
    block = ConvUpsampleBlock(768 + 384, 256, cond_ch=384, cross_attn_heads=4)
    x    = torch.randn(2, 768, 8, 8)
    skip = torch.randn(2, 384, 16, 16)
    out  = block(x, skip, cond=skip)
    assert out.shape == (2, 256, 16, 16)

def test_upsample_block_cross_attn_identity_at_init():
    """CrossAttentionBlock to_out zero-init → cross-attn path ≈ GroupNorm path at step 0."""
    from src.models.huggingface_gan.gen import ConvUpsampleBlock
    torch.manual_seed(0)
    block_ca = ConvUpsampleBlock(256, 128, cond_ch=192, cross_attn_heads=4)
    block_gn = ConvUpsampleBlock(256, 128, cond_ch=0)
    # Copy all non-attn weights so only attn differs
    block_ca.conv1.weight.data = block_gn.conv1.weight.data.clone()
    block_ca.conv2.weight.data = block_gn.conv2.weight.data.clone()
    block_ca.shortcut.weight.data = block_gn.shortcut.weight.data.clone()
    block_ca.eval(); block_gn.eval()
    x   = torch.randn(2, 256, 16, 16)
    ctx = torch.randn(2, 192, 32, 32)
    with torch.no_grad():
        out_ca = block_ca(x, cond=ctx)
        out_gn = block_gn(x)
    assert out_ca.shape == out_gn.shape == (2, 128, 32, 32)
```

- [ ] **Step 2: Run tests to verify they fail**

```
.venv/Scripts/python -m pytest tests/test_hfgan_gen.py::test_upsample_block_with_cross_attn tests/test_hfgan_gen.py::test_upsample_block_cross_attn_identity_at_init -v
```
Expected: FAIL — `ConvUpsampleBlock` doesn't accept `cross_attn_heads`.

- [ ] **Step 3: Remove SPADENorm and rewrite ConvUpsampleBlock in gen.py**

Delete the entire `SPADENorm` class from gen.py.

Replace the entire `ConvUpsampleBlock` class:

```python
class ConvUpsampleBlock(nn.Module):
    """Bilinear upsample + optional skip concat + two-conv block with residual.

    When cond_ch > 0, uses CrossAttentionBlock (Q=features, K/V=cond) instead
    of GroupNorm after each conv. cond_ch=0 → plain GroupNorm (no conditioning).
    """
    def __init__(self, in_ch: int, out_ch: int, cond_ch: int = 0,
                 cross_attn_heads: int = 4):
        super().__init__()
        num_groups      = min(out_ch // 4, 8)
        self.pad1       = nn.ReflectionPad2d(1)
        self.conv1      = nn.Conv2d(in_ch,  out_ch, 3, bias=False)
        self.pad2       = nn.ReflectionPad2d(1)
        self.conv2      = nn.Conv2d(out_ch, out_ch, 3, bias=False)
        self.shortcut   = nn.Conv2d(in_ch,  out_ch, 1, bias=False)
        self.act        = nn.GELU()
        self.use_cross_attn = cond_ch > 0
        if self.use_cross_attn:
            self.attn1 = CrossAttentionBlock(out_ch, cond_ch, nhead=cross_attn_heads)
            self.attn2 = CrossAttentionBlock(out_ch, cond_ch, nhead=cross_attn_heads)
        else:
            self.norm1 = nn.GroupNorm(num_groups, out_ch)
            self.norm2 = nn.GroupNorm(num_groups, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor = None,
                cond: torch.Tensor = None) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        h = self.conv1(self.pad1(x))
        h = self.act(self.attn1(h, cond) if self.use_cross_attn else self.norm1(h))
        h = self.conv2(self.pad2(h))
        h = self.act(self.attn2(h, cond) if self.use_cross_attn else self.norm2(h))
        return h + self.shortcut(x)
```

- [ ] **Step 4: Run all ConvUpsampleBlock tests**

```
.venv/Scripts/python -m pytest tests/test_hfgan_gen.py -k "upsample" -v
```
Expected: all PASS (existing + new).

---

## Task 5: Update HFGenerator

**Files:**
- Modify: `src/models/huggingface_gan/gen.py` (HFGenerator class)
- Test: `tests/test_hfgan_gen.py`

- [ ] **Step 1: Write failing integration test**

Add to `tests/test_hfgan_gen.py`:

```python
def test_generator_cross_attn_enabled(device):
    from src.models.huggingface_gan.gen import HFGenerator
    from omegaconf import OmegaConf
    cfg = OmegaConf.load('src/models/huggingface_gan/config.yaml')
    gen = HFGenerator(cfg, encoder=MockBackbone()).to(device).eval()
    sar = torch.randn(2, 1, 256, 256, device=device)
    with torch.no_grad():
        out = gen(sar)
    assert out.shape == (2, 3, 256, 256)
    assert out.min() >= -1.0 - 1e-5
    assert out.max() <=  1.0 + 1e-5

def test_generator_sar_bottleneck_enc_exists(device):
    from src.models.huggingface_gan.gen import HFGenerator
    from omegaconf import OmegaConf
    cfg = OmegaConf.load('src/models/huggingface_gan/config.yaml')
    gen = HFGenerator(cfg, encoder=MockBackbone()).to(device)
    assert hasattr(gen, 'sar_bottleneck_enc') and gen.sar_bottleneck_enc is not None
```

- [ ] **Step 2: Run tests to verify they fail**

```
.venv/Scripts/python -m pytest tests/test_hfgan_gen.py::test_generator_cross_attn_enabled tests/test_hfgan_gen.py::test_generator_sar_bottleneck_enc_exists -v
```
Expected: FAIL or AttributeError.

- [ ] **Step 3: Rewrite HFGenerator in gen.py**

Replace the entire `HFGenerator` class:

```python
class HFGenerator(nn.Module):
    """ConvNeXtV2 U-Net generator with bottleneck cross-modal attention
    (SAR↔optical) and decoder cross-attention at up4/up3.

    Args:
        cfg: OmegaConf config with model.gen.{backbone, out_indices,
             bottleneck_*, sar_skip_channels, use_cross_attn, cross_attn_heads}
        encoder: optional pre-built backbone (used in tests to avoid HF download)
    """
    def __init__(self, cfg, encoder=None):
        super().__init__()
        self.channel_adapter = ChannelAdapterV2()

        if encoder is not None:
            self.encoder = encoder
        else:
            from transformers import AutoBackbone
            self.encoder = AutoBackbone.from_pretrained(
                cfg.model.gen.backbone,
                out_indices=tuple(cfg.model.gen.out_indices),
            )

        dim              = cfg.model.gen.bottleneck_dim          # 768
        use_cross_attn   = getattr(cfg.model.gen, 'use_cross_attn', False)
        cross_attn_heads = getattr(cfg.model.gen, 'cross_attn_heads', 4)
        self.use_cross_attn = use_cross_attn

        self.bottleneck = BottleneckAttention(
            dim=dim,
            nhead=cfg.model.gen.bottleneck_heads,
            num_layers=cfg.model.gen.bottleneck_layers,
            sar_cross_attn=use_cross_attn,
            cross_attn_heads=cross_attn_heads,
        )

        sk = cfg.model.gen.sar_skip_channels                     # [32, 16]
        self.sar_skip_pyramid = SARSkipPyramid(ch_128=sk[0], ch_256=sk[1])

        # SAR bottleneck encoder for cross-modal attention
        self.sar_bottleneck_enc = (
            SARBottleneckEncoder(out_dim=dim) if use_cross_attn else None
        )

        # Decoder: cross-attn at up4/up3, GroupNorm at up2/up1/up0
        ca4 = 384 if use_cross_attn else 0
        ca3 = 192 if use_cross_attn else 0

        self.up4 = ConvUpsampleBlock(dim + 384,   256, cond_ch=ca4, cross_attn_heads=cross_attn_heads)
        self.up3 = ConvUpsampleBlock(256 + 192,   128, cond_ch=ca3, cross_attn_heads=cross_attn_heads)
        self.up2 = ConvUpsampleBlock(128 +  96,    64)
        self.up1 = ConvUpsampleBlock( 64 + sk[0],  32)
        self.up0 = ConvUpsampleBlock( 32 + sk[1],  16)

        self.head = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(16, 3, kernel_size=7),
            nn.Tanh(),
        )

    def forward(self, sar: torch.Tensor) -> torch.Tensor:
        x              = self.channel_adapter(sar)
        s0, s1, s2, s3 = self.encoder(pixel_values=x).feature_maps

        if self.use_cross_attn and self.sar_bottleneck_enc is not None:
            sar_bot = self.sar_bottleneck_enc(sar)
            s3      = self.bottleneck(s3, sar_feat=sar_bot)
        else:
            s3      = self.bottleneck(s3)

        f128, f256 = self.sar_skip_pyramid(sar)

        x = self.up4(s3,  s2,   cond=s2)    # cross-attn (if enabled)
        x = self.up3(x,   s1,   cond=s1)    # cross-attn (if enabled)
        x = self.up2(x,   s0)               # GroupNorm
        x = self.up1(x,   f128)             # GroupNorm
        x = self.up0(x,   f256)             # GroupNorm
        return self.head(x)
```

Also update the `__main__` block at the bottom of gen.py to test with `use_cross_attn`:

```python
if __name__ == '__main__':
    from omegaconf import OmegaConf
    from types import SimpleNamespace

    cfg = OmegaConf.load('src/models/huggingface_gan/config.yaml')

    class _MockEnc(nn.Module):
        def forward(self, pixel_values=None, **_):
            B = pixel_values.shape[0]
            return SimpleNamespace(feature_maps=(
                torch.zeros(B,  96, 64, 64),
                torch.zeros(B, 192, 32, 32),
                torch.zeros(B, 384, 16, 16),
                torch.zeros(B, 768,  8,  8),
            ))

    g = HFGenerator(cfg, encoder=_MockEnc())
    x = torch.zeros(1, 1, 256, 256)
    out = g(x)
    print(f'input:  {x.shape}')
    print(f'output: {out.shape}')
    assert out.shape == (1, 3, 256, 256)
    cross_attn_enabled = g.use_cross_attn
    print(f'cross_attn_enabled: {cross_attn_enabled}')
    print('Architecture OK.')
```

- [ ] **Step 4: Run full generator test suite**

```
.venv/Scripts/python -m pytest tests/test_hfgan_gen.py -v
```
Expected: all PASS.

- [ ] **Step 5: Run quick architecture check**

```
.venv/Scripts/python -m src.models.huggingface_gan.gen
```
Expected output includes: `output: torch.Size([1, 3, 256, 256])` and `cross_attn_enabled: True`

---

## Task 6: Update config.yaml and changelog.md

**Files:**
- Modify: `src/models/huggingface_gan/config.yaml`
- Modify: `changelog.md`

- [ ] **Step 1: Update config.yaml**

In `config.yaml`, make these changes:

Under `model.gen`, change/add:
```yaml
model:
  gen:
    use_spade:         false    # was true — SPADE removed
    use_cross_attn:    true     # new: cross-attn at bottleneck + up4/up3
    cross_attn_heads:  4
```

Under `loss`, change:
```yaml
loss:
  fm_weight:         10.0    # was 5.0 — match cfrwd-36 best run
```

Under `system`, change:
```yaml
system:
  tb_version:    "hfgan-10"  # was "hfgan-9"
  weights_ckpt:  null
  resume_ckpt:   null
```

- [ ] **Step 2: Add hfgan-10 entry to changelog.md**

Add at the top of the changelog entries:

```markdown
## hfgan-10 (in progress)

**Architecture:** Cross-modal SAR↔optical attention at ConvNeXtV2 bottleneck + cross-attention decoder (up4/up3). First explicit cross-modal reasoning in any sar2opt_light experiment.

**Key changes vs hfgan-9:**
- `SARBottleneckEncoder`: 5-block strided CNN compresses SAR (1×256²) to (768×8²); feeds K/V for bottleneck cross-attn
- `CrossAttentionBlock`: Q=decoder, K/V=encoder skip + 2D sin-cos pos embed; `to_out` zero-init → identity at step 0
- `BottleneckAttention`: adds optional cross-attn stream (optical Q, SAR K/V) after self-attn
- `ConvUpsampleBlock`: SPADE removed; cross-attn at up4/up3 (≤32² tokens), GroupNorm at up2/up1/up0
- Loss: fm_weight 5.0 → 10.0 (matches cfrwd-36 best run)
- Fresh start, no warm checkpoint

**Deferred:** ProjectedDiscriminator (DINOv2) → hfgan-11
```

- [ ] **Step 3: Run full test suite to verify nothing broke**

```
.venv/Scripts/python -m pytest tests/test_hfgan_gen.py tests/test_hfgan_main.py -v
```
Expected: all PASS.

---

## Task 7: Final commit

- [ ] **Step 1: Stage files**

```bash
git add src/models/huggingface_gan/gen.py \
        src/models/huggingface_gan/config.yaml \
        tests/test_hfgan_gen.py \
        changelog.md \
        docs/superpowers/specs/2026-05-16-hfgan10-design.md \
        docs/superpowers/plans/2026-05-16-hfgan10-implementation.md
```

- [ ] **Step 2: Verify staged diff is correct**

```bash
git diff --cached --stat
```
Expected: gen.py, config.yaml, test_hfgan_gen.py, changelog.md changed.

- [ ] **Step 3: Commit**

```bash
git commit -m "feat(hfgan-10): cross-modal SAR↔optical attention at bottleneck and decoder

Replace SPADE with CrossAttentionBlock at up4/up3 (global receptive field,
to_out zero-init for training stability). Add SARBottleneckEncoder that
compresses raw SAR to 768ch@8x8 for cross-modal attention at ConvNeXtV2
bottleneck — first explicit SAR-optical cross-modal coupling in any
sar2opt_light experiment. fm_weight 5→10 matching cfrwd-36.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Self-Review Checklist

- [x] `SARBottleneckEncoder` produces (B, 768, 8, 8) — matches bottleneck dim
- [x] `CrossAttentionBlock` query_dim must be divisible by nhead — asserted in __init__
- [x] `to_out` zero-init confirmed in CrossAttentionBlock and inherited by BottleneckAttention.cross_attn
- [x] `build_2d_sincos_pos_embed` called with `Cc` (context_dim before projection) — correct
- [x] up2/up1/up0 called without `cond=` — use GroupNorm, no crash
- [x] `use_cross_attn=False` (default) → HFGenerator behaves identically to hfgan-9 without SPADE
- [x] Existing test `test_upsample_block_with_skip` calls `ConvUpsampleBlock(768+384, 256)` — no cond_ch, still GroupNorm path ✅
- [x] `BottleneckAttention` backward-compatible — `sar_cross_attn=False` default
- [x] Config `use_spade: false` prevents any old SPADE code path
