# cfrwd-38 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Revive HFCF branch collapse and push toward PSNR ≥17 dB, SSIM ≥0.35 — via db4 DWT, FAB filters, fixed HFMaskedFFTLoss, MS-SSIM loss, and soft routing penalty.

**Architecture:** Drop-in replacement of HaarDown with DaubechiesDown inside existing DWTBlock; add FrequencyAttentionBlock (FAB) after each HFCF stream's ResBlocks; fix 4× diluted masked loss; add MS-SSIM as 4th AdaptiveLoss component; replace inactive hinge routing with always-active soft L2.

**Tech Stack:** PyTorch 2.8, Lightning 2.5, pytorch-msssim, existing cfrwd architecture in `src/models/cfrwd/`

---

## File Map

| File | Change |
|------|--------|
| `src/models/cfrwd/gen.py` | Add `DaubechiesDown`, update `DWTBlock`, add `FrequencyAttentionBlock`, update `HFCFBranch` |
| `src/models/cfrwd/losses.py` | Fix `HFMaskedFFTLoss.forward` (line 209), add `MSSSIMLoss` |
| `src/models/cfrwd/factory.py` | Add `MSSSIMLoss` import + `'MSSSIM'` criterion |
| `src/models/cfrwd/main.py` | Routing soft L2, 4th AdaptiveLoss eta, new TensorBoard keys, `val/msssim` |
| `src/models/cfrwd/config.yaml` | 4 param changes |
| `requirements.txt` | Add `pytorch-msssim` |
| `tests/models/cfrwd/test_cfrwd38.py` | New test file (TDD) |

---

## Task 1: Install pytorch-msssim

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Add dependency**

```
pytorch-msssim==1.0.0
```

Append to `requirements.txt` (after `torchvision` line).

- [ ] **Step 2: Install**

Run: `pip install pytorch-msssim==1.0.0`
Expected: Successfully installed pytorch-msssim-1.0.0

- [ ] **Step 3: Smoke test import**

Run: `python -c "from pytorch_msssim import ms_ssim; print('ok')"`
Expected: `ok`

- [ ] **Step 4: Commit**

```bash
git add requirements.txt
git commit -m "chore(deps): add pytorch-msssim 1.0.0 for MS-SSIM loss"
```

---

## Task 2: DaubechiesDown + DWTBlock update

**Files:**
- Modify: `src/models/cfrwd/gen.py` (after line 298, before `DWTBlock`)
- Create: `tests/models/cfrwd/test_cfrwd38.py`

- [ ] **Step 1: Write failing tests**

Create `tests/models/cfrwd/test_cfrwd38.py`:

```python
"""
cfrwd-38 unit tests.

Run from repo root:
    python -m pytest tests/models/cfrwd/test_cfrwd38.py -v
"""
import pytest
import torch
import torch.nn as nn


# ─── Task 2: DaubechiesDown ──────────────────────────────────────────────────

def test_daubechies_output_shapes_match_haar():
    """DaubechiesDown must produce same output shapes as HaarDown."""
    from src.models.cfrwd.gen import DaubechiesDown, HaarDown
    x = torch.randn(2, 1, 64, 64)
    haar   = HaarDown(in_channels=1)
    db4    = DaubechiesDown(in_channels=1)
    ll_h, lh_h, hl_h, hh_h = haar(x)
    ll_d, lh_d, hl_d, hh_d = db4(x)
    assert ll_d.shape == ll_h.shape, f"LL: {ll_d.shape} != {ll_h.shape}"
    assert lh_d.shape == lh_h.shape
    assert hl_d.shape == hl_h.shape
    assert hh_d.shape == hh_h.shape


def test_daubechies_lo_filter_unit_norm():
    """Lo_D filter must have L2 norm = 1 (orthonormal)."""
    from src.models.cfrwd.gen import DaubechiesDown
    db4 = DaubechiesDown(in_channels=1)
    lo_norm = db4.lo.norm().item()
    assert abs(lo_norm - 1.0) < 1e-5, f"Lo_D L2 norm = {lo_norm}, expected 1.0"


def test_daubechies_hi_filter_unit_norm():
    """Hi_D filter must have L2 norm = 1 (orthonormal)."""
    from src.models.cfrwd.gen import DaubechiesDown
    db4 = DaubechiesDown(in_channels=1)
    hi_norm = db4.hi.norm().item()
    assert abs(hi_norm - 1.0) < 1e-5, f"Hi_D L2 norm = {hi_norm}, expected 1.0"


def test_daubechies_no_trainable_params():
    """DaubechiesDown must have zero trainable parameters (register_buffer)."""
    from src.models.cfrwd.gen import DaubechiesDown
    db4 = DaubechiesDown(in_channels=1)
    n_params = sum(p.numel() for p in db4.parameters())
    assert n_params == 0, f"Expected 0 trainable params, got {n_params}"


def test_dwtblock_with_db4_shapes():
    """DWTBlock(db4) must return g1 at W/4, g2 at W/4 (3C), g3 at W/2 (3C)."""
    from src.models.cfrwd.gen import DWTBlock
    x = torch.randn(2, 1, 256, 256)
    dwt = DWTBlock(in_channels=1)
    g1, g2, g3 = dwt(x)
    assert g1.shape == (2, 1,  64,  64), f"g1={g1.shape}"
    assert g2.shape == (2, 3,  64,  64), f"g2={g2.shape}"
    assert g3.shape == (2, 3, 128, 128), f"g3={g3.shape}"


def test_daubechies_no_nan():
    """DaubechiesDown must not produce NaN/Inf on normal input."""
    from src.models.cfrwd.gen import DaubechiesDown
    db4 = DaubechiesDown(in_channels=3)
    x = torch.randn(4, 3, 256, 256)
    ll, lh, hl, hh = db4(x)
    for name, t in [('LL', ll), ('LH', lh), ('HL', hl), ('HH', hh)]:
        assert not torch.isnan(t).any(), f"NaN in {name}"
        assert not torch.isinf(t).any(), f"Inf in {name}"
```

- [ ] **Step 2: Run to verify tests fail**

Run: `python -m pytest tests/models/cfrwd/test_cfrwd38.py -k "daubechies or dwtblock" -v`
Expected: ImportError or AttributeError — `DaubechiesDown` does not exist yet.

- [ ] **Step 3: Implement DaubechiesDown in gen.py**

Insert the following class immediately after `HaarDown` (after the commented-out `HaarUp` block, before `DWTBlock`). Current `DWTBlock` starts at line 318.

```python
class DaubechiesDown(nn.Module):
    """
    2D Daubechies db4 DWT — separable convolution with reflect padding.

    Applies 1D Lo_D/Hi_D filters row-wise then column-wise (stride=2 each),
    producing four subbands: (LL, LH, HL, HH) each at (B, C, H//2, W//2).

    8-tap orthonormal filter: 4 vanishing moments — better edge/texture
    representation than Haar, cleaner LL/detail subband separation for SAR.

    Fixed filters via register_buffer — non-trainable, like HaarDown.
    HaarDown is kept for ablation reference but no longer instantiated.
    """
    # db4 analysis filters (pywt convention, orthonormal: ||h||_2 = 1)
    _LO = [ 0.23037781330885523,  0.71484657055254152,  0.63088076792959040,
           -0.02798376941685985, -0.18703481171888114,  0.03084138183598697,
            0.03288301166698295, -0.01059740178499728]
    _HI = [-0.01059740178499728, -0.03288301166698295,  0.03084138183598697,
            0.18703481171888114, -0.02798376941685985, -0.63088076792959040,
            0.71484657055254152, -0.23037781330885523]

    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.register_buffer('lo', torch.tensor(self._LO, dtype=torch.float32))
        self.register_buffer('hi', torch.tensor(self._HI, dtype=torch.float32))

    def _apply_rows(self, x: torch.Tensor, filt: torch.Tensor) -> torch.Tensor:
        """Apply 1D filter along W (columns), stride=2, reflect padding."""
        B, C, H, W = x.shape
        L   = filt.shape[0]          # 8
        pad = (L - 2) // 2           # 3
        x   = torch.nn.functional.pad(x, (pad, pad), mode='reflect')
        # Grouped conv: treat each channel independently
        w   = filt.view(1, 1, 1, L).expand(C, 1, 1, L).contiguous()
        # Reshape B×C×H×(W+2pad) → (B*H)×C×1×(W+2pad) for conv2d
        out = torch.nn.functional.conv2d(
            x.reshape(B * H, C, 1, W + 2 * pad),
            w, stride=(1, 2), groups=C
        )
        return out.reshape(B, C, H, -1)  # B×C×H×(W//2)

    def _apply_cols(self, x: torch.Tensor, filt: torch.Tensor) -> torch.Tensor:
        """Apply 1D filter along H (rows), stride=2, reflect padding."""
        return self._apply_rows(x.transpose(-2, -1), filt).transpose(-2, -1)

    def forward(self, x: torch.Tensor):
        lo_r = self._apply_rows(x, self.lo)   # B×C×H×(W//2)
        hi_r = self._apply_rows(x, self.hi)
        LL = self._apply_cols(lo_r, self.lo)  # B×C×(H//2)×(W//2)
        LH = self._apply_cols(lo_r, self.hi)
        HL = self._apply_cols(hi_r, self.lo)
        HH = self._apply_cols(hi_r, self.hi)
        return LL, LH, HL, HH
```

- [ ] **Step 4: Update DWTBlock to use DaubechiesDown**

In `DWTBlock.__init__`, change one line:

```python
# Before:
self.dwt = HaarDown(in_channels=in_channels)

# After:
self.dwt = DaubechiesDown(in_channels=in_channels)
```

- [ ] **Step 5: Run tests to verify pass**

Run: `python -m pytest tests/models/cfrwd/test_cfrwd38.py -k "daubechies or dwtblock" -v`
Expected: 6 PASSED

- [ ] **Step 6: Smoke test gen.py**

Run: `python src/models/cfrwd/gen.py`
Expected: No errors, prints `out: torch.Size([1, 3, 256, 256])`

- [ ] **Step 7: Commit**

```bash
git add src/models/cfrwd/gen.py tests/models/cfrwd/test_cfrwd38.py
git commit -m "feat(gen): Haar → Daubechies db4 in DWTBlock, add unit tests"
```

---

## Task 3: FrequencyAttentionBlock (FAB) in HFCFBranch

**Files:**
- Modify: `src/models/cfrwd/gen.py`
- Modify: `tests/models/cfrwd/test_cfrwd38.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/models/cfrwd/test_cfrwd38.py`:

```python
# ─── Task 3: FrequencyAttentionBlock ────────────────────────────────────────

def test_fab_output_shape():
    """FAB must preserve input shape B×C×H×W."""
    from src.models.cfrwd.gen import FrequencyAttentionBlock
    fab = FrequencyAttentionBlock(channels=64, h=64, w=64)
    x = torch.randn(2, 64, 64, 64)
    out = fab(x)
    assert out.shape == x.shape, f"FAB output {out.shape} != input {x.shape}"


def test_fab_identity_init():
    """At init (weight_real=1, weight_imag=0), FAB must be identity: output ≈ input."""
    from src.models.cfrwd.gen import FrequencyAttentionBlock
    fab = FrequencyAttentionBlock(channels=64, h=64, w=64)
    x = torch.randn(2, 64, 64, 64)
    with torch.no_grad():
        out = fab(x)
    assert torch.allclose(out, x, atol=1e-4), \
        f"FAB identity init failed: max diff = {(out - x).abs().max().item():.6f}"


def test_fab_gradient_flows():
    """Gradient must flow through FAB to weight_real and weight_imag."""
    from src.models.cfrwd.gen import FrequencyAttentionBlock
    fab = FrequencyAttentionBlock(channels=16, h=32, w=32)
    x = torch.randn(1, 16, 32, 32)
    out = fab(x)
    out.mean().backward()
    assert fab.weight_real.grad is not None, "No gradient on weight_real"
    assert fab.weight_imag.grad is not None, "No gradient on weight_imag"


def test_fab_no_nan():
    """FAB must not produce NaN/Inf."""
    from src.models.cfrwd.gen import FrequencyAttentionBlock
    fab = FrequencyAttentionBlock(channels=64, h=64, w=64)
    x = torch.randn(4, 64, 64, 64)
    out = fab(x)
    assert not torch.isnan(out).any(), "NaN in FAB output"
    assert not torch.isinf(out).any(), "Inf in FAB output"


def test_hfcf_branch_has_fab_attributes():
    """HFCFBranch must expose fab_low, fab_mid, fab_high."""
    from src.models.cfrwd.gen import HFCFBranch
    branch = HFCFBranch(in_channels=1, hidden_dim=64)
    assert hasattr(branch, 'fab_low'),  "HFCFBranch missing fab_low"
    assert hasattr(branch, 'fab_mid'),  "HFCFBranch missing fab_mid"
    assert hasattr(branch, 'fab_high'), "HFCFBranch missing fab_high"


def test_hfcf_branch_output_shape_unchanged():
    """HFCFBranch output shape must remain B×32×256×256 after FAB insertion."""
    from src.models.cfrwd.gen import HFCFBranch
    branch = HFCFBranch(in_channels=1, hidden_dim=64)
    x = torch.randn(2, 1, 256, 256)
    with torch.no_grad():
        out = branch(x)
    assert out.shape == (2, 32, 256, 256), f"HFCFBranch output shape {out.shape}"
```

- [ ] **Step 2: Run to verify tests fail**

Run: `python -m pytest tests/models/cfrwd/test_cfrwd38.py -k "fab or hfcf_branch" -v`
Expected: ImportError — `FrequencyAttentionBlock` does not exist yet.

- [ ] **Step 3: Implement FrequencyAttentionBlock in gen.py**

Insert the following class immediately before `HFCFBranch` (before line 486):

```python
class FrequencyAttentionBlock(nn.Module):
    """
    FFNet-style learnable frequency-domain attention.

    Applies a per-channel complex filter K in rfft2 space:
        x_f = rfft2(x)
        x_f = x_f * K          where K = weight_real + i*weight_imag
        out = irfft2(x_f)

    Initialized as identity (weight_real=1, weight_imag=0) so early training
    is unaffected. Learns which frequency components to amplify or suppress.

    h, w must match the spatial dimensions of the forward input.
    All three HFCFBranch streams are 64×64 after ResBlocks — use h=64, w=64.

    Not a Conv2d/InstanceNorm2d, so _initialize_weights() skips it automatically.
    """
    def __init__(self, channels: int, h: int, w: int):
        super().__init__()
        w_freq = w // 2 + 1           # rfft2 output width: 33 for w=64
        # Identity init: K = 1 + 0j → output = input at epoch 0
        self.weight_real = nn.Parameter(torch.ones (1, channels, h, w_freq))
        self.weight_imag = nn.Parameter(torch.zeros(1, channels, h, w_freq))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_f    = torch.fft.rfft2(x.float(), norm='ortho')
        weight = torch.complex(self.weight_real, self.weight_imag)
        x_f    = x_f * weight
        return torch.fft.irfft2(x_f, s=(H, W), norm='ortho').to(x.dtype)
```

- [ ] **Step 4: Add FAB instances to HFCFBranch.__init__**

In `HFCFBranch.__init__`, add the three FAB instances after `self.high_stream` and before `self.merge`. Current `self.merge` is around line 542.

```python
        # --- Frequency Attention Blocks (after ResBlocks, before merge) ---
        # All three streams output (B, 64, 64, 64) — same h=64, w=64 for all.
        self.fab_low  = FrequencyAttentionBlock(hidden_dim, 64, 64)
        self.fab_mid  = FrequencyAttentionBlock(hidden_dim, 64, 64)
        self.fab_high = FrequencyAttentionBlock(hidden_dim, 64, 64)
```

- [ ] **Step 5: Update HFCFBranch.forward to apply FABs**

Current `forward` (lines 557–568):
```python
    def forward(self, x):
        g1, g2, g3 = self.dwt(x)

        logger.debug(f'DWT shapes: g1={g1.shape}, g2={g2.shape}, g3={g3.shape}', once=True)

        ll   = self.ll_stream(self.ll_cbam(self.ll_proj(g1)))
        mid  = self.mid_stream(self.mid_cbam(self.mid_proj(g2)))
        high = self.high_stream(self.high_cbam(self.high_proj(g3)))

        merged = self.merge(torch.cat([ll, mid, high], dim=1))
        return self.decoder(merged)
```

Replace with:
```python
    def forward(self, x):
        g1, g2, g3 = self.dwt(x)

        logger.debug(f'DWT shapes: g1={g1.shape}, g2={g2.shape}, g3={g3.shape}', once=True)

        ll   = self.fab_low (self.ll_stream  (self.ll_cbam  (self.ll_proj  (g1))))
        mid  = self.fab_mid (self.mid_stream (self.mid_cbam (self.mid_proj (g2))))
        high = self.fab_high(self.high_stream(self.high_cbam(self.high_proj(g3))))

        merged = self.merge(torch.cat([ll, mid, high], dim=1))
        return self.decoder(merged)
```

- [ ] **Step 6: Run tests to verify pass**

Run: `python -m pytest tests/models/cfrwd/test_cfrwd38.py -k "fab or hfcf_branch" -v`
Expected: 6 PASSED

- [ ] **Step 7: Full gen.py smoke test**

Run: `python src/models/cfrwd/gen.py`
Expected: No errors. Output shape printed.

- [ ] **Step 8: Commit**

```bash
git add src/models/cfrwd/gen.py tests/models/cfrwd/test_cfrwd38.py
git commit -m "feat(gen): add FrequencyAttentionBlock, insert FAB in HFCFBranch streams"
```

---

## Task 4: Fix HFMaskedFFTLoss + Add MSSSIMLoss

**Files:**
- Modify: `src/models/cfrwd/losses.py`
- Modify: `tests/models/cfrwd/test_cfrwd38.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/models/cfrwd/test_cfrwd38.py`:

```python
# ─── Task 4: HFMaskedFFTLoss fix + MSSSIMLoss ───────────────────────────────

def test_hf_masked_loss_greater_than_unmasked_naive():
    """
    After fix: masked mean (÷ active bins only) must be larger than naive
    mean (÷ all bins) for the same mismatch.
    """
    from src.models.cfrwd.losses import HFMaskedFFTLoss
    import torch.nn as nn

    pred   = torch.zeros(2, 3, 64, 64)
    target = torch.randn(2, 3, 64, 64)

    # Compute both variants manually to compare
    loss_fn = HFMaskedFFTLoss(freq_threshold=0.25)
    masked_loss = loss_fn(pred, target).item()

    # Naive: divide by all bins
    pred_f = torch.fft.rfft2(pred.float(), norm='ortho')
    tgt_f  = torch.fft.rfft2(target.float(), norm='ortho')
    naive_loss = (pred_f - tgt_f).abs().mean().item()

    assert masked_loss > naive_loss, \
        f"Masked mean {masked_loss:.4f} should be > naive mean {naive_loss:.4f}"


def test_hf_masked_loss_zero_on_identical():
    """HFMaskedFFTLoss must still be 0 on identical inputs after fix."""
    from src.models.cfrwd.losses import HFMaskedFFTLoss
    loss_fn = HFMaskedFFTLoss(freq_threshold=0.25)
    t = torch.randn(2, 3, 64, 64)
    assert loss_fn(t, t).item() < 1e-6


def test_hf_masked_loss_gradient_flows():
    """Gradient must flow through fixed HFMaskedFFTLoss."""
    from src.models.cfrwd.losses import HFMaskedFFTLoss
    loss_fn = HFMaskedFFTLoss(freq_threshold=0.25)
    pred = torch.randn(2, 3, 64, 64, requires_grad=True)
    target = torch.randn(2, 3, 64, 64)
    loss_fn(pred, target).backward()
    assert pred.grad is not None


def test_msssim_loss_zero_on_identical():
    """MSSSIMLoss must be 0 on identical inputs."""
    from src.models.cfrwd.losses import MSSSIMLoss
    loss_fn = MSSSIMLoss()
    t = torch.rand(2, 3, 256, 256) * 2 - 1  # [-1, 1]
    val = loss_fn(t, t).item()
    assert val < 1e-4, f"MSSSIMLoss on identical = {val}, expected ~0"


def test_msssim_loss_range():
    """MSSSIMLoss must be in [0, 1] for normal inputs in [-1, 1]."""
    from src.models.cfrwd.losses import MSSSIMLoss
    loss_fn = MSSSIMLoss()
    pred   = torch.rand(2, 3, 256, 256) * 2 - 1
    target = torch.rand(2, 3, 256, 256) * 2 - 1
    val = loss_fn(pred, target).item()
    assert 0.0 <= val <= 1.0, f"MSSSIMLoss = {val}, expected [0, 1]"


def test_msssim_loss_gradient_flows():
    """Gradient must flow through MSSSIMLoss."""
    from src.models.cfrwd.losses import MSSSIMLoss
    loss_fn = MSSSIMLoss()
    pred   = torch.rand(2, 3, 256, 256, requires_grad=True) * 2 - 1
    target = torch.rand(2, 3, 256, 256) * 2 - 1
    loss_fn(pred, target).backward()
    assert pred.grad is not None
```

- [ ] **Step 2: Run to verify tests fail**

Run: `python -m pytest tests/models/cfrwd/test_cfrwd38.py -k "hf_masked or msssim" -v`
Expected: `test_hf_masked_loss_greater_than_unmasked_naive` FAIL (loss values equal or naive wins); `test_msssim_loss_*` ImportError.

- [ ] **Step 3: Fix HFMaskedFFTLoss.forward (line 209)**

In `src/models/cfrwd/losses.py`, replace the last line of `HFMaskedFFTLoss.forward`:

```python
# Before (line 209):
        return ((pred_f - tgt_f).abs() * hf_mask).mean()

# After (masked mean — divides only by active HF bins × B × C):
        diff_hf  = (pred_f - tgt_f).abs() * hf_mask
        n_active = hf_mask.sum() * pred_f.shape[0] * pred_f.shape[1]  # ×B ×C
        return diff_hf.sum() / n_active.clamp(min=1)
```

- [ ] **Step 4: Add MSSSIMLoss to losses.py**

Add the import at the top of `src/models/cfrwd/losses.py` (after existing imports):

```python
from pytorch_msssim import ms_ssim
```

Add the class after `HFMaskedFFTLoss`:

```python
class MSSSIMLoss(nn.Module):
    """
    Multi-Scale SSIM loss (Wang et al., 2003).
    Loss = 1 - MS-SSIM, so 0 = perfect, 1 = worst.

    Uses 5 scales with default win_size=11, K=(0.01, 0.03).
    Directly optimizes the SSIM gap (cfrwd-37: 0.186 vs paper 0.562).
    The 5-scale structure is architecturally aligned with the two-level DWT.

    Input: B×3×H×W in [-1, 1] (tanh output). data_range=2.0 for [-1,1].
    Requires H, W ≥ 160 (5 scales × 2^4 downsampling). Train at 256×256 — OK.
    """
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return 1.0 - ms_ssim(pred.float(), target.float(),
                              data_range=2.0, size_average=True)
```

- [ ] **Step 5: Run tests to verify pass**

Run: `python -m pytest tests/models/cfrwd/test_cfrwd38.py -k "hf_masked or msssim" -v`
Expected: 6 PASSED

- [ ] **Step 6: Commit**

```bash
git add src/models/cfrwd/losses.py tests/models/cfrwd/test_cfrwd38.py
git commit -m "fix(losses): masked mean in HFMaskedFFTLoss; feat: add MSSSIMLoss"
```

---

## Task 5: Update factory.py

**Files:**
- Modify: `src/models/cfrwd/factory.py`
- Modify: `tests/models/cfrwd/test_cfrwd38.py`

- [ ] **Step 1: Write failing test**

Append to `tests/models/cfrwd/test_cfrwd38.py`:

```python
# ─── Task 5: factory.py ──────────────────────────────────────────────────────

def test_factory_criterions_has_msssim():
    """build_criterions must return 'MSSSIM' key after cfrwd-38."""
    from src.models.cfrwd.factory import build_criterions

    class _FakeBackbone(nn.Module):
        def forward(self, x, y):
            return torch.zeros(x.shape[0])

    crits = build_criterions(lpips_backbone=_FakeBackbone())
    assert 'MSSSIM' in crits, "build_criterions must contain 'MSSSIM'"
    assert 'FOCAL_FREQ' in crits
    assert 'HF_AUX'     in crits
    assert 'GAN'        in crits
    assert 'FM'         in crits
    assert 'L1'         in crits
```

- [ ] **Step 2: Run to verify test fails**

Run: `python -m pytest tests/models/cfrwd/test_cfrwd38.py -k "factory" -v`
Expected: FAIL — `MSSSIM` not in crits.

- [ ] **Step 3: Update factory.py**

In `src/models/cfrwd/factory.py`, update the import line:

```python
# Before:
from src.models.cfrwd.losses import FeatureMatchingLoss, GANLoss, L1Loss, FocalFrequencyLoss, HFMaskedFFTLoss, LPIPSLoss

# After:
from src.models.cfrwd.losses import FeatureMatchingLoss, GANLoss, L1Loss, FocalFrequencyLoss, HFMaskedFFTLoss, LPIPSLoss, MSSSIMLoss
```

In `build_criterions`, add `'MSSSIM'` criterion after `'HF_AUX'`:

```python
    crits = {
        'GAN':        GANLoss(use_lsgan=True),
        'FM':         FeatureMatchingLoss(),
        'L1':         L1Loss(),
        'FOCAL_FREQ': FocalFrequencyLoss(),
        'HF_AUX':     HFMaskedFFTLoss(freq_threshold=cfg.loss.get('hf_freq_threshold', 0.25)),
        'MSSSIM':     MSSSIMLoss(),
    }
```

Also update the return type annotation to include `'MSSSIM'`:

```python
def build_criterions(lpips_backbone=None) -> dict[Literal['GAN', 'FM', 'L1', 'FOCAL_FREQ', 'HF_AUX', 'LPIPS', 'MSSSIM'], nn.Module]:
```

- [ ] **Step 4: Run test to verify pass**

Run: `python -m pytest tests/models/cfrwd/test_cfrwd38.py -k "factory" -v`
Expected: 1 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/models/cfrwd/factory.py tests/models/cfrwd/test_cfrwd38.py
git commit -m "feat(factory): add MSSSIMLoss criterion"
```

---

## Task 6: Update main.py

**Files:**
- Modify: `src/models/cfrwd/main.py`
- Modify: `tests/models/cfrwd/test_cfrwd38.py`

### 6a. _n_recon and __init__ changes

- [ ] **Step 1: Write failing tests**

Append to `tests/models/cfrwd/test_cfrwd38.py`:

```python
# ─── Task 6: main.py routing + adaptive loss ─────────────────────────────────

def test_routing_soft_l2_always_active():
    """Soft L2 routing must be > 0 even for moderately unbalanced routing."""
    # w_hfcf = 0.7 (above 0.5) — was not penalized by old hinge
    weights = torch.zeros(2, 2, 8, 8)
    weights[:, 0, :, :] = 0.3   # CFR
    weights[:, 1, :, :] = 0.7   # HFCF
    routing_balance_weight = 0.1
    loss = (weights[:, 1].mean() - 0.5).pow(2) * routing_balance_weight
    assert loss.item() > 0, "Soft L2 must penalize w_hfcf=0.7"


def test_routing_soft_l2_zero_at_balance():
    """Soft L2 routing must be exactly 0 when w_hfcf = 0.5."""
    weights = torch.full((2, 2, 8, 8), 0.5)
    routing_balance_weight = 0.1
    loss = (weights[:, 1].mean() - 0.5).pow(2) * routing_balance_weight
    assert loss.item() < 1e-10, f"Soft L2 must be 0 at balance, got {loss.item()}"


def test_n_recon_is_4_with_both_flags():
    """_n_recon must equal 4 when both use_lpips=true and use_msssim=true."""
    # Simulate the _n_recon logic from main.py __init__
    class FakeCfg:
        class loss:
            @staticmethod
            def get(key, default):
                return {'use_lpips': True, 'use_msssim': True}.get(key, default)
    cfg = FakeCfg()
    _n_recon = 2
    if cfg.loss.get('use_lpips',  False): _n_recon += 1
    if cfg.loss.get('use_msssim', False): _n_recon += 1
    assert _n_recon == 4, f"Expected _n_recon=4, got {_n_recon}"
```

- [ ] **Step 2: Run tests to verify routing tests pass (logic check)**

Run: `python -m pytest tests/models/cfrwd/test_cfrwd38.py -k "routing_soft or n_recon" -v`
Expected: The routing tests use standalone math (no imports) — should PASS already. The `n_recon` test also uses standalone logic — should PASS. These are verification tests for the logic we're about to add to main.py.

- [ ] **Step 3: Update __init__ in main.py**

In `SAR2OPTGANLightningModule.__init__`, find and update:

```python
# Before (line 38-39):
        _n_recon = 3 if self.cfg.loss.get('use_lpips', False) else 2
        self.adaptive_loss = AdaptiveLoss(n_losses=_n_recon)

# After:
        _n_recon = 2
        if self.cfg.loss.get('use_lpips',  False): _n_recon += 1
        if self.cfg.loss.get('use_msssim', False): _n_recon += 1
        self.adaptive_loss = AdaptiveLoss(n_losses=_n_recon)
```

Also add the val/msssim accumulator in `__init__` (after `self.fixed_opt = None`):

```python
        self._val_msssim_acc: list = []
```

### 6b. training_step changes

- [ ] **Step 4: Replace routing entropy penalty (training_step)**

Find this block in `training_step` (lines 143–145):

```python
        eps = 1e-8
        H_spatial    = -(fusion_weights * (fusion_weights + eps).log()).sum(dim=1)  # B×H×W
        loss_routing = (0.347 - H_spatial.mean()).clamp(min=0.0) * self.cfg.loss.get('routing_entropy_weight', 0.005)
```

Replace with:

```python
        # Soft L2 toward balance: always active, penalizes any deviation from w_hfcf=0.5.
        # (Old hinge at 0.347 required near-degenerate 0.12/0.88 split to activate — never triggered.)
        loss_routing = (fusion_weights[:, 1].mean() - 0.5).pow(2) * self.cfg.loss.routing_balance_weight
        # Routing entropy kept for diagnostics (not used in loss)
        _H_routing = -(fusion_weights * (fusion_weights + 1e-8).log()).sum(dim=1).mean()
```

- [ ] **Step 5: Add MSSSIM to recon_losses in training_step**

Find this block (lines 127–133):

```python
        recon_losses = [loss_l1, loss_focal_freq]
        if 'LPIPS' in self.criterions:
            loss_lpips = self.criterions['LPIPS'](fake_opt, real_opt)
            recon_losses.append(loss_lpips)
        loss_recon = self.adaptive_loss(recon_losses)
```

Replace with:

```python
        recon_losses = [loss_l1, loss_focal_freq]
        if 'LPIPS' in self.criterions:
            loss_lpips = self.criterions['LPIPS'](fake_opt, real_opt)
            recon_losses.append(loss_lpips)
        if self.cfg.loss.get('use_msssim', False):
            loss_msssim = self.criterions['MSSSIM'](fake_opt, real_opt)
            recon_losses.append(loss_msssim)
        loss_recon = self.adaptive_loss(recon_losses)
```

### 6c. log_dict update in training_step

- [ ] **Step 6: Update log_dict to add new keys**

Find the existing `self.log_dict({...})` block (lines 160–188). Replace the entire block:

```python
        # --- HFCF diagnostics ---
        with torch.no_grad():
            g1, g2, g3 = self.netG.hfcf_branch.dwt(real_sar)

        _msssim_log = ({'train/loss_msssim': loss_msssim,
                        'loss/eta_msssim':   self.adaptive_loss.eta[3],
                        'loss/w_msssim':     torch.exp(-self.adaptive_loss.eta[3]),
                       } if self.cfg.loss.get('use_msssim', False) else {})

        self.log('train/g_loss', g_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=real_sar.size(0))
        self.log_dict({
            'train/loss_fm':         loss_fm,
            'train/loss_gan':        loss_gan,
            'train/loss_d':          d_loss,
            'train/loss_l1':         loss_l1,
            'train/loss_focal_freq': loss_focal_freq,
            'feats/d_real_mean': real_means.mean(),
            'feats/d_fake_mean': fake_means.mean(),
            'loss/eta_l1':         self.adaptive_loss.eta[0],
            'loss/eta_focal_freq': self.adaptive_loss.eta[1],
            'loss/w_l1':           torch.exp(-self.adaptive_loss.eta[0]),
            'loss/w_focal_freq':   torch.exp(-self.adaptive_loss.eta[1]),
            **({'train/loss_lpips': loss_lpips,
                'loss/eta_lpips':   self.adaptive_loss.eta[2],
                'loss/w_lpips':     torch.exp(-self.adaptive_loss.eta[2]),
               } if 'LPIPS' in self.criterions else {}),
            **_msssim_log,
            'loss/hfcf_aux':        loss_hfcf_aux,
            'loss/routing_entropy': _H_routing,
            'loss/routing_loss':    loss_routing,
            'fusion/w_hfcf':        fusion_weights[:, 1].mean(),
            'fusion/spatial_std':   fusion_weights[:, 1].std(dim=[1, 2]).mean(),
            'fusion/temperature':   self.netG.adaptive_fusion.temperature.item(),
            # HFCF collapse indicators
            'hfcf/out_spatial_std': hfcf_out.std(dim=[2, 3]).mean(),
            'hfcf/out_mean':        hfcf_out.mean(),
            # DWT subband energies (db4 quality check)
            'hfcf/g1_energy': g1.pow(2).mean(),
            'hfcf/g2_energy': g2.pow(2).mean(),
            'hfcf/g3_energy': g3.pow(2).mean(),
        }, prog_bar=False, on_step=False, on_epoch=True, batch_size=real_sar.size(0))
```

### 6d. validation_step + on_validation_epoch_end

- [ ] **Step 7: Add val/msssim accumulation to validation_step**

Add import at top of main.py (with other imports):

```python
from pytorch_msssim import ms_ssim as pytorch_ms_ssim
```

In `validation_step`, after the line `self.ergas.update(fake_01, real_01)` (line 226), add:

```python
        # MS-SSIM: accumulate per-batch values; compute epoch mean in on_validation_epoch_end.
        # Raw MS-SSIM (higher=better), NOT 1-ms_ssim.
        if self.cfg.loss.get('use_msssim', False):
            val_msssim = pytorch_ms_ssim(fake_opt_f32, real_opt_f32,
                                         data_range=2.0, size_average=True)
            self._val_msssim_acc.append(val_msssim.item())
```

In `on_validation_epoch_end`, after `self.lpips_metric.reset()` and before `gc.collect()`:

```python
        if self._val_msssim_acc:
            self.log('val/msssim', torch.tensor(self._val_msssim_acc).mean(), prog_bar=True)
            self._val_msssim_acc = []
```

### 6e. on_train_epoch_end: FAB diagnostics

- [ ] **Step 8: Add FAB filter logging to on_train_epoch_end**

In `on_train_epoch_end`, add the following block before the `if self.fixed_sar is None: return` guard (so it always runs):

```python
        # FAB filter magnitude — deviates from 1.0 as the branch learns frequency selectivity
        hfcf = self.netG.hfcf_branch
        for tag, fab in [('low', hfcf.fab_low), ('mid', hfcf.fab_mid), ('high', hfcf.fab_high)]:
            with torch.no_grad():
                mag = torch.complex(fab.weight_real, fab.weight_imag).abs()
            self.log(f'fab/filter_mag_{tag}', mag.mean(), on_step=False, on_epoch=True)
            self.log(f'fab/filter_std_{tag}', mag.std(),  on_step=False, on_epoch=True)
```

- [ ] **Step 9: Run all tests**

Run: `python -m pytest tests/models/cfrwd/test_cfrwd38.py -v`
Expected: All tests PASS.

- [ ] **Step 10: Commit**

```bash
git add src/models/cfrwd/main.py tests/models/cfrwd/test_cfrwd38.py
git commit -m "feat(main): soft L2 routing, MS-SSIM loss, HFCF/FAB TensorBoard diagnostics"
```

---

## Task 7: Config update + end-to-end smoke test

**Files:**
- Modify: `src/models/cfrwd/config.yaml`

- [ ] **Step 1: Update config.yaml**

Replace the `loss` block in `config.yaml`:

```yaml
loss:
  gan_weight: 1
  fm_weight: 10
  # L1, FOCAL_FREQ, LPIPS, MSSSIM auto-balanced via AdaptiveLoss (Kendall et al., 2018).
  use_lpips: true   # 3rd AdaptiveLoss component
  use_msssim: true  # 4th AdaptiveLoss component — targets SSIM gap (0.186 vs paper 0.562)
  # cfrwd-38 additions/fixes:
  aux_hfcf_weight_start: 0.5     # was 0.3 — stronger initial HF supervision signal
  aux_hfcf_weight_end: 0.1       # unchanged
  routing_balance_weight: 0.1    # replaces routing_entropy_weight: 0.005
                                  # soft L2 formula: (w_hfcf - 0.5)^2 × weight
  hf_freq_threshold: 0.25         # HFMaskedFFTLoss: supervise freqs above 25% Nyquist
```

Also update `system.tb_version` and `system.resume_ckpt`:

```yaml
system:
  ...
  tb_version: "cfrwd-38"
  resume_ckpt: null
```

- [ ] **Step 2: Verify gen.py smoke test**

Run: `python src/models/cfrwd/gen.py`
Expected: No errors, model forward pass completes, `out: torch.Size([1, 3, 256, 256])` printed.

- [ ] **Step 3: Verify discriminator.py smoke test**

Run: `python src/models/cfrwd/discriminator.py`
Expected: No errors.

- [ ] **Step 4: Run full test suite**

Run: `python -m pytest tests/models/cfrwd/ -v`
Expected: All tests in `test_cfrwd37.py` and `test_cfrwd38.py` PASS.

- [ ] **Step 5: Run train.py quick integration check**

Run with reduced batches to verify the training loop completes one full epoch without error:

```bash
python -m src.models.cfrwd.train
```

Check `config.yaml` has `limit_train_batches: 0.1` and `limit_val_batches: 0.1` during this test. Expected: at least one training step and validation step complete. TensorBoard keys `hfcf/out_spatial_std`, `fab/filter_mag_low`, `val/msssim`, `loss/routing_loss` appear in CSV log.

Restore `limit_train_batches: 1.0` and `limit_val_batches: 1.0` after the check.

- [ ] **Step 6: Commit**

```bash
git add src/models/cfrwd/config.yaml
git commit -m "feat(config): cfrwd-38 params — use_msssim, routing_balance_weight, aux_weight_start=0.5, tb_version"
```

---

## Task 8: Changelog entry

**Files:**
- Modify: `changelog.md`

- [ ] **Step 1: Add cfrwd-38 entry**

Add a new experiment entry at the top of `changelog.md`:

```markdown
## cfrwd-38 — 2026-04-09

**Goal:** Revive HFCF branch collapse; push SSIM from 0.186 toward ≥0.35.

**Changes:**
- `gen.py`: Haar → Daubechies db4 (`DaubechiesDown`, 8-tap separable conv, reflect padding)
- `gen.py`: Add `FrequencyAttentionBlock` (learnable rfft2 filter, identity init) after each HFCFBranch stream
- `losses.py`: Fix `HFMaskedFFTLoss` masked mean (÷ active bins only, was ÷ all bins = 4× dilution)
- `losses.py`: Add `MSSSIMLoss` (1 − MS-SSIM, data_range=2.0, 5 scales)
- `factory.py`: Add `'MSSSIM'` criterion
- `main.py`: Replace hinge routing penalty with soft L2 `(w_hfcf − 0.5)² × 0.1`
- `main.py`: 4th AdaptiveLoss eta for MSSSIM; `_n_recon` logic generalized
- `main.py`: New TensorBoard keys: `hfcf/out_*`, `hfcf/g{1,2,3}_energy`, `fab/filter_*`, `val/msssim`
- `config.yaml`: `aux_hfcf_weight_start: 0.5`, `routing_balance_weight: 0.1`, `use_msssim: true`, `tb_version: cfrwd-38`

**Clean start:** Yes (cfrwd-37 discarded — HFCF weights stuck in flat minima).

**Results:** (to be filled after training)
```

- [ ] **Step 2: Commit**

```bash
git add changelog.md
git commit -m "docs(changelog): add cfrwd-38 entry"
```

---

## Post-Implementation Verification Checklist

After all tasks are complete, verify:

- [ ] `python -m pytest tests/models/cfrwd/ -v` — all tests pass
- [ ] `python src/models/cfrwd/gen.py` — no errors, output `[1, 3, 256, 256]`
- [ ] `python src/models/cfrwd/discriminator.py` — no errors
- [ ] Training step produces `hfcf/out_spatial_std > 0` from epoch 1 (branch not grey blob)
- [ ] `loss/routing_loss > 0` from epoch 0 (was permanently 0 in cfrwd-37)
- [ ] `loss/hfcf_aux` starts ~4× higher than cfrwd-37 (masked mean fix)
- [ ] `fab/filter_mag_*` starts near 1.0, deviates after ep10
- [ ] `val/msssim` appears in TensorBoard from epoch 1
