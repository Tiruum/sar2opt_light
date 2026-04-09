# cfrwd-38 Design: HFCF Revival via db4 + FAB + Loss Fixes

**Date:** 2026-04-09
**Branch:** audit → cfrwd-38 experiment
**Goal:** Revive collapsed HFCF branch and push toward SOTA PSNR ≥17 dB, SSIM ≥0.35 on SEN1-2 S5.
**Clean start:** Yes — cfrwd-37 checkpoint discarded (HFCF weights stuck in flat minima).

---

## Problem Statement (from cfrwd-37 audit)

| Issue | Evidence | Root Cause |
|-------|----------|------------|
| HFCF branch outputs grey blob every epoch | Visual: 0 spatial variation across all 60 epochs | Three compounding bugs below |
| `HFMaskedFFTLoss` 4× diluted | `loss/hfcf_aux ≈ 0.012`; raw loss ≈ 0.042 (should be ~0.168) | `.mean()` divides by ALL freq bins incl. ~75% zero-masked |
| `routing_loss` always 0.0 | 62 epochs of logs | Hinge at 0.347 requires collapse to 0.12/0.88 split; entropy currently ~0.59 |
| SSIM = 0.186 (paper: 0.562) | val/ssim | No structural loss; L1 causes blur by averaging outputs |
| `w_hfcf` = 0.722, drifting up | fusion/w_hfcf | HFCF receives no HF-specific supervision gradient; trivially wins fusion |

**Paper baseline (Wei et al., Remote Sens. 2023):**
- Loss: LSGAN + FM only (no pixel loss)
- PSNR 18.97 dB, SSIM 0.562, LPIPS 0.399 on SEN1-2 S5
- WD branch output: edge map, not full image

**cfrwd-37 at ep61:** PSNR 15.14, SSIM 0.186, LPIPS 0.374. PSNR plateau forming before LR decay (ep200).

---

## Architecture Status — What Is NOT Changing

- CFRBranch, CFRBlock, AdaptiveFusion, Discriminator — unchanged
- HFCFBranch decoder, merge (1×1 Conv 192→64), HFCFPreprocess — unchanged
- WDResBlock, RedBlock, CBAM — unchanged (CBAM stays in its original position: before ResBlocks)
- Training loop structure (manual optimization, EMA, LR schedule) — unchanged
- AdaptiveLoss formula `Σ(L_i × exp(-η_i) + η_i)` — unchanged

---

## Change 1 — `gen.py`: Haar → Daubechies db4 in DWTBlock

**File:** `src/models/cfrwd/gen.py`

**Problem:** Haar (1 vanishing moment) poorly approximates smooth SAR regions; detail subbands have energy leakage between edge and smooth content. Daubechies db4 (4 vanishing moments) provides cleaner LL/LH/HL/HH separation and better sparse representation of curvilinear structures (roads, building edges) — key features in both SAR and optical imagery.

**Implementation:** Replace `HaarDown` with `DaubechiesDown`. Same interface:
- `forward(x: Tensor) → (LL, LH, HL, HH)` each shape `(B, C, H//2, W//2)`
- Fixed filters via `register_buffer` (non-trainable, same as HaarDown)
- Reflect padding (3-tap on each side for length-8 db4 filters)
- Separable 2D convolution: apply 1D Lo_D/Hi_D row-wise then column-wise with stride=2

**db4 analysis filter coefficients (orthonormal):**
```python
LO_D = [ 0.23037781,  0.71484657,  0.63088075, -0.02798377,
        -0.18703481,  0.03084138,  0.03288301, -0.01059740]
HI_D = [-0.01059740, -0.03288301,  0.03084138,  0.18703481,
        -0.02798377, -0.63088075,  0.71484657, -0.23037781]
```

`DWTBlock` replaces `self.dwt = HaarDown()` with `self.dwt = DaubechiesDown()`. No other changes to DWTBlock or downstream shapes (g1, g2, g3 shapes are identical to Haar output).

`HaarDown` class is kept in the file but no longer instantiated.

`_initialize_weights` must NOT be called on `DaubechiesDown` (same invariant as HaarDown: fixed buffers, not trained weights).

---

## Change 2 — `gen.py`: FrequencyAttentionBlock (FAB) in HFCFBranch streams

**File:** `src/models/cfrwd/gen.py`

**Problem:** HFCFBranch streams process wavelet coefficients but have no explicit mechanism to learn which frequency components to amplify or suppress. After ResBlocks learn spatial features, FAB adds a frequency-domain refinement step before the three-way merge.

**Architecture verified:** Current per-stream order is `Preprocess → CBAM → ResBlocks`. FAB is inserted AFTER ResBlocks, before merge. CBAM position is preserved (speckle suppression before computation).

```
Low:  HFCFPreprocess → CBAM → WDResBlock×2 → FAB(64, 64, 64)
Mid:  HFCFPreprocess → CBAM → WDResBlock×6 → FAB(64, 64, 64)
High: HFCFPreprocess → CBAM → RedBlock×2   → FAB(64, 64, 64)
```

All three streams output `(B, 64, 64, 64)` after FAB — matches existing merge input.

**FAB implementation:**
```python
class FrequencyAttentionBlock(nn.Module):
    """
    FFNet-style frequency-domain attention.
    Learns a per-channel complex spectral filter K in rfft2 space.
    Initialized as identity (weight_real=1, weight_imag=0).
    Spatial size (H, W) must match forward input — fixed for 64×64 streams.
    """
    def __init__(self, channels: int, h: int, w: int):
        super().__init__()
        w_h = w // 2 + 1
        self.weight_real = nn.Parameter(torch.ones(1, channels, h, w_h))
        self.weight_imag = nn.Parameter(torch.zeros(1, channels, h, w_h))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_f = torch.fft.rfft2(x, norm='ortho')
        weight = torch.complex(self.weight_real, self.weight_imag)
        x_f = x_f * weight
        return torch.fft.irfft2(x_f, s=(H, W), norm='ortho')
```

FAB instances in HFCFBranch (all streams at 64×64):
```python
self.fab_low  = FrequencyAttentionBlock(64, 64, 64)
self.fab_mid  = FrequencyAttentionBlock(64, 64, 64)
self.fab_high = FrequencyAttentionBlock(64, 64, 64)
```

`_initialize_weights` must NOT be called on FABs (their init is intentional identity).

---

## Change 3 — `losses.py`: Fix HFMaskedFFTLoss + Add MSSSIMLoss

**File:** `src/models/cfrwd/losses.py`

### 3a. Fix `HFMaskedFFTLoss.forward()` (line 209)

The `.mean()` divides by all `H × (W//2+1)` frequency bins including ~75% that are zero-masked. This dilutes the HF supervision signal ~4×.

```python
# Before:
return ((pred_f - tgt_f).abs() * hf_mask).mean()

# After (masked mean — divide only by active HF bins):
diff_hf = (pred_f - tgt_f).abs() * hf_mask
n_active = hf_mask.sum() * pred_f.shape[0] * pred_f.shape[1]  # ×B ×C
return diff_hf.sum() / n_active.clamp(min=1)
```

### 3b. Add `MSSSIMLoss`

MS-SSIM (Wang et al., 2003) measures structural similarity across 5 scales. Directly targets the SSIM gap (0.186 vs paper 0.562). The 5-scale structure is architecturally aligned with the multi-scale DWT decomposition in HFCFBranch.

```python
from pytorch_msssim import ms_ssim

class MSSSIMLoss(nn.Module):
    """
    Multi-Scale SSIM loss. Loss = 1 - MS-SSIM (so 0 is perfect).
    Images must be in [-1, 1] — data_range=2.0.
    win_size=11, K=(0.01, 0.03) are standard defaults.
    """
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return 1.0 - ms_ssim(pred, target, data_range=2.0, size_average=True)
```

`pytorch-msssim` is a lightweight dependency (no new heavy models). Add to `requirements.txt`.

---

## Change 4 — `factory.py`: Wire new criterions

**File:** `src/models/cfrwd/factory.py`

Add to `build_criterions`:
```python
criterions["MSSSIM"] = MSSSIMLoss()
```

`HF_AUX` criterion stays (`HFMaskedFFTLoss` — now with fixed mean).

---

## Change 5 — `main.py`: Routing penalty, 4th AdaptiveLoss component, new logging

**File:** `src/models/cfrwd/main.py`

### 5a. Routing penalty — soft L2 toward balance

```python
# Before (hinge, permanently inactive; also computed H_spatial unnecessarily):
eps = 1e-8
H_spatial = -(fusion_weights * (fusion_weights + eps).log()).sum(dim=1)
loss_routing = F.relu(0.347 - H_spatial.mean()) * self.cfg.loss.routing_entropy_weight

# After (soft L2, always active; remove H_spatial computation entirely):
loss_routing = (fusion_weights[:, 1].mean() - 0.5).pow(2) * self.cfg.loss.routing_balance_weight
```

`routing_entropy_weight` → `routing_balance_weight` in config (renamed to match new formula semantics).
`H_spatial` variable and its computation are removed from `training_step`.
`loss/routing_entropy` log key **remains** — it still logs `fusion_weights` entropy for diagnostics, but computed inline: `H = -(fusion_weights * (fusion_weights + 1e-8).log()).sum(dim=1).mean()`.

### 5b. AdaptiveLoss — add MS-SSIM as 4th component

G-update uses `[loss_l1, loss_focal_freq, loss_lpips, loss_msssim]` in `AdaptiveLoss`.
`adaptive_loss` must be re-initialized with `n=4` learnable etas (was `n=3` in cfrwd-37).
**Verify `AdaptiveLoss.__init__` signature in `losses.py`** and update the constructor call in `main.py` or `factory.py` accordingly.

```python
loss_msssim = self.criterions["MSSSIM"](fake_opt, real_opt)
loss_recon  = self.adaptive_loss([loss_l1, loss_focal_freq, loss_lpips, loss_msssim])
```

### 5c. New TensorBoard logging

**In `training_step` (G-update block), after `return_branches=True` forward:**
```python
# HFCF branch health — key collapse indicators
self.log('hfcf/out_spatial_std', hfcf_out.std(dim=[2, 3]).mean())
self.log('hfcf/out_mean',        hfcf_out.mean())

# DWTBlock subband energies (detach — diagnostic only)
# Attribute path: verify via CFRWDGenerator → HFCFBranch → DWTBlock attr name
with torch.no_grad():
    g1, g2, g3 = self.netG.hfcf_branch.dwt(real_sar)   # verify attr names in gen.py
    self.log('hfcf/g1_energy', g1.pow(2).mean())
    self.log('hfcf/g2_energy', g2.pow(2).mean())
    self.log('hfcf/g3_energy', g3.pow(2).mean())

# New AdaptiveLoss component
self.log('train/loss_msssim',  loss_msssim)
self.log('loss/eta_msssim',    self.adaptive_loss.eta[3])
self.log('loss/w_msssim',      torch.exp(-self.adaptive_loss.eta[3]))
```

Note: attribute path `self.netG.hfcf_branch.dwt` is indicative — **verify exact names in `gen.py`** during implementation. The DWTBlock is the `HFCFBranch` field that wraps `DaubechiesDown`.

**In `on_train_epoch_end`:**
```python
# FAB filter diagnostics (logged once per epoch)
# fab_low/mid/high are the FAB instances added to HFCFBranch — verify attr names
hfcf = self.netG.hfcf_branch  # verify attr name in CFRWDGenerator
for tag, fab in [('low', hfcf.fab_low), ('mid', hfcf.fab_mid), ('high', hfcf.fab_high)]:
    with torch.no_grad():
        mag = torch.complex(fab.weight_real, fab.weight_imag).abs()
        self.log(f'fab/filter_mag_{tag}', mag.mean())
        self.log(f'fab/filter_std_{tag}', mag.std())
```

**In `validation_step` (accumulate) and `on_validation_epoch_end` (log):**
```python
# In validation_step — after getting fake_opt:
from pytorch_msssim import ms_ssim
val_msssim = ms_ssim(fake_opt, real_opt, data_range=2.0, size_average=True)
self._val_msssim_acc.append(val_msssim.item())

# In on_validation_epoch_end:
self.log('val/msssim', torch.tensor(self._val_msssim_acc).mean())
self._val_msssim_acc = []
```

Add `self._val_msssim_acc: list[float] = []` in `__init__`. No extra torchmetrics object needed.
Note: `val/msssim` is raw MS-SSIM (higher = better, range [0,1]), NOT `1 - ms_ssim` (which is the training loss).

---

## Change 6 — `config.yaml`: 3 parameter changes

```yaml
loss:
  aux_hfcf_weight_start: 0.5      # was 0.3 — stronger initial HF supervision signal
  aux_hfcf_weight_end: 0.1        # unchanged
  routing_balance_weight: 0.1     # replaces routing_entropy_weight: 0.005
                                  # renamed + recalibrated for soft L2 formula
  hf_freq_threshold: 0.25         # unchanged

system:
  tb_version: "cfrwd-38"
  resume_ckpt: null
```

`routing_entropy_weight` key is removed. All other params unchanged.

---

## Expected Behavior After Changes

| Metric / Behavior | cfrwd-37 (broken) | cfrwd-38 (expected) |
|---|---|---|
| `hfcf/out_spatial_std` | ~0 (grey blob) | >0.05 from ep5 |
| `hfcf/out_mean` | ~0.5 (tanh midpoint) | ~0.0 (balanced) |
| `fusion/w_hfcf` | 0.55→0.72 (drifting up) | Stabilizes 0.45–0.60 |
| `loss/routing_loss` | 0.0 (permanently) | Active from ep0, ~0.005 |
| `loss/hfcf_aux` | 0.012 (flat, 4× diluted) | Starts ~0.05, decays |
| `fab/filter_mag_*` | N/A | Deviates from 1.0 after ep10 |
| `val/ssim` | 0.186 | Target ≥0.35 |
| `val/psnr` | 15.14 (plateau) | Target ≥17.0 post ep200 |
| `val/msssim` | N/A | Target ≥0.60 |
| `eta_focal_freq` | −0.98 (stable) | Remains stable ~−1.0 to −1.5 |

---

## Files Changed

| File | Change | Lines (est.) |
|------|--------|-------------|
| `src/models/cfrwd/gen.py` | `DaubechiesDown`, update `DWTBlock`, add `FrequencyAttentionBlock`, insert FAB in `HFCFBranch` | ~80 |
| `src/models/cfrwd/losses.py` | Fix `HFMaskedFFTLoss.forward()`, add `MSSSIMLoss` | ~25 |
| `src/models/cfrwd/factory.py` | Add `'MSSSIM'` criterion | ~5 |
| `src/models/cfrwd/main.py` | Routing penalty, 4th eta, new logging, `val/msssim` metric | ~40 |
| `src/models/cfrwd/config.yaml` | 3 params | ~5 |
| `requirements.txt` | Add `pytorch-msssim` | ~1 |

**Total:** ~155 lines of new/changed code across 6 files.

---

## Verification Checklist

- [ ] `python src/models/cfrwd/gen.py` — smoke test passes; verify `DaubechiesDown` output shapes match `HaarDown`
- [ ] `python src/models/cfrwd/losses.py` — smoke test: `HFMaskedFFTLoss` masked mean > plain mean value
- [ ] `python src/models/cfrwd/discriminator.py` — unchanged, still passes
- [ ] `hfcf/out_spatial_std > 0` from ep0 in TensorBoard
- [ ] `loss/routing_loss > 0` from ep0 (soft L2, always active)
- [ ] `fab/filter_mag_*` deviates from 1.0 after ep5–10
- [ ] `val/msssim` appears in TensorBoard validation metrics
- [ ] `fusion/w_hfcf` does not exceed 0.65 by ep30
- [ ] `eta_focal_freq` does not go below −2.0 (stability check)
- [ ] Visual: `hfcf_out` column in epoch images shows non-grey content by ep10

---

## Non-Goals (explicit)

- NSST (Non-Subsampled Shearlet Transform) — deferred to cfrwd-39
- CFRBranch modifications
- Discriminator changes
- GAN loss or FM loss weight changes
- Any change to the pix2pix model (legacy, read-only)
