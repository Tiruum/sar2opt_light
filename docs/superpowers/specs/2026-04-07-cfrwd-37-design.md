# cfrwd-37 Design: Natural HFCF via Frequency-Specialized Supervision

**Date:** 2026-04-07  
**Branch:** audit → cfrwd-37 experiment  
**Goal:** Fix HFCF branch collapse and improve generation quality by aligning supervision with architectural role.

---

## Problem Statement

In cfrwd-36, the HFCF (wavelet) branch collapsed: `spatial_std` fell from 0.122 to 0.067 while `w_hfcf` rose to 0.75. The branch produces near-uniform output (edge map expected, grey blob observed).

Three confirmed root causes:

| # | Cause | Evidence |
|---|-------|---------|
| 1 | **Gradient starvation via fusion multiplication** | `∂loss/∂hfcf_feats = gradient × w_hfcf`. When w_hfcf ≈ 0, HFCF starves even though the path exists. `gen.py:602` |
| 2 | **Wrong supervision target** | HFCF receives gradient only from fused output. Asking it to compete with CFR on full-image pixel loss is architecturally wrong. CFR wins. |
| 3 | **FFT eta spiral dominates CFR** | `eta_fft → −3.07` → `w_fft ≈ 21.6`. FFT monopolizes AdaptiveLoss gradient. CFR handles all frequency matching → HFCF redundant. |

**Secondary issues:**
- Generation plateau since epoch ~200; LR decay starts at epoch 400 (too late)
- `return_branches=True` wraps cfr_out/hfcf_out in `torch.no_grad()` — prevents aux loss from having any effect

**Paper baseline (Wei et al., Remote Sens. 2023):**  
- Loss: `L_LSGAN + 10×L_FM` only (no pixel loss)  
- Result: PSNR 18.97 dB, SSIM 0.562, LPIPS 0.399 on SEN1-2 S5  
- WD branch output (Figure 10c): produces **edge map**, not full image  
- WD contribution to PSNR: +0.31 dB (largest impact on dense-urban scenes: +1.62 dB)

**Our cfrwd-36 result:** PSNR 15.42 dB, SSIM 0.195, LPIPS 0.359 (epoch 266 of 800, EMA-bugged checkpoints)

---

## Architecture Status (what is NOT changing)

The HFCFBranch already exceeds the paper's implementation:
- LL2 retained as g1 (paper also keeps LL, contrary to old CLAUDE.md)
- CBAM on all three wavelet streams (paper does not have CBAM)
- Spatial AdaptiveFusion (B×2×H×W softmax) vs paper's scalar coefficient

No architectural surgery. All 4 changes are in loss system, training loop, and config.

---

## Change 1 — `gen.py`: Remove `torch.no_grad()` from `return_branches` path

**File:** `src/models/cfrwd/gen.py`  
**Lines:** ~649–654

**Current:**
```python
if return_branches:
    with torch.no_grad():
        cfr_out = torch.tanh(self.final(cfr_feats))
        hfcf_out = torch.tanh(self.final(hfcf_feats))
    return out, cfr_out, hfcf_out, fusion_weights
```

**New:**
```python
if return_branches:
    cfr_out = torch.tanh(self.final(cfr_feats))
    hfcf_out = torch.tanh(self.final(hfcf_feats))
    return out, cfr_out, hfcf_out, fusion_weights
```

**Rationale:** The `no_grad` context was correct when branch outputs were visualization-only. For cfrwd-37, `hfcf_out` is needed in the training step G-update with full gradients for the aux loss. The `no_grad` context moves to the **caller** — `on_train_epoch_end` in `main.py` wraps its visualization call explicitly.

---

## Change 2 — `losses.py`: Two new loss classes

**File:** `src/models/cfrwd/losses.py`

### 2a. `FocalFrequencyLoss` (replaces `FFTLoss` in `AdaptiveLoss`)

**Problem being solved:** `FFTLoss` computes uniform L1 on FFT magnitude. By epoch 5, `loss_fft ≈ 0.046` (very small). `eta_fft` converges to `−log(1/0.046) ≈ −3.07` → weight ×21.6. This monopolizes the AdaptiveLoss gradient, CFR handles all frequency reconstruction, HFCF becomes redundant.

**Solution:** Focal Frequency Loss (Jiang et al., ICCV 2021) weights each frequency component by its **relative** reconstruction error. Well-reconstructed frequencies get low weight; hard ones get high weight. The eta equilibrium won't spiral to −3 because the loss has inherent per-frequency normalization — the absolute value of FocalFrequencyLoss is stable regardless of how well the model reconstructs easy frequencies.

```python
class FocalFrequencyLoss(nn.Module):
    """
    Jiang et al., 'Focal Frequency Loss for Image Reconstruction and Synthesis', ICCV 2021.

    Per-frequency adaptive weighting: weight_f = |diff_f| / (|pred_f| + eps).
    Amplifies frequencies where the model is failing; suppresses already-correct ones.
    More stable than plain FFT L1 because the weight normalizes by prediction magnitude,
    preventing the absolute loss value from collapsing to near-zero and causing eta spiral.
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_f = torch.fft.rfft2(pred, norm='ortho')
        tgt_f  = torch.fft.rfft2(target, norm='ortho')
        diff   = (pred_f - tgt_f).abs()
        with torch.no_grad():
            weight = diff / (pred_f.abs().detach() + 1e-8)
        return (weight * diff).mean()
```

`FocalFrequencyLoss` replaces `FFTLoss` as the second component in `AdaptiveLoss`:
- `build_criterions` creates `criterions["FOCAL_FREQ"] = FocalFrequencyLoss()`
- `AdaptiveLoss` in training step receives `[loss_l1, loss_focal_freq, loss_lpips]`
- `FFTLoss` is kept in the file but removed from the AdaptiveLoss composition

### 2b. `HFMaskedFFTLoss` (new, for HFCF aux supervision)

**Rationale:** The paper (Figure 10c) shows HFCF output is an **edge map** — it produces high-frequency optical features, not the full image. Supervising `hfcf_out` on the full optical image (pixel L1 or full FFT) is wrong because HFCF competes with CFR on low-frequency content where CFR has more capacity. The correct target is **only** the high-frequency content of `real_opt`.

`freq_threshold=0.25` means: penalize only spatial frequencies above 25% of Nyquist (the edge/texture band). Coarse structure and color (below threshold) are CFR's domain.

```python
class HFMaskedFFTLoss(nn.Module):
    """
    Frequency-band-selective supervision for HFCF branch.

    Penalizes only high-frequency reconstruction error (spatial freq > threshold).
    Used as auxiliary loss on hfcf_out to give the HFCF branch an independent
    training signal aligned with its architectural role (edge/detail recovery).

    freq_threshold=0.25: supervise frequencies above 25% of Nyquist.
    Low-frequency content (color, coarse structure) is excluded — that is CFR's domain.
    """
    def __init__(self, freq_threshold: float = 0.25):
        super().__init__()
        self.freq_threshold = freq_threshold

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_f = torch.fft.rfft2(pred, norm='ortho')
        tgt_f  = torch.fft.rfft2(target, norm='ortho')
        H, W_h = pred_f.shape[-2], pred_f.shape[-1]
        W = (W_h - 1) * 2
        fy = torch.fft.fftfreq(H, device=pred.device).abs()    # H
        fx = torch.fft.rfftfreq(W, device=pred.device)          # W//2+1
        freq_mag = (fy[:, None]**2 + fx[None, :]**2).sqrt()    # H×(W//2+1)
        hf_mask = (freq_mag > self.freq_threshold).float()
        return ((pred_f - tgt_f).abs() * hf_mask).mean()
```

---

## Change 3 — `main.py`: Training step changes

**File:** `src/models/cfrwd/main.py`

### 3a. G update: call `return_branches=True`

In the G-update section of `training_step`, change the generator forward call:

```python
# Before (returns fused output only):
fake_opt, fusion_weights = self.netG(real_sar)

# After (returns branches with full gradients):
fake_opt, cfr_out, hfcf_out, fusion_weights = self.netG(real_sar, return_branches=True)
```

`cfr_out` is not used for any loss — it's available but only for potential future use. `hfcf_out` is used in 3b.

Note: The D-update forward call is under `torch.no_grad()` and uses only `fake_opt` — it remains unchanged: `fake_opt, _ = self.netG(real_sar)`.

### 3b. HFCF auxiliary loss with epoch-scheduled weight

```python
# Auxiliary loss: supervise HFCF on HF content of real_opt
aux_weight = self._aux_hfcf_weight()
loss_hfcf_aux = self.criterions["HF_AUX"](hfcf_out, real_opt) * aux_weight

# Add to g_loss
g_loss = loss_gan * self.loss_weights['gan'] \
       + loss_fm  * self.loss_weights['fm']  \
       + loss_recon                           \
       + loss_hfcf_aux                        \
       + loss_routing  # see 3c
```

`_aux_hfcf_weight()` method on the Lightning module:
```python
def _aux_hfcf_weight(self) -> float:
    """Linear decay: aux_hfcf_weight_start → aux_hfcf_weight_end over max_epochs."""
    start = self.cfg.loss.aux_hfcf_weight_start   # 0.3
    end   = self.cfg.loss.aux_hfcf_weight_end     # 0.1
    frac  = min(self.current_epoch / self.cfg.system.max_epochs, 1.0)
    return start + (end - start) * frac
```

### 3c. Routing entropy regularization

Computes entropy of the B×2×H×W spatial softmax fusion weights. Penalizes when mean entropy falls below 50% of the maximum (log(2) ≈ 0.693 for 2-branch system → threshold = 0.347).

```python
# Routing diversity: penalize if spatial entropy of fusion weights is too low
eps = 1e-8
H_spatial = -(fusion_weights * (fusion_weights + eps).log()).sum(dim=1)  # B×H×W
loss_routing = F.relu(0.347 - H_spatial.mean()) * self.cfg.loss.routing_entropy_weight
```

This is a soft lower bound on routing entropy — zero cost when entropy is healthy, penalizes collapse toward uniform `w_hfcf ≈ 0` or `w_hfcf ≈ 1`.

### 3d. Log new loss terms

Replace `'train/loss_fft'` / `'loss/eta_fft'` / `'loss/w_fft'` keys with focal-freq variants, and add new keys:
```python
'train/loss_focal_freq': loss_focal_freq,
'loss/eta_focal_freq':   self.adaptive_loss.eta[1],
'loss/w_focal_freq':     torch.exp(-self.adaptive_loss.eta[1]),
'loss/hfcf_aux':         loss_hfcf_aux.item(),
'loss/routing_entropy':  H_spatial.mean().item(),
'loss/routing_loss':     loss_routing.item(),
```

Note: `on_train_epoch_end` at `main.py:246` already has `with torch.no_grad()` wrapping the visualization forward — no change needed there.

---

## Change 4 — `config.yaml`: 5 parameter updates

**File:** `src/models/cfrwd/config.yaml`

```yaml
loss:
  fm_weight: 10                    # unchanged — paper validated this value; ablation shows FM is largest contributor
  aux_hfcf_weight_start: 0.3       # NEW: aux loss weight at epoch 0
  aux_hfcf_weight_end: 0.1         # NEW: aux loss weight at epoch 800 (linear decay)
  routing_entropy_weight: 0.005    # NEW: entropy regularization weight
  hf_freq_threshold: 0.25          # NEW: HFMaskedFFTLoss frequency cutoff

scheduler:
  linear_decay_epochs: 600         # was 400. Decay starts epoch 200 (= 800 - 600), not epoch 400.

system:
  tb_version: "cfrwd-37"
  resume_ckpt: null                # fresh start — cfrwd-36 checkpoints have EMA-bugged state
```

---

## Expected behavior after changes

| Metric / Behavior | cfrwd-36 (broken) | cfrwd-37 (expected) |
|---|---|---|
| `fusion/spatial_std` | Falls 0.122→0.067 (uniform routing) | Stays ≥0.08 (spatial specialization) |
| `fusion/w_hfcf` mean | Rises to 0.75 (compensating dead branch) | Stabilizes 0.4–0.6 (meaningful weight) |
| `hfcf_out` visual | Uniform grey | Edge/texture map (matches paper Figure 10c) |
| `loss_hfcf_aux` | N/A | Decreases from ~0.3 → ~0.1 over training |
| `eta_fft` (now `eta_focal_freq`) | −3.07 (weight ×21) | ~−1.5 (weight ×4–5, stable) |
| PSNR plateau | Epoch ~200 | Delayed — LR decay starts epoch 200 |
| Generation quality | PSNR 15.4 dB, LPIPS 0.359 | Target: PSNR ≥17 dB, LPIPS ≤0.35 |

---

## Files changed

| File | Change |
|------|--------|
| `src/models/cfrwd/gen.py` | Remove `torch.no_grad()` from `return_branches` path (lines 652–654, ~3 lines) |
| `src/models/cfrwd/losses.py` | Add `FocalFrequencyLoss` and `HFMaskedFFTLoss` classes (~45 lines) |
| `src/models/cfrwd/factory.py` | Import new losses; replace `'FFT': FFTLoss()` with `'FOCAL_FREQ': FocalFrequencyLoss()` and add `'HF_AUX': HFMaskedFFTLoss()`; update return-type annotation (~8 lines) |
| `src/models/cfrwd/main.py` | G-update: `return_branches=True`; add `_aux_hfcf_weight()` method; add aux/routing losses + logging; rename FFT→focal_freq keys (~40 lines) |
| `src/models/cfrwd/config.yaml` | 5 new parameters + version string + LR schedule change |

Total: ~95 lines of new/changed code across 5 files. No architectural changes.

---

## Verification checklist

- [ ] `python src/models/cfrwd/gen.py` — generator smoke test passes
- [ ] `python src/models/cfrwd/losses.py` — loss smoke test passes  
- [ ] Check `loss_hfcf_aux` appears in TensorBoard from epoch 0
- [ ] Check `fusion/spatial_std` does not monotonically decrease
- [ ] Check `eta_focal_freq` does not go below −2.0 (stability check)
- [ ] Visual: `hfcf_out` in epoch images shows edges, not grey blob
- [ ] PSNR improves past epoch 200 (LR decay effect)
