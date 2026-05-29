# llwt_v5 — Self-Aligning Wavelet GAN with Physics-Anchored Alignment (SAWG-Φ)

**Date:** 2026-05-29
**Module:** `src/models/llwt_v5` (fresh copy of `llwt_v45`; old diffusion `llwt_v5` removed)
**Status:** Design — approved, pending spec review → implementation plan

---

## 1. Problem

Paired SEN1-2 SAR↔optical patches are geometrically **misregistered** (sub-pixel to
several-pixel shifts; sensor parallax, terrain, resampling). Every pixel-fidelity loss in
`llwt_v45` (`per_band` L1, MS-SSIM, FFL, LPIPS) compares the generated optical against the
ground-truth optical **at the same coordinates**. When the pair is shifted, the loss
penalizes that geometric offset as if it were a *content* error. The generator's
least-cost response is to **hedge with blur** — smear high-frequency structure so the
misaligned penalty averages out.

Measured ceiling (llwt_v45 / v0.4.6 runs): val SSIM plateaus ~0.23, PSNR ~15–18 dB. PatchNCE
(a *soft* misalignment-robust contrastive loss) helps but does not remove the cap.

## 2. Core claim / novelty

**Register the target to the prediction before measuring fidelity, and anchor that
registration to electromagnetic scattering centers — the only reliable cross-modal landmarks
under speckle.**

Two fused contributions:
- **① Self-Aligning Wavelet GAN:** a learned dense deformation field φ warps the GT optical
  into the generator's predicted geometry; φ is estimated in the **speckle-robust LL (low-freq)
  Haar band**. Pixel-fidelity losses then compare against the *aligned* target → G's training
  signal stops punishing misalignment → G learns true structure instead of hedging blur.
- **② Physics-Anchored Alignment (Φ-GAN-inspired):** photometric LL-correlation is ambiguous
  on flat/repetitive regions. Bright persistent SAR scatterers (Point Scattering Centers) mark
  real physical structure that *must* exist in the optical. PSCs constrain φ where registration
  is physically trustworthy, plus a backscatter↔structure consistency loss suppresses
  hallucinated detail where SAR is flat.

This fuses into **one coherent mechanism**, not two bolted-on parts: speckle-robust LL gives the
coarse warp; scattering-center correspondence makes it physically correct.

## 3. Why the headline metric (raw-GT) improves

The aligner fixes the generator's **training signal**, not the deployed output. With a clean
(aligned) target, G is no longer rewarded for blur and learns sharp, correct structure →
**raw-GT PSNR/SSIM/LPIPS rise vs the llwt_v45 baseline.** That delta is the thesis result.

The *aligned-GT* metrics are a **diagnostic only** — they quantify residual misregistration
and serve as an upper bound "given perfect registration." They are NOT the headline: at
inference there is no GT to align to, so the deployed output is `fake` in SAR geometry.
Reported honestly as a diagnostic.

## 4. Architecture

Generator (`LLWv4Generator`), discriminator, and the whole loss/optimizer stack are inherited
from `llwt_v45` unchanged. New pieces only.

### 4.1 `align.py` (new module)

**`DeformationAligner`**
- Input: `fake_LL`, `opt_LL` — the LL (low-frequency) Haar subband of fake and GT optical,
  at H/2 (reuse the existing `HaarDown`). LL-only by decision (speckle-robust; clean ablation;
  not LL+features, which would entangle the wavelet-grounded claim).
- Body: small conv encoder on `concat(fake_LL, opt_LL)` → predicts φ at **H/4**
  (`(B, 2, H/4, W/4)`) → bilinear-upsample to full H. Deformation fields are low-frequency, so
  predicting coarse + upsampling is cheaper and naturally smooth (implicit regularization).
- Output bounding: `φ = tanh(raw) * max_disp_px` (default 8 px, normalized to grid coords).
- **Zero-init final conv → φ = 0 → identity warp at step 0.** Preserves the warm-start contract:
  a v45 checkpoint loaded into v5 behaves identically at step 0.

**`warp(img, φ)`** — `F.grid_sample(img, identity_grid + φ, mode='bilinear',
padding_mode='border', align_corners=False)`.

**`psc_detect(sar) -> M_psc`** — Point Scattering Center heatmap.
- Source: **despeckled SAR** (reuse the SARAdapter log-domain adaptive-Lee output; or compute
  inline) — speckle suppressed so maxima are true scatterers, not noise spikes.
- Detect: local maxima above an adaptive threshold (e.g. mean + k·std), keep top-K (default 64)
  per patch.
- Render: Gaussian-splat the K points → soft heatmap `M_psc (B,1,H,W)` ∈ [0,1].
- **Deterministic, non-learnable** for v1 — cheap, interpretable, no extra params. A learnable
  PSC head is explicit future work.

### 4.2 `losses.py` additions

**`DeformationRegLoss(φ)`** — *load-bearing.* Without it φ collapses: an unconstrained warp can
move GT to match `fake` exactly, driving recon loss → 0 while G learns nothing.
- TV-smoothness: `mean(|∇_x φ| + |∇_y φ|)` (weight `reg_smooth_weight`, default 10.0).
- Magnitude: `mean(‖φ‖²)` (weight `reg_mag_weight`, default 1.0) — keeps shifts small/rigid-ish.

**`PSCAnchorLoss` (contribution A)** — grounds φ in physics.
- Fidelity re-weighting: scale the per-pixel fidelity residual by `(1 + λ·M_psc)` so alignment
  and match are enforced hardest at scattering centers.
- Correspondence term: maximize `corr(M_psc, |∇ opt_aligned|)` — scatterers should land on
  optical structure after warping.
- Weight `psc_anchor_weight` (default 1.0).

**`BackscatterStructureLoss` (contribution B)** — anti-hallucination, no new module (~20 lines).
- Low SAR backscatter (flat: water, road) → penalize `|∇ fake|` (no invented edges).
- High SAR backscatter → encourage optical structure presence.
- Weight `bsc_weight` (default 0.5).

### 4.3 `main.py` `training_step` routing

```
fake [, predicted_sub] = G(sar)             # return_internals iff per_band active
fake_LL = HaarDown(fake).LL                  # H/2
opt_LL  = HaarDown(opt).LL
phi        = aligner(fake_LL, opt_LL)
opt_aligned = aligner.warp(opt, phi)
M_psc      = psc_detect(sar)                 # detached / no-grad
```

| Loss | Target |
|---|---|
| `per_band` (vs `HaarDown(opt_aligned)`), MS-SSIM, FFL, LPIPS | **`opt_aligned`** |
| GAN main + FM (discriminator) | **`opt`** (raw — realism supervision stays honest; D never sees a warped GT) |
| PatchNCE | **`opt`** (raw — soft align complements the hard aligner; avoid double-correcting) |
| `DeformationRegLoss(phi)`, `PSCAnchorLoss`, `BackscatterStructureLoss` | new terms |

Aligner parameters join the **`opt_g` param group** (small net, fewer moving parts). Split into
a separate optimizer only if training destabilizes.

### 4.4 Validation — dual evaluation

Compute PSNR/SSIM/LPIPS/FID **twice**:
- **vs raw `opt`** → headline, comparable to all prior work and the llwt_v45 baseline.
- **vs `opt_aligned`** → diagnostic, "given registration" upper bound.

Log both (e.g. `val/psnr`, `val/psnr_aligned`). Thesis frames raw as the result, aligned as the
misalignment-cost quantifier.

## 5. Config (`align:` block)

```yaml
align:
  enabled:            true     # false -> llwt_v5 == llwt_v45 exactly (baseline ablation)
  max_disp_px:        8        # tanh-bounded max deformation
  phi_resolution:     4        # predict phi at H/4, upsample to H
  reg_smooth_weight:  10.0     # TV smoothness on phi (load-bearing)
  reg_mag_weight:     1.0      # magnitude penalty on phi
  psc_anchor_weight:  1.0      # contribution A
  psc_topk:           64       # scattering centers per patch
  bsc_weight:         0.5      # contribution B
```

## 6. Warm-start

Load the llwt_v45 best checkpoint → v5 G + D via the existing `weights_ckpt` path
(`strict=False`; obs 2462 confirmed v4→v45 loads with 0 missing/unexpected keys, identical
generator). Aligner starts fresh, zero-init → identity warp → step-0 output ≡ v45.

## 7. Ablation ladder (one variable at a time)

1. `align.enabled=false` → **≡ llwt_v45 baseline**
2. align on, `psc_anchor_weight=0`, `bsc_weight=0` → **pure photometric self-alignment**
3. + A (`psc_anchor_weight>0`) → physics-anchored alignment
4. + B (`bsc_weight>0`) → + backscatter-structure consistency
5. full (A + B)

Primary ablation = align on/off. Secondary = physics-anchor on/off.

## 8. Smoke tests (CPU, no data)

- `DeformationAligner` zero-init → φ ≈ 0, `warp(opt, 0) == opt` (identity).
- `warp` output shape `(B,3,H,W)`, in-range.
- `psc_detect` returns `(B,1,H,W)` ∈ [0,1], K maxima present.
- `DeformationRegLoss`, `PSCAnchorLoss`, `BackscatterStructureLoss` finite on random input.
- Full `training_step` on one mock batch runs end-to-end; step-0 output ≡ v45 (zero-init).

## 9. Scope

1 new module (`align.py`: aligner + PSC detector) + 3 losses + ~40-line `training_step` edit +
dual-eval validation + config block. Generator/discriminator untouched. Master-sized, single
clean primary ablation.

## 10. Out of scope (future work)

- Learnable PSC parameter-estimation head (full Φ-GAN PSC model).
- Differentiable optical→SAR speckle forward model / cycle-consistency.
- PSC-aware discriminator conditioning (deferred — D-dominance risk observed in QXS run).
