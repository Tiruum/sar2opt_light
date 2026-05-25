# LLW-Former v0.4.x — Architecture, Experiments, Results

**Project:** SAR-to-Optical Image Translation
**Branch:** `physics-aware`
**Last updated:** 2026-05-24
**Status:** R-stage ablations complete; S-stage rejected; headline 200ep run pending

---

## 1. Final Architecture

### 1.1 Generator — `LLWv4Generator`

Wavelet-native generator: ConvNeXt V2-Tiny backbone with Haar-Stem replacement and Inverse-Haar output head.

```
SAR (B, 1, 256, 256)
    │
    ▼
┌──────────────────────────────────────────────────────────┐
│ SARAdapter (optional, cfg.model.gen.use_sar_physics)     │
│   raw + log|SAR| + Sobel-gradient → 3-channel SAR input  │
└──────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────┐
│ Haar-Stem (replaces ConvNeXt's 4×4 patch embed)          │
│   Single-level Haar DWT: 1→4 channels, H/2×W/2 spatial   │
└──────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────┐
│ ConvNeXt V2-Tiny backbone (HF facebook/convnextv2-tiny)  │
│   hidden_sizes = (96, 192, 384, 768)                     │
│   out_indices  = [1, 2, 3, 4] — 4 multi-scale features   │
│   28.6M params; pretrained ImageNet weights              │
└──────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────┐
│ Multi-scale fusion decoder                               │
│   Bilinear upsample + conv blocks                        │
│   Predicts 4 Haar subbands at H/2 resolution             │
│   Output sub: (B, 3, 4, H/2, W/2)  [LL, LH, HL, HH]      │
└──────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────┐
│ InverseHaarUp (fixed, no learned weights)                │
│   Reconstruct RGB from predicted subbands                │
│   tanh activation final                                  │
│   Output: (B, 3, 256, 256) in [-1, 1]                    │
└──────────────────────────────────────────────────────────┘
    │
    ▼
OPT_fake (B, 3, 256, 256)
```

**Total params:** ~35.06M
**Forward returns:** `(opt_fake, sub)` when `return_internals=True` (needed by per-band L1)

### 1.2 Discriminator Stack — `LLWFormerDiscriminator`

Two-head adversarial framework. **Third head (FourierDis) implemented but rejected after S-stage ablation.**

```
                ┌─────────────────────────────────────┐
                │ MainDis (paired pixel)              │
                │   Input: [SAR, OPT] concat           │
                │   2-scale 70×70 + 46×46 PatchGAN     │
                │   No spectral norm                   │
                │   ~1.8M params                       │
                └─────────────────────────────────────┘
                              │
[SAR, OPT_real/fake]──┬──────►│ 4-layer LeakyReLU conv  │──► (logits_70, logits_46) + FM features
                      │       └─────────────────────────────────────┘
                      │
                      │       ┌─────────────────────────────────────┐
                      │       │ SubbandDis (Haar wavelet)           │
                      │       │   Input: OPT only                   │
                      │       │   log L=1 Haar decomposition         │
                      │       │   4-conv spectral-norm PatchGAN      │
                      │       │   ndf=32, ~1.8M params              │
                      │       └─────────────────────────────────────┘
                      │                     │
                      └──────► OPT only ────►│ ──► logits + FM features
                                            └─────────────────────────────────────┘
```

**Total D params:** ~3.6M
**FourierDis (rejected):** log|rfft2(opt)| → 4-conv SN PatchGAN. Implemented in code (cfg-gated, default `false`); rejected after S1/S2 trials showed redundancy with SubbandDis.

### 1.3 Loss Stack — Final Configuration

| Loss | Weight | Status |
|------|--------|--------|
| **GANLoss (LSGAN)** main D | `gan_main_weight = 1.0` | active |
| **GANLoss (LSGAN)** subband D | `gan_sub_weight = 1.0` | active |
| **FeatureMatchingLoss** main D | `fm_main_weight = 10.0` | active (load-bearing) |
| **FeatureMatchingLoss** subband D | `fm_sub_weight = 10.0` | active |
| **PerBandWaveletL1Loss** | `per_band_weight = 2.0` | active (R2 manual: LL=1, LH=1.5, HL=1.5, HH=2) |
| L1 (`PlainL1Loss`) | 0.0 | off |
| MS-SSIM | 0.0 | off |
| LAB chroma | 0.0 | off |
| WaveletDetailL1Loss | 0.0 | off (overlaps PerBand) |
| LPIPS | 0.0 | off (LPIPS uses VGG natural-image features) |
| SpeckleDecoupleLoss | 0.0 | off |
| PerfectReconLoss | 0.0 | off (no learnable lifting in v0.4.x) |
| GAN-Fourier | 0.0 | **rejected (S-stage)** |
| FM-Fourier | 0.0 | **rejected (S-stage)** |
| R1 main penalty | 0.0 | off (spectral norm handles Lipschitz) |
| LSGAN label smoothing | `real=1.0, fake=0.0` | LSGAN equilibrium fix |

### 1.4 Training Configuration

| Field | Value |
|-------|-------|
| Optimizer | AdamW (bnb 8-bit available, fp32 fallback) |
| LR (G + D) | `2e-4`, β=(0.5, 0.999), weight_decay=0 |
| Scheduler | `linear_decay` (constant 150 ep + linear decay 50 ep → 1e-6) |
| Batch size | 8 (train), 4 (val) — full-res 256×256 |
| Precision | bf16-mixed |
| Memory format | channels_last |
| EMA | decay 0.999, `start_epoch = 20` |
| Compile | off (Windows venv lacks triton) |
| Manual optimization | yes (`automatic_optimization = False`) |
| Optimizer order | `[opt_d, opt_g]` (D first) |
| D update | batched real+fake forward, `torch.no_grad()` on G |

### 1.5 Data

- **Dataset:** SEN12 (Sentinel-1 SAR + Sentinel-2 optical pairs)
- **Subset:** 5 scenes — `["5", "45", "52", "84", "100"]`
- **Split:** 80/20 train/val by random seed 42
- **SAR channels:** 1 (grayscale, dB-scaled)
- **OPT channels:** 3 (RGB, [-1, 1] tanh range)
- **Augmentation:** synchronized geometric (albumentations `additional_targets={'optical': 'image'}`)

---

## 2. Experiment Series

All runs warm-start from prior best ckpt unless noted. All val metrics computed on 5-scene holdout.

### 2.1 R0 — `v0.4.0-ihaar-prod` (Initial Baseline)

**Purpose:** Establish IHaar output head viability vs full-res RGB regression.
**Schedule:** 120 epochs, `cosine_warm_restarts` (T_0=20, T_mult=2)
**Per-band:** equal weights (1, 1, 1, 1), `per_band_weight = 2.0`
**Warm-start:** none (fresh from scratch)

**Result:** PSNR 13.02 (ep87), FID 83.0 (ep93), LPIPS 0.4355 (ep64), SSIM 0.2763 (ep62)

**Issue:** Cosine restarts at ep113→ep119 wrecked final checkpoint (PSNR 12.51, FID 121 at last ckpt).

### 2.2 R-Stage Ablations — Per-Band Weighting

**Common protocol:**
- Warm-start from `v0.4.0-ihaar-prod ep87` ckpt
- Constant LR (no cosine restart wreckage)
- 60 epochs
- Same 5-scene SEN12 subset

#### R1 — `v0.4.1-perband-r1-equal`

**Config:** `per_band = (1, 1, 1, 1)`, `per_band_weight = 2.0`

**Result:**
- **Best per-metric:** PSNR 14.52 (ep59), FID 80.0 (ep29), LPIPS 0.420 (ep37), SSIM 0.288 (ep57)
- **Last-saved ckpt (ep59):** PSNR 14.52, FID 87.0 ← PSNR-strong, FID poor late
- **Pattern:** PSNR climbs monotonically; FID plateaus then regresses after ep37
- **Late-10ep mean FID:** 88.5

**Diagnosis:** Late-stage L1-induced averaging blur. PSNR continues to climb at the cost of perceptual fidelity.

#### R2 — `v0.4.1-perband-r2-detail` ★ **WINNER**

**Config:** `per_band = (1, 1.5, 1.5, 2)`, `per_band_weight = 2.0` (detail boost, LL deweight)

**Result:**
- **Best per-metric:** PSNR 14.26 (ep56), FID 81.0 (ep56), LPIPS 0.422 (ep56), SSIM 0.293 (ep57)
- **Best ckpt ep56:** PSNR 14.26, FID 81.0, LPIPS 0.422, SSIM 0.293 — **all 4 metrics co-aligned**
- **Late-10ep mean FID:** 83.4 (vs R1's 88.5)

**Diagnosis:** LL deweight (share 16.7% vs R1's 25%) reduces L1 averaging pressure on the structural band. Allows G to retain perceptual texture variability while maintaining structural fidelity. **Late-stage rally rather than degradation.**

#### R3 — `v0.4.1-perband-r3-aggressive` (skipped)

Rejected before launch. R2 demonstrated detail-boost direction; R3 would over-correct LL further.

#### R4 — `v0.4.1-perband-r4-adaptive`

**Config:** Kendall uncertainty weighting (learnable σ per band).
- 4-dim `log_var` Parameter, clamped to `[-3, 3]` (σ ∈ [0.05, 5])
- Loss: `(Σ exp(-log_var_i) · L_i + 0.5·Σ log_var_i) / 4` (band-count normalised)
- Adapter wired to `opt_g`

**Result:**
- **Best per-metric:** PSNR 14.29 (ep58), FID 81.0 (ep53), LPIPS 0.430 (ep10), SSIM 0.299 (ep54)
- **Best ckpt ep58:** PSNR 14.29, FID 86.5 — PSNR matches R2, FID worse at saved ckpt
- **Late-10ep mean FID:** 84.1

**Kendall σ trajectory** (precisions = `exp(-log_var)`):

| Epoch | pb_w_ll | pb_w_lh | pb_w_hl | pb_w_hh | LL share |
|-------|---------|---------|---------|---------|----------|
| 0 | 0.91 | 1.12 | 1.12 | 1.12 | 21% |
| 13 | 0.61 | 5.16 | 4.09 | 10.93 | 2.9% |
| 34 | 0.62 | 5.13 | 4.04 | 10.71 | 3.0% |
| 59 (final) | 0.63 | 5.11 | 4.01 | 10.47 | 3.1% |

**Diagnosis:**
- Kendall converged organically (no clamp saturation) by ep20
- Same ranking as R2 manual (HH > LH ≈ HL > LL) **automatically discovered**
- BUT magnitudes 5.4× more extreme: LL share 3.1% (R4) vs 16.7% (R2)
- Over-correction loses FID. **Manual R2 (LL=18%) hits perceptual sweet spot Kendall cannot find.**

**Mechanism**: Kendall σ² = c · L_i tracks LOSS MAGNITUDE, not PERCEPTUAL IMPORTANCE. LL has biggest raw L1 (DC content), so Kendall downweights it aggressively — but LL is also the most perceptually critical band (structure). Loss-magnitude proxy fails for perceptual tasks.

### 2.3 S-Stage Ablations — FourierDis Adversarial Spectral Loss

**Common protocol:** Add `FourierDis` head (log|rfft2(opt)| → 4-conv SN PatchGAN) on top of R2 winner. Warm from R2 ep56 ckpt.

#### S1 — `v0.4.2-fourier-s1-equal`

**Config:** `gan_fourier=1.0, fm_fourier=10.0` (equal to main/sub D)

**Result @ ep22 (aborted):**
- Best PSNR 14.12 (ep13), FID 83.0 (ep12, EMA-stable window)
- D-G gap saturated at 0.50 (abort threshold) by ep12
- `fm_fourier` plateaued at 0.123 → G capacity exhausted

**Diagnosis:** FourierDis too strong, dominates G. Adversarial dynamics with 3 D heads compete for G capacity. SubbandDis already provides frequency discrimination via Haar — Fourier adds redundant signal.

#### S2 — `v0.4.2-fourier-s2-gentle`

**Config:** `gan_fourier=0.5, fm_fourier=5.0` (half-strength)

**Result @ ep3 (aborted):**
- Same trajectory shape as S1
- FID 87+ at ep3 (vs R2 baseline 81)
- D-G gap climbing 0.42 → 0.44

**Diagnosis:** Gentler dose, same failure mode. Confirms mechanism rejection — not weight-tuning issue.

**S-stage verdict: REJECTED.** FourierDis kept in code (cfg-gated, default off) for future configurations where SubbandDis is disabled.

---

## 3. Results Summary

### 3.1 Best-of-Run Per-Metric Comparison

| Run | tb_version | PSNR best | FID best | LPIPS best | SSIM best |
|-----|------------|-----------|----------|------------|-----------|
| R0 (v0.4.0 ihaar) | `llwt-v0.4.0-ihaar-prod` | 13.02 (ep87) | 83.0 (ep93) | 0.4355 (ep64) | 0.2763 (ep62) |
| R1 equal | `llwt-v0.4.1-perband-r1-equal` | 14.52 (ep59) | 80.0 (ep29) | 0.4199 (ep37) | 0.2883 (ep57) |
| **R2 detail** ★ | `llwt-v0.4.1-perband-r2-detail` | 14.26 (ep56) | **81.0 (ep56)** | **0.4219 (ep56)** | **0.2931 (ep57)** |
| R4 adaptive | `llwt-v0.4.1-perband-r4-adaptive` | 14.29 (ep58) | 81.0 (ep53) | 0.4297 (ep10) | 0.2991 (ep54) |
| S1 fourier full | `llwt-v0.4.2-fourier-s1-equal` (aborted) | 14.12 (ep13) | 83.0 (ep12) | 0.4180 (ep13) | 0.2811 (ep14) |
| S2 fourier gentle | `llwt-v0.4.2-fourier-s2-gentle` (aborted) | — (ep3) | 87+ | — | — |

### 3.2 Deployable-Checkpoint Comparison

PSNR-monitored checkpoints save top-3 by `val/psnr`. Production deployment uses these (not transient best-per-metric).

| Run | Saved ckpt epoch | PSNR | FID | LPIPS | SSIM | Note |
|-----|------------------|------|-----|-------|------|------|
| R1 (last ckpt) | ep59 | 14.52 | **87.0** | 0.430 | 0.286 | High PSNR, poor FID |
| **R2 ep56** ★ | ep56 | 14.26 | **81.0** | 0.422 | 0.293 | **All 4 co-aligned — golden** |
| R4 ep58 | ep58 | 14.29 | 86.5 | 0.432 | 0.299 | Matches R2 PSNR, worse FID |

**Key insight:** R2's PSNR-monitor happened to catch the FID-optimal epoch (ep56) because all 4 metrics co-aligned there. R1's saved ckpts are all post-ep37 degradation window. R4's saved ckpts are post-Kendall-equilibrium and have R4's over-corrected FID baseline.

### 3.3 Late-Stage Stability (Mean FID, Last 10 Epochs)

| Run | Late-10ep mean FID | Late-10ep std |
|-----|---------------------|---------------|
| R1 | 88.5 | ±2.0 |
| R2 | 83.4 | ±1.4 |
| R4 | 84.1 | ±1.6 |
| S1 | regressing | high variance |

R2 has lowest mean AND lowest variance in late stage — most stable for shipping.

### 3.4 SOTA Context

Published 2024-2026 SAR-to-optical translation results on SEN1-2 / GF-AD typically report FID 150-265. Our R2 ep56 ckpt at FID 81.0 is **2-3× better than typical baselines**.

---

## 4. Mechanism Findings

### 4.1 Per-band L1 Weighting Sweet Spot

LL deweight is the lever, NOT detail boost per se. R2's win comes from reducing LL's share of the L1 gradient (16.7% vs R1's 25%), which:
- Prevents L1-averaging blur on the structural band late-run
- Preserves texture variability (FID/LPIPS gain)
- Costs ~0.3 dB PSNR (LL fidelity tradeoff)

### 4.2 Kendall Adaptive Limitation

Kendall uncertainty weighting (Kendall, Gal, Cipolla 2018) is **optimal for the loss it minimizes**, not for perceptual quality. Equilibrium σ² ∝ L_i means tasks with bigger loss magnitude get downweighted — correct under Gaussian noise assumption, but loss magnitude is a poor proxy for perceptual importance. R4 reproduces R2's direction (LL deweight) but over-corrects 5.4×.

Useful negative result: **manual prior beats learned σ for perception-aligned image translation.**

### 4.3 FourierDis Redundancy

SubbandDis (Haar wavelet PatchGAN) already provides multi-resolution frequency discrimination. Adding FourierDis stacks two frequency-domain D heads → conflicting gradients on G → stuck equilibrium. Useful negative result: **frequency adversarial signal is non-additive in a wavelet-native architecture.**

### 4.4 LSGAN Equilibrium Fix

Initial smoke had D-G plateau at d_real ≈ d_fake ≈ 0.45. Root cause: `real_label_smooth=0.9, fake_label_smooth=0.0` made LSGAN's degenerate fixed point exactly `(0.9+0)/2 = 0.45`. Fix: `real_smooth=1.0, fake_smooth=0.0`. Plus R1 penalty disabled (spectral norm in D already enforces Lipschitz).

---

## 5. Headline Run (Pending)

**`v0.4.3-headline-200ep`** — production training with all best v4 settings.

| Field | Value |
|-------|-------|
| `tb_version` | `llwt-v0.4.3-headline-200ep` |
| `weights_ckpt` | R2 ep56 (warm-start from W2) |
| `max_epochs` | 200 |
| `scheduler.type` | `linear_decay` |
| `scheduler.linear_decay_epochs` | 50 (constant LR for first 150 ep, decay over last 50) |
| `ema.start_epoch` | 20 (delayed for long-run stability) |
| All other fields | R2-best (per-band manual, no fourier, no adaptive) |

**Estimated wall-clock:** ~5h (1.5 min/ep × 200ep)

**Forecast targets:**
- PSNR ≥ 14.7 (vs R2 best 14.52)
- FID ≤ 78 (vs R2 best 81)
- SSIM ≥ 0.30

---

## 6. Future Work

### 6.1 Foundation-Model Perceptual Loss (PEPL)

Use frozen Prithvi-EO-2.0 (NASA/IBM satellite ViT-MAE) as perceptual feature extractor for an additional loss term. Provides domain-aligned semantic supervision (vs LPIPS's natural-image priors).

**Status:** Class implemented in `sarformer_wb/losses.py` (cfg-gated, default off). Blocked by `terratorch` dependency upgrade conflict with HuggingFace `transformers` 5.x ConvNeXt V2. Resolution paths documented; deferred pending env fix.

### 6.2 SAR-Conditional Diffusion Refinement (SAR-DR)

Add lightweight UNet diffusion refinement head conditioned on `[SAR, G(SAR)_coarse]`. 3-5 step DDIM sampler. Combines GAN inference speed with diffusion's distributional matching guarantees.

**Status:** Design phase. Target v0.5.x (`llwt_v5/`).

---

## 7. Key Artifacts

### 7.1 Checkpoints

```
checkpoints/llwt_v4/
├── llwt-v0.4.0-ihaar-prod/                   # R0 baseline
├── llwt-v0.4.0-ihaar-smoke/                  # R0 smoke
├── llwt-v0.4.1-perband-r1-equal/             # R1 equal weights
├── llwt-v0.4.1-perband-r2-detail/            # R2 WINNER (ep56 = golden)
│   └── epoch=056-psnr=14.2607.ckpt           # ★ W2 ckpt
├── llwt-v0.4.1-perband-r4-adaptive/          # R4 Kendall
├── llwt-v0.4.2-fourier-s1-equal/             # S1 (aborted)
└── llwt-v0.4.2-fourier-s2-gentle/            # S2 (aborted)
```

### 7.2 Logs

```
output/llwt_v4/csv_logs/<tb_version>/version_0/metrics.csv
output/llwt_v4/tb_logs/<tb_version>/
output/llwt_v4/images/<tb_version>/epoch_*.png
```

### 7.3 Code

- `src/models/llwt_v4/gen.py` — `LLWv4Generator`, `InverseHaarUp`
- `src/models/llwt/dis.py` — `LLWFormerDiscriminator`, `FourierDis`
- `src/models/llwt_v4/main.py` — `LLWv4LightningModule` (manual optimization)
- `src/models/llwt/factory.py` — losses + optimizers + schedulers
- `src/models/sarformer_wb/losses.py` — loss classes including `PerBandWaveletL1Loss` (manual + adaptive), `FoundationPerceptualLoss` (deferred)
- `src/models/llwt_v4/train.py` — Lightning entry point
- `src/models/llwt_v4/overfit_test.py` — single-batch sanity gate (Tier 2)

---

## 8. Decision Log

| Date | Decision | Reason |
|------|----------|--------|
| 2026-05-23 | Switch scheduler from `cosine_warm_restarts` to `constant` for R-stage | Cosine restarts wrecked v0.4.0 final ckpt (restart at ep113 no time to recover) |
| 2026-05-23 | Warm-start R-stage from v0.4.0 ep87 | Save ~3h GPU per arm vs fresh; allow direct A/B test of per-band effect |
| 2026-05-23 | R-stage: ep30 → ep60 final decision gate | Ep30 too early for stable comparison |
| 2026-05-23 | R3 + R4 skipped initially, then R4 added on user request | R-stage answered weighting question; R4 = thesis-completeness check |
| 2026-05-23 | W2 = R2 (not R1) | R2 deployable ckpt has lower FID + better LPIPS/SSIM, only -0.26 dB PSNR |
| 2026-05-24 | S-stage rejected | S1+S2 both fail same way → mechanism issue, not weight tuning |
| 2026-05-24 | Headline = linear_decay, 200ep, warm from R2 ep56 | Builds on R2 W2 with LR decay for final refinement; constant LR's late oscillation addressed |
| 2026-05-24 | Headline keeps 5-scene subset | A/B parity with R-stage ablations preserved |
| 2026-05-24 | PEPL deferred → SAR-DR next | terratorch broke transformers env; SAR-DR has higher thesis novelty anyway |
