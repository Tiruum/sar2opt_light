# LLW-Former — Architecture, Experiments, and Results

**Task:** SAR-to-Optical Image Translation (Sentinel-1 → Sentinel-2)
**Model lineage:** `llwt_v4` (wavelet-native conditional GAN) → `llwt_v45` (final, current)
**Branch:** `physics-aware`
**Last updated:** 2026-05-25
**Status:** v0.4.x R-stage ablations complete; v0.4.5 overhaul built, screened, and locked; 200-epoch headline run pending.

---

## Abstract

We address single-channel SAR-to-optical image translation on the SEN1-2 benchmark. Our generator, **LLW-Former**, is *wavelet-native*: it replaces the patch-embedding stem of a pretrained ConvNeXt V2-Tiny backbone with a fixed orthonormal **Haar-Stem**, and replaces the final RGB regression head with a fixed **Inverse-Haar** reconstruction head, so that every loss acts on an explicit, known frequency-decomposed basis. Training is adversarial (conditional PatchGAN) with manual optimisation.

The first production line (`llwt_v4`) plateaued at **PSNR ≈ 14.5 dB, SSIM ≈ 0.28, FID ≈ 81**. A code-level audit traced the plateau to two coupled causes: (i) an objective that was ≈ 85 % adversarial with almost no structural/perceptual fidelity signal, and (ii) a *silently broken* wavelet discriminator whose logits had exploded by ten orders of magnitude and were masked only by gradient clipping. The successor (`llwt_v45`) is a self-contained module that rebalances the objective with non-blurring fidelity losses (MS-SSIM, LPIPS, Focal Frequency Loss), adds a misalignment-robust contrastive loss (PatchNCE) targeting the inherent SAR↔optical co-registration gap, replaces the redundant log-amplitude SAR channel with a physically-motivated adaptive **Lee despeckle** channel, disables the broken wavelet discriminator, and adds the KID metric. Each change was validated by a sub-one-hour screening protocol (single-batch overfit + controlled 10-epoch A/B) before being adopted.

---

## 1. Final Architecture (`llwt_v45`)

The generator architecture is shared with the `llwt_v4` lineage; `llwt_v45` differs in three places — the SAR adapter's second channel (now adaptive despeckle), the discriminator configuration (wavelet head disabled), and the loss stack (rebalanced perceptual suite). Those differences are flagged inline below and motivated in §2.

### 1.1 Generator — `LLWv4Generator`

A single-branch encoder–decoder: a SAR-physics adapter feeds a fixed Haar wavelet stem, a pretrained ConvNeXt V2-Tiny backbone produces a four-scale feature pyramid, a PixelShuffle decoder reconstructs to half resolution, and a fixed inverse-Haar head lifts to full resolution.

![Generator architecture — full pipeline](figures/fig_generator.pdf)
> **Figure 1.** End-to-end `LLWv4Generator`. SAR `(B,1,256,256)` → SARAdapter (3-channel physics) → Haar-Stem (replaces the ConvNeXt 4×4 patch embed) → ConvNeXt V2-Tiny stages 1–4 → PixelShuffle decoder with skip connections → 12-channel subband head → Inverse-Haar reconstruction → `tanh`, giving optical `(B,3,256,256)` in `[-1,1]`. *(TikZ source compiled separately; drop the PDF at the path above.)*

**Total params:** ≈ 35.1 M (generator). **Forward contract:** `forward(sar, return_internals=True)` returns the triple `(out, sub, raw_feat)`, where `sub` is the predicted subband tensor `(B,3,4,H/2,W/2)` in order `[LL, LH, HL, HH]` (consumed by the per-band loss) and `raw_feat` is the post-stem 96-channel map.

#### 1.1.1 SAR-physics adapter — `SARAdapter`

Rather than a naïve `Conv(1→3)`, the SAR amplitude is expanded into three physically meaningful channels before the ImageNet-pretrained stem ever sees it:

```
channel 0 :  raw SAR amplitude              (dB, normalised to [-1,1])
channel 1 :  adaptive Lee despeckle  ← v0.4.5  (was log|x| in v4; see §2.5.4)
channel 2 :  Sobel gradient magnitude       (reflect-padded, fp32)
```

The three channels are concatenated and projected by a reflect-padded `Conv(3→3)` followed by GELU. The despeckle channel (`channel 1`) is the v0.4.5 change: it is a zero-parameter, edge-preserving filter (see §2.5.4), so the module's `state_dict` is unchanged and v4 checkpoints warm-load with **0 missing / 0 unexpected** keys.

#### 1.1.2 Haar-Stem — `HaarStemProjection`

The backbone's learned 4×4 stride-4 patch embedding is replaced by a **fixed two-level orthonormal Haar DWT**. The first level decomposes the 3-channel input into `[LL₁, LH₁, HL₁, HH₁]` at H/2; the second level decomposes `LL₁` into `[LL₂, LH₂, HL₂, HH₂]` at H/4. The three first-level detail bands are average-pooled to H/4 and concatenated with the four second-level bands (`7 × in_channels = 21` channels), then projected by a `1×1` conv to 96 channels with channel-wise LayerNorm. This produces the exact `(B, 96, H/4, W/4)` map the ConvNeXt stages expect.

A `token_output` flag makes the same stem emit the SwinV2 `(tokens, (H/4, W/4))` contract, enabling a clean one-variable backbone ablation (§2.5.5). **Design decision:** the Haar transform is *fixed* (orthonormal, no learnable lifting); an earlier learnable-lifting variant (v0.1.x) was tried and lost to the fixed orthonormal transform.

#### 1.1.3 Backbone — ConvNeXt V2-Tiny

```
facebook/convnextv2-tiny-22k-224   (HF AutoBackbone)
hidden_sizes = (96, 192, 384, 768)
out_indices  = [1, 2, 3, 4]  → four scales s0..s3 at H/4, H/8, H/16, H/32
≈ 28.6 M params, ImageNet-pretrained
```

ConvNeXt V2 blocks are purely convolutional (depthwise 7×7 → LayerNorm → pointwise → GELU → **GRN** → pointwise, with a residual). The **Global Response Normalisation (GRN)** layer performs channel-wise feature recalibration — this is why an additional CBAM channel-attention block was evaluated and **rejected as redundant** (see §4.7).

![ConvNeXt V2 block](figures/fig_convnextv2_block.pdf)
> **Figure 2.** A single ConvNeXt V2 block: depthwise 7×7 conv → LayerNorm → pointwise expand → GELU → GRN (global response normalisation) → pointwise project → residual add. GRN supplies the channel recalibration that makes a separate attention block redundant.

#### 1.1.4 PixelShuffle decoder

Four `ConvUpsampleBlock`s upsample the H/32 bottleneck back to H/2, each fusing the corresponding encoder skip. Every block does a sub-pixel (PixelShuffle) 2× upsample, optional skip concat, then a two-conv GroupNorm+GELU residual stack:

```
up4:  768 (+ skip s2=384) → 256   H/32 → H/16
up3:  256 (+ skip s1=192) → 128   H/16 → H/8
up2:  128 (+ skip s0=96)  →  64   H/8  → H/4
up1:   64 (no skip)       →  32   H/4  → H/2   (terminal feature map)
```

![Decoder block with PixelShuffle upsampling](figures/fig_decoder_pixelshuffle.pdf)
> **Figure 3.** One `ConvUpsampleBlock`: `1×1 Conv` to `4·C` channels → PixelShuffle 2× → concat encoder skip → two reflect-padded 3×3 convs (GroupNorm + GELU) with a 1×1 residual shortcut.

PixelShuffle (sub-pixel convolution) upsamples by reshaping channels into space — a `(B, 4C, H, W)` tensor becomes `(B, C, 2H, 2W)` — which avoids the checkerboard artefacts of transposed convolution.

![PixelShuffle operation](figures/fig_pixelshuffle.pdf)
> **Figure 4.** PixelShuffle: a `1×1` conv produces `r²` channels per output channel, which are then periodically shuffled into a higher-resolution grid (here upscale factor `r = 2`). No learned upsampling kernel, no checkerboard artefact.

#### 1.1.5 Subband head + Inverse-Haar output — `InverseHaarUp`

The 32-channel H/2 feature map is mapped by a zero-initialised 7×7 conv to **12 channels = 3 RGB × 4 Haar subbands**, reshaped to `(B,3,4,H/2,W/2)`, and reconstructed to full resolution by the fixed orthonormal inverse Haar transform (`Wᵀ` einsum + PixelShuffle), then `tanh`.

- **Why predict subbands instead of RGB?** The inverse is fixed and orthonormal, so each loss downstream acts on a *known frequency basis*: LL carries DC/low-frequency colour, the three detail bands carry horizontal/vertical/diagonal edges. This is what makes the per-band wavelet-L1 loss (§2.2) well-posed.
- **Zero-init warm-start trick:** with the subband head zero-initialised, predicted subbands = 0 → `IDWT(0) = 0` → `tanh(0)` = mid-gray. The generator therefore starts from a neutral image rather than random garbage, denying the discriminator a trivial early win. The same trick makes the optional detail-residual head (§2.5, default off) and v4-checkpoint warm-loading safe.

### 1.2 Discriminator — `LLWFormerDiscriminator`

A multi-head adversarial framework. In `llwt_v45` only the **MainDis** head is active; the wavelet **SubbandDis** head is *disabled* (it was found to be numerically broken — §2.5.6), and the **FourierDis** head was rejected earlier (§2.3).

![Discriminator stack](figures/fig_discriminator.pdf)
> **Figure 5.** Discriminator heads. **MainDis** (active): conditional two-scale 70×70 + 46×46 PatchGAN on the `[SAR, optical]` pair, no spectral norm. **SubbandDis** (disabled in v0.4.5): Haar-coefficient PatchGAN on the optical image with spectral norm. **FourierDis** (rejected): `log|rfft2|` PatchGAN. Each head returns `(logits, FM-features)` for the feature-matching loss.

| Head | Input | Structure | Status in v0.4.5 |
|------|-------|-----------|------------------|
| **MainDis** | `[SAR, OPT]` concat | 2-scale conditional PatchGAN, 4-layer LeakyReLU, **no** spectral norm | **active** |
| SubbandDis | OPT only | L=1 Haar PatchGAN, ndf=32, spectral norm | **disabled** (logits exploded; §2.5.6) |
| FourierDis | `log\|rfft2(OPT)\|` | 4-conv SN PatchGAN | rejected (§2.3) |

**Total D params:** ≈ 3.6 M.

### 1.3 Loss stack — final (`llwt_v45`)

The objective was rebalanced from raw measured magnitudes (warm R2 checkpoint, fp32) so that **every active term contributes ≈ 0.3–0.9** to the generator loss — no single term dominates. Measured contribution shown in parentheses.

| Loss | Weight (contribution) | Role |
|------|----------------------|------|
| **GANLoss (LSGAN)** — MainDis | `gan_main = 1.0` (0.61) | adversarial realism |
| **FeatureMatchingLoss** — MainDis | `fm_main = 10.0` (0.74) | stable adversarial training (load-bearing) |
| **MS-SSIM** (`1 − MS-SSIM`) | `msssim = 1.0` (0.70) | direct, non-blurring structural lever |
| **LPIPS** (AlexNet) | `lpips = 2.0` (0.88) | perceptual realism (primary photorealism driver) |
| **Focal Frequency Loss** (Jiang, ICCV 2021) | `ffl = 10.0` (0.32) | global FFT-domain sharpener |
| **PerBandWaveletL1Loss** | `per_band = 2.0` (0.54) | local Haar-subband fidelity (LL=1, LH=1.5, HL=1.5, HH=2) |
| **MultiLayerPatchNCE** (CUT) | `patchnce = 0.1` (0.49) | misalignment-robust contrastive supervision |
| GANLoss/FM — SubbandDis | `0.0` | **disabled** (§2.5.6) |
| GAN/FM — FourierDis | `0.0` | rejected (§2.3) |
| L1, LAB-chroma, WaveletDetail, SpeckleDecouple, PerfectRecon | `0.0` | off |
| R1 penalty | `0.0` | off (LSGAN + arch handle stability) |
| LSGAN label smoothing | `real = 1.0, fake = 0.0` | LSGAN equilibrium fix (§4.4) |

### 1.4 Training configuration

| Field | Value |
|-------|-------|
| Optimiser | AdamW (bitsandbytes 8-bit, fp32 fallback), β = (0.5, 0.999), wd = 0 |
| Learning rate (G + D) | `2e-4` (equal; no TTUR) |
| Scheduler | `linear_decay` — constant for first 150 ep, linear decay to `1e-6` over last 50 |
| Batch size | 8 train / 4 val, full-res 256×256 |
| Precision / memory format | bf16-mixed / channels_last (NHWC) |
| EMA | decay 0.999, `start_epoch = 20` |
| Compile | off (Windows venv lacks Triton) |
| Optimisation | manual (`automatic_optimization = False`), order `[opt_d, opt_g]` |
| D update | batched real+fake forward, `torch.no_grad()` on G |
| Warm-start | R2 best checkpoint `llwt-v0.4.1-perband-r2-detail/epoch=056-psnr=14.2607.ckpt` |
| Schedule | 200 epochs |
| Trainable params | netG 35.1 M + netD 3.6 M + criterions 3.1 M (PatchNCE MLPs + frozen LPIPS) |

### 1.5 Data

- **Dataset:** SEN1-2 (Sentinel-1 SAR + Sentinel-2 optical pairs).
- **Subset:** 5 scenes `["5","45","52","84","100"]` for screening; 9-scene list `["5","10","25","35","45","47","52","84","100"]` reserved for the headline run.
- **Split:** 80/20 train/val, seed 42.
- **Channels:** SAR = 1 (grayscale, dB-scaled, pre-normalised by the dataset authors); OPT = 3 (RGB, `[-1,1]`).
- **Augmentation:** synchronised geometric (albumentations `additional_targets={'optical':'image'}`).
- **Known limitation:** single-channel GRD amplitude only — no interferometric coherence, no dual-polarisation. SAR↔optical pairs are imperfectly co-registered, which caps pixel-aligned SSIM and pushes any pixel-loss GAN toward blur. This motivates the PatchNCE loss (§2.5.3).

---

## 2. Experimental Journey: v0.4.0 → v0.4.5

All runs warm-start from the prior best checkpoint unless noted; all validation metrics are on the 5-scene holdout.

### 2.1 v0.4.0 baseline — `v0.4.0-ihaar-prod` (R0)

**Purpose:** validate the Inverse-Haar output head against direct full-res RGB regression. **Schedule:** 120 ep, `cosine_warm_restarts` (T₀=20, T_mult=2), per-band equal weights, fresh from scratch.

**Result:** PSNR 13.02 (ep87), FID 83.0 (ep93), LPIPS 0.4355 (ep64), SSIM 0.2763 (ep62). **Issue:** cosine restarts at ep113→ep119 wrecked the final checkpoint (PSNR 12.51, FID 121). → switch to a non-restarting schedule for all later runs.

### 2.2 R-stage — per-band wavelet-L1 weighting

**Protocol:** warm-start from R0 ep87, constant LR, 60 ep, same 5-scene subset. The free variable is the relative weighting of the four predicted Haar subbands.

#### R1 — `perband-r1-equal` — `(1,1,1,1)`
PSNR 14.52 (ep59), FID 80.0 (ep29), LPIPS 0.420 (ep37), SSIM 0.288 (ep57). **Pattern:** PSNR climbs monotonically while FID plateaus then regresses after ep37 (late-10ep mean FID 88.5). **Diagnosis:** late-stage L1-averaging blur — PSNR keeps climbing at the cost of perceptual fidelity.

#### R2 — `perband-r2-detail` — `(1, 1.5, 1.5, 2)` ★ **WINNER**
PSNR 14.26 (ep56), FID 81.0 (ep56), LPIPS 0.4219 (ep56), SSIM 0.2931 (ep57) — **all four metrics co-aligned at ep56** (late-10ep mean FID 83.4, lowest mean *and* variance). **Diagnosis:** de-weighting the LL band (16.7 % share vs R1's 25 %) reduces L1-averaging pressure on the structural band, preserving texture variability while holding structural fidelity. The ep56 checkpoint is the "golden" warm-start for everything downstream.

#### R3 — skipped (would over-correct LL further; R2 already established the direction).

#### R4 — `perband-r4-adaptive` — Kendall uncertainty weighting
Learnable per-band σ (4-dim `log_var`, clamped `[-3,3]`). PSNR 14.29 (ep58), FID 81.0 (ep53), SSIM 0.2991 (ep54), but the saved ckpt FID is worse (86.5). The learned weights **rediscover R2's ranking automatically** (HH > LH ≈ HL > LL) but over-correct 5.4× (LL share 3.1 % vs R2's 16.7 %). **Negative result:** Kendall σ² tracks *loss magnitude*, not *perceptual importance* — LL has the largest raw L1 (DC content) so it is down-weighted hardest, yet LL is the most perceptually critical band. Manual R2 beats learned σ for this perception-aligned task.

### 2.3 S-stage — FourierDis adversarial spectral loss (rejected)

Adding a third discriminator head on `log|rfft2(opt)|` (warm from R2 ep56). Both the full-strength arm (S1: `gan_fourier=1.0, fm_fourier=10.0`) and the half-strength arm (S2) collapsed the same way — the D-G gap saturated at the abort threshold by ep12 (S1) / ep3 (S2) and FID regressed past baseline. **Negative result:** frequency adversarial signal is *non-additive* in a wavelet-native architecture — SubbandDis (Haar PatchGAN) already provides multi-resolution frequency discrimination, so a second frequency-domain head produces conflicting gradients on G. FourierDis was kept in code (config-gated, default off) but rejected.

### 2.4 The v4 plateau — root-cause diagnosis

After R/S stages, `llwt_v4` was stuck at **PSNR ≈ 14.5 / SSIM ≈ 0.28 / FID ≈ 81**. A code-level audit (not metric-chasing) found two coupled causes:

1. **The objective was ≈ 85 % adversarial.** `gan_main·1 + fm_main·10 + gan_sub·1 + fm_sub·10` dominated; the only fidelity anchor was per-band wavelet-L1 at ≈ 0.07 of the generator loss. With almost no structural or perceptual signal, SSIM had no lever to climb — it sat at 0.28 against a field norm of 0.4–0.7.
2. **SEN1-2 mis-registration** (single-channel amplitude, no coherence) caps pixel-aligned SSIM and pushes the GAN toward blur.

These two findings defined the v0.4.5 work programme: add non-blurring fidelity signal, and add a supervision term that tolerates mis-registration.

### 2.5 v0.4.5 overhaul → `llwt_v45`

#### 2.5.1 Isolation — self-contained module (copy-then-own)
To guarantee that improving the thesis model could never break the other models in the repository, `llwt_v45` was created as a *fully self-contained* copy of `llwt_v4` (the dotted name `v4.5` is not an importable Python module, hence `llwt_v45`; version strings use `v0.4.5`). Every cross-model dependency was vendored: `SARAdapter / HaarDown / ConvUpsampleBlock` → `blocks.py`; the discriminator → `dis.py`; the loss suite + LPIPS/FFL/PatchNCE → `losses.py`; the CUT sampler → `patchnce.py`; a self-contained `factory.py`. An isolation grep confirms the module imports nothing from other model packages.

#### 2.5.2 Loss overhaul — non-blurring fidelity
Three fidelity losses that add structure *without* the L1-averaging blur that the prior audit had correctly removed:
- **MS-SSIM** (`1 − MS-SSIM`) — the direct, structural, non-blurring lever for the SSIM metric.
- **LPIPS** (AlexNet, expects `[-1,1]`) — perceptual realism; the main photorealism driver.
- **Focal Frequency Loss** (Jiang et al., ICCV 2021) — a global FFT-domain sharpener that penalises hard-to-synthesise frequencies, complementary to the *local* per-band Haar loss.

#### 2.5.3 PatchNCE — misalignment-robust contrastive loss
`MultiLayerPatchNCE` (CUT, Park et al., 2020) applies contrastive supervision in the generator's **own** four-scale encoder feature space (no external VGG): the query is the encoded fake (with gradient), the key is the encoded real (no-grad). Because it matches *features* rather than pixels, it tolerates the sub-pixel SAR↔optical shift identified in §2.4. This required adding `LLWv4Generator.encode_optical()`; the sampler MLP parameters register into `opt_g`. Cost: ≈ 1.5× generator-step backbone compute.

#### 2.5.4 SAR despeckle channel — `SARAdapter` channel-2 A/B
The v4 second channel `log|x|` is a **folded, redundant double-log**: SEN1-2 SAR is already dB-scaled and normalised to `[-1,1]`, so `log(|x|)` conflates the darkest and brightest pixels. It was replaced with a **log-domain adaptive Lee despeckle** filter (`lee_despeckle`):

```
out = mean + w · (x − mean),   w = clamp(1 − σ_n² / var_local, 0, 1)
```

where `σ_n²` (log-domain speckle variance) is estimated *per image* as the 5th-percentile of the local-variance map (homogeneous patches ≈ pure speckle). It is O(N), edge-preserving, and **zero-parameter** — so the `state_dict` is unchanged. The choice of Lee (over Frost/Kuan/SRAD and DL despeckling) follows the standard log-domain homomorphic pipeline, which converts multiplicative speckle into additive noise that an adaptive linear filter handles well.

**Controlled A/B** (10 ep; for fairness the `SARAdapter.proj` was re-initialised in *both* arms, so only channel-2 differs — see the warm-start confound note in §4.6):

| channel-2 | FID | KID | LPIPS | SSIM | PSNR |
|-----------|-----|-----|-------|------|------|
| `log\|x\|` (v4) | 201 | 0.068 | 0.365 | 0.305 | 15.33 |
| **despeckle** (v0.4.5) | **180** | **0.048** (−29 %) | **0.355** | **0.310** | 15.27 |

Despeckle wins on photorealism (FID −10 %, KID −29 %, LPIPS −3 %, SSIM +) at flat PSNR → `sar_channel2: despeckle` is the default. *(These are relative screening numbers under a deliberately handicapped A/B — proj re-init, 10 ep — not production-scale metrics.)*

#### 2.5.5 Backbone ablation — SwinV2 vs ConvNeXt V2
The Haar-stem's `token_output` mode lets the identical front-end feed either backbone family; SwinV2-Tiny and ConvNeXt V2-Tiny share `hidden_sizes = (96,192,384,768)`, so the decoder is untouched — a clean one-variable ablation. In the single-batch overfit capacity screen (§2.6), **ConvNeXt V2 reached 35.38 dB smoothly** while **SwinV2 reached only 34.71 dB with instability spikes** (transformers need LR warmup). → **ConvNeXt V2 retained**; SwinV2 rejected.

#### 2.5.6 Critical discovery — SubbandDis numerical explosion
A loss-balance probe (fp32-confirmed, not a bf16 artefact) revealed that the inherited v4 **SubbandDis** was *silently broken*: its logits had exploded to **≈ 1.8 × 10⁵** (MainDis healthy at `[-0.6, 1.6]`) and its last feature-matching layer to **≈ 1.2 × 10⁵**. Consequently `gan_sub ≈ 3.5 × 10¹⁰` and `fm_sub ≈ 10³` — roughly **ten orders of magnitude** above every perceptual loss. R2 only "trained" because gradient clipping (`norm = 1`) renormalised the total gradient; its effective adversarial signal was almost entirely the clipped, exploding wavelet head, and **any added perceptual loss would have been completely swamped (zero effect).** Spectral norm is present on the head but does not bound it once warm-loaded.

**Fix:** disable SubbandDis (`subband.enabled = false`, `gan_sub = fm_sub = 0`) and drive realism through the healthy MainDis plus the balanced perceptual suite (§1.3). This is *the* change that makes the rest of the v0.4.5 overhaul effective.

#### 2.5.7 Added metric — KID
Kernel Inception Distance (`subset_size = 50`, guarded against tiny-validation NaN) was added alongside PSNR/SSIM/LPIPS/FID. KID is less biased than FID at small sample counts, which matters on the 5-scene holdout.

### 2.6 Overfit capacity screen (health smoke)

Pure-L1 single-batch overfit (lr 5e-4, 3000 iterations) verifies architectural capacity before committing GPU time to full runs:

| Arm | Overfit PSNR | Note |
|-----|--------------|------|
| ConvNeXt V2 | **35.38 dB** | smooth |
| ConvNeXt V2 + detail-residual head | 35.42 dB | capacity-neutral (IHaar path already fits L1) |
| SwinV2 | 34.71 dB | unstable (spikes at it 2000/2500) |

The ceiling is iteration-bound and shared, so the screen confirms *health*, not fine architectural ranking. It justified keeping ConvNeXt V2 and shipping the detail-residual head as an opt-in (default off).

---

## 3. Results

### 3.1 v4 R/S-stage — best-of-run per metric

| Run | tb_version | PSNR best | FID best | LPIPS best | SSIM best |
|-----|------------|-----------|----------|------------|-----------|
| R0 ihaar | `llwt-v0.4.0-ihaar-prod` | 13.02 (ep87) | 83.0 (ep93) | 0.4355 (ep64) | 0.2763 (ep62) |
| R1 equal | `llwt-v0.4.1-perband-r1-equal` | 14.52 (ep59) | 80.0 (ep29) | 0.4199 (ep37) | 0.2883 (ep57) |
| **R2 detail ★** | `llwt-v0.4.1-perband-r2-detail` | 14.26 (ep56) | **81.0 (ep56)** | **0.4219 (ep56)** | **0.2931 (ep57)** |
| R4 adaptive | `llwt-v0.4.1-perband-r4-adaptive` | 14.29 (ep58) | 81.0 (ep53) | 0.4297 (ep10) | 0.2991 (ep54) |
| S1 fourier (aborted) | `llwt-v0.4.2-fourier-s1-equal` | 14.12 (ep13) | 83.0 (ep12) | 0.4180 (ep13) | 0.2811 (ep14) |
| S2 fourier (aborted) | `llwt-v0.4.2-fourier-s2-gentle` | — | 87+ | — | — |

### 3.2 Deployable-checkpoint comparison (top-3 by `val/psnr`)

| Run | ckpt | PSNR | FID | LPIPS | SSIM | Note |
|-----|------|------|-----|-------|------|------|
| R1 (last) | ep59 | 14.52 | 87.0 | 0.430 | 0.286 | high PSNR, poor FID |
| **R2 ★** | ep56 | 14.26 | **81.0** | 0.422 | 0.293 | **all four co-aligned — golden** |
| R4 | ep58 | 14.29 | 86.5 | 0.432 | 0.299 | matches R2 PSNR, worse FID |

R2's PSNR-monitor happened to catch the FID-optimal epoch because all four metrics co-aligned at ep56. This is the warm-start anchor for `llwt_v45`.

### 3.3 v0.4.5 screening results
- **Overfit capacity:** ConvNeXt V2 35.38 dB (smooth) > SwinV2 34.71 dB (unstable). → ConvNeXt V2.
- **SubbandDis probe:** logits ≈ 1.8e5 vs MainDis `[-0.6,1.6]` → wavelet head disabled.
- **Despeckle A/B:** FID 201→180, KID −29 %, LPIPS −3 %, SSIM + (PSNR flat). → despeckle default.
- **Recipe A/B (overhauled perceptual suite vs v4 recipe):** the overhauled recipe led the v4 recipe on PSNR, LPIPS and FID at both the 1-epoch smoke and the 10-epoch checkpoint, confirming the rebalanced objective is the right direction before the 200-epoch run.

### 3.4 SOTA context
Published 2024–2026 SAR-to-optical results on SEN1-2 / GF-AD typically report **FID 150–265**. The R2 ep56 checkpoint at **FID 81.0** is already 2–3× better than typical baselines; the v0.4.5 objective is designed to extend that lead on photorealism while finally moving SSIM.

---

## 4. Mechanism Findings

### 4.1 Per-band L1 sweet spot
LL **de-weighting** (not detail boost per se) is the lever. Reducing LL's share of the L1 gradient (16.7 % vs 25 %) prevents late-run averaging blur on the structural band and preserves texture variability (FID/LPIPS gain) at the cost of ≈ 0.3 dB PSNR.

### 4.2 Kendall adaptive limitation
Kendall uncertainty weighting is optimal for the *loss* it minimises, not for *perceptual quality*: equilibrium σ² ∝ loss magnitude, so the perceptually-critical LL band (largest raw L1) is down-weighted hardest. **Manual prior beats learned σ for perception-aligned image translation.**

### 4.3 FourierDis redundancy
Frequency adversarial signal is non-additive in a wavelet-native architecture: stacking a Fourier head on top of the Haar SubbandDis produces conflicting gradients and a stuck equilibrium.

### 4.4 LSGAN equilibrium fix
The initial smoke plateaued at `d_real ≈ d_fake ≈ 0.45` because `real_smooth=0.9, fake_smooth=0.0` placed the LSGAN degenerate fixed point at exactly `(0.9+0)/2 = 0.45`. Fix: `real_smooth=1.0, fake_smooth=0.0`, with the R1 penalty disabled.

### 4.5 The objective was the bottleneck, not the backbone
The v4 plateau came from an ≈ 85 %-adversarial objective with negligible fidelity signal — *and* from a discriminator head that had silently exploded by ten orders of magnitude. Both are objective/optimisation faults, not capacity faults; the overfit screen confirms the architecture can reach 35 dB. This is why the v0.4.5 effort targeted losses and discriminator health, not a bigger backbone.

### 4.6 Warm-start A/B confound
The first despeckle A/B *appeared* to lose because the warm-loaded `SARAdapter.proj` was tuned for the log channel. Re-initialising `proj` in **both** arms removed the confound and reversed the result. **Lesson:** when A/B-testing an input-representation change behind a warm-started layer, re-initialise that layer in every arm.

### 4.7 CBAM rejected — redundant with GRN
Adding a CBAM channel-attention block was evaluated and rejected: ConvNeXt V2's GRN already performs global channel recalibration, so CBAM's channel branch is redundant. (Likewise, learnable Haar lifting was tried in v0.1.x and lost to the fixed orthonormal transform.)

---

## 5. Final Model — `llwt_v45`

The current best model is the `llwt_v45` module with the locked recipe below. It is a self-contained copy of `llwt_v4` with: despeckle SAR channel, MainDis-only discriminator, the balanced MS-SSIM + LPIPS + FFL + per-band + PatchNCE objective, and KID instrumentation.

| Field | Value |
|-------|-------|
| Config | `src/models/llwt_v45/config.yaml` |
| `tb_version` | `llwt-v0.4.5-cnx-overhaul` |
| Backbone | ConvNeXt V2-Tiny (`facebook/convnextv2-tiny-22k-224`) |
| SAR channel-2 | `despeckle` (adaptive Lee) |
| Discriminator | MainDis only (SubbandDis disabled) |
| Objective | gan_main 1.0 / fm_main 10 / msssim 1.0 / lpips 2.0 / ffl 10 / per_band 2.0 / patchnce 0.1 |
| Warm-start | R2 ep056 (0 missing / 0 unexpected keys verified) |
| Schedule | 200 ep, linear_decay (50 decay), EMA from ep20 |
| Metrics | PSNR, SSIM, LPIPS, FID, **KID** |
| Verification | overfit PASS; `fast_dev_run` PASS (all 7 criterions finite) |

**Launch command:**
```powershell
python -m src.models.llwt_v45.train
```

**Forecast targets (vs R2 best):** PSNR ≥ 14.7 (R2 14.52), FID ≤ 78 (R2 81), SSIM ≥ 0.40 (R2 0.29) — the SSIM lift being the headline goal of the loss overhaul. **Estimated wall-clock:** ≈ 5 h (≈ 1.5 min/ep × 200).

---

## 6. Future Work

### 6.1 v0.4.6 queue (after the 200-epoch baseline)
1. **Gated bottleneck self-attention on s3** (top candidate): the s3 feature map is 8×8 = 64 tokens, so global self-attention is nearly free and supplies the cross-region context that the purely-convolutional ConvNeXt lacks. Add as a zero-init gate `s3 + γ·Attn(s3)` with `γ = 0` so warm-loading is unchanged.
2. Enable the **detail-residual head** (already built, zero-init, default off) — a direct full-resolution SAR-conditioned high-frequency path.
3. **Fix and re-enable SubbandDis** (proper spectral/output normalisation, fresh non-warm D training) to restore the Haar-coefficient adversarial novelty.

Each candidate must pass the same screening gate (overfit + matched 10-epoch A/B) before adoption. **Rejected:** CBAM (redundant with GRN), learnable Haar (v0.1.x lost to fixed orthonormal), SwinV2 (lower + unstable), FourierDis (non-additive with SubbandDis).

### 6.2 Foundation-model perceptual loss (deferred)
Frozen Prithvi-EO-2.0 (NASA/IBM satellite ViT-MAE) as a domain-aligned perceptual feature extractor (vs LPIPS's natural-image priors). Implemented but blocked by a `terratorch` ↔ `transformers` dependency conflict.

### 6.3 SAR-conditional diffusion refinement (`llwt_v5`)
A lightweight UNet diffusion-refinement head (3–5 step DDIM) on the frozen GAN coarse output — combines GAN inference speed with diffusion's distributional matching. A structurally-faithful v0.4.5 base directly raises this refiner's ceiling.

---

## 7. Key Artifacts

```
src/models/llwt_v45/            # FINAL module (self-contained)
  gen.py        LLWv4Generator, HaarStemProjection, InverseHaarUp
  dis.py        LLWFormerDiscriminator (MainDis active)
  blocks.py     SARAdapter (+ lee_despeckle), HaarDown, ConvUpsampleBlock
  losses.py     GAN/FM/MS-SSIM/LPIPS/FFL/PerBandWaveletL1
  patchnce.py   MultiLayerPatchNCE (CUT sampler)
  factory.py    build_criterions / build_optimizers / build_lr_schedulers
  main.py       LLWv4LightningModule (manual optimisation, KID)
  config.yaml   locked v0.4.5 recipe
  train.py      entry point

checkpoints/llwt_v4/llwt-v0.4.1-perband-r2-detail/epoch=056-psnr=14.2607.ckpt   # warm-start anchor
output/llwt_v45/{tb_logs,csv_logs,images}/<tb_version>/                           # v0.4.5 outputs

figures/        # TikZ-compiled PDFs referenced by Figures 1–5
```

---

## 8. Decision Log

| Date | Decision | Reason |
|------|----------|--------|
| 2026-05-23 | Drop `cosine_warm_restarts` for R-stage | restarts wrecked the R0 final ckpt |
| 2026-05-23 | Warm-start R-stage from R0 ep87 | direct A/B of per-band effect, saves GPU |
| 2026-05-23 | W2 = R2 (not R1) | lower FID + better LPIPS/SSIM at only −0.26 dB PSNR |
| 2026-05-24 | Reject S-stage (FourierDis) | S1+S2 fail identically → mechanism, not tuning |
| 2026-05-25 | Build `llwt_v45` as a vendored, self-contained module | improving the thesis model must never break other models |
| 2026-05-25 | Add MS-SSIM + LPIPS + FFL + PatchNCE | objective was ≈ 85 % adversarial; needs non-blurring fidelity + misalignment-robust supervision |
| 2026-05-25 | **Disable SubbandDis** | logits exploded ≈ 1.8e5, swamping all perceptual losses under grad-clip |
| 2026-05-25 | SAR channel-2 = adaptive Lee despeckle | `log\|x\|` is a redundant double-log; despeckle wins FID −10 % / KID −29 % |
| 2026-05-25 | Keep ConvNeXt V2; reject SwinV2 | overfit 35.38 (smooth) vs 34.71 (unstable) |
| 2026-05-25 | Add KID metric | less biased than FID on the small holdout |
| 2026-05-25 | Defer bottleneck self-attention to v0.4.6 | adopt only after the headline baseline is established |
