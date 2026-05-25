# Stage 3 — Fourier-D Spectral Discriminator (LLW-Former v0.4.2)

**Status:** Design approved 2026-05-23. Implementation pending.
**Author:** Brainstormed via `superpowers:brainstorming` skill.
**Plan file:** to be generated via `superpowers:writing-plans` skill.

## Context

The LLW-Former v0.4.x roadmap (memory: `project_llwt_v4_stage2_built.md`,
`project_llwt_v3_ideas.md`) has three sequential architectural axes:

1. **Stage 1 — Inverse-Haar output head** (v0.4.0). Shipped. Currently
   training: matches v3 best PSNR (12.91 dB) in ~16 epochs at limit=0.3.
2. **Stage 2 — Per-band wavelet L1 loss** (v0.4.1). Shipped and Tier 1
   unit-tested. GPU validation (Tier 2-4) deferred until v0.4.0 prod finishes.
3. **Stage 3 — Fourier-D spectral discriminator** (v0.4.2). **This spec.**

Stage 3 adds a third adversarial head — a small PatchGAN that scores the
realism of the **log-magnitude FFT** of the generated optical image. The
existing discriminator wrapper already has a two-head structure
(`MainDis` pixel-space, `SubbandDis` Haar-wavelet); Fourier-D becomes the
third frequency-axis adversarial signal: spatial → wavelet → spectral.

**Thesis story:** "Tri-domain adversarial framework — three independent
adversarial signals across the spatial, wavelet, and spectral domains. Each
head specialises: MainDis enforces paired pixel realism, SubbandDis enforces
wavelet-band distribution match, FourierDis enforces global frequency
distribution match."

**Motivation:** Per-band L1 (stage 2) gives explicit per-band gradient on a
4-band Haar basis. FFT magnitude has a much finer frequency basis (~33k bins
at 256×256). A spectral discriminator can pressure G into matching texture
statistics that a 4-band wavelet decomposition collapses.

**Goal:** close the residual FID gap if stage 2 does not — provide a finer-
grained adversarial pressure on the high-frequency tail of the generated
optical distribution.

## Constraints

- **VRAM:** v0.4.0 prod at bs=8 uses ~23GB/24GB. Stage 3 must fit in <1GB
  additional headroom. Target: ~500MB activations + ~300k params.
- **GPU time:** prod run ~7h. Stage 3 must not balloon step time by >25%.
- **No interference with current runs:** v0.4.0 prod and v0.4.1 stage 2
  must remain unaffected. Achieved via `cfg.model.dis.fourier.enabled=false`
  default. Existing configs unchanged.
- **Thesis defendability:** every component must have an interpretable
  story. Black-box architectures rejected.

## Design

### Module — `FourierDis`

New class in `src/models/llwt/dis.py` parallel to `MainDis` and `SubbandDis`.

**Input:** optical RGB tensor `(B, 3, H, W)` in `[-1, 1]`.

**Internal pipeline:**

```
opt: (B, 3, H, W)
    │
    ▼
torch.fft.rfft2(opt, dim=(-2, -1))    # complex, (B, 3, H, W//2+1)
    │
    ▼
log(|F| + 1e-8)                       # real, dynamic-range compressed
                                       # squashes the 1/f power law tail
    │
    ▼
5-conv spectral-norm PatchGAN
  (kernel=4, stride=2, ndf=32, LeakyReLU 0.2)
    │
    ▼
logits: (B, 1, h', w')   +   feats: list of 4 LeakyReLU activations (FM)
```

**Why log-magnitude:** natural-image power spectra follow ~1/f power law.
Raw |F| saturates the first conv (DC bin dwarfs everything else). Log
compresses dynamic range and matches the standard pattern for spectral GANs
(StyleGAN3 spectral analyses, Focal Frequency Loss).

**Why unconditional (opt-only):** symmetric with `SubbandDis`. SAR spectrum
has speckle-dominated statistics — concatenating SAR + opt spectra adds
noise without clear semantic value. The conditional MainDis already covers
paired realism. Each head specialises.

**Why spectral norm + ndf=32:** mirrors `SubbandDis`. Spectral norm enforces
1-Lipschitz constraint (stabilises GAN dynamics, especially important when
adding a third head that could destabilise the existing equilibrium). ndf=32
keeps the param + VRAM cost low.

**Why 5-conv:** matches `SubbandDis` depth. Spectrum tensor at 256×256 input
is (B, 3, 256, 129); after 5 stride-2 convs the receptive field covers
the whole spectrum.

### Wrapper changes — `LLWFormerDiscriminator`

Extend forward signature to return a **6-tuple**:

```python
def forward(self, sar, opt):
    """Returns (main_pair, sub_logits, fourier_logits,
                feats_main, feats_sub, feats_fourier).

    All four "logits" slots may be None when the corresponding head is
    disabled. All three feature lists may be empty when disabled.
    """
```

Backwards-compat strategy: when `cfg.model.dis.fourier.enabled=false`
(default), `fourier_logits=None` and `feats_fourier=[]` — same pattern as
the existing `SubbandDis` disabled path.

**Caller-site impact:** every site that unpacks the discriminator output
must change from 4-tuple to 6-tuple. Inventory:

- `src/models/llwt_v4/main.py` — D-update R1 path, D-update batched path,
  G-update path (3 unpack sites).
- `src/models/llwt_v4/overfit_test.py` — D path, G path (2 unpack sites).

All updates are mechanical (extend tuple unpack with two extra `_, _` slots
or named variables). Same PR.

### Training-loop additions — `main.py training_step`

Mirror the existing `use_subband_d` pattern. New flags:

```python
self.use_fourier_d = self.netD.use_fourier   # set in __init__
self.gan_fourier_weight = float(getattr(cfg.loss, 'gan_fourier_weight', 1.0))
self.fm_fourier_weight  = float(getattr(cfg.loss, 'fm_fourier_weight',  10.0))
```

D-update (both R1 and batched paths) adds the third GAN term:

```python
if self.use_fourier_d:
    d_loss_fourier = 0.5 * (
        self.criterions['gan'](real_fourier, is_real=True) +
        self.criterions['gan'](fake_fourier, is_real=False)
    )
    d_loss = d_loss + d_loss_fourier
    self.log('train/d_loss_fourier', d_loss_fourier.detach(), on_step=False, on_epoch=True)
```

G-update adds GAN_fourier + FM_fourier terms:

```python
if self.use_fourier_d:
    l_gan_fourier = self.criterions['gan'](fake_fourier_g, is_real=True, for_d=False)
    g_loss = g_loss + l_gan_fourier * self.gan_fourier_weight
    self.log('train/gan_fourier', l_gan_fourier.detach(), on_step=False, on_epoch=True)
    if fake_feats_fourier:
        l_fm_fourier = self.criterions['fm'](fake_feats_fourier, real_feats_fourier)
        g_loss = g_loss + l_fm_fourier * self.fm_fourier_weight
        self.log('train/fm_fourier', l_fm_fourier.detach(), on_step=False, on_epoch=True)
```

R1 penalty on Fourier-D: **not added** in v0.4.2. Spectral norm + opt-only
input means the FFT is a deterministic function of the input image; R1
would require gradients through the FFT, which is supported by PyTorch but
adds noticeable wall-clock cost. Defer to v0.4.3 if observed instability
warrants.

### Config schema additions

Under `cfg.model.dis`:

```yaml
fourier:
  enabled: false   # gated default OFF — opt-in via stage-3 cfg variants
  ndf: 32          # base width; matches SubbandDis. Drop to 16 if VRAM tight.
  use_sn: true     # spectral norm; keep on for tri-D stability
```

Under `cfg.loss`:

```yaml
# Fourier-D adversarial weights (v0.4.2 stage 3). Mirror gan_sub/fm_sub
# scale by default — start equal, ablate later.
gan_fourier_weight: 0.0   # set 1.0 to match gan_sub when enabled
fm_fourier_weight:  0.0   # set 10.0 to match fm_sub when enabled
```

All four new fields default to off/zero. Existing configs (v0.4.0,
v0.4.1) unaffected.

## Validation strategy

### Tier 1 — unit smoke (CPU, <10s)

`python -m src.models.llwt.dis` (extend the existing `__main__` if any, or add):
- Build `FourierDis(ndf=32)`.
- Feed `opt = torch.randn(2, 3, 256, 256)`.
- Assert output shapes: logits `(2, 1, h', w')` non-empty; feats list
  length 4, each element a tensor.
- Assert no NaN / inf in logits or features.
- Assert log-magnitude pipeline handles a saturated input (opt = ones) and
  a zero input (opt = zeros) without NaN/inf.

### Tier 2 — GPU overfit (5min)

Extend `overfit_test.py` to handle the optional Fourier-D path. Run with
`cfg.model.dis.fourier.enabled=true`, `gan_fourier_weight=1.0`,
`fm_fourier_weight=10.0`:
- PSNR ≥ 22 dB at step 800 (existing gate).
- `train/d_loss_fourier` decreases monotonically over steps 25–200.
- No NaN strikes on G or D.

### Tier 3 — short prod sanity (1h)

20 epochs at `limit_train/val_batches=0.3`. Compare ep10 to v0.4.1 R1
control (or v0.4.0 prod ep10 if R1 not yet run):
- PSNR Δ ≥ 0 dB (no regression).
- FID Δ ≤ -5% (Fourier-D moves the needle).
- D-G gap (`d_real_mean - d_fake_mean`) stable below +0.20.
- D-D-D gap stability: all three heads' `d_real - d_fake` should sit in
  the same order of magnitude. If one head goes to +0.5 while others stay
  at +0.05, that head is winning — likely Fourier-D overpowering — abort.

### Tier 4 — ablation matrix (~28h sequential)

Run sequentially after stage 2's R1/R2/R3 finish. Each at full prod config
(`limit=1.0`, `max_epochs=60`):

| Run | tb_version | Stage 2 cfg | Stage 3 cfg | Purpose |
|-----|-----------|-------------|-------------|---------|
| **S0** | `llwt-v0.4.1-perband-r1-equal` | per_band=2, equal | fourier off | Control (v0.4.1 stage-2 winner) |
| **S1** | `llwt-v0.4.2-fourier-default` | per_band=2, equal | gan_f=1.0, fm_f=10.0 | Does Fourier-D help on top of stage 2? |
| **S2** | `llwt-v0.4.2-fourier-gentle` | per_band=2, equal | gan_f=0.5, fm_f=5.0 | Gentler weights — does it still help? |

Decision rule: S1 must beat S0 on FID at ep30 to justify shipping
Fourier-D. If FID Δ < 3% at ep30, reject the hypothesis; v0.4.2 stays in
the codebase as opt-in but is not the default.

## Risks

| Risk | Detection | Mitigation |
|------|-----------|------------|
| **Three-head D destabilises GAN equilibrium** | Any single head's `d_real - d_fake > 0.30` while others stay < 0.15 (head dominance) | Existing mode-collapse abort gate; reduce that head's weight by 50% |
| **VRAM overflow at bs=8** | OOM at first training step | Fall back to `ndf=16` (4× smaller activations) or `bs=6` |
| **FFT log-magnitude NaN on saturated images** | First training step throws RuntimeError | `+1e-8` eps already in design; clamp log output to `[-30, +30]` as belt-and-suspenders |
| **6-tuple unpacking breaks resume** | First resume step throws ValueError on unpack | All 5 unpack sites updated in same PR; full smoke + overfit before merging |
| **rfft2 dtype mismatch under bf16 autocast** | First step throws RuntimeError | `torch.fft.rfft2` is autocast-aware in PyTorch >= 2.0; verify in Tier 1 |
| **Fourier-D rewards "spectrum match" while losing pixel quality** | FID improves but PSNR/SSIM regress sharply | Tier-3 abort threshold: PSNR Δ < -0.5 dB → halt |
| **3rd head adds >25% step time** | Training step ms increases >25% vs v0.4.1 baseline | Profile in Tier 3; ndf=16 fallback or skip-every-other-step pattern |
| **G can't satisfy three D heads simultaneously** | All three GAN losses oscillate or rise after ep20 | Reduce all GAN weights proportionally (gan_main=0.5, gan_sub=0.5, gan_fourier=0.5) |

## Out of scope (defer to v0.4.3+)

- **R1 penalty on Fourier-D** — adds compute, deferred unless instability observed.
- **Conditional Fourier-D (SAR-conditioned)** — rejected at brainstorm (SAR spectrum statistics differ too much from opt).
- **DCT-based spectral D** — rejected as alternative input domain; log-FFT is more standard.
- **Multi-scale Fourier-D** (FFT at multiple resolutions) — possible v0.4.3 enhancement; skip for stage 3.
- **Focal Frequency Loss as a non-adversarial supplement** — rejected as
  alternative path (would replace stage 3 rather than add to it).

## Critical files

- `src/models/llwt/dis.py` — new `FourierDis` class + extend `LLWFormerDiscriminator` wrapper to 6-tuple return.
- `src/models/llwt_v4/main.py` — 3 unpack sites + `use_fourier_d` flag + D-update + G-update additions.
- `src/models/llwt_v4/overfit_test.py` — 2 unpack sites + optional Fourier-D path.
- `src/models/llwt_v4/config.yaml` — 4 new fields, default off.
- `src/models/llwt/factory.py` — no change required (criterions reused; D wrapper is what gets extended).
