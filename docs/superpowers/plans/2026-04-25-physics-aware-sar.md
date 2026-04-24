# Physics-Aware SAR-to-Optical GAN Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Option B physics-aware improvements to CFRWD-GAN: scalar fusion weight fix, SpeckleAwareModule heteroscedastic gating, wavelet-domain HFCF supervision, and CFR aux L1 supervision.

**Architecture:** Replace the spatial-softmax AdaptiveFusion with a single learnable sigmoid scalar (w_hfcf, init=0.5). Insert SpeckleAwareModule after the DWT step in HFCFBranch — it estimates per-pixel local variance of each wavelet subband and uses a small network to produce a per-channel attention gate ∈ [0,1] (high local variance = speckle-dominated = attenuate; low = coherent edge = pass). Supervise the HFCF branch output with WaveletSupervisionLoss (L1 across all 6 detail subbands of the Haar DWT) instead of the architecturally-wrong full-image L1. Add standard full-image L1 aux supervision to the CFR branch (valid: CFR processes the full image). Log-domain preprocessing for HFCF is implemented but disabled by default for SEN12 (thesis empirical finding: worsens |ρ| from 0.16→0.77; enable for QXSLAB).

**Tech Stack:** PyTorch 2.x, PyTorch Lightning, HaarDown (in-codebase Haar DWT in gen.py), OmegaConf, cfrwd-36 branch base.

---

## File Map

| File | Change type | Summary |
|------|-------------|---------|
| `src/models/cfrwd/gen.py` | Modify | Add `SpeckleAwareModule`; modify `HFCFBranch` (log-preprocess option + SAM gating); replace `AdaptiveFusion` with scalar `_fusion_logit` in `CFRWDGenerator`; update `_initialize_weights` |
| `src/models/cfrwd/losses.py` | Modify | Add `WaveletSupervisionLoss` |
| `src/models/cfrwd/factory.py` | Modify | Pass `use_log_preprocess` to `CFRWDGenerator`; add `'WAVELET'` criterion |
| `src/models/cfrwd/main.py` | Modify | Remove `loss_hfcf_aux`; capture `cfr_out`; add `loss_cfr_aux` + `loss_wavelet`; update logging |
| `src/models/cfrwd/config.yaml` | Modify | `tb_version: "physics-1"`; add `cfr_aux_weight`, `wavelet_weight`, `use_log_preprocessing` |
| `src/models/cfrwd/inference.py` | Modify | Handle scalar `w_hfcf`; show speckle gate stats in subplot 6 |

---

## Task 1: Create git branch

**Files:** none (git only)

- [ ] **Step 1.1: Create physics-aware branch from current state**

```bash
git checkout -b physics-aware
```

Expected output: `Switched to a new branch 'physics-aware'`

- [ ] **Step 1.2: Verify branch**

```bash
git branch --show-current
```

Expected: `physics-aware`

---

## Task 2: Bug fixes — scalar fusion + remove HFCF L1 aux + config bump

**Files:**
- Modify: `src/models/cfrwd/gen.py:570-654`
- Modify: `src/models/cfrwd/main.py:104-150`
- Modify: `src/models/cfrwd/config.yaml`

### 2a — gen.py: replace AdaptiveFusion usage with scalar sigmoid weight

- [ ] **Step 2a.1: In `CFRWDGenerator.__init__` (gen.py line 606), replace `adaptive_fusion` with scalar parameter**

Replace:
```python
class CFRWDGenerator(nn.Module):
    def __init__(self, in_channels=1):
        super(CFRWDGenerator, self).__init__()
        self.cfr_branch = CFRBranch(in_channels=in_channels)
        self.hfcf_branch = HFCFBranch(in_channels=in_channels)
        self.adaptive_fusion = AdaptiveFusion(feat_channels=32)
        # Per-branch decoders: each branch decodes its own feature space.
        # Fusion happens at RGB logit level so each decoder trains on its own distribution.
        self.cfr_final = FinalDecoderBlock(32, 3, kernel_size=7)
        self.hfcf_final = FinalDecoderBlock(32, 3, kernel_size=7)

        self._initialize_weights()
```

With:
```python
class CFRWDGenerator(nn.Module):
    def __init__(self, in_channels=1, use_log_preprocess=False):
        super(CFRWDGenerator, self).__init__()
        self.cfr_branch = CFRBranch(in_channels=in_channels)
        self.hfcf_branch = HFCFBranch(in_channels=in_channels, use_log_preprocess=use_log_preprocess)
        # Scalar learnable fusion weight: sigmoid(0) = 0.5 → equal init weighting.
        # Replaces spatial AdaptiveFusion (spatial softmax degenerated to scalar in cfrwd-38).
        self._fusion_logit = nn.Parameter(torch.zeros(1))
        # Per-branch decoders: each branch decodes its own feature space.
        # Fusion happens at RGB logit level so each decoder trains on its own distribution.
        self.cfr_final = FinalDecoderBlock(32, 3, kernel_size=7)
        self.hfcf_final = FinalDecoderBlock(32, 3, kernel_size=7)

        self._initialize_weights()
```

- [ ] **Step 2a.2: In `CFRWDGenerator.forward` (gen.py line 641), replace AdaptiveFusion call with scalar weight**

Replace:
```python
    def forward(self, x, return_branches=False):
        cfr_feats = self.cfr_branch(x)    # B×32×H×W
        hfcf_feats = self.hfcf_branch(x)  # B×32×H×W

        _, fusion_weights = self.adaptive_fusion(cfr_feats, hfcf_feats)
        cfr_logits  = self.cfr_final(cfr_feats)    # B×3×H×W, pre-tanh
        hfcf_logits = self.hfcf_final(hfcf_feats)  # B×3×H×W, pre-tanh
        out = torch.tanh(
            fusion_weights[:, 0:1] * cfr_logits + fusion_weights[:, 1:2] * hfcf_logits
        )

        if return_branches:
            return out, torch.tanh(cfr_logits), torch.tanh(hfcf_logits), fusion_weights
        return out, fusion_weights
```

With:
```python
    def forward(self, x, return_branches=False):
        cfr_feats  = self.cfr_branch(x)    # B×32×H×W
        hfcf_feats = self.hfcf_branch(x)   # B×32×H×W

        w_hfcf = torch.sigmoid(self._fusion_logit)  # scalar ∈ (0, 1), init=0.5
        w_cfr  = 1.0 - w_hfcf
        cfr_logits  = self.cfr_final(cfr_feats)     # B×3×H×W, pre-tanh
        hfcf_logits = self.hfcf_final(hfcf_feats)   # B×3×H×W, pre-tanh
        out = torch.tanh(w_cfr * cfr_logits + w_hfcf * hfcf_logits)

        if return_branches:
            cfr_out  = torch.tanh(cfr_logits)
            hfcf_out = torch.tanh(hfcf_logits)
            return out, cfr_out, hfcf_out, w_hfcf
        return out, w_hfcf
```

### 2b — main.py: remove hfcf_aux, update G-update unpacking and logging

- [ ] **Step 2b.1: Update G-update forward call (main.py line 108) to capture cfr_out and scalar w_hfcf**

Replace:
```python
        fake_opt, _, hfcf_out, fusion_weights = self.netG(real_sar, return_branches=True)
```

With:
```python
        fake_opt, cfr_out, hfcf_out, w_hfcf = self.netG(real_sar, return_branches=True)
```

- [ ] **Step 2b.2: Remove loss_hfcf_aux computation and update g_loss (main.py lines 119-129)**

Replace:
```python
        # Auxiliary supervision for HFCF branch — prevents wavelet branch atrophy
        loss_hfcf_aux = self.criterions['L1'](hfcf_out, real_opt) * self.cfg.loss.get('hfcf_aux_weight', 1.0)

        loss_l1  = self.criterions['L1'](fake_opt, real_opt)
        loss_fft = self.criterions['FFT'](fake_opt, real_opt)

        g_loss = (
            loss_gan      * self.loss_weights['gan'] +
            loss_fm       * self.loss_weights['fm'] +
            loss_hfcf_aux
        )
```

With:
```python
        loss_l1  = self.criterions['L1'](fake_opt, real_opt)
        loss_fft = self.criterions['FFT'](fake_opt, real_opt)

        g_loss = (
            loss_gan * self.loss_weights['gan'] +
            loss_fm  * self.loss_weights['fm']
        )
```

- [ ] **Step 2b.3: Update training step logging (main.py lines 135-150)**

Replace:
```python
        self.log('train/g_loss', g_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=real_sar.size(0))
        self.log_dict({
            'train/loss_fm':  loss_fm,
            'train/loss_gan': loss_gan,
            'train/loss_d':   d_loss,
            'train/loss_l1':  loss_l1,
            'train/loss_fft': loss_fft,
            'train/loss_hfcf_aux': loss_hfcf_aux,
            'feats/d_real_mean': real_means.mean(),
            'feats/d_fake_mean': fake_means.mean(),
            'fusion/w_hfcf':     fusion_weights[:, 1].mean(),
            'fusion/spatial_std': fusion_weights[:, 1].std(dim=[1, 2]).mean(),
        }, prog_bar=False, on_step=False, on_epoch=True, batch_size=real_sar.size(0))
```

With:
```python
        self.log('train/g_loss', g_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=real_sar.size(0))
        self.log_dict({
            'train/loss_fm':  loss_fm,
            'train/loss_gan': loss_gan,
            'train/loss_d':   d_loss,
            'train/loss_l1':  loss_l1,
            'train/loss_fft': loss_fft,
            'feats/d_real_mean': real_means.mean(),
            'feats/d_fake_mean': fake_means.mean(),
            'fusion/w_hfcf': w_hfcf.squeeze(),
        }, prog_bar=False, on_step=False, on_epoch=True, batch_size=real_sar.size(0))
```

### 2c — config.yaml: bump version, remove hfcf_aux_weight

- [ ] **Step 2c.1: Update config.yaml**

In the `loss:` section, replace:
```yaml
loss:
  gan_weight: 1
  fm_weight: 10
  hfcf_aux_weight: 1.0  # L1 weight on HFCF branch output — prevents wavelet branch atrophy
  # L1 и FFT авто-балансируются через AdaptiveLoss (Kendall et al., 2018).
  # Ручные веса не нужны — eta обучаются вместе с генератором.
  use_lpips: false  # true → добавляет LPIPS в AdaptiveLoss третьим компонентом [L1, FFT, LPIPS]
```

With:
```yaml
loss:
  gan_weight: 1
  fm_weight: 10
  cfr_aux_weight: 0.3   # L1 on CFR branch output (full-image, valid for CFR)
  wavelet_weight: 0.5   # L1 on Haar detail subbands of HFCF output
  use_log_preprocessing: false  # log(1+relu(x)) before DWT — worsens SEN12 |rho| (0.16→0.77), enable only for QXSLAB
  use_lpips: false
```

In the `system:` section, change:
```yaml
  tb_version: "cfrwd-38"
  resume_ckpt: null
```
To:
```yaml
  tb_version: "physics-1"
  resume_ckpt: null
```

- [ ] **Step 2d: Verify gen.py runs clean (no import errors, correct output shapes)**

```bash
python src/models/cfrwd/gen.py
```

Expected: prints `out: torch.Size([1, 3, 256, 256])` with no errors.

- [ ] **Step 2e: Commit bug fixes**

```bash
git add src/models/cfrwd/gen.py src/models/cfrwd/main.py src/models/cfrwd/config.yaml
git commit -m "HFCF AUX LOSS & OTHER FIXED

- Replace AdaptiveFusion spatial softmax with scalar sigmoid fusion weight
  (init=0.5, paper-faithful; spatial fusion degenerated to scalar in cfrwd-38)
- Remove wrong HFCF full-image L1 aux loss (HF-only branch cannot reconstruct
  full image — LL2 is discarded in DWT, no LF signal available)
- Capture cfr_out from return_branches for upcoming CFR aux loss
- Update fusion logging to scalar w_hfcf
- Bump config to physics-1, remove hfcf_aux_weight, add cfr_aux/wavelet placeholders"
```

---

## Task 3: Add SpeckleAwareModule to gen.py

**Files:**
- Modify: `src/models/cfrwd/gen.py` — add class after CBAM; modify HFCFBranch.__init__ and forward; update CFRWDGenerator._initialize_weights

**Physics rationale:** SAR speckle is multiplicative (I = R·n, n ~ Gamma). In Haar wavelet detail subbands, real structural edges produce low local variance (consistent, localized). Speckle produces high local variance (spatially uncorrelated). The module estimates local variance via `E[x²] - E[x]²` (using AvgPool2d) and maps it to an attention gate: high variance → attenuate (speckle-dominated); low variance → pass through (coherent signal). Gate initializes near 1.0 (no effect at start, network learns when to attenuate).

- [ ] **Step 3.1: Add `SpeckleAwareModule` class to gen.py, after the `CBAM` class (after line 483)**

Insert after line 483 (end of CBAM class):
```python

class SpeckleAwareModule(nn.Module):
    """
    Heteroscedastic attention gate for wavelet detail subbands.

    Estimates per-pixel local variance via E[x²] - E[x]² (AvgPool2d approximation),
    then maps variance → gate ∈ [0,1] via a 2-layer bottleneck network.

    Physics: SAR speckle is multiplicative (I = R·n, n ~ Gamma(L,L)).
    In wavelet domain: real edges → low local variance (structured, localized).
    Speckle noise → high local variance (spatially uncorrelated).
    Gate attenuation of high-variance regions extracts signal from speckle.

    Gate initializes near 1.0 (pass-through) via bias init in CFRWDGenerator._initialize_weights.
    """
    def __init__(self, in_channels: int, kernel_size: int = 7):
        super().__init__()
        reduced = max(in_channels // 4, 8)
        self.pool = nn.AvgPool2d(kernel_size, stride=1, padding=kernel_size // 2)
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, reduced, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, in_channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        # Local variance estimate: E[x²] - E[x]²
        mu  = self.pool(x)
        mu2 = self.pool(x * x)
        var = (mu2 - mu * mu).clamp(min=0.0)  # B×C×H×W, non-negative
        gate = self.gate(var)                  # B×C×H×W, ∈ [0,1]
        return x * gate, gate
```

- [ ] **Step 3.2: Modify `HFCFBranch.__init__` to accept `use_log_preprocess` and register two SpeckleAwareModules**

Replace the class signature and `__init__` opening lines:
```python
class HFCFBranch(nn.Module):
    """
    ...
    """
    def __init__(self, in_channels=1, hidden_dim=64):
        super(HFCFBranch, self).__init__()
        logger.debug('HFCF BRANCH INIT')
        freq_c = 3 * in_channels  # каналы cat-группы высокочастотных подполос

        self.dwt = DWTBlock(in_channels=in_channels)
```

With:
```python
class HFCFBranch(nn.Module):
    """
    Wavelet Decomposition branch с тремя независимыми потоками.

    Исправления по аудиту архитектуры:
    - Ранняя сумма g2+g3 до потоков устранена: каждая группа обрабатывается независимо.
    - LL2 добавлен как отдельный Low-поток (структурный контекст), что помогает
      Mid/High потокам ориентироваться в пространстве и снижает вероятность
      "затухания" HFCF-ветки (fusion_weight → 0).
    - CBAM на входе каждого потока фильтрует спекл-шум SAR до ResBlocks.
    - SpeckleAwareModule: heteroscedastic gate на сырых вейвлет-подполосах до CBAM+проекции.

    Архитектура потоков (все приводятся к W/4 перед слиянием):

      Low  (LL2 @ W/4)       : HFCFPreprocess → CBAM → WDResBlock×2      ──┐
      Mid  ([LH2,HL2,HH2] W/4): SAM → HFCFPreprocess → CBAM → WDResBlock×6 ──┤ → merge(1×1) → decoder → 3ch
      High ([LH1,HL1,HH1] W/2): SAM → HFCFPreprocess(↓) → CBAM → RedBlock×2 ──┘

    SAM = SpeckleAwareModule (heteroscedastic variance gate)
    """
    def __init__(self, in_channels=1, hidden_dim=64, use_log_preprocess=False):
        super(HFCFBranch, self).__init__()
        logger.debug('HFCF BRANCH INIT')
        self.use_log_preprocess = use_log_preprocess
        freq_c = 3 * in_channels  # каналы cat-группы высокочастотных подполос

        self.dwt = DWTBlock(in_channels=in_channels)

        # Physics-aware speckle variance gates on detail subbands (NOT on LL2 — structural, less speckle)
        self.speckle_mid  = SpeckleAwareModule(in_channels=freq_c, kernel_size=7)
        self.speckle_high = SpeckleAwareModule(in_channels=freq_c, kernel_size=7)
```

- [ ] **Step 3.3: Modify `HFCFBranch.forward` to apply log preprocessing and SpeckleAwareModule gating**

Replace:
```python
    def forward(self, x):
        g1, g2, g3 = self.dwt(x)  # g1=LL2 @ W/4, g2=[LH2,HL2,HH2] @ W/4, g3=[LH1,HL1,HH1] @ W/2

        logger.debug(f'DWT shapes: g1={g1.shape}, g2={g2.shape}, g3={g3.shape}', once=True)

        # Три полностью независимых потока — никакого преждевременного смешения
        ll   = self.ll_stream(self.ll_cbam(self.ll_proj(g1)))
        mid  = self.mid_stream(self.mid_cbam(self.mid_proj(g2)))
        high = self.high_stream(self.high_cbam(self.high_proj(g3)))  # W/2 → W/4

        merged = self.merge(torch.cat([ll, mid, high], dim=1))
        return self.decoder(merged)
```

With:
```python
    def forward(self, x):
        if self.use_log_preprocess:
            # log(1 + relu(x)): converts multiplicative speckle toward additive domain.
            # Disabled for SEN12 by default — empirically worsens |rho| (0.16→0.77).
            # Enable via config.loss.use_log_preprocessing for QXSLAB.
            x = torch.log1p(torch.relu(x))

        g1, g2, g3 = self.dwt(x)  # g1=LL2 @ W/4, g2=[LH2,HL2,HH2] @ W/4, g3=[LH1,HL1,HH1] @ W/2

        logger.debug(f'DWT shapes: g1={g1.shape}, g2={g2.shape}, g3={g3.shape}', once=True)

        # Physics-aware speckle gating on raw detail subbands (before InstanceNorm in proj).
        # Gate acts on raw wavelet statistics: high local variance = speckle → attenuate.
        g2, g2_gate = self.speckle_mid(g2)
        g3, g3_gate = self.speckle_high(g3)
        # Store mean gate activations for logging (side-effect, accessed by training loop)
        self._last_g2_gate_mean = g2_gate.mean().detach()
        self._last_g3_gate_mean = g3_gate.mean().detach()

        # Три полностью независимых потока — никакого преждевременного смешения
        ll   = self.ll_stream(self.ll_cbam(self.ll_proj(g1)))
        mid  = self.mid_stream(self.mid_cbam(self.mid_proj(g2)))
        high = self.high_stream(self.high_cbam(self.high_proj(g3)))  # W/2 → W/4

        merged = self.merge(torch.cat([ll, mid, high], dim=1))
        return self.decoder(merged)
```

- [ ] **Step 3.4: Update `CFRWDGenerator._initialize_weights` to set SpeckleAwareModule gate biases to near-pass-through**

After the existing Xavier block (after line 638 `logger.info(...)`), add:
```python
        # SpeckleAwareModule: init gate[-2] bias large positive → sigmoid ≈ 0.95 (near pass-through).
        # The Kaiming pass above set these biases to 0 → sigmoid(0)=0.5 would block half the signal.
        # Starting open lets the network learn when to attenuate, not force attenuation from epoch 0.
        for m in self.modules():
            if isinstance(m, SpeckleAwareModule):
                nn.init.constant_(m.gate[-2].bias, 3.0)  # sigmoid(3.0) ≈ 0.95
```

- [ ] **Step 3.5: Verify gen.py shape test still passes**

```bash
python src/models/cfrwd/gen.py
```

Expected: prints `out: torch.Size([1, 3, 256, 256])` with no errors.

- [ ] **Step 3.6: Commit SpeckleAwareModule**

```bash
git add src/models/cfrwd/gen.py
git commit -m "feat: SpeckleAwareModule — heteroscedastic speckle variance gating

Add SpeckleAwareModule applied to Haar detail subbands g2 (LH2/HL2/HH2)
and g3 (LH1/HL1/HH1) before HFCFPreprocess+CBAM processing.

Physics basis: SAR speckle is multiplicative (I = R·n, n ~ Gamma).
In wavelet domain: coherent edges → low local variance; speckle → high
local variance (spatially uncorrelated). Gate learns to attenuate
speckle-dominated coefficients and pass coherent signal through.

Gate initializes near pass-through (bias=3.0 → sigmoid≈0.95) to avoid
forcing attenuation from epoch 0. Stores _last_g2/g3_gate_mean for logging.

Add optional log(1+relu(x)) preprocessing before DWT (disabled by default
for SEN12 — empirically worsens heteroscedasticity; see THESIS_REPORT.md)."
```

---

## Task 4: Add WaveletSupervisionLoss to losses.py

**Files:**
- Modify: `src/models/cfrwd/losses.py`

**Design:** Compute 2-level Haar DWT of both prediction and target (both B×3×H×W in [-1,1]). Compare the 6 detail subbands (LH1, HL1, HH1, LH2, HL2, HH2) via mean L1. Skip LL (structural, handled by CFR and GAN). Use float32 to match FFTLoss stability practice.

- [ ] **Step 4.1: Add `import torch.nn.functional as F` to losses.py if not present**

At the top of losses.py, after `import torch.nn as nn`, add:
```python
import torch.nn.functional as F
```

- [ ] **Step 4.2: Add `WaveletSupervisionLoss` class at the end of losses.py**

```python
class WaveletSupervisionLoss(nn.Module):
    """
    Wavelet-domain L1 supervision for the HFCF branch output.

    Computes 2-level Haar DWT of pred and target, then L1 on all 6 detail
    subbands (LH1, HL1, HH1, LH2, HL2, HH2). LL subbands are excluded:
    the HFCF branch discards LL2 in its own DWT, so supervising LL would
    penalise content the branch cannot model from its inputs.

    Architecturally matched: HFCF processes wavelet coefficients → supervise
    its output in the same domain. Unlike full-image L1, this does not force
    the HF-only branch to reconstruct LF structure.

    Input: B×3×H×W tensors in [-1, 1] (tanh output range).
    """
    def __init__(self):
        super().__init__()
        # HaarDown uses register_buffer — auto-moved with .to(device).
        # in_channels arg is unused in forward computation (works for any C).
        from src.models.cfrwd.gen import HaarDown
        self._haar = HaarDown(in_channels=1)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # float32: avoid bf16 precision loss in DWT (same rationale as FFTLoss)
        p = pred.float()
        t = target.float()

        # Level-1 decomposition
        ll1_p, lh1_p, hl1_p, hh1_p = self._haar(p)
        ll1_t, lh1_t, hl1_t, hh1_t = self._haar(t)

        # Level-2 decomposition on LL1
        _, lh2_p, hl2_p, hh2_p = self._haar(ll1_p)
        _, lh2_t, hl2_t, hh2_t = self._haar(ll1_t)

        # Mean L1 across all 6 detail subbands (equal weighting, no LL)
        return (
            F.l1_loss(lh1_p, lh1_t) + F.l1_loss(hl1_p, hl1_t) + F.l1_loss(hh1_p, hh1_t) +
            F.l1_loss(lh2_p, lh2_t) + F.l1_loss(hl2_p, hl2_t) + F.l1_loss(hh2_p, hh2_t)
        ) / 6.0
```

- [ ] **Step 4.3: Verify losses.py import works**

```bash
python -c "from src.models.cfrwd.losses import WaveletSupervisionLoss; import torch; l = WaveletSupervisionLoss().cuda(); x = torch.randn(2,3,256,256).cuda(); y = torch.randn(2,3,256,256).cuda(); print(l(x,y))"
```

Expected: prints a scalar tensor, e.g. `tensor(0.6832, device='cuda:0', grad_fn=<DivBackward0>)`. No errors.

- [ ] **Step 4.4: Commit**

```bash
git add src/models/cfrwd/losses.py
git commit -m "feat: WaveletSupervisionLoss — wavelet-domain HFCF supervision

Supervises HFCF output via L1 on 6 Haar detail subbands (LH1/HL1/HH1
at level 1, LH2/HL2/HH2 at level 2). LL excluded (HFCF branch discards
LL2 in its own DWT — no LF signal available to that branch).

Replaces the architecturally-wrong full-image L1 (hfcf_aux) which forced
the HF-only branch to reconstruct LF content → mean/gray output → inflated
fusion weight domination (w_hfcf→0.92 by epoch 17 in cfrwd-38).

Reuses HaarDown (fixed Haar weights, register_buffer) from gen.py."
```

---

## Task 5: Update factory.py

**Files:**
- Modify: `src/models/cfrwd/factory.py`

- [ ] **Step 5.1: Update imports in factory.py**

Replace:
```python
from src.models.cfrwd.losses import FeatureMatchingLoss, GANLoss, L1Loss, FFTLoss, LPIPSLoss
```

With:
```python
from src.models.cfrwd.losses import FeatureMatchingLoss, GANLoss, L1Loss, FFTLoss, LPIPSLoss, WaveletSupervisionLoss
```

- [ ] **Step 5.2: Update `build_models` to pass `use_log_preprocess` from config**

Replace:
```python
def build_models():
    cfg = _load_cfg()
    netG = CFRWDGenerator(in_channels=cfg.model.gen.in_channels)
    netD = CFRWDPatchDis(in_channels=cfg.model.dis.in_channels, ndf=cfg.model.dis.ndf, return_features=True)
    return netG, netD
```

With:
```python
def build_models():
    cfg = _load_cfg()
    use_log = getattr(cfg.loss, 'use_log_preprocessing', False)
    netG = CFRWDGenerator(in_channels=cfg.model.gen.in_channels, use_log_preprocess=use_log)
    netD = CFRWDPatchDis(in_channels=cfg.model.dis.in_channels, ndf=cfg.model.dis.ndf, return_features=True)
    return netG, netD
```

- [ ] **Step 5.3: Add `WaveletSupervisionLoss` to `build_criterions`**

Replace:
```python
def build_criterions(lpips_backbone=None) -> dict[Literal['GAN', 'FM', 'L1', 'FFT', 'LPIPS'], nn.Module]:
    cfg = _load_cfg()
    crits = {
        'GAN': GANLoss(use_lsgan=True),
        'FM':  FeatureMatchingLoss(),
        'L1':  L1Loss(),
        'FFT': FFTLoss(),
    }
```

With:
```python
def build_criterions(lpips_backbone=None) -> dict[Literal['GAN', 'FM', 'L1', 'FFT', 'WAVELET', 'LPIPS'], nn.Module]:
    cfg = _load_cfg()
    crits = {
        'GAN':     GANLoss(use_lsgan=True),
        'FM':      FeatureMatchingLoss(),
        'L1':      L1Loss(),
        'FFT':     FFTLoss(),
        'WAVELET': WaveletSupervisionLoss(),
    }
```

- [ ] **Step 5.4: Commit**

```bash
git add src/models/cfrwd/factory.py
git commit -m "feat: add WaveletSupervisionLoss to build_criterions; pass use_log_preprocess to generator"
```

---

## Task 6: Update main.py — add cfr_aux + wavelet losses + logging

**Files:**
- Modify: `src/models/cfrwd/main.py`

- [ ] **Step 6.1: Add cfr_aux and wavelet losses to G-update, after the current g_loss block**

Replace:
```python
        loss_l1  = self.criterions['L1'](fake_opt, real_opt)
        loss_fft = self.criterions['FFT'](fake_opt, real_opt)

        g_loss = (
            loss_gan * self.loss_weights['gan'] +
            loss_fm  * self.loss_weights['fm']
        )
```

With:
```python
        loss_l1  = self.criterions['L1'](fake_opt, real_opt)
        loss_fft = self.criterions['FFT'](fake_opt, real_opt)

        # CFR branch: full-image L1 is valid (CFR processes the complete spatial context)
        loss_cfr_aux = self.criterions['L1'](cfr_out, real_opt) * self.cfg.loss.get('cfr_aux_weight', 0.3)

        # HFCF branch: wavelet-domain L1 on 6 detail subbands — architecturally matched
        # (HFCF discards LL2; supervising LL would penalise content it cannot model)
        loss_wavelet = self.criterions['WAVELET'](hfcf_out, real_opt) * self.cfg.loss.get('wavelet_weight', 0.5)

        g_loss = (
            loss_gan     * self.loss_weights['gan'] +
            loss_fm      * self.loss_weights['fm'] +
            loss_cfr_aux +
            loss_wavelet
        )
```

- [ ] **Step 6.2: Update training step logging to include new losses and speckle gate stats**

Replace:
```python
        self.log('train/g_loss', g_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=real_sar.size(0))
        self.log_dict({
            'train/loss_fm':  loss_fm,
            'train/loss_gan': loss_gan,
            'train/loss_d':   d_loss,
            'train/loss_l1':  loss_l1,
            'train/loss_fft': loss_fft,
            'feats/d_real_mean': real_means.mean(),
            'feats/d_fake_mean': fake_means.mean(),
            'fusion/w_hfcf': w_hfcf.squeeze(),
        }, prog_bar=False, on_step=False, on_epoch=True, batch_size=real_sar.size(0))
```

With:
```python
        self.log('train/g_loss', g_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=real_sar.size(0))
        self.log_dict({
            'train/loss_fm':       loss_fm,
            'train/loss_gan':      loss_gan,
            'train/loss_d':        d_loss,
            'train/loss_l1':       loss_l1,
            'train/loss_fft':      loss_fft,
            'train/loss_cfr_aux':  loss_cfr_aux,
            'train/loss_wavelet':  loss_wavelet,
            'feats/d_real_mean':   real_means.mean(),
            'feats/d_fake_mean':   fake_means.mean(),
            'fusion/w_hfcf':       w_hfcf.squeeze(),
            # SpeckleAwareModule gate mean activations: 1.0 = pass-through, 0.0 = full attenuation
            'fusion/speckle_gate_mid':  self.netG.hfcf_branch._last_g2_gate_mean,
            'fusion/speckle_gate_high': self.netG.hfcf_branch._last_g3_gate_mean,
        }, prog_bar=False, on_step=False, on_epoch=True, batch_size=real_sar.size(0))
```

- [ ] **Step 6.3: Commit**

```bash
git add src/models/cfrwd/main.py
git commit -m "feat: add CFR aux L1 + wavelet supervision losses; log speckle gate stats

G-loss is now: GAN + FM×10 + L1_cfr×0.3 + L_wavelet×0.5

- loss_cfr_aux: L1(cfr_out, real_opt) — CFR branch processes full image,
  full-image L1 is architecturally valid and anchors it to optical appearance
- loss_wavelet: WaveletSupervisionLoss(hfcf_out, real_opt) — L1 on 6 Haar
  detail subbands, matching the HFCF branch's own wavelet processing domain
- Log fusion/speckle_gate_mid and _high to monitor heteroscedastic attenuation
  (target: should drift below 0.95 in regions with strong speckle)"
```

---

## Task 7: Update inference.py for scalar fusion weight

**Files:**
- Modify: `src/models/cfrwd/inference.py`

- [ ] **Step 7.1: Update forward call unpacking and fw handling in inference.py**

Replace (lines 52-60):
```python
    with torch.no_grad():
        fused, cfr_out, hfcf_out, fw = netG(sar, return_branches=True)

    sar_np = sar.detach().cpu().numpy()
    opt_np = opt.detach().cpu().numpy()
    fused_np = fused.detach().cpu().numpy()
    cfr_np = cfr_out.detach().cpu().numpy()
    hfcf_np = hfcf_out.detach().cpu().numpy()
    fw_np = fw.detach().cpu().numpy()
```

With:
```python
    with torch.no_grad():
        fused, cfr_out, hfcf_out, w_hfcf = netG(sar, return_branches=True)

    sar_np   = sar.detach().cpu().numpy()
    opt_np   = opt.detach().cpu().numpy()
    fused_np = fused.detach().cpu().numpy()
    cfr_np   = cfr_out.detach().cpu().numpy()
    hfcf_np  = hfcf_out.detach().cpu().numpy()
    w_hfcf_val = w_hfcf.item()  # scalar float
    gate_mid_val  = netG.hfcf_branch._last_g2_gate_mean.item()
    gate_high_val = netG.hfcf_branch._last_g3_gate_mean.item()
```

- [ ] **Step 7.2: Update subplot 6 rendering (replaces spatial heatmap with scalar stats panel)**

Replace (lines 96-104):
```python
        w_hfcf = fw_np[i, 1]
        im = axes[5].imshow(w_hfcf, cmap='viridis', vmin=0, vmax=1)
        cbar = plt.colorbar(im, ax=axes[5])
        mean_w = w_hfcf.mean()
        axes[5].set_title(f"w_hfcf (mean={mean_w:.3f})")
        axes[5].axis('off')

        w_hfcf_values.append(w_hfcf)
```

With:
```python
        axes[5].set_facecolor('#1a1a2e')
        axes[5].text(0.5, 0.55,
            f"w_hfcf  = {w_hfcf_val:.3f}\n"
            f"gate_mid  = {gate_mid_val:.3f}\n"
            f"gate_high = {gate_high_val:.3f}",
            ha='center', va='center', fontsize=11, color='white',
            transform=axes[5].transAxes, family='monospace')
        axes[5].set_title("Fusion & Speckle Gate")
        axes[5].axis('off')
```

- [ ] **Step 7.3: Update per-image print and overall summary (removes old w_hfcf_values list)**

Replace (lines 111-118):
```python
        std_w = w_hfcf.std()
        min_w = w_hfcf.min()
        max_w = w_hfcf.max()
        print(f"Image {i:03d}  mean_w_hfcf={mean_w:.3f}  std={std_w:.3f}  min={min_w:.3f}  max={max_w:.3f}  saved → {out_path}")

    print("━" * 40)
    overall_mean = torch.stack([torch.tensor(w) for w in w_hfcf_values]).mean().item()
    print(f"Overall mean_w_hfcf: {overall_mean:.3f}  (across {n} images)")
```

With:
```python
        print(f"Image {i:03d}  w_hfcf={w_hfcf_val:.3f}  gate_mid={gate_mid_val:.3f}  gate_high={gate_high_val:.3f}  saved → {out_path}")

    print("━" * 40)
    print(f"w_hfcf (scalar, learned): {w_hfcf_val:.3f}  |  gate_mid: {gate_mid_val:.3f}  gate_high: {gate_high_val:.3f}")
```

- [ ] **Step 7.4: Remove the now-unused `w_hfcf_values` list initialisation (line 62)**

Replace:
```python
    w_hfcf_values = []
    n = len(sar)
```

With:
```python
    n = len(sar)
```

- [ ] **Step 7.5: Commit**

```bash
git add src/models/cfrwd/inference.py
git commit -m "fix: update inference.py for scalar fusion weight and speckle gate display"
```

---

## Task 8: Smoke test — 1 epoch to verify no crashes

**Files:**
- Temporarily modify: `src/models/cfrwd/config.yaml` (revert after test)

- [ ] **Step 8.1: Set smoke-test config**

In `config.yaml`, change these fields temporarily:
```yaml
system:
  max_epochs: 1
  limit_train_batches: 0.005   # ~4-5 batches for sen12_full
  limit_val_batches: 0.01
```

- [ ] **Step 8.2: Run 1-epoch smoke test**

```bash
python -m src.models.cfrwd.train
```

Expected output (all must be present in the log):
- `train/g_loss` — scalar
- `train/loss_cfr_aux` — scalar, typically 0.3–0.5
- `train/loss_wavelet` — scalar, typically 0.1–0.3
- `fusion/w_hfcf` — scalar, should be ≈ 0.5 at epoch 0 (init)
- `fusion/speckle_gate_mid` — scalar, should be ≈ 0.95 (near pass-through at init)
- `fusion/speckle_gate_high` — scalar, should be ≈ 0.95
- `val/psnr`, `val/ssim` — computed and logged
- No CUDA OOM, no NaN losses, no KeyError in criterions

- [ ] **Step 8.3: Revert smoke-test config to full training values**

In `config.yaml`, restore:
```yaml
system:
  max_epochs: 400
  limit_train_batches: 1.0
  limit_val_batches: 1.0
```

- [ ] **Step 8.4: Commit final config**

```bash
git add src/models/cfrwd/config.yaml
git commit -m "config: physics-1 full training config (400 epochs, full batches)"
```

---

## Expected Parameter Count Delta

| Component | New trainable params |
|-----------|---------------------|
| `_fusion_logit` | 1 |
| `SpeckleAwareModule` (mid, freq_c=3) | Conv(3→8,1) + Conv(8→3,1,bias) = 24 + 27 = 51 |
| `SpeckleAwareModule` (high, freq_c=3) | same = 51 |
| `WaveletSupervisionLoss._haar` | 0 (register_buffer, fixed Haar weights) |
| **Total new** | **103 params** |

Run `python src/models/cfrwd/gen.py` and use `ModelSummary(model, max_depth=-1)` to confirm exact total.

## New TensorBoard Metrics to Monitor

| Metric | Target healthy range | Warning sign |
|--------|---------------------|--------------|
| `fusion/w_hfcf` | Drifts slowly toward 0.5–0.7 | > 0.9 (HFCF domination) |
| `fusion/speckle_gate_mid` | Gradually decreases below 0.9 | Stays at 0.95 (gate not activating) or collapses to 0.2 (over-gating) |
| `fusion/speckle_gate_high` | Same as above | Same |
| `train/loss_wavelet` | Decreasing over epochs | Stagnant or increasing (HFCF not learning HF content) |
| `train/loss_cfr_aux` | Decreasing over epochs | Stagnant (CFR not learning structure) |
| `val/psnr` | > 15 dB by epoch 50 | < 13 dB at epoch 50 (restart with adjusted weights) |
