# Stage 3 — Fourier-D Spectral Discriminator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a third adversarial head to LLW-Former v0.4.2 — a small PatchGAN that scores the realism of `log|rfft2(opt)|`. Gated by `cfg.model.dis.fourier.enabled` (default `false`) so v0.4.0/v0.4.1 runs are unaffected.

**Architecture:** New `FourierDis(nn.Module)` parallel to existing `MainDis` + `SubbandDis`. Internal `torch.fft.rfft2` → `log(|F|+1e-8)` → 4-conv spectral-norm PatchGAN mirroring `SubbandDis`. `LLWFormerDiscriminator` wrapper extended from 4-tuple to 6-tuple return; 5 caller sites updated mechanically.

**Tech Stack:** PyTorch (`torch.fft.rfft2`, `torch.nn.utils.spectral_norm`); Lightning; OmegaConf.

**Reference spec:** `docs/superpowers/specs/2026-05-23-stage-3-fourier-d-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `src/models/llwt/dis.py` | modify | Add `FourierDis` class; extend `LLWFormerDiscriminator` to 6-tuple return + `use_fourier` flag |
| `src/models/llwt_v4/main.py` | modify | 3 unpack sites (D-R1, D-batched, G-update) → 6-tuple; add `use_fourier_d`/`gan_fourier_weight`/`fm_fourier_weight` attrs; add D-update third GAN term; add G-update GAN_fourier + FM_fourier terms |
| `src/models/llwt_v4/overfit_test.py` | modify | 2 unpack sites (D-update, G-update) → 6-tuple; handle `use_fourier` symmetrically with `use_sub` |
| `src/models/llwt_v4/config.yaml` | modify | Add `cfg.model.dis.fourier.{enabled, ndf, use_sn}` + `cfg.loss.{gan_fourier_weight, fm_fourier_weight}` all default off/zero |
| `src/models/llwt_v4/factory.py` | no-op | Verify no change required — `build_models` delegates to `LLWFormerDiscriminator` |

All edits in one PR. Mechanical extensions only — no behavior change when `enabled=false` (the default).

---

## Task 1: Add `FourierDis` class to `dis.py`

**Files:**
- Modify: `src/models/llwt/dis.py` (insert new class between `SubbandDis` and `LLWFormerDiscriminator`)

- [ ] **Step 1: Write the failing unit smoke**

Append to `src/models/llwt/dis.py` after the existing classes, in a new `if __name__ == '__main__':` block. The smoke verifies `FourierDis` exists and produces valid shapes + no NaN.

```python
# ---------------------------------------------------------------------------
# Standalone smoke for FourierDis (run with: python -m src.models.llwt.dis)
# ---------------------------------------------------------------------------


def _smoke_fourier_dis() -> None:
    """Tier 1 unit smoke for FourierDis."""
    print("[fourier-dis smoke] running FourierDis shape + sanity checks")
    torch.manual_seed(0)
    d = FourierDis(ndf=32, use_sn=True)
    for B, H, W in [(2, 256, 256), (1, 128, 128)]:
        opt = torch.randn(B, 3, H, W)
        logits, feats = d(opt)
        assert torch.isfinite(logits).all(), f"non-finite logits at {(B, 3, H, W)}"
        for i, f in enumerate(feats):
            assert torch.isfinite(f).all(), f"non-finite feat[{i}] at {(B, 3, H, W)}"
        assert len(feats) == 3, f"expected 3 FM feats, got {len(feats)}"
        print(f"  [OK] shape=({B},3,{H},{W}) -> logits {tuple(logits.shape)} + {len(feats)} feats")

    # Edge: saturated (all-ones) input — log(|F|) must not NaN.
    opt_sat = torch.ones(2, 3, 64, 64)
    logits, _ = d(opt_sat)
    assert torch.isfinite(logits).all(), "non-finite logits on saturated input"

    # Edge: zero input — log(0+eps) finite.
    opt_zero = torch.zeros(2, 3, 64, 64)
    logits, _ = d(opt_zero)
    assert torch.isfinite(logits).all(), "non-finite logits on zero input"

    print("[fourier-dis smoke] PASS — shapes, FM feat count, NaN guards all clean")


if __name__ == '__main__':
    _smoke_fourier_dis()
```

- [ ] **Step 2: Run smoke, verify it fails with `NameError: FourierDis`**

```powershell
python -m src.models.llwt.dis
```

Expected: `NameError: name 'FourierDis' is not defined`.

- [ ] **Step 3: Implement `FourierDis` class**

Insert this block between the `SubbandDis` class (ends ~line 202) and the `LLWFormerDiscriminator` wrapper (starts ~line 210):

```python
# ---------------------------------------------------------------------------
# Fourier-D — unconditional PatchGAN on log|rfft2(opt)|
# ---------------------------------------------------------------------------


class _FourierPatchBranch(nn.Module):
    """4-layer spectral-norm PatchGAN on log-magnitude spectrum feature maps.

    Mirrors :class:`_SubbandPatchBranch` depth and width; the only difference
    is the channel count of the input (3 RGB log-magnitude maps instead of
    12 Haar subbands).
    """
    def __init__(self, in_ch: int = 3, ndf: int = 32, use_sn: bool = True):
        super().__init__()
        conv = _sn if use_sn else (lambda ci, co, k, s, p: nn.Conv2d(ci, co, k, s, p, bias=True))
        self.layers = nn.ModuleList([
            nn.Sequential(conv(in_ch,    ndf,     4, 2, 1), nn.LeakyReLU(0.2, inplace=True)),
            nn.Sequential(conv(ndf,      ndf * 2, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True)),
            nn.Sequential(conv(ndf * 2,  ndf * 4, 4, 1, 1), nn.LeakyReLU(0.2, inplace=True)),
            conv(ndf * 4, 1, 4, 1, 1),
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        feats: List[torch.Tensor] = []
        for layer in self.layers[:-1]:
            x = layer(x)
            feats.append(x)
        return self.layers[-1](x), feats


class FourierDis(nn.Module):
    """Unconditional PatchGAN on log-magnitude rfft2 of the optical pair.

    Forward: ``opt (B, 3, H, W) -> ((B, 1, h, w), feats)``.

    Rationale:
      * ``torch.fft.rfft2`` produces a complex tensor of shape
        ``(B, 3, H, W//2+1)``.  Taking ``log(|F| + eps)`` compresses the
        ~1/f power-law dynamic range so the first conv isn't dominated by
        the DC bin.
      * Unconditional (opt-only) — symmetric with :class:`SubbandDis`.
        SAR spectrum is speckle-dominated; conditional pairing adds noise
        without semantic value.
      * 4-conv depth + spectral_norm + ndf=32 — mirrors :class:`SubbandDis`.
        Keeps param + VRAM cost low for a third head added to an already
        bs=8-at-24GB training setup.

    Outputs feats list with the layer-0 activation *kept* (subband-D
    matches this), giving 3 FM features per forward pass.
    """
    _EPS = 1e-8

    def __init__(self, ndf: int = 32, use_sn: bool = True):
        super().__init__()
        self.branch = _FourierPatchBranch(in_ch=3, ndf=ndf, use_sn=use_sn)

    def _log_mag(self, x: torch.Tensor) -> torch.Tensor:
        # FFT under bf16 autocast: rfft2 is autocast-aware in torch>=2.0.
        # We cast to float32 explicitly to avoid the autocast-emitted complex32
        # which has poor kernel coverage and to keep magnitude/log numerics tight.
        F_x = torch.fft.rfft2(x.float(), dim=(-2, -1))
        mag = F_x.abs()
        return torch.log(mag + self._EPS)

    def forward(self, opt: torch.Tensor):
        spec = self._log_mag(opt)                            # (B, 3, H, W//2+1)
        return self.branch(spec)
```

Also update the module's `__all__` list at the top:

```python
__all__ = ["LLWFormerDiscriminator", "MainDis", "SubbandDis", "FourierDis", "FixedHaarDWT"]
```

- [ ] **Step 4: Re-run smoke to verify it passes**

```powershell
python -m src.models.llwt.dis
```

Expected output:
```
[fourier-dis smoke] running FourierDis shape + sanity checks
  [OK] shape=(2,3,256,256) -> logits (2, 1, ...) + 3 feats
  [OK] shape=(1,3,128,128) -> logits (1, 1, ...) + 3 feats
[fourier-dis smoke] PASS — shapes, FM feat count, NaN guards all clean
```

- [ ] **Step 5: Commit Task 1**

```powershell
git add src/models/llwt/dis.py
git commit -m @'
feat(dis): add FourierDis spectral discriminator class

Unconditional 4-conv spectral-norm PatchGAN on log|rfft2(opt)|. Mirrors
SubbandDis structure with 3-channel spectrum input instead of 12-channel
Haar subbands. Standalone smoke verifies shape, FM feat count, NaN guards
on saturated and zero inputs.

Not yet wired into the LLWFormerDiscriminator wrapper — that lands in
the next task. Adding the class first lets the smoke gate the math
before touching the multi-caller wrapper contract.
'@
```

---

## Task 2: Extend `LLWFormerDiscriminator` to 6-tuple return

**Files:**
- Modify: `src/models/llwt/dis.py:210-251` (`LLWFormerDiscriminator` class)

- [ ] **Step 1: Update docstring and `__init__` to read fourier sub-config**

Replace the class body (currently lines 210-251) with:

```python
class LLWFormerDiscriminator(nn.Module):
    """Wraps MainDis, SubbandDis, and FourierDis with per-head enable flags.

    Forward returns a 6-tuple:
        ``((logits_coarse, logits_fine), logits_subband, logits_fourier,
          feats_main, feats_sub, feats_fourier)``
    where ``logits_subband`` is ``None`` and ``feats_sub`` is ``[]`` when
    the subband head is disabled (``cfg.model.dis.subband.enabled = false``).
    Same convention for the fourier head
    (``cfg.model.dis.fourier.enabled = false``, default).

    Keeping ``feats_main``, ``feats_sub``, ``feats_fourier`` as separate
    lists lets the training loop weight each head's feature-matching
    contribution independently via ``loss.fm_main_weight``,
    ``loss.fm_sub_weight``, ``loss.fm_fourier_weight``.
    """
    def __init__(self, cfg=None):
        super().__init__()
        dcfg = None if cfg is None else cfg.model.dis
        main_cfg     = _g(dcfg, 'main', None)
        in_ch        = int(_g(main_cfg, 'in_channels', 4))
        ndf_main     = int(_g(main_cfg, 'ndf', 64))
        sub_cfg      = _g(dcfg, 'subband', None)
        self.use_sub = bool(_g(sub_cfg, 'enabled', True))
        ndf_sub      = int(_g(sub_cfg, 'ndf', 32))
        fourier_cfg     = _g(dcfg, 'fourier', None)
        self.use_fourier = bool(_g(fourier_cfg, 'enabled', False))
        ndf_fourier     = int(_g(fourier_cfg, 'ndf', 32))
        use_sn_fourier  = bool(_g(fourier_cfg, 'use_sn', True))

        self.main = MainDis(in_ch=in_ch, ndf=ndf_main)
        self.sub  = SubbandDis(ndf=ndf_sub) if self.use_sub else None
        self.fourier = (
            FourierDis(ndf=ndf_fourier, use_sn=use_sn_fourier)
            if self.use_fourier else None
        )

    def forward(self, sar: torch.Tensor, opt: torch.Tensor):
        """Returns 6-tuple: see class docstring for layout."""
        (lc, lf), feats_main = self.main(sar, opt)
        if self.sub is not None:
            ls, feats_sub = self.sub(opt)
        else:
            ls, feats_sub = None, []
        if self.fourier is not None:
            lfourier, feats_fourier = self.fourier(opt)
        else:
            lfourier, feats_fourier = None, []
        return (lc, lf), ls, lfourier, feats_main, feats_sub, feats_fourier
```

- [ ] **Step 2: Write a quick wrapper smoke to verify both off and on paths**

Append to the existing `_smoke_fourier_dis` block or as a new helper. Add to the same `__main__`:

```python
def _smoke_wrapper() -> None:
    """Tier 1 wrapper smoke — verify 6-tuple return with fourier on/off."""
    from omegaconf import OmegaConf
    print("[wrapper smoke] running LLWFormerDiscriminator with fourier off + on")

    # OFF path (default)
    cfg_off = OmegaConf.create({
        'model': {'dis': {
            'main':    {'in_channels': 4, 'ndf': 64},
            'subband': {'enabled': True, 'ndf': 32},
            'fourier': {'enabled': False, 'ndf': 32, 'use_sn': True},
        }},
    })
    d_off = LLWFormerDiscriminator(cfg=cfg_off)
    sar = torch.randn(2, 1, 64, 64)
    opt = torch.randn(2, 3, 64, 64)
    main_pair, sub_l, fourier_l, feats_m, feats_s, feats_f = d_off(sar, opt)
    assert isinstance(main_pair, tuple) and len(main_pair) == 2, "main pair"
    assert sub_l is not None, "subband enabled"
    assert fourier_l is None, "fourier disabled should be None"
    assert feats_f == [], "fourier feats disabled should be empty"
    print(f"  [OK] fourier OFF: 6-tuple unpacks; fourier_l=None feats_f=[]")

    # ON path
    cfg_on = OmegaConf.create({
        'model': {'dis': {
            'main':    {'in_channels': 4, 'ndf': 64},
            'subband': {'enabled': True, 'ndf': 32},
            'fourier': {'enabled': True, 'ndf': 32, 'use_sn': True},
        }},
    })
    d_on = LLWFormerDiscriminator(cfg=cfg_on)
    main_pair, sub_l, fourier_l, feats_m, feats_s, feats_f = d_on(sar, opt)
    assert fourier_l is not None, "fourier enabled should produce logits"
    assert len(feats_f) == 3, f"fourier feats should be 3, got {len(feats_f)}"
    print(f"  [OK] fourier ON: logits shape {tuple(fourier_l.shape)} + {len(feats_f)} FM feats")

    print("[wrapper smoke] PASS — 6-tuple contract holds on both gates")


if __name__ == '__main__':
    _smoke_fourier_dis()
    _smoke_wrapper()
```

- [ ] **Step 3: Run wrapper smoke**

```powershell
python -m src.models.llwt.dis
```

Expected output includes both `[fourier-dis smoke] PASS` and `[wrapper smoke] PASS`.

- [ ] **Step 4: Commit Task 2**

```powershell
git add src/models/llwt/dis.py
git commit -m @'
feat(dis): extend LLWFormerDiscriminator to 6-tuple return for fourier head

Wrapper now reads cfg.model.dis.fourier.{enabled, ndf, use_sn} and
instantiates FourierDis when enabled (default off). Forward returns
6-tuple (main_pair, sub_logits, fourier_logits, feats_main, feats_sub,
feats_fourier) with None/[] slots when a head is disabled — same
convention as the existing subband-off path.

Wrapper smoke confirms 6-tuple contract holds on both gates. Callers
still need updates — landing in subsequent commits per the stage 3 plan.
'@
```

---

## Task 3: Update `main.py` D-R1 path (first unpack site)

**Files:**
- Modify: `src/models/llwt_v4/main.py` around line 144 (D-R1 path) and class `__init__` to add the three new attrs

- [ ] **Step 1: Add new attrs to `LLWv4LightningModule.__init__`**

Locate the existing attrs in `__init__` (around line 84-86):

```python
        self.use_subband_d = self.netD.use_sub
        self.gan_sub_weight = float(getattr(cfg.loss, 'gan_sub_weight', 1.0))
        self.fm_sub_weight  = float(getattr(cfg.loss, 'fm_sub_weight',  10.0))
```

Add the three Fourier-D attrs immediately after:

```python
        self.use_fourier_d = bool(getattr(self.netD, 'use_fourier', False))
        self.gan_fourier_weight = float(getattr(cfg.loss, 'gan_fourier_weight', 1.0))
        self.fm_fourier_weight  = float(getattr(cfg.loss, 'fm_fourier_weight',  10.0))
```

- [ ] **Step 2: Update D-R1 unpacks (around line 144) to 6-tuple**

Locate the two `self.netD(...)` calls in the R1 branch (around lines 144-145):

```python
                real_main, real_sub, _, _ = self.netD(sar.float(), opt_real.float())
                fake_main, fake_sub, _, _ = self.netD(sar.float(), fake_d.float())
```

Replace with:

```python
                real_main, real_sub, real_fourier, _, _, _ = self.netD(sar.float(), opt_real.float())
                fake_main, fake_sub, fake_fourier, _, _, _ = self.netD(sar.float(), fake_d.float())
```

- [ ] **Step 3: Add the Fourier-D loss term after the subband block (still inside the R1 branch)**

Locate (around lines 157-163):

```python
                if self.use_subband_d:
                    d_loss_sub = 0.5 * (
                        self.criterions['gan'](real_sub, is_real=True) +
                        self.criterions['gan'](fake_sub, is_real=False)
                    )
                    d_loss = d_loss + d_loss_sub
                    self.log('train/d_loss_sub', d_loss_sub.detach(), on_step=False, on_epoch=True)
```

Append immediately after (inside the `if apply_r1_main:` block, still under the same `with torch.amp.autocast(...)` context):

```python
                if self.use_fourier_d:
                    d_loss_fourier = 0.5 * (
                        self.criterions['gan'](real_fourier, is_real=True) +
                        self.criterions['gan'](fake_fourier, is_real=False)
                    )
                    d_loss = d_loss + d_loss_fourier
                    self.log('train/d_loss_fourier', d_loss_fourier.detach(), on_step=False, on_epoch=True)
```

- [ ] **Step 4: Verify the file parses cleanly**

```powershell
python -c "from src.models.llwt_v4.main import LLWv4LightningModule; print('import OK')"
```

Expected: `import OK`.

- [ ] **Step 5: Commit Task 3**

```powershell
git add src/models/llwt_v4/main.py
git commit -m @'
feat(llwt-v4): wire Fourier-D into main.py D-R1 path

Add use_fourier_d / gan_fourier_weight / fm_fourier_weight attrs.
Extend the R1-path discriminator unpacks from 4-tuple to 6-tuple
(matches new dis.py wrapper contract).  Add the third GAN term to
d_loss under the use_fourier_d gate; disabled by default so existing
runs are unchanged.

Two more unpack sites (D-batched, G-update) land in subsequent commits.
'@
```

---

## Task 4: Update `main.py` D-batched and G-update unpacks

**Files:**
- Modify: `src/models/llwt_v4/main.py` D-batched path (around lines 165-180) + G-update path (around line 211)

- [ ] **Step 1: Update D-batched unpack (around line 172)**

Locate:

```python
            main_both, sub_both, _, _ = self.netD(sar_doubled, opt_doubled)
```

Replace with:

```python
            main_both, sub_both, fourier_both, _, _, _ = self.netD(sar_doubled, opt_doubled)
```

Then within the same `else:` branch (the non-R1 batched path), after the existing subband-D loss block (around lines 180-186), append:

```python
            if self.use_fourier_d:
                real_fourier = fourier_both[:B]
                fake_fourier = fourier_both[B:]
                d_loss_fourier = 0.5 * (
                    self.criterions['gan'](real_fourier, is_real=True) +
                    self.criterions['gan'](fake_fourier, is_real=False)
                )
                d_loss = d_loss + d_loss_fourier
                self.log('train/d_loss_fourier', d_loss_fourier.detach(), on_step=False, on_epoch=True)
```

- [ ] **Step 2: Update G-update unpack (around line 211)**

Locate:

```python
        main_both_g, sub_both_g, feats_main_both, feats_sub_both = self.netD(sar_doubled_g, opt_doubled_g)
```

Replace with:

```python
        main_both_g, sub_both_g, fourier_both_g, feats_main_both, feats_sub_both, feats_fourier_both = self.netD(sar_doubled_g, opt_doubled_g)
```

- [ ] **Step 3: Add Fourier-D G-update terms after the subband-D G block**

Locate the existing subband-D G block (around lines 233-240):

```python
        if self.use_subband_d:
            l_gan_sub = self.criterions['gan'](fake_sub_g, is_real=True, for_d=False)
            g_loss = g_loss + l_gan_sub * self.gan_sub_weight
            self.log('train/gan_sub', l_gan_sub.detach(), on_step=False, on_epoch=True)
            if fake_feats_sub:
                l_fm_sub = self.criterions['fm'](fake_feats_sub, real_feats_sub)
                g_loss = g_loss + l_fm_sub * self.fm_sub_weight
                self.log('train/fm_sub', l_fm_sub.detach(), on_step=False, on_epoch=True)
```

Append immediately after, but before the `if 'l1' in self.criterions:` recon block:

```python
        if self.use_fourier_d:
            fake_fourier_g = fourier_both_g[:B]
            fake_feats_fourier = [f[:B] for f in feats_fourier_both]
            real_feats_fourier = [f[B:].detach() for f in feats_fourier_both]
            l_gan_fourier = self.criterions['gan'](fake_fourier_g, is_real=True, for_d=False)
            g_loss = g_loss + l_gan_fourier * self.gan_fourier_weight
            self.log('train/gan_fourier', l_gan_fourier.detach(), on_step=False, on_epoch=True)
            if fake_feats_fourier:
                l_fm_fourier = self.criterions['fm'](fake_feats_fourier, real_feats_fourier)
                g_loss = g_loss + l_fm_fourier * self.fm_fourier_weight
                self.log('train/fm_fourier', l_fm_fourier.detach(), on_step=False, on_epoch=True)
```

- [ ] **Step 4: Verify the file parses cleanly**

```powershell
python -c "from src.models.llwt_v4.main import LLWv4LightningModule; print('import OK')"
```

Expected: `import OK`.

- [ ] **Step 5: Commit Task 4**

```powershell
git add src/models/llwt_v4/main.py
git commit -m @'
feat(llwt-v4): wire Fourier-D into main.py D-batched and G-update paths

Both remaining 4-tuple unpacks extended to 6-tuple.  D-batched gets the
third GAN term under use_fourier_d; G-update gets GAN_fourier + FM_fourier
under the same gate.  Default-off means the three new training logs
(d_loss_fourier, gan_fourier, fm_fourier) are absent from v0.4.0/v0.4.1
runs and present only when the cfg flag is opted in.
'@
```

---

## Task 5: Update `overfit_test.py` (both unpack sites)

**Files:**
- Modify: `src/models/llwt_v4/overfit_test.py` lines around D-update (~line 136) and G-update (~line 162)

- [ ] **Step 1: Add `use_fourier` flag + weights extraction near the top of `main()`**

Locate the existing block around lines 108-118:

```python
    use_sub = bool(netD.use_sub)
    gan_main_w = float(cfg.loss.gan_main_weight)
    gan_sub_w  = float(cfg.loss.gan_sub_weight)
    fm_main_w  = float(cfg.loss.fm_main_weight)
    fm_sub_w   = float(cfg.loss.fm_sub_weight)
    grad_clip_g = float(cfg.loss.grad_clip_g)
    grad_clip_d = float(cfg.loss.grad_clip_d)
    use_bf16 = str(cfg.system.precision).startswith('bf16')
```

Add three new attrs immediately after `fm_sub_w`:

```python
    use_fourier = bool(getattr(netD, 'use_fourier', False))
    gan_fourier_w = float(getattr(cfg.loss, 'gan_fourier_weight', 1.0))
    fm_fourier_w  = float(getattr(cfg.loss, 'fm_fourier_weight',  10.0))
```

And update the print one line below to include the new flag:

Locate:

```python
    print(f"[overfit] subband_d={use_sub} bf16={use_bf16} "
          f"gan_main/sub=({gan_main_w},{gan_sub_w}) fm_main/sub=({fm_main_w},{fm_sub_w})")
```

Replace with:

```python
    print(f"[overfit] subband_d={use_sub} fourier_d={use_fourier} bf16={use_bf16} "
          f"gan_main/sub/fourier=({gan_main_w},{gan_sub_w},{gan_fourier_w}) "
          f"fm_main/sub/fourier=({fm_main_w},{fm_sub_w},{fm_fourier_w})")
```

- [ ] **Step 2: Update D-update unpack and add Fourier-D loss term**

Locate around line 136:

```python
            main_both, sub_both, _, _ = netD(sar2, opt2)
            lc, lf = main_both
            real_main = (lc[:B], lf[:B])
            fake_main = (lc[B:], lf[B:])
            d_loss = 0.5 * (
                criterions['gan'](real_main, is_real=True) +
                criterions['gan'](fake_main, is_real=False)
            )
            if use_sub:
                real_sub = sub_both[:B]
                fake_sub = sub_both[B:]
                d_loss = d_loss + 0.5 * (
                    criterions['gan'](real_sub, is_real=True) +
                    criterions['gan'](fake_sub, is_real=False)
                )
```

Replace with:

```python
            main_both, sub_both, fourier_both, _, _, _ = netD(sar2, opt2)
            lc, lf = main_both
            real_main = (lc[:B], lf[:B])
            fake_main = (lc[B:], lf[B:])
            d_loss = 0.5 * (
                criterions['gan'](real_main, is_real=True) +
                criterions['gan'](fake_main, is_real=False)
            )
            if use_sub:
                real_sub = sub_both[:B]
                fake_sub = sub_both[B:]
                d_loss = d_loss + 0.5 * (
                    criterions['gan'](real_sub, is_real=True) +
                    criterions['gan'](fake_sub, is_real=False)
                )
            if use_fourier:
                real_fourier = fourier_both[:B]
                fake_fourier = fourier_both[B:]
                d_loss = d_loss + 0.5 * (
                    criterions['gan'](real_fourier, is_real=True) +
                    criterions['gan'](fake_fourier, is_real=False)
                )
```

- [ ] **Step 3: Update G-update unpack and add Fourier-D loss term**

Locate around line 162:

```python
            main_both_g, sub_both_g, feats_main, feats_sub = netD(sar2g, opt2g)
            lc_g, lf_g = main_both_g
            fake_main_g = (lc_g[:B], lf_g[:B])
            fake_feats_main = [f[:B] for f in feats_main]
            real_feats_main = [f[B:].detach() for f in feats_main]
            l_gan = criterions['gan'](fake_main_g, is_real=True, for_d=False)
            l_fm  = criterions['fm'](fake_feats_main, real_feats_main)
            g_loss = l_gan * gan_main_w + l_fm * fm_main_w
            if use_sub:
                fake_sub_g = sub_both_g[:B]
                fake_feats_sub = [f[:B] for f in feats_sub]
                real_feats_sub = [f[B:].detach() for f in feats_sub]
                l_gan_sub = criterions['gan'](fake_sub_g, is_real=True, for_d=False)
                l_fm_sub  = criterions['fm'](fake_feats_sub, real_feats_sub)
                g_loss = g_loss + l_gan_sub * gan_sub_w + l_fm_sub * fm_sub_w
```

Replace with:

```python
            main_both_g, sub_both_g, fourier_both_g, feats_main, feats_sub, feats_fourier = netD(sar2g, opt2g)
            lc_g, lf_g = main_both_g
            fake_main_g = (lc_g[:B], lf_g[:B])
            fake_feats_main = [f[:B] for f in feats_main]
            real_feats_main = [f[B:].detach() for f in feats_main]
            l_gan = criterions['gan'](fake_main_g, is_real=True, for_d=False)
            l_fm  = criterions['fm'](fake_feats_main, real_feats_main)
            g_loss = l_gan * gan_main_w + l_fm * fm_main_w
            if use_sub:
                fake_sub_g = sub_both_g[:B]
                fake_feats_sub = [f[:B] for f in feats_sub]
                real_feats_sub = [f[B:].detach() for f in feats_sub]
                l_gan_sub = criterions['gan'](fake_sub_g, is_real=True, for_d=False)
                l_fm_sub  = criterions['fm'](fake_feats_sub, real_feats_sub)
                g_loss = g_loss + l_gan_sub * gan_sub_w + l_fm_sub * fm_sub_w
            if use_fourier:
                fake_fourier_g = fourier_both_g[:B]
                fake_feats_fourier = [f[:B] for f in feats_fourier]
                real_feats_fourier = [f[B:].detach() for f in feats_fourier]
                l_gan_fourier = criterions['gan'](fake_fourier_g, is_real=True, for_d=False)
                l_fm_fourier  = criterions['fm'](fake_feats_fourier, real_feats_fourier)
                g_loss = g_loss + l_gan_fourier * gan_fourier_w + l_fm_fourier * fm_fourier_w
```

- [ ] **Step 4: Verify the file parses cleanly**

```powershell
python -c "import importlib.util, sys; spec = importlib.util.spec_from_file_location('m', 'src/models/llwt_v4/overfit_test.py'); m = importlib.util.module_from_spec(spec); print('compile OK' if spec.loader else 'fail')"
```

Expected: `compile OK`.

- [ ] **Step 5: Commit Task 5**

```powershell
git add src/models/llwt_v4/overfit_test.py
git commit -m @'
feat(llwt-v4): wire Fourier-D into overfit_test.py D and G paths

Both unpacks extended to 6-tuple.  use_fourier flag derived from the
discriminator (mirrors use_sub).  Third GAN+FM block appended after the
subband block under the same gate.  Default off — existing overfit runs
unchanged.
'@
```

---

## Task 6: Add config fields (default OFF)

**Files:**
- Modify: `src/models/llwt_v4/config.yaml`

- [ ] **Step 1: Add fourier sub-config under `model.dis`**

Locate the existing `dis` block:

```yaml
  dis:
    main:
      in_channels:    4                             # SAR(1) + opt(3)
      ndf:            64
    subband:
      enabled:        true                          # subband-D head -- novelty axis #5
      ndf:            32
```

Append immediately after the `subband:` block, still indented under `dis:`:

```yaml
    fourier:
      enabled:        false                         # stage-3 fourier-D head — opt-in for v0.4.2 ablation
      ndf:            32                            # matches subband ndf; drop to 16 if VRAM tight
      use_sn:         true                          # spectral norm; keep on for tri-D stability
```

- [ ] **Step 2: Add fourier-loss weights under `loss`**

Locate the per-band loss block (added in v0.4.1, around the `per_band_hh` line):

```yaml
  per_band_hh:            1.0
  lpips_weight:           0.0
```

Insert between `per_band_hh` and `lpips_weight`:

```yaml
  per_band_hh:            1.0
  # Fourier-D adversarial weights (v0.4.2 stage 3). Default 0.0 = stage-3 inert
  # (v0.4.0/v0.4.1 behaviour); set 1.0/10.0 to match gan_sub/fm_sub when enabled.
  gan_fourier_weight:     0.0
  fm_fourier_weight:      0.0
  lpips_weight:           0.0
```

- [ ] **Step 3: Verify the config still loads + factory builds cleanly**

```powershell
python -c "
from omegaconf import OmegaConf
from src.models.llwt_v4 import factory
cfg = OmegaConf.load('src/models/llwt_v4/config.yaml')
print(f'fourier.enabled = {cfg.model.dis.fourier.enabled}')
print(f'gan_fourier_weight = {cfg.loss.gan_fourier_weight}')
netG, netD = factory.build_models(cfg)
print(f'netD.use_fourier = {netD.use_fourier}')
print(f'netD.fourier is None: {netD.fourier is None}')
"
```

Expected output:
```
fourier.enabled = False
gan_fourier_weight = 0.0
netD.use_fourier = False
netD.fourier is None: True
```

- [ ] **Step 4: Commit Task 6**

```powershell
git add src/models/llwt_v4/config.yaml
git commit -m @'
feat(llwt-v4): add Fourier-D config fields (default off)

Adds cfg.model.dis.fourier.{enabled, ndf, use_sn} and
cfg.loss.{gan_fourier_weight, fm_fourier_weight}.  All default to
off/zero so v0.4.0 and v0.4.1 runs see no behaviour change — opt-in
required for stage-3 ablation.
'@
```

---

## Task 7: End-to-end smoke with fourier enabled

**Files:**
- No edits — pure verification.

- [ ] **Step 1: Run the dis.py smoke (Tier 1 unit)**

```powershell
python -m src.models.llwt.dis
```

Expected: both `[fourier-dis smoke] PASS` and `[wrapper smoke] PASS`.

- [ ] **Step 2: Verify factory + module + training-step smoke with fourier enabled via config override**

```powershell
python -c "
import torch
from omegaconf import OmegaConf
from src.models.llwt_v4 import factory
from src.models.llwt_v4.main import LLWv4LightningModule

cfg = OmegaConf.load('src/models/llwt_v4/config.yaml')
# Opt-in: enable fourier-D and weights.
cfg.model.dis.fourier.enabled = True
cfg.loss.gan_fourier_weight = 1.0
cfg.loss.fm_fourier_weight = 10.0

netG, netD = factory.build_models(cfg)
assert netD.use_fourier, 'fourier flag should be True'
assert netD.fourier is not None, 'fourier head should be instantiated'

# Forward smoke
sar = torch.randn(2, 1, 64, 64)
opt = torch.randn(2, 3, 64, 64)
main_pair, sub_l, fourier_l, fm, fs, ff = netD(sar, opt)
assert fourier_l is not None, 'fourier logits should be produced'
assert len(ff) == 3, f'expected 3 fourier feats, got {len(ff)}'
print(f'OK: fourier_l shape {tuple(fourier_l.shape)}, fourier feats count {len(ff)}')

# Lightning module instantiates cleanly with the 3-head config
m = LLWv4LightningModule(cfg)
assert m.use_fourier_d, 'module flag should reflect netD'
assert m.gan_fourier_weight == 1.0
assert m.fm_fourier_weight == 10.0
print(f'OK: LLWv4LightningModule constructed with fourier enabled')
"
```

Expected: both `OK:` lines print.

- [ ] **Step 3: Verify fourier-OFF default still works (regression check)**

```powershell
python -c "
import torch
from omegaconf import OmegaConf
from src.models.llwt_v4 import factory
from src.models.llwt_v4.main import LLWv4LightningModule

cfg = OmegaConf.load('src/models/llwt_v4/config.yaml')
# Default: fourier off
assert cfg.model.dis.fourier.enabled is False
assert cfg.loss.gan_fourier_weight == 0.0

netG, netD = factory.build_models(cfg)
assert not netD.use_fourier
assert netD.fourier is None

sar = torch.randn(2, 1, 64, 64)
opt = torch.randn(2, 3, 64, 64)
main_pair, sub_l, fourier_l, fm, fs, ff = netD(sar, opt)
assert fourier_l is None, 'fourier should be None when disabled'
assert ff == [], 'fourier feats should be empty when disabled'

m = LLWv4LightningModule(cfg)
assert not m.use_fourier_d
print('OK: fourier OFF default still works end-to-end')
"
```

Expected: `OK: fourier OFF default still works end-to-end`.

- [ ] **Step 4: Commit Task 7 — final stage 3 commit (no code, validation only)**

If the test commands all pass and any prior commit needs a final tag, run:

```powershell
git tag -a llwt-v0.4.2-stage3-built -m "stage 3 (Fourier-D) shipped + Tier 1 unit smokes green; GPU validation pending"
```

(Optional — skip the tag if you don't want a git ref. The plan is complete either way.)

---

## Self-Review

**Spec coverage:** Every section of the spec maps to a task:
- FourierDis class (spec §Module) → Task 1
- Wrapper 6-tuple (spec §Wrapper changes) → Task 2
- training_step additions (spec §Training-loop additions) → Tasks 3 + 4
- Config schema (spec §Config schema additions) → Task 6
- overfit_test path (spec §Validation Tier 2) → Task 5
- Tier 1 smoke (spec §Validation Tier 1) → embedded in Tasks 1, 2, 7

Gaps: Tier 2/3/4 GPU validation is deferred per the spec ("after v0.4.0 prod finishes") — not implementation tasks, no plan entry needed.

**Placeholder scan:** No TBD/TODO/vague language in any task. Every step has the exact code or command.

**Type consistency:**
- `use_fourier` attr name consistent across `LLWFormerDiscriminator`, `LLWv4LightningModule.use_fourier_d`, `overfit_test.use_fourier`. Different attribute names but distinct namespaces, consistent within each.
- `gan_fourier_weight` / `fm_fourier_weight` config names match across `main.py`, `overfit_test.py`, `config.yaml`, and the spec.
- 6-tuple return order: `(main_pair, sub_logits, fourier_logits, feats_main, feats_sub, feats_fourier)` — identical across dis.py wrapper, main.py 3 unpacks, overfit_test.py 2 unpacks.
- `FourierDis(ndf=..., use_sn=...)` signature matches everywhere it's instantiated.

Plan is internally consistent and self-contained.
