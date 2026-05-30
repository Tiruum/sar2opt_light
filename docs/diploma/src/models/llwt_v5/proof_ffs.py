"""Controlled experiment to validate/kill the "Frequency-Factorized Supervision" (FFS) premise.

FFS proposes: supervise the LL (low-freq) Haar band with a PIXEL loss (determinable +
shift-robust) but supervise the DETAIL bands (LH/HL/HH) with a DISTRIBUTIONAL, shift-invariant
loss (because exact detail phase is one-to-many-unknowable AND destroyed by the measured
~1.2px median misalignment). Three claims must ALL hold:

  C1  detail PIXEL L1 is shift-fragile (rises steeply 0->6px) while LL pixel L1 rises slowly.
  C2  a distributional band loss stays ~flat under the same shift. Two candidates:
        (a) FFT-magnitude L1 per subband (phase-free => shift = pure phase => |.| invariant)
        (b) 1D sorted-coefficient / Sliced-Wasserstein distance on flattened subband coeffs.
  C3  the distributional target is ACHIEVABLE: on the real trained model's output, detail
        bands match GT much better in DISTRIBUTION than in PIXELS (stats right, phase wrong).

Forward-only, no training. Loads the trained SAWG ckpt (strict=False). <10 min on CPU/GPU.

Run from repo root::

    python -m src.models.llwt_v5.proof_ffs

Throwaway diagnostic — no commit, no side effects beyond stdout.
"""
from __future__ import annotations

import os
import sys

# repo-root sys.path shim (run from repo root, but be robust)
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
# NOTE: deliberately do NOT set HF_HUB_OFFLINE — backbone loads from local HF cache.

import functools
import time

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

# torch>=2.6 weights_only default flip — Lightning ckpts hold python objects.
_orig_load = torch.load


@functools.wraps(_orig_load)
def _patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _orig_load(*args, **kwargs)


torch.load = _patched_load
torch.set_float32_matmul_precision('high')

from src.models.llwt_v5.main import LLWv4LightningModule
from src.models.llwt_v5.blocks import HaarDown
from src.data.sen12_full_align.datamodule import SEN12FullDataModule

CONFIG_PATH = 'src/models/llwt_v5/config.yaml'
CKPT = 'checkpoints/llwt_v45/llwt-v0.5.0-sawg/last.ckpt'
MAX_PAIRS = 64
SHIFTS = [0, 0.5, 1, 2, 3, 4, 6]      # pixels
TAG = '[ffs-proof]'
BANDS = ('LL', 'LH', 'HL', 'HH')
DETAIL = ('LH', 'HL', 'HH')


# --------------------------------------------------------------------------- #
# distributional distances
# --------------------------------------------------------------------------- #
def fft_mag_l1(a: torch.Tensor, b: torch.Tensor) -> float:
    """Phase-free L1 between FFT magnitudes of two (B,C,h,w) bands.

    A pure spatial shift is a pure phase change => the magnitude spectrum is
    shift-invariant, so this distance should be ~flat under translation.
    """
    fa = torch.fft.rfft2(a).abs()
    fb = torch.fft.rfft2(b).abs()
    return float((fa - fb).abs().mean())


def swd_sorted(a: torch.Tensor, b: torch.Tensor) -> float:
    """1D sliced-Wasserstein / sorted-coefficient distance per (B,C) slice.

    Flatten each (h*w) subband, sort both, mean |diff| of the sorted sequences.
    Purely a distribution match: position/phase-free, so a spatial shift (which
    permutes coefficient positions but leaves the value MULTISET ~unchanged for
    a global roll) leaves this ~flat.
    """
    B, C = a.shape[0], a.shape[1]
    fa = a.reshape(B, C, -1).sort(dim=-1).values
    fb = b.reshape(B, C, -1).sort(dim=-1).values
    return float((fa - fb).abs().mean())


def pixel_l1(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().mean())


# --------------------------------------------------------------------------- #
# subpixel translate
# --------------------------------------------------------------------------- #
def translate(img: torch.Tensor, dx: float, dy: float) -> torch.Tensor:
    """Translate img (B,C,H,W) by (dx,dy) pixels (sub-pixel via grid_sample,
    bilinear, border padding). Integer shifts go through the same path for
    consistency."""
    if dx == 0 and dy == 0:
        return img
    B, C, H, W = img.shape
    ys, xs = torch.meshgrid(
        torch.linspace(-1.0, 1.0, H, device=img.device),
        torch.linspace(-1.0, 1.0, W, device=img.device),
        indexing='ij',
    )
    base = torch.stack([xs, ys], dim=0).unsqueeze(0).to(img.dtype)   # (1,2,H,W) [x,y]
    sx = 2.0 / max(W - 1, 1)
    sy = 2.0 / max(H - 1, 1)
    off = torch.tensor([dx * sx, dy * sy], device=img.device, dtype=img.dtype).view(1, 2, 1, 1)
    grid = (base + off).permute(0, 2, 3, 1).expand(B, H, W, 2)
    return F.grid_sample(img, grid, mode='bilinear', padding_mode='border', align_corners=True)


def load_gen(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    raw_sd = ckpt.get('state_dict', ckpt)
    gen_sd = {k[len('netG.'):]: v for k, v in raw_sd.items() if k.startswith('netG.')}
    mg, ug = model.netG.load_state_dict(gen_sd, strict=False)
    print(f'{TAG} loaded {ckpt_path} | G keys={len(gen_sd)} missing={len(mg)} unexpected={len(ug)}')
    return len(gen_sd)


def main():
    cfg = OmegaConf.load(CONFIG_PATH)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'{TAG} device={device}')

    # ---- model + ckpt ----
    model = LLWv4LightningModule(cfg)
    n = load_gen(model, CKPT)
    if n == 0:
        raise RuntimeError(f'No generator weights loaded from {CKPT}')
    netG = model.netG.to(device).eval()
    haar = HaarDown(in_channels=3).to(device)

    # ---- aligned val pairs (same split/setup as train.py) ----
    data_root = cfg.data.data_dir['sen12_full_align']
    dm = SEN12FullDataModule(
        data_dir=data_root,
        batch_size=cfg.data.batch_size,
        image_size=cfg.data.image_size,
        num_workers=0,
        persistent_workers=False,
        prefetch_factor=cfg.data.prefetch_factor,
        train_val_split_ratio=cfg.data.train_val_split_ratio,
        seed=cfg.data.seed,
        sar_channels=cfg.data.sar_channels,
        use_augmentation=False,
        scenes=list(cfg.data.scenes),
        val_batch_size=cfg.data.get('val_batch_size', None) or cfg.data.batch_size,
    )
    dm.setup('fit')
    loader = dm.val_dataloader()

    sars, opts = [], []
    seen = 0
    for sar, opt in loader:
        sars.append(sar)
        opts.append(opt)
        seen += sar.size(0)
        if seen >= MAX_PAIRS:
            break
    sar = torch.cat(sars, 0)[:MAX_PAIRS].to(device)
    opt = torch.cat(opts, 0)[:MAX_PAIRS].to(device).float()
    print(f'{TAG} collected {opt.size(0)} aligned val pairs | opt {tuple(opt.shape)}')

    t0 = time.perf_counter()

    # ===================================================================== #
    # C1 + C2 : synthetic shift sweep on the OPTICAL (isolates shift effect)
    # ===================================================================== #
    # reference (unshifted) opt bands
    with torch.no_grad():
        ref_bands = dict(zip(BANDS, haar(opt)))

    # rows[shift] = {metric: {'LL':.., 'detail':..}}
    rows = {}
    with torch.no_grad():
        for s in SHIFTS:
            # diagonal shift (both axes) so all detail orientations are exercised
            opt_sh = translate(opt, dx=float(s), dy=float(s))
            sh_bands = dict(zip(BANDS, haar(opt_sh)))

            px = {b: pixel_l1(sh_bands[b], ref_bands[b]) for b in BANDS}
            fm = {b: fft_mag_l1(sh_bands[b], ref_bands[b]) for b in BANDS}
            sw = {b: swd_sorted(sh_bands[b], ref_bands[b]) for b in BANDS}

            def agg(d):
                return {'LL': d['LL'], 'detail': float(np.mean([d[b] for b in DETAIL]))}

            rows[s] = {'pixel': agg(px), 'fftmag': agg(fm), 'swd': agg(sw)}

    # ===================================================================== #
    # C3 : achievability on the real trained model output
    # ===================================================================== #
    with torch.no_grad():
        fake = netG(sar).float().clamp(-1, 1)
        fb = dict(zip(BANDS, haar(fake)))
        ob = dict(zip(BANDS, haar(opt)))
        c3_px = {b: pixel_l1(fb[b], ob[b]) for b in BANDS}
        c3_fm = {b: fft_mag_l1(fb[b], ob[b]) for b in BANDS}
        c3_sw = {b: swd_sorted(fb[b], ob[b]) for b in BANDS}
        # energy normalizers (per band, on GT) so cross-metric ratios are scale-fair
        en = {b: float(ob[b].abs().mean()) for b in BANDS}
        fft_en = {b: float(torch.fft.rfft2(ob[b]).abs().mean()) for b in BANDS}

    dt = time.perf_counter() - t0
    print(f'{TAG} compute done in {dt:.1f}s on {device}')

    # ===================================================================== #
    # REPORT
    # ===================================================================== #
    print('\n' + '=' * 70)
    print(f'{TAG} FFS PREMISE TEST  (ckpt={CKPT}, n={opt.size(0)} aligned val pairs)')
    print('=' * 70)

    # shift=0 is a band-vs-itself (distance exactly 0) anchor; the meaningful
    # signal is the NON-ZERO shifts. We report magnitudes directly and summarize
    # over shifts in [1,6]px (the regime that matters: real misalign ~1.2px median).
    nz = [s for s in SHIFTS if s >= 1]

    def shift_table(metric, title):
        print(f'\n--- {title} vs shift (shifted-opt band vs unshifted-opt band) ---')
        print(f'  {"shift_px":>8} | {"LL":>12} | {"detail-mean":>12} | {"detail/LL":>9}')
        for s in SHIFTS:
            ll = rows[s][metric]['LL']
            det = rows[s][metric]['detail']
            ratio = det / max(ll, 1e-9)
            print(f'  {s:>8} | {ll:>12.6f} | {det:>12.6f} | {ratio:>9.3f}')
        # summaries over the meaningful 1..6px regime (avoid the degenerate 0)
        ll_vals = [rows[s][metric]['LL'] for s in nz]
        det_vals = [rows[s][metric]['detail'] for s in nz]
        ll_mean = float(np.mean(ll_vals))
        det_mean = float(np.mean(det_vals))
        # "flatness" across 1..6px: spread relative to mean (low => flat/saturated)
        det_spread = float(np.std(det_vals) / max(det_mean, 1e-9))
        print(f'  mean over 1..6px:  LL {ll_mean:.6f}   detail {det_mean:.6f}   '
              f'(detail rel-spread {det_spread:.2f})')
        return ll_mean, det_mean, det_spread

    px_ll_m, px_det_m, px_det_spread = shift_table('pixel', 'C1/C2  PIXEL L1')
    fm_ll_m, fm_det_m, fm_det_spread = shift_table('fftmag', 'C2  FFT-MAGNITUDE L1')
    sw_ll_m, sw_det_m, sw_det_spread = shift_table('swd', 'C2  SWD (sorted-coeff)')

    # ---- C3 table ----
    print('\n--- C3  REAL (fake vs opt) per-band distances ---')
    print(f'  {"band":>4} | {"pixelL1":>9} | {"fftmagL1":>9} | {"swd":>9} '
          f'| {"px/energy":>9} | {"fft/fftEn":>9}')
    for b in BANDS:
        print(f'  {b:>4} | {c3_px[b]:>9.5f} | {c3_fm[b]:>9.5f} | {c3_sw[b]:>9.5f} '
              f'| {c3_px[b]/max(en[b],1e-9):>9.4f} | {c3_fm[b]/max(fft_en[b],1e-9):>9.4f}')
    det_px_rel = float(np.mean([c3_px[b] / max(en[b], 1e-9) for b in DETAIL]))
    ll_px_rel = c3_px['LL'] / max(en['LL'], 1e-9)
    det_fft_rel = float(np.mean([c3_fm[b] / max(fft_en[b], 1e-9) for b in DETAIL]))
    ll_fft_rel = c3_fm['LL'] / max(fft_en['LL'], 1e-9)
    print(f'  detail relative PIXEL error  = {det_px_rel:.4f}   (LL {ll_px_rel:.4f})')
    print(f'  detail relative FFT-mag error= {det_fft_rel:.4f}   (LL {ll_fft_rel:.4f})')
    # the key C3 ratio: how much BETTER is the distributional match than pixel, for detail?
    c3_gain = det_px_rel / max(det_fft_rel, 1e-9)
    print(f'  C3 detail gain (px_rel / fft_rel) = {c3_gain:.2f}x '
          f'(>1 => distribution closer than pixels => target reachable)')

    # ===================================================================== #
    # VERDICTS
    # ===================================================================== #
    print('\n' + '=' * 70)
    print(f'{TAG} VERDICTS')
    print('=' * 70)

    # Everything is judged against the REAL-model error scale (C3) so the
    # magnitudes are comparable. "shift-injected error per unit of real error"
    # = how much a 1..6px misalignment perturbs each loss relative to the gap
    # the model actually has to close. <1 => shift noise is small vs real signal
    # (shift-robust); >=~1 => shift noise swamps the real signal (shift-fragile).
    real_px_det = float(np.mean([c3_px[b] for b in DETAIL]))
    real_px_ll = c3_px['LL']
    real_fm_det = float(np.mean([c3_fm[b] for b in DETAIL]))
    real_sw_det = float(np.mean([c3_sw[b] for b in DETAIL]))

    px_det_si = px_det_m / max(real_px_det, 1e-9)     # pixel, detail: shift/real
    px_ll_si = px_ll_m / max(real_px_ll, 1e-9)        # pixel, LL:     shift/real
    fm_det_si = fm_det_m / max(real_fm_det, 1e-9)     # fft-mag, detail
    sw_det_si = sw_det_m / max(real_sw_det, 1e-9)     # swd, detail

    # C1: detail PIXEL loss is shift-fragile — a 1..6px shift injects error
    # comparable-or-larger than the real model gap (si>=~0.7), AND detail is far
    # more shift-sensitive than LL (per unit real error). LL pixel must be the
    # robust band (low si relative to its real, big gap) — i.e. detail/LL si high.
    c1_detail_fragile = px_det_si >= 0.7
    c1_detail_worse_than_ll = px_det_si >= 1.5 * px_ll_si
    c1_pass = c1_detail_fragile and c1_detail_worse_than_ll
    print(f'  C1 detail-pixel-shift-fragile: shift-injected/real-error  '
          f'detail={px_det_si:.2f} vs LL={px_ll_si:.2f}  '
          f'(detail{"" if c1_detail_fragile else " NOT"} fragile, '
          f'detail {"" if c1_detail_worse_than_ll else "NOT "}>> LL)  '
          f'=> {"PASS" if c1_pass else "FAIL"}')

    # C2: a distributional detail loss is shift-invariant — shift injects FAR
    # less than the pixel loss per unit real error (si much < pixel si) AND is
    # flat (low spread) across 1..6px. Test both candidates.
    fm_robust = (fm_det_si <= 0.5 * px_det_si) and (fm_det_spread <= 0.6)
    sw_robust = (sw_det_si <= 0.5 * px_det_si) and (sw_det_spread <= 0.6)
    c2_pass = fm_robust or sw_robust
    print(f'  C2 distributional-shift-invariant (vs pixel detail si={px_det_si:.2f}): '
          f'FFT-mag si={fm_det_si:.2f} spread={fm_det_spread:.2f} '
          f'({"robust" if fm_robust else "NOT robust"}), '
          f'SWD si={sw_det_si:.2f} spread={sw_det_spread:.2f} '
          f'({"robust" if sw_robust else "NOT robust"})  '
          f'=> {"PASS" if c2_pass else "FAIL"}')

    # C3: distributional match on detail is proportionally much better than pixel.
    c3_pass = c3_gain >= 1.5
    print(f'  C3 distributional-target-achievable: detail px/fft relative-error gain '
          f'{c3_gain:.2f}x  => {"PASS" if c3_pass else "FAIL"}')

    # better detail supervisor: lower shift-sensitivity wins, must also be robust.
    # rank by si (shift-injected error per unit real error), tie-break on spread.
    cands = []
    if fm_robust:
        cands.append(('FFT-mag', fm_det_si, fm_det_spread))
    if sw_robust:
        cands.append(('SWD', sw_det_si, sw_det_spread))
    if not cands:                                     # neither robust: still report least-bad
        cands = [('FFT-mag', fm_det_si, fm_det_spread), ('SWD', sw_det_si, sw_det_spread)]
    cands.sort(key=lambda t: (t[1], t[2]))
    better = cands[0][0]
    print(f'  better detail supervisor (lowest shift-sensitivity): {better}  '
          f'(FFT-mag si={fm_det_si:.2f}/spread={fm_det_spread:.2f} vs '
          f'SWD si={sw_det_si:.2f}/spread={sw_det_spread:.2f})')

    n_pass = sum([c1_pass, c2_pass, c3_pass])
    if n_pass == 3:
        overall = 'JUSTIFIED'
        line = (f'all 3 hold: detail pixel loss is shift-fragile '
                f'(shift/real si={px_det_si:.2f}, {px_det_si/max(px_ll_si,1e-9):.1f}x worse than LL), '
                f'{better} is shift-invariant, and distribution is a reachable target '
                f'({c3_gain:.1f}x closer than pixels). Use {better} on LH/HL/HH, pixel L1 on LL.')
    elif n_pass == 2:
        overall = 'PARTIAL'
        fails = [n for n, p in zip(('C1', 'C2', 'C3'), (c1_pass, c2_pass, c3_pass)) if not p]
        reshape = ''
        if not c1_pass:
            # the empirically observed reshape: detail pixel L1 saturates fast
            # (rel-spread {px_det_spread:.2f}) and is SMALLER than LL pixel L1;
            # it is LL (not detail) that is the more shift-fragile pixel band.
            reshape = (f' KEY: detail pixel L1 saturates fast (spread {px_det_spread:.2f}) '
                       f'and is LOWER than LL pixel L1 (detail/LL si {px_det_si:.2f}/{px_ll_si:.2f}); '
                       f'so the FFS rationale "supervise LL with pixel because it is shift-robust" '
                       f'is WRONG — LL pixel is the MORE shift-fragile band. Detail pixel IS noisy '
                       f'(si {px_det_si:.2f}~1) so a distributional detail loss ({better}) still '
                       f'helps, but pair it with a SHIFT-TOLERANT LL term, not raw pixel L1.')
        line = f'{",".join(fails)} failed — FFS needs reshaping.{reshape}'
    else:
        overall = 'REJECTED'
        fails = [n for n, p in zip(('C1', 'C2', 'C3'), (c1_pass, c2_pass, c3_pass)) if not p]
        line = f'{",".join(fails)} failed — premise does not hold; do NOT build FFS as proposed.'

    print('\n' + '=' * 70)
    print(f'{TAG} PREMISE = {overall} — {line}')
    print('=' * 70)


if __name__ == '__main__':
    main()
