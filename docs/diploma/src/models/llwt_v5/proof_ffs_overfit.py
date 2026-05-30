"""DECISIVE overfit verification for Frequency-Factorized Supervision (FFS).

THROWAWAY diagnostic (uncommitted). The fair, in-model test of the FFS premise
on MISALIGNED SAR->optical pairs:

    Does L1-detail supervision overfit to BLUR while SWD-detail overfits to SHARP?

We proved at the loss level that per-band L1 REWARDS blur under the dataset's
~1.2px median misalignment (a blurry pred smears low-magnitude detail energy that
overlaps the smeared GT -> small per-position |Δ|), while WaveletDetailSWDLoss
(sliced-Wasserstein on the SORTED detail-coefficient distribution) is
position/phase-free and PENALIZES the collapsed detail distribution of a blur.

This script confirms it IN-MODEL by OVERFITTING the generator to a FIXED tiny set
of RAW (misaligned) pairs to near-convergence under each detail loss, fidelity
only (NO discriminator, NO warm-optimizer confound, NO GAN noise), and measuring
output sharpness on the very pairs it overfit.

Two arms, EACH from a FRESH re-load of the same warm G (identical init):
  ARM L1 : detail supervised by per-band L1 (ALL bands, config ratios).
           + MS-SSIM (coarse structure).
  ARM SWD: LL supervised by L1 (per_band restricted to ll=1, lh=hl=hh=0);
           detail supervised by WaveletDetailSWDLoss, weight CALIBRATED so its
           initial contribution ~ the per-band DETAIL contribution in ARM L1.
           + the SAME MS-SSIM.
Everything else identical (init / lr / steps / data / LL treatment / MS-SSIM).

Metrics (every 200 steps + final), measured ON THE FIXED TRAIN PAIRS (overfit):
  * mid-frequency radial power ratio fake/opt (1/6..1/2 Nyquist) — the BLUR metric
    (higher=sharper, GT-matched ~ 1.0). PRIMARY sharpness judge.
  * highest-band power ratio fake/opt (1/2..Nyq) — want ~1.0, NOT >>1 (incoherent
    noise). The COHERENCE guard.
  * detail Haar-band relative L1 (detail L1 / detail energy).
  * LPIPS / PSNR / SSIM vs the (misaligned) GT. NOTE: PSNR may FAVOR the blurry L1
    arm BECAUSE the GT is misaligned -> do NOT judge sharpness by PSNR; use
    mid-freq power ratio + LPIPS.

Run from repo root::

    python -m src.models.llwt_v5.proof_ffs_overfit
"""
from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
# NOTE: deliberately do NOT set HF_HUB_OFFLINE — backbone loads from local HF cache.

import copy
import functools
import time

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

_orig_load = torch.load


@functools.wraps(_orig_load)
def _patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _orig_load(*args, **kwargs)


torch.load = _patched_load
torch.set_float32_matmul_precision('high')

from src.models.llwt_v5.main import LLWv4LightningModule
from src.models.llwt_v5.blocks import HaarDown
from src.models.llwt_v5.losses import PerBandWaveletL1Loss, WaveletDetailSWDLoss
from src.data.sen12_full.datamodule import SEN12FullDataModule

CONFIG_PATH = 'src/models/llwt_v5/config.yaml'
CKPT = 'checkpoints/llwt_v45/llwt-v0.5.0-sawg/last.ckpt'
TAG = '[ffs-overfit]'

STEPS = 1200              # generator updates per arm (approach overfit; >=800)
LOG_EVERY = 200
BS = 4                   # 2 batches of bs 4 = 8 fixed RAW misaligned pairs
N_TRAIN_BATCHES = 2
LR = 2.0e-4
SEED = 42


# --------------------------------------------------------------------------- #
# measurement helpers
# --------------------------------------------------------------------------- #
def _radial_power_profile(x: torch.Tensor, n_bins: int = 64):
    """Mean radial power spectrum of x (B,C,H,W). Returns (centers[0..1], power).

    Radius normalized so r=0.5 == Nyquist (image edge), r=1.0 == diagonal corner.
    """
    x = x.float()
    B, C, H, W = x.shape
    fft = torch.fft.fftshift(torch.fft.fft2(x), dim=(-2, -1))
    power = (fft.abs() ** 2).mean(dim=(0, 1))            # (H,W) averaged over B,C
    cy, cx = H // 2, W // 2
    ys = torch.arange(H, device=x.device).view(-1, 1) - cy
    xs = torch.arange(W, device=x.device).view(1, -1) - cx
    r = torch.sqrt((ys.float() ** 2 + xs.float() ** 2))
    r_max = float(r.max())
    r_norm = r / r_max
    bins = torch.linspace(0.0, 1.0, n_bins + 1, device=x.device)
    centers = 0.5 * (bins[1:] + bins[:-1])
    prof = torch.zeros(n_bins, device=x.device)
    idx = torch.bucketize(r_norm.flatten(), bins) - 1
    idx = idx.clamp(0, n_bins - 1)
    p_flat = power.flatten()
    prof.scatter_add_(0, idx, p_flat)
    counts = torch.zeros(n_bins, device=x.device).scatter_add_(
        0, idx, torch.ones_like(p_flat))
    prof = prof / counts.clamp_min(1.0)
    return centers.cpu().numpy(), prof.cpu().numpy()


def band_power_ratios(fake: torch.Tensor, opt: torch.Tensor):
    """fake/opt mean power ratio in MID (1/6..1/2 Nyq) and HIGH (1/2..Nyq) bands.

    Radius normalized so r=0.5 == Nyquist. MID band = r in [0.5/6, 0.5/2] =
    [0.0833, 0.25]. HIGH band = r in [0.4, 0.5] (toward Nyquist).
    """
    c, pf = _radial_power_profile(fake)
    _, po = _radial_power_profile(opt)
    nyq = 0.5
    mid = (c >= nyq / 6.0) & (c <= nyq / 2.0)
    high = (c >= 0.8 * nyq) & (c <= nyq)
    mid_ratio = float(pf[mid].mean() / max(po[mid].mean(), 1e-12))
    high_ratio = float(pf[high].mean() / max(po[high].mean(), 1e-12))
    return mid_ratio, high_ratio


def detail_rel_l1(fake: torch.Tensor, opt: torch.Tensor, haar: HaarDown):
    """Detail-band (LH/HL/HH) L1(fake,opt) / detail energy(opt). Lower = closer.

    ``HaarDown.forward`` returns a 4-tuple ``(LL, LH, HL, HH)``, each (B,C,h,w).
    """
    fb = haar(fake.float())
    ob = haar(opt.float())
    det_fake = torch.stack(fb[1:], dim=2)
    det_real = torch.stack(ob[1:], dim=2)
    det_l1 = (det_fake - det_real).abs().mean()
    det_en = det_real.abs().mean().clamp_min(1e-9)
    return float(det_l1 / det_en)


@torch.no_grad()
def measure(netG, sar_f, opt_f, haar, lpips_net, psnr_m, ssim_m):
    """Measure on the FIXED TRAIN pairs (it's an overfit — measure the fit)."""
    was_training = netG.training
    netG.eval()
    fake = netG(sar_f).float().clamp(-1, 1)
    mid_r, high_r = band_power_ratios(fake, opt_f)
    drel = detail_rel_l1(fake, opt_f, haar)
    lp = float(lpips_net(fake, opt_f).mean())
    ps = float(psnr_m(fake, opt_f))
    ss = float(ssim_m(fake, opt_f))
    if was_training:
        netG.train()
    return dict(midfreq=mid_r, highfreq=high_r, detail_rel=drel, lpips=lp, psnr=ps, ssim=ss)


# --------------------------------------------------------------------------- #
# checkpoint load (G only) — strict=False warm start
# --------------------------------------------------------------------------- #
def load_G(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    raw = ckpt.get('state_dict', ckpt)
    g_sd = {k[len('netG.'):]: v for k, v in raw.items() if k.startswith('netG.')}
    mg, ug = model.netG.load_state_dict(g_sd, strict=False)
    return len(g_sd), len(mg), len(ug)


def fmt(m):
    return (f"mid={m['midfreq']:.4f} high={m['highfreq']:.4f} "
            f"detRel={m['detail_rel']:.4f} lpips={m['lpips']:.4f} "
            f"psnr={m['psnr']:.3f} ssim={m['ssim']:.4f}")


# --------------------------------------------------------------------------- #
# one overfit arm (fidelity-only, no D)
# --------------------------------------------------------------------------- #
def run_arm(name, cfg, device, train_batches, sar_f, opt_f,
            haar, lpips_net, psnr_m, ssim_m,
            per_band_weights, msssim_weight, swd_weight):
    print(f'\n{TAG} ===== ARM {name} ===== fresh warm-G reload')
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # fresh module + fresh warm G (identical init across arms)
    model = LLWv4LightningModule(cfg)
    ng, mg, ug = load_G(model, CKPT)
    print(f'{TAG} {name}: G keys={ng} (missing {mg}, unexpected {ug})')
    netG = model.netG.to(device).train()

    # losses (built fresh on device — per_band is a buffer-only module, no params)
    per_band = PerBandWaveletL1Loss(band_weights=per_band_weights).to(device)
    swd = WaveletDetailSWDLoss().to(device) if swd_weight > 0 else None
    from src.models.llwt_v5.losses import MSSSIMLoss
    msssim = MSSSIMLoss().to(device)

    opt_g = torch.optim.Adam(netG.parameters(), lr=LR,
                             betas=(float(cfg.optimizer.beta1), float(cfg.optimizer.beta2)))

    traj = []   # (step, metrics)
    init = measure(netG, sar_f, opt_f, haar, lpips_net, psnr_m, ssim_m)
    traj.append((0, init))
    print(f'{TAG} {name} step    0/{STEPS}: {fmt(init)}')

    nb = len(train_batches)
    pb_detail_seen = pb_ll_seen = swd_seen = ms_seen = 0.0
    t0 = time.perf_counter()
    for step in range(1, STEPS + 1):
        sar, opt = train_batches[(step - 1) % nb]
        sar = sar.to(device)
        opt = opt.to(device).float()

        fake, predicted_sub, _ = netG(sar, return_internals=True)

        # LL (+detail in L1 arm) per-band L1
        l_pb = per_band(predicted_sub, opt)
        g_loss = l_pb * float(cfg.loss.per_band_weight)
        # telemetry: split LL vs detail contribution (sum-of-weights normalized)
        pb = per_band.last_per_band
        wsum = sum(per_band_weights.values())
        ll_share = per_band_weights['ll'] * float(pb['ll']) / max(wsum, 1e-9)
        det_share = (per_band_weights['lh'] * float(pb['lh']) +
                     per_band_weights['hl'] * float(pb['hl']) +
                     per_band_weights['hh'] * float(pb['hh'])) / max(wsum, 1e-9)
        pb_ll_seen += float(cfg.loss.per_band_weight) * ll_share
        pb_detail_seen += float(cfg.loss.per_band_weight) * det_share

        if swd is not None:
            l_swd = swd(predicted_sub, opt)
            g_loss = g_loss + l_swd * swd_weight
            swd_seen += float(l_swd.detach()) * swd_weight

        l_ms = msssim(fake, opt)
        g_loss = g_loss + l_ms * msssim_weight
        ms_seen += float(l_ms.detach()) * msssim_weight

        opt_g.zero_grad(set_to_none=True)
        g_loss.backward()
        torch.nn.utils.clip_grad_norm_(netG.parameters(), 1.0)
        opt_g.step()

        if step % LOG_EVERY == 0 or step == STEPS:
            m = measure(netG, sar_f, opt_f, haar, lpips_net, psnr_m, ssim_m)
            traj.append((step, m))
            print(f'{TAG} {name} step {step:4d}/{STEPS}: {fmt(m)}  '
                  f'g_loss={float(g_loss):.4f}')

    dt = time.perf_counter() - t0
    print(f'{TAG} {name}: contributions/step  per_band_LL~{pb_ll_seen/STEPS:.4f} '
          f'per_band_detail~{pb_detail_seen/STEPS:.4f} '
          f'swd~{swd_seen/STEPS:.4f} msssim~{ms_seen/STEPS:.4f}  ({dt:.1f}s)')

    del model, netG, opt_g, per_band, swd, msssim
    torch.cuda.empty_cache()
    return traj


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'{TAG} device={device} | STEPS={STEPS} | RAW MISALIGNED pairs (sen12_full)')
    if device != 'cuda':
        print(f'{TAG} WARNING: CUDA not available — running on CPU will be slow.')

    base_cfg = OmegaConf.load(CONFIG_PATH)
    # disable online aligner / patchnce / lab / compile / channels_last — fidelity-only
    if 'align' in base_cfg and base_cfg.align is not None:
        base_cfg.align.enabled = False
    base_cfg.loss.patchnce_weight = 0.0
    base_cfg.loss.lab_chroma_weight = 0.0
    base_cfg.system.compile = False
    base_cfg.system.channels_last = False

    msssim_weight = float(base_cfg.loss.msssim_weight)   # identical in both arms

    # ---- FIXED RAW MISALIGNED pairs from sen12_full (NOT the aligned mirror) ----
    data_root = base_cfg.data.data_dir['sen12_full']
    dm = SEN12FullDataModule(
        data_dir=data_root,
        batch_size=BS,
        image_size=base_cfg.data.image_size,
        num_workers=0,
        persistent_workers=False,
        prefetch_factor=base_cfg.data.prefetch_factor,
        train_val_split_ratio=base_cfg.data.train_val_split_ratio,
        seed=base_cfg.data.seed,
        sar_channels=base_cfg.data.sar_channels,
        use_augmentation=False,                          # fixed pairs, no aug noise
        scenes=list(base_cfg.data.scenes),
        val_batch_size=BS,
    )
    dm.setup('fit')
    train_loader = dm.train_dataloader()

    train_batches = []
    for b in train_loader:
        train_batches.append((b[0].clone(), b[1].clone()))
        if len(train_batches) >= N_TRAIN_BATCHES:
            break
    # the FIXED full set (8 pairs) on device, for measurement (overfit -> measure the fit)
    sar_f = torch.cat([b[0] for b in train_batches], 0).to(device)
    opt_f = torch.cat([b[1] for b in train_batches], 0).to(device).float()
    print(f'{TAG} fixed RAW train pool={len(train_batches)} batches '
          f'(bs={train_batches[0][0].size(0)}); total {sar_f.size(0)} misaligned pairs')

    haar = HaarDown(in_channels=3).to(device)
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
    lpips_net = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=False).to(device).eval()
    for p in lpips_net.parameters():
        p.requires_grad_(False)
    psnr_m = PeakSignalNoiseRatio(data_range=2.0).to(device)
    ssim_m = StructuralSimilarityIndexMeasure(data_range=2.0).to(device)

    # ---- arm band-weight specs ----
    pb_all = dict(ll=float(base_cfg.loss.per_band_ll), lh=float(base_cfg.loss.per_band_lh),
                  hl=float(base_cfg.loss.per_band_hl), hh=float(base_cfg.loss.per_band_hh))
    pb_ll_only = dict(ll=1.0, lh=0.0, hl=0.0, hh=0.0)

    # ---- calibrate swd_weight so its initial contribution ~ per_band DETAIL
    # contribution in ARM L1 (one warm forward on the fixed pairs) ----
    _calib = LLWv4LightningModule(base_cfg)
    load_G(_calib, CKPT)
    _g = _calib.netG.to(device).eval()
    with torch.no_grad():
        _, _sub, _ = _g(sar_f, return_internals=True)
        _pb = PerBandWaveletL1Loss(band_weights=pb_all).to(device)
        _ = _pb(_sub, opt_f)
        _per = _pb.last_per_band
        _wsum = sum(pb_all.values())
        _detail_share = (pb_all['lh'] * float(_per['lh']) + pb_all['hl'] * float(_per['hl']) +
                         pb_all['hh'] * float(_per['hh'])) / max(_wsum, 1e-9)
        _base_detail_contrib = float(base_cfg.loss.per_band_weight) * _detail_share
        _swd_raw = float(WaveletDetailSWDLoss().to(device)(_sub, opt_f))
    _swd_w = _base_detail_contrib / max(_swd_raw, 1e-9)
    swd_weight = float(np.clip(_swd_w, 0.5, 50.0))
    print(f'{TAG} calibration: ARM-L1 per_band DETAIL contribution ~ {_base_detail_contrib:.4f} '
          f'(pbw={float(base_cfg.loss.per_band_weight)}, detail-share L1={_detail_share:.4f}); '
          f'raw SWD={_swd_raw:.4f} -> swd_weight={swd_weight:.3f} '
          f'(raw target {_swd_w:.3f}, clamped [0.5,50])')
    del _calib, _g
    torch.cuda.empty_cache()

    # ---- arm configs (per_band_weight master kept identical; only band ratios differ) ----
    cfg_l1 = copy.deepcopy(base_cfg)
    cfg_swd = copy.deepcopy(base_cfg)

    traj_l1 = run_arm('L1', cfg_l1, device, train_batches, sar_f, opt_f,
                      haar, lpips_net, psnr_m, ssim_m,
                      per_band_weights=pb_all, msssim_weight=msssim_weight, swd_weight=0.0)
    traj_swd = run_arm('SWD', cfg_swd, device, train_batches, sar_f, opt_f,
                       haar, lpips_net, psnr_m, ssim_m,
                       per_band_weights=pb_ll_only, msssim_weight=msssim_weight,
                       swd_weight=swd_weight)

    # ===================================================================== #
    # REPORT
    # ===================================================================== #
    print('\n' + '=' * 84)
    print(f'{TAG} MID-FREQ POWER RATIO TRAJECTORY (fake/opt; GT=1.0; higher=sharper)')
    print('=' * 84)
    print(f'  {"step":>5} | {"ARM-L1":>10} | {"ARM-SWD":>10}')
    steps_l1 = {s: m for s, m in traj_l1}
    steps_swd = {s: m for s, m in traj_swd}
    all_steps = sorted(set(steps_l1) | set(steps_swd))
    for s in all_steps:
        a = steps_l1.get(s, {}).get('midfreq', float('nan'))
        b = steps_swd.get(s, {}).get('midfreq', float('nan'))
        print(f'  {s:>5} | {a:>10.4f} | {b:>10.4f}')

    print('\n' + '=' * 84)
    print(f'{TAG} HIGH-BAND POWER RATIO TRAJECTORY (fake/opt; GT=1.0; want ~1, NOT >>1)')
    print('=' * 84)
    print(f'  {"step":>5} | {"ARM-L1":>10} | {"ARM-SWD":>10}')
    for s in all_steps:
        a = steps_l1.get(s, {}).get('highfreq', float('nan'))
        b = steps_swd.get(s, {}).get('highfreq', float('nan'))
        print(f'  {s:>5} | {a:>10.4f} | {b:>10.4f}')

    l1 = traj_l1[-1][1]
    sw = traj_swd[-1][1]
    print('\n' + '=' * 84)
    print(f'{TAG} FINAL COMPARISON (after {STEPS} overfit steps, on the FIXED misaligned pairs)')
    print('=' * 84)
    hdr = (f'  {"metric":>16} | {"ARM-L1":>10} | {"ARM-SWD":>10} | note')
    print(hdr)
    print('  ' + '-' * 80)
    print(f'  {"midfreq (GT=1)":>16} | {l1["midfreq"]:>10.4f} | {sw["midfreq"]:>10.4f} | '
          f'PRIMARY blur metric; closer to 1.0 = sharper')
    print(f'  {"highband (GT=1)":>16} | {l1["highfreq"]:>10.4f} | {sw["highfreq"]:>10.4f} | '
          f'coherence guard; want ~1, >>1 = noise')
    print(f'  {"detail relL1":>16} | {l1["detail_rel"]:>10.4f} | {sw["detail_rel"]:>10.4f} | '
          f'lower=closer (L1 itself rewards blur)')
    print(f'  {"LPIPS":>16} | {l1["lpips"]:>10.4f} | {sw["lpips"]:>10.4f} | '
          f'lower=better perceptual (sharpness-aware)')
    print(f'  {"PSNR":>16} | {l1["psnr"]:>10.3f} | {sw["psnr"]:>10.3f} | '
          f'NOT a sharpness judge (GT misaligned -> rewards blur)')
    print(f'  {"SSIM":>16} | {l1["ssim"]:>10.4f} | {sw["ssim"]:>10.4f} | structural')

    # ===================================================================== #
    # VERDICT
    # ===================================================================== #
    # SHARPNESS: SWD mid-freq closer to 1.0 (sharper) than L1?
    l1_mid_gap = abs(l1['midfreq'] - 1.0)
    sw_mid_gap = abs(sw['midfreq'] - 1.0)
    if sw['midfreq'] > l1['midfreq'] + 1e-3 and sw_mid_gap < l1_mid_gap - 1e-3:
        sharp = 'SHARPER'
    elif l1['midfreq'] > sw['midfreq'] + 1e-3 and l1_mid_gap < sw_mid_gap - 1e-3:
        sharp = 'BLURRIER'
    elif abs(sw['midfreq'] - l1['midfreq']) < 1e-3:
        sharp = 'EQUAL'
    else:
        # one is higher but the OTHER is closer to 1.0 (over- vs under-shoot)
        sharp = 'SHARPER' if sw['midfreq'] > l1['midfreq'] else 'BLURRIER'

    # COHERENCE: is SWD's high-band NOT wildly inflated vs L1?
    coherent = sw['highfreq'] <= max(l1['highfreq'] * 2.0, l1['highfreq'] + 1.0)
    lpips_better = sw['lpips'] < l1['lpips']

    print('\n' + '=' * 84)
    print(f'{TAG} VERDICT')
    print('=' * 84)
    print(f'  SHARPNESS : ARM-SWD mid-freq {sw["midfreq"]:.4f} vs ARM-L1 {l1["midfreq"]:.4f} '
          f'(GT=1.0) -> SWD is {sharp}')
    print(f'  COHERENCE : ARM-SWD high-band {sw["highfreq"]:.4f} vs ARM-L1 {l1["highfreq"]:.4f} '
          f'-> {"coherent (not blown up)" if coherent else "INFLATED (possible noise)"}')
    print(f'  LPIPS     : ARM-SWD {sw["lpips"]:.4f} vs ARM-L1 {l1["lpips"]:.4f} '
          f'-> SWD {"better" if lpips_better else "worse"}')

    if sharp == 'SHARPER' and coherent:
        verdict = 'FFS VERIFIED in-model: SWD overfits SHARPER than L1 on misaligned pairs (and stays coherent).'
    elif sharp == 'SHARPER' and not coherent:
        verdict = 'SWD sharper BUT high-band inflated — check it is real detail, not incoherent noise.'
    elif sharp == 'EQUAL':
        verdict = 'NO in-model separation: SWD ~ L1 on sharpness here.'
    else:
        verdict = 'FFS NOT supported in-model here: SWD is not sharper than L1.'
    print(f'\n{TAG} SWD {sharp} than L1 (midfreq SWD={sw["midfreq"]:.4f} L1={l1["midfreq"]:.4f}, GT=1.0); '
          f'highband SWD={sw["highfreq"]:.4f} L1={l1["highfreq"]:.4f}; '
          f'LPIPS SWD={sw["lpips"]:.4f} L1={l1["lpips"]:.4f}')
    print(f'{TAG} {verdict}')


if __name__ == '__main__':
    main()
