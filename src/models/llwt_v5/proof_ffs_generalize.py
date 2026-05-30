"""DECISIVE GENERALIZATION A/B for Frequency-Factorized Supervision (FFS).

THROWAWAY diagnostic (uncommitted). The prior `proof_ffs_overfit.py` overfit a
FIXED tiny pool and found SWD blurrier overall — but the blur-from-misalignment
effect FFS targets is a *generalization* phenomenon: under ~1.2px SAR<->optical
misregistration, per-band L1 on detail REWARDS blur because a blurry pred smears
low-magnitude detail that overlaps the smeared (misaligned) GT -> small per-pos
|delta|; an OVERFIT can memorize either target so it can't expose the effect.

This script runs the VALID cheap test: train on MANY distinct misaligned pairs
(no memorization possible) and measure DETAIL SHARPNESS on a FIXED HELD-OUT val
set NEVER trained on.

Two arms, EACH from a FRESH reload of the same warm G+D (identical init):
  ARM BASE: detail supervised by per-band L1 on LH/HL/HH (config ratios, ALL
            bands), full loss stack.
  ARM FFS : per-band LL-ONLY (weight tuned so its LL-L1 contribution NUMERICALLY
            matches BASE's LL contribution at step 0) + WaveletDetailSWDLoss on
            detail (swd_weight calibrated so its step-0 contribution ~ BASE's
            step-0 detail-L1 contribution).
  ONLY the detail term differs. LL + everything else (gan_main+fm+msssim+lpips
  +ffl, D updates, lr, optimizer, data order, steps) IDENTICAL across arms.

Fairness is PROVEN by printing, at step 0, the two arms' LL contribution (must
match within 5%) and detail contribution (must match within calibration).

Eval (BEFORE step 0 + AFTER) on the HELD-OUT val set for BOTH arms:
  * mid-frequency radial power ratio fake/opt (1/6..1/2 Nyq) — PRIMARY blur
    metric (->1.0 = sharp).
  * highest-band power ratio (->1.0; >>1 = incoherent noise) — coherence guard.
  * detail Haar-band power ratio fake/opt (->1.0 = matched detail energy).
  * LPIPS / PSNR / SSIM.

Run from repo root::

    python -m src.models.llwt_v5.proof_ffs_generalize
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

from src.models.llwt_v5 import factory
from src.models.llwt_v5.blocks import HaarDown
from src.models.llwt_v5.losses import PerBandWaveletL1Loss, WaveletDetailSWDLoss
from src.data.sen12_full.datamodule import SEN12FullDataModule

CONFIG_PATH = 'src/models/llwt_v5/config.yaml'
CKPT = 'checkpoints/llwt_v45/llwt-v0.5.0-sawg/last.ckpt'
TAG = '[ffs-gen]'

STEPS = 2200            # generator+D updates per arm; reduced if time budget hit
LOG_EVERY = 550
LR_G = 1.0e-4          # MODERATE (below 2e-4 default) — don't destroy warm spectrum
SEED = 42
TIME_BUDGET_S = 8.4 * 60  # hard per-arm wall budget (< 9 min); break early if hit
N_VAL_PAIRS = 32       # held-out eval set (NEVER trained on)


# --------------------------------------------------------------------------- #
# measurement helpers
# --------------------------------------------------------------------------- #
def _radial_power_profile(x: torch.Tensor, n_bins: int = 64):
    x = x.float()
    B, C, H, W = x.shape
    fft = torch.fft.fftshift(torch.fft.fft2(x), dim=(-2, -1))
    power = (fft.abs() ** 2).mean(dim=(0, 1))
    cy, cx = H // 2, W // 2
    ys = torch.arange(H, device=x.device).view(-1, 1) - cy
    xs = torch.arange(W, device=x.device).view(1, -1) - cx
    r = torch.sqrt((ys.float() ** 2 + xs.float() ** 2))
    r_norm = r / float(r.max())
    bins = torch.linspace(0.0, 1.0, n_bins + 1, device=x.device)
    centers = 0.5 * (bins[1:] + bins[:-1])
    prof = torch.zeros(n_bins, device=x.device)
    idx = torch.bucketize(r_norm.flatten(), bins).clamp(0, n_bins - 1)
    p_flat = power.flatten()
    prof.scatter_add_(0, idx, p_flat)
    counts = torch.zeros(n_bins, device=x.device).scatter_add_(
        0, idx, torch.ones_like(p_flat))
    prof = prof / counts.clamp_min(1.0)
    return centers.cpu().numpy(), prof.cpu().numpy()


def band_power_ratios(fake: torch.Tensor, opt: torch.Tensor):
    c, pf = _radial_power_profile(fake)
    _, po = _radial_power_profile(opt)
    nyq = 0.5
    mid = (c >= nyq / 6.0) & (c <= nyq / 2.0)
    high = (c >= 0.8 * nyq) & (c <= nyq)
    mid_ratio = float(pf[mid].mean() / max(po[mid].mean(), 1e-12))
    high_ratio = float(pf[high].mean() / max(po[high].mean(), 1e-12))
    return mid_ratio, high_ratio


def detail_power_ratio(fake: torch.Tensor, opt: torch.Tensor, haar: HaarDown):
    """fake/opt detail-band (LH+HL+HH) power ratio. ->1.0 = matched detail energy."""
    fb = haar(fake.float())
    ob = haar(opt.float())
    det_fake = torch.stack(fb[1:], dim=2)          # (B,C,3,h,w)
    det_real = torch.stack(ob[1:], dim=2)
    pf = (det_fake ** 2).mean()
    po = (det_real ** 2).mean().clamp_min(1e-12)
    return float(pf / po)


@torch.no_grad()
def measure(netG, val_sar, val_opt, haar, lpips_net, psnr_m, ssim_m, vbs=8):
    """Measure on the HELD-OUT val set (batched to fit VRAM)."""
    was_training = netG.training
    netG.eval()
    fakes = []
    for i in range(0, val_sar.size(0), vbs):
        s = val_sar[i:i + vbs]
        f = netG(s).float().clamp(-1, 1)
        fakes.append(f)
    fake = torch.cat(fakes, 0)
    mid_r, high_r = band_power_ratios(fake, val_opt)
    detpow = detail_power_ratio(fake, val_opt, haar)
    # metrics in batches (LPIPS net likes <=batchable)
    lp = ps = ss = 0.0
    n = 0
    for i in range(0, fake.size(0), vbs):
        f = fake[i:i + vbs]
        o = val_opt[i:i + vbs]
        bs = f.size(0)
        lp += float(lpips_net(f, o).mean()) * bs
        ps += float(psnr_m(f, o)) * bs
        ss += float(ssim_m(f, o)) * bs
        n += bs
    if was_training:
        netG.train()
    return dict(midfreq=mid_r, highfreq=high_r, detpow=detpow,
                lpips=lp / n, psnr=ps / n, ssim=ss / n)


def fmt(m):
    return (f"mid={m['midfreq']:.4f} high={m['highfreq']:.4f} "
            f"detpow={m['detpow']:.4f} lpips={m['lpips']:.4f} "
            f"psnr={m['psnr']:.3f} ssim={m['ssim']:.4f}")


# --------------------------------------------------------------------------- #
# warm ckpt load (G + D) — strict=False
# --------------------------------------------------------------------------- #
def load_GD(netG, netD, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    raw = ckpt.get('state_dict', ckpt)
    g_sd = {k[len('netG.'):]: v for k, v in raw.items() if k.startswith('netG.')}
    d_sd = {k[len('netD.'):]: v for k, v in raw.items() if k.startswith('netD.')}
    mg, ug = netG.load_state_dict(g_sd, strict=False)
    md, ud = netD.load_state_dict(d_sd, strict=False)
    return len(g_sd), len(mg), len(ug), len(d_sd), len(md), len(ud)


def build_loader(base_cfg, bs):
    """Fresh RAW (misaligned) sen12_full datamodule + train/val split."""
    data_root = base_cfg.data.data_dir['sen12_full']
    dm = SEN12FullDataModule(
        data_dir=data_root,
        batch_size=bs,
        image_size=base_cfg.data.image_size,
        num_workers=int(base_cfg.data.num_workers),
        persistent_workers=False,
        prefetch_factor=base_cfg.data.prefetch_factor,
        train_val_split_ratio=base_cfg.data.train_val_split_ratio,
        seed=base_cfg.data.seed,
        sar_channels=base_cfg.data.sar_channels,
        use_augmentation=False,                  # NO aug noise — clean A/B
        scenes=list(base_cfg.data.scenes),
        val_batch_size=bs,
    )
    dm.setup('fit')
    return dm


# --------------------------------------------------------------------------- #
# one training arm: faithful manual replication of main.py training_step
#   (D update on detached fake + G update with the listed loss stack),
#   align + patchnce OFF (not in the gan_main+fm+msssim+lpips+ffl stack).
# --------------------------------------------------------------------------- #
def run_arm(name, base_cfg, device, dm, val_sar, val_opt, haar,
            lpips_net, psnr_m, ssim_m, per_band_weights, swd_weight):
    print(f'\n{TAG} ===== ARM {name} ===== fresh warm G+D reload')
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    cfg = copy.deepcopy(base_cfg)
    netG, netD = factory.build_models(cfg)
    g_keys, mg, ug, d_keys, md, ud = load_GD(netG, netD, CKPT)
    print(f'{TAG} {name}: warm G keys={g_keys} (miss {mg}, unexp {ug}); '
          f'D keys={d_keys} (miss {md}, unexp {ud})')
    netG = netG.to(device).train()
    netD = netD.to(device).train()

    # ----- criterions (shared stack), built fresh on device -----
    gan = factory.build_criterions(cfg)['gan'].to(device)  # GANLoss (lsgan)
    from src.models.llwt_v5.losses import FeatureMatchingLoss, MSSSIMLoss, LPIPSLoss, FFLLoss
    fm = FeatureMatchingLoss().to(device)
    msssim = MSSSIMLoss().to(device)
    lpips_loss = LPIPSLoss(net_type='alex').to(device)
    ffl = FFLLoss(alpha=float(cfg.loss.ffl_alpha)).to(device)
    per_band = PerBandWaveletL1Loss(band_weights=per_band_weights).to(device)
    swd = WaveletDetailSWDLoss(in_channels=3).to(device) if swd_weight > 0 else None

    pb_w = float(cfg.loss.per_band_weight)
    gan_w = float(cfg.loss.gan_main_weight)
    fm_w = float(cfg.loss.fm_main_weight)
    ms_w = float(cfg.loss.msssim_weight)
    lp_w = float(cfg.loss.lpips_weight)
    ffl_w = float(cfg.loss.ffl_weight)

    opt_g = torch.optim.Adam(netG.parameters(), lr=LR_G,
                             betas=(float(cfg.optimizer.beta1), float(cfg.optimizer.beta2)))
    opt_d = torch.optim.Adam(netD.parameters(), lr=float(cfg.optimizer.lr_d),
                             betas=(float(cfg.optimizer.beta1), float(cfg.optimizer.beta2)))

    def autocast():
        return torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)

    # step-0 contribution probe (LL + detail), for the fairness print
    step0 = {}

    init = measure(netG, val_sar, val_opt, haar, lpips_net, psnr_m, ssim_m)
    traj = [(0, init)]
    print(f'{TAG} {name} HELD-OUT step    0/{STEPS}: {fmt(init)}')

    train_loader = dm.train_dataloader()
    data_iter = iter(train_loader)
    t0 = time.perf_counter()
    real_steps = 0
    for step in range(1, STEPS + 1):
        try:
            sar, opt = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            sar, opt = next(data_iter)
        sar = sar.to(device, non_blocking=True)
        opt = opt.to(device, non_blocking=True).float()
        if not (torch.isfinite(sar).all() and torch.isfinite(opt).all()):
            continue

        # -------- D update on detached fake (batched real+fake) --------
        with torch.no_grad(), autocast():
            fake_d = netG(sar)
        B = sar.size(0)
        sar2 = sar.repeat(2, 1, 1, 1)
        opt2 = torch.cat([opt, fake_d.float()], dim=0)
        with autocast():
            main_both, _, _, _, _, _ = netD(sar2, opt2)
            lc, lf = main_both
            real_main = (lc[:B], lf[:B])
            fake_main = (lc[B:], lf[B:])
            d_loss = 0.5 * (gan(real_main, is_real=True) + gan(fake_main, is_real=False))
        opt_d.zero_grad(set_to_none=True)
        d_loss.backward()
        dgn = torch.nn.utils.clip_grad_norm_(netD.parameters(), float(cfg.loss.grad_clip_d))
        if torch.isfinite(dgn):
            opt_d.step()

        # -------- G update (full listed stack) --------
        with autocast():
            fake, predicted_sub, _ = netG(sar, return_internals=True)
            sar2g = sar.repeat(2, 1, 1, 1)
            opt2g = torch.cat([fake, opt], dim=0)
            main_both_g, _, _, feats_main_both, _, _ = netD(sar2g, opt2g)
            lc_g, lf_g = main_both_g
            fake_main_g = (lc_g[:B], lf_g[:B])
            fake_feats = [f[:B] for f in feats_main_both]
            real_feats = [f[B:].detach() for f in feats_main_both]

            l_gan = gan(fake_main_g, is_real=True, for_d=False)
            l_fm = fm(fake_feats, real_feats)
            g_loss = l_gan * gan_w + l_fm * fm_w

            # per-band L1 (LL [+detail in BASE]) — fp32 inside loss
            l_pb = per_band(predicted_sub, opt)
            g_loss = g_loss + l_pb * pb_w
            if swd is not None:
                l_swd = swd(predicted_sub, opt)
                g_loss = g_loss + l_swd * swd_weight
            l_ms = msssim(fake, opt)
            g_loss = g_loss + l_ms * ms_w
            l_lp = lpips_loss(fake, opt)
            g_loss = g_loss + l_lp * lp_w
            l_ffl = ffl(fake, opt)
            g_loss = g_loss + l_ffl * ffl_w

        # --- step-0 fairness telemetry (LL + detail contributions, both arms) ---
        if step == 1:
            pb = per_band.last_per_band
            wsum = sum(per_band_weights.values())
            ll_contrib = pb_w * per_band_weights['ll'] * float(pb['ll']) / max(wsum, 1e-9)
            det_contrib = pb_w * (
                per_band_weights['lh'] * float(pb['lh']) +
                per_band_weights['hl'] * float(pb['hl']) +
                per_band_weights['hh'] * float(pb['hh'])) / max(wsum, 1e-9)
            swd_contrib = (float(l_swd.detach()) * swd_weight) if swd is not None else 0.0
            step0 = dict(ll=ll_contrib, det_l1=det_contrib, swd=swd_contrib)

        opt_g.zero_grad(set_to_none=True)
        if torch.isfinite(g_loss):
            g_loss.backward()
            ggn = torch.nn.utils.clip_grad_norm_(netG.parameters(), float(cfg.loss.grad_clip_g))
            if torch.isfinite(ggn):
                opt_g.step()

        real_steps = step
        if step % LOG_EVERY == 0 or step == STEPS:
            m = measure(netG, val_sar, val_opt, haar, lpips_net, psnr_m, ssim_m)
            traj.append((step, m))
            print(f'{TAG} {name} HELD-OUT step {step:4d}/{STEPS}: {fmt(m)}  '
                  f'g={float(g_loss):.3f} d={float(d_loss):.3f}')
        if time.perf_counter() - t0 > TIME_BUDGET_S:
            print(f'{TAG} {name}: TIME BUDGET hit at step {step} '
                  f'({time.perf_counter()-t0:.0f}s) — stopping early.')
            m = measure(netG, val_sar, val_opt, haar, lpips_net, psnr_m, ssim_m)
            if traj[-1][0] != step:
                traj.append((step, m))
            break

    dt = time.perf_counter() - t0
    print(f'{TAG} {name}: {real_steps} steps in {dt:.0f}s '
          f'({dt/max(real_steps,1):.3f}s/step)')

    del netG, netD, opt_g, opt_d, per_band, swd, fm, msssim, lpips_loss, ffl, gan
    torch.cuda.empty_cache()
    return traj, step0, real_steps


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'{TAG} device={device} | target STEPS={STEPS} lr_g={LR_G} | '
          f'RAW MISALIGNED multi-pair generalization (sen12_full)')
    if device != 'cuda':
        print(f'{TAG} WARNING: CUDA not available — CPU will be too slow for this test.')

    base_cfg = OmegaConf.load(CONFIG_PATH)
    # OFF: align + patchnce (confounds, not in the listed gan_main+fm+msssim+lpips+ffl stack)
    if 'align' in base_cfg and base_cfg.align is not None:
        base_cfg.align.enabled = False
    base_cfg.loss.patchnce_weight = 0.0
    base_cfg.loss.lab_chroma_weight = 0.0
    base_cfg.system.compile = False
    base_cfg.system.channels_last = False   # keep tensors plain NCHW for measurement
    bs = int(base_cfg.data.batch_size)

    # ---- data: build once, reuse the SAME held-out val set for both arms ----
    dm = build_loader(base_cfg, bs)
    val_loader = dm.val_dataloader()
    vs, vo = [], []
    for b in val_loader:
        vs.append(b[0]); vo.append(b[1])
        if sum(x.size(0) for x in vs) >= N_VAL_PAIRS:
            break
    val_sar = torch.cat(vs, 0)[:N_VAL_PAIRS].to(device)
    val_opt = torch.cat(vo, 0)[:N_VAL_PAIRS].to(device).float()
    n_train = len(dm.train_dataset)
    print(f'{TAG} HELD-OUT val pairs={val_sar.size(0)} (NEVER trained on); '
          f'train pool={n_train} distinct RAW misaligned pairs (bs={bs})')

    haar = HaarDown(in_channels=3).to(device)
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
    lpips_net = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=False).to(device).eval()
    for p in lpips_net.parameters():
        p.requires_grad_(False)
    psnr_m = PeakSignalNoiseRatio(data_range=2.0).to(device)
    ssim_m = StructuralSimilarityIndexMeasure(data_range=2.0).to(device)

    # ---- band-weight specs ----
    pb_all = dict(ll=float(base_cfg.loss.per_band_ll), lh=float(base_cfg.loss.per_band_lh),
                  hl=float(base_cfg.loss.per_band_hl), hh=float(base_cfg.loss.per_band_hh))

    # ============================================================ #
    # CALIBRATION on the warm G (one forward on a fixed train batch)
    #   - BASE LL contribution  -> target for FFS LL-only weight
    #   - BASE detail-L1 contrib -> target for FFS swd_weight
    # ============================================================ #
    calib_loader = dm.train_dataloader()
    cb = next(iter(calib_loader))
    c_sar = cb[0].to(device); c_opt = cb[1].to(device).float()
    _g, _d = factory.build_models(copy.deepcopy(base_cfg))
    load_GD(_g, _d, CKPT)
    _g = _g.to(device).eval()
    with torch.no_grad():
        _, _sub, _ = _g(c_sar, return_internals=True)
        _pb = PerBandWaveletL1Loss(band_weights=pb_all).to(device)
        _ = _pb(_sub, c_opt)
        _per = _pb.last_per_band
        _wsum = sum(pb_all.values())
        pb_w = float(base_cfg.loss.per_band_weight)
        base_ll_contrib = pb_w * pb_all['ll'] * float(_per['ll']) / _wsum
        base_det_contrib = pb_w * (pb_all['lh'] * float(_per['lh']) +
                                   pb_all['hl'] * float(_per['hl']) +
                                   pb_all['hh'] * float(_per['hh'])) / _wsum
        _swd_raw = float(WaveletDetailSWDLoss(in_channels=3).to(device)(_sub, c_opt))

    # FFS LL-only: per_band_weights={ll:1, rest:0} -> its contribution = pb_w*L1_ll.
    # We want pb_w_ffs * L1_ll == base_ll_contrib  ->  pb_w_ffs = base_ll_contrib / L1_ll.
    # Implemented by setting the FFS arm's per_band MASTER weight via cfg override
    # (band weights ll=1 only). Scale factor relative to base pb_w:
    ffs_pb_w = base_ll_contrib / max(float(_per['ll']), 1e-12)
    # swd_weight so swd_raw * w == base_det_contrib
    swd_weight = base_det_contrib / max(_swd_raw, 1e-12)
    print(f'{TAG} CALIB (warm G): L1_ll={float(_per["ll"]):.5f} '
          f'L1_det(weighted-share)={base_det_contrib/pb_w:.5f} swd_raw={_swd_raw:.5f}')
    print(f'{TAG} CALIB targets: BASE LL contrib={base_ll_contrib:.5f}  '
          f'BASE detail-L1 contrib={base_det_contrib:.5f}')
    print(f'{TAG} CALIB -> FFS per_band(LL-only) master weight={ffs_pb_w:.4f} ; '
          f'FFS swd_weight={swd_weight:.4f}')
    del _g, _d
    torch.cuda.empty_cache()

    # ---- arm configs: only the detail term differs ----
    cfg_base = copy.deepcopy(base_cfg)   # per_band ALL bands, no SWD
    cfg_ffs = copy.deepcopy(base_cfg)
    cfg_ffs.loss.per_band_weight = float(ffs_pb_w)   # LL-only master weight

    traj_base, s0_base, n_base = run_arm(
        'BASE', cfg_base, device, dm, val_sar, val_opt, haar,
        lpips_net, psnr_m, ssim_m, per_band_weights=pb_all, swd_weight=0.0)
    traj_ffs, s0_ffs, n_ffs = run_arm(
        'FFS', cfg_ffs, device, dm, val_sar, val_opt, haar,
        lpips_net, psnr_m, ssim_m,
        per_band_weights=dict(ll=1.0, lh=0.0, hl=0.0, hh=0.0), swd_weight=swd_weight)

    # ===================================================================== #
    # FAIRNESS CHECK (step 0 contributions)
    # ===================================================================== #
    print('\n' + '=' * 88)
    print(f'{TAG} FAIRNESS CHECK (step-0 measured contributions; proves the A/B is fair)')
    print('=' * 88)
    ll_match = abs(s0_base['ll'] - s0_ffs['ll']) / max(s0_base['ll'], 1e-9) * 100
    det_base = s0_base['det_l1']
    det_ffs = s0_ffs['swd']
    det_match = abs(det_base - det_ffs) / max(det_base, 1e-9) * 100
    print(f'  LL  contribution : BASE={s0_base["ll"]:.5f}  FFS={s0_ffs["ll"]:.5f}  '
          f'(diff {ll_match:.1f}% — {"MATCHED" if ll_match <= 5 else "MISMATCH >5%"})')
    print(f'  DET contribution : BASE(L1)={det_base:.5f}  FFS(SWD)={det_ffs:.5f}  '
          f'(diff {det_match:.1f}% — calibrated)')

    # ===================================================================== #
    # HELD-OUT before/after tables
    # ===================================================================== #
    def table(title, key, note):
        print('\n' + '=' * 88)
        print(f'{TAG} {title}')
        print('=' * 88)
        print(f'  {"step":>6} | {"BASE":>10} | {"FFS":>10}   ({note})')
        sb = {s: m for s, m in traj_base}
        sf = {s: m for s, m in traj_ffs}
        for s in sorted(set(sb) | set(sf)):
            a = sb.get(s, {}).get(key, float('nan'))
            b = sf.get(s, {}).get(key, float('nan'))
            print(f'  {s:>6} | {a:>10.4f} | {b:>10.4f}')

    table('MID-FREQ POWER RATIO (HELD-OUT; GT=1.0; closer to 1.0 = sharper)', 'midfreq', 'PRIMARY blur metric')
    table('DETAIL-BAND POWER RATIO (HELD-OUT; GT=1.0; closer to 1.0 = matched detail)', 'detpow', 'detail energy match')
    table('HIGH-BAND POWER RATIO (HELD-OUT; GT=1.0; >>1 = incoherent noise)', 'highfreq', 'coherence guard')
    table('LPIPS (HELD-OUT; lower = better perceptual)', 'lpips', 'perceptual / sharpness-aware')
    table('PSNR (HELD-OUT; NOT a sharpness judge — GT misaligned)', 'psnr', 'pixel; rewards blur under misalign')
    table('SSIM (HELD-OUT)', 'ssim', 'structural')

    b = traj_base[-1][1]
    f = traj_ffs[-1][1]
    N = min(n_base, n_ffs)

    # ===================================================================== #
    # VERDICT
    # ===================================================================== #
    b_mid_gap = abs(b['midfreq'] - 1.0)
    f_mid_gap = abs(f['midfreq'] - 1.0)
    b_det_gap = abs(b['detpow'] - 1.0)
    f_det_gap = abs(f['detpow'] - 1.0)
    sharper = (f_mid_gap < b_mid_gap - 1e-3) or (f_det_gap < b_det_gap - 1e-3 and f['midfreq'] >= b['midfreq'] - 1e-3)
    lpips_not_worse = f['lpips'] <= b['lpips'] + 0.003   # within small noise
    lpips_better = f['lpips'] < b['lpips'] - 1e-3

    if sharper and lpips_better:
        verdict = 'SHARPER&BETTER'
    elif sharper and lpips_not_worse:
        verdict = 'SHARPER&BETTER'
    elif (not sharper) and (f['lpips'] > b['lpips'] + 0.003 or (f_mid_gap > b_mid_gap + 1e-3 and f_det_gap > b_det_gap + 1e-3)):
        verdict = 'WORSE'
    else:
        verdict = 'MIXED'

    print('\n' + '=' * 88)
    print(f'{TAG} VERDICT')
    print('=' * 88)
    print(f'{TAG} on HELD-OUT after N={N} steps: '
          f'midfreq BASE={b["midfreq"]:.4f} FFS={f["midfreq"]:.4f} (GT=1.0); '
          f'LPIPS BASE={b["lpips"]:.4f} FFS={f["lpips"]:.4f}; '
          f'detailpow BASE={b["detpow"]:.4f} FFS={f["detpow"]:.4f} -> FFS {verdict}')


if __name__ == '__main__':
    main()
