"""In-model head-to-head: does FFS (WaveletDetailSWDLoss) un-blur vs per-band L1?

THROWAWAY diagnostic (uncommitted). From the SAME warm-started generator, run two
SHORT, matched trainings and measure whether the FFS detail loss produces SHARPER
output than the baseline per-band pixel L1, at matched gradient steps.

Two arms (each from a FRESH reload of the warm ckpt -> identical start):
  BASE : per_band L1 ON (config band weights), swd OFF.
  FFS  : swd_weight>0 on the 3 detail bands, per_band restricted to LL-only
         (lh=hl=hh=0, ll=1). Everything else (gan/fm/msssim/lpips/ffl + D updates)
         is IDENTICAL across arms.

Each arm runs the SAME fixed set of train batches for STEPS generator updates
(with matched D updates). Measurement on a FIXED val batch, BEFORE and AFTER:
  * mid-frequency radial power ratio fake/opt (blur metric; higher->sharper, 1.0=match)
  * highest-band power ratio fake/opt (1.0=match; >1 = incoherent over-energy)
  * detail-band relative L1 error (per_band detail L1 / detail energy)
  * LPIPS, PSNR, SSIM

This is a MECHANISM PROBE at ~300-400 steps, not convergence. We report the
DIRECTION/trend and whether FFS sharpens MORE than BASE at matched steps.

Run from repo root::

    python -m src.models.llwt_v5.proof_ffs_inmodel
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
from src.data.sen12_full_align.datamodule import SEN12FullDataModule

CONFIG_PATH = 'src/models/llwt_v5/config.yaml'
CKPT = 'checkpoints/llwt_v45/llwt-v0.5.0-sawg/last.ckpt'
TAG = '[ffs-inmodel]'

STEPS = 320               # generator updates per arm (~mechanism probe, not convergence)
N_TRAIN_BATCHES = 8       # fixed pool of train batches, cycled for STEPS
SWD_WEIGHT = 4.0          # FFS detail weight (calibrated below to match per_band detail contribution)
SEED = 42


# --------------------------------------------------------------------------- #
# measurement helpers
# --------------------------------------------------------------------------- #
def _radial_power_profile(x: torch.Tensor, n_bins: int = 64):
    """Mean radial power spectrum of x (B,C,H,W). Returns (centers[0..1], power)."""
    x = x.float()
    B, C, H, W = x.shape
    fft = torch.fft.fftshift(torch.fft.fft2(x), dim=(-2, -1))
    power = (fft.abs() ** 2).mean(dim=(0, 1))            # (H,W) averaged over B,C
    cy, cx = H // 2, W // 2
    ys = torch.arange(H, device=x.device).view(-1, 1) - cy
    xs = torch.arange(W, device=x.device).view(1, -1) - cx
    r = torch.sqrt((ys.float() ** 2 + xs.float() ** 2))
    r_max = float(r.max())
    r_norm = r / r_max                                   # 0..1, 1 = corner (>Nyquist)
    # bin by normalized radius up to Nyquist (=0.5 of the diagonal corner ~ edge)
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
    """fake/opt mean power ratio in MID (1/6..1/2 Nyquist) and HIGH (>=0.8 Nyq) bands.

    Radius is normalized so r=0.5 == Nyquist (image edge). MID band = r in
    [0.5/6, 0.5/2] = [0.0833, 0.25]. HIGH band = r in [0.4, 0.5].
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
    fb = haar(fake.float())                              # (LL, LH, HL, HH)
    ob = haar(opt.float())
    det_fake = torch.stack(fb[1:], dim=2)                # (B,C,3,h,w)
    det_real = torch.stack(ob[1:], dim=2)
    det_l1 = (det_fake - det_real).abs().mean()
    det_en = det_real.abs().mean().clamp_min(1e-9)
    return float(det_l1 / det_en)


@torch.no_grad()
def measure(netG, sar_v, opt_v, haar, lpips_net, psnr_m, ssim_m):
    netG.eval()
    fake = netG(sar_v).float().clamp(-1, 1)
    mid_r, high_r = band_power_ratios(fake, opt_v)
    drel = detail_rel_l1(fake, opt_v, haar)
    lp = float(lpips_net(fake, opt_v).mean())
    ps = float(psnr_m(fake, opt_v))
    ss = float(ssim_m(fake, opt_v))
    netG.train()
    return dict(midfreq=mid_r, highfreq=high_r, detail_rel=drel, lpips=lp, psnr=ps, ssim=ss)


# --------------------------------------------------------------------------- #
# checkpoint load (G + D) — strict=False warm start
# --------------------------------------------------------------------------- #
def load_GD(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    raw = ckpt.get('state_dict', ckpt)
    g_sd = {k[len('netG.'):]: v for k, v in raw.items() if k.startswith('netG.')}
    d_sd = {k[len('netD.'):]: v for k, v in raw.items() if k.startswith('netD.')}
    mg, ug = model.netG.load_state_dict(g_sd, strict=False)
    md, ud = model.netD.load_state_dict(d_sd, strict=False)
    return len(g_sd), len(mg), len(ug), len(d_sd), len(md), len(ud)


# --------------------------------------------------------------------------- #
# one matched arm
# --------------------------------------------------------------------------- #
def run_arm(name, cfg_arm, device, train_batches, sar_v, opt_v,
            haar, lpips_net, psnr_m, ssim_m):
    print(f'\n{TAG} ===== ARM {name} ===== build + warm-load')
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model = LLWv4LightningModule(cfg_arm)
    ng, mg, ug, nd, md, ud = load_GD(model, CKPT)
    print(f'{TAG} {name}: G keys={ng} (missing {mg}, unexpected {ug}); '
          f'D keys={nd} (missing {md}, unexpected {ud})')

    netG = model.netG.to(device).train()
    netD = model.netD.to(device).train()
    crit = model.criterions.to(device)
    loss_cfg = cfg_arm.loss
    have_pb = 'per_band' in crit
    have_swd = 'wavelet_swd' in crit
    use_sub = bool(getattr(netD, 'use_sub', False))
    print(f'{TAG} {name}: criterions={sorted(list(crit.keys()))} | '
          f'per_band={have_pb} wavelet_swd={have_swd} subD={use_sub}')

    # Optimizers: plain Adam on G(+per_band params) and D. lr from cfg.
    g_params = list(netG.parameters())
    if have_pb:
        g_params += list(crit['per_band'].parameters())
    opt_g = torch.optim.Adam(g_params, lr=float(cfg_arm.optimizer.lr_g),
                             betas=(float(cfg_arm.optimizer.beta1), float(cfg_arm.optimizer.beta2)))
    opt_d = torch.optim.Adam(netD.parameters(), lr=float(cfg_arm.optimizer.lr_d),
                             betas=(float(cfg_arm.optimizer.beta1), float(cfg_arm.optimizer.beta2)))

    before = measure(netG, sar_v, opt_v, haar, lpips_net, psnr_m, ssim_m)
    print(f'{TAG} {name} BEFORE: '
          f"midfreq={before['midfreq']:.4f} highfreq={before['highfreq']:.4f} "
          f"detail_rel={before['detail_rel']:.4f} lpips={before['lpips']:.4f} "
          f"psnr={before['psnr']:.3f} ssim={before['ssim']:.4f}")

    nb = len(train_batches)
    swd_seen = pb_seen = 0.0
    t0 = time.perf_counter()
    for step in range(STEPS):
        sar, opt = train_batches[step % nb]
        sar = sar.to(device)
        opt = opt.to(device)
        B = sar.size(0)

        # ---- D update (matched across arms; identical D init + data order) ----
        with torch.no_grad():
            fake_d = netG(sar)
        sar2 = sar.repeat(2, 1, 1, 1)
        opt2 = torch.cat([opt, fake_d], dim=0)
        main_both, sub_both, _f, _a, _b, _c = netD(sar2, opt2)
        lc, lf = main_both
        real_main = (lc[:B], lf[:B]); fake_main = (lc[B:], lf[B:])
        d_loss = 0.5 * (crit['gan'](real_main, is_real=True) +
                        crit['gan'](fake_main, is_real=False))
        if use_sub:
            real_sub = sub_both[:B]; fake_sub = sub_both[B:]
            d_loss = d_loss + 0.5 * (crit['gan'](real_sub, is_real=True) +
                                     crit['gan'](fake_sub, is_real=False))
        opt_d.zero_grad(); d_loss.backward()
        torch.nn.utils.clip_grad_norm_(netD.parameters(), 5.0)
        opt_d.step()

        # ---- G update ----
        need_int = have_pb or have_swd
        if need_int:
            fake, predicted_sub, _ = netG(sar, return_internals=True)
        else:
            fake = netG(sar); predicted_sub = None

        main_both_g, sub_both_g, _fg, fm_main_both, fm_sub_both, _cg = netD(sar2, torch.cat([fake, opt], dim=0))
        lcg, lfg = main_both_g
        fake_main_g = (lcg[:B], lfg[:B])
        fake_feats_main = [f[:B] for f in fm_main_both]
        real_feats_main = [f[B:].detach() for f in fm_main_both]

        l_gan = crit['gan'](fake_main_g, is_real=True, for_d=False)
        l_fm = crit['fm'](fake_feats_main, real_feats_main)
        g_loss = (l_gan * float(loss_cfg.gan_main_weight) +
                  l_fm * float(loss_cfg.fm_main_weight))
        if use_sub:
            fake_sub_g = sub_both_g[:B]
            l_gan_sub = crit['gan'](fake_sub_g, is_real=True, for_d=False)
            g_loss = g_loss + l_gan_sub * float(getattr(loss_cfg, 'gan_sub_weight', 1.0))
            ffs = [f[:B] for f in fm_sub_both]; rfs = [f[B:].detach() for f in fm_sub_both]
            if ffs:
                g_loss = g_loss + crit['fm'](ffs, rfs) * float(getattr(loss_cfg, 'fm_sub_weight', 10.0))

        if 'msssim' in crit:
            g_loss = g_loss + crit['msssim'](fake, opt) * float(loss_cfg.msssim_weight)
        if have_pb:
            l_pb = crit['per_band'](predicted_sub, opt)
            g_loss = g_loss + l_pb * float(loss_cfg.per_band_weight)
            pb_seen += float(l_pb.detach()) * float(loss_cfg.per_band_weight)
        if have_swd:
            l_swd = crit['wavelet_swd'](predicted_sub, opt)
            g_loss = g_loss + l_swd * float(loss_cfg.swd_weight)
            swd_seen += float(l_swd.detach()) * float(loss_cfg.swd_weight)
        if 'lpips' in crit:
            g_loss = g_loss + crit['lpips'](fake, opt) * float(loss_cfg.lpips_weight)
        if 'ffl' in crit:
            g_loss = g_loss + crit['ffl'](fake, opt) * float(loss_cfg.ffl_weight)

        opt_g.zero_grad(); g_loss.backward()
        torch.nn.utils.clip_grad_norm_(g_params, 1.0)
        opt_g.step()

        if step % 80 == 0 or step == STEPS - 1:
            print(f'{TAG} {name} step {step:4d}/{STEPS}  g_loss={float(g_loss):.4f} '
                  f'd_loss={float(d_loss):.4f}')

    dt = time.perf_counter() - t0
    if have_pb:
        print(f'{TAG} {name}: mean per_band detail+LL contribution (w*loss) ~ {pb_seen / STEPS:.4f}')
    if have_swd:
        print(f'{TAG} {name}: mean SWD contribution (w*loss) ~ {swd_seen / STEPS:.4f}')

    after = measure(netG, sar_v, opt_v, haar, lpips_net, psnr_m, ssim_m)
    print(f'{TAG} {name} AFTER ({STEPS} steps, {dt:.1f}s): '
          f"midfreq={after['midfreq']:.4f} highfreq={after['highfreq']:.4f} "
          f"detail_rel={after['detail_rel']:.4f} lpips={after['lpips']:.4f} "
          f"psnr={after['psnr']:.3f} ssim={after['ssim']:.4f}")

    del model, netG, netD, crit, opt_g, opt_d
    torch.cuda.empty_cache()
    return before, after


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'{TAG} device={device} | STEPS={STEPS} | swd_weight=auto-calibrated below')
    base_cfg = OmegaConf.load(CONFIG_PATH)
    # Disable the online aligner in BOTH arms: it would train jointly with G and
    # add run-to-run noise that is unrelated to the detail-loss question. With it
    # off, opt_aligned == opt and both arms are identical except the detail loss.
    if 'align' in base_cfg and base_cfg.align is not None:
        base_cfg.align.enabled = False
    # Drop patchnce (needs encoder_dims; irrelevant to detail sharpness and adds
    # a sampler MLP that differs run-to-run) and lab_chroma to keep arms lean+identical.
    base_cfg.loss.patchnce_weight = 0.0
    base_cfg.loss.lab_chroma_weight = 0.0
    base_cfg.system.compile = False
    base_cfg.system.channels_last = False

    # ---- fixed data: train pool + val batch (aligned mirror, no aug) ----
    data_root = base_cfg.data.data_dir['sen12_full_align']
    dm = SEN12FullDataModule(
        data_dir=data_root,
        batch_size=base_cfg.data.batch_size,
        image_size=base_cfg.data.image_size,
        num_workers=0,
        persistent_workers=False,
        prefetch_factor=base_cfg.data.prefetch_factor,
        train_val_split_ratio=base_cfg.data.train_val_split_ratio,
        seed=base_cfg.data.seed,
        sar_channels=base_cfg.data.sar_channels,
        use_augmentation=False,
        scenes=list(base_cfg.data.scenes),
        val_batch_size=base_cfg.data.get('val_batch_size', None) or base_cfg.data.batch_size,
    )
    dm.setup('fit')
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()

    train_batches = []
    for b in train_loader:
        train_batches.append((b[0].clone(), b[1].clone()))
        if len(train_batches) >= N_TRAIN_BATCHES:
            break
    vb = next(iter(val_loader))
    sar_v = vb[0].to(device)
    opt_v = vb[1].to(device).float()
    print(f'{TAG} fixed train pool={len(train_batches)} batches '
          f'(bs={train_batches[0][0].size(0)}); val batch={sar_v.size(0)} imgs')

    haar = HaarDown(in_channels=3).to(device)
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
    lpips_net = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=False).to(device).eval()
    for p in lpips_net.parameters():
        p.requires_grad_(False)
    psnr_m = PeakSignalNoiseRatio(data_range=2.0).to(device)
    ssim_m = StructuralSimilarityIndexMeasure(data_range=2.0).to(device)

    # ---- calibrate FFS swd_weight so SWD's detail contribution ~ matches what
    # per_band's DETAIL bands contributed in BASE (one warm forward). ----
    from src.models.llwt_v5.losses import PerBandWaveletL1Loss, WaveletDetailSWDLoss
    _calib = LLWv4LightningModule(base_cfg)
    load_GD(_calib, CKPT)
    _g = _calib.netG.to(device).eval()
    with torch.no_grad():
        _, _sub, _ = _g(sar_v, return_internals=True)
        # BASE per_band detail-only contribution: master per_band_weight * (weighted
        # detail L1 share). Approx with the shipped band ratios, detail share only.
        _pbw = float(base_cfg.loss.per_band_weight)
        _bw = dict(ll=float(base_cfg.loss.per_band_ll), lh=float(base_cfg.loss.per_band_lh),
                   hl=float(base_cfg.loss.per_band_hl), hh=float(base_cfg.loss.per_band_hh))
        _pb = PerBandWaveletL1Loss(band_weights=_bw).to(device)
        _ = _pb(_sub, opt_v)
        _per = _pb.last_per_band                       # raw L1 per band
        _wsum = sum(_bw.values())
        _detail_share = (_bw['lh'] * float(_per['lh']) + _bw['hl'] * float(_per['hl']) +
                         _bw['hh'] * float(_per['hh'])) / max(_wsum, 1e-9)
        _base_detail_contrib = _pbw * _detail_share
        _swd_raw = float(WaveletDetailSWDLoss().to(device)(_sub, opt_v))
    _swd_w = _base_detail_contrib / max(_swd_raw, 1e-9)
    swd_weight = float(np.clip(_swd_w, 0.5, 50.0))
    print(f'{TAG} calibration: BASE per_band detail contribution ~ {_base_detail_contrib:.4f} '
          f'(pbw={_pbw}, detail-share L1={_detail_share:.4f}); raw SWD={_swd_raw:.4f} '
          f'-> swd_weight={swd_weight:.3f} (raw target ~ {_swd_w:.3f}, clamped [0.5,50])')
    del _calib, _g
    torch.cuda.empty_cache()

    # ---- arm configs ----
    # BASE: per_band ON (config band weights), swd OFF.
    cfg_base = copy.deepcopy(base_cfg)
    cfg_base.loss.swd_weight = 0.0
    # ensure per_band on (config default is 2.0); keep its band weights as shipped.

    # FFS: swd ON; per_band restricted to LL-only so detail is supervised by SWD.
    cfg_ffs = copy.deepcopy(base_cfg)
    cfg_ffs.loss.swd_weight = swd_weight
    cfg_ffs.loss.per_band_ll = 1.0
    cfg_ffs.loss.per_band_lh = 0.0
    cfg_ffs.loss.per_band_hl = 0.0
    cfg_ffs.loss.per_band_hh = 0.0
    cfg_ffs.loss.per_band_adaptive = False

    base_before, base_after = run_arm('BASE', cfg_base, device, train_batches,
                                      sar_v, opt_v, haar, lpips_net, psnr_m, ssim_m)
    ffs_before, ffs_after = run_arm('FFS', cfg_ffs, device, train_batches,
                                    sar_v, opt_v, haar, lpips_net, psnr_m, ssim_m)

    # ===================================================================== #
    # REPORT
    # ===================================================================== #
    def row(tag, b, a):
        return (f'  {tag:>4} | mid {b["midfreq"]:.4f}->{a["midfreq"]:.4f} '
                f'(d{a["midfreq"]-b["midfreq"]:+.4f}) | '
                f'high {b["highfreq"]:.4f}->{a["highfreq"]:.4f} '
                f'(d{a["highfreq"]-b["highfreq"]:+.4f}) | '
                f'detRel {b["detail_rel"]:.4f}->{a["detail_rel"]:.4f} '
                f'(d{a["detail_rel"]-b["detail_rel"]:+.4f})\n'
                f'       | lpips {b["lpips"]:.4f}->{a["lpips"]:.4f} '
                f'(d{a["lpips"]-b["lpips"]:+.4f}) | '
                f'psnr {b["psnr"]:.3f}->{a["psnr"]:.3f} '
                f'(d{a["psnr"]-b["psnr"]:+.3f}) | '
                f'ssim {b["ssim"]:.4f}->{a["ssim"]:.4f} '
                f'(d{a["ssim"]-b["ssim"]:+.4f})')

    print('\n' + '=' * 78)
    print(f'{TAG} BEFORE/AFTER ({STEPS} matched G steps from the SAME warm ckpt)')
    print('=' * 78)
    print(row('BASE', base_before, base_after))
    print(row('FFS', ffs_before, ffs_after))

    # blur metric = mid-freq power ratio toward 1.0; sharper = higher (when <1).
    base_mid_d = base_after['midfreq'] - base_before['midfreq']
    ffs_mid_d = ffs_after['midfreq'] - ffs_before['midfreq']
    # incoherent over-energy = how far highfreq ratio sits ABOVE 1.0; want it to drop.
    base_high_excess_d = abs(base_after['highfreq'] - 1.0) - abs(base_before['highfreq'] - 1.0)
    ffs_high_excess_d = abs(ffs_after['highfreq'] - 1.0) - abs(ffs_before['highfreq'] - 1.0)
    base_det_d = base_after['detail_rel'] - base_before['detail_rel']
    ffs_det_d = ffs_after['detail_rel'] - ffs_before['detail_rel']

    print('\n' + '=' * 78)
    print(f'{TAG} VERDICT')
    print('=' * 78)
    print(f'  midfreq power ratio d (toward 1.0 / higher=sharper when starting <1): '
          f'BASE {base_mid_d:+.4f}  FFS {ffs_mid_d:+.4f}')
    print(f'  highfreq |ratio-1| d (negative=less incoherent over-energy=better): '
          f'BASE {base_high_excess_d:+.4f}  FFS {ffs_high_excess_d:+.4f}')
    print(f'  detail relative-L1 d (lower=closer, but L1 itself rewards blur): '
          f'BASE {base_det_d:+.4f}  FFS {ffs_det_d:+.4f}')

    # Sharper-more = FFS gains more mid-freq power AND/OR reduces incoherent
    # high-freq over-energy more than BASE.
    mid_better = ffs_mid_d > base_mid_d + 1e-4
    high_better = ffs_high_excess_d < base_high_excess_d - 1e-4
    if mid_better and high_better:
        verdict = 'SHARPENS'
    elif mid_better or high_better:
        verdict = 'SHARPENS (partial)'
    elif abs(ffs_mid_d - base_mid_d) < 5e-3 and abs(ffs_high_excess_d - base_high_excess_d) < 5e-3:
        verdict = 'NEUTRAL'
    else:
        verdict = 'WORSE'
    print(f'\n{TAG} FFS {verdict} vs BASE: '
          f'midfreq BASE d={base_mid_d:+.4f} FFS d={ffs_mid_d:+.4f} | '
          f'high-excess BASE d={base_high_excess_d:+.4f} FFS d={ffs_high_excess_d:+.4f}')
    print(f'{TAG} NOTE: {STEPS}-step probe -- report is DIRECTIONAL, not converged.')


if __name__ == '__main__':
    main()
