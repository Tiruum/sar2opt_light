"""DECISIVE GENERALIZATION A/B for the HIGH-FREQUENCY DISCRIMINATOR (HF-D).

THROWAWAY diagnostic (uncommitted). Same validated harness as
``proof_ffs_generalize.py``: train on MANY distinct RAW misaligned pairs (no
memorization) and measure DETAIL SHARPNESS on a FIXED HELD-OUT val set NEVER
trained on.

Evidence chain: the model UNDER-produces mid-frequency (blur, power ratio
~0.66) and OVER-produces INCOHERENT high-frequency (speckle-noise, ~1.14).
Distributional losses (sliced-Wasserstein) FAILED to sharpen — they match
energy budgets but don't carve coherent structure. The PROVEN GAN-sharpening
mechanism is a discriminator that SEES high frequencies, forcing COHERENT
detail. This tests whether a HF-D beats the baseline on held-out sharpness.

Two arms, EACH from a FRESH reload of the same warm G+D (identical init):
  ARM BASE: current full loss stack (gan_main+fm+msssim+lpips+ffl+per_band),
            HF-D OFF. Real main-D updates.
  ARM HFD : identical + HF-D on highpass(opt). The D-phase ALSO updates the
            HF-D on highpass(real) vs highpass(fake.detach()); the G-phase adds
            hfd_weight * gan_loss(HF_D(highpass(fake)), real=True). HF-D
            fresh-initialized, updated each step. hfd_weight≈1.0 (matches
            gan_main scale).
  ONLY the HF-D differs. Everything else (loss stack, main-D updates, lr,
  optimizer, data order, steps) IDENTICAL across arms.

Eval (BEFORE step 0 + AFTER) on the HELD-OUT val set for BOTH arms:
  * mid-frequency radial power ratio fake/opt (1/6..1/2 Nyq) — PRIMARY blur
    metric (->1.0 = sharp).
  * highest-band power ratio (->1.0; >>1 = incoherent noise) — coherence guard.
  * detail Haar-band power ratio fake/opt (->1.0 = matched detail energy).
  * LPIPS / PSNR / SSIM.

D-stability watch: HF-D d_loss is logged each LOG_EVERY; if its logits go
NaN/huge it is REPORTED (the known subband-D failure mode — spectral norm
should prevent it).

Run from repo root::

    python -m src.models.llwt_v5.proof_hfd_generalize
"""
from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

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
from src.models.llwt_v5.dis import HighFreqDis
from src.models.llwt_v5.losses import PerBandWaveletL1Loss
from src.data.sen12_full.datamodule import SEN12FullDataModule

CONFIG_PATH = 'src/models/llwt_v5/config.yaml'
CKPT = 'checkpoints/llwt_v45/llwt-v0.5.0-sawg/last.ckpt'
TAG = '[hfd-gen]'

STEPS = 2200            # generator+D updates per arm; reduced if time budget hit
LOG_EVERY = 550
LR_G = 1.0e-4          # MODERATE (below 2e-4 default) — don't destroy warm spectrum
SEED = 42
TIME_BUDGET_S = 8.4 * 60  # hard per-arm wall budget (< 9 min); break early if hit
N_VAL_PAIRS = 32       # held-out eval set (NEVER trained on)
HFD_WEIGHT = 1.0       # G-side HF-D weight (≈ gan_main scale)
HFD_SIGMA = 2.0        # gaussian-blur sigma for the high-pass split


# --------------------------------------------------------------------------- #
# measurement helpers (verbatim from proof_ffs_generalize.py)
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
    det_fake = torch.stack(fb[1:], dim=2)
    det_real = torch.stack(ob[1:], dim=2)
    pf = (det_fake ** 2).mean()
    po = (det_real ** 2).mean().clamp_min(1e-12)
    return float(pf / po)


@torch.no_grad()
def measure(netG, val_sar, val_opt, haar, lpips_net, psnr_m, ssim_m, vbs=8):
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
        use_augmentation=False,
        scenes=list(base_cfg.data.scenes),
        val_batch_size=bs,
    )
    dm.setup('fit')
    return dm


# --------------------------------------------------------------------------- #
# one training arm: faithful replication of main.py training_step
#   (main-D on detached fake + G with the full listed stack), align + patchnce
#   OFF. When use_hfd: a fresh HF-D is built + updated each step on
#   highpass(real) vs highpass(fake.detach()); G adds hfd_weight * gan(HF-D fake).
# --------------------------------------------------------------------------- #
def run_arm(name, base_cfg, device, dm, val_sar, val_opt, haar,
            lpips_net, psnr_m, ssim_m, per_band_weights, use_hfd, return_model=False):
    print(f'\n{TAG} ===== ARM {name} ===== fresh warm G+D reload '
          f'(HF-D {"ON" if use_hfd else "OFF"})')
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    cfg = copy.deepcopy(base_cfg)
    netG, netD = factory.build_models(cfg)
    g_keys, mg, ug, d_keys, md, ud = load_GD(netG, netD, CKPT)
    print(f'{TAG} {name}: warm G keys={g_keys} (miss {mg}, unexp {ug}); '
          f'D keys={d_keys} (miss {md}, unexp {ud})')
    netG = netG.to(device).train()
    netD = netD.to(device).train()

    # HF-D: fresh-initialized conditional SN-PatchGAN on optical high-pass.
    in_ch = int(cfg.model.dis.main.in_channels)
    hfd = None
    opt_hfd = None
    if use_hfd:
        hfd = HighFreqDis(in_ch=in_ch, ndf=64, sigma=HFD_SIGMA).to(device).train()
        opt_hfd = torch.optim.Adam(hfd.parameters(), lr=float(cfg.optimizer.lr_d),
                                   betas=(float(cfg.optimizer.beta1), float(cfg.optimizer.beta2)))

    gan = factory.build_criterions(cfg)['gan'].to(device)
    from src.models.llwt_v5.losses import FeatureMatchingLoss, MSSSIMLoss, LPIPSLoss, FFLLoss
    fm = FeatureMatchingLoss().to(device)
    msssim = MSSSIMLoss().to(device)
    lpips_loss = LPIPSLoss(net_type='alex').to(device)
    ffl = FFLLoss(alpha=float(cfg.loss.ffl_alpha)).to(device)
    per_band = PerBandWaveletL1Loss(band_weights=per_band_weights).to(device)

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

    init = measure(netG, val_sar, val_opt, haar, lpips_net, psnr_m, ssim_m)
    traj = [(0, init)]
    print(f'{TAG} {name} HELD-OUT step    0/{STEPS}: {fmt(init)}')

    train_loader = dm.train_dataloader()
    data_iter = iter(train_loader)
    t0 = time.perf_counter()
    real_steps = 0
    hfd_unstable = False
    last_hfd_dloss = float('nan')
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

        # -------- main-D update on detached fake (batched real+fake) --------
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

        # -------- HF-D update on highpass(real) vs highpass(fake.detach()) --------
        if use_hfd:
            hp_real = hfd.highpass(opt, HFD_SIGMA)            # fp32 inside
            hp_fake = hfd.highpass(fake_d.float(), HFD_SIGMA)
            hp2 = torch.cat([hp_real, hp_fake], dim=0)
            hf_both, _ = hfd(sar2.float(), hp2)
            real_hf = hf_both[:B]
            fake_hf = hf_both[B:]
            d_loss_hf = 0.5 * (gan(real_hf, is_real=True) + gan(fake_hf, is_real=False))
            last_hfd_dloss = float(d_loss_hf.detach())
            if not torch.isfinite(d_loss_hf) or abs(last_hfd_dloss) > 1e4:
                hfd_unstable = True
            opt_hfd.zero_grad(set_to_none=True)
            d_loss_hf.backward()
            hgn = torch.nn.utils.clip_grad_norm_(hfd.parameters(), float(cfg.loss.grad_clip_d))
            if torch.isfinite(hgn):
                opt_hfd.step()

        # -------- G update (full listed stack [+ HF-D adversarial]) --------
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

            l_pb = per_band(predicted_sub, opt)
            g_loss = g_loss + l_pb * pb_w
            l_ms = msssim(fake, opt)
            g_loss = g_loss + l_ms * ms_w
            l_lp = lpips_loss(fake, opt)
            g_loss = g_loss + l_lp * lp_w
            l_ffl = ffl(fake, opt)
            g_loss = g_loss + l_ffl * ffl_w

            if use_hfd:
                hp_fake_g = hfd.highpass(fake, HFD_SIGMA)
                fake_hf_g, _ = hfd(sar.float(), hp_fake_g)
                l_gan_hf = gan(fake_hf_g, is_real=True, for_d=False)
                g_loss = g_loss + l_gan_hf * HFD_WEIGHT

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
            hfd_str = f' hfd_d={last_hfd_dloss:.3f}' if use_hfd else ''
            print(f'{TAG} {name} HELD-OUT step {step:4d}/{STEPS}: {fmt(m)}  '
                  f'g={float(g_loss):.3f} d={float(d_loss):.3f}{hfd_str}')
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
    if use_hfd:
        if hfd_unstable:
            print(f'{TAG} {name}: *** HF-D INSTABILITY DETECTED *** '
                  f'(NaN/huge d_loss seen; spectral-norm did NOT contain it)')
        else:
            print(f'{TAG} {name}: HF-D stable (last d_loss={last_hfd_dloss:.4f}, '
                  f'bounded throughout — spectral norm held)')

    del netD, opt_g, opt_d, per_band, fm, msssim, lpips_loss, ffl, gan
    if hfd is not None:
        del hfd, opt_hfd
    if return_model:
        torch.cuda.empty_cache()
        return traj, real_steps, hfd_unstable, netG
    del netG
    torch.cuda.empty_cache()
    return traj, real_steps, hfd_unstable


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'{TAG} device={device} | target STEPS={STEPS} lr_g={LR_G} '
          f'hfd_weight={HFD_WEIGHT} sigma={HFD_SIGMA} | '
          f'RAW MISALIGNED multi-pair generalization (sen12_full)')
    if device != 'cuda':
        print(f'{TAG} WARNING: CUDA not available — CPU will be too slow.')

    base_cfg = OmegaConf.load(CONFIG_PATH)
    if 'align' in base_cfg and base_cfg.align is not None:
        base_cfg.align.enabled = False
    base_cfg.loss.patchnce_weight = 0.0
    base_cfg.loss.lab_chroma_weight = 0.0
    base_cfg.system.compile = False
    base_cfg.system.channels_last = False
    bs = int(base_cfg.data.batch_size)

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

    pb_all = dict(ll=float(base_cfg.loss.per_band_ll), lh=float(base_cfg.loss.per_band_lh),
                  hl=float(base_cfg.loss.per_band_hl), hh=float(base_cfg.loss.per_band_hh))

    traj_base, n_base, _ = run_arm(
        'BASE', base_cfg, device, dm, val_sar, val_opt, haar,
        lpips_net, psnr_m, ssim_m, per_band_weights=pb_all, use_hfd=False)
    traj_hfd, n_hfd, hfd_unstable = run_arm(
        'HFD', base_cfg, device, dm, val_sar, val_opt, haar,
        lpips_net, psnr_m, ssim_m, per_band_weights=pb_all, use_hfd=True)

    # ===================================================================== #
    # HELD-OUT before/after tables
    # ===================================================================== #
    def table(title, key, note):
        print('\n' + '=' * 88)
        print(f'{TAG} {title}')
        print('=' * 88)
        print(f'  {"step":>6} | {"BASE":>10} | {"HFD":>10}   ({note})')
        sb = {s: m for s, m in traj_base}
        sf = {s: m for s, m in traj_hfd}
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
    f = traj_hfd[-1][1]
    N = min(n_base, n_hfd)

    # ===================================================================== #
    # VERDICT
    # ===================================================================== #
    b_mid_gap = abs(b['midfreq'] - 1.0)
    f_mid_gap = abs(f['midfreq'] - 1.0)
    b_high_gap = abs(b['highfreq'] - 1.0)
    f_high_gap = abs(f['highfreq'] - 1.0)
    # SHARPER = mid-freq moves toward 1.0 (less blur) ...
    mid_sharper = f_mid_gap < b_mid_gap - 1e-3
    # ... AND high-band did NOT explode into incoherent noise (worse than BASE
    # by a margin AND > 1.0 = over-production of high-freq).
    high_exploded = (f['highfreq'] > b['highfreq'] + 0.05) and (f['highfreq'] > 1.15)
    coherent = not high_exploded
    lpips_not_worse = f['lpips'] <= b['lpips'] + 0.003
    lpips_better = f['lpips'] < b['lpips'] - 1e-3

    if hfd_unstable:
        verdict = 'WORSE (HF-D UNSTABLE)'
    elif mid_sharper and coherent and lpips_not_worse:
        verdict = 'SHARPER&COHERENT'
    elif (not mid_sharper) and (f['lpips'] > b['lpips'] + 0.003 or high_exploded):
        verdict = 'WORSE'
    else:
        verdict = 'MIXED'

    print('\n' + '=' * 88)
    print(f'{TAG} VERDICT')
    print('=' * 88)
    print(f'{TAG} HELD-OUT after N={N} steps: '
          f'midfreq BASE={b["midfreq"]:.4f} HFD={f["midfreq"]:.4f} (GT=1.0); '
          f'highband BASE={b["highfreq"]:.4f} HFD={f["highfreq"]:.4f}; '
          f'LPIPS BASE={b["lpips"]:.4f} HFD={f["lpips"]:.4f} -> HFD {verdict}')
    if hfd_unstable:
        print(f'{TAG} NOTE: HF-D produced NaN/huge logits during training — the known '
              f'subband-D failure mode. Spectral norm did NOT contain it here.')


if __name__ == '__main__':
    main()
