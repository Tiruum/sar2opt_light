"""GPU error-profiling diagnostic for the trained LLWv4/SAWG generator.

Forward-only, no training. Loads the most-trained checkpoint (strict=False),
runs ~200-400 aligned val patches, and answers WHERE the model loses quality:

  1. Overall PSNR / SSIM / LPIPS baseline.
  2. Per-Haar-band L1 + relative error (LL/LH/HL/HH) -> low- vs high-freq failure.
  3. Color vs structure: LAB L (luma) vs a,b (chroma) error + per-RGB bias.
  4. High-frequency deficit: radial power spectrum ratio fake/opt (low/mid/high).
  5. Per-scene PSNR breakdown.
  6. Worst vs best patch PSNR distribution (min/p10/median/p90/max).

Run from repo root::

    python -m src.models.llwt_v5.diag_errorprofile

Throwaway diagnostic — no commit, no side effects beyond stdout.
"""
from __future__ import annotations

import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

import functools
import time
from collections import defaultdict

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
CKPT_CANDIDATES = [
    'checkpoints/llwt_v45/llwt-v0.5.0-sawg/last.ckpt',
    'checkpoints/llwt_v45/llwt-v0.4.6-align-ws/last.ckpt',
]
MAX_PATCHES = 320          # ~budget; forward-only
TAG = '[errorprofile]'


def _load_gen_weights(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    raw_sd = ckpt.get('state_dict', ckpt)
    gen_sd = {k[len('netG.'):]: v for k, v in raw_sd.items() if k.startswith('netG.')}
    mg, ug = model.netG.load_state_dict(gen_sd, strict=False)
    print(f'{TAG} loaded {ckpt_path} | G keys={len(gen_sd)} missing={len(mg)} unexpected={len(ug)}')
    return len(gen_sd)


def _rgb_to_lab_np(rgb01: np.ndarray) -> np.ndarray:
    """rgb in [0,1] (H,W,3) -> CIELAB via sRGB->XYZ->Lab (D65). Vectorized."""
    # inverse sRGB gamma
    a = 0.055
    lin = np.where(rgb01 <= 0.04045, rgb01 / 12.92, ((rgb01 + a) / (1 + a)) ** 2.4)
    M = np.array([[0.4124, 0.3576, 0.1805],
                  [0.2126, 0.7152, 0.0722],
                  [0.0193, 0.1192, 0.9505]], dtype=np.float64)
    xyz = lin.reshape(-1, 3) @ M.T
    # D65 white
    xyz /= np.array([0.95047, 1.0, 1.08883])
    eps = 216 / 24389
    kap = 24389 / 27
    f = np.where(xyz > eps, np.cbrt(xyz), (kap * xyz + 16) / 116)
    fx, fy, fz = f[:, 0], f[:, 1], f[:, 2]
    L = 116 * fy - 16
    aa = 500 * (fx - fy)
    bb = 200 * (fy - fz)
    return np.stack([L, aa, bb], axis=-1).reshape(rgb01.shape)


def _radial_power(img01: torch.Tensor) -> np.ndarray:
    """Radially-averaged power spectrum of a batch of grayscale [0,1] images.

    img01: (B,1,H,W). Returns 1D array length = nbins (radial profile, averaged
    over batch). Uses log power.
    """
    B, _, H, W = img01.shape
    f = torch.fft.fftshift(torch.fft.fft2(img01.squeeze(1)), dim=(-2, -1))
    power = (f.abs() ** 2)                      # (B,H,W)
    cy, cx = H // 2, W // 2
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    r = torch.sqrt((yy - cy).float() ** 2 + (xx - cx).float() ** 2)
    r = r.round().long().to(power.device)
    nbins = int(r.max().item()) + 1
    prof = torch.zeros(nbins, device=power.device)
    cnt = torch.zeros(nbins, device=power.device)
    pflat = power.mean(0).reshape(-1)           # avg over batch
    rflat = r.reshape(-1)
    prof.scatter_add_(0, rflat, pflat)
    cnt.scatter_add_(0, rflat, torch.ones_like(pflat))
    prof = prof / cnt.clamp_min(1)
    return prof.cpu().numpy()


def main():
    cfg = OmegaConf.load(CONFIG_PATH)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ---- model + ckpt ----
    model = LLWv4LightningModule(cfg)
    ckpt_used = None
    for c in CKPT_CANDIDATES:
        if os.path.isfile(c):
            try:
                n = _load_gen_weights(model, c)
                if n > 0:
                    ckpt_used = c
                    break
            except Exception as e:                       # noqa: BLE001
                print(f'{TAG} failed {c}: {e}')
    if ckpt_used is None:
        raise RuntimeError('No checkpoint loaded.')
    netG = model.netG.to(device).eval()
    use_cl = bool(getattr(cfg.system, 'channels_last', False))
    if use_cl:
        netG = netG.to(memory_format=torch.channels_last)

    haar = HaarDown(in_channels=3).to(device)

    # ---- aligned val dataloader (same split as train.py) ----
    dataset_name = 'sen12_full_align'
    data_root = cfg.data.data_dir[dataset_name]
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
    val_ds = dm.val_dataset
    # scene id per val item (items tuple = (season, s1_dir, s2_dir, s1f, s2f))
    scene_of = [it[1][3:] for it in val_ds.items]   # strip "s1_"
    print(f'{TAG} dataset={dataset_name} root={data_root} | val_n={len(val_ds)} '
          f'scenes={sorted(set(scene_of))}')

    loader = dm.val_dataloader()
    bs = dm.val_batch_size

    # ---- accumulators ----
    import torchmetrics.functional as tmf

    band_l1 = {b: 0.0 for b in ('LL', 'LH', 'HL', 'HH')}
    band_energy = {b: 0.0 for b in ('LL', 'LH', 'HL', 'HH')}
    lab_err = {'L': 0.0, 'a': 0.0, 'b': 0.0}
    rgb_err = np.zeros(3)
    rgb_bias = np.zeros(3)               # mean(fake)-mean(opt) per channel, in [0,1]
    psnr_sum = ssim_sum = lpips_sum = 0.0
    n_imgs = 0
    n_lab_imgs = 0

    rp_fake_sum = None
    rp_opt_sum = None
    rp_batches = 0

    per_scene_psnr = defaultdict(list)
    all_psnr = []

    lpips_net = model.lpips.to(device)
    lpips_net.eval()

    t0 = time.perf_counter()
    seen = 0
    with torch.no_grad():
        for bi, (sar, opt) in enumerate(loader):
            if seen >= MAX_PATCHES:
                break
            sar = sar.to(device, non_blocking=True)
            opt = opt.to(device, non_blocking=True)
            if use_cl:
                sar = sar.contiguous(memory_format=torch.channels_last)
                opt = opt.contiguous(memory_format=torch.channels_last)
            fake = netG(sar)                       # [-1,1]
            fake = fake.float().clamp(-1, 1)
            opt = opt.float()
            B = fake.size(0)

            # scene ids for this batch
            idx0 = bi * bs
            batch_scenes = scene_of[idx0:idx0 + B]

            # ---- per-image PSNR (for distribution + per-scene) ----
            for i in range(B):
                p = float(tmf.peak_signal_noise_ratio(
                    fake[i:i+1], opt[i:i+1], data_range=2.0))
                all_psnr.append(p)
                if i < len(batch_scenes):
                    per_scene_psnr[batch_scenes[i]].append(p)

            # ---- overall metrics (batched) ----
            psnr_sum += float(tmf.peak_signal_noise_ratio(fake, opt, data_range=2.0)) * B
            ssim_sum += float(tmf.structural_similarity_index_measure(fake, opt, data_range=2.0)) * B
            lpips_sum += float(lpips_net(fake, opt)) * B   # normalize=False, expects [-1,1]

            # ---- per-Haar-band L1 + energy ----
            fb = haar(fake)      # tuple of 4 (B,3,H/2,W/2)
            ob = haar(opt)
            for name, ff, oo in zip(('LL', 'LH', 'HL', 'HH'), fb, ob):
                band_l1[name] += float((ff - oo).abs().mean()) * B
                band_energy[name] += float(oo.abs().mean()) * B

            # ---- color/structure (RGB + LAB), on [0,1] ----
            fake01 = ((fake + 1) / 2).clamp(0, 1)
            opt01 = ((opt + 1) / 2).clamp(0, 1)
            rgb_err += (fake01 - opt01).abs().mean(dim=(0, 2, 3)).cpu().numpy() * B
            rgb_bias += (fake01.mean(dim=(0, 2, 3)) - opt01.mean(dim=(0, 2, 3))).cpu().numpy() * B

            # LAB on a subset (CPU, slower) — cap to keep <10 min
            if n_lab_imgs < 160:
                fnp = fake01.permute(0, 2, 3, 1).cpu().numpy()
                onp = opt01.permute(0, 2, 3, 1).cpu().numpy()
                for i in range(B):
                    if n_lab_imgs >= 160:
                        break
                    fl = _rgb_to_lab_np(fnp[i])
                    ol = _rgb_to_lab_np(onp[i])
                    d = np.abs(fl - ol).mean(axis=(0, 1))   # [dL, da, db]
                    lab_err['L'] += float(d[0])
                    lab_err['a'] += float(d[1])
                    lab_err['b'] += float(d[2])
                    n_lab_imgs += 1

            # ---- radial power spectrum (grayscale luma proxy = mean RGB) ----
            fg = fake01.mean(dim=1, keepdim=True)
            og = opt01.mean(dim=1, keepdim=True)
            rpf = _radial_power(fg)
            rpo = _radial_power(og)
            L = min(len(rpf), len(rpo))
            if rp_fake_sum is None:
                rp_fake_sum = np.zeros(L)
                rp_opt_sum = np.zeros(L)
            rp_fake_sum[:L] += rpf[:L]
            rp_opt_sum[:L] += rpo[:L]
            rp_batches += 1

            n_imgs += B
            seen += B

    dt = time.perf_counter() - t0
    print(f'{TAG} forward done: {n_imgs} patches in {dt:.1f}s '
          f'({n_imgs/dt:.1f} img/s) on {device}')

    # ================= REPORT =================
    print('\n' + '=' * 64)
    print(f'{TAG} ERROR PROFILE  (ckpt={ckpt_used}, n={n_imgs} aligned val patches)')
    print('=' * 64)

    # 1. overall
    print('\n--- 1. OVERALL METRICS (baseline) ---')
    print(f'  PSNR  = {psnr_sum/n_imgs:7.3f} dB')
    print(f'  SSIM  = {ssim_sum/n_imgs:7.4f}')
    print(f'  LPIPS = {lpips_sum/n_imgs:7.4f}  (lower=better)')

    # 2. per-band
    print('\n--- 2. PER-WAVELET-BAND ERROR (Haar, fake vs opt) ---')
    print('  band      L1        rel(L1/energy)')
    rel = {}
    for b in ('LL', 'LH', 'HL', 'HH'):
        l1 = band_l1[b] / n_imgs
        en = band_energy[b] / n_imgs
        r = l1 / max(en, 1e-8)
        rel[b] = r
        print(f'  {b:3s}   {l1:8.5f}   {r:8.4f}   (opt energy {en:.5f})')
    ll_l1 = band_l1['LL'] / n_imgs
    detail_l1 = sum(band_l1[b] for b in ('LH', 'HL', 'HH')) / n_imgs / 3
    print(f'  ratio  mean(detail-L1)/LL-L1 = {detail_l1/max(ll_l1,1e-8):.3f}')
    print(f'  ratio  mean(detail-rel)/LL-rel = '
          f'{(sum(rel[b] for b in ("LH","HL","HH"))/3)/max(rel["LL"],1e-8):.3f}')

    # 3. color vs structure
    print('\n--- 3. COLOR vs STRUCTURE (LAB + per-RGB) ---')
    for k in ('L', 'a', 'b'):
        print(f'  LAB mean|err| {k}: {lab_err[k]/max(n_lab_imgs,1):8.4f}'
              f'   (n={n_lab_imgs})')
    re = rgb_err / n_imgs
    rb = rgb_bias / n_imgs
    print(f'  per-RGB mean|err| (0-1 scale): R={re[0]:.4f} G={re[1]:.4f} B={re[2]:.4f}')
    print(f'  per-RGB bias mean(fake-opt):   R={rb[0]:+.4f} G={rb[1]:+.4f} B={rb[2]:+.4f}')
    print(f'  global brightness bias = {rb.mean():+.4f}  (0-1 scale)')

    # 4. high-freq deficit
    print('\n--- 4. HIGH-FREQUENCY DEFICIT (radial power, fake/opt) ---')
    rpf = rp_fake_sum / rp_batches
    rpo = rp_opt_sum / rp_batches
    L = len(rpf)
    lo = slice(1, max(2, L // 6))                 # skip DC bin 0
    mid = slice(L // 6, L // 2)
    hi = slice(L // 2, L)
    def _ratio(s):
        return float(rpf[s].sum() / max(rpo[s].sum(), 1e-12))
    print(f'  freq band   fake/opt power ratio')
    print(f'  low  (DC..1/6 Nyq):  {_ratio(lo):.4f}')
    print(f'  mid  (1/6..1/2):     {_ratio(mid):.4f}')
    print(f'  high (1/2..Nyq):     {_ratio(hi):.4f}')
    print(f'  -> ratio<1 = model UNDER-produces energy at that band (blur)')

    # 5. per-scene
    print('\n--- 5. PER-SCENE PSNR ---')
    scene_means = []
    for s in sorted(per_scene_psnr, key=lambda x: int(x) if x.isdigit() else x):
        arr = np.array(per_scene_psnr[s])
        scene_means.append((s, arr.mean(), len(arr)))
        print(f'  scene {s:>4s}: PSNR={arr.mean():6.2f} dB  (n={len(arr)})')
    scene_means.sort(key=lambda t: t[1])
    if scene_means:
        print(f'  WORST scene: {scene_means[0][0]} ({scene_means[0][1]:.2f} dB) | '
              f'BEST scene: {scene_means[-1][0]} ({scene_means[-1][1]:.2f} dB)')

    # 6. distribution
    print('\n--- 6. PER-PATCH PSNR DISTRIBUTION ---')
    a = np.array(all_psnr)
    print(f'  min={a.min():.2f}  p10={np.percentile(a,10):.2f}  '
          f'median={np.median(a):.2f}  p90={np.percentile(a,90):.2f}  max={a.max():.2f}')
    # worst-10% scene composition
    thr = np.percentile(a, 10)
    worst_idx = [i for i, v in enumerate(all_psnr) if v <= thr]
    worst_scenes = defaultdict(int)
    for i in worst_idx:
        if i < len(scene_of_seen := []):
            pass
    # map worst patches back to scenes via per-image order
    flat_scene = []
    for s, lst in [(s, per_scene_psnr[s]) for s in per_scene_psnr]:
        pass
    # rebuild ordered scene list matching all_psnr order
    # (all_psnr appended per-image in loader order = scene_of order, truncated)
    ordered_scenes = scene_of[:len(all_psnr)]
    wc = defaultdict(int)
    for i in worst_idx:
        wc[ordered_scenes[i]] += 1
    comp = ', '.join(f'{k}:{v}' for k, v in sorted(wc.items(),
                     key=lambda kv: -kv[1]))
    print(f'  worst-10% (<= {thr:.2f} dB, n={len(worst_idx)}) scene composition: {comp}')

    # ---- verdict ----
    print('\n' + '=' * 64)
    detail_rel = sum(rel[b] for b in ('LH', 'HL', 'HH')) / 3
    hi_ratio = _ratio(hi)
    mid_ratio = _ratio(mid)
    luma = lab_err['L'] / max(n_lab_imgs, 1)
    chroma = (lab_err['a'] + lab_err['b']) / 2 / max(n_lab_imgs, 1)
    bits = []
    if hi_ratio < 0.7:
        bits.append(f'HIGH-FREQ TEXTURE DEFICIT (hi-band power ratio {hi_ratio:.2f}<<1 = blurry)')
    if detail_rel > rel['LL'] * 1.2:
        bits.append(f'detail bands worse than LL (rel ratio '
                    f'{detail_rel/max(rel["LL"],1e-8):.2f}) = loses edges/texture')
    if chroma > luma:
        bits.append(f'CHROMA error ({chroma:.2f}) > luma ({luma:.2f}) = loses COLOR')
    else:
        bits.append(f'luma error ({luma:.2f}) >= chroma ({chroma:.2f}) = structure-bound')
    if abs(rb.mean()) > 0.02:
        bits.append(f'global brightness bias {rb.mean():+.3f}')
    verdict = '; '.join(bits) if bits else 'no single dominant failure'
    print(f'{TAG} BOTTLENECK = {verdict}')
    print('=' * 64)


if __name__ == '__main__':
    main()
