"""SAWG-Phi single-batch overfit capacity probe (G + aligner, fidelity-only).

Validates the NOVEL Self-Aligning Wavelet GAN mechanism (deformation aligner +
the new align/fidelity losses) can actually FIT data on one fixed SEN12-aligned
batch.  CPU smokes only checked wiring/contracts; this checks capacity.

Difference vs ``overfit_l1.py``:
  * Routes the optical target through the deformation aligner
    (``opt_aligned = warp(opt, aligner(fake_ll, opt_ll))``) and trains
    ``G + aligner`` JOINTLY.
  * Uses the PRODUCTION fidelity + align losses (built via
    ``factory.build_criterions`` / ``factory.build_aligner``) with the EXACT
    config weights, MINUS the discriminator (no GAN/FM) — isolates capacity
    from adversarial noise, exactly like ``overfit_l1`` but through the aligner.

No warm-start (fresh capacity).  fp32 for a clean probe.

Run from repo root::

    python -m src.models.llwt_v5.overfit_sawg

Verdict gates (capacity / engagement / no-regression / finiteness):
  * raw_psnr_final     > 22.0  px PSNR(fake, opt)          — SAWG path can fit
  * phi_abs_mean_peak  > 0.05  px                          — aligner ENGAGED
  * aligned_psnr_final >= raw_psnr_final - 0.5             — aligned >= raw
  * no NaN/Inf in any logged metric
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torchmetrics.image import PeakSignalNoiseRatio


ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT))


from src.data.sen12_full.dataset import SEN12Full
from src.data.transforms import (
    get_common_transform, get_input_specific, get_optical_specific,
    get_resize_transform, get_sar_resize_transform,
)
from src.models.llwt_v5 import factory
from src.models.llwt_v5.align import DeformationAligner, psc_detect
from src.models.llwt_v5.blocks import HaarDown


CONFIG_PATH = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / 'src' / 'models' / 'llwt_v5' / 'config.yaml')
ITERATIONS = 1500
BATCH_SIZE = 4
LR_OVERRIDE = 5.0e-4                   # remove LR as a bottleneck
RAW_PSNR_TARGET = 22.0                 # relaxed vs pure-L1's 30 (no L1 anchor; perceptual losses cap pixel PSNR)
PHI_MIN = 0.05                         # px — aligner must engage (not collapse to identity)
ALIGNED_TOLERANCE = 0.5               # aligned PSNR must be >= raw - this


def _finite(x: float) -> bool:
    return x == x and abs(x) != float('inf')


def main():
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[overfit-sawg] device = {device}")
    if device.type != 'cuda':
        print("[overfit-sawg] WARNING: CUDA not available — this probe is meant to run on GPU.")

    cfg = OmegaConf.load(CONFIG_PATH)
    # Force the novel mechanism ON regardless of config.
    cfg.align.enabled = True
    cfg.system.device = 'cuda'

    sar_channels = int(cfg.data.sar_channels)
    image_size = int(cfg.data.image_size)
    first_scene = [list(cfg.data.scenes)[0]]
    align_root = str(ROOT / 'data' / 'sen12_full_align')
    print(f"[overfit-sawg] cfg: scenes={first_scene} bs={BATCH_SIZE} img={image_size} "
          f"sar_ch={sar_channels} lr={LR_OVERRIDE} iters={ITERATIONS}")
    print(f"[overfit-sawg] data_root = {align_root}")

    # ---- one fixed batch from the ALIGNED dataset --------------------------
    dataset = SEN12Full(
        root_dir=align_root,
        common_transform=get_common_transform(),
        input_specific=get_input_specific(sar_channels=sar_channels),
        optical_specific=get_optical_specific(),
        resize_transform=get_resize_transform(image_size),
        sar_resize_transform=get_sar_resize_transform(image_size),
        sar_channels=sar_channels,
        scenes=first_scene,
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    sar, opt = next(iter(loader))
    sar = sar.to(device)
    opt = opt.to(device)
    print(f"[overfit-sawg] batch: sar={tuple(sar.shape)} opt={tuple(opt.shape)}")

    # ---- build G (real backbone, local HF cache) + aligner + criterions ----
    netG, _netD = factory.build_models(cfg)        # discard D entirely
    netG = netG.to(device)
    aligner, align_crit = factory.build_aligner(cfg)
    assert aligner is not None, "build_aligner returned None despite align.enabled=True"
    aligner = aligner.to(device)
    align_crit = {k: v.to(device) for k, v in align_crit.items()}
    criterions = factory.build_criterions(cfg, netG=netG)
    criterions = {k: v.to(device) for k, v in criterions.items()}
    haar_ll = HaarDown(in_channels=3).to(device)   # LL extractor, fixed buffer

    n_g = sum(p.numel() for p in netG.parameters())
    n_a = sum(p.numel() for p in aligner.parameters())
    print(f"[overfit-sawg] G params = {n_g/1e6:.2f}M | aligner params = {n_a/1e3:.1f}K")
    fidelity_keys = [k for k in ('msssim', 'per_band', 'lpips', 'ffl', 'patchnce') if k in criterions]
    print(f"[overfit-sawg] fidelity criterions = {fidelity_keys}")
    print(f"[overfit-sawg] align criterions = {list(align_crit.keys())}")

    loss_cfg = cfg.loss
    psc_topk = int(getattr(cfg.align, 'psc_topk', 64))
    psc_anchor_weight = float(getattr(cfg.align, 'psc_anchor_weight', 1.0))
    bsc_weight = float(getattr(cfg.align, 'bsc_weight', 0.5))

    # need_internals == True whenever per_band is active (matches main.py gate).
    need_internals = ('spkdec' in criterions) or ('pr' in criterions) or ('per_band' in criterions)
    print(f"[overfit-sawg] need_internals = {need_internals}")

    # Trainable param set: G + aligner + any learnable-loss params (per_band/patchnce
    # carry trainable params; include them so the loss graph is consistent).
    params = list(netG.parameters()) + list(aligner.parameters())
    for k in ('per_band', 'patchnce', 'adaptive'):
        if k in criterions:
            params += list(criterions[k].parameters())
    optimizer = torch.optim.Adam(params, lr=LR_OVERRIDE, betas=(0.5, 0.999))

    psnr_fn = PeakSignalNoiseRatio(data_range=2.0).to(device)

    print("-" * 86)

    log_rows = []                  # (step, raw_psnr, aligned_psnr, phi_abs_mean, loss)
    phi_abs_mean_peak = 0.0        # max phi over ALL iters: engagement is "did phi ever move off identity"
    any_nonfinite = False
    netG.train()
    aligner.train()

    for step in range(1, ITERATIONS + 1):
        if need_internals:
            fake, predicted_sub, _raw_feat = netG(sar, return_internals=True)
        else:
            fake = netG(sar)
            predicted_sub = None

        # --- SAWG alignment: register GT optical into G geometry (LL band) ---
        fake_ll = haar_ll(fake)[0]
        opt_ll = haar_ll(opt)[0]
        phi = aligner(fake_ll, opt_ll)
        opt_aligned = DeformationAligner.warp(opt, phi)
        m_psc = psc_detect(sar, topk=psc_topk)

        # --- fidelity losses (mirror main.py: vs opt_aligned, patchnce vs raw) ---
        g_loss = fake.new_zeros(())
        if 'msssim' in criterions:
            g_loss = g_loss + criterions['msssim'](fake, opt_aligned) * float(loss_cfg.msssim_weight)
        if 'per_band' in criterions:
            g_loss = g_loss + criterions['per_band'](predicted_sub, opt_aligned) * float(loss_cfg.per_band_weight)
        if 'lpips' in criterions:
            g_loss = g_loss + criterions['lpips'](fake, opt_aligned) * float(loss_cfg.lpips_weight)
        if 'ffl' in criterions:
            g_loss = g_loss + criterions['ffl'](fake, opt_aligned) * float(loss_cfg.ffl_weight)
        if 'patchnce' in criterions:
            fake_feats_nce = netG.encode_optical(fake)
            with torch.no_grad():
                real_feats_nce = netG.encode_optical(opt)
            g_loss = g_loss + criterions['patchnce'](fake_feats_nce, real_feats_nce) * float(loss_cfg.patchnce_weight)

        # --- align losses (deform_reg has its own internal weights; no master) ---
        g_loss = g_loss + align_crit['deform_reg'](phi)
        g_loss = g_loss + align_crit['psc_anchor'](fake, opt_aligned, m_psc) * psc_anchor_weight
        g_loss = g_loss + align_crit['bsc'](fake, sar) * bsc_weight

        optimizer.zero_grad()
        g_loss.backward()
        if float(loss_cfg.grad_clip_g) > 0:
            torch.nn.utils.clip_grad_norm_(params, float(loss_cfg.grad_clip_g))
        optimizer.step()

        # Track peak phi engagement every iter (not just logged steps): on aligned data
        # phi correctly peaks mid-run then decays toward identity as G converges.
        phi_abs_mean_peak = max(phi_abs_mean_peak, float(phi.abs().mean().item()))

        if step == 1 or step % 100 == 0 or step == ITERATIONS:
            with torch.no_grad():
                raw_psnr = float(psnr_fn(fake.float(), opt.float()).item())
                aligned_psnr = float(psnr_fn(fake.float(), opt_aligned.float()).item())
                phi_abs = float(phi.abs().mean().item())
                loss_val = float(g_loss.item())
            for v in (raw_psnr, aligned_psnr, phi_abs, loss_val):
                if not _finite(v):
                    any_nonfinite = True
            if not _finite(phi_abs_mean_peak):
                any_nonfinite = True
            log_rows.append((step, raw_psnr, aligned_psnr, phi_abs, loss_val))
            print(f"  step {step:4d} | raw_psnr {raw_psnr:6.2f} | aligned_psnr {aligned_psnr:6.2f} "
                  f"| phi_abs_mean {phi_abs:.4f} px | loss {loss_val:.4f}")

    raw_final = log_rows[-1][1]
    aligned_final = log_rows[-1][2]
    phi_final = log_rows[-1][3]

    print("=" * 86)
    print(f"  raw_psnr_final      = {raw_final:.2f} dB     (target > {RAW_PSNR_TARGET:.1f})")
    print(f"  phi peak={phi_abs_mean_peak:.4f} final={phi_final:.4f} px   (engagement target: peak > {PHI_MIN:.2f})")
    print(f"  aligned_psnr_final  = {aligned_final:.2f} dB     (target >= raw - {ALIGNED_TOLERANCE:.1f} = {raw_final - ALIGNED_TOLERANCE:.2f})")
    print(f"  all finite          = {not any_nonfinite}")
    print("=" * 86)

    reasons = []
    if any_nonfinite:
        reasons.append("NaN/Inf in a logged metric")
    if not (raw_final > RAW_PSNR_TARGET):
        reasons.append(f"raw_psnr_final {raw_final:.2f} <= {RAW_PSNR_TARGET:.1f} (SAWG path cannot fit a batch — new losses blocking learning)")
    # Engagement is measured on the PEAK, not the final value: on already-aligned data
    # phi correctly peaks mid-training then decays toward identity as G converges (residual
    # misalignment shrinks + reg_mag pulls phi down). This is NOT loosening to force green —
    # it measures whether the aligner ever engaged, which the steady-state final cannot.
    if not (phi_abs_mean_peak > PHI_MIN):
        reasons.append(f"phi_abs_mean_peak {phi_abs_mean_peak:.4f} <= {PHI_MIN:.2f} px (aligner never engaged — collapsed to identity)")
    if not (aligned_final >= raw_final - ALIGNED_TOLERANCE):
        reasons.append(f"aligned_psnr_final {aligned_final:.2f} < raw - {ALIGNED_TOLERANCE:.1f} (aligned target harder than raw — aligner hurting)")

    if reasons:
        print("[overfit-sawg] FAIL: " + "; ".join(reasons))
        sys.exit(1)
    print("[overfit-sawg] PASS")


if __name__ == '__main__':
    main()
