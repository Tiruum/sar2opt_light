"""THROWAWAY GPU proof — which fix makes the SAWG deformation aligner ENGAGE?

The aligner stayed inert (phi ~ 0.07px) during training even though the data has
REAL misalignment (median 1.23px, p90 5.3px). Two diagnosed causes:
  B (reg over-damping) — already loosened in config (reg_smooth 10->3, reg_mag 2->0.5).
  C (no direct registration objective) — fidelity loss on a warm/good G gives phi a
    weak gradient, so loosening reg alone may not be enough.

This proof compares, with G FROZEN (warm v0.4.6 ckpt) and the aligner optimized
ALONE on REAL misaligned sen12_full batches:

  Variant 1 (reg-loosen only): loosened reg (3 / 0.5) + the existing align losses
    (psc_anchor anchored-L1 + corr) + the fidelity losses routed through opt_aligned
    (msssim/lpips/ffl/per_band, cfg weights). == exactly what the committed config does.

  Variant 2 (+ registration term): Variant 1 PLUS a new direct registration loss
    = 1 - LNCC(grad_mag(despeckle(sar)), grad_mag(opt_aligned)), weight 5.0. A strong,
    G-INDEPENDENT, modality-robust gradient on phi (online version of align_pairs.py's
    gradient-NCC registration).

NOT a training run. 200 aligner-only steps on 3 fixed batches (G frozen) — seconds.
The registration loss lives ONLY here (no config / model edits).

Run from repo root::

    python -m src.models.llwt_v5.proof_registration
"""
from __future__ import annotations

import sys
from pathlib import Path

# Windows console defaults to cp1252 and chokes on em-dash / unicode in banners.
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT))

from omegaconf import OmegaConf  # noqa: E402

from src.models.llwt_v5.main import LLWv4LightningModule  # noqa: E402
from src.models.llwt_v5.align import DeformationAligner, psc_detect, gaussian_blur  # noqa: E402
from src.models.llwt_v5.losses import DeformationRegLoss  # noqa: E402
from src.data.sen12_full.datamodule import SEN12FullDataModule  # noqa: E402


CONFIG_PATH = str(ROOT / 'src' / 'models' / 'llwt_v5' / 'config.yaml')
# v0.4.6 warm ckpt (fresh aligner — reproduces the training START), NOT the failed v0.5.0.
CKPT_PATH = str(ROOT / 'checkpoints' / 'llwt_v45' / 'llwt-v0.4.6-align-ws' / 'last.ckpt')

STEPS = 200
BATCH_SIZE = 4
N_BATCHES = 3
LR = 1e-3
REG_SMOOTH = 3.0          # loosened (matches committed config)
REG_MAG = 0.5
REG_LOSS_WEIGHT = 5.0     # Variant 2 registration-term weight
LNCC_WIN = 9
DESPECKLE_SIGMA = 1.0
PHI_ENGAGE_PX = 0.8       # decision threshold toward the 1.23px median misalignment


def banner(title: str) -> None:
    print('\n' + '=' * 84)
    print(title)
    print('=' * 84)


# ----------------------------------------------------------------- fp32 helpers


def grad_mag(x: torch.Tensor) -> torch.Tensor:
    """Per-channel finite-difference gradient magnitude, mean over channels -> (B,1,H,W)."""
    x = x.float()
    gx = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs()
    gy = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs()
    gx = F.pad(gx, (0, 1, 0, 0))
    gy = F.pad(gy, (0, 0, 0, 1))
    return (gx + gy).mean(dim=1, keepdim=True)


def despeckle(sar: torch.Tensor) -> torch.Tensor:
    """Cheap despeckle proxy: gaussian blur (sigma 1.0), reusing align.gaussian_blur."""
    return gaussian_blur(sar.float(), sigma=DESPECKLE_SIGMA)


def lncc(a: torch.Tensor, b: torch.Tensor, win: int = LNCC_WIN) -> torch.Tensor:
    """Differentiable local normalized cross-correlation over win x win windows.

    Returns a scalar mean (over pixels and batch) in ~[-1, 1]. Uses avg_pool2d for
    local means/vars (standard LNCC = mean of cov / (sqrt(var_a * var_b) + eps))."""
    a = a.float()
    b = b.float()
    pad = win // 2
    eps = 1e-4   # variance floor: keeps sqrt's 1/(2*sqrt(x)) gradient finite on flat patches

    def local_mean(t):
        return F.avg_pool2d(t, kernel_size=win, stride=1, padding=pad)

    mu_a = local_mean(a)
    mu_b = local_mean(b)
    mu_aa = local_mean(a * a)
    mu_bb = local_mean(b * b)
    mu_ab = local_mean(a * b)
    # eps INSIDE the sqrt (not added after) so the sqrt argument never reaches 0 —
    # sqrt'(0)=inf was the NaN source on flat optical patches (var->0).
    var_a = (mu_aa - mu_a * mu_a).clamp_min(0.0)
    var_b = (mu_bb - mu_b * mu_b).clamp_min(0.0)
    cov = mu_ab - mu_a * mu_b
    cc = cov / torch.sqrt(var_a * var_b + eps)
    return cc.mean()


def registration_loss(sar: torch.Tensor, opt_aligned: torch.Tensor) -> torch.Tensor:
    """1 - LNCC(grad_mag(despeckle(sar)), grad_mag(opt_aligned)). Differentiable in opt_aligned."""
    return 1.0 - lncc(grad_mag(despeckle(sar)), grad_mag(opt_aligned))


def reg_ncc(sar: torch.Tensor, img: torch.Tensor) -> float:
    """Registration-quality METRIC (no grad): gradient-domain LNCC between SAR and img."""
    with torch.no_grad():
        return float(lncc(grad_mag(despeckle(sar)), grad_mag(img)).item())


def fresh_aligner(cfg, device) -> DeformationAligner:
    """Re-init a zero-init DeformationAligner (phi=0 at step 0) — identical start per variant."""
    ac = cfg.align
    aligner = DeformationAligner(
        ll_channels=3,
        max_disp_px=float(getattr(ac, 'max_disp_px', 8.0)),
        hidden=int(getattr(ac, 'phi_hidden', 64)),
    ).to(device)
    return aligner


def fidelity_loss(module, fake, predicted_sub, opt_aligned, loss_cfg):
    """Sum of fidelity losses routed through opt_aligned — mirrors main.py weights exactly."""
    crit = module.criterions
    total = fake.new_zeros(())
    if 'msssim' in crit:
        total = total + crit['msssim'](fake, opt_aligned) * float(loss_cfg.msssim_weight)
    if 'per_band' in crit and predicted_sub is not None:
        total = total + crit['per_band'](predicted_sub, opt_aligned) * float(loss_cfg.per_band_weight)
    if 'lpips' in crit:
        total = total + crit['lpips'](fake, opt_aligned) * float(loss_cfg.lpips_weight)
    if 'ffl' in crit:
        total = total + crit['ffl'](fake, opt_aligned) * float(loss_cfg.ffl_weight)
    return total


# ---------------------------------------------------------------- variant runner


def run_variant(name, with_reg_term, module, batches, fakes, predicted_subs,
                deform_reg, psc_anchor, psc_anchor_weight, loss_cfg, cfg, device):
    """200 aligner-only steps cycling the fixed batches. G frozen (precomputed fakes).

    Returns dict with final phi mean/max and final reg_ncc (raw opt vs warped opt_aligned)."""
    banner(f"[proof] VARIANT: {name}  (registration_term={'ON' if with_reg_term else 'OFF'})")
    torch.manual_seed(0)
    aligner = fresh_aligner(cfg, device)
    aligner.train()
    optim = torch.optim.Adam(aligner.parameters(), lr=LR)
    haar_ll = module._haar_ll
    psc_topk = int(getattr(cfg.align, 'psc_topk', 64))

    # constant raw reg_ncc reference per batch (opt is unwarped; same every step)
    raw_regncc = [reg_ncc(s, o) for (s, o) in batches]

    last = {}
    for step in range(1, STEPS + 1):
        bi = (step - 1) % len(batches)
        sar, opt = batches[bi]
        fake = fakes[bi]
        predicted_sub = predicted_subs[bi]

        with torch.autocast('cuda', enabled=False):
            fake_ll = haar_ll(fake)[0]
            opt_ll = haar_ll(opt)[0]
            phi = aligner(fake_ll, opt_ll)
            opt_aligned = DeformationAligner.warp(opt, phi)
            m_psc = psc_detect(sar, topk=psc_topk)

            l_fid = fidelity_loss(module, fake, predicted_sub, opt_aligned, loss_cfg)
            l_psc = psc_anchor(fake, opt_aligned, m_psc) * psc_anchor_weight
            l_reg = deform_reg(phi)
            loss = l_fid + l_psc + l_reg
            if with_reg_term:
                l_regterm = registration_loss(sar, opt_aligned) * REG_LOSS_WEIGHT
                loss = loss + l_regterm

        optim.zero_grad(set_to_none=True)
        loss.backward()
        if float(loss_cfg.grad_clip_g) > 0:
            torch.nn.utils.clip_grad_norm_(aligner.parameters(), float(loss_cfg.grad_clip_g))
        optim.step()

        if step == 1 or step % 25 == 0 or step == STEPS:
            with torch.no_grad():
                phi_mean = float(phi.abs().mean().item())
                phi_max = float(phi.abs().max().item())
                ncc_raw = raw_regncc[bi]
                ncc_aln = reg_ncc(sar, opt_aligned)
            print(f"  step {step:3d} (b{bi}) | phi.mean={phi_mean:.4f}px phi.max={phi_max:.4f}px "
                  f"| regncc raw={ncc_raw:+.4f} aligned={ncc_aln:+.4f} (d={ncc_aln - ncc_raw:+.4f}) "
                  f"| fid={float(l_fid):.3f} psc={float(l_psc):.3f} reg={float(l_reg):.4f}"
                  + (f" regterm={float(l_regterm):.4f}" if with_reg_term else ""))
            last = dict(phi_mean=phi_mean, phi_max=phi_max,
                        ncc_raw_mean=sum(raw_regncc) / len(raw_regncc))

    # final stats averaged over ALL batches at the trained aligner (G frozen)
    phi_means, phi_maxs, raw_nccs, aln_nccs = [], [], [], []
    with torch.no_grad(), torch.autocast('cuda', enabled=False):
        for bi, (sar, opt) in enumerate(batches):
            fake = fakes[bi]
            fake_ll = haar_ll(fake)[0]
            opt_ll = haar_ll(opt)[0]
            phi = aligner(fake_ll, opt_ll)
            opt_aligned = DeformationAligner.warp(opt, phi)
            phi_means.append(float(phi.abs().mean().item()))
            phi_maxs.append(float(phi.abs().max().item()))
            raw_nccs.append(reg_ncc(sar, opt))
            aln_nccs.append(reg_ncc(sar, opt_aligned))
    last['phi_mean'] = sum(phi_means) / len(phi_means)
    last['phi_max'] = max(phi_maxs)
    last['regncc_raw'] = sum(raw_nccs) / len(raw_nccs)
    last['regncc_aligned'] = sum(aln_nccs) / len(aln_nccs)
    last['regncc_delta'] = last['regncc_aligned'] - last['regncc_raw']
    last['engaged'] = last['phi_mean'] >= PHI_ENGAGE_PX
    print(f"  [final] phi.mean={last['phi_mean']:.4f}px phi.max={last['phi_max']:.4f}px "
          f"regncc raw={last['regncc_raw']:+.4f} aligned={last['regncc_aligned']:+.4f} "
          f"d={last['regncc_delta']:+.4f}  engaged(>={PHI_ENGAGE_PX}px)={last['engaged']}")
    return last


def main():
    torch.manual_seed(0)
    assert torch.cuda.is_available(), "proof needs CUDA"
    device = torch.device('cuda')
    cfg = OmegaConf.load(CONFIG_PATH)
    cfg.align.enabled = True
    cfg.system.device = 'cuda'
    loss_cfg = cfg.loss

    banner('[proof] SETUP — build module + load WARM v0.4.6 ckpt (fresh aligner)')
    module = LLWv4LightningModule(cfg).to(device)
    module.eval()
    assert module.use_align, "align not enabled in module"
    sd = torch.load(CKPT_PATH, map_location='cpu', weights_only=False)['state_dict']
    missing, unexpected = module.load_state_dict(sd, strict=False)
    n_align_in_ckpt = sum(1 for k in sd if k.startswith('aligner.'))
    n_g_missing = sum(1 for k in missing if k.startswith('netG.'))
    print(f"  ckpt = {Path(CKPT_PATH).parent.name}/{Path(CKPT_PATH).name}")
    print(f"  ckpt state_dict keys     = {len(sd)} (aligner.* in ckpt = {n_align_in_ckpt})")
    print(f"  load missing total       = {len(missing)} (netG.* missing={n_g_missing})")
    print(f"  load unexpected total    = {len(unexpected)}")
    # confirm the module aligner is fresh/zero-init (v0.4.6 has no trained aligner)
    w = module.aligner.net[-1].weight.detach()
    print(f"  module aligner final-conv W abs.max = {w.abs().max().item():.3e} "
          f"(0 => fresh zero-init, the training start)")

    # FREEZE netG so only the aligner moves.
    for p in module.netG.parameters():
        p.requires_grad_(False)
    module.netG.eval()

    banner(f'[proof] DATA — {N_BATCHES} fixed RAW sen12_full batches (bs={BATCH_SIZE})')
    dm = SEN12FullDataModule(
        data_dir=str(ROOT / 'data' / 'sen12_full'),
        batch_size=BATCH_SIZE,
        image_size=int(cfg.data.image_size),
        num_workers=0,
        persistent_workers=False,
        prefetch_factor=2,
        train_val_split_ratio=float(cfg.data.train_val_split_ratio),
        seed=int(cfg.data.seed),
        sar_channels=int(cfg.data.sar_channels),
        use_augmentation=False,
        scenes=list(cfg.data.scenes),
    )
    dm.setup('fit')
    it = iter(dm.train_dataloader())
    batches = [next(it) for _ in range(N_BATCHES)]
    batches = [(s.to(device), o.to(device)) for (s, o) in batches]
    s0, o0 = batches[0]
    print(f"  batch0: sar={tuple(s0.shape)} opt={tuple(o0.shape)} "
          f"sar[{s0.min():.2f},{s0.max():.2f}] opt[{o0.min():.2f},{o0.max():.2f}]")
    # baseline raw SAR<->opt registration (constant reference)
    for bi, (s, o) in enumerate(batches):
        print(f"  batch{bi} raw reg_ncc(sar,opt) = {reg_ncc(s, o):+.4f}")

    # Precompute frozen-G fakes (+ predicted_sub) ONCE — fair, identical across variants.
    banner('[proof] precompute frozen-G outputs (no_grad)')
    need_internals = ('per_band' in module.criterions)
    fakes, predicted_subs = [], []
    with torch.no_grad(), torch.autocast('cuda', enabled=False):
        for (sar, opt) in batches:
            if need_internals:
                fake, predicted_sub, _ = module.netG(sar, return_internals=True)
                predicted_subs.append(predicted_sub.detach())
            else:
                fake = module.netG(sar)
                predicted_subs.append(None)
            fakes.append(fake.detach())
    print(f"  need_internals={need_internals}  fakes={len(fakes)}  "
          f"fake0={tuple(fakes[0].shape)}")
    fid_keys = [k for k in ('msssim', 'per_band', 'lpips', 'ffl') if k in module.criterions]
    print(f"  fidelity criterions routed through opt_aligned = {fid_keys}")

    # Shared loss objects: LOOSENED reg (3 / 0.5) + the config psc_anchor.
    deform_reg = DeformationRegLoss(smooth_weight=REG_SMOOTH, mag_weight=REG_MAG).to(device)
    psc_anchor = module.align_criterions['psc_anchor']
    psc_anchor_weight = float(getattr(cfg.align, 'psc_anchor_weight', 1.0))
    print(f"  reg = DeformationRegLoss(smooth={REG_SMOOTH}, mag={REG_MAG})  "
          f"psc_anchor_weight={psc_anchor_weight}  reg_term_weight(V2)={REG_LOSS_WEIGHT}")

    # --------------------------------------------------------------- run both
    v1 = run_variant('1 reg-loosen ONLY', False, module, batches, fakes, predicted_subs,
                     deform_reg, psc_anchor, psc_anchor_weight, loss_cfg, cfg, device)
    v2 = run_variant('2 reg-loosen + REGISTRATION term', True, module, batches, fakes, predicted_subs,
                     deform_reg, psc_anchor, psc_anchor_weight, loss_cfg, cfg, device)

    # ----------------------------------------------------------------- compare
    banner('[proof] COMPARISON TABLE')
    hdr = f"  {'variant':<34} {'phi.mean':>9} {'phi.max':>9} {'regncc d':>10} {'engaged>=0.8px':>14}"
    print(hdr)
    print('  ' + '-' * (len(hdr) - 2))
    for label, v in (('V1 reg-loosen only', v1), ('V2 + registration term', v2)):
        print(f"  {label:<34} {v['phi_mean']:>8.4f}px {v['phi_max']:>8.4f}px "
              f"{v['regncc_delta']:>+10.4f} {str(v['engaged']):>14}")

    # ----------------------------------------------------------------- verdict
    banner('[proof] VERDICT')
    v1_ok = v1['phi_mean'] >= PHI_ENGAGE_PX and v1['regncc_delta'] > 0
    v2_better = v2['phi_mean'] >= 1.0 and v2['regncc_delta'] > 0
    if v1_ok:
        fix = 'reg-only'
    elif (v1['phi_mean'] < 0.5) and v2_better:
        fix = 'needs-registration-term'
    else:
        fix = 'inconclusive (see trajectories)'
    print(f"[proof] Variant1 phi={v1['phi_mean']:.4f} px (regncc d={v1['regncc_delta']:+.4f}), "
          f"Variant2 phi={v2['phi_mean']:.4f} px (regncc d={v2['regncc_delta']:+.4f}) "
          f"-> FIX = {fix}")


if __name__ == '__main__':
    main()
