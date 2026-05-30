"""DECISIVE proof: does pixel-L1 on detail bands REWARD blur, and does the
Sliced-Wasserstein detail loss REWARD sharpness instead?

Throwaway experiment (NOT committed, NOT wired into training).

Run from repo root::

    python -m src.models.llwt_v5.proof_swd_blur

Setup
-----
Take ~32 ALIGNED optical patches from ``sen12_full_align``. For each patch the
supervision TARGET is the aligned GT optical (unshifted). We construct two
candidate "predictions" of that target:

  * blurry-aligned  : gaussian-blur(GT, sigma=1.5)  -> the model's hedged blur,
                      CORRECT position.
  * sharp-misaligned: GT subpixel-shifted by 1.5 px  -> a SHARP, correct-content
                      prediction that is merely misregistered (the thing we WANT
                      the model to be free to produce under ~1.2 px misalignment).

We then score each candidate against the unshifted GT using:

  * per-band L1 on detail subbands (LH,HL,HH)  — the current supervision.
  * WaveletDetailSWDLoss                         — the proposed distributional loss.

Both losses take ``(sub_pred, opt)`` where ``sub_pred`` is the candidate's Haar
subband tensor ``(B,C,4,H/2,W/2)`` and ``opt`` is the GT image — identical to
how ``main.py`` calls them (``predicted_sub`` vs ``HaarDown(opt_aligned)``).

Verdict logic
-------------
  * L1_detail(blurry) < L1_detail(sharp-misaligned)  -> L1 PREFERS BLUR (wrong).
  * SWD(sharp-misaligned) < SWD(blurry)              -> SWD PREFERS SHARP (right).
If both hold, the Frequency-Faithful Supervision (FFS) idea is JUSTIFIED.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from omegaconf import OmegaConf

from src.models.llwt_v5.losses import PerBandWaveletL1Loss, WaveletDetailSWDLoss
from src.models.llwt_v5.blocks import HaarDown
from src.models.llwt_v5.align import gaussian_blur


N_PATCHES = 32
SIGMA = 1.5            # blur strength of the hedged-blur candidate
SHIFT_PX = 1.5         # subpixel misregistration of the sharp candidate


def _load_optical_patches(n: int) -> torch.Tensor:
    """Grab ~n aligned optical patches as a (n,3,H,W) tensor in [-1,1]-ish norm."""
    cfg = OmegaConf.load('src/models/llwt_v5/config.yaml')
    from src.data.sen12_full_align.datamodule import SEN12FullDataModule

    dc = cfg.data
    dm = SEN12FullDataModule(
        data_dir=dc.data_dir['sen12_full_align'],
        batch_size=n,
        image_size=int(dc.image_size),
        num_workers=0,
        persistent_workers=False,
        sar_channels=int(dc.sar_channels),
        use_augmentation=False,             # deterministic patches
        scenes=list(dc.scenes),
    )
    dm.setup('fit')
    loader = dm.val_dataloader()
    _sar, opt = next(iter(loader))
    return opt[:n].float()


def _subpixel_shift(x: torch.Tensor, dx: float, dy: float) -> torch.Tensor:
    """Shift x by (dx,dy) pixels via bilinear grid_sample (border-padded)."""
    B, C, H, W = x.shape
    ys, xs = torch.meshgrid(
        torch.linspace(-1, 1, H, device=x.device),
        torch.linspace(-1, 1, W, device=x.device),
        indexing='ij',
    )
    # convert pixel shift to normalized-coord shift (grid range is 2 over the dim)
    gx = xs + (2.0 * dx / max(W - 1, 1))
    gy = ys + (2.0 * dy / max(H - 1, 1))
    grid = torch.stack([gx, gy], dim=-1).unsqueeze(0).expand(B, H, W, 2)
    return F.grid_sample(x, grid, mode='bilinear', padding_mode='border', align_corners=True)


def _detail_l1(sub_pred: torch.Tensor, opt: torch.Tensor) -> float:
    """Per-band L1 restricted to the detail bands (LH,HL,HH) — matches what the
    blur-incentive argument is about (LL excluded, supervised by pixel L1)."""
    loss = PerBandWaveletL1Loss(band_weights={'ll': 0.0, 'lh': 1.0, 'hl': 1.0, 'hh': 1.0})
    return float(loss(sub_pred, opt).item())


def _mid_freq_power_ratio(img: torch.Tensor, ref: torch.Tensor) -> float:
    """Ratio of detail-band (LH+HL+HH) energy: img vs ref. <1 => img lower-freq."""
    haar = HaarDown(in_channels=3)
    _ll_i, lh_i, hl_i, hh_i = haar(img)
    _ll_r, lh_r, hl_r, hh_r = haar(ref)
    e_img = sum(t.pow(2).mean() for t in (lh_i, hl_i, hh_i))
    e_ref = sum(t.pow(2).mean() for t in (lh_r, hl_r, hh_r))
    return float((e_img / e_ref.clamp_min(1e-12)).item())


def main() -> None:
    torch.manual_seed(0)
    opt = _load_optical_patches(N_PATCHES)             # (N,3,H,W) GT target
    print(f"[swd-proof] loaded {opt.shape[0]} aligned optical patches, shape={tuple(opt.shape)}, "
          f"range=[{opt.min():.2f}, {opt.max():.2f}]")

    haar = HaarDown(in_channels=3)

    # ---- candidates ------------------------------------------------------
    blurry = gaussian_blur(opt, sigma=SIGMA)                       # hedged blur, aligned
    sharp_mis = _subpixel_shift(opt, dx=SHIFT_PX, dy=SHIFT_PX)     # sharp, misregistered

    # predicted-subband tensors (B,3,4,H/2,W/2) — same shape main.py feeds.
    def _subs(x: torch.Tensor) -> torch.Tensor:
        ll, lh, hl, hh = haar(x)
        return torch.stack([ll, lh, hl, hh], dim=2)

    sub_blurry = _subs(blurry)
    sub_sharp = _subs(sharp_mis)

    # ---- L1 on detail bands ---------------------------------------------
    l1_blurry = _detail_l1(sub_blurry, opt)
    l1_sharp = _detail_l1(sub_sharp, opt)

    # ---- Sliced-Wasserstein detail --------------------------------------
    swd = WaveletDetailSWDLoss()
    swd_blurry = float(swd(sub_blurry, opt).item())
    swd_sharp = float(swd(sub_sharp, opt).item())

    # ---- secondary: mid-freq power ratio (confirm blurry is lower-freq) --
    pr_blurry = _mid_freq_power_ratio(blurry, opt)
    pr_sharp = _mid_freq_power_ratio(sharp_mis, opt)

    # ---- report ----------------------------------------------------------
    print("\n[swd-proof] ===================== RESULTS =====================")
    print(f"  L1_detail(blurry-aligned,   GT) = {l1_blurry:.6f}")
    print(f"  L1_detail(sharp-misaligned, GT) = {l1_sharp:.6f}")
    print(f"  SWD_detail(blurry-aligned,   GT) = {swd_blurry:.6f}")
    print(f"  SWD_detail(sharp-misaligned, GT) = {swd_sharp:.6f}")
    print(f"  mid-freq power ratio: blurry/GT = {pr_blurry:.3f}, sharp-mis/GT = {pr_sharp:.3f} "
          f"(blurry should be <1, sharp ~1)")

    l1_prefers = 'blur' if l1_blurry < l1_sharp else 'sharp'
    swd_prefers = 'sharp' if swd_sharp < swd_blurry else 'blur'
    print(f"\n  L1  ranking : prefers {l1_prefers}  "
          f"({'blurry' if l1_prefers=='blur' else 'sharp'} has the LOWER L1)")
    print(f"  SWD ranking : prefers {swd_prefers}  "
          f"({'sharp' if swd_prefers=='sharp' else 'blurry'} has the LOWER SWD)")

    justified = (l1_prefers == 'blur') and (swd_prefers == 'sharp')
    verdict = 'JUSTIFIED' if justified else 'REJECTED'
    print(f"\n[swd-proof] L1 prefers={l1_prefers}, SWD prefers={swd_prefers} -> FFS {verdict}")


if __name__ == '__main__':
    main()
