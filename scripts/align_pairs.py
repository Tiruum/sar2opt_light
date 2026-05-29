"""SAR<->optical pair alignment for SEN1-2 (sen12_full -> sen12_full_align).

Builds an aligned mirror of ``data/sen12_full`` at ``data/sen12_full_align`` with
the **identical** directory layout (``<season>/s1_<scene>/`` + ``s2_<scene>/``).
The SAR image is the reference frame and is copied byte-for-byte unchanged; the
paired OPTICAL image is warped (Euclidean: translation + rotation) into the SAR
frame so the pair is co-registered.

Why this method (grounded in SAR<->optical registration literature)
-------------------------------------------------------------------
Raw-intensity similarity (NCC / MI on pixel values) is unreliable across SAR and
optical because the two sensors have fundamentally different imaging mechanisms
(radar backscatter vs reflectance) -- the same scene has *inverted* or unrelated
intensities.  What the two modalities DO share is **structure / edges**.  So both
the registration objective AND the quality metrics operate on *gradient-structure*
images, not raw intensity:

  1. SAR is despeckled (log-domain adaptive Lee, mirroring the model's
     ``blocks.lee_despeckle``) then reduced to a Sobel gradient-magnitude map.
  2. Optical is converted to gray then to a Sobel gradient-magnitude map.
  3. Coarse sub-pixel translation is estimated with ``cv2.phaseCorrelate`` on the
     (Hann-windowed) gradient maps.
  4. A Euclidean warp (tx, ty, theta) is refined with ``cv2.findTransformECC``
     (``MOTION_EUCLIDEAN``) on the gradient maps -- ECC maximises the enhanced
     correlation coefficient, robust to the residual radiometric difference left
     in the gradient domain.
  5. The best candidate warp (ECC vs phase-translation vs identity) is selected by
     post-warp gradient-NCC, then GATED: a pair is only re-written if alignment
     *improves* gradient-NCC and the estimated shift/rotation stays within sane
     bounds.  Featureless patches (flat SAR gradient -> registration undefined)
     and any pair the warp would not improve are copied through unchanged.  The
     pipeline can therefore never make a pair worse than it started.

References (see also the run header printed at start):
  * Phase-congruency / gradient structural descriptors (HOPC, CFOG) for
    multimodal optical-SAR registration -- Ye et al.
  * SAR-SIFT: a speckle-robust gradient definition for SAR.
  * ``phaseCorrelate`` + ECC = the standard coarse->fine OpenCV registration path.

Proof-of-improvement metrics (no ground-truth registration available)
---------------------------------------------------------------------
For every pair we log gradient-domain **NCC** and **Mutual Information** BEFORE and
AFTER alignment.  The aggregate report shows mean dNCC, mean dMI, %improved and
%accepted -- a positive mean delta is the evidence that alignment made the pairs
measurably better.  Checkerboard QA overlays (SAR-gradient vs warped-optical
gradient) are written for a sample so the improvement can be eyeballed.

Outputs (``./output/align_pairs/<tag>/``):
  report.csv  per-pair: scene, file, tx, ty, theta_deg, ncc_before/after,
              mi_before/after, accepted, method
  report.txt  aggregate summary (the headline before/after numbers)
  qa/*.png    [before | after] checkerboard overlays for a sample of pairs

Run from repo root::

    python scripts/align_pairs.py --limit 200 --no-write   # fast dry-run, metrics only
    python scripts/align_pairs.py                           # full run -> writes mirror
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from src.data.sen12_full.dataset import SEN12Full  # reuse the exact pairing logic

CONFIG_PATH = 'src/models/llwt_v45/config.yaml'

_EPS = 1e-6


# --------------------------------------------------------------------------- #
# Structure extraction
# --------------------------------------------------------------------------- #
def lee_despeckle_np(x: np.ndarray, win: int = 5, q: float = 0.05) -> np.ndarray:
    """Adaptive Lee despeckle in the log/dB domain (additive-noise form).

    Numpy/cv2 port of ``src/models/llwt_v45/blocks.lee_despeckle`` so the
    registration sees the same edge-preserving denoise the model's SARAdapter
    uses.  SEN1-2 SAR PNGs are already dB/log-scaled, so multiplicative speckle
    behaves as ~additive constant-variance noise here:

        out = mean + w * (x - mean),  w = clamp(1 - sigma_n2 / var_local, 0, 1)

    ``sigma_n2`` is estimated per-image as the low quantile of the local-variance
    map (homogeneous patches ~= pure speckle).  No params, no dataset tuning.
    """
    x = x.astype(np.float32, copy=False)
    k = (1.0 / (win * win)) * np.ones((win, win), np.float32)
    mean = cv2.filter2D(x, -1, k, borderType=cv2.BORDER_REFLECT)
    mean_sq = cv2.filter2D(x * x, -1, k, borderType=cv2.BORDER_REFLECT)
    var = np.maximum(mean_sq - mean * mean, _EPS)
    sigma_n2 = float(np.quantile(var, q))
    w = np.clip(1.0 - sigma_n2 / var, 0.0, 1.0)
    return mean + w * (x - mean)


def grad_mag(gray: np.ndarray) -> np.ndarray:
    """Normalised Sobel gradient-magnitude map (float32, ~[0,1])."""
    gray = gray.astype(np.float32, copy=False)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    m = np.sqrt(gx * gx + gy * gy)
    return m / (float(m.max()) + _EPS)


# --------------------------------------------------------------------------- #
# Cross-modal QA metrics (computed on gradient-structure images)
# --------------------------------------------------------------------------- #
def ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Zero-mean normalised cross-correlation in [-1, 1]."""
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = math.sqrt(float((a * a).sum()) * float((b * b).sum())) + 1e-12
    return float((a * b).sum() / denom)


def mutual_information(a: np.ndarray, b: np.ndarray, bins: int = 32) -> float:
    """MI (nats) from the joint histogram of two [0,1] gradient images."""
    hist, _, _ = np.histogram2d(
        a.ravel(), b.ravel(), bins=bins, range=[[0.0, 1.0], [0.0, 1.0]]
    )
    pab = hist / (hist.sum() + _EPS)
    pa = pab.sum(axis=1, keepdims=True)
    pb = pab.sum(axis=0, keepdims=True)
    nz = pab > 0
    denom = (pa * pb)[nz]
    return float((pab[nz] * np.log((pab[nz] + _EPS) / (denom + _EPS))).sum())


# --------------------------------------------------------------------------- #
# Alignment
# --------------------------------------------------------------------------- #
def _warp(img: np.ndarray, M: np.ndarray, size_wh) -> np.ndarray:
    # ECC's matrix and our translation matrices are both used with
    # WARP_INVERSE_MAP (matrix maps dst->src), per OpenCV's findTransformECC
    # usage contract.  REFLECT border avoids black wedges that would poison
    # the GAN at the patch edges.
    return cv2.warpAffine(
        img, M, size_wh,
        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REFLECT,
    )


def estimate_warp(sar_g, sar_gs, opt_g, opt_gs, hann, criteria):
    """Return (best_M 2x3 float32, method, warped_opt_grad) maximising grad-NCC.

    Candidates: identity, ECC-euclidean-from-identity, phase-correlation
    translation (both sign conventions).  Sign ambiguity in phaseCorrelate is
    resolved by simply scoring every candidate with grad-NCC and keeping the
    best -- so a wrong guess can never win.
    """
    H, W = sar_g.shape
    size_wh = (W, H)
    eye = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)

    candidates = [('identity', eye.copy(), opt_g)]

    # ECC Euclidean refine from identity.
    try:
        warp = eye.copy()
        cv2.findTransformECC(sar_gs, opt_gs, warp, cv2.MOTION_EUCLIDEAN,
                             criteria, None, 5)
        candidates.append(('ecc', warp, _warp(opt_g, warp, size_wh)))
    except cv2.error:
        pass

    # Coarse phase-correlation translation (try both sign conventions).
    try:
        (dx, dy), _resp = cv2.phaseCorrelate(sar_gs, opt_gs, hann)
        for sgn, tag in ((1.0, 'phase'), (-1.0, 'phase_neg')):
            Mt = np.array([[1, 0, sgn * dx], [0, 1, sgn * dy]], dtype=np.float32)
            candidates.append((tag, Mt, _warp(opt_g, Mt, size_wh)))
    except cv2.error:
        pass

    best = max(candidates, key=lambda c: ncc(sar_g, c[2]))
    return best


def decompose(M: np.ndarray):
    """Return (tx, ty, theta_deg) from a 2x3 Euclidean/affine matrix."""
    tx = float(M[0, 2])
    ty = float(M[1, 2])
    theta = math.degrees(math.atan2(float(M[1, 0]), float(M[0, 0])))
    return tx, ty, theta


def checkerboard(a: np.ndarray, b: np.ndarray, tile: int = 32) -> np.ndarray:
    """Alternate tiles of a/b (uint8 gray) so misalignment breaks edges."""
    H, W = a.shape
    yy = (np.arange(H)[:, None] // tile)
    xx = (np.arange(W)[None, :] // tile)
    mask = ((yy + xx) % 2).astype(bool)
    return np.where(mask, a, b).astype(np.uint8)


def _u8(g: np.ndarray) -> np.ndarray:
    g = g - g.min()
    g = g / (g.max() + _EPS)
    return (g * 255.0).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def main():
    cfg = OmegaConf.load(CONFIG_PATH)

    ap = argparse.ArgumentParser(description="SAR<->optical pair alignment (sen12_full -> sen12_full_align)")
    ap.add_argument('--src', default=None, help='source root (default: cfg.data.data_dir.sen12_full)')
    ap.add_argument('--dst', default='./data/sen12_full_align', help='aligned mirror root')
    ap.add_argument('--scenes', default=None,
                    help='comma-list of scene ids, or "all". default: cfg.data.scenes')
    ap.add_argument('--limit', type=int, default=0, help='process only first N pairs (0=all)')
    ap.add_argument('--no-write', action='store_true', help='metrics + QA only, do not write mirror')
    ap.add_argument('--out', default='./output/align_pairs', help='report/QA output dir')
    ap.add_argument('--tag', default='run', help='sub-folder under --out')
    ap.add_argument('--qa-count', type=int, default=24, help='checkerboard overlays to save')
    # tuning knobs (defaults sane for 256px SEN1-2 patches)
    ap.add_argument('--shift-max', type=float, default=48.0, help='max |translation| px to accept')
    ap.add_argument('--theta-max', type=float, default=8.0, help='max |rotation| deg to accept')
    ap.add_argument('--ncc-eps', type=float, default=1e-3, help='min grad-NCC gain to accept a warp')
    ap.add_argument('--featureless-std', type=float, default=5e-3,
                    help='skip (identity) if SAR gradient std below this')
    ap.add_argument('--lee-win', type=int, default=5, help='Lee despeckle window')
    args = ap.parse_args()

    src_root = args.src or str(cfg.data.data_dir.sen12_full)
    dst_root = args.dst
    if args.scenes is None:
        scenes = list(cfg.data.scenes)
    elif args.scenes.strip().lower() == 'all':
        scenes = None
    else:
        scenes = [s.strip() for s in args.scenes.split(',') if s.strip()]

    out_dir = Path(args.out) / args.tag
    qa_dir = out_dir / 'qa'
    out_dir.mkdir(parents=True, exist_ok=True)
    qa_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("SAR<->optical pair alignment  (Euclidean: translation + rotation)")
    print("  register + score on GRADIENT-STRUCTURE images (cross-modal intensity")
    print("  NCC/MI is unreliable; SAR & optical share edges, not intensities).")
    print("  coarse phaseCorrelate -> ECC(MOTION_EUCLIDEAN) refine; gated by grad-NCC gain.")
    print("=" * 78)
    print(f"src       : {src_root}")
    print(f"dst       : {dst_root}  ({'DRY-RUN, not writing' if args.no_write else 'WRITING mirror'})")
    print(f"scenes    : {scenes if scenes is not None else 'ALL'}")
    print(f"gates     : |shift|<={args.shift_max}px  |theta|<={args.theta_max}deg  dNCC>{args.ncc_eps}")
    print(f"reports   : {out_dir}")
    print("-" * 78)

    if not os.path.isdir(src_root):
        raise FileNotFoundError(f"source root not found: {src_root}")

    # Reuse the dataset's exact pairing logic (s1->s2 filename substitution).
    ds = SEN12Full(root_dir=src_root, sar_channels=1, scenes=scenes)
    items = ds.items
    if args.limit > 0:
        items = items[:args.limit]
    n = len(items)
    if n == 0:
        raise RuntimeError(f"0 pairs found in {src_root} (scenes={scenes}).")
    print(f"pairs     : {n}")

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-4)

    rows = []
    qa_saved = 0
    hann = None
    t0 = time.perf_counter()

    for i, (season, s1_d, s2_d, s1_fname, s2_fname) in enumerate(items):
        sar_path = os.path.join(src_root, season, s1_d, s1_fname)
        opt_path = os.path.join(src_root, season, s2_d, s2_fname)

        sar_gray = cv2.imread(sar_path, cv2.IMREAD_GRAYSCALE)
        opt_bgr = cv2.imread(opt_path, cv2.IMREAD_COLOR)
        if sar_gray is None or opt_bgr is None:
            print(f"  !! unreadable, skipping: {s1_fname}")
            continue
        H, W = sar_gray.shape
        opt_gray = cv2.cvtColor(opt_bgr, cv2.COLOR_BGR2GRAY)

        # Structure maps.
        sar_ds = lee_despeckle_np(sar_gray.astype(np.float32), win=args.lee_win)
        sar_g = grad_mag(sar_ds)
        opt_g = grad_mag(opt_gray)
        sar_gs = cv2.GaussianBlur(sar_g, (5, 5), 0)
        opt_gs = cv2.GaussianBlur(opt_g, (5, 5), 0)

        if hann is None or hann.shape != (H, W):
            hann = cv2.createHanningWindow((W, H), cv2.CV_32F)

        ncc_before = ncc(sar_g, opt_g)
        mi_before = mutual_information(sar_g, opt_g)

        featureless = float(sar_g.std()) < args.featureless_std

        method = 'identity'
        M = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        warped_g = opt_g
        if not featureless:
            method, M, warped_g = estimate_warp(sar_g, sar_gs, opt_g, opt_gs, hann, criteria)

        tx, ty, theta = decompose(M)
        ncc_after_cand = ncc(sar_g, warped_g)
        shift_mag = math.hypot(tx, ty)

        accept = (
            (not featureless)
            and method != 'identity'
            and (ncc_after_cand > ncc_before + args.ncc_eps)
            and (shift_mag <= args.shift_max)
            and (abs(theta) <= args.theta_max)
        )

        if accept:
            opt_out_bgr = _warp(opt_bgr, M, (W, H))
            ncc_after = ncc_after_cand
            mi_after = mutual_information(sar_g, warped_g)
        else:
            # Identity: pair copied through unchanged -> metrics == before.
            method = 'identity' if not featureless else 'featureless'
            opt_out_bgr = opt_bgr
            tx = ty = theta = 0.0
            warped_g = opt_g
            ncc_after = ncc_before
            mi_after = mi_before

        # Write mirror (SAR copied unchanged; optical warped or copied).
        if not args.no_write:
            dst_s1_dir = os.path.join(dst_root, season, s1_d)
            dst_s2_dir = os.path.join(dst_root, season, s2_d)
            os.makedirs(dst_s1_dir, exist_ok=True)
            os.makedirs(dst_s2_dir, exist_ok=True)
            shutil.copy2(sar_path, os.path.join(dst_s1_dir, s1_fname))  # SAR: reference, untouched
            dst_opt = os.path.join(dst_s2_dir, s2_fname)
            if accept:
                cv2.imwrite(dst_opt, opt_out_bgr)
            else:
                shutil.copy2(opt_path, dst_opt)  # byte-identical passthrough

        # QA overlay for a sample (prefer accepted pairs so the snap is visible).
        if qa_saved < args.qa_count and (accept or i < args.qa_count):
            before = checkerboard(_u8(sar_g), _u8(opt_g))
            after = checkerboard(_u8(sar_g), _u8(warped_g))
            sep = np.full((H, 4), 255, np.uint8)
            panel = np.hstack([before, sep, after])
            cv2.imwrite(str(qa_dir / f"{season}_{s1_d}_{Path(s1_fname).stem}.png"), panel)
            qa_saved += 1

        rows.append(dict(
            season=season, scene=s1_d, file=s1_fname, method=method,
            tx=round(tx, 3), ty=round(ty, 3), theta_deg=round(theta, 4),
            ncc_before=round(ncc_before, 5), ncc_after=round(ncc_after, 5),
            d_ncc=round(ncc_after - ncc_before, 5),
            mi_before=round(mi_before, 5), mi_after=round(mi_after, 5),
            d_mi=round(mi_after - mi_before, 5),
            accepted=int(accept),
        ))

        if (i + 1) % 200 == 0 or (i + 1) == n:
            rate = (i + 1) / (time.perf_counter() - t0 + _EPS)
            print(f"  [{i+1:>6}/{n}] {rate:5.1f} pairs/s  last={method:11s} dNCC={ncc_after-ncc_before:+.4f}")

    # ----------------------------------------------------------------------- #
    # Reports
    # ----------------------------------------------------------------------- #
    csv_path = out_dir / 'report.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    n_rows = len(rows)
    n_acc = sum(r['accepted'] for r in rows)
    n_imp = sum(1 for r in rows if r['d_ncc'] > 0)
    arr = lambda key: np.array([r[key] for r in rows], dtype=np.float64)
    ncc_b, ncc_a = arr('ncc_before'), arr('ncc_after')
    mi_b, mi_a = arr('mi_before'), arr('mi_after')
    acc_mask = arr('accepted').astype(bool)

    def fmt_block(mask, label):
        if mask.sum() == 0:
            return f"  {label:<22}: (none)"
        return (f"  {label:<22}: grad-NCC {ncc_b[mask].mean():.4f} -> {ncc_a[mask].mean():.4f} "
                f"(d={ (ncc_a[mask]-ncc_b[mask]).mean():+.4f})   "
                f"MI {mi_b[mask].mean():.4f} -> {mi_a[mask].mean():.4f} "
                f"(d={ (mi_a[mask]-mi_b[mask]).mean():+.4f})")

    tx_a, ty_a, th_a = arr('tx'), arr('ty'), arr('theta_deg')
    shift = np.hypot(tx_a, ty_a)

    lines = []
    lines.append("SAR<->optical pair alignment report")
    lines.append("=" * 60)
    lines.append(f"source            : {src_root}")
    lines.append(f"dest              : {dst_root}  ({'dry-run' if args.no_write else 'written'})")
    lines.append(f"scenes            : {scenes if scenes is not None else 'ALL'}")
    lines.append(f"pairs processed   : {n_rows}")
    lines.append(f"accepted (warped) : {n_acc}  ({100.0*n_acc/max(n_rows,1):.1f}%)")
    lines.append(f"grad-NCC improved : {n_imp}  ({100.0*n_imp/max(n_rows,1):.1f}%)")
    lines.append("")
    lines.append("Before -> After (gradient-domain QA; higher = better aligned):")
    lines.append(fmt_block(np.ones(n_rows, bool), "ALL pairs"))
    lines.append(fmt_block(acc_mask, "accepted only"))
    lines.append("")
    lines.append("Accepted-warp geometry:")
    if acc_mask.sum() > 0:
        lines.append(f"  |shift| px            : mean {shift[acc_mask].mean():.2f}  "
                     f"med {np.median(shift[acc_mask]):.2f}  max {shift[acc_mask].max():.2f}")
        lines.append(f"  theta deg             : mean {np.abs(th_a[acc_mask]).mean():.3f}  "
                     f"max {np.abs(th_a[acc_mask]).max():.3f}")
    else:
        lines.append("  (no accepted warps)")
    lines.append("")
    lines.append("Per-scene (ALL pairs):")
    for sc in sorted(set(r['scene'] for r in rows)):
        m = np.array([r['scene'] == sc for r in rows], dtype=bool)
        na = int(arr('accepted')[m].sum())
        lines.append(f"  {sc:<10} n={int(m.sum()):>5}  acc={na:>5}  "
                     f"NCC {ncc_b[m].mean():.4f}->{ncc_a[m].mean():.4f} "
                     f"({(ncc_a[m]-ncc_b[m]).mean():+.4f})  "
                     f"MI {mi_b[m].mean():.4f}->{mi_a[m].mean():.4f} "
                     f"({(mi_a[m]-mi_b[m]).mean():+.4f})")
    report = "\n".join(lines)

    with open(out_dir / 'report.txt', 'w') as f:
        f.write(report + "\n")

    print("-" * 78)
    print(report)
    print("-" * 78)
    print(f"csv : {csv_path}")
    print(f"qa  : {qa_dir}  ({qa_saved} overlays)")
    if args.no_write:
        print("DRY-RUN: no aligned images written. Re-run without --no-write to build the mirror.")


if __name__ == '__main__':
    main()
