"""Measure ACTUAL SAR<->optical misalignment magnitude in raw sen12_full.

Throwaway diagnostic (no GPU, no training). Samples ~300 paired patches from the
5 training scenes, estimates the SAR->optical translation shift in pixels using
the SAME gradient-domain phaseCorrelate->ECC estimator that scripts/align_pairs.py
uses to build the aligned mirror, and reports the shift-magnitude distribution.

Decides whether the SAWG deformation aligner's target phi should be ~0.2px
(data already near-aligned -> aligner correctly settles tiny) or ~2px (real
misalignment the aligner must be strengthened to catch).

Run from repo root (either form works):
    python -m src.models.llwt_v5.diag_misalign
    python src/models/llwt_v5/diag_misalign.py
"""
from __future__ import annotations

import math
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from omegaconf import OmegaConf

# --- repo-root sys.path shim (so `python src/.../diag_misalign.py` also works) ---
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Reuse the EXACT estimator + structure helpers from the alignment pipeline.
from scripts.align_pairs import (
    estimate_warp,
    decompose,
    lee_despeckle_np,
    grad_mag,
)
from src.data.sen12_full.dataset import SEN12Full

CONFIG_PATH = "src/models/llwt_v5/config.yaml"
N_SAMPLE = 300
SEED = 42


def pct(arr, p):
    return float(np.percentile(arr, p))


def main():
    cfg = OmegaConf.load(CONFIG_PATH)
    src_root = str(cfg.data.data_dir.sen12_full)
    scenes = list(cfg.data.scenes)
    sar_channels = int(cfg.data.sar_channels)
    image_size = int(cfg.data.image_size)

    print("=" * 78)
    print("RAW sen12_full SAR<->optical misalignment diagnostic")
    print(f"  src        : {src_root}")
    print(f"  scenes     : {scenes}   (sar_channels={sar_channels}, image_size={image_size})")
    print(f"  estimator  : align_pairs.estimate_warp (gradient-domain phaseCorrelate -> ECC)")
    print(f"  sample     : up to {N_SAMPLE} pairs  (seed={SEED})")
    print("=" * 78)

    if not os.path.isdir(src_root):
        raise FileNotFoundError(f"source root not found: {src_root}")

    # Exact pairing logic (s1_/s2_ dirs, _s1_->_s2_ filename substitution).
    ds = SEN12Full(root_dir=src_root, sar_channels=1, scenes=scenes)
    items = ds.items
    n_total = len(items)
    if n_total == 0:
        raise RuntimeError(f"0 pairs found in {src_root} (scenes={scenes}).")

    rng = random.Random(SEED)
    sample = items if n_total <= N_SAMPLE else rng.sample(items, N_SAMPLE)
    print(f"paired files found : {n_total}  -> sampling {len(sample)}")

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-4)
    hann = None

    txs, tys, mags, thetas, methods = [], [], [], [], []
    n_skipped = 0

    for i, (season, s1_d, s2_d, s1_fname, s2_fname) in enumerate(sample):
        sar_path = os.path.join(src_root, season, s1_d, s1_fname)
        opt_path = os.path.join(src_root, season, s2_d, s2_fname)

        sar_gray = cv2.imread(sar_path, cv2.IMREAD_GRAYSCALE)
        opt_bgr = cv2.imread(opt_path, cv2.IMREAD_COLOR)
        if sar_gray is None or opt_bgr is None:
            n_skipped += 1
            continue
        opt_gray = cv2.cvtColor(opt_bgr, cv2.COLOR_BGR2GRAY)

        # Match training resolution (256). INTER_AREA for downscale = clean.
        if sar_gray.shape != (image_size, image_size):
            sar_gray = cv2.resize(sar_gray, (image_size, image_size), interpolation=cv2.INTER_AREA)
        if opt_gray.shape != (image_size, image_size):
            opt_gray = cv2.resize(opt_gray, (image_size, image_size), interpolation=cv2.INTER_AREA)

        # Identical structure pipeline to align_pairs.py.
        sar_ds = lee_despeckle_np(sar_gray.astype(np.float32), win=5)
        sar_g = grad_mag(sar_ds)
        opt_g = grad_mag(opt_gray)
        sar_gs = cv2.GaussianBlur(sar_g, (5, 5), 0)
        opt_gs = cv2.GaussianBlur(opt_g, (5, 5), 0)

        H, W = sar_g.shape
        if hann is None or hann.shape != (H, W):
            hann = cv2.createHanningWindow((W, H), cv2.CV_32F)

        # Skip featureless SAR (registration undefined) — same gate as the pipeline.
        if float(sar_g.std()) < 5e-3:
            n_skipped += 1
            continue

        method, M, _warped = estimate_warp(sar_g, sar_gs, opt_g, opt_gs, hann, criteria)
        tx, ty, theta = decompose(M)

        txs.append(abs(tx))
        tys.append(abs(ty))
        mags.append(math.hypot(tx, ty))
        thetas.append(abs(theta))
        methods.append(method)

        if (i + 1) % 50 == 0:
            print(f"  [{i+1:>4}/{len(sample)}] last method={method:10s} "
                  f"shift={math.hypot(tx, ty):.3f}px")

    mags = np.array(mags, dtype=np.float64)
    txs = np.array(txs, dtype=np.float64)
    tys = np.array(tys, dtype=np.float64)
    thetas = np.array(thetas, dtype=np.float64)
    n = len(mags)
    if n == 0:
        raise RuntimeError("No estimable pairs (all skipped/featureless).")

    def frac_gt(t):
        return float((mags > t).mean())

    method_counts = {m: methods.count(m) for m in sorted(set(methods))}

    print("-" * 78)
    print(f"estimable pairs    : {n}   (skipped/unreadable/featureless: {n_skipped})")
    print(f"method mix         : {method_counts}")
    print("")
    print("Shift magnitude  sqrt(tx^2+ty^2)  [pixels @ 256]:")
    print(f"  mean   : {mags.mean():.3f}")
    print(f"  median : {np.median(mags):.3f}")
    print(f"  p25    : {pct(mags,25):.3f}")
    print(f"  p75    : {pct(mags,75):.3f}")
    print(f"  p90    : {pct(mags,90):.3f}")
    print(f"  max    : {mags.max():.3f}")
    print("")
    print("Per-axis (absolute):")
    print(f"  mean |tx| : {txs.mean():.3f}    mean |ty| : {tys.mean():.3f}")
    print("")
    print("Fraction of pairs with shift magnitude exceeding threshold:")
    print(f"  > 0.5 px : {100*frac_gt(0.5):5.1f}%")
    print(f"  > 1.0 px : {100*frac_gt(1.0):5.1f}%")
    print(f"  > 2.0 px : {100*frac_gt(2.0):5.1f}%")
    print("")
    print(f"Rotation |theta| deg : mean {thetas.mean():.3f}  p90 {pct(thetas,90):.3f}  max {thetas.max():.3f}")
    print("-" * 78)

    med = float(np.median(mags))
    p90 = pct(mags, 90)
    if med < 0.5 and p90 < 1.0:
        phi = 0.2
        verdict_extra = "near-aligned -> aligner correctly settles tiny; novelty needs a more-misaligned dataset"
    elif med > 1.0 or p90 > 2.0:
        phi = 2.0
        verdict_extra = "genuinely misaligned -> STRENGTHEN aligner (lower reg / direct registration term / drop warm-start)"
    else:
        phi = max(0.5, round(med, 2))
        verdict_extra = "mild misalignment -> moderate phi target"

    print(f"[misalign] median shift = {med:.3f} px, p90 = {p90:.3f} px -> target phi ~{phi} px")
    print(f"[misalign] verdict: {verdict_extra}")


if __name__ == "__main__":
    main()
