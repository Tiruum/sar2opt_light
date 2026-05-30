# src/data/qxslab/dataset_analysis.py

import argparse
import os
import random
from collections import defaultdict

import cv2
import numpy as np


def _discover_variants(root_dir):
    variants = []
    for d in sorted(os.listdir(root_dir)):
        if d.startswith('sar_') and os.path.isdir(os.path.join(root_dir, d)):
            suffix = d[4:]
            if os.path.isdir(os.path.join(root_dir, 'opt_' + suffix)):
                variants.append(suffix)
    return variants


def _collect_items(root_dir, variants):
    items = []
    for variant in variants:
        sar_dir = os.path.join(root_dir, 'sar_' + variant)
        opt_dir = os.path.join(root_dir, 'opt_' + variant)
        for fname in sorted(os.listdir(sar_dir)):
            if os.path.isfile(os.path.join(opt_dir, fname)):
                items.append((variant, fname, fname))
    return items


def _pixel_stats(paths: list, read_flag: int, n_channels: int, sample: int) -> dict:
    sampled = random.sample(paths, min(sample, len(paths)))
    all_vals = [[] for _ in range(n_channels)]
    for p in sampled:
        img = cv2.imread(p, read_flag)
        if img is None:
            continue
        if img.ndim == 2:
            img = img[..., None]
        elif read_flag == cv2.IMREAD_COLOR:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for c in range(min(n_channels, img.shape[2])):
            all_vals[c].append(img[:, :, c].flatten())
    stats = {}
    for c in range(n_channels):
        if not all_vals[c]:
            continue
        v = np.concatenate(all_vals[c]).astype(np.float32)
        stats[c] = {
            'min': float(v.min()), 'max': float(v.max()),
            'mean': float(v.mean()), 'std': float(v.std()),
            **{f'p{p}': float(np.percentile(v, p)) for p in [1, 5, 50, 95, 99]},
        }
    return stats


def _print_stats(label: str, stats: dict):
    print(f"\n  {label}:")
    for c, s in stats.items():
        print(f"    ch{c}: min={s['min']:.1f} max={s['max']:.1f} "
              f"mean={s['mean']:.2f} std={s['std']:.2f} | "
              f"p1={s['p1']:.1f} p5={s['p5']:.1f} p50={s['p50']:.1f} "
              f"p95={s['p95']:.1f} p99={s['p99']:.1f}")


def main():
    parser = argparse.ArgumentParser(description="QXSLAB dataset analysis")
    parser.add_argument('--data_dir', required=True, help='Path to QXSLAB root')
    parser.add_argument('--sample', type=int, default=500,
                        help='Max images to sample for pixel stats')
    parser.add_argument('--variants', nargs='*', default=None,
                        help='Variant suffixes to include, e.g. 256_oc_0.2 (default: all)')
    parser.add_argument('--sar_channels', type=int, default=1, choices=[1, 3])
    args = parser.parse_args()

    variants = args.variants or _discover_variants(args.data_dir)
    items = _collect_items(args.data_dir, variants)

    print("=" * 60)
    print("QXSLAB Dataset Analysis")
    print("=" * 60)
    print(f"  root_dir : {args.data_dir}")
    print(f"  variants : {variants}")
    print(f"  total pairs: {len(items)}")

    # Breakdown by variant
    breakdown: dict = defaultdict(int)
    sar_paths, opt_paths = [], []
    for variant, sar_fname, opt_fname in items:
        breakdown[variant] += 1
        sar_paths.append(os.path.join(args.data_dir, 'sar_' + variant, sar_fname))
        opt_paths.append(os.path.join(args.data_dir, 'opt_' + variant, opt_fname))

    print("\n  Breakdown (variant -> count):")
    for v in sorted(breakdown):
        print(f"    {v}: {breakdown[v]}")

    # Integrity
    print("\n  Integrity check...")
    missing_sar = sum(1 for p in sar_paths if not os.path.isfile(p))
    missing_opt = sum(1 for p in opt_paths if not os.path.isfile(p))
    print(f"    missing SAR files : {missing_sar}")
    print(f"    missing OPT files : {missing_opt}")

    # Image size check (first 20)
    sizes = set()
    for p in sar_paths[:20]:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            sizes.add(img.shape)
    print(f"    SAR image shapes (first 20 files): {sizes}")

    # Pixel stats
    print(f"\n  Pixel stats (sampled {min(args.sample, len(sar_paths))} images):")
    sar_flag = cv2.IMREAD_GRAYSCALE if args.sar_channels == 1 else cv2.IMREAD_COLOR
    sar_stats = _pixel_stats(sar_paths, sar_flag, args.sar_channels, args.sample)
    opt_stats = _pixel_stats(opt_paths, cv2.IMREAD_COLOR, 3, args.sample)
    _print_stats("SAR", sar_stats)
    _print_stats("Optical", opt_stats)

    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()
