# src/data/sen12_full/dataset_analysis.py

import argparse
import os
import random
from collections import defaultdict

import cv2
import numpy as np


_ALLOWED_EXT = frozenset({'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'})


def _collect_items(root_dir, seasons=None, scenes=None):
    scene_set = set(scenes) if scenes is not None else None
    all_seasons = seasons or sorted(
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d)) and not d.startswith('.')
    )
    items = []
    for season in all_seasons:
        season_dir = os.path.join(root_dir, season)
        if not os.path.isdir(season_dir):
            continue
        for s1_d in sorted(d for d in os.listdir(season_dir) if d.startswith('s1_') and os.path.isdir(os.path.join(season_dir, d))):
            scene_id = s1_d[3:]
            if scene_set is not None and scene_id not in scene_set:
                continue
            s2_d = 's2_' + scene_id
            s1_dir = os.path.join(season_dir, s1_d)
            s2_dir = os.path.join(season_dir, s2_d)
            if not os.path.isdir(s2_dir):
                continue
            for fname in sorted(f for f in os.listdir(s1_dir) if os.path.splitext(f)[1].lower() in _ALLOWED_EXT):
                s2_fname = fname.replace('_s1_', '_s2_') if '_s1_' in fname else fname
                if os.path.isfile(os.path.join(s2_dir, s2_fname)):
                    items.append((season, s1_d, s2_d, fname, s2_fname))
    return all_seasons, items


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
    parser = argparse.ArgumentParser(description="SEN12Full dataset analysis")
    parser.add_argument('--data_dir', required=True, help='Path to SEN1-2 root')
    parser.add_argument('--sample', type=int, default=500,
                        help='Max images to sample for pixel stats')
    parser.add_argument('--scenes', nargs='*', default=None,
                        help='Scene IDs to include (default: all)')
    parser.add_argument('--seasons', nargs='*', default=None,
                        help='Season folder names to include (default: all)')
    parser.add_argument('--sar_channels', type=int, default=1, choices=[1, 3])
    args = parser.parse_args()

    all_seasons, items = _collect_items(args.data_dir, args.seasons, args.scenes)

    print("=" * 60)
    print("SEN12Full Dataset Analysis")
    print("=" * 60)
    print(f"  root_dir : {args.data_dir}")
    print(f"  seasons  : {all_seasons}")
    print(f"  scenes   : {sorted(args.scenes) if args.scenes else 'all'}")
    print(f"  total pairs: {len(items)}")

    # Breakdown by season x scene
    breakdown: dict = defaultdict(lambda: defaultdict(int))
    sar_paths, opt_paths = [], []
    for season, s1_d, s2_d, s1_f, s2_f in items:
        breakdown[season][s1_d[3:]] += 1  # strip s1_
        sar_paths.append(os.path.join(args.data_dir, season, s1_d, s1_f))
        opt_paths.append(os.path.join(args.data_dir, season, s2_d, s2_f))

    print("\n  Breakdown (season x scene -> count):")
    for season in sorted(breakdown):
        for scene_id in sorted(breakdown[season], key=lambda x: int(x) if x.isdigit() else x):
            print(f"    {season} / scene {scene_id}: {breakdown[season][scene_id]}")

    # Integrity: verify all paths exist
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
