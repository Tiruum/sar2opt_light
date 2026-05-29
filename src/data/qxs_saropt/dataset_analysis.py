# src/data/qxs_saropt/dataset_analysis.py

import argparse
import os
import random

import cv2
import numpy as np


_ALLOWED_EXT = frozenset({'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'})


def _collect_items(root_dir, sar_subdir, opt_subdir):
    sar_dir = os.path.join(root_dir, sar_subdir)
    opt_dir = os.path.join(root_dir, opt_subdir)
    if not os.path.isdir(sar_dir):
        raise FileNotFoundError(f"SAR subdir missing: {sar_dir}")
    if not os.path.isdir(opt_dir):
        raise FileNotFoundError(f"Optical subdir missing: {opt_dir}")
    sar_files = sorted(
        f for f in os.listdir(sar_dir)
        if os.path.splitext(f)[1].lower() in _ALLOWED_EXT
    )
    items = [
        f for f in sar_files
        if os.path.isfile(os.path.join(opt_dir, f))
    ]
    return sar_dir, opt_dir, items


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
            'skew': float(((v - v.mean()) ** 3).mean() / (v.std() ** 3 + 1e-8)),
            'kurt': float(((v - v.mean()) ** 4).mean() / (v.std() ** 4 + 1e-8) - 3.0),
            **{f'p{p}': float(np.percentile(v, p)) for p in [1, 5, 50, 95, 99]},
        }
    return stats


def _print_stats(label: str, stats: dict):
    print(f"\n  {label}:")
    for c, s in stats.items():
        print(f"    ch{c}: min={s['min']:.1f} max={s['max']:.1f} "
              f"mean={s['mean']:.2f} std={s['std']:.2f} "
              f"skew={s['skew']:+.2f} kurt={s['kurt']:+.2f} | "
              f"p1={s['p1']:.1f} p5={s['p5']:.1f} p50={s['p50']:.1f} "
              f"p95={s['p95']:.1f} p99={s['p99']:.1f}")


def _logscale_stats(paths: list, sample: int) -> dict:
    """Compute log1p-normalized SAR distribution to compare against linear."""
    sampled = random.sample(paths, min(sample, len(paths)))
    vals = []
    for p in sampled:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        v = np.log1p(img.astype(np.float32))
        v = v / np.log1p(255.0) * 255.0
        vals.append(v.flatten())
    v = np.concatenate(vals).astype(np.float32)
    return {0: {
        'min': float(v.min()), 'max': float(v.max()),
        'mean': float(v.mean()), 'std': float(v.std()),
        'skew': float(((v - v.mean()) ** 3).mean() / (v.std() ** 3 + 1e-8)),
        'kurt': float(((v - v.mean()) ** 4).mean() / (v.std() ** 4 + 1e-8) - 3.0),
        **{f'p{p}': float(np.percentile(v, p)) for p in [1, 5, 50, 95, 99]},
    }}


def main():
    parser = argparse.ArgumentParser(description="QXSLAB SAR-OPT dataset analysis")
    parser.add_argument('--data_dir', default='./data/QXSLAB_SAROPT',
                        help='Path to QXSLAB_SAROPT root')
    parser.add_argument('--sar_subdir', default='sar_256_oc_0.2')
    parser.add_argument('--opt_subdir', default='opt_256_oc_0.2')
    parser.add_argument('--sample', type=int, default=500,
                        help='Max images to sample for pixel stats')
    parser.add_argument('--sar_channels', type=int, default=1, choices=[1, 3])
    args = parser.parse_args()

    sar_dir, opt_dir, items = _collect_items(args.data_dir, args.sar_subdir, args.opt_subdir)

    print("=" * 60)
    print("QXSLAB SAR-OPT Dataset Analysis")
    print("=" * 60)
    print(f"  root_dir   : {args.data_dir}")
    print(f"  sar_subdir : {args.sar_subdir}")
    print(f"  opt_subdir : {args.opt_subdir}")
    print(f"  total pairs: {len(items)}")

    sar_paths = [os.path.join(sar_dir, f) for f in items]
    opt_paths = [os.path.join(opt_dir, f) for f in items]

    # Integrity
    print("\n  Integrity check...")
    missing_sar = sum(1 for p in sar_paths if not os.path.isfile(p))
    missing_opt = sum(1 for p in opt_paths if not os.path.isfile(p))
    print(f"    missing SAR files : {missing_sar}")
    print(f"    missing OPT files : {missing_opt}")

    sizes_sar = set()
    sizes_opt = set()
    for p in sar_paths[:20]:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            sizes_sar.add(img.shape)
    for p in opt_paths[:20]:
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is not None:
            sizes_opt.add(img.shape)
    print(f"    SAR image shapes (first 20): {sizes_sar}")
    print(f"    OPT image shapes (first 20): {sizes_opt}")

    print(f"\n  Pixel stats (sampled {min(args.sample, len(sar_paths))} images):")
    sar_flag = cv2.IMREAD_GRAYSCALE if args.sar_channels == 1 else cv2.IMREAD_COLOR
    sar_stats = _pixel_stats(sar_paths, sar_flag, args.sar_channels, args.sample)
    opt_stats = _pixel_stats(opt_paths, cv2.IMREAD_COLOR, 3, args.sample)
    _print_stats("SAR (linear)", sar_stats)
    _print_stats("Optical (linear)", opt_stats)

    # Log-norm comparison — informs whether use_lognorm should be enabled.
    print(f"\n  Log-norm comparison (sampled {min(args.sample, len(sar_paths))} SAR):")
    sar_log_stats = _logscale_stats(sar_paths, args.sample)
    _print_stats("SAR (log1p, rescaled to [0,255])", sar_log_stats)

    print("\n  Interpretation:")
    s_lin = sar_stats[0]
    s_log = sar_log_stats[0]
    print(f"    Linear skew={s_lin['skew']:+.2f} kurt={s_lin['kurt']:+.2f}")
    print(f"    Log1p  skew={s_log['skew']:+.2f} kurt={s_log['kurt']:+.2f}")
    if abs(s_log['skew']) < abs(s_lin['skew']) and abs(s_log['kurt']) < abs(s_lin['kurt']):
        print("    => Log-norm reduces skew AND kurtosis: consider use_lognorm=True.")
    elif abs(s_lin['skew']) < 0.5:
        print("    => Linear distribution already near-symmetric: log-norm NOT needed.")
    else:
        print("    => Mixed signal; eyeball percentiles before deciding.")

    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()
