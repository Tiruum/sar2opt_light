# src/data/qxslab/showcase.py
# Usage: python src/data/qxslab/showcase.py --data_dir data/QXSLAB_SAROPT --out output/showcase_qxslab.png

import argparse
import random
import os
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def discover_variants(root_dir):
    variants = []
    for d in sorted(os.listdir(root_dir)):
        if d.startswith('sar_') and os.path.isdir(os.path.join(root_dir, d)):
            suffix = d[4:]
            if os.path.isdir(os.path.join(root_dir, 'opt_' + suffix)):
                variants.append(suffix)
    return variants


def collect_items(root_dir, variants):
    items = []
    for variant in variants:
        sar_dir = os.path.join(root_dir, 'sar_' + variant)
        opt_dir = os.path.join(root_dir, 'opt_' + variant)
        for fname in sorted(os.listdir(sar_dir)):
            if os.path.isfile(os.path.join(opt_dir, fname)):
                items.append((variant, fname))
    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--out', default='output/showcase_qxslab.png')
    parser.add_argument('--n', type=int, default=8)
    parser.add_argument('--variants', nargs='*', default=None)
    args = parser.parse_args()

    variants = args.variants or discover_variants(args.data_dir)
    items = collect_items(args.data_dir, variants)
    print(f"QXSLAB | variants={variants} | total pairs={len(items)}")

    sample = random.sample(items, min(args.n, len(items)))
    fig, axes = plt.subplots(len(sample), 2, figsize=(6, 2.5 * len(sample)))
    if len(sample) == 1:
        axes = [axes]

    for row, (variant, fname) in enumerate(sample):
        sar = cv2.imread(os.path.join(args.data_dir, 'sar_' + variant, fname), cv2.IMREAD_GRAYSCALE)
        opt = cv2.cvtColor(cv2.imread(os.path.join(args.data_dir, 'opt_' + variant, fname)), cv2.COLOR_BGR2RGB)

        axes[row][0].imshow(sar, cmap='gray', vmin=0, vmax=255)
        axes[row][0].set_title(f'SAR  {fname}', fontsize=7)
        axes[row][0].axis('off')
        axes[row][1].imshow(opt)
        axes[row][1].set_title(f'OPT  {fname}', fontsize=7)
        axes[row][1].axis('off')

    plt.suptitle(f'QXSLAB — {variants}', fontsize=9)
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    plt.savefig(args.out, dpi=120, bbox_inches='tight')
    print(f'Saved -> {args.out}')


if __name__ == '__main__':
    main()
