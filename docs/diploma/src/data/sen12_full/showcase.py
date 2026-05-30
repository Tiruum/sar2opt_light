# src/data/sen12_full/showcase.py
# Usage: python src/data/sen12_full/showcase.py --data_dir data/SEN1-2 --out output/showcase_sen12full_paper.png

import argparse
import random
import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def collect_items(root_dir, seasons=None, scenes=None):
    scene_set = set(scenes) if scenes else None
    items = []
    all_seasons = seasons or sorted(
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d)) and not d.startswith('.')
        and not d.endswith('.sha512') and not d.endswith('.txt')
    )
    for season in all_seasons:
        season_dir = os.path.join(root_dir, season)
        if not os.path.isdir(season_dir):
            continue
        for s1_d in sorted(d for d in os.listdir(season_dir) if d.startswith('s1_')):
            scene_id = s1_d[3:]
            if scene_set and scene_id not in scene_set:
                continue
            s2_d = 's2_' + scene_id
            s1_dir = os.path.join(season_dir, s1_d)
            s2_dir = os.path.join(season_dir, s2_d)
            if not os.path.isdir(s2_dir):
                continue
            for fname in sorted(os.listdir(s1_dir)):
                s2_fname = fname.replace('_s1_', '_s2_')
                if os.path.isfile(os.path.join(s2_dir, s2_fname)):
                    items.append((season, s1_d, s2_d, fname, s2_fname))
    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--out', default='output/showcase_sen12full_paper.png')
    parser.add_argument('--n', type=int, default=8)
    parser.add_argument('--scenes', nargs='*', default=None)
    parser.add_argument('--seasons', nargs='*', default=None)
    args = parser.parse_args()

    items = collect_items(args.data_dir, args.seasons, args.scenes)
    print(f"SEN12Full | scenes={args.scenes or 'all'} | total pairs={len(items)}")

    sample = random.sample(items, min(args.n, len(items)))
    fig, axes = plt.subplots(len(sample), 2, figsize=(6, 2.5 * len(sample)))
    if len(sample) == 1:
        axes = [axes]

    for row, (season, s1_d, s2_d, s1_f, s2_f) in enumerate(sample):
        sar = cv2.imread(os.path.join(args.data_dir, season, s1_d, s1_f), cv2.IMREAD_GRAYSCALE)
        opt_bgr = cv2.imread(os.path.join(args.data_dir, season, s2_d, s2_f))
        if sar is None or opt_bgr is None:
            continue
        opt = cv2.cvtColor(opt_bgr, cv2.COLOR_BGR2RGB)

        p1, p99 = np.percentile(sar, 1), np.percentile(sar, 99)
        sar_disp = np.clip(sar, p1, p99)

        axes[row][0].imshow(sar_disp, cmap='gray')
        axes[row][0].set_title(f'{season}/{s1_d}/{s1_f}', fontsize=5)
        axes[row][0].axis('off')
        axes[row][1].imshow(opt)
        axes[row][1].set_title(f'{s2_d}/{s2_f}', fontsize=5)
        axes[row][1].axis('off')

    plt.suptitle(f'SEN12Full — scenes={args.scenes or "all"}', fontsize=9)
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    plt.savefig(args.out, dpi=120, bbox_inches='tight')
    print(f'Saved -> {args.out}')


if __name__ == '__main__':
    main()
