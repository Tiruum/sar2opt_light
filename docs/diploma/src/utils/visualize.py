from collections import defaultdict
from typing import Literal, Optional
import random
import torch
import torchvision.transforms.functional as TF
from torchvision.utils import save_image


def pin_diverse_batch(datamodule, n_per_class: int = 2, device='cpu', seed: Optional[int] = 42):
    """
    Sample n_per_class items from each SEN12 class for a reproducibly diverse
    epoch-end visualisation batch.  Replaces the alphabetical next(iter()) bias
    that caused physics-10 to show only agri/roads in every epoch image.

    Args:
        datamodule: SEN12Datamodule (must have called setup('fit') first)
        n_per_class: how many samples to draw from each class
        device: target device for the returned tensors
        seed: fixed RNG seed for reproducibility across calls

    Returns:
        (sar, opt)  each [N_total, C, H, W] on `device`
    """
    dataset = datamodule.train_dataset
    by_class = defaultdict(list)
    for idx, item in enumerate(dataset.items):
        # SEN12: (class, s1_file, s2_file) — 3-tuple
        # SEN12Full: (season, s1_dir, s2_dir, s1_fname, s2_fname) — 5-tuple
        key = item[0] if len(item) == 3 else item[1]
        by_class[key].append(idx)

    rng = random.Random(seed)
    selected = []
    for cls in sorted(by_class.keys()):
        pool = by_class[cls]
        chosen = rng.sample(pool, min(n_per_class, len(pool)))
        selected.extend(chosen)

    sar_list, opt_list = [], []
    for idx in selected:
        sar, opt = dataset[idx]
        sar_list.append(sar)
        opt_list.append(opt)

    return (
        torch.stack(sar_list).to(device),
        torch.stack(opt_list).to(device),
    )

def visualize_batch(
    real_sar, fake_optical, real_optical,
    save_path='results/batch_output.png',
    max_rows=8, cmap='inferno', mode: Literal['quality', 'fast'] = 'quality',
    title: str = None,
    cfr_out=None, hfcf_out=None, fusion_weight=None
):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    batch_size = min(real_sar.size(0), max_rows)

    if mode == 'quality':
        diff_maps = []
        for i in range(batch_size):
            gen_img = (fake_optical[i].cpu() + 1) / 2.0
            gt_img = (real_optical[i].cpu() + 1) / 2.0
            diff = torch.abs(gen_img - gt_img)
            diff_gray = torch.mean(diff, dim=0)
            diff_maps.append(diff_gray)

        all_diffs = torch.stack(diff_maps)
        vmin, vmax = all_diffs.min().item(), all_diffs.max().item()

        # Если переданы промежуточные выходы, добавляем колонки (обе ветки или только CFR)
        show_both     = cfr_out is not None and hfcf_out is not None
        show_cfr_only = cfr_out is not None and hfcf_out is None
        show_branches = show_both or show_cfr_only
        num_cols      = 7 if show_both else 6 if show_cfr_only else 5
        width_ratios  = ([1, 1, 1, 1, 1, 1, 0.05] if show_both else
                         [1, 1, 1, 1, 1, 0.05]    if show_cfr_only else
                         [1, 1, 1, 1, 0.05])
        fig_width     = 12 if show_both else 10 if show_cfr_only else 8

        fig = plt.figure(figsize=(fig_width, batch_size * 2))

        if title:
            if show_branches and fusion_weight is not None:
                title += f" | Fusion Weight: {fusion_weight:.4f}"
            fig.suptitle(title, fontsize=14, y=0.99) # ⬅️ Вернул заголовок наверх

        gs = gridspec.GridSpec(
            batch_size, num_cols,
            width_ratios=width_ratios,
            wspace=0.01,    # ⬅️ почти склеить по горизонтали
            hspace=0.15     # ⬅️ небольшой вертикальный отступ
        )

        for idx in range(batch_size):
            sar_img = (real_sar[idx].cpu() + 1) / 2.0
            gen_img = (fake_optical[idx].cpu() + 1) / 2.0
            gt_img = (real_optical[idx].cpu() + 1) / 2.0
            diff_gray = diff_maps[idx]

            images = [
                TF.to_pil_image(sar_img).convert("RGB"),
                TF.to_pil_image(gen_img),
                TF.to_pil_image(gt_img)
            ]
            titles = ['SAR', 'Generated', 'Ground Truth', 'Difference']
            
            if show_both:
                cfr_img  = (cfr_out[idx].cpu()  + 1) / 2.0
                hfcf_img = (hfcf_out[idx].cpu() + 1) / 2.0
                images.insert(1, TF.to_pil_image(cfr_img))
                images.insert(2, TF.to_pil_image(hfcf_img))
                titles.insert(1, 'CFR Branch')
                titles.insert(2, 'HFCF Branch')
            elif show_cfr_only:
                cfr_img = (cfr_out[idx].cpu() + 1) / 2.0
                images.insert(1, TF.to_pil_image(cfr_img))
                titles.insert(1, 'CFR Branch')

            num_image_cols = 5 if show_both else 4 if show_cfr_only else 3

            for j in range(num_image_cols):
                ax = fig.add_subplot(gs[idx, j])
                ax.imshow(images[j])
                ax.axis('off')
                if idx == 0:
                    ax.set_title(titles[j], fontsize=10)

            ax = fig.add_subplot(gs[idx, num_image_cols])
            im = ax.imshow(diff_gray, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.axis('off')
            if idx == 0:
                ax.set_title(titles[-1], fontsize=10)

        cbar_ax = fig.add_subplot(gs[:, num_cols - 1])
        fig.colorbar(im, cax=cbar_ax)
        cbar_ax.set_title('Diff', fontsize=9)

        # Сильно увеличиваем верхний отступ (top_margin), чтобы картинки уехали вниз
        top_margin = 0.94 if title else 0.97
        plt.subplots_adjust(left=0.01, right=0.94, top=top_margin, bottom=0.03)

        plt.savefig(save_path, dpi=300)
        plt.close()
    elif mode == 'fast':
        # Преобразуем SAR в RGB, если он одноканальный
        sar_rgb = (real_sar.repeat(1, 3, 1, 1) + 1) / 2.0 if real_sar.size(1) == 1 else (real_sar + 1) / 2.0
        
        # Нормализуем остальные изображения
        fake_rgb = (fake_optical + 1) / 2.0
        real_rgb = (real_optical + 1) / 2.0
        
        # Конкатенируем все тензоры в один список
        concatenated = torch.cat(
            [sar_rgb, fake_rgb, real_rgb],
            dim=2  # Конкатенация по ширине
        )

        save_image(concatenated, save_path)
    else:
        raise ValueError("Invalid mode. Use 'quality' or 'fast'.")