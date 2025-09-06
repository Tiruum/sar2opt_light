from typing import Literal
import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import matplotlib.gridspec as gridspec
from torchvision.utils import save_image

def visualize_batch(
    real_sar, fake_optical, real_optical,
    save_path='results/batch_output.png',
    max_rows=8, cmap='inferno', mode: Literal['quality', 'fast'] = 'quality',
    title: str = None
):
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

        fig = plt.figure(figsize=(8, batch_size * 2))

        if title:
            fig.suptitle(title, fontsize=14, y=0.99)

        gs = gridspec.GridSpec(
            batch_size, 5,
            width_ratios=[1, 1, 1, 1, 0.05],
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

            for j in range(3):
                ax = fig.add_subplot(gs[idx, j])
                ax.imshow(images[j])
                ax.axis('off')
                if idx == 0:
                    ax.set_title(titles[j], fontsize=10)

            ax = fig.add_subplot(gs[idx, 3])
            im = ax.imshow(diff_gray, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.axis('off')
            if idx == 0:
                ax.set_title(titles[3], fontsize=10)

        cbar_ax = fig.add_subplot(gs[:, 4])
        fig.colorbar(im, cax=cbar_ax)
        cbar_ax.set_title('Diff', fontsize=9)

        # Минимальные отступы по краям
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