"""Standalone inference entry for LLW-Former v0.3.0.

Loads ``CHECKPOINT`` (Lightning ckpt), strips ``netG.`` prefix, applies the
v4 generator to ``N_IMAGES`` SAR samples, writes side-by-side PNGs with
PSNR/SSIM.  Mirrors ``src/models/llwt/inference.py``.
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from omegaconf import OmegaConf
from torchmetrics.image import (
    PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure,
)

from src.data.sen12_full.datamodule import SEN12FullDataModule
from src.models.llwt_v4.gen import LLWv4Generator


CHECKPOINT = "checkpoints/llwt_v4/llwt-v0.4.1-perband-r4-adaptive/last.ckpt"
N_IMAGES = 10
SPLIT = "val"  # "train" or "val"
OUTPUT_DIR = f"./src/models/llwt_v4/output/{SPLIT}"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_LIVE_WEIGHTS = False


def main():
    cfg = OmegaConf.load('./src/models/llwt_v4/config.yaml')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    dm = SEN12FullDataModule(
        data_dir=cfg.data.data_dir.sen12_full,
        batch_size=cfg.data.get('val_batch_size', cfg.data.batch_size),
        image_size=cfg.data.image_size,
        num_workers=cfg.data.num_workers,
        persistent_workers=cfg.data.persistent_workers,
        prefetch_factor=cfg.data.prefetch_factor,
        train_val_split_ratio=cfg.data.train_val_split_ratio,
        seed=cfg.data.seed,
        sar_channels=cfg.data.sar_channels,
        use_augmentation=cfg.data.use_train_common_transform,
        scenes=list(cfg.data.scenes),
        train_crop_size=None,
        val_batch_size=cfg.data.get('val_batch_size', None),
    )
    dm.setup("fit")

    loader = dm.val_dataloader() if SPLIT == "val" else dm.train_dataloader()
    sar, opt = next(iter(loader))
    sar = sar.to(DEVICE)
    opt = opt.to(DEVICE)

    ckpt = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
    netG = LLWv4Generator(cfg)
    src_dict = (
        ckpt.get('current_model_state', ckpt['state_dict'])
        if USE_LIVE_WEIGHTS
        else ckpt['state_dict']
    )
    state_dict = {
        k[len('netG.'):]: v for k, v in src_dict.items() if k.startswith('netG.')
    }
    src_label = 'live (current_model_state)' if USE_LIVE_WEIGHTS else 'EMA-or-live (state_dict)'
    print(f"[ckpt] loaded {src_label} netG weights ({len(state_dict)} tensors)")
    netG.load_state_dict(state_dict)
    netG = netG.to(DEVICE).eval()

    with torch.no_grad():
        generated = netG(sar)

    sar_np = sar.detach().cpu().numpy()
    opt_np = opt.detach().cpu().numpy()
    gen_np = generated.detach().cpu().numpy()

    gen_01 = (generated.detach().clamp(-1, 1) + 1.0) / 2.0
    opt_01 = (opt.detach() + 1.0) / 2.0

    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(DEVICE)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)

    n = min(N_IMAGES, len(sar))
    avg_psnr = 0.0
    avg_ssim = 0.0
    for i in range(n):
        curr_gen = gen_01[i:i + 1]
        curr_opt = opt_01[i:i + 1]
        psnr_val = psnr_metric(curr_gen, curr_opt).item()
        ssim_val = ssim_metric(curr_gen, curr_opt).item()
        avg_psnr += psnr_val
        avg_ssim += ssim_val

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        sar_img = (sar_np[i, 0] + 1) / 2
        axes[0].imshow(sar_img, cmap='gray')
        axes[0].set_title("SAR input")
        axes[0].axis('off')
        gen_img = (gen_np[i] + 1) / 2
        gen_img = gen_img.transpose(1, 2, 0)
        axes[1].imshow(gen_img)
        axes[1].set_title(f"Generated (PSNR: {psnr_val:.2f}, SSIM: {ssim_val:.3f})")
        axes[1].axis('off')
        gt_img = (opt_np[i] + 1) / 2
        gt_img = gt_img.transpose(1, 2, 0)
        axes[2].imshow(gt_img)
        axes[2].set_title("GT optical")
        axes[2].axis('off')
        plt.tight_layout()
        out_path = os.path.join(OUTPUT_DIR, f"img_{i:03d}.png")
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Image {i:03d} | PSNR: {psnr_val:6.2f} dB, SSIM: {ssim_val:5.3f} | saved -> {out_path}")

    avg_psnr /= n
    avg_ssim /= n
    print("=" * 60)
    print(f"Average   | PSNR: {avg_psnr:6.2f} dB, SSIM: {avg_ssim:5.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
