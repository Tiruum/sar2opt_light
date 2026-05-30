"""Full-validation metric eval for the ConvNeXtV2+U-Net baseline (huggingface_gan).

Computes PSNR / SSIM / LPIPS / FID over the ENTIRE 5-scene validation split
(the same split the model trained/validated on) using the best-PSNR checkpoint.
Training only ever logged PSNR/SSIM; this script adds the missing LPIPS+FID so
the thesis §5.3.1 baseline numbers are verifiable.

Run from repo root::

    python -m src.models.huggingface_gan.eval_metrics
"""
import os

# Offline-HF guards (backbone cached from training); mirrors llwt_v5/inference.py.
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
if os.environ.get('HF_HUB_OFFLINE') == '1':
    import transformers.models.auto.auto_factory as _af
    _af.repo_exists = lambda *a, **k: True

import torch
from omegaconf import OmegaConf
from torchmetrics.image import (
    PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure,
    LearnedPerceptualImagePatchSimilarity,
)
from torchmetrics.image.fid import FrechetInceptionDistance

from src.models.huggingface_gan.gen_legacy import HFGenerator  # babb3d0c arch (bilinear, matches 800-ep ckpts)
from src.data.sen12_full.datamodule import SEN12FullDataModule

CHECKPOINT = "checkpoints/huggingface-gan/epoch=798-psnr=17.9483.ckpt"  # 800-ep run, best val/psnr; = ablation v0
CONFIG_PATH = './src/models/huggingface_gan/config.yaml'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    cfg = OmegaConf.load(CONFIG_PATH)

    dm = SEN12FullDataModule(
        data_dir=cfg.data.data_dir.sen12_full,
        batch_size=cfg.data.batch_size,
        image_size=cfg.data.image_size,
        num_workers=cfg.data.num_workers,
        persistent_workers=cfg.data.persistent_workers,
        prefetch_factor=cfg.data.prefetch_factor,
        train_val_split_ratio=cfg.data.train_val_split_ratio,
        seed=cfg.data.seed,
        sar_channels=cfg.data.sar_channels,
        use_augmentation=cfg.data.use_train_common_transform,
        scenes=list(cfg.data.scenes),
    )
    dm.setup("fit")
    loader = dm.val_dataloader()

    ckpt = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
    netG = HFGenerator(cfg)
    state_dict = {k[len('netG.'):]: v for k, v in ckpt['state_dict'].items() if k.startswith('netG.')}
    netG.load_state_dict(state_dict)
    netG = netG.to(DEVICE).eval()
    print(f"[ckpt] {len(state_dict)} netG tensors from {CHECKPOINT}")

    psnr_m = PeakSignalNoiseRatio(data_range=1.0).to(DEVICE)
    ssim_m = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)
    lpips_m = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=False).to(DEVICE)
    fid_m = FrechetInceptionDistance(feature=2048, normalize=True).to(DEVICE)

    n = 0
    with torch.no_grad():
        for sar, opt in loader:
            sar = sar.to(DEVICE)
            opt = opt.to(DEVICE)
            gen = netG(sar).clamp(-1, 1)

            gen01 = (gen + 1.0) / 2.0
            opt01 = (opt + 1.0) / 2.0

            psnr_m.update(gen01, opt01)
            ssim_m.update(gen01, opt01)
            lpips_m.update(gen.float(), opt.float())
            fid_m.update(opt01.float(), real=True)
            fid_m.update(gen01.float(), real=False)
            n += sar.size(0)

    print("=" * 60)
    print(f"ConvNeXtV2+U-Net baseline — full val ({n} samples, 5 scenes)")
    print("=" * 60)
    print(f"PSNR : {psnr_m.compute().item():.2f} dB")
    print(f"SSIM : {ssim_m.compute().item():.4f}")
    print(f"LPIPS: {lpips_m.compute().item():.4f}")
    print(f"FID  : {fid_m.compute().item():.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
