"""Canonical full-val evaluation of the final LLW-Former under ONE protocol.

Recomputes PSNR / SSIM / LPIPS / FID for the production checkpoint on the FULL
validation split, with a single, explicit convention so the number is directly
comparable across tables:

  * generator weights = ``ckpt['state_dict']`` (EMA-or-live), ``netG.`` stripped;
  * PSNR/SSIM on [0,1] images ((x+1)/2, clamped), data_range=1.0;
  * LPIPS on [-1,1] images, normalize=False (AlexNet);
  * FID = ONE global pool over the whole val set (not per-scene), normalize=True,
    feature=2048 — fakes vs real optical.

All metrics accumulate over every batch and reduce once at the end.

Run from repo root::

    python -m src.models.llwt_v5.eval_full
"""
import os
import time

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

import torch
from omegaconf import OmegaConf
from torchmetrics.image import (
    PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure,
    FrechetInceptionDistance,
)
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from src.models.llwt_v5 import factory
from src.models.llwt_v5.inference import _build_datamodule


CHECKPOINT = "checkpoints/llwt_v45/llwt-v0.5.1-hfd/epoch=097-psnr=17.1615.ckpt"
SPLIT = "val"
DATASET = None  # None = use config; or override: "sen12_full" / "sen12_full_align"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    cfg = OmegaConf.load('./src/models/llwt_v5/config.yaml')
    if DATASET is not None:
        cfg.data.dataset = DATASET
    cfg.data.num_workers = 0  # Windows-safe; full pass is GPU-bound anyway

    dm = _build_datamodule(cfg)
    dm.setup("fit")
    loader = dm.val_dataloader() if SPLIT == "val" else dm.train_dataloader()

    ckpt = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
    netG, _ = factory.build_models(cfg)
    sd = ckpt['state_dict']
    netG.load_state_dict({k[len('netG.'):]: v for k, v in sd.items() if k.startswith('netG.')})
    netG = netG.to(DEVICE).eval()

    psnr = PeakSignalNoiseRatio(data_range=1.0).to(DEVICE)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)
    lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=False).to(DEVICE)
    fid = FrechetInceptionDistance(feature=2048, reset_real_features=True, normalize=True).to(DEVICE)

    n = 0
    t0 = time.time()
    with torch.no_grad():
        for bi, (sar, opt) in enumerate(loader):
            sar, opt = sar.to(DEVICE), opt.to(DEVICE)
            fake = netG(sar)

            gen01 = ((fake.clamp(-1, 1) + 1) / 2).float()
            opt01 = ((opt + 1) / 2).clamp(0, 1).float()

            psnr.update(gen01, opt01)
            ssim.update(gen01, opt01)
            lpips.update(fake.clamp(-1, 1).float(), opt.float())  # [-1,1]
            fid.update(opt01, real=True)
            fid.update(gen01, real=False)

            n += sar.size(0)
            if bi % 20 == 0:
                print(f"  ... {n} pairs ({time.time() - t0:.0f}s)")

    res = {
        'PSNR': float(psnr.compute()),
        'SSIM': float(ssim.compute()),
        'LPIPS': float(lpips.compute()),
        'FID': float(fid.compute()),
    }
    print("=" * 60)
    print(f"dataset={cfg.data.dataset} split={SPLIT} pairs={n} ckpt={os.path.basename(CHECKPOINT)}")
    print(f"protocol: PSNR/SSIM [0,1] dr=1.0 | LPIPS [-1,1] | FID global pool normalize=True")
    print(f"  PSNR  = {res['PSNR']:.2f} dB")
    print(f"  SSIM  = {res['SSIM']:.4f}")
    print(f"  LPIPS = {res['LPIPS']:.4f}")
    print(f"  FID   = {res['FID']:.2f}")
    print(f"  ({time.time() - t0:.0f}s total)")
    print("=" * 60)


if __name__ == "__main__":
    main()
