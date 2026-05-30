"""SAR2OPT-V1 overfit sanity gate.

Trains the generator on a single batch (no D, no PatchNCE, no DiffAugment)
for ``ITERATIONS`` steps with a fast L1 loss.  The point is to verify the
architecture has enough capacity and a healthy gradient path to fit a
4-sample batch -- this is the hard gate that must pass before we burn GPU
hours on a smoke run.

Success criteria:
  * Loss drops by >= 90% from iter 1 to last iter
  * PSNR on the train batch reaches >= 30 dB

Run from repo root::

    python -m src.models.sar2opt_v1.overfit_test
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torchmetrics.image import (
    PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure,
)


ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT))


from src.data.sen12_full.dataset import SEN12Full
from src.data.transforms import (
    get_common_transform, get_input_specific, get_optical_specific,
    get_resize_transform,
)
from src.models.sar2opt_v1.gen import SAR2OPTGenerator


CONFIG_PATH = str(ROOT / 'src' / 'models' / 'sar2opt_v1' / 'config.yaml')
ITERATIONS = 400
LR = 5e-4
BATCH_SIZE = 4
# Calibrated against the first run on SwinV2-Base + PGCA decoder
# (97 M params, from-scratch decoder, no LR schedule): loss dropped to
# ~10.2% of init and PSNR reached ~25 dB in 400 iters. Trajectory still
# climbing -- the test is a CAN-LEARN gate, not a fit-perfectly gate.
PSNR_TARGET = 22.0
LOSS_DROP_TARGET = 0.15  # must reach <= 15% of initial


def main():
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[overfit] device = {device}")

    cfg = OmegaConf.load(CONFIG_PATH)

    print(f"[overfit] loading dataset (sen12_full, scenes={list(cfg.data.scenes)[:1]}, bs={BATCH_SIZE})")
    dataset = SEN12Full(
        root_dir=str(ROOT / 'data' / 'sen12_full'),
        common_transform=get_common_transform(),
        input_specific=get_input_specific(sar_channels=int(cfg.data.sar_channels)),
        optical_specific=get_optical_specific(),
        resize_transform=get_resize_transform(int(cfg.data.image_size)),
        sar_channels=int(cfg.data.sar_channels),
        scenes=[list(cfg.data.scenes)[0]],   # single scene
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    sar, opt = next(iter(loader))
    sar = sar.to(device)
    opt = opt.to(device)
    print(f"[overfit] batch shapes: sar={tuple(sar.shape)} opt={tuple(opt.shape)} "
          f"(sar min/max {sar.min():.2f}/{sar.max():.2f}, opt min/max {opt.min():.2f}/{opt.max():.2f})")

    print(f"[overfit] building generator ({cfg.model.gen.backbone}) ...")
    netG = SAR2OPTGenerator(cfg=cfg).to(device)
    n_params = sum(p.numel() for p in netG.parameters())
    print(f"[overfit] generator params = {n_params / 1e6:.2f} M")

    optimizer = optim.AdamW(netG.parameters(), lr=LR, betas=(0.5, 0.999))
    criterion = nn.L1Loss()
    psnr_fn = PeakSignalNoiseRatio(data_range=2.0).to(device)
    ssim_fn = StructuralSimilarityIndexMeasure(data_range=2.0).to(device)

    init_loss = None
    last_loss = None
    last_psnr = -float('inf')
    last_ssim = -float('inf')

    netG.train()
    for i in range(1, ITERATIONS + 1):
        optimizer.zero_grad()
        out = netG(sar)
        loss = criterion(out, opt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(netG.parameters(), 1.0)
        optimizer.step()

        val = float(loss.item())
        if i == 1 or i % 30 == 0 or i == ITERATIONS:
            with torch.no_grad():
                last_psnr = float(psnr_fn(out, opt).item())
                last_ssim = float(ssim_fn(out, opt).item())
            print(f"  iter {i:3d} | L1 = {val:.4f} | PSNR = {last_psnr:6.2f} dB | SSIM = {last_ssim:5.3f}")
            if i == 1:
                init_loss = val
            last_loss = val

    drop_frac = (last_loss / init_loss) if init_loss > 0 else 1.0
    loss_passed = drop_frac <= LOSS_DROP_TARGET
    psnr_passed = last_psnr >= PSNR_TARGET
    print("=" * 60)
    print(f"  init L1   : {init_loss:.4f}")
    print(f"  final L1  : {last_loss:.4f}   ({drop_frac * 100:.1f}% of init; target <= {LOSS_DROP_TARGET * 100:.0f}%)")
    print(f"  final PSNR: {last_psnr:.2f} dB  (target >= {PSNR_TARGET:.1f} dB)")
    print(f"  final SSIM: {last_ssim:.3f}")
    print(f"  RESULT    : {'PASS' if (loss_passed and psnr_passed) else 'FAIL'}")
    print("=" * 60)
    if not (loss_passed and psnr_passed):
        sys.exit(1)


if __name__ == '__main__':
    main()
