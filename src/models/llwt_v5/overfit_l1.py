"""LLW-Former v0.3.x pure-L1 overfit (capacity diagnostic).

Trains the generator alone on a single batch with ONLY a pixel L1 loss --
no D, no FM, no LSGAN signal.  Purpose is to measure G's architectural
ceiling on the overfit problem without adversarial gradient noise.

Compare against ``overfit_test.py`` (GAN+FM, same iters / batch / LR):
  * If pure-L1 PSNR is much higher, the GAN+FM stack is leaving pixel
    capacity on the table -- L1 anchor would help.
  * If pure-L1 PSNR is similar, the architecture itself caps at that
    point (input/feature representation, decoder, etc.).

Run from repo root::

    python -m src.models.llwt_v3.overfit_l1
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torchmetrics.image import PeakSignalNoiseRatio


ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT))


from src.data.sen12_full.dataset import SEN12Full
from src.data.transforms import (
    get_common_transform, get_input_specific, get_optical_specific,
    get_resize_transform, get_sar_resize_transform,
)
from src.models.llwt_v5 import factory


CONFIG_PATH = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / 'src' / 'models' / 'llwt_v45' / 'config.yaml')
ITERATIONS = 3000                      # arch-ceiling probe — long enough to plateau
BATCH_SIZE = 4
LR_OVERRIDE = 5.0e-4                   # 5x cfg.optimizer.lr_g to remove LR as bottleneck
PSNR_TARGET = 30.0                     # pure L1 should easily clear this on 1 batch


def main():
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[overfit-l1] device = {device}")

    cfg = OmegaConf.load(CONFIG_PATH)
    sar_channels = int(cfg.data.sar_channels)
    image_size = int(cfg.data.image_size)
    first_scene = [list(cfg.data.scenes)[0]]
    print(f"[overfit-l1] cfg: scenes={first_scene} bs={BATCH_SIZE} img={image_size} "
          f"lr_g={cfg.optimizer.lr_g}")

    dataset = SEN12Full(
        root_dir=str(ROOT / 'data' / 'sen12_full'),
        common_transform=get_common_transform(),
        input_specific=get_input_specific(sar_channels=sar_channels),
        optical_specific=get_optical_specific(),
        resize_transform=get_resize_transform(image_size),
        sar_resize_transform=get_sar_resize_transform(image_size),
        sar_channels=sar_channels,
        scenes=first_scene,
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    sar, opt = next(iter(loader))
    sar = sar.to(device)
    opt = opt.to(device)
    print(f"[overfit-l1] batch: sar={tuple(sar.shape)} opt={tuple(opt.shape)}")

    # Build G only -- skip D entirely.
    netG, _netD = factory.build_models(cfg)
    netG = netG.to(device)
    n_g = sum(p.numel() for p in netG.parameters())
    print(f"[overfit-l1] G params = {n_g/1e6:.2f}M")

    # LR override removes LR as a confound: if pure L1 at 5e-4 / 3000 iters
    # still plateaus at ~25 dB, the cap is architectural, not optimization.
    lr_use = LR_OVERRIDE if LR_OVERRIDE else float(cfg.optimizer.lr_g)
    print(f"[overfit-l1] lr_use={lr_use} (override={LR_OVERRIDE is not None})")
    optimizer = torch.optim.AdamW(
        netG.parameters(),
        lr=lr_use,
        betas=(float(cfg.optimizer.beta1), float(cfg.optimizer.beta2)),
        weight_decay=float(cfg.optimizer.weight_decay_g),
        fused=True,
    )
    criterion = nn.L1Loss()
    psnr_fn = PeakSignalNoiseRatio(data_range=2.0).to(device)
    grad_clip_g = float(cfg.loss.grad_clip_g)
    use_bf16 = str(cfg.system.precision).startswith('bf16')

    print(f"[overfit-l1] bf16={use_bf16} grad_clip_g={grad_clip_g}")
    print("-" * 78)

    init_psnr = None
    last_psnr = -float('inf')
    netG.train()
    for step in range(1, ITERATIONS + 1):
        with torch.amp.autocast(device_type='cuda', enabled=use_bf16, dtype=torch.bfloat16):
            fake = netG(sar)
            l_l1 = criterion(fake, opt)
        optimizer.zero_grad()
        l_l1.backward()
        if grad_clip_g > 0:
            torch.nn.utils.clip_grad_norm_(netG.parameters(), grad_clip_g)
        optimizer.step()

        if step == 1 or step % 100 == 0 or step == ITERATIONS:
            with torch.no_grad():
                with torch.amp.autocast(device_type='cuda', enabled=use_bf16, dtype=torch.bfloat16):
                    fake_eval = netG(sar)
                psnr_val = float(psnr_fn(fake_eval.float(), opt.float()).item())
            print(f"  step {step:3d} | L1 = {float(l_l1):.4f} | PSNR = {psnr_val:5.2f} dB")
            if init_psnr is None:
                init_psnr = psnr_val
            last_psnr = psnr_val

    psnr_pass = last_psnr >= PSNR_TARGET
    print("=" * 78)
    print(f"  init PSNR : {init_psnr:.2f} dB")
    print(f"  final PSNR: {last_psnr:.2f} dB     (target >= {PSNR_TARGET:.1f})")
    print(f"  RESULT    : {'PASS' if psnr_pass else 'FAIL'}")
    print("=" * 78)
    if not psnr_pass:
        sys.exit(1)


if __name__ == '__main__':
    main()
