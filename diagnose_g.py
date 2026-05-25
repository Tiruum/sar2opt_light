import torch
import torch.nn as nn
import os
from omegaconf import OmegaConf
from src.models.huggingface_gan.main import SAR2OPTLightningModule
from src.data.sen12_full.datamodule import SEN12FullDatamodule

# Load config
cfg = OmegaConf.load('src/models/huggingface_gan/config.yaml')

# Load checkpoint
ckpt_path = 'checkpoints/huggingface-gan/hfgan-14/last.ckpt'
module = SAR2OPTLightningModule.load_from_checkpoint(ckpt_path, cfg=cfg)
netG = module.netG.eval()
device = torch.device('cuda')
netG = netG.to(device)

# Setup dataset
dm = SEN12FullDatamodule(cfg)
dm.setup('validate')
val_loader = dm.val_dataloader()

# Get one batch
batch = next(iter(val_loader))
sar, opt = batch
sar, opt = sar.to(device), opt.to(device)

print(f"Input SAR shape: {sar.shape}, range: [{sar.min():.4f}, {sar.max():.4f}]")
print(f"Target OPT shape: {opt.shape}, range: [{opt.min():.4f}, {opt.max():.4f}]")

# Forward pass
with torch.no_grad():
    fake = netG(sar)

print(f"\n=== GENERATOR OUTPUT ANALYSIS ===")
print(f"Fake shape: {fake.shape}")
print(f"Fake range: [{fake.min():.6f}, {fake.max():.6f}]")
print(f"Fake mean: {fake.mean():.6f}")
print(f"Fake std:  {fake.std():.6f}")

# Per-channel stats
for c in range(fake.shape[1]):
    ch = fake[:, c]
    print(f"Channel {c}: min={ch.min():.6f}, max={ch.max():.6f}, mean={ch.mean():.6f}, std={ch.std():.6f}")

# Spatial variation (std across H,W per channel per sample)
print(f"\nSpatial variation (std across H,W):")
for b in range(min(2, fake.shape[0])):
    for c in range(fake.shape[1]):
        spatial_std = fake[b, c].std().item()
        print(f"  Sample {b}, Ch {c}: {spatial_std:.6f}")

# Unique values / saturation
unique_count = len(torch.unique(fake))
saturated_pos = (fake > 0.99).sum().item()
saturated_neg = (fake < -0.99).sum().item()
total_pixels = fake.numel()
print(f"\nUnique values: {unique_count} / {total_pixels}")
print(f"Pixels at +1 (≥0.99): {saturated_pos} ({100*saturated_pos/total_pixels:.2f}%)")
print(f"Pixels at -1 (≤-0.99): {saturated_neg} ({100*saturated_neg/total_pixels:.2f}%)")

# Compare vs target: MSE, optimal constant output PSNR
mse = ((fake - opt) ** 2).mean().item()
print(f"\nMSE vs target: {mse:.6f}")
print(f"Implied PSNR: {10 * torch.log10(torch.tensor(4.0 / mse)):.3f} dB")

# Optimal constant output = mean of target
mean_opt = opt.mean().item()
mse_const = ((mean_opt - opt) ** 2).mean().item()
psnr_const = 10 * torch.log10(torch.tensor(4.0 / mse_const))
print(f"Optimal constant output (mean={mean_opt:.4f}): MSE={mse_const:.6f}, PSNR={psnr_const:.3f} dB")

# Is fake near constant?
fake_std_spatial = fake.view(fake.shape[0], fake.shape[1], -1).std(dim=2).mean()
print(f"\nFake spatial std (avg across C,B): {fake_std_spatial:.6f}")

# Contrast: fake std / target std
target_std = opt.view(opt.shape[0], opt.shape[1], -1).std(dim=2).mean()
print(f"Target spatial std (avg across C,B): {target_std:.6f}")
print(f"Contrast ratio fake/target: {(fake_std_spatial / target_std):.4f}x")
