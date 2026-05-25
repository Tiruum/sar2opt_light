"""Rigorous hallucination audit for SAR-to-optical generators.

Quantifies how much a trained generator INVENTS structures that are NOT
supported by the SAR input.  Compares G output's structural alignment to
SAR against ground-truth OPT's structural alignment to SAR — if real OPT
preserves SAR edges more than fake OPT does, the generator is hallucinating.

Reports five metrics per image + aggregate:

  1. ``edge_iou_real`` — IoU(sobel(SAR), sobel(luma(opt_real)))
  2. ``edge_iou_fake`` — IoU(sobel(SAR), sobel(luma(opt_fake)))
  3. ``hall_score``  — (iou_real - iou_fake) / max(iou_real, 1e-6)
                        > 0 = fake STRUCTURALLY diverges from SAR more
                              than real does (i.e. hallucinating structure)
                        ~ 0 = structurally aligned with real
                        < 0 = fake actually MORE SAR-aligned than real (unusual)
  4. ``pix_corr_real`` — Pearson r between |SAR| and luma(opt_real)
  5. ``pix_corr_fake`` — Pearson r between |SAR| and luma(opt_fake)
                        Both should be similar magnitude; large gap = mismatch.
  6. ``color_kl``     — KL divergence between (opt_fake, opt_real) per-channel
                        histograms.  High = color bias / palette drift.
  7. ``psnr``, ``ssim`` — reference quality metrics.

Outputs:

  ./output/hallucination_audit/<model_tag>/metrics.csv
  ./output/hallucination_audit/<model_tag>/triplet_grid.png
  ./output/hallucination_audit/<model_tag>/edge_overlay_*.png

Usage::

    python scripts/hallucination_audit.py --model v4 \\
        --ckpt checkpoints/llwt_v4/llwt-v0.4.1-perband-r2-detail/epoch=056-psnr=14.2607.ckpt \\
        --n_batches 6

    python scripts/hallucination_audit.py --model hfgan \\
        --ckpt checkpoints/huggingface-gan/hfgan-18/last.ckpt \\
        --n_batches 6
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))


from src.data.sen12_full.datamodule import SEN12FullDataModule


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ---------------------------------------------------------------------------
# Model loaders — keep one entry per supported model.  Each returns a callable
# ``forward(sar) -> opt_fake`` plus the matching DataModule cfg.
# ---------------------------------------------------------------------------


def _strip_prefix(state_dict, prefix='netG.'):
    return {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}


def load_v4(ckpt_path: str, config_path: str = 'src/models/llwt_v4/config.yaml'):
    from src.models.llwt_v4.gen import LLWv4Generator
    cfg = OmegaConf.load(config_path)
    netG = LLWv4Generator(cfg=cfg)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    # ``state_dict`` is the EMA-averaged version that produced the saved val
    # metric; ``current_model_state`` is the LIVE training-step weights
    # (lower-quality outputs).  Use the EMA version to reproduce training-
    # time val PSNR.
    sd = ckpt.get('state_dict', ckpt)
    g_sd = _strip_prefix(sd, 'netG.')
    missing, unexpected = netG.load_state_dict(g_sd, strict=False)
    print(f"[v4] loaded {ckpt_path}")
    print(f"     missing={len(missing)} unexpected={len(unexpected)}")
    # Match training context: channels_last + cuda; bf16 autocast applied
    # inside the audit loop so metric ops still get fp32 inputs.
    netG = netG.to(DEVICE).to(memory_format=torch.channels_last).eval()
    return netG, cfg


def load_hfgan(ckpt_path: str, config_path: str = 'src/models/huggingface_gan/config.yaml'):
    from src.models.huggingface_gan.gen import HFGenerator
    cfg = OmegaConf.load(config_path)
    netG = HFGenerator(cfg)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = ckpt.get('state_dict', ckpt)
    g_sd = _strip_prefix(sd, 'netG.')
    missing, unexpected = netG.load_state_dict(g_sd, strict=False)
    print(f"[hfgan] loaded {ckpt_path}")
    print(f"        missing={len(missing)} unexpected={len(unexpected)}")
    netG = netG.to(DEVICE).eval()
    return netG, cfg


MODEL_REGISTRY = {
    'v4':    load_v4,
    'hfgan': load_hfgan,
}


# ---------------------------------------------------------------------------
# Metric primitives.  Pure tensor ops, no torchmetrics dep so it runs even
# in a stripped-down env.
# ---------------------------------------------------------------------------


def luma(rgb: torch.Tensor) -> torch.Tensor:
    """ITU-R BT.601 luma from RGB in [-1, 1] -> [-1, 1] scalar channel."""
    r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
    return 0.299 * r + 0.587 * g + 0.114 * b


_SOBEL_X = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
_SOBEL_Y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)


def sobel_magnitude(x: torch.Tensor) -> torch.Tensor:
    """Per-channel Sobel gradient magnitude.  Returns (B, 1, H, W)."""
    # Reduce to single channel via luma if multi-channel.
    if x.size(1) == 3:
        x = luma(x)
    sx = F.conv2d(x, _SOBEL_X.to(x.device, x.dtype), padding=1)
    sy = F.conv2d(x, _SOBEL_Y.to(x.device, x.dtype), padding=1)
    return torch.sqrt(sx * sx + sy * sy + 1e-12)


def edge_mask(mag: torch.Tensor, percentile: float = 75.0) -> torch.Tensor:
    """Binarise a gradient magnitude map at the per-image percentile threshold."""
    B = mag.size(0)
    flat = mag.view(B, -1)
    thr = torch.quantile(flat, percentile / 100.0, dim=1, keepdim=True)
    thr = thr.view(B, 1, 1, 1)
    return (mag > thr).float()


def iou(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Per-image IoU between two binary masks.  Returns (B,)."""
    a = a.float()
    b = b.float()
    inter = (a * b).sum(dim=(1, 2, 3))
    union = ((a + b) - a * b).sum(dim=(1, 2, 3))
    return inter / (union + eps)


def pixel_correlation(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Per-image Pearson correlation between two single-channel maps.  Returns (B,)."""
    B = x.size(0)
    xf = x.view(B, -1)
    yf = y.view(B, -1)
    xf = xf - xf.mean(dim=1, keepdim=True)
    yf = yf - yf.mean(dim=1, keepdim=True)
    num = (xf * yf).sum(dim=1)
    den = torch.sqrt((xf * xf).sum(dim=1) * (yf * yf).sum(dim=1) + eps)
    return num / (den + eps)


def color_kl(fake: torch.Tensor, real: torch.Tensor, bins: int = 32) -> torch.Tensor:
    """Per-image symmetric KL between (fake, real) RGB histograms.  Returns (B,)."""
    B = fake.size(0)
    out = torch.zeros(B, device=fake.device)
    eps = 1e-8
    for b in range(B):
        kl_acc = 0.0
        for c in range(3):
            fh = torch.histc(fake[b, c], bins=bins, min=-1.0, max=1.0) + eps
            rh = torch.histc(real[b, c], bins=bins, min=-1.0, max=1.0) + eps
            fh = fh / fh.sum()
            rh = rh / rh.sum()
            kl = (fh * (fh.log() - rh.log())).sum() + (rh * (rh.log() - fh.log())).sum()
            kl_acc += float(kl)
        out[b] = kl_acc / 3.0
    return out


def psnr_db(fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
    """Per-image PSNR in dB on [-1, 1] tensors (data_range=2)."""
    mse = (fake - real).pow(2).mean(dim=(1, 2, 3))
    return 10.0 * torch.log10(4.0 / mse.clamp_min(1e-12))


# ---------------------------------------------------------------------------
# Audit loop.
# ---------------------------------------------------------------------------


def run_audit(
    model_tag: str,
    ckpt_path: str,
    n_batches: int,
    output_dir: Path,
    seed: int = 42,
):
    if model_tag not in MODEL_REGISTRY:
        raise ValueError(
            f"model_tag must be one of {list(MODEL_REGISTRY.keys())}, got {model_tag}"
        )
    loader_fn = MODEL_REGISTRY[model_tag]
    netG, cfg = loader_fn(ckpt_path)

    # Build val loader using each model's own cfg.
    dm = SEN12FullDataModule(
        data_dir=cfg.data.data_dir.sen12_full,
        batch_size=cfg.data.get('val_batch_size', cfg.data.batch_size),
        image_size=cfg.data.image_size,
        num_workers=0,                              # avoid Windows worker spawn cost
        persistent_workers=False,
        prefetch_factor=2,
        train_val_split_ratio=cfg.data.train_val_split_ratio,
        seed=seed,
        sar_channels=cfg.data.sar_channels,
        use_augmentation=cfg.data.use_train_common_transform,
        scenes=list(cfg.data.scenes),
        train_crop_size=None,
        val_batch_size=cfg.data.get('val_batch_size', None),
    )
    dm.setup('fit')
    val_loader = dm.val_dataloader()

    rows = []
    saved_triplets = []
    saved_overlays = []

    print(f"[audit] running on val ({n_batches} batches)")
    with torch.no_grad():
        for batch_idx, (sar, opt) in enumerate(val_loader):
            if batch_idx >= n_batches:
                break
            sar = sar.to(DEVICE).contiguous(memory_format=torch.channels_last)
            opt = opt.to(DEVICE).contiguous(memory_format=torch.channels_last)
            # bf16 autocast mirrors the training/val precision regime so the
            # frozen ckpt produces the same logits it did during training.
            with torch.amp.autocast(device_type='cuda', enabled=True, dtype=torch.bfloat16):
                fake = netG(sar)
                if isinstance(fake, (tuple, list)):
                    fake = fake[0]
            fake = fake.float().clamp(-1.0, 1.0)
            sar  = sar.float()
            opt  = opt.float()

            # Bring SAR to a comparable "structural" channel for sobel.
            sar_struct = sar
            if sar_struct.size(1) == 3:
                sar_struct = luma(sar_struct)
            sar_mag  = sobel_magnitude(sar_struct)
            real_mag = sobel_magnitude(opt)
            fake_mag = sobel_magnitude(fake)

            sar_em  = edge_mask(sar_mag)
            real_em = edge_mask(real_mag)
            fake_em = edge_mask(fake_mag)

            iou_real_b = iou(sar_em, real_em)
            iou_fake_b = iou(sar_em, fake_em)
            hall_b     = (iou_real_b - iou_fake_b) / iou_real_b.clamp_min(1e-6)

            sar_norm  = sar_struct.abs()
            corr_real = pixel_correlation(sar_norm, luma(opt))
            corr_fake = pixel_correlation(sar_norm, luma(fake))

            kl_b      = color_kl(fake, opt)
            psnr_b    = psnr_db(fake, opt)

            for i in range(sar.size(0)):
                rows.append({
                    'batch': batch_idx,
                    'image': i,
                    'edge_iou_real': float(iou_real_b[i]),
                    'edge_iou_fake': float(iou_fake_b[i]),
                    'hall_score':    float(hall_b[i]),
                    'pix_corr_real': float(corr_real[i]),
                    'pix_corr_fake': float(corr_fake[i]),
                    'color_kl':      float(kl_b[i]),
                    'psnr':          float(psnr_b[i]),
                })

            # Save first batch's triplets + overlays for visual sanity.
            if batch_idx == 0:
                saved_triplets.append((sar.cpu(), fake.cpu(), opt.cpu()))
                saved_overlays.append(
                    (sar.cpu(), sar_em.cpu(), real_em.cpu(), fake_em.cpu())
                )

    # ------------------------------------------------------------------ write
    output_dir.mkdir(parents=True, exist_ok=True)
    import csv
    csv_path = output_dir / 'metrics.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Aggregate report.
    def _agg(key):
        vals = np.array([r[key] for r in rows], dtype=np.float64)
        return float(vals.mean()), float(vals.std())

    report = {k: _agg(k) for k in (
        'edge_iou_real', 'edge_iou_fake', 'hall_score',
        'pix_corr_real', 'pix_corr_fake', 'color_kl', 'psnr',
    )}

    print(f"\n[audit] aggregate over {len(rows)} images:")
    for k, (m, s) in report.items():
        print(f"  {k:18s} = {m:+.4f}  ± {s:.4f}")

    # Save visualizations.
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        sar_b, fake_b, opt_b = saved_triplets[0]
        N = min(4, sar_b.size(0))
        fig, axes = plt.subplots(N, 3, figsize=(9, 3 * N))
        for i in range(N):
            if N == 1:
                row = axes
            else:
                row = axes[i]
            row[0].imshow(sar_b[i, 0].numpy(), cmap='gray', vmin=-1, vmax=1)
            row[0].set_title('SAR'); row[0].axis('off')
            row[1].imshow(((fake_b[i].permute(1, 2, 0).numpy() + 1) * 0.5).clip(0, 1))
            row[1].set_title(f'{model_tag} fake'); row[1].axis('off')
            row[2].imshow(((opt_b[i].permute(1, 2, 0).numpy() + 1) * 0.5).clip(0, 1))
            row[2].set_title('OPT real'); row[2].axis('off')
        plt.tight_layout()
        fig.savefig(output_dir / 'triplet_grid.png', dpi=120)
        plt.close(fig)

        sar_b, sar_em, real_em, fake_em = saved_overlays[0]
        N = min(4, sar_b.size(0))
        fig, axes = plt.subplots(N, 4, figsize=(12, 3 * N))
        for i in range(N):
            row = axes if N == 1 else axes[i]
            row[0].imshow(sar_b[i, 0].numpy(), cmap='gray', vmin=-1, vmax=1)
            row[0].set_title('SAR'); row[0].axis('off')
            row[1].imshow(sar_em[i, 0].numpy(), cmap='gray', vmin=0, vmax=1)
            row[1].set_title('SAR edges'); row[1].axis('off')
            row[2].imshow(real_em[i, 0].numpy(), cmap='gray', vmin=0, vmax=1)
            row[2].set_title('OPT real edges'); row[2].axis('off')
            row[3].imshow(fake_em[i, 0].numpy(), cmap='gray', vmin=0, vmax=1)
            row[3].set_title(f'{model_tag} fake edges'); row[3].axis('off')
        plt.tight_layout()
        fig.savefig(output_dir / 'edge_overlay.png', dpi=120)
        plt.close(fig)
        print(f"[audit] viz saved -> {output_dir}/triplet_grid.png, edge_overlay.png")
    except Exception as e:
        print(f"[audit] viz step skipped: {e}")

    print(f"[audit] metrics csv -> {csv_path}")
    return report


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, choices=list(MODEL_REGISTRY.keys()),
                        help='Model tag: v4 (LLW-Former v0.4.x) or hfgan (pure ConvNeXt V2).')
    parser.add_argument('--ckpt', required=True,
                        help='Lightning ckpt path for the generator.')
    parser.add_argument('--n_batches', type=int, default=6,
                        help='Number of val batches to audit (default 6 -> ~24 images at bs=4).')
    parser.add_argument('--output_dir', default=None,
                        help='Override output dir.  Default: output/hallucination_audit/<model_tag>/')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    out = Path(args.output_dir) if args.output_dir else \
          ROOT / 'output' / 'hallucination_audit' / args.model
    run_audit(
        model_tag=args.model,
        ckpt_path=args.ckpt,
        n_batches=args.n_batches,
        output_dir=out,
        seed=args.seed,
    )
