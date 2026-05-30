"""Side-by-side comparison: llwt_v45 baseline vs llwt_v5 (HF-D) generator.

Runs the SAME validation samples through BOTH generators, computes per-sample
PSNR/SSIM/LPIPS for each, and saves the TOP_K panels where v5 most improves
LPIPS over the baseline (i.e. where the high-frequency discriminator's
sharpening is most visible). Also prints full-split aggregate means for both
checkpoints — the honest headline comparison.

Panel layout per sample::

    [ SAR input | v45 baseline | v5 (HF-D) | GT optical ]

with PSNR/SSIM/LPIPS annotated under each generated image.

Run from repo root::

    python -m src.models.llwt_v5.compare_v45_v5
"""
import os
import heapq

# Same offline-HF guards as inference.py — the backbone is cached from training.
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
if os.environ.get('HF_HUB_OFFLINE') == '1':
    import transformers.models.auto.auto_factory as _af
    _af.repo_exists = lambda *a, **k: True

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from omegaconf import OmegaConf
from torchmetrics.image import (
    PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure,
    LearnedPerceptualImagePatchSimilarity,
)

from src.models.llwt_v5.gen import LLWv4Generator


CKPT_V45 = "checkpoints/llwt_v45/llwt-v0.4.6-align-ws/last.ckpt"          # baseline (warm-start parent)
CKPT_V5  = "checkpoints/llwt_v45/llwt-v0.5.1-hfd/epoch=097-psnr=17.1615.ckpt"  # HF-D, best PSNR
TOP_K    = 8                      # panels to save (largest LPIPS improvement)
SPLIT    = "val"
OUTPUT_DIR = f"./src/models/llwt_v5/output/compare_{SPLIT}"
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"


def _build_datamodule(cfg):
    """Dispatch on cfg.data.dataset — mirrors inference.py."""
    dataset_name = str(cfg.data.dataset)
    if dataset_name == 'sen12_full':
        from src.data.sen12_full.datamodule import SEN12FullDataModule as _DM
    elif dataset_name == 'sen12_full_align':
        from src.data.sen12_full_align.datamodule import SEN12FullDataModule as _DM
    elif dataset_name == 'qxs_saropt':
        from src.data.qxs_saropt.datamodule import QXSSAROPTDataModule as _DM
    else:
        raise ValueError(f"cfg.data.dataset='{dataset_name}' unsupported")
    data_root = cfg.data.data_dir[dataset_name]
    scenes_arg = None if dataset_name == 'qxs_saropt' else list(cfg.data.scenes)
    dm_kwargs = dict(
        data_dir=data_root,
        batch_size=cfg.data.get('val_batch_size', cfg.data.batch_size),
        image_size=cfg.data.image_size,
        num_workers=cfg.data.num_workers,
        persistent_workers=cfg.data.persistent_workers,
        prefetch_factor=cfg.data.prefetch_factor,
        train_val_split_ratio=cfg.data.train_val_split_ratio,
        seed=cfg.data.seed,
        sar_channels=cfg.data.sar_channels,
        use_augmentation=cfg.data.use_train_common_transform,
        scenes=scenes_arg,
        train_crop_size=None,
        val_batch_size=cfg.data.get('val_batch_size', None),
    )
    if dataset_name == 'qxs_saropt':
        dm_kwargs['sar_lognorm'] = bool(cfg.data.get('sar_lognorm', False))
    return _DM(**dm_kwargs)


def _load_generator(cfg, ckpt_path):
    """Load EMA-or-live netG weights (state_dict, netG.-prefixed) into a fresh G."""
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    src = ckpt['state_dict']
    sd = {k[len('netG.'):]: v for k, v in src.items() if k.startswith('netG.')}
    g = LLWv4Generator(cfg)
    g.load_state_dict(sd)
    print(f"[ckpt] {len(sd)} netG tensors from {ckpt_path}")
    return g.to(DEVICE).eval()


def main():
    cfg = OmegaConf.load('./src/models/llwt_v5/config.yaml')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    dm = _build_datamodule(cfg)
    dm.setup("fit")
    loader = dm.val_dataloader() if SPLIT == "val" else dm.train_dataloader()

    g45 = _load_generator(cfg, CKPT_V45)
    g5  = _load_generator(cfg, CKPT_V5)

    psnr_m = PeakSignalNoiseRatio(data_range=1.0).to(DEVICE)
    ssim_m = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)
    # normalize=False -> inputs expected in [-1, 1] (matches main.py val loop).
    lpips_m = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=False).to(DEVICE)

    def metrics(gen, opt):
        """Per-image PSNR/SSIM (data in [0,1]) and LPIPS (data in [-1,1])."""
        g01 = (gen.clamp(-1, 1) + 1.0) / 2.0
        o01 = (opt + 1.0) / 2.0
        return (
            psnr_m(g01, o01).item(),
            ssim_m(g01, o01).item(),
            lpips_m(gen.clamp(-1, 1).float(), opt.float()).item(),
        )

    # min-heap of (delta_lpips, tiebreak, payload); delta = lpips45 - lpips5 (>0 = v5 sharper)
    heap = []
    n = 0
    agg = {'p45': 0.0, 's45': 0.0, 'l45': 0.0, 'p5': 0.0, 's5': 0.0, 'l5': 0.0}

    with torch.no_grad():
        for sar, opt in loader:
            sar = sar.to(DEVICE)
            opt = opt.to(DEVICE)
            gen45 = g45(sar)
            gen5 = g5(sar)
            for i in range(sar.size(0)):
                o = opt[i:i + 1]
                p45, s45, l45 = metrics(gen45[i:i + 1], o)
                p5, s5, l5 = metrics(gen5[i:i + 1], o)
                agg['p45'] += p45; agg['s45'] += s45; agg['l45'] += l45
                agg['p5'] += p5; agg['s5'] += s5; agg['l5'] += l5
                delta = l45 - l5
                payload = dict(
                    sar=sar[i].cpu(), g45=gen45[i].cpu(), g5=gen5[i].cpu(), gt=opt[i].cpu(),
                    p45=p45, s45=s45, l45=l45, p5=p5, s5=s5, l5=l5, delta=delta,
                )
                if len(heap) < TOP_K:
                    heapq.heappush(heap, (delta, n, payload))
                elif delta > heap[0][0]:
                    heapq.heapreplace(heap, (delta, n, payload))
                n += 1

    # ---- aggregate (full-split means) ----
    for k in agg:
        agg[k] /= max(n, 1)
    print("=" * 78)
    print(f"FULL-SPLIT MEANS over {n} {SPLIT} samples (same samples, both generators)")
    print("=" * 78)
    print(f"{'':14s}{'PSNR':>8s}{'SSIM':>8s}{'LPIPS':>9s}")
    print(f"{'v45 baseline':14s}{agg['p45']:8.3f}{agg['s45']:8.4f}{agg['l45']:9.4f}")
    print(f"{'v5 (HF-D)':14s}{agg['p5']:8.3f}{agg['s5']:8.4f}{agg['l5']:9.4f}")
    print(f"{'delta (v5-v45)':14s}{agg['p5']-agg['p45']:+8.3f}{agg['s5']-agg['s45']:+8.4f}{agg['l5']-agg['l45']:+9.4f}")
    print("  (LPIPS: lower=better -> negative delta = v5 better)")
    print("=" * 78)

    # ---- save top-K improvement panels ----
    ranked = sorted(heap, key=lambda t: t[0], reverse=True)
    for rank, (delta, idx, p) in enumerate(ranked):
        fig, axes = plt.subplots(1, 4, figsize=(16, 4.4))
        axes[0].imshow((p['sar'][0].numpy() + 1) / 2, cmap='gray')
        axes[0].set_title("SAR input"); axes[0].axis('off')
        axes[1].imshow(((p['g45'].numpy() + 1) / 2).transpose(1, 2, 0).clip(0, 1))
        axes[1].set_title(f"v45 baseline\nPSNR {p['p45']:.2f} SSIM {p['s45']:.3f} LPIPS {p['l45']:.3f}")
        axes[1].axis('off')
        axes[2].imshow(((p['g5'].numpy() + 1) / 2).transpose(1, 2, 0).clip(0, 1))
        axes[2].set_title(f"v5 HF-D\nPSNR {p['p5']:.2f} SSIM {p['s5']:.3f} LPIPS {p['l5']:.3f}")
        axes[2].axis('off')
        axes[3].imshow(((p['gt'].numpy() + 1) / 2).transpose(1, 2, 0).clip(0, 1))
        axes[3].set_title("GT optical"); axes[3].axis('off')
        fig.suptitle(f"#{rank:02d}  LPIPS improvement (v45-v5) = {delta:+.4f}", fontsize=12)
        plt.tight_layout()
        out = os.path.join(OUTPUT_DIR, f"cmp_{rank:02d}_sample{idx:04d}_dLPIPS{delta:.3f}.png")
        plt.savefig(out, dpi=140, bbox_inches='tight')
        plt.close(fig)
        print(f"#{rank:02d} sample {idx:04d} | dLPIPS {delta:+.4f} | "
              f"v45 L{p['l45']:.3f} -> v5 L{p['l5']:.3f} | saved -> {out}")


if __name__ == "__main__":
    main()
