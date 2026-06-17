"""Standalone quick-look inference for WaveNeXt v0.4.5.

Loads ``CHECKPOINT['state_dict']`` (defaults to the production ``last.ckpt``),
strips the ``netG.`` prefix, applies the generator to ONE batch of SAR samples
(``N_IMAGES`` capped at the batch size), and writes side-by-side PNGs with
PSNR/SSIM.  For full-split top-K-per-scene scoring use ``best_inference.py``.

Run from repo root::

    python -m src.models.wavenext.inference
"""
import os

# Skip the online HF repo_exists probe that SSL-times-out on a flaky/offline
# network; the backbone is already cached from training. setdefault keeps an
# escape hatch: ``$env:HF_HUB_OFFLINE=0`` forces online.
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

# transformers' AutoBackbone.from_pretrained calls repo_exists() unconditionally
# (a network probe to pick HF-vs-timm). Offline that *raises*; online on a flaky
# net it SSL-times-out. Our backbone is a cached HF model, so force the HF branch.
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
)

from src.models.wavenext.gen import WaveNeXtGenerator


# Base default (matches config.yaml backbone). Tiny ckpt: checkpoints/llwt_v45/llwt-v0.5.1-hfd/epoch=097-psnr=17.1615.ckpt
CHECKPOINT = "checkpoints/llwt_v45_base/llwt-v0.4.6-base/epoch=199-psnr=18.5361.ckpt"
N_IMAGES = 10
SPLIT = "val"  # "train" or "val"
OUTPUT_DIR = f"./src/models/wavenext/output/{SPLIT}"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_LIVE_WEIGHTS = False


def _build_datamodule(cfg):
    """Dispatch on cfg.data.dataset — mirrors train.py selection logic."""
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


def load_generator(ckpt_path, cfg, device="cuda", use_live_weights=False):
    """Build ``WaveNeXtGenerator(cfg)`` and load its ``netG.``-prefixed weights from a
    Lightning ``.ckpt``. Decoupled from the LightningModule / data / training config —
    needs only the generator block of ``cfg``. EMA-swapped weights live under
    ``state_dict``; pass ``use_live_weights=True`` for the non-EMA snapshot.

    Returns the generator in eval mode on ``device``. Reused by ``export_hf.py``.
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    src_dict = (
        ckpt.get('current_model_state', ckpt['state_dict'])
        if use_live_weights
        else ckpt['state_dict']
    )
    state_dict = {
        k[len('netG.'):]: v for k, v in src_dict.items() if k.startswith('netG.')
    }
    g = WaveNeXtGenerator(cfg)
    missing, unexpected = g.load_state_dict(state_dict, strict=False)
    src_label = 'live (current_model_state)' if use_live_weights else 'EMA-or-live (state_dict)'
    print(f"[load_generator] {len(state_dict)} netG tensors ({src_label}) | "
          f"missing={len(missing)} unexpected={len(unexpected)}")
    return g.to(device).eval()


def main():
    cfg = OmegaConf.load('./src/models/wavenext/config.yaml')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    dm = _build_datamodule(cfg)
    dm.setup("fit")

    loader = dm.val_dataloader() if SPLIT == "val" else dm.train_dataloader()
    sar, opt = next(iter(loader))
    sar = sar.to(DEVICE)
    opt = opt.to(DEVICE)

    netG = load_generator(CHECKPOINT, cfg, DEVICE, use_live_weights=USE_LIVE_WEIGHTS)

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
