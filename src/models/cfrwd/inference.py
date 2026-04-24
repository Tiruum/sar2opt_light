import os
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from src.models.cfrwd.gen import CFRWDGenerator
from src.data.sen12.datamodule import SEN12Datamodule

# Architecture changed (separate per-branch decoders) — incompatible with cfrwd-36 checkpoint.
# Update this path after retraining with the new architecture.
CHECKPOINT = "checkpoints/cfrwd/cfrwd-38/last.ckpt"
N_IMAGES = 8
SPLIT = "val"
OUTPUT_DIR = "outputs/cfr_analysis"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    cfg = OmegaConf.load("src/models/cfrwd/config.yaml")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    dm = SEN12Datamodule(
        data_dir=cfg.data.data_dir.sen12,
        batch_size=N_IMAGES,
        image_size=cfg.data.image_size,
        num_workers=0,
        persistent_workers=False,
        prefetch_factor=None,
        train_val_split_ratio=cfg.data.train_val_split_ratio,
        seed=cfg.data.seed,
        sar_channels=cfg.data.sar_channels,
        use_augmentation=False
    )
    dm.setup("fit")

    if SPLIT == "val":
        loader = dm.val_dataloader()
    else:
        loader = dm.train_dataloader()

    sar, opt = next(iter(loader))
    sar = sar.to(DEVICE)
    opt = opt.to(DEVICE)

    ckpt = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
    netG = CFRWDGenerator(in_channels=cfg.model.gen.in_channels)
    state_dict = {k[len('netG.'):]: v for k, v in ckpt['state_dict'].items() if k.startswith('netG.')}
    netG.load_state_dict(state_dict)
    netG = netG.to(DEVICE).eval()

    with torch.no_grad():
        fused, cfr_out, hfcf_out, fw = netG(sar, return_branches=True)

    sar_np = sar.detach().cpu().numpy()
    opt_np = opt.detach().cpu().numpy()
    fused_np = fused.detach().cpu().numpy()
    cfr_np = cfr_out.detach().cpu().numpy()
    hfcf_np = hfcf_out.detach().cpu().numpy()
    fw_np = fw.detach().cpu().numpy()

    # fw is a scalar (1,) tensor — same value for all images in this batch
    w_scalar = fw_np[0]  # shape () or (1,) — the single fusion weight value
    w_val = float(w_scalar) if hasattr(w_scalar, '__float__') else float(w_scalar.item())

    n = len(sar)

    for i in range(n):
        fig, axes = plt.subplots(1, 6, figsize=(22, 4))

        sar_img = (sar_np[i, 0] + 1) / 2
        axes[0].imshow(sar_img, cmap='gray')
        axes[0].set_title("SAR input")
        axes[0].axis('off')

        cfr_img = (cfr_np[i] + 1) / 2
        cfr_img = cfr_img.transpose(1, 2, 0)
        axes[1].imshow(cfr_img)
        axes[1].set_title("CFR branch")
        axes[1].axis('off')

        hfcf_img = (hfcf_np[i] + 1) / 2
        hfcf_img = hfcf_img.transpose(1, 2, 0)
        axes[2].imshow(hfcf_img)
        axes[2].set_title("HFCF branch")
        axes[2].axis('off')

        fused_img = (fused_np[i] + 1) / 2
        fused_img = fused_img.transpose(1, 2, 0)
        axes[3].imshow(fused_img)
        axes[3].set_title("Fused output")
        axes[3].axis('off')

        gt_img = (opt_np[i] + 1) / 2
        gt_img = gt_img.transpose(1, 2, 0)
        axes[4].imshow(gt_img)
        axes[4].set_title("GT optical")
        axes[4].axis('off')

        # Scalar fusion weight — same for all images (learnable parameter, not per-pixel)
        axes[5].text(0.5, 0.5, f"w_hfcf\n{w_val:.4f}", ha='center', va='center',
                     fontsize=18, transform=axes[5].transAxes)
        axes[5].set_title("Fusion weight (scalar)")
        axes[5].axis('off')

        plt.tight_layout()
        out_path = os.path.join(OUTPUT_DIR, f"img_{i:03d}.png")
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Image {i:03d}  w_hfcf={w_val:.4f}  saved → {out_path}")

    print("━" * 40)
    print(f"w_hfcf scalar: {w_val:.4f}  (same across all {n} images — learnable parameter)")


if __name__ == "__main__":
    main()
