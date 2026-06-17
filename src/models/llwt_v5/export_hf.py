"""Export the LLW-Former generator for Hugging Face (route-1: weights + config).

Loads a Lightning ``.ckpt`` via :func:`inference.load_generator`, then writes a
clean generator ``state_dict`` (safetensors if available, else ``.pt``) plus a
minimal ``config.json`` of generator-only hyperparameters. No training/data keys
leak into the export.

Run from repo root::

    python -m src.models.llwt_v5.export_hf \
        --ckpt checkpoints/llwt_v45_base/llwt-v0.4.6-base/epoch=199-psnr=18.5361.ckpt \
        --out  output/hf_export
"""
import argparse
import json
import os

# Mirror inference.py offline-HF guards (backbone is cached from training).
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
if os.environ.get('HF_HUB_OFFLINE') == '1':
    import transformers.models.auto.auto_factory as _af
    _af.repo_exists = lambda *a, **k: True

import torch
from omegaconf import OmegaConf

from src.models.llwt_v5.inference import load_generator

CONFIG = "./src/models/llwt_v5/config.yaml"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="output/hf_export")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--live", action="store_true", help="export live (non-EMA) weights")
    args = ap.parse_args()

    cfg = OmegaConf.load(CONFIG)
    g = load_generator(args.ckpt, cfg, args.device, use_live_weights=args.live).cpu()
    os.makedirs(args.out, exist_ok=True)

    sd = g.state_dict()
    try:
        from safetensors.torch import save_file
        save_file(sd, os.path.join(args.out, "generator.safetensors"))
        weights_file = "generator.safetensors"
    except Exception as e:  # safetensors missing or non-contiguous tensors
        print(f"[export_hf] safetensors unavailable ({e}); falling back to .pt")
        torch.save(sd, os.path.join(args.out, "generator.pt"))
        weights_file = "generator.pt"

    gen_cfg = {
        "architecture": "LLWv4Generator",
        "backbone": str(cfg.model.gen.backbone),
        "sar_channels": int(cfg.data.sar_channels),
        "image_size": int(cfg.data.image_size),
        "use_sar_physics": bool(cfg.model.gen.get("use_sar_physics", True)),
        "weights_file": weights_file,
        "num_tensors": len(sd),
    }
    with open(os.path.join(args.out, "config.json"), "w", encoding="utf-8") as f:
        json.dump(gen_cfg, f, indent=2)
    print(f"[export_hf] wrote {weights_file} + config.json to {args.out}")


if __name__ == "__main__":
    main()
