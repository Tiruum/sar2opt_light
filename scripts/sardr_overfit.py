"""Tier 2 GPU overfit smoke for v5 SAR-DR Phase 1.

Loads the v5 module via factory, picks one batch, trains SAR-DR for
``ITERATIONS`` steps on that single batch, and reports:

  * ``loss_diff`` trajectory (should decrease monotonically)
  * coarse PSNR — fixed (G is frozen)
  * refined PSNR via 3-step DDIM at checkpoints (should approach
    or exceed coarse PSNR)

Pass criteria:
  * Final ``loss_diff`` < 0.1 (SAR-DR learned to predict noise)
  * Final refined PSNR >= coarse PSNR - 0.5 dB
    (diffusion at minimum shouldn't HURT the coarse output)

Run from repo root::

    python scripts/sardr_overfit.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))


from src.data.sen12_full.datamodule import SEN12FullDataModule
from src.models.llwt_v5 import factory
from src.models.llwt_v5.diffusion import ddim_sample
from src.models.llwt_v5.main import _load_g_from_ckpt


CONFIG_PATH = 'src/models/llwt_v5/config.yaml'
ITERATIONS  = 800
BATCH_SIZE  = 4
CHECK_EVERY = 100
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'


def psnr_db(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().clamp(-1, 1)
    b = b.float().clamp(-1, 1)
    mse = (a - b).pow(2).mean()
    return float(10.0 * torch.log10(torch.tensor(4.0) / mse.clamp_min(1e-12)))


def main():
    cfg = OmegaConf.load(CONFIG_PATH)
    assert cfg.sardr.enabled, "set cfg.sardr.enabled = true before overfit"

    print(f"[overfit] device={DEVICE}  iterations={ITERATIONS}  bs={BATCH_SIZE}")
    print(f"[overfit] sardr cfg: base_dim={cfg.sardr.base_dim} ch_mult={list(cfg.sardr.ch_mult)} "
          f"inference_steps={cfg.sardr.num_inference_steps}")

    # Build models.  Factory returns 3-tuple when sardr enabled.
    netG, netD, sardr = factory.build_models(cfg)

    # Load frozen v4 G from headline ckpt.
    _load_g_from_ckpt(netG, cfg.system.weights_ckpt)
    for p in netG.parameters():
        p.requires_grad = False
    netG = netG.to(DEVICE).to(memory_format=torch.channels_last).eval()
    sardr = sardr.to(DEVICE).to(memory_format=torch.channels_last)

    diffusion = factory.build_diffusion(cfg).to(DEVICE)
    opt_sardr = factory.build_sardr_optimizer(cfg, sardr)

    n_sardr = sum(p.numel() for p in sardr.parameters())
    print(f"[overfit] params: G(frozen)={sum(p.numel() for p in netG.parameters())/1e6:.2f}M  "
          f"sardr={n_sardr/1e6:.2f}M")

    # One fixed batch from val loader.
    dm = SEN12FullDataModule(
        data_dir=cfg.data.data_dir.sen12_full,
        batch_size=BATCH_SIZE,
        image_size=cfg.data.image_size,
        num_workers=0,
        persistent_workers=False,
        prefetch_factor=2,
        train_val_split_ratio=cfg.data.train_val_split_ratio,
        seed=cfg.data.seed,
        sar_channels=cfg.data.sar_channels,
        use_augmentation=False,
        scenes=list(cfg.data.scenes),
        train_crop_size=None,
        val_batch_size=BATCH_SIZE,
    )
    dm.setup('fit')
    sar, opt = next(iter(dm.val_dataloader()))
    sar = sar.to(DEVICE).contiguous(memory_format=torch.channels_last)
    opt = opt.to(DEVICE).contiguous(memory_format=torch.channels_last)

    # Coarse PSNR (fixed — G frozen).
    with torch.no_grad():
        with torch.amp.autocast(device_type='cuda', enabled=True, dtype=torch.bfloat16):
            coarse = netG(sar)
    coarse = coarse.float().clamp(-1, 1)
    coarse_psnr = psnr_db(coarse, opt)
    print(f"[overfit] coarse PSNR (G frozen): {coarse_psnr:.2f} dB")
    print("=" * 78)

    sardr.train()
    loss_history = []
    for step in range(1, ITERATIONS + 1):
        B = sar.size(0)
        t = torch.randint(0, diffusion.num_train_timesteps, (B,), device=DEVICE, dtype=torch.long)
        noise = torch.randn_like(opt)
        x_t = diffusion.q_sample(opt, t, noise)

        with torch.amp.autocast(device_type='cuda', enabled=True, dtype=torch.bfloat16):
            eps_pred = sardr(sar, coarse, x_t, t)
        l_diff = F.mse_loss(eps_pred.float(), noise)

        opt_sardr.zero_grad()
        l_diff.backward()
        torch.nn.utils.clip_grad_norm_(sardr.parameters(), float(cfg.sardr.get('grad_clip', 1.0)))
        opt_sardr.step()

        loss_history.append(l_diff.item())

        if step % CHECK_EVERY == 0 or step == 1:
            # DDIM sample and compute refined PSNR.
            sardr.eval()
            with torch.no_grad():
                with torch.amp.autocast(device_type='cuda', enabled=True, dtype=torch.bfloat16):
                    refined = ddim_sample(
                        sardr, diffusion, sar, coarse,
                        num_steps=int(cfg.sardr.num_inference_steps),
                        eta=float(cfg.sardr.ddim_eta),
                        init_strength=float(cfg.sardr.init_strength),
                    )
            refined_psnr = psnr_db(refined, opt)
            avg_loss = sum(loss_history[-CHECK_EVERY:]) / min(CHECK_EVERY, len(loss_history))
            print(f"  step {step:4d} | L_diff={avg_loss:.4f} | "
                  f"coarse PSNR={coarse_psnr:.2f} | refined PSNR={refined_psnr:.2f} | "
                  f"delta={refined_psnr - coarse_psnr:+.2f} dB")
            sardr.train()

    print("=" * 78)
    final_loss = sum(loss_history[-100:]) / 100
    sardr.eval()
    with torch.no_grad():
        with torch.amp.autocast(device_type='cuda', enabled=True, dtype=torch.bfloat16):
            refined = ddim_sample(
                sardr, diffusion, sar, coarse,
                num_steps=int(cfg.sardr.num_inference_steps),
                eta=float(cfg.sardr.ddim_eta),
                init_strength=float(cfg.sardr.init_strength),
            )
    final_refined_psnr = psnr_db(refined, opt)

    print(f"  final L_diff (avg last 100 steps): {final_loss:.4f}")
    print(f"  coarse PSNR:  {coarse_psnr:.2f} dB")
    print(f"  refined PSNR: {final_refined_psnr:.2f} dB  (delta={final_refined_psnr - coarse_psnr:+.2f})")

    pass_loss   = final_loss < 0.1
    pass_no_hurt = final_refined_psnr >= coarse_psnr - 0.5

    print()
    print(f"  L_diff < 0.1                 : {'PASS' if pass_loss else 'FAIL'}")
    print(f"  refined >= coarse - 0.5 dB   : {'PASS' if pass_no_hurt else 'FAIL'}")
    print(f"  OVERALL                      : {'PASS' if pass_loss and pass_no_hurt else 'FAIL'}")
    print("=" * 78)


if __name__ == '__main__':
    main()
