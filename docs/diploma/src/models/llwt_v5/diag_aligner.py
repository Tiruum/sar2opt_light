"""THROWAWAY GPU diagnostic — why did the SAWG deformation aligner stay inert?

train/phi_abs_mean ~ 0.05px flat for 130 epochs on RAW (unaligned) sen12_full,
where it should reach 1.5-4px. Three hypotheses:
  A (quantization): bf16 autocast quantizes the warp grid -> logged phi floors.
  B (reg over-damping): reg_smooth=10 + reg_mag=2 pin phi->0 (reg grad >> fid grad).
  C (weak aligner gradient): warm G already good -> fidelity ~0 -> no signal to phi.

Run from repo root:  python -m src.models.llwt_v5.diag_aligner

NOT a training run. A few batches, ~50 backward passes total. Seconds.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT))

from omegaconf import OmegaConf

from src.models.llwt_v5.main import LLWv4LightningModule
from src.models.llwt_v5.align import DeformationAligner
from src.data.sen12_full.datamodule import SEN12FullDataModule

CONFIG_PATH = str(ROOT / 'src' / 'models' / 'llwt_v5' / 'config.yaml')
CKPT_PATH = str(ROOT / 'checkpoints' / 'llwt_v45' / 'llwt-v0.5.0-sawg' / 'last.ckpt')


def banner(title: str) -> None:
    print('\n' + '=' * 80)
    print(title)
    print('=' * 80)


def fidelity_loss(module, fake, predicted_sub, opt_aligned, loss_cfg):
    """Sum of the fidelity losses that flow gradient through opt_aligned (and fake).
    Mirrors main.py weights exactly. No reg/psc/bsc here — pure fidelity."""
    crit = module.criterions
    total = fake.new_zeros(())
    if 'msssim' in crit:
        total = total + crit['msssim'](fake, opt_aligned) * float(loss_cfg.msssim_weight)
    if 'per_band' in crit:
        total = total + crit['per_band'](predicted_sub, opt_aligned) * float(loss_cfg.per_band_weight)
    if 'lpips' in crit:
        total = total + crit['lpips'](fake, opt_aligned) * float(loss_cfg.lpips_weight)
    if 'ffl' in crit:
        total = total + crit['ffl'](fake, opt_aligned) * float(loss_cfg.ffl_weight)
    return total


def aligner_grad_norm(module):
    sq = 0.0
    for p in module.aligner.parameters():
        if p.grad is not None:
            sq += float(p.grad.detach().float().pow(2).sum().item())
    return sq ** 0.5


def main():
    torch.manual_seed(0)
    assert torch.cuda.is_available(), "diag needs CUDA"
    device = torch.device('cuda')
    cfg = OmegaConf.load(CONFIG_PATH)
    cfg.align.enabled = True
    loss_cfg = cfg.loss

    banner('[diag] SETUP — build module + load checkpoint')
    module = LLWv4LightningModule(cfg).to(device)
    module.eval()
    assert module.use_align, "align not enabled in module"
    sd = torch.load(CKPT_PATH, map_location='cpu', weights_only=False)['state_dict']
    missing, unexpected = module.load_state_dict(sd, strict=False)
    n_align_missing = sum(1 for k in missing if k.startswith('aligner.'))
    n_g_missing = sum(1 for k in missing if k.startswith('netG.'))
    print(f"  ckpt state_dict keys     = {len(sd)}")
    print(f"  load missing total       = {len(missing)} (aligner.* missing={n_align_missing}, netG.* missing={n_g_missing})")
    print(f"  load unexpected total    = {len(unexpected)}")
    # report aligner final-conv weight stats (zero-init => 0; trained => nonzero)
    w = module.aligner.net[-1].weight.detach()
    print(f"  aligner final-conv W: abs.mean={w.abs().mean().item():.3e} abs.max={w.abs().max().item():.3e}")

    banner('[diag] DATA — 2 raw sen12_full batches (bs=4)')
    dm = SEN12FullDataModule(
        data_dir=str(ROOT / 'data' / 'sen12_full'),
        batch_size=4,
        image_size=int(cfg.data.image_size),
        num_workers=0,
        persistent_workers=False,
        prefetch_factor=2,
        train_val_split_ratio=float(cfg.data.train_val_split_ratio),
        seed=int(cfg.data.seed),
        sar_channels=int(cfg.data.sar_channels),
        use_augmentation=False,
        scenes=list(cfg.data.scenes),
    )
    dm.setup('fit')
    loader = dm.train_dataloader()
    it = iter(loader)
    batches = [next(it), next(it)]
    batches = [(s.to(device), o.to(device)) for (s, o) in batches]
    sar0, opt0 = batches[0]
    print(f"  batch0: sar={tuple(sar0.shape)} opt={tuple(opt0.shape)}  "
          f"sar[{sar0.min():.2f},{sar0.max():.2f}] opt[{opt0.min():.2f},{opt0.max():.2f}]")

    netG = module.netG
    aligner = module.aligner

    # ===================================================================== TEST A
    banner('[diag] TEST A — quantization (fp32 phi vs bf16 phi)')
    sar, opt = batches[0]
    with torch.no_grad():
        for label, use_bf16 in (('fp32', False), ('bf16', True)):
            ctx = torch.autocast('cuda', dtype=torch.bfloat16) if use_bf16 else torch.autocast('cuda', enabled=False)
            with ctx:
                fake = netG(sar)
                fake_ll = module._haar_ll(fake)[0]
                opt_ll = module._haar_ll(opt)[0]
                phi = aligner(fake_ll, opt_ll)
                ident_err = (DeformationAligner.warp(opt, torch.zeros_like(phi)) - opt).abs().max()
            print(f"  [{label}] phi.abs().mean={phi.float().abs().mean().item():.5f}px  "
                  f"phi.abs().max={phi.float().abs().max().item():.5f}px  "
                  f"identity-warp.max_err={ident_err.float().item():.3e}  phi.dtype={phi.dtype}")

    # ================================================================ TEST B / C
    banner('[diag] TEST B/C — gradient balance at current phi (fp32, G frozen)')
    sar, opt = batches[0]
    with torch.autocast('cuda', enabled=False):
        with torch.no_grad():
            need_int = ('per_band' in module.criterions)
            if need_int:
                fake, predicted_sub, _ = netG(sar, return_internals=True)
            else:
                fake = netG(sar); predicted_sub = None
        fake = fake.detach()
        if predicted_sub is not None:
            predicted_sub = predicted_sub.detach()

        # fidelity grad to aligner
        aligner.zero_grad(set_to_none=True)
        fake_ll = module._haar_ll(fake)[0]
        opt_ll = module._haar_ll(opt)[0]
        phi = aligner(fake_ll, opt_ll)
        opt_aligned = DeformationAligner.warp(opt, phi)
        l_fid = fidelity_loss(module, fake, predicted_sub, opt_aligned, loss_cfg)
        l_fid.backward()
        g_fid = aligner_grad_norm(module)

        # reg grad to aligner
        aligner.zero_grad(set_to_none=True)
        phi2 = aligner(fake_ll.detach(), opt_ll.detach())
        l_reg = module.align_criterions['deform_reg'](phi2)
        l_reg.backward()
        g_reg = aligner_grad_norm(module)

    print(f"  current phi.abs().mean   = {phi.detach().abs().mean().item():.5f} px")
    print(f"  fidelity loss            = {float(l_fid):.5f}")
    print(f"  reg loss                 = {float(l_reg):.5f}")
    print(f"  g_fid (||grad_aligner||) = {g_fid:.4e}")
    print(f"  g_reg (||grad_aligner||) = {g_reg:.4e}")
    ratio = g_reg / g_fid if g_fid > 0 else float('inf')
    print(f"  g_reg / g_fid            = {ratio:.3f}")

    # ============================================================= TEST (grow?)
    banner('[diag] TEST grow — can phi grow with reg REMOVED + G frozen? (50 steps, fp32)')
    sar, opt = batches[0]
    for p in netG.parameters():
        p.requires_grad_(False)
    opt_adam = torch.optim.Adam(aligner.parameters(), lr=1e-3)
    with torch.no_grad(), torch.autocast('cuda', enabled=False):
        if ('per_band' in module.criterions):
            fake, predicted_sub, _ = netG(sar, return_internals=True)
        else:
            fake = netG(sar); predicted_sub = None
    fake = fake.detach()
    predicted_sub = predicted_sub.detach() if predicted_sub is not None else None
    opt_ll = module._haar_ll(opt)[0].detach()
    fake_ll = module._haar_ll(fake)[0].detach()
    print(f"  step  0 | phi.abs().mean = (init) {aligner(fake_ll, opt_ll).abs().mean().item():.5f} px")
    for step in range(1, 51):
        with torch.autocast('cuda', enabled=False):
            phi = aligner(fake_ll, opt_ll)
            opt_aligned = DeformationAligner.warp(opt, phi)
            loss = fidelity_loss(module, fake, predicted_sub, opt_aligned, loss_cfg)  # NO reg
        opt_adam.zero_grad(set_to_none=True)
        loss.backward()
        opt_adam.step()
        if step % 10 == 0:
            print(f"  step {step:2d} | phi.abs().mean = {phi.detach().abs().mean().item():.5f} px  "
                  f"phi.max={phi.detach().abs().max().item():.4f}  fid={float(loss):.5f}")

    # ===================================================== TEST grow+reg (control)
    banner('[diag] TEST grow+reg — same 50 steps WITH reg (the production loss), G frozen')
    # Re-fetch a fresh aligner state so this is comparable to the no-reg run's start.
    # (aligner was just trained 50 steps above; reload its ckpt weights to reset.)
    aligner.load_state_dict({k[len('aligner.'):]: v for k, v in sd.items() if k.startswith('aligner.')})
    aligner = aligner.to(device)
    opt_adam2 = torch.optim.Adam(aligner.parameters(), lr=1e-3)
    print(f"  step  0 | phi.abs().mean = (reset) {aligner(fake_ll, opt_ll).abs().mean().item():.5f} px")
    for step in range(1, 51):
        with torch.autocast('cuda', enabled=False):
            phi = aligner(fake_ll, opt_ll)
            opt_aligned = DeformationAligner.warp(opt, phi)
            loss = fidelity_loss(module, fake, predicted_sub, opt_aligned, loss_cfg)
            loss = loss + module.align_criterions['deform_reg'](phi)   # WITH reg
        opt_adam2.zero_grad(set_to_none=True)
        loss.backward()
        opt_adam2.step()
        if step % 10 == 0:
            print(f"  step {step:2d} | phi.abs().mean = {phi.detach().abs().mean().item():.5f} px  "
                  f"phi.max={phi.detach().abs().max().item():.4f}  total={float(loss):.5f}")

    # ===================================================================== VERDICT
    banner('[diag] VERDICT')
    print(f"  g_fid={g_fid:.3e}  g_reg={g_reg:.3e}  ratio(reg/fid)={ratio:.2f}")
    print("  (interpret: A if fp32 phi >> bf16 phi; B if reg/fid>>1 AND grow-test moves phi; "
          "C if g_fid~0 AND grow-test phi stays flat)")


if __name__ == '__main__':
    main()
