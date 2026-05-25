"""LLW-Former v0.5.2 (A3) — adversarial residual refiner on a frozen v4 G.

Lightning module that trains a small UNet refiner on top of a frozen v4
generator, ADVERSARIALLY against the v4 discriminator stack (MainDis +
SubbandDis) with GAN + feature-matching + a light L1 anchor.  Manual
optimisation, two optimisers, batched D real+fake forward — same loop shape
as ``llwt_v4/main.py`` with the refiner standing in for the generator.

Why adversarial (replaces the v0.5.1 pure-L1 refiner):
  * L1's minimiser is the conditional mean E[opt|sar]; for the ill-posed
    SAR->opt mapping that estimator is inherently blurry.  Pure-L1 refinement
    slides along the PSNR<->FID Pareto front toward the (blurry) PSNR corner
    — measured v0.5.1: PSNR +1.0 dB but FID 143->167, LPIPS 0.570->0.613.
  * Only a distributional (adversarial) signal can ADVANCE the front past the
    frozen-G FID corner.  ``cfg.loss.l1_weight`` is the leash: a light pixel
    anchor that bounds adversarial hallucination and keeps the refiner near
    the data.  It is the explicit PSNR<->FID dial (drop toward 0 for more
    sharpness, raise for more fidelity).

Pipeline:

    coarse   = G_frozen(SAR)                      (B, 3, H, W) in [-1, 1]
    residual = ResidualRefiner(SAR, coarse)       (B, 3, H, W)
    refined  = (coarse + residual).clamp(-1, 1)   (B, 3, H, W)
    D sees   [SAR, refined.detach()] vs [SAR, opt]
    G(refiner) loss = gan_main + gan_sub + fm_main + fm_sub + l1_weight * L1

Validation:
  PSNR / SSIM / LPIPS / FID on ``refined``.  Also logs ``val/coarse_psnr``
  (fixed, G frozen) for direct comparison — refined should beat or tie it.
"""
from __future__ import annotations

from typing import Optional

import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.image import (
    LearnedPerceptualImagePatchSimilarity,
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)
from torchmetrics.image.fid import FrechetInceptionDistance

from src.models.llwt_v5 import factory


__all__ = ["LLWv5RefinerModule"]


def _logit_mean(logits) -> torch.Tensor:
    if isinstance(logits, (list, tuple)):
        return torch.stack([l.mean() for l in logits]).mean()
    return logits.mean()


def _grads_finite(params) -> bool:
    for p in params:
        if p.grad is None:
            continue
        if not torch.isfinite(p.grad).all():
            return False
    return True


def _load_g_from_ckpt(netG: nn.Module, ckpt_path: str) -> None:
    """Load v4 G weights from a Lightning ckpt, stripping the ``netG.`` prefix."""
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state_dict = ckpt.get('state_dict', ckpt)
    g_state = {}
    for k, v in state_dict.items():
        if k.startswith('netG.'):
            g_state[k[len('netG.'):]] = v
    missing, unexpected = netG.load_state_dict(g_state, strict=False)
    print(f"[v5] loaded frozen G from {ckpt_path}")
    if missing:
        print(f"[v5]   missing keys ({len(missing)}): {list(missing)[:5]} ...")
    if unexpected:
        print(f"[v5]   unexpected keys ({len(unexpected)}): {list(unexpected)[:5]} ...")


class LLWv5RefinerModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.automatic_optimization = False

        out = factory.build_models(cfg)
        if len(out) != 3:
            raise RuntimeError(
                "v5 LLWv5RefinerModule requires cfg.refiner.enabled=true so "
                "that build_models returns (G, D, refiner).  Got 2-tuple."
            )
        netG, netD, refiner = out
        self.netG    = netG
        self.netD    = netD          # A3: trained adversarially vs the refiner
        self.refiner = refiner

        ckpt_path = getattr(cfg.system, 'weights_ckpt', None)
        if ckpt_path is None:
            raise ValueError(
                "v5 requires cfg.system.weights_ckpt pointing at a v4 ckpt "
                "(recommended: headline best — PSNR 14.85)."
            )
        _load_g_from_ckpt(self.netG, ckpt_path)
        for p in self.netG.parameters():
            p.requires_grad = False
        self.netG.eval()

        self.use_channels_last = bool(getattr(cfg.system, 'channels_last', False))
        if self.use_channels_last:
            self.netG    = self.netG.to(memory_format=torch.channels_last)
            self.netD    = self.netD.to(memory_format=torch.channels_last)
            self.refiner = self.refiner.to(memory_format=torch.channels_last)

        self.criterions = nn.ModuleDict(factory.build_refiner_criterions(cfg))

        # Loss weights / grad clips from cfg.loss.
        loss_cfg = cfg.loss
        self.gan_main_weight = float(getattr(loss_cfg, 'gan_main_weight', 1.0))
        self.gan_sub_weight  = float(getattr(loss_cfg, 'gan_sub_weight',  1.0))
        self.fm_main_weight  = float(getattr(loss_cfg, 'fm_main_weight',  10.0))
        self.fm_sub_weight   = float(getattr(loss_cfg, 'fm_sub_weight',   10.0))
        self.l1_weight       = float(getattr(loss_cfg, 'l1_weight',       5.0))
        self.grad_clip_g     = float(getattr(loss_cfg, 'grad_clip_g',     1.0))
        self.grad_clip_d     = float(getattr(loss_cfg, 'grad_clip_d',     5.0))
        self.use_subband_d   = bool(getattr(self.netD, 'use_sub', False))

        # Mode-collapse abort-gate state (light L1 anchor lowers the risk vs
        # pure GAN+FM, but the gate is cheap insurance).
        self._prev_val_psnr: Optional[float] = None
        self._mode_collapse_strikes = 0

        # Validation metrics — data range 2.0 because images are in [-1, 1].
        self.psnr  = PeakSignalNoiseRatio(data_range=2.0)
        self.ssim  = StructuralSimilarityIndexMeasure(data_range=2.0)
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=False)
        self.fid   = FrechetInceptionDistance(feature=2048, reset_real_features=True, normalize=True)

        self.fixed_sar: Optional[torch.Tensor] = None
        self.fixed_opt: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------ optim

    def configure_optimizers(self):
        opt_d, opt_g = factory.build_refiner_optimizers(self.cfg, self.netD, self.refiner)
        sched_d, sched_g = factory.build_lr_schedulers(self.cfg, opt_d, opt_g)
        self.sched_d = sched_d
        self.sched_g = sched_g
        return [opt_d, opt_g]

    # ------------------------------------------------------------------ setup

    def setup(self, stage: str) -> None:
        if stage != 'fit':
            return
        train_loader = self.trainer.datamodule.train_dataloader()
        sar, opt = next(iter(train_loader))
        self.fixed_sar = sar
        self.fixed_opt = opt

    # ------------------------------------------------------------------ train

    def training_step(self, batch, batch_idx):
        sar, opt = batch
        opt_d, opt_g = self.optimizers()

        if not (torch.isfinite(sar).all() and torch.isfinite(opt).all()):
            self.log('train/bad_batch_skip', 1.0, on_step=False, on_epoch=True, reduce_fx='sum')
            return

        if self.use_channels_last:
            sar = sar.contiguous(memory_format=torch.channels_last)
            opt = opt.contiguous(memory_format=torch.channels_last)

        # Frozen-G coarse output — deterministic, shared by both phases.
        with torch.no_grad():
            coarse = self.netG(sar)

        B = sar.size(0)

        # -------- D update on detached refined --------
        with torch.no_grad():
            residual_d = self.refiner(sar, coarse)
            fake_d = (coarse + residual_d).clamp(-1.0, 1.0)

        sar_doubled = sar.repeat(2, 1, 1, 1)
        opt_doubled = torch.cat([opt, fake_d], dim=0)
        main_both, sub_both, _, _, _, _ = self.netD(sar_doubled, opt_doubled)
        lc_both, lf_both = main_both
        real_main = (lc_both[:B], lf_both[:B])
        fake_main = (lc_both[B:], lf_both[B:])
        d_loss = 0.5 * (
            self.criterions['gan'](real_main, is_real=True) +
            self.criterions['gan'](fake_main, is_real=False)
        )
        if self.use_subband_d:
            real_sub = sub_both[:B]
            fake_sub = sub_both[B:]
            d_loss_sub = 0.5 * (
                self.criterions['gan'](real_sub, is_real=True) +
                self.criterions['gan'](fake_sub, is_real=False)
            )
            d_loss = d_loss + d_loss_sub
            self.log('train/d_loss_sub', d_loss_sub.detach(), on_step=False, on_epoch=True)

        opt_d.zero_grad()
        self.manual_backward(d_loss)
        if not _grads_finite(self.netD.parameters()):
            self.log('train/d_grad_nan_skip', 1.0, on_step=False, on_epoch=True, reduce_fx='sum')
            opt_d.zero_grad()
        else:
            if self.grad_clip_d > 0:
                torch.nn.utils.clip_grad_norm_(self.netD.parameters(), self.grad_clip_d)
            opt_d.step()

        # -------- G (refiner) update --------
        residual = self.refiner(sar, coarse)
        refined = (coarse + residual).clamp(-1.0, 1.0)

        sar_doubled_g = sar.repeat(2, 1, 1, 1)
        opt_doubled_g = torch.cat([refined, opt], dim=0)
        main_both_g, sub_both_g, _, feats_main_both, feats_sub_both, _ = self.netD(
            sar_doubled_g, opt_doubled_g,
        )
        lc_both_g, lf_both_g = main_both_g
        fake_main_g = (lc_both_g[:B], lf_both_g[:B])
        fake_feats_main = [f[:B] for f in feats_main_both]
        real_feats_main = [f[B:].detach() for f in feats_main_both]

        l_gan_main = self.criterions['gan'](fake_main_g, is_real=True, for_d=False)
        l_fm_main = self.criterions['fm'](fake_feats_main, real_feats_main)
        g_loss = (
            l_gan_main * self.gan_main_weight +
            l_fm_main * self.fm_main_weight
        )
        self.log('train/gan_main', l_gan_main.detach(), on_step=False, on_epoch=True)
        self.log('train/fm_main', l_fm_main.detach(), on_step=False, on_epoch=True)

        if self.use_subband_d:
            fake_sub_g = sub_both_g[:B]
            fake_feats_sub = [f[:B] for f in feats_sub_both]
            real_feats_sub = [f[B:].detach() for f in feats_sub_both]
            l_gan_sub = self.criterions['gan'](fake_sub_g, is_real=True, for_d=False)
            g_loss = g_loss + l_gan_sub * self.gan_sub_weight
            self.log('train/gan_sub', l_gan_sub.detach(), on_step=False, on_epoch=True)
            if fake_feats_sub:
                l_fm_sub = self.criterions['fm'](fake_feats_sub, real_feats_sub)
                g_loss = g_loss + l_fm_sub * self.fm_sub_weight
                self.log('train/fm_sub', l_fm_sub.detach(), on_step=False, on_epoch=True)

        # Light L1 anchor — the PSNR<->FID leash on adversarial drift.
        if self.l1_weight > 0:
            l_l1 = F.l1_loss(refined, opt)
            g_loss = g_loss + l_l1 * self.l1_weight
            self.log('train/loss_l1', l_l1.detach(), on_step=False, on_epoch=True)

        opt_g.zero_grad()
        if not torch.isfinite(g_loss):
            self.log('train/g_loss_nan_skip', 1.0, on_step=False, on_epoch=True, reduce_fx='sum')
        else:
            self.manual_backward(g_loss)
            if not _grads_finite(self.refiner.parameters()):
                self.log('train/g_grad_nan_skip', 1.0, on_step=False, on_epoch=True, reduce_fx='sum')
                opt_g.zero_grad()
            else:
                if self.grad_clip_g > 0:
                    torch.nn.utils.clip_grad_norm_(self.refiner.parameters(), self.grad_clip_g)
                opt_g.step()

        with torch.no_grad():
            d_real_mean = _logit_mean(real_main).detach()
            d_fake_mean = _logit_mean(fake_main).detach()
            residual_mean = residual.abs().mean().detach()
        self.log_dict({
            'train/d_loss': d_loss.detach(),
            'train/g_loss': g_loss.detach(),
            'train/d_real_mean': d_real_mean,
            'train/d_fake_mean': d_fake_mean,
            'train/residual_mean': residual_mean,
        }, prog_bar=True, on_step=False, on_epoch=True)

    # ------------------------------------------------------------------ val

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        sar, opt = batch
        if self.use_channels_last:
            sar = sar.contiguous(memory_format=torch.channels_last)
            opt = opt.contiguous(memory_format=torch.channels_last)

        coarse = self.netG(sar)
        residual = self.refiner(sar, coarse)
        refined = (coarse + residual).clamp(-1.0, 1.0)

        refined_f = refined.float()
        coarse_f  = coarse.float().clamp(-1.0, 1.0)
        opt_f     = opt.float().clamp(-1.0, 1.0)

        self.psnr.update(refined_f, opt_f)
        self.ssim.update(refined_f, opt_f)
        self.lpips.update(refined_f, opt_f)

        refined_01 = (refined_f + 1.0) * 0.5
        opt_01     = (opt_f + 1.0) * 0.5
        self.fid.update(opt_01, real=True)
        self.fid.update(refined_01, real=False)

        # Coarse PSNR for direct comparison vs refined.
        coarse_mse = F.mse_loss(coarse_f, opt_f)
        coarse_psnr = 10.0 * torch.log10(torch.tensor(4.0, device=coarse_f.device)
                                          / coarse_mse.clamp_min(1e-12))
        self.log('val/coarse_psnr', coarse_psnr, on_epoch=True, sync_dist=True)

    def on_validation_epoch_end(self):
        psnr_now = float(self.psnr.compute())
        self.log('val/psnr',  psnr_now, prog_bar=True)
        self.log('val/ssim',  self.ssim.compute(),  prog_bar=True)
        self.log('val/lpips', self.lpips.compute(), prog_bar=True)
        self.log('val/fid',   self.fid.compute(),   prog_bar=True)
        self.psnr.reset();  self.ssim.reset();  self.lpips.reset();  self.fid.reset()

        # Mode-collapse abort gate (val side) — flat PSNR across checks.
        if self._prev_val_psnr is not None:
            if abs(psnr_now - self._prev_val_psnr) < 1e-4 and self.current_epoch >= 3:
                self._mode_collapse_strikes += 1
                print(
                    f"[ABORT-GATE] val/psnr identical across checks "
                    f"({psnr_now:.4f}, strike #{self._mode_collapse_strikes} "
                    f"at ep{self.current_epoch})."
                )
            else:
                self._mode_collapse_strikes = 0
        self._prev_val_psnr = psnr_now

    # ------------------------------------------------------------------ epoch end

    def on_train_epoch_end(self):
        self.sched_d.step()
        self.sched_g.step()

        # Mode-collapse abort gate (train side): D winning too hard.
        cm = self.trainer.callback_metrics
        d_real = cm.get('train/d_real_mean')
        d_fake = cm.get('train/d_fake_mean')
        if d_real is not None and d_fake is not None and self.current_epoch >= 3:
            diff = float(d_real) - float(d_fake)
            if diff < 0.05:
                print(
                    f"[ABORT-GATE] D collapse signature: d_real - d_fake = "
                    f"{diff:.4f} at ep{self.current_epoch} (< 0.05). Consider "
                    f"raising l1_weight or lowering lr_d."
                )

        self._maybe_save_viz()

    # ------------------------------------------------------------------ viz

    def _maybe_save_viz(self) -> None:
        img_freq = int(getattr(self.cfg.system, 'image_freq', 0))
        if img_freq <= 0 or self.fixed_sar is None:
            return
        if (self.current_epoch + 1) % img_freq != 0:
            return

        self.refiner.eval()
        self.netG.eval()
        with torch.no_grad():
            sar = self.fixed_sar.to(self.device)
            opt = self.fixed_opt.to(self.device)
            if self.use_channels_last:
                sar = sar.contiguous(memory_format=torch.channels_last)
                opt = opt.contiguous(memory_format=torch.channels_last)
            coarse = self.netG(sar)
            residual = self.refiner(sar, coarse)
            refined = (coarse + residual).clamp(-1.0, 1.0)
        self.refiner.train()

        if self.logger is not None and hasattr(self.logger, 'experiment'):
            import torchvision.utils as vutils
            grid = vutils.make_grid(
                torch.cat([
                    sar.expand(-1, 3, -1, -1).clamp(-1, 1),
                    coarse.clamp(-1, 1),
                    refined.clamp(-1, 1),
                    opt.clamp(-1, 1),
                ], dim=0),
                nrow=sar.size(0), normalize=True, value_range=(-1, 1),
            )
            try:
                self.logger.experiment.add_image(
                    'fixed/sar_coarse_refined_opt', grid, self.current_epoch,
                )
            except Exception:
                pass
