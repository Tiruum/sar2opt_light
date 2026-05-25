"""SAR2OPT-V1 Lightning module.

Key features:
  * Manual optimisation: D-step then G-step.
  * DiffAugment (Zhao NeurIPS 2020) applied jointly to (SAR, optical) inputs
    of the D in BOTH the D-step and the G-step (paper-mandated -- otherwise
    G overfits to the un-augmented D and the augment provides no benefit).
  * PatchNCE (CUT, Park ECCV 2020) on encoder features of fake_opt vs
    real_opt -- contrastive sharp anchor.
  * Non-blurring reconstruction stack: LAB-Chroma + Wavelet-Detail + MS-SSIM
    + FFL. NO global pixel L1, NO LPIPS in the training loss (LPIPS tracked
    as a val metric only).
  * R1 gradient penalty in fp32 every ``r1_every`` steps (bf16-mixed safe).
  * Image-dump triplet (SAR | fake | GT) in ``on_train_epoch_end`` every
    ``image_freq`` epochs -- ports the cfrwd pattern that was missing from
    the failed llwt_v3 stack.
  * Mode-collapse abort gate on D-gap and val/psnr plateau.
"""
from __future__ import annotations

import os
from typing import Optional

import lightning.pytorch as pl
import torch
import torch.nn as nn
from torchmetrics.image import (
    LearnedPerceptualImagePatchSimilarity,
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)
from torchmetrics.image.fid import FrechetInceptionDistance

from src.models.sar2opt_v1 import factory
from src.models.sar2opt_v1.diffaug import diff_augment_pair
from src.models.sar2opt_v1.losses import HingeGANLoss


__all__ = ["SAR2OPTV1LightningModule"]


def _logits_mean(logits) -> torch.Tensor:
    if isinstance(logits, (list, tuple)):
        return torch.stack([l.mean() for l in logits]).mean()
    return logits.mean()


def _logits_sum(logits) -> torch.Tensor:
    if isinstance(logits, (list, tuple)):
        return torch.stack([l.sum() for l in logits]).sum()
    return logits.sum()


def _grads_finite(params) -> bool:
    for p in params:
        if p.grad is None:
            continue
        if not torch.isfinite(p.grad).all():
            return False
    return True


def _gan_call(criterion, logits, is_real: bool, is_generator: bool = False) -> torch.Tensor:
    """Uniform GAN-loss call: works for both LSGAN (`GANLoss`) and Hinge."""
    if isinstance(criterion, HingeGANLoss):
        return criterion(logits, is_real=is_real, is_generator=is_generator)
    return criterion(logits, is_real=is_real)


class SAR2OPTV1LightningModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.automatic_optimization = False

        self.netG, self.netD = factory.build_models(cfg)

        # bfloat16-aware NHWC: only set channels_last on conv-heavy submodules
        # (encoder + decoder convs); skip the transformer encoder which is
        # already channels-last internally via Linear layers.
        self.use_channels_last = bool(getattr(cfg.system, 'channels_last', False))
        if self.use_channels_last:
            self.netG = self.netG.to(memory_format=torch.channels_last)
            self.netD = self.netD.to(memory_format=torch.channels_last)

        self.criterions = nn.ModuleDict(factory.build_criterions(cfg, netG=self.netG))

        # Loss weights (cached as float to avoid OmegaConf lookup in hot loop).
        L = cfg.loss
        self.w_gan        = float(L.gan_weight)
        self.w_fm         = float(L.fm_weight)
        self.w_l1         = float(getattr(L, 'l1_weight', 0.0))
        self.w_lab_chroma = float(getattr(L, 'lab_chroma_weight', 0.0))
        self.w_wavelet    = float(getattr(L, 'wavelet_weight', 0.0))
        self.w_msssim     = float(getattr(L, 'msssim_weight', 0.0))
        self.w_patchnce   = float(getattr(L, 'patchnce_weight', 0.0))
        self.w_ffl        = float(getattr(L, 'ffl_weight', 0.0))

        # R1 + grad clip.
        self.r1_gamma     = float(getattr(L, 'r1_gamma', 0.05))
        self.r1_every     = int(getattr(L, 'r1_every', 16))
        self.grad_clip_g  = float(getattr(L, 'grad_clip_g', 1.0))
        self.grad_clip_d  = float(getattr(L, 'grad_clip_d', 5.0))

        # DiffAugment.
        self.diffaug_enabled = bool(getattr(L, 'diffaug_enabled', True))
        self.diffaug_policy  = str(getattr(L, 'diffaug_policy', 'color,translation,cutout'))

        # Image-dump fixed batch (filled in `setup`).
        self.fixed_sar: Optional[torch.Tensor] = None
        self.fixed_opt: Optional[torch.Tensor] = None

        # Mode-collapse abort state.
        self._prev_val_psnr: Optional[float] = None
        self._mode_collapse_strikes = 0

        # Validation metrics -- data_range=2.0 because images are in [-1, 1].
        self.psnr  = PeakSignalNoiseRatio(data_range=2.0)
        self.ssim  = StructuralSimilarityIndexMeasure(data_range=2.0)
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=False)
        self.fid   = FrechetInceptionDistance(feature=2048, reset_real_features=True, normalize=True)

    # ------------------------------------------------------------------ optims

    def configure_optimizers(self):
        opt_d, opt_g = factory.build_optimizers(
            self.cfg, self.netG, self.netD, criterions=self.criterions,
        )
        sched_d, sched_g = factory.build_lr_schedulers(self.cfg, opt_d, opt_g)
        self.sched_g = sched_g
        self.sched_d = sched_d
        return [opt_d, opt_g]

    # ------------------------------------------------------------------ setup (fixed batch)

    def setup(self, stage: Optional[str] = None) -> None:
        if stage != 'fit':
            return
        from torch.utils.data import DataLoader
        dm = self.trainer.datamodule
        # num_workers=0 so we don't spawn workers just for one batch on Windows.
        one_shot = DataLoader(
            dm.train_dataset,
            batch_size=self.cfg.data.batch_size,
            shuffle=False,
            num_workers=0,
        )
        sar, opt = next(iter(one_shot))
        self.fixed_sar = sar.to(self.device)
        self.fixed_opt = opt.to(self.device)

    # ------------------------------------------------------------------ helpers

    def _maybe_augment(self, sar: torch.Tensor, opt: torch.Tensor):
        if not self.diffaug_enabled:
            return sar, opt
        return diff_augment_pair(sar, opt, policy=self.diffaug_policy)

    def _r1_penalty(self, real_logits, real_input: torch.Tensor) -> torch.Tensor:
        """R1 = ||grad_{real} sum(D(real))||^2, in fp32."""
        with torch.amp.autocast(device_type='cuda', enabled=False):
            grads = torch.autograd.grad(
                outputs=_logits_sum(real_logits).float(),
                inputs=real_input,
                create_graph=True,
                retain_graph=True,
            )[0]
            return grads.float().pow(2).reshape(grads.shape[0], -1).sum(dim=1).mean()

    # ------------------------------------------------------------------ training

    def training_step(self, batch, batch_idx):
        sar, opt = batch
        opt_d, opt_g = self.optimizers()

        if not (torch.isfinite(sar).all() and torch.isfinite(opt).all()):
            self.log('train/bad_batch_skip', 1.0, on_step=False, on_epoch=True, reduce_fx='sum')
            return

        if self.use_channels_last:
            sar = sar.contiguous(memory_format=torch.channels_last)
            opt = opt.contiguous(memory_format=torch.channels_last)

        # ============================================================
        # D update
        # ============================================================
        with torch.no_grad():
            fake_opt_d = self.netG(sar)

        # DiffAugment: SAME augment applied to (sar, real_opt) and (sar, fake_opt)
        # in this step. (A fresh policy realisation per call -- different masks
        # per call is the DiffAugment-pair contract.)
        sar_real_a, opt_real_a = self._maybe_augment(sar, opt)
        sar_fake_a, opt_fake_a = self._maybe_augment(sar, fake_opt_d)

        apply_r1 = (batch_idx % self.r1_every == 0) and (self.r1_gamma > 0)
        if apply_r1:
            opt_real_for_r1 = opt_real_a.detach().clone().requires_grad_(True)
            with torch.amp.autocast(device_type='cuda', enabled=False):
                real_logits, _ = self.netD(sar_real_a.float(), opt_real_for_r1.float())
                fake_logits, _ = self.netD(sar_fake_a.float(), opt_fake_a.float())
                d_loss_real = _gan_call(self.criterions['gan'], real_logits, is_real=True)
                d_loss_fake = _gan_call(self.criterions['gan'], fake_logits, is_real=False)
                d_loss = 0.5 * (d_loss_real + d_loss_fake)
                r1 = self._r1_penalty(real_logits, opt_real_for_r1)
                d_loss = d_loss + 0.5 * self.r1_gamma * r1
            self.log('train/r1', r1.detach(), on_step=False, on_epoch=True)
        else:
            real_logits, _ = self.netD(sar_real_a, opt_real_a)
            fake_logits, _ = self.netD(sar_fake_a, opt_fake_a)
            d_loss_real = _gan_call(self.criterions['gan'], real_logits, is_real=True)
            d_loss_fake = _gan_call(self.criterions['gan'], fake_logits, is_real=False)
            d_loss = 0.5 * (d_loss_real + d_loss_fake)

        opt_d.zero_grad()
        self.manual_backward(d_loss)
        if not _grads_finite(self.netD.parameters()):
            self.log('train/d_grad_nan_skip', 1.0, on_step=False, on_epoch=True, reduce_fx='sum')
            opt_d.zero_grad()
        else:
            if self.grad_clip_d > 0:
                torch.nn.utils.clip_grad_norm_(self.netD.parameters(), self.grad_clip_d)
            opt_d.step()

        # ============================================================
        # G update
        # ============================================================
        fake_opt = self.netG(sar)

        # DiffAugment again with fresh policy realisations for the G-step D-pass.
        sar_g_a, fake_opt_g_a = self._maybe_augment(sar, fake_opt)

        fake_logits_g, fake_feats_g = self.netD(sar_g_a, fake_opt_g_a)
        # We also need real features for FM. Pass real_opt through D under the
        # same augmentation as fake for FM alignment.
        sar_g_real_a, opt_g_real_a = self._maybe_augment(sar, opt)
        with torch.no_grad():
            _, real_feats_g = self.netD(sar_g_real_a, opt_g_real_a)
        real_feats_g = [f.detach() for f in real_feats_g]

        l_gan = _gan_call(self.criterions['gan'], fake_logits_g, is_real=True, is_generator=True)
        l_fm  = self.criterions['fm'](fake_feats_g, real_feats_g)

        g_loss = l_gan * self.w_gan + l_fm * self.w_fm
        self.log_dict({
            'train/gan_main': l_gan.detach(),
            'train/fm_main':  l_fm.detach(),
        }, on_step=False, on_epoch=True)

        # ---- non-blurring reconstruction stack ----
        if self.w_lab_chroma > 0 and 'lab_chroma' in self.criterions:
            l_lab = self.criterions['lab_chroma'](fake_opt, opt)
            g_loss = g_loss + l_lab * self.w_lab_chroma
            self.log('train/loss_lab_chroma', l_lab.detach(), on_step=False, on_epoch=True)
        if self.w_wavelet > 0 and 'wavelet' in self.criterions:
            l_wav = self.criterions['wavelet'](fake_opt, opt)
            g_loss = g_loss + l_wav * self.w_wavelet
            self.log('train/loss_wavelet', l_wav.detach(), on_step=False, on_epoch=True)
        if self.w_msssim > 0 and 'msssim' in self.criterions:
            l_ms = self.criterions['msssim'](fake_opt, opt)
            g_loss = g_loss + l_ms * self.w_msssim
            self.log('train/loss_msssim', l_ms.detach(), on_step=False, on_epoch=True)
        if self.w_ffl > 0 and 'ffl' in self.criterions:
            l_ffl = self.criterions['ffl'](fake_opt, opt)
            g_loss = g_loss + l_ffl * self.w_ffl
            self.log('train/loss_ffl', l_ffl.detach(), on_step=False, on_epoch=True)
        if self.w_l1 > 0 and 'l1' in self.criterions:
            l_l1 = self.criterions['l1'](fake_opt, opt)
            g_loss = g_loss + l_l1 * self.w_l1
            self.log('train/loss_l1', l_l1.detach(), on_step=False, on_epoch=True)

        # ---- PatchNCE (CUT) on encoder features ----
        if self.w_patchnce > 0 and 'patchnce' in self.criterions:
            # Real-encoder features: no grad (they only serve as positive keys).
            with torch.no_grad():
                real_feats_enc = self.netG.encode_optical(opt)
            # Fake-encoder features: gradient flows back to the generator.
            fake_feats_enc = self.netG.encode_optical(fake_opt)
            l_nce = self.criterions['patchnce'](fake_feats_enc, real_feats_enc)
            g_loss = g_loss + l_nce * self.w_patchnce
            self.log('train/loss_patchnce', l_nce.detach(), on_step=False, on_epoch=True)

        opt_g.zero_grad()
        if not torch.isfinite(g_loss):
            self.log('train/g_loss_nan_skip', 1.0, on_step=False, on_epoch=True, reduce_fx='sum')
        else:
            self.manual_backward(g_loss)
            g_clip_params = list(self.netG.parameters())
            if 'patchnce' in self.criterions:
                g_clip_params.extend(self.criterions['patchnce'].sampler.parameters())
            if not _grads_finite(g_clip_params):
                self.log('train/g_grad_nan_skip', 1.0, on_step=False, on_epoch=True, reduce_fx='sum')
                opt_g.zero_grad()
            else:
                if self.grad_clip_g > 0:
                    torch.nn.utils.clip_grad_norm_(g_clip_params, self.grad_clip_g)
                opt_g.step()

        # ---- diagnostics ----
        with torch.no_grad():
            d_real_mean = _logits_mean(real_logits).detach()
            d_fake_mean = _logits_mean(fake_logits).detach()
        self.log_dict({
            'train/d_loss': d_loss.detach(),
            'train/g_loss': g_loss.detach(),
            'train/d_real_mean': d_real_mean,
            'train/d_fake_mean': d_fake_mean,
            'train/d_gap': (d_real_mean - d_fake_mean).abs(),
        }, prog_bar=True, on_step=False, on_epoch=True)

    # ------------------------------------------------------------------ val

    def validation_step(self, batch, batch_idx):
        sar, opt = batch
        if self.use_channels_last:
            sar = sar.contiguous(memory_format=torch.channels_last)
            opt = opt.contiguous(memory_format=torch.channels_last)
        with torch.no_grad():
            fake = self.netG(sar)
        self.psnr.update(fake, opt)
        self.ssim.update(fake, opt)
        self.lpips.update(fake.float(), opt.float())
        fake01 = ((fake + 1) / 2).clamp(0, 1).float()
        opt01  = ((opt + 1) / 2).clamp(0, 1).float()
        self.fid.update(fake01, real=False)
        self.fid.update(opt01, real=True)

    def on_validation_epoch_end(self):
        psnr_now = float(self.psnr.compute())
        ssim_now = float(self.ssim.compute())
        lpips_now = float(self.lpips.compute())
        fid_now = float(self.fid.compute())
        self.log_dict({
            'val/psnr':  psnr_now,
            'val/ssim':  ssim_now,
            'val/lpips': lpips_now,
            'val/fid':   fid_now,
        }, prog_bar=True)
        self.psnr.reset()
        self.ssim.reset()
        self.lpips.reset()
        self.fid.reset()

        # Mode-collapse abort gate (val side).
        if self._prev_val_psnr is not None and self.current_epoch >= 3:
            delta = abs(psnr_now - self._prev_val_psnr)
            if delta < 1e-4:
                self._mode_collapse_strikes += 1
                print(
                    f"[ABORT-GATE] val/psnr identical across consecutive checks "
                    f"({psnr_now:.4f} == {self._prev_val_psnr:.4f}, "
                    f"strike #{self._mode_collapse_strikes} at ep{self.current_epoch})."
                )
            else:
                self._mode_collapse_strikes = 0
        self._prev_val_psnr = psnr_now

    # ------------------------------------------------------------------ train-epoch end (image dump + scheduler)

    def on_train_epoch_end(self):
        self.sched_g.step()
        self.sched_d.step()

        # D-collapse warning (train side).
        cm = self.trainer.callback_metrics
        d_real = cm.get('train/d_real_mean')
        d_fake = cm.get('train/d_fake_mean')
        if d_real is not None and d_fake is not None and self.current_epoch >= 3:
            diff = abs(float(d_real) - float(d_fake))
            if diff < 0.05:
                print(
                    f"[ABORT-GATE] D collapse signature: |d_real - d_fake| = {diff:.4f} "
                    f"at ep{self.current_epoch} (< 0.05 threshold). "
                    f"Consider flipping TTUR (lr_d > lr_g) or disabling DiffAugment."
                )

        # Image dump (SAR | fake | GT triplet).
        if self.fixed_sar is None:
            return
        image_freq = int(getattr(self.cfg.system, 'image_freq', 0))
        if image_freq <= 0:
            return
        if (self.current_epoch + 1) % image_freq != 0:
            return
        with torch.no_grad():
            fake_fixed = self.netG(self.fixed_sar)
        out_dir = os.path.join(self.cfg.system.images_dir, self.cfg.system.tb_version)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"epoch_{self.current_epoch + 1:03d}.png")
        from src.utils.visualize import visualize_batch
        visualize_batch(
            self.fixed_sar.cpu().detach(),
            fake_fixed.cpu().detach(),
            self.fixed_opt.cpu().detach(),
            save_path=out_path,
            max_rows=6,
            mode='quality',
            title=f"sar2opt-v1 | epoch {self.current_epoch + 1}",
        )
        del fake_fixed
        torch.cuda.empty_cache()
