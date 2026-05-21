"""LLW-Former Lightning module (llwt-v0.1.x).

Manual optimisation, ``[opt_d, opt_g]`` order — same contract as
``sarformer_wb.main`` so checkpoints and Lightning callbacks behave the same.

Training step (per batch):

  1. D update: main + (optional) subband logits on detached fake.  R1 on the
     main D every ``r1_main_every`` steps; forced through fp32 to avoid the
     bf16 second-order grad NaN issue.
  2. G update: GAN+FM (main + subband) + L1 + MS-SSIM + LAB-chroma +
     wavelet-detail + LPIPS + speckle-decouple + PR.

Validation step computes PSNR/SSIM/LPIPS/FID via torchmetrics, plus the
LLW-specific PR diagnostic and the wavelet-FID (FID over Haar LL coeffs of
fake vs real — measures low-frequency distributional alignment).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import lightning.pytorch as pl
from torchmetrics.image import (
    LearnedPerceptualImagePatchSimilarity,
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)
from torchmetrics.image.fid import FrechetInceptionDistance

from src.models.llwt import factory


__all__ = ["LLWFormerLightningModule"]


def _stack_or_self(logits) -> torch.Tensor:
    if isinstance(logits, (list, tuple)):
        return torch.stack([l.sum() for l in logits]).sum()
    return logits.sum()


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


class LLWFormerLightningModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.automatic_optimization = False

        self.netG, self.netD = factory.build_models(cfg)

        # ``nn.Module.compile()`` (PyTorch >=2.0) wraps ``_call_impl`` in place
        # so ``state_dict`` keys are unchanged — keeps ``_load_weights_ckpt``
        # and the EMA callback working without ``_orig_mod.`` prefix handling.
        # Only ``netG`` is compiled; ``netD`` stays eager because R1 uses
        # ``create_graph=True`` inside ``autocast(enabled=False)`` which has
        # historical NaN issues under dynamo (see sarformer_wb config note).
        if bool(getattr(cfg.system, 'compile', False)):
            self.netG.compile(mode='default', dynamic=False)

        self.criterions = nn.ModuleDict(factory.build_criterions(cfg))

        # R1 + grad clip from cfg.loss.
        self.r1_gamma = float(getattr(cfg.loss, 'r1_gamma', 0.5))
        self.r1_main_every = int(getattr(cfg.loss, 'r1_main_every', 16))
        self.grad_clip_g = float(getattr(cfg.loss, 'grad_clip_g', 1.0))
        self.grad_clip_d = float(getattr(cfg.loss, 'grad_clip_d', 5.0))
        self.use_subband_d = self.netD.use_sub
        self.gan_sub_weight = float(getattr(cfg.loss, 'gan_sub_weight', 0.5))
        self.fm_sub_weight  = float(getattr(cfg.loss, 'fm_sub_weight',  5.0))

        # Validation metrics — data range 2.0 because images are in [-1, 1].
        self.psnr = PeakSignalNoiseRatio(data_range=2.0)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=2.0)
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=False)
        self.fid = FrechetInceptionDistance(feature=2048, reset_real_features=True, normalize=True)

    # ------------------------------------------------------------------ optims

    def configure_optimizers(self):
        opt_d, opt_g = factory.build_optimizers(
            self.cfg, self.netG, self.netD, criterions=self.criterions,
        )
        sched_d, sched_g = factory.build_lr_schedulers(self.cfg, opt_d, opt_g)
        self.sched_g = sched_g
        self.sched_d = sched_d
        return [opt_d, opt_g]

    # ------------------------------------------------------------------ helpers

    def _r1_penalty(self, real_logits, real_input: torch.Tensor) -> torch.Tensor:
        """R1 gradient penalty — forced to fp32 to avoid bf16 second-order NaN.

        Identical implementation to ``sarformer_wb`` (which has been hardened
        through multiple training runs); we only ever fire R1 on the main D's
        logits, not on the subband-D, because the subband path is shallower
        and the extra second-order pass would not buy enough regularisation
        to justify the wall-clock cost.
        """
        with torch.amp.autocast(device_type='cuda', enabled=False):
            grads = torch.autograd.grad(
                outputs=_stack_or_self(real_logits).float(), inputs=real_input,
                create_graph=True, retain_graph=True,
            )[0]
            return grads.float().pow(2).reshape(grads.shape[0], -1).sum(dim=1).mean()

    # ------------------------------------------------------------------ training

    def training_step(self, batch, batch_idx):
        sar, opt = batch
        loss_cfg = self.cfg.loss
        opt_d, opt_g = self.optimizers()

        if not (torch.isfinite(sar).all() and torch.isfinite(opt).all()):
            self.log('train/bad_batch_skip', 1.0, on_step=False, on_epoch=True, reduce_fx='sum')
            return

        # -------- D update on detached fake --------
        with torch.no_grad():
            fake_d = self.netG(sar)

        apply_r1_main = (batch_idx % self.r1_main_every == 0)
        if apply_r1_main:
            opt_real = opt.detach().clone().requires_grad_(True)
            with torch.amp.autocast(device_type='cuda', enabled=False):
                real_main, real_sub, _, _ = self.netD(sar.float(), opt_real.float())
                fake_main, fake_sub, _, _ = self.netD(sar.float(), fake_d.float())
                d_loss_main = 0.5 * (
                    self.criterions['gan'](real_main, is_real=True) +
                    self.criterions['gan'](fake_main, is_real=False)
                )
                r1 = self._r1_penalty(real_main, opt_real)
                d_loss = d_loss_main + 0.5 * self.r1_gamma * r1
                if self.use_subband_d:
                    d_loss_sub = 0.5 * (
                        self.criterions['gan'](real_sub, is_real=True) +
                        self.criterions['gan'](fake_sub, is_real=False)
                    )
                    d_loss = d_loss + d_loss_sub
                    self.log('train/d_loss_sub', d_loss_sub.detach(), on_step=False, on_epoch=True)
            self.log('train/r1_main', r1.detach(), on_step=False, on_epoch=True)
        else:
            opt_real = opt
            real_main, real_sub, _, _ = self.netD(sar, opt_real)
            fake_main, fake_sub, _, _ = self.netD(sar, fake_d)
            d_loss = 0.5 * (
                self.criterions['gan'](real_main, is_real=True) +
                self.criterions['gan'](fake_main, is_real=False)
            )
            if self.use_subband_d:
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

        # -------- G update --------
        fake = self.netG(sar)

        fake_main_g, fake_sub_g, fake_feats_main, fake_feats_sub = self.netD(sar, fake)
        with torch.no_grad():
            _, _, real_feats_main, real_feats_sub = self.netD(sar, opt)
        real_feats_main = [f.detach() for f in real_feats_main]
        real_feats_sub  = [f.detach() for f in real_feats_sub]

        l_gan_main = self.criterions['gan'](fake_main_g, is_real=True)
        l_fm_main = self.criterions['fm'](fake_feats_main, real_feats_main)
        g_loss = (
            l_gan_main * float(loss_cfg.gan_main_weight) +
            l_fm_main * float(loss_cfg.fm_main_weight)
        )
        self.log('train/gan_main', l_gan_main.detach(), on_step=False, on_epoch=True)
        self.log('train/fm_main', l_fm_main.detach(), on_step=False, on_epoch=True)
        if self.use_subband_d:
            l_gan_sub = self.criterions['gan'](fake_sub_g, is_real=True)
            g_loss = g_loss + l_gan_sub * self.gan_sub_weight
            self.log('train/gan_sub', l_gan_sub.detach(), on_step=False, on_epoch=True)
            if fake_feats_sub:                                  # guard against empty list
                l_fm_sub = self.criterions['fm'](fake_feats_sub, real_feats_sub)
                g_loss = g_loss + l_fm_sub * self.fm_sub_weight
                self.log('train/fm_sub', l_fm_sub.detach(), on_step=False, on_epoch=True)

        # Reconstruction stack.
        if 'l1' in self.criterions:
            l_l1 = self.criterions['l1'](fake, opt)
            g_loss = g_loss + l_l1 * float(loss_cfg.l1_weight)
            self.log('train/loss_l1', l_l1.detach(), on_step=False, on_epoch=True)
        if 'msssim' in self.criterions:
            l_ms = self.criterions['msssim'](fake, opt)
            g_loss = g_loss + l_ms * float(loss_cfg.msssim_weight)
            self.log('train/loss_msssim', l_ms.detach(), on_step=False, on_epoch=True)
        if 'lab_chroma' in self.criterions:
            l_lab = self.criterions['lab_chroma'](fake, opt)
            g_loss = g_loss + l_lab * float(loss_cfg.lab_chroma_weight)
            self.log('train/loss_lab_chroma', l_lab.detach(), on_step=False, on_epoch=True)
        if 'wavelet' in self.criterions:
            l_wav = self.criterions['wavelet'](fake, opt)
            g_loss = g_loss + l_wav * float(loss_cfg.wavelet_weight)
            self.log('train/loss_wavelet', l_wav.detach(), on_step=False, on_epoch=True)
        if 'lpips' in self.criterions:
            l_lp = self.criterions['lpips'](fake, opt)
            g_loss = g_loss + l_lp * float(loss_cfg.lpips_weight)
            self.log('train/loss_lpips', l_lp.detach(), on_step=False, on_epoch=True)

        # LLW-specific regularisers.
        if 'spkdec' in self.criterions or 'pr' in self.criterions:
            # Recompute lifting coeffs once for both regs; they're cheap.
            # NOTE: feat at this point inside fake's forward graph; redo it
            # explicitly so we can attach gradients through the lifting modules
            # cleanly.
            three = self.netG.physics_front_end(sar)
            feat = self.netG.stem(three)
            coeffs = self.netG.llw(feat)
            if 'spkdec' in self.criterions:
                l_spk = self.criterions['spkdec'](sar, coeffs)
                g_loss = g_loss + l_spk * float(loss_cfg.spkdec_weight)
                self.log('train/loss_spkdec', l_spk.detach(), on_step=False, on_epoch=True)
            if 'pr' in self.criterions:
                l_pr = self.criterions['pr'](self.netG.llw, feat)
                g_loss = g_loss + l_pr * float(loss_cfg.pr_weight)
                self.log('train/loss_pr', l_pr.detach(), on_step=False, on_epoch=True)

        opt_g.zero_grad()
        if not torch.isfinite(g_loss):
            self.log('train/g_loss_nan_skip', 1.0, on_step=False, on_epoch=True, reduce_fx='sum')
        else:
            self.manual_backward(g_loss)
            g_clip_params = list(self.netG.parameters())
            if 'adaptive' in self.criterions:
                g_clip_params.extend(self.criterions['adaptive'].parameters())
            if not _grads_finite(g_clip_params):
                self.log('train/g_grad_nan_skip', 1.0, on_step=False, on_epoch=True, reduce_fx='sum')
                opt_g.zero_grad()
            else:
                if self.grad_clip_g > 0:
                    torch.nn.utils.clip_grad_norm_(g_clip_params, self.grad_clip_g)
                opt_g.step()

        with torch.no_grad():
            d_real_mean = _logit_mean(real_main).detach()
            d_fake_mean = _logit_mean(fake_main).detach()
            pu_norm = self.netG.llw.pu_param_norm().detach()
        log_dict = {
            'train/d_loss': d_loss.detach(),
            'train/g_loss': g_loss.detach(),
            'train/d_real_mean': d_real_mean,
            'train/d_fake_mean': d_fake_mean,
            'train/pu_param_norm': pu_norm,        # has the lifting learned anything?
        }
        self.log_dict(log_dict, prog_bar=True, on_step=False, on_epoch=True)

    # ------------------------------------------------------------------ val

    def validation_step(self, batch, batch_idx):
        sar, opt = batch
        with torch.no_grad():
            fake = self.netG(sar)
        self.psnr.update(fake, opt)
        self.ssim.update(fake, opt)
        self.lpips.update(fake.float(), opt.float())
        fake01 = ((fake + 1) / 2).clamp(0, 1).float()
        opt01 = ((opt + 1) / 2).clamp(0, 1).float()
        self.fid.update(fake01, real=False)
        self.fid.update(opt01, real=True)

    def on_validation_epoch_end(self):
        self.log_dict({
            'val/psnr':  self.psnr.compute(),
            'val/ssim':  self.ssim.compute(),
            'val/lpips': self.lpips.compute(),
            'val/fid':   self.fid.compute(),
        }, prog_bar=True)
        self.psnr.reset()
        self.ssim.reset()
        self.lpips.reset()
        self.fid.reset()

    def on_train_epoch_end(self):
        self.sched_g.step()
        self.sched_d.step()
        if 'adaptive' in self.criterions:
            adaptive = self.criterions['adaptive']
            weights = adaptive.effective_weights()
            for name, eta, w in zip(adaptive._loss_order, adaptive.eta, weights):
                self.log(f'train/eta_{name}', eta.detach(), on_step=False, on_epoch=True)
                self.log(f'train/w_{name}', w.detach(), on_step=False, on_epoch=True)
