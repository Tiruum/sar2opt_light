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

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
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

        # NHWC (channels_last) memory format: cuDNN benchmark picks faster conv
        # kernels on Ada/Ampere when both weights and activations are NHWC.
        # Free 10-15% throughput on the conv-heavy lifting + subband-D paths;
        # 1-channel SAR is essentially a no-op but the 3-channel optical/RGB
        # path and the embed_dim=96 PSwE/decoder tensors benefit fully.
        self.use_channels_last = bool(getattr(cfg.system, 'channels_last', False))
        if self.use_channels_last:
            self.netG = self.netG.to(memory_format=torch.channels_last)
            self.netD = self.netD.to(memory_format=torch.channels_last)

        self.criterions = nn.ModuleDict(factory.build_criterions(cfg))

        # R1 + grad clip from cfg.loss.
        self.r1_gamma = float(getattr(cfg.loss, 'r1_gamma', 0.5))
        self.r1_main_every = int(getattr(cfg.loss, 'r1_main_every', 16))
        self.grad_clip_g = float(getattr(cfg.loss, 'grad_clip_g', 1.0))
        self.grad_clip_d = float(getattr(cfg.loss, 'grad_clip_d', 5.0))
        self.use_subband_d = self.netD.use_sub
        self.gan_sub_weight = float(getattr(cfg.loss, 'gan_sub_weight', 0.5))
        self.fm_sub_weight  = float(getattr(cfg.loss, 'fm_sub_weight',  5.0))

        # Mode-collapse abort-gate state (pure GAN+FM recipe has no pixel
        # anchor; without these checks we can train through a silent collapse).
        self._prev_val_psnr: Optional[float] = None
        self._mode_collapse_strikes = 0

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

        if self.use_channels_last:
            sar = sar.contiguous(memory_format=torch.channels_last)
            opt = opt.contiguous(memory_format=torch.channels_last)

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
            # Batched D real+fake forward: D is per-sample (spectral_norm +
            # F.instance_norm(sar) + Conv2d + LeakyReLU; no BN/GN), so processing
            # 2B samples in one call is bit-identical to two separate D calls
            # but pays one set of kernel launches instead of two.
            opt_real = opt
            B = sar.size(0)
            sar_doubled = sar.repeat(2, 1, 1, 1)
            opt_doubled = torch.cat([opt_real, fake_d], dim=0)
            main_both, sub_both, _, _ = self.netD(sar_doubled, opt_doubled)
            lc_both, lf_both = main_both
            real_main = (lc_both[:B], lf_both[:B])
            fake_main = (lc_both[B:], lf_both[B:])
            if self.use_subband_d:
                real_sub = sub_both[:B]
                fake_sub = sub_both[B:]
            else:
                real_sub, fake_sub = None, None
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
        # If spkdec or PR is active, ask the generator to return the pre-encoder
        # raw lifting coeffs and the post-stem feature tensor.  Both losses
        # previously re-ran ``physics_front_end -> stem -> llw`` outside the
        # compiled forward, costing one extra (uncompiled) lifting pass per
        # step.  Caching them removes that redundancy at no math cost.
        need_internals = ('spkdec' in self.criterions) or ('pr' in self.criterions)
        if need_internals:
            fake, raw_coeffs, raw_feat = self.netG(sar, return_internals=True)
        else:
            fake = self.netG(sar)

        # Batched D forward in G phase: fake (with grad) + opt (we want detached
        # features for FM).  Slice fake half of the output / features for the
        # G-loss path, .detach() the real half for FM target.  D's grad
        # accumulates from the fake half only (opt has no requires_grad source
        # and the real_feats slice is detached before FM); same as the previous
        # two-call layout, just one set of kernel launches.
        B = sar.size(0)
        sar_doubled_g = sar.repeat(2, 1, 1, 1)
        opt_doubled_g = torch.cat([fake, opt], dim=0)
        main_both_g, sub_both_g, feats_main_both, feats_sub_both = self.netD(sar_doubled_g, opt_doubled_g)
        lc_both_g, lf_both_g = main_both_g
        fake_main_g = (lc_both_g[:B], lf_both_g[:B])
        fake_feats_main = [f[:B] for f in feats_main_both]
        real_feats_main = [f[B:].detach() for f in feats_main_both]
        if self.use_subband_d:
            fake_sub_g = sub_both_g[:B]
            fake_feats_sub = [f[:B] for f in feats_sub_both]
            real_feats_sub = [f[B:].detach() for f in feats_sub_both]
        else:
            fake_sub_g = None
            fake_feats_sub = []
            real_feats_sub = []

        l_gan_main = self.criterions['gan'](fake_main_g, is_real=True, for_d=False)
        l_fm_main = self.criterions['fm'](fake_feats_main, real_feats_main)
        g_loss = (
            l_gan_main * float(loss_cfg.gan_main_weight) +
            l_fm_main * float(loss_cfg.fm_main_weight)
        )
        self.log('train/gan_main', l_gan_main.detach(), on_step=False, on_epoch=True)
        self.log('train/fm_main', l_fm_main.detach(), on_step=False, on_epoch=True)
        if self.use_subband_d:
            l_gan_sub = self.criterions['gan'](fake_sub_g, is_real=True, for_d=False)
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

        # LLW-specific regularisers — reuse the raw coeffs + feat already
        # produced by the G forward above (return_internals=True).  Gradients
        # still flow through the lifting modules because raw_coeffs and
        # raw_feat are tracked tensors from the same compiled forward graph.
        if need_internals:
            if 'spkdec' in self.criterions:
                l_spk = self.criterions['spkdec'](sar, raw_coeffs)
                g_loss = g_loss + l_spk * float(loss_cfg.spkdec_weight)
                self.log('train/loss_spkdec', l_spk.detach(), on_step=False, on_epoch=True)
            if 'pr' in self.criterions:
                # Inline PR to skip a second ``self.netG.llw(raw_feat)`` call;
                # raw_coeffs is already that result.  Only the inverse pass is
                # needed for the |x - iLLW(LLW(x))| target.
                x_hat = self.netG.llw.inverse(raw_coeffs)
                l_pr = F.l1_loss(x_hat, raw_feat)
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
        if self.use_channels_last:
            sar = sar.contiguous(memory_format=torch.channels_last)
            opt = opt.contiguous(memory_format=torch.channels_last)
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

        # Mode-collapse abort gate (val side): exact PSNR repeat to 4 decimals
        # across consecutive val epochs = G output is deterministic given SAR
        # = mode collapse.  Only the pure GAN+FM recipe is vulnerable; pixel
        # anchors used to mask this with noisy L1 contributions.
        if self._prev_val_psnr is not None:
            delta = abs(psnr_now - self._prev_val_psnr)
            if delta < 1e-4 and self.current_epoch >= 3:
                self._mode_collapse_strikes += 1
                print(
                    f"[ABORT-GATE] val/psnr identical across consecutive checks "
                    f"({psnr_now:.4f} == {self._prev_val_psnr:.4f}, "
                    f"strike #{self._mode_collapse_strikes} at ep{self.current_epoch}). "
                    f"Pure GAN+FM mode-collapse signature — consider §15.6 anchored fallback."
                )
            else:
                self._mode_collapse_strikes = 0
        self._prev_val_psnr = psnr_now

    def on_train_epoch_end(self):
        self.sched_g.step()
        self.sched_d.step()
        if 'adaptive' in self.criterions:
            adaptive = self.criterions['adaptive']
            weights = adaptive.effective_weights()
            for name, eta, w in zip(adaptive._loss_order, adaptive.eta, weights):
                self.log(f'train/eta_{name}', eta.detach(), on_step=False, on_epoch=True)
                self.log(f'train/w_{name}', w.detach(), on_step=False, on_epoch=True)

        # Mode-collapse abort gate (train side): D no longer distinguishes
        # real from fake → adversarial signal dead.  trainer.callback_metrics
        # holds the on_epoch-aggregated values logged in training_step.
        cm = self.trainer.callback_metrics
        d_real = cm.get('train/d_real_mean')
        d_fake = cm.get('train/d_fake_mean')
        if d_real is not None and d_fake is not None and self.current_epoch >= 3:
            diff = float(d_real) - float(d_fake)
            if diff < 0.05:
                print(
                    f"[ABORT-GATE] D collapse signature: d_real - d_fake = {diff:.4f} "
                    f"at ep{self.current_epoch} (< 0.05 threshold). "
                    f"Pure GAN+FM training has no pixel anchor — consider §15.6 fallback."
                )
