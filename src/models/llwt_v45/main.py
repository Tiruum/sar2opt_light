"""LLW-Former v0.4.0 Lightning module.

Clone of ``src/models/llwt_v3/main.py`` with two differences:
  * Imports ``factory`` from ``src.models.llwt_v4`` (Haar-Stem ConvNeXt V2 +
    Inverse-Haar-head generator + reused LLW discriminator).
  * Drops the ``pu_param_norm`` log line and inline-PR helper — v0.4.0 has no
    learnable lifting wavelet, so those LLW-only diagnostics would AttributeError.
    The ``spkdec`` / ``pr`` criterion code paths remain (config-gated by weight > 0)
    so the rollback path in §15.6 keeps working if we later swap learnable lifting
    back in for an ablation.

Stage-2 active when ``loss.per_band_weight`` > 0: ``training_step`` asks the
generator for ``return_internals`` so the per-band L1 loss can score the
predicted subband tensor (``predicted_sub``, shape ``(B, 3, 4, H/2, W/2)``,
dim=2 order ``[LL, LH, HL, HH]``) against ``HaarDown(opt)``.  Default config
ships ``per_band_weight = 0.0`` (no behaviour change vs v0.4.0); explicit
opt-in per the v0.4.1 ablation matrix.

All other behaviour (manual optimisation, batched D, R1 in fp32, channels_last,
mode-collapse abort gates) is identical to v0.3.x.
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

from src.models.llwt_v45 import factory


__all__ = ["LLWv4LightningModule"]


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


class LLWv4LightningModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.automatic_optimization = False

        self.netG, self.netD = factory.build_models(cfg)

        # ``nn.Module.compile()`` (PyTorch >=2.0) wraps ``_call_impl`` in place
        # so ``state_dict`` keys are unchanged.
        if bool(getattr(cfg.system, 'compile', False)):
            self.netG.compile(mode='default', dynamic=False)

        # NHWC (channels_last) memory format: cuDNN benchmark picks faster conv
        # kernels on Ada/Ampere when both weights and activations are NHWC.
        self.use_channels_last = bool(getattr(cfg.system, 'channels_last', False))
        if self.use_channels_last:
            self.netG = self.netG.to(memory_format=torch.channels_last)
            self.netD = self.netD.to(memory_format=torch.channels_last)

        self.criterions = nn.ModuleDict(factory.build_criterions(cfg, netG=self.netG))

        # R1 + grad clip from cfg.loss.
        self.r1_gamma = float(getattr(cfg.loss, 'r1_gamma', 0.5))
        self.r1_main_every = int(getattr(cfg.loss, 'r1_main_every', 16))
        self.grad_clip_g = float(getattr(cfg.loss, 'grad_clip_g', 1.0))
        self.grad_clip_d = float(getattr(cfg.loss, 'grad_clip_d', 5.0))
        self.use_subband_d = self.netD.use_sub
        self.gan_sub_weight = float(getattr(cfg.loss, 'gan_sub_weight', 1.0))
        self.fm_sub_weight  = float(getattr(cfg.loss, 'fm_sub_weight',  10.0))
        # Fourier-D (v0.4.2 stage 3) — gated by cfg.model.dis.fourier.enabled.
        # Default False on both sides keeps v0.4.0/v0.4.1 runs unchanged.
        self.use_fourier_d = bool(getattr(self.netD, 'use_fourier', False))
        self.gan_fourier_weight = float(getattr(cfg.loss, 'gan_fourier_weight', 1.0))
        self.fm_fourier_weight  = float(getattr(cfg.loss, 'fm_fourier_weight',  10.0))

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
        """R1 gradient penalty — fp32 to avoid bf16 second-order NaN."""
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
                real_main, real_sub, real_fourier, _, _, _ = self.netD(sar.float(), opt_real.float())
                fake_main, fake_sub, fake_fourier, _, _, _ = self.netD(sar.float(), fake_d.float())
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
                if self.use_fourier_d:
                    d_loss_fourier = 0.5 * (
                        self.criterions['gan'](real_fourier, is_real=True) +
                        self.criterions['gan'](fake_fourier, is_real=False)
                    )
                    d_loss = d_loss + d_loss_fourier
                    self.log('train/d_loss_fourier', d_loss_fourier.detach(), on_step=False, on_epoch=True)
            self.log('train/r1_main', r1.detach(), on_step=False, on_epoch=True)
        else:
            # Batched D real+fake forward (bit-identical to two calls; one set
            # of kernel launches).
            opt_real = opt
            B = sar.size(0)
            sar_doubled = sar.repeat(2, 1, 1, 1)
            opt_doubled = torch.cat([opt_real, fake_d], dim=0)
            main_both, sub_both, fourier_both, _, _, _ = self.netD(sar_doubled, opt_doubled)
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
            if self.use_fourier_d:
                real_fourier = fourier_both[:B]
                fake_fourier = fourier_both[B:]
                d_loss_fourier = 0.5 * (
                    self.criterions['gan'](real_fourier, is_real=True) +
                    self.criterions['gan'](fake_fourier, is_real=False)
                )
                d_loss = d_loss + d_loss_fourier
                self.log('train/d_loss_fourier', d_loss_fourier.detach(), on_step=False, on_epoch=True)
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
        # v0.4.0 had no consumers of the internals path (spkdec/pr both gated
        # to weight 0).  v0.4.1 adds ``per_band`` which DOES consume the
        # predicted subband tensor — extend the gate.  When True, the
        # generator's triple-return contract is:
        #   (out, predicted_sub, raw_feat)
        # where ``predicted_sub`` is the (B, 3, 4, H/2, W/2) Haar subband
        # prediction BEFORE InverseHaarUp.  Order on dim=2 is [LL, LH, HL, HH].
        need_internals = (
            ('spkdec' in self.criterions)
            or ('pr' in self.criterions)
            or ('per_band' in self.criterions)
        )
        if need_internals:
            fake, predicted_sub, raw_feat = self.netG(sar, return_internals=True)
        else:
            fake = self.netG(sar)
            predicted_sub = None

        # Batched D forward in G phase.
        B = sar.size(0)
        sar_doubled_g = sar.repeat(2, 1, 1, 1)
        opt_doubled_g = torch.cat([fake, opt], dim=0)
        main_both_g, sub_both_g, fourier_both_g, feats_main_both, feats_sub_both, feats_fourier_both = self.netD(sar_doubled_g, opt_doubled_g)
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
            if fake_feats_sub:
                l_fm_sub = self.criterions['fm'](fake_feats_sub, real_feats_sub)
                g_loss = g_loss + l_fm_sub * self.fm_sub_weight
                self.log('train/fm_sub', l_fm_sub.detach(), on_step=False, on_epoch=True)
        if self.use_fourier_d:
            # v0.4.2 stage 3: third adversarial head on log|rfft2(opt)|.
            # Mirrors the subband-D G-update block above.
            fake_fourier_g = fourier_both_g[:B]
            fake_feats_fourier = [f[:B] for f in feats_fourier_both]
            real_feats_fourier = [f[B:].detach() for f in feats_fourier_both]
            l_gan_fourier = self.criterions['gan'](fake_fourier_g, is_real=True, for_d=False)
            g_loss = g_loss + l_gan_fourier * self.gan_fourier_weight
            self.log('train/gan_fourier', l_gan_fourier.detach(), on_step=False, on_epoch=True)
            if fake_feats_fourier:
                l_fm_fourier = self.criterions['fm'](fake_feats_fourier, real_feats_fourier)
                g_loss = g_loss + l_fm_fourier * self.fm_fourier_weight
                self.log('train/fm_fourier', l_fm_fourier.detach(), on_step=False, on_epoch=True)

        # Optional reconstruction stack (all zero-weighted in v0.4.0; kept for
        # the §15.6 anchored-fallback path if pure GAN+FM collapses).
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
        if 'per_band' in self.criterions:
            # v0.4.1 stage 2: per-band L1 on the predicted Haar subbands
            # (pre-IHaar) vs HaarDown(opt).  ``predicted_sub`` is populated
            # above when ``need_internals`` is True — which the factory
            # guarantees whenever 'per_band' is in self.criterions.
            pb_loss_fn = self.criterions['per_band']
            l_pb = pb_loss_fn(predicted_sub, opt)
            g_loss = g_loss + l_pb * float(loss_cfg.per_band_weight)
            self.log('train/loss_per_band', l_pb.detach(), on_step=False, on_epoch=True)
            # v0.4.1 R4: always log effective per-band weights (precisions in
            # adaptive mode, manual ratios in manual mode) so we can plot the
            # learned σ trajectory and compare R-stage runs head-to-head.
            with torch.no_grad():
                for band, value in pb_loss_fn.last_precision.items():
                    self.log(f'train/pb_w_{band}', value, on_step=False, on_epoch=True)
                if bool(getattr(self.cfg.system, 'debug', False)):
                    for band, value in pb_loss_fn.last_per_band.items():
                        self.log(f'train/loss_pb_{band}', value, on_step=False, on_epoch=True)
        if 'lpips' in self.criterions:
            l_lp = self.criterions['lpips'](fake, opt)
            g_loss = g_loss + l_lp * float(loss_cfg.lpips_weight)
            self.log('train/loss_lpips', l_lp.detach(), on_step=False, on_epoch=True)
        if 'ffl' in self.criterions:
            # v0.4.5: Focal Frequency Loss on the full-res RGB output (global
            # FFT spectrum). Complementary to per_band (local Haar subbands).
            l_ffl = self.criterions['ffl'](fake, opt)
            g_loss = g_loss + l_ffl * float(loss_cfg.ffl_weight)
            self.log('train/loss_ffl', l_ffl.detach(), on_step=False, on_epoch=True)
        if 'patchnce' in self.criterions:
            # v0.4.5: misalignment-robust contrastive supervision (CUT/PatchNCE)
            # in the generator's own encoder space. Query = encoded fake (grad
            # flows to G + sampler MLP), key = encoded real (no_grad; detached
            # inside the loss). Tolerant to sub-pixel SAR<->optical shift.
            fake_feats_nce = self.netG.encode_optical(fake)
            with torch.no_grad():
                real_feats_nce = self.netG.encode_optical(opt)
            l_nce = self.criterions['patchnce'](fake_feats_nce, real_feats_nce)
            g_loss = g_loss + l_nce * float(loss_cfg.patchnce_weight)
            self.log('train/loss_patchnce', l_nce.detach(), on_step=False, on_epoch=True)

        opt_g.zero_grad()
        if not torch.isfinite(g_loss):
            self.log('train/g_loss_nan_skip', 1.0, on_step=False, on_epoch=True, reduce_fx='sum')
        else:
            self.manual_backward(g_loss)
            g_clip_params = list(self.netG.parameters())
            if 'adaptive' in self.criterions:
                g_clip_params.extend(self.criterions['adaptive'].parameters())
            if 'per_band' in self.criterions:
                g_clip_params.extend(self.criterions['per_band'].parameters())
            if 'patchnce' in self.criterions:
                g_clip_params.extend(self.criterions['patchnce'].parameters())
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
        log_dict = {
            'train/d_loss': d_loss.detach(),
            'train/g_loss': g_loss.detach(),
            'train/d_real_mean': d_real_mean,
            'train/d_fake_mean': d_fake_mean,
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

        # Mode-collapse abort gate (val side).
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

        # Mode-collapse abort gate (train side).
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
