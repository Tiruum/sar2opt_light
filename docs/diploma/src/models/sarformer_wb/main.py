"""SARFormer-WB Lightning module (sarformer-wb-3-simple).

Manual optimisation, ``[opt_d, opt_g]`` order.  Training step pattern (per batch):

    1. main-D update on detached fake; R1 every ``r1_main_every`` steps.
    2. G update: GAN+FM (main), L1, MS-SSIM, LAB-chroma, wavelet-detail.

The Phi-D path and speckle-consistency / uncertainty losses from earlier
revisions have been removed; the generator returns a single tensor instead of
``(out, log_var, s_spk)``.

Validation step computes PSNR/SSIM/LPIPS/FID via torchmetrics.
"""
import torch
import torch.nn as nn
import lightning.pytorch as pl
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.fid import FrechetInceptionDistance

from src.models.sarformer_wb import factory


def _stack_or_self(logits) -> torch.Tensor:
    """For tuple-of-tensors, return mean of per-tensor sums (for R1 scalarisation)."""
    if isinstance(logits, (list, tuple)):
        return torch.stack([l.sum() for l in logits]).sum()
    return logits.sum()


def _logit_mean(logits) -> torch.Tensor:
    if isinstance(logits, (list, tuple)):
        return torch.stack([l.mean() for l in logits]).mean()
    return logits.mean()


def _grads_finite(params) -> bool:
    """Check that every parameter's gradient is finite (no NaN/Inf).

    Returns True if all gradients are clean.  Skipping the optimiser step on a
    NaN-grad batch is a strict architectural safeguard: applying a NaN
    gradient (even after grad-clip, which silently propagates NaN through the
    norm scaling) poisons Adam's moment buffers and corrupts every subsequent
    update.  One bad batch must NOT be allowed to wreck the run.
    """
    for p in params:
        if p.grad is None:
            continue
        if not torch.isfinite(p.grad).all():
            return False
    return True


class SARFormerWBLightningModule(pl.LightningModule):
    def __init__(self, cfg, encoder=None):
        super().__init__()
        self.cfg = cfg
        self.automatic_optimization = False

        self.netG, self.netD = factory.build_models(cfg, encoder=encoder)
        self.criterions = nn.ModuleDict(factory.build_criterions(cfg))

        # R1 schedule.
        self.r1_gamma = float(getattr(cfg.loss, 'r1_gamma', 0.5))
        self.r1_main_every = int(getattr(cfg.loss, 'r1_main_every', 16))
        # Gradient clipping (manual-opt mode bypasses Trainer's gradient_clip_val,
        # so we must call torch.nn.utils.clip_grad_norm_ explicitly before each
        # optimiser step).
        self.grad_clip_g = float(getattr(cfg.loss, 'grad_clip_g', 1.0))
        self.grad_clip_d = float(getattr(cfg.loss, 'grad_clip_d', 5.0))

        # Validation metrics — data range 2.0 because images are in [-1, 1].
        self.psnr = PeakSignalNoiseRatio(data_range=2.0)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=2.0)
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=False)
        # ``reset_real_features=True`` so each val epoch builds a fresh real
        # distribution from the *full* val split (see hfgan-1 audit).
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

    def _r1_penalty(self, real_logits: torch.Tensor, real_input: torch.Tensor) -> torch.Tensor:
        """R1 gradient penalty (Mescheder et al. 2018).

        R1 squares the gradient norm — under bf16-mixed precision the squaring
        of small gradient components silently underflows to zero, and the
        squaring of larger ones can overflow.  Both pollute the second-order
        graph (``create_graph=True``) and accumulate into NaN over the ~70 R1
        fires per epoch at BS=8 / 256x256.  Forcing the R1 path through fp32
        avoids the bf16 numerical edge cases without sacrificing the speed
        benefit on the rest of the forward.
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

        # Defense-in-depth: a single batch with NaN/Inf input (corrupted file,
        # bad augmentation interaction, etc.) will silently poison weights
        # across the rest of the epoch.  Detect once at step entry and skip.
        if not (torch.isfinite(sar).all() and torch.isfinite(opt).all()):
            self.log('train/bad_batch_skip', 1.0, on_step=False, on_epoch=True, reduce_fx='sum')
            return

        # -------- D update on detached fake --------
        with torch.no_grad():
            fake_d = self.netG(sar)

        # On R1 batches, force the WHOLE D forward + R1 second-order grad
        # through fp32.  Otherwise the bf16 D forward caches bf16 activations
        # in the R1 backward graph, and squaring them in `_r1_penalty`
        # accumulates NaN over ~70 R1 fires per epoch.  On non-R1 batches we
        # stay in bf16-mixed for speed and VRAM.
        apply_r1_main = (batch_idx % self.r1_main_every == 0)
        if apply_r1_main:
            opt_real = opt.detach().clone().requires_grad_(True)
            with torch.amp.autocast(device_type='cuda', enabled=False):
                real_logits, _ = self.netD(sar.float(), opt_real.float())
                fake_logits, _ = self.netD(sar.float(), fake_d.float())
                d_loss = 0.5 * (
                    self.criterions['gan'](real_logits, is_real=True) +
                    self.criterions['gan'](fake_logits, is_real=False)
                )
                r1 = self._r1_penalty(real_logits, opt_real)
                d_loss = d_loss + 0.5 * self.r1_gamma * r1
            self.log('train/r1_main', r1.detach(), on_step=False, on_epoch=True)
        else:
            opt_real = opt
            real_logits, _ = self.netD(sar, opt_real)
            fake_logits, _ = self.netD(sar, fake_d)
            d_loss = 0.5 * (
                self.criterions['gan'](real_logits, is_real=True) +
                self.criterions['gan'](fake_logits, is_real=False)
            )
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

        # Main-D adversarial signal.
        fake_logits_g, fake_feats_g = self.netD(sar, fake)
        with torch.no_grad():
            _, real_feats_g = self.netD(sar, opt)
        real_feats_g = [f.detach() for f in real_feats_g]
        l_gan = self.criterions['gan'](fake_logits_g, is_real=True)
        l_fm = self.criterions['fm'](fake_feats_g, real_feats_g)
        g_loss = (
            l_gan * float(loss_cfg.gan_main_weight) +
            l_fm * float(loss_cfg.fm_main_weight)
        )
        self.log('train/gan_main', l_gan.detach(), on_step=False, on_epoch=True)
        self.log('train/fm_main', l_fm.detach(), on_step=False, on_epoch=True)

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

        opt_g.zero_grad()
        # Guard against NaN/Inf in the composite G loss BEFORE backward.
        if not torch.isfinite(g_loss):
            self.log('train/g_loss_nan_skip', 1.0, on_step=False, on_epoch=True, reduce_fx='sum')
        else:
            self.manual_backward(g_loss)
            g_clip_params = list(self.netG.parameters())
            if 'adaptive' in self.criterions:
                g_clip_params.extend(self.criterions['adaptive'].parameters())
            # Even with finite loss, backward can produce NaN grads (bf16
            # underflow squared in R1, fused-op overflow, etc.) — those would
            # propagate through grad-clip (NaN-norm scaling) into Adam's m, v
            # buffers and poison the run.  Hard-skip the step on NaN grads.
            if not _grads_finite(g_clip_params):
                self.log('train/g_grad_nan_skip', 1.0, on_step=False, on_epoch=True, reduce_fx='sum')
                opt_g.zero_grad()
            else:
                if self.grad_clip_g > 0:
                    torch.nn.utils.clip_grad_norm_(g_clip_params, self.grad_clip_g)
                opt_g.step()

        with torch.no_grad():
            d_real_mean = _logit_mean(real_logits).detach()
            d_fake_mean = _logit_mean(fake_logits).detach()
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
        with torch.no_grad():
            fake = self.netG(sar)
        self.psnr.update(fake, opt)
        self.ssim.update(fake, opt)
        self.lpips.update(fake.float(), opt.float())
        fake01 = ((fake + 1) / 2).clamp(0, 1).float()
        opt01 = ((opt + 1) / 2).clamp(0, 1).float()
        # Accumulate BOTH real and fake features on every val batch so the
        # FID covariances are estimated on the full val distribution, not on
        # a single batch's worth of samples.
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
        self.fid.reset()  # reset_real_features=True — clears both, full re-accumulation next epoch

    def on_train_epoch_end(self):
        self.sched_g.step()
        self.sched_d.step()
        if 'adaptive' in self.criterions:
            adaptive = self.criterions['adaptive']
            weights = adaptive.effective_weights()
            for name, eta, w in zip(adaptive._loss_order, adaptive.eta, weights):
                self.log(f'train/eta_{name}', eta.detach(), on_step=False, on_epoch=True)
                self.log(f'train/w_{name}', w.detach(), on_step=False, on_epoch=True)
