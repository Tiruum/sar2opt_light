"""Factory for SARFormer-WB (sarformer-wb-3-simple): model + criterions +
optimisers + schedulers + EMA.

Compared with sarformer-wb-2-rebal, the Phi-D physics discriminator and the
speckle-consistency loss are gone, so:

  * ``build_models`` returns ``(netG, netD)`` instead of three modules.
  * ``build_optimizers`` returns ``(opt_d, opt_g)`` — Phi-D opt dropped.
  * ``build_lr_schedulers`` returns ``(sched_g, sched_d)`` — Phi-D sched dropped.
  * ``build_criterions`` constructs plain L1 in place of uncertainty L1 and
    drops the speckle-consistency loss.
"""
import torch.optim as optim
from torch.optim.lr_scheduler import (
    LinearLR, SequentialLR, ConstantLR, CosineAnnealingWarmRestarts,
)

from src.models.sarformer_wb.gen import SARFormerWBGenerator
from src.models.sarformer_wb.dis import MSPatchGANDis
from src.models.sarformer_wb.losses import (
    GANLoss, FeatureMatchingLoss, MSSSIMLoss,
    LABChromaL1Loss, WaveletDetailL1Loss,
    PlainL1Loss, AdaptiveLoss,
)


def build_models(cfg, encoder=None):
    """Returns ``(netG, netD)``."""
    netG = SARFormerWBGenerator(cfg, encoder=encoder)
    netD = MSPatchGANDis(
        in_ch=int(cfg.model.dis.main.in_channels),
        ndf=int(cfg.model.dis.main.ndf),
    )
    return netG, netD


def build_criterions(cfg) -> dict:
    """Instantiate active losses based on weights in cfg.

    ``l1`` is the plain pixel-wise L1 loss (replaces the previous uncertainty-
    weighted L1 in sarformer-wb-2-rebal).
    """
    loss_cfg = cfg.loss
    criterions = {
        'gan': GANLoss(),
        'fm':  FeatureMatchingLoss(),
    }
    if loss_cfg.get('l1_weight', 0.0) > 0:
        criterions['l1'] = PlainL1Loss()
    if loss_cfg.get('msssim_weight', 0.0) > 0:
        criterions['msssim'] = MSSSIMLoss()
    if loss_cfg.get('lab_chroma_weight', 0.0) > 0:
        criterions['lab_chroma'] = LABChromaL1Loss()
    if loss_cfg.get('wavelet_weight', 0.0) > 0:
        criterions['wavelet'] = WaveletDetailL1Loss()
    if loss_cfg.get('adaptive_balance', False):
        # Honoured but not wired into main.py by default; surfaced for ablation A3.
        # Wrap any reconstruction-only losses (l1 + msssim + lab_chroma + wavelet)
        # that are active.
        recon_keys = [k for k in ('l1', 'msssim', 'lab_chroma', 'wavelet') if k in criterions]
        if len(recon_keys) >= 2:
            criterions['adaptive'] = AdaptiveLoss(
                n_losses=len(recon_keys),
                eta_max=float(loss_cfg.get('adaptive_eta_max', 5.0)),
            )
            criterions['adaptive']._loss_order = recon_keys
    return criterions


def _encoder_param_ids(netG) -> set:
    enc_params = list(netG.encoder.parameters())
    return {id(p) for p in enc_params}


def build_optimizers(cfg, netG, netD, criterions: dict = None):
    """AdamW for G (encoder at lr*scale, rest at lr); Adam for D.

    Generator param groups:
      group 0: encoder params at ``lr_g * encoder_lr_scale``
      group 1: every other G param at ``lr_g`` — physics front-end, adapter,
               decoder, head, plus any AdaptiveLoss eta params.

    A post-build assertion verifies every G parameter ends up in some group.

    Returns ``(opt_d, opt_g)`` — order matches what ``configure_optimizers``
    consumes (D first, then G).
    """
    enc_ids = _encoder_param_ids(netG)
    enc_params = [p for p in netG.parameters() if id(p) in enc_ids]
    fresh_params = [p for p in netG.parameters() if id(p) not in enc_ids]

    # Loss modules that own trainable parameters live in the G optimizer too.
    if criterions is not None and 'adaptive' in criterions:
        fresh_params = fresh_params + list(criterions['adaptive'].parameters())

    encoder_scale = float(cfg.model.gen.get('encoder_lr_scale', 0.1))
    opt_g = optim.AdamW(
        [
            {'params': enc_params,   'lr': float(cfg.optimizer.lr_g) * encoder_scale},
            {'params': fresh_params, 'lr': float(cfg.optimizer.lr_g)},
        ],
        betas=(float(cfg.optimizer.beta1), float(cfg.optimizer.beta2)),
        weight_decay=float(cfg.optimizer.weight_decay_g),
    )

    # Param-coverage assertion: every G param + every criterion-owned param
    # (adaptive eta) must be in some opt_g group.
    expected_ids = {id(p) for p in netG.parameters()}
    if criterions is not None and 'adaptive' in criterions:
        expected_ids.update(id(p) for p in criterions['adaptive'].parameters())
    opt_param_ids = {id(p) for pg in opt_g.param_groups for p in pg['params']}
    missing = expected_ids - opt_param_ids
    assert not missing, (
        f"build_optimizers: {len(missing)} G/criterion parameter(s) not in any "
        f"opt_g group (missing ids: {sorted(missing)[:5]}...)"
    )

    opt_d = optim.Adam(
        netD.parameters(),
        lr=float(cfg.optimizer.lr_d),
        betas=(float(cfg.optimizer.beta1), float(cfg.optimizer.beta2)),
    )

    return opt_d, opt_g


def build_lr_schedulers(cfg, opt_d, opt_g):
    """LR schedulers per ``cfg.scheduler.type``.

    Supported types:

      * ``"cosine_warm_restarts"`` (default) — ``CosineAnnealingWarmRestarts``
        with ``T_0`` initial period and ``T_mult`` period multiplier per restart.
      * ``"linear_decay"`` — constant LR for ``max_epochs -
        linear_decay_epochs`` then linear decay to ``eta_min``.

    Returns ``(sched_d, sched_g)``.
    """
    sched_type = str(cfg.scheduler.get('type', 'cosine_warm_restarts')).lower()
    eta_min = float(cfg.scheduler.eta_min)

    def make_sched(opt, base_lr):
        if sched_type == 'cosine_warm_restarts':
            return CosineAnnealingWarmRestarts(
                opt,
                T_0=int(cfg.scheduler.get('t_0', 20)),
                T_mult=int(cfg.scheduler.get('t_mult', 2)),
                eta_min=eta_min,
            )
        if sched_type == 'linear_decay':
            decay = int(cfg.scheduler.linear_decay_epochs)
            max_epochs = int(cfg.system.max_epochs)
            warmup = max(max_epochs - decay, 0)
            end_factor = eta_min / max(base_lr, 1e-10)
            linear = LinearLR(opt, start_factor=1.0, end_factor=end_factor, total_iters=decay)
            if warmup == 0:
                return linear
            return SequentialLR(
                opt,
                schedulers=[ConstantLR(opt, factor=1.0, total_iters=warmup), linear],
                milestones=[warmup],
            )
        raise ValueError(f"Unknown scheduler.type='{sched_type}' (expected 'cosine_warm_restarts' or 'linear_decay')")

    sched_g = make_sched(opt_g, float(cfg.optimizer.lr_g))
    sched_d = make_sched(opt_d, float(cfg.optimizer.lr_d))
    return sched_d, sched_g


def build_ema_callback(cfg):
    """Returns an EMAWeightAveraging callback when cfg.ema.use_ema, else None."""
    ema_cfg = getattr(cfg, 'ema', None)
    if ema_cfg is None or not getattr(ema_cfg, 'use_ema', False):
        return None
    from src.utils.callbacks import EMAWeightAveraging
    return EMAWeightAveraging(
        decay=float(ema_cfg.decay),
        update_starting_at_epoch=int(ema_cfg.start_epoch),
    )
