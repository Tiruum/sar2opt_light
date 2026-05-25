"""Factory for LLW-Former v0.4.0.

Same contract as ``src/models/llwt_v3/factory.py`` but:
  * ``build_models`` returns ``(LLWv4Generator, LLWFormerDiscriminator)`` —
    the generator is the new Haar-Stem ConvNeXt V2 + Inverse-Haar-head design
    (see ``src/models/llwt_v4/gen.py``); the discriminator is reused verbatim
    from ``llwt/dis.py`` (pixel-space, decoupled from G's output basis).
  * Loss + optimiser + scheduler builders re-exported from the v0.2.x factory
    so we inherit the bnb 8-bit Adam path, cosine-warm-restarts scheduler,
    and EMA callback wiring without duplication.

Returns ``(opt_d, opt_g)`` order — same Lightning consumer contract.
"""
from __future__ import annotations

from src.models.llwt.factory import (    # noqa: F401  (intentional re-export)
    build_criterions,
    build_ema_callback,
    build_lr_schedulers,
    build_optimizers,
)

from src.models.llwt.dis import LLWFormerDiscriminator
from src.models.llwt_v4.gen import LLWv4Generator


__all__ = [
    "build_models",
    "build_criterions",
    "build_optimizers",
    "build_lr_schedulers",
    "build_ema_callback",
]


def build_models(cfg):
    """Returns ``(netG, netD)`` — v0.4.0 generator + reused LLW discriminator."""
    netG = LLWv4Generator(cfg=cfg)
    netD = LLWFormerDiscriminator(cfg=cfg)
    return netG, netD
