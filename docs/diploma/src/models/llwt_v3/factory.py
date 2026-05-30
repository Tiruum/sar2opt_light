"""Factory for LLW-Former v0.3.0.

Same contract as ``src/models/llwt/factory.py`` but:
  * ``build_models`` returns ``(LLWv3Generator, LLWFormerDiscriminator)`` — the
    generator is the new Haar-Stem ConvNeXt V2 design; the discriminator is
    reused verbatim from ``llwt/dis.py`` (Section F of the dis review:
    pixel-space, decoupled from the generator's coefficient format).
  * Loss + optimiser + scheduler builders are imported from the v0.2.x factory
    so we inherit the bnb 8-bit Adam path, the cosine-warm-restarts scheduler,
    and the EMA callback wiring without duplication.

Returns ``(opt_d, opt_g)`` order — same Lightning consumer contract.
"""
from __future__ import annotations

# Re-export everything from the v0.2.x factory so callers can swap the import
# path in one line.  We only override ``build_models``.
from src.models.llwt.factory import (    # noqa: F401  (intentional re-export)
    build_criterions,
    build_ema_callback,
    build_lr_schedulers,
    build_optimizers,
)

from src.models.llwt.dis import LLWFormerDiscriminator
from src.models.llwt_v3.gen import LLWv3Generator


__all__ = [
    "build_models",
    "build_criterions",
    "build_optimizers",
    "build_lr_schedulers",
    "build_ema_callback",
]


def build_models(cfg):
    """Returns ``(netG, netD)`` — v0.3.0 generator + reused LLW discriminator."""
    netG = LLWv3Generator(cfg=cfg)
    netD = LLWFormerDiscriminator(cfg=cfg)
    return netG, netD
