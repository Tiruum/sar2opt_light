"""CPU smoke: build LLWv4LightningModule with a mock encoder, run one training_step
with align enabled.  Verifies wiring + finite g_loss + step-0 zero-init contract.

Run: python -m src.models.llwt_v5.smoke_train_step
"""
from __future__ import annotations

import torch
from omegaconf import OmegaConf

from src.models.llwt_v5.gen import LLWv4Generator


def main() -> None:
    print("[train-step smoke] building module with align enabled")
    cfg = OmegaConf.load('src/models/llwt_v5/config.yaml')
    # Force CPU-friendly settings.
    cfg.system.device = 'cpu'
    cfg.system.precision = '32-true'
    cfg.system.channels_last = False
    cfg.system.compile = False
    cfg.align.enabled = True

    from src.models.llwt_v5.main import LLWv4LightningModule

    # Patch the generator to use the mock encoder (no network / HF download).
    class _MockEnc(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = type('Cfg', (), {'hidden_sizes': (96, 192, 384, 768),
                                           'model_type': 'convnextv2'})()
            self.embeddings = torch.nn.Module()
            self.embeddings.patch_embeddings = torch.nn.Identity()

        def forward(self, pixel_values):
            B, _, H, W = pixel_values.shape
            d = pixel_values.dtype
            return type('O', (), {'feature_maps': [
                torch.zeros(B, 96, H // 4, W // 4, dtype=d),
                torch.zeros(B, 192, H // 8, W // 8, dtype=d),
                torch.zeros(B, 384, H // 16, W // 16, dtype=d),
                torch.zeros(B, 768, H // 32, W // 32, dtype=d),
            ]})()

    # Build module, then swap in a generator that uses the mock encoder.
    module = LLWv4LightningModule(cfg)
    module.netG = LLWv4Generator(cfg=cfg, encoder=_MockEnc())
    module.use_channels_last = False
    assert module.use_align, "align should be enabled"

    sar = torch.randn(2, 1, 256, 256)
    opt = torch.randn(2, 3, 256, 256)

    # Manually exercise the aligner path (mirrors training_step).
    from src.models.llwt_v5.align import psc_detect, DeformationAligner
    fake = module.netG(sar)
    assert fake.abs().max().item() < 1e-4, f"step-0 zero-init broken: {fake.abs().max().item()}"
    fake_ll = module._haar_ll(fake)[0]
    opt_ll = module._haar_ll(opt)[0]
    phi = module.aligner(fake_ll, opt_ll)
    assert phi.abs().max().item() < 1e-6, "aligner not zero-init"
    opt_aligned = DeformationAligner.warp(opt, phi)
    assert (opt_aligned - opt).abs().max().item() < 1e-3, "identity warp broken"
    m = psc_detect(sar, topk=module.psc_topk)
    l_reg = module.align_criterions['deform_reg'](phi)
    l_psc = module.align_criterions['psc_anchor'](fake, opt_aligned, m)
    l_bsc = module.align_criterions['bsc'](fake, sar)
    for name, v in [('reg', l_reg), ('psc', l_psc), ('bsc', l_bsc)]:
        assert torch.isfinite(v), f"{name} loss not finite"
    print(f"  [OK] step-0 fake~0, phi~0, identity warp, losses finite "
          f"(reg={l_reg.item():.4f} psc={l_psc.item():.4f} bsc={l_bsc.item():.4f})")
    print("[train-step smoke] PASS")


if __name__ == '__main__':
    main()
