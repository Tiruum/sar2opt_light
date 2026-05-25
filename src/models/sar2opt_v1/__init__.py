"""SAR2OPT-V1: Swin V2-Base + Physics-Guided Cross-Attention decoder.

Greenfield architecture (May 2026) replacing the failed llwt_v3 stack.

Generator: SARAdapter (raw / log|SAR| / Sobel-grad) -> ConvNeXt V2 / SwinV2
pretrained encoder -> U-Net PixelShuffle decoder with one Physics-Guided
Cross-Attention (PGCA) block per decoder scale.

Discriminator: re-export of HFGANDiscriminator (3-scale 70/46/22 RF,
spectral-norm, asymmetric SAR-only InstanceNorm) -- proven in hfgan-18.

Loss recipe (no global pixel-L1): GAN + FM + LAB-Chroma + Wavelet-Detail +
MS-SSIM + PatchNCE (CUT) + FFL (Jiang 2021).
"""
