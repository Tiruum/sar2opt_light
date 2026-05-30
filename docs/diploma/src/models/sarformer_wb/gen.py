"""SARFormer-WB generator (sarformer-wb-3-simple): Swin V2 encoder + GRN decoder.

Forward returns ``out`` only — a ``(B, 3, H, W)`` optical tensor in ``[-1, 1]``.

Composition (top to bottom):
  1. SARPhysicsFrontEnd          (raw + log + Sobel -> 3 ch)
  2. 1x1 adapter                 (3 ch -> 3 ch, for backbone stem)
  3. Swin V2 encoder             (HF AutoBackbone, 4 stages)
  4. Identity bottleneck         (no-op; the wavelet bottleneck was removed in
                                   sarformer-wb-3-simple because its residual
                                   gate stayed dormant during training)
  5. PixelShuffle/GRN decoder    (5 stages: 8->16->32->64->128->256)
  6. RGB head (tanh)             -> (3, 256, 256) in [-1, 1]

Removed in sarformer-wb-3-simple vs prior revisions:
  * ``WaveletBottleneck`` (and its sub-modules ``MiniSwinV2Block``,
    ``SpeckleGatedConvStack``, ``HaarDWT2d``) — residual gate never opened.
  * ``SARPyramid`` + ``SARCrossAttnSkip`` (32/64 windowed cross-attn) — produced
    a checkerboard / window-grid artifact in outputs and gates stayed at 0.
  * Uncertainty (``head_unc``) head and its return value ``log_var``.
  * ``speckle_head`` inside ``SARPhysicsFrontEnd`` and its return value
    ``s_spk`` — the Phi-D physics path was retired.
"""
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. SAR Physics Front-End
# ---------------------------------------------------------------------------


class SARPhysicsFrontEnd(nn.Module):
    """Builds a 3-channel SAR tensor:

        ch0: raw SAR  (already in [-1, 1] after LogNormSAR)
        ch1: log1p(|SAR|) — collapses dynamic range
        ch2: reflect-padded Sobel gradient magnitude

    The earlier (sarformer-wb-2-rebal) revision also predicted a spatial speckle
    log-variance ``s_spk`` from a small conv head, used by the Phi-D and the
    speckle-consistency loss.  Both downstream consumers have been removed for
    sarformer-wb-3-simple so the head is dropped as well — the front-end is now
    purely deterministic from its input.
    """
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1., 0., 1.],
                                [-2., 0., 2.],
                                [-1., 0., 1.]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = sobel_x.transpose(-1, -2).contiguous()
        self.register_buffer('sx', sobel_x)
        self.register_buffer('sy', sobel_y)

    def forward(self, sar: torch.Tensor) -> torch.Tensor:
        """Returns (B, 3, H, W)."""
        eps = 1e-6
        log_amp = torch.log1p(sar.abs())
        with torch.amp.autocast(device_type='cuda', enabled=False):
            xf = sar.float()
            xf_pad = F.pad(xf, (1, 1, 1, 1), mode='reflect')
            gx = F.conv2d(xf_pad, self.sx)
            gy = F.conv2d(xf_pad, self.sy)
            grad = torch.sqrt(gx * gx + gy * gy + eps).to(sar.dtype)
        return torch.cat([sar, log_amp, grad], dim=1)


# ---------------------------------------------------------------------------
# 2. Decoder block — ConvNeXt V2 GRN block + PixelShuffle upsample
# ---------------------------------------------------------------------------


class GRN(nn.Module):
    """Global Response Normalization (Woo et al. 2023, arXiv:2301.00808).

    Operates in channel-last (B, H, W, C) layout. Identity at init because
    gamma and beta are zero-initialised.

    ``eps`` is set to ``1e-3`` so the normalisation is stable under bf16-mixed
    precision: the original ``1e-6`` falls below bf16's epsilon (~7.8e-3) and
    can blow up the ``gx / (gx.mean() + eps)`` term when ``gx.mean()`` is small.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)         # (B, 1, 1, C)
        nx = gx / (gx.mean(dim=-1, keepdim=True) + 1e-3)
        return self.gamma * (x * nx) + self.beta + x


class DropPath(nn.Module):
    """Stochastic depth (Huang et al. 2016). Identity in eval."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x * mask / keep


class ConvNeXtV2GRNBlock(nn.Module):
    """ConvNeXt V2 block (DWConv7x7 -> LN -> Linear -> GELU -> GRN -> Linear -> DropPath + residual).

    Zero-init on ``pwconv2`` makes the whole residual branch start at zero, so
    the block is a *true* identity at init (the GRN gamma/beta zero-init alone
    is not enough — pwconv2's random init would otherwise emit O(1) noise into
    the residual sum, perturbing all downstream losses at epoch 0).  Combined
    with the zero-init RGB head, this gives the generator a defined "no-op"
    initialisation across the entire decoder.
    """
    def __init__(self, dim: int, drop_path: float = 0.0, expand: int = 4):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, expand * dim)
        self.act = nn.GELU()
        self.grn = GRN(expand * dim)
        self.pwconv2 = nn.Linear(expand * dim, dim)
        nn.init.zeros_(self.pwconv2.weight)
        nn.init.zeros_(self.pwconv2.bias)
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)                                 # B,C,H,W -> B,H,W,C
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)                                 # B,H,W,C -> B,C,H,W
        return identity + self.drop_path(x)


def _icnr_init(conv: nn.Conv2d, scale: int = 2) -> None:
    """Sub-pixel ICNR init for PixelShuffle (Aitken et al. 2017)."""
    co, ci, kh, kw = conv.weight.shape
    sub = co // (scale * scale)
    w = torch.empty(sub, ci, kh, kw, device=conv.weight.device, dtype=conv.weight.dtype)
    nn.init.kaiming_normal_(w, mode='fan_in', nonlinearity='linear')
    w = w.repeat_interleave(scale * scale, dim=0)
    conv.weight.data.copy_(w)


class DecoderStage(nn.Module):
    """PixelShuffle 2x upsample (ICNR init) + optional skip concat + ConvNeXtV2GRN block."""
    def __init__(self, up_ch: int, skip_ch: int, out_ch: int, drop_path: float = 0.0):
        super().__init__()
        self.up_conv = nn.Conv2d(up_ch, up_ch * 4, kernel_size=1, bias=False)
        _icnr_init(self.up_conv, scale=2)
        self.pixel_shuffle = nn.PixelShuffle(2)
        in_ch = up_ch + skip_ch
        # Project to out_ch first (so subsequent GRN block runs at out_ch).
        self.project = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.block = ConvNeXtV2GRNBlock(out_ch, drop_path=drop_path, expand=4)
        # Residual shortcut for stability (concat -> 1x1 -> out_ch).
        self.shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.pixel_shuffle(self.up_conv(x))
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        y = self.project(x)
        y = self.block(y)
        return y + self.shortcut(x)


# ---------------------------------------------------------------------------
# 3. Generator
# ---------------------------------------------------------------------------


class SARFormerWBGenerator(nn.Module):
    """Hybrid SAR->Optical generator (sarformer-wb-3-simple).

    Architecture summary (input/output sizes assume 256x256):

      sar (1ch)
        -> SARPhysicsFrontEnd      -> (3ch)
        -> 1x1 adapter             -> (3ch)
        -> Swin V2-Tiny encoder    -> s0=(96,64,64), s1=(192,32,32), s2=(384,16,16), s3=(768,8,8)
        -> Identity bottleneck     -> b3' = (768, 8, 8)  (no-op)
        -> DecoderStage 8 ->16     -> (256, 16, 16), skip = s2
        -> DecoderStage 16->32     -> (128, 32, 32), skip = s1
        -> DecoderStage 32->64     -> ( 64, 64, 64), skip = s0
        -> DecoderStage 64->128    -> ( 32,128,128)
        -> DecoderStage 128->256   -> ( 16,256,256)
        -> ReflectionPad+Conv7x7   -> (3, 256, 256) logits -> tanh

    ``encoder`` argument lets callers (and tests) inject a pre-built backbone
    so HF downloads are avoided in CI.
    """
    def __init__(self, cfg, encoder: Optional[nn.Module] = None):
        super().__init__()
        gen_cfg = cfg.model.gen
        self.image_size = int(cfg.data.image_size)

        # Physics front-end and adapter.  Front-end now outputs 3 channels
        # (raw + log + Sobel); the adapter is left as a Conv2d(3,3,1) so the
        # backbone stem still sees a learnable colour-space remap.
        self.physics_front_end = SARPhysicsFrontEnd()
        self.adapter = nn.Conv2d(3, 3, kernel_size=1, bias=True)

        # Swin V2 encoder.
        if encoder is not None:
            self.encoder = encoder
        else:
            from transformers import AutoBackbone
            self.encoder = AutoBackbone.from_pretrained(
                gen_cfg.backbone,
                out_indices=tuple(gen_cfg.out_indices),
            )

        # Backbone channel layout: read from the actual backbone when possible
        # so Tiny/Small/Base variants all work.  The decoder dim choices below
        # were tuned for Swin V2-Tiny (96, 192, 384, 768); a non-Tiny variant
        # still works but the decoder channel budget may need re-tuning.
        ch_attr = getattr(self.encoder, 'channels', None)
        if ch_attr is None:
            import warnings
            warnings.warn(
                "Encoder has no `.channels` attribute; defaulting to Swin V2-Tiny "
                "layout (96, 192, 384, 768). If you swapped the backbone, set "
                "encoder.channels or pass a custom encoder.", stacklevel=2,
            )
            ch = (96, 192, 384, 768)
        else:
            ch = tuple(int(c) for c in ch_attr)
        assert len(ch) == 4, (
            f"SARFormerWB expects exactly 4 encoder stages; got {len(ch)} "
            f"from encoder.channels={ch_attr}. Use a 4-stage hierarchical "
            f"backbone (Swin V2, ConvNeXt V2, etc.)."
        )
        if ch != (96, 192, 384, 768):
            import warnings
            warnings.warn(
                f"Encoder channel layout {ch} differs from the Swin V2-Tiny "
                f"reference (96, 192, 384, 768) the decoder was tuned for. "
                f"The model will run but channel-budget / VRAM may shift.",
                stacklevel=2,
            )
        self.enc_channels = ch
        c0, c1, c2, c3 = ch

        # Bottleneck removed in sarformer-wb-3-simple.  Keep an Identity so the
        # forward chain stays intact and any saved-state lookups for the
        # ``bottleneck`` attribute do not KeyError (the WaveletBottleneck's
        # parameters won't be in the checkpoint at all going forward, but
        # ``self.bottleneck`` still resolves).
        self.bottleneck = nn.Identity()

        # Decoder. Channels chosen so the heaviest stages live at the lowest
        # resolutions and total decoder params stay near the budget in the plan.
        d_cfg = gen_cfg.decoder
        dp = float(d_cfg.get('droppath', 0.1))
        self.dec_8_16 = DecoderStage(up_ch=c3,  skip_ch=c2, out_ch=256, drop_path=dp)
        self.dec_16_32 = DecoderStage(up_ch=256, skip_ch=c1, out_ch=128, drop_path=dp)
        self.dec_32_64 = DecoderStage(up_ch=128, skip_ch=c0, out_ch=64,  drop_path=dp)
        self.dec_64_128 = DecoderStage(up_ch=64,  skip_ch=0,  out_ch=32,  drop_path=dp)
        self.dec_128_256 = DecoderStage(up_ch=32,  skip_ch=0,  out_ch=16,  drop_path=dp)

        # Output head.
        self.head_rgb = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(16, 3, kernel_size=7),
        )
        # Zero-init the RGB head's last conv: at init, ``logits = 0`` →
        # ``tanh(0) = 0`` → generator outputs mid-grey ((-1+1)/2-equivalent).
        # This gives every reconstruction loss a *defined* neutral start point
        # instead of random-vs-target, so loss magnitudes at epoch 0 reflect
        # the actual loss scale, not init noise.  Encoder/decoder features are
        # still randomly initialised — only the output projection is zeroed.
        nn.init.zeros_(self.head_rgb[-1].weight)
        nn.init.zeros_(self.head_rgb[-1].bias)

    def _encoder_forward(self, x3: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Return (s0, s1, s2, s3) — robust to different HF backbone APIs."""
        out = self.encoder(pixel_values=x3)
        feats = getattr(out, 'feature_maps', None) or out
        return tuple(feats)

    def forward(self, sar: torch.Tensor) -> torch.Tensor:
        """Returns a single tensor ``out`` of shape ``(B, 3, H, W)`` in ``[-1, 1]``.

        The earlier revisions also returned ``log_var`` (uncertainty L1 head)
        and ``s_spk`` (physics front-end speckle log-variance).  Both have been
        removed in sarformer-wb-3-simple.
        """
        three = self.physics_front_end(sar)                       # (B, 3, H, W)
        x3 = self.adapter(three)                                  # (B, 3, H, W)
        s0, s1, s2, s3 = self._encoder_forward(x3)
        b3 = self.bottleneck(s3)                                  # Identity

        d = self.dec_8_16(b3, skip=s2)                            # (B, 256, 16, 16)
        d = self.dec_16_32(d, skip=s1)                            # (B, 128, 32, 32)
        d = self.dec_32_64(d, skip=s0)                            # (B,  64, 64, 64)
        d = self.dec_64_128(d)                                    # (B,  32, 128, 128)
        d = self.dec_128_256(d)                                   # (B,  16, 256, 256)

        logits = self.head_rgb(d)
        return torch.tanh(logits)
