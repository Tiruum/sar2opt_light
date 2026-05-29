"""Loss suite for SARFormer-WB (sarformer-wb-3-simple).

Active losses:
  * ``GANLoss`` — LSGAN with label smoothing.
  * ``FeatureMatchingLoss`` — element-weighted L1 across D feature layers.
  * ``MSSSIMLoss`` — 1 - MS-SSIM on [-1, 1] tensors.
  * ``LABChromaL1Loss`` — L1 on (a*, b*) in CIE LAB.
  * ``WaveletDetailL1Loss`` — scalar L1 on Haar detail subbands (LH/HL/HH) of
    the full-res output.  Drops LL.  Use for any generator without an
    IHaar output head.
  * ``PerBandWaveletL1Loss`` — per-band L1 (LL+LH+HL+HH) on the pre-IHaar
    predicted subband tensor of v0.4.x generators.  Internally re-derives the
    target subbands from ``HaarDown(opt)`` and supports per-band ratio
    weighting.  Mutually exclusive with ``WaveletDetailL1Loss`` at the factory
    level (overlapping detail-band gradient).
  * ``PlainL1Loss`` — thin wrapper around ``F.l1_loss`` (replaces the previous
    ``UncertaintyL1Loss``, removed because its log-variance head was dominating
    the recon stack and starving the adversarial signal).
  * ``AdaptiveLoss`` — homoscedastic uncertainty weighting (opt-in, off by default).

Removed in sarformer-wb-3-simple:
  * ``UncertaintyL1Loss`` — replaced by plain L1.
  * ``SpeckleConsistencyLoss`` — Phi-D path retired, no consumer.

The package is intentionally self-contained: do not import from huggingface_gan.
"""
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveLoss(nn.Module):
    """Homoscedastic uncertainty weighting (Kendall et al., 2018).

    Balances N loss components via learnable log-variances `eta`:

        L_total = sum_i ( L_i * exp(-eta_i) + eta_i )

    Symmetrically clamped to ``[-eta_max, +eta_max]`` (drift to -inf would let
    one recon term dominate adversarial signal; +inf would zero a component).
    Default off; opt in via ``cfg.loss.adaptive_balance``.
    """
    def __init__(self, n_losses: int, eta_max: float = 5.0):
        super().__init__()
        self.eta = nn.Parameter(torch.zeros(n_losses))
        self.eta_max = float(eta_max)

    def forward(self, losses: List[torch.Tensor]) -> torch.Tensor:
        assert len(losses) == self.eta.numel(), (
            f"AdaptiveLoss got {len(losses)} losses, expected {self.eta.numel()}"
        )
        eta_clamped = self.eta.clamp(min=-self.eta_max, max=self.eta_max)
        return sum(L * torch.exp(-e) + e for L, e in zip(losses, eta_clamped))

    @torch.no_grad()
    def effective_weights(self) -> torch.Tensor:
        return torch.exp(-self.eta.clamp(min=-self.eta_max, max=self.eta_max))


class GANLoss(nn.Module):
    """LSGAN-with-smoothing or hinge.  Accepts a tensor or a tuple of tensors.

    ``gan_type='lsgan'``  (default): MSE against smoothed labels.
    ``gan_type='hinge'``  : D = relu(1 - D(real)) on real / relu(1 + D(fake)) on
                            fake; G = -D(fake).mean().  Pass ``for_d=False`` from
                            the G update to switch to the G branch — LSGAN
                            ignores the flag (its real/fake math is symmetric).
    """
    def __init__(self, real_smooth: float = 0.9, fake_smooth: float = 0.0,
                 gan_type: str = 'lsgan'):
        super().__init__()
        gan_type = str(gan_type).lower()
        if gan_type not in ('lsgan', 'hinge'):
            raise ValueError(f"GANLoss: gan_type must be 'lsgan' or 'hinge', got '{gan_type}'")
        self.gan_type = gan_type
        self.criterion = nn.MSELoss()
        self.real_smooth = float(real_smooth)
        self.fake_smooth = float(fake_smooth)

    def _lsgan_loss(self, logit: torch.Tensor, is_real: bool) -> torch.Tensor:
        val = self.real_smooth if is_real else self.fake_smooth
        return self.criterion(logit, torch.full_like(logit, val))

    def _hinge_loss(self, logit: torch.Tensor, is_real: bool, for_d: bool) -> torch.Tensor:
        if not for_d:
            return -logit.mean()
        if is_real:
            return F.relu(1.0 - logit).mean()
        return F.relu(1.0 + logit).mean()

    def _loss_one(self, logit: torch.Tensor, is_real: bool, for_d: bool) -> torch.Tensor:
        if self.gan_type == 'hinge':
            return self._hinge_loss(logit, is_real, for_d)
        return self._lsgan_loss(logit, is_real)

    def forward(self, logits, is_real: bool, for_d: bool = True) -> torch.Tensor:
        if isinstance(logits, (list, tuple)):
            return sum(self._loss_one(l, is_real, for_d) for l in logits) / len(logits)
        return self._loss_one(logits, is_real, for_d)


class FeatureMatchingLoss(nn.Module):
    """Element-weighted L1 across discriminator feature layers.

    ``sum(|fake_i - real_i|.sum()) / sum(numel_i)`` so each activation
    contributes equally — a 32×32 layer is not weighted the same as a 128×128.
    Real features are detached.
    """
    def forward(self, fake_feats: list, real_feats: list) -> torch.Tensor:
        total = sum(
            F.l1_loss(f, r.detach(), reduction='sum')
            for f, r in zip(fake_feats, real_feats)
        )
        n_elem = sum(f.numel() for f in fake_feats)
        return total / max(n_elem, 1)


class MSSSIMLoss(nn.Module):
    """1 − MS-SSIM on float32 tensors in [-1, 1] (data_range=2.0)."""
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        from torchmetrics.functional.image import (
            multiscale_structural_similarity_index_measure as ms_ssim,
        )
        score = ms_ssim(pred.float(), target.float(), data_range=2.0)
        return 1.0 - score


class LABChromaL1Loss(nn.Module):
    """L1 in CIE LAB on (a*, b*) only; drops L* (covered by pixel L1 / MS-SSIM).

    Architectural note: a*, b* live in roughly ``[-128, 127]`` per pixel, so an
    un-normalised L1 between two random RGB images can sit around ~30-50 — two
    orders of magnitude larger than pixel-domain losses (L1, MS-SSIM, wavelet)
    that operate in ``[-1, 1]`` and produce ~0-2.  Without rescaling, an
    intended ``lab_chroma_weight=5`` would actually contribute ~150-250 to the
    composite G loss — 30x what the weight number suggests — and dominate the
    gradient direction regardless of the other terms.

    We divide the L1 by ``ab_norm`` (default 50.0 = mid-range a*/b* spread for
    natural-image distributions) so the output magnitude lands in roughly
    ``[0, 1]`` and the configured weight expresses the *intended* priority
    relative to other reconstruction losses.  Tunable in case the dataset's
    chroma distribution differs (e.g. arid SAR-paired imagery may have a
    smaller chroma spread).
    """
    _RGB2XYZ = torch.tensor([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ], dtype=torch.float32)
    _D65_WHITE = torch.tensor([0.95047, 1.0, 1.08883], dtype=torch.float32)

    def __init__(self, ab_norm: float = 50.0):
        super().__init__()
        self.ab_norm = float(ab_norm)
        self.register_buffer('rgb2xyz', self._RGB2XYZ.clone())
        self.register_buffer('d65_white', self._D65_WHITE.clone())
        try:
            from kornia.color import rgb_to_lab  # noqa: F401
            self._has_kornia = True
        except Exception:
            self._has_kornia = False

    @staticmethod
    def _srgb_to_linear(rgb: torch.Tensor) -> torch.Tensor:
        threshold = 0.04045
        linear_lo = rgb / 12.92
        linear_hi = ((rgb + 0.055) / 1.055).clamp_min(0.0) ** 2.4
        return torch.where(rgb <= threshold, linear_lo, linear_hi)

    @staticmethod
    def _f_lab(t: torch.Tensor) -> torch.Tensor:
        delta = 6.0 / 29.0
        delta3 = delta ** 3
        f_hi = t.clamp_min(0.0) ** (1.0 / 3.0)
        f_lo = t / (3.0 * delta * delta) + 4.0 / 29.0
        return torch.where(t > delta3, f_hi, f_lo)

    def _manual_rgb_to_lab(self, rgb01: torch.Tensor) -> torch.Tensor:
        linear = self._srgb_to_linear(rgb01)
        B, _, H, W = linear.shape
        flat = linear.reshape(B, 3, H * W)
        xyz = torch.matmul(self.rgb2xyz, flat).reshape(B, 3, H, W)
        white = self.d65_white.view(1, 3, 1, 1)
        f_xyz = self._f_lab(xyz / white)
        fx, fy, fz = f_xyz[:, 0:1], f_xyz[:, 1:2], f_xyz[:, 2:3]
        L = 116.0 * fy - 16.0
        a = 500.0 * (fx - fy)
        b = 200.0 * (fy - fz)
        return torch.cat([L, a, b], dim=1)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred01 = ((pred.float() + 1.0) / 2.0).clamp(0.0, 1.0)
        target01 = ((target.float() + 1.0) / 2.0).clamp(0.0, 1.0)
        if self._has_kornia:
            from kornia.color import rgb_to_lab
            pred_lab = rgb_to_lab(pred01)
            target_lab = rgb_to_lab(target01)
        else:
            pred_lab = self._manual_rgb_to_lab(pred01)
            target_lab = self._manual_rgb_to_lab(target01)
        # Normalise L1 by the typical a*/b* spread so the loss output is
        # commensurate with pixel-domain losses (see class docstring).
        return F.l1_loss(pred_lab[:, 1:], target_lab[:, 1:]) / self.ab_norm


class WaveletDetailL1Loss(nn.Module):
    """L1 on Haar DWT detail subbands (LH/HL/HH) of pred vs target.

    Orthonormal Haar (Parseval-preserving). Per-channel.
    """
    _HAAR = torch.tensor([
        [ 1.0,  1.0,  1.0,  1.0],   # LL
        [-1.0,  1.0, -1.0,  1.0],   # LH
        [-1.0, -1.0,  1.0,  1.0],   # HL
        [ 1.0, -1.0, -1.0,  1.0],   # HH
    ], dtype=torch.float32) * 0.5

    def __init__(self):
        super().__init__()
        self.register_buffer('haar_weights', self._HAAR.clone())

    def _dwt(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_blocks = F.pixel_unshuffle(x, 2).view(B, C, 4, H // 2, W // 2)
        return torch.einsum('bcihw, oi -> bcohw', x_blocks, self.haar_weights)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_w = self._dwt(pred.float())[:, :, 1:]
        target_w = self._dwt(target.float())[:, :, 1:]
        return F.l1_loss(pred_w, target_w)


class PerBandWaveletL1Loss(nn.Module):
    """Per-band L1 on predicted Haar subbands (pre-IHaar) vs HaarDown(target).

    Designed for LLW-Former v0.4.x (``LLWv4Generator`` with ``InverseHaarUp``
    output head).  Targets the ``(B, 3, 4, H/2, W/2)`` ``sub`` tensor returned
    by ``LLWv4Generator.forward(sar, return_internals=True)`` — dim=2 order is
    ``[LL, LH, HL, HH]``, matching the orthonormal Haar matrix used by
    :class:`~src.models.huggingface_gan.gen.HaarDown`.

    Target subbands are derived inline by applying ``HaarDown`` to the full-res
    optical target.  By the orthonormality of Haar, this matches what the
    generator's IHaar would invert — so per-band L1 against this target is
    mathematically equivalent to per-band L1 on the IHaar output (modulo the
    ``tanh`` nonlinearity at the very end).

    Two modes, switched at construction time:

    * **Manual (default, adaptive=False)**: ``band_weights`` are RATIOS
      (not absolute scales).  Internally normalised by the sum of weights so
      flipping all weights by ``k`` leaves the loss magnitude unchanged.
      Only the *master* ``per_band_weight`` in the loss config (a multiplier
      applied at the call site in ``main.py``) controls contribution scale.

    * **Adaptive (adaptive=True, v0.4.1 R4)**: per-band weights are learned
      via Kendall uncertainty weighting (Kendall, Gal, Cipolla 2018):
      ``L = Σᵢ exp(-log_varᵢ) · Lᵢ + 0.5 · Σᵢ log_varᵢ``.  ``log_var`` is a
      4-vector ``nn.Parameter`` (one entry per band), clamped to ``[-3, 3]``
      for σ-stability (precision ∈ [0.05, 5]).  ``band_weights`` arg is
      ignored.  Caller (factory) is responsible for adding ``log_var`` to
      ``opt_g``.

    Mutually exclusive with :class:`WaveletDetailL1Loss` at the factory
    level — both losses overlap on the Haar detail bands and would
    double-count the gradient.
    """
    _HAAR = torch.tensor([
        [ 1.0,  1.0,  1.0,  1.0],   # LL
        [-1.0,  1.0, -1.0,  1.0],   # LH
        [-1.0, -1.0,  1.0,  1.0],   # HL
        [ 1.0, -1.0, -1.0,  1.0],   # HH
    ], dtype=torch.float32) * 0.5
    _BAND_ORDER = ('ll', 'lh', 'hl', 'hh')
    _LOG_VAR_CLAMP = (-3.0, 3.0)

    def __init__(
        self,
        band_weights: dict | None = None,
        adaptive: bool = False,
        log_var_init: float = 0.0,
    ):
        super().__init__()
        self.adaptive = bool(adaptive)
        if self.adaptive:
            self.log_var = nn.Parameter(
                torch.full((4,), float(log_var_init), dtype=torch.float32)
            )
            self.register_buffer('band_weights', torch.ones(4, dtype=torch.float32))
        else:
            if band_weights is None:
                band_weights = {b: 1.0 for b in self._BAND_ORDER}
            weights = torch.tensor(
                [float(band_weights[b]) for b in self._BAND_ORDER],
                dtype=torch.float32,
            )
            self.register_buffer('band_weights', weights)
        self.register_buffer('haar_weights', self._HAAR.clone())
        self.last_per_band: dict[str, torch.Tensor] = {}
        self.last_precision: dict[str, torch.Tensor] = {}

    def _dwt(self, x: torch.Tensor) -> torch.Tensor:
        # channels_last -> NCHW contiguous; required by the .view below.
        x = x.contiguous()
        B, C, H, W = x.shape
        x_blocks = F.pixel_unshuffle(x, 2).view(B, C, 4, H // 2, W // 2)
        # bf16 autocast safety — mirror InverseHaarUp's dtype cast.
        w = self.haar_weights.to(dtype=x.dtype)
        return torch.einsum('bcihw, oi -> bcohw', x_blocks, w)

    def forward(self, sub_pred: torch.Tensor, opt: torch.Tensor) -> torch.Tensor:
        # fp32 cast — matches MSSSIMLoss / LPIPSLoss convention.  Trivial
        # VRAM cost on (B, 3, 4, H/2, W/2) and avoids any bf16 numeric drift.
        sub_pred_f = sub_pred.float()
        sub_real = self._dwt(opt.float())               # (B, C, 4, H/2, W/2)
        per_band = [
            F.l1_loss(sub_pred_f[:, :, i], sub_real[:, :, i])
            for i in range(4)
        ]
        for name, value in zip(self._BAND_ORDER, per_band):
            self.last_per_band[name] = value.detach()

        if self.adaptive:
            # Kendall uncertainty:  L = Σ exp(-log_var) L_i + 0.5 Σ log_var
            # Clamp σ ∈ [0.05, 5] prevents collapse (σ→0, weight→∞) and
            # blow-up (σ→∞, band ignored).
            #
            # Output divided by the band count (4) so that the scale matches
            # the manual-mode sum-of-weights normalisation at log_var=0 init
            # (precision=1 → task = Σ L_i; / 4 = mean(L_i) = manual equal-
            # weight output).  Both task and reg are divided by the same
            # constant, which preserves the Kendall equilibrium σ² = 2 L_i
            # exactly — only the absolute loss/gradient magnitude is rescaled
            # to keep warm-start parity with R1/R2 ckpts.
            log_var = self.log_var.clamp(*self._LOG_VAR_CLAMP)
            # Regulariser kept in fp32 (log_var's native dtype) for the sum
            # — bf16 mantissa would lose ~0.1 on a 4-element [-3, 3] sum,
            # which would otherwise perturb the Kendall gradient on log_var
            # at the same magnitude as a single SGD step.
            reg = (0.5 * log_var.sum()).to(dtype=sub_pred_f.dtype)
            log_var_c = log_var.to(dtype=sub_pred_f.dtype)
            precision = torch.exp(-log_var_c)
            task = sum(precision[i] * per_band[i] for i in range(4))
            for name, p in zip(self._BAND_ORDER, precision.detach()):
                self.last_precision[name] = p
            return (task + reg) / float(len(self._BAND_ORDER))

        weights = self.band_weights.to(dtype=sub_pred_f.dtype)
        for name, w in zip(self._BAND_ORDER, weights):
            self.last_precision[name] = w.detach()
        weighted_sum = sum(w * L for w, L in zip(weights, per_band))
        return weighted_sum / weights.sum().clamp_min(1e-8)


class FoundationPerceptualLoss(nn.Module):
    """Perceptual loss using a frozen pretrained foundation model.

    Forward both ``fake = G(SAR)`` and ``real`` through a HuggingFace
    ``AutoModel`` and compute L1 (or cosine) distance between intermediate
    transformer hidden states.  Backbone weights are frozen — only gradient
    flow is through the activations back to G.  Optional 3-channel-to-N
    learnable adapter handles backbones that expect more bands than RGB
    (e.g. Prithvi-EO HLS 6-band).

    Two practical backbones supported out of the box:

    * ``dinov2`` (default) — ``facebook/dinov2-base`` (86M, ImageNet self-
      supervised, well-supported by ``transformers.AutoModel``).  Provides a
      strong perceptual signal but trained on natural images.
    * ``prithvi-2.0-300m`` — ``ibm-nasa-geospatial/Prithvi-EO-2.0-300M``
      (300M, NASA/IBM satellite ViT-MAE, 6-band HLS).  Maximum domain
      alignment for SAR-to-optical translation, but the official loader is
      ``terratorch.registry.BACKBONE_REGISTRY``; vanilla ``AutoModel`` may
      fail.  Provided as opt-in; install ``terratorch`` first if needed.

    Implementation notes:
    * G output is assumed to live in ``[-1, 1]`` (tanh).  Mapped to
      ``[0, 1]`` then ImageNet-normalised for DINOv2; passed through raw
      for satellite backbones that expect reflectance values.
    * ``real`` features are computed under ``no_grad`` since the target
      tensor is detached from the autograd graph anyway.
    * Image is bilinearly resized to the backbone's expected ``input_size``
      (224 by default).  Trivial cost on (B, 3-6, 224, 224).
    """

    _PRESETS = {
        'dinov2': {
            'loader':      'transformers',
            'hf_name':     'facebook/dinov2-base',
            'in_channels': 3,
            'input_size':  224,
            'imgnet_norm': True,
        },
        'dinov2-small': {
            'loader':      'transformers',
            'hf_name':     'facebook/dinov2-small',
            'in_channels': 3,
            'input_size':  224,
            'imgnet_norm': True,
        },
        'convnextv2-tiny': {
            'loader':      'transformers',
            'hf_name':     'facebook/convnextv2-tiny-22k-224',
            'in_channels': 3,
            'input_size':  224,
            'imgnet_norm': True,
        },
        'prithvi-2.0-300m': {
            'loader':      'terratorch',
            'tt_name':     'prithvi_eo_v2_300',
            'in_channels': 6,                   # HLS: Blue, Green, Red, NarrowNIR, SWIR1, SWIR2
            'input_size':  224,
            'imgnet_norm': False,               # HLS reflectance values, no ImageNet stats
        },
        'prithvi-2.0-600m': {
            'loader':      'terratorch',
            'tt_name':     'prithvi_eo_v2_600',
            'in_channels': 6,
            'input_size':  224,
            'imgnet_norm': False,
        },
    }

    _IMGNET_MEAN = (0.485, 0.456, 0.406)
    _IMGNET_STD  = (0.229, 0.224, 0.225)

    def __init__(
        self,
        backbone: str = 'dinov2',
        distance: str = 'l1',
        channel_adapter: str = 'learnable',
        layer_idxs: list = None,
    ):
        super().__init__()
        if backbone not in self._PRESETS:
            raise ValueError(
                f"Unknown PEPL backbone='{backbone}'. "
                f"Available: {list(self._PRESETS.keys())}"
            )
        preset = self._PRESETS[backbone]
        self.backbone_name = backbone
        self.loader_type   = str(preset['loader'])
        self.in_channels   = int(preset['in_channels'])
        self.input_size    = int(preset['input_size'])
        self.imgnet_norm   = bool(preset['imgnet_norm'])
        self.distance      = str(distance).lower()
        self.layer_idxs    = list(layer_idxs) if layer_idxs else [-1]

        if self.distance not in ('l1', 'cosine'):
            raise ValueError(f"distance must be 'l1' or 'cosine', got {distance}")

        # Branch on loader: HuggingFace ``transformers`` for natural-image
        # backbones, ``terratorch`` for NASA/IBM satellite models.  Each path
        # exposes a uniform interface via ``self._features``.
        if self.loader_type == 'transformers':
            try:
                from transformers import AutoModel
            except ImportError as e:
                raise ImportError(
                    "FoundationPerceptualLoss requires `transformers` for this "
                    "backbone. Install with `pip install transformers`."
                ) from e
            self.backbone = AutoModel.from_pretrained(preset['hf_name'])
        elif self.loader_type == 'terratorch':
            try:
                from terratorch.registry import BACKBONE_REGISTRY
            except ImportError as e:
                raise ImportError(
                    "FoundationPerceptualLoss with a Prithvi-EO backbone "
                    "requires `terratorch`.  Install with `pip install terratorch`."
                ) from e
            self.backbone = BACKBONE_REGISTRY.build(
                preset['tt_name'], pretrained=True,
            )
        else:
            raise ValueError(f"Unknown loader_type='{self.loader_type}'")

        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

        # ImageNet norm buffers (only when needed).
        if self.imgnet_norm:
            mean = torch.tensor(self._IMGNET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
            std  = torch.tensor(self._IMGNET_STD,  dtype=torch.float32).view(1, 3, 1, 1)
            self.register_buffer('imgnet_mean', mean)
            self.register_buffer('imgnet_std',  std)

        # 3-channel-to-N adapter for satellite backbones.  Learnable 1x1
        # conv (18 params for 3->6) — initialised so first 3 channels = RGB
        # passthrough and any extras = mean(RGB).  Trains jointly with G.
        if self.in_channels == 3:
            self.channel_adapter = None
        elif channel_adapter == 'learnable':
            self.channel_adapter = nn.Conv2d(3, self.in_channels, kernel_size=1, bias=False)
            with torch.no_grad():
                self.channel_adapter.weight.zero_()
                for i in range(3):
                    self.channel_adapter.weight[i, i, 0, 0] = 1.0
                for i in range(3, self.in_channels):
                    self.channel_adapter.weight[i, :, 0, 0] = 1.0 / 3.0
        elif channel_adapter == 'replicate':
            self.channel_adapter = None  # handled in _prep
        else:
            raise ValueError(
                f"channel_adapter must be 'learnable' or 'replicate', got {channel_adapter}"
            )
        self._adapter_mode = channel_adapter

    def trainable_parameters(self):
        """Return only the trainable params (adapter), not the frozen backbone."""
        if self.channel_adapter is not None:
            yield from self.channel_adapter.parameters()

    def _prep(self, x: torch.Tensor) -> torch.Tensor:
        """Map G output [-1, 1] to backbone-ready input."""
        # G typically runs channels_last (NHWC) for ada-conv speedup; force
        # contiguous NCHW here — some frozen ViT backbones (Prithvi) trip
        # ``CUDNN_STATUS_BAD_PARAM`` on channels_last inputs.
        x = x.contiguous(memory_format=torch.contiguous_format)
        # [-1, 1] -> [0, 1]
        x = (x + 1.0) * 0.5
        # ImageNet norm if needed (DINOv2, ConvNeXt).  3-channel only.
        if self.imgnet_norm and x.size(1) == 3:
            x = (x - self.imgnet_mean) / self.imgnet_std
        # 3 -> N channel adapt.
        if self.channel_adapter is not None:
            x = self.channel_adapter(x)
        elif self.in_channels != 3:
            reps = (self.in_channels + 2) // 3
            x = x.repeat(1, reps, 1, 1)[:, :self.in_channels]
        # Resize to backbone's expected input size.
        if x.shape[-1] != self.input_size or x.shape[-2] != self.input_size:
            x = F.interpolate(
                x, size=self.input_size, mode='bilinear', align_corners=False,
            )
        # Re-assert contiguous after interpolate / channel adapt.
        return x.contiguous(memory_format=torch.contiguous_format)

    def _features(self, x: torch.Tensor) -> list:
        """Forward through frozen backbone and select hidden states.

        Two backend conventions handled:

        * ``transformers`` AutoModel returns a ``ModelOutput`` with
          ``hidden_states`` tuple when ``output_hidden_states=True``.
        * ``terratorch`` BACKBONE_REGISTRY returns a plain ``list`` of
          per-layer tensors directly from forward — no kwarg needed.

        Disables outer autocast: Prithvi's PrithviViT internals hit
        ``CUDNN_STATUS_BAD_PARAM`` under bf16-mixed autocast.  Cheaper to
        force fp32 here than to debug each cuDNN op individually — the
        backbone is frozen so no training-time precision is gained from
        running it in bf16 anyway.
        """
        device_type = 'cuda' if x.is_cuda else 'cpu'
        with torch.amp.autocast(device_type=device_type, enabled=False):
            x = x.float()
            if self.loader_type == 'transformers':
                outputs = self.backbone(x, output_hidden_states=True)
                hidden = getattr(outputs, 'hidden_states', None)
                if hidden is None:
                    hidden = (outputs.last_hidden_state,)
            else:  # terratorch
                hidden = self.backbone(x)
                if not isinstance(hidden, (list, tuple)):
                    hidden = (hidden,)
        return [hidden[i] for i in self.layer_idxs]

    def forward(self, fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        # fp32 cast for backbone forward (bf16 autocast can break some HF
        # models; safer to upcast and let the outer autocast scope decide
        # what to do with the scalar loss).
        fake_p = self._prep(fake.float())
        with torch.no_grad():
            real_p = self._prep(real.float())
        feats_fake = self._features(fake_p)
        with torch.no_grad():
            feats_real = self._features(real_p)
        losses = []
        for ff, fr in zip(feats_fake, feats_real):
            if self.distance == 'l1':
                losses.append(F.l1_loss(ff, fr))
            else:  # cosine
                cos = F.cosine_similarity(ff, fr, dim=-1)
                losses.append((1.0 - cos).mean())
        return sum(losses) / float(len(losses))


class PlainL1Loss(nn.Module):
    """Thin wrapper around ``F.l1_loss`` for style consistency with the suite.

    Replaces the previous ``UncertaintyL1Loss`` in sarformer-wb-3-simple — the
    uncertainty (log-variance) head was dominating the loss stack and starving
    the adversarial signal, with no measurable PSNR/SSIM benefit.
    """
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(pred, target)


# ---------------------------------------------------------------------------
# Tier-1 unit smoke for PerBandWaveletL1Loss.
# Run with::  python -m src.models.sarformer_wb.losses
# ---------------------------------------------------------------------------


def _smoke_perband() -> None:
    """Verify the contracts of :class:`PerBandWaveletL1Loss`."""
    torch.manual_seed(0)
    print("[perband smoke] running PerBandWaveletL1Loss contract checks")

    # Build an ideal-target tensor: sub_pred = HaarDown(opt) -> loss should be 0.
    opt = torch.randn(2, 3, 64, 64)
    loss = PerBandWaveletL1Loss()  # default equal weights
    sub_real = loss._dwt(opt)      # (B, 3, 4, H/2, W/2)
    err_round_trip = float(loss(sub_real, opt).item())
    status_a = 'OK' if err_round_trip < 1e-5 else 'FAIL'
    print(f"  [{status_a}] round-trip: loss(HaarDown(opt), opt) = {err_round_trip:.2e}  (expect ~0)")
    assert err_round_trip < 1e-5, f"round-trip mismatch: {err_round_trip}"

    # Zero prediction should give a nonzero loss equal to mean(|HaarDown(opt)|)
    # after sum-of-weights normalisation.
    sub_zero = torch.zeros_like(sub_real)
    err_zero = float(loss(sub_zero, opt).item())
    status_b = 'OK' if err_zero > 0.01 else 'FAIL'
    print(f"  [{status_b}] zero-pred: loss(0, opt) = {err_zero:.4f}  (expect > 0.01)")
    assert err_zero > 0.01, f"zero-pred loss suspiciously small: {err_zero}"

    # Normalisation invariance: doubling all band weights leaves the loss
    # unchanged (sum-of-weights normalisation).
    loss_2x = PerBandWaveletL1Loss(band_weights={'ll': 2.0, 'lh': 2.0, 'hl': 2.0, 'hh': 2.0})
    err_zero_2x = float(loss_2x(sub_zero, opt).item())
    diff = abs(err_zero - err_zero_2x)
    status_c = 'OK' if diff < 1e-6 else 'FAIL'
    print(f"  [{status_c}] normalisation: 1x vs 2x weights diff = {diff:.2e}  (expect ~0)")
    assert diff < 1e-6, f"normalisation contract broken: 1x={err_zero}, 2x={err_zero_2x}"

    # Per-band breakdown is populated for debug logging.
    bands = loss.last_per_band
    assert set(bands.keys()) == {'ll', 'lh', 'hl', 'hh'}, f"bad band keys: {set(bands.keys())}"
    print(f"  [OK] per-band cache populated: "
          f"ll={float(bands['ll']):.4f} lh={float(bands['lh']):.4f} "
          f"hl={float(bands['hl']):.4f} hh={float(bands['hh']):.4f}")

    # Detail-emphasised weights: same target, different scale than equal — but
    # also normalised, so flipping ratios changes magnitude only via the
    # *relative* per-band L1s.  Sanity-check: detail-emph loss differs from
    # equal-weight loss on a non-zero prediction.
    sub_noisy = sub_real + 0.05 * torch.randn_like(sub_real)
    loss_equal  = PerBandWaveletL1Loss(band_weights={'ll': 1.0, 'lh': 1.0, 'hl': 1.0, 'hh': 1.0})
    loss_detail = PerBandWaveletL1Loss(band_weights={'ll': 1.0, 'lh': 1.5, 'hl': 1.5, 'hh': 2.0})
    eq = float(loss_equal(sub_noisy, opt).item())
    de = float(loss_detail(sub_noisy, opt).item())
    print(f"  [info] equal-weight loss = {eq:.4f}, detail-emph loss = {de:.4f}  (should differ slightly)")

    # --- Adaptive mode (v0.4.1 R4) contract checks -------------------------
    print("[perband smoke] adaptive-mode contract checks")
    loss_ad = PerBandWaveletL1Loss(adaptive=True)
    # Must expose log_var as the only learnable parameter (4 entries).
    params = list(loss_ad.parameters())
    assert len(params) == 1 and params[0].numel() == 4, \
        f"adaptive must expose exactly 4 learnable params (log_var); got {[p.numel() for p in params]}"
    print(f"  [OK] adaptive exposes log_var Parameter shape={tuple(params[0].shape)}")

    # log_var init = 0 -> precision = 1, reg = 0.  Output divided by band
    # count (4) for warm-start parity → adaptive(log_var=0) == manual(equal
    # weights, sum-of-weights normalised) == mean(L_i).
    sub_pred = sub_real + 0.1 * torch.randn_like(sub_real)
    ad_init = float(loss_ad(sub_pred, opt).item())
    expected_mean = sum(
        F.l1_loss(sub_pred[:, :, i], sub_real[:, :, i]).item() for i in range(4)
    ) / 4.0
    diff = abs(ad_init - expected_mean)
    assert diff < 1e-5, f"adaptive init mismatch: got {ad_init}, expected {expected_mean}"
    # Cross-check: adaptive(log_var=0) must equal manual(equal weights).
    loss_eq = PerBandWaveletL1Loss(band_weights={'ll': 1.0, 'lh': 1.0, 'hl': 1.0, 'hh': 1.0})
    manual_eq = float(loss_eq(sub_pred, opt).item())
    parity_diff = abs(ad_init - manual_eq)
    assert parity_diff < 1e-5, f"adaptive≠manual at init: ad={ad_init}, manual={manual_eq}"
    print(f"  [OK] adaptive init: loss = mean(per_band) = {ad_init:.4f}  (matches manual equal-w)")

    # Gradient flows to log_var.
    sub_pred_g = (sub_real + 0.1 * torch.randn_like(sub_real)).requires_grad_(True)
    loss_ad2 = PerBandWaveletL1Loss(adaptive=True)
    out = loss_ad2(sub_pred_g, opt)
    out.backward()
    assert loss_ad2.log_var.grad is not None, "log_var.grad is None — adaptive disconnected"
    assert loss_ad2.log_var.grad.shape == (4,), f"log_var.grad shape: {loss_ad2.log_var.grad.shape}"
    print(f"  [OK] log_var receives gradient: {loss_ad2.log_var.grad.tolist()}")

    # Telemetry: last_precision populated in both modes.
    assert set(loss_ad2.last_precision.keys()) == {'ll', 'lh', 'hl', 'hh'}, \
        f"adaptive last_precision keys: {set(loss_ad2.last_precision.keys())}"
    assert set(loss_equal.last_precision.keys()) == {'ll', 'lh', 'hl', 'hh'}, \
        f"manual last_precision keys: {set(loss_equal.last_precision.keys())}"
    print("  [OK] last_precision telemetry populated in both manual and adaptive modes")

    # Clamp safety: extreme log_var initialisation does not produce NaN/inf.
    loss_clamp = PerBandWaveletL1Loss(adaptive=True, log_var_init=10.0)  # would blow up unclamped
    out_clamp = float(loss_clamp(sub_pred, opt).item())
    assert torch.isfinite(torch.tensor(out_clamp)), f"clamp failed: out={out_clamp}"
    print(f"  [OK] log_var clamp [-3,3] holds at init=10.0: out = {out_clamp:.4f} (finite)")

    print("[perband smoke] PASS — manual + adaptive contracts all clean")


class DeformationRegLoss(nn.Module):
    """Regularizer for the deformation field phi (B,2,H,W) in pixel units.

    Load-bearing: without it the aligner collapses (warps GT to match fake, recon -> 0).
    L = smooth_w * TV(phi) + mag_w * mean(||phi||^2).
    """
    def __init__(self, smooth_weight: float = 10.0, mag_weight: float = 1.0):
        super().__init__()
        self.smooth_weight = float(smooth_weight)
        self.mag_weight = float(mag_weight)

    def forward(self, phi: torch.Tensor) -> torch.Tensor:
        dx = (phi[:, :, :, 1:] - phi[:, :, :, :-1]).abs().mean()
        dy = (phi[:, :, 1:, :] - phi[:, :, :-1, :]).abs().mean()
        tv = dx + dy
        mag = (phi ** 2).mean()
        return self.smooth_weight * tv + self.mag_weight * mag


def _smoke_v5_losses() -> None:
    """Tier-1 smokes for the SAWG-Phi loss extensions (Tasks 3-5)."""
    print("[loss smoke] DeformationRegLoss")
    _reg = DeformationRegLoss(smooth_weight=10.0, mag_weight=1.0)
    _phi = torch.randn(2, 2, 256, 256, requires_grad=True)
    _l = _reg(_phi)
    assert torch.isfinite(_l), "reg loss not finite"
    _l.backward()
    assert _phi.grad is not None, "no grad to phi"
    assert _reg(torch.zeros(2, 2, 64, 64)).item() == 0.0, "zero phi should give zero reg"
    print(f"  [OK] DeformationRegLoss = {_l.item():.4f}, zero-phi=0")


if __name__ == '__main__':
    _smoke_perband()
    _smoke_v5_losses()


# ---------------------------------------------------------------------------
# llwt_v5 vendored extensions: LPIPS (perceptual), Focal Frequency Loss,
# and MultiLayerPatchNCE (alignment-robust contrastive). Single self-contained
# loss suite for v0.4.5.
# ---------------------------------------------------------------------------

from typing import Sequence as _Sequence

from src.models.llwt_v5.patchnce import PatchNCELoss, PatchSampleF


class LPIPSLoss(nn.Module):
    """AlexNet LPIPS (Zhang et al. 2018) via torchmetrics. Inputs in [-1, 1].

    Eager-built so AlexNet params register as children at __init__ (a lazy
    version desynced Lightning's EMA WeightAveraging shadow-zip).
    """
    def __init__(self, net_type: str = 'alex'):
        super().__init__()
        self.net_type = net_type
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
        self._lpips = LearnedPerceptualImagePatchSimilarity(net_type=net_type, normalize=False)
        self._lpips.eval()
        for p in self._lpips.parameters():
            p.requires_grad_(False)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Call the internal LPIPS network directly instead of the torchmetrics
        # Metric.forward(). As a Metric, forward() ACCUMULATES global state (a
        # cat-list of per-batch scores) on every call and never frees it when
        # used as a per-step loss (no reset) — leaking ~1 tensor/step. That
        # growing autograd/Python bookkeeping is the +15s/epoch train-time ramp
        # (GPU mem stays flat because the leaked tensors are tiny). ``net(...)``
        # is stateless and bit-identical to the metric value (verified). With
        # normalize=False the inputs are already in [-1, 1], exactly what
        # ``_lpips.net`` expects. ``.net`` keeps the same submodule, so the
        # module state_dict (and warm-start/resume) is unchanged.
        return self._lpips.net(pred.float(), target.float()).mean()


class FFLLoss(nn.Module):
    """Focal Frequency Loss (Jiang ICCV 2021) over the official pkg.

    Penalises hard-to-synthesise frequency components via focal re-weighting
    in the FFT domain. Inputs auto-cast to fp32 for bf16-mixed safety.
    """
    def __init__(self, alpha: float = 1.0, patch_factor: int = 1, ave_spectrum: bool = False):
        super().__init__()
        try:
            from focal_frequency_loss import FocalFrequencyLoss as FFL
        except ImportError as e:
            raise ImportError(
                "FFLLoss requires `focal-frequency-loss` "
                "(pip install focal-frequency-loss; Jiang ICCV 2021)."
            ) from e
        self._ffl = FFL(
            loss_weight=1.0, alpha=float(alpha), patch_factor=int(patch_factor),
            ave_spectrum=bool(ave_spectrum), log_matrix=False, batch_matrix=False,
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self._ffl(pred.float(), target.float())


class MultiLayerPatchNCE(nn.Module):
    """Owns the PatchSampleF MLP head + per-layer PatchNCE losses (CUT, Park 2020).

    forward(fake_feats, real_feats) -> scalar mean PatchNCE across the picked
    layers. Patch IDs sampled on the query (fake) side and reused on the key
    (real) side so positive pairs align by spatial position (paired-translation
    adaptation, alignment-robust in feature space).
    """
    def __init__(self, feature_dims: _Sequence, layer_ids: _Sequence, nc: int = 256,
                 num_patches: int = 256, temperature: float = 0.07, batch_size: int = 1):
        super().__init__()
        assert len(feature_dims) == len(layer_ids), (
            f"feature_dims ({len(feature_dims)}) and layer_ids ({len(layer_ids)}) length mismatch"
        )
        self.layer_ids = tuple(int(i) for i in layer_ids)
        self.num_patches = int(num_patches)
        self.sampler = PatchSampleF(feature_dims=[int(d) for d in feature_dims], nc=int(nc))
        self.nce_losses = nn.ModuleList([
            PatchNCELoss(temperature=temperature, batch_size=batch_size)
            for _ in feature_dims
        ])

    @staticmethod
    def _pick(feats, layer_ids):
        return [feats[i] for i in layer_ids]

    def forward(self, fake_feats_full, real_feats_full):
        fake_feats = self._pick(fake_feats_full, self.layer_ids)
        real_feats = self._pick(real_feats_full, self.layer_ids)
        fq_sampled, patch_ids = self.sampler(fake_feats, num_patches=self.num_patches)
        fk_sampled, _ = self.sampler(real_feats, num_patches=self.num_patches, patch_ids=patch_ids)
        total = 0.0
        for fq, fk, loss_fn in zip(fq_sampled, fk_sampled, self.nce_losses):
            total = total + loss_fn(fq, fk)
        return total / max(len(self.nce_losses), 1)
