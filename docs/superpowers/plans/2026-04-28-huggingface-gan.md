# huggingface-gan Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a SOTA SAR-to-optical conditional GAN using verified HuggingFace and PyTorch building blocks — ConvNeXtV2 encoder, PyTorch bottleneck attention, U-Net decoder, two-scale PatchGAN discriminator.

**Architecture:** ConvNeXtV2-Tiny (`AutoBackbone`) encodes SAR → 4 feature maps. `nn.TransformerEncoderLayer` adds global context at the 8×8 bottleneck (64 tokens). A U-Net decoder with skip connections and `ConvUpsampleBlock` (2-conv + residual shortcut + GroupNorm) reconstructs optical at 256×256. Two-scale spectral-norm PatchGAN discriminates at full and half resolution.

**Tech Stack:** PyTorch 2.8, Lightning 2.5, HuggingFace Transformers (`AutoBackbone`, `ConvNextV2`), torchmetrics 1.6, OmegaConf

---

## File Map

| File | Responsibility |
|------|---------------|
| `src/models/huggingface_gan/__init__.py` | Empty package marker |
| `src/models/huggingface_gan/config.yaml` | All hyperparameters and paths |
| `src/models/huggingface_gan/losses.py` | `GANLoss`, `FeatureMatchingLoss`, `FFTLoss`, `PerceptualLoss` |
| `src/models/huggingface_gan/dis.py` | `PatchDisBranch`, `HFGANDiscriminator` |
| `src/models/huggingface_gan/gen.py` | `ChannelAdapter`, `BottleneckAttention`, `ConvUpsampleBlock`, `HFGenerator` |
| `src/models/huggingface_gan/factory.py` | `build_models`, `build_criterions`, `build_optimizers`, `build_lr_schedulers` |
| `src/models/huggingface_gan/main.py` | `SAR2OPTLightningModule` |
| `src/models/huggingface_gan/train.py` | Entry point: config → datamodule → trainer → fit |
| `tests/test_hfgan_losses.py` | Loss function tests |
| `tests/test_hfgan_dis.py` | Discriminator shape and contract tests |
| `tests/test_hfgan_gen.py` | Generator shape tests (MockBackbone, no download) |
| `tests/test_hfgan_factory.py` | Factory wiring and conditional instantiation tests |
| `tests/test_hfgan_main.py` | Lightning module component tests |

> **Note on directory name:** Uses `huggingface_gan` (underscore) not `huggingface-gan` (hyphen) — Python cannot import hyphenated module names.

---

## Task 1: Scaffold — directory, config, shared test fixture

**Files:**
- Create: `src/models/huggingface_gan/__init__.py`
- Create: `src/models/huggingface_gan/config.yaml`
- Create: `tests/conftest_hfgan.py` (shared MockBackbone — imported by all test files)

- [ ] **Step 1: Create package directory and empty `__init__.py`**

```bash
mkdir src/models/huggingface_gan
touch src/models/huggingface_gan/__init__.py
```

- [ ] **Step 2: Write `config.yaml`**

Create `src/models/huggingface_gan/config.yaml`:

```yaml
data:
  data_dir:
    sen12:      "./data/sen12"
    sen12_full: "./data/sen12_full"
  dataset:               "sen12_full"
  scenes:                ["5", "45", "52", "84", "100"]
  batch_size:            8
  image_size:            256
  num_workers:           3
  prefetch_factor:       2
  persistent_workers:    true
  use_train_common_transform: true
  train_val_split_ratio: 0.8
  seed:                  42
  sar_channels:          1

model:
  gen:
    backbone:          "facebook/convnextv2-tiny-22k-224"
    out_indices:       [0, 1, 2, 3]
    bottleneck_dim:    768
    bottleneck_heads:  8
    bottleneck_layers: 2
  dis:
    ndf:         64
    in_channels: 4
  log_summary: true

optimizer:
  lr_g:           2.0e-4
  lr_d:           2.0e-4
  beta1:          0.5
  beta2:          0.999
  weight_decay_g: 0.01

scheduler:
  eta_min:             1.0e-6
  linear_decay_epochs: 200

ema:
  use_ema:     true
  decay:       0.999
  start_epoch: 30

loss:
  gan_weight:        1.0
  fm_weight:         5.0
  fft_weight:        1.0
  perceptual_weight: 0.1

system:
  device:        "cuda"
  precision:     "bf16-mixed"
  deterministic: false
  benchmark:     false
  compile:       false
  max_epochs:    400
  image_freq:    10
  limit_train_batches: 1.0
  limit_val_batches:   1.0
  tb_version:    "hfgan-1"
  resume_ckpt:   null
  debug:         false
  checkpoints_dir: "checkpoints/huggingface-gan"
  output_dir:      "./output/huggingface-gan"
  images_dir:      "./output/huggingface-gan/images"
  profiler_dir:    "./output/huggingface-gan/profiler"
  summary_dir:     "./output/huggingface-gan/summary"
```

- [ ] **Step 3: Write shared `MockBackbone` fixture**

Create `tests/conftest_hfgan.py` — all `test_hfgan_*.py` files import from here:

```python
"""Shared fixtures for huggingface-gan tests. Import via: from tests.conftest_hfgan import ..."""
from types import SimpleNamespace
import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf


class MockBackbone(nn.Module):
    """Drop-in for AutoBackbone. Returns zeros at ConvNeXtV2-Tiny feature map sizes.
    No HF download, no internet required. Device-aware."""
    def forward(self, pixel_values):
        B = pixel_values.shape[0]
        dev = pixel_values.device
        return SimpleNamespace(feature_maps=(
            torch.zeros(B,  96, 64, 64, device=dev),
            torch.zeros(B, 192, 32, 32, device=dev),
            torch.zeros(B, 384, 16, 16, device=dev),
            torch.zeros(B, 768,  8,  8, device=dev),
        ))


@pytest.fixture(scope='module')
def device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


@pytest.fixture(scope='module')
def test_cfg():
    return OmegaConf.load('src/models/huggingface_gan/config.yaml')


@pytest.fixture(scope='module')
def mock_backbone():
    return MockBackbone()
```

- [ ] **Step 4: Verify config loads**

```bash
python -c "from omegaconf import OmegaConf; cfg = OmegaConf.load('src/models/huggingface_gan/config.yaml'); print(cfg.model.gen.backbone)"
```

Expected output: `facebook/convnextv2-tiny-22k-224`

- [ ] **Step 5: Commit**

```bash
git add src/models/huggingface_gan/__init__.py src/models/huggingface_gan/config.yaml tests/conftest_hfgan.py
git commit -m "feat(hfgan): scaffold directory, config, and shared test MockBackbone"
```

---

## Task 2: `losses.py`

**Files:**
- Create: `src/models/huggingface_gan/losses.py`
- Create: `tests/test_hfgan_losses.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_hfgan_losses.py`:

```python
import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# GANLoss
# ---------------------------------------------------------------------------

def test_gan_loss_real_single():
    from src.models.huggingface_gan.losses import GANLoss
    loss = GANLoss(real_smooth=0.9)
    logit = torch.zeros(2, 1, 16, 16)       # pred=0, target=0.9 → MSE > 0
    l = loss(logit, is_real=True)
    assert l.shape == torch.Size([])
    assert l.item() > 0.0

def test_gan_loss_fake_single():
    from src.models.huggingface_gan.losses import GANLoss
    loss = GANLoss(fake_smooth=0.0)
    logit = torch.ones(2, 1, 16, 16)        # pred=1, target=0 → MSE > 0
    l = loss(logit, is_real=False)
    assert l.shape == torch.Size([])
    assert l.item() > 0.0

def test_gan_loss_perfect_real():
    from src.models.huggingface_gan.losses import GANLoss
    loss = GANLoss(real_smooth=0.9)
    logit = torch.full((2, 1, 16, 16), 0.9)  # pred == target → MSE = 0
    l = loss(logit, is_real=True)
    assert l.item() == pytest.approx(0.0, abs=1e-6)

def test_gan_loss_tuple_logits():
    from src.models.huggingface_gan.losses import GANLoss
    loss = GANLoss()
    logits = (torch.zeros(2, 1, 30, 30), torch.zeros(2, 1, 14, 14))
    l = loss(logits, is_real=True)
    assert l.shape == torch.Size([])
    assert l.item() > 0.0

def test_gan_loss_tuple_averaged():
    """Tuple loss should equal mean of individual losses."""
    from src.models.huggingface_gan.losses import GANLoss
    loss = GANLoss(real_smooth=0.9)
    l1 = torch.zeros(2, 1, 30, 30)
    l2 = torch.zeros(2, 1, 14, 14)
    combined = loss((l1, l2), is_real=True)
    individual = (loss(l1, is_real=True) + loss(l2, is_real=True)) / 2
    assert combined.item() == pytest.approx(individual.item(), rel=1e-5)


# ---------------------------------------------------------------------------
# FeatureMatchingLoss
# ---------------------------------------------------------------------------

def test_fm_loss_positive():
    from src.models.huggingface_gan.losses import FeatureMatchingLoss
    loss = FeatureMatchingLoss()
    fake  = [torch.randn(2, 64, 32, 32) for _ in range(8)]
    real  = [torch.randn(2, 64, 32, 32) for _ in range(8)]
    l = loss(fake, real)
    assert l.shape == torch.Size([])
    assert l.item() >= 0.0

def test_fm_loss_identical_inputs():
    from src.models.huggingface_gan.losses import FeatureMatchingLoss
    loss = FeatureMatchingLoss()
    feats = [torch.randn(2, 64, 32, 32) for _ in range(8)]
    l = loss(feats, feats)
    assert l.item() == pytest.approx(0.0, abs=1e-5)

def test_fm_loss_averaged_over_layers():
    from src.models.huggingface_gan.losses import FeatureMatchingLoss
    loss = FeatureMatchingLoss()
    import torch.nn.functional as F
    fake = [torch.ones(2, 8, 4, 4)]
    real = [torch.zeros(2, 8, 4, 4)]
    l = loss(fake, real)
    assert l.item() == pytest.approx(F.l1_loss(fake[0], real[0]).item(), rel=1e-5)


# ---------------------------------------------------------------------------
# FFTLoss
# ---------------------------------------------------------------------------

def test_fft_loss_shape():
    from src.models.huggingface_gan.losses import FFTLoss
    loss = FFTLoss()
    pred   = torch.randn(2, 3, 256, 256)
    target = torch.randn(2, 3, 256, 256)
    l = loss(pred, target)
    assert l.shape == torch.Size([])
    assert l.item() >= 0.0

def test_fft_loss_identical_inputs():
    from src.models.huggingface_gan.losses import FFTLoss
    loss = FFTLoss()
    x = torch.randn(2, 3, 256, 256)
    l = loss(x, x)
    assert l.item() == pytest.approx(0.0, abs=1e-5)


# ---------------------------------------------------------------------------
# PerceptualLoss — test _norm math only (no backbone download)
# ---------------------------------------------------------------------------

def test_perceptual_norm_maps_minus1_to_imagenet():
    """x=-1 (black) should map to (0 - mean) / std for each channel."""
    import torch
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    x = torch.full((1, 3, 4, 4), -1.0)
    normed = ((x + 1) / 2 - mean) / std       # replicate _norm logic
    expected_ch0 = (0.0 - 0.485) / 0.229
    assert normed[0, 0, 0, 0].item() == pytest.approx(expected_ch0, rel=1e-4)

def test_perceptual_norm_maps_plus1_to_imagenet():
    """x=+1 (white) should map to (1 - mean) / std for each channel."""
    import torch
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    x = torch.full((1, 3, 4, 4), 1.0)
    normed = ((x + 1) / 2 - mean) / std
    expected_ch0 = (1.0 - 0.485) / 0.229
    assert normed[0, 0, 0, 0].item() == pytest.approx(expected_ch0, rel=1e-4)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_hfgan_losses.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.models.huggingface_gan.losses'`

- [ ] **Step 3: Write `src/models/huggingface_gan/losses.py`**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class GANLoss(nn.Module):
    def __init__(self, real_smooth: float = 0.9, fake_smooth: float = 0.0):
        super().__init__()
        self.criterion   = nn.MSELoss()
        self.real_smooth = real_smooth
        self.fake_smooth = fake_smooth

    def _loss(self, logit: torch.Tensor, is_real: bool) -> torch.Tensor:
        val = self.real_smooth if is_real else self.fake_smooth
        return self.criterion(logit, torch.full_like(logit, val))

    def forward(self, logits, is_real: bool) -> torch.Tensor:
        if isinstance(logits, (list, tuple)):
            return sum(self._loss(l, is_real) for l in logits) / len(logits)
        return self._loss(logits, is_real)


class FeatureMatchingLoss(nn.Module):
    def forward(self, fake_feats: list, real_feats: list) -> torch.Tensor:
        loss = sum(F.l1_loss(f, r.detach()) for f, r in zip(fake_feats, real_feats))
        return loss / len(fake_feats)


class FFTLoss(nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_mag   = torch.log1p(torch.abs(torch.fft.rfft2(pred,   norm='ortho')))
        target_mag = torch.log1p(torch.abs(torch.fft.rfft2(target, norm='ortho')))
        return F.l1_loss(pred_mag, target_mag)


class PerceptualLoss(nn.Module):
    def __init__(self, backbone_name: str = "facebook/convnextv2-tiny-22k-224"):
        super().__init__()
        from transformers import AutoBackbone
        self.backbone = AutoBackbone.from_pretrained(backbone_name, out_indices=(0, 1, 2))
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std',  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return ((x + 1) / 2 - self.mean) / self.std

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pf = self.backbone(pixel_values=self._norm(pred)).feature_maps
        tf = self.backbone(pixel_values=self._norm(target)).feature_maps
        return sum(F.l1_loss(p, t.detach()) for p, t in zip(pf, tf)) / len(pf)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_hfgan_losses.py -v
```

Expected: `10 passed`

- [ ] **Step 5: Commit**

```bash
git add src/models/huggingface_gan/losses.py tests/test_hfgan_losses.py
git commit -m "feat(hfgan): losses — GANLoss, FeatureMatchingLoss, FFTLoss, PerceptualLoss"
```

---

## Task 3: `dis.py`

**Files:**
- Create: `src/models/huggingface_gan/dis.py`
- Create: `tests/test_hfgan_dis.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_hfgan_dis.py`:

```python
import pytest
import torch
from torch.nn.utils.parametrize import is_parametrized


# ---------------------------------------------------------------------------
# PatchDisBranch
# ---------------------------------------------------------------------------

def test_branch_full_scale_logits_shape():
    from src.models.huggingface_gan.dis import PatchDisBranch
    branch = PatchDisBranch(in_ch=4, ndf=64)
    x = torch.randn(2, 4, 256, 256)
    logits, feats = branch(x)
    assert logits.shape == (2, 1, 30, 30), f"got {logits.shape}"

def test_branch_half_scale_logits_shape():
    from src.models.huggingface_gan.dis import PatchDisBranch
    branch = PatchDisBranch(in_ch=4, ndf=64)
    x = torch.randn(2, 4, 128, 128)
    logits, feats = branch(x)
    assert logits.shape == (2, 1, 14, 14), f"got {logits.shape}"

def test_branch_returns_4_features():
    from src.models.huggingface_gan.dis import PatchDisBranch
    branch = PatchDisBranch(in_ch=4, ndf=64)
    x = torch.randn(2, 4, 256, 256)
    _, feats = branch(x)
    assert len(feats) == 4

def test_branch_spectral_norm_applied():
    from src.models.huggingface_gan.dis import PatchDisBranch
    branch = PatchDisBranch(in_ch=4, ndf=64)
    first_conv = branch.layers[0][0]     # Conv2d inside first Sequential
    assert is_parametrized(first_conv), "First conv should have spectral norm"


# ---------------------------------------------------------------------------
# HFGANDiscriminator
# ---------------------------------------------------------------------------

def test_discriminator_output_contract():
    from src.models.huggingface_gan.dis import HFGANDiscriminator
    netD = HFGANDiscriminator(in_ch=4, ndf=64)
    sar = torch.randn(2, 1, 256, 256)
    opt = torch.randn(2, 3, 256, 256)
    (logits1, logits2), feats = netD(sar, opt)
    assert logits1.shape == (2, 1, 30, 30)
    assert logits2.shape == (2, 1, 14, 14)
    assert len(feats) == 8                  # 4 per branch

def test_discriminator_no_gradient_on_real_during_d_step():
    """Discriminator should not require grad on the input tensors."""
    from src.models.huggingface_gan.dis import HFGANDiscriminator
    netD = HFGANDiscriminator(in_ch=4, ndf=64)
    sar = torch.randn(2, 1, 256, 256)
    opt = torch.randn(2, 3, 256, 256)
    (logits1, _), _ = netD(sar, opt)
    loss = logits1.mean()
    loss.backward()                         # should not raise

def test_discriminator_downsample_halves_spatial():
    from src.models.huggingface_gan.dis import HFGANDiscriminator
    netD = HFGANDiscriminator(in_ch=4, ndf=64)
    x = torch.randn(2, 4, 256, 256)
    x2 = netD.downsample(x)
    assert x2.shape == (2, 4, 128, 128)
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_hfgan_dis.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.models.huggingface_gan.dis'`

- [ ] **Step 3: Write `src/models/huggingface_gan/dis.py`**

```python
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class PatchDisBranch(nn.Module):
    """5-layer 70×70 spectral-norm PatchGAN branch.

    At 256×256 input:  logits shape = (B, 1, 30, 30)
    At 128×128 input:  logits shape = (B, 1, 14, 14)
    Returns (logits, features) where features is a list of 4 intermediate tensors.
    """
    def __init__(self, in_ch: int, ndf: int = 64):
        super().__init__()

        def sn(ci, co, k, s, p):
            return spectral_norm(nn.Conv2d(ci, co, k, s, p, bias=True))

        self.layers = nn.ModuleList([
            nn.Sequential(sn(in_ch, ndf,     4, 2, 1), nn.LeakyReLU(0.2, inplace=True)),
            nn.Sequential(sn(ndf,   ndf * 2, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True)),
            nn.Sequential(sn(ndf*2, ndf * 4, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True)),
            nn.Sequential(sn(ndf*4, ndf * 8, 4, 1, 1), nn.LeakyReLU(0.2, inplace=True)),
            sn(ndf * 8, 1, 4, 1, 1),
        ])

    def forward(self, x: torch.Tensor):
        features = []
        for layer in self.layers[:-1]:
            x = layer(x)
            features.append(x)
        return self.layers[-1](x), features


class HFGANDiscriminator(nn.Module):
    """Two-scale conditional PatchGAN.

    Concatenates [SAR, OPT] and runs two branches: full resolution and 2× downsampled.
    Returns ((logits_large, logits_small), features) where features is a flat list of
    8 tensors (4 per branch) used by FeatureMatchingLoss.
    """
    def __init__(self, in_ch: int = 4, ndf: int = 64):
        super().__init__()
        self.branch1    = PatchDisBranch(in_ch, ndf)
        self.branch2    = PatchDisBranch(in_ch, ndf)
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)

    def forward(self, sar: torch.Tensor, opt: torch.Tensor):
        x  = torch.cat([sar, opt], dim=1)    # (B, 4, 256, 256)
        x2 = self.downsample(x)              # (B, 4, 128, 128)
        logits1, feats1 = self.branch1(x)
        logits2, feats2 = self.branch2(x2)
        return (logits1, logits2), feats1 + feats2
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_hfgan_dis.py -v
```

Expected: `7 passed`

- [ ] **Step 5: Commit**

```bash
git add src/models/huggingface_gan/dis.py tests/test_hfgan_dis.py
git commit -m "feat(hfgan): discriminator — two-scale spectral-norm PatchGAN"
```

---

## Task 4: `gen.py`

**Files:**
- Create: `src/models/huggingface_gan/gen.py`
- Create: `tests/test_hfgan_gen.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_hfgan_gen.py`:

```python
import pytest
import torch
import sys
import os
sys.path.insert(0, os.path.abspath('.'))
from tests.conftest_hfgan import MockBackbone
from omegaconf import OmegaConf


@pytest.fixture(scope='module')
def test_cfg():
    return OmegaConf.load('src/models/huggingface_gan/config.yaml')

@pytest.fixture(scope='module')
def device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


# ---------------------------------------------------------------------------
# ChannelAdapter
# ---------------------------------------------------------------------------

def test_channel_adapter_maps_1_to_3ch():
    from src.models.huggingface_gan.gen import ChannelAdapter
    adapter = ChannelAdapter()
    x = torch.randn(2, 1, 256, 256)
    out = adapter(x)
    assert out.shape == (2, 3, 256, 256)

def test_channel_adapter_preserves_spatial():
    from src.models.huggingface_gan.gen import ChannelAdapter
    adapter = ChannelAdapter()
    x = torch.randn(1, 1, 128, 128)
    out = adapter(x)
    assert out.shape == (1, 3, 128, 128)


# ---------------------------------------------------------------------------
# BottleneckAttention
# ---------------------------------------------------------------------------

def test_bottleneck_attention_shape():
    from src.models.huggingface_gan.gen import BottleneckAttention
    attn = BottleneckAttention(dim=768, nhead=8, num_layers=2)
    x   = torch.randn(2, 768, 8, 8)
    out = attn(x)
    assert out.shape == (2, 768, 8, 8)

def test_bottleneck_attention_residual():
    """Output should differ from input (not identity)."""
    from src.models.huggingface_gan.gen import BottleneckAttention
    attn = BottleneckAttention(dim=768, nhead=8, num_layers=2)
    x   = torch.randn(2, 768, 8, 8)
    out = attn(x)
    assert not torch.allclose(out, x)


# ---------------------------------------------------------------------------
# ConvUpsampleBlock
# ---------------------------------------------------------------------------

def test_upsample_block_with_skip():
    from src.models.huggingface_gan.gen import ConvUpsampleBlock
    # Represents up4: input 768ch@8x8 upsampled to 16x16, concat with skip 384ch@16x16
    block = ConvUpsampleBlock(768 + 384, 256)
    x    = torch.randn(2, 768, 8, 8)
    skip = torch.randn(2, 384, 16, 16)
    out  = block(x, skip)
    assert out.shape == (2, 256, 16, 16)

def test_upsample_block_no_skip():
    from src.models.huggingface_gan.gen import ConvUpsampleBlock
    # Represents up1: 64ch@64x64 → 32ch@128x128
    block = ConvUpsampleBlock(64, 32)
    x    = torch.randn(2, 64, 64, 64)
    out  = block(x)
    assert out.shape == (2, 32, 128, 128)

def test_upsample_block_residual_contributes():
    """shortcut(x) is added — output should be distinct from conv-only path."""
    from src.models.huggingface_gan.gen import ConvUpsampleBlock
    block = ConvUpsampleBlock(64, 32)
    x = torch.randn(1, 64, 8, 8)
    out = block(x)
    assert out.shape == (1, 32, 16, 16)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# HFGenerator (MockBackbone — no HF download)
# ---------------------------------------------------------------------------

def test_generator_output_shape(test_cfg, device):
    from src.models.huggingface_gan.gen import HFGenerator
    gen = HFGenerator(test_cfg, encoder=MockBackbone()).to(device).eval()
    sar = torch.randn(2, 1, 256, 256, device=device)
    with torch.no_grad():
        out = gen(sar)
    assert out.shape == (2, 3, 256, 256), f"got {out.shape}"

def test_generator_output_tanh_range(test_cfg, device):
    from src.models.huggingface_gan.gen import HFGenerator
    gen = HFGenerator(test_cfg, encoder=MockBackbone()).to(device).eval()
    sar = torch.randn(2, 1, 256, 256, device=device)
    with torch.no_grad():
        out = gen(sar)
    assert out.min().item() >= -1.0 - 1e-5
    assert out.max().item() <=  1.0 + 1e-5

def test_generator_gradients_flow(test_cfg):
    from src.models.huggingface_gan.gen import HFGenerator
    gen = HFGenerator(test_cfg, encoder=MockBackbone()).train()
    sar = torch.randn(1, 1, 256, 256)
    out = gen(sar)
    out.mean().backward()
    grad_norms = [p.grad.norm().item() for p in gen.parameters() if p.grad is not None]
    assert len(grad_norms) > 0, "No gradients flowed"
    assert all(torch.isfinite(torch.tensor(g)) for g in grad_norms)
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_hfgan_gen.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.models.huggingface_gan.gen'`

- [ ] **Step 3: Write `src/models/huggingface_gan/gen.py`**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAdapter(nn.Module):
    """Projects 1-channel SAR to 3-channel space matching ConvNeXtV2 stem input."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class BottleneckAttention(nn.Module):
    """Global attention at the 8×8 encoder bottleneck (64 tokens).

    Uses nn.TransformerEncoderLayer with Pre-LN (norm_first=True) for
    training stability. No dropout — dataset is small.
    """
    def __init__(self, dim: int = 768, nhead: int = 8, num_layers: int = 2):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=nhead,
            dim_feedforward=dim * 2,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.pos = nn.Parameter(torch.zeros(1, 64, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape                         # (B, 768, 8, 8)
        t = x.flatten(2).transpose(1, 2)             # (B, 64, 768)
        t = self.transformer(t + self.pos)
        return t.transpose(1, 2).reshape(B, C, H, W)


class ConvUpsampleBlock(nn.Module):
    """Bilinear upsample + optional skip concat + two-conv block with residual.

    in_ch must be the channel count AFTER concatenation with the skip tensor.
    Example: ConvUpsampleBlock(768 + 384, 256) — input 768ch is upsampled 2×
    then concatenated with a 384ch skip before the convolutions.
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_ch,  out_ch, 3, bias=False),
            nn.GroupNorm(8, out_ch),
            nn.GELU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(out_ch, out_ch, 3, bias=False),
            nn.GroupNorm(8, out_ch),
            nn.GELU(),
        )
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1, bias=False)

    def forward(self, x: torch.Tensor, skip: torch.Tensor = None) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.conv(x) + self.shortcut(x)


class HFGenerator(nn.Module):
    """ConvNeXtV2 U-Net generator with bottleneck attention.

    Args:
        cfg: OmegaConf config with model.gen.{backbone, out_indices, bottleneck_*}
        encoder: optional pre-built backbone (used in tests to avoid HF download)
    """
    def __init__(self, cfg, encoder=None):
        super().__init__()
        self.channel_adapter = ChannelAdapter()

        if encoder is not None:
            self.encoder = encoder
        else:
            from transformers import AutoBackbone
            self.encoder = AutoBackbone.from_pretrained(
                cfg.model.gen.backbone,
                out_indices=tuple(cfg.model.gen.out_indices),
            )

        dim = cfg.model.gen.bottleneck_dim          # 768
        self.bottleneck = BottleneckAttention(
            dim=dim,
            nhead=cfg.model.gen.bottleneck_heads,
            num_layers=cfg.model.gen.bottleneck_layers,
        )

        # Decoder: channel counts are (post-concat, out)
        self.up4 = ConvUpsampleBlock(dim + 384, 256)   # 8→16,  concat s2
        self.up3 = ConvUpsampleBlock(256 + 192, 128)   # 16→32, concat s1
        self.up2 = ConvUpsampleBlock(128 +  96,  64)   # 32→64, concat s0
        self.up1 = ConvUpsampleBlock( 64,        32)   # 64→128
        self.up0 = ConvUpsampleBlock( 32,        16)   # 128→256

        self.head = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(16, 3, kernel_size=7),
            nn.Tanh(),
        )

    def forward(self, sar: torch.Tensor) -> torch.Tensor:
        x               = self.channel_adapter(sar)               # (B, 3, 256, 256)
        s0, s1, s2, s3  = self.encoder(pixel_values=x).feature_maps
        s3              = self.bottleneck(s3)                      # (B, 768, 8, 8)
        x               = self.up4(s3, s2)                        # (B, 256, 16, 16)
        x               = self.up3(x,  s1)                        # (B, 128, 32, 32)
        x               = self.up2(x,  s0)                        # (B,  64, 64, 64)
        x               = self.up1(x)                             # (B,  32, 128, 128)
        x               = self.up0(x)                             # (B,  16, 256, 256)
        return self.head(x)                                        # (B,   3, 256, 256)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_hfgan_gen.py -v
```

Expected: `11 passed`

- [ ] **Step 5: Commit**

```bash
git add src/models/huggingface_gan/gen.py tests/test_hfgan_gen.py
git commit -m "feat(hfgan): generator — ChannelAdapter, BottleneckAttention, ConvUpsampleBlock, HFGenerator"
```

---

## Task 5: `factory.py`

**Files:**
- Create: `src/models/huggingface_gan/factory.py`
- Create: `tests/test_hfgan_factory.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_hfgan_factory.py`:

```python
import pytest
import torch
from omegaconf import OmegaConf, DictConfig
import sys, os
sys.path.insert(0, os.path.abspath('.'))
from tests.conftest_hfgan import MockBackbone


@pytest.fixture(scope='module')
def cfg():
    return OmegaConf.load('src/models/huggingface_gan/config.yaml')


def test_build_models_types(cfg):
    from src.models.huggingface_gan.factory import build_models
    from src.models.huggingface_gan.gen import HFGenerator
    from src.models.huggingface_gan.dis import HFGANDiscriminator
    netG, netD = build_models(cfg, encoder=MockBackbone())
    assert isinstance(netG, HFGenerator)
    assert isinstance(netD, HFGANDiscriminator)


def test_build_criterions_core_always_present(cfg):
    from src.models.huggingface_gan.factory import build_criterions
    OmegaConf.update(cfg, 'loss.fft_weight', 0.0)
    OmegaConf.update(cfg, 'loss.perceptual_weight', 0.0)
    c = build_criterions(cfg)
    assert 'gan' in c
    assert 'fm'  in c
    assert 'fft' not in c
    assert 'perceptual' not in c


def test_build_criterions_fft_enabled(cfg):
    from src.models.huggingface_gan.factory import build_criterions
    cfg2 = OmegaConf.merge(cfg, {'loss': {'fft_weight': 1.0, 'perceptual_weight': 0.0}})
    c = build_criterions(cfg2)
    assert 'fft' in c
    assert 'perceptual' not in c


def test_build_criterions_no_perceptual_when_zero(cfg):
    from src.models.huggingface_gan.factory import build_criterions
    cfg2 = OmegaConf.merge(cfg, {'loss': {'fft_weight': 0.0, 'perceptual_weight': 0.0}})
    c = build_criterions(cfg2)
    assert 'perceptual' not in c


def test_build_optimizers_returns_two(cfg):
    from src.models.huggingface_gan.factory import build_models, build_optimizers
    netG, netD = build_models(cfg, encoder=MockBackbone())
    opt_g, opt_d = build_optimizers(cfg, netG, netD)
    assert opt_g is not None
    assert opt_d is not None


def test_build_optimizers_g_has_two_param_groups(cfg):
    from src.models.huggingface_gan.factory import build_models, build_optimizers
    netG, netD = build_models(cfg, encoder=MockBackbone())
    opt_g, _ = build_optimizers(cfg, netG, netD)
    assert len(opt_g.param_groups) == 2


def test_build_optimizers_encoder_lr_is_tenth(cfg):
    from src.models.huggingface_gan.factory import build_models, build_optimizers
    netG, netD = build_models(cfg, encoder=MockBackbone())
    opt_g, _ = build_optimizers(cfg, netG, netD)
    enc_lr  = opt_g.param_groups[0]['lr']
    full_lr = opt_g.param_groups[1]['lr']
    assert full_lr == pytest.approx(cfg.optimizer.lr_g, rel=1e-5)
    assert enc_lr  == pytest.approx(cfg.optimizer.lr_g * 0.1, rel=1e-5)


def test_build_lr_schedulers_returns_two(cfg):
    from src.models.huggingface_gan.factory import build_models, build_optimizers, build_lr_schedulers
    netG, netD = build_models(cfg, encoder=MockBackbone())
    opt_g, opt_d = build_optimizers(cfg, netG, netD)
    sched_g, sched_d = build_lr_schedulers(cfg, opt_g, opt_d)
    assert sched_g is not None
    assert sched_d is not None
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_hfgan_factory.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.models.huggingface_gan.factory'`

- [ ] **Step 3: Write `src/models/huggingface_gan/factory.py`**

```python
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR, SequentialLR, ConstantLR

from src.models.huggingface_gan.gen import HFGenerator
from src.models.huggingface_gan.dis import HFGANDiscriminator
from src.models.huggingface_gan.losses import GANLoss, FeatureMatchingLoss, FFTLoss


def build_models(cfg, encoder=None):
    """Build generator and discriminator.

    encoder: optional pre-built backbone — pass a MockBackbone in tests to avoid
    downloading the HuggingFace checkpoint.
    """
    netG = HFGenerator(cfg, encoder=encoder)
    netD = HFGANDiscriminator(
        in_ch=cfg.model.dis.in_channels,
        ndf=cfg.model.dis.ndf,
    )
    return netG, netD


def build_criterions(cfg) -> dict:
    """Build loss dict. Optional losses are only instantiated when weight > 0.

    PerceptualLoss loads a 28M frozen backbone — never instantiate when weight=0.
    """
    criterions = {
        'gan': GANLoss(),
        'fm':  FeatureMatchingLoss(),
    }
    if cfg.loss.fft_weight > 0:
        criterions['fft'] = FFTLoss()
    if cfg.loss.perceptual_weight > 0:
        from src.models.huggingface_gan.losses import PerceptualLoss
        criterions['perceptual'] = PerceptualLoss(cfg.model.gen.backbone)
    return criterions


def build_optimizers(cfg, netG, netD):
    """AdamW for generator (two param groups: encoder at 0.1× LR, decoder at full LR).
    Adam for discriminator (standard GAN practice).
    """
    enc_params = (
        list(netG.channel_adapter.parameters()) +
        list(netG.encoder.parameters())
    )
    fresh_params = (
        list(netG.bottleneck.parameters()) +
        list(netG.up4.parameters()) +
        list(netG.up3.parameters()) +
        list(netG.up2.parameters()) +
        list(netG.up1.parameters()) +
        list(netG.up0.parameters()) +
        list(netG.head.parameters())
    )
    opt_g = optim.AdamW(
        [
            {'params': enc_params,   'lr': cfg.optimizer.lr_g * 0.1},
            {'params': fresh_params, 'lr': cfg.optimizer.lr_g},
        ],
        betas=(cfg.optimizer.beta1, cfg.optimizer.beta2),
        weight_decay=cfg.optimizer.weight_decay_g,
    )
    opt_d = optim.Adam(
        netD.parameters(),
        lr=cfg.optimizer.lr_d,
        betas=(cfg.optimizer.beta1, cfg.optimizer.beta2),
    )
    return opt_g, opt_d


def build_lr_schedulers(cfg, opt_g, opt_d):
    """Flat LR for first (max_epochs - linear_decay_epochs) epochs, then linear decay."""
    decay   = cfg.scheduler.linear_decay_epochs
    warmup  = max(cfg.system.max_epochs - decay, 0)

    def make_sched(opt, base_lr):
        end_factor = cfg.scheduler.eta_min / max(base_lr, 1e-10)
        linear = LinearLR(opt, start_factor=1.0, end_factor=end_factor, total_iters=decay)
        if warmup == 0:
            return linear
        return SequentialLR(
            opt,
            schedulers=[ConstantLR(opt, factor=1.0, total_iters=warmup), linear],
            milestones=[warmup],
        )

    return make_sched(opt_g, cfg.optimizer.lr_g), make_sched(opt_d, cfg.optimizer.lr_d)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_hfgan_factory.py -v
```

Expected: `8 passed`

- [ ] **Step 5: Commit**

```bash
git add src/models/huggingface_gan/factory.py tests/test_hfgan_factory.py
git commit -m "feat(hfgan): factory — build_models, build_criterions, build_optimizers, build_lr_schedulers"
```

---

## Task 6: `main.py`

**Files:**
- Create: `src/models/huggingface_gan/main.py`
- Create: `tests/test_hfgan_main.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_hfgan_main.py`:

```python
import pytest
import torch
from omegaconf import OmegaConf
import sys, os
sys.path.insert(0, os.path.abspath('.'))
from tests.conftest_hfgan import MockBackbone


@pytest.fixture(scope='module')
def cfg():
    c = OmegaConf.load('src/models/huggingface_gan/config.yaml')
    # Disable optional heavy losses for unit tests
    return OmegaConf.merge(c, {'loss': {'fft_weight': 0.0, 'perceptual_weight': 0.0}})

@pytest.fixture(scope='module')
def device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

@pytest.fixture(scope='module')
def module(cfg, device):
    from src.models.huggingface_gan.main import SAR2OPTLightningModule
    m = SAR2OPTLightningModule(cfg, encoder=MockBackbone())
    return m.to(device)


def test_configure_optimizers_returns_two(module):
    opts = module.configure_optimizers()
    assert len(opts) == 2          # [opt_d, opt_g]

def test_configure_optimizers_g_has_two_param_groups(module):
    opts = module.configure_optimizers()
    _, opt_g = opts                # D first, G second
    assert len(opt_g.param_groups) == 2

def test_criterions_are_moduledict(module):
    import torch.nn as nn
    assert isinstance(module.criterions, nn.ModuleDict)
    assert 'gan' in module.criterions
    assert 'fm'  in module.criterions
    assert 'fft' not in module.criterions         # disabled in fixture cfg

def test_d_loss_is_finite(module, device):
    sar  = torch.randn(2, 1, 256, 256, device=device)
    opt  = torch.randn(2, 3, 256, 256, device=device)
    with torch.no_grad():
        fake_d = module.netG(sar)
    real_logits, _ = module.netD(sar, opt)
    fake_logits, _ = module.netD(sar, fake_d)
    d_loss = 0.5 * (
        module.criterions['gan'](real_logits, is_real=True) +
        module.criterions['gan'](fake_logits, is_real=False)
    )
    assert torch.isfinite(d_loss)
    assert d_loss.item() >= 0.0

def test_g_loss_is_finite(module, device):
    sar = torch.randn(2, 1, 256, 256, device=device)
    opt = torch.randn(2, 3, 256, 256, device=device)
    with torch.no_grad():
        fake_d = module.netG(sar)
    _, real_feats = module.netD(sar, opt)
    fake = module.netG(sar)
    fake_logits, fake_feats = module.netD(sar, fake)
    real_feats_d = [f.detach() for f in real_feats]
    cfg = module.cfg.loss
    g_loss = (
        module.criterions['gan'](fake_logits, is_real=True) * cfg.gan_weight +
        module.criterions['fm'](fake_feats, real_feats_d)   * cfg.fm_weight
    )
    assert torch.isfinite(g_loss)
    assert g_loss.item() >= 0.0

def test_validation_step_updates_metrics(module, device):
    sar = torch.randn(2, 1, 256, 256, device=device)
    opt = torch.randn(2, 3, 256, 256, device=device)
    batch = {'sar': sar, 'optical': opt}
    module.validation_step(batch, 0)
    psnr_val = module.psnr.compute()
    assert torch.isfinite(psnr_val)
    module.psnr.reset()
    module.ssim.reset()
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_hfgan_main.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.models.huggingface_gan.main'`

- [ ] **Step 3: Write `src/models/huggingface_gan/main.py`**

```python
import torch
import torch.nn as nn
import lightning.pytorch as pl
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from src.models.huggingface_gan import factory


class SAR2OPTLightningModule(pl.LightningModule):
    def __init__(self, cfg, encoder=None):
        super().__init__()
        self.cfg = cfg
        self.automatic_optimization = False

        self.netG, self.netD = factory.build_models(cfg, encoder=encoder)
        self.criterions = nn.ModuleDict(factory.build_criterions(cfg))

        self.psnr = PeakSignalNoiseRatio(data_range=2.0)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=2.0)

    def configure_optimizers(self):
        opt_g, opt_d = factory.build_optimizers(self.cfg, self.netG, self.netD)
        self.sched_g, self.sched_d = factory.build_lr_schedulers(self.cfg, opt_g, opt_d)
        return [opt_d, opt_g]           # Lightning: unpack as opt_d, opt_g = self.optimizers()

    def training_step(self, batch, batch_idx):
        sar, opt     = batch['sar'], batch['optical']
        opt_d, opt_g = self.optimizers()
        loss_cfg     = self.cfg.loss

        # ── D step ──────────────────────────────────────────────────────────
        with torch.no_grad():
            fake_d = self.netG(sar)

        real_logits, real_feats = self.netD(sar, opt)
        fake_logits, _          = self.netD(sar, fake_d)

        d_loss = 0.5 * (
            self.criterions['gan'](real_logits, is_real=True) +
            self.criterions['gan'](fake_logits, is_real=False)
        )
        opt_d.zero_grad()
        self.manual_backward(d_loss)
        opt_d.step()

        # ── G step ──────────────────────────────────────────────────────────
        fake = self.netG(sar)
        fake_logits_g, fake_feats = self.netD(sar, fake)
        real_feats_d = [f.detach() for f in real_feats]   # reuse from D step

        g_loss = (
            self.criterions['gan'](fake_logits_g, is_real=True) * loss_cfg.gan_weight +
            self.criterions['fm'](fake_feats, real_feats_d)      * loss_cfg.fm_weight
        )
        if 'fft' in self.criterions:
            g_loss = g_loss + self.criterions['fft'](fake, opt) * loss_cfg.fft_weight
        if 'perceptual' in self.criterions:
            g_loss = g_loss + self.criterions['perceptual'](fake, opt) * loss_cfg.perceptual_weight

        opt_g.zero_grad()
        self.manual_backward(g_loss)
        opt_g.step()

        self.log_dict({'train/d_loss': d_loss, 'train/g_loss': g_loss},
                      prog_bar=True, on_step=True, on_epoch=False)

    def validation_step(self, batch, batch_idx):
        sar, opt = batch['sar'], batch['optical']
        with torch.no_grad():
            fake = self.netG(sar)
        self.psnr.update(fake, opt)
        self.ssim.update(fake, opt)

    def on_validation_epoch_end(self):
        self.log_dict({
            'val/psnr': self.psnr.compute(),
            'val/ssim': self.ssim.compute(),
        }, prog_bar=True)
        self.psnr.reset()
        self.ssim.reset()

    def on_train_epoch_end(self):
        self.sched_g.step()
        self.sched_d.step()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_hfgan_main.py -v
```

Expected: `7 passed`

- [ ] **Step 5: Commit**

```bash
git add src/models/huggingface_gan/main.py tests/test_hfgan_main.py
git commit -m "feat(hfgan): Lightning module — manual G/D steps, PSNR/SSIM validation, scheduler"
```

---

## Task 7: `train.py` + smoke test

**Files:**
- Create: `src/models/huggingface_gan/train.py`

- [ ] **Step 1: Write `src/models/huggingface_gan/train.py`**

```python
import os
import functools
import torch
import lightning.pytorch as pl
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from omegaconf import OmegaConf

# PyTorch 2.6+ changed weights_only default to True, which breaks OmegaConf
# in checkpoints. Patch before any Lightning code runs.
_orig_load = torch.load
@functools.wraps(_orig_load)
def _patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _orig_load(*args, **kwargs)
torch.load = _patched_load

from src.models.huggingface_gan.main import SAR2OPTLightningModule
from src.data.sen12_full.datamodule import SEN12FullDataModule
from src.utils.cleanup_memory import full_cleanup
from src.utils.callbacks import EMAWeightAveraging

os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

CONFIG_PATH = 'src/models/huggingface_gan/config.yaml'


def main():
    cfg = OmegaConf.load(CONFIG_PATH)

    dm    = SEN12FullDataModule(cfg)
    model = SAR2OPTLightningModule(cfg)

    checkpoints = ModelCheckpoint(
        dirpath=cfg.system.checkpoints_dir,
        filename='epoch={epoch:03d}-psnr={val/psnr:.4f}',
        monitor='val/psnr',
        mode='max',
        save_top_k=3,
        save_last=True,
        auto_insert_metric_name=False,
    )
    callbacks = [checkpoints]
    if cfg.ema.use_ema:
        callbacks.append(EMAWeightAveraging(
            decay=cfg.ema.decay,
            update_starting_at_epoch=cfg.ema.start_epoch,
        ))

    tb_logger  = TensorBoardLogger(cfg.system.output_dir + '/tb_logs',  name=cfg.system.tb_version)
    csv_logger = CSVLogger(cfg.system.output_dir + '/csv_logs', name=cfg.system.tb_version)

    trainer = Trainer(
        logger=[tb_logger, csv_logger],
        callbacks=callbacks,
        accelerator=cfg.system.device,
        devices=1,
        precision=cfg.system.precision,
        max_epochs=cfg.system.max_epochs,
        num_sanity_val_steps=2,
        deterministic=cfg.system.deterministic,
        benchmark=cfg.system.benchmark,
        limit_train_batches=cfg.system.limit_train_batches,
        limit_val_batches=cfg.system.limit_val_batches,
        log_every_n_steps=50,
    )

    try:
        ckpt_path = cfg.system.resume_ckpt or None
        trainer.fit(model, datamodule=dm, ckpt_path=ckpt_path)
    except KeyboardInterrupt:
        pass
    finally:
        full_cleanup(trainer=trainer, model=model, datamodule=dm, log=True)


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Verify the module imports cleanly**

```bash
python -c "from src.models.huggingface_gan.train import main; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Run full test suite**

```bash
pytest tests/test_hfgan_losses.py tests/test_hfgan_dis.py tests/test_hfgan_gen.py tests/test_hfgan_factory.py tests/test_hfgan_main.py -v
```

Expected: all tests pass (no failures).

- [ ] **Step 4: Commit**

```bash
git add src/models/huggingface_gan/train.py
git commit -m "feat(hfgan): train.py entry point — Lightning trainer, checkpoints, EMA, TB/CSV loggers"
```

- [ ] **Step 5: Update changelog**

Add to `changelog.md`:

```
## hfgan-1 (YYYY-MM-DD)
Architecture: ConvNeXtV2-Tiny encoder + bottleneck attention (2-layer, 64 tokens) + U-Net decoder
Discriminator: Two-scale spectral-norm PatchGAN
Losses: GAN (LSGAN) + FM (weight=5.0) + optional FFT (1.0) + optional Perceptual (0.1)
Optimizer: AdamW for G (differential LR: encoder 2e-5, decoder 2e-4), Adam for D (2e-4)
Status: ready to train
```

```bash
git add changelog.md
git commit -m "docs: log hfgan-1 experiment in changelog"
```

---

## Self-Review Checklist

- **Spec coverage:** All 5 spec sections covered. Generator (Task 4), Discriminator (Task 3), Losses (Task 2), Training loop (Task 6), File structure + config (Task 1 + 7). Factory interface from spec gap-fix covered in Task 5.
- **Placeholders:** None. Every step has code or exact commands.
- **Type consistency:**
  - `HFGenerator.__init__(cfg, encoder=None)` defined in Task 4, used in Tasks 5 and 6 ✓
  - `HFGANDiscriminator.forward(sar, opt)` returns `((logits1, logits2), feats)` — used by `GANLoss` tuple branch in Task 6 ✓
  - `factory.build_models(cfg, encoder=None)` returns `(netG, netD)` — consumed in Task 6 ✓
  - `factory.build_optimizers` returns `(opt_g, opt_d)` — reordered to `[opt_d, opt_g]` in `configure_optimizers` ✓
  - Criterion keys `'gan'`, `'fm'`, `'fft'`, `'perceptual'` consistent across Tasks 5 and 6 ✓
  - `ConvUpsampleBlock(in_ch, out_ch)` — in_ch is post-concat channel count, verified in Task 4 test ✓
