# huggingface-gan Design Spec
**Date:** 2026-04-28  
**Task:** SOTA SAR-to-optical GAN using HuggingFace building blocks  
**Location:** `src/models/huggingface-gan/`

---

## Goal

Create a SOTA-level conditional GAN for SAR-to-optical image translation (256×256, 1ch SAR → 3ch RGB) using verified HuggingFace and PyTorch building blocks. Priority: no custom block bugs — use only battle-tested components at high abstraction level.

---

## Architecture Decision

**Approach B selected:** ConvNeXtV2 encoder + PyTorch bottleneck attention + U-Net decoder.

- **Why ConvNeXtV2:** FCMAE pretraining (masked reconstruction, closer to our task than ImageNet classification), GRN normalization (inter-channel competition naturally suppresses SAR speckle-like noise), pure CNN (torch.compile friendly, no attention padding issues), `AutoBackbone` support with clean `feature_maps` API.
- **Why bottleneck attention:** SAR→optical requires hallucinating coherent global structures (roads, buildings) from noisy local radar returns. 8×8 bottleneck = 64 tokens: trivially cheap, zero risk.
- **Why not LeViT:** Classification-only architecture, reaches only 4×4 spatial resolution at deepest stage (would require 64× upsampling), no `AutoBackbone` support.
- **Why not Swin at this stage:** Higher complexity, window-padding overhead, torch.compile friction. Will be evaluated as a future alternative once this baseline is established.

---

## Section 1 — Generator (`gen.py`)

### ChannelAdapter
```python
nn.Sequential(
    nn.Conv2d(1, 3, kernel_size=3, padding=1, bias=False),
    nn.GELU(),
)
```
Projects 1ch SAR to 3ch so the pretrained ConvNeXtV2 stem receives its expected shape. No `ignore_mismatched_sizes` needed — backbone sees clean 3ch input.

### Encoder — AutoBackbone (ConvNeXtV2-Tiny, 22k-pretrained)
```python
encoder = AutoBackbone.from_pretrained(
    "facebook/convnextv2-tiny-22k-224",
    out_indices=(0, 1, 2, 3),
)
# outputs.feature_maps → 4 native (B,C,H,W) tensors:
# s0: (B,  96, 64, 64)
# s1: (B, 192, 32, 32)
# s2: (B, 384, 16, 16)
# s3: (B, 768,  8,  8)  ← bottleneck input
```
FCMAE pretrained weights kept. Trained end-to-end (no frozen layers) with differential LR.

### BottleneckAttention
```python
class BottleneckAttention(nn.Module):
    def __init__(self, dim=768, nhead=8, num_layers=2):
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=nhead,
            dim_feedforward=dim * 2,   # 1536 — conservative for small dataset
            dropout=0.0,
            batch_first=True,
            norm_first=True,           # Pre-LN: stable when training from near-scratch
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers)
        self.pos = nn.Parameter(torch.zeros(1, 64, dim))  # learned, init=0

    def forward(self, x):
        B, C, H, W = x.shape
        t = x.flatten(2).transpose(1, 2)          # (B, 64, 768)
        t = self.transformer(t + self.pos)
        return t.transpose(1, 2).reshape(B, C, H, W)
```
`nn.TransformerEncoderLayer` is PyTorch's own tested implementation. 64 tokens × 768 dim is computationally trivial.

### ConvUpsampleBlock (decoder building block)
```python
class ConvUpsampleBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_ch,  out_ch, 3, bias=False), nn.GroupNorm(8, out_ch), nn.GELU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(out_ch, out_ch, 3, bias=False), nn.GroupNorm(8, out_ch), nn.GELU(),
        )
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1, bias=False)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.conv(x) + self.shortcut(x)
```
- Two convs per block (standard UNet capacity)
- Residual shortcut (1×1) for gradient flow during from-scratch training
- ReflectionPad2d throughout (GAN best practice, prevents border artifacts)
- GroupNorm(8) — not InstanceNorm (project audit: InstanceNorm causes feature amplitude suppression)

### Decoder + Head
```python
# Stages after bottleneck (8×8):
up4 = ConvUpsampleBlock(768 + 384, 256)  # → 16×16, concat s2
up3 = ConvUpsampleBlock(256 + 192, 128)  # → 32×32, concat s1
up2 = ConvUpsampleBlock(128 +  96,  64)  # → 64×64, concat s0
up1 = ConvUpsampleBlock( 64,        32)  # → 128×128
up0 = ConvUpsampleBlock( 32,        16)  # → 256×256

head = nn.Sequential(
    nn.ReflectionPad2d(3),
    nn.Conv2d(16, 3, kernel_size=7),
    nn.Tanh(),
)
```

### Full forward
```python
def forward(self, sar):                             # (B, 1, 256, 256)
    x = self.channel_adapter(sar)                  # (B, 3, 256, 256)
    s0, s1, s2, s3 = self.encoder(pixel_values=x).feature_maps
    s3 = self.bottleneck(s3)
    x = self.up4(s3, s2)
    x = self.up3(x,  s1)
    x = self.up2(x,  s0)
    x = self.up1(x)
    x = self.up0(x)
    return self.head(x)                            # (B, 3, 256, 256)
```

**Parameter budget:** ConvNeXtV2-Tiny ~28.6M + bottleneck ~9.6M + decoder ~8M = **~46M total**.

---

## Section 2 — Discriminator (`dis.py`)

Two-scale conditional spectral-norm PatchGAN. No HF components — standard architecture, fully proven.

```python
class PatchDisBranch(nn.Module):
    """5-layer 70×70 spectral-norm PatchGAN."""
    def __init__(self, in_ch, ndf=64):
        def sn(ci, co, k, s, p):
            return spectral_norm(nn.Conv2d(ci, co, k, s, p, bias=True))

        self.layers = nn.ModuleList([
            nn.Sequential(sn(in_ch, ndf,    4,2,1), nn.LeakyReLU(0.2, True)),  # 256→128
            nn.Sequential(sn(ndf,   ndf*2,  4,2,1), nn.LeakyReLU(0.2, True)),  # 128→64
            nn.Sequential(sn(ndf*2, ndf*4,  4,2,1), nn.LeakyReLU(0.2, True)),  # 64→32
            nn.Sequential(sn(ndf*4, ndf*8,  4,1,1), nn.LeakyReLU(0.2, True)),  # 32→30
            sn(ndf*8, 1, 4, 1, 1),                                              # logits
        ])

    def forward(self, x):
        features = []
        for layer in self.layers[:-1]:
            x = layer(x)
            features.append(x)
        return self.layers[-1](x), features


class HFGANDiscriminator(nn.Module):
    def __init__(self, in_ch=4, ndf=64):
        self.branch1    = PatchDisBranch(in_ch, ndf)
        self.branch2    = PatchDisBranch(in_ch, ndf)
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)

    def forward(self, sar, opt):
        x  = torch.cat([sar, opt], dim=1)    # (B, 4, 256, 256)
        x2 = self.downsample(x)              # (B, 4, 128, 128)
        logits1, feats1 = self.branch1(x)
        logits2, feats2 = self.branch2(x2)
        return (logits1, logits2), feats1 + feats2
```

No InstanceNorm/GroupNorm alongside spectral norm — adding another norm causes training instability. SN + LeakyReLU only.

Return contract: `((logits_large, logits_small), features_list)` where `features_list` is 8 tensors (4 per branch) for FM loss.

---

## Section 3 — Losses (`losses.py`)

### GANLoss — LSGAN with label smoothing
```python
class GANLoss(nn.Module):
    def __init__(self, real_smooth=0.9, fake_smooth=0.0):
        self.criterion   = nn.MSELoss()
        self.real_smooth = real_smooth
        self.fake_smooth = fake_smooth

    def forward(self, logits, is_real: bool):
        if isinstance(logits, (list, tuple)):
            return sum(self._loss(l, is_real) for l in logits) / len(logits)
        return self._loss(logits, is_real)

    def _loss(self, logit, is_real):
        val = self.real_smooth if is_real else self.fake_smooth
        return self.criterion(logit, torch.full_like(logit, val))
```

### FeatureMatchingLoss
```python
class FeatureMatchingLoss(nn.Module):
    def forward(self, fake_feats, real_feats):
        loss = sum(F.l1_loss(f, r.detach()) for f, r in zip(fake_feats, real_feats))
        return loss / len(fake_feats)
```

### FFTLoss (optional — disable via `fft_weight: 0.0`)
```python
class FFTLoss(nn.Module):
    def forward(self, pred, target):
        pred_mag   = torch.log1p(torch.abs(torch.fft.rfft2(pred,   norm='ortho')))
        target_mag = torch.log1p(torch.abs(torch.fft.rfft2(target, norm='ortho')))
        return F.l1_loss(pred_mag, target_mag)
```

### PerceptualLoss (optional — HF-native, disable via `perceptual_weight: 0.0`)
```python
class PerceptualLoss(nn.Module):
    def __init__(self, backbone_name="facebook/convnextv2-tiny-22k-224"):
        self.backbone = AutoBackbone.from_pretrained(backbone_name, out_indices=(0, 1, 2))
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std',  torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def _norm(self, x):
        return ((x + 1) / 2 - self.mean) / self.std

    def forward(self, pred, target):
        pf = self.backbone(pixel_values=self._norm(pred)).feature_maps
        tf = self.backbone(pixel_values=self._norm(target)).feature_maps
        return sum(F.l1_loss(p, t.detach()) for p, t in zip(pf, tf)) / len(pf)
```
Uses frozen ConvNeXtV2 stages 0–2. Better than VGG: same architecture family as generator, FCMAE pretraining biased toward reconstruction. VRAM note: ~28M frozen params — disable if VRAM is tight.

---

## Section 4 — Training Loop (`main.py`)

### Optimizers
```python
def configure_optimizers(self):
    enc_params   = list(self.netG.channel_adapter.parameters()) \
                 + list(self.netG.encoder.parameters())
    fresh_params = list(self.netG.bottleneck.parameters()) \
                 + list(self.netG.up4.parameters()) \
                 + list(self.netG.up3.parameters()) \
                 + list(self.netG.up2.parameters()) \
                 + list(self.netG.up1.parameters()) \
                 + list(self.netG.up0.parameters()) \
                 + list(self.netG.head.parameters())

    opt_g = AdamW([
        {'params': enc_params,   'lr': cfg.optimizer.lr_g * 0.1},
        {'params': fresh_params, 'lr': cfg.optimizer.lr_g},
    ], betas=(0.5, 0.999), weight_decay=cfg.optimizer.weight_decay_g)

    opt_d = Adam(self.netD.parameters(),
                 lr=cfg.optimizer.lr_d, betas=(0.5, 0.999))

    return [opt_d, opt_g]
```
- AdamW for G (weight decay regularizes pretrained encoder)
- Differential LR: encoder at 0.1× to preserve pretrained init
- Adam for D (standard GAN discriminator optimizer)

### Training step — real features reused D→G
```python
def training_step(self, batch, batch_idx):
    sar, opt     = batch['sar'], batch['optical']
    opt_d, opt_g = self.optimizers()

    # D step
    with torch.no_grad():
        fake_d = self.netG(sar)
    real_logits, real_feats = self.netD(sar, opt)
    fake_logits, _          = self.netD(sar, fake_d)
    d_loss = 0.5 * (
        self.gan_loss(real_logits, is_real=True) +
        self.gan_loss(fake_logits, is_real=False)
    )
    opt_d.zero_grad(); self.manual_backward(d_loss); opt_d.step()

    # G step — reuse real_feats from D step (no extra D forward)
    fake = self.netG(sar)
    fake_logits_g, fake_feats = self.netD(sar, fake)
    real_feats_d = [f.detach() for f in real_feats]

    g_loss = (
        self.gan_loss(fake_logits_g, is_real=True) * cfg.loss.gan_weight  +
        self.fm_loss(fake_feats, real_feats_d)      * cfg.loss.fm_weight  +
        self.fft_loss(fake, opt)                    * cfg.loss.fft_weight +
        self.perc_loss(fake, opt)                   * cfg.loss.perceptual_weight
    )
    opt_g.zero_grad(); self.manual_backward(g_loss); opt_g.step()
```

### Validation
- PSNR, SSIM: stateful torchmetrics (`data_range=2.0`)
- SAM, ERGAS: functional per-batch, accumulated manually (prevent VRAM doubling)

---

## Section 5 — File Structure & Config

```
src/models/huggingface-gan/
├── config.yaml
├── gen.py
├── dis.py
├── losses.py
├── factory.py
├── main.py
└── train.py
```

### config.yaml
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
  tb_version:    "hfgan-1"
  resume_ckpt:   null
  debug:         false
  checkpoints_dir: "checkpoints/huggingface-gan"
  output_dir:      "./output/huggingface-gan"
  images_dir:      "./output/huggingface-gan/images"
  profiler_dir:    "./output/huggingface-gan/profiler"
  summary_dir:     "./output/huggingface-gan/summary"
```

---

## factory.py Interface

`factory.py` is the single wiring point — `main.py` calls nothing else at construction time.

```python
def build_models(cfg) -> tuple[HFGenerator, HFGANDiscriminator]:
    netG = HFGenerator(cfg)
    netD = HFGANDiscriminator(in_ch=cfg.model.dis.in_channels, ndf=cfg.model.dis.ndf)
    return netG, netD

def build_criterions(cfg) -> dict[str, nn.Module]:
    losses = {
        'gan': GANLoss(),
        'fm':  FeatureMatchingLoss(),
    }
    if cfg.loss.fft_weight > 0:
        losses['fft'] = FFTLoss()
    if cfg.loss.perceptual_weight > 0:
        losses['perceptual'] = PerceptualLoss(cfg.model.gen.backbone)
    return losses

def build_optimizers(cfg, netG, netD) -> tuple[AdamW, Adam]: ...
def build_lr_schedulers(cfg, opt_g, opt_d) -> tuple[LRScheduler, LRScheduler]: ...
```

Conditional instantiation is critical: `PerceptualLoss` loads a frozen 28M-param backbone — it must not be created when `perceptual_weight: 0.0`. Same for `FFTLoss` (cheaper, but keeps the dict clean).

---

## Key Constraints & Gotchas

1. **GroupNorm, not InstanceNorm** — project audit documented InstanceNorm causes feature amplitude suppression in this codebase.
2. **`pixel_values=x`** — AutoBackbone forward uses keyword arg `pixel_values`, not positional.
3. **Conditional loss instantiation** — `FFTLoss` and `PerceptualLoss` must only be created in `factory.py` when their config weight is `> 0`. Multiplying by zero is not enough — `PerceptualLoss` loads a 28M frozen backbone even if its output is zeroed.
4. **PerceptualLoss renormalization** — input is [-1,1]; backbone expects ImageNet normalization. The `_norm()` method handles this.
5. **Differential LR** — encoder at 0.1× base LR. Do not use a single flat LR for the generator or pretrained weights drift immediately.
6. **bias=False with norm layers** — throughout generator (norm absorbs bias).
7. **No norm in discriminator** — spectral norm + LeakyReLU only; additional norm causes instability.
8. **Real feats reuse** — `real_feats` captured in D step, detached and reused in G step. Avoids a redundant D forward pass on real images.
