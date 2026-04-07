# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Active development scope

Only `src/models/cfrwd` is under active development. `src/models/pix2pix` is a legacy artifact kept for reproducibility — do not add features there.

## Commands

```powershell
# Install dependencies
python -m venv .venv && .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Train CFRWD (must be run from repo root)
python -m src.models.cfrwd.train

# Monitor training
tensorboard --logdir output/cfrwd/tb_logs

# Clean/aggregate CSV logs
python src/utils/clean_csv_logs.py

# Quick generator architecture test (no data needed)
python src/models/cfrwd/gen.py

# Quick discriminator test
python src/models/cfrwd/discriminator.py
```

## Configuration

**`src/models/cfrwd/config.yaml` must exist before running anything.** This path is hardcoded in `train.py`, `main.py`, `factory.py`, `logger.py`, and `clean_csv_logs.py`. There is no CLI argument override.

`factory.py` loads the config via `@lru_cache`, so it is read once per process. Changing it mid-run has no effect.

Key config fields to know:
- `system.tb_version` — version string used as directory name for all outputs (TensorBoard, CSV, checkpoints, images). Change this for each new experiment.
- `system.resume_ckpt` — set to a checkpoint path to resume training; `null` to start fresh.
- `system.compile` — enables `torch.compile()` (first epoch slower).
- `system.image_freq` — save visualization grid every N epochs; `0` disables.
- `system.debug` — enables verbose `logger.debug()` output from model internals.
- `scheduler.linear_decay_epochs` — how many final epochs to linearly decay LR to `scheduler.eta_min`.
- `ema.use_ema` — enable EMA weight averaging via `EMAWeightAveraging` callback.

Reference the pix2pix `config.yaml` at `src/models/pix2pix/config.yaml` for the expected schema structure.

## Architecture

### Generator: `CFRWDGenerator` (`src/models/cfrwd/gen.py`)

Two parallel branches fused by a learnable scalar `fusion_weight` (initialized to 1.0):

```
output = tanh(cfr_out_logits + fusion_weight * hfcf_out_logits)
```

**CFRBranch** — spatial reasoning:
- Encoder: 4× `EncoderBlock` (Conv + InstanceNorm + LeakyReLU) → 64ch feature map
- `CFRBlock`: HRNet-style multi-scale cross-fusion over 3 stages
  - Channels distributed by resolution: c1=16 (full), c2=32 (½), c3=64 (¼), c4=128 (⅛)
  - Final fusion upsamples all branches to full resolution before concat
- Decoder: `DecoderBlock` (64→32) + `FinalDecoderBlock` (32→3, 7×7 conv, no activation)

**HFCFBranch** — high-frequency wavelet branch:
- `DWTBlock`: two-level Haar DWT; **keeps LL2 as g1** (low-low subband at level 2), uses LH/HL/HH detail subbands at both levels → g1=LL2 (64×64), g2=level-2 details (64×64), g3=level-1 details (128×128)
- Three independent streams: Low (g1/LL2 via `WDResBlock×2`), Mid (g2 via `WDResBlock×6`), High (g3 via `RedBlock×2`); CBAM on each
- Streams concatenated, decoded via bilinear upsampling back to 256×256
- `FinalDecoderBlock` (32→3, 7×7 conv, no activation)

`HaarDown` uses `register_buffer` — its weights are fixed (not trained). Do not call `_initialize_weights` on it.

`forward(x, return_branches=False, hfcf_out_only=False)`:
- Default: returns `(out, fusion_weights)`
- `hfcf_out_only=True`: returns `(out, hfcf_out, fusion_weights)` — training path (avoids computing cfr_out)
- `return_branches=True`: returns `(out, cfr_out, hfcf_out, fusion_weights)` — visualization path

### Discriminator: `CFRWDPatchDis` (`src/models/cfrwd/discriminator.py`)

Two-scale conditional PatchGAN. Both branches (`CFRWDPatchDisBranch`) are identical 5-layer spectral-norm PatchGAN. Input is concatenated `[SAR, optical]` pair — `in_channels` in config must equal `sar_channels + 3`.

Returns `(outputs, features)` where `outputs = (large_logits, small_logits)` and `features` is the list of intermediate LeakyReLU activations from both branches combined (used by `FeatureMatchingLoss`).

### Training loop: `SAR2OPTGANLightningModule` (`src/models/cfrwd/main.py`)

Manual optimization (`automatic_optimization = False`). Optimizer order from `configure_optimizers`: `[optD, optG]`, unpacked as `opt_d, opt_g = self.optimizers()`.

D update uses `torch.no_grad()` on the generator forward to avoid building the G computation graph. G update runs a fresh forward pass with gradients.

`on_train_epoch_end` steps both LR schedulers and optionally saves a visualization grid + sends Telegram notification.

`setup(stage)` grabs one batch from the train dataloader to hold as `fixed_sar`/`fixed_opt` for consistent epoch-end visualizations.

### Losses (`src/models/cfrwd/losses.py`)

- `GANLoss`: LSGAN (MSELoss) by default. Supports label smoothing via `real_label_smooth`/`fake_label_smooth` args.
- `FeatureMatchingLoss`: mean L1 across all discriminator intermediate feature layers.
- `L1Loss`: standard pixel-wise L1.

Loss weights are in config under `loss.gan_weight`, `loss.fm_weight`, `loss.l1_weight`.

### Data: `SEN12` dataset (`src/data/sen12/dataset.py`)

Pairs SAR (`s1/`) and optical (`s2/`) images by filename substitution (`_s1_` → `_s2_`). Supports `sar_channels=1` (grayscale) or `sar_channels=3` (color). Uses albumentations transforms; `common_transform` must use `additional_targets={'optical': 'image'}` for synchronized geometric augmentations.

`SEN12Datamodule` performs a single filesystem scan and passes `classes`/`items` directly to train/val dataset constructors to avoid redundant scans. Sets `drop_last=True` on the train loader (required for stable BatchNorm if used). Automatically disables `persistent_workers` when `num_workers=0`.

Expected data layout:
```
data/sen12/<class>/s1/<file>
data/sen12/<class>/s2/<file>
```

## Artifacts

All outputs are keyed by `cfg.system.tb_version`:
- Checkpoints: `checkpoints/cfrwd/<tb_version>/` — top-3 by `val/psnr` + `last.ckpt`
- TensorBoard: `output/cfrwd/tb_logs/<tb_version>/`
- CSV logs: `output/cfrwd/csv_logs/<tb_version>/`
- Epoch images: `cfg.system.images_dir/<tb_version>/epoch_N.png`
- Profiler: `cfg.system.profiler_dir/<tb_version>.txt`
- Model summary: `cfg.system.summary_dir/<tb_version>.txt` (if `model.log_summary: true`)

## Experiment tracking

Log every experiment in `changelog.md` with the run ID, changes, and results. The version string convention is `vX.Y.Z` (major.minor.patch).

## Utilities

- `src/utils/logger.py`: Custom colorized console logger; reads `system.debug` from config. `logger.debug(..., once=True)` suppresses repeated identical calls. Module-level `Logger` instances in `gen.py` fire at import time.
- `src/utils/notification.py`: Telegram notifications via `.env` keys `TELEGRAM_BOT_TOKEN` / `TELEGRAM_RECIEVER_USER_ID`. Bot instance is a singleton. Silent if keys are absent.
- `src/utils/callbacks.py`: `EMAWeightAveraging` wraps Lightning's `WeightAveraging` with step- and epoch-based update gating.
- `src/utils/cleanup_memory.py`: `full_cleanup()` should be called in the `finally` block to release CUDA memory and stop dataloader workers.
