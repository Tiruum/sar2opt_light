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
- `DWTBlock`: two-level Haar DWT; discards LL, uses LH/HL/HH detail subbands at both levels → g2 (64×64), g3 (128×128)
- Two processing streams: top (`WDResBlock` bottlenecks on g2+g3 fused) and bottom (`RedBlock` on g3)
- Streams merged, decoded via bilinear upsampling back to 256×256
- `FinalDecoderBlock` (32→3, 7×7 conv, no activation)

`HaarDown` uses `register_buffer` — its weights are fixed (not trained). Do not call `_initialize_weights` on it.

`forward(x, return_branches=True)` returns `(fused_out, cfr_tanh, hfcf_tanh)` for visualization.

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

<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **sar2opt_light** (698 symbols, 1704 relationships, 49 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

> If any GitNexus tool warns the index is stale, run `npx gitnexus analyze` in terminal first.

## Always Do

- **MUST run impact analysis before editing any symbol.** Before modifying a function, class, or method, run `gitnexus_impact({target: "symbolName", direction: "upstream"})` and report the blast radius (direct callers, affected processes, risk level) to the user.
- **MUST run `gitnexus_detect_changes()` before committing** to verify your changes only affect expected symbols and execution flows.
- **MUST warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding with edits.
- When exploring unfamiliar code, use `gitnexus_query({query: "concept"})` to find execution flows instead of grepping. It returns process-grouped results ranked by relevance.
- When you need full context on a specific symbol — callers, callees, which execution flows it participates in — use `gitnexus_context({name: "symbolName"})`.

## When Debugging

1. `gitnexus_query({query: "<error or symptom>"})` — find execution flows related to the issue
2. `gitnexus_context({name: "<suspect function>"})` — see all callers, callees, and process participation
3. `READ gitnexus://repo/sar2opt_light/process/{processName}` — trace the full execution flow step by step
4. For regressions: `gitnexus_detect_changes({scope: "compare", base_ref: "main"})` — see what your branch changed

## When Refactoring

- **Renaming**: MUST use `gitnexus_rename({symbol_name: "old", new_name: "new", dry_run: true})` first. Review the preview — graph edits are safe, text_search edits need manual review. Then run with `dry_run: false`.
- **Extracting/Splitting**: MUST run `gitnexus_context({name: "target"})` to see all incoming/outgoing refs, then `gitnexus_impact({target: "target", direction: "upstream"})` to find all external callers before moving code.
- After any refactor: run `gitnexus_detect_changes({scope: "all"})` to verify only expected files changed.

## Never Do

- NEVER edit a function, class, or method without first running `gitnexus_impact` on it.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis.
- NEVER rename symbols with find-and-replace — use `gitnexus_rename` which understands the call graph.
- NEVER commit changes without running `gitnexus_detect_changes()` to check affected scope.

## Tools Quick Reference

| Tool | When to use | Command |
|------|-------------|---------|
| `query` | Find code by concept | `gitnexus_query({query: "auth validation"})` |
| `context` | 360-degree view of one symbol | `gitnexus_context({name: "validateUser"})` |
| `impact` | Blast radius before editing | `gitnexus_impact({target: "X", direction: "upstream"})` |
| `detect_changes` | Pre-commit scope check | `gitnexus_detect_changes({scope: "staged"})` |
| `rename` | Safe multi-file rename | `gitnexus_rename({symbol_name: "old", new_name: "new", dry_run: true})` |
| `cypher` | Custom graph queries | `gitnexus_cypher({query: "MATCH ..."})` |

## Impact Risk Levels

| Depth | Meaning | Action |
|-------|---------|--------|
| d=1 | WILL BREAK — direct callers/importers | MUST update these |
| d=2 | LIKELY AFFECTED — indirect deps | Should test |
| d=3 | MAY NEED TESTING — transitive | Test if critical path |

## Resources

| Resource | Use for |
|----------|---------|
| `gitnexus://repo/sar2opt_light/context` | Codebase overview, check index freshness |
| `gitnexus://repo/sar2opt_light/clusters` | All functional areas |
| `gitnexus://repo/sar2opt_light/processes` | All execution flows |
| `gitnexus://repo/sar2opt_light/process/{name}` | Step-by-step execution trace |

## Self-Check Before Finishing

Before completing any code modification task, verify:
1. `gitnexus_impact` was run for all modified symbols
2. No HIGH/CRITICAL risk warnings were ignored
3. `gitnexus_detect_changes()` confirms changes match expected scope
4. All d=1 (WILL BREAK) dependents were updated

## Keeping the Index Fresh

After committing code changes, the GitNexus index becomes stale. Re-run analyze to update it:

```bash
npx gitnexus analyze
```

If the index previously included embeddings, preserve them by adding `--embeddings`:

```bash
npx gitnexus analyze --embeddings
```

To check whether embeddings exist, inspect `.gitnexus/meta.json` — the `stats.embeddings` field shows the count (0 means no embeddings). **Running analyze without `--embeddings` will delete any previously generated embeddings.**

> Claude Code users: A PostToolUse hook handles this automatically after `git commit` and `git merge`.

## CLI

| Task | Read this skill file |
|------|---------------------|
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus/gitnexus-exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus/gitnexus-debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus/gitnexus-refactoring/SKILL.md` |
| Tools, resources, schema reference | `.claude/skills/gitnexus/gitnexus-guide/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.claude/skills/gitnexus/gitnexus-cli/SKILL.md` |

<!-- gitnexus:end -->

<!-- code-review-graph MCP tools -->
## MCP Tools: code-review-graph

**IMPORTANT: This project has a knowledge graph. ALWAYS use the
code-review-graph MCP tools BEFORE using Grep/Glob/Read to explore
the codebase.** The graph is faster, cheaper (fewer tokens), and gives
you structural context (callers, dependents, test coverage) that file
scanning cannot.

### When to use graph tools FIRST

- **Exploring code**: `semantic_search_nodes` or `query_graph` instead of Grep
- **Understanding impact**: `get_impact_radius` instead of manually tracing imports
- **Code review**: `detect_changes` + `get_review_context` instead of reading entire files
- **Finding relationships**: `query_graph` with callers_of/callees_of/imports_of/tests_for
- **Architecture questions**: `get_architecture_overview` + `list_communities`

Fall back to Grep/Glob/Read **only** when the graph doesn't cover what you need.

### Key Tools

| Tool | Use when |
|------|----------|
| `detect_changes` | Reviewing code changes — gives risk-scored analysis |
| `get_review_context` | Need source snippets for review — token-efficient |
| `get_impact_radius` | Understanding blast radius of a change |
| `get_affected_flows` | Finding which execution paths are impacted |
| `query_graph` | Tracing callers, callees, imports, tests, dependencies |
| `semantic_search_nodes` | Finding functions/classes by name or keyword |
| `get_architecture_overview` | Understanding high-level codebase structure |
| `refactor_tool` | Planning renames, finding dead code |

### Workflow

1. The graph auto-updates on file changes (via hooks).
2. Use `detect_changes` for code review.
3. Use `get_affected_flows` to understand impact.
4. Use `query_graph` pattern="tests_for" to check coverage.
