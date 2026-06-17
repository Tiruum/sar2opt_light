# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Active development scope

`src/models/wavenext` is the single canonical model module (WaveNeXt). The full
experimental lineage (cfrwd, huggingface_gan, llwt, llwt_v3/v4/v45, llwt_v45_base,
sar2opt_v1, sarformer_wb, pix2pix) was removed during consolidation and lives in git
tag `archive/full-lineage-v1` — restore from there to reproduce the diploma ablation
table. `src/models/wavenext/ARCHITECTURE.md` is the authoritative architecture reference (RU).

Capacity (tiny/base) and the HF-D novelty are pure config switches — see Configuration below.

## Commands

```powershell
# Install dependencies
python -m venv .venv && .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Train (must be run from repo root; config.yaml = base backbone + HF-D)
python -m src.models.wavenext.train

# One-step smoke (tiny/fast, uses config_smoke.yaml)
python -m src.models.wavenext.smoke_train_step

# Monitor training
tensorboard --logdir output/llwt_v45/tb_logs

# Quick-look inference (loads the base checkpoint via load_generator)
python -m src.models.wavenext.inference

# Export the generator for Hugging Face (ckpt -> safetensors + config.json)
python -m src.models.wavenext.export_hf --ckpt <path/to.ckpt> --out output/hf_export

# Generator architecture smoke (no data; mock encoder)
python -m src.models.wavenext.gen
```

## Configuration

**`src/models/wavenext/config.yaml` must exist before running anything.** It is hardcoded in `train.py`/`main.py`/`factory.py`; there is no CLI override.

Capacity and the HF-D novelty are pure config switches:
- `model.gen.backbone` — `facebook/convnextv2-base-22k-224` (default, ~88M, set `data.batch_size: 6`) or `...-tiny-22k-224` (~29M, `batch_size: 8`). `gen.py` derives stage channels from the backbone, so no code change is needed to switch capacity.
- `loss.hfd_weight` + `model.dis.highfreq.enabled` — the high-frequency discriminator (HF-D, the thesis novelty). Set `hfd_weight: 0` to disable it (reproduces the plain baseline / ablation arm).
- `system.tb_version` — directory name for all outputs; change per experiment.
- `system.weights_ckpt` — warm-start G/D (strict=False); `system.resume_ckpt` — full resume, or `null` to start fresh.

## Architecture

Full detail with `file:line` maps is in `src/models/wavenext/ARCHITECTURE.md`. Summary:

### Generator: `WaveNeXtGenerator` (`src/models/wavenext/gen.py`)

Single branch (no fusion weight): `SARAdapter` (raw + log-domain Lee despeckle + Sobel gradient → 3ch) → 2-level Haar stem (replaces the ConvNeXt patch-embed) → **ConvNeXt V2 backbone** (tiny or base, channels auto-derived from the HF `AutoBackbone`) → PixelShuffle U-Net decoder with skip-concats → 1-level **Inverse-Haar head** (zero-init, predicts subbands) → `tanh`. `HaarDown`/Haar weights use `register_buffer` (fixed, not trained). `forward(x, return_internals=True)` returns `(ŷ, subbands, raw_feat)` for visualization.

### Discriminator: `WaveNeXtDiscriminator` (`src/models/wavenext/dis.py`)

Holds several heads; the winning config runs exactly two:
- **`MainDis`** — two-scale conditional PatchGAN (pix2pixHD-style, coarse + fine), **no spectral norm**. Input `cat([InstanceNorm(SAR), optical])`. Returns `((coarse, fine), feats)`; feats feed Feature Matching.
- **`HighFreqDis` (HF-D, the novelty)** — spectral-norm PatchGAN judging the high-frequency residual `h(x) = x − gaussian_blur(x, σ=2)`. Conditional, LSGAN, **train-only** (not in the main `forward`; called in `training_step`). Gated by `model.dis.highfreq.enabled` AND `loss.hfd_weight > 0`; additive and ablation-clean (λ=0 ≡ baseline).

`SubbandDis`/`FourierDis`/`DeformationAligner` exist but are disabled (logit explosion / no gain).

### Training loop & losses (`main.py`, `losses.py`)

Manual optimization (`automatic_optimization = False`); `configure_optimizers → [opt_d, opt_g]`. D steps on detached fakes, then G steps on a fresh forward. EMA (decay 0.999, from epoch 20; final eval uses EMA), bf16/channels-last, 200 epochs, linear-decay LR tail. G loss is additive (criterion built only when weight > 0): LSGAN(1) + FM(10) + **HF-D(1)** + MS-SSIM(1) + per-band Haar L1(2) + LPIPS-AlexNet(2) + FFL(10) + PatchNCE(0.1). **No pixel L1** — L1 lives only in the wavelet basis.

### Data (`src/data/sen12_full`, `src/data/sen12_full_align`)

`data.dataset` selects the datamodule: `sen12_full` (raw) or `sen12_full_align` (gradient-domain ECC-aligned mirror; SAR byte-identical, optical warped). `transforms.py` holds the albumentations pipelines (synchronized geometric aug via `additional_targets={'optical': 'image'}`).

## Artifacts

All outputs are keyed by `cfg.system.tb_version`. The path prefix is `llwt_v45` (legacy,
from the warm-start lineage — see `system.*_dir` in config):
- Checkpoints: `checkpoints/llwt_v45/<tb_version>/` — top-k by `val/psnr` + `last.ckpt`
- TensorBoard: `output/llwt_v45/tb_logs/<tb_version>/`
- Epoch images: `output/llwt_v45/images/<tb_version>/`
- Profiler / summary: `output/llwt_v45/profiler/`, `output/llwt_v45/summary/`

## Experiment tracking

Log every experiment in `changelog.md` with the run ID, changes, and results. The version string convention is `vX.Y.Z` (major.minor.patch).

## Utilities

- `src/utils/logger.py`: Custom colorized console logger; reads `system.debug` from config. `logger.debug(..., once=True)` suppresses repeated identical calls. Module-level `Logger` instances in `gen.py` fire at import time.
- `src/utils/notification.py`: Telegram notifications via `.env` keys `TELEGRAM_BOT_TOKEN` / `TELEGRAM_RECIEVER_USER_ID`. Bot instance is a singleton. Silent if keys are absent.
- `src/utils/callbacks.py`: `EMAWeightAveraging` wraps Lightning's `WeightAveraging` with step- and epoch-based update gating.
- `src/utils/cleanup_memory.py`: `full_cleanup()` should be called in the `finally` block to release CUDA memory and stop dataloader workers.
- `src/utils/nsst_torch.py` + `NSST.py`: shearlet-experiment scaffolding kept for possible future work. Their lazy import of `SpeckleAwareModule`/`CBAM` from the archived `cfrwd` is dangling — restore those from tag `archive/full-lineage-v1` before use.

<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **sar2opt_light** (856 symbols, 1911 relationships, 68 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

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
