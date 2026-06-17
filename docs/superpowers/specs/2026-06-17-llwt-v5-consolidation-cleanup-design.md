# Design: llwt_v5 consolidation + repository cleanup

**Date:** 2026-06-17
**Branch:** `physics-aware` (work directly here; no extra branch)
**Status:** Approved (design), pending spec review

## Goal

Reduce the repository to a single canonical model module (`src/models/llwt_v5`)
that is:

1. **Capacity-switchable** — tiny vs base ConvNeXt V2 backbone via config (default **base**).
2. **HF-D toggleable** — high-frequency discriminator on/off via `loss.hfd_weight`
   (and `model.dis.highfreq.enabled`); default **on** (best architecture).
3. **Clean** — dead model lineages, research scaffolding, and top-level junk removed
   from the working tree (preserved in a git archive tag).
4. **Hugging Face ready (light)** — inference path self-contained and decoupled from
   Lightning/training config, so a later HF push is low-friction. Full HF packaging is
   a separate, deferred task.

## Key findings (from investigation)

- `llwt_v5/gen.py` and `llwt_v5/blocks.py` are **byte-identical** to the
  `llwt_v45_base` copies except for the import path. The generator already derives
  stage channels from the HuggingFace `AutoBackbone`, so **base is purely a config
  switch** — no generator code change needed.
- `llwt_v5` is a **feature superset** of `llwt_v45_base` (it adds HF-D, FFS, aligner;
  all but HF-D disabled). So `llwt_v45_base` is redundant after consolidation.
- HF-D is already a clean additive toggle: gated by **both** `model.dis.highfreq.enabled`
  and `loss.hfd_weight > 0`. `hfd_weight: 0` reproduces the plain baseline exactly.
- Base checkpoint on disk (gitignored, survives cleanup):
  `checkpoints/llwt_v45_base/llwt-v0.4.6-base/epoch=199-psnr=18.5361.ckpt`.
- **Tests depend on dead dirs** — `tests/test_hfgan_*`, `tests/test_llwt_*`,
  `tests/test_sarformer_*`, `tests/models/cfrwd/*` import modules slated for deletion.
  These tests must be removed too, or test collection breaks.
- `physics-aware` is **112 commits ahead of `master`** and has a **dirty working tree**:
  modified diploma `.tex/.pdf` (the user's defended-thesis edits) + an already-deleted
  bundled code copy under `docs/diploma/src/`. Thesis edits must be preserved.
- Core llwt_v5 imports are all repo-package-absolute (`from src.models.llwt_v5.X`,
  `src.utils`, `src.data`). Fine for HF route-1; relative imports needed only for
  HF route-2 (`from_pretrained` custom code).

## Phase A — Pre-cleanup hygiene

1. Commit pending diploma edits as their own commit so thesis work is preserved and
   separated from cleanup: `docs: thesis edits + drop bundled src copy`.
   - Scope this commit to `docs/diploma/**` only (targeted `git add`).
2. Create archive tag on that commit capturing the **full lineage** (all 11 model dirs,
   research scaffolding, all tests): `git tag archive/full-lineage-v1`.
   - Tag message records what it preserves and why.
3. No new branch — continue on `physics-aware`.

## Phase B — Consolidate llwt_v5 (config + redundant-dir removal)

1. `src/models/llwt_v5/config.yaml` edits for **base default**:
   - `model.gen.backbone` → `facebook/convnextv2-base-22k-224`
   - `data.batch_size` → `6` (base VRAM; tiny used 8)
   - `system.weights_ckpt` → base checkpoint path (warm-start the base generator)
   - `system.tb_version` → new string (e.g. `llwt-v0.6.0-base-hfd`)
   - HF-D left **on** (`model.dis.highfreq.enabled: true`, `loss.hfd_weight: 1.0`)
2. Add in-file switch comments documenting both capacities and the HF-D off path:
   - tiny: `backbone=...-tiny-22k-224`, `batch_size=8`
   - base: `backbone=...-base-22k-224`, `batch_size=6`
   - HF-D off (ablation/baseline): `loss.hfd_weight: 0`
3. Delete `src/models/llwt_v45_base/` (redundant — base is now a config switch).

## Phase C — Deletions (working tree; archive tag retains everything)

- Dead model dirs: `cfrwd, huggingface_gan, llwt, llwt_v3, llwt_v4, llwt_v45,
  llwt_v45_base, sar2opt_v1, sarformer_wb`, and legacy `pix2pix`.
- Research scaffolding in `llwt_v5/`: `proof_*.py, diag_*.py, overfit_*.py,
  compare_v45_v5.py, thesis_hfd_section.tex, _*.log, img_to_find/,
  config_smoke_off.yaml`. (Keep `viz_*.py`, **and keep `smoke_train_step.py` +
  `config_smoke.yaml`** as the ongoing smoke-test harness.)
- Tests for deleted modules: `tests/test_hfgan_*, tests/test_llwt_*,
  tests/test_sarformer_*, tests/models/cfrwd/`.
- Top-level junk: `dynamo_log.txt, diagnose_g.py, LLWT_V4_REPORT.md, _ablation_eval.py,
  _hfd_ablation_full.py, _hfd_full.log, output_channels/`.
- `docs/diploma/src/` bundled copy (already deleted in working tree — finalize).

## Phase D — Keep

- `src/models/llwt_v5/` core (`gen, blocks, dis, main, factory, losses, patchnce,
  align, inference, best_inference, eval_full, train, config.yaml, ARCHITECTURE.md,
  viz_*.py`).
- `src/data/{sen12_full, sen12_full_align, transforms.py}` (datasets the model uses),
  `src/utils/*`, `scripts/`.
- `README.md, CLAUDE.md, AGENTS.md, requirements.txt, pytest.ini, changelog.md,
  docs/diploma/, docs/presentation/`.
- Update `CLAUDE.md`: active scope `cfrwd` → `llwt_v5`; refresh architecture/commands
  sections to match the surviving module.

## Phase E — HF-readiness (bake in now; full push deferred)

1. **F1** — Ensure `inference.py` does not import any deleted scaffolding. Factor a
   `load_generator(ckpt_path) -> nn.Module` helper that:
   - strips the `netG.` prefix, selects EMA-or-live weights deterministically,
   - builds the generator from gen-only config keys (no data/training keys required).
2. **F2** — Add `src/models/llwt_v5/export_hf.py`: Lightning `.ckpt` →
   `generator.safetensors` + minimal `config.json` (generator hyperparameters only).
   Decouples shippable weights from the training stack.
3. **F3** — Confirm core imports stay clean. Document (do **not** force now) that HF
   route-2 (`from_pretrained` + `trust_remote_code`) would need relative imports
   (`from .blocks import …`) and a `PreTrainedModel`/config wrapper.

### Deferred to a separate "HF push" task

- Model card (`README.md` for the HF repo) — source from `ARCHITECTURE.md` + metric tables.
- `LICENSE` (account for ConvNeXt V2 backbone + SEN12 dataset terms).
- `huggingface_hub` upload script; optional Gradio Space demo.
- Optional `from_pretrained` custom-architecture packaging (route-2).

## Phase F — Verification

- `python -m src.models.llwt_v5.gen` (generator smoke, base backbone).
- **Smoke train** — `python -m src.models.llwt_v5.smoke_train_step` (or `train.py` with
  `config_smoke.yaml`): one full D+G step on base backbone with HF-D on, no crash,
  finite losses.
- **Smoke inference** — `python -m src.models.llwt_v5.inference` loads the base
  checkpoint and produces output without error.
- `pytest` — surviving tests green, collection has no import errors from deleted dirs.
- `git status` clean and scoped to intended deletions; archive tag resolves.

## Out of scope

- 278 GB `checkpoints/` and 5.3 GB `output/` are gitignored (local disk, not the repo) —
  not touched by this cleanup. Optional separate disk cleanup if requested.
- Local non-repo dotfiles (`.venv_wsl`, `.history`, `.continue`, `.qwen`, etc.) are
  gitignored — not part of repo cleanup.
- No generator/architecture code changes; consolidation is config-only plus deletions.

## Risks

- Deleting test files alongside model dirs: mitigated by auditing `tests/` imports
  before deletion (Phase F catches stragglers).
- Base + HF-D is an untrained combination: inference with the existing base checkpoint
  works (HF-D is train-only); fresh base+HF-D training is a future run, not part of cleanup.
- Dirty working tree mixing thesis edits with cleanup: mitigated by the scoped Phase A commit.
