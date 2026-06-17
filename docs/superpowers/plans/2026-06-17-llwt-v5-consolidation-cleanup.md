# llwt_v5 Consolidation + Repository Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce the repo to one canonical model module (`src/models/llwt_v5`) — base-default backbone switch, HF-D toggle, dead lineage removed (archive-tagged), inference HF-push-ready.

**Architecture:** No generator code changes. Base is a config switch (generator derives channels from the HuggingFace `AutoBackbone`). HF-D is an additive toggle (`loss.hfd_weight`). Everything dead is preserved in a git tag, then removed from the working tree. Inference is decoupled from Lightning/training config for later HF export.

**Tech Stack:** PyTorch, PyTorch Lightning, HuggingFace `transformers` (ConvNeXt V2 `AutoBackbone`), OmegaConf, safetensors, pytest.

## Global Constraints

- Work directly on branch `physics-aware`. No new branch.
- All commands run from repo root. The model entrypoints are package modules
  (`python -m src.models.llwt_v5.<x>`), not file paths.
- `src/models/llwt_v5/config.yaml` is hardcoded in `train.py`/`main.py`/`factory.py`;
  there is no CLI config override.
- Do NOT touch `docs/diploma/**` thesis content beyond Task 0 (the user's defended work).
- gitignored and out of scope: `checkpoints/`, `output/`, `data/`, all dotfiles.
- Base checkpoint (on disk, survives): `checkpoints/llwt_v45_base/llwt-v0.4.6-base/epoch=199-psnr=18.5361.ckpt`.
- Commit message footer on every commit:
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`

---

## File Structure

**Modified:**
- `src/models/llwt_v5/config.yaml` — base-default backbone/batch/warm-start/version + switch comments.
- `src/models/llwt_v5/inference.py` — extract a Lightning-decoupled `load_generator()`.
- `CLAUDE.md` — active scope `cfrwd` → `llwt_v5`; refresh commands/architecture.

**Created:**
- `src/models/llwt_v5/export_hf.py` — `.ckpt` → `generator.safetensors` + `config.json`.

**Deleted (archive tag retains):**
- 10 model dirs: `cfrwd, huggingface_gan, llwt, llwt_v3, llwt_v4, llwt_v45, llwt_v45_base, sar2opt_v1, sarformer_wb, pix2pix`.
- llwt_v5 scaffolding: `proof_*.py, diag_*.py, overfit_*.py, compare_v45_v5.py, thesis_hfd_section.tex, _*.log, img_to_find/, config_smoke_off.yaml`.
- Tests: `tests/test_hfgan_*, tests/test_llwt_*, tests/test_sarformer_*, tests/models/cfrwd/`.
- Top-level: `dynamo_log.txt, diagnose_g.py, LLWT_V4_REPORT.md, _ablation_eval.py, _hfd_ablation_full.py, _hfd_full.log, output_channels/`.

**Kept (do not delete):** `llwt_v5/{gen,blocks,dis,main,factory,losses,patchnce,align,inference,best_inference,eval_full,train,smoke_train_step}.py`, `config.yaml`, `config_smoke.yaml`, `ARCHITECTURE.md`, `viz_*.py`.

---

### Task 0: Pre-cleanup hygiene — preserve thesis edits + archive tag

**Files:**
- Commit: `docs/diploma/**` (pending working-tree changes)
- Tag: `archive/full-lineage-v1`

**Interfaces:**
- Produces: a clean working tree and an archive tag that all later deletions rely on for recoverability.

- [ ] **Step 1: Inspect the pending working-tree changes**

```bash
git status --short | grep -vE '^\?\?' | head -60
```
Expected: modified `docs/diploma/*.tex/.pdf/.aux/...` and deleted `docs/diploma/src/*`. Confirm all are under `docs/diploma/`.

- [ ] **Step 2: Stage and commit only the diploma changes**

```bash
git add docs/diploma
git commit -m "docs: thesis edits + drop bundled src copy

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```
Expected: commit succeeds; `git status --short | grep -vE '^\?\?'` now empty (only untracked remain).

- [ ] **Step 3: Create the archive tag on this full-lineage commit**

```bash
git tag -a archive/full-lineage-v1 -m "Full experimental lineage before consolidation: all 11 model dirs (cfrwd, hfgan, llwt v3/v4/v45/v45_base/v5, sar2opt_v1, sarformer_wb, pix2pix), research scaffolding, and full test suite. Reproduces diploma ablation table."
```

- [ ] **Step 4: Verify the tag resolves to current HEAD**

```bash
git rev-parse archive/full-lineage-v1 && git rev-parse HEAD
```
Expected: identical SHAs.

---

### Task 1: config.yaml — base default + capacity/HF-D switch comments

**Files:**
- Modify: `src/models/llwt_v5/config.yaml`

**Interfaces:**
- Produces: a config whose default is base backbone + HF-D on, warm-started from the base checkpoint. Later smoke/inference tasks load this.

- [ ] **Step 1: Read the current gen/data/system blocks**

```bash
grep -nE 'backbone|batch_size|tb_version|weights_ckpt|resume_ckpt' src/models/llwt_v5/config.yaml
```

- [ ] **Step 2: Edit the backbone line to base + add a capacity switch comment**

Set `model.gen.backbone` to base and document the tiny alternative inline:
```yaml
    backbone:         "facebook/convnextv2-base-22k-224"   # SWITCH capacity: base=...-base-22k-224 (batch 6) | tiny=...-tiny-22k-224 (batch 8). Generator derives stage channels from this.
```

- [ ] **Step 3: Set base batch size**

Set `data.batch_size: 6` (base VRAM). Leave `val_batch_size` as-is.

- [ ] **Step 4: Point warm-start at the base checkpoint + bump version**

```yaml
  tb_version:    "llwt-v0.6.0-base-hfd"
  weights_ckpt:  "checkpoints/llwt_v45_base/llwt-v0.4.6-base/epoch=199-psnr=18.5361.ckpt"   # warm-start G (strict=False); base generator trained without HF-D
  resume_ckpt:   null
```

- [ ] **Step 5: Add the HF-D off ablation comment next to `hfd_weight`**

```yaml
  hfd_weight:             1.0    # HF-D headline lever. SET 0 to disable HF-D (reproduces plain baseline / ablation arm).
```

- [ ] **Step 6: Verify the generator builds with the base backbone**

Run: `python -m src.models.llwt_v5.gen`
Expected: prints generator summary, no exception; channel dims reflect base (128/256/512/1024). If it downloads the backbone, that is expected on first run.

- [ ] **Step 7: Commit**

```bash
git add src/models/llwt_v5/config.yaml
git commit -m "feat(llwt_v5): default to base backbone + document tiny/HF-D switches

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Remove redundant `llwt_v45_base`

**Files:**
- Delete: `src/models/llwt_v45_base/`

**Interfaces:**
- Consumes: Task 1 (base is now reachable via llwt_v5 config).

- [ ] **Step 1: Confirm nothing in surviving code imports it**

```bash
grep -rn "llwt_v45_base" src/ tests/ --include=*.py | grep -v '/llwt_v45_base/'
```
Expected: no hits (the only references are inside the dir itself, which is being removed).

- [ ] **Step 2: Delete the directory**

```bash
git rm -r src/models/llwt_v45_base
```

- [ ] **Step 3: Verify llwt_v5 still imports clean**

Run: `python -c "import src.models.llwt_v5.factory"`
Expected: no ImportError.

- [ ] **Step 4: Commit**

```bash
git commit -m "refactor: drop redundant llwt_v45_base (base now a llwt_v5 config switch)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Remove dead model lineages

**Files:**
- Delete: `src/models/{cfrwd,huggingface_gan,llwt,llwt_v3,llwt_v4,llwt_v45,sar2opt_v1,sarformer_wb,pix2pix}/`

**Interfaces:**
- Consumes: archive tag (Task 0) for recoverability.

- [ ] **Step 1: Confirm no surviving non-test code imports these**

```bash
grep -rnE "models\.(cfrwd|huggingface_gan|llwt|llwt_v3|llwt_v4|llwt_v45|sar2opt_v1|sarformer_wb|pix2pix)\b" src/ --include=*.py | grep -vE '/(cfrwd|huggingface_gan|llwt|llwt_v3|llwt_v4|llwt_v45|sar2opt_v1|sarformer_wb|pix2pix)/'
```
Expected: no hits outside the dirs themselves. (`llwt_v45` matches must not be `llwt_v45_base` — already gone; and must not be `llwt_v5`.)

- [ ] **Step 2: Delete the directories**

```bash
git rm -r src/models/cfrwd src/models/huggingface_gan src/models/llwt src/models/llwt_v3 src/models/llwt_v4 src/models/llwt_v45 src/models/sar2opt_v1 src/models/sarformer_wb src/models/pix2pix
```

- [ ] **Step 3: Verify the package still imports**

Run: `python -c "import src.models.llwt_v5.factory; import src.models.llwt_v5.main"`
Expected: no ImportError.

- [ ] **Step 4: Commit**

```bash
git commit -m "chore: remove dead model lineages (preserved in archive/full-lineage-v1)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Strip research scaffolding from llwt_v5 (keep smoke + viz)

**Files:**
- Delete in `src/models/llwt_v5/`: `proof_*.py, diag_*.py, overfit_*.py, compare_v45_v5.py, thesis_hfd_section.tex, _*.log, img_to_find/, config_smoke_off.yaml`

**Interfaces:**
- Produces: a llwt_v5 dir containing only the runnable model + smoke + viz tooling.

- [ ] **Step 1: List what will be removed (sanity check before deletion)**

```bash
ls src/models/llwt_v5/{proof_*.py,diag_*.py,overfit_*.py,compare_v45_v5.py,thesis_hfd_section.tex,config_smoke_off.yaml} src/models/llwt_v5/_*.log 2>/dev/null; ls -d src/models/llwt_v5/img_to_find 2>/dev/null
```
Expected: lists the scaffolding files only — NOT `smoke_train_step.py`, `viz_*.py`, `config_smoke.yaml`, or any core module.

- [ ] **Step 2: Confirm core modules do not import the scaffolding**

```bash
grep -rnE "import (proof_|diag_|overfit_|compare_v45_v5)" src/models/llwt_v5/{gen,blocks,dis,main,factory,losses,patchnce,align,inference,best_inference,eval_full,train,smoke_train_step}.py
```
Expected: no hits.

- [ ] **Step 3: Delete the scaffolding (git rm tracked, rm untracked)**

```bash
git rm -r --ignore-unmatch src/models/llwt_v5/proof_*.py src/models/llwt_v5/diag_*.py src/models/llwt_v5/overfit_*.py src/models/llwt_v5/compare_v45_v5.py src/models/llwt_v5/thesis_hfd_section.tex src/models/llwt_v5/config_smoke_off.yaml src/models/llwt_v5/img_to_find
rm -f src/models/llwt_v5/_*.log
```

- [ ] **Step 4: Verify smoke + factory still import clean**

Run: `python -c "import src.models.llwt_v5.factory; import src.models.llwt_v5.smoke_train_step"`
Expected: no ImportError (smoke_train_step import may execute its guard — if it runs a step on import, instead run `python -c "import src.models.llwt_v5.factory"` and defer smoke to Task 9).

- [ ] **Step 5: Commit**

```bash
git add -A src/models/llwt_v5
git commit -m "chore(llwt_v5): remove research scaffolding (keep smoke + viz harness)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: Remove tests for deleted modules

**Files:**
- Delete: `tests/test_hfgan_*.py, tests/test_llwt_*.py, tests/test_sarformer_*.py, tests/models/cfrwd/`

**Interfaces:**
- Consumes: Tasks 2–3 (modules these tests cover are gone).
- Produces: a test suite that collects without import errors.

- [ ] **Step 1: Identify llwt_v5-specific tests to KEEP (if any)**

```bash
grep -rln "llwt_v5" tests/ --include=*.py
```
Expected: note any `llwt_v5` tests — these are KEPT. `test_llwt_*.py` (no `_v5`) target the old `llwt` dir and are removed.

- [ ] **Step 2: Delete the dead test files**

```bash
git rm --ignore-unmatch tests/test_hfgan_*.py tests/test_llwt_dis.py tests/test_llwt_factory.py tests/test_llwt_gen.py tests/test_llwt_lifting.py tests/test_llwt_losses.py tests/test_llwt_main.py tests/test_sarformer_*.py
git rm -r --ignore-unmatch tests/models/cfrwd
```
NOTE: if Step 1 found a `test_llwt_v5*.py`, do NOT delete it — adjust the globs to spare it.

- [ ] **Step 3: Verify pytest collects with no import errors**

Run: `python -m pytest --collect-only -q`
Expected: collection completes; no `ModuleNotFoundError` / `ImportError` for removed modules. (Pre-existing unrelated collection issues, if any, are out of scope — note them.)

- [ ] **Step 4: Commit**

```bash
git add -A tests
git commit -m "test: remove tests for deleted model lineages

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: Remove top-level junk

**Files:**
- Delete: `dynamo_log.txt, diagnose_g.py, LLWT_V4_REPORT.md, _ablation_eval.py, _hfd_ablation_full.py, _hfd_full.log, output_channels/`

- [ ] **Step 1: Delete tracked + untracked junk**

```bash
git rm --ignore-unmatch dynamo_log.txt diagnose_g.py LLWT_V4_REPORT.md
git rm -r --ignore-unmatch output_channels
rm -f _ablation_eval.py _hfd_ablation_full.py _hfd_full.log
```

- [ ] **Step 2: Verify working tree shows only intended removals**

```bash
git status --short
```
Expected: deletions of the listed files only; no surprise modifications.

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "chore: remove top-level debug logs and one-off scripts

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 7: Decouple inference from Lightning — `load_generator()` helper

**Files:**
- Modify: `src/models/llwt_v5/inference.py`

**Interfaces:**
- Produces: `load_generator(ckpt_path: str, cfg, device="cuda") -> torch.nn.Module` —
  builds `LLWv4Generator(cfg)`, loads `netG.`-prefixed (EMA-swapped) weights from a
  Lightning `.ckpt`, returns `.to(device).eval()`. Reused by Task 8 and inference.

- [ ] **Step 1: Read the current weight-loading block in inference.py**

```bash
grep -nE "state_dict|netG\.|load_state_dict|def main|OmegaConf.load" src/models/llwt_v5/inference.py | head
```

- [ ] **Step 2: Add the standalone helper (mirrors the verified loader)**

Insert near the top-level functions of `inference.py`:
```python
def load_generator(ckpt_path, cfg, device="cuda"):
    """Build LLWv4Generator from gen-config and load netG.-prefixed (EMA-swapped)
    weights from a Lightning .ckpt. No data/training config required."""
    import torch
    from src.models.llwt_v5.gen import LLWv4Generator
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    src_sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    sd = {k[len("netG."):]: v for k, v in src_sd.items() if k.startswith("netG.")}
    g = LLWv4Generator(cfg)
    missing, unexpected = g.load_state_dict(sd, strict=False)
    print(f"[load_generator] {len(sd)} netG tensors | missing={len(missing)} unexpected={len(unexpected)}")
    return g.to(device).eval()
```

- [ ] **Step 3: Route the existing inference path through the helper**

In `main()` of `inference.py`, replace the inline generator construction + `load_state_dict`
with a call to `load_generator(CKPT_PATH, cfg, DEVICE)`. Keep the datamodule/output code unchanged.

- [ ] **Step 4: Verify import + signature (no run yet)**

Run: `python -c "from src.models.llwt_v5.inference import load_generator; print('ok')"`
Expected: prints `ok`, no ImportError.

- [ ] **Step 5: Commit**

```bash
git add src/models/llwt_v5/inference.py
git commit -m "refactor(llwt_v5): extract Lightning-decoupled load_generator() for HF export

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 8: `export_hf.py` — ckpt → safetensors + config.json

**Files:**
- Create: `src/models/llwt_v5/export_hf.py`

**Interfaces:**
- Consumes: `load_generator()` (Task 7).
- Produces: a CLI that writes `generator.safetensors` + `config.json` to an output dir.

- [ ] **Step 1: Write the export script**

```python
"""Export the LLW-Former generator for Hugging Face (route-1: weights + config).

Loads a Lightning .ckpt via inference.load_generator, then writes a clean
generator state_dict (safetensors if available, else .pt) plus a minimal
config.json of generator-only hyperparameters. No training/data keys leak.

Run from repo root::

    python -m src.models.llwt_v5.export_hf \
        --ckpt checkpoints/llwt_v45_base/llwt-v0.4.6-base/epoch=199-psnr=18.5361.ckpt \
        --out  output/hf_export
"""
import argparse
import json
import os

import torch
from omegaconf import OmegaConf

from src.models.llwt_v5.inference import load_generator

CONFIG = "./src/models/llwt_v5/config.yaml"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="output/hf_export")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = OmegaConf.load(CONFIG)
    g = load_generator(args.ckpt, cfg, args.device).cpu()
    os.makedirs(args.out, exist_ok=True)

    sd = g.state_dict()
    try:
        from safetensors.torch import save_file
        save_file(sd, os.path.join(args.out, "generator.safetensors"))
        weights_file = "generator.safetensors"
    except Exception as e:
        print(f"[export_hf] safetensors unavailable ({e}); falling back to .pt")
        torch.save(sd, os.path.join(args.out, "generator.pt"))
        weights_file = "generator.pt"

    gen_cfg = {
        "architecture": "LLWv4Generator",
        "backbone": str(cfg.model.gen.backbone),
        "sar_channels": int(cfg.data.sar_channels),
        "image_size": int(cfg.data.image_size),
        "use_sar_physics": bool(cfg.model.gen.get("use_sar_physics", True)),
        "weights_file": weights_file,
        "num_tensors": len(sd),
    }
    with open(os.path.join(args.out, "config.json"), "w", encoding="utf-8") as f:
        json.dump(gen_cfg, f, indent=2)
    print(f"[export_hf] wrote {weights_file} + config.json to {args.out}")


if __name__ == "__main__":
    main()
```
NOTE during execution: verify the exact config keys (`cfg.model.gen.use_sar_physics`, `cfg.data.sar_channels`, `cfg.data.image_size`) exist; adjust `.get(...)` paths to match the real config schema if they differ.

- [ ] **Step 2: Verify it imports**

Run: `python -c "import src.models.llwt_v5.export_hf; print('ok')"`
Expected: prints `ok`.

- [ ] **Step 3: Run the export against the base checkpoint**

Run:
```bash
python -m src.models.llwt_v5.export_hf --ckpt checkpoints/llwt_v45_base/llwt-v0.4.6-base/epoch=199-psnr=18.5361.ckpt --out output/hf_export --device cpu
```
Expected: prints tensor count + `wrote ... to output/hf_export`; `output/hf_export/config.json` exists and lists the base backbone.

- [ ] **Step 4: Commit (script only — output/ is gitignored)**

```bash
git add src/models/llwt_v5/export_hf.py
git commit -m "feat(llwt_v5): add export_hf.py (ckpt -> safetensors + config.json)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 9: Update CLAUDE.md to the consolidated module

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update the active-development scope line**

Replace the "Active development scope" section so it states `src/models/llwt_v5` is the
canonical module, base backbone is the default, tiny/HF-D are config switches, and the
legacy lineage lives in tag `archive/full-lineage-v1`. Remove the `cfrwd`/`pix2pix` wording.

- [ ] **Step 2: Update the Commands block**

Point train/test/smoke commands at llwt_v5:
```
python -m src.models.llwt_v5.train          # train (config.yaml = base + HF-D)
python -m src.models.llwt_v5.smoke_train_step   # one-step smoke (uses config_smoke.yaml)
python -m src.models.llwt_v5.inference      # inference with base checkpoint
python -m src.models.llwt_v5.export_hf --ckpt <path> --out output/hf_export
```

- [ ] **Step 3: Verify no stale references to deleted modules remain**

```bash
grep -nE "cfrwd|huggingface_gan|sarformer|pix2pix" CLAUDE.md
```
Expected: no hits (or only an intentional mention of the archive tag).

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: point CLAUDE.md at consolidated llwt_v5 module

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 10: Full verification — smoke train, smoke inference, suite, tree

**Files:** none (verification only)

- [ ] **Step 1: Generator smoke on base backbone**

Run: `python -m src.models.llwt_v5.gen`
Expected: builds, no exception, base channel dims.

- [ ] **Step 2: Smoke train — one D+G step, base backbone, HF-D on**

Run: `python -m src.models.llwt_v5.smoke_train_step`
Expected: completes one training step, no crash, finite d/g losses printed.
NOTE: `config_smoke.yaml` is tiny/fast by design. If a base-specific smoke is wanted,
temporarily set its `backbone` to base + `batch_size` lower; otherwise the tiny smoke
validates the train loop and Task 1 already validated the base generator build.

- [ ] **Step 3: Smoke inference — load base checkpoint, produce output**

Run: `python -m src.models.llwt_v5.inference`
Expected: loads the base checkpoint via `load_generator`, writes output images, no error.

- [ ] **Step 4: Test suite collects + runs**

Run: `python -m pytest -q`
Expected: no import errors from deleted modules; surviving tests pass (note any
pre-existing unrelated failures rather than fixing out of scope).

- [ ] **Step 5: Final tree + tag check**

```bash
git status --short
git tag -l 'archive/*'
du -sh src/models/*
```
Expected: clean tree (only intended state), `archive/full-lineage-v1` present, `src/models/` contains only `llwt_v5` (+ `__init__.py`).

- [ ] **Step 6: Re-index GitNexus (index went stale after deletions)**

Run: `npx gitnexus analyze` (preserve embeddings with `--embeddings` if `.gitnexus/meta.json` shows `stats.embeddings > 0`).
Expected: re-analysis completes.

---

## Self-Review

**Spec coverage:**
- Phase A → Task 0 ✓ | Phase B → Tasks 1–2 ✓ | Phase C → Tasks 3–6 ✓ |
  Phase D (keep + CLAUDE.md) → Task 9 ✓ | Phase E F1/F2 → Tasks 7–8, F3 noted in spec ✓ |
  Phase F verification → Task 10 ✓. All spec phases mapped.

**Placeholder scan:** No TBD/TODO. Two explicit "verify exact schema during execution"
notes (Task 8 config keys, Task 5 llwt_v5 test sparing) are guardrails, not deferrals —
each has a concrete fallback action.

**Type consistency:** `load_generator(ckpt_path, cfg, device)` defined in Task 7 is
consumed with matching signature in Task 8. `archive/full-lineage-v1` tag name consistent
across Tasks 0, 3, 9, 10. Base checkpoint path identical everywhere.
