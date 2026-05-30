"""LLW-Former v0.4.5 best-per-scene inference.

Iterates the full val (or train) split, scores every sample by the chosen
``METRIC`` (PSNR | SSIM | LPIPS), and writes the **top-K** SAR/Gen/GT
side-by-side PNGs **per scene**.  Replaces the ``inference.py`` quick-look flow
where the ``N_IMAGES`` knob is silently capped at one batch.

Loads ``CHECKPOINT['state_dict']`` (EMA-swapped weights for a finished run) and
strips the ``netG.`` prefix.  Defaults to the production ``last.ckpt``.

Output layout::

    OUTPUT_DIR/
      scene_5/
        top1_idx00042_psnr18.45_ssim0.712_lpips0.301.png
        top2_idx00310_psnr17.92_ssim0.698_lpips0.318.png
        ...
      scene_45/
        ...

Run from repo root::

    python -m src.models.llwt_v45.best_inference
"""
from __future__ import annotations

import heapq
import os
from collections import defaultdict

# Use locally-cached HF weights; skip the online repo_exists probe that SSL-
# times-out on an offline/blocked network (training already cached the backbone).
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

# transformers' AutoBackbone.from_pretrained calls repo_exists() unconditionally
# (a network probe to pick HF-vs-timm). Offline that *raises*; online on a flaky
# net it SSL-times-out. Our backbone is a cached HF model, so force the HF branch.
if os.environ.get('HF_HUB_OFFLINE') == '1':
    import transformers.models.auto.auto_factory as _af
    _af.repo_exists = lambda *a, **k: True

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torchmetrics.image import (
    PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure,
    LearnedPerceptualImagePatchSimilarity,
)
from torchmetrics.image.fid import FrechetInceptionDistance

from src.models.llwt_v45.inference import _build_datamodule
from src.models.llwt_v45.gen import LLWv4Generator


CHECKPOINT       = "checkpoints/llwt_v45/llwt-v0.4.5-cnx-overhaul/last.ckpt"
TOP_K            = 3
SPLIT            = "val"          # "val" or "train"
METRIC           = "ssim"         # "psnr" | "ssim" | "lpips" — score used to rank
OUTPUT_DIR       = f"./src/models/llwt_v45/output/best_{SPLIT}"
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"
USE_LIVE_WEIGHTS = False          # False = EMA-or-live state_dict; True = raw current_model_state
COMPUTE_FID      = True           # per-scene FID; uses InceptionV3 (~+100 MB VRAM)
COMPUTE_CLIP     = True           # per-scene CLIP image<->image cosine sim (semantic fidelity)
CLIP_MODEL       = "openai/clip-vit-base-patch32"

# CLIP image-encoder normalisation (OpenAI CLIP preprocessing).
_CLIP_MEAN = (0.48145466, 0.45782750, 0.40821073)
_CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)


def _scene_of(dataset, vidx: int) -> str:
    """Return scene id for a dataset index.

    SEN12Full layout: ``items[vidx] = (season, s1_<scene>, s2_<scene>, s1_fname, s2_fname)``.
    QXS layout: ``items[vidx] = <filename>`` (flat, no scene partition) — bucket all
    samples under the single label ``'qxs'`` so the per-scene aggregation degrades to a
    single global aggregation.
    """
    entry = dataset.items[vidx]
    if isinstance(entry, str):
        return 'qxs'
    s1_d = entry[1]
    return s1_d[len('s1_'):]


def _save_triplet(sar_t, opt_t, gen_t, psnr_val, ssim_val, lpips_val, scene, out_path):
    sar_np = sar_t.float().numpy()
    opt_np = opt_t.float().numpy()
    gen_np = gen_t.float().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    sar_img = ((sar_np[0] + 1) / 2).clip(0, 1)
    axes[0].imshow(sar_img, cmap='gray')
    axes[0].set_title(f"SAR (scene {scene})")
    axes[0].axis('off')

    gen_img = ((gen_np + 1) / 2).transpose(1, 2, 0).clip(0, 1)
    axes[1].imshow(gen_img)
    axes[1].set_title(f"Gen (PSNR={psnr_val:.2f}, SSIM={ssim_val:.3f}, LPIPS={lpips_val:.3f})")
    axes[1].axis('off')

    opt_img = ((opt_np + 1) / 2).transpose(1, 2, 0).clip(0, 1)
    axes[2].imshow(opt_img)
    axes[2].set_title("GT optical")
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


@torch.no_grad()
def _clip_cosine(clip_model, a_m1p1, b_m1p1, mean, std):
    """Cosine similarity of CLIP image embeddings for two [-1, 1] RGB batches.

    Semantic fidelity proxy: high cosine = fake and GT optical land in the same
    region of CLIP's image-embedding space (same scene content), tolerant to the
    sub-pixel SAR<->optical misregistration that depresses PSNR/SSIM.  Returns a
    ``(B,)`` tensor of per-sample cosine sims in roughly [-1, 1] (higher better).
    """
    def prep(x):
        x = (x.clamp(-1, 1) + 1.0) / 2.0                          # -> [0, 1]
        x = F.interpolate(x, size=224, mode='bicubic', align_corners=False)
        return (x - mean) / std
    fa = clip_model.get_image_features(pixel_values=prep(a_m1p1.float()))
    fb = clip_model.get_image_features(pixel_values=prep(b_m1p1.float()))
    fa = F.normalize(fa, dim=-1)
    fb = F.normalize(fb, dim=-1)
    return (fa * fb).sum(dim=-1)


def main():
    cfg = OmegaConf.load('./src/models/llwt_v45/config.yaml')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    dm = _build_datamodule(cfg)
    dm.setup("fit")

    dataset = dm.val_dataset if SPLIT == "val" else dm.train_dataset
    loader  = dm.val_dataloader() if SPLIT == "val" else dm.train_dataloader()
    print(f"[best_inference] split={SPLIT} | {len(dataset)} samples in {len(loader)} batches")

    # ----- load generator weights -----
    ckpt = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
    src_dict = (
        ckpt.get('current_model_state', ckpt['state_dict'])
        if USE_LIVE_WEIGHTS
        else ckpt['state_dict']
    )
    state_dict = {
        k[len('netG.'):]: v for k, v in src_dict.items() if k.startswith('netG.')
    }
    src_label = 'live (current_model_state)' if USE_LIVE_WEIGHTS else 'EMA-or-live (state_dict)'
    print(f"[ckpt] {src_label} netG weights: {len(state_dict)} tensors from {CHECKPOINT}")

    netG = LLWv4Generator(cfg).to(DEVICE).eval()
    netG.load_state_dict(state_dict)

    psnr_metric  = PeakSignalNoiseRatio(data_range=1.0).to(DEVICE)
    ssim_metric  = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)
    # LPIPS: inputs already in [-1, 1] → normalize=False.  ``.float()`` cast
    # is harmless and avoids torchmetrics dtype warnings.
    lpips_metric = LearnedPerceptualImagePatchSimilarity(
        net_type='alex', normalize=False,
    ).to(DEVICE)
    # Per-scene FID accumulators.  Built lazily as new scenes are seen so we
    # only pay InceptionV3 VRAM for scenes that actually appear in the split.
    fid_metrics: dict[str, FrechetInceptionDistance] = {}

    # CLIP image<->image semantic-similarity (eval-only). Loaded once; disabled
    # gracefully if the model can't be fetched (offline / no cache).
    clip_model = None
    clip_mean = clip_std = None
    if COMPUTE_CLIP:
        try:
            from transformers import CLIPModel
            clip_model = CLIPModel.from_pretrained(CLIP_MODEL).to(DEVICE).eval()
            clip_mean = torch.tensor(_CLIP_MEAN, device=DEVICE).view(1, 3, 1, 1)
            clip_std  = torch.tensor(_CLIP_STD,  device=DEVICE).view(1, 3, 1, 1)
            print(f"[clip] loaded {CLIP_MODEL} for paired image<->image semantic similarity")
        except Exception as e:
            print(f"[clip] DISABLED — load failed: {type(e).__name__}: {e}")
            clip_model = None

    # per-scene min-heap of length <= TOP_K, ordered by (score, vidx).
    # heapq is a min-heap, so the *smallest* score stays at index 0 — for
    # PSNR/SSIM ``score`` is the metric value (larger = better, we want
    # TOP_K largest).  For LPIPS lower is better, so we negate the metric
    # before pushing.
    per_scene: dict[str, list] = defaultdict(list)
    scene_psnr_running:  dict[str, list[float]] = defaultdict(list)
    scene_ssim_running:  dict[str, list[float]] = defaultdict(list)
    scene_lpips_running: dict[str, list[float]] = defaultdict(list)
    scene_clip_running:  dict[str, list[float]] = defaultdict(list)

    idx_global = 0
    with torch.no_grad():
        for batch_idx, (sar, opt) in enumerate(loader):
            sar = sar.to(DEVICE)
            opt = opt.to(DEVICE)
            gen = netG(sar)
            gen_01 = (gen.clamp(-1, 1) + 1.0) / 2.0
            opt_01 = (opt + 1.0) / 2.0
            B = sar.size(0)
            # Batched CLIP cosine for the whole batch (one forward pair), indexed
            # per-sample below.
            clip_sims = None
            if clip_model is not None:
                clip_sims = _clip_cosine(clip_model, gen, opt, clip_mean, clip_std).detach().cpu()
            for j in range(B):
                vidx = idx_global + j
                if vidx >= len(dataset):
                    break
                scene = _scene_of(dataset, vidx)
                p = psnr_metric(gen_01[j:j + 1], opt_01[j:j + 1]).item()
                s = ssim_metric(gen_01[j:j + 1], opt_01[j:j + 1]).item()
                lp = lpips_metric(
                    gen[j:j + 1].float().clamp(-1, 1),
                    opt[j:j + 1].float(),
                ).item()
                scene_psnr_running[scene].append(p)
                scene_ssim_running[scene].append(s)
                scene_lpips_running[scene].append(lp)
                if clip_sims is not None:
                    scene_clip_running[scene].append(float(clip_sims[j]))
                # Score is the value used to rank for the *top* heap.  LPIPS
                # is lower-is-better, so flip sign so the heap's "largest"
                # corresponds to the best perceptual match.
                if METRIC == 'psnr':
                    score = p
                elif METRIC == 'ssim':
                    score = s
                elif METRIC == 'lpips':
                    score = -lp
                else:
                    raise ValueError(f"unknown METRIC='{METRIC}' (psnr|ssim|lpips)")
                # entry: (score, vidx, p, s, lp, sar_cpu, opt_cpu, gen_cpu)
                # second slot is a tiebreaker for stable heap ordering.
                entry = (score, vidx, p, s, lp,
                         sar[j].detach().cpu(),
                         opt[j].detach().cpu(),
                         gen[j].detach().cpu())
                heap = per_scene[scene]
                if len(heap) < TOP_K:
                    heapq.heappush(heap, entry)
                else:
                    heapq.heappushpop(heap, entry)
                # FID needs normalised [0, 1] inputs with ``normalize=True``.
                # Accumulate per-scene so each scene's FID is over its own
                # real+fake distribution.
                if COMPUTE_FID:
                    if scene not in fid_metrics:
                        fid_metrics[scene] = FrechetInceptionDistance(
                            feature=2048, reset_real_features=True, normalize=True,
                        ).to(DEVICE)
                    fid_metrics[scene].update(gen_01[j:j + 1].float(), real=False)
                    fid_metrics[scene].update(opt_01[j:j + 1].float(), real=True)
            idx_global += B
            if (batch_idx + 1) % 10 == 0:
                print(f"  processed batch {batch_idx + 1}/{len(loader)} "
                      f"(scenes seen: {len(per_scene)})")

    print(f"[best_inference] scoring complete; {len(per_scene)} scenes found")

    # ----- save top-K per scene + print summary -----
    for scene in sorted(per_scene.keys()):
        scene_dir = os.path.join(OUTPUT_DIR, f"scene_{scene}")
        os.makedirs(scene_dir, exist_ok=True)
        ranked = sorted(per_scene[scene], key=lambda e: -e[0])  # desc by score
        for rank, (score, vidx, p, s, lp, sar_t, opt_t, gen_t) in enumerate(ranked, start=1):
            out_path = os.path.join(
                scene_dir,
                f"top{rank}_idx{vidx:05d}_psnr{p:.2f}_ssim{s:.3f}_lpips{lp:.3f}.png",
            )
            _save_triplet(sar_t, opt_t, gen_t, p, s, lp, scene, out_path)
            print(f"  scene={scene} rank={rank} idx={vidx} "
                  f"PSNR={p:.2f} SSIM={s:.3f} LPIPS={lp:.3f} → {out_path}")

    print("=" * 78)
    print(f"Per-scene summary (METRIC={METRIC}, K={TOP_K}):")
    for scene in sorted(per_scene.keys()):
        ranked = sorted(per_scene[scene], key=lambda e: -e[0])
        top_scores = [f"{e[0]:+.2f}" for e in ranked]  # LPIPS uses negated score
        n_scene = len(scene_psnr_running[scene])
        mean_psnr  = sum(scene_psnr_running[scene])  / n_scene
        mean_ssim  = sum(scene_ssim_running[scene])  / n_scene
        mean_lpips = sum(scene_lpips_running[scene]) / n_scene
        fid_str = ""
        if COMPUTE_FID and scene in fid_metrics:
            try:
                fid_val = float(fid_metrics[scene].compute().item())
                fid_str = f" FID={fid_val:.1f}"
            except Exception as e:
                fid_str = f" FID=ERR({type(e).__name__})"
        clip_str = ""
        if scene_clip_running.get(scene):
            mean_clip = sum(scene_clip_running[scene]) / len(scene_clip_running[scene])
            clip_str = f" CLIP={mean_clip:.4f}"
        print(f"  scene {scene:>4s} | n={n_scene:>4d} | "
              f"top-{TOP_K} {METRIC}=[{', '.join(top_scores)}] | "
              f"mean PSNR={mean_psnr:.2f} SSIM={mean_ssim:.3f} LPIPS={mean_lpips:.3f}{fid_str}{clip_str}")
    # Overall semantic headline: dataset-wide mean CLIP image<->image cosine.
    if COMPUTE_CLIP and any(scene_clip_running.values()):
        all_clip = [v for vs in scene_clip_running.values() for v in vs]
        print("-" * 78)
        print(f"  OVERALL CLIP image<->image cosine = {sum(all_clip) / len(all_clip):.4f}  "
              f"(n={len(all_clip)}; higher = better semantic match)")
    print("=" * 78)
    if COMPUTE_FID and fid_metrics:
        # Reset to free InceptionV3 buffers before exit.
        for m in fid_metrics.values():
            m.reset()


if __name__ == "__main__":
    main()
