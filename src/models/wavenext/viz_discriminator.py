"""Discriminator visualization for WaveNeXt v0.5.1-hfd.

Loads the trained generator AND discriminator (Main-D + HF-D) from one
checkpoint, runs them on a batch, and writes a slide-ready 8-panel per-sample
figure (Russian titles):

    SAR | optical (real) | generated | opt-gauss(opt) |
    Main-D real | Main-D fake | HF-D real | HF-D fake

Each head shows BOTH the real pass D(SAR, opt) and the fake pass D(SAR, gen)
under a SHARED relative [0,1] scale, so the LSGAN comparison reads directly:
the real pass is redder (higher score), the fake pass bluer (lower).  This is
LSGAN, so the raw discriminator output is an unbounded regression score, NOT a
probability — the absolute scale (which runs to ~1e5 on the uncalibrated HF-D
head) is deliberately hidden; only the relative real>fake gap is shown.  The
colorbar legend reads "синтетика <-> реалистичнее".

``opt - gauss(opt)`` is exactly the high-pass residual the HF-D consumes
(``HighFreqDis.highpass``); it is shown amplified so the carved high-frequency
structure is legible.

Run from repo root::

    python -m src.models.wavenext.viz_discriminator
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf

from src.models.wavenext import factory
from src.models.wavenext.gen import WaveNeXtGenerator
# importing _build_datamodule also installs the offline-HF env shims
from src.models.wavenext.inference import _build_datamodule


CHECKPOINT = "checkpoints/llwt_v45/llwt-v0.5.1-hfd/epoch=097-psnr=17.1615.ckpt"
N_IMAGES = 6
SPLIT = "val"  # "train" or "val"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HP_AMP = 5.0  # display gain for the high-pass residual panel
SAVE_INDIVIDUAL = True  # also dump each panel as its own PNG under sample_NNN/
OUTPUT_DIR = f"./src/models/wavenext/output/pres_{SPLIT}"


def _load_weights(netG, netD, ckpt):
    """Strip the Lightning ``netG.`` / ``netD.`` prefixes and load both nets."""
    sd = ckpt['state_dict']
    g_sd = {k[len('netG.'):]: v for k, v in sd.items() if k.startswith('netG.')}
    d_sd = {k[len('netD.'):]: v for k, v in sd.items() if k.startswith('netD.')}
    netG.load_state_dict(g_sd)
    netD.load_state_dict(d_sd)
    print(f"[ckpt] netG tensors: {len(g_sd)} | netD tensors: {len(d_sd)}")


def _to_rgb(t_chw):
    """[-1,1] CHW tensor -> [0,1] HWC numpy."""
    arr = (t_chw.detach().cpu().numpy() + 1.0) / 2.0
    return np.clip(arr.transpose(1, 2, 0), 0.0, 1.0)


def _hp_to_rgb(hp_chw):
    """High-pass residual (centered ~0) -> amplified, clipped [0,1] HWC."""
    arr = hp_chw.detach().cpu().numpy().transpose(1, 2, 0)
    return np.clip(arr * HP_AMP + 0.5, 0.0, 1.0)


def _logit_map(t):
    """(1,1,h,w) logits tensor -> 2D numpy."""
    return t.detach().cpu().numpy()[0, 0]


def _rel(x, vmin, vmax):
    """Relative position of scalar/array x within [vmin, vmax] -> [0, 1]."""
    return (x - vmin) / ((vmax - vmin) or 1.0)


def _heat(fig, ax, score_map, vmin, vmax, title, add_cbar=False):
    """Draw one PatchGAN score map rescaled *relative* to a shared [vmin, vmax].

    Both the эталон (real) and генерация (fake) passes of a head are drawn with
    the SAME (vmin, vmax) so the eye reads the LSGAN comparison directly: the
    real pass sits redder (higher score) than the fake pass.  The absolute
    LSGAN score (which runs to ~1e5 on the uncalibrated HF-D head) never reaches
    the slide — only "more synthetic <-> more realistic".
    """
    im = ax.imshow(_rel(score_map, vmin, vmax), cmap='RdBu_r',
                   vmin=0.0, vmax=1.0, interpolation='bilinear')
    ax.set_title(title, fontsize=10)
    if add_cbar:
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=[0.04, 0.96])
        cb.ax.set_yticklabels(['синтетика', 'реалистичнее'], fontsize=8,
                              rotation=90, va='center')
    return im


def _save_img(path, arr, cmap=None):
    """Save a single image panel — clean, no axes/title/padding."""
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(arr, cmap=cmap)
    ax.axis('off')
    fig.savefig(path, dpi=150, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def _save_heat(path, score_map, vmin, vmax, title):
    """Save a single score-map panel with the synthetic<->realistic legend."""
    fig, ax = plt.subplots(figsize=(4.6, 4))
    im = ax.imshow(_rel(score_map, vmin, vmax), cmap='RdBu_r',
                   vmin=0.0, vmax=1.0, interpolation='bilinear')
    ax.set_title(title, fontsize=11)
    ax.axis('off')
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=[0.04, 0.96])
    cb.ax.set_yticklabels(['синтетика', 'реалистичнее'], fontsize=8,
                          rotation=90, va='center')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    cfg = OmegaConf.load('./src/models/wavenext/config.yaml')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    dm = _build_datamodule(cfg)
    dm.setup("fit")
    loader = dm.val_dataloader() if SPLIT == "val" else dm.train_dataloader()
    sar, opt = next(iter(loader))
    sar = sar.to(DEVICE)
    opt = opt.to(DEVICE)

    ckpt = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
    netG, netD = factory.build_models(cfg)
    _load_weights(netG, netD, ckpt)
    netG = netG.to(DEVICE).eval()
    netD = netD.to(DEVICE).eval()

    if getattr(netD, 'highfreq', None) is None:
        raise RuntimeError("HF-D head not present — check cfg.model.dis.highfreq.enabled")
    hf = netD.highfreq

    with torch.no_grad():
        fake = netG(sar)

        # Main-D conditional logits (use the fine/high-res scale), real vs fake.
        (_, main_lf_real), *_ = netD(sar.float(), opt.float())
        (_, main_lf_fake), *_ = netD(sar.float(), fake.float())

        # HF-D logits on the high-pass residual, real vs fake.
        hp_real = hf.highpass(opt, hf.sigma)
        hp_fake = hf.highpass(fake, hf.sigma)
        hf_logits_real, _ = hf(sar.float(), hp_real)
        hf_logits_fake, _ = hf(sar.float(), hp_fake)

    n = min(N_IMAGES, sar.shape[0])
    cols = 8
    for i in range(n):
        fig, axes = plt.subplots(1, cols, figsize=(3.6 * cols, 4.8))

        # 1) SAR input
        axes[0].imshow((sar[i, 0].cpu().numpy() + 1) / 2, cmap='gray')
        axes[0].set_title("SAR (вход)", fontsize=10)

        # 2) GT optical — изначальное изображение
        axes[1].imshow(_to_rgb(opt[i]))
        axes[1].set_title("Оптический эталон", fontsize=10)

        # 3) Generated — сгенерированное
        axes[2].imshow(_to_rgb(fake[i]))
        axes[2].set_title("Сгенерированное", fontsize=10)

        # 4) opt - gauss(opt) — HF-D input, high-pass residual of real optical
        axes[3].imshow(_hp_to_rgb(hp_real[i]))
        axes[3].set_title(f"ВЧ-остаток opt − gauss(opt)\n(усиление ×{HP_AMP:g})", fontsize=10)

        # 5-6) Main-D: эталон vs генерация под ОБЩЕЙ шкалой (как сравнивает LSGAN)
        m_real = _logit_map(main_lf_real[i:i + 1])
        m_fake = _logit_map(main_lf_fake[i:i + 1])
        mvmin = float(min(m_real.min(), m_fake.min()))
        mvmax = float(max(m_real.max(), m_fake.max()))
        _heat(fig, axes[4], m_real, mvmin, mvmax, "Main-D · эталон")
        _heat(fig, axes[5], m_fake, mvmin, mvmax,
              f"Main-D · генерация\n(отн.: эталон {_rel(m_real.mean(), mvmin, mvmax):.2f} / "
              f"ген {_rel(m_fake.mean(), mvmin, mvmax):.2f})", add_cbar=True)

        # 7-8) HF-D: эталон vs генерация под ОБЩЕЙ шкалой
        h_real = _logit_map(hf_logits_real[i:i + 1])
        h_fake = _logit_map(hf_logits_fake[i:i + 1])
        hvmin = float(min(h_real.min(), h_fake.min()))
        hvmax = float(max(h_real.max(), h_fake.max()))
        _heat(fig, axes[6], h_real, hvmin, hvmax, "HF-D (ВЧ) · эталон")
        _heat(fig, axes[7], h_fake, hvmin, hvmax,
              f"HF-D (ВЧ) · генерация\n(отн.: эталон {_rel(h_real.mean(), hvmin, hvmax):.2f} / "
              f"ген {_rel(h_fake.mean(), hvmin, hvmax):.2f})", add_cbar=True)

        for ax in axes:
            ax.axis('off')
        fig.suptitle(
            "LSGAN: дискриминатор сравнивает оценки эталона и генерации "
            "(цель: эталон → 0.9, генерация → 0.0). Общая шкала на пару: "
            "эталон краснее (реалистичнее), генерация синее (синтетика); "
            "разрыв оценок — состязательный сигнал генератору.",
            fontsize=12, y=1.02)
        plt.tight_layout()
        out_path = os.path.join(OUTPUT_DIR, f"pres_{i:03d}.png")
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        if SAVE_INDIVIDUAL:
            sdir = os.path.join(OUTPUT_DIR, f"sample_{i:03d}")
            os.makedirs(sdir, exist_ok=True)
            _save_img(os.path.join(sdir, "1_sar.png"),
                      (sar[i, 0].cpu().numpy() + 1) / 2, cmap='gray')
            _save_img(os.path.join(sdir, "2_optical_real.png"), _to_rgb(opt[i]))
            _save_img(os.path.join(sdir, "3_generated.png"), _to_rgb(fake[i]))
            _save_img(os.path.join(sdir, "4_highpass.png"), _hp_to_rgb(hp_real[i]))
            _save_heat(os.path.join(sdir, "5_mainD_real.png"), m_real, mvmin, mvmax, "Main-D · эталон")
            _save_heat(os.path.join(sdir, "6_mainD_fake.png"), m_fake, mvmin, mvmax, "Main-D · генерация")
            _save_heat(os.path.join(sdir, "7_hfD_real.png"), h_real, hvmin, hvmax, "HF-D (ВЧ) · эталон")
            _save_heat(os.path.join(sdir, "8_hfD_fake.png"), h_fake, hvmin, hvmax, "HF-D (ВЧ) · генерация")

        print(f"[{i:03d}] Main-D rel real/fake = "
              f"{_rel(m_real.mean(), mvmin, mvmax):.2f}/{_rel(m_fake.mean(), mvmin, mvmax):.2f}"
              f" | HF-D rel real/fake = "
              f"{_rel(h_real.mean(), hvmin, hvmax):.2f}/{_rel(h_fake.mean(), hvmin, hvmax):.2f}"
              f" -> {out_path}")

    print("=" * 60)
    print(f"Saved {n} discriminator panels to {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
