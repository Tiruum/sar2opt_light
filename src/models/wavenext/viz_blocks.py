"""Per-block visualization of the WaveNeXt GENERATOR and DISCRIMINATOR.

Registers forward hooks on every block of ``WaveNeXtGenerator`` and of
``WaveNeXtDiscriminator`` (Main-D coarse + fine branches, HF-D), runs ONE
forward pass on a SAR sample, reduces each intermediate activation to an image,
and emits a single HTML page laying every pipeline out left-to-right with
arrows, Russian "после блока X" captions, per-block explanations, encoder→
decoder skip-connection badges, the SAR conditioning input of the
discriminator, and a final real-vs-generated LSGAN comparison.

Generator flow (skip connections s0/s1/s2 feed the decoder):
    SAR (вход) → SAR-adapter → Haar-Stem → enc s0/s1/s2/s3 →
    dec up4(+s2)/up3(+s1)/up2(+s0)/up1 → subband_head → IHaar → tanh (выход)

Discriminator flows (conditional PatchGAN — SAR is the condition):
    [SAR условие] ⊕ [генерация]   → conv0.. → logits   (Main-D coarse / fine)
    [SAR условие] ⊕ [ВЧ-остаток]  → conv0.. → logits   (HF-D)

Feature maps: grayscale grid of the first ``N_CHANNELS`` example channels.
True-RGB tensors (sar3, logits, output) shown directly.  Logit maps: blue→red
ramp (blue = синтетика, red = реалистичнее).

Run from repo root::

    python -m src.models.wavenext.viz_blocks

Then open ``src/models/wavenext/output/blocks/index.html``.
"""
import math
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf

from src.models.wavenext import factory
# importing _build_datamodule also installs the offline-HF env shims
from src.models.wavenext.inference import _build_datamodule


CHECKPOINT = "checkpoints/llwt_v45/llwt-v0.5.1-hfd/epoch=097-psnr=17.1615.ckpt"
N_IMAGES = 2
SPLIT = "val"  # "train" or "val"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HP_AMP = 5.0
N_CHANNELS = 4  # example channels per feature block (grid)
OUTPUT_DIR = "./src/models/wavenext/output/blocks"

STAGE_COLOR = {
    'вход':    '#6b7280',
    'энкодер': '#2563eb',
    'декодер': '#16a34a',
    'голова':  '#d97706',
    'Main-D':  '#7c3aed',
    'HF-D':    '#db2777',
}
MODE_LABEL = {
    'gray': 'SAR', 'out': 'RGB', 'rgb': 'RGB', 'feat': f'каналы 0–{N_CHANNELS - 1}',
    'logits': 'оценка ↓синт/↑реал', 'hp': 'ВЧ ×5',
}

# Per-conv explanation reused across discriminator branches.
_CONV_DESC = "Conv 4×4 + LeakyReLU. Стрид-2 слои делят разрешение ×½ и удваивают каналы — извлекают патч-признаки."
_LOGIT_DESC = "Финальный conv → карта 1 канал. Каждый патч получает LSGAN-оценку реалистичности (не вероятность)."


def _load_models(cfg, ckpt):
    netG, netD = factory.build_models(cfg)
    sd = ckpt['state_dict']
    netG.load_state_dict({k[len('netG.'):]: v for k, v in sd.items() if k.startswith('netG.')})
    netD.load_state_dict({k[len('netD.'):]: v for k, v in sd.items() if k.startswith('netD.')})
    return netG.to(DEVICE).eval(), netD.to(DEVICE).eval()


# ----------------------------------------------------------------- reductions

def _feat_montage(t, k=N_CHANNELS, ncols=2):
    """(C,H,W) -> grayscale grid of the first k channels (each min-max normed)."""
    t = t.detach().float()
    C = t.shape[0]
    k = min(k, C)
    chans = []
    for c in range(k):
        m = t[c].cpu().numpy()
        m = (m - m.min()) / (np.ptp(m) + 1e-8)
        chans.append(np.pad(m, 1, constant_values=1.0))
    nrows = math.ceil(k / ncols)
    while len(chans) < nrows * ncols:
        chans.append(np.ones_like(chans[0]))
    rows = [np.hstack(chans[r * ncols:(r + 1) * ncols]) for r in range(nrows)]
    return np.vstack(rows)


def _rgb_norm(t):
    arr = t.detach().float().permute(1, 2, 0).cpu().numpy()
    mn = arr.min(axis=(0, 1), keepdims=True)
    mx = arr.max(axis=(0, 1), keepdims=True)
    return (arr - mn) / (mx - mn + 1e-8)


def _rel(x, vmin, vmax):
    return (x - vmin) / ((vmax - vmin) or 1.0)


def _render_image(kind, tensor):
    if kind == 'gray':
        return (tensor[0].detach().cpu().numpy() + 1) / 2, 'gray'
    if kind == 'out':
        return np.clip((tensor.detach().float().permute(1, 2, 0).cpu().numpy() + 1) / 2, 0, 1), None
    if kind == 'rgb':
        return _rgb_norm(tensor), None
    if kind == 'hp':
        arr = tensor.detach().float().permute(1, 2, 0).cpu().numpy()
        return np.clip(arr * HP_AMP + 0.5, 0, 1), None
    if kind == 'logits':
        m = tensor.detach().float()[0].cpu().numpy()
        return (m - m.min()) / (np.ptp(m) + 1e-8), 'RdBu_r'
    return _feat_montage(tensor), 'gray'  # 'feat'


def _save_panel(path, img, cmap=None):
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(img, cmap=cmap)
    ax.axis('off')
    fig.savefig(path, dpi=120, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


# ----------------------------------------------------------------- hooks

def _hook_list(module_list):
    store = [None] * len(module_list)
    handles = []
    for k, m in enumerate(module_list):
        def mk(idx):
            def h(_mod, _inp, out):
                store[idx] = out
            return h
        handles.append(m.register_forward_hook(mk(k)))
    return store, handles


def _collect_gen(netG, sar):
    """Run generator forward with hooks; return (seq, out_tensor).

    seq entries: (label, stage, tensor, kind, desc, skip).
    """
    acts = {}

    def hook(name):
        def _h(_m, _i, out):
            acts[name] = out
        return _h

    handles = []
    adapter = netG.sar_adapter if netG.sar_adapter is not None else netG.naive_proj
    handles.append(adapter.register_forward_hook(hook('sar3')))
    handles.append(netG.encoder.embeddings.patch_embeddings.register_forward_hook(hook('stem')))
    handles.append(netG.encoder.register_forward_hook(hook('encoder')))
    for nm in ('up4', 'up3', 'up2', 'up1'):
        handles.append(getattr(netG, nm).register_forward_hook(hook(nm)))
    handles.append(netG.subband_head.register_forward_hook(hook('subband_head')))
    handles.append(netG.ihaar.register_forward_hook(hook('ihaar')))

    with torch.no_grad():
        out = netG(sar)
    for h in handles:
        h.remove()

    f = acts['encoder'].feature_maps
    seq = [
        ("SAR (вход)", 'вход', sar[0], 'gray',
         "Входной радиолокационный снимок (1 канал, амплитуда). Геометрия сцены, без оптического спектра.", ""),
        ("SAR-adapter → sar3", 'вход', acts['sar3'][0], 'rgb',
         "Физический адаптер 1→3 канала (амплитуда + log + производные). Готовит SAR под 3-канальный вход бэкбона.", ""),
        ("Haar-Stem", 'энкодер', acts['stem'][0], 'feat',
         "Вейвлет-стем вместо patch-embed: Haar-разложение → 7×C субполос → 1×1 conv. ×¼ разрешения, кодирует частоты с 1-го слоя.", ""),
        ("Энкодер s0 (H/4)", 'энкодер', f[0][0], 'feat',
         "ConvNeXt V2, стадия 1. Низкоуровневые признаки: края, текстура.", ""),
        ("Энкодер s1 (H/8)", 'энкодер', f[1][0], 'feat',
         "Стадия 2. Средний уровень: формы, границы объектов.", ""),
        ("Энкодер s2 (H/16)", 'энкодер', f[2][0], 'feat',
         "Стадия 3. Высокий уровень: структуры и контекст.", ""),
        ("Энкодер s3 (H/32)", 'энкодер', f[3][0], 'feat',
         "Стадия 4, самое глубокое представление. Семантика сцены, минимум разрешения.", ""),
        ("Декодер up4 (H/16)", 'декодер', acts['up4'][0], 'feat',
         "PixelShuffle ×2 + skip-связь s2. Восстанавливает разрешение, вливает детали энкодера.", "s2 (H/16)"),
        ("Декодер up3 (H/8)", 'декодер', acts['up3'][0], 'feat',
         "PixelShuffle ×2 + skip-связь s1.", "s1 (H/8)"),
        ("Декодер up2 (H/4)", 'декодер', acts['up2'][0], 'feat',
         "PixelShuffle ×2 + skip-связь s0.", "s0 (H/4)"),
        ("Декодер up1 (H/2)", 'декодер', acts['up1'][0], 'feat',
         "Финальный апскейл декодера (без skip), 32 канала на H/2.", ""),
        ("subband_head (12ch)", 'голова', acts['subband_head'][0], 'feat',
         "7×7 conv: 32→12 = 3 RGB × 4 субполосы Haar (LL,LH,HL,HH) на H/2. Zero-init → старт с серого.", ""),
        ("IHaar → logits", 'голова', acts['ihaar'][0], 'rgb',
         "Обратный Haar: 12 субполос → RGB на полном H. Замыкает вейвлет-петлю (стем-вход ↔ ihaar-выход).", ""),
        ("tanh → выход", 'голова', out[0], 'out',
         "Сжатие в [-1,1] → готовое оптическое изображение.", ""),
    ]
    return seq, out


def _branch_items(store, sar_t, img_t, img_label, img_kind, color, head_desc):
    """Flow items for a PatchGAN branch: SAR condition ⊕ image → conv.. → logits.

    Returns list of nodes; a node is a card tuple or the connector marker '⊕'.
    Card tuple: (label, color, tensor, kind, desc, skip).
    """
    n = len(store)
    items = [
        ("Условие: SAR", color, sar_t, 'gray',
         "SAR подаётся как УСЛОВИЕ (после InstanceNorm) и склеивается с изображением по каналам. "
         "Привязывает оценку к геометрии входа — D conditional.", ""),
        '⊕',
        (img_label, color, img_t, img_kind, head_desc, ""),
    ]
    for k in range(n - 1):
        items.append((f"после conv{k}", color, store[k][0], 'feat', _CONV_DESC, ""))
    items.append(("logits (карта оценок)", color, store[n - 1][0], 'logits', _LOGIT_DESC, ""))
    return items


def _collect_dis(netD, sar, fake):
    """Fake pass through Main-D (coarse+fine) and HF-D with per-layer hooks."""
    hf = netD.highfreq
    sc, hc = _hook_list(netD.main.coarse.layers)
    sf, hf_ = _hook_list(netD.main.fine.layers)
    sh, hh = _hook_list(hf.layers)
    with torch.no_grad():
        netD.main(sar, fake)
        hp = hf.highpass(fake, hf.sigma)
        hf(sar, hp)
    for h in hc + hf_ + hh:
        h.remove()

    main_desc = ("Условный PatchGAN. Вход = [InstanceNorm(SAR) ⊕ изображение] (4 канала). "
                 "Две ветки масштабов: coarse 70×70 RF, fine 46×46 RF.")
    hf_desc = ("Условный SN-PatchGAN на ВЧ-остатке opt−gauss(opt). "
               "Судит когерентность высокой частоты; SAR — условие.")
    return [
        ("Main-D · coarse (70×70 RF)",
         _branch_items(sc, sar[0], fake[0], "Изображение: генерация", 'out', 'Main-D', main_desc)),
        ("Main-D · fine (46×46 RF)",
         _branch_items(sf, sar[0], fake[0], "Изображение: генерация", 'out', 'Main-D', main_desc)),
        ("HF-D · высокочастотный",
         _branch_items(sh, sar[0], hp[0], "ВЧ-остаток: генерация", 'hp', 'HF-D', hf_desc)),
    ]


# ----------------------------------------------------------------- html

def _emit_card(sdir, sdir_rel, fname, label, colorkey, tensor, kind, desc="", skip=""):
    img, cmap = _render_image(kind, tensor)
    _save_panel(os.path.join(sdir, fname), img, cmap=cmap)
    shape = "×".join(str(s) for s in tuple(tensor.shape))
    color = STAGE_COLOR[colorkey]
    skip_html = f'<div class="skip">⟲ skip-связь ← {skip}</div>' if skip else ''
    desc_html = f'<div class="desc">{desc}</div>' if desc else ''
    return (
        f'<div class="card" style="border-color:{color}">'
        f'{skip_html}'
        f'<img src="{sdir_rel}/{fname}" />'
        f'<div class="cap" style="background:{color}">после блока:<br><b>{label}</b></div>'
        f'<div class="meta">{shape} · {MODE_LABEL[kind]}</div>'
        f'{desc_html}'
        f'</div>'
    )


def _flow_div(nodes):
    """nodes: list of card-html strings or connector markers ('→' default, '⊕')."""
    inner = []
    prev_card = False
    for nd in nodes:
        if nd in ('→', '⊕'):
            inner.append(f'<div class="arrow">{nd}</div>')
            prev_card = False
            continue
        if prev_card:
            inner.append('<div class="arrow">→</div>')
        inner.append(nd)
        prev_card = True
    return f'<div class="flow">{"".join(inner)}</div>'


def _emit_compare(sdir, sdir_rel, fid, head, color, real_map, fake_map):
    """Two logit maps (эталон vs генерация) under a SHARED relative scale."""
    vmin = float(min(real_map.min(), fake_map.min()))
    vmax = float(max(real_map.max(), fake_map.max()))
    rr = _rel(real_map.mean(), vmin, vmax)
    rf = _rel(fake_map.mean(), vmin, vmax)
    cards = []
    for tag, m, val in (("эталон (real)", real_map, rr), ("генерация (fake)", fake_map, rf)):
        fname = f"cmp{fid}_{'r' if 'real' in tag else 'f'}.png"
        _save_panel(os.path.join(sdir, fname), _rel(m, vmin, vmax), cmap='RdBu_r')
        cards.append(
            f'<div class="card" style="border-color:{color}">'
            f'<img src="{sdir_rel}/{fname}" />'
            f'<div class="cap" style="background:{color}">{head}<br><b>{tag}</b></div>'
            f'<div class="meta">отн. оценка μ = {val:.2f}</div>'
            f'</div>'
        )
    note = (f'<div class="cmpnote">LSGAN-цель: эталон → 0.9, генерация → 0.0. '
            f'Разрыв оценок (эталон {rr:.2f} ≫ генерация {rf:.2f}) — состязательный сигнал генератору. '
            f'Общая шкала: эталон краснее (реалистичнее), генерация синее (синтетика).</div>')
    return f'<div class="flow">{cards[0]}<div class="arrow">vs</div>{cards[1]}{note}</div>'


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cfg = OmegaConf.load('./src/models/wavenext/config.yaml')
    cfg.data.num_workers = 0

    dm = _build_datamodule(cfg)
    dm.setup("fit")
    loader = dm.val_dataloader() if SPLIT == "val" else dm.train_dataloader()
    sar, opt = next(iter(loader))
    sar, opt = sar.to(DEVICE), opt.to(DEVICE)

    ckpt = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
    netG, netD = _load_models(cfg, ckpt)
    hf = netD.highfreq

    n = min(N_IMAGES, sar.shape[0])
    sections = []
    for i in range(n):
        sar_i, opt_i = sar[i:i + 1], opt[i:i + 1]
        sdir = os.path.join(OUTPUT_DIR, f"sample_{i:03d}")
        os.makedirs(sdir, exist_ok=True)
        sdir_rel = f"sample_{i:03d}"

        # --- generator (with skip badges + explanations) ---
        gen_seq, out = _collect_gen(netG, sar_i)
        gcards = [_emit_card(sdir, sdir_rel, f"g_{j:02d}.png", lbl, st, t, kd, desc, skip)
                  for j, (lbl, st, t, kd, desc, skip) in enumerate(gen_seq)]
        block = [f'<h2>Образец #{i} — Генератор</h2>', _flow_div(gcards)]

        # --- discriminator fake pass (SAR condition shown) ---
        dis_flows = _collect_dis(netD, sar_i, out)
        for fi, (title, items) in enumerate(dis_flows):
            nodes = []
            cardc = 0
            for it in items:
                if it == '⊕':
                    nodes.append('⊕')
                    continue
                lbl, st, t, kd, desc, skip = it
                nodes.append(_emit_card(sdir, sdir_rel, f"d{fi}_{cardc:02d}.png",
                                        lbl, st, t, kd, desc, skip))
                cardc += 1
            block.append(f'<h3>Дискриминатор — {title}</h3>')
            block.append(_flow_div(nodes))

        # --- real-vs-generated comparison (how LSGAN compares) ---
        with torch.no_grad():
            main_real = netD(sar_i, opt_i)[0][1]   # fine logits, real
            main_fake = netD(sar_i, out)[0][1]      # fine logits, fake
            hf_real, _ = hf(sar_i, hf.highpass(opt_i, hf.sigma))
            hf_fake, _ = hf(sar_i, hf.highpass(out, hf.sigma))
        mr, mf = main_real[0, 0].cpu().numpy(), main_fake[0, 0].cpu().numpy()
        hr, hfk = hf_real[0, 0].cpu().numpy(), hf_fake[0, 0].cpu().numpy()
        block.append('<h3>Как дискриминатор сравнивает реальное и сгенерированное (LSGAN)</h3>')
        block.append(_emit_compare(sdir, sdir_rel, 'main', "Main-D (fine)", STAGE_COLOR['Main-D'], mr, mf))
        block.append(_emit_compare(sdir, sdir_rel, 'hf', "HF-D (ВЧ)", STAGE_COLOR['HF-D'], hr, hfk))

        sections.append("".join(block))
        print(f"[{i:03d}] gen {len(gen_seq)} + dis {sum(len(it) for _, it in dis_flows)} + compare 4 -> {sdir}")

    legend = "".join(
        f'<span class="lg" style="background:{c}">{name}</span>'
        for name, c in STAGE_COLOR.items()
    )
    html = f"""<!DOCTYPE html><html lang="ru"><head><meta charset="utf-8">
<title>WaveNeXt: визуализация по блокам (генератор + дискриминатор)</title>
<style>
  body {{ font-family:-apple-system,Segoe UI,Roboto,sans-serif; margin:24px; color:#111; background:#fafafa; }}
  h1 {{ font-size:22px; }} h2 {{ font-size:17px; margin-top:30px; color:#111; border-top:2px solid #ddd; padding-top:12px; }}
  h3 {{ font-size:14px; margin:16px 0 6px; color:#374151; }}
  .legend {{ margin:8px 0; }}
  .lg {{ color:#fff; padding:3px 10px; border-radius:6px; margin-right:6px; font-size:12px; }}
  .flow {{ display:flex; flex-wrap:wrap; align-items:flex-start; gap:4px; }}
  .card {{ width:172px; border:3px solid #999; border-radius:10px; overflow:hidden;
           background:#fff; box-shadow:0 1px 3px rgba(0,0,0,.12); }}
  .card img {{ width:172px; height:172px; object-fit:cover; display:block; }}
  .skip {{ background:#0ea5e9; color:#fff; font-size:10px; font-weight:bold; padding:3px 6px; }}
  .cap {{ color:#fff; font-size:11px; padding:5px 6px; line-height:1.25; }}
  .meta {{ font-size:10px; color:#6b7280; padding:3px 6px; font-family:monospace; }}
  .desc {{ font-size:10px; color:#374151; padding:4px 6px 7px; line-height:1.32; }}
  .arrow {{ font-size:24px; color:#9ca3af; padding:60px 2px 0; }}
  .cmpnote {{ font-size:12px; color:#374151; max-width:340px; padding:6px 10px; line-height:1.4;
              background:#fff7ed; border-left:3px solid #d97706; border-radius:6px; align-self:center; }}
</style></head><body>
<h1>WaveNeXt — активации после каждого блока: генератор + дискриминатор</h1>
<div class="legend">{legend}</div>
<p style="font-size:13px;color:#4b5563;max-width:1100px;">
  Признаковые карты — сетка первых {N_CHANNELS} каналов (примеры) в оттенках серого; sar3 / logits генератора /
  выход — настоящий RGB. <b>⟲ skip-связь</b> — пропуск признаков энкодера в декодер (s0/s1/s2).
  Дискриминатор — <b>условный</b> PatchGAN: SAR подаётся как условие (⊕ склейка по каналам) и показан отдельной
  карточкой. Дискриминатор-потоки на fake-проходе; финальная секция — сравнение эталона и генерации по LSGAN
  (синий = синтетика, красный = реалистичнее). Стрелки — поток данных.
</p>
{''.join(sections)}
</body></html>"""

    html_path = os.path.join(OUTPUT_DIR, "index.html")
    with open(html_path, "w", encoding="utf-8") as fp:
        fp.write(html)
    print("=" * 60)
    print(f"HTML -> {html_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
