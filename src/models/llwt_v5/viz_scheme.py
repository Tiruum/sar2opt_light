"""Overall LLW-Former scheme: data flow + outputs + loss wiring.

Emits a single static HTML page (inline SVG diagram + tables) showing:
  * the whole model data flow: SAR / GT optical -> Generator -> (fake, subbands)
    -> conditional discriminators (Main-D, HF-D) -> logit + feature maps;
  * every model output and which images feed which loss;
  * a loss table — what each ACTIVE loss compares with what, in which domain,
    on which update (G / D), with its weight.

Loss rows are generated from the live ``config.yaml`` weights (weight > 0 only),
so the scheme stays in sync if you re-tune the recipe.  No checkpoint / torch
forward needed — pure config read.

Run from repo root::

    python -m src.models.llwt_v5.viz_scheme

Then open ``src/models/llwt_v5/output/scheme.html``.
"""
import os

from omegaconf import OmegaConf

OUTPUT = "./src/models/llwt_v5/output/scheme.html"

C = {  # palette (matches viz_blocks stages)
    'in': '#6b7280', 'gen': '#2563eb', 'out': '#16a34a',
    'main': '#7c3aed', 'hf': '#db2777', 'loss': '#d97706',
}


def _box(x, y, w, h, title, sub, color):
    sub_t = (f'<text x="{x + w/2}" y="{y + h - 8}" class="sub">{sub}</text>'
             if sub else '')
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" '
        f'fill="#fff" stroke="{color}" stroke-width="3"/>'
        f'<text x="{x + w/2}" y="{y + 22}" class="bt" fill="{color}">{title}</text>'
        f'{sub_t}'
    )


def _arrow(x1, y1, x2, y2, label='', dashed=False, color='#555'):
    dash = 'stroke-dasharray="6 4"' if dashed else ''
    lab = (f'<text x="{(x1 + x2) / 2}" y="{(y1 + y2) / 2 - 6}" class="al">{label}</text>'
           if label else '')
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
        f'stroke-width="2" marker-end="url(#ah)" {dash}/>{lab}'
    )


def _svg():
    p = ['<svg viewBox="0 0 1180 600" width="100%" style="max-width:1180px">']
    p.append('<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" '
             'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
             '<path d="M0,0 L10,5 L0,10 z" fill="#555"/></marker></defs>')

    # boxes
    p.append(_box(30, 250, 130, 64, 'SAR', 'вход 1×256×256', C['in']))
    p.append(_box(30, 470, 130, 64, 'Оптич. эталон', 'GT 3×256×256', C['in']))
    p.append(_box(210, 200, 220, 150,
                  'Генератор',
                  '', C['gen']))
    # generator internal pipeline text
    p.append('<text x="320" y="248" class="pl">Haar-Stem →</text>')
    p.append('<text x="320" y="266" class="pl">ConvNeXtV2 энкодер →</text>')
    p.append('<text x="320" y="284" class="pl">декодер (+skip s0/s1/s2) →</text>')
    p.append('<text x="320" y="302" class="pl">subband_head → IHaar →</text>')
    p.append('<text x="320" y="320" class="pl">tanh</text>')

    p.append(_box(480, 205, 150, 56, 'fake', '3×256×256 ∈[-1,1]', C['out']))
    p.append(_box(480, 300, 150, 56, 'субполосы sub', '3×4×128×128', C['out']))

    p.append(_box(720, 150, 170, 70, 'Main-D', 'coarse 70² + fine 46²', C['main']))
    p.append(_box(720, 320, 170, 70, 'HF-D', 'ВЧ-остаток opt−gauss', C['hf']))

    p.append(_box(960, 158, 180, 56, 'logits + признаки', 'оценки по патчам', C['main']))
    p.append(_box(960, 328, 180, 56, 'logits + признаки', 'оценки ВЧ', C['hf']))

    # arrows: inputs -> generator
    p.append(_arrow(160, 282, 210, 278, 'вход'))
    # SAR as CONDITION to both D (dashed)
    p.append(_arrow(95, 250, 720, 195, 'SAR = условие', dashed=True, color=C['in']))
    p.append(_arrow(95, 314, 720, 360, 'SAR = условие', dashed=True, color=C['in']))
    # generator -> outputs
    p.append(_arrow(430, 250, 480, 238))
    p.append(_arrow(430, 300, 480, 326))
    # fake -> D heads
    p.append(_arrow(630, 230, 720, 190, 'fake'))
    p.append(_arrow(630, 250, 720, 350, 'fake → highpass'))
    # GT optical -> D heads (real pass)
    p.append(_arrow(160, 480, 720, 210, 'real', dashed=True, color=C['out']))
    p.append(_arrow(160, 500, 720, 380, 'real → highpass', dashed=True, color=C['out']))
    # D -> logits
    p.append(_arrow(890, 185, 960, 186))
    p.append(_arrow(890, 355, 960, 356))
    p.append('</svg>')
    return ''.join(p)


def _loss_rows(L):
    """Build active-loss rows from cfg.loss (weight>0). Returns list of dicts."""
    def w(k):
        return float(L.get(k, 0.0))

    rows = []
    if w('gan_main_weight') > 0:
        rows.append(('Adversarial Main-D', 'G',
                     'Main-D(SAR, <b>fake</b>)', 'цель «real» = 0.9',
                     'LSGAN, патчи', w('gan_main_weight'), C['main']))
        rows.append(('Adversarial Main-D', 'D',
                     'Main-D(SAR, <b>real</b>) → 0.9', 'Main-D(SAR, <b>fake</b>) → 0.0',
                     'LSGAN — разводит оценки', w('gan_main_weight'), C['main']))
    if w('hfd_weight') > 0:
        rows.append(('Adversarial HF-D', 'G',
                     'HF-D(SAR, hp(<b>fake</b>))', 'цель «real» = 0.9',
                     'ВЧ-остаток, LSGAN', w('hfd_weight'), C['hf']))
        rows.append(('Adversarial HF-D', 'D',
                     'HF-D(SAR, hp(<b>real</b>)) → 0.9', 'HF-D(SAR, hp(<b>fake</b>)) → 0.0',
                     'ВЧ-остаток, LSGAN', w('hfd_weight'), C['hf']))
    if w('fm_main_weight') > 0:
        rows.append(('Feature Matching', 'G',
                     'признаки Main-D(<b>real</b>)', 'признаки Main-D(<b>fake</b>)',
                     'L1 по слоям дискриминатора', w('fm_main_weight'), C['main']))
    if w('msssim_weight') > 0:
        rows.append(('MS-SSIM', 'G', '<b>fake</b>', '<b>real</b>',
                     '1−MS-SSIM, многомасштабная структура', w('msssim_weight'), C['out']))
    if w('lpips_weight') > 0:
        rows.append(('LPIPS', 'G', '<b>fake</b>', '<b>real</b>',
                     'перцептивное (AlexNet-признаки)', w('lpips_weight'), C['out']))
    if w('ffl_weight') > 0:
        rows.append(('FFL', 'G', 'FFT(<b>fake</b>)', 'FFT(<b>real</b>)',
                     'частотный домен (focal frequency)', w('ffl_weight'), C['out']))
    if w('per_band_weight') > 0:
        bands = (f"LL×{w('per_band_ll'):g} LH×{w('per_band_lh'):g} "
                 f"HL×{w('per_band_hl'):g} HH×{w('per_band_hh'):g}")
        rows.append(('Per-band wavelet L1', 'G',
                     'субполосы sub (предсказ., до IHaar)', 'Haar(<b>real</b>)',
                     f'L1 по полосам: {bands}', w('per_band_weight'), C['gen']))
    if w('wavelet_weight') > 0:
        rows.append(('Wavelet detail L1', 'G', 'детали(<b>fake</b>)', 'детали(<b>real</b>)',
                     'Haar LH/HL/HH', w('wavelet_weight'), C['gen']))
    if w('lab_chroma_weight') > 0:
        rows.append(('LAB chroma L1', 'G', 'ab(<b>fake</b>)', 'ab(<b>real</b>)',
                     'цветность Lab', w('lab_chroma_weight'), C['out']))
    if w('l1_weight') > 0:
        rows.append(('Pixel L1', 'G', '<b>fake</b>', '<b>real</b>',
                     'пиксельный L1', w('l1_weight'), C['out']))
    if w('patchnce_weight') > 0:
        rows.append(('PatchNCE', 'G', 'энкодер(<b>fake</b>)', 'энкодер(<b>real</b>)',
                     'контрастив (InfoNCE), слои 0-3 — устойчив к рассинхрону пар',
                     w('patchnce_weight'), C['gen']))
    return rows


def main():
    cfg = OmegaConf.load('./src/models/llwt_v5/config.yaml')
    L = cfg.loss
    rows = _loss_rows(L)

    tr = []
    for name, who, a, b, dom, wt, col in rows:
        who_bg = '#1d4ed8' if who == 'G' else '#b91c1c'
        tr.append(
            f'<tr style="border-left:5px solid {col}">'
            f'<td><b>{name}</b></td>'
            f'<td><span class="upd" style="background:{who_bg}">{who}</span></td>'
            f'<td class="op">{a}</td><td class="vs">↔</td><td class="op">{b}</td>'
            f'<td>{dom}</td><td class="wt">{wt:g}</td></tr>'
        )

    outputs_tbl = """
    <table class="io"><tr><th>Что</th><th>Тензор</th><th>Описание</th></tr>
    <tr><td>SAR (вход)</td><td>1×256×256</td><td>радиолокационная амплитуда — единственный реальный вход на инференсе</td></tr>
    <tr><td>Оптич. эталон (GT)</td><td>3×256×256</td><td>цель обучения; на инференсе не нужен</td></tr>
    <tr><td><b>fake</b> (выход G)</td><td>3×256×256</td><td>сгенерированное оптическое, tanh ∈ [-1,1] — итоговый продукт</td></tr>
    <tr><td><b>sub</b> (выход G)</td><td>3×4×128×128</td><td>предсказанные Haar-субполосы [LL,LH,HL,HH] до обратного Haar; для per-band лосса</td></tr>
    <tr><td>Main-D logits</td><td>coarse+fine карты</td><td>оценка реалистичности по патчам (вход [SAR⊕изобр.])</td></tr>
    <tr><td>HF-D logits</td><td>карта</td><td>оценка когерентности ВЧ (вход [SAR⊕hp(изобр.)])</td></tr>
    </table>"""

    html = f"""<!DOCTYPE html><html lang="ru"><head><meta charset="utf-8">
<title>LLW-Former — общая схема модели и лоссов</title>
<style>
  body {{ font-family:-apple-system,Segoe UI,Roboto,sans-serif; margin:24px; color:#111; background:#fafafa; }}
  h1 {{ font-size:22px; }} h2 {{ font-size:16px; margin-top:26px; color:#374151; }}
  text {{ font-family:Segoe UI,sans-serif; }}
  .bt {{ font-size:14px; font-weight:bold; text-anchor:middle; }}
  .sub {{ font-size:10px; fill:#6b7280; text-anchor:middle; }}
  .pl {{ font-size:11px; fill:#1f2937; text-anchor:middle; }}
  .al {{ font-size:11px; fill:#374151; text-anchor:middle; }}
  table {{ border-collapse:collapse; width:100%; max-width:1180px; font-size:13px; background:#fff; }}
  th,td {{ border:1px solid #e5e7eb; padding:6px 9px; text-align:left; vertical-align:top; }}
  th {{ background:#f3f4f6; }}
  .op {{ font-family:monospace; font-size:12px; }}
  .vs {{ text-align:center; color:#9ca3af; font-size:16px; }}
  .wt {{ text-align:center; font-weight:bold; }}
  .upd {{ color:#fff; padding:2px 8px; border-radius:5px; font-size:12px; font-weight:bold; }}
  .io td:first-child {{ font-weight:600; }}
  .leg span {{ padding:2px 9px; border-radius:5px; color:#fff; margin-right:6px; font-size:12px; }}
  .note {{ font-size:12px; color:#4b5563; max-width:1000px; }}
</style></head><body>
<h1>LLW-Former — общая схема: потоки данных, выходы и лоссы</h1>
<p class="leg">
  <span style="background:{C['in']}">вход</span>
  <span style="background:{C['gen']}">генератор</span>
  <span style="background:{C['out']}">выходы / реконструкция</span>
  <span style="background:{C['main']}">Main-D</span>
  <span style="background:{C['hf']}">HF-D</span>
</p>
<p class="note">Пунктир = условие/real-проход. Дискриминатор <b>условный</b>: SAR склеивается с изображением по каналам
([SAR ⊕ opt] для Main-D, [SAR ⊕ hp(opt)] для HF-D). Каждая D-голова прогоняется дважды — на <b>real</b> и на <b>fake</b>.</p>
{_svg()}

<h2>Выходы и картинки</h2>
{outputs_tbl}

<h2>Функция потерь — что с чем сравнивается ({len(rows)} активных членов)</h2>
<table>
<tr><th>Лосс</th><th>Обновл.</th><th>Сравнивает</th><th></th><th>с</th><th>Домен / смысл</th><th>Вес</th></tr>
{''.join(tr)}
</table>
<p class="note">«Обновл.»: <b>G</b> — член входит в потерю генератора, <b>D</b> — дискриминатора.
Итоговая потеря = взвешенная сумма (веса = столбец «Вес», из config.yaml). Отключённые члены (вес 0) не показаны.</p>
</body></html>"""

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"active loss terms: {len(rows)}")
    for r in rows:
        print(f"  [{r[1]}] {r[0]:22s} w={r[5]:g}")
    print("=" * 60)
    print(f"HTML -> {OUTPUT}")


if __name__ == "__main__":
    main()
