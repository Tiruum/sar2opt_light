# RESUME — заметка будущему себе

> Если ты открыл это через полгода-год и ничего не помнишь — читай отсюда.
> Это карта проекта: что сделано, где лежит, что нельзя ломать.

## TL;DR — что это

**WaveNeXt** — нейросеть, переводящая радар (**SAR**, Sentinel-1, 1 канал) в
оптику (**Sentinel-2-like**, 3 канала), 256×256. Это магистерская ВКР
(защищена на 9/10, июнь 2026).

Название = **Wave**let + Conv**NeXt**:
- фиксированный Хаар-вейвлет на входе/выходе (не обучается, `register_buffer`);
- бэкбон ConvNeXt V2-Base (~98M параметров, перенос с ImageNet-22k);
- GAN-обучение с **высокочастотным дискриминатором HF-D** (новизна работы).

Опубликовано: **[huggingface.co/umpaoflumpia/WaveNeXt](https://huggingface.co/umpaoflumpia/WaveNeXt)**
+ **[github.com/Tiruum/sar2opt_light](https://github.com/Tiruum/sar2opt_light)**.

Метрики (SEN1-2 held-out): **Base** PSNR 18.54 / SSIM 0.432 / FID 58.5 / LPIPS 0.241.

---

## Где что лежит (КАРТА)

### В GitHub (всё запушено в `master`)
- `src/models/wavenext/` — **единственная** модель. Всё остальное удалено.
  - `gen.py` — генератор `WaveNeXtGenerator`
  - `dis.py` — дискриминаторы `WaveNeXtDiscriminator` (Main + HF-D)
  - `main.py` — Lightning-модуль `WaveNeXtLightningModule` (цикл, лоссы, EMA)
  - `losses.py`, `blocks.py`, `factory.py`, `train.py`, `inference.py`
  - `export_hf.py`, `export_onnx.py` — экспорт на Hugging Face
  - `config.yaml` — **главный конфиг** (путь захардкожен, CLI-оверрайда нет)
  - `ARCHITECTURE.md` — полный референс архитектуры (RU, с картой `файл:строка`)
- `src/data/sen12_full/`, `src/data/sen12_full_align/` — датамодули
- `src/utils/` — логгер, EMA-коллбек, очистка памяти, Telegram-нотификации
- `docs/diploma/` — текст ВКР (LaTeX)
- `README.md` — публичное описание репо
- `CLAUDE.md` / `AGENTS.md` — инструкции для AI-ассистентов

### Только локально (НЕ в гите, потеряется если удалить папку!)
| Что | Размер | Зачем |
|--|--|--|
| `checkpoints/llwt_v45_base/llwt-v0.4.6-base/epoch=199-psnr=18.5361.ckpt` | ~1.8G | **Веса за Base-моделью на HF.** Главный чекпоинт. |
| `checkpoints/llwt_v45/llwt-v0.5.1-hfd/` | — | Tiny-вариант |
| `output/hf_export/` | 747M | Готовые артефакты для HF (safetensors, onnx, model card) |
| `data/` | 52G | Датасет SEN1-2 |
| `.env` | 1K | Telegram-токены (gitignored) |
| `.venv` / `.venv_wsl` | ~9.5G | Виртуалки |

> **ВАЖНО:** чекпоинты и `.env` НЕ в гите. Прежде чем удалять/переустанавливать
> папку — сохрани `checkpoints/` + `.env`. Остальное (код) восстановится из
> `git clone`, датасет — скачивается заново.

### На Hugging Face (`umpaoflumpia/WaveNeXt`)
- `generator.safetensors` (391M, 425 тензоров) — веса Base
- `model.onnx` (391M, fp32, opset 17) — рантайм-независимый граф
- `config.json`, `LICENSE` (CC-BY-NC-4.0), `README.md` (model card)
- `hfd-showcase.jpg` — картинка-демо (baseline vs HF-D)

---

## Старое наследие — git-тег `archive/full-lineage-v1`

Весь экспериментальный путь (cfrwd, llwt v3/v4/v45, sarformer, pix2pix,
diffusion и пр.) **удалён из рабочего дерева** и сохранён в теге
`archive/full-lineage-v1` (есть и на origin). Оттуда воспроизводится
абляционная таблица диплома. Восстановить:
```powershell
git checkout archive/full-lineage-v1 -- <path>   # отдельный файл
git worktree add ../sar2opt_archive archive/full-lineage-v1   # всё дерево
```

---

## Как запустить (из корня репо)

```powershell
# окружение
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt

# обучение (config.yaml = Base + HF-D, тёплый старт)
python -m src.models.wavenext.train

# быстрый смоук (один шаг, config_smoke.yaml)
python -m src.models.wavenext.smoke_train_step

# инференс (грузит Base-чекпоинт)
python -m src.models.wavenext.inference

# экспорт на HF
python -m src.models.wavenext.export_hf  --ckpt <path.ckpt> --out output/hf_export
python -m src.models.wavenext.export_onnx --ckpt <path.ckpt> --out output/hf_export

# мониторинг
tensorboard --logdir output/llwt_v45/tb_logs
```

---

## Переключатели (всё в `config.yaml`, без правки кода)

- **Ёмкость:** `model.gen.backbone` → `...convnextv2-base-22k-224` (Base, `batch_size: 6`)
  или `...tiny-22k-224` (Tiny, `batch_size: 8`). Каналы стадий авто-выводятся из бэкбона.
- **HF-D (новизна):** `loss.hfd_weight` = `1.0` (вкл) / `0` (выкл → чистый baseline для абляции).
  Также `model.dis.highfreq.enabled`. λ=0 ≡ baseline байт-в-байт.
- `system.tb_version` — имя всех артефактов, меняй на каждый эксперимент.
- `system.weights_ckpt` — тёплый старт (strict=False); `system.resume_ckpt` — полное возобновление / `null`.
- `data.dataset` — `sen12_full` (raw) или `sen12_full_align` (ECC-выровненный).

---

## ⚠️ Что НЕЛЬЗЯ ломать (грабли, на которые уже наступали)

1. **`lee_despeckle()` в `blocks.py` использует sort-based квантиль, НЕ `torch.quantile`.**
   `torch.quantile` не экспортируется в ONNX (`aten::quantile`). Замена численно
   идентична (diff 4.8e-7). НЕ откатывай на `torch.quantile` — сломает ONNX-экспорт.
2. **ONNX экспортируется с `dynamo=False`** (legacy-экспортёр). Новый dynamo-путь
   падает на `onnxscript ModuleNotFoundError` в torch 2.9.
3. **fp16/int8 ONNX НЕ работают** — fp16 падает в onnxconverter-common, int8 убивает
   выход (parity 1.5 для conv-модели). Опубликован только **fp32**. Флаги `--fp16/--int8`
   опциональны и по умолчанию выключены — не включай без причины.
4. **Префиксы путей артефактов = `llwt_v45`** (легаси от тёплого старта). Это
   намеренно, не переименовывай — иначе порвутся пути в конфиге.
5. **Нет pixel-L1 в лоссе** — L1 живёт только в вейвлет-базисе (per-band Haar L1).
   Это часть дизайна (anti-blur), не баг.
6. **`src/utils/nsst_torch.py` + `NSST.py`** — оставлены под будущие эксперименты,
   но их lazy-import `SpeckleAwareModule`/`CBAM` из удалённого `cfrwd` — висячий.
   Восстанови из тега `archive/full-lineage-v1` перед использованием.

---

## Данные (SEN1-2)

Структура: `data/sen12_full/<scene>/s1/<file>` (SAR) + `.../s2/<file>` (оптика).
Пары по имени (`_s1_` → `_s2_`). Датасет: [SEN1-2](https://mediatum.ub.tum.de/1436631), research-only.

**Обучались на репрезентативном подмножестве из 5 сцен** (`5, 45, 52, 84, 100`,
поле `data.scenes`) — взято из статьи CFRWD-GAN как representative. Не на полном
датасете. Генерализация на незнакомые регионы/сенсоры не гарантирована.

---

## Журнал

Все эксперименты — в `changelog.md` (run id, изменения, результат, версии `vX.Y.Z`).
Если запускаешь новое — пиши туда.
