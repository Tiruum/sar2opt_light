# WaveNeXt — SAR → Optical

Перевод одноканального радиолокационного изображения (**SAR**, Sentinel-1) в
трёхканальную оптику (**Sentinel-2-like**). Магистерская ВКР.

**WaveNeXt** = *Wavelet + ConvNeXt*. Имя точное:

- **Вейвлет** — фиксированный двухуровневый Хаар-стем (вместо patch-embed) и
  обратный Хаар на голове (`register_buffer`, **не** обучаемый лифтинг).
- **ConvNeXt V2** — свёрточный бэкбон (**не** трансформер).
- Состязательное обучение GAN; главная новизна — **высокочастотный дискриминатор
  HF-D** (судит остаток `x − gaussian_blur(x)`, только на обучении, нулевая
  стоимость на инференсе).

Полное устройство с картой `файл:строка` — `src/models/wavenext/ARCHITECTURE.md`.

## Результаты (SEN1-2 held-out val)

| Вариант | PSNR↑ | SSIM↑ | FID↓ | LPIPS↓ |
|--|--|--|--|--|
| **WaveNeXt Base** | **18.54** | **0.432** | **58.5** | **0.241** |
| WaveNeXt Tiny | 17.28 | 0.369 | 73.0 | 0.311 |

Веса (Base) + ONNX + model card: **[umpaoflumpia/WaveNeXt](https://huggingface.co/umpaoflumpia/WaveNeXt)** на Hugging Face.

## Структура репозитория

- `src/models/wavenext/` — единственная модель: генератор, дискриминаторы (Main + HF-D),
  лоссы, тренировочный цикл, инференс, экспорт. `ARCHITECTURE.md` — референс.
- `src/data/sen12_full/`, `src/data/sen12_full_align/` — датамодули SEN1-2 (raw / ECC-выровненный).
- `src/utils/` — логгер, EMA-коллбек, очистка памяти, нотификации.
- `scripts/` — вспомогательные скрипты (выравнивание пар и т.п.).
- `docs/diploma/` — текст ВКР.

> Полное наследие экспериментов (cfrwd, llwt v3/v4/v45, sarformer и др.) удалено из
> рабочего дерева и сохранено в git-теге **`archive/full-lineage-v1`** — оттуда
> воспроизводится абляционная таблица диплома.

## Установка

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

GPU с CUDA нужен для обучения (RTX-класс, ~18.5 ГБ VRAM для Base @ bf16).

## Быстрый старт

```powershell
# обучение (config.yaml = Base backbone + HF-D, тёплый старт)
python -m src.models.wavenext.train

# одношаговый смоук (быстрый, config_smoke.yaml)
python -m src.models.wavenext.smoke_train_step

# инференс (грузит Base-чекпоинт через load_generator)
python -m src.models.wavenext.inference

# экспорт весов для Hugging Face (ckpt -> safetensors + config.json)
python -m src.models.wavenext.export_hf  --ckpt <path/to.ckpt> --out output/hf_export

# экспорт в ONNX (fp32, opset 17, с parity-проверкой)
python -m src.models.wavenext.export_onnx --ckpt <path/to.ckpt> --out output/hf_export

# TensorBoard
tensorboard --logdir output/llwt_v45/tb_logs
```

## Конфиг — переключатели (без правки кода)

`src/models/wavenext/config.yaml` (путь захардкожен в `train.py`/`main.py`/`factory.py`):

- **Ёмкость** — `model.gen.backbone`: `facebook/convnextv2-base-22k-224` (Base, дефолт,
  `data.batch_size: 6`) или `...-tiny-22k-224` (Tiny, `batch_size: 8`). Каналы стадий
  авто-выводятся из бэкбона.
- **HF-D** (новизна) — `loss.hfd_weight`: `1.0` (вкл) или `0` (выкл → чистый baseline / абляция).
- `system.tb_version` — имя для всех артефактов; меняй на каждый эксперимент.
- `system.weights_ckpt` — тёплый старт G/D (`strict=False`); `system.resume_ckpt` — полное возобновление или `null`.

## Данные (SEN1-2)

`data.dataset` выбирает датамодуль: `sen12_full` (raw) или `sen12_full_align`
(градиентный ECC-выровненный зеркальный набор; SAR байт-в-байт, оптика деформирована).

```text
data/sen12_full/<scene>/s1/<file>   # SAR
data/sen12_full/<scene>/s2/<file>   # optical
```
Пары сопоставляются по имени (`_s1_` → `_s2_`). [SEN1-2](https://mediatum.ub.tum.de/1436631), research-only.

> **Обучение только на подмножестве из 5 сцен** SEN1-2 (`5, 45, 52, 84, 100`,
> поле `data.scenes`), не на полном датасете. Это узкоспециализированная тезисная
> модель — лучшие результаты на похожей местности (снег/лёд), генерализация на
> неизвестные регионы не гарантируется.

## Артефакты

По `cfg.system.tb_version` (префикс путей `llwt_v45` — легаси от тёплого старта):

- чекпоинты: `checkpoints/llwt_v45/<tb_version>/` (top-k по `val/psnr` + `last`)
- TensorBoard: `output/llwt_v45/tb_logs/<tb_version>/`
- изображения эпох: `output/llwt_v45/images/<tb_version>/`

## Прочее

- **Telegram** (опц.): `.env` ключи `TELEGRAM_BOT_TOKEN` / `TELEGRAM_RECIEVER_USER_ID`. Нет ключей — молча выкл.
- **Журнал экспериментов**: `changelog.md` (run id, изменения, результат; версии `vX.Y.Z`).

## Лицензия

**CC-BY-NC-4.0** (non-commercial). Веса производны от ConvNeXt V2 (Meta, CC-BY-NC-4.0)
и обучены на SEN1-2 (research-only) — non-commercial-условия наследуются.
