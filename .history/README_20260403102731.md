# SAR2OPT Light

## Статус проекта

Проект в активной разработке только по направлению **CFRWD**.

- Все новые эксперименты, изменения архитектуры и оптимизации делаются в CFRWD-пайплайне.
- `pix2pix` сохранен как исторический артефакт для воспроизводимости и сравнений.
- Новая функциональность в `pix2pix` не разрабатывается (кроме редких правок совместимости).

## TL;DR

1. Рабочее направление: `src/models/cfrwd`.
2. Базовый запуск обучения: `python -m src.models.cfrwd.train`.
3. Логи и артефакты: `output/cfrwd` и `checkpoints/cfrwd` (через значения из конфига).
4. `pix2pix` считается legacy-слоем и не является текущей целевой веткой исследований.

## Что в репозитории

Ключевые каталоги:

- `src/models/cfrwd` - активная модель и тренировочный цикл.
- `src/data/sen12` - датасет и datamodule для SEN12.
- `docs/cfrwd` - материалы по архитектуре CFRWD.
- `output/cfrwd` - TensorBoard/CSV/изображения/профилирование.
- `checkpoints/cfrwd` - чекпоинты активных запусков.
- `src/models/pix2pix`, `docs/pix2pix`, `output/pix2pix`, `checkpoints/pix2pix` - архивный legacy-контур.

## Требования окружения

- Python (рекомендуется 3.10+).
- Установленные зависимости из `requirements.txt`.
- GPU с CUDA для полноценного обучения (CPU-режим возможен, но медленный).

Пример установки:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Быстрый старт (CFRWD)

1. Подготовьте окружение и зависимости.
2. Проверьте структуру данных в `data/sen12`.
3. Убедитесь, что конфиг доступен по пути `src/models/cfrwd/config.yaml`.
4. Запустите обучение.

Команда запуска:

```powershell
python -m src.models.cfrwd.train
```

## Важный момент по конфигу

Код CFRWD ожидает конфиг в фиксированном пути:

- `src/models/cfrwd/config.yaml`

Этот путь используется в нескольких местах (`train.py`, `main.py`, `factory.py`, `logger.py`, `clean_csv_logs.py`).
Если файла нет, запуск обучения и часть утилит не стартуют.

Рекомендуемая практика:

1. Держать один канонический конфиг в `src/models/cfrwd/config.yaml`.
2. Менять `tb_version` для каждого нового эксперимента.
3. Для возобновления обучения задавать `system.resume_ckpt`.

## Ожидаемая структура данных SEN12

Минимально ожидаемая структура (`src/data/sen12/dataset.py`):

```text
data/sen12/
	agri/
		s1/
		s2/
	barrenland/
		s1/
		s2/
	grassland/
		s1/
		s2/
	urban/
		s1/
		s2/
```

Примечания:

- Датасет сопоставляет пары SAR/Optical по имени файла (например, `_s1_` -> `_s2_`).
- Поддерживаются основные форматы изображений: `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`, `.bmp`.

## Артефакты и мониторинг

Во время обучения сохраняются:

- чекпоинты (`save_top_k=3` + `last`) в `checkpoints/cfrwd/<tb_version>`;
- TensorBoard-логи в `output/cfrwd/tb_logs/<tb_version>`;
- CSV-логи в `output/cfrwd/csv_logs/<tb_version>`;
- профилировщик в директорию `cfg.system.profiler_dir`;
- визуализации эпох в `cfg.system.images_dir/<tb_version>` при `system.image_freq != 0`.

Запуск TensorBoard:

```powershell
tensorboard --logdir output/cfrwd/tb_logs
```

Очистка и агрегация CSV-логов:

```powershell
python src/utils/clean_csv_logs.py
```

## Уведомления в Telegram (опционально)

Поддерживаются уведомления через `.env`:

- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_RECIEVER_USER_ID`

Если переменные не заданы, уведомления будут неактивны.

## Статус инференса

Файл `src/models/cfrwd/inference.py` присутствует, но на текущий момент не содержит рабочего CLI-пайплайна.
Текущий фокус репозитория - обучение/валидация и исследовательские итерации CFRWD.

## Политика вклада

1. Все новые эксперименты и улучшения направлять в CFRWD.
2. Изменения в `pix2pix` делать только при необходимости совместимости или воспроизводимости.
3. Для каждого нового запуска фиксировать цель, изменения и результат в журнале экспериментов (`changelog.md`).

## Legacy-зона: pix2pix

`pix2pix` оставлен как исторический артефакт:

- для ретроспективных сравнений;
- для воспроизводимости старых результатов;
- без активного feature-development.