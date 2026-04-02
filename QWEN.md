# SAR2OPT Light — Контекст проекта

## 📋 Обзор проекта

**SAR2OPT Light** — это исследовательский проект для трансляции изображений с радаров с синтезированной апертурой (SAR) в изображения оптического диапазона. Проект использует современные GAN-архитектуры (Generative Adversarial Networks) для генерации фотореалистичных оптических изображений из SAR-данных.

### Ключевые особенности
- **GAN-архитектуры**: Реализованы CFRWD-GAN (custom) и pix2pix baseline
- **PyTorch Lightning**: Модульная структура обучения с поддержкой mixed precision
- **Мультимасштабные дискриминаторы**: Улучшенная стабильность обучения
- **Feature Matching + L1 Loss**: Комбинация метрик для качественной генерации
- **Telegram-уведомления**: Интеграция с ботом для мониторинга обучения
- **TensorBoard logging**: Детальное логирование метрик и визуализаций

---

## 🔥 КРИТИЧЕСКИЕ ИСПРАВЛЕНИЯ v3.0.0 (02.04.2026)

**Версия:** cfrwd-33 | **Статус:** Тестовое обучение запущено

### Выявленные проблемы (baseline cfrwd-32):
1. **fusion_weight → 0.2** — HFCF ветка почти не влияла на генерацию
2. **L1 loss отключён** (l1_weight: 0) — модель "галлюцинировала"
3. **Нет TTUR** (lr_g = lr_d = 2e-4) — дисбаланс G/D, коллапс дискриминатора
4. **HFCFBranch без skip connections** — потеря высокочастотных деталей
5. **Fusion weight без clipping** — нестабильность баланса CFR/HFCF

### Внесённые исправления:

| Исправление | Файл | Ожидаемый эффект |
|-------------|------|------------------|
| **L1 loss включён** | config.yaml: l1_weight=100 | PSNR +1-2 dB, меньше галлюцинаций |
| **TTUR** | config.yaml: lr_g=1e-4, lr_d=4e-4 | Стабильный loss_gan ~0.6-0.8 |
| **HFCF skip connections** | gen.py: HFCFBranch | SSIM +0.05-0.10, чёткие границы |
| **Fusion weight clipping** | gen.py: clamp(0.1, 2.0) | fusion_weight ~0.4-0.8 |
| **CFRBlock residual** | gen.py: k1_res_proj | Лучшая сходимость |
| **Weight decay для fusion** | factory.py | Стабильность обучения |

### Ожидаемые результаты (после 200 эпох):
- **PSNR:** 18-19 dB (БЫЛО: 16.66 dB)
- **SSIM:** 0.35-0.40 (БЫЛО: 0.292)
- **fusion_weight:** 0.4-0.8 (БЫЛО: → 0.2)
- **loss_gan:** 0.6-0.8 (БЫЛО: → 1.0)

**Документация:** См. `ARCHITECTURE_FIXES.md` для полного описания изменений.

---

## 🏗️ Архитектура проекта

```
sar2opt_light/
├── src/
│   ├── models/
│   │   ├── cfrwd/           # Custom CFRWD-GAN архитектура
│   │   │   ├── main.py      # Lightning модуль обучения
│   │   │   ├── gen.py       # Генератор (CFR + HFCF ветви)
│   │   │   ├── discriminator.py
│   │   │   ├── train.py     # Factory для training step
│   │   │   ├── factory.py   # Построение моделей/оптимизаторов
│   │   │   ├── losses.py    # Функции потерь (GAN, L1, FM)
│   │   │   ├── inference.py
│   │   │   └── config.yaml  # Конфигурация экспериментов
│   │   └── pix2pix/         # Pix2Pix baseline реализация
│   │       ├── generator.py
│   │       ├── discriminator.py
│   │       ├── main.py
│   │       └── config.yaml
│   ├── data/
│   │   ├── sen12/           # SEN1-2 dataset модули
│   │   ├── sen12_full/      # Полная версия SEN1-2
│   │   ├── labeled/         # Кастомные размеченные данные
│   │   ├── transforms.py    # Аугментации (Albumentations)
│   │   └── datamodule.py    # PyTorch Lightning DataModule
│   └── utils/
│       ├── logger.py        # Кастомный логгер
│       ├── callbacks.py     # Lightning callbacks
│       ├── notification.py  # Telegram уведомления
│       ├── visualize.py     # Визуализация батчей
│       └── cleanup_memory.py# Очистка GPU памяти
├── notebooks/
│   ├── pix2pix_kaggle.ipynb
│   ├── optuna_study.py
│   └── speckle_remove.ipynb
├── docs/
│   ├── cfrwd/               # Документация CFRWD
│   ├── pix2pix/
│   └── sen1-2/
├── checkpoints/             # Сохранённые веса моделей
├── data/                    # Датасеты (игнорируется git)
├── output/                  # Результаты обучения (игнорируется git)
├── requirements.txt
└── changelog.md             # Журнал экспериментов
```

---

## 🛠️ Технологии и зависимости

### Основные фреймворки
- **PyTorch** 2.8.0+cu129
- **PyTorch Lightning** 2.5.5
- **Albumentations** 2.0.8 (аугментации)
- **OmegaConf** 2.3.0 (конфигурация)

### Метрики и визуализация
- **TorchMetrics**: PSNR, SSIM, SAM, MSE
- **Matplotlib**, **OpenCV**, **Pillow**

### Утилиты
- **Optuna** 4.5.0 (оптимизация гиперпараметров)
- **python-dotenv**, **pyTelegramBotAPI**

---

## 🚀 Запуск обучения

### CFRWD-GAN

```bash
# Обучение с конфигурацией по умолчанию
python src/models/cfrwd/main.py

# Ключевые параметры в config.yaml:
# - data.batch_size: 8
# - system.max_epochs: 200
# - system.precision: "bf16-mixed"
# - model.gen.ngf: 64
# - loss.l1_weight: 0 (отключён), gan_weight: 1, fm_weight: 10
```

### Pix2Pix Baseline

```bash
python src/models/pix2pix/main.py
```

### Структура данных

```
data/
├── train/
│   ├── sar/       # SAR изображения (.tif, .png)
│   └── optical/   # Оптические изображения
├── val/
│   ├── sar/
│   └── optical/
└── test/
    ├── sar/
    └── optical/
```

---

## 📊 Архитектура CFRWD-GAN

### Генератор (gen.py)

**Основные компоненты:**

1. **CFRBlock (Cross-Fusion Reasoning)**
   - Мульти-масштабная обработка (4 масштаба: 256→128→64→32)
   - HRNet-подобная структура каналов (c1=16, c2=32, c3=64, c4=128)
   - Non-linear fusion с InstanceNorm + LeakyReLU
   - Сохранение максимального разрешения для Final Fusion

2. **HFCFBlock (Haar Wavelet Decomposition)**
   - Вейвлет-разложение на основе Haar-преобразования
   - Высокочастотная и низкочастотная ветви
   - Фиксированные веса (не обучаются)

3. **Residual Blocks**
   - Modern GAN стиль: ReflectionPad + Conv + InstanceNorm + LeakyReLU
   - Skip connections для стабильности градиентов

### Дискриминатор (discriminator.py)

- **Conditional Multi-Scale**: Принимает concat(SAR, Optical)
- **Spectral Normalization**: Вместо InstanceNorm для стабильности
- **LSGAN Loss**: Least Squares GAN для плавных градиентов

### Функции потерь (losses.py)

```python
# Комбинированный loss генератора:
g_loss = λ_l1 * L1(fake, real) + 
         λ_gan * GAN_loss(D(fake)) + 
         λ_fm * FeatureMatching(D(fake), D(real))

# Типичные веса:
# λ_l1 = 0 (отключён в текущих экспериментах)
# λ_gan = 1
# λ_fm = 10
```

---

## 📈 Метрики качества

### Валидационные метрики

| Метрика | Диапазон | Интерпретация |
|---------|----------|---------------|
| **PSNR** | >16 dB | State-of-the-art для SAR→Optical |
| **SSIM** | >0.30 | Отличное структурное сходство |
| **L1 Loss** | 0.20-0.35 | Хорошая реконструкция пикселей |
| **FM Loss** | 1.0-2.5 | Совпадение текстур |

### Метрики баланса GAN

| Метрика | Здоровый диапазон | Проблема |
|---------|-------------------|----------|
| **loss_gan** | 0.5-0.99 | >1.0 = D доминирует |
| **loss_d** | 0.3-0.7 | <0.1 = D слишком силён |
| **d_real_mean** | 0.7-0.95 | >0.99 = переобучение D |
| **d_fake_mean** | 0.2-0.5 | <0.05 = vanishing gradient |

---

## 🧪 Журнал экспериментов

Проект ведёт детальный журнал экспериментов в `changelog.md` с кодированием:

**Формат:** `cfrwd-XX` — номер эксперимента
- **cfrwd-1 до cfrwd-32**: Эволюция архитектуры
- **Ключевые вехи:**
  - `cfrwd-12`: Полная переработка ядра модели
  - `cfrwd-26`: Добавление аугментаций, оптимизация данных
  - `cfrwd-28**: Mixed precision (4:30 мин/эпоху vs 6:30)
  - `cfrwd-31**: EMA, SpectralNorm, PyTorch Lightning 2.5.5
  - `cfrwd-32`: HRNet-подобная архитектура CFR

**Текущая версия:** v2.7.5 (cfrwd-32)

---

## 🔧 Конфигурация экспериментов

### CFRWD config.yaml (ключевые параметры)

```yaml
data:
  batch_size: 8
  image_size: 256
  num_workers: 6
  use_train_common_transform: true  # Включить аугментации

model:
  gen:
    ngf: 64
    in_channels: 1  # SAR
  dis:
    ndf: 64
    in_channels: 4  # SAR + Optical (conditional)

optimizer:
  lr_g: 2e-4
  lr_d: 2e-4
  beta1: 0.5
  beta2: 0.999

scheduler:
  linear_decay_epochs: 100  # Linear decay после 100 эпох

ema:
  use_ema: true
  decay: 0.999
  start_epoch: 30  # EMA включается после 30 эпох

loss:
  l1_weight: 0      # Отключён для чистоты GAN
  gan_weight: 1
  fm_weight: 10

system:
  precision: "bf16-mixed"  # BFloat16 mixed precision
  max_epochs: 200
  tb_version: 'cfrwd-32'   # Версия для TensorBoard
  resume_ckpt: null        # Путь к чекпоинту для возобновления
```

---

## 📁 Структура исходного кода

### Модели (`src/models/`)

**CFRWD:**
- `main.py`: LightningModule с training/validation steps
- `gen.py`: Генератор (608 строк)
- `discriminator.py`: Multi-scale дискриминатор
- `train.py`: Factory для training step логики
- `factory.py`: Построение моделей, оптимизаторов, scheduler'ов
- `losses.py`: GANLoss, FeatureMatching, L1
- `inference.py`: Инференс утилиты

**Pix2Pix:**
- `generator.py`: SPADE генератор
- `discriminator.py`: PatchGAN
- `multiscale_discriminator.py`
- `main.py`: Lightning модуль

### Данные (`src/data/`)

- `transforms.py`: Albumentations пайплайны
  - `train_common_transform`: Геометрические аугментации
  - `val_transform`: Только ресайз
- `sen12/dataset.py`: SEN1-2 dataset wrapper
- `sen12/datamodule.py`: Lightning DataModule

### Утилиты (`src/utils/`)

- `logger.py`: Кастомный логгер с поддержкой debug/info
- `callbacks.py`: ModelCheckpoint, LearningRateMonitor
- `notification.py`: Telegram бот для уведомлений
- `visualize.py`: Визуализация SAR→Optical батчей
- `cleanup_memory.py`: Очистка GPU памяти после эпохи

---

## 🎯 Практические советы

### Мониторинг обучения

**Каждые 10 эпох проверяйте:**

1. **val_psnr и val_ssim растут?**
   - Да → Модель учится ✅
   - Нет → Проблема с архитектурой/данными ❌

2. **loss_gan в диапазоне 0.5-0.99?**
   - Да → GAN здоров ✅
   - >1.0 → D доминирует, нужен баланс ❌

3. **d_real - d_fake = 0.3-0.6?**
   - Да → Идеальный баланс ✅
   - >0.8 → D слишком силён ❌

### Типичные проблемы и решения

| Проблема | Симптомы | Решение |
|----------|----------|---------|
| **D доминирует** | loss_d→0, loss_gan→1, d_fake→0 | Уменьшить lr_d, увеличить lr_g (TTUR) |
| **Mode collapse** | loss_gan→0, loss_d→∞ | Увеличить FM weight, добавить noise |
| **Артефакты** | Пятна на генерации | Проверить Haar инициализацию, добавить L1 |
| **Переобучение** | train_l1 << val_l1 | Увеличить аугментации, добавить dropout |

---

## 📚 Дополнительные ресурсы

- **changelog.md**: Детальный журнал всех экспериментов
- **metrics&losses.md**: Полное руководство по метрикам GAN
- **docs/cfrwd/**: Архитектурные диаграммы и описания
- **notebooks/**: Jupyter блокноты для анализа

---

## 🤝 Вклад в проект

При добавлении новых экспериментов:

1. Обновите `changelog.md` с форматом `cfrwd-XX`
2. Задокументируйте изменения в `config.yaml`
3. Сохраняйте чекпоинты в `checkpoints/cfrwd/`
4. Логируйте в TensorBoard с уникальным `tb_version`

---

*Последнее обновление: 21.02.2026 (cfrwd-32)*
