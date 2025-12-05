# HFCF Branch Implementation Summary

## Задача (Task)

Реализовать ветвь HFCF (High-Frequency Coding and Filtering) для модели CFRWD-GAN согласно статье:

**Wei, J.; Zou, H.; Sun, L.; Cao, X.; He, S.; Liu, S.; Zhang, Y.** "CFRWD-GAN for SAR-to-Optical Image Translation." *Remote Sensing*, 2023, 15, 2547.

## Что было сделано (What Was Done)

### 1. Исправлена архитектура HFCF Branch

**Проблема**: Предыдущая реализация объединяла высокочастотные компоненты g2 и g3 вместе, что не соответствовало архитектуре из статьи.

**Решение**: 
- ✅ g2 (LH₂, HL₂, HH₂) обрабатывается через Upper Branch (ResNet101-style)
- ✅ g3 (LH₁, HL₁, HH₁) обрабатывается через Lower Branch (ResNet18-style)
- ✅ Выходы выравниваются по пространственному разрешению перед конкатенацией
- ✅ Правильное upsampling 8×8 → 256×256 через 5 слоев

### 2. Детальная трассировка размерностей

```
Input SAR:        B × 1 × 256 × 256

DWT:
├── g1:           B × 1 × 64 × 64    (LL₂)
├── g2:           B × 3 × 64 × 64    (LH₂, HL₂, HH₂)
└── g3:           B × 3 × 128 × 128  (LH₁, HL₁, HH₁)

Preprocessing:
├── g2 → prep:    B × 32 × 32 × 32
└── g3 → prep:    B × 32 × 64 × 64

Branch Processing:
├── Upper (g2):   B × 32 × 32 × 32 → B × 128 × 8 × 8
└── Lower (g3):   B × 32 × 64 × 64 → B × 32 × 64 × 64

Alignment:
├── Upper:        B × 128 × 8 × 8
└── Lower:        B × 32 × 64 × 64 → B × 32 × 8 × 8 (3× AvgPool)

Concatenation:    B × 160 × 8 × 8

Upsampling:       B × 160 × 8 × 8 → B × 3 × 256 × 256 (5× TConv)

Output Optical:   B × 3 × 256 × 256 (RGB)
```

### 3. Блоки обработки (Processing Blocks)

#### Upper Branch (Yellow → Blue → Blue → Yellow → Blue → Blue)
- **Yellow Block**: Bottleneck с downsampling (stride=2)
  - 3 Conv слоя: 3×3, s=1 → 3×3, s=1 → 3×3, s=2
  - Skip connection: 1×1 Conv, s=2
  - Увеличивает каналы, уменьшает разрешение

- **Blue Block**: Bottleneck без downsampling
  - 3 Conv слоя: 3×3, s=1 → 3×3, s=1 → 3×3, s=1
  - Skip connection: Identity
  - Сохраняет каналы и разрешение

#### Lower Branch (Red → Red)
- **Red Block**: BasicBlock
  - 2 Conv слоя: 3×3, s=1 → 3×3, s=1
  - Skip connection: Identity
  - Сохраняет каналы и разрешение

### 4. Добавлена полная документация

Создано 3 подробных документа:

1. **HFCF_IMPLEMENTATION.md**
   - Обзор архитектуры
   - Детальное описание компонентов
   - Обоснование дизайна
   - Инструкции по тестированию

2. **HFCF_ARCHITECTURE_FLOW.md**
   - Визуальные диаграммы потока данных
   - Трассировка размерностей
   - Таблицы изменений разрешения и каналов
   - Сравнение до/после исправления

3. **HFCF_USAGE_GUIDE.md**
   - Примеры использования
   - Рекомендации по обучению
   - Отладка и мониторинг
   - Оптимизация производительности
   - Решение частых проблем
   - Метрики оценки

## Ключевые изменения в коде (Key Code Changes)

### HFCFBranch.forward() - ДО (Before)

```python
# Неправильно: объединение g2 и g3 перед обработкой
hfcf_g2 = self.HFCF_g2_prep(g2)
hfcf_g3 = self.HFCF_g3_prep(g3)
hfcf_g2 = torch.cat([hfcf_g2, hfcf_g3], dim=1)  # ❌ Неверно

out = torch.cat([upper_branch(hfcf_g2), lower_branch(hfcf_g3)], dim=1)  # ❌ Неверно
```

### HFCFBranch.forward() - ПОСЛЕ (After)

```python
# Правильно: раздельная обработка g2 и g3
hfcf_g2 = self.HFCF_g2_prep(g2)  # ✅ Предобработка g2
hfcf_g3 = self.HFCF_g3_prep(g3)  # ✅ Предобработка g3

upper_out = self.upper_branch(hfcf_g2)        # ✅ Upper branch обрабатывает g2
lower_out = self.lower_branch(hfcf_g3)        # ✅ Lower branch обрабатывает g3
lower_out_aligned = self.align_lower(lower_out)  # ✅ Выравнивание

out = torch.cat([upper_out, lower_out_aligned], dim=1)  # ✅ Конкатенация
```

### HFCFUpconvBlock - ДО (Before)

```python
# Неправильно: недостаточно upsampling слоев
TConvBlock(in_channels, in_channels // 2),
TConvBlock(in_channels // 2, in_channels // 4),
TConvBlock(in_channels // 4, in_channels // 8),
# Только 3 upsampling → 32×32 → 256×256 ❌
```

### HFCFUpconvBlock - ПОСЛЕ (After)

```python
# Правильно: 5 upsampling слоев для 8×8 → 256×256
ConvBlock(160, 128, stride=1),        # Подготовка
TConvBlock(128, 128),                 # 8 → 16
TConvBlock(128, 64),                  # 16 → 32
TConvBlock(64, 32),                   # 32 → 64
TConvBlock(32, 16),                   # 64 → 128
TConvBlock(16, 8),                    # 128 → 256
FinalTConvBlock(8, 3)                 # Финальная конволюция + Tanh
```

## Проверка корректности (Correctness Verification)

### Соответствие статье (Paper Compliance)

| Требование | Статус |
|-----------|--------|
| 2-уровневая вейвлет-декомпозиция Haar | ✅ Реализовано |
| Раздельная обработка g2 и g3 | ✅ Исправлено |
| ResNet101-style блоки для g2 | ✅ Реализовано |
| ResNet18-style блоки для g3 | ✅ Реализовано |
| Выравнивание пространственных размеров | ✅ Добавлено |
| Upsampling до исходного разрешения | ✅ Исправлено |
| Feature matching loss (λ=10) | ✅ В config |

### Трассировка размерностей (Dimension Tracking)

Все размерности проверены и задокументированы в `HFCF_ARCHITECTURE_FLOW.md`.

Ключевые проверки:
- ✅ DWT: 256×256 → 64×64, 128×128
- ✅ Upper branch: 32×32 → 8×8
- ✅ Lower branch: 64×64 → 64×64
- ✅ Alignment: 64×64 → 8×8
- ✅ Upsampling: 8×8 → 256×256

## Как использовать (How to Use)

### Быстрый старт (Quick Start)

```python
import torch
from src.models.cfrwd.gen import CFRWDGenerator

# Инициализация
generator = CFRWDGenerator(in_channels=1)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
generator = generator.to(device)

# Тест
sar_input = torch.randn(1, 1, 256, 256).to(device)
optical_output = generator(sar_input)

print(f"Input: {sar_input.shape}")    # (1, 1, 256, 256)
print(f"Output: {optical_output.shape}")  # (1, 3, 256, 256)
```

### Обучение (Training)

См. детали в `HFCF_USAGE_GUIDE.md`:
- Рекомендуемые гиперпараметры
- Функции потерь
- Оптимизация производительности
- Мониторинг и отладка

### Ожидаемые результаты (Expected Results)

На основе статьи (SEN1-2 dataset):
- RMSE: ~32.0
- PSNR: ~19.0 dB
- SSIM: ~0.56
- LPIPS: ~0.40

## Следующие шаги (Next Steps)

1. **Запустить обучение** с исправленной архитектурой
   ```bash
   python src/models/cfrwd/train.py
   ```

2. **Мониторить fusion coefficient**
   - Должен находиться в диапазоне 0.3-0.7
   - Можно проверить: `print(generator.fusion_coeff.item())`

3. **Визуализировать промежуточные результаты**
   - Использовать debug режим в config.yaml
   - См. примеры в HFCF_USAGE_GUIDE.md

4. **Оценить результаты**
   - SSIM, PSNR, LPIPS метрики
   - Визуальное качество переведенных изображений

## Возможные проблемы и решения (Troubleshooting)

### Проблема 1: Out of Memory
**Решение**: Уменьшить batch_size до 1, использовать gradient checkpointing

### Проблема 2: Размытые результаты
**Решение**: Увеличить fm_weight до 20, проверить что fusion_coeff учится

### Проблема 3: Нестабильность обучения
**Решение**: Gradient clipping, label smoothing, spectral normalization

Полный список решений в `HFCF_USAGE_GUIDE.md`.

## Заключение (Conclusion)

Реализация ветви HFCF теперь полностью соответствует архитектуре из статьи CFRWD-GAN:

✅ **Исправлена** раздельная обработка высокочастотных компонентов  
✅ **Добавлено** правильное выравнивание пространственных размерностей  
✅ **Исправлен** upsampling для корректного восстановления разрешения  
✅ **Задокументирована** полная архитектура с детальными диаграммами  
✅ **Подготовлено** руководство по использованию и отладке  

Модель готова к обучению и должна показать результаты, близкие к указанным в статье.

## Контакты автора статьи (Paper Authors)

Для вопросов по оригинальной статье:
- **Corresponding Author**: Huanxin Zou (zouhuanxin@nudt.edu.cn)
- **Institution**: National University of Defense Technology, China

## Ссылки (References)

1. [Оригинальная статья](https://doi.org/10.3390/rs15102547)
2. [ResNet Paper](https://arxiv.org/abs/1512.03385)
3. [Wavelet Transform Theory](https://en.wikipedia.org/wiki/Discrete_wavelet_transform)
4. [GAN Training Tips](https://github.com/soumith/ganhacks)
