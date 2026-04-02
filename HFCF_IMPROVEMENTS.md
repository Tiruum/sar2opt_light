# Улучшения архитектуры HFCF ветви для SAR-to-Optical трансляции

## Проблема
HFCF (High-Frequency Cross-Fusion) ветвь имела коэффициент обучения ~0.2, что означало, что модель считала её бесполезной и практически не использовала.

## Решение

### 1. Адаптивное пространственное объединение ветвей (AdaptiveBranchFusion)

**Было:** Скалярный `fusion_weight`, который мог свободно уменьшаться до минимума (0.1), позволяя модели полностью игнорировать HFCF ветвь.

**Стало:** Пространственно-вариативная attention маска, которая:
- Вынуждает сеть использовать ОБЕ ветви
- Выбирает лучшую ветвь для КАЖДОГО пикселя отдельно
- Не позволяет "обнулить" всю ветвь целиком

```python
class AdaptiveBranchFusion(nn.Module):
    # Генерирует attention маску [B, 3, H, W] из обеих ветвей
    # fused = attention * CFR + (1 - attention) * HFCF
```

**Инициализация:** Attention инициализируется со смещением к CFR (~0.7), но HFCF получает ~0.3, что обеспечивает баланс на старте.

### 2. Channel Attention для DWT коэффициентов

Добавлено адаптивное взвешивание частотных полос:
```python
self.attention_g2 = ChannelAttention(freq_c, reduction=4)
self.attention_g3 = ChannelAttention(freq_c, reduction=4)
```

Это позволяет сети автоматически определять важные частоты для SAR-to-Optical задачи.

### 3. Cross-Frequency Interaction Blocks

**Было:** Независимые streams (top/bottom), которые обрабатывали частоты отдельно.

**Стало:** Три блока cross-frequency interaction с bidirectional attention:
```python
class CrossFrequencyBlock(nn.Module):
    # high_proc и mid_proc обмениваются информацией через gating механизмы
    # high_out = high_feat + gate_high * mid_proc
    # mid_out = mid_feat + gate_mid * high_proc
```

### 4. Упрощенный Preprocess с лучшим градиентным потоком

**Было:** Сложный `HFCFPreprocess` с множеством слоев.

**Стало:** Простой последовательный блок:
```python
self.pre_g2 = nn.Sequential(
    nn.Conv2d(freq_c, hidden_dim, kernel_size=3, stride=1, padding=1),
    nn.InstanceNorm2d(hidden_dim),
    nn.ReLU(inplace=True)
)
```

### 5. Dense Skip Connections от DWT

Добавлены прямые skip connections со всех уровней DWT:
- `skip_g2_fine` - для уровня 64x64
- `skip_g2_mid` - для уровня 128x128  
- `skip_g3_coarse` - для уровня 256x256

### 6. Residual Connection в декодере

Добавлен residual путь для улучшения градиентного потока:
```python
base_out = self.final(dec)
residual = self.residual_proj(dec)
out = base_out + residual * 0.1
```

## Ожидаемые результаты

1. **Fusion weight больше не будет падать до 0.1** - attention маска вынуждена выбирать между ветвями для каждого пикселя
2. **HFCF ветвь будет использоваться для текстур и краев** - там где высокочастотные детали критичны
3. **CFR ветвь останется доминирующей для глобальной структуры** - attention изначально смещен к CFR
4. **Улучшенная видимость в логах** - attention_map визуализируется вместе с результатами

## Изменения в файлах

### `src/models/cfrwd/gen.py`
- Добавлен класс `ChannelAttention`
- Добавлен класс `GatedFusion`
- Полностью переписан `HFCFBranch`
- Добавлен класс `CrossFrequencyBlock`
- Добавлен класс `AdaptiveBranchFusion`
- Модифицирован `CFRWDGenerator` для использования adaptive fusion

### `src/models/cfrwd/main.py`
- Обновлен `on_train_epoch_end` для обработки attention_map

### `src/utils/visualize.py`
- Добавлена поддержка визуализации attention_map
- Расширена сетка для отображения дополнительных колонок

## Мониторинг

В логах теперь доступны:
- `fusion/fusion_weight` - среднее значение attention маски (показывает баланс ветвей)
- Визуализация attention_map каждые N эпох
- Отдельные изображения для CFR и HFCF ветвей

## Рекомендации по обучению

1. **Первые 10-20 эпох:** Attention будет около 0.5-0.7 (CFR доминирует)
2. **Эпохи 20-50:** Network научится использовать HFCF для текстур
3. **Эпохи 50+:** Stable balance с пространственной вариативностью

Если attention сходится к крайним значениям (>0.9 или <0.1), рассмотрите:
- Увеличение `fm_weight` для улучшения feature matching
- Добавление diversity loss для поощрения использования обеих ветвей
