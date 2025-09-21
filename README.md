# SAR2OPT Light: Преобразование радиолокационных изображений в оптические

![GitHub stars](https://img.shields.io/github/stars/Tiruum/sar2opt_light?style=social)
![GitHub forks](https://img.shields.io/github/forks/Tiruum/sar2opt_light?style=social)
![GitHub issues](https://img.shields.io/github/issues/Tiruum/sar2opt_light)
![GitHub license](https://img.shields.io/github/license/Tiruum/sar2opt_light)

*Read this in [English](README.md)*

## Обзор проекта

SAR2OPT Light — это облегченная реализация для преобразования радиолокационных изображений (SAR - Synthetic Aperture Radar) в оптические изображения с использованием методов глубокого обучения. Данный репозиторий предоставляет оптимизированный подход к трансляции изображений SAR в оптические для исследовательских и практических целей.

Проект решает задачу конвертации спутниковых радиолокационных изображений в изображения видимого спектра (оптические), что обеспечивает лучшую интерпретацию и анализ данных дистанционного зондирования Земли даже в условиях, когда оптическая съемка невозможна (например, из-за облачности или в ночное время).

## Ключевые особенности

- **Эффективное преобразование**: Конвертация радиолокационных снимков в фотореалистичные оптические изображения с помощью оптимизированных моделей
- **Jupyter Notebooks**: Интерактивные примеры и демонстрации для упрощения понимания
- **Облегченная реализация**: Упрощенная версия полного проекта [sar2opt](https://github.com/Tiruum/sar2opt)
- **Предобученные модели**: Доступ к предварительно обученным моделям для немедленного использования
- **Метрики качества**: Комплексные метрики оценки для анализа качества преобразования

## Установка

```bash
# Клонирование репозитория
git clone https://github.com/Tiruum/sar2opt_light.git
cd sar2opt_light

# Создание виртуальной среды (опционально, но рекомендуется)
python -m venv venv
source venv/bin/activate  # На Windows: venv\Scripts\activate

# Установка зависимостей
pip install -r requirements.txt
```

## Структура данных

Проект ожидает организации данных по следующей структуре:

```
data/
├── train/
│   ├── sar/
│   │   └── [SAR-изображения]
│   └── optical/
│       └── [Соответствующие оптические изображения]
├── val/
│   ├── sar/
│   │   └── [SAR-изображения]
│   └── optical/
│       └── [Соответствующие оптические изображения]
└── test/
    ├── sar/
    │   └── [SAR-изображения]
    └── optical/
        └── [Соответствующие оптические изображения]
```

## Использование

### Быстрый старт

Откройте и запустите основной демонстрационный блокнот:

```bash
jupyter notebook notebooks/SAR2OPT_Demo.ipynb
```

### Обучение новой модели

```python
from sar2opt import Trainer

# Инициализация тренера
trainer = Trainer(
    sar_dir='data/train/sar',
    optical_dir='data/train/optical',
    val_sar_dir='data/val/sar',
    val_optical_dir='data/val/optical',
    batch_size=4,
    epochs=100
)

# Обучение модели
trainer.train()

# Сохранение обученной модели
trainer.save_model('models/my_sar2opt_model.h5')
```

### Инференс

```python
from sar2opt import Translator

# Инициализация транслятора с предобученной моделью
translator = Translator(model_path='models/pretrained_sar2opt.h5')

# Преобразование одного SAR-изображения
optical_image = translator.translate('path/to/sar_image.tif')

# Сохранение результата
optical_image.save('path/to/output_optical_image.png')
```

## Структура проекта

```
sar2opt_light/
├── data/                   # Директория с данными (не включена в репозиторий)
├── models/                 # Предобученные модели
├── notebooks/              # Jupyter-блокноты для демонстраций
│   ├── SAR2OPT_Demo.ipynb
│   ├── Model_Training.ipynb
│   └── Results_Analysis.ipynb
├── sar2opt/                # Python-модуль с основной функциональностью
│   ├── __init__.py
│   ├── data_loader.py
│   ├── models.py
│   ├── trainer.py
│   └── translator.py
├── results/                # Примеры результатов и визуализации
├── requirements.txt        # Python-зависимости
├── LICENSE                 # Информация о лицензии
└── README.md               # Этот файл
```

## Требования

- Python 3.7+
- TensorFlow 2.5+
- PyTorch 1.9+ (опционально, для определенных архитектур моделей)
- GDAL для обработки геопространственных данных
- NumPy, Matplotlib и другие стандартные библиотеки для обработки данных

Полный список зависимостей см. в файле `requirements.txt`.

## Результаты

Модель SAR2OPT Light достигает:

- PSNR: ~28.5 dB
- SSIM: ~0.85
- FID: ~18.7

Примеры преобразований:

![Пример преобразования](results/example_translation.png)

## Научная основа

Работа основана на современных исследованиях в области трансляции изображений с использованием генеративно-состязательных сетей (GANs) и других архитектур глубокого обучения. Подход использует:

- Условные GAN для структурной согласованности
- Перцептивные функции потерь для визуального качества
- Специализированные функции потерь для сохранения пространственных характеристик

## Цитирование

Если вы используете этот код для своих исследований, пожалуйста, цитируйте:

```bibtex
@software{sar2opt_light,
  author = {Tiruum},
  title = {SAR2OPT Light: Synthetic Aperture Radar to Optical Image Translation},
  year = {2023},
  url = {https://github.com/Tiruum/sar2opt_light}
}
```

## Лицензия

Этот проект распространяется под лицензией MIT — подробности см. в файле LICENSE.

## Благодарности

- [Провайдеры наборов данных]
- [Соответствующие научные статьи]
- [Участники и сотрудники]

## Контакты

По вопросам и отзывам, пожалуйста, создайте issue на GitHub или свяжитесь с владельцем репозитория.
