import pandas as pd
from omegaconf import OmegaConf
import os

cfg = OmegaConf.load('src/models/cfrwd/config.yaml')


def clean_csv_logs(exp_name):
    """
    Читает CSV-логи Lightning CSVLogger и объединяет строки одной эпохи в одну.
    
    Lightning пишет train и val метрики в разные строки (они логируются в разные моменты).
    Эта функция мержит их по колонке 'epoch', беря первое непустое значение для каждой метрики.

    Args:
        exp_name: Имя эксперимента (название папки внутри csv_logs/).
    
    Returns:
        pd.DataFrame с одной строкой на эпоху, или None при ошибке.
    """
    log_dir = f'{cfg.system.output_dir}/csv_logs/{exp_name}'
    csv_path = os.path.join(log_dir, 'metrics.csv')
    output_csv_path = os.path.join(log_dir, 'clean_metrics.csv')

    if not os.path.exists(csv_path):
        print(f"❌ metrics.csv не найден: {csv_path}")
        return None

    print(f"📁 Читаю: {csv_path}")

    try:
        df = pd.read_csv(csv_path)

        if 'epoch' not in df.columns:
            print("❌ В CSV нет колонки 'epoch'.")
            return None

        rows_before = len(df)

        # Убираем колонки, которые дублируют ручное логирование (lr-Adam от LearningRateMonitor)
        lr_monitor_cols = [c for c in df.columns if c.startswith('lr-')]
        if lr_monitor_cols:
            df = df.drop(columns=lr_monitor_cols)

        # Объединяем строки по эпохе: берём первое непустое значение для каждой колонки
        df_clean = df.groupby('epoch', dropna=False).first().reset_index()

        # Убираем строки без номера эпохи (бывают от initial lr log)
        df_clean = df_clean.dropna(subset=['epoch']).reset_index(drop=True)
        df_clean['epoch'] = df_clean['epoch'].astype(int)

        # Убираем полностью пустые колонки
        df_clean = df_clean.dropna(axis=1, how='all')

        print(f"📈 {rows_before} строк → {len(df_clean)} строк (по одной на эпоху)")

        # Сохраняем
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        df_clean.to_csv(output_csv_path, index=False)
        print(f"✅ Сохранено: {output_csv_path}")

        # Статистика
        metric_cols = [c for c in df_clean.columns if c not in ('epoch', 'step')]
        print(f"📊 Эпох: {len(df_clean)}, метрик: {len(metric_cols)}")

        return df_clean

    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return None


def analyze(exp_name, df):
    """
    Выводит лучшие значения ключевых метрик.

    Args:
        exp_name: Имя эксперимента.
        df: DataFrame после clean_csv_logs().
    """
    if df is None or df.empty:
        print("❌ Нет данных для анализа")
        return

    print(f"\n🏆 Лучшие метрики [{exp_name}]:")

    # (колонка, режим, формат, единица)
    targets = [
        ('val/psnr',    'max', '.2f', 'dB'),
        ('val/ssim',    'max', '.4f', ''),
        ('val/rmse',    'min', '.4f', ''),
        ('val/sam',     'min', '.4f', ''),
        ('val/loss_l1', 'min', '.4f', ''),
    ]

    for col, mode, fmt, unit in targets:
        if col not in df.columns:
            continue
        idx = df[col].idxmax() if mode == 'max' else df[col].idxmin()
        val = df.loc[idx, col]
        epoch = int(df.loc[idx, 'epoch'])
        print(f"  • {col}: {val:{fmt}}{(' ' + unit) if unit else ''} (epoch {epoch})")


if __name__ == '__main__':
    experiments = ['cfrwd-34']

    for exp in experiments:
        print(f"\n{'='*60}")
        print(f"Обработка: {exp}")
        print(f"{'='*60}")

        df_clean = clean_csv_logs(exp)
        if df_clean is not None:
            analyze(exp, df_clean)