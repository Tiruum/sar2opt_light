from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
from omegaconf import OmegaConf
import os

cfg = OmegaConf.load('src/models/cfrwd/config.yaml')

def tb2csv(exp_name):
    """
    Экспортирует все скалярные метрики из TensorBoard логов в один CSV файл.
    Корректно обрабатывает метрики с разным количеством записей и объединяет
    строки с одинаковым шагом.

    Args:
        exp_name: Имя эксперимента.
    """
    log_dir = f'{cfg.system.output_dir}/tb_logs/{exp_name}'
    output_csv_path = f'{cfg.system.output_dir}/tb_logs_csv/{exp_name}.csv'
    
    # Проверяем существование директории
    if not os.path.exists(log_dir):
        print(f"❌ Директория с логами не найдена: {log_dir}")
        return
    
    print(f"📁 Ищу логи в: {log_dir}")
    
    try:
        # Инициализируем аккумулятор событий
        event_data = event_accumulator.EventAccumulator(
            log_dir,
            size_guidance={event_accumulator.SCALARS: 0}
        )
        event_data.Reload()
        
        # Получаем список всех скалярных метрик
        scalar_tags = event_data.Tags()['scalars']
        if not scalar_tags:
            print("⚠️ В логах не найдено скалярных метрик.")
            return
        
        print(f"📊 Найдено {len(scalar_tags)} метрик")
        
        # Создаем словарь для сбора данных
        data_by_step = {}
        
        # Для каждой метрики извлекаем шаг, время и значение
        for tag in scalar_tags:
            scalar_events = event_data.Scalars(tag)
            
            for event in scalar_events:
                key = (event.step, event.wall_time)
                
                if key not in data_by_step:
                    data_by_step[key] = {'step': event.step, 'wall_time': event.wall_time}
                
                data_by_step[key][tag] = event.value
        
        # Преобразуем в список и создаем DataFrame
        data_list = list(data_by_step.values())
        if not data_list:
            print("❌ Не удалось извлечь данные.")
            return
        
        df = pd.DataFrame(data_list)
        
        # --- КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ: Объединяем строки с одинаковым шагом ---
        print("\n🔧 Объединяю строки с одинаковым шагом...")
        
        # 1. Сортируем по шагу и времени
        df = df.sort_values(['step', 'wall_time']).reset_index(drop=True)
        
        # 2. Создаем список для очищенных данных
        cleaned_data = []
        
        # 3. Группируем по шагу и объединяем все данные для каждого шага
        for step, group in df.groupby('step'):
            if len(group) == 1:
                # Если только одна строка на этот шаг
                cleaned_data.append(group.iloc[0].to_dict())
            else:
                # Объединяем все строки для этого шага
                merged_row = {'step': step}
                
                # Используем wall_time из первой строки
                merged_row['wall_time'] = group.iloc[0]['wall_time']
                
                # Для каждой колонки берем первое не-NaN значение
                for column in group.columns:
                    if column not in ['step', 'wall_time']:
                        non_null_values = group[column].dropna()
                        if not non_null_values.empty:
                            merged_row[column] = non_null_values.iloc[0]
                
                cleaned_data.append(merged_row)
        
        # 4. Создаем новый DataFrame из очищенных данных
        df_clean = pd.DataFrame(cleaned_data)
        
        # 5. Сортируем по шагу
        df_clean = df_clean.sort_values('step').reset_index(drop=True)
        
        # 6. Заполняем пропуски в колонках гиперпараметров и эпох
        #    (они логируются редко, поэтому заполняем значениями из предыдущих строк)
        for col in df_clean.columns:
            if any(keyword in col for keyword in ['hp_', 'lr-', 'epoch']):
                df_clean[col] = df_clean[col].ffill()
        
        print(f"📈 До обработки: {len(df)} строк")
        print(f"📈 После обработки: {len(df_clean)} строк (объединены по шагам)")
        
        # Создаем директорию для вывода
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        
        # Сохраняем в CSV
        df_clean.to_csv(output_csv_path, index=False)
        print(f"✅ Данные успешно экспортированы в {output_csv_path}")
        
        # Статистика по данным
        print(f"\n📊 Статистика данных:")
        print(f"• Шаги: {df_clean['step'].min()} - {df_clean['step'].max()}")
        print(f"• Всего шагов: {len(df_clean)}")
        print(f"• Колонок: {len(df_clean.columns)}")
        
        # Группируем метрики по типу
        metric_types = {}
        for col in df_clean.columns:
            if col not in ['step', 'wall_time']:
                prefix = col.split('/')[0] if '/' in col else col.split('_')[0]
                metric_types.setdefault(prefix, []).append(col)
        
        print(f"\n📋 Группы метрик:")
        for mtype, metrics in metric_types.items():
            filled_count = df_clean[metrics].notna().sum().mean()
            pct_filled = filled_count / len(df_clean) * 100
            print(f"  {mtype}: {len(metrics)} метрик, заполнено ~{pct_filled:.1f}% строк")
        
        # Возвращаем очищенный DataFrame для дальнейшего использования
        return df_clean
        
    except Exception as e:
        print(f"❌ Ошибка при обработке логов: {e}")
        import traceback
        traceback.print_exc()
        return None


def analyze_and_plot(exp_name, df_clean):
    """
    Анализирует и визуализирует метрики из очищенного DataFrame.
    
    Args:
        exp_name: Имя эксперимента (для заголовков графиков)
        df_clean: Очищенный DataFrame с метриками
    """
    if df_clean is None or df_clean.empty:
        print("❌ Нет данных для анализа")
        return
    
    # Анализ лучших значений
    print(f"\n🏆 Лучшие метрики эксперимента '{exp_name}':")
    
    if 'val/psnr' in df_clean.columns:
        best_psnr_idx = df_clean['val/psnr'].idxmax()
        best_psnr = df_clean.loc[best_psnr_idx, 'val/psnr']
        best_psnr_step = df_clean.loc[best_psnr_idx, 'step']
        best_psnr_epoch = df_clean.loc[best_psnr_idx, 'epoch']
        print(f"  • Наивысший PSNR: {best_psnr:.2f} dB (эпоха {best_psnr_epoch}, шаг {best_psnr_step})")
    
    if 'val/ssim' in df_clean.columns:
        best_ssim_idx = df_clean['val/ssim'].idxmax()
        best_ssim = df_clean.loc[best_ssim_idx, 'val/ssim']
        best_ssim_step = df_clean.loc[best_ssim_idx, 'step']
        best_ssim_epoch = df_clean.loc[best_ssim_idx, 'epoch']
        print(f"  • Наивысший SSIM: {best_ssim:.4f} (эпоха {best_ssim_epoch}, шаг {best_ssim_step})")
    
    if 'val/loss_l1' in df_clean.columns:
        best_l1_idx = df_clean['val/loss_l1'].idxmin()
        best_l1 = df_clean.loc[best_l1_idx, 'val/loss_l1']
        best_l1_step = df_clean.loc[best_l1_idx, 'step']
        best_l1_epoch = df_clean.loc[best_l1_idx, 'epoch']
        print(f"  • Наименьший L1 Loss: {best_l1:.4f} (эпоха {best_l1_epoch}, шаг {best_l1_step})")


if __name__ == '__main__':
    experiments = ['cfrwd-25']
    
    for exp in experiments:
        print(f"\n{'='*60}")
        print(f"Обработка эксперимента: {exp}")
        print(f"{'='*60}")
        
        # Экспортируем и очищаем данные
        df_clean = tb2csv(exp)
        
        # Анализируем результаты
        if df_clean is not None:
            analyze_and_plot(exp, df_clean)
            
            # Пример: дополнительный анализ
            if 'train/g_loss' in df_clean.columns and 'train/loss_d' in df_clean.columns:
                # Найдем, когда G и D losses сбалансированы
                g_d_ratio = df_clean['train/g_loss'] / df_clean['train/loss_d']
                balanced_idx = (g_d_ratio - 1.0).abs().idxmin()
                balanced_step = df_clean.loc[balanced_idx, 'step']
                balanced_ratio = g_d_ratio[balanced_idx]
                print(f"  • Наиболее сбалансированные G/D losses: ratio={balanced_ratio:.2f} (шаг {balanced_step})")