import torch
import gc
import multiprocessing


def cleanup_memory(log: bool = False):
    """Очистка GPU-кэша и сборка мусора Python."""

    gc.collect()

    if torch.cuda.is_available():
        if log:
            old_allocated = torch.cuda.memory_allocated() / 1024**3
            old_cached = torch.cuda.memory_reserved() / 1024**3
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        if log:
            allocated = torch.cuda.memory_allocated() / 1024**3
            cached = torch.cuda.memory_reserved() / 1024**3
            print("Memory after cleanup:")
            print(f'\tallocated {old_allocated:.2f} -> {allocated:.2f} GB')
            print(f'\tcached    {old_cached:.2f} -> {cached:.2f} GB')


def kill_dataloader_workers(log: bool = False):
    """Принудительно убивает все дочерние процессы (DataLoader workers).
    
    На Windows DataLoader workers — это spawn-процессы, которые могут
    оставаться живыми после Ctrl+C, накапливая RAM при перезапусках.
    """
    children = multiprocessing.active_children()
    if log and children:
        print(f"Found {len(children)} child process(es), terminating...")
    for child in children:
        child.terminate()
    for child in children:
        child.join(timeout=5)
        if child.is_alive():
            child.kill()
    if log and children:
        print("All child processes terminated.")


def full_cleanup(trainer=None, model=None, datamodule=None, log: bool = False):
    """Полная очистка ресурсов после обучения.
    
    Правильный порядок:
    1. Удаляем фиксированные тензоры из модели (GPU RAM)
    2. Удаляем модель и датамодуль (освобождаем ссылки)
    3. Убиваем DataLoader worker-процессы (RAM)
    4. Чистим CUDA кэш и собираем мусор
    """
    # 1. Убираем ссылки на фиксированные тензоры модели
    if model is not None:
        for attr in ('fixed_sar', 'fixed_opt'):
            if hasattr(model, attr):
                setattr(model, attr, None)

    # 2. Принудительный teardown DataLoader workers через trainer
    #    _teardown() уничтожает стратегию, акселератор, и dataloaders
    if trainer is not None:
        try:
            # Это закрывает dataloaders и их workers
            if hasattr(trainer, 'train_dataloader') and trainer.train_dataloader is not None:
                pass  # Lightning управляет ими через _teardown
            trainer._teardown()
        except Exception:
            pass

    # 3. Удаляем Python-объекты
    del trainer, model, datamodule

    # 4. Принудительно убиваем зависшие worker-процессы
    kill_dataloader_workers(log=log)

    # 5. Сборка мусора + очистка CUDA
    cleanup_memory(log=log)


if __name__ == "__main__":
    cleanup_memory(log=True)
    kill_dataloader_workers(log=True)