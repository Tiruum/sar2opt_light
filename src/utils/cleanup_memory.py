import torch
import gc

def cleanup_memory(log: bool = False):
    """Основная функция очистки памяти"""

    # Сборка мусора Python
    gc.collect()
    
    # Очистка кэша CUDA (если используется GPU)
    if torch.cuda.is_available():
        if log:
            old_allocated = torch.cuda.memory_allocated() / 1024**3
            old_cached = torch.cuda.memory_reserved() / 1024**3
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        if log:
            allocated = torch.cuda.memory_allocated() / 1024**3
            cached = torch.cuda.memory_reserved() / 1024**3
            print(f"Память после очистки:")
            print(f'\tВыделено {old_allocated:.2f} -> {allocated:.2f}GB')
            print(f'\tКэш {old_cached:.2f} -> {cached:.2f}GB')

if __name__ == "__main__":
    cleanup_memory(log=True)