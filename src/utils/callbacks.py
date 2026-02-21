from pytorch_lightning.callbacks import WeightAveraging
from torch.optim.swa_utils import get_ema_avg_fn

class EMAWeightAveraging(WeightAveraging):
    """
    Exponential Moving Average (EMA) Callback for PyTorch Lightning.
    Updates an averaged model after each training step.
    """
    def __init__(self, decay=0.999, start_epoch=0):
        super().__init__(avg_fn=get_ema_avg_fn(decay))
        self.start_epoch = start_epoch

    def should_update(self, step_idx=None, epoch_idx=None):
        # Начинаем обновлять EMA только после достижения start_epoch
        return (epoch_idx is not None) and (epoch_idx >= self.start_epoch)
