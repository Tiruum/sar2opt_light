import torch
from typing import Any, Optional, Union
from lightning.pytorch.callbacks import WeightAveraging
from torch.optim.swa_utils import get_ema_avg_fn

class EMAWeightAveraging(WeightAveraging):
    """Exponential Moving Average (EMA) Weight Averaging callback."""

    def __init__(
        self,
        device: Optional[Union[torch.device, str, int]] = None,
        use_buffers: bool = True,
        decay: float = 0.999,
        update_every_n_steps: int = 1,
        update_starting_at_step: Optional[int] = None,
        update_starting_at_epoch: Optional[int] = None,
        **kwargs: Any,
    ):
        super().__init__(
            device=device,
            use_buffers=use_buffers,
            **kwargs,
            avg_fn=get_ema_avg_fn(decay=decay),
        )

        self.update_every_n_steps = update_every_n_steps
        self.update_starting_at_step = update_starting_at_step
        self.update_starting_at_epoch = update_starting_at_epoch

    def should_update(self, step_idx: Optional[int] = None, epoch_idx: Optional[int] = None) -> bool:
        """Decide when to update the model weights.

        Args:
            step_idx: The current step index.
            epoch_idx: The current epoch index.
        Returns:
            bool: True if the model weights should be updated, False otherwise.

        """
        if step_idx is not None:
            # Check step-based conditions only if we have a valid step_idx
            meets_step_requirement = self.update_starting_at_step is None or step_idx >= self.update_starting_at_step
            meets_step_frequency = self.update_every_n_steps > 0 and step_idx % self.update_every_n_steps == 0
            if meets_step_requirement and meets_step_frequency:
                return True

        if epoch_idx is not None:
            # Check epoch-based condition only if we specify one
            meets_epoch_requirement = (
                self.update_starting_at_epoch is not None and epoch_idx >= self.update_starting_at_epoch
            )
            if meets_epoch_requirement:
                return True

        return False
