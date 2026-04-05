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

        if update_every_n_steps < 1:
            raise ValueError(f"update_every_n_steps must be >= 1, got {update_every_n_steps}")
        self.update_every_n_steps = update_every_n_steps
        self.update_starting_at_step = update_starting_at_step
        self.update_starting_at_epoch = update_starting_at_epoch

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Override to inject current epoch into should_update so the epoch gate is enforced.

        The parent WeightAveraging.on_train_batch_end calls should_update(step_idx=N) without
        an epoch_idx, which means update_starting_at_epoch is never evaluated. We replicate the
        parent's guard logic here and pass epoch_idx so the epoch gate fires correctly.

        Note: uses self._average_model (Lightning internal) — verified against Lightning 2.x.
        If a future Lightning version renames this attribute, this method will raise AttributeError.
        """
        # Mirror the parent's zero-based step index convention.
        step_idx = trainer.global_step - 1
        if (trainer.global_step > self._latest_update_step) and self.should_update(
            step_idx=step_idx, epoch_idx=trainer.current_epoch
        ):
            assert self._average_model is not None
            self._average_model.update_parameters(pl_module)
            self._latest_update_step = trainer.global_step

    def should_update(self, step_idx: Optional[int] = None, epoch_idx: Optional[int] = None) -> bool:
        """Decide when to update the EMA weights.

        update_starting_at_epoch acts as a global gate: if the current epoch is before the
        configured start epoch, no update happens regardless of step conditions.

        Args:
            step_idx: The current step index (zero-based).
            epoch_idx: The current epoch index (zero-based).
        Returns:
            bool: True if EMA weights should be updated, False otherwise.
        """
        # Epoch gate: block all updates before the configured start epoch.
        if self.update_starting_at_epoch is not None:
            if epoch_idx is None or epoch_idx < self.update_starting_at_epoch:
                return False

        # Step gate: check step requirement and frequency.
        if step_idx is not None:
            meets_step_requirement = (
                self.update_starting_at_step is None
                or step_idx >= self.update_starting_at_step
            )
            meets_step_frequency = (
                self.update_every_n_steps > 0
                and step_idx % self.update_every_n_steps == 0
            )
            return meets_step_requirement and meets_step_frequency

        # Called without step_idx (epoch-only call); epoch gate already passed above.
        return True
