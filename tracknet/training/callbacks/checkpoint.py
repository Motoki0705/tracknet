"""Model checkpointing helper.

Tracks a monitored metric and determines when to save a new best checkpoint.
The actual save operation is provided by the caller via a callable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional

import math


Mode = Literal["min", "max"]


@dataclass
class CheckpointConfig:
    """Configuration for model checkpointing.

    Attributes:
        mode: ``"min"`` for minimizing metrics (e.g., loss) or ``"max"``.
        save_best_only: If True, only save when improvement occurs.
    """

    mode: Mode = "min"
    save_best_only: bool = True


class ModelCheckpoint:
    """Decide when to save the best model based on a monitored metric."""

    def __init__(self, cfg: CheckpointConfig, save_fn: Callable[[], None]) -> None:
        self.cfg = cfg
        self.save_fn = save_fn
        self.best = math.inf if cfg.mode == "min" else -math.inf

    def step(self, value: float) -> bool:
        """Evaluate whether to save a new checkpoint.

        Args:
            value: The monitored metric.

        Returns:
            ``True`` if a checkpoint was saved.
        """

        improved = (
            (self.cfg.mode == "min" and value < self.best) or
            (self.cfg.mode == "max" and value > self.best)
        )
        if improved:
            self.best = value
            self.save_fn()
            return True
        if not self.cfg.save_best_only:
            self.save_fn()
            return True
        return False

