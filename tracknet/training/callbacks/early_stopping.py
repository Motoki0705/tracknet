"""Early stopping utility.

Stops training when a monitored metric has stopped improving.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

Mode = Literal["min", "max"]


@dataclass
class EarlyStoppingConfig:
    """Configuration for early stopping.

    Attributes:
        patience: Number of epochs to wait without improvement.
        min_delta: Minimum change to qualify as an improvement.
        mode: ``"min"`` for losses, ``"max"`` for metrics.
    """

    patience: int = 10
    min_delta: float = 0.0
    mode: Mode = "min"


class EarlyStopping:
    """Track a metric and decide when to stop early."""

    def __init__(self, cfg: EarlyStoppingConfig) -> None:
        self.cfg = cfg
        self.best = math.inf if cfg.mode == "min" else -math.inf
        self.num_bad = 0

    def step(self, value: float) -> bool:
        """Update with a new metric value.

        Args:
            value: The monitored metric value.

        Returns:
            ``True`` if early stopping should trigger, ``False`` otherwise.
        """

        improved = (
            self.cfg.mode == "min" and value < self.best - self.cfg.min_delta
        ) or (self.cfg.mode == "max" and value > self.best + self.cfg.min_delta)
        if improved:
            self.best = value
            self.num_bad = 0
            return False
        self.num_bad += 1
        return self.num_bad > self.cfg.patience
