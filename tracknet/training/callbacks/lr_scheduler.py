"""Learning-rate scheduler integration callback.

Provides thin wrappers to step PyTorch schedulers at epoch or batch boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

When = Literal["epoch", "batch"]


@dataclass
class LRSchedulerConfig:
    """Configuration for learning-rate scheduling.

    Attributes:
        when: Step the scheduler at epoch end or per batch.
    """

    when: When = "epoch"


class LRSchedulerCallback:
    """Minimal wrapper to step a scheduler based on a configured cadence."""

    def __init__(self, scheduler: torch.optim.lr_scheduler._LRScheduler, cfg: LRSchedulerConfig) -> None:  # type: ignore[attr-defined]
        self.scheduler = scheduler
        self.cfg = cfg

    def on_batch_end(self) -> None:
        """Step the scheduler if configured for per-batch updates."""

        if self.cfg.when == "batch":
            self.scheduler.step()

    def on_epoch_end(self, metric: float | None = None) -> None:
        """Step the scheduler if configured for per-epoch updates.

        Args:
            metric: Optional monitored metric for schedulers that accept it
                (e.g., ``ReduceLROnPlateau``).
        """

        if self.cfg.when == "epoch":
            try:
                # Some schedulers accept a metric argument.
                if metric is not None:
                    self.scheduler.step(metric)
                else:
                    self.scheduler.step()
            except TypeError:
                self.scheduler.step()
