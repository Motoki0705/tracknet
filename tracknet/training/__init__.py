# tracknet/training/__init__.py
"""
Training package for TrackNet.

Includes losses, metrics, and callbacks used by the trainer.
Lazily exposes symbols at package top-level to reduce import time.
"""

from __future__ import annotations

import importlib
import threading
from typing import TYPE_CHECKING, Any

__all__ = [
    # losses
    "HeatmapMSELoss",
    "HeatmapFocalLoss",
    "HeatmapLossConfig",
    "build_heatmap_loss",
    # metrics
    "heatmap_argmax_coords",
    "heatmap_soft_argmax_coords",
    "l2_error",
    "pck_at_r",
    "visible_from_mask",
    # callbacks
    "EarlyStopping",
    "EarlyStoppingConfig",
    "ModelCheckpoint",
    "CheckpointConfig",
    "LRSchedulerCallback",
    "LRSchedulerConfig",
]

# name -> "relative.module.path:attribute" mapping
_lazy_specs: dict[str, str] = {
    # losses
    "HeatmapMSELoss": ".losses.heatmap_loss:HeatmapMSELoss",
    "HeatmapFocalLoss": ".losses.heatmap_loss:HeatmapFocalLoss",
    "HeatmapLossConfig": ".losses.heatmap_loss:HeatmapLossConfig",
    "build_heatmap_loss": ".losses.heatmap_loss:build_heatmap_loss",
    # metrics
    "heatmap_argmax_coords": ".metrics:heatmap_argmax_coords",
    "heatmap_soft_argmax_coords": ".metrics:heatmap_soft_argmax_coords",
    "l2_error": ".metrics:l2_error",
    "pck_at_r": ".metrics:pck_at_r",
    "visible_from_mask": ".metrics:visible_from_mask",
    # callbacks
    "EarlyStopping": ".callbacks.early_stopping:EarlyStopping",
    "EarlyStoppingConfig": ".callbacks.early_stopping:EarlyStoppingConfig",
    "ModelCheckpoint": ".callbacks.checkpoint:ModelCheckpoint",
    "CheckpointConfig": ".callbacks.checkpoint:CheckpointConfig",
    "LRSchedulerCallback": ".callbacks.lr_scheduler:LRSchedulerCallback",
    "LRSchedulerConfig": ".callbacks.lr_scheduler:LRSchedulerConfig",
}

_lazy_cache: dict[str, Any] = {}
_lazy_lock = threading.RLock()


def __getattr__(name: str) -> Any:
    # Return cached if already resolved
    obj = _lazy_cache.get(name)
    if obj is not None:
        return obj

    spec = _lazy_specs.get(name)
    if spec is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    mod_path, attr = spec.split(":")
    with _lazy_lock:
        # Double-check under lock
        obj = _lazy_cache.get(name)
        if obj is not None:
            return obj
        mod = importlib.import_module(mod_path, __name__)
        obj = getattr(mod, attr)
        _lazy_cache[name] = obj
        globals()[name] = obj  # Cache into module globals for subsequent direct access
        return obj


def __dir__():
    # Better REPL/IDE experience
    return sorted(set(list(globals().keys()) + __all__))


# Static type checkers / IDEs see the symbols without importing at runtime
if TYPE_CHECKING:
    from .callbacks.checkpoint import CheckpointConfig, ModelCheckpoint
    from .callbacks.early_stopping import EarlyStopping, EarlyStoppingConfig
    from .callbacks.lr_scheduler import LRSchedulerCallback, LRSchedulerConfig
    from .losses.heatmap_loss import (
        HeatmapFocalLoss,
        HeatmapLossConfig,
        HeatmapMSELoss,
        build_heatmap_loss,
    )
    from .metrics import (
        heatmap_argmax_coords,
        heatmap_soft_argmax_coords,
        l2_error,
        pck_at_r,
        visible_from_mask,
    )
