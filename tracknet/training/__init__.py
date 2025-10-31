"""Training package for TrackNet.

Includes losses, metrics, and callbacks used by the trainer.
"""

# Lazy imports to improve startup time
def _import_losses():
    from .losses.heatmap_loss import (
        HeatmapMSELoss,
        HeatmapFocalLoss,
        HeatmapLossConfig,
        build_heatmap_loss,
    )
    return HeatmapMSELoss, HeatmapFocalLoss, HeatmapLossConfig, build_heatmap_loss

def _import_metrics():
    from .metrics import (
        heatmap_argmax_coords,
        heatmap_soft_argmax_coords,
        l2_error,
        pck_at_r,
        visible_from_mask,
    )
    return heatmap_argmax_coords, heatmap_soft_argmax_coords, l2_error, pck_at_r, visible_from_mask

def _import_callbacks():
    from .callbacks.early_stopping import EarlyStopping, EarlyStoppingConfig
    from .callbacks.checkpoint import ModelCheckpoint, CheckpointConfig
    from .callbacks.lr_scheduler import LRSchedulerCallback, LRSchedulerConfig
    return EarlyStopping, EarlyStoppingConfig, ModelCheckpoint, CheckpointConfig, LRSchedulerCallback, LRSchedulerConfig

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

