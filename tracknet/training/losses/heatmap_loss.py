"""Heatmap losses with visibility masking.

Provides MSE and focal-style heatmap losses. All losses accept a visibility
mask to exclude missing targets from contributing to the objective.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn


def _reduce_with_mask(
    loss: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """Reduce a per-pixel loss using a visibility mask.

    Args:
        loss: Per-pixel loss tensor of shape ``[B, 1, H, W]``.
        mask: Visibility mask of shape ``[B, 1, H, W]`` where ``1`` includes the
            pixel in the loss and ``0`` excludes it.
        eps: Small value to avoid division by zero.

    Returns:
        Scalar tensor of the masked mean loss.
    """

    masked = loss * mask
    denom = mask.sum().clamp_min(eps)
    return masked.sum() / denom


class HeatmapMSELoss(nn.Module):
    """Mean squared error for heatmaps with visibility masking.

    Computes ``mean((pred - target)^2)`` over visible pixels only.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute masked MSE loss.

        Args:
            pred: Predicted heatmaps ``[B, 1, H, W]``.
            target: Target heatmaps ``[B, 1, H, W]``.
            mask: Visibility mask ``[B, 1, H, W]``.

        Returns:
            Scalar tensor of the masked MSE.
        """

        loss = (pred - target) ** 2
        return _reduce_with_mask(loss, mask)


class HeatmapFocalLoss(nn.Module):
    """Focal-style loss for heatmap regression with masking.

    This implementation follows a common variant used in keypoint heatmaps.
    For target values in ``[0, 1]``, the loss encourages accurate peaks while
    down-weighting easy negatives.
    """

    def __init__(self, alpha: float = 2.0, beta: float = 4.0) -> None:
        """Initialize focal loss parameters.

        alpha: Exponent for positive terms.
        beta: Exponent for negative terms.
        """

        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute masked focal-style heatmap loss.

        Args:
            pred: Predicted heatmaps ``[B, 1, H, W]`` (logits or probabilities).
            target: Target heatmaps ``[B, 1, H, W]`` in ``[0, 1]``.
            mask: Visibility mask ``[B, 1, H, W]``.

        Returns:
            Scalar tensor of the masked focal loss.
        """

        pred_sigmoid = pred.sigmoid().clamp(1e-6, 1 - 1e-6)
        pos_loss = (
            -((1 - pred_sigmoid) ** self.alpha) * torch.log(pred_sigmoid) * target
        )
        neg_loss = -(
            (pred_sigmoid**self.alpha)
            * ((1 - target) ** self.beta)
            * torch.log(1 - pred_sigmoid)
        )
        loss = pos_loss + (1 - target) * neg_loss
        return _reduce_with_mask(loss, mask)


LossName = Literal["mse", "focal"]


@dataclass
class HeatmapLossConfig:
    """Configuration for selecting a heatmap loss.

    Attributes:
        name: One of ``"mse"`` or ``"focal"``.
        alpha: Focal alpha (only for ``name="focal"``).
        beta: Focal beta (only for ``name="focal"``).
    """

    name: LossName = "mse"
    alpha: float = 2.0
    beta: float = 4.0


def build_heatmap_loss(cfg: HeatmapLossConfig) -> nn.Module:
    """Factory to build a heatmap loss criterion.

    Args:
        cfg: Loss configuration.

    Returns:
        An instance of ``nn.Module`` implementing the selected loss.
    """

    if cfg.name == "mse":
        return HeatmapMSELoss()
    if cfg.name == "focal":
        return HeatmapFocalLoss(alpha=cfg.alpha, beta=cfg.beta)
    raise ValueError(f"Unsupported loss: {cfg.name}")
