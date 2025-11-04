"""Heatmap prediction head.

Applies a 1x1 convolution to map decoder features to a single-channel heatmap.
Activation (e.g., sigmoid) is intentionally omitted; the loss function decides
the appropriate activation handling.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class HeatmapHead(nn.Module):
    """Project feature maps to 1-channel heatmaps via 1x1 Conv."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Feature tensor ``[B, C, H, W]``.

        Returns:
            Heatmap tensor ``[B, 1, H, W]``.
        """

        return self.conv(x)
