"""Upsampling decoder to map patch tokens to feature maps.

Takes a patch token grid ``[B, H_p, W_p, C_in]``, converts to NCHW, and applies
staged upsampling with Conv layers to reach a target spatial resolution. The
final feature map can then be passed to a prediction head.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class UpsamplingDecoder(nn.Module):
    """Multi-stage upsampling decoder with Conv + interpolation.

    Args:
        channels: Channel plan including input dim, e.g., ``[384, 256, 128, 64]``.
        upsample: Upsample factors between stages, length ``len(channels)-1`` or
            smaller. If a factor is 1, no interpolation is applied before the conv.
        out_size: Optional target size as ``(H, W)`` to force final interpolation.
    """

    def __init__(
        self,
        channels: Sequence[int],
        upsample: Sequence[int] | None = None,
        out_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        super().__init__()
        assert len(channels) >= 2, "channels must include input and at least one output stage"
        self.channels = list(channels)
        self.upsample = list(upsample) if upsample is not None else [2] * (len(channels) - 1)
        self.out_size = out_size

        layers: List[nn.Module] = []
        in_c = channels[0]
        for i, out_c in enumerate(channels[1:]):
            k = 3
            p = 1
            layers.append(nn.Conv2d(in_c, out_c, kernel_size=k, padding=p))
            layers.append(nn.ReLU(inplace=True))
            in_c = out_c
        self.proj = nn.Sequential(*layers)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            tokens: Patch token grid ``[B, H_p, W_p, C_in]``.

        Returns:
            Feature map ``[B, C_out, H_out, W_out]``.
        """

        x = tokens.permute(0, 3, 1, 2).contiguous()  # [B,C,H_p,W_p]

        # Progressive upsampling with conv projections per stage
        c_idx = 0
        for i, factor in enumerate(self.upsample):
            if factor and factor != 1:
                x = F.interpolate(x, scale_factor=factor, mode="bilinear", align_corners=False)
            # Apply corresponding conv+relu stage
            # Use a single sequential over all stages by slicing
            conv = self.proj[2 * i]
            act = self.proj[2 * i + 1]
            x = act(conv(x))

        if self.out_size is not None:
            x = F.interpolate(x, size=self.out_size, mode="bilinear", align_corners=False)
        return x

