"""FPN-like decoder for multi-scale ConvNeXt features.

Given a list of feature maps [C3, C4, C5(, C2)], builds top-down pyramid:
    Lateral 1x1 convs -> top-down upsample+sum -> 3x3 refine -> fuse to target size.

Fuse options:
- 'sum': upsample each P* to out_size and sum
- 'concat': concatenate then 1x1 conv to reduce to fpn_dim
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


FuseType = Literal["sum", "concat"]


@dataclass
class FPNDecoderConfig:
    lateral_dim: int = 256
    use_p2: bool = False
    fuse: FuseType = "sum"
    out_size: Optional[Tuple[int, int]] = None  # (H, W)


class FPNDecoder(nn.Module):
    """Build an FPN top-down pathway and fuse to target resolution."""

    def __init__(self, in_channels: Sequence[int], cfg: FPNDecoderConfig) -> None:
        super().__init__()
        assert len(in_channels) >= 3, "Expect at least [C3,C4,C5]"
        self.cfg = cfg
        self.lateral = nn.ModuleList([nn.Conv2d(c, cfg.lateral_dim, kernel_size=1) for c in in_channels])
        self.refine = nn.ModuleList([nn.Conv2d(cfg.lateral_dim, cfg.lateral_dim, kernel_size=3, padding=1) for _ in in_channels])
        if cfg.fuse == "concat":
            self.fuse_conv = nn.Conv2d(cfg.lateral_dim * len(in_channels), cfg.lateral_dim, kernel_size=1)
        else:
            self.fuse_conv = None  # type: ignore[assignment]

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        """Compute FPN features and fuse to ``out_size`` if provided.

        Args:
            feats: List of tensors [C3, C4, C5(, C2)] in NCHW.

        Returns:
            Tensor [B, C, H_out, W_out] where C = lateral_dim and size matches
            cfg.out_size if given, otherwise that of the highest-resolution pyramid map.
        """

        # Apply lateral convs: convert to same channel dim
        lat = [l(f) for l, f in zip(self.lateral, feats)]  # same order as feats

        # Top-down: start from the last (coarsest) and upsample-add
        p = [None] * len(lat)
        p[-1] = lat[-1]
        for i in range(len(lat) - 2, -1, -1):
            up = F.interpolate(p[i + 1], size=lat[i].shape[-2:], mode="bilinear", align_corners=False)
            p[i] = lat[i] + up

        # 3x3 refine
        P = [ref(m) for ref, m in zip(self.refine, p)]

        # Determine output size
        if self.cfg.out_size is not None:
            out_h, out_w = self.cfg.out_size
        else:
            # Highest resolution is the first element (C3)
            out_h, out_w = P[0].shape[-2:]

        # Fuse maps
        up_maps = [F.interpolate(m, size=(out_h, out_w), mode="bilinear", align_corners=False) for m in P]
        if self.cfg.fuse == "sum":
            out = torch.stack(up_maps, dim=0).sum(dim=0)
        else:  # concat
            out = torch.cat(up_maps, dim=1)
            out = self.fuse_conv(out)  # type: ignore[operator]
        return out

