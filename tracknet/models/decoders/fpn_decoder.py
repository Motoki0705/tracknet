"""FPN-like decoder for multi-scale ConvNeXt features.

Given a list of feature maps [C1, C2, C3, C4, C5] (high -> low resolution),
builds a top-down pyramid:
  Lateral 1x1 convs -> top-down upsample+sum -> 3x3 refine -> fuse to target size.

Fuse options:
- 'sum'   : upsample each P* to out_size and sum
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
    in_channels: Sequence[int] = (3, 128, 256, 512, 1024)
    lateral_dim: int = 256
    fuse: FuseType = "sum"
    out_size: Optional[Tuple[int, int]] = None  # (H, W)

class FPNDecoder(nn.Module):
    """Build an FPN top-down pathway and fuse to target resolution.

    Expects features ordered as [C1, C2, C3, C4, C5] (highest -> lowest resolution).
    """

    def __init__(self, cfg: FPNDecoderConfig) -> None:
        super().__init__()
        assert len(cfg.in_channels) >= 3, "Expect at least 3 scales (e.g., [C1,C2,C3])"
        self.cfg = cfg

        # Lateral 1x1 projections to a unified channel dim
        self.lateral = nn.ModuleList(
            [nn.Conv2d(c, cfg.lateral_dim, kernel_size=1) for c in cfg.in_channels]
        )

        # 3x3 conv to refine each pyramid level after top-down sum
        self.refine = nn.ModuleList(
            [nn.Conv2d(cfg.lateral_dim, cfg.lateral_dim, kernel_size=3, padding=1)
             for _ in cfg.in_channels]
        )

        # Fuse head (only used when concat)
        if cfg.fuse == "concat":
            self.fuse_conv = nn.Conv2d(cfg.lateral_dim * len(cfg.in_channels),
                                       cfg.lateral_dim, kernel_size=1)
        else:
            self.fuse_conv = None  # type: ignore[assignment]

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        """Compute FPN features and fuse to ``out_size`` if provided.

        Args:
            feats: List of tensors [C1, C2, C3, C4, C5] in NCHW (high -> low resolution).

        Returns:
            Tensor [B, C, H_out, W_out] where C = lateral_dim and size matches
            cfg.out_size if given, otherwise that of the highest-resolution pyramid map (C1).
        """
        assert len(feats) == len(self.lateral), \
            f"len(in_channels)={len(self.lateral)} but len(feats)={len(feats)}"

        # Apply lateral convs: convert to same channel dim
        lat = [l(f) for l, f in zip(self.lateral, feats)]  # same order as feats: C1..C5

        # Top-down pathway: start from the last (coarsest, C5) and upsample-add
        p = [None] * len(lat)
        p[-1] = lat[-1]  # coarsest
        for i in range(len(lat) - 2, -1, -1):  # from C4 down to C1
            up = F.interpolate(p[i + 1], size=lat[i].shape[-2:], mode="bilinear", align_corners=False)
            p[i] = lat[i] + up

        # 3x3 refine on each pyramid level
        P = [ref(m) for ref, m in zip(self.refine, p)]

        # Determine output size (default: highest resolution, i.e., P1 which aligns to C1)
        if self.cfg.out_size is not None:
            out_h, out_w = self.cfg.out_size
        else:
            out_h, out_w = P[0].shape[-2:]

        # Fuse maps
        up_maps = [F.interpolate(m, size=(out_h, out_w), mode="bilinear", align_corners=False) for m in P]
        if self.cfg.fuse == "sum":
            out = torch.stack(up_maps, dim=0).sum(dim=0)
        else:  # concat
            out = torch.cat(up_maps, dim=1)
            out = self.fuse_conv(out)  # type: ignore[operator]
        return out