from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import FeaturePyramidNetwork

FuseType = Literal["sum", "concat"]


@dataclass
class FPNDecoderConfig:
    in_channels: list[int]  # 例: ConvNeXt の [C2..C5]
    lateral_dim: int = 256
    fuse: FuseType = "sum"
    out_size: tuple[int, int] | None = None  # (H,W)


class FPNDecoderTorchvision(nn.Module):
    """
    torchvision.ops.FeaturePyramidNetwork を使った簡易版。
    入力は [C2, C3, C4, C5]（高→低解像度の順）でも [C1..C5] でも可。
    """

    def __init__(self, cfg: FPNDecoderConfig):
        super().__init__()
        self.cfg = cfg
        self.fpn = FeaturePyramidNetwork(cfg.in_channels, cfg.lateral_dim)
        if cfg.fuse == "concat":
            self.reduce = nn.Conv2d(
                cfg.lateral_dim * len(cfg.in_channels), cfg.lateral_dim, 1
            )
        else:
            self.reduce = None

    def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
        x = OrderedDict((f"c{i}", t) for i, t in enumerate(feats))  # 高→低の順に保持
        pyr = self.fpn(x)  # OrderedDict（高解像→低解像の順で返る）
        maps = list(pyr.values())

        # 出力解像度
        H, W = self.cfg.out_size or maps[0].shape[-2:]
        up = [
            F.interpolate(m, size=(H, W), mode="bilinear", align_corners=False)
            for m in maps
        ]

        if self.cfg.fuse == "sum":
            return torch.stack(up, 0).sum(0)
        out = torch.cat(up, dim=1)
        return self.reduce(out)


if __name__ == "__main__":
    # simple test
    B = 2
    C_ins = [128, 256, 512, 1024]
    Hs = [64, 32, 16, 8]
    Ws = [64, 32, 16, 8]
    feats = [torch.randn(B, c, h, w) for c, h, w in zip(C_ins, Hs, Ws, strict=False)]

    cfg = FPNDecoderConfig(
        in_channels=C_ins,
        lateral_dim=256,
        fuse="concat",
        out_size=(128, 128),
    )
    model = FPNDecoderTorchvision(cfg)
    out = model(feats)
    print(out.shape)  # expect [B, 256, 128, 128]
