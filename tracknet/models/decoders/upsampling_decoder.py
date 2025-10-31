"""Upsampling decoder to map patch tokens to feature maps (slightly enhanced).

- 入力:  tokens [B, H_p, W_p, C_in]
- 出力:  feature map [B, C_out, H_out, W_out]
- 従来の Conv+ReLU を軽量の Residual Block(Depthwise separable + Norm + GELU) に置換
- 既定では SE無効、ブロック数1、GroupNorm を使用（変更可能）

使い方:
    dec = UpsamplingDecoder(
        channels=[384, 256, 128, 64],
        upsample=[2, 2, 2],           # 省略可。短い場合は残りを 1 で埋める
        out_size=(H, W),              # 任意
        blocks_per_stage=1,           # “少し”だけ強化
        use_depthwise=True,
        use_se=False,
        norm='gn',                    # 'bn' も可
        activation='gelu',
        dropout=0.0
    )
"""

from __future__ import annotations
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_norm(c: int, kind: str) -> nn.Module:
    kind = kind.lower()
    if kind == "bn":
        return nn.BatchNorm2d(c)
    if kind == "gn":
        # 小バッチでも安定。グループ数はチャネル以下の2の冪に丸める
        g = 32
        while g > 1 and c % g != 0:
            g //= 2
        return nn.GroupNorm(num_groups=max(1, g), num_channels=c)
    raise ValueError(f"Unsupported norm: {kind}")


class SE(nn.Module):
    """Very small SE block (optional)."""
    def __init__(self, c: int, reduction: int = 8) -> None:
        super().__init__()
        r = max(1, c // reduction)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(c, r, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(r, c, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.fc(self.avg(x))
        return x * w


class ResidualDSBlock(nn.Module):
    """Depthwise-separable residual block: DW(3x3) -> Norm -> GELU -> PW(1x1)."""
    def __init__(
        self,
        c: int,
        norm: str = "gn",
        activation: str = "gelu",
        use_depthwise: bool = True,
        use_se: bool = False,
        se_reduction: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.use_se = use_se
        self.drop = nn.Dropout2d(dropout) if dropout and dropout > 0 else nn.Identity()

        if activation.lower() == "relu":
            act = nn.ReLU(inplace=True)
        elif activation.lower() == "gelu":
            act = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # Depthwise 3x3（オフにすれば通常の3x3Conv）
        self.dw = nn.Conv2d(c, c, kernel_size=3, padding=1, groups=(c if use_depthwise else 1), bias=False)
        self.norm = _make_norm(c, norm)
        self.act = act
        # Pointwise 1x1
        self.pw = nn.Conv2d(c, c, kernel_size=1, bias=False)
        self.se = SE(c, se_reduction) if use_se else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        y = self.dw(x)
        y = self.norm(y)
        y = self.act(y)
        y = self.pw(y)
        y = self.se(y)
        y = self.drop(y)
        return y + identity


class _Stage(nn.Module):
    """(optional upsample) -> channel projection -> N residual blocks."""
    def __init__(
        self,
        in_c: int,
        out_c: int,
        num_blocks: int,
        norm: str,
        activation: str,
        use_depthwise: bool,
        use_se: bool,
        se_reduction: int,
        dropout: float,
    ) -> None:
        super().__init__()
        # チャネルを合わせるための 1x1 プロジェクション
        self.proj = nn.Conv2d(in_c, out_c, kernel_size=1, bias=False) if in_c != out_c else nn.Identity()
        # ブロック本体
        blocks = [
            ResidualDSBlock(
                c=out_c,
                norm=norm,
                activation=activation,
                use_depthwise=use_depthwise,
                use_se=use_se,
                se_reduction=se_reduction,
                dropout=dropout,
            )
            for _ in range(max(1, num_blocks))
        ]
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = self.blocks(x)
        return x


class UpsamplingDecoder(nn.Module):
    """Multi-stage upsampling decoder with lightweight residual blocks.

    Args:
        channels: e.g., [C_in, 256, 128, 64].
        upsample: factors between stages (len <= len(channels)-1 allowed).
        out_size: final spatial size (H, W), if given forces last interpolation.
        blocks_per_stage: number of residual blocks per stage (default 1).
        norm: 'gn' (default) or 'bn'.
        activation: 'gelu' (default) or 'relu'.
        use_depthwise: enable depthwise in the residual block (default True).
        use_se: enable tiny SE attention (default False).
        se_reduction: SE reduction ratio.
        dropout: dropout2d in residual block (default 0.0).
    """

    def __init__(
        self,
        channels: Sequence[int],
        upsample: Sequence[int] | None = None,
        out_size: Optional[Tuple[int, int]] = None,
        *,
        blocks_per_stage: int = 1,
        norm: str = "gn",
        activation: str = "gelu",
        use_depthwise: bool = True,
        use_se: bool = False,
        se_reduction: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert len(channels) >= 2, "channels must include input and at least one output stage"
        self.channels = list(channels)
        num_stages = len(channels) - 1

        base = list(upsample) if upsample is not None else [2] * num_stages
        # upsample が短い場合は足りない分を 1 で埋める（=補間なし）
        self.upsample: List[int] = base + [1] * max(0, num_stages - len(base))
        self.out_size = out_size

        self.stages = nn.ModuleList(
            [
                _Stage(
                    in_c=channels[i],
                    out_c=channels[i + 1],
                    num_blocks=blocks_per_stage,
                    norm=norm,
                    activation=activation,
                    use_depthwise=use_depthwise,
                    use_se=use_se,
                    se_reduction=se_reduction,
                    dropout=dropout,
                )
                for i in range(num_stages)
            ]
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                if hasattr(m, "weight") and m.weight is not None:
                    nn.init.ones_(m.weight)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Args:
            tokens: [B, H_p, W_p, C_in]
        Returns:
            feature map: [B, C_out, H_out, W_out]
        """
        # to NCHW
        x = tokens.permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]

        # Progressive upsample → stage convs
        for i, factor in enumerate(self.upsample):
            if factor and factor != 1:
                x = F.interpolate(x, scale_factor=factor, mode="bilinear", align_corners=False)
            x = self.stages[i](x)

        if self.out_size is not None:
            x = F.interpolate(x, size=self.out_size, mode="bilinear", align_corners=False)
        return x
