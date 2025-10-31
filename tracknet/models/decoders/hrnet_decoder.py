from __future__ import annotations
from dataclasses import dataclass
from typing import List, Sequence, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x3_bn_relu(in_ch, out_ch, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )

def conv1x1_bn_relu(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )

class ExchangeUnit(nn.Module):
    """
    HRNet の multi-resolution fusion を簡略化。
    入力: 複数解像度の特徴 [x0(高), x1, x2, x3]
    出力: 同じ解像度リスト [y0, y1, y2, y3]
    """
    def __init__(self, widths: Sequence[int]):
        super().__init__()
        self.widths = list(widths)
        branches = []
        for w in widths:
            branches.append(conv3x3_bn_relu(w, w, stride=1))
        self.branch_ops = nn.ModuleList(branches)

        # ペア(i -> k)ごとの変換レイヤ
        # - i == k: Identity
        # - i <  k: downsample (3x3 s=2) を (k-i) 回
        # - i >  k: 1x1 でチャネル合わせ → 後段でアップサンプル
        fuse_layers = []
        n = len(widths)
        for k in range(n):
            fuse_k = nn.ModuleList()
            for i in range(n):
                if i == k:
                    fuse_k.append(nn.Identity())
                elif i < k:
                    ops = []
                    in_c = widths[i]
                    # 1段下げるごとに 3x3 s=2、最後の出力チャネルは target 幅
                    for d in range(k - i):
                        out_c = widths[k]
                        ops.append(conv3x3_bn_relu(in_c, out_c, stride=2))
                        in_c = out_c
                    fuse_k.append(nn.Sequential(*ops))
                else:  # i > k
                    fuse_k.append(conv1x1_bn_relu(widths[i], widths[k]))
            fuse_layers.append(fuse_k)
        self.fuse_layers = nn.ModuleList(fuse_layers)

    def forward(self, xs: List[torch.Tensor]) -> List[torch.Tensor]:
        # 各分岐で軽く整形
        xs = [op(x) for op, x in zip(self.branch_ops, xs)]
        # 融合
        ys: List[torch.Tensor] = []
        for k, fuse_k in enumerate(self.fuse_layers):
            y = None
            target_size = xs[k].shape[-2:]
            for i, trans in enumerate(fuse_k):
                z = trans(xs[i])
                if i > k:
                    # 低解像→高解像へ：アップサンプルで空間サイズ合わせ
                    z = F.interpolate(z, size=target_size, mode="bilinear", align_corners=False)
                # （i<k のときは stride=2 系で既に target_size になっている）
                y = z if y is None else (y + z)
            ys.append(y)
        return ys

@dataclass
class HRDecoderConfig:
    in_channels: Sequence[int]             # 例: ConvNeXt の [C2..C5] チャネル
    widths: Sequence[int] = (64, 128, 256, 512)   # 各分岐のチャネル幅（高→低）
    num_units: int = 2                     # ExchangeUnit の反復回数
    out_channels: Optional[int] = None     # 最後に 3x3 で出力チャネルへ変換（省略可）

class HRDecoder(nn.Module):
    """
    ConvNeXt [C2..C5] -> HRNet 風の交換ユニットを通し、C2 解像度だけ出力
    """
    def __init__(self, cfg: HRDecoderConfig):
        super().__init__()
        assert len(cfg.in_channels) == len(cfg.widths), "in_channels と widths の長さを合わせてください"
        self.cfg = cfg

        # Lateral: 各段を所定の幅に合わせる（1x1）
        self.lateral = nn.ModuleList([
            conv1x1_bn_relu(cin, w) for cin, w in zip(cfg.in_channels, cfg.widths)
        ])

        # Exchange units（HRNetの stage 内 fusion を複数回）
        self.units = nn.ModuleList([ExchangeUnit(cfg.widths) for _ in range(cfg.num_units)])

        # 最終出力（C2 解像度の分岐のみ使用）
        out_ch = cfg.out_channels or cfg.widths[0]
        self.head = nn.Conv2d(cfg.widths[0], out_ch, kernel_size=3, padding=1)

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        """
        feats: [C2, C3, C4, C5] （高→低解像度）
        返り値: C2 解像度の特徴（[B, out_ch, H_C2, W_C2]）
        """
        assert len(feats) == len(self.lateral)
        xs = [lat(f) for lat, f in zip(self.lateral, feats)]

        # 交換ユニットを反復
        for unit in self.units:
            xs = unit(xs)

        # 高解像度分岐（index=0）のみを出力ヘッドへ
        out = self.head(xs[0])
        return out

if __name__ == "__main__":
    B = 2
    # ConvNeXt 一般例：C2..C5 = stride [4, 8, 16, 32]
    H, W = 512, 512
    shapes = [(H//4, W//4), (H//8, W//8), (H//16, W//16), (H//32, W//32)]
    C_ins  = [192, 384, 768, 1536]  # ConvNeXt-L 相当の例（モデルにより要調整）
    feats = [torch.randn(B, c, h, w) for (h,w), c in zip(shapes, C_ins)]

    cfg = HRDecoderConfig(
        in_channels=C_ins,
        widths=(128, 256, 256, 256),  # 好みで。高解像だけ細め/低解像は太めでもOK
        num_units=2,
        out_channels=256
    )
    dec = HRDecoder(cfg)
    y = dec(feats)
    print(f"HRDecoder output shape: {y.shape}")  # [B, 256, H//4, W//4] = C2 解像度
