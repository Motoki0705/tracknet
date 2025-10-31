# ConvNeXt FPN Heatmap モデル

ConvNeXt バックボーン＋FPN デコーダ＋ヒートマップヘッドからなる構成。

## バックボーン
- モジュール: `tracknet/models/backbones/convnext_backbone.py`
- `ConvNeXtBackboneConfig(pretrained_model_name, return_stages=(0,1,2,3,4), device_map="auto", local_files_only=True)`
- Hugging Face `AutoModel` をローカルキャッシュから読み込み、`hidden_states` から `[C1, C2, C3, C4, C5]`（高解像度→低解像度）を返却。
- `return_stages` を変更することで利用する層を制御可能。

## FPN デコーダ
- モジュール: `tracknet/models/decoders/fpn_decoder.py`
- `FPNDecoder(in_channels, FPNDecoderConfig(lateral_dim=256, fuse="sum", out_size=None))`
- 各スケールを 1x1 ラテラルで射影→トップダウンで補間加算→3x3 リファイン。
- `fuse="sum"`（既定）で加算融合、`fuse="concat"` で連結＋1x1 再投影。
- `out_size=(Hh,Wh)` を指定すると最終解像度へリサイズ。

## ヘッド
- モジュール: `tracknet/models/heads/heatmap_head.py`
- `[B,C,H,W]` → `[B,1,H,W]` を生成。

## 主なコンフィグ項目
- `model.pretrained_model_name`: HF ConvNeXt 識別子（例: `facebook/dinov3-convnext-base-pretrain-lvd1689m`）
- `model.backbone.{freeze, return_stages, device_map, local_files_only}`
- `model.fpn.{lateral_dim, fuse}`
- `model.heatmap.{size=[W,H], sigma}`

## 使用例
```python
import torch
from tracknet.models import (
    ConvNeXtBackbone, ConvNeXtBackboneConfig,
    FPNDecoder, FPNDecoderConfig,
    HeatmapHead,
)

B, C, H, W = 2, 3, 720, 1280
x = torch.randn(B, C, H, W)

backbone = ConvNeXtBackbone(ConvNeXtBackboneConfig(
    pretrained_model_name="facebook/dinov3-convnext-base-pretrain-lvd1689m",
    return_stages=(0, 1, 2, 3, 4),
    device_map="auto",
    local_files_only=True,
))
features = backbone(x)  # [C1..C5]

fpn = FPNDecoder(
    in_channels=[f.shape[1] for f in features],
    cfg=FPNDecoderConfig(lateral_dim=256, fuse="sum", out_size=(144, 256)),
)
pyramid = fpn(features)

head = HeatmapHead(256)
heatmaps = head(pyramid)  # [B,1,144,256]
print(heatmaps.shape)
```
