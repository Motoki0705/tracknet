# ViT Heatmap モデル

ViT バックボーン＋アップサンプリングデコーダ＋ヒートマップヘッドからなる構成。

## バックボーン
- モジュール: `tracknet/models/backbones/vit_backbone.py`
- `ViTBackboneConfig(pretrained_model_name, device_map="auto", local_files_only=True, patch_size=16)`
- Hugging Face `AutoModel` をローカルキャッシュから読み込み、入力 `[B,3,H,W]`（`H`,`W` は `patch_size` の倍数）を `[B, H//patch_size, W//patch_size, C]` のパッチグリッドに変換。
- クラストークンとレジスタトークンを除去したパッチのみを返す。画像リサイズが不正な場合は例外を送出。

## アップサンプリングデコーダ
- モジュール: `tracknet/models/decoders/upsampling_decoder.py`
- `UpsamplingDecoder(channels, upsample, out_size=None, blocks_per_stage=1, norm="gn", activation="gelu", use_depthwise=True, use_se=False, se_reduction=8, dropout=0.0)`
- 各ステージで 1x1 プロジェクション → Residual Depthwise Block（Norm→GELU→Pointwise）を適用。
- `blocks_per_stage` でブロック数を増やせる。`norm` は `gn` か `bn`。`use_se=True` で軽量 SE を挿入。
- `out_size=(Hh, Wh)` を指定すると最終ヒートマップ解像度へ補間。

## ヘッド
- モジュール: `tracknet/models/heads/heatmap_head.py`
- `HeatmapHead(in_channels)` により `[B,C,H,W]` → `[B,1,H,W]` を生成（活性化はロス側で処理）。

## 主なコンフィグ項目
- `model.pretrained_model_name`: HF モデル識別子（例: `facebook/dinov3-vits16-pretrain-lvd1689m`）
- `model.backbone.{freeze, device_map, local_files_only, patch_size}`
- `model.decoder.{channels, upsample, blocks_per_stage, norm, activation, use_depthwise, use_se, se_reduction, dropout}`
- `model.heatmap.{size=[W,H], sigma}`

## 使用例
```python
import torch
from tracknet.models import (
    ViTBackbone, ViTBackboneConfig,
    UpsamplingDecoder,
    HeatmapHead,
)

B, C, H, W = 2, 3, 720, 1280
x = torch.randn(B, C, H, W)

backbone = ViTBackbone(ViTBackboneConfig(
    pretrained_model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
    device_map="auto",
    local_files_only=True,
    patch_size=16,
))
tokens = backbone(x)  # [B,45,80,384]

decoder = UpsamplingDecoder(
    channels=[384, 256, 128, 64],
    upsample=[2, 2, 2],
    out_size=(144, 256),
    blocks_per_stage=1,
    norm="gn",
    activation="gelu",
    use_depthwise=True,
    use_se=False,
)
features = decoder(tokens)

head = HeatmapHead(64)
heatmaps = head(features)  # [B,1,144,256]
print(heatmaps.shape)
```
