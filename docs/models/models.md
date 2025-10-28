# モデル仕様（Section 3）

ViTバックボーン＋アップサンプリングデコーダ＋ヒートマップヘッドの概要を示す。

## バックボーン
- `tracknet/models/backbones/vit_backbone.py`
  - `ViTBackboneConfig(pretrained_model_name, use_pretrained=True, fallback_dim=384, patch_size=16)`
  - `ViTBackbone`: 入力 `[B,C,H,W]` → 出力 `[B,Hp,Wp,C]`（パッチトークン）
  - オンライン/キャッシュあり: Hugging Face `AutoImageProcessor` + `AutoModel`（`local_files_only=True`）。
  - オフライン: Conv2d（stride=patch_size）で疑似パッチ埋め込み（fallback）。

## デコーダ
- `tracknet/models/decoders/upsampling_decoder.py`
  - `UpsamplingDecoder(channels=[384,256,128,64], upsample=[2,2,2], out_size=(Hh,Wh))`
  - 入力 `[B,Hp,Wp,C]` → 出力 `[B,C_out,H, W]`
  - 各ステージで `interpolate(scale)` → `Conv3x3+ReLU` を適用。

## ヘッド
- `tracknet/models/heads/heatmap_head.py`
  - `HeatmapHead(in_channels)`
  - 入力 `[B,C,H,W]` → 出力 `[B,1,H,W]`（活性化はロス側で処理）。

## 形状の要点
- 例: 画像 `1280x720`、patch=16 → `Hp=45, Wp=80`（fallbackの場合）
- デコーダの `out_size` をヒートマップ解像度 `Hh x Wh`（例 `72x128`）に合わせる。

## 簡易使用例
```python
import torch
from tracknet.models import ViTBackbone, ViTBackboneConfig, UpsamplingDecoder, HeatmapHead

B, C, H, W = 2, 3, 720, 1280
x = torch.randn(B, C, H, W)

backbone = ViTBackbone(ViTBackboneConfig(
    pretrained_model_name="facebook/dinov3-vitb16-pretrain-lvd1689m",
    use_pretrained=False,  # オフライン時はfallback
))

tokens = backbone(x)                         # [B,Hp,Wp,384]
dec = UpsamplingDecoder([384,256,128,64], [2,2,2], out_size=(72,128))
feat = dec(tokens)                            # [B,64,72,128]
head = HeatmapHead(64)
out = head(feat)                              # [B,1,72,128]
print(out.shape)
```

