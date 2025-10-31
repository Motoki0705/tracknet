# HRNet Decoder

HRNet風のマルチ解像度融合デコーダーです。ConvNeXtなどのバックボーンから出力される複数スケールの特徴マップを、高解像度のまま情報交換を行いながら融合します。

## アーキテクチャ

### 特徴
- **マルチ解像度融合**: 複数の解像度の特徴マップを並列に処理
- **情報交換ユニット**: 異なる解像度間で情報を交換・融合
- **高解像度出力**: 最終的に最高解像度（C2相当）の特徴マップを出力

### 構成要素
1. **Lateral Convs**: 各スケールの特徴マップを指定されたチャネル幅に変換
2. **Exchange Units**: HRNetのmulti-resolution fusionを簡略化したもの
3. **Head Conv**: 最終出力チャネル数への変換

## コンフィグパラメータ

```yaml
hrnet:
  in_channels: [128, 256, 512, 1024]    # ConvNeXtの各ステージ出力チャネル
  widths: [64, 128, 256, 512]           # 各分岐のチャネル幅（高→低解像度）
  num_units: 2                          # ExchangeUnitの反復回数
  out_channels: 256                     # 最終出力チャネル数（省略時はwidths[0]）
```

### パラメータ詳細
- **in_channels**: バックボーンからの入力チャネル [C2, C3, C4, C5]
- **widths**: 各解像度分岐のチャネル幅。高解像度ほど細かく、低解像度ほど太く設定
- **num_units**: 情報交換を繰り返す回数。多いほど複雑な融合が可能
- **out_channels**: ヒートマップヘッドへの入力チャネル数

## 使用例

### コンフィグ例
```yaml
model_name: "convnext_hrnet_heatmap"
pretrained_model_name: facebook/dinov3-convnext-base-pretrain-lvd1689m

backbone:
  freeze: true
  return_stages: [1, 2, 3, 4]
  device_map: auto
  local_files_only: true

hrnet:
  in_channels: [128, 256, 512, 1024]
  widths: [64, 128, 256, 512]
  num_units: 2
  out_channels: 256

heatmap:
  size: [256, 144]
  sigma: 2.0
```

### Pythonでの使用
```python
from tracknet.models.build import build_model
from omegaconf import OmegaConf

# コンフィグ読み込み
cfg = OmegaConf.load("configs/model/convnext_hrnet_heatmap.yaml")

# モデル構築
model = build_model(cfg.model)

# フォワードパス
images = torch.randn(2, 3, 224, 224)
heatmaps = model(images)  # [2, 1, 144, 256]
```

## 利点

1. **高解像度維持**: 低解像度の特徴をアップサンプルせず、高解像度のまま処理
2. **効率的な融合**: 異なるスケールの情報を効果的に交換
3. **スケーラブル**: widthsやnum_unitsでモデル容量を調整可能
4. **計算効率**: FPNに比べて不要なアップサンプルが少ない

## 注意点

- ConvNeXtなどの階層的バックボーンとの組み合わせを想定
- 入力は [C2, C3, C4, C5] の順（高→低解像度）
- 出力はC2解像度（入力の1/4サイズ）
