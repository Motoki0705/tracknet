# モデルビルド (`tracknet.models.build`)

## 目的
`tracknet.models.build` はモデル組み立て処理を一箇所に集約し、トレーナーやスクリプト側の責務を軽量化する。`build_model(model_cfg)` を呼び出すだけで、ViT／ConvNeXt いずれの構成でも適切なバックボーン・デコーダ・ヘッドが初期化される。

## 主なエントリ
- `HeatmapModel(model_cfg)`  
  - コンフィグを受け取り、内部でバックボーン／デコーダ／ヘッドを束ねた `nn.Module` を構築。
  - `model_cfg.decoder` が存在すれば ViT + Upsampling、`model_cfg.fpn` が存在すれば ConvNeXt + FPN を採用。
- `build_model(model_cfg)`  
  - `HeatmapModel` の薄いラッパー。トレーナー／スクリプトはこの関数のみを使用する。

## 処理概要
1. `heatmap.size` から最終解像度 `(Hh, Wh)` を算出。
2. `model.backbone` の設定（`freeze`, `device_map`, など）に基づきバックボーンを初期化。
3. それぞれの構成に応じてアップサンプラ（`UpsamplingDecoder`）または FPN（`FPNDecoder`）を準備。
4. `HeatmapHead` を付与し、必要に応じてバックボーンの勾配を停止。

## 利用例
```python
from omegaconf import OmegaConf
from tracknet.models import build_model

cfg = OmegaConf.load("configs/model/vit_heatmap.yaml")
model = build_model(cfg)
```

## トレーナーとの連携
`tracknet/training/trainer.py` では `build_model(self.cfg.model)` を呼び出し、返却された `nn.Module` を `.to(device)` して使用する。これによりトレーニング側はアーキテクチャの詳細から切り離される。
