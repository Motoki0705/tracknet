# データセット仕様（Section 2）

本ドキュメントは、`tracknet/datasets` 実装の要点と入出力仕様をまとめる。

## 基本設計
- サンプルは「画像テンソル（C,H,W）」「ボール座標（x,y：ピクセル）」「可視性（int）」を返す。
  - 可視性は `0` が非可視、`>0` は可視として扱う（`2`/`3` もマスク上は可視）。
- ヒートマップ生成は `collate` 段で行う（柔軟な解像度に対応するため）。
- 前処理は `PreprocessConfig` で制御（`resize`/`normalize`/`flip_prob`）。`resize` は `(width, height)` のタプル。 

## 主要モジュール
- `tracknet/datasets/base/image_dataset.py`
  - 単画像サンプル用の抽象クラス `BaseImageDataset`。
  - `__getitem__` は `image`/`coord`/`visibility`/`meta` を返す。
- `tracknet/datasets/base/sequence_dataset.py`
  - 時系列サンプル用の抽象クラス `BaseSequenceDataset`。
  - `images: [T,C,H,W]` と `coords/visibility` の系列を返す。
- `tracknet/datasets/utils/augmentations.py`
  - `apply_augmentations_single`（水平反転・リサイズ、座標整合）
  - `to_tensor_and_normalize`（ImageNet正規化）
- `tracknet/datasets/utils/collate.py`
  - `collate_frames` / `collate_sequences`（バッチ化＋ガウスヒートマップ生成）
  - `gaussian_2d`（ヒートマップ核）
- `tracknet/datasets/tracknet_frame.py`
  - `data/tracknet/game*/Clip*/Label.csv` を読み、単画像データセットを構築。
- `tracknet/datasets/tracknet_sequence.py`
  - クリップ内でスライディングウィンドウを作成し、時系列データセットを構築。

## 返却フィールド
- フレーム：
  - `image` (FloatTensor `[C,H,W]`)、`coord` (tuple `(x,y)`)、`visibility` (int)
  - `meta.size = (W,H)`（オリジナル画像サイズ）
- 時系列：
  - `images` (FloatTensor `[T,C,H,W]`)、`coords` (list of `(x,y)`)、`visibility` (list of int)
  - `meta.sizes = [(W,H)] * T`

## ヒートマップ生成
- `collate_*` に `heatmap_size=(W,H)` と `sigma` を渡す（内部で `[H,W]` 形状のヒートマップを生成）。
- 画像座標 `(x,y)` を `heatmap_size` にスケールし、2Dガウスを生成。
- `visibility==0` の場合、ヒートマップはゼロ、マスクもゼロ（ロスから除外）。

## 分割・サブセット
- `configs/data/tracknet.yaml` の `split.train_games` / `split.val_games` を想定し、
  対応するゲームフォルダのみ読み込む。

## 簡易使用例
```python
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tracknet.datasets import (
    TrackNetFrameDataset, TrackNetFrameDatasetConfig,
    collate_frames, PreprocessConfig,
)

data_cfg = OmegaConf.load("configs/data/tracknet.yaml")
pp = PreprocessConfig(
    resize=None,
    normalize=data_cfg.preprocess.get("normalize", True),
    flip_prob=float(data_cfg.preprocess.get("flip_prob", 0.0)),
)
ds = TrackNetFrameDataset(TrackNetFrameDatasetConfig(
    root=data_cfg.root,
    games=data_cfg.split.train_games,
    preprocess=pp,
))
dl = DataLoader(ds, batch_size=4, shuffle=True,
                collate_fn=lambda b: collate_frames(
                    b,
                    heatmap_size=tuple(OmegaConf.load("configs/model/vit_heatmap.yaml").heatmap.size),
                    sigma=float(OmegaConf.load("configs/model/vit_heatmap.yaml").heatmap.sigma),
                ))

batch = next(iter(dl))
print(batch["images"].shape, batch["heatmaps"].shape, batch["masks"].shape)
```

## 形状の想定
- 画像: `[B,C,H,W]`
- ヒートマップ（フレーム）: `[B,1,Hh,Wh]`
- ヒートマップ（時系列）: `[B,T,1,Hh,Wh]`

