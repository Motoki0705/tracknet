# TrackNetデータセット可視化ツール

## 概要

`visualize_dataset.py`ツールはTrackNetフレームデータセットの可視化機能を提供します。OmegaConf設定から`TrackNetFrameDataset`をインスタンス化し、ヒートマップを元画像に重ねて表示することで、データ読み込みとヒートマップ生成を確認できます。

## 機能

- **OmegaConf連携**: YAMLファイルからデータセット設定を読み込み
- **ヒートマップ生成**: 目標座標からガウシアンヒートマップを生成
- **オーバーレイ可視化**: ヒートマップを元画像に重ねて表示（透明度調整可能）
- **バッチ可視化**: 複数サンプルをグリッドレイアウトで表示
- **柔軟な設定**: ヒートマップサイズ、シグマ、可視化パラメータをカスタマイズ可能

## 使用方法

### 基本的な使用方法

```bash
uv run python -m tracknet.tools.visualize_dataset --config tracknet/tools/tracknet_frame_config.yaml
```

### 詳細な使用方法

```bash
uv run python -m tracknet.tools.visualize_dataset \
    --config tracknet/tools/tracknet_frame_config.yaml \
    --num-samples 12 \
    --heatmap-size 128 128 \
    --sigma 3.0 \
    --save visualization.png
```

### コマンドライン引数

- `--config`: データセット設定ファイルのパス（必須）
- `--num-samples`: 可視化するサンプル数（デフォルト: 8）
- `--heatmap-size`: ヒートマップサイズを幅 高さで指定（デフォルト: 64 64）
- `--sigma`: ヒートマップ生成のガウシアンシグマ（デフォルト: 2.0）
- `--save`: 可視化画像の保存先パス（オプション）

## 設定ファイル形式

設定ファイルは以下の構造に従います：

```yaml
# tracknet/tools/tracknet_frame_config.yaml
dataset:
  root: "data/tracknet"           # TrackNetデータのルートディレクトリ
  games: ["game1"]                # 含めるゲーム
  preprocess:
    resize: [640, 360]           # 可視化用のリサイズ（オプション）
    normalize: false             # 可視化のために[0,1]範囲を維持
    flip_prob: 0.0               # 可視化のために拡張なし
```

## 出力

ツールは以下を含む画像グリッドを表示します：
- ヒートマップオーバーレイ付きの元画像
- 目標座標と可視性ステータス
- 各サンプルのゲームとクリップ情報
- 可視性を向上させるための調整可能な透明度

各サブプロットに表示：
- サンプルインデックス
- 目標座標 (x, y)
- 可視性フラグ (0または1)
- ゲームとクリップ識別子

## 実装の詳細

### データ読み込み
- OmegaConfからインスタンス化された`TrackNetFrameDataset`を使用
- TrackNetデータセットから画像と対応する座標を読み込み
- 設定で指定された前処理を適用

### ヒートマップ生成
- 一貫したヒートマップ生成のために`collate_frames`ユーティリティを使用
- 目標座標を中心としたガウシアンヒートマップ
- 設定可能なヒートマップサイズとシグマパラメータ

### 可視化
- ヒートマップ用のjetカラーマップを使用したmatplotlibベースの可視化
- オーバーレイ用の調整可能なアルファブレンディング
- 自動サブプロットグリッド配置
- オプションのファイル保存

## 依存関係

- `matplotlib`: 可視化とプロット用
- `numpy`: 配列操作用
- `torch`: テンソル操作とヒートマップ生成用
- `omegaconf`: 設定読み込み用
- `PIL`: 画像処理用（データセット経由）

## 出力例

正常に実行されると、ツールは以下を行います：
1. 指定された設定からデータセットを読み込み
2. 要求された数のサンプルのヒートマップを生成
3. グリッドレイアウトで重ね合わせた可視化を表示
4. オプションで結果を指定されたパスに保存

可視化は以下を確認するのに役立ちます：
- データセットからの正しいデータ読み込み
- 適切な座標スケーリングとヒートマップ生成
- 適切な前処理設定
- 異なるゲーム/クリップ間のデータ品質とカバレッジ

## プログラムでの使用

可視化関数はプログラムからも使用できます：

```python
from tracknet.tools.visualize_dataset import (
    load_dataset_from_config,
    generate_heatmap_for_sample,
    overlay_heatmap_on_image,
    visualize_samples,
)

# データセット読み込み
dataset = load_dataset_from_config("tracknet/tools/tracknet_frame_config.yaml")

# 単一サンプルのヒートマップを生成
sample = dataset[0]
heatmap = generate_heatmap_for_sample(sample, heatmap_size=(64, 64), sigma=2.0)

# オーバーレイ可視化を作成
overlay = overlay_heatmap_on_image(sample["image"], heatmap, alpha=0.4)

# 複数サンプルを可視化
visualize_samples(dataset, num_samples=8, save_path="output.png")
```
