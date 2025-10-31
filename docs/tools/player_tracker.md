# Player Tracker

TrackNetデータセットのクリップ/ゲームでプレイヤー追跡を実行するツール。

## 概要

TrackNetデータセット構造に対応したプレイヤー追跡ツール。クリップ単位またはゲーム単位でYOLOv8 + DeepSORTを使用してプレイヤーを追跡し、トラックIDを選択する。

## ファイル構成

```
tracknet/tools/
├── player_tracker.py             # プレイヤー追跡ツール
└── utils/
    └── ui/
        ├── player_selector.py    # プレイヤー選択UI
        └── court_selector.py     # コートアノテーションUI
```

## 主要クラス

### PlayerTracker

YOLOv8 + DeepSORTを使用してプレイヤーを追跡するメインクラス。

**メソッド:**
- `track_video(video_path)`: 動画からプレイヤーを追跡
- `select_track_ids(track_history)`: UIでトラックIDを選択
- `save_tracking_results(track_history, output_path)`: 追跡結果を保存
- `get_video_info(video_path)`: 動画情報を取得
- `extract_track_ids(video_path, output_dir, launch_ui=True)`: 追跡とID抽出をまとめて実行

### PlayerSelectorUI

フレーム単位のプレイヤー選択UI。

**機能:**
- フレームスライダーでの動画ナビゲーション
- トラックIDのチェックボックス選択
- リアルタイムのトラッキング結果プレビュー
- 再生/一時停止コントロール

### ClipPlayerUI

クリップ単位のプレイヤー選択UI。

**機能:**
- クリップベースのトラッキング確認
- トラックタイムラインの可視化
- IDジャンプ（ギャップ）の検出
- 長期的なトラッキング継続性の確認

## 使用方法

### 基本的な使用方法

```bash
# データセットのクリップに対してプレイヤー追跡を実行
uv run python -m tracknet.tools.player_tracker --clip data/tracknet/game1/Clip1 --output-dir tracking_output

# データセットのゲーム全体に対してプレイヤー追跡を実行
uv run python -m tracknet.tools.player_tracker --game data/tracknet/game1 --output-dir tracking_output

# プレイヤー追跡とID抽出を実行
uv run python -m tracknet.tools.player_tracker --clip data/tracknet/game1/Clip1 --output-dir tracking_output --extract-ids

# UIなしで自動的に全トラックを選択
uv run python -m tracknet.tools.player_tracker --clip data/tracknet/game1/Clip1 --output-dir tracking_output --extract-ids --skip-ui
```

### トラッキング結果の確認

```bash
# 追跡結果ファイル
tracking_output/Clip1_tracking.json    # トラッキング結果
tracking_output/Clip1_selected.json    # 選択されたトラックID
```

### COCO形式への統合

```bash
# 追跡結果をCOCO形式に追加
uv run python -m tracknet.tools.coco_converter \
  --dataset-root data/tracknet \
  --output data/tracknet/annotations.json \
  --load-existing data/tracknet/annotations.json \
  --add-players tracking_output/Clip1_tracking.json Clip1 tracking_output/Clip1_selected.json
```

### コマンドライン引数

- `--clip`: データセットのクリップディレクトリパス
- `--game`: データセットのゲームディレクトリパス  
- `--model`: YOLOモデルパス（デフォルト: yolov8x.pt）
- `--output-dir`: 追跡結果の保存先ディレクトリ（必須）
- `--extract-ids`: トラックID抽出を実行
- `--skip-ui`: プレイヤー選択UIをスキップ（自動選択）

**注意:** `--clip` または `--game` のいずれかを指定する必要があります。

## 出力形式

### 追跡結果 (JSON)

フレームごとのトラッキングデータ：
```json
[
  [
    {"id": 1, "bbx_xyxy": [x1, y1, x2, y2]},
    {"id": 2, "bbx_xyxy": [x1, y1, x2, y2]}
  ],
  ...
]
```

### 選択されたトラックID (JSON)

UIで選択されたトラックIDのリスト：
```json
[1, 3, 7]
```

### COCO形式での出力

COCO形式に変換されたプレイヤーアノテーション：
```json
{
  "id": 1,
  "image_id": 1,
  "category_id": 2,
  "segmentation": [],
  "area": 2400,
  "bbox": [100, 50, 60, 40],
  "iscrowd": 0,
  "track_id": 1
}
```

## UI操作方法

### PlayerSelectorUI

1. **フレームナビゲーション**: スライダーでフレームを移動
2. **プレイヤー選択**: チェックボックスでトラックIDを選択
3. **再生コントロール**: Play/Pauseボタンで動画を再生
4. **確定**: Doneボタンで選択を完了

### ClipPlayerUI

1. **クリップ選択**: クリップスライダーでクリップを移動
2. **フレーム選択**: フレームスライダーでクリップ内フレームを移動
3. **タイムライン確認**: トラックの継続性とギャップを視覚的に確認
4. **プレイヤー選択**: チェックボックスでトラックIDを選択

## 技術仕様

### YOLO設定

- モデル: YOLOv8x
- 検出クラス: 人間（クラス0のみ）
- 信頼度閾値: 0.5
- デバイス: CUDA（利用可能な場合）

### トラッキング処理

1. 動画をフレーム単位で処理
2. YOLOで人物を検出
3. DeepSORTアルゴリズムでトラッキング
4. フレームごとにトラックIDとバウンディングボックスを記録

### UI実装

- フレームワーク: matplotlib
- ウィジェット: Slider, Button, CheckButtons
- 画像処理: OpenCV
- リアルタイム更新: matplotlibイベントループ

## 依存パッケージ

```bash
uv add ultralytics opencv-python matplotlib
```

## 注意事項

- CUDA環境での実行を推奨（処理速度向上）
- 長時間の動画の場合はメモリ使用量に注意
- UIはmatplotlibベースのため、一部環境で表示が異なる場合あり
- トラックIDは動画ごとに再割り当てされるため、一貫性はない
