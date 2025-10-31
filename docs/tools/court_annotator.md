# Court Annotation Tool

TrackNetデータセット拡張用のコートキーポイントアノテーションツール。

## 概要

テニスコートのキーポイントをアノテーションするツール。カメラ位置がゲーム単位で固定であるため、ゲームレベルでのアノテーションを行い、そのゲーム内の全クリップに適用できる。

## ファイル構成

```
tracknet/tools/
├── court_annotator.py            # メインのコートアノテーションツール
└── utils/ui/
    └── court_selector.py         # コートアノテーションUI
```

## 主要クラス

### CourtKeypointAnnotator

コートキーポイントアノテーションクラス。

**メソッド:**
- `annotate_game(video_path, existing_annotation)`: ゲーム動画のコートをアノテーション
- `save_annotation(annotation, output_path)`: アノテーションを保存
- `load_annotation(annotation_path)`: アノテーションを読み込み
- `validate_annotation(annotation)`: アノテーションの完全性を検証
- `visualize_annotation(video_path, annotation)`: アノテーションを可視化
- `apply_to_game_clips(game_annotation, game_dir, output_dir)`: ゲーム内全クリップに適用

### CourtAnnotationUI

インタラクティブなコートアノテーションUI。

**機能:**
- クリックでのキーポイント配置
- リアルタイムのコートライン可視化
- フレームナビゲーション
- キーポイントリストとステータス管理
- 既存アノテーションの編集

### CourtReviewUI

既存アノテーションの確認・編集UI。

**機能:**
- アノテーションの確認
- キーポイントの選択と削除
- コートラインの可視化
- 変更の保存

## 標準コートキーポイント

15個の標準的なテニスコートキーポイント：

| インデックス | キーポイント名                     |
| :----------- | :--------------------------------- |
| 0            | `far doubles corner left`          |
| 1            | `far doubles corner right`         |
| 2            | `near doubles corner left`         |
| 3            | `near doubles corner right`        |
| 4            | `far singles corner left`          |
| 5            | `near singles corner left`         |
| 6            | `far singles corner right`         |
| 7            | `near singles corner right`        |
| 8            | `far service-line endpoint left`   |
| 9            | `far service-line endpoint right`  |
| 10           | `near service-line endpoint left`  |
| 11           | `near service-line endpoint right` |
| 12           | `far service T`                    |
| 13           | `near service T`                   |
| 14           | `net center`                       |

## スケルトン定義

以下のキーポイントペアが接続され、コートのラインを形成します：

```json
[
  [1, 2],   # far doubles line
  [3, 4],   # near doubles line  
  [1, 3],   # left doubles sideline
  [2, 4],   # right doubles sideline
  [5, 6],   # left singles sideline
  [7, 8],   # right singles sideline
  [9, 10],  # far service line
  [11, 12], # near service line
  [13, 14]  # service T to net center
]
```

## 使用方法

### 基本的な使用方法

```bash
# 新規アノテーション
uv run python -m tracknet.tools.court_annotator --video path/to/game_video.mp4 --output court_annotation.json

# 既存アノテーションの編集
uv run python -m tracknet.tools.court_annotator --video path/to/game_video.mp4 --edit existing.json --output updated.json

# ゲーム内全クリップに適用
uv run python -m tracknet.tools.court_annotator --video path/to/game_video.mp4 --output court.json --apply-to-clips path/to/game_dir/

# 検証と可視化
uv run python -m tracknet.tools.court_annotator --video path/to/game_video.mp4 --output court.json --validate --visualize
```

### コマンドライン引数

- `--video`: ゲーム動画ファイルのパス（必須）
- `--output`: コートアノテーションの保存先（JSON形式）（必須）
- `--edit`: 編集する既存アノテーションファイルのパス
- `--visualize`: 保存後にアノテーション可視化を表示
- `--validate`: 保存前にアノテーションを検証
- `--apply-to-clips`: 指定されたゲームディレクトリ内の全クリップに適用

## UI操作方法

### CourtAnnotationUI

1. **キーポイント配置**: 画像上をクリックして現在のキーポイントを配置
2. **フレーム選択**: スライダーで最適なフレームを見つける
3. **キーポイント削除**: 右クリックで現在のキーポイントを削除
4. **キーポイント移動**: Prev/Nextボタンで前後のキーポイントに移動
5. **クリア**: Clearボタンで現在のキーポイントをクリア
6. **完了**: Doneボタンでアノテーションを完了

### キーボードショートカット

- **左クリック**: キーポイント配置
- **右クリック**: キーポイント削除
- **Prev/Nextボタン**: キーポイント移動
- **Start/Middle/Endボタン**: フレームジャンプ

## 出力形式

### コートアノテーション (JSON)

キーポイント名と座標のマッピング：
```json
{
  "far doubles corner left": [x, y],
  "far doubles corner right": [x, y],
  "near doubles corner left": [x, y],
  "near doubles corner right": [x, y],
  "far singles corner left": [x, y],
  "near singles corner left": [x, y],
  "far singles corner right": [x, y],
  "near singles corner right": [x, y],
  "far service-line endpoint left": [x, y],
  "far service-line endpoint right": [x, y],
  "near service-line endpoint left": [x, y],
  "near service-line endpoint right": [x, y],
  "far service T": [x, y],
  "near service T": [x, y],
  "net center": [x, y]
}
```

### COCO形式への統合

コートアノテーションはCOCO形式のJSONに統合されます：

```json
{
  "id": 1,
  "image_id": 1,
  "category_id": 3,
  "keypoints": [x1, y1, 2, x2, y2, 2, ...],
  "num_keypoints": 15,
  "bbox": [x_min, y_min, width, height],
  "area": width * height
}
```

- `keypoints`: 15個のキーポイント座標と可視性フラグ（2=可視）
- `num_keypoints`: アノテーションされたキーポイント数
- `bbox`: すべてのキーポイントを含むバウンディングボックス

## コートラインの定義

標準的なテニスコートのライン接続（スケルトン）：

```python
skeleton = [
    [1, 2],   # far doubles line
    [3, 4],   # near doubles line  
    [1, 3],   # left doubles sideline
    [2, 4],   # right doubles sideline
    [5, 6],   # left singles sideline
    [7, 8],   # right singles sideline
    [9, 10],  # far service line
    [11, 12], # near service line
    [13, 14], # service T to net center
]
```

## TrackNetデータセットとの連携

### COCO形式への移行

既存のCSV形式からCOCO JSON形式へ移行：

```bash
# CSVからCOCO形式に変換
uv run python -m tracknet.tools.coco_converter \
  --dataset-root data/tracknet \
  --output data/tracknet/annotations.json
```

### データ構造の拡張

```
data/tracknet/
├── annotations.json           # COCO形式の統合アノテーション
├── game1/
│   ├── Clip1/
│   │   ├── Label.csv           # 既存のボールアノテーション（COCO変換後は不要）
│   │   ├── players.json        # プレイヤー追跡結果
│   │   └── court.json          # コートアノテーション
│   └── ...
└── game2/
    └── ...
```

### COCO形式のカテゴリ

```json
{
  "categories": [
    {"id": 1, "name": "ball", "keypoints": 0},
    {"id": 2, "name": "player", "keypoints": 0},
    {"id": 3, "name": "court", "keypoints": 15, "skeleton": [...]}
  ]
}
```

## 注意事項

- カメラ位置が固定であることを前提
- アノテーションは一度ゲーム単位で実施すれば、そのゲーム内の全クリップに適用可能
- キーポイントの順序と命名は標準仕様に準拠
- UIはmatplotlibベースのため、表示環境によっては調整が必要な場合あり
