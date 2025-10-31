# COCO Format Converter

TrackNetデータセットのCSVアノテーションをCOCO JSON形式に変換するツール。

## 概要

既存のTrackNet CSV形式のボールアノテーションをCOCO JSON形式に変換し、プレイヤー追跡とコートキーポイントアノテーションを統合する。COCO形式は標準的なアノテーション形式で、様々な深層学習フレームワークでサポートされている。

## ファイル構成

```
tracknet/tools/
└── coco_converter.py             # COCO形式変換ツール
```

## 主要クラス

### COCOConverter

TrackNetデータセットをCOCO形式に変換するメインクラス。

**メソッド:**
- `convert_csv_to_coco(output_path)`: CSVアノテーションをCOCO形式に変換
- `add_player_tracking(tracking_file, clip_dir, selected_ids)`: プレイヤー追跡結果を追加
- `add_court_annotation(court_file, game_dir)`: コートアノテーションを追加
- `save_coco_data(output_path)`: COCOデータを保存
- `load_existing_coco(coco_file)`: 既存のCOCOファイルを読み込み

## COCOカテゴリ定義

```json
{
  "categories": [
    {
      "id": 1,
      "name": "ball",
      "keypoints": 0,
      "skeleton": []
    },
    {
      "id": 2,
      "name": "player",
      "keypoints": 0,
      "skeleton": []
    },
    {
      "id": 3,
      "name": "court",
      "keypoints": 15,
      "skeleton": [
        [1, 2], [3, 4], [1, 3], [2, 4],
        [5, 6], [7, 8], [9, 10], [11, 12], [13, 14]
      ]
    }
  ]
}
```

## 使用方法

### 基本的な変換

```bash
# CSVからCOCO形式に変換
uv run python -m tracknet.tools.coco_converter \
  --dataset-root data/tracknet \
  --output data/tracknet/annotations.json
```

### プレイヤー追跡結果の追加

```bash
# 既存のCOCOファイルにプレイヤー追跡結果を追加
uv run python -m tracknet.tools.coco_converter \
  --dataset-root data/tracknet \
  --output data/tracknet/annotations.json \
  --load-existing data/tracknet/annotations.json \
  --add-players tracking.json Clip1 selected_ids.json
```

### コートアノテーションの追加

```bash
# コートアノテーションを追加
uv run python -m tracknet.tools.coco_converter \
  --dataset-root data/tracknet \
  --output data/tracknet/annotations.json \
  --load-existing data/tracknet/annotations.json \
  --add-court court.json data/tracknet/game1
```

### 完全な統合

```bash
# すべてのアノテーションを統合
uv run python -m tracknet.tools.coco_converter \
  --dataset-root data/tracknet \
  --output data/tracknet/annotations.json
```

## コマンドライン引数

- `--dataset-root`: TrackNetデータセットのルートディレクトリ（必須）
- `--output`: COCO JSONファイルの出力先（必須）
- `--add-players`: プレイヤー追跡結果を追加（3つの引数: tracking_file clip_dir selected_ids_json）
- `--add-court`: コートアノテーションを追加（2つの引数: court_file game_dir）
- `--load-existing`: 既存のCOCOファイルを読み込む

## 出力形式

### COCOアノテーション構造

```json
{
  "info": {
    "description": "TrackNet Tennis Dataset",
    "version": "2.0",
    "year": 2024,
    "contributor": "TrackNet Project",
    "date_created": "2024-01-01"
  },
  "licenses": [{"id": 1, "name": "Unknown License", "url": ""}],
  "images": [
    {
      "id": 1,
      "width": 1280,
      "height": 720,
      "file_name": "game1/Clip1/frame_001.jpg",
      "license": 1,
      "flickr_url": "",
      "coco_url": "",
      "date_captured": ""
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "segmentation": [],
      "area": 0,
      "bbox": [320.5, 180.2, 1, 1],
      "iscrowd": 0
    },
    {
      "id": 2,
      "image_id": 1,
      "category_id": 2,
      "segmentation": [],
      "area": 2400,
      "bbox": [100, 50, 60, 40],
      "iscrowd": 0,
      "track_id": 1
    },
    {
      "id": 3,
      "image_id": 1,
      "category_id": 3,
      "segmentation": [],
      "area": 50000,
      "bbox": [50, 100, 500, 100],
      "iscrowd": 0,
      "keypoints": [x1, y1, 2, x2, y2, 2, ...],
      "num_keypoints": 15
    }
  ],
  "categories": [...]
}
```

### 各カテゴリのアノテーション形式

#### ボール (category_id: 1)
- `bbox`: [x, y, 1, 1] - 点アノテーションとして小さなバウンディングボックス
- `area`: 0 - 点アノテーションは面積なし

#### プレイヤー (category_id: 2)
- `bbox`: [x1, y1, width, height] - バウンディングボックス
- `area`: width * height
- `track_id`: 追跡ID（追加情報）

#### コート (category_id: 3)
- `keypoints`: [x1, y1, v1, x2, y2, v2, ...] - 15個のキーポイント
- `num_keypoints`: アノテーションされたキーポイント数
- `bbox`: すべてのキーポイントを含むバウンディングボックス
- `area`: バウンディングボックスの面積

## 技術仕様

### 変換処理

1. **CSV読み込み**: pandasを使用してLabel.csvを読み込み
2. **画像検証**: 各フレームに対応する画像ファイルを検索
3. **寸法取得**: PILを使用して画像の幅と高さを取得
4. **COCO形式変換**: COCO仕様に従ってJSON構造を生成

### ID管理

- 画像IDとアノテーションIDは自動的にインクリメント
- 既存のCOCOファイルを読み込む場合、IDを継続
- 重複を避けるため最大ID+1から開始

### エラーハンドリング

- CSVファイルが見つからない場合の警告
- 画像ファイルが見つからない場合のスキップ
- 不正な座標値の検証
- パスの相対・絶対パス対応

## 依存パッケージ

```bash
uv add pandas pillow
```

## ワークフロー例

### 完全なデータセット変換

```bash
# 1. CSVからCOCO形式に変換
uv run python -m tracknet.tools.coco_converter \
  --dataset-root data/tracknet \
  --output data/tracknet/annotations.json

# 2. プレイヤー追跡とID抽出
uv run python -m tracknet.tools.player_tracker \
  --video data/tracknet/game1/Clip1/Clip1.mp4 \
  --output-dir outputs/tracking \
  --extract-ids

# 3. プレイヤー追跡結果をCOCOに追加
uv run python -m tracknet.tools.coco_converter \
  --dataset-root data/tracknet \
  --output data/tracknet/annotations.json \
  --load-existing data/tracknet/annotations.json \
  --add-players outputs/tracking/Clip1_tracking.json data/tracknet/game1/Clip1 outputs/tracking/Clip1_selected.json

# 4. コートアノテーションを追加
uv run python -m tracknet.tools.coco_converter \
  --dataset-root data/tracknet \
  --output data/tracknet/annotations.json \
  --load-existing data/tracknet/annotations.json \
  --add-court data/tracknet/game1/court.json data/tracknet/game1
```

### バッチ処理

```bash
# 複数のクリップでプレイヤー追跡
for clip in data/tracknet/game1/Clip*/; do
  clip_name=$(basename "$clip")
  uv run python -m tracknet.tools.player_tracker \
    --video "$clip/$clip_name.mp4" \
    --output-dir "outputs/tracking/$clip_name" \
    --extract-ids --skip-ui
done

# すべての追跡結果をCOCOに追加
for clip in data/tracknet/game1/Clip*/; do
  clip_name=$(basename "$clip")
  uv run python -m tracknet.tools.coco_converter \
    --dataset-root data/tracknet \
    --output data/tracknet/annotations.json \
    --load-existing data/tracknet/annotations.json \
    --add-players "outputs/tracking/$clip_name/${clip_name}_tracking.json" "data/tracknet/game1/$clip_name" "outputs/tracking/$clip_name/${clip_name}_selected.json"
done
```

## 検証と品質確認

### 変換結果の確認

```python
import json

# COCOファイルを読み込み
with open("data/tracknet/annotations.json", "r") as f:
    coco_data = json.load(f)

# 統計情報を表示
print(f"Images: {len(coco_data['images'])}")
print(f"Annotations: {len(coco_data['annotations'])}")

# カテゴリごとのアノテーション数をカウント
category_counts = {}
for ann in coco_data['annotations']:
    cat_id = ann['category_id']
    category_counts[cat_id] = category_counts.get(cat_id, 0) + 1

print("Category counts:", category_counts)
```

### 可視化

```bash
# COCO形式のデータセットを可視化
uv run python -m tracknet.tools.visualize_coco \
  --coco-file data/tracknet/annotations.json \
  --num-samples 8
```

## 注意事項

- **メモリ使用**: 大規模なデータセットの場合、メモリ使用量に注意
- **パス一貫性**: すべてのファイルパスはデータセットルートからの相対パス
- **座標系**: ピクセル座標系（原点は左上）
- **互換性**: COCO仕様に準拠しているため、様々なフレームワークで使用可能

## 拡張機能

### カスタムカテゴリの追加

```python
# 新しいカテゴリをCOCOデータに追加
new_category = {
    "id": 4,
    "name": "racket",
    "keypoints": 0,
    "skeleton": []
}
converter.coco_data["categories"].append(new_category)
```

### メタデータの追加

```python
# カスタムメタデータを追加
converter.coco_data["info"]["custom_field"] = "custom_value"
```
