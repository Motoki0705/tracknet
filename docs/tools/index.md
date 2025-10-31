# TrackNet Tools

TrackNetデータセット拡張用ツール群のドキュメント。

## 概要

TrackNetプロジェクトのデータセットを拡張するためのツール群。ボール検出に加えて、プレイヤー追跡とコートキーポイントアノテーション機能を提供する。

## ツール一覧

### データセット可視化ツール

- **[visualize_dataset.md](visualize_dataset.md)**: TrackNetデータセットのボール検出結果を可視化

### COCO形式変換ツール

- **[coco_converter.md](coco_converter.md)**: CSVアノテーションをCOCO JSON形式に変換

### プレイヤー追跡ツール

- **[batch_player_tracker.md](batch_player_tracker.md)**: 全データセットの一括プレイヤー追跡

### コートアノテーションツール

- **[court_annotator.md](court_annotator.md)**: テニスコートのキーポイントアノテーションツール

## 共通の特徴

### UI実装
- matplotlibベースのインタラクティブUI
- リアルタイムプレビュー機能
- 直感的な操作体系

### 出力形式
- COCO JSON形式でのデータ保存
- 既存TrackNet CSV形式との互換性
- 拡張性のある構造

### データセット連携
- 既存のTrackNetデータセット構造を拡張
- ボール、プレイヤー、コート情報の統合
- ゲーム/クリップ階層構造の維持

## インストール

```bash
# 必要な依存パッケージをインストール
uv add ultralytics opencv-python matplotlib
```

## ワークフロー

### 1. データセット確認
```bash
# 既存データセットを可視化
uv run python -m tracknet.tools.visualize_dataset --config configs/dataset/tracknet.yaml
```

### 2. COCO形式への変換
```bash
# CSVからCOCO形式に変換
uv run python -m tracknet.tools.coco_converter --dataset-root data/tracknet --output annotations.json
```

### 3. 一括プレイヤー追跡
```bash
# 全データセットで一括プレイヤー追跡
uv run python -m tracknet.tools.batch_player_tracker \
  --dataset-root data/tracknet \
  --output-dir outputs/batch_tracking \
  --extract-ids --skip-ui
```

### 4. COCO形式へのマージ
```bash
# すべてのアノテーションをCOCO形式に統合
uv run python -m tracknet.tools.coco_converter \
  --dataset-root data/tracknet \
  --output data/tracknet/annotations.json \
  --load-existing data/tracknet/annotations.json \
  --add-players outputs/batch_tracking/game1/Clip1/Clip1_tracking.json Clip1 outputs/batch_tracking/game1/Clip1/Clip1_selected.json \
  # ... (すべてのクリップを追加)
```

### 5. コートアノテーション
```bash
# ゲーム単位でコートをアノテーション
uv run python -m tracknet.tools.court_annotator --video game_video.mp4 --output court.json
```

## 拡張データセット構造

```
data/tracknet/
├── annotations.json           # COCO形式の統合アノテーション
├── game1/
│   ├── Clip1/
│   │   ├── Label.csv           # 既存：ボール座標（COCO変換後は不要）
│   │   ├── players.json        # 新規：プレイヤー追跡結果
│   │   └── court.json          # 新規：コートキーポイント
│   ├── Clip2/
│   │   ├── Label.csv
│   │   ├── players.json
│   │   └── court.json
│   └── court.json              # ゲーム単位のコートアノテーション
├── game2/
│   └── ...
└── clip_annotations/           # 生成されたクリップ別アノテーション
    ├── Clip1_court.json
    ├── Clip2_court.json
    └── ...
```

## 技術仕様

### プレイヤー追跡
- **アルゴリズム**: YOLOv8 + DeepSORT
- **検出対象**: 人物（クラス0）
- **出力**: トラックIDとバウンディングボックス
- **UI機能**: フレーム/クリップ単位の確認

### コートアノテーション
- **キーポイント数**: 15個の標準コートポイント
- **適用単位**: ゲーム単位（カメラ固定）
- **出力**: キーポイント座標とスケルトン接続情報
- **UI機能**: クリック配置、リアルタイム可視化

### データ形式
- **座標系**: ピクセル座標（原点は左上）
- **フォーマット**: COCO JSON形式
- **カテゴリ**: ball(1), player(2), court(3)
- **コートキーポイント**: 15個の標準キーポイントとスケルトン定義

## 使用例

### 完全なワークフロー例

```bash
# 1. 既存データセットを確認
uv run python -m tracknet.tools.visualize_dataset --config configs/dataset/tracknet.yaml --num-samples 8

# 2. CSVからCOCO形式に変換
uv run python -m tracknet.tools.coco_converter --dataset-root data/tracknet --output data/tracknet/annotations.json

# 3. 全データセットで一括プレイヤー追跡とID選択
uv run python -m tracknet.tools.batch_player_tracker \
  --dataset-root data/tracknet \
  --output-dir outputs/batch_tracking \
  --extract-ids --skip-ui

# 4. COCO形式にマージ（自動生成コマンドを使用）
python -c "
import json
with open('outputs/batch_tracking/batch_summary.json', 'r') as f:
    summary = json.load(f)

print('uv run python -m tracknet.tools.coco_converter \\\\')
print('  --dataset-root data/tracknet \\\\')
print('  --output data/tracknet/annotations.json \\\\')
print('  --load-existing data/tracknet/annotations.json')

for clip_key in sorted(summary['selected_ids'].keys()):
    game_name, clip_name = clip_key.split('/')
    tracking_file = summary['tracking_results'][clip_key]['tracking_file']
    selected_file = summary['tracking_results'][clip_key]['selected_file']
    print(f'  --add-players {tracking_file} {clip_name} {selected_file} \\\\')
" | bash

# 5. 結果を確認
python -c "
import json
with open('data/tracknet/annotations.json', 'r') as f:
    coco = json.load(f)
print(f'Images: {len(coco[\"images\"])}')
print(f'Annotations: {len(coco[\"annotations\"])}')
"
```

## 注意事項

- **計算リソース**: CUDA環境での実行を推奨
- **メモリ**: 長時間動画の場合はメモリ使用量に注意
- **品質**: アノテーション品質がモデル性能に直接影響
- **一貫性**: ゲーム内でのコートアノテーションの一貫性が重要

## 今後の拡張

- マルチカメラ対応
- 自動コート検出機能
- プレイヤー行動分析
- リアルタイムアノテーション支援
