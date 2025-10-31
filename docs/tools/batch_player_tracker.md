# Batch Player Tracker

TrackNetデータセットの全ゲーム・クリップで一括プレイヤー追跡を実行するツール。

## 概要

データセット内のすべてのクリップに対してプレイヤー追跡を一括実行し、結果を保存する。その後、ID選択とCOCO形式へのマージを効率的に行うためのワークフローを提供する。

## ファイル構成

```
tracknet/tools/
└── batch_player_tracker.py      # バッチプレイヤー追跡ツール
```

## ワークフロー

### ステップ1: 一括追跡実行

```bash
# 全クリップでプレイヤー追跡を実行
uv run python -m tracknet.tools.batch_player_tracker \
  --dataset-root data/tracknet \
  --output-dir outputs/batch_tracking
```

### ステップ2: ID選択

```bash
# UIでID選択を実行
uv run python -m tracknet.tools.batch_player_tracker \
  --dataset-root data/tracknet \
  --output-dir outputs/batch_tracking \
  --extract-ids

# 自動で全IDを選択
uv run python -m tracknet.tools.batch_player_tracker \
  --dataset-root data/tracknet \
  --output-dir outputs/batch_tracking \
  --extract-ids --skip-ui
```

### ステップ3: COCO形式にマージ

```bash
# すべての結果をCOCO形式にマージ
uv run python -m tracknet.tools.coco_converter \
  --dataset-root data/tracknet \
  --output data/tracknet/annotations.json \
  --load-existing data/tracknet/annotations.json \
  --add-players outputs/batch_tracking/game1/Clip1/Clip1_tracking.json Clip1 outputs/batch_tracking/game1/Clip1/Clip1_selected.json \
  --add-players outputs/batch_tracking/game1/Clip2/Clip2_tracking.json Clip2 outputs/batch_tracking/game1/Clip2/Clip2_selected.json \
  # ... (すべてのクリップを追加)
```

## コマンドライン引数

- `--dataset-root`: TrackNetデータセットのルートディレクトリ（デフォルト: data/tracknet）
- `--output-dir`: バッチ処理結果の保存先（デフォルト: outputs/batch_tracking）
- `--model`: YOLOモデルパス（デフォルト: yolov8x.pt）
- `--extract-ids`: 追跡後にID選択を実行
- `--skip-ui`: UIをスキップして自動選択
- `--force`: 既存の結果を上書き

## 出力構造

```
outputs/batch_tracking/
├── batch_summary.json           # 処理サマリー
├── game1/
│   ├── Clip1/
│   │   ├── Clip1_tracking.json  # 追跡結果
│   │   └── Clip1_selected.json  # 選択されたID
│   ├── Clip2/
│   │   ├── Clip2_tracking.json
│   │   └── Clip2_selected.json
│   └── ...
├── game2/
│   └── ...
└── ...
```

## 完全な実行例

### 1. 全データセットの一括処理

```bash
# ステップ1: 一括追跡
uv run python -m tracknet.tools.batch_player_tracker \
  --dataset-root data/tracknet \
  --output-dir outputs/batch_tracking

# ステップ2: ID選択（自動）
uv run python -m tracknet.tools.batch_player_tracker \
  --dataset-root data/tracknet \
  --output-dir outputs/batch_tracking \
  --extract-ids --skip-ui
```

### 2. COCO形式へのマージ

バッチ処理完了後、自動生成されるコマンドを使用：

```bash
# batch_summary.jsonからマージコマンドを生成
python -c "
import json
with open('outputs/batch_tracking/batch_summary.json', 'r') as f:
    summary = json.load(f)

print('uv run python -m tracknet.tools.coco_converter \\')
print('  --dataset-root data/tracknet \\')
print('  --output data/tracknet/annotations.json \\')
print('  --load-existing data/tracknet/annotations.json')

for clip_key in sorted(summary['selected_ids'].keys()):
    game_name, clip_name = clip_key.split('/')
    tracking_file = summary['tracking_results'][clip_key]['tracking_file']
    selected_file = summary['tracking_results'][clip_key]['selected_file']
    print(f'  --add-players {tracking_file} {clip_name} {selected_file} \\')
print('  --add-court data/tracknet/game1/court.json data/tracknet/game1')
" > merge_command.sh

# 実行
bash merge_command.sh
```

### 3. 結果の確認

```bash
# COCOファイルの統計を確認
python -c "
import json
with open('data/tracknet/annotations.json', 'r') as f:
    coco = json.load(f)

print(f'Images: {len(coco[\"images\"])}')
print(f'Annotations: {len(coco[\"annotations\"])}')

# カテゴリごとのカウント
from collections import defaultdict
counts = defaultdict(int)
for ann in coco['annotations']:
    counts[ann['category_id']] += 1

print('By category:')
for cat_id, count in sorted(counts.items()):
    cat_name = next(c['name'] for c in coco['categories'] if c['id'] == cat_id)
    print(f'  {cat_name} (ID {cat_id}): {count}')
"
```

## 処理時間の目安

- **単一クリップ**: 約1-3分（クリップ長による）
- **1ゲーム（10クリップ）**: 約10-30分
- **全データセット（10ゲーム）**: 約1-3時間

※GPU環境により大幅に高速化可能

## メモリ使用量

- **追跡処理**: 約2-4GB（モデル読み込み込み）
- **バッチ処理**: 追跡結果を逐次保存するため、メモリ使用量は一定

## エラーハンドリング

### 動画ファイルが見つからない場合
```
Warning: No video file found in data/tracknet/game1/Clip1
```
→ 該当クリップをスキップして続行

### 既存の結果がある場合
```
Skipping game1/Clip1 (already exists)
```
→ `--force` オプションで上書き可能

### 追跡エラー
```
Error processing game1/Clip2: [エラー詳細]
```
→ 該当クリップをスキップして続行

## 中断と再開

処理はクリップ単位で独立しているため、途中で中断しても再開可能：

```bash
# 特定のゲームから再開
uv run python -m tracknet.tools.batch_player_tracker \
  --dataset-root data/tracknet \
  --output-dir outputs/batch_tracking \
  --force  # 未完了のクリップのみ処理
```

## カスタマイズ

### 特定のゲームのみ処理

```python
# batch_player_tracker.py を変更
def find_all_clips(dataset_root: Path, target_games: List[str] = None) -> List[Tuple[str, str, Path]]:
    clips = []
    game_dirs = [d for d in dataset_root.iterdir() 
                if d.is_dir() and d.name.startswith("game")]
    
    if target_games:
        game_dirs = [d for d in game_dirs if d.name in target_games]
    
    # ... 残りの処理
```

### 並列処理

大規模データセットの場合、並列処理で高速化：

```bash
# GNU parallelを使用（例）
find data/tracknet -name "Clip*" -type d | \
  parallel -j 4 'uv run python -m tracknet.tools.player_tracker --clip {} --output-dir outputs/parallel/$(basename $(dirname {}))/$(basename {}) --extract-ids --skip-ui'
```
