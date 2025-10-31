# スクリプト（Section 6）

## train
- モジュール: `tracknet/scripts/train.py`
- 役割: OmegaConfで `cfg` を構築し、`Trainer(cfg).train()` を実行。
- 例:
```
uv run python -m tracknet.scripts.train \
  --data tracknet --model vit_heatmap --training default \
  training.epochs=1 training.batch_size=2 \
  training.limit_train_batches=2 training.limit_val_batches=2
```

## train_sequential
- モジュール: `tracknet/scripts/train_sequential.py`
- 役割: 複数のモデルを順次学習する。デフォルトは `vit -> convnext_fpn -> convnext_hrnet`。
- 主な機能:
  - モデル列を指定して順次学習実行
  - 各モデルの学習時間と成功/失敗を記録
  - エラー時の継続実行オプション
  - ドライランモード対応
  - 学習設定の一括上書き対応
- 基本使用例:
```
# デフォルトシーケンスで学習
uv run python -m tracknet.scripts.train_sequential

# ドライランで確認
uv run python -m tracknet.scripts.train_sequential --dry-run

# カスタムエポック数とバッチサイズ
uv run python -m tracknet.scripts.train_sequential --epochs 20 --batch-size 8

# 特定のモデルのみ学習
uv run python -m tracknet.scripts.train_sequential --models vit_heatmap convnext_fpn_heatmap

# 学習設定を上書き
uv run python -m tracknet.scripts.train_sequential training.optimizer.lr=1e-4 training.precision=fp16

# エラーでも継続実行
uv run python -m tracknet.scripts.train_sequential --continue-on-error

# モデル間で待機時間を設定
uv run python -m tracknet.scripts.train_sequential --sleep-between 30
```
- オプション:
  - `--data`: データ設定名（デフォルト: tracknet）
  - `--training`: 学習設定名（デフォルト: default）
  - `--models`: 学習するモデル列（デフォルト: vit_heatmap convnext_fpn_heatmap convnext_hrnet_heatmap）
  - `--epochs`: エポック数（training.epochsを上書き）
  - `--batch-size`: バッチサイズ（training.batch_sizeを上書き）
  - `--dry-run`: 実行せずコマンドのみ表示
  - `--continue-on-error`: 失敗しても次のモデルを学習
  - `--sleep-between`: モデル間の待機時間（秒）
  - `--output-dir`: 出力ディレクトリ（training.output_dirを上書き）
- 出力形式:
```
=== TrackNet Sequential Training ===
Models: vit_heatmap -> convnext_fpn_heatmap -> convnext_hrnet_heatmap
Data config: tracknet
Training config: default
Epochs: 20
Batch size: 8

=== Training Model 1/3: vit_heatmap ===
Executing: uv run python -m tracknet.scripts.train --data tracknet --model vit_heatmap --training default training.epochs=20 training.batch_size=8
✓ Completed successfully in 1234.56s

=== Training Model 2/3: convnext_fpn_heatmap ===
...

=== Training Summary ===
✓ vit_heatmap: 1234.56s (exit: 0)
✓ convnext_fpn_heatmap: 987.65s (exit: 0)
✓ convnext_hrnet_heatmap: 876.54s (exit: 0)

Overall: 3/3 models successful
Total time: 3098.75s
```

## eval
- モジュール: `tracknet/scripts/eval.py`
- 役割: `cfg` 構築→チェックポイント読込→検証データで評価。
- 例:
```
uv run python -m tracknet.scripts.eval \
  --data tracknet --model vit_heatmap --training default \
  --checkpoint outputs/checkpoints/best_*.pt
```
- 出力: `eval_loss`, `l2`, `pck@3`

## predict
- モジュール: `tracknet/scripts/predict.py`
- 役割: 単画像で前向きし、ヒートマップ画像とオーバーレイを保存。
- 例:
```
uv run python -m tracknet.scripts.predict \
  --data tracknet --model vit_heatmap --training default \
  --image data/tracknet/game1/Clip1/0000.jpg \
  --checkpoint outputs/checkpoints/best_*.pt
```
- 出力: `cfg.runtime.output_root/predictions/<stem>_heatmap.png`, `<stem>_overlay.png`

