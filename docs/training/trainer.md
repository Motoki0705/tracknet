# トレーナ（Section 5）

本書は `tracknet/training/trainer.py` の概要と使用方法を示す。

## 機能概要
- データ: `TrackNetFrameDataset` / `TrackNetSequenceDataset` を構築し、`collate_frames`/`collate_sequences` でヒートマップ・マスクを生成。
- モデル: コンフィグに応じて以下を切替。
  - ViT + UpsamplingDecoder + HeatmapHead
  - ConvNeXt + FPNDecoder + HeatmapHead
- 損失: `build_heatmap_loss`（既定: MSE）
- 最適化: `AdamW`（既定）、CosineAnnealingLR（既定）
- ループ: AMP（任意）、勾配クリップ（任意）、検証、ベストチェックポイント保存

## コンフィグの主な項目
- `data.preprocess.{resize, normalize, flip_prob}`
- `data.sequence.{enabled, length, stride}`
- `model.heatmap.{size=[W,H], sigma}`
- `model.decoder.*` or `model.fpn.*`（選択的）
- `training.{batch_size, epochs, amp, grad_clip}`
- `training.optimizer.{name, lr, weight_decay}`
- `training.scheduler.{name, warmup_epochs}`（一部未使用）
- 任意: `training.limit_train_batches`, `training.limit_val_batches`（スモーク用途）

## 使用例
```
uv run python -m tracknet.scripts.train \
  --data tracknet --model vit_heatmap --training default \
  training.epochs=1 training.batch_size=2 \
  training.limit_train_batches=2 training.limit_val_batches=2
```

## 出力
- 標準出力: 各エポックの `train_loss` / `val_loss`
- チェックポイント: `outputs/checkpoints/best_*.pt`（最良指標で更新）

