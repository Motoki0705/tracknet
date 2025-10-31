# トレーナ（Section 5）

本書は `tracknet/training/trainer.py` の概要と使用方法を示す。

## 機能概要
- データ: `TrackNetFrameDataset` / `TrackNetSequenceDataset` を構築し、`collate_frames`/`collate_sequences` でヒートマップ・マスクを生成。
- モデル: `tracknet.models.build_model(cfg.model)` を用いて組み立て、コンフィグに応じて以下を切替。
  - ViT + UpsamplingDecoder + HeatmapHead
  - ConvNeXt + FPNDecoder + HeatmapHead
- 損失: `build_heatmap_loss`（`training.loss` で `mse`/`focal` を切替、既定: MSE）
- 最適化: `AdamW`（既定）、CosineAnnealingLR（既定）
- ループ: 精度選択（`training.precision` または legacy `amp`）、勾配クリップ、検証、ベストチェックポイント保存

## コンフィグの主な項目
- `data.preprocess.{resize, normalize, flip_prob}`
- `data.sequence.{enabled, length, stride}`
- `model.heatmap.{size=[W,H], sigma}`
- `model.decoder.*`（ViT: `channels`, `upsample`, `blocks_per_stage`, `norm`, `activation`, `use_depthwise`, `use_se`, `se_reduction`, `dropout`）
- `model.fpn.*`（ConvNeXt: `lateral_dim`, `fuse`）
- `training.{batch_size, epochs, precision, amp, grad_clip}`
- 損失設定: `training.loss.{name, alpha, beta}`（任意）
- `training.optimizer.{name, lr, weight_decay}`
- `training.scheduler.{name, warmup_epochs}`（一部未使用）
- 任意: `training.limit_train_batches`, `training.limit_val_batches`（スモーク用途）
- パフォーマンス: `training.data_loader.{num_workers,pin_memory,persistent_workers,prefetch_factor,drop_last}`

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
