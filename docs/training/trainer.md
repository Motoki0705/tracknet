# トレーナ（Section 5）

本書は PyTorch Lightning ベースの学習基盤の概要と使用方法を示す。

エントリポイントは `tracknet/scripts/train.py`、学習ロジックは以下に分離される。
- LightningModule: `tracknet/training/lightning_module.py`（`PLHeatmapModule`）
- DataModule: `tracknet/training/lightning_datamodule.py`（`TrackNetDataModule`）

## 機能概要
- データ: 既存 `TrackNetFrameDataset` / `TrackNetSequenceDataset` と `collate_frames` / `collate_sequences` をそのまま利用。
- モデル: `tracknet.models.build_model(cfg.model)` で組み立て。
  - ViT + UpsamplingDecoder + HeatmapHead
  - ConvNeXt + FPNDecoder + HeatmapHead
- 損失: `build_heatmap_loss`（`training.loss` で `mse`/`focal` 切替、既定: MSE）
- 最適化: `AdamW`（既定）、CosineAnnealingLR（既定）
- ループ: Lightningの`Trainer`に委譲（AMP、分散、勾配クリップ、進捗バー、検証、ベストチェックポイント）
- コールバック: Lightning標準
  - `ModelCheckpoint(monitor='val/loss', mode='min', save_top_k=1)`
  - `LearningRateMonitor`
  - （任意）`EarlyStopping(monitor='val/loss')`

### 改善点（本更新）
- バックボーンの段階的フリーズ/解凍:
  - `training.backbone_freeze_epochs: int` の間はバックボーンを凍結し、その後自動で解凍します。
  - Optimizerは全パラメータを保持しているため、解凍後も再構築不要で学習が継続します。
- 学習率ウォームアップ:
  - `training.scheduler.warmup_epochs: int` と `training.scheduler.warmup_start_factor: float` を追加。
  - `LinearLR`（ウォームアップ）→`CosineAnnealingLR` の逐次スケジューラを採用します。

## 精度（precision）
- `training.precision` ∈ {`fp32`, `fp16`, `bf16`} を Lightning の precision にマッピング
  - `fp32` → `32-true`
  - `fp16` → `16-mixed`（CUDAのみ、CPUでは自動で`32-true`にフォールバック）
  - `bf16` → `bf16-mixed`（対応GPUのみ）
- 互換のため `training.amp=true` は `fp16` 相当として扱われる。

## コンフィグの主な項目
- `data.preprocess.{resize, normalize, flip_prob}`
- `data.sequence.{enabled, length, stride}`
- `model.heatmap.{size=[W,H], sigma}`
- `model.decoder.*`（ViT: `channels`, `upsample`, `blocks_per_stage`, `norm`, `activation`, `use_depthwise`, `use_se`, `se_reduction`, `dropout`）
- `model.fpn.*`（ConvNeXt: `lateral_dim`, `fuse`）
- `training.{batch_size, epochs, precision, amp, grad_clip}`
- 損失設定: `training.loss.{name, alpha, beta}`（任意）
- `training.optimizer.{name, lr, weight_decay}`
- `training.scheduler.{name}`（Cosineが既定）
- `training.backbone_freeze_epochs: int`（先頭Nエポックのみバックボーン凍結）
- `training.scheduler.{warmup_epochs, warmup_start_factor}`（線形ウォームアップ）
- 任意: `training.limit_train_batches`, `training.limit_val_batches`（0または未指定で全バッチ）
- パフォーマンス: `training.data_loader.{num_workers,pin_memory,persistent_workers,prefetch_factor,drop_last}`

## 使用例
```
uv run python -m tracknet.scripts.train \
  --data tracknet --model vit_heatmap --training default \
  training.epochs=10 training.batch_size=2 training.precision=fp16 \
  training.backbone_freeze_epochs=2 \
  training.scheduler.warmup_epochs=3 training.scheduler.warmup_start_factor=0.1 \
  training.limit_train_batches=2 training.limit_val_batches=2
```

## 出力
- ログ（CSV）: `outputs/logs/<run_id>/version_*/metrics.csv`（Lightning `CSVLogger`）
- プレビュー画像: `outputs/logs/<run_id>/(inputs|overlays)/epoch***/sample_*.png`
- チェックポイント: `outputs/checkpoints/best_<run_id>.ckpt`（最良指標で更新）

## 実装ファイルの要点
- `tracknet/training/lightning_module.py`
  - `training_step` / `validation_step` で損失を計算し `train/loss` / `val/loss` を `self.log`。
  - 先頭検証バッチで入力と予測ヒートマップのプレビューPNGを保存。
  - `configure_optimizers` で Optimizer と Scheduler を返却。
- `tracknet/training/lightning_datamodule.py`
  - `setup(stage)` で Dataset を構築し、`train_dataloader` / `val_dataloader` を返す。
  - DataLoader設定は `cfg.training.data_loader.*` から反映。
