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

