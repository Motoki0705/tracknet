# 損失・メトリクス・コールバック（Section 4）

## 損失
- `tracknet/training/losses/heatmap_loss.py`
  - `HeatmapMSELoss`: マスク付きMSE。`[B,1,H,W]` の可視マスクで加重平均。
  - `HeatmapFocalLoss(alpha=2,beta=4)`: CenterNet系のヒートマップ向けFocal変種。
    - 正例: `-(1 - p)^alpha * log(p)`
    - 負例: `-(p^alpha) * (1 - t)^beta * log(1 - p)`
  - `build_heatmap_loss(HeatmapLossConfig)` で選択可能。

## メトリクス
- `tracknet/training/metrics/__init__.py`
  - `heatmap_argmax_coords(hm)` → `[B,2] (x,y)`
  - `heatmap_soft_argmax_coords(hm, beta)` → `[B,2] (x,y)`
  - `l2_error(pred_xy, tgt_xy, visible)` → 平均L2誤差（ヒートマップ座標）
  - `pck_at_r(pred_xy, tgt_xy, visible, r)` → PCK@r（ヒートマップ座標，rピクセル）
  - `visible_from_mask(mask)` → `[B]` の可視フラグ抽出

## コールバック
- `tracknet/training/callbacks/early_stopping.py`
  - `EarlyStopping(patience, min_delta, mode)` → `step(value) -> stop:bool`
- `tracknet/training/callbacks/checkpoint.py`
  - `ModelCheckpoint(mode, save_best_only, save_fn)` → `step(value)` で改善時に保存
- `tracknet/training/callbacks/lr_scheduler.py`
  - `LRSchedulerCallback(scheduler, when)` → `on_batch_end`/`on_epoch_end(metric)`

## 使い方（例）
```python
import torch
from tracknet.training import (
  HeatmapMSELoss, HeatmapFocalLoss, heatmap_argmax_coords,
  l2_error, pck_at_r, visible_from_mask,
)

B,H,W = 4,72,128
pred = torch.rand(B,1,H,W)
true = torch.rand(B,1,H,W)
mask = torch.ones(B,1,H,W)

mse = HeatmapMSELoss()(pred,true,mask)
focal = HeatmapFocalLoss()(pred,true,mask)

pred_xy = heatmap_argmax_coords(pred)
tgt_xy  = heatmap_argmax_coords(true)
vis = visible_from_mask(mask)
print(float(l2_error(pred_xy,tgt_xy,vis)))
print(float(pck_at_r(pred_xy,tgt_xy,vis,r=3.0)))
```

