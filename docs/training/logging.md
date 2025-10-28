# ログ・可視化（Section 7）

## 概要
- `tracknet/utils/logging.py` に、スカラーのロガー（CSV/TensorBoard任意）と可視化ユーティリティを提供。
- トレーナは各エポックで `train/loss`, `val/loss` を `outputs/logs/scalars.csv` に記録し、検証の最初のバッチから最大4枚のオーバーレイ（`outputs/logs/overlays/epochXXX/*.png`）を保存する。

## モジュール
- `Logger`: `LoggerConfig(log_dir, use_tensorboard=False)`
  - `log_scalar(tag, value, step)` → CSV（`scalars.csv`）およびTensorBoard（利用可能時）へ記録
  - `close()`
- `tensor_to_pil(img, denormalize=True, mean, std)`
- `save_heatmap_png(hm, path)`
- `save_overlay_from_tensor(img_t, hm, path, alpha=0.5, denormalize=True)`

## 使い方（Trainer内での利用）
- 自動的に `outputs/logs` 配下にスカラーとオーバーレイが保存される。
- ImageNet正規化を用いている場合は自動で逆正規化してオーバーレイを生成。

## 補足
- TensorBoard出力は `torch.utils.tensorboard` が利用可能な場合のみ有効化。既定ではCSVのみ。
- 追加の可視化（例: 予測ヒートマップ単体保存）は `save_heatmap_png` を利用。

