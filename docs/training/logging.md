# ログ・可視化（Section 7）

## 概要
- `tracknet/utils/logging.py` は、スカラーログ（CSV/TensorBoard）と、画像・ヒートマップ保存ユーティリティを提供する。
- Logger は実験ごとに固有のサブディレクトリ（`<log_root>/<run_id>/`）を自動作成し、他の実験と混ざらないようにする。`run_id` が未指定の場合はタイムスタンプを用いる。
- トレーナは検証バッチからランダムに抽出したサンプルを用いて、元画像(`inputs/epochXXX/`)とヒートマップオーバーレイ(`overlays/epochXXX/`)を保存する。サンプル数は最大4枚で、エポック内では同じインデックスを繰り返さない。

## モジュール
- `Logger`: `LoggerConfig(log_dir, run_id=None, use_tensorboard=False)`
  - `log_scalar(tag, value, step)` → 実験サブディレクトリ配下の `scalars.csv` および TensorBoard（利用可能時）へ記録する。
  - `close()` → TensorBoard の flush/close を実行。
- `tensor_to_pil(img, denormalize=True, mean, std)` → テンソル画像をPILに変換。
- `save_image_from_tensor(img, path, denormalize=True, mean, std)` → 逆正規化後にPNG保存。
- `save_heatmap_png(hm, path)` → ヒートマップを擬似カラーPNGとして保存。
- `save_overlay_from_tensor(img_t, hm, path, alpha=0.5, denormalize=True)` → 画像とヒートマップをブレンドして保存。

## 使い方（Trainer内での利用）
- `Trainer` は runtime 設定の `run_id` を Logger に渡し、`runtime.log_dir` 配下に `run_id` サブディレクトリを生成する。
- 各エポックの検証フェーズで最大4枚のサンプルをランダム抽出し、`inputs/epochXXX/` と `overlays/epochXXX/` に保存する。ImageNet正規化を使う場合は自動で逆正規化される。

## 補足
- 同一秒内に複数回実行した場合もサフィックス付きのディレクトリ（例: `run-20240101-120000-01`）を生成し、衝突を防ぐ。
- TensorBoard出力は `torch.utils.tensorboard` が利用可能な場合にのみ有効化される。
- 追加の可視化（例: ヒートマップ単体の保存、グレースケール保存）は用意されたユーティリティを組み合わせて実装する。
