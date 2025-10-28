# 設計書（ViT＋アップサンプリングデコーダ TrackNet）

本設計書は、ViTバックボーンのパッチトークンをアップサンプリングデコーダでヒートマップ化し、そのピークをボール位置として推定する学習システムの全体像をまとめる。エントリポイントは `uv run python -m tracknet.scripts.train` を基本とする。

## 全体アーキテクチャ
- 入力: 単画像または時系列画像（クリップ）
- 前処理: HF `AutoImageProcessor` による正規化・リサイズ（ViTの前処理に一致）
- バックボーン: HF `AutoModel`（例: `facebook/dinov3-vitb16-pretrain-lvd1689m`）
  - 出力 `last_hidden_state` を CLS、register tokens、patch tokensに分離
  - パッチトークンを `[B, H_p, W_p, C]` にリシェイプ
- デコーダ: パッチトークンから段階的アップサンプル（ConvTranspose/Interpolation + Conv）
- ヘッド: 1chヒートマップ出力（活性化はロス側で制御）
- ターゲット: ガウス分布で生成したヒートマップ（visibility=0 は座標欠損としてマスク）
- 損失: MSE または Focal（マスク対応）
- メトリクス: ヒートマップから座標推定（argmax/soft-argmax）→ L2誤差、PCK@r

## 主要モジュール
- `tracknet/models/backbones/vit_backbone.py`
  - `AutoImageProcessor.from_pretrained(pretrained_model_name)`
  - `AutoModel.from_pretrained(pretrained_model_name)`
  - 前処理済みテンソルを受け、`last_hidden_state` から `[B, H_p, W_p, C]` を返すユーティリティを提供
- `tracknet/models/decoders/upsampling_decoder.py`
  - 入力 `[B, H_p, W_p, C]` を `[B, C_d, H, W]` に転置後、数段のアップサンプルで `H_out×W_out` に到達
  - スキップ接続は最小構成では省略。必要であればMLP/Convでチャネル調整
- `tracknet/models/heads/heatmap_head.py`
  - 最終的に1ch（もしくは時系列なら `T×1ch`）のヒートマップを出力
- `tracknet/datasets/base/sequence_dataset.py` / `base/image_dataset.py`
  - 入力取得インターフェースを定める（`__getitem__` が画像（列）とターゲット、マスク、メタを返却）
- `tracknet/datasets/tracknet_frame.py` / `tracknet/datasets/tracknet_sequence.py`
  - `data/tracknet/` の `game*/Clip*/{frame}.jpg` と `Label.csv` を読み込み
  - visibility=0 のフレームは座標欠損として扱い、ロスをマスク
- `tracknet/datasets/utils/augmentations.py`
  - 幾何（リサイズ、中心クロップ、ランダムフリップ等）と色変換
  - 幾何変換は座標とヒートマップに反映（解像度変換の整合を保証）
- `tracknet/datasets/utils/collate.py`
  - バッチ整形、ガウスヒートマップ生成、可視性マスク作成
- `tracknet/training/losses/heatmap_loss.py`
  - `mse(target, pred)` or `focal_heatmap_loss`（可視性マスクを乗算）
- `tracknet/training/metrics/`
  - 予測ヒートマップからピーク座標を復元し、L2誤差などを算出
- `tracknet/training/trainer.py`
  - cfgからデータ・モデル・最適化・スケジューラ・ロギングを初期化
  - 学習/検証ループ、チェックポイント保存、シード固定、AMP、勾配クリップ
- `tracknet/utils/config.py`
  - `configs/data/`, `configs/model/`, `configs/training/` を OmegaConf で読み込み、統合cfgを返す
  - CLI引数（`--data`, `--model`, `--training`）やキー単位のオーバーライド対応
- `tracknet/utils/geometry.py`
  - 画像座標↔ヒートマップ座標のスケーリング、ガウスカーネル生成ヘルパー
- `tracknet/utils/logging.py`
  - 標準出力、TensorBoard/W&B等のロガー（任意）

## 形状とスケーリングの要点
- 例: 入力画像 1280×720、ViT/16 → パッチグリッドは概ね 80×45（前処理サイズに依存）
- `AutoImageProcessor` の `size` により実際の入力解像度が決まるため、ターゲットヒートマップは cfg で `heatmap.size` を明示
- デコーダは `[B, H_p, W_p, C] -> [B, 1, H_out, W_out]` に変換
- ターゲットヒートマップ生成時は、元の座標（Label.csvのピクセル単位）を `H_out×W_out` にスケーリングしてガウスを描画

## コンフィグ設計（OmegaConf）
- `configs/data/tracknet.yaml`
  - `root: data/tracknet`
  - `split: { train_games: [...], val_games: [...] }` 等
  - 前処理・拡張: `resize`, `normalize`, `flip_prob`, `sequence: { length, stride }`
- `configs/model/vit_heatmap.yaml`
  - `pretrained_model_name: facebook/dinov3-vitb16-pretrain-lvd1689m`
  - `decoder: { channels: [C, 256, 128, 64], upsample: [2,2,2] }`
  - `heatmap: { size: [W,H], sigma: 2.0 }`
- `configs/training/default.yaml`
  - `optimizer: { name: adamw, lr: 5e-4, weight_decay: 0.05 }`
  - `scheduler: { name: cosine, warmup_epochs: 5 }`
  - `batch_size, epochs, amp, grad_clip, log_dir, ckpt_dir`

`uv run python -m tracknet.scripts.train --data tracknet --model vit_heatmap --training default` で起動。キー単位のオーバーライドは `uv run python -m tracknet.scripts.train training.optimizer.lr=1e-4` のように渡す想定。

## 学習・評価フロー
1. cfg構築（OmegaConf）
2. データセット/データローダ初期化（train/val）
3. モデル作成（ViT+Decoder+Head）
4. ロス/メトリクス初期化
5. ループ実行（学習→検証→ログ/保存）
6. 必要に応じてベストckptで評価/推論

## リスク・留意点
- 可視性ラベルの不均衡（`visibility=1` が多い）→ ロス重み/サンプリングで緩和
- 大きな入力解像度に伴うメモリ使用量 → 勾配チェックポイント/AMP/小バッチで対処
- ViTの前処理サイズとターゲット解像度の不一致 → cfgで明示的に管理
- 位置が欠損（`visibility=0`）の扱い → マスク必須、評価から除外

## 実行例（uv）
- 学習: `uv run python -m tracknet.scripts.train --data tracknet --model vit_heatmap --training default`
- ドライラン: `uv run python -m tracknet.scripts.train --dry-run`
- インポート確認: `uv run python -c "import tracknet"`

以上。
