# 実装チケット（チェックポイント）

以下は、ViTバックボーン＋アップサンプリングデコーダによるヒートマップ出力モデルと学習システムを構築するための実装タスク群です。完了時に各チェックをオンにしてください。

## 0. 初期セットアップ
- [x] `tracknet/` パッケージのスケルトン作成（`__init__.py` 含む）
- [x] `tracknet/models/` `training/` `utils/` `scripts/` `datasets/` のディレクトリ作成
- [x] `configs/` 配下に `data/` `model/` `training/` ディレクトリ作成
- [x] 既存 `demo/vit_demo.py` の参照確認（`pretrained_model_name` を共有）

完了条件:
- [x] 主要ディレクトリが作成され、インポート可能（`uv run python -c "import tracknet"`）

## 1. コンフィグ読み込み（OmegaConf）
- [x] `tracknet/utils/config.py` 実装
  - [x] `configs/data/<name>.yaml` `configs/model/<name>.yaml` `configs/training/<name>.yaml` を読み込み、マージして `cfg` を返す
  - [x] CLI引数（例: `--data tracknet --model vit_heatmap --training default`）で指定可
  - [x] ランダムシード、出力ディレクトリの初期化
- [x] サンプル設定ファイル追加
  - [x] `configs/data/tracknet.yaml`（データルート、分割、前処理）
  - [x] `configs/model/vit_heatmap.yaml`（ViT設定、デコーダ、ヒートマップ設定）
  - [x] `configs/training/default.yaml`（最適化、スケジューラ、ロギング、チェックポイント）

完了条件:
- [x] `uv run python -m tracknet.scripts.train --dry-run` 相当で `cfg` の構築ログが出力される
- [x] 使用をdocs/config/に作成。

## 2. データセット層（tracknet/datasets）
- [x] 抽象ベース実装
  - [x] `tracknet/datasets/base/sequence_dataset.py`（時系列サンプル用）
  - [x] `tracknet/datasets/base/image_dataset.py`（単画像サンプル用）
- [x] ユーティリティ
  - [x] `tracknet/datasets/utils/augmentations.py`（幾何・色変換。座標との整合維持）
  - [x] `tracknet/datasets/utils/collate.py`（バッチ整形、ヒートマップ生成、可視性マスク）
  - [x] ヒートマップ生成はガウス分布（σや出力解像度はコンフィグ）
- [x] TrackNet具象実装
  - [x] `tracknet/datasets/tracknet_frame.py`（単画像＋ヒートマップ）
  - [x] `tracknet/datasets/tracknet_sequence.py`（時系列ウィンドウ＋時系列ヒートマップ）
  - [x] `data/tracknet/` 配下の `game*/Clip*/<frame>.jpg` と `Label.csv` をパース
  - [x] `visibility=0` など座標欠損時はロスをマスク

完了条件:
- [x] DataLoaderで1バッチ取得し、画像テンソル・ヒートマップ・マスクの形状が想定通り
- [x] サブセット読み込み・分割（train/val）が機能
- [x] 仕様をdocs/datasets/に作成。

## 3. モデル（ViT＋アップサンプリングデコーダ）
- [x] バックボーン
  - [x] `tracknet/models/backbones/vit_backbone.py`（HF Transformersの `AutoImageProcessor` + `AutoModel`）
  - [x] `pretrained_model_name` は `demo/vit_demo.py` と同一指定が可能
  - [x] 出力: パッチトークンを `[B, H_p, W_p, C]` に整形して返す
- [x] デコーダ
  - [x] `tracknet/models/decoders/upsampling_decoder.py`（ConvTranspose/Interpolation + Convで段階的アップサンプル）
  - [x] 可変のヒートマップ解像度（例: 128×72）に対応
- [x] ヘッド
  - [x] `tracknet/models/heads/heatmap_head.py`（1chヒートマップ出力。活性化は後段のロスで対応）
- [x] `tracknet/models/__init__.py` 整備

完了条件:
- [x] ダミー入力からヒートマップ出力まで前向き計算が通る（形状チェック）
- [x] 仕様をdocs/models/に作成。

### 3.1 ConvNeXt＋FPN モデル（差し込み）
- [x] バックボーン
  - [x] `tracknet/models/backbones/convnext_backbone.py`（HF `AutoBackbone`／`AutoModel` で hidden_states 取得）
  - [x] `pretrained_model_name` は `demo/convnext_demo.py` と同一指定が可能
  - [x] 出力: マルチスケール特徴列 `[C3, C4, C5(, C2)]` を返す（各 `[B, C_i, H_i, W_i]`）
- [x] デコーダ（FPN）
  - [x] `tracknet/models/decoders/fpn_decoder.py`（1x1ラテラル→トップダウンup→3x3精錬→融合）
  - [x] ヒートマップ解像度（例: 128×72）へ各段をリサイズし `sum` or `concat+1x1` で融合可能
  - [x] 既存 `HeatmapHead` を利用して1chヒートマップ出力
- [x] コンフィグ
  - [x] `configs/model/convnext_fpn_heatmap.yaml`（`pretrained_model_name`, `fpn.lateral_dim`, `fpn.use_p2`, `fuse` 等）
- [x] `tracknet/models/__init__.py` 整備（公開シンボル追加）

完了条件:
- [x] ダミー入力からヒートマップ出力まで前向き計算が通る（形状チェック）
- [x] ViT/ConvNeXt をコンフィグで切替可能（`--model vit_heatmap` / `--model convnext_fpn_heatmap`）
- [x] 仕様をdocs/models/に追記。

## 4. 損失・メトリクス・コールバック
- [x] `tracknet/training/losses/heatmap_loss.py`（MSE・Focalの選択、マスク対応）
- [x] `tracknet/training/metrics/`（argmax/soft-argmaxで座標推定、L2誤差、PCK@r）
- [x] `tracknet/training/callbacks/`（モデル選択・早期終了・LRスケジューラ連携）

完了条件:
- [x] 単体テストでロス・メトリクスの入出力が妥当（ローカルスモーク確認）
- [x] 仕様をdocs/training/に作成。


## 5. トレーナ（オーケストレーション）
- [x] `tracknet/training/trainer.py`
  - [x] コンフィグからデータセット・モデル・最適化器・スケジューラを初期化
  - [x] AMP（任意）、マルチGPU/単GPU/CPU対応（本実装は単GPU/CPUの最小構成）
  - [x] 学習・検証ループ、ロギング、チェックポイント保存
  - [x] 再現性（seed固定）、勾配クリップ、混合精度のON/OFF

完了条件:
- [x] 小規模サンプルで1〜2エポックのスモークテストが通る
- [x] 仕様をdocs/training/に作成。

## 6. スクリプト
- [x] `tracknet/scripts/train.py`（エントリポイント）
  - [x] OmegaConfでcfgを構築し、`trainer.train(cfg)` を起動
  - [x] `--data` `--model` `--training` と一部オーバーライドのCLIを受け付け
- [x] `tracknet/scripts/eval.py`（任意）
- [x] `tracknet/scripts/predict.py`（任意。単画像/動画での推論デモ）

完了条件:
- [x] `uv run python -m tracknet.scripts.train --data tracknet --model vit_heatmap --training default` が起動
- [x] 使用をdocs/scripts/に作成。

## 7. ログ・可視化
- [ ] `tracknet/utils/logging.py`（標準出力ログ＋TensorBoard/W&Bは任意）
- [ ] 検証時にヒートマップと元画像のオーバーレイをサンプル保存

完了条件:
- [ ] エポック毎に主要メトリクスが記録され、可視化が確認できる

## 8. テスト（最小限）
- [ ] `tests/test_datasets.py`（単バッチ取得、形状検証、マスクロジック）
- [ ] `tests/test_models.py`（前向き形状、勾配の有無）
- [ ] `tests/test_trainer.py`（短いスモーク学習）

完了条件:
- [ ] 主要テストがローカルで通過（CIは任意）
- [ ] 仕様をdocs/tests/に作成。

---

メモ:
- 画像サイズとパッチ解像度の関係（例: 1280×720 + ViT/16 → 80×45 パッチ）を念頭に、デコーダ側でターゲットヒートマップ解像度に揃える。
- `pretrained_model_name` は `demo/vit_demo.py` と同名（例: `facebook/dinov3-vitb16-pretrain-lvd1689m`）を既定値にし、コンフィグで変更可能にする。
- `visibility=0` は座標欠損のためロスからマスク。クラス不均衡（visibility/status）への重み付けは拡張タスクとする。
