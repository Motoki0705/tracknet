# ディレクトリ構成案

ViTバックボーン＋アップサンプリングデコーダによるヒートマップ出力モデル、およびその学習パイプラインを実装するためのプロジェクト構成案を示す。

## tracknet/

- `models/`
  - `backbones/vit_backbone.py`  
    `demo/vit_demo.py` を基にしたViTラッパー。`pretrained_model_name`など既存デモと同一の初期化パラメータを維持する。
  - `backbones/convnext_backbone.py`  
    ConvNeXtラッパー。`AutoBackbone` から各ステージの特徴（hidden states）を取得し、FPN用に公開。
  - `decoders/upsampling_decoder.py`  
    パッチトークンを2次元ヒートマップへアップサンプリングするデコーダ。
  - `decoders/fpn_decoder.py`  
    FPNライクなトップダウン経路＋ラテラル畳み込みでマルチスケール融合を行うデコーダ。
  - `heads/heatmap_head.py`  
    デコーダ出力を加工し、最終ヒートマップや信頼度マップを生成する層。
  - `__init__.py`  
    主要クラスの公開。
- `training/`
  - `trainer.py`  
    学習オーケストレーション。Loss計算、最適化、スケジューラ、ロギングを統括。
  - `losses/heatmap_loss.py`  
    ヒートマップ用損失（MSEやfocal等）と可視性マスク処理。
  - `metrics/`  
    座標誤差やヒートマップ評価のメトリクス群。
  - `callbacks/`  
    チェックポイント保存、学習率スケジューラ、早期終了など。
- `utils/`
  - `logging.py`  
    TensorBoard/Weights & Biases等のロギング用ユーティリティ。
  - `config.py`  
    OmegaConfで `configs/data/`, `configs/model/`, `configs/training/` のYAMLを読み込み、CLI引数や実験用オーバーライドを統合するコンフィグユーティリティ。
  - `geometry.py`  
    ヒートマップと画像座標の相互変換関数。
- `scripts/`
  - `train.py`  
    `uv run python -m tracknet.scripts.train` の入口。OmegaConf設定を注入して `training/trainer.py` を呼び出す。
  - `eval.py`  
    学習済みモデルで検証を実行。
  - `predict.py`  
    単発推論またはデモ用推論スクリプト。
- `__init__.py`

## configs/

- `data/`  
  データソース別設定（例: `tracknet.yaml`）。データルート、分割、前処理に関するパラメータを定義。
- `model/`  
  ViTバックボーンやデコーダ構成、ヒートマップ関連のハイパーパラメータ。
  - 例: `vit_heatmap.yaml`, `convnext_fpn_heatmap.yaml`
- `training/`  
  最適化設定、スケジューラ、ロギング、チェックポイントに関する設定。

各カテゴリのYAMLをOmegaConfで読み取り、エントリポイント `tracknet/scripts/train.py` から `utils/config.py` のヘルパーを通じてマージした統合設定（`cfg`）として利用する。

## tests/

- `test_datasets.py`  
  データローダ・前処理の単体テスト。
- `test_models.py`  
  モデル入出力形状や勾配計算を検証。
- `test_trainer.py`  
  学習オーケストレーションのスモークテスト。

## docs/

- `dataset.md`（既存）  
  データセット仕様。`data/tracknet/` を参照するよう更新。
- `model.md`（将来追加）  
  ViT＋デコーダモデルの設計指針、ハイパーパラメータの記録。

## data/

- `tracknet/`  
  実データ（JPEGフレーム＋`Label.csv`）の格納先。以前の `data/processed/` から移設。
- その他のデータセットを追加する場合は `data/<dataset名>/` を同階層に配置する。

## tracknet/datasets/

- `base/sequence_dataset.py`（仮称）  
  時系列画像列を返す抽象クラス。クリップ長やステップ幅、ヒートマップ生成用メタ情報を定義。
- `base/image_dataset.py`（仮称）  
  単体画像＋ラベルを返す抽象クラス。
- `tracknet_sequence.py` / `tracknet_frame.py`（仮称）  
  `data/tracknet/` を参照する具象データセット実装。
- `utils/augmentations.py`  
  データ拡張・正規化など、画像とヒートマップの整合を取る処理。
- `utils/collate.py`  
  バッチ整形、ヒートマップターゲット生成、可視性マスク付与。
- `__init__.py`  
  公開データセットクラスのエクスポート。

> 備考: `tracknet` パッケージ配下に `datasets/` を置き、同一名前空間でモデルやトレーナーから直接インポートできるようにする。必要に応じて `pyproject.toml` の `packages` 設定に `tracknet` を含める。

## demo/

- `vit_demo.py`  
  ViT利用例。バックボーン構築の参考として維持。
- `convnext_demo.py`  
  他バックボーン例。

## third_party/（既存）

- 外部コードの集約場所。ライセンス対応に注意。

## 追加メモ

- ログ・チェックポイント出力先は後日 `configs/data/tracknet.yaml` で指定し、Git管理対象外パスを `.gitignore` に登録する。
- 抽象クラスを `datasets/base/` に分けることで、将来的なデータソース拡張にも対応しやすい。
- OmegaConfにより設定の階層化や型チェックを行い、CLIオプションからの上書きも視野に入れる。
