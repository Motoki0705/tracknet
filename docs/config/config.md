# コンフィグ仕様（OmegaConf）

本仕様は、TrackNetにおける設定管理の要件と構造を定義する。`OmegaConf` を用いて `configs/` 直下のYAMLを階層化・統合し、CLIからの上書き（ドットリスト）に対応する。

## 目的と範囲
- `configs/data/`, `configs/model/`, `configs/training/` のYAMLを読み込み、統合cfg（`cfg`）を構築する。
- ランタイム情報（シード、出力先など）を `cfg.runtime` に集約する。
- 入口は `uv run python -m tracknet.scripts.train`。Section 1ではドライラン（cfg構築・表示）を提供する。

## 主要API
- `tracknet.utils.config.build_cfg(...)`
  - 引数: `data_name`, `model_name`, `training_name`, `overrides`, `seed`, `output_dir`, `dry_run`
  - 戻り値: `DictConfig`（`data`, `model`, `training`, `runtime` を含む）
  - 機能:
    - 各カテゴリのYAMLを読み込み、`OmegaConf.create`/`merge`で統合
    - ドットリストによる上書き（例:`training.optimizer.lr=1e-4`）
    - シード初期化（`random`, `PYTHONHASHSEED`, `torch`）
    - 出力系ディレクトリ（`output_root`, `log_dir`, `ckpt_dir`）の解決・作成（`dry_run=True` 時は未作成）
- `tracknet.utils.config.add_config_cli_arguments(parser)`
  - `--data`, `--model`, `--training`, `--seed`, `--output-dir`, `--dry-run` を追加

## CLI仕様
- 基本起動: `uv run python -m tracknet.scripts.train`
- ドライラン: `--dry-run` を付与（cfg構築と表示のみ）
- カテゴリ指定: `--data <name> --model <name> --training <name>`
- オーバーライド: 解析されなかった引数はドットリストとして `build_cfg(overrides=...)` に渡す
  - 例: `uv run python -m tracknet.scripts.train training.optimizer.lr=1e-4 --dry-run`

## YAMLスキーマ
- `configs/data/<name>.yaml`
  - `root: str` データルート
  - `split: { train_games: list[str], val_games: list[str] }`
  - `preprocess: { resize: int|null, normalize: bool, flip_prob: float }`
  - `sequence: { enabled: bool, length: int, stride: int }`
- `configs/model/<name>.yaml`
  - `pretrained_model_name: str` HFモデル識別子
  - `decoder: { channels: list[int], upsample: list[int] }`
  - `heatmap: { size: [int, int], sigma: float }`
- `configs/training/<name>.yaml`
  - `seed: int`
  - `batch_size: int`, `epochs: int`, `amp: bool`, `grad_clip: float`
  - `optimizer: { name: str, lr: float, weight_decay: float }`
  - `scheduler: { name: str, warmup_epochs: int }`
  - `output_dir: str`, `log_dir: str`, `ckpt_dir: str`

## 生成されるランタイム情報（cfg.runtime）
- `seed: int` 最終決定シード（優先度: CLI `--seed` > `training.seed` > 既定42）
- `project_root: str` リポジトリルート
- `output_root: str` 出力ルート（`training.output_dir` もしくはCLI指定）
- `log_dir: str`, `ckpt_dir: str` ログ・チェックポイント出力先
- `timestamp: str` `YYYYMMDD-HHMMSS`
- `run_id: str` `run-<timestamp>-s<seed>`

## エラーハンドリング
- 存在しないYAML: `FileNotFoundError` を送出し失敗させる
- ディレクトリ作成: `dry_run=True` の場合は作成せずにスキップ

## 実行例
- 既定構成（ドライラン）
  - `uv run python -m tracknet.scripts.train --dry-run`
- カテゴリ指定
  - `uv run python -m tracknet.scripts.train --data tracknet --model vit_heatmap --training default --dry-run`
- ハイパーパラメータ上書き
  - `uv run python -m tracknet.scripts.train training.optimizer.lr=1e-4 --dry-run`

## 拡張方針
- 新たなデータ・モデル・トレーニング設定は、`configs/<category>/<name>.yaml` を追加し、CLIの `--<category> <name>` で切替
- 型バリデーションが必要になった場合は `OmegaConf.structured` や `pydantic` の導入を検討

以上。
