# TrackNet

テニスの画像列から各種コンポーネント（選手、ボール、コートなど）を検知するモデルを学習・推論するためのプロジェクトです。高品質なデータと安定した学習パイプラインを提供し、モデル開発・検証・運用を効率化します。

## 主な目標

- 高精度なオブジェクト検出モデルの開発
- 学習・評価・推論を再現性高く行える環境構築
- 推論パイプラインの軽量化と高速化
- **メモリ効率の良いファインチューニング**（LoRA・量子化対応）

## 技術スタック

- **言語**: Python 3.11+
- **フレームワーク**: PyTorch, PyTorch Lightning
- **設定管理**: OmegaConf
- **ロギング/可視化**: TensorBoard, tqdm
- **モデル関連**: HuggingFace Transformers, PEFT, bitsandbytes
- **最適化**: LoRA (Low-Rank Adaptation), INT4/FP4量子化
- **CI/CD**: GitHub Actions

## 開発環境セットアップ

### 前提条件

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (依存関係管理ツール)

### インストール手順

1. リポジトリをクローンします:
```bash
git clone <repository-url>
cd tracknet
```

2. 依存関係をインストールします:
```bash
# 本番用依存関係
uv sync

# 開発用依存関係（コードスタイルツール、テストツールなど）
uv sync --dev
```

3. 開発環境を有効化します:
```bash
source .venv/bin/activate
```

### 開発ツール

このプロジェクトでは以下のコード品質管理ツールを使用しています:

#### Linting & Formatting

- **Ruff**: 高速なPython linterとformatter

使用方法:
```bash
# Linting (問題の検出)
uv run ruff check .

# Linting (問題の自動修正)
uv run ruff check . --fix

# Formatting
uv run ruff format .
```

#### Type Checking

- **Mypy**: 静的型チェック

使用方法:
```bash
# 型チェックの実行
uv run mypy tracknet
```

#### Pre-commit Hooks

ローカル開発でコミット前に自動的に品質チェックを実行するには:

```bash
# Pre-commit hooksのインストール
uv run pre-commit install

# 手動で全てのhooksを実行
uv run pre-commit run --all-files
```

### テスト

このプロジェクトでは3層のテスト戦略を採用しています:

- **Unit Tests**: 個別の関数やクラスをテスト (tests/unit/)
- **Integration Tests**: 複数のコンポーネント連携をテスト (tests/integration/)  
- **E2E Tests**: 完全なワークフローをテスト (tests/e2e/)

```bash
# 全てのテストを実行
uv run pytest

# カバレッジレポート付きでテストを実行
uv run pytest --cov=tracknet --cov-report=html

# 特定のテスト層を実行
uv run pytest tests/unit/ -v          # Unit tests only
uv run pytest tests/integration/ -v   # Integration tests only
uv run pytest tests/e2e/ -v           # E2E tests only

# マーカー付きテストを実行
uv run pytest -m unit                  # Unit tests only
uv run pytest -m integration           # Integration tests only
uv run pytest -m e2e                   # E2E tests only
uv run pytest -m "not slow"            # Skip slow tests
```

#### カバレッジ目標

- 全体カバレッジ: 75%以上
- コアモジュール: 85-90%以上
  - tracknet.utils: 90%以上
  - tracknet.models: 85%以上  
  - tracknet.datasets: 85%以上

カバレッジレポートは `htmlcov/index.html` で確認できます。

## プロジェクト構成

```
tracknet/
├── tracknet/                 # メインパッケージ
│   ├── datasets/            # データセット関連
│   ├── models/              # モデル定義
│   ├── training/            # トレーニングロジック
│   ├── tools/               # ツールとユーティリティ
│   ├── utils/               # 汎用ユーティリティ
│   └── configs/             # 設定ファイル
├── demo/                    # デモスクリプト
├── docs/                    # ドキュメント
├── tests/                   # テストコード
│   ├── unit/                # 単体テスト
│   ├── integration/         # 統合テスト
│   ├── e2e/                 # エンドツーエンドテスト
│   ├── tools/               # ツール関連テスト
│   ├── conftest.py          # pytest fixtures
│   └── utils.py             # テストユーティリティ
├── configs/                 # 設定ファイル
├── .github/workflows/       # CI/CDワークフロー
└── openspec/               # 仕様管理
```

## コーディング規約

- **命名規約**:
  - 変数・関数: `snake_case`
  - クラス: `PascalCase`
  - Configキー: `lower_case_with_underscores`

- **ドキュメント**: 全ての関数とクラスにはGoogleスタイルのdocstringが必要

- **型ヒント**: 新しいコードには型ヒントを追加することが推奨されます

## CI/CD

GitHub Actionsを使用して以下の自動チェックを実行しています:

- Linting (Ruff)
- Formatting (Ruff Format)
- Type Checking (Mypy)
- Tests (Pytest) with Coverage Reporting
- Coverage Upload to Codecov

これらのチェックはプルリクエスト作成時とmain/developブランチへのプッシュ時に自動的に実行されます。

カバレッジが75%未満の場合、CIは失敗します。コアモジュールにはより高いカバレッジ要件が適用されます。

## LoRAと量子化

TrackNetでは、大規模な事前学習済みモデルを効率的にファインチューニングするための**LoRA (Low-Rank Adaptation)** と**量子化**機能をサポートしています。

### 特徴

- **メモリ効率**: INT4量子化によりメモリ使用量を約75%削減
- **高速なファインチューニング**: LoRAにより学習パラメータを大幅に削減
- **設定駆動**: YAML設定ファイルで簡単に有効化・無効化
- **下位互換性**: 既存の設定はそのまま動作

### 使用例

#### LoRAのみを使用する場合

```yaml
# configs/model/vit_lora_heatmap.yaml
model_name: "vit_heatmap"
pretrained_model_name: facebook/dinov3-vits16-pretrain-lvd1689m

backbone:
  freeze: false  # LoRAで効率的に学習
  device_map: auto
  local_files_only: true
  patch_size: 16

decoder:
  channels: [384, 256, 128, 64]
  upsample: [2, 2, 2]
  # ... その他設定

heatmap:
  size: [256, 144]
  sigma: 2.0

# LoRA設定
lora:
  enabled: true
  r: 16                    # ランク（低いほどパラメータが少ない）
  lora_alpha: 32           # スケーリング係数
  lora_dropout: 0.05       # ドロップアウト率
  target_modules: ["query", "key", "value", "dense"]  # 対象モジュール（自動検出も可能）
  bias: "none"
  task_type: "FEATURE_EXTRACTION"

quantization:
  enabled: false  # 量子化は無効化
```

#### 量子化 + LoRA (QLoRA) を使用する場合

```yaml
# configs/model/vit_qlora_heatmap.yaml
model_name: "vit_heatmap"
pretrained_model_name: facebook/dinov3-vits16-pretrain-lvd1689m

# ... backbone, decoder, heatmap設定は同じ ...

# LoRA設定
lora:
  enabled: true
  r: 16
  lora_alpha: 32
  lora_dropout: 0.05
  target_modules: ["query", "key", "value", "dense"]
  bias: "none"
  task_type: "FEATURE_EXTRACTION"

# 量子化設定
quantization:
  enabled: true
  quant_type: "nf4"           # "nf4" または "fp4"
  compute_dtype: "bfloat16"   # 計算精度
  skip_modules: []             # 量子化をスキップするモジュール
  mode: "manual"               # "manual" または "hf"
  compress_statistics: true
  use_double_quant: true
```

### 学習の実行

```bash
# LoRAモデルで学習
uv run python train.py --config configs/model/vit_lora_heatmap.yaml

# QLoRAモデルで学習
uv run python train.py --config configs/model/vit_qlora_heatmap.yaml
```

### パフォーマンス比較

| モデル | メモリ使用量 | 学習パラメータ数 | 推論速度 |
|--------|-------------|------------------|----------|
| 通常    | 100%        | 100%             | 基準     |
| LoRA   | ~90%        | ~5%              | 基準     |
| QLoRA  | ~25%        | ~5%              | 基準     |

### 詳細設定

#### LoRAパラメータ

- `r`: LoRAのランク（8, 16, 32, 64など）
- `lora_alpha`: スケーリング係数（通常rの2倍）
- `lora_dropout`: ドロップアウト率（0.0-0.1）
- `target_modules`: LoRAを適用するモジュールリスト（Noneで自動検出）

#### 量子化パラメータ

- `quant_type`: 量子化タイプ（"nf4"推奨、"fp4"も利用可能）
- `compute_dtype`: 計算精度（"bfloat16"推奨、"float16"も利用可能）
- `mode`: 量子化モード（"manual"推奨、"hf"はHuggingFace依存）

## 貢献方法

1. 機能用のブランチを作成: `git checkout -b feat/your-feature-name`
2. 変更を開発: コーディング規約を守り、テストを追加
3. 品質チェック: `uv run ruff check . && uv run ruff format . && uv run mypy tracknet && uv run pytest`
4. プルリクエストを作成

## ライセンス

[ライセンス情報をここに追加]