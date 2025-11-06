# Project Context

## Purpose
テニスの画像列から各種コンポーネント（選手、ボール、コートなど）を検知するモデルを学習・推論することを目的とする。
高品質なデータと安定した学習パイプラインを提供し、モデル開発・検証・運用を効率化する。

主な目標:
- 高精度なオブジェクト検出モデルの開発
- 学習・評価・推論を再現性高く行える環境構築
- 推論パイプラインの軽量化と高速化

---

## Tech Stack
- **言語**: Python
- **フレームワーク**: PyTorch, PyTorch Lightning
- **設定管理**: OmegaConf
- **ロギング/可視化**: TensorBoard, tqdm
- **モデル関連**: HuggingFace Transformers, PEFT (Parameter-Efficient Fine-Tuning), bitsandbytes（量子化）
- **CI/CD**: GitHub Actions

---

## Project Conventions

### Code Style
- **Linter/Formatter**: Ruff
- **型チェック**: Mypy
- **命名規約**:
  - 変数・関数: `snake_case`
  - クラス: `PascalCase`
  - Configキー: `lower_case_with_underscores`
- **構成規約**:
  - `tracknet/` 以下に `data/`, `models/`, `training/`, `evaluation/`, `configs/`
  - モジュールごとに明確な責務分離（データ処理、モデル、トレーニング、評価）

---

### Architecture Patterns
- **パターン**: クリーンアーキテクチャを参考にしたモジュール構成
  - `domain`: データ型・モデル定義
  - `application`: トレーニングや推論フロー
  - `infrastructure`: データローディング・保存・ログ
- **特徴**:
  - OmegaConf による設定の明示的管理
  - LightningModule により学習ロジックを分離
  - ログ・メトリクスを TensorBoard で統合管理

---

### Testing Strategy
- **テストレイヤ**:
  - **Unit**: 関数/クラス単位のロジック検証（Pytest）
  - **Integration/Contract**: 学習・推論パイプライン、設定ロード、モデル保存など
  - **E2E**: データ → 学習 → 評価 → 推論の一連テスト（小規模データで再現確認）

- **カバレッジ目標**:
  - 全体: 75–85%
  - コアロジック: 90%（branch coverage 重視）

- **方針**:
  - 全体カバレッジを維持しつつ、特に推論・評価ロジックを重点的に検証
  - 異常系（設定不備、欠損データ、GPUリソース不足など）を必ず1件以上カバー

- **テストツール**:
  - Pytest + Coverage
  - CI上で自動実行（GitHub Actions）

---

### Git Workflow
- **ブランチモデル**: trunk-based（短命ブランチ）
  - 命名: `feat/`, `fix/`, `refactor/`, `test/`
- **コミット規約**: Conventional Commits
  - 例: `feat(model): add detection head`
- **レビュー運用**:
  - mainブランチはPR経由のみ
  - CIでpytest・lint・format自動チェック
- **CI/CD**: GitHub Actions による自動テスト実行と成果物アーティファクト保存

---

## External Dependencies
- **主要ライブラリ/API**:
  - PyTorch, Lightning, HuggingFace Transformers
  - bitsandbytes（量子化）, PEFT（微調整）
- **CIサービス**: GitHub Actions
- **外部ログ/監視**: TensorBoard（ローカル or Remote Logger）
