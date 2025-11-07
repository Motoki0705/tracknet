# 貢献ガイドライン

このドキュメントではTrackNetプロジェクトへの貢献方法について説明します。

## 開始方法

### 開発環境のセットアップ

1. フォークしてリポジトリをクローン:
```bash
git clone <your-forked-repo>
cd tracknet
```

2. 開発依存関係をインストール:
```bash
uv sync --dev
```

3. Pre-commit hooksをインストール:
```bash
uv run pre-commit install
```

## 開発ワークフロー

### 1. ブランチ戦略

- `main`: 安定版リリース用
- `develop`: 開発版統合用
- `feat/機能名`: 新機能開発用
- `fix/修正名`: バグ修正用
- `refactor/対象名`: リファクタリング用

### 2. コーディング規約

#### 命名規約

```python
# 変数・関数: snake_case
def process_data(input_data):
    max_iterations = 100
    return processed_data

# クラス: PascalCase
class DataProcessor:
    def __init__(self):
        self.config_manager = ConfigManager()

# 定数: UPPER_SNAKE_CASE
MAX_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 0.001

# Configキー: lower_case_with_underscores
config = {
    "model_name": "vit_base",
    "batch_size": 16,
    "learning_rate": 0.001
}
```

#### ドキュメント

Googleスタイルのdocstringを使用:

```python
def calculate_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """損失を計算します。

    Args:
        predictions: モデルの予測値 [batch_size, num_classes]
        targets: 正解ラベル [batch_size]

    Returns:
        計算された損失値

    Raises:
        ValueError: 入力テンソルの形状が一致しない場合
    """
    if predictions.shape[0] != targets.shape[0]:
        raise ValueError("Batch sizes must match")

    return torch.nn.functional.cross_entropy(predictions, targets)
```

#### 型ヒント

積極的に型ヒントを使用:

```python
from typing import List, Dict, Optional, Tuple
import torch

def train_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    epochs: int = 10,
    device: Optional[torch.device] = None
) -> Dict[str, List[float]]:
    """モデルをトレーニングします。"""
    # 実装...
    return {"loss": [], "accuracy": []}
```

### 3. コード品質チェック

#### 必須チェック

コミット前には必ず以下を実行:

```bash
# Lintingと自動修正
uv run ruff check . --fix

# フォーマットting
uv run ruff format .

# 型チェック
uv run mypy tracknet

# テスト
uv run pytest
```

#### 一括実行

```bash
# 全ての品質チェックを一度に実行
uv run ruff check . --fix && uv run ruff format . && uv run mypy tracknet && uv run pytest
```

### 4. テスト

#### テストの書き方

```python
import pytest
import torch
from tracknet.models import create_model

def test_model_creation():
    """モデル作成のテスト。"""
    model = create_model("vit_base")
    assert isinstance(model, torch.nn.Module)

    # ダミー入力でforward passをテスト
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    assert output.shape == (1, num_classes)

def test_model_with_invalid_config():
    """無効な設定でのエラーをテスト。"""
    with pytest.raises(ValueError):
        create_model("invalid_model_name")
```

#### テストカバレッジ

- 新機能には必ずテストを追加
- カバレッジ目標: 75-85%
- クリティカルなロジック: 90%以上

### 5. コミットメッセージ

Conventional Commits規約に従います:

```
feat(model): add new attention mechanism
fix(training): resolve gradient explosion issue
docs(readme): update installation instructions
style(lint): fix ruff violations
refactor(config): simplify configuration loading
test(models): add unit tests for ViT model
```

### 6. プルリクエスト

#### PRの作成手順

1. ブランチを最新のdevelopに同期:
```bash
git checkout develop
git pull upstream develop
git checkout feat/your-feature
git rebase develop
```

2. プルリクエストを作成:
   - 明確なタイトルと説明
   - 関連するIssueを参照
   - 変更内容の要約

#### PRテンプレート

```markdown
## 変更内容
- 機能の概要
- 主要な変更点

## 関連Issue
closes #123

## テスト
- [x] 単体テストを追加
- [x] 統合テストを実行
- [x] 手動テストを実行

## チェックリスト
- [x] コードがプロジェクト規約に従っている
- [x] テストが通過する
- [x] ドキュメントを更新
- [x] CI/CDが通過する
```

## コードレビュー

### レビュアーの責任

- コードの論理的正しさを確認
- プロジェクト規約への準拠をチェック
- パフォーマンスとセキュリティを評価
- 建設的なフィードバックを提供

### レビュー受領者の責任

- フィードバックに適切に対応
- 議論をオープンかつ建設的に進行
- 必要に応じて実装を修正

## 開発ツール詳細

### Ruff

高速なPython linter兼formatter:

```bash
# 問題のある箇所を特定
uv run ruff check path/to/file.py

# 自動修正可能な問題を修正
uv run ruff check path/to/file.py --fix

# フォーマットを適用
uv run ruff format path/to/file.py

# フォーマットをチェックのみ
uv run ruff format --check path/to/file.py

# 特定のルールを無視
uv run ruff check path/to/file.py --ignore=E501,F403

# 特定のファイルのみ
uv run ruff check tracknet/models/
uv run ruff format tracknet/models/
```

### MyPy

静的型チェッカー:

```bash
# 基本的な型チェック
uv run mypy tracknet

# 厳格モードでチェック
uv run mypy tracknet --strict

# 特定のファイルのみ
uv run mypy tracknet/models/vit.py
```

## ヘルプとサポート

- Issueを作成して質問や問題を報告
- Slack/Discordでディスカッション
- メンテナーに直接連絡

ありがとうございます！皆様の貢献を心より歓迎します。
