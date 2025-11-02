<!-- OPENSPEC:START -->
# OpenSpec Instructions

These instructions are for AI assistants working in this project.

Always open `@/openspec/AGENTS.md` when the request:
- Mentions planning or proposals (words like proposal, spec, change, plan)
- Introduces new capabilities, breaking changes, architecture shifts, or big performance/security work
- Sounds ambiguous and you need the authoritative spec before coding

Use `@/openspec/AGENTS.md` to learn:
- How to create and apply change proposals
- Spec format and conventions
- Project structure and guidelines

Keep this managed block so 'openspec update' can refresh the instructions.

<!-- OPENSPEC:END -->

## プロジェクトの概要。
テニスのボールを検知するモデルをトレーニング、利用することを目標にする。

## ディレクトリ構成
- demo/
  - モデルのロード、使用法の具体例。
- docs/dataset.md
  - データセットの内容(data/tracknet/がデータセットの実態。)
- spec/
  - 実装にあたっての要件やチケット、目指すディレクトリ構成

## 実行環境
仮想環境.venvを用いている。
実行は`uv run python`で行う。例えば、`uv run python demo/vit_demo.py`, `uv run pytest`など。
モジュールが未導入である等のエラーに対しては`uv add`で導入。例えば、`uv add pytest`など。

## コーディングスタイル。
必ずdocstringを書く。（google style）
モジュラーな構想を意識。
可読性の高さを優先。


