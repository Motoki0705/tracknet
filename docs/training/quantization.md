# 量子化学習 (Quantization Training)

TrackNetでは、メモリ使用量を削減し、より大きなモデルやバッチサイズで学習するためにint8量子化とQLoRAをサポートしています。

## 概要

- **int8量子化**: 8ビット量子化を使用してモデルのメモリフットプリントを約50%削減
- **QLoRA**: 4ビット量子化とLoRAアダプターを組み合わせ、効率的なファインチューニングを実現
- **自動マイクロバッチング**: OOM（Out of Memory）を自動的に検出し、バッチサイズを動的に調整

## 依存関係

量子化学習には以下のライブラリが必要です：

```bash
# pyproject.tomlに追加済み
bitsandbytes>=0.48.0
peft>=0.17.0
accelerate>=1.11.0
```

## int8量子化学習

### 特徴

- モデルの重みを8ビットに量子化
- メモリ使用量を約50%削減
- 推論時の高速化
- ほとんどのモデルで精度の低下が最小限

### 設定ファイル

`configs/training/int8.yaml`:

```yaml
# int8量子化設定
quantization:
  enabled: true
  load_in_8bit: true
  llm_int8_threshold: 6.0
  llm_int8_has_fp16_weight: false
  llm_int8_skip_modules: []  # 量子化をスキップするモジュール

# 基本学習設定
batch_size: 16  # 通常より小さめに設定
epochs: 20
precision: fp16
```

### 実行方法

```bash
# int8量子化で学習
uv run python -m tracknet.scripts.train --data tracknet --model vit_heatmap --training int8

# CLIでパラメータを上書き
uv run python -m tracknet.scripts.train --data tracknet --model vit_heatmap --training int8 training.batch_size=8
```

### 設定パラメータ

| パラメータ | 説明 | デフォルト値 |
|-----------|------|-------------|
| `enabled` | 量子化を有効にする | `true` |
| `load_in_8bit` | 8ビット量子化を有効にする | `true` |
| `llm_int8_threshold` | 量子化のしきい値 | `6.0` |
| `llm_int8_has_fp16_weight` | FP16重みを保持するか | `false` |
| `llm_int8_skip_modules` | 量子化をスキップするモジュールリスト | `[]` |

## QLoRA学習

### 特徴

- 4ビット量子化とLoRAアダプターの組み合わせ
- パラメータ効率の良いファインチューニング（PEFT）
- 最小限の学習可能パラメータで高い精度
- GPUメモリが限定的な環境での大規模モデル学習に適

### 設定ファイル

`configs/training/qlora.yaml`:

```yaml
# 4ビット量子化設定
quantization:
  enabled: true
  load_in_4bit: true
  bnb_4bit_compute_dtype: bf16
  bnb_4bit_use_double_quant: true
  bnb_4bit_quant_type: "nf4"

# LoRA設定
lora:
  enabled: true
  r: 16  # LoRAランク
  lora_alpha: 32  # スケーリングパラメータ
  target_modules: []  # 自動検出
  lora_dropout: 0.1
  bias: "none"
  task_type: "FEATURE_EXTRACTION"
```

### 実行方法

```bash
# QLoRAで学習
uv run python -m tracknet.scripts.train --data tracknet --model vit_heatmap --training qlora

# LoRAパラメータをカスタマイズ
uv run python -m tracknet.scripts.train --data tracknet --model vit_heatmap --training qlora training.lora.r=32 training.lora.lora_alpha=64
```

### LoRA設定パラメータ

| パラメータ | 説明 | デフォルト値 | 推奨範囲 |
|-----------|------|-------------|----------|
| `r` | LoRAランク（低いほどパラメータが少ない） | `16` | `8-64` |
| `lora_alpha` | スケーリングパラメータ | `32` | `16-128` |
| `lora_dropout` | LoRAドロップアウト率 | `0.1` | `0.0-0.3` |
| `target_modules` | LoRAを適用するモジュール | `[]`（自動） | `["query", "value"]`など |

## メモリ最適化

### 勾配チェックポイントング

```yaml
memory_optimization:
  gradient_checkpointing: true  # メモリ使用量を削減 but 計算コスト増加
```

### CPUオフロード

```yaml
memory_optimization:
  use_cpu_offload: true  # GPUメモリ不足時にCPUを使用
```

### ディスクオフロード

```yaml
memory_optimization:
  use_disk_offload: true  # 最終手段としてディスクを使用（非常に低速）
```

## 自動マイクロバッチング

量子化学習と組み合わせて使用すると効果的です：

```yaml
adaptive_micro_batch: true
min_micro_batch_size: 1  # QLoRAの場合は1に設定
mb_backoff_factor: 2
oom_retries: 5  # QLoRAの場合はリトライ回数を増やす
```

## パフォーマンス比較

| 手法 | メモリ使用量 | 学習速度 | 精度 | 推奨用途 |
|------|-------------|----------|------|----------|
| 通常（FP16） | 100% | 基準 | 100% | 十分なGPUメモリがある場合 |
| int8量子化 | ~50% | 1.2x | 98-99% | メモリ削減が優先の場合 |
| QLoRA | ~25% | 0.8x | 97-99% | 限定的なGPUメモリ環境 |

## モニタリング

学習中のメモリ使用量はTensorBoardで監視できます：

- `memory/allocated_gb`: 現在の確保メモリ量
- `memory/cached_gb`: キャッシュされたメモリ量
- `train/loss`: 学習損失

```bash
# TensorBoardで監視
tensorboard --logdir=outputs/logs
```

## ベストプラクティス

### int8量子化の場合

1. **バッチサイズ**: 通常より小さく設定（16→8）
2. **学習率**: 少し低めに設定（5e-4→3e-4）
3. **ウォームアップ**: 短めに設定（3→2エポック）

### QLoRAの場合

1. **バッチサイズ**: 小さく設定（8→4）
2. **学習率**: 低く設定（5e-4→1e-4）
3. **エポック数**: 多めに設定（20→30）
4. **ウォームアップ**: 長めに設定（3→5エポック）

### 共通

1. **混合精度**: `fp16`または`bf16`を使用
2. **勾配チェックポイントング**: メモリが厳しい場合は有効化
3. **マイクロバッチング**: OOM対策として必ず有効化

## トラブルシューティング

### OOMエラーが頻発する場合

```yaml
# 対策1: バッチサイズを削減
batch_size: 4

# 対策2: マイクロバッチ設定を調整
adaptive_micro_batch: true
min_micro_batch_size: 1
oom_retries: 5

# 対策3: メモリ最適化を有効化
memory_optimization:
  gradient_checkpointing: true
  use_cpu_offload: true
```

### 精度が低下する場合

```yaml
# 対策1: 量子化のしきい値を調整
quantization:
  llm_int8_threshold: 4.0  # より保守的に

# 対策2: LoRAランクを増やす
lora:
  r: 32
  lora_alpha: 64

# 対策3: 学習率を調整
optimizer:
  lr: 2.0e-4
```

### 学習が遅い場合

```yaml
# 対策1: メモリ最適化を無効化
memory_optimization:
  gradient_checkpointing: false
  use_cpu_offload: false

# 対策2: バッチサイズを増やす
batch_size: 32

# 対策3: 通常学習に切り替える
# training: default
```

## サンプルコマンド

```bash
# int8量子化でViTモデルを学習
uv run python -m tracknet.scripts.train \
  --data tracknet \
  --model vit_heatmap \
  --training int8 \
  training.batch_size=8 \
  training.optimizer.lr=3e-4

# QLoRAでConvNeXtモデルを学習
uv run python -m tracknet.scripts.train \
  --data tracknet \
  --model convnext_fpn_heatmap \
  --training qlora \
  training.lora.r=32 \
  training.lora.lora_alpha=64 \
  training.batch_size=4

# メモリ最適化を最大限に活用
uv run python -m tracknet.scripts.train \
  --data tracknet \
  --model vit_heatmap \
  --training qlora \
  training.memory_optimization.gradient_checkpointing=true \
  training.memory_optimization.use_cpu_offload=true \
  training.min_micro_batch_size=1
```

## 注意事項

1. **互換性**: 量子化はCUDA GPUでのみサポートされています
2. **モデル**: Vision Transformer系モデルで最も効果的です
3. **データセット**: 大規模データセットほど量子化の効果が大きいです
4. **保存**: 量子化されたモデルは特別な保存・読み込み手続きが必要です
5. **推論**: 推論時も同じ量子化設定を使用する必要があります
