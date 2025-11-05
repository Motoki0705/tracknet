# LoRAと量子化機能への移行ガイド

このガイドでは、既存のTrackNet設定をLoRAと量子化機能に移行する方法を説明します。

## 概要

TrackNet v0.1.0以降では、大規模な事前学習済みモデルを効率的にファインチューニングするためのLoRA（Low-Rank Adaptation）と量子化機能が追加されました。これにより、メモリ使用量を削減し、学習速度を向上させることができます。

## 主な変更点

- 新機能: LoRAによるパラメータ効率の良いファインチューニング
- 新機能: INT4/FP4量子化によるメモリ削減
- 下位互換性: 既存の設定は変更なしで動作
- 新設定オプション: `lora`と`quantization`セクション

## 移行手順

### ステップ1: 既存設定の確認

まず、現在のモデル設定を確認します。

```yaml
# 既存の設定例
model_name: "vit_heatmap"
pretrained_model_name: facebook/dinov3-vits16-pretrain-lvd1689m

backbone:
  freeze: true
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
```

### ステップ2: LoRAを有効化する

メモリ効率を改善したい場合、LoRAを有効化します。

```yaml
# LoRAを有効化した設定
model_name: "vit_heatmap"
pretrained_model_name: facebook/dinov3-vits16-pretrain-lvd1689m

backbone:
  freeze: false  # LoRAを使用する場合はfalseに設定
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

# 新規追加: LoRA設定
lora:
  enabled: true
  r: 16
  lora_alpha: 32
  lora_dropout: 0.05
  target_modules: null  # 自動検出を使用
  bias: "none"
  task_type: "FEATURE_EXTRACTION"
```

### ステップ3: 量子化を有効化する（オプション）

さらにメモリを削減したい場合、量子化を有効化します。

```yaml
# QLoRA（量子化 + LoRA）設定
model_name: "vit_heatmap"
pretrained_model_name: facebook/dinov3-vits16-pretrain-lvd1689m

backbone:
  freeze: false
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

lora:
  enabled: true
  r: 16
  lora_alpha: 32
  lora_dropout: 0.05
  target_modules: null
  bias: "none"
  task_type: "FEATURE_EXTRACTION"

# 新規追加: 量子化設定
quantization:
  enabled: true
  quant_type: "nf4"
  compute_dtype: "bfloat16"
  skip_modules: []
  mode: "manual"
  compress_statistics: true
  use_double_quant: true
```

## 設定オプション詳細

### LoRA設定

| パラメータ | 型 | デフォルト | 説明 |
|-----------|----|-----------|------|
| `enabled` | bool | false | LoRAを有効化するかどうか |
| `r` | int | 16 | LoRAのランク（8, 16, 32, 64など） |
| `lora_alpha` | int | 32 | スケーリング係数（通常rの2倍） |
| `lora_dropout` | float | 0.05 | ドロップアウト率（0.0-0.1） |
| `target_modules` | list | null | LoRAを適用するモジュールリスト（nullで自動検出） |
| `bias` | str | "none" | バイアス設定（"none", "all", "lora_only"） |
| `task_type` | str | "FEATURE_EXTRACTION" | タスクタイプ |

### 量子化設定

| パラメータ | 型 | デフォルト | 説明 |
|-----------|----|-----------|------|
| `enabled` | bool | false | 量子化を有効化するかどうか |
| `quant_type` | str | "nf4" | 量子化タイプ（"nf4", "fp4"） |
| `compute_dtype` | str | "bfloat16" | 計算精度（"bfloat16", "float16"） |
| `skip_modules` | list | [] | 量子化をスキップするモジュールリスト |
| `mode` | str | "manual" | 量子化モード（"manual", "hf"） |
| `compress_statistics` | bool | true | 統計情報を圧縮するか（manualモードのみ） |
| `use_double_quant` | bool | true | 二重量子化を使用するか（HFモードのみ） |

## パフォーマンス比較

| 設定 | メモリ使用量 | 学習パラメータ数 | 推論速度 | 推奨用途 |
|------|-------------|------------------|----------|----------|
| 従来通り | 100% | 100% | 基準 | 小規模モデル、十分なメモリがある場合 |
| LoRAのみ | ~90% | ~5% | 基準 | 中規模モデル、学習速度を重視する場合 |
| QLoRA | ~25% | ~5% | 基準 | 大規模モデル、メモリ制約がある場合 |

## よくある質問

### Q: 既存のモデルは引き続き使用できますか？

A: はい、既存の設定は変更なしで動作します。LoRAと量子化はオプション機能です。

### Q: LoRAと量子化を同時に使用する必要がありますか？

A: いいえ、それぞれ独立して使用できます。ただし、量子化を使用する場合はLoRAも併用することを推奨します（QLoRA）。

### Q: どのモデルタイプでLoRAと量子化が使用できますか？

A: 現在はViTバックボーンモデルで主にサポートされています。将来的には他のモデルタイプにも拡張される予定です。

### Q: 学習済みのLoRAアダプターを保存するには？

A: 通常のモデル保存と同じ方法で保存できます。LoRAアダプターはモデルに含まれています。

### Q: 推論時にも量子化は有効ですか？

A: はい、推論時も量子化されたモデルを使用することでメモリ効率が向上します。

## トラブルシューティング

### メモリ不足エラー

```bash
# 解決策1: 量子化を有効化する
quantization:
  enabled: true

# 解決策2: LoRAのランクを下げる
lora:
  r: 8  # 16から8に減少

# 解決策3: バッチサイズを小さくする
training:
  batch_size: 4  # 8から4に減少
```

### 学習が不安定な場合

```yaml
# 解決策1: LoRAのドロップアウトを調整
lora:
  lora_dropout: 0.1  # 0.05から0.1に増加

# 解決策2: 学習率を下げる
training:
  optimizer:
    lr: 1e-5  # 1e-4から1e-5に減少
```

### 精度が低下した場合

```yaml
# 解決策1: LoRAのランクを増やす
lora:
  r: 32  # 16から32に増加

# 解決策2: 量子化を無効化する
quantization:
  enabled: false
```

## サンプル設定

### 小規模データセット向け

```yaml
lora:
  enabled: true
  r: 8
  lora_alpha: 16
  lora_dropout: 0.1

quantization:
  enabled: false
```

### 大規模データセット向け

```yaml
lora:
  enabled: true
  r: 32
  lora_alpha: 64
  lora_dropout: 0.05

quantization:
  enabled: true
  quant_type: "nf4"
  compute_dtype: "bfloat16"
```

### メモリ制約が厳しい環境向け

```yaml
lora:
  enabled: true
  r: 8
  lora_alpha: 16
  lora_dropout: 0.05

quantization:
  enabled: true
  quant_type: "nf4"
  compute_dtype: "float16"  # bfloat16よりメモリ効率が良い
```

## サポート

移行に関する問題や質問がある場合は、以下のリソースを参照してください：

- [README.md](../README.md) - 基本的な使用方法
- [テストコード](../tests/) - 実装例
- [GitHub Issues](https://github.com/your-repo/tracknet/issues) - バグ報告や機能リクエスト
