# 量子化学習クイックスタートガイド

TrackNetのint8量子化とQLoRA機能をすぐに使い始めるためのガイドです。

## 🚀 クイックスタート

### 1. int8量子化学習

```bash
# 基本的なint8学習
uv run python -m tracknet.scripts.train \
  --data tracknet \
  --model vit_heatmap \
  --training int8

# メモリ使用量をさらに削減
uv run python -m tracknet.scripts.train \
  --data tracknet \
  --model vit_heatmap \
  --training int8 \
  training.batch_size=8 \
  training.memory_optimization.gradient_checkpointing=true
```

### 2. QLoRA学習

```bash
# 基本的なQLoRA学習
uv run python -m tracknet.scripts.train \
  --data tracknet \
  --model vit_heatmap \
  --training qlora

# LoRAパラメータをカスタマイズ
uv run python -m tracknet.scripts.train \
  --data tracknet \
  --model vit_heatmap \
  --training qlora \
  training.lora.r=32 \
  training.lora.lora_alpha=64 \
  training.batch_size=4
```

### 3. メモリ最適化付きQLoRA

```bash
# 最大限のメモリ効率
uv run python -m tracknet.scripts.train \
  --data tracknet \
  --model vit_heatmap \
  --training qlora \
  training.memory_optimization.gradient_checkpointing=true \
  training.memory_optimization.use_cpu_offload=true \
  training.min_micro_batch_size=1 \
  training.batch_size=2
```

## 📊 どの手法を選ぶべきか？

| GPUメモリ | 推奨手法 | バッチサイズ | 精度 | 速度 |
|-----------|----------|-------------|------|------|
| 16GB+ | 通常学習 (FP16) | 32 | 100% | 基準 |
| 12GB | int8量子化 | 16 | 98-99% | 1.2x |
| 8GB | QLoRA | 8 | 97-99% | 0.8x |
| 4GB | QLoRA + CPUオフロード | 4 | 95-98% | 0.5x |

## ⚙️ 設定ファイルの場所

- `configs/training/int8.yaml` - int8量子化設定
- `configs/training/qlora.yaml` - QLoRA設定
- `configs/training/default.yaml` - 通常学習設定

## 🔧 よくあるカスタマイズ

### バッチサイズの調整

```bash
# int8の場合
training.batch_size=8  # 16→8に削減

# QLoRAの場合
training.batch_size=4  # 8→4に削減
```

### LoRAパラメータの調整

```bash
# より高い精度が必要な場合
training.lora.r=32
training.lora.lora_alpha=64

# より少ないメモリ使用量が必要な場合
training.lora.r=8
training.lora.lora_alpha=16
```

### 学習率の調整

```bash
# int8の場合
training.optimizer.lr=3e-4

# QLoRAの場合
training.optimizer.lr=1e-4
```

## 📈 モニタリング

学習の進捗とメモリ使用量を監視：

```bash
# TensorBoardを起動
tensorboard --logdir=outputs/logs

# 監視するメトリクス:
# - train/loss: 学習損失
# - memory/allocated_gb: 確保済みGPUメモリ
# - memory/cached_gb: キャッシュGPUメモリ
```

## 🆘 トラブルシューティング

### OOMエラー

```bash
# 対策1: バッチサイズを削減
training.batch_size=4

# 対策2: マイクロバッチを有効化
training.adaptive_micro_batch=true
training.min_micro_batch_size=1

# 対策3: メモリ最適化を有効化
training.memory_optimization.gradient_checkpointing=true
training.memory_optimization.use_cpu_offload=true
```

### 精度が低下

```bash
# 対策1: LoRAランクを増やす
training.lora.r=32
training.lora.lora_alpha=64

# 対策2: 学習率を調整
training.optimizer.lr=2e-4

# 対策3: エポック数を増やす
training.epochs=30
```

## 📝 詳細ドキュメント

- [`quantization.md`](quantization.md) - 詳細な技術ドキュメント
- [`micro_batching.md`](micro_batching.md) - 自動マイクロバッチング
- [`trainer.md`](trainer.md) - トレーナー設定

## 🎯 ベストプラクティス

### int8量子化のベストプラクティス

1. **バッチサイズ**: 16以下に設定
2. **学習率**: 3e-4程度に調整
3. **ウォームアップ**: 2エポック程度
4. **勾配クリッピング**: 1.0を推奨

### QLoRAのベストプラクティス

1. **バッチサイズ**: 8以下に設定
2. **LoRAランク**: 16-32の範囲で調整
3. **学習率**: 1e-4程度に調整
4. **エポック数**: 30以上を推奨
5. **ウォームアップ**: 5エポック程度

### 共通のベストプラクティス

1. **混合精度**: fp16またはbf16を使用
2. **マイクロバッチング**: 常に有効化
3. **メモリ監視**: TensorBoardで監視
4. **段階的なスケールアップ**: 小さい設定から始める

## 🔄 モデルの保存と読み込み

### int8量子化モデル

```python
# 保存
torch.save(quantized_model.state_dict(), "int8_model.pth")

# 読み込み
model = build_model(model_cfg, training_cfg)
model.load_state_dict(torch.load("int8_model.pth"))
```

### QLoRAモデル

```python
# LoRAアダプターのみを保存
model.save_pretrained("lora_adapters")

# 読み込み
from peft import PeftModel
base_model = build_model(model_cfg)
model = PeftModel.from_pretrained(base_model, "lora_adapters")
```

## 📞 サポート

問題が発生した場合：

1. まずこのガイドのトラブルシューティングセクションを確認
2. [`quantization.md`](quantization.md)の詳細ドキュメントを参照
3. GPUメモリと依存関係を確認
4. 必要に応じてGitHubでIssueを作成

---

**Happy Training! 🎾**
