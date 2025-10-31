# 自動マイクロバッチング

TrackNetの学習におけるメモリ効率を向上させるための自動マイクロバッチング機能について説明します。

## 概要

自動マイクロバッチングは、OOM（Out of Memory）エラーを検知すると自動でバッチを分割し、メモリ不足を回避する機能です。

### 主な特徴

- **通常時は分割なし**: 元のバッチサイズで高速に実行されます
- **OOM時自動分割**: `B → B/2 → B/4 ...` のように自動でバッチサイズを縮小
- **成功サイズ記憶**: 一度成功したマイクロバッチサイズは記憶され、以後も使用
- **解凍時に再探索**: バックボーン解凍時に最適なサイズを再探索
- **AMP互換**: Lightningのprecision pluginに完全対応

## 設定パラメータ

```yaml
training:
  # 基本設定
  adaptive_micro_batch: true      # 自動マイクロバッチングを有効化
  min_micro_batch_size: 4         # これ以下にはバッチサイズを縮小しない
  mb_backoff_factor: 2            # バックオフ係数（2で半減）
  oom_retries: 3                  # OOM時の再試行回数
  micro_batch_size: 0             # 0より大きい値で固定サイズ（自動探索を無効化）
  grad_clip_norm: 1.0             # 勾配クリップのノルム値
```

### パラメータ詳細

| パラメータ | 型 | デフォルト | 説明 |
|-----------|----|-----------|------|
| `adaptive_micro_batch` | bool | `true` | 自動マイクロバッチングを有効にするかどうか |
| `min_micro_batch_size` | int | `1` | バッチサイズの最小値。これより小さくはならない |
| `mb_backoff_factor` | int | `2` | OOM時のバックオフ係数。`2`なら半減、`4`なら1/4になる |
| `oom_retries` | int | `3` | OOM時の最大再試行回数 |
| `micro_batch_size` | int | `0` | `0`より大きい値を設定すると、そのサイズで固定される |
| `grad_clip_norm` | float | `0.0` | 勾配クリップのノルム値。`0.0`で無効化 |

## 動作原理

### 学習ステップの流れ

1. **通常実行**: まず元のバッチサイズ `B` で実行を試みる
2. **OOM検知**: OOMが発生した場合、自動でマイクロバッチングを開始
3. **バックオフ**: `B/mb_backoff_factor` にサイズを縮小して再試行
4. **成功記録**: 成功したサイズを `_runtime_mb` として記録
5. **継続使用**: 以後のバッチでは記録されたサイズを使用

### 勾配蓄積の仕組み

```python
# マイクロバッチごとに損失を計算し、勾配を蓄積
for i in range(0, B, mb):
    loss_mb = self.criterion(out, yb, mbb) / num_micro  # 平均化
    self.manual_backward(loss_mb)  # 勾配蓄積のみ

# 全マイクロバッチ終了後に1回だけoptimizer.step()
opt.step()
```

### バックボーン解凍時の再探索

```python
def on_train_epoch_start(self):
    if self.current_epoch == self._freeze_epochs:  # 解凍タイミング
        self._runtime_mb = None  # 記憶をリセットして再探索
```

## 使用例

### 基本的な使用法

```yaml
# configs/training/default.yaml
training:
  adaptive_micro_batch: true
  min_micro_batch_size: 4
  mb_backoff_factor: 2
  oom_retries: 3
```

### 固定マイクロバッチサイズ

```yaml
training:
  micro_batch_size: 8  # 自動探索を無効化し、8で固定
```

### メモリが厳しい環境向け

```yaml
training:
  adaptive_micro_batch: true
  min_micro_batch_size: 2    # より小さなサイズまで許容
  mb_backoff_factor: 4       # より aggressive に縮小
  oom_retries: 5             # より多く再試行
```

## 注意事項

### Lightningのaccumulate_grad_batchesとの併用

- **併用しないでください**: 本実装は1 DataLoaderバッチ内での分割のみ行います
- 複数バッチにまたがる見かけの大バッチが必要な場合は、別途`global_accum_steps`を実装してください

### OOM後の処理

- OOM後はCUDAがエラーステートになるため、必ず勾配解放と`empty_cache()`を実行
- `_handle_oom()`メソッドがこの処理を自動化

### 検証時の動作

- 検証時も同様の自動分割が適用されます
- プレビュー保存時のOOMはtry/exceptで保護されています

## パフォーマンスへの影響

### 通常時
- オーバーヘッドはほぼなし
- 元のバッチサイズと等価の速度で実行

### OOM時
- バッチ分割によるオーバーヘッドが発生
- ただし学習を継続できるため、全体的なスループットは向上

### メモリ使用量
- 最大で元のバッチサイズと同じメモリを使用
- OOM時は自動で縮小されるため、安全に実行可能

## トラブルシューティング

### よくある問題

1. **OOMが頻発する**
   - `min_micro_batch_size`を小さくする
   - `mb_backoff_factor`を大きくする（4など）
   - 元の`batch_size`を小さくする

2. **学習が遅すぎる**
   - `micro_batch_size`で固定サイズを試す
   - `min_micro_batch_size`を大きくする

3. **メモリが十分にあるのに分割される**
   - `adaptive_micro_batch: false`で無効化
   - `micro_batch_size`に大きな値を設定

### ログでの確認

```python
# 現在のマイクロバッチサイズはログで確認可能
self.log("train/micro_batch_size", mb, on_step=True)
```

## 実装の詳細

### 主要メソッド

- `_run_micro_batches()`: マイクロバッチ実行の本体
- `_handle_oom()`: OOM時のクリーンアップ処理
- `training_step()`: OOM検知と再試行ロジック
- `validation_step()`: 検証時の自動分割

### 手動最適化への移行

本機能ではLightningの`automatic_optimization = False`を使用し、手動で勾配蓄積とoptimizer.step()を制御しています。これにより、マイクロバッチングの柔軟な実装が可能になっています。
