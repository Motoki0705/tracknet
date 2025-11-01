"""Quantization training demonstration script.

This script demonstrates how to use int8 quantization and QLoRA for memory-efficient training.
"""

from __future__ import annotations

import torch
from omegaconf import OmegaConf

from tracknet.models import build_model
from tracknet.utils.quantization import (
    apply_int8_quantization,
    apply_lora,
    apply_qlora_quantization,
    get_memory_usage,
    setup_quantization_training,
)


def demo_int8_quantization():
    """Demonstrate int8 quantization setup."""
    print("=== int8量子化デモ ===")
    
    # モデル設定
    model_cfg = OmegaConf.create({
        "model_name": "vit_heatmap",
        "pretrained_model_name": "facebook/dinov3-vits16-pretrain-lvd1689m",
        "heatmap": {"size": [40, 40]},
        "backbone": {"patch_size": 16},
        "decoder": {
            "channels": [384, 192, 96, 1],
            "upsample": [4, 4, 2],
            "blocks_per_stage": 1,
            "norm": "gn",
            "activation": "gelu",
            "use_depthwise": True,
            "dropout": 0.0,
        },
    })
    
    # int8量子化設定
    training_cfg = OmegaConf.create({
        "quantization": {
            "enabled": True,
            "load_in_8bit": True,
            "llm_int8_threshold": 6.0,
            "llm_int8_has_fp16_weight": False,
            "llm_int8_skip_modules": [],
        }
    })
    
    # モデル構築と量子化適用
    print("モデルを構築中...")
    model = build_model(model_cfg)
    
    print("量子化前のメモリ使用量:")
    if torch.cuda.is_available():
        model = model.cuda()
        memory_before = get_memory_usage()
        print(f"  確保メモリ: {memory_before['allocated']:.2f} GB")
        print(f"  キャッシュメモリ: {memory_before['cached']:.2f} GB")
    
    print("int8量子化を適用中...")
    try:
        quantized_model = apply_int8_quantization(model, training_cfg.quantization)
        print("int8量子化が正常に適用されました")
        
        print("量子化後のメモリ使用量:")
        if torch.cuda.is_available():
            memory_after = get_memory_usage()
            print(f"  確保メモリ: {memory_after['allocated']:.2f} GB")
            print(f"  キャッシュメモリ: {memory_after['cached']:.2f} GB")
            
            reduction = (memory_before['allocated'] - memory_after['allocated']) / memory_before['allocated'] * 100
            print(f"  メモリ削減率: {reduction:.1f}%")
            
    except ImportError as e:
        print(f"エラー: {e}")
        print("bitsandbytesがインストールされていることを確認してください")


def demo_qlora():
    """Demonstrate QLoRA setup."""
    print("\n=== QLoRAデモ ===")
    
    # モデル設定
    model_cfg = OmegaConf.create({
        "model_name": "vit_heatmap",
        "pretrained_model_name": "facebook/dinov3-vits16-pretrain-lvd1689m",
        "heatmap": {"size": [40, 40]},
        "backbone": {"patch_size": 16},
        "decoder": {
            "channels": [384, 192, 96, 1],
            "upsample": [4, 4, 2],
            "blocks_per_stage": 1,
            "norm": "gn",
            "activation": "gelu",
            "use_depthwise": True,
            "dropout": 0.0,
        },
    })
    
    # QLoRA設定
    training_cfg = OmegaConf.create({
        "quantization": {
            "enabled": True,
            "load_in_4bit": True,
            "bnb_4bit_compute_dtype": "float16",
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_quant_type": "nf4",
        },
        "lora": {
            "enabled": True,
            "r": 16,
            "lora_alpha": 32,
            "target_modules": [],  # 自動検出
            "lora_dropout": 0.1,
            "bias": "none",
            "task_type": "FEATURE_EXTRACTION",
        }
    })
    
    # モデル構築とQLoRA適用
    print("モデルを構築中...")
    model = build_model(model_cfg)
    
    print("QLoRA適用前のメモリ使用量:")
    if torch.cuda.is_available():
        model = model.cuda()
        memory_before = get_memory_usage()
        print(f"  確保メモリ: {memory_before['allocated']:.2f} GB")
        print(f"  キャッシュメモリ: {memory_before['cached']:.2f} GB")
    
    print("QLoRAを適用中...")
    try:
        qlora_model = apply_qlora_quantization(
            model, 
            training_cfg.quantization, 
            training_cfg.lora
        )
        print("QLoRAが正常に適用されました")
        
        print("QLoRA適用後のメモリ使用量:")
        if torch.cuda.is_available():
            memory_after = get_memory_usage()
            print(f"  確保メモリ: {memory_after['allocated']:.2f} GB")
            print(f"  キャッシュメモリ: {memory_after['cached']:.2f} GB")
            
            reduction = (memory_before['allocated'] - memory_after['allocated']) / memory_before['allocated'] * 100
            print(f"  メモリ削減率: {reduction:.1f}%")
            
        # 学習可能パラメータ数を表示
        total_params = sum(p.numel() for p in qlora_model.parameters())
        trainable_params = sum(p.numel() for p in qlora_model.parameters() if p.requires_grad)
        
        print(f"総パラメータ数: {total_params:,}")
        print(f"学習可能パラメータ数: {trainable_params:,}")
        print(f"学習可能パラメータ率: {trainable_params/total_params*100:.2f}%")
        
    except ImportError as e:
        print(f"エラー: {e}")
        print("bitsandbytesとPEFTがインストールされていることを確認してください")


def demo_unified_setup():
    """Demonstrate unified quantization setup function."""
    print("\n=== 統一量子化セットアップデモ ===")
    
    # 完全な設定
    cfg = OmegaConf.create({
        "model": {
            "model_name": "vit_heatmap",
            "pretrained_model_name": "facebook/dinov3-vits16-pretrain-lvd1689m",
            "heatmap": {"size": [40, 40]},
            "backbone": {"patch_size": 16},
            "decoder": {
                "channels": [384, 192, 96, 1],
                "upsample": [4, 4, 2],
                "blocks_per_stage": 1,
                "norm": "gn",
                "activation": "gelu",
                "use_depthwise": True,
                "dropout": 0.0,
            },
        },
        "training": {
            "quantization": {
                "enabled": True,
                "load_in_8bit": True,
                "llm_int8_threshold": 6.0,
            },
            "lora": {
                "enabled": False,  # int8のみ
            }
        }
    })
    
    print("統一セットアップ関数を使用してモデルを構築中...")
    try:
        model = build_model(cfg.model, cfg.training)
        print("int8量子化モデルが正常に構築されました")
        
        # QLoRAに切り替え
        cfg.training.quantization.load_in_8bit = False
        cfg.training.quantization.load_in_4bit = True
        cfg.training.lora.enabled = True
        
        model = build_model(cfg.model, cfg.training)
        print("QLoRAモデルが正常に構築されました")
        
    except ImportError as e:
        print(f"エラー: {e}")


def main():
    """Main demonstration function."""
    print("TrackNet量子化学習デモ")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("警告: CUDAが利用できません。量子化のデモはGPUでのみ実行できます。")
    
    try:
        demo_int8_quantization()
        demo_qlora()
        demo_unified_setup()
        
        print("\n" + "=" * 50)
        print("デモ完了！")
        print("\n実用的な学習コマンド:")
        print("  int8: uv run python -m tracknet.scripts.train --data tracknet --model vit_heatmap --training int8")
        print("  QLoRA: uv run python -m tracknet.scripts.train --data tracknet --model vit_heatmap --training qlora")
        
    except Exception as e:
        print(f"デモ実行中にエラーが発生しました: {e}")
        print("依存関係が正しくインストールされていることを確認してください。")


if __name__ == "__main__":
    main()
