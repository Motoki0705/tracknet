"""Quantization utilities for int8 and QLoRA training.

This module provides functions to apply int8 quantization and LoRA adapters
to models using bitsandbytes and PEFT libraries.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from omegaconf import DictConfig

try:
    import bitsandbytes as bnb
    from peft import LoraConfig, get_peft_model, TaskType
    from transformers import BitsAndBytesConfig
    BNB_AVAILABLE = True
except ImportError:
    BNB_AVAILABLE = False


def apply_int8_quantization(model: nn.Module, config: DictConfig) -> nn.Module:
    """Apply int8 quantization to a model using bitsandbytes.
    
    Args:
        model: PyTorch model to quantize.
        config: Configuration dictionary with quantization settings.
        
    Returns:
        Quantized model.
        
    Raises:
        ImportError: If bitsandbytes is not available.
    """
    if not BNB_AVAILABLE:
        raise ImportError("bitsandbytes is required for int8 quantization")
    
    # Create quantization config
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=float(config.get("llm_int8_threshold", 6.0)),
        llm_int8_has_fp16_weight=bool(config.get("llm_int8_has_fp16_weight", False)),
        llm_int8_skip_modules=config.get("llm_int8_skip_modules", []),
    )
    
    # For vision models, we need to manually apply quantization
    # since transformers BitsAndBytesConfig is mainly for LLMs
    return _apply_int8_to_vision_model(model, config)


def _apply_int8_to_vision_model(model: nn.Module, config: DictConfig) -> nn.Module:
    """Apply int8 quantization to vision transformer models.
    
    Args:
        model: Vision model to quantize.
        config: Quantization configuration.
        
    Returns:
        Quantized model.
    """
    # Replace linear layers with int8 quantized versions
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Skip modules specified in config
            skip_modules = config.get("llm_int8_skip_modules", [])
            if any(skip_mod in name for skip_mod in skip_modules):
                continue
                
            # Create int8 linear layer
            parent_module = model
            for part in name.split('.')[:-1]:
                parent_module = getattr(parent_module, part)
            
            layer_name = name.split('.')[-1]
            quantized_layer = bnb.nn.Int8Params(
                module.weight.data, 
                requires_grad=False, 
                has_fp16_weights=config.get("llm_int8_has_fp16_weight", False)
            )
            
            # Replace the layer
            setattr(parent_module, layer_name, quantized_layer)
    
    return model


def apply_lora(
    model: nn.Module, 
    config: DictConfig
) -> nn.Module:
    """Apply LoRA adapters to a model using PEFT.
    
    Args:
        model: PyTorch model to apply LoRA to.
        config: Configuration dictionary with LoRA settings.
        
    Returns:
        Model with LoRA adapters applied.
        
    Raises:
        ImportError: If PEFT is not available.
    """
    if not BNB_AVAILABLE:
        raise ImportError("PEFT is required for LoRA training")
    
    # Create LoRA config
    lora_config = LoraConfig(
        r=int(config.get("r", 16)),
        lora_alpha=int(config.get("lora_alpha", 32)),
        target_modules=config.get("target_modules", None),  # None for auto-detection
        lora_dropout=float(config.get("lora_dropout", 0.1)),
        bias=config.get("bias", "none"),
        task_type=TaskType.FEATURE_EXTRACTION,  # Suitable for vision models
    )
    
    # Apply LoRA to model
    lora_model = get_peft_model(model, lora_config)
    
    # Print trainable parameters info
    lora_model.print_trainable_parameters()
    
    return lora_model


def apply_qlora_quantization(
    model: nn.Module, 
    quantization_config: DictConfig,
    lora_config: DictConfig
) -> nn.Module:
    """Apply QLoRA (4-bit quantization + LoRA) to a model.
    
    Args:
        model: PyTorch model to quantize and apply LoRA to.
        quantization_config: 4-bit quantization configuration.
        lora_config: LoRA configuration.
        
    Returns:
        Model with QLoRA applied.
    """
    if not BNB_AVAILABLE:
        raise ImportError("bitsandbytes and PEFT are required for QLoRA training")
    
    # Create 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=getattr(torch, quantization_config.get("bnb_4bit_compute_dtype", "float16")),
        bnb_4bit_use_double_quant=bool(quantization_config.get("bnb_4bit_use_double_quant", True)),
        bnb_4bit_quant_type=quantization_config.get("bnb_4bit_quant_type", "nf4"),
    )
    
    # For vision models, we need a different approach
    # since transformers BitsAndBytesConfig is mainly for LLMs
    model = _apply_4bit_to_vision_model(model, quantization_config)
    
    # Apply LoRA
    lora_model = apply_lora(model, lora_config)
    
    return lora_model


def _apply_4bit_to_vision_model(model: nn.Module, config: DictConfig) -> nn.Module:
    """Apply 4-bit quantization to vision transformer models.
    
    Args:
        model: Vision model to quantize.
        config: 4-bit quantization configuration.
        
    Returns:
        Quantized model.
    """
    # Replace linear layers with 4-bit quantized versions
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Create 4-bit quantized linear layer
            parent_module = model
            for part in name.split('.')[:-1]:
                parent_module = getattr(parent_module, part)
            
            layer_name = name.split('.')[-1]
            
            # Use bitsandbytes 4-bit quantization
            quantized_layer = bnb.nn.Params4bit(
                module.weight.data,
                requires_grad=False,
                compress_statistics=config.get("bnb_4bit_use_double_quant", True),
                quant_type=config.get("bnb_4bit_quant_type", "nf4")
            )
            
            # Replace the layer
            setattr(parent_module, layer_name, quantized_layer)
    
    return model


def setup_quantization_training(model: nn.Module, cfg: DictConfig) -> nn.Module:
    """Setup quantization training based on configuration.
    
    Args:
        model: PyTorch model to setup quantization for.
        cfg: Training configuration.
        
    Returns:
        Model with appropriate quantization applied.
    """
    quant_cfg = cfg.get("quantization", {})
    lora_cfg = cfg.get("lora", {})
    
    # Check if quantization is enabled
    if not quant_cfg.get("enabled", False):
        return model
    
    # int8 quantization
    if quant_cfg.get("load_in_8bit", False):
        model = apply_int8_quantization(model, quant_cfg)
        print("Applied int8 quantization to model")
    
    # QLoRA (4-bit + LoRA)
    elif quant_cfg.get("load_in_4bit", False) and lora_cfg.get("enabled", False):
        model = apply_qlora_quantization(model, quant_cfg, lora_cfg)
        print("Applied QLoRA (4-bit quantization + LoRA) to model")
    
    # LoRA only
    elif lora_cfg.get("enabled", False):
        model = apply_lora(model, lora_cfg)
        print("Applied LoRA adapters to model")
    
    return model


def get_memory_usage() -> Dict[str, float]:
    """Get current GPU memory usage.
    
    Returns:
        Dictionary with memory usage statistics.
    """
    if not torch.cuda.is_available():
        return {"available": 0.0, "used": 0.0, "total": 0.0}
    
    return {
        "allocated": torch.cuda.memory_allocated() / 1024**3,  # GB
        "cached": torch.cuda.memory_reserved() / 1024**3,     # GB
        "max_allocated": torch.cuda.max_memory_allocated() / 1024**3,  # GB
    }


def clear_gpu_cache() -> None:
    """Clear GPU cache to free memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
