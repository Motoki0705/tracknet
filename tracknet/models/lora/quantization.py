"""Quantization utilities for TrackNet models.

This module provides functions to apply INT4/FP4 quantization
to pretrained models, enabling memory-efficient training.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn

from tracknet.models.lora.config import QuantizationConfig


def _get_parent_and_attr(model: nn.Module, name: str) -> tuple[nn.Module, str]:
    """Get parent module and attribute name from a full module name.
    
    Args:
        model: Root PyTorch model.
        name: Full module name (e.g., "backbone.model.layers.0.linear").
        
    Returns:
        tuple[nn.Module, str]: (parent_module, attribute_name)
    """
    parent = model
    parts = name.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def convert_linear_to_int4_manual(
    model: nn.Module,
    *,
    skip_modules: Iterable[str] | None = None,
    quant_type: str = "nf4",
    compute_dtype: torch.dtype = torch.bfloat16,
    compress_statistics: bool = True,
) -> nn.Module:
    """Convert linear layers to INT4 using manual bitsandbytes approach.
    
    Args:
        model: PyTorch model to quantize.
        skip_modules: Iterable of module names to skip during quantization.
        quant_type: Type of 4-bit quantization ("nf4" or "fp4").
        compute_dtype: Data type for computations.
        compress_statistics: Whether to compress statistics.
        
    Returns:
        nn.Module: Quantized model.
        
    Raises:
        ImportError: If bitsandbytes is not available.
        RuntimeError: If quantization fails.
    """
    try:
        import bitsandbytes as bnb
    except ImportError as e:
        raise ImportError(
            "bitsandbytes is required for quantization. "
            "Install with: pip install bitsandbytes"
        ) from e
    
    skip_modules = list(skip_modules or [])
    
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear):
            # Skip specified modules
            if any(skip_name in name for skip_name in skip_modules):
                continue
            
            try:
                # Get parent module and attribute name
                parent, attr = _get_parent_and_attr(model, name)
                
                # Create quantized linear layer
                quantized_layer = bnb.nn.Linear4bit(
                    module.in_features,
                    module.out_features,
                    bias=(module.bias is not None),
                    compute_dtype=compute_dtype,
                    quant_type=quant_type,
                    compress_statistics=compress_statistics,
                )
                
                # Copy weights and bias
                with torch.no_grad():
                    quantized_layer.weight.data.copy_(module.weight.data)
                    if module.bias is not None:
                        quantized_layer.bias.data.copy_(module.bias.data)
                
                # Replace original layer with quantized version
                setattr(parent, attr, quantized_layer)
                
            except Exception as e:
                raise RuntimeError(f"Failed to quantize module {name}: {e}") from e
    
    return model


def apply_hf_quantization(
    model: nn.Module,
    model_name: str,
    *,
    quant_type: str = "nf4",
    compute_dtype: torch.dtype = torch.bfloat16,
    use_double_quant: bool = True,
    device_map: str | None = None,
) -> nn.Module:
    """Apply quantization using HuggingFace BitsAndBytesConfig.

    Args:
        model: PyTorch model to quantize.
        model_name: Name of the model for loading with quantization.
        quant_type: Type of 4-bit quantization ("nf4" or "fp4").
        compute_dtype: Compute dtype for quantized operations.
        use_double_quant: Whether to use double quantization.
        device_map: Device mapping strategy.

    Returns:
        nn.Module: Quantized model.

    Raises:
        ImportError: If required libraries are not available.
        RuntimeError: If quantization fails.
    """
    try:
        from transformers import BitsAndBytesConfig
    except ImportError as e:
        raise ImportError(
            "transformers is required for HF quantization. "
            "Install with: pip install transformers"
        ) from e
    
    try:
        # Create BitsAndBytesConfig
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=quant_type,
            bnb_4bit_use_double_quant=use_double_quant,
        )
        
        # Note: In actual usage, this would reload the model with quantization
        # For now, we return the original model as a placeholder
        print(f"Warning: HF quantization mode requires model reloading. "
              f"Returning original model for {model_name}")
        return model
        
    except Exception as e:
        raise RuntimeError(f"Failed to apply HF quantization: {e}") from e


def apply_quantization(
    model: nn.Module,
    config: QuantizationConfig,
    model_name: str | None = None,
) -> nn.Module:
    """Apply quantization to a PyTorch model based on configuration.
    
    Args:
        model: PyTorch model to quantize.
        config: Quantization configuration.
        model_name: Name/path of the pretrained model (required for HF mode).
        
    Returns:
        nn.Module: Quantized model.
        
    Raises:
        ValueError: If configuration is invalid.
        ImportError: If required libraries are not available.
        RuntimeError: If quantization fails.
    """
    if not config.enabled:
        return model
    
    if config.mode == "manual":
        return convert_linear_to_int4_manual(
            model,
            skip_modules=config.skip_modules,
            quant_type=config.quant_type,
            compute_dtype=config.compute_dtype,
            compress_statistics=config.compress_statistics,
        )
    
    elif config.mode == "hf":
        if model_name is None:
            raise ValueError("model_name is required for HF quantization mode")
        
        return apply_hf_quantization(
            model,
            model_name=model_name,
            quant_type=config.quant_type,
            compute_dtype=config.compute_dtype,
            use_double_quant=config.use_double_quant,
        )
    
    else:
        raise ValueError(f"Unsupported quantization mode: {config.mode}")


def validate_quantization_compatibility(
    model: nn.Module,
    skip_modules: Iterable[str] | None = None,
) -> bool:
    """Validate if model is compatible with quantization.

    Args:
        model: PyTorch model to validate.
        skip_modules: Iterable of module names to skip during validation.

    Returns:
        bool: True if model is compatible, False otherwise.
    """
    if skip_modules is None:
        skip_modules = []
    
    for name, module in model.named_modules():
        if (
            isinstance(module, nn.Linear)
            and not any(skip_name in name for skip_name in skip_modules)
            and (module.in_features <= 0 or module.out_features <= 0)
        ):
            print(f"Warning: Module {name} has invalid dimensions: "
                  f"{module.in_features} -> {module.out_features}")
            return False
    
    return True


def get_quantization_memory_info(model: nn.Module) -> dict[str, int]:
    """Get memory usage information for quantized model.
    
    Args:
        model: PyTorch model (quantized or not).
        
    Returns:
        dict[str, int]: Dictionary with memory usage information.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Estimate memory usage (rough approximation)
    # FP32: 4 bytes per parameter
    # INT4: 0.5 bytes per parameter (4-bit)
    fp32_memory_mb = total_params * 4 // (1024 * 1024)
    int4_memory_mb = total_params // (2 * 1024 * 1024)  # 4-bit = 0.5 bytes
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "fp32_memory_mb": fp32_memory_mb,
        "int4_memory_mb": int4_memory_mb,
        "memory_reduction_percent": 100 * (fp32_memory_mb - int4_memory_mb) // fp32_memory_mb if fp32_memory_mb > 0 else 0,
    }
