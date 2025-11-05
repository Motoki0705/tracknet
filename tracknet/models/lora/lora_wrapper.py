"""LoRA wrapper utilities for TrackNet models.

This module provides functions to apply LoRA (Low-Rank Adaptation)
to pretrained models, enabling parameter-efficient fine-tuning.
"""

from __future__ import annotations

import torch.nn as nn

from tracknet.models.lora.config import LoRAConfig


def auto_target_modules(model: nn.Module) -> list[str]:
    """Automatically detect target modules for LoRA.
    
    Args:
        model: PyTorch model to analyze.
        
    Returns:
        List[str]: List of module names suitable for LoRA.
    """
    candidates = [
        "q_proj",
        "k_proj", 
        "v_proj",
        "o_proj",
        "qkv",
        "attn.proj",
        "attention.query",
        "attention.key",
        "attention.value",
        "attention.output.dense",
        "mlp.fc1",
        "mlp.fc2",
        "intermediate.dense",
        "output.dense",
        "fc1",
        "fc2",
        "proj",
    ]
    
    module_names = [name for name, _ in model.named_modules()]
    target_modules = sorted({
        candidate for candidate in candidates 
        if any(candidate in name for name in module_names)
    })
    
    # Filter out normalization layers
    target_modules = [module for module in target_modules if "norm" not in module.lower()]
    
    # Fallback to basic linear layers if no specific modules found
    if not target_modules:
        target_modules = ["fc1", "fc2"]
    
    return target_modules


def apply_lora_to_model(
    model: nn.Module,
    config: LoRAConfig,
    target_modules: list[str] | None = None,
) -> nn.Module:
    """Apply LoRA to a PyTorch model.

    Args:
        model: PyTorch model to apply LoRA to.
        config: LoRA configuration.
        target_modules: List of target module names. If None, auto-detect.

    Returns:
        nn.Module: Model with LoRA applied.

    Raises:
        ImportError: If PEFT library is not available.
        RuntimeError: If LoRA application fails.
    """
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as e:
        raise ImportError(
            "PEFT library is required for LoRA. "
            "Install with: pip install peft"
        ) from e

    try:
        # Auto-detect target modules if not specified
        if target_modules is None:
            target_modules = auto_target_modules(model)

        # Create PEFT LoRA config
        peft_config = LoraConfig(
            r=config.r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=target_modules,
            bias=config.bias,
            task_type=TaskType.FEATURE_EXTRACTION,
        )

        # Apply LoRA to model
        lora_model = get_peft_model(model, peft_config)
        return lora_model

    except Exception as e:
        raise RuntimeError(f"Failed to apply LoRA to model: {e}") from e


def prepare_model_for_kbit_training(model: nn.Module) -> nn.Module:
    """Prepare model for k-bit training (quantization-aware).
    
    This function should be called after quantization but before LoRA
    to ensure proper gradient handling.
    
    Args:
        model: Quantized PyTorch model.
    
    Returns:
        nn.Module: Model prepared for k-bit training.
    
    Raises:
        ImportError: If PEFT library is not available.
    """
    try:
        from peft import prepare_model_for_kbit_training
    except ImportError as e:
        raise ImportError(
            "PEFT library is required for k-bit training preparation. "
            "Install with: pip install peft"
        ) from e

    try:
        return prepare_model_for_kbit_training(model)
    except Exception as e:
        raise RuntimeError(f"Failed to prepare model for k-bit training: {e}") from e


def get_lora_trainable_parameters(model: nn.Module) -> tuple[int, int]:
    """Get count of trainable and total parameters in LoRA model.
    
    Args:
        model: LoRA-enhanced PyTorch model.
        
    Returns:
        tuple[int, int]: (trainable_params, total_params)
    """
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    total_params = sum(p.numel() for p in model.parameters())
    
    return trainable_params, total_params


def print_lora_trainable_parameters(model: nn.Module) -> None:
    """Print information about trainable parameters in LoRA model.
    
    Args:
        model: LoRA-enhanced PyTorch model.
    """
    trainable_params, total_params = get_lora_trainable_parameters(model)
    percentage = 100 * trainable_params / total_params if total_params > 0 else 0
    
    print(
        f"trainable params: {trainable_params:,} || "
        f"all params: {total_params:,} || "
        f"trainable%: {percentage:.2f}"
    )
