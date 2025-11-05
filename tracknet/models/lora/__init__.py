"""LoRA and Quantization modules for TrackNet.

This package provides utilities for applying LoRA (Low-Rank Adaptation)
and quantization to pretrained models, enabling efficient fine-tuning
with reduced memory footprint.
"""

from __future__ import annotations

from tracknet.models.lora.config import LoRAConfig, QuantizationConfig
from tracknet.models.lora.lora_wrapper import apply_lora_to_model
from tracknet.models.lora.quantization import apply_quantization

__all__ = [
    "LoRAConfig",
    "QuantizationConfig", 
    "apply_lora_to_model",
    "apply_quantization",
]
