"""Configuration dataclasses for LoRA and quantization.

This module defines the configuration classes used to control
LoRA and quantization behavior in TrackNet models.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class LoRAConfig:
    """Configuration for LoRA (Low-Rank Adaptation).

    Args:
        r: Rank of the update matrices. Lower rank means fewer parameters.
        lora_alpha: Scaling factor for LoRA weights.
        lora_dropout: Dropout probability for LoRA layers.
        target_modules: List of module names to apply LoRA to. If None,
            auto-detection will be used.
        bias: LoRA bias configuration ("none", "all", or "lora_only").
        task_type: Type of task for LoRA configuration.
    """

    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] | None = None
    bias: str = "none"
    task_type: str = "FEATURE_EXTRACTION"

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.r <= 0:
            raise ValueError("LoRA rank 'r' must be positive")
        if self.lora_alpha <= 0:
            raise ValueError("LoRA alpha must be positive")
        if not 0.0 <= self.lora_dropout <= 1.0:
            raise ValueError("LoRA dropout must be between 0.0 and 1.0")
        if self.bias not in ["none", "all", "lora_only"]:
            raise ValueError("bias must be one of: 'none', 'all', 'lora_only'")


@dataclass
class QuantizationConfig:
    """Configuration for model quantization.

    Args:
        enabled: Whether quantization is enabled.
        quant_type: Type of quantization ("nf4" or "fp4").
        compute_dtype: Data type for computations.
        skip_modules: List of module names to skip during quantization.
        mode: Quantization mode ("manual" or "hf").
        compress_statistics: Whether to compress statistics (manual mode only).
        use_double_quant: Whether to use double quantization (HF mode only).
    """

    enabled: bool = False
    quant_type: str = "nf4"
    compute_dtype: torch.dtype = torch.bfloat16
    skip_modules: list[str] | None = field(default_factory=list)
    mode: str = "manual"
    compress_statistics: bool = True
    use_double_quant: bool = True

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.quant_type not in ["nf4", "fp4"]:
            raise ValueError("quant_type must be one of: 'nf4', 'fp4'")
        if self.mode not in ["manual", "hf"]:
            raise ValueError("mode must be one of: 'manual', 'hf'")
        if self.compute_dtype not in [torch.float16, torch.bfloat16]:
            raise ValueError("compute_dtype must be float16 or bfloat16")


def parse_dtype(dtype_str: str) -> torch.dtype:
    """Parse string to torch dtype.

    Args:
        dtype_str: String representation of dtype ("float16", "bfloat16").

    Returns:
        torch.dtype: Corresponding torch dtype.

    Raises:
        ValueError: If dtype_str is not recognized.
    """
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }

    if dtype_str.lower() not in dtype_map:
        raise ValueError(f"Unsupported dtype: {dtype_str}. Use 'float16' or 'bfloat16'")

    return dtype_map[dtype_str.lower()]
