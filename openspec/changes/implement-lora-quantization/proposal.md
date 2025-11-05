# Implement LoRA and Quantization Modules for TrackNet

## Summary

This proposal introduces LoRA (Low-Rank Adaptation) and quantization capabilities to the TrackNet project, enabling efficient fine-tuning of large pretrained models with reduced memory footprint and computational requirements.

## Problem Statement

The current TrackNet implementation supports full fine-tuning of backbone models, which can be memory-intensive and computationally expensive for large ViT models. There is a need for:

1. **Memory-efficient training**: Reduce GPU memory usage during training
2. **Faster fine-tuning**: Enable quicker adaptation to new tennis ball detection tasks
3. **Flexible quantization**: Support different quantization schemes (INT4, FP4, etc.)
4. **Configurable LoRA**: Allow selective application of LoRA to specific layers

## Proposed Solution

Create a new `tracknet/models/lora/` module that provides:

1. **LoRA integration**: Wrapper classes for applying LoRA to pretrained model linear layers
2. **Quantization support**: INT4/FP4 quantization with configurable skip modules
3. **Configuration-driven**: YAML-based configuration for LoRA and quantization settings
4. **ViT backbone focus**: Primary support for ViT models with extensible architecture

## Key Features

- **LoRA module**: `tracknet/models/lora/lora_wrapper.py` - Apply LoRA to linear layers
- **Quantization module**: `tracknet/models/lora/quantization.py` - INT4/FP4 quantization utilities
- **Configuration integration**: Extend build.py to support LoRA/quantization configs
- **Training compatibility**: Ensure Lightning module works with quantized+LoRA models

## Benefits

- **Reduced memory usage**: INT4 quantization + LoRA reduces memory by ~75%
- **Faster training**: Fewer trainable parameters accelerate convergence
- **Flexible deployment**: Support both quantized and full-precision inference
- **Backward compatibility**: Existing configs and models continue to work

## Scope

This change focuses on ViT backbone models but maintains an extensible architecture for future model types. The implementation follows patterns established in the demo code while integrating cleanly with the existing TrackNet architecture.
