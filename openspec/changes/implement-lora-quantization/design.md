# LoRA and Quantization Design

## Architecture Overview

This design introduces a modular approach to LoRA and quantization that integrates cleanly with the existing TrackNet architecture.

### Core Components

#### 1. LoRA Module (`tracknet/models/lora/`)

```
tracknet/models/lora/
├── __init__.py
├── lora_wrapper.py      # LoRA application utilities
├── quantization.py      # Quantization utilities
└── config.py           # Configuration dataclasses
```

#### 2. Integration Points

- **Model Builder**: Extend `tracknet/models/build.py` to support LoRA/quantization configs
- **Configuration**: New YAML configs for LoRA-enabled models
- **Training**: Ensure `tracknet/training/lightning_module.py` compatibility

### Technical Design

#### LoRA Wrapper Design

```python
@dataclass
class LoRAConfig:
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] | None = None
    bias: str = "none"

def apply_lora_to_model(
    model: nn.Module,
    config: LoRAConfig,
    target_modules: List[str] | None = None
) -> nn.Module
```

#### Quantization Design

```python
@dataclass
class QuantizationConfig:
    enabled: bool = False
    quant_type: str = "nf4"  # "nf4" or "fp4"
    compute_dtype: torch.dtype = torch.bfloat16
    skip_modules: List[str] | None = None
    mode: str = "manual"  # "manual" or "hf"

def apply_quantization(
    model: nn.Module,
    config: QuantizationConfig
) -> nn.Module
```

#### Configuration Integration

Extended model configs will include:

```yaml
model_name: "vit_lora_heatmap"
# ... existing vit config ...
lora:
  enabled: true
  r: 16
  lora_alpha: 32
  lora_dropout: 0.05
  target_modules: ["query", "key", "value", "dense"]
quantization:
  enabled: true
  quant_type: "nf4"
  compute_dtype: "bfloat16"
  skip_modules: ["pooler.dense"]
  mode: "manual"
```

### Implementation Strategy

#### Phase 1: Core LoRA/Quantization Modules
1. Create `tracknet/models/lora/` package
2. Implement `lora_wrapper.py` based on demo code patterns
3. Implement `quantization.py` with manual and HF quantization modes
4. Add configuration dataclasses

#### Phase 2: Model Builder Integration
1. Extend `build.py` to detect LoRA/quantization configs
2. Add conditional model wrapping logic
3. Ensure backward compatibility with existing configs

#### Phase 3: Training Compatibility
1. Verify Lightning module works with quantized+LoRA models
2. Add proper parameter filtering for optimizers
3. Handle gradient checkpointing if needed

#### Phase 4: Configuration and Testing
1. Create example YAML configs
2. Add unit tests for LoRA/quantization functionality
3. Integration tests with training pipeline

### Key Design Decisions

1. **Modular Approach**: LoRA/quantization as separate modules that wrap existing models
2. **Configuration-Driven**: All behavior controlled via YAML configs
3. **Demo-Based Implementation**: Leverage proven patterns from `hf_vit_qlora_demo_v3.py`
4. **Backward Compatibility**: Existing configs continue to work unchanged
5. **ViT-First Focus**: Primary support for ViT models with extensible architecture

### Dependencies

- `peft`: For LoRA implementation
- `bitsandbytes`: For INT4 quantization
- `transformers`: Already required for ViT models

### Error Handling

- Graceful fallback when LoRA/quantization libraries unavailable
- Clear error messages for configuration mismatches
- Validation of quantization compatibility with target modules

### Performance Considerations

- Lazy loading of LoRA/quantization libraries
- Efficient parameter filtering for optimizer setup
- Memory-efficient training with quantized backbones
