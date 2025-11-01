# ConvNeXt + Deformable DETR + FPN Heatmap Model

## Overview

The `convnext_deformable_fpn_heatmap` model combines ConvNeXt backbone features with Deformable DETR's multi-scale deformable attention mechanism and Feature Pyramid Network (FPN) for heatmap generation. This architecture is particularly effective for tennis ball detection tasks that require multi-scale feature fusion and precise localization.

## Architecture

### 1. ConvNeXt Backbone
- Uses pretrained ConvNeXt models from Hugging Face
- Extracts multi-scale features from stages [1, 2, 3, 4]
- Output channels: [128, 256, 512, 1024] for base model
- Features are ordered from high to low resolution

### 2. Deformable Transformer Encoder
- Projects ConvNeXt features to transformer dimension (default: 256)
- Adds positional embeddings using sine encoding
- Adds level embeddings to distinguish feature scales
- Applies multi-scale deformable attention for cross-scale feature fusion
- Uses multiple encoder layers (default: 3) for deep feature integration

### 3. Feature Pyramid Network (FPN)
- Projects encoded features back to FPN dimension
- Applies standard FPN with lateral connections
- Upsamples all features to target heatmap resolution
- Combines multi-scale features through summation

### 4. Heatmap Head
- Simple convolutional head for final heatmap prediction
- Outputs Gaussian heatmaps for tennis ball localization

## Key Features

### Multi-Scale Deformable Attention
- Leverages Deformable DETR's attention mechanism for adaptive feature sampling
- Enables cross-scale information flow between different ConvNeXt stages
- Improves feature fusion compared to simple FPN concatenation

### Position and Level Embeddings
- Positional embeddings provide spatial information
- Level embeddings distinguish between different feature scales
- Combined embeddings guide the attention mechanism effectively

### Flexible Configuration
- Configurable transformer dimensions, heads, and layers
- Adjustable number of sampling points for deformable attention
- Customizable FPN output dimensions

## Configuration

```yaml
model_name: "convnext_deformable_fpn_heatmap"
pretrained_model_name: "facebook/dinov3-convnext-base-pretrain-lvd1689m"

backbone:
  freeze: true
  return_stages: [1, 2, 3, 4]
  device_map: auto
  local_files_only: true

deformable_encoder:
  d_model: 256              # Transformer hidden dimension
  nhead: 8                  # Number of attention heads
  num_encoder_layers: 3     # Number of encoder layers
  num_feature_levels: 4     # Number of feature scales
  n_points: 4               # Sampling points per head
  lateral_dim: 256          # FPN lateral dimension

heatmap:
  size: [256, 144]          # Target heatmap size [W, H]
  sigma: 2.0                # Gaussian sigma for heatmaps
```

## Performance Characteristics

### Advantages
- **Multi-scale fusion**: Effectively combines features from different resolutions
- **Adaptive attention**: Deformable attention focuses on relevant regions
- **Precise localization**: FPN provides high-resolution feature maps
- **Pretrained backbone**: Leverages large-scale ConvNeXt pretraining

### Computational Cost
- Higher computational cost than standard FPN due to transformer encoder
- Memory usage scales with number of feature levels and encoder layers
- Suitable for applications requiring high accuracy over real-time performance

## Usage

### Building the Model
```python
from tracknet.models.build import build_model
import hydra
from omegaconf import DictConfig

# Load configuration
cfg = ...  # Load your config
model = build_model(cfg.model)

# Forward pass
images = torch.randn(2, 3, 224, 224)
heatmaps = model(images)
print(f"Output shape: {heatmaps.shape}")  # [2, 144, 256]
```

### Training
The model can be trained using the standard TrackNet training pipeline:
```bash
uv run python tracknet/scripts/train.py --config configs/model/convnext_deformable_fpn_heatmap.yaml
```

### Inference
```python
model.eval()
with torch.no_grad():
    heatmaps = model(images)
    # Post-process heatmaps for ball detection
```

## Implementation Details

### Deformable Attention Mechanism
- Uses MSDeformAttn from Deformable DETR
- Supports multiple feature levels with adaptive sampling
- Each attention head samples n_points per feature level

### Feature Flow
1. ConvNeXt extracts multi-scale features [C2, C3, C4, C5]
2. Features are projected to transformer dimension
3. Position and level embeddings are added
4. Deformable transformer encoder processes features
5. Features are reshaped and projected to FPN dimension
6. FPN generates pyramid features
7. All features are upsampled and summed
8. Heatmap head produces final output

## Dependencies

- PyTorch
- torchvision (for FPN)
- transformers (for ConvNeXt backbone)
- Deformable DETR components (included in third_party)

## Comparison with Other Models

| Model | Multi-scale Fusion | Attention Mechanism | Parameters | Accuracy |
|-------|-------------------|-------------------|------------|----------|
| convnext_fpn_heatmap | FPN concatenation/sum | None | Medium | Good |
| convnext_deformable_fpn_heatmap | Deformable attention + FPN | Multi-scale deformable | High | Excellent |
| vit_heatmap | Upsampling | Self-attention | High | Good |

## File Structure

```
tracknet/models/
├── decoders/
│   └── deformable_decoder.py    # Deformable DETR decoder implementation
├── build.py                     # Model factory with new variant
└── ...

configs/model/
└── convnext_deformable_fpn_heatmap.yaml  # Model configuration

docs/models/
└── convnext_deformable_fpn_heatmap.md    # This documentation
```

## Future Improvements

- Add support for different backbone architectures
- Implement decoder-only variant for faster inference
- Add attention visualization tools
- Optimize memory usage for larger input sizes
