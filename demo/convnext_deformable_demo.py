"""Demo script for ConvNeXt + Deformable DETR + FPN heatmap model."""

import torch
from omegaconf import OmegaConf

from tracknet.models.build import build_model


def main():
    """Test the ConvNeXt Deformable FPN model."""
    print("=== ConvNeXt + Deformable DETR + FPN Demo ===")
    
    # Create model configuration
    model_cfg = OmegaConf.create({
        'model_name': 'convnext_deformable_fpn_heatmap',
        'pretrained_model_name': 'facebook/dinov3-convnext-base-pretrain-lvd1689m',
        'backbone': {
            'freeze': True,
            'return_stages': [1, 2, 3, 4],
            'device_map': 'auto',
            'local_files_only': True
        },
        'deformable_encoder': {
            'd_model': 256,
            'nhead': 8,
            'num_encoder_layers': 3,
            'num_feature_levels': 4,
            'n_points': 4,
            'lateral_dim': 256
        },
        'heatmap': {
            'size': [256, 144],
            'sigma': 2.0
        }
    })
    
    try:
        # Build model
        print("Building model...")
        model = build_model(model_cfg)
        print(f"✓ Model built successfully: {model.variant}")
        
        # Create dummy input
        batch_size = 1
        dummy_input = torch.randn(batch_size, 3, 224, 224)
        print(f"Input shape: {dummy_input.shape}")
        
        # Test forward pass
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"✓ Forward pass successful!")
        print(f"Output shape: {output.shape}")
        
        # Verify output size
        expected_h, expected_w = model_cfg.heatmap.size[1], model_cfg.heatmap.size[0]
        assert output.shape == (batch_size, expected_h, expected_w)
        print(f"✓ Output matches expected size: {expected_w}x{expected_h}")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        print("\n=== Demo completed successfully! ===")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
