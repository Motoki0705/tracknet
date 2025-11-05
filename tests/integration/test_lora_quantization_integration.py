"""Integration tests for LoRA and quantization with training pipeline.

This module tests the end-to-end functionality of LoRA and quantization
with the TrackNet training pipeline.
"""

import pytest
import torch
from omegaconf import OmegaConf

from tracknet.models.build import build_model
from tracknet.training.lightning_module import PLHeatmapModule


class TestModelBuildingIntegration:
    """Test model building with LoRA and quantization configurations."""

    def test_vit_model_with_lora_config(self):
        """Test building ViT model with LoRA configuration."""
        model_cfg = OmegaConf.create({
            "model_name": "vit_heatmap",
            "pretrained_model_name": "facebook/dinov3-vits16-pretrain-lvd1689m",
            "backbone": {
                "freeze": False,
                "device_map": None,  # CPU for testing
                "local_files_only": False,  # Allow download for testing
                "patch_size": 16,
            },
            "decoder": {
                "channels": [384, 256, 128, 64],
                "upsample": [2, 2, 2],
                "blocks_per_stage": 1,
                "norm": "gn",
                "activation": "gelu",
                "use_depthwise": True,
                "use_se": False,
                "se_reduction": 8,
                "dropout": 0.0,
            },
            "heatmap": {
                "size": [64, 36],  # Smaller for testing
                "sigma": 2.0,
            },
            "lora": {
                "enabled": True,
                "r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.1,
                "target_modules": None,  # Auto-detect
                "bias": "none",
                "task_type": "FEATURE_EXTRACTION",
            },
            "quantization": {
                "enabled": False,  # Disable for basic test
            }
        })
        
        try:
            model = build_model(model_cfg)
            
            # Check model structure
            assert hasattr(model, 'backbone')
            assert hasattr(model, 'decoder')
            assert hasattr(model, 'head')
            
            # Check that model is not None
            assert model is not None
            
            # Test forward pass with dummy input
            dummy_input = torch.randn(1, 3, 144, 256)  # Small batch for testing
            with torch.no_grad():
                output = model(dummy_input)
            
            # Check output shape
            assert output.shape[0] == 1  # Batch size
            assert output.shape[1] == 1  # Single heatmap channel
            assert output.shape[2] == 36  # Height
            assert output.shape[3] == 64  # Width
            
        except ImportError as e:
            pytest.skip(f"Required libraries not available: {e}")

    def test_vit_model_with_quantization_config(self):
        """Test building ViT model with quantization configuration."""
        model_cfg = OmegaConf.create({
            "model_name": "vit_heatmap",
            "pretrained_model_name": "facebook/dinov3-vits16-pretrain-lvd1689m",
            "backbone": {
                "freeze": False,
                "device_map": None,
                "local_files_only": False,
                "patch_size": 16,
            },
            "decoder": {
                "channels": [384, 256, 128, 64],
                "upsample": [2, 2, 2],
                "blocks_per_stage": 1,
                "norm": "gn",
                "activation": "gelu",
                "use_depthwise": True,
                "use_se": False,
                "se_reduction": 8,
                "dropout": 0.0,
            },
            "heatmap": {
                "size": [64, 36],
                "sigma": 2.0,
            },
            "lora": {
                "enabled": False,  # Disable for quantization-only test
            },
            "quantization": {
                "enabled": True,
                "quant_type": "nf4",
                "compute_dtype": "bfloat16",
                "skip_modules": [],
                "mode": "manual",
                "compress_statistics": True,
                "use_double_quant": True,
            }
        })
        
        try:
            model = build_model(model_cfg)
            
            # Check model structure
            assert hasattr(model, 'backbone')
            assert hasattr(model, 'decoder')
            assert hasattr(model, 'head')
            
            # Test forward pass
            dummy_input = torch.randn(1, 3, 144, 256)
            with torch.no_grad():
                output = model(dummy_input)
            
            # Check output shape
            assert output.shape == (1, 1, 36, 64)
            
        except ImportError as e:
            pytest.skip(f"Required libraries not available: {e}")

    def test_backward_compatibility(self):
        """Test that existing configurations still work."""
        model_cfg = OmegaConf.create({
            "model_name": "vit_heatmap",
            "pretrained_model_name": "facebook/dinov3-vits16-pretrain-lvd1689m",
            "backbone": {
                "freeze": True,
                "device_map": None,
                "local_files_only": False,
                "patch_size": 16,
            },
            "decoder": {
                "channels": [384, 256, 128, 64],
                "upsample": [2, 2, 2],
                "blocks_per_stage": 1,
                "norm": "gn",
                "activation": "gelu",
                "use_depthwise": True,
                "use_se": False,
                "se_reduction": 8,
                "dropout": 0.0,
            },
            "heatmap": {
                "size": [64, 36],
                "sigma": 2.0,
            }
            # No lora or quantization sections
        })
        
        try:
            model = build_model(model_cfg)
            
            # Check that backbone is frozen
            backbone_trainable = any(p.requires_grad for p in model.backbone.parameters())
            assert not backbone_trainable
            
            # Test forward pass
            dummy_input = torch.randn(1, 3, 144, 256)
            with torch.no_grad():
                output = model(dummy_input)
            
            assert output.shape == (1, 1, 36, 64)
            
        except ImportError as e:
            pytest.skip(f"Required libraries not available: {e}")


class TestLightningModuleIntegration:
    """Test Lightning module integration with LoRA and quantization."""

    def test_lightning_module_with_lora(self):
        """Test Lightning module with LoRA model."""
        cfg = OmegaConf.create({
            "model": {
                "model_name": "vit_heatmap",
                "pretrained_model_name": "facebook/dinov3-vits16-pretrain-lvd1689m",
                "backbone": {
                    "freeze": False,
                    "device_map": None,
                    "local_files_only": False,
                    "patch_size": 16,
                },
                "decoder": {
                    "channels": [384, 256, 128, 64],
                    "upsample": [2, 2, 2],
                    "blocks_per_stage": 1,
                    "norm": "gn",
                    "activation": "gelu",
                    "use_depthwise": True,
                    "use_se": False,
                    "se_reduction": 8,
                    "dropout": 0.0,
                },
                "heatmap": {
                    "size": [64, 36],
                    "sigma": 2.0,
                },
                "lora": {
                    "enabled": True,
                    "r": 8,
                    "lora_alpha": 16,
                    "lora_dropout": 0.1,
                    "target_modules": None,
                    "bias": "none",
                    "task_type": "FEATURE_EXTRACTION",
                },
                "quantization": {
                    "enabled": False,
                }
            },
            "training": {
                "optimizer": {
                    "name": "adamw",
                    "lr": 1e-4,
                    "weight_decay": 0.01,
                },
                "backbone_freeze_epochs": 0,
            },
            "runtime": {
                "run_id": "test_run",
            }
        })
        
        try:
            lightning_module = PLHeatmapModule(cfg)
            
            # Check that model is built
            assert hasattr(lightning_module, 'model')
            assert hasattr(lightning_module, 'criterion')
            
            # Test forward pass
            dummy_input = torch.randn(2, 3, 144, 256)
            with torch.no_grad():
                output = lightning_module(dummy_input)
            
            assert output.shape == (2, 1, 36, 64)
            
            # Test optimizer configuration
            optimizer = lightning_module.configure_optimizers()
            if isinstance(optimizer, dict):
                optimizer = optimizer["optimizer"]
            
            # Check that optimizer has parameters
            assert len(optimizer.param_groups) > 0
            
        except ImportError as e:
            pytest.skip(f"Required libraries not available: {e}")

    def test_lightning_module_parameter_filtering(self):
        """Test that optimizer correctly filters trainable parameters."""
        cfg = OmegaConf.create({
            "model": {
                "model_name": "vit_heatmap",
                "pretrained_model_name": "facebook/dinov3-vits16-pretrain-lvd1689m",
                "backbone": {
                    "freeze": False,
                    "device_map": None,
                    "local_files_only": False,
                    "patch_size": 16,
                },
                "decoder": {
                    "channels": [384, 256, 128, 64],
                    "upsample": [2, 2, 2],
                    "blocks_per_stage": 1,
                    "norm": "gn",
                    "activation": "gelu",
                    "use_depthwise": True,
                    "use_se": False,
                    "se_reduction": 8,
                    "dropout": 0.0,
                },
                "heatmap": {
                    "size": [64, 36],
                    "sigma": 2.0,
                },
                "lora": {
                    "enabled": True,
                    "r": 8,
                    "lora_alpha": 16,
                    "lora_dropout": 0.1,
                    "target_modules": None,
                    "bias": "none",
                    "task_type": "FEATURE_EXTRACTION",
                },
                "quantization": {
                    "enabled": False,
                }
            },
            "training": {
                "optimizer": {
                    "name": "adamw",
                    "lr": 1e-4,
                    "weight_decay": 0.01,
                },
                "backbone_freeze_epochs": 0,
            },
            "runtime": {
                "run_id": "test_run",
            }
        })
        
        try:
            lightning_module = PLHeatmapModule(cfg)
            
            # Count trainable parameters
            total_params = sum(p.numel() for p in lightning_module.model.parameters())
            trainable_params = sum(p.numel() for p in lightning_module.model.parameters() if p.requires_grad)
            
            # With LoRA, trainable params should be less than total
            assert trainable_params < total_params
            
            # Check percentage is reasonable (should be small for LoRA)
            trainable_percentage = 100 * trainable_params / total_params
            assert trainable_percentage < 50  # Should be much less than 50%
            
        except ImportError as e:
            pytest.skip(f"Required libraries not available: {e}")


class TestConfigurationValidation:
    """Test configuration validation and error handling."""

    def test_invalid_lora_config(self):
        """Test handling of invalid LoRA configuration."""
        model_cfg = OmegaConf.create({
            "model_name": "vit_heatmap",
            "pretrained_model_name": "facebook/dinov3-vits16-pretrain-lvd1689m",
            "backbone": {
                "freeze": False,
                "device_map": None,
                "local_files_only": True,  # Use local files only to avoid download
                "patch_size": 16,
            },
            "decoder": {
                "channels": [384, 256, 128, 64],
                "upsample": [2, 2, 2],
                "blocks_per_stage": 1,
                "norm": "gn",
                "activation": "gelu",
                "use_depthwise": True,
                "use_se": False,
                "se_reduction": 8,
                "dropout": 0.0,
            },
            "heatmap": {
                "size": [64, 36],
                "sigma": 2.0,
            },
            "lora": {
                "enabled": True,
                "r": -1,  # Invalid
                "lora_alpha": 16,
                "lora_dropout": 0.1,
                "target_modules": None,
                "bias": "none",
                "task_type": "FEATURE_EXTRACTION",
            },
            "quantization": {
                "enabled": False,
            }
        })
        
        with pytest.raises(ValueError, match="LoRA rank 'r' must be positive"):
            build_model(model_cfg)

    def test_invalid_quantization_config(self):
        """Test handling of invalid quantization configuration."""
        model_cfg = OmegaConf.create({
            "model_name": "vit_heatmap",
            "pretrained_model_name": "facebook/dinov3-vits16-pretrain-lvd1689m",
            "backbone": {
                "freeze": False,
                "device_map": None,
                "local_files_only": True,  # Use local files only
                "patch_size": 16,
            },
            "decoder": {
                "channels": [384, 256, 128, 64],
                "upsample": [2, 2, 2],
                "blocks_per_stage": 1,
                "norm": "gn",
                "activation": "gelu",
                "use_depthwise": True,
                "use_se": False,
                "se_reduction": 8,
                "dropout": 0.0,
            },
            "heatmap": {
                "size": [64, 36],
                "sigma": 2.0,
            },
            "lora": {
                "enabled": False,
            },
            "quantization": {
                "enabled": True,
                "quant_type": "invalid",  # Invalid
                "compute_dtype": "bfloat16",
                "skip_modules": [],
                "mode": "manual",
                "compress_statistics": True,
                "use_double_quant": True,
            }
        })
        
        with pytest.raises(ValueError, match="quant_type must be one of"):
            build_model(model_cfg)


if __name__ == "__main__":
    pytest.main([__file__])
