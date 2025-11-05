"""Unit tests for LoRA functionality.

This module tests the LoRA wrapper utilities and configuration
to ensure they work correctly with different model types and configurations.
"""

from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn as nn

from tracknet.models.lora.config import LoRAConfig, parse_dtype
from tracknet.models.lora.lora_wrapper import (
    apply_lora_to_model,
    auto_target_modules,
    get_lora_trainable_parameters,
    prepare_model_for_kbit_training,
    print_lora_trainable_parameters,
)


class TestLoRAConfig:
    """Test LoRA configuration parsing and validation."""

    def test_default_config(self):
        """Test default LoRA configuration."""
        config = LoRAConfig()
        assert config.r == 16
        assert config.lora_alpha == 32
        assert config.lora_dropout == 0.05
        assert config.target_modules is None
        assert config.bias == "none"
        assert config.task_type == "FEATURE_EXTRACTION"

    def test_custom_config(self):
        """Test custom LoRA configuration."""
        config = LoRAConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["query", "key"],
            bias="lora_only",
            task_type="FEATURE_EXTRACTION"
        )
        assert config.r == 8
        assert config.lora_alpha == 16
        assert config.lora_dropout == 0.1
        assert config.target_modules == ["query", "key"]
        assert config.bias == "lora_only"

    def test_config_validation(self):
        """Test configuration validation."""
        # Test invalid rank
        with pytest.raises(ValueError, match="LoRA rank 'r' must be positive"):
            LoRAConfig(r=0)

        # Test invalid alpha
        with pytest.raises(ValueError, match="LoRA alpha must be positive"):
            LoRAConfig(lora_alpha=-1)

        # Test invalid dropout
        with pytest.raises(ValueError, match="LoRA dropout must be between 0.0 and 1.0"):
            LoRAConfig(lora_dropout=1.5)

        # Test invalid bias
        with pytest.raises(ValueError, match="bias must be one of"):
            LoRAConfig(bias="invalid")


class TestParseDtype:
    """Test dtype parsing utility."""

    def test_parse_bfloat16(self):
        """Test parsing bfloat16 dtype."""
        dtype = parse_dtype("bfloat16")
        assert dtype == torch.bfloat16

    def test_parse_float16(self):
        """Test parsing float16 dtype."""
        dtype = parse_dtype("float16")
        assert dtype == torch.float16

    def test_parse_fp16(self):
        """Test parsing fp16 alias."""
        dtype = parse_dtype("fp16")
        assert dtype == torch.float16

    def test_parse_bf16(self):
        """Test parsing bf16 alias."""
        dtype = parse_dtype("bf16")
        assert dtype == torch.bfloat16

    def test_parse_invalid_dtype(self):
        """Test parsing invalid dtype."""
        with pytest.raises(ValueError, match="Unsupported dtype"):
            parse_dtype("invalid")


class TestAutoTargetModules:
    """Test automatic target module detection."""

    def test_vit_like_modules(self):
        """Test target module detection for ViT-like models."""
        # Create a mock model with ViT-like module names
        model = Mock()
        model.named_modules.return_value = [
            ("backbone.layers.0.attention.query", Mock()),
            ("backbone.layers.0.attention.key", Mock()),
            ("backbone.layers.0.attention.value", Mock()),
            ("backbone.layers.0.attention.dense", Mock()),
            ("backbone.layers.0.mlp.fc1", Mock()),
            ("backbone.layers.0.mlp.fc2", Mock()),
            ("backbone.layernorm", Mock()),  # Should be filtered out
        ]

        targets = auto_target_modules(model)
        
        # Should include attention and MLP modules (check for substrings)
        assert any("query" in t for t in targets)
        assert any("key" in t for t in targets)
        assert any("value" in t for t in targets)
        assert any("dense" in t for t in targets)
        assert any("fc1" in t for t in targets)
        assert any("fc2" in t for t in targets)
        
        # Should not include normalization layers
        assert not any("norm" in t for t in targets)

    def test_convnext_like_modules(self):
        """Test target module detection for ConvNeXt-like models."""
        model = Mock()
        model.named_modules.return_value = [
            ("backbone.stages.0.blocks.0.fc1", Mock()),
            ("backbone.stages.0.blocks.0.fc2", Mock()),
            ("backbone.norm", Mock()),  # Should be filtered out
        ]

        targets = auto_target_modules(model)
        
        assert "fc1" in targets
        assert "fc2" in targets
        assert "norm" not in targets

    def test_fallback_modules(self):
        """Test fallback to basic modules when no specific modules found."""
        model = Mock()
        model.named_modules.return_value = [
            ("some.random.module", Mock()),
            ("another.module", Mock()),
        ]

        targets = auto_target_modules(model)
        
        # Should fallback to basic linear layer names
        assert "fc1" in targets
        assert "fc2" in targets


class MockModel(nn.Module):
    """Mock model for testing LoRA functionality."""

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 20)
        self.linear2 = nn.Linear(20, 5)
        self.norm = nn.LayerNorm(5)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        x = self.norm(x)
        return x


class TestApplyLoRA:
    """Test LoRA application to models."""

    @patch('tracknet.models.lora.lora_wrapper.get_peft_model')
    @patch('tracknet.models.lora.lora_wrapper.LoraConfig')
    def test_apply_lora_success(self, mock_lora_config, mock_get_peft_model):
        """Test successful LoRA application."""
        # Setup mocks
        mock_config = LoRAConfig()
        mock_lora_instance = Mock()
        mock_lora_config.return_value = mock_lora_instance
        mock_lora_model = Mock()
        mock_get_peft_model.return_value = mock_lora_model

        # Create test model
        model = MockModel()
        
        # Apply LoRA
        result = apply_lora_to_model(model, mock_config, ["linear1", "linear2"])

        # Verify calls
        mock_lora_config.assert_called_once()
        mock_get_peft_model.assert_called_once_with(model, mock_lora_instance)
        assert result == mock_lora_model

    @patch('tracknet.models.lora.lora_wrapper.get_peft_model')
    @patch('tracknet.models.lora.lora_wrapper.LoraConfig')
    def test_apply_lora_with_auto_target(self, mock_lora_config, mock_get_peft_model):
        """Test LoRA application with automatic target detection."""
        mock_config = LoRAConfig()
        mock_lora_instance = Mock()
        mock_lora_config.return_value = mock_lora_instance
        mock_lora_model = Mock()
        mock_get_peft_model.return_value = mock_lora_model

        model = MockModel()
        
        # Apply LoRA without specifying target modules
        apply_lora_to_model(model, mock_config, target_modules=None)

        # Should still call get_peft_model
        mock_get_peft_model.assert_called_once()

    def test_apply_lora_missing_peft(self):
        """Test LoRA application when PEFT is not available."""
        with patch('builtins.__import__') as mock_import:
            mock_import.side_effect = ImportError("No module named 'peft'")
            
            model = MockModel()
            config = LoRAConfig()
            
            with pytest.raises(ImportError, match="PEFT library is required"):
                apply_lora_to_model(model, config)


class TestPrepareKbitTraining:
    """Test k-bit training preparation."""

    @patch('tracknet.models.lora.lora_wrapper.prepare_model_for_kbit_training')
    def test_prepare_kbit_success(self, mock_prepare):
        """Test successful k-bit training preparation."""
        mock_model = MockModel()  # Use real model instead of Mock
        mock_prepared = Mock()
        mock_prepare.return_value = mock_prepared

        result = prepare_model_for_kbit_training(mock_model)
        
        mock_prepare.assert_called_once_with(mock_model)
        assert result == mock_prepared

    def test_prepare_kbit_missing_peft(self):
        """Test k-bit preparation when PEFT is not available."""
        with patch('builtins.__import__') as mock_import:
            mock_import.side_effect = ImportError("No module named 'peft'")
            
            model = MockModel()
            
            with pytest.raises(ImportError, match="PEFT library is required"):
                prepare_model_for_kbit_training(model)


class TestLoRAParameterUtils:
    """Test LoRA parameter utilities."""

    def test_get_trainable_parameters(self):
        """Test getting trainable parameter counts."""
        model = MockModel()
        
        # Freeze some parameters
        for param in model.linear1.parameters():
            param.requires_grad = False
        
        trainable, total = get_lora_trainable_parameters(model)
        
        # Should count only trainable parameters
        assert trainable < total
        assert total == sum(p.numel() for p in model.parameters())

    def test_print_trainable_parameters(self, capsys):
        """Test printing trainable parameter information."""
        model = MockModel()
        
        # Freeze some parameters to simulate LoRA
        for param in model.linear1.parameters():
            param.requires_grad = False
        
        print_lora_trainable_parameters(model)
        
        captured = capsys.readouterr()
        assert "trainable params:" in captured.out
        assert "all params:" in captured.out
        assert "trainable%:" in captured.out


if __name__ == "__main__":
    pytest.main([__file__])
