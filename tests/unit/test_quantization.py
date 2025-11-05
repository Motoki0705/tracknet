"""Unit tests for quantization functionality.

This module tests the quantization utilities and configuration
to ensure they work correctly with different model types and settings.
"""

from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn as nn

from tracknet.models.lora.config import QuantizationConfig
from tracknet.models.lora.quantization import (
    _get_parent_and_attr,
    apply_hf_quantization,
    apply_quantization,
    convert_linear_to_int4_manual,
    get_quantization_memory_info,
    validate_quantization_compatibility,
)


class TestQuantizationConfig:
    """Test quantization configuration parsing and validation."""

    def test_default_config(self):
        """Test default quantization configuration."""
        config = QuantizationConfig()
        assert config.enabled is False
        assert config.quant_type == "nf4"
        assert config.compute_dtype == torch.bfloat16
        assert config.skip_modules == []
        assert config.mode == "manual"
        assert config.compress_statistics is True
        assert config.use_double_quant is True

    def test_custom_config(self):
        """Test custom quantization configuration."""
        config = QuantizationConfig(
            enabled=True,
            quant_type="fp4",
            compute_dtype=torch.float16,
            skip_modules=["attention", "mlp"],
            mode="hf",
            compress_statistics=False,
            use_double_quant=False,
        )
        assert config.enabled is True
        assert config.quant_type == "fp4"
        assert config.compute_dtype == torch.float16
        assert config.skip_modules == ["attention", "mlp"]
        assert config.mode == "hf"
        assert config.compress_statistics is False
        assert config.use_double_quant is False

    def test_config_validation(self):
        """Test configuration validation."""
        # Test invalid quant_type
        with pytest.raises(ValueError, match="quant_type must be one of"):
            QuantizationConfig(quant_type="invalid")

        # Test invalid mode
        with pytest.raises(ValueError, match="mode must be one of"):
            QuantizationConfig(mode="invalid")

        # Test invalid compute_dtype
        with pytest.raises(ValueError, match="compute_dtype must be"):
            QuantizationConfig(compute_dtype=torch.float32)


class TestGetParentAndAttr:
    """Test parent and attribute extraction utility."""

    def test_simple_module_name(self):
        """Test simple module name extraction."""
        model = Mock()
        child = Mock()
        model.child = child

        parent, attr = _get_parent_and_attr(model, "child")

        assert parent is model
        assert attr == "child"

    def test_nested_module_name(self):
        """Test nested module name extraction."""
        model = Mock()
        intermediate = Mock()
        child = Mock()
        model.backbone = intermediate
        intermediate.layer = child

        parent, attr = _get_parent_and_attr(model, "backbone.layer")

        assert parent is intermediate
        assert attr == "layer"

    def test_deeply_nested_module_name(self):
        """Test deeply nested module name extraction."""
        model = Mock()
        level1 = Mock()
        level2 = Mock()
        level3 = Mock()
        child = Mock()

        model.level1 = level1
        level1.level2 = level2
        level2.level3 = level3
        level3.child = child

        parent, attr = _get_parent_and_attr(model, "level1.level2.level3.child")

        assert parent is level3
        assert attr == "child"


class MockQuantizableModel(nn.Module):
    """Mock model for testing quantization."""

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 20)
        self.linear2 = nn.Linear(20, 5)
        self.to_skip = nn.Linear(5, 3)  # This should be skipped
        self.norm = nn.LayerNorm(3)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.to_skip(x)
        x = self.norm(x)
        return x


class TestConvertLinearToInt4Manual:
    """Test manual INT4 conversion functionality."""

    @patch("tracknet.models.lora.quantization.bnb")
    def test_convert_success(self, mock_bnb):
        """Test successful INT4 conversion."""
        # Setup mock bitsandbytes
        mock_linear4bit = Mock()
        mock_bnb.nn.Linear4bit = mock_linear4bit

        model = MockQuantizableModel()

        # Convert to INT4
        convert_linear_to_int4_manual(
            model,
            skip_modules=["to_skip"],
            quant_type="nf4",
            compute_dtype=torch.bfloat16,
            compress_statistics=True,
        )

        # Should have created Linear4bit layers
        assert mock_linear4bit.call_count == 2  # linear1 and linear2, not to_skip

        # Verify the calls
        calls = mock_linear4bit.call_args_list
        for call in calls:
            kwargs = call[1]
            assert kwargs["compute_dtype"] == torch.bfloat16
            assert kwargs["quant_type"] == "nf4"
            assert kwargs["compress_statistics"] is True

    @patch("tracknet.models.lora.quantization.bnb")
    def test_convert_with_skip_modules(self, mock_bnb):
        """Test INT4 conversion with skip modules."""
        mock_linear4bit = Mock()
        mock_bnb.nn.Linear4bit = mock_linear4bit

        model = MockQuantizableModel()

        # Convert with skip modules
        convert_linear_to_int4_manual(
            model,
            skip_modules=["linear1", "to_skip"],
        )

        # Should only convert linear2
        assert mock_linear4bit.call_count == 1

    def test_convert_missing_bitsandbytes(self):
        """Test conversion when bitsandbytes is not available."""
        with patch("tracknet.models.lora.quantization.BNB_AVAILABLE", False):
            model = MockQuantizableModel()

            with pytest.raises(ImportError, match="bitsandbytes is required"):
                convert_linear_to_int4_manual(model)

    @patch("tracknet.models.lora.quantization.bnb")
    def test_convert_with_exception(self, mock_bnb):
        """Test conversion with exception during layer replacement."""
        mock_linear4bit = Mock()
        mock_linear4bit.side_effect = [Mock(), RuntimeError("Test error")]
        mock_bnb.nn.Linear4bit = mock_linear4bit

        model = MockQuantizableModel()

        with pytest.raises(RuntimeError, match="Failed to quantize module"):
            convert_linear_to_int4_manual(model)


class TestApplyHFQuantization:
    """Test HuggingFace quantization functionality."""

    @patch("tracknet.models.lora.quantization.BitsAndBytesConfig")
    def test_apply_hf_success(self, mock_bnb_config):
        """Test successful HF quantization."""
        model = Mock()

        apply_hf_quantization(
            model,
            model_name="test/model",
            quant_type="nf4",
            compute_dtype=torch.bfloat16,
            use_double_quant=True,
        )

        # Should create BitsAndBytesConfig
        mock_bnb_config.assert_called_once_with(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    def test_apply_hf_missing_transformers(self):
        """Test HF quantization when transformers is not available."""
        with patch("tracknet.models.lora.quantization.HF_AVAILABLE", False):
            model = Mock()

            with pytest.raises(ImportError, match="transformers is required"):
                apply_hf_quantization(model, "test/model")

    def test_apply_hf_missing_bitsandbytes(self):
        """Test HF quantization when bitsandbytes is not available."""
        with patch("tracknet.models.lora.quantization.BNB_AVAILABLE", False):
            model = Mock()

            with pytest.raises(ImportError, match="bitsandbytes is required"):
                apply_hf_quantization(model, "test/model")


class TestApplyQuantization:
    """Test main quantization application function."""

    def test_apply_disabled(self):
        """Test applying quantization when disabled."""
        config = QuantizationConfig(enabled=False)
        model = Mock()

        result = apply_quantization(model, config)

        assert result is model  # Should return unchanged model

    @patch("tracknet.models.lora.quantization.convert_linear_to_int4_manual")
    def test_apply_manual_mode(self, mock_convert):
        """Test applying quantization in manual mode."""
        config = QuantizationConfig(
            enabled=True,
            mode="manual",
            quant_type="nf4",
            compute_dtype=torch.bfloat16,
        )
        model = Mock()
        mock_quantized = Mock()
        mock_convert.return_value = mock_quantized

        result = apply_quantization(model, config)

        mock_convert.assert_called_once_with(
            model,
            skip_modules=[],
            quant_type="nf4",
            compute_dtype=torch.bfloat16,
            compress_statistics=True,
        )
        assert result is mock_quantized

    @patch("tracknet.models.lora.quantization.apply_hf_quantization")
    def test_apply_hf_mode(self, mock_apply_hf):
        """Test applying quantization in HF mode."""
        config = QuantizationConfig(
            enabled=True,
            mode="hf",
            quant_type="nf4",
            compute_dtype=torch.bfloat16,
        )
        model = Mock()
        mock_quantized = Mock()
        mock_apply_hf.return_value = mock_quantized

        result = apply_quantization(model, config, model_name="test/model")

        mock_apply_hf.assert_called_once_with(
            model,
            model_name="test/model",
            quant_type="nf4",
            compute_dtype=torch.bfloat16,
            use_double_quant=True,
        )
        assert result is mock_quantized

    def test_apply_hf_mode_no_model_name(self):
        """Test HF quantization without model name."""
        config = QuantizationConfig(enabled=True, mode="hf")
        model = Mock()

        with pytest.raises(
            ValueError, match="model_name is required for HF quantization mode"
        ):
            apply_quantization(model, config)

    def test_apply_invalid_mode(self):
        """Test applying quantization with invalid mode."""
        config = QuantizationConfig(enabled=True, mode="invalid")
        model = Mock()

        with pytest.raises(ValueError, match="Unsupported quantization mode"):
            apply_quantization(model, config)


class TestValidateQuantizationCompatibility:
    """Test quantization compatibility validation."""

    def test_valid_model(self):
        """Test validation with valid model."""
        model = MockQuantizableModel()

        result = validate_quantization_compatibility(model)

        assert result is True

    def test_invalid_dimensions(self):
        """Test validation with invalid dimensions."""
        model = Mock()
        model.named_modules.return_value = [
            ("invalid_linear", Mock(spec=nn.Linear, in_features=-1, out_features=10)),
        ]

        result = validate_quantization_compatibility(model)

        assert result is False

    def test_validation_with_skip_modules(self):
        """Test validation with skip modules."""
        model = Mock()
        model.named_modules.return_value = [
            ("invalid_linear", Mock(spec=nn.Linear, in_features=-1, out_features=10)),
            ("valid_linear", Mock(spec=nn.Linear, in_features=10, out_features=5)),
        ]

        # Skip the invalid module
        result = validate_quantization_compatibility(
            model, skip_modules=["invalid_linear"]
        )

        assert result is True


class TestGetQuantizationMemoryInfo:
    """Test memory information calculation."""

    def test_memory_info_calculation(self):
        """Test memory information calculation."""
        model = MockQuantizableModel()

        # Freeze some parameters
        for param in model.linear1.parameters():
            param.requires_grad = False

        info = get_quantization_memory_info(model)

        # Check required fields
        assert "total_parameters" in info
        assert "trainable_parameters" in info
        assert "fp32_memory_mb" in info
        assert "int4_memory_mb" in info
        assert "memory_reduction_percent" in info

        # Check values make sense
        assert info["total_parameters"] > 0
        assert info["trainable_parameters"] < info["total_parameters"]
        assert info["fp32_memory_mb"] > info["int4_memory_mb"]
        assert info["memory_reduction_percent"] > 0

    def test_memory_info_empty_model(self):
        """Test memory info with empty model."""
        model = nn.Sequential()

        info = get_quantization_memory_info(model)

        assert info["total_parameters"] == 0
        assert info["trainable_parameters"] == 0
        assert info["fp32_memory_mb"] == 0
        assert info["int4_memory_mb"] == 0


if __name__ == "__main__":
    pytest.main([__file__])
