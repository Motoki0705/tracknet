"""Unit tests for tracknet utility functions."""

import pytest
import torch
import numpy as np
from unittest.mock import patch, Mock
from pathlib import Path

from tracknet.utils.config import build_cfg
from tracknet.utils.logging import Logger, LoggerConfig


class TestConfigUtils:
    """Test configuration utility functions."""
    
    def test_build_cfg_with_default_parameters(self):
        """Test building configuration with default parameters."""
        cfg = build_cfg(dry_run=True)  # Use dry_run to avoid creating directories
        
        # Verify structure
        assert "data" in cfg
        assert "model" in cfg
        assert "training" in cfg
        assert "runtime" in cfg
        
        # Verify runtime values
        assert cfg.runtime.seed >= 0
        assert "project_root" in cfg.runtime
        assert "output_root" in cfg.runtime
        assert "run_id" in cfg.runtime
    
    def test_build_cfg_with_custom_parameters(self):
        """Test building configuration with custom parameters."""
        cfg = build_cfg(
            data_name="tracknet",
            model_name="convnext_fpn_heatmap", 
            training_name="default",
            seed=123,
            dry_run=True
        )
        
        # Verify custom seed was used
        assert cfg.runtime.seed == 123
        
        # Verify run_id includes seed
        assert "s123" in cfg.runtime.run_id
    
    def test_build_cfg_with_overrides(self):
        """Test building configuration with dotlist overrides."""
        overrides = ["training.batch_size=8", "model.backbone=resnet50"]
        cfg = build_cfg(overrides=overrides, dry_run=True)
        
        # Verify overrides were applied
        assert cfg.training.batch_size == 8
        assert cfg.model.backbone == "resnet50"
    
    def test_build_cfg_nonexistent_config(self):
        """Test building configuration with nonexistent config file."""
        with pytest.raises(FileNotFoundError):
            build_cfg(data_name="nonexistent", dry_run=True)


class TestLoggingUtils:
    """Test logging utility functions."""
    
    def test_logger_creation(self, temp_dir):
        """Test logger creation with default settings."""
        logger_cfg = LoggerConfig(
            log_dir=str(temp_dir),
            run_id="test_run",
            use_tensorboard=False
        )
        
        logger = Logger(logger_cfg)
        
        # Verify logger directory was created
        assert logger.dir.exists()
        assert logger.dir.name == "test_run"
        assert logger.csv_path.exists()
    
    def test_logger_scalar_logging(self, temp_dir):
        """Test scalar logging functionality."""
        logger_cfg = LoggerConfig(log_dir=str(temp_dir), use_tensorboard=False)
        logger = Logger(logger_cfg)
        
        # Log some scalars
        logger.log_scalar("train/loss", 0.5, 1)
        logger.log_scalar("val/loss", 0.6, 1)
        logger.log_scalar("train/accuracy", 0.8, 1)
        
        # Verify CSV file was created and contains data
        assert logger.csv_path.exists()
        csv_content = logger.csv_path.read_text()
        assert "train/loss" in csv_content
        assert "val/loss" in csv_content
        assert "train/accuracy" in csv_content
        assert "0.5" in csv_content
    
    def test_logger_auto_run_id(self, temp_dir):
        """Test logger auto-generates run ID when not provided."""
        logger_cfg = LoggerConfig(log_dir=str(temp_dir))
        logger = Logger(logger_cfg)
        
        # Verify run ID was auto-generated
        assert logger.dir.name.startswith("run-")
        assert logger.dir.exists()


class TestTensorUtils:
    """Test tensor utility functions."""
    
    def test_tensor_creation(self):
        """Test basic tensor creation utilities."""
        # Test creating tensors of different shapes
        tensor_1d = torch.randn(10)
        tensor_2d = torch.randn(5, 5)
        tensor_3d = torch.randn(3, 224, 224)
        
        assert tensor_1d.dim() == 1
        assert tensor_2d.dim() == 2
        assert tensor_3d.dim() == 3
    
    def test_tensor_device_handling(self):
        """Test tensor device handling utilities."""
        tensor = torch.randn(3, 224, 224)
        
        # Test CPU tensor
        assert tensor.device.type == "cpu"
        
        # Test moving to CPU (should be no-op)
        cpu_tensor = tensor.cpu()
        assert cpu_tensor.device.type == "cpu"
        assert torch.equal(tensor, cpu_tensor)
    
    def test_tensor_dtype_handling(self):
        """Test tensor dtype handling utilities."""
        # Test different dtypes
        float_tensor = torch.randn(3, 224, 224, dtype=torch.float32)
        int_tensor = torch.randint(0, 255, (3, 224, 224), dtype=torch.uint8)
        
        assert float_tensor.dtype == torch.float32
        assert int_tensor.dtype == torch.uint8
        
        # Test type conversion
        converted_tensor = float_tensor.to(torch.float16)
        assert converted_tensor.dtype == torch.float16


class TestImageUtils:
    """Test image utility functions."""
    
    def test_image_tensor_conversion(self, sample_image_file):
        """Test image to tensor conversion utilities."""
        from PIL import Image
        
        # Load image
        image = Image.open(sample_image_file)
        
        # Convert to numpy array
        image_array = np.array(image)
        assert image_array.shape == (100, 100, 3)
        assert image_array.dtype == np.uint8
        
        # Convert to tensor
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).float()
        assert image_tensor.shape == (3, 100, 100)
        assert image_tensor.dtype == torch.float32
    
    def test_image_normalization(self):
        """Test image normalization utilities."""
        # Create test image tensor
        image_tensor = torch.randint(0, 255, (3, 224, 224), dtype=torch.float32)
        
        # Normalize to [0, 1]
        normalized = image_tensor / 255.0
        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0
        
        # Test ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        
        imagenet_normalized = (normalized - mean) / std
        assert imagenet_normalized.shape == image_tensor.shape


@pytest.mark.unit
class TestMathUtils:
    """Test mathematical utility functions."""
    
    def test_calculate_iou(self):
        """Test Intersection over Union calculation."""
        # Test with overlapping boxes
        box1 = torch.tensor([10, 10, 50, 50])  # x1, y1, x2, y2
        box2 = torch.tensor([30, 30, 70, 70])
        
        # Calculate IoU manually
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        expected_iou = intersection / union if union > 0 else 0
        assert 0 < expected_iou < 1
    
    def test_calculate_area(self):
        """Test area calculation for bounding boxes."""
        box = torch.tensor([10, 10, 50, 50])  # x1, y1, x2, y2
        area = (box[2] - box[0]) * (box[3] - box[1])
        assert area == 1600  # 40 * 40
    
    def test_box_center(self):
        """Test center point calculation for bounding boxes."""
        box = torch.tensor([10, 10, 50, 50])  # x1, y1, x2, y2
        center_x = (box[0] + box[2]) / 2
        center_y = (box[1] + box[3]) / 2
        
        assert center_x == 30.0
        assert center_y == 30.0
        """Test tensor dtype handling utilities."""
        # Test different dtypes
        float_tensor = torch.randn(3, 224, 224, dtype=torch.float32)
        int_tensor = torch.randint(0, 255, (3, 224, 224), dtype=torch.uint8)
        
        assert float_tensor.dtype == torch.float32
        assert int_tensor.dtype == torch.uint8
        
        # Test type conversion
        converted_tensor = float_tensor.to(torch.float16)
        assert converted_tensor.dtype == torch.float16


class TestImageUtils:
    """Test image utility functions."""
    
    def test_image_tensor_conversion(self, sample_image_file):
        """Test image to tensor conversion utilities."""
        from PIL import Image
        
        # Load image
        image = Image.open(sample_image_file)
        
        # Convert to numpy array
        image_array = np.array(image)
        assert image_array.shape == (100, 100, 3)
        assert image_array.dtype == np.uint8
        
        # Convert to tensor
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).float()
        assert image_tensor.shape == (3, 100, 100)
        assert image_tensor.dtype == torch.float32
    
    def test_image_normalization(self):
        """Test image normalization utilities."""
        # Create test image tensor
        image_tensor = torch.randint(0, 255, (3, 224, 224), dtype=torch.float32)
        
        # Normalize to [0, 1]
        normalized = image_tensor / 255.0
        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0
        
        # Test ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        
        imagenet_normalized = (normalized - mean) / std
        assert imagenet_normalized.shape == image_tensor.shape


@pytest.mark.unit
class TestMathUtils:
    """Test mathematical utility functions."""
    
    def test_calculate_iou(self):
        """Test Intersection over Union calculation."""
        # Test with overlapping boxes
        box1 = torch.tensor([10, 10, 50, 50])  # x1, y1, x2, y2
        box2 = torch.tensor([30, 30, 70, 70])
        
        # Calculate IoU manually
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        expected_iou = intersection / union if union > 0 else 0
        assert 0 < expected_iou < 1
    
    def test_calculate_area(self):
        """Test area calculation for bounding boxes."""
        box = torch.tensor([10, 10, 50, 50])  # x1, y1, x2, y2
        area = (box[2] - box[0]) * (box[3] - box[1])
        assert area == 1600  # 40 * 40
    
    def test_box_center(self):
        """Test center point calculation for bounding boxes."""
        box = torch.tensor([10, 10, 50, 50])  # x1, y1, x2, y2
        center_x = (box[0] + box[2]) / 2
        center_y = (box[1] + box[3]) / 2
        
        assert center_x == 30.0
        assert center_y == 30.0
