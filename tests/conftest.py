"""Pytest configuration and fixtures for tracknet testing."""

import os
import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def mock_config():
    """Create a mock configuration for testing."""
    return {
        "model": {
            "name": "test_model",
            "backbone": "resnet18",
            "num_classes": 10,
        },
        "training": {
            "batch_size": 2,
            "learning_rate": 0.001,
            "epochs": 1,
        },
        "data": {
            "train_path": "/tmp/train",
            "val_path": "/tmp/val",
        },
    }


@pytest.fixture
def mock_tensor():
    """Create a mock tensor for testing."""
    return torch.randn(1, 3, 224, 224)


@pytest.fixture
def sample_image_file(temp_dir):
    """Create a sample image file for testing."""
    import numpy as np
    from PIL import Image
    
    # Create a simple test image
    img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    img_path = temp_dir / "test_image.jpg"
    img.save(img_path)
    return img_path


@pytest.fixture
def mock_yolo_model():
    """Create a mock YOLO model for testing."""
    mock_model = Mock()
    mock_model.predict.return_value = [
        Mock(
            boxes=Mock(xyxy=torch.tensor([[10, 10, 50, 50]])),
            conf=torch.tensor([0.9]),
            cls=torch.tensor([0])
        )
    ]
    return mock_model


@pytest.fixture(scope="session")
def test_data_dir():
    """Get the test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up environment variables for testing."""
    # Set environment variables for testing
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    yield
    # Clean up after test
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
