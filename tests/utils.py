"""Test utilities and helper functions."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np
import torch
from PIL import Image


def create_mock_video_frame(width=640, height=480):
    """Create a mock video frame tensor."""
    return torch.randn(1, 3, height, width)


def create_test_image(path: Path, size=(100, 100)):
    """Create a test image file at the given path."""
    img_array = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    img.save(path)
    return path


def create_mock_dataloader(batch_size=2, num_batches=3):
    """Create a mock DataLoader for testing."""
    mock_dataloader = Mock()
    mock_dataloader.batch_size = batch_size
    
    # Create mock batches
    batches = []
    for _ in range(num_batches):
        batch = {
            "images": torch.randn(batch_size, 3, 224, 224),
            "targets": torch.randint(0, 10, (batch_size,))
        }
        batches.append(batch)
    
    mock_dataloader.__iter__ = Mock(return_value=iter(batches))
    mock_dataloader.__len__ = Mock(return_value=num_batches)
    
    return mock_dataloader


def create_mock_model(num_classes=10):
    """Create a mock model for testing."""
    mock_model = Mock()
    mock_model.num_classes = num_classes
    mock_model.training = False
    
    # Mock forward pass
    mock_model.forward = Mock(return_value=torch.randn(2, num_classes))
    mock_model.train = Mock()
    mock_model.eval = Mock()
    
    return mock_model


def skip_if_no_gpu():
    """Skip test if GPU is not available."""
    return pytest.skip("GPU not available")


def create_temp_config(config_dict):
    """Create a temporary config file from a dictionary."""
    import yaml
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_dict, f)
        return Path(f.name)


def assert_tensors_close(tensor1, tensor2, rtol=1e-5, atol=1e-8):
    """Assert that two tensors are close within tolerance."""
    assert torch.allclose(tensor1, tensor2, rtol=rtol, atol=atol)


def mock_torch_functions():
    """Context manager to mock expensive torch functions."""
    with patch('torch.load') as mock_load, \
         patch('torch.save') as mock_save, \
         patch('torch.cuda.is_available', return_value=False):
        mock_load.return_value = {"state_dict": {}}
        yield mock_load, mock_save


class MockDataset:
    """Mock dataset for testing."""
    
    def __init__(self, size=10, num_classes=10):
        self.size = size
        self.num_classes = num_classes
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return {
            "image": torch.randn(3, 224, 224),
            "label": torch.randint(0, self.num_classes, (1,)).item(),
            "index": idx
        }
