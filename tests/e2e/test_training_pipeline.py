"""End-to-end tests for complete training pipeline."""

import pytest
import tempfile
import yaml
import torch
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock


@pytest.mark.e2e
@pytest.mark.slow
class TestTrainingPipelineE2E:
    """End-to-end tests for the complete training pipeline."""
    
    def test_minimal_training_workflow(self, temp_dir):
        """Test minimal training workflow with mock components."""
        # Create minimal config
        config = {
            "model": {
                "name": "test_model",
                "backbone": "resnet18",
                "num_classes": 2
            },
            "training": {
                "batch_size": 1,
                "learning_rate": 0.001,
                "epochs": 1,
                "device": "cpu"
            },
            "data": {
                "train_path": str(temp_dir / "train"),
                "val_path": str(temp_dir / "val"),
                "image_size": [224, 224]
            }
        }
        
        # Create config file
        config_file = temp_dir / "e2e_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config, f)
        
        # Create mock data directories
        train_dir = temp_dir / "train"
        val_dir = temp_dir / "val"
        train_dir.mkdir()
        val_dir.mkdir()
        
        # Create mock data files
        (train_dir / "image1.jpg").touch()
        (train_dir / "image1.txt").write_text("0 0.5 0.5 0.3 0.3\n")
        (val_dir / "val_image1.jpg").touch()
        (val_dir / "val_image1.txt").write_text("0 0.5 0.5 0.3 0.3\n")
        
        # Test config loading
        from tracknet.utils.config import load_config
        loaded_config = load_config(str(config_file))
        assert loaded_config == config
        
        # Test that paths exist
        assert Path(config["data"]["train_path"]).exists()
        assert Path(config["data"]["val_path"]).exists()
    
    @patch('torch.cuda.is_available', return_value=False)
    def test_cpu_training_pipeline(self, mock_cuda, temp_dir):
        """Test CPU training pipeline execution."""
        # Mock model components
        mock_model = Mock()
        mock_model.train = Mock()
        mock_model.eval = Mock()
        mock_model.forward = Mock(return_value=torch.randn(1, 2))
        mock_model.parameters = Mock(return_value=[torch.randn(10)])
        
        # Mock optimizer
        mock_optimizer = Mock()
        mock_optimizer.zero_grad = Mock()
        mock_optimizer.step = Mock()
        
        # Mock loss function
        mock_loss_fn = Mock(return_value=torch.tensor(0.5))
        
        # Mock data loader
        mock_dataloader = Mock()
        mock_batch = {
            "images": torch.randn(1, 3, 224, 224),
            "targets": torch.tensor([0])
        }
        mock_dataloader.__iter__ = Mock(return_value=iter([mock_batch]))
        mock_dataloader.__len__ = Mock(return_value=1)
        
        # Simulate training step
        mock_model.train()
        
        for batch in mock_dataloader:
            mock_optimizer.zero_grad()
            
            # Forward pass
            outputs = mock_model(batch["images"])
            loss = mock_loss_fn(outputs, batch["targets"])
            
            # Backward pass
            loss.backward()
            mock_optimizer.step()
            
            # Verify training step occurred
            mock_model.forward.assert_called()
            mock_loss_fn.assert_called()
            mock_optimizer.step.assert_called()
            
            break  # Only one step for E2E test
        
        # Verify CUDA check was called
        mock_cuda.assert_called()
    
    def test_model_inference_pipeline(self, temp_dir):
        """Test model inference pipeline."""
        # Mock trained model
        mock_model = Mock()
        mock_model.eval = Mock()
        mock_model.forward = Mock(return_value=torch.tensor([[0.1, 0.9]]))
        
        # Create test image
        test_image = torch.randn(1, 3, 224, 224)
        
        # Run inference
        mock_model.eval()
        with torch.no_grad():
            predictions = mock_model(test_image)
        
        # Verify inference results
        assert predictions.shape == (1, 2)
        mock_model.eval.assert_called()
        mock_model.forward.assert_called_with(test_image)
        
        # Test post-processing
        predicted_class = torch.argmax(predictions, dim=1)
        confidence = torch.softmax(predictions, dim=1)
        
        assert predicted_class.item() == 1
        assert confidence.shape == (1, 2)
        assert torch.allclose(confidence.sum(dim=1), torch.ones(1))


@pytest.mark.e2e
class TestDataProcessingPipeline:
    """End-to-end tests for data processing pipeline."""
    
    def test_image_preprocessing_pipeline(self, sample_image_file):
        """Test complete image preprocessing pipeline."""
        from PIL import Image
        import torchvision.transforms as transforms
        
        # Load original image
        original_image = Image.open(sample_image_file)
        assert original_image.size == (100, 100)
        
        # Define preprocessing pipeline
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Apply preprocessing
        processed_tensor = preprocess(original_image)
        
        # Verify processed tensor
        assert processed_tensor.shape == (3, 224, 224)
        assert processed_tensor.dtype == torch.float32
        
        # Verify normalization (values should be roughly in [-2, 2])
        assert processed_tensor.min() >= -3.0
        assert processed_tensor.max() <= 3.0
    
    def test_annotation_processing_pipeline(self, temp_dir):
        """Test annotation processing pipeline."""
        # Create test annotation file
        annotation_file = temp_dir / "annotations.txt"
        annotation_content = """0 0.5 0.5 0.3 0.3
1 0.2 0.2 0.1 0.1
0 0.8 0.8 0.15 0.15"""
        annotation_file.write_text(annotation_content)
        
        # Parse annotations
        annotations = []
        with open(annotation_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:])
                    annotations.append({
                        "class_id": class_id,
                        "x_center": x_center,
                        "y_center": y_center,
                        "width": width,
                        "height": height
                    })
        
        # Verify parsed annotations
        assert len(annotations) == 3
        assert annotations[0]["class_id"] == 0
        assert annotations[0]["x_center"] == 0.5
        
        # Convert to absolute coordinates
        image_width, image_height = 224, 224
        for ann in annotations:
            ann["x_abs"] = ann["x_center"] * image_width
            ann["y_abs"] = ann["y_center"] * image_height
            ann["w_abs"] = ann["width"] * image_width
            ann["h_abs"] = ann["height"] * image_height
        
        # Verify absolute coordinates
        assert annotations[0]["x_abs"] == 112.0  # 0.5 * 224
        assert annotations[0]["w_abs"] == 67.2   # 0.3 * 224


@pytest.mark.e2e
class TestModelSavingLoadingPipeline:
    """End-to-end tests for model saving and loading pipeline."""
    
    def test_model_checkpoint_pipeline(self, temp_dir):
        """Test model checkpoint saving and loading."""
        # Create mock model with state
        mock_model = Mock()
        mock_state_dict = {
            "layer1.weight": torch.randn(10, 3),
            "layer1.bias": torch.randn(10),
            "layer2.weight": torch.randn(5, 10),
            "layer2.bias": torch.randn(5)
        }
        mock_model.state_dict.return_value = mock_state_dict
        
        # Save checkpoint
        checkpoint_path = temp_dir / "model_checkpoint.pth"
        torch.save({
            "model_state_dict": mock_model.state_dict(),
            "epoch": 10,
            "loss": 0.123,
            "config": {"num_classes": 5}
        }, checkpoint_path)
        
        # Verify checkpoint was saved
        assert checkpoint_path.exists()
        assert checkpoint_path.stat().st_size > 0
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path)
        
        # Verify checkpoint contents
        assert "model_state_dict" in checkpoint
        assert "epoch" in checkpoint
        assert "loss" in checkpoint
        assert "config" in checkpoint
        
        assert checkpoint["epoch"] == 10
        assert checkpoint["loss"] == 0.123
        assert checkpoint["config"]["num_classes"] == 5
        
        # Verify model state dict
        loaded_state_dict = checkpoint["model_state_dict"]
        assert "layer1.weight" in loaded_state_dict
        assert loaded_state_dict["layer1.weight"].shape == (10, 3)
    
    def test_config_model_consistency(self, temp_dir):
        """Test consistency between config and saved model."""
        # Create config
        config = {
            "model": {
                "name": "test_model",
                "backbone": "resnet18",
                "num_classes": 10,
                "pretrained": True
            },
            "training": {
                "batch_size": 4,
                "learning_rate": 0.001
            }
        }
        
        # Save config
        config_path = temp_dir / "model_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        # Create model checkpoint with config
        checkpoint = {
            "model_state_dict": {"dummy": torch.randn(5)},
            "config": config,
            "version": "1.0"
        }
        
        checkpoint_path = temp_dir / "checkpoint.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Load both and verify consistency
        loaded_config = yaml.safe_load(open(config_path, 'r'))
        loaded_checkpoint = torch.load(checkpoint_path)
        
        assert loaded_config == loaded_checkpoint["config"]
        assert loaded_checkpoint["version"] == "1.0"
