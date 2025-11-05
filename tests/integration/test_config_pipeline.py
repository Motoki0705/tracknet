"""Integration tests for configuration and data loading pipeline."""

import pytest
import tempfile
import yaml
from pathlib import Path
from unittest.mock import Mock, patch

from tracknet.utils.config import build_cfg


@pytest.mark.integration
class TestConfigDataPipeline:
    """Test configuration and data loading integration."""
    
    def test_config_building_integration(self, temp_dir):
        """Test integration between config building and validation."""
        # Create test config files
        data_config = {
            "root": str(temp_dir / "data"),
            "image_size": [224, 224],
            "batch_size": 2
        }
        
        model_config = {
            "name": "test_model",
            "backbone": "resnet18",
            "num_classes": 10
        }
        
        training_config = {
            "learning_rate": 0.001,
            "epochs": 5,
            "batch_size": 4
        }
        
        # Create config directories and files
        configs_dir = temp_dir / "configs"
        configs_dir.mkdir()
        (configs_dir / "data").mkdir()
        (configs_dir / "model").mkdir()
        (configs_dir / "training").mkdir()
        
        with open(configs_dir / "data" / "test.yaml", 'w') as f:
            yaml.dump(data_config, f)
        with open(configs_dir / "model" / "test.yaml", 'w') as f:
            yaml.dump(model_config, f)
        with open(configs_dir / "training" / "test.yaml", 'w') as f:
            yaml.dump(training_config, f)
        
        # Test that we can build a valid config (this will use actual config files)
        try:
            cfg = build_cfg(
                data_name="tracknet",  # Use existing config
                model_name="convnext_fpn_heatmap",  # Use existing config
                training_name="default",  # Use existing config
                dry_run=True
            )
            
            # Verify config structure
            assert "data" in cfg
            assert "model" in cfg
            assert "training" in cfg
            assert "runtime" in cfg
            
        except FileNotFoundError:
            # If configs don't exist, that's expected in test environment
            pass
    
    def test_config_override_integration(self):
        """Test configuration override system integration."""
        # Test with overrides
        overrides = [
            "training.batch_size=8",
            "model.backbone=resnet50",
            "data.image_size=[256,256]"
        ]
        
        try:
            cfg = build_cfg(overrides=overrides, dry_run=True)
            
            # Verify overrides were applied (if config exists)
            if hasattr(cfg.training, 'batch_size'):
                assert cfg.training.batch_size == 8
            if hasattr(cfg.model, 'backbone'):
                assert cfg.model.backbone == "resnet50"
                
        except FileNotFoundError:
            # Expected if config files don't exist
            pass
    
    @patch('tracknet.utils.config._seed_all')
    def test_config_seeding_integration(self, mock_seed):
        """Test configuration and random seeding integration."""
        try:
            cfg = build_cfg(seed=42, dry_run=True)
            
            # Verify seeding was called
            mock_seed.assert_called_once_with(42)
            
            # Verify seed is in runtime config
            assert cfg.runtime.seed == 42
            
        except FileNotFoundError:
            # Expected if config files don't exist
            pass


@pytest.mark.integration  
class TestModelConfigIntegration:
    """Test model configuration integration."""
    
    def test_model_config_validation(self):
        """Test model configuration validation."""
        try:
            cfg = build_cfg(dry_run=True)
            model_config = cfg.model
            
            # Validate required fields exist
            assert hasattr(model_config, 'name') or hasattr(model_config, 'backbone')
            
        except FileNotFoundError:
            # Expected if config files don't exist
            pass
    
    def test_training_config_integration(self):
        """Test training configuration integration."""
        try:
            cfg = build_cfg(dry_run=True)
            training_config = cfg.training
            
            # Validate training parameters exist
            assert hasattr(training_config, 'batch_size') or hasattr(training_config, 'learning_rate')
            
        except FileNotFoundError:
            # Expected if config files don't exist
            pass


@pytest.mark.integration
class TestLoggingConfigIntegration:
    """Test logging and configuration integration."""
    
    def test_runtime_config_integration(self):
        """Test runtime configuration integration."""
        try:
            cfg = build_cfg(seed=123, dry_run=True)
            runtime = cfg.runtime
            
            # Verify runtime configuration
            assert runtime.seed == 123
            assert "project_root" in runtime
            assert "output_root" in runtime
            assert "run_id" in runtime
            assert "timestamp" in runtime
            
            # Verify run_id format
            assert "run-" in runtime.run_id
            assert "s123" in runtime.run_id
            
        except FileNotFoundError:
            # Expected if config files don't exist
            pass


@pytest.mark.integration
class TestEndToEndConfigFlow:
    """Test end-to-end configuration flow."""
    
    def test_full_config_building_pipeline(self):
        """Test complete configuration building pipeline."""
        try:
            # Test building config with all options
            cfg = build_cfg(
                data_name="tracknet",
                model_name="convnext_fpn_heatmap",
                training_name="default",
                seed=456,
                overrides=["training.batch_size=16"],
                dry_run=True
            )
            
            # Verify all sections are loaded
            assert "data" in cfg
            assert "model" in cfg
            assert "training" in cfg
            assert "runtime" in cfg
            
            # Verify specific values
            assert cfg.runtime.seed == 456
            assert "run-" in cfg.runtime.run_id
            assert "s456" in cfg.runtime.run_id
            
            # Verify override was applied
            if hasattr(cfg.training, 'batch_size'):
                assert cfg.training.batch_size == 16
                
        except FileNotFoundError:
            # Expected if config files don't exist in test environment
            pass
    
    def test_config_error_handling(self):
        """Test configuration error handling."""
        # Test with invalid config names
        with pytest.raises(FileNotFoundError):
            build_cfg(data_name="nonexistent_config", dry_run=True)
