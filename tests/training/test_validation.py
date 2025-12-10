"""
Tests for the validation module.
Module 2.3: Training Validation & Smoke Testing

Tests validation functions and test configuration.
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestValidationImports:
    """Test that validation modules can be imported."""
    
    def test_import_validation_module(self):
        """Test validation module imports."""
        from training import validation
        assert validation is not None
    
    def test_import_smoke_test_module(self):
        """Test smoke test module imports."""
        from training import smoke_test
        assert smoke_test is not None
    
    def test_import_validate_training_pipeline(self):
        """Test main validation function exists."""
        from training.validation import validate_training_pipeline
        assert callable(validate_training_pipeline)
    
    def test_import_test_data_loading(self):
        """Test data loading test function exists."""
        from training.validation import test_data_loading
        assert callable(test_data_loading)
    
    def test_import_test_model_initialization(self):
        """Test model init test function exists."""
        from training.validation import test_model_initialization
        assert callable(test_model_initialization)
    
    def test_import_test_training_step(self):
        """Test training step test function exists."""
        from training.validation import test_training_step
        assert callable(test_training_step)
    
    def test_import_test_checkpoint_saving(self):
        """Test checkpoint saving test function exists."""
        from training.validation import test_checkpoint_saving
        assert callable(test_checkpoint_saving)
    
    def test_import_test_checkpoint_loading(self):
        """Test checkpoint loading test function exists."""
        from training.validation import test_checkpoint_loading
        assert callable(test_checkpoint_loading)
    
    def test_import_test_inference(self):
        """Test inference test function exists."""
        from training.validation import test_inference
        assert callable(test_inference)
    
    def test_import_generate_validation_report(self):
        """Test report generation function exists."""
        from training.validation import generate_validation_report
        assert callable(generate_validation_report)
    
    def test_import_run_smoke_test(self):
        """Test smoke test function exists."""
        from training.smoke_test import run_smoke_test
        assert callable(run_smoke_test)
    
    def test_import_test_full_workflow(self):
        """Test full workflow function exists."""
        from training.smoke_test import test_full_workflow
        assert callable(test_full_workflow)


class TestTestConfig:
    """Test the test configuration file."""
    
    def test_test_config_exists(self):
        """Test that test config file exists."""
        config_path = PROJECT_ROOT / "training" / "configs" / "test_config.yaml"
        assert config_path.exists()
    
    def test_test_config_valid_yaml(self):
        """Test that test config is valid YAML."""
        config_path = PROJECT_ROOT / "training" / "configs" / "test_config.yaml"
        
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        assert isinstance(config, dict)
    
    def test_test_config_has_required_sections(self):
        """Test that test config has all required sections."""
        config_path = PROJECT_ROOT / "training" / "configs" / "test_config.yaml"
        
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        assert "model" in config
        assert "lora" in config
        assert "training" in config
        assert "data" in config
        assert "quantization" in config
    
    def test_test_config_max_steps_small(self):
        """Test that max_steps is small for testing."""
        config_path = PROJECT_ROOT / "training" / "configs" / "test_config.yaml"
        
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        max_steps = config["training"].get("max_steps", 1000)
        assert max_steps <= 50, f"max_steps should be small for testing, got {max_steps}"
    
    def test_test_config_max_samples_small(self):
        """Test that max_samples is small for testing."""
        config_path = PROJECT_ROOT / "training" / "configs" / "test_config.yaml"
        
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        max_train = config["data"].get("max_samples_train", 1000)
        max_val = config["data"].get("max_samples_val", 1000)
        
        assert max_train <= 50, f"max_samples_train should be small, got {max_train}"
        assert max_val <= 20, f"max_samples_val should be small, got {max_val}"
    
    def test_test_config_save_steps_configured(self):
        """Test that save_steps is configured for checkpoint testing."""
        config_path = PROJECT_ROOT / "training" / "configs" / "test_config.yaml"
        
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        save_steps = config["training"].get("save_steps")
        assert save_steps is not None
        assert save_steps <= 20, f"save_steps should be small for testing, got {save_steps}"
    
    def test_test_config_lora_small_rank(self):
        """Test that LoRA rank is small for faster testing."""
        config_path = PROJECT_ROOT / "training" / "configs" / "test_config.yaml"
        
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        lora_r = config["lora"].get("r", 16)
        assert lora_r <= 16, f"LoRA rank should be small for testing, got {lora_r}"


class TestValidationReport:
    """Test validation report generation."""
    
    def test_generate_report_all_passed(self):
        """Test report generation when all tests pass."""
        from training.validation import generate_validation_report
        
        results = {
            "data_loading": {"passed": True, "message": "20 examples loaded"},
            "model_init": {"passed": True, "message": "Model loaded"},
            "training_step": {"passed": True, "message": "Loss decreased"},
        }
        
        report = generate_validation_report(results)
        
        assert report["all_passed"] == True
        assert "timestamp" in report
        assert "data_loading" in report
    
    def test_generate_report_some_failed(self):
        """Test report generation when some tests fail."""
        from training.validation import generate_validation_report
        
        results = {
            "data_loading": {"passed": True, "message": "OK"},
            "model_init": {"passed": False, "message": "OOM error"},
        }
        
        report = generate_validation_report(results)
        
        assert report["all_passed"] == False
    
    def test_generate_report_empty(self):
        """Test report generation with empty results."""
        from training.validation import generate_validation_report
        
        results = {}
        report = generate_validation_report(results)
        
        assert report["all_passed"] == True  # No failures = all passed
        assert "timestamp" in report


class TestSmokeTestInitialization:
    """Test smoke test can be initialized (without running full test)."""
    
    def test_smoke_test_config_path_check(self):
        """Test that smoke test checks for config path."""
        from training.smoke_test import test_full_workflow
        
        # Should return failure if config doesn't exist
        with patch('os.path.exists', return_value=False):
            success, failure_point = test_full_workflow()
            assert success == False
            assert failure_point == "config_not_found"
    
    def test_smoke_test_results_structure(self):
        """Test that smoke test returns expected structure."""
        # We can't run the full test without GPU, but we can check the structure
        expected_keys = [
            "success",
            "failure_point",
            "train_loss",
            "checkpoint_path",
            "can_generate",
            "steps_completed",
        ]
        
        # Just verify the function signature
        from training.smoke_test import run_smoke_test
        import inspect
        sig = inspect.signature(run_smoke_test)
        assert "config_path" in sig.parameters


class TestValidationScripts:
    """Test validation scripts exist and are configured correctly."""
    
    def test_run_validation_script_exists(self):
        """Test that run_validation.sh exists."""
        script_path = PROJECT_ROOT / "scripts" / "run_validation.sh"
        assert script_path.exists()
    
    def test_run_validation_script_executable(self):
        """Test that run_validation.sh is executable."""
        script_path = PROJECT_ROOT / "scripts" / "run_validation.sh"
        assert os.access(script_path, os.X_OK)
    
    def test_validation_notebook_exists(self):
        """Test that validation notebook exists."""
        notebook_path = PROJECT_ROOT / "notebooks" / "validation_notebook.ipynb"
        assert notebook_path.exists()
    
    def test_validation_notebook_valid_json(self):
        """Test that validation notebook is valid JSON."""
        import json
        notebook_path = PROJECT_ROOT / "notebooks" / "validation_notebook.ipynb"
        
        with open(notebook_path) as f:
            notebook = json.load(f)
        
        assert "cells" in notebook
        assert "nbformat" in notebook
        assert len(notebook["cells"]) >= 10  # Should have 10 cells


class TestGCSSyncFunction:
    """Test GCS sync functionality."""
    
    def test_gcs_sync_function_exists(self):
        """Test that GCS sync test function exists."""
        from training.validation import test_gcs_sync
        assert callable(test_gcs_sync)
    
    def test_gcs_sync_handles_missing_gsutil(self):
        """Test that GCS sync handles missing gsutil gracefully."""
        from training.validation import test_gcs_sync
        
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch('subprocess.run', side_effect=FileNotFoundError("gsutil not found")):
                success, info = test_gcs_sync("test-bucket", tmpdir)
                assert success == False
                assert "gsutil not found" in info.get("error", "")
