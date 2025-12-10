#!/usr/bin/env python3
"""
Tests for training environment setup and validation.
"""

import pytest
import yaml
from pathlib import Path

from training.environment_setup import (
    check_python_version,
    check_conda_environment,
    check_package_version,
    check_gpu_specs,
    check_data_files,
    estimate_memory_requirements,
    load_training_config,
    validate_environment,
    REQUIRED_PACKAGES,
)

# Project paths
PROJECT_DIR = Path(__file__).parent.parent.parent
CONFIG_DIR = PROJECT_DIR / "training" / "configs"
DATA_DIR = PROJECT_DIR / "data" / "processed"


class TestPythonVersion:
    """Tests for Python version checking."""
    
    def test_check_python_version_returns_tuple(self):
        """Test that check_python_version returns a tuple."""
        result = check_python_version()
        assert isinstance(result, tuple)
        assert len(result) == 2
    
    def test_check_python_version_returns_bool_and_string(self):
        """Test return types."""
        is_valid, version_str = check_python_version()
        assert isinstance(is_valid, bool)
        assert isinstance(version_str, str)
    
    def test_python_version_format(self):
        """Test version string format."""
        _, version_str = check_python_version()
        parts = version_str.split(".")
        assert len(parts) >= 2


class TestCondaEnvironment:
    """Tests for conda environment checking."""
    
    def test_check_conda_returns_tuple(self):
        """Test that check_conda_environment returns a tuple."""
        result = check_conda_environment()
        assert isinstance(result, tuple)
        assert len(result) == 2
    
    def test_check_conda_returns_bool_and_string(self):
        """Test return types."""
        is_active, env_name = check_conda_environment()
        assert isinstance(is_active, bool)
        assert isinstance(env_name, str)


class TestPackageVersions:
    """Tests for package version checking."""
    
    def test_check_installed_package(self):
        """Test checking an installed package."""
        # pytest should be installed
        is_valid, version = check_package_version("pytest", "1.0.0")
        assert is_valid is True
        assert version != "Not installed"
    
    def test_check_missing_package(self):
        """Test checking a non-existent package."""
        is_valid, version = check_package_version("nonexistent-package-xyz", "1.0.0")
        assert is_valid is False
        assert version == "Not installed"
    
    def test_required_packages_defined(self):
        """Test that required packages are defined."""
        assert len(REQUIRED_PACKAGES) > 0
        assert "torch" in REQUIRED_PACKAGES
        assert "transformers" in REQUIRED_PACKAGES
        assert "peft" in REQUIRED_PACKAGES


class TestGPUSpecs:
    """Tests for GPU specification checking."""
    
    def test_check_gpu_specs_returns_dict(self):
        """Test that check_gpu_specs returns a dictionary."""
        specs = check_gpu_specs()
        assert isinstance(specs, dict)
    
    def test_gpu_specs_has_required_keys(self):
        """Test that GPU specs has required keys."""
        specs = check_gpu_specs()
        required_keys = ["cuda_available", "gpu_count", "gpu_name", "gpu_memory_gb"]
        for key in required_keys:
            assert key in specs
    
    def test_gpu_specs_types(self):
        """Test GPU specs value types."""
        specs = check_gpu_specs()
        assert isinstance(specs["cuda_available"], bool)
        assert isinstance(specs["gpu_count"], int)


class TestDataFiles:
    """Tests for data file checking."""
    
    def test_check_data_files_returns_dict(self):
        """Test that check_data_files returns a dictionary."""
        stats = check_data_files()
        assert isinstance(stats, dict)
    
    def test_data_files_has_required_keys(self):
        """Test that data stats has required keys."""
        stats = check_data_files()
        assert "all_present" in stats
        assert "files" in stats
    
    def test_data_files_checks_train_val_test(self):
        """Test that all required files are checked."""
        stats = check_data_files()
        assert "train_postgres.jsonl" in stats["files"]
        assert "val_postgres.jsonl" in stats["files"]
        assert "test_postgres.jsonl" in stats["files"]
    
    def test_data_files_exist(self):
        """Test that data files exist (if pipeline was run)."""
        stats = check_data_files()
        # At least check the structure is correct
        for filename, info in stats["files"].items():
            assert "exists" in info
            assert "size_mb" in info
            assert "examples" in info


class TestMemoryEstimation:
    """Tests for memory requirement estimation."""
    
    def test_estimate_memory_returns_dict(self):
        """Test that estimate_memory_requirements returns a dictionary."""
        estimates = estimate_memory_requirements()
        assert isinstance(estimates, dict)
    
    def test_memory_has_breakdown(self):
        """Test that memory estimates has breakdown."""
        estimates = estimate_memory_requirements()
        assert "breakdown" in estimates
        assert "total_estimated_gb" in estimates
    
    def test_memory_breakdown_components(self):
        """Test memory breakdown has expected components."""
        estimates = estimate_memory_requirements()
        breakdown = estimates["breakdown"]
        assert "base_model_4bit" in breakdown
        assert "lora_adapters" in breakdown
        assert "optimizer_states" in breakdown
    
    def test_memory_total_is_sum(self):
        """Test that total is approximately sum of breakdown."""
        estimates = estimate_memory_requirements()
        breakdown_sum = sum(estimates["breakdown"].values())
        assert abs(estimates["total_estimated_gb"] - breakdown_sum) < 0.5
    
    def test_memory_fits_in_a100(self):
        """Test memory estimation for A100."""
        estimates = estimate_memory_requirements()
        # With 4-bit quantization, should fit in A100 40GB
        assert estimates["fits_in_a100_40gb"] is True
    
    def test_memory_custom_batch_size(self):
        """Test memory estimation with custom batch size."""
        estimates_small = estimate_memory_requirements(batch_size=2)
        estimates_large = estimate_memory_requirements(batch_size=8)
        # Both should return valid estimates
        assert estimates_small["total_estimated_gb"] > 0
        assert estimates_large["total_estimated_gb"] > 0


class TestTrainingConfig:
    """Tests for training configuration loading."""
    
    def test_config_file_exists(self):
        """Test that training config file exists."""
        config_path = CONFIG_DIR / "mistral_lora_config.yaml"
        assert config_path.exists(), "Training config file not found"
    
    def test_load_training_config(self):
        """Test loading training configuration."""
        config = load_training_config()
        if config is None:
            pytest.skip("Config file not found")
        assert isinstance(config, dict)
    
    def test_config_has_model_section(self):
        """Test config has model section."""
        config = load_training_config()
        if config is None:
            pytest.skip("Config file not found")
        assert "model" in config
        assert "name" in config["model"]
    
    def test_config_has_lora_section(self):
        """Test config has LoRA section."""
        config = load_training_config()
        if config is None:
            pytest.skip("Config file not found")
        assert "lora" in config
        assert "r" in config["lora"]
        assert "lora_alpha" in config["lora"]
    
    def test_config_has_training_section(self):
        """Test config has training section."""
        config = load_training_config()
        if config is None:
            pytest.skip("Config file not found")
        assert "training" in config
        assert "num_train_epochs" in config["training"]
        assert "learning_rate" in config["training"]
    
    def test_config_has_data_section(self):
        """Test config has data section."""
        config = load_training_config()
        if config is None:
            pytest.skip("Config file not found")
        assert "data" in config
        assert "train_file" in config["data"]


class TestGCPConfig:
    """Tests for GCP configuration."""
    
    def test_gcp_config_file_exists(self):
        """Test that GCP config file exists."""
        config_path = CONFIG_DIR / "gcp_compute.yaml"
        assert config_path.exists(), "GCP config file not found"
    
    def test_gcp_config_valid_yaml(self):
        """Test that GCP config is valid YAML."""
        config_path = CONFIG_DIR / "gcp_compute.yaml"
        if not config_path.exists():
            pytest.skip("GCP config file not found")
        
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        assert isinstance(config, dict)
    
    def test_gcp_config_has_compute_section(self):
        """Test GCP config has compute section."""
        config_path = CONFIG_DIR / "gcp_compute.yaml"
        if not config_path.exists():
            pytest.skip("GCP config file not found")
        
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        assert "compute" in config
        assert "machine_type" in config["compute"]
        assert "gpu_type" in config["compute"]
    
    def test_gcp_config_has_cost_section(self):
        """Test GCP config has cost estimates."""
        config_path = CONFIG_DIR / "gcp_compute.yaml"
        if not config_path.exists():
            pytest.skip("GCP config file not found")
        
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        assert "cost" in config


class TestRequirementsFile:
    """Tests for requirements.txt file."""
    
    def test_requirements_file_exists(self):
        """Test that training requirements file exists."""
        req_path = PROJECT_DIR / "training" / "requirements.txt"
        assert req_path.exists(), "Training requirements.txt not found"
    
    def test_requirements_has_torch(self):
        """Test requirements includes torch."""
        req_path = PROJECT_DIR / "training" / "requirements.txt"
        if not req_path.exists():
            pytest.skip("Requirements file not found")
        
        content = req_path.read_text()
        assert "torch" in content
    
    def test_requirements_has_transformers(self):
        """Test requirements includes transformers."""
        req_path = PROJECT_DIR / "training" / "requirements.txt"
        if not req_path.exists():
            pytest.skip("Requirements file not found")
        
        content = req_path.read_text()
        assert "transformers" in content
    
    def test_requirements_has_peft(self):
        """Test requirements includes peft."""
        req_path = PROJECT_DIR / "training" / "requirements.txt"
        if not req_path.exists():
            pytest.skip("Requirements file not found")
        
        content = req_path.read_text()
        assert "peft" in content
    
    def test_requirements_has_bitsandbytes(self):
        """Test requirements includes bitsandbytes."""
        req_path = PROJECT_DIR / "training" / "requirements.txt"
        if not req_path.exists():
            pytest.skip("Requirements file not found")
        
        content = req_path.read_text()
        assert "bitsandbytes" in content


class TestValidateEnvironment:
    """Tests for full environment validation."""
    
    def test_validate_environment_returns_bool(self):
        """Test that validate_environment returns a boolean."""
        result = validate_environment(verbose=False)
        assert isinstance(result, bool)
    
    def test_validate_environment_runs_without_error(self):
        """Test that validate_environment runs without raising exceptions."""
        # Should not raise any exceptions
        try:
            validate_environment(verbose=False)
        except Exception as e:
            pytest.fail(f"validate_environment raised an exception: {e}")
