#!/usr/bin/env python3
"""
Training Environment Setup and Validation.

This module validates the training environment, checks dependencies,
and estimates memory requirements for fine-tuning.
"""

import importlib.metadata
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml

# Project paths
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_DIR = SCRIPT_DIR.parent
CONFIG_DIR = SCRIPT_DIR / "configs"
DATA_DIR = PROJECT_DIR / "data" / "processed"

# Required packages with minimum versions
REQUIRED_PACKAGES = {
    "torch": "2.1.0",
    "transformers": "4.36.0",
    "peft": "0.7.0",
    "accelerate": "0.25.0",
    "bitsandbytes": "0.41.0",
    "trl": "0.7.0",
    "datasets": "2.15.0",
    "tokenizers": "0.15.0",
    "wandb": "0.16.0",
    "tensorboard": "2.15.0",
    "scipy": "1.11.0",
    "scikit-learn": "1.3.0",
    "pyyaml": "6.0",
}

# Data files required for training
REQUIRED_DATA_FILES = [
    "train_postgres.jsonl",
    "val_postgres.jsonl",
    "test_postgres.jsonl",
]


def check_python_version() -> tuple[bool, str]:
    """Check if Python version is compatible."""
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    
    if version.major == 3 and version.minor >= 10:
        return True, version_str
    return False, version_str


def check_conda_environment() -> tuple[bool, str]:
    """Check if conda environment is activated."""
    conda_env = os.environ.get("CONDA_DEFAULT_ENV", "")
    if conda_env:
        return True, conda_env
    return False, "No conda environment detected"


def check_package_version(package: str, min_version: str) -> tuple[bool, str]:
    """Check if a package is installed with minimum version."""
    try:
        installed_version = importlib.metadata.version(package)
        # Simple version comparison (works for most cases)
        installed_parts = [int(x) for x in installed_version.split(".")[:3]]
        required_parts = [int(x) for x in min_version.split(".")[:3]]
        
        # Pad with zeros
        while len(installed_parts) < 3:
            installed_parts.append(0)
        while len(required_parts) < 3:
            required_parts.append(0)
        
        is_valid = installed_parts >= required_parts
        return is_valid, installed_version
    except importlib.metadata.PackageNotFoundError:
        return False, "Not installed"


def check_gpu_specs() -> dict[str, Any]:
    """
    Check GPU specifications.
    
    Returns:
        Dictionary with GPU specs or error info
    """
    specs = {
        "cuda_available": False,
        "gpu_count": 0,
        "gpu_name": None,
        "gpu_memory_gb": 0,
        "cuda_version": None,
        "cudnn_version": None,
    }
    
    try:
        import torch
        
        specs["cuda_available"] = torch.cuda.is_available()
        
        if specs["cuda_available"]:
            specs["gpu_count"] = torch.cuda.device_count()
            specs["gpu_name"] = torch.cuda.get_device_name(0)
            specs["gpu_memory_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / (1024**3), 1
            )
            specs["cuda_version"] = torch.version.cuda
            
            # Try to get cuDNN version
            if hasattr(torch.backends.cudnn, "version"):
                specs["cudnn_version"] = torch.backends.cudnn.version()
    except Exception as e:
        specs["error"] = str(e)
    
    return specs


def check_data_files() -> dict[str, Any]:
    """
    Check if required data files exist.
    
    Returns:
        Dictionary with file statistics
    """
    stats = {
        "all_present": True,
        "files": {},
    }
    
    for filename in REQUIRED_DATA_FILES:
        filepath = DATA_DIR / filename
        file_info = {
            "exists": filepath.exists(),
            "size_mb": 0,
            "examples": 0,
        }
        
        if filepath.exists():
            file_info["size_mb"] = round(filepath.stat().st_size / (1024 * 1024), 1)
            
            # Count lines (examples)
            with open(filepath, "r") as f:
                file_info["examples"] = sum(1 for _ in f)
        else:
            stats["all_present"] = False
        
        stats["files"][filename] = file_info
    
    return stats


def estimate_memory_requirements(
    batch_size: int = 4,
    gradient_accumulation: int = 4,
    seq_length: int = 2048,
) -> dict[str, Any]:
    """
    Estimate GPU memory requirements for training.
    
    Args:
        batch_size: Per-device batch size
        gradient_accumulation: Gradient accumulation steps
        seq_length: Maximum sequence length
        
    Returns:
        Memory estimation dictionary
    """
    # Memory estimates in GB
    estimates = {
        "base_model_4bit": 3.5,      # Mistral-7B in 4-bit
        "lora_adapters": 0.1,         # LoRA parameters
        "optimizer_states": 2.0,      # AdamW states
        "activations": 8.0,           # Forward pass activations
        "gradient_accumulation": 4.0, # Gradient storage
        "buffer": 2.0,                # Safety buffer
    }
    
    total = sum(estimates.values())
    
    return {
        "breakdown": estimates,
        "total_estimated_gb": round(total, 1),
        "recommended_gpu_memory_gb": 40,  # A100 40GB
        "fits_in_a100_40gb": total < 40,
        "fits_in_v100_16gb": total < 16,
        "fits_in_t4_16gb": total < 16,
    }


def load_training_config() -> dict[str, Any] | None:
    """Load training configuration from YAML."""
    config_path = CONFIG_DIR / "mistral_lora_config.yaml"
    
    if not config_path.exists():
        return None
    
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def validate_environment(verbose: bool = True) -> bool:
    """
    Validate the complete training environment.
    
    Args:
        verbose: Whether to print detailed output
        
    Returns:
        True if all checks pass, False otherwise
    """
    all_passed = True
    
    if verbose:
        print("\n" + "━" * 50)
        print("🔍 Training Environment Validation")
        print("━" * 50)
    
    # Check Python version
    py_ok, py_version = check_python_version()
    if verbose:
        status = "✅" if py_ok else "❌"
        print(f"\nPython version: {py_version} {status}")
    if not py_ok:
        all_passed = False
    
    # Check conda environment
    conda_ok, conda_env = check_conda_environment()
    if verbose:
        status = "✅" if conda_ok else "⚠️"
        print(f"Conda environment: {conda_env} {status}")
    
    # Check required packages
    if verbose:
        print("\n📦 Required Packages:")
    
    packages_ok = True
    for package, min_version in REQUIRED_PACKAGES.items():
        pkg_ok, installed_version = check_package_version(package, min_version)
        if verbose:
            status = "✅" if pkg_ok else "❌"
            print(f"  {status} {package}: {installed_version} (required: >={min_version})")
        if not pkg_ok:
            packages_ok = False
    
    if not packages_ok:
        all_passed = False
    
    # Check GPU
    if verbose:
        print("\n🖥️  GPU Status:")
    
    gpu_specs = check_gpu_specs()
    if verbose:
        if gpu_specs["cuda_available"]:
            print(f"  ✅ CUDA available: {gpu_specs['cuda_version']}")
            print(f"  ✅ GPU: {gpu_specs['gpu_name']}")
            print(f"  ✅ GPU Memory: {gpu_specs['gpu_memory_gb']} GB")
        else:
            print("  ⚠️  CUDA not available (expected on Mac, required for training)")
    
    # Check data files
    if verbose:
        print("\n📁 Data Files:")
    
    data_stats = check_data_files()
    for filename, info in data_stats["files"].items():
        if verbose:
            status = "✅" if info["exists"] else "❌"
            if info["exists"]:
                print(f"  {status} {filename}: {info['examples']:,} examples ({info['size_mb']} MB)")
            else:
                print(f"  {status} {filename}: Not found")
    
    if not data_stats["all_present"]:
        all_passed = False
    
    # Memory estimation
    if verbose:
        print("\n💾 Memory Estimates (for GCP A100):")
    
    memory = estimate_memory_requirements()
    if verbose:
        for component, gb in memory["breakdown"].items():
            print(f"  {component.replace('_', ' ').title()}: {gb} GB")
        print(f"  {'─' * 30}")
        print(f"  Total estimated: {memory['total_estimated_gb']} GB")
        
        if memory["fits_in_a100_40gb"]:
            print(f"  ✅ Fits in A100 40GB")
        else:
            print(f"  ❌ May not fit in A100 40GB")
    
    # Load and validate config
    config = load_training_config()
    if verbose:
        print("\n⚙️  Configuration:")
        if config:
            print(f"  ✅ Training config loaded")
            print(f"     Model: {config['model']['name']}")
            print(f"     LoRA rank: {config['lora']['r']}")
            print(f"     Epochs: {config['training']['num_train_epochs']}")
            print(f"     Batch size: {config['training']['per_device_train_batch_size']}")
        else:
            print(f"  ❌ Training config not found")
            all_passed = False
    
    # Summary
    if verbose:
        print("\n" + "━" * 50)
        if all_passed:
            print("✅ Environment validation passed!")
        else:
            print("❌ Environment validation failed - see errors above")
        print("━" * 50)
    
    return all_passed


def print_next_steps():
    """Print next steps after environment setup."""
    print("\n📋 Next Steps:")
    print("  1. Review training config: training/configs/mistral_lora_config.yaml")
    print("  2. Setup GCP instance: ./scripts/gcp/create_training_instance.sh")
    print("  3. Or proceed with local testing (limited without GPU)")
    print("  4. Run training: python -m training.train")


def main() -> int:
    """Main entry point for environment validation."""
    print("🔧 SQL Codegen Training Environment Setup")
    
    is_valid = validate_environment(verbose=True)
    
    if is_valid:
        print_next_steps()
        return 0
    else:
        print("\n⚠️  Please fix the issues above before proceeding.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
