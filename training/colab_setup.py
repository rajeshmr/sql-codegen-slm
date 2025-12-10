#!/usr/bin/env python3
"""
Colab Environment Setup for SQL Codegen SLM Training.

This module provides utilities for setting up and validating
the Google Colab Pro+ training environment with Google Cloud Storage.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


# Default GCS configuration
DEFAULT_PROJECT_ID = "your-gcp-project-id"
DEFAULT_BUCKET_NAME = "sql-codegen-slm-data"


def is_colab() -> bool:
    """Check if running in Google Colab."""
    try:
        import google.colab
        return True
    except ImportError:
        return False


def authenticate_gcp() -> dict[str, Any]:
    """
    Authenticate with Google Cloud in Colab.
    
    Returns:
        Dictionary with authentication status
    """
    result = {
        "authenticated": False,
        "project_id": None,
        "error": None,
    }
    
    if not is_colab():
        result["error"] = "Not running in Colab"
        return result
    
    try:
        from google.colab import auth
        auth.authenticate_user()
        result["authenticated"] = True
        print("✅ Authenticated with Google Cloud")
    except Exception as e:
        result["error"] = str(e)
        print(f"❌ Authentication failed: {e}")
    
    return result


def setup_gcs(project_id: str, bucket_name: str) -> dict[str, Any]:
    """
    Setup Google Cloud Storage bucket and configure gcloud.
    
    Args:
        project_id: GCP project ID
        bucket_name: GCS bucket name
        
    Returns:
        Dictionary with GCS setup status
    """
    result = {
        "configured": False,
        "project_id": project_id,
        "bucket_name": bucket_name,
        "bucket_url": f"gs://{bucket_name}",
        "error": None,
    }
    
    try:
        # Set project
        subprocess.run(
            ["gcloud", "config", "set", "project", project_id],
            check=True, capture_output=True
        )
        print(f"✅ GCP project set: {project_id}")
        
        # Check if bucket exists, create if not
        check_bucket = subprocess.run(
            ["gsutil", "ls", f"gs://{bucket_name}"],
            capture_output=True
        )
        
        if check_bucket.returncode != 0:
            print(f"   Creating bucket: {bucket_name}")
            subprocess.run(
                ["gsutil", "mb", "-l", "us", f"gs://{bucket_name}"],
                check=True, capture_output=True
            )
            print(f"✅ Bucket created: gs://{bucket_name}")
        else:
            print(f"✅ Bucket exists: gs://{bucket_name}")
        
        result["configured"] = True
        result["project_id"] = project_id
        
    except subprocess.CalledProcessError as e:
        result["error"] = e.stderr.decode() if e.stderr else str(e)
        print(f"❌ GCS setup failed: {result['error']}")
    except Exception as e:
        result["error"] = str(e)
        print(f"❌ GCS setup failed: {e}")
    
    return result


def download_data_from_gcs(
    bucket_name: str,
    local_path: str = "/content/data",
    gcs_prefix: str = "data/"
) -> dict[str, Any]:
    """
    Download training data from GCS to local storage.
    
    Args:
        bucket_name: GCS bucket name
        local_path: Local directory to download to
        gcs_prefix: Prefix in GCS bucket
        
    Returns:
        Dictionary with download status
    """
    result = {
        "downloaded": False,
        "files": {},
        "local_path": local_path,
        "error": None,
    }
    
    required_files = [
        "train_postgres.jsonl",
        "val_postgres.jsonl",
        "test_postgres.jsonl",
    ]
    
    # Create local directory
    Path(local_path).mkdir(parents=True, exist_ok=True)
    
    try:
        # Download from GCS
        gcs_path = f"gs://{bucket_name}/{gcs_prefix}"
        print(f"📥 Downloading data from {gcs_path}")
        
        subprocess.run(
            ["gsutil", "-m", "cp", "-r", f"{gcs_path}*", local_path],
            check=True
        )
        
        # Verify files
        all_present = True
        for filename in required_files:
            filepath = Path(local_path) / filename
            if filepath.exists():
                size_mb = round(filepath.stat().st_size / (1024 * 1024), 1)
                lines = sum(1 for _ in open(filepath))
                result["files"][filename] = {
                    "exists": True,
                    "size_mb": size_mb,
                    "examples": lines,
                }
                print(f"   ✅ {filename}: {lines:,} examples ({size_mb} MB)")
            else:
                result["files"][filename] = {"exists": False}
                all_present = False
                print(f"   ❌ {filename}: NOT FOUND")
        
        result["downloaded"] = all_present
        
    except subprocess.CalledProcessError as e:
        result["error"] = str(e)
        print(f"❌ Download failed: {e}")
        print("")
        print("📋 To upload data to GCS, run locally:")
        print(f"   gsutil -m cp data/processed/*.jsonl gs://{bucket_name}/{gcs_prefix}")
    except Exception as e:
        result["error"] = str(e)
        print(f"❌ Download failed: {e}")
    
    return result


def upload_checkpoint_to_gcs(
    local_path: str,
    bucket_name: str,
    gcs_prefix: str = "models/"
) -> dict[str, Any]:
    """
    Upload checkpoint/model to GCS.
    
    Args:
        local_path: Local checkpoint directory
        bucket_name: GCS bucket name
        gcs_prefix: Prefix in GCS bucket
        
    Returns:
        Dictionary with upload status
    """
    result = {
        "uploaded": False,
        "gcs_path": f"gs://{bucket_name}/{gcs_prefix}",
        "error": None,
    }
    
    try:
        gcs_path = f"gs://{bucket_name}/{gcs_prefix}"
        print(f"📤 Uploading to {gcs_path}")
        
        subprocess.run(
            ["gsutil", "-m", "cp", "-r", local_path, gcs_path],
            check=True
        )
        
        result["uploaded"] = True
        print(f"✅ Uploaded to {gcs_path}")
        
    except Exception as e:
        result["error"] = str(e)
        print(f"❌ Upload failed: {e}")
    
    return result


def check_gpu() -> dict[str, Any]:
    """
    Check GPU allocation in Colab.
    
    Returns:
        Dictionary with GPU specifications
    """
    specs = {
        "available": False,
        "name": None,
        "memory_gb": 0,
        "cuda_version": None,
        "is_a100": False,
        "is_v100": False,
        "is_t4": False,
        "warning": None,
    }
    
    try:
        import torch
        
        specs["available"] = torch.cuda.is_available()
        
        if specs["available"]:
            specs["name"] = torch.cuda.get_device_name(0)
            specs["memory_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / (1024**3), 1
            )
            specs["cuda_version"] = torch.version.cuda
            
            # Detect GPU type
            name_lower = specs["name"].lower()
            specs["is_a100"] = "a100" in name_lower
            specs["is_v100"] = "v100" in name_lower
            specs["is_t4"] = "t4" in name_lower
            
            # Print status
            if specs["is_a100"]:
                print(f"✅ GPU: {specs['name']} ({specs['memory_gb']} GB)")
                print("   🎉 Got A100 - optimal for training!")
            elif specs["is_v100"]:
                print(f"✅ GPU: {specs['name']} ({specs['memory_gb']} GB)")
                print("   ⚠️  Got V100 - good, but A100 is faster")
                specs["warning"] = "V100 allocated instead of A100"
            elif specs["is_t4"]:
                print(f"⚠️  GPU: {specs['name']} ({specs['memory_gb']} GB)")
                print("   ⚠️  Got T4 - training will be slower")
                print("   💡 Tip: Disconnect and reconnect to try for A100")
                specs["warning"] = "T4 allocated - consider reconnecting for A100"
            else:
                print(f"✅ GPU: {specs['name']} ({specs['memory_gb']} GB)")
        else:
            print("❌ No GPU available!")
            print("   Go to Runtime > Change runtime type > GPU")
            specs["warning"] = "No GPU - enable in Runtime settings"
            
    except Exception as e:
        specs["error"] = str(e)
        print(f"❌ Error checking GPU: {e}")
    
    return specs


def setup_local_dirs() -> dict[str, Any]:
    """
    Setup local directories for training.
    
    Returns:
        Dictionary with directory status
    """
    result = {
        "created": False,
        "directories": [],
        "error": None,
    }
    
    dirs = [
        "/content/data",
        "/content/models",
        "/content/logs",
        "/content/tensorboard",
    ]
    
    try:
        for d in dirs:
            Path(d).mkdir(parents=True, exist_ok=True)
            result["directories"].append(d)
        result["created"] = True
        print("✅ Local directories created")
        for d in dirs:
            print(f"   {d}")
    except Exception as e:
        result["error"] = str(e)
        print(f"❌ Failed to create directories: {e}")
    
    return result


def setup_colab_environment(
    project_id: str = None,
    bucket_name: str = None,
) -> dict[str, Any]:
    """
    Complete Colab environment setup with GCS.
    
    Args:
        project_id: GCP project ID (or set via environment)
        bucket_name: GCS bucket name
        
    Returns:
        Dictionary with full environment status
    """
    print("🔧 Setting up Colab Training Environment (GCS)")
    print("=" * 50)
    print("")
    
    # Get config from environment or defaults
    project_id = project_id or os.environ.get("GCP_PROJECT_ID", DEFAULT_PROJECT_ID)
    bucket_name = bucket_name or os.environ.get("GCS_BUCKET", DEFAULT_BUCKET_NAME)
    
    status = {
        "is_colab": is_colab(),
        "auth": None,
        "gcs": None,
        "gpu": None,
        "data": None,
        "dirs": None,
        "ready": False,
    }
    
    # Check if in Colab
    if not status["is_colab"]:
        print("⚠️  Not running in Google Colab")
        print("   This setup is designed for Colab Pro+")
        print("   Some features may not work locally")
        print("")
    
    # Authenticate with GCP
    print("🔐 Authenticating with Google Cloud...")
    status["auth"] = authenticate_gcp()
    print("")
    
    # Setup GCS
    if status["auth"] and status["auth"].get("authenticated", False):
        print("☁️  Setting up Google Cloud Storage...")
        status["gcs"] = setup_gcs(project_id, bucket_name)
        print("")
    
    # Check GPU
    print("🖥️  Checking GPU allocation...")
    status["gpu"] = check_gpu()
    print("")
    
    # Setup local directories
    print("📁 Setting up local directories...")
    status["dirs"] = setup_local_dirs()
    print("")
    
    # Download data from GCS
    if status["gcs"] and status["gcs"].get("configured", False):
        print("📥 Downloading data from GCS...")
        status["data"] = download_data_from_gcs(bucket_name)
        print("")
    
    # Summary
    print("=" * 50)
    
    all_good = (
        (status["auth"] and status["auth"].get("authenticated", False) or not status["is_colab"]) and
        (status["gpu"] and status["gpu"].get("available", False)) and
        (status["data"] and status["data"].get("downloaded", False)) and
        (status["dirs"] and status["dirs"].get("created", False))
    )
    
    status["ready"] = all_good
    
    if all_good:
        print("✅ Environment ready for training!")
        print("")
        print("Next steps:")
        print("  1. Run training: !python -m training.train")
        print("  2. Monitor: %tensorboard --logdir /content/tensorboard")
        print(f"  3. After training, models saved to: gs://{bucket_name}/models/")
    else:
        print("❌ Environment not ready - fix issues above")
        if not (status["data"] and status["data"].get("downloaded", False)):
            print("")
            print("📋 To upload data to GCS from your local machine:")
            print(f"   gsutil -m cp data/processed/*.jsonl gs://{bucket_name}/data/")
    
    print("=" * 50)
    
    return status


def sync_checkpoints_to_gcs(
    local_path: str = "/content/models",
    bucket_name: str = None,
) -> dict[str, Any]:
    """
    Sync local checkpoints to GCS (call periodically during training).
    
    Args:
        local_path: Local checkpoint directory
        bucket_name: GCS bucket name
        
    Returns:
        Dictionary with sync status
    """
    bucket_name = bucket_name or os.environ.get("GCS_BUCKET", DEFAULT_BUCKET_NAME)
    return upload_checkpoint_to_gcs(local_path, bucket_name, "models/")


def estimate_training_time(
    num_examples: int = 6016,
    batch_size: int = 4,
    gradient_accumulation: int = 4,
    epochs: int = 3,
) -> dict[str, Any]:
    """
    Estimate training time and compute units.
    
    Args:
        num_examples: Number of training examples
        batch_size: Per-device batch size
        gradient_accumulation: Gradient accumulation steps
        epochs: Number of epochs
        
    Returns:
        Training time estimates
    """
    effective_batch = batch_size * gradient_accumulation
    steps_per_epoch = num_examples // effective_batch
    total_steps = steps_per_epoch * epochs
    
    # Rough estimates based on A100
    seconds_per_step_a100 = 2.5  # ~2.5 seconds per step on A100
    seconds_per_step_v100 = 4.0  # ~4 seconds per step on V100
    seconds_per_step_t4 = 8.0    # ~8 seconds per step on T4
    
    estimates = {
        "total_steps": total_steps,
        "steps_per_epoch": steps_per_epoch,
        "a100": {
            "hours": round(total_steps * seconds_per_step_a100 / 3600, 1),
            "compute_units": round(total_steps * seconds_per_step_a100 / 3600 * 10, 0),
        },
        "v100": {
            "hours": round(total_steps * seconds_per_step_v100 / 3600, 1),
            "compute_units": round(total_steps * seconds_per_step_v100 / 3600 * 8, 0),
        },
        "t4": {
            "hours": round(total_steps * seconds_per_step_t4 / 3600, 1),
            "compute_units": round(total_steps * seconds_per_step_t4 / 3600 * 5, 0),
        },
    }
    
    print("⏱️  Training Time Estimates")
    print(f"   Total steps: {total_steps:,}")
    print(f"   Steps per epoch: {steps_per_epoch:,}")
    print("")
    print(f"   A100: ~{estimates['a100']['hours']} hours (~{estimates['a100']['compute_units']:.0f} compute units)")
    print(f"   V100: ~{estimates['v100']['hours']} hours (~{estimates['v100']['compute_units']:.0f} compute units)")
    print(f"   T4:   ~{estimates['t4']['hours']} hours (~{estimates['t4']['compute_units']:.0f} compute units)")
    
    return estimates


if __name__ == "__main__":
    setup_colab_environment()
