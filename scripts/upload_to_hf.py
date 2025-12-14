#!/usr/bin/env python3
"""
Upload trained model from GCS to HuggingFace Hub.

This script:
1. Downloads the model from GCS to a local temp directory
2. Copies the MODEL_CARD.md as README.md
3. Uploads everything to HuggingFace Hub

Usage:
    python scripts/upload_to_hf.py

Requirements:
    - gsutil installed and authenticated
    - huggingface_hub installed
    - HF_TOKEN environment variable or logged in via `huggingface-cli login`
"""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    from huggingface_hub import HfApi, login, whoami
except ImportError:
    print("Error: huggingface_hub not installed. Run: pip install huggingface_hub")
    sys.exit(1)


# Configuration
GCS_MODEL_PATH = "gs://sql-codegen-slm-data/models/mistral-sql-final/final_model/"
HF_REPO_ID = "rajeshmanikka/mistral-7b-text-to-sql"
MODEL_CARD_PATH = "docs/MODEL_CARD.md"


def check_gsutil():
    """Check if gsutil is available."""
    try:
        result = subprocess.run(
            ["gsutil", "version"],
            capture_output=True,
            text=True,
            check=True
        )
        print(f"✅ gsutil available: {result.stdout.strip().split()[0]}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ gsutil not found. Please install Google Cloud SDK.")
        print("   https://cloud.google.com/sdk/docs/install")
        return False


def check_hf_auth():
    """Check HuggingFace authentication."""
    # Check for HF_TOKEN environment variable
    hf_token = os.environ.get("HF_TOKEN")
    
    if hf_token:
        print("✅ Using HF_TOKEN from environment")
        login(token=hf_token)
        return True
    
    # Try to use cached credentials
    try:
        user_info = whoami()
        print(f"✅ Logged in to HuggingFace as: {user_info['name']}")
        return True
    except Exception:
        print("⚠️  Not logged in to HuggingFace")
        print("   Please run: huggingface-cli login")
        print("   Or set HF_TOKEN environment variable")
        
        # Prompt for login
        response = input("Would you like to login now? (y/n): ")
        if response.lower() == 'y':
            login()
            return True
        return False


def download_from_gcs(gcs_path: str, local_dir: str) -> bool:
    """Download model files from GCS to local directory."""
    print(f"\n📥 Downloading from GCS...")
    print(f"   Source: {gcs_path}")
    print(f"   Destination: {local_dir}")
    
    try:
        result = subprocess.run(
            ["gsutil", "-m", "cp", "-r", gcs_path + "*", local_dir],
            capture_output=True,
            text=True,
            check=True
        )
        
        # List downloaded files
        files = list(Path(local_dir).glob("*"))
        print(f"✅ Downloaded {len(files)} files:")
        for f in files:
            size = f.stat().st_size / (1024 * 1024)  # MB
            print(f"   - {f.name} ({size:.2f} MB)")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ GCS download failed: {e.stderr}")
        return False


def copy_model_card(local_dir: str) -> bool:
    """Copy MODEL_CARD.md as README.md into the model directory."""
    # Find the project root (where docs/ is located)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    model_card_path = project_root / MODEL_CARD_PATH
    
    if not model_card_path.exists():
        print(f"❌ Model card not found: {model_card_path}")
        return False
    
    readme_dest = Path(local_dir) / "README.md"
    shutil.copy(model_card_path, readme_dest)
    print(f"✅ Copied MODEL_CARD.md as README.md")
    
    return True


def upload_to_hub(local_dir: str, repo_id: str) -> str:
    """Upload model files to HuggingFace Hub."""
    print(f"\n📤 Uploading to HuggingFace Hub...")
    print(f"   Repository: {repo_id}")
    
    api = HfApi()
    
    # Create repo if it doesn't exist
    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type="model",
            exist_ok=True,
            private=False
        )
        print(f"✅ Repository ready: {repo_id}")
    except Exception as e:
        print(f"⚠️  Repo creation note: {e}")
    
    # Upload all files
    try:
        api.upload_folder(
            folder_path=local_dir,
            repo_id=repo_id,
            repo_type="model",
            commit_message="Upload fine-tuned Mistral-7B text-to-SQL model (LoRA adapters)"
        )
        
        hub_url = f"https://huggingface.co/{repo_id}"
        print(f"✅ Upload complete!")
        print(f"   URL: {hub_url}")
        
        return hub_url
        
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return None


def main():
    """Main upload workflow."""
    print("=" * 60)
    print("HuggingFace Model Upload")
    print("=" * 60)
    print(f"Model: {HF_REPO_ID}")
    print(f"Source: {GCS_MODEL_PATH}")
    print("=" * 60)
    
    # Pre-flight checks
    print("\n🔍 Pre-flight checks...")
    
    if not check_gsutil():
        sys.exit(1)
    
    if not check_hf_auth():
        sys.exit(1)
    
    # Create temp directory
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"\n📁 Using temp directory: {temp_dir}")
        
        # Download from GCS
        if not download_from_gcs(GCS_MODEL_PATH, temp_dir):
            sys.exit(1)
        
        # Copy model card
        if not copy_model_card(temp_dir):
            sys.exit(1)
        
        # Upload to HuggingFace
        hub_url = upload_to_hub(temp_dir, HF_REPO_ID)
        
        if not hub_url:
            sys.exit(1)
    
    # Cleanup happens automatically with tempfile
    print("\n🧹 Cleaned up temp files")
    
    # Final summary
    print("\n" + "=" * 60)
    print("✅ UPLOAD COMPLETE")
    print("=" * 60)
    print(f"Model URL: https://huggingface.co/{HF_REPO_ID}")
    print(f"\nTo use this model:")
    print(f'  from peft import PeftModel')
    print(f'  model = PeftModel.from_pretrained(base_model, "{HF_REPO_ID}")')
    print("=" * 60)


if __name__ == "__main__":
    main()
