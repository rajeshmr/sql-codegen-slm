# Deployment Guide

This guide covers deploying the trained text-to-SQL model to HuggingFace Hub and creating a demo on HuggingFace Spaces.

## Prerequisites

### Required Tools

1. **Google Cloud SDK** (for gsutil)
   ```bash
   # macOS
   brew install google-cloud-sdk
   
   # Or download from https://cloud.google.com/sdk/docs/install
   ```

2. **HuggingFace CLI**
   ```bash
   pip install huggingface_hub
   ```

3. **Git** (for Spaces deployment)
   ```bash
   # Usually pre-installed on macOS/Linux
   git --version
   ```

### Authentication

1. **Google Cloud** (for accessing GCS bucket)
   ```bash
   gcloud auth login
   gcloud config set project YOUR_PROJECT_ID
   ```

2. **HuggingFace**
   ```bash
   # Option 1: CLI login (recommended)
   huggingface-cli login
   
   # Option 2: Environment variable
   export HF_TOKEN=hf_your_token_here
   ```
   
   Get your token from: https://huggingface.co/settings/tokens

## Step 1: Upload Model to HuggingFace Hub

### Using the Upload Script

```bash
# From project root
python scripts/upload_to_hf.py
```

This script will:
1. Download model from `gs://sql-codegen-slm-data/models/mistral-sql-final/final_model/`
2. Copy `docs/MODEL_CARD.md` as `README.md`
3. Create HuggingFace repo `rajeshmanikka/mistral-7b-text-to-sql`
4. Upload all files

### Expected Output

```
============================================================
HuggingFace Model Upload
============================================================
Model: rajeshmanikka/mistral-7b-text-to-sql
Source: gs://sql-codegen-slm-data/models/mistral-sql-final/final_model/
============================================================

🔍 Pre-flight checks...
✅ gsutil available
✅ Logged in to HuggingFace as: rajeshmanikka

📥 Downloading from GCS...
✅ Downloaded 6 files:
   - adapter_model.safetensors (163.87 MB)
   - adapter_config.json (0.00 MB)
   - tokenizer.json (1.72 MB)
   - tokenizer.model (0.47 MB)
   - tokenizer_config.json (0.00 MB)
   - special_tokens_map.json (0.00 MB)

✅ Copied MODEL_CARD.md as README.md

📤 Uploading to HuggingFace Hub...
✅ Repository ready: rajeshmanikka/mistral-7b-text-to-sql
✅ Upload complete!
   URL: https://huggingface.co/rajeshmanikka/mistral-7b-text-to-sql

============================================================
✅ UPLOAD COMPLETE
============================================================
Model URL: https://huggingface.co/rajeshmanikka/mistral-7b-text-to-sql
============================================================
```

### Manual Upload (Alternative)

If the script fails, you can upload manually:

```bash
# 1. Download from GCS
mkdir -p ./model_upload
gsutil -m cp -r gs://sql-codegen-slm-data/models/mistral-sql-final/final_model/* ./model_upload/

# 2. Copy model card
cp docs/MODEL_CARD.md ./model_upload/README.md

# 3. Upload to HuggingFace
huggingface-cli upload rajeshmanikka/mistral-7b-text-to-sql ./model_upload --repo-type model

# 4. Cleanup
rm -rf ./model_upload
```

## Step 2: Deploy Demo to HuggingFace Spaces

### Using the Deployment Script

```bash
# From project root
bash scripts/deploy_to_spaces.sh
```

This script will:
1. Clone/create the Spaces repository
2. Copy files from `spaces/` directory
3. Commit and push to HuggingFace Spaces

### Expected Output

```
============================================================
HuggingFace Spaces Deployment
============================================================
Space: https://huggingface.co/spaces/rajeshmanikka/text-to-sql-demo
Source: /path/to/sql-codegen-slm/spaces
============================================================

🔍 Checking required files...
   ✅ app.py
   ✅ requirements.txt
   ✅ README.md

🔐 Checking HuggingFace authentication...
   ✅ Logged in as: rajeshmanikka

📥 Setting up Spaces repository...
   ✅ Cloned existing Space

📋 Copying files...
'spaces/app.py' -> './app.py'
'spaces/requirements.txt' -> './requirements.txt'
'spaces/README.md' -> './README.md'

💾 Committing changes...
🚀 Pushing to HuggingFace Spaces...
   ✅ Pushed successfully!

============================================================
✅ DEPLOYMENT COMPLETE
============================================================

Your Space is now deploying at:
   https://huggingface.co/spaces/rajeshmanikka/text-to-sql-demo

Note: It may take a few minutes for the Space to build.
============================================================
```

### Manual Deployment (Alternative)

```bash
# 1. Clone the Spaces repo
git clone https://huggingface.co/spaces/rajeshmanikka/text-to-sql-demo
cd text-to-sql-demo

# 2. Copy files
cp ../sql-codegen-slm/spaces/app.py .
cp ../sql-codegen-slm/spaces/requirements.txt .
cp ../sql-codegen-slm/spaces/README.md .

# 3. Commit and push
git add -A
git commit -m "Deploy text-to-SQL demo"
git push
```

## Step 3: Verification

### Verify Model Upload

1. Visit: https://huggingface.co/rajeshmanikka/mistral-7b-text-to-sql
2. Check that all files are present:
   - `README.md` (model card)
   - `adapter_model.safetensors`
   - `adapter_config.json`
   - Tokenizer files
3. Verify the model card renders correctly

### Verify Spaces Deployment

1. Visit: https://huggingface.co/spaces/rajeshmanikka/text-to-sql-demo
2. Check the "Logs" tab for build status
3. Wait for build to complete (2-5 minutes)
4. Test the demo with an example query

### Test Model Loading

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# This should work after upload
tokenizer = AutoTokenizer.from_pretrained("rajeshmanikka/mistral-7b-text-to-sql")
print("✅ Tokenizer loaded successfully")
```

## Troubleshooting

### Authentication Issues

**Problem**: "Not logged in to HuggingFace"
```bash
# Solution: Login via CLI
huggingface-cli login

# Or set token
export HF_TOKEN=hf_your_token_here
```

**Problem**: "Permission denied" on HuggingFace
- Ensure your token has write permissions
- Check token at: https://huggingface.co/settings/tokens

### GCS Download Issues

**Problem**: "gsutil not found"
```bash
# Install Google Cloud SDK
brew install google-cloud-sdk  # macOS
# Or: https://cloud.google.com/sdk/docs/install
```

**Problem**: "Access denied to bucket"
```bash
# Re-authenticate
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```

### Spaces Build Errors

**Problem**: Space fails to build
1. Check the "Logs" tab on HuggingFace Spaces
2. Common issues:
   - Missing dependencies in `requirements.txt`
   - Python version incompatibility
   - Memory limits exceeded

**Problem**: Model loading timeout
- The free tier has limited resources
- First load may take 2-3 minutes
- Consider upgrading to GPU Spaces for faster inference

### Upload Failures

**Problem**: "Repository not found"
```bash
# Create the repo first
huggingface-cli repo create rajeshmanikka/mistral-7b-text-to-sql --type model
```

**Problem**: Large file upload fails
- HuggingFace uses Git LFS for large files
- Ensure Git LFS is installed: `git lfs install`

## Post-Deployment Checklist

- [ ] Model visible at HuggingFace Hub URL
- [ ] Model card (README.md) renders correctly
- [ ] All model files present (adapter, tokenizer)
- [ ] Spaces demo is accessible
- [ ] Demo generates SQL correctly
- [ ] Example queries work as expected
- [ ] GitHub README updated with live links
- [ ] Pushed all changes to GitHub

## Updating the Deployment

### Update Model

```bash
# Re-run upload script
python scripts/upload_to_hf.py
```

### Update Demo

```bash
# Edit spaces/app.py locally, then:
bash scripts/deploy_to_spaces.sh
```

### Update Model Card

1. Edit `docs/MODEL_CARD.md`
2. Re-run `python scripts/upload_to_hf.py`

## Resource Links

- **HuggingFace Hub Docs**: https://huggingface.co/docs/hub
- **Spaces Docs**: https://huggingface.co/docs/hub/spaces
- **PEFT Docs**: https://huggingface.co/docs/peft
- **Gradio Docs**: https://www.gradio.app/docs
