# Training Setup Guide - Google Colab Pro+ with GCS

This guide covers setting up the training environment on Google Colab Pro+ with Google Cloud Storage for fine-tuning Mistral-7B on SQL code generation.

## Why Colab Pro+ with GCS?

| Factor | GCP A100 VM | Colab Pro+ |
|--------|-------------|------------|
| **Upfront cost** | $0 | $58.99/month |
| **Cost per training run** | $35-53 | Included |
| **5 training runs** | $175-265 | $58.99 |
| **Setup complexity** | High | Low |
| **Storage** | Persistent disk | GCS (fast, reliable) |
| **Best for** | Production | Learning/experimentation |

**Why GCS over Google Drive?**
- **Faster** - Direct gsutil transfers vs Drive sync
- **More reliable** - No sync issues or quota limits
- **Better integration** - Native Colab support
- **Negligible cost** - ~$0.02/GB/month

## Prerequisites

1. **Google Colab Pro+ subscription** ($58.99/month)
   - Sign up at [colab.research.google.com](https://colab.research.google.com)
   
2. **GCP Project** with Cloud Storage enabled
   - Create at [console.cloud.google.com](https://console.cloud.google.com)
   - Enable Cloud Storage API
   
3. **Google Cloud SDK** installed locally
   - Install: `brew install google-cloud-sdk` (Mac)
   - Or: https://cloud.google.com/sdk/docs/install

## Quick Start

1. Upload data to GCS: `./scripts/prepare_data_for_upload.sh`
2. Open `notebooks/train_colab.ipynb` in Colab
3. Set your `PROJECT_ID` and `BUCKET_NAME`
4. Run all cells in order
5. Estimated time: 8-12 hours on A100

## Step-by-Step Setup

### 1. Setup GCP Project

```bash
# Authenticate with GCP
gcloud auth login

# Set your project
gcloud config set project YOUR_PROJECT_ID

# Enable Cloud Storage API
gcloud services enable storage.googleapis.com
```

### 2. Upload Data to GCS

```bash
# On your local machine
cd sql-codegen-slm

# Set your bucket name (optional, defaults to sql-codegen-slm-data)
export GCS_BUCKET=your-bucket-name

# Upload data to GCS
./scripts/prepare_data_for_upload.sh
```

This uploads to `gs://your-bucket-name/data/`:
- `train_postgres.jsonl` (6,016 examples)
- `val_postgres.jsonl` (332 examples)
- `test_postgres.jsonl` (333 examples)

### 3. Open Colab Notebook

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. File > Open notebook > GitHub
3. Enter your repo URL
4. Select `notebooks/train_colab.ipynb`

### 4. Configure Runtime

1. Runtime > Change runtime type
2. Select **GPU**
3. For Pro+ users: Select **A100** if available

### 5. Run Training

Execute cells in order:
1. **Check GPU** - Verify A100/V100 allocation
2. **Configure GCP** - Set PROJECT_ID and BUCKET_NAME
3. **Authenticate** - Login to GCP
4. **Clone Repo** - Get latest code
5. **Download Data** - Copy from GCS to local
6. **Verify Environment** - Check everything is ready
7. **Start Training** - Begin fine-tuning
8. **Sync to GCS** - Backup checkpoints
9. **Monitor** - View TensorBoard

## GPU Allocation Tips

Colab Pro+ gives priority access to A100 GPUs, but availability varies.

### Getting A100

1. **Time of day**: Try early morning (US time) for better availability
2. **Reconnect**: If you get T4, disconnect and reconnect
3. **Background execution**: Pro+ allows background execution - start training and close browser

### GPU Comparison

| GPU | VRAM | Training Time | Notes |
|-----|------|---------------|-------|
| A100 | 40GB | 8-12 hours | Optimal |
| V100 | 16GB | 12-18 hours | Good |
| T4 | 16GB | 20-30 hours | Slow but works |

## Session Management

### Handling Disconnects

Colab sessions can disconnect after ~12 hours idle or 24 hours total. Our setup handles this:

1. **Checkpoints every 500 steps** - Saved to Google Drive
2. **Resume training** - Run with `--resume` flag
3. **Background execution** - Pro+ keeps running when browser closes

### Resuming Training

If disconnected:

```python
# In Colab, after reconnecting:
!python -m training.train --config training/configs/mistral_lora_config.yaml --resume
```

The training will automatically find the latest checkpoint in Drive.

## Checkpoint Recovery

Checkpoints are saved to `/content/drive/MyDrive/sql-codegen-models/`:

```
sql-codegen-models/
├── checkpoint-500/
├── checkpoint-1000/
├── config.json
├── adapter_config.json
└── adapter_model.bin
```

To manually load a checkpoint:

```python
from peft import PeftModel
model = PeftModel.from_pretrained(
    base_model,
    "/content/drive/MyDrive/sql-codegen-models/checkpoint-1000"
)
```

## Cost Breakdown

### Colab Pro+ ($58.99/month)

- **Compute units**: 600/month
- **Per training run**: ~80-120 units (8-12 hours)
- **Runs per month**: 5-7 full training runs
- **Background execution**: Included
- **Priority GPU access**: Included

### Comparison with GCP

| Scenario | GCP Cost | Colab Pro+ Cost |
|----------|----------|-----------------|
| 1 training run | $35-53 | $58.99 |
| 3 training runs | $105-159 | $58.99 |
| 5 training runs | $175-265 | $58.99 |

**Break-even**: ~1.5 training runs

## Configuration Options

### Adjusting for Different GPUs

If you get a V100 or T4 instead of A100, adjust the config:

```yaml
# For V100/T4 (16GB VRAM)
training:
  per_device_train_batch_size: 2  # Reduced from 4
  gradient_accumulation_steps: 8   # Increased to maintain effective batch
```

### Faster Training (Less Accuracy)

```yaml
training:
  num_train_epochs: 1              # Quick test
  save_steps: 200                  # More frequent saves
  eval_steps: 200
```

### More Thorough Training

```yaml
training:
  num_train_epochs: 5              # More epochs
  learning_rate: 0.0001            # Lower LR for stability
```

## Troubleshooting

### "CUDA out of memory"

1. Reduce batch size:
   ```yaml
   per_device_train_batch_size: 2
   ```
2. Increase gradient accumulation:
   ```yaml
   gradient_accumulation_steps: 8
   ```
3. Restart runtime and try again

### "No GPU available"

1. Go to Runtime > Change runtime type
2. Select GPU
3. If still no GPU, Colab may be at capacity - try later

### "Session crashed"

1. Your checkpoints are safe in Drive
2. Reconnect to Colab
3. Run setup cells (1-5)
4. Resume training with `--resume` flag

### "Drive not mounting"

1. Clear browser cookies for Google
2. Try incognito mode
3. Re-authorize Drive access

### "Data files not found"

1. Verify data uploaded to Drive
2. Check path: `/content/drive/MyDrive/sql-codegen-data/`
3. Run the data upload cell again

## TensorBoard Monitoring

View training progress in real-time:

```python
%load_ext tensorboard
%tensorboard --logdir /content/drive/MyDrive/sql-codegen-tensorboard
```

Metrics to watch:
- **train/loss**: Should decrease steadily
- **eval/loss**: Should decrease, watch for overfitting
- **learning_rate**: Follows cosine schedule

## After Training

### Download Model

```python
# Zip model for download
!zip -r model.zip /content/drive/MyDrive/sql-codegen-models/

from google.colab import files
files.download('model.zip')
```

### Test Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "/content/drive/MyDrive/sql-codegen-models",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(
    "/content/drive/MyDrive/sql-codegen-models"
)

# Generate SQL
prompt = "[INST] Write SQL to find all customers [/INST]"
outputs = model.generate(tokenizer(prompt, return_tensors="pt").input_ids)
print(tokenizer.decode(outputs[0]))
```

## Next Steps

After training completes:

1. **Evaluate model** - Run on test set
2. **Export for deployment** - Convert to GGUF for local inference
3. **Build API** - Create FastAPI backend
4. **Build UI** - Create Next.js frontend
