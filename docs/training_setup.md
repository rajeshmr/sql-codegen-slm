# Training Setup Guide

This guide covers setting up the training environment for fine-tuning Mistral-7B on SQL code generation.

## Hardware Requirements

### Minimum Requirements (GCP A100)
- **GPU**: NVIDIA A100 40GB (recommended)
- **RAM**: 52GB system memory
- **Disk**: 200GB SSD
- **CUDA**: 12.1+

### Alternative Options
| GPU | VRAM | Training Time | Cost/Hour | Total Cost |
|-----|------|---------------|-----------|------------|
| A100 40GB | 40GB | 8-12 hours | $4.41 | $35-53 |
| T4 16GB | 16GB | 24-36 hours | $1.50 | $36-54 |
| V100 16GB | 16GB | 16-24 hours | $2.50 | $40-60 |

## Installation

### Local Setup (for testing)

```bash
# 1. Activate conda environment
conda activate sql-codegen

# 2. Run setup script
./scripts/setup_training_env.sh

# 3. Validate environment
python -m training.environment_setup
```

### GCP Setup (for actual training)

```bash
# 1. Set your GCP project ID
export GCP_PROJECT_ID=your-project-id

# 2. Create training instance
./scripts/gcp/create_training_instance.sh

# 3. Sync data to instance
./scripts/gcp/sync_data_to_gcp.sh

# 4. Connect to instance
./scripts/gcp/connect_to_instance.sh

# 5. On the instance, start training
cd ~/sql-codegen-slm
pip install -r training/requirements.txt
python -m training.train

# 6. After training, stop instance to save costs
./scripts/gcp/stop_instance.sh
```

## Configuration

### Training Configuration (`training/configs/mistral_lora_config.yaml`)

#### Model Settings
```yaml
model:
  name: "mistralai/Mistral-7B-v0.1"  # Base model
  max_seq_length: 2048               # Maximum input length
  torch_dtype: "float16"             # Precision
```

#### LoRA Settings
```yaml
lora:
  r: 16              # Rank - lower = fewer params, faster training
  lora_alpha: 32     # Scaling factor (alpha/r = 2 is standard)
  lora_dropout: 0.05 # Regularization
  target_modules:    # Which layers to adapt
    - "q_proj"       # Query projection
    - "k_proj"       # Key projection
    - "v_proj"       # Value projection
    - "o_proj"       # Output projection
    - "gate_proj"    # MLP gate
    - "up_proj"      # MLP up projection
    - "down_proj"    # MLP down projection
```

#### Quantization Settings
```yaml
quantization:
  load_in_4bit: true              # Enable 4-bit quantization
  bnb_4bit_compute_dtype: "float16"
  bnb_4bit_use_double_quant: true # Double quantization for more savings
  bnb_4bit_quant_type: "nf4"      # NormalFloat4 - best for LLMs
```

#### Training Hyperparameters
```yaml
training:
  num_train_epochs: 3
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4   # Effective batch = 4 * 4 = 16
  learning_rate: 0.0002            # 2e-4, standard for LoRA
  warmup_ratio: 0.03               # 3% warmup
  lr_scheduler_type: "cosine"
```

### Understanding the Configuration

| Parameter | Value | Explanation |
|-----------|-------|-------------|
| `lora.r` | 16 | LoRA rank - controls adapter size |
| `lora_alpha/r` | 2 | Scaling ratio - controls adaptation strength |
| `batch_size × grad_accum` | 16 | Effective batch size |
| `learning_rate` | 2e-4 | Standard for LoRA fine-tuning |
| `warmup_ratio` | 0.03 | 3% of steps for LR warmup |

## Memory Estimation

With 4-bit quantization and LoRA:

| Component | Memory |
|-----------|--------|
| Base model (4-bit) | 3.5 GB |
| LoRA adapters | 0.1 GB |
| Optimizer states | 2.0 GB |
| Activations | 8.0 GB |
| Gradient accumulation | 4.0 GB |
| Buffer | 2.0 GB |
| **Total** | **~18 GB** |

This fits comfortably in an A100 40GB GPU.

## GCP Setup Walkthrough

### 1. Prerequisites

```bash
# Install Google Cloud SDK
# https://cloud.google.com/sdk/docs/install

# Authenticate
gcloud auth login

# Set project
gcloud config set project YOUR_PROJECT_ID
```

### 2. Create Instance

```bash
export GCP_PROJECT_ID=your-project-id
./scripts/gcp/create_training_instance.sh
```

This creates:
- n1-highmem-8 machine (8 vCPUs, 52GB RAM)
- NVIDIA A100 40GB GPU
- 200GB SSD boot disk
- Deep Learning VM with CUDA 12.1

### 3. Sync Data

```bash
./scripts/gcp/sync_data_to_gcp.sh
```

This copies:
- Training data (train/val/test JSONL files)
- Configuration files
- Training code

### 4. Connect and Train

```bash
./scripts/gcp/connect_to_instance.sh
```

This opens SSH with port forwarding for:
- TensorBoard: http://localhost:6006
- Jupyter: http://localhost:8888

On the instance:
```bash
cd ~/sql-codegen-slm
pip install -r training/requirements.txt
python -m training.train
```

### 5. Monitor Training

In a new terminal:
```bash
# TensorBoard is available at http://localhost:6006
# Weights & Biases dashboard at https://wandb.ai
```

### 6. Stop Instance

```bash
./scripts/gcp/stop_instance.sh
```

**Important**: Always stop the instance when not training to save costs!

## Cost Optimization

### Tips to Reduce Costs

1. **Use preemptible instances**: 60-80% cheaper but can be terminated
2. **Stop instance when not training**: Saves ~$4/hour
3. **Use T4 GPU**: Slower but cheaper ($1.50/hour vs $4.41/hour)
4. **Reduce epochs**: Start with 1 epoch to validate setup

### Cost Breakdown

| Resource | Cost/Hour |
|----------|-----------|
| n1-highmem-8 | $0.47 |
| A100 40GB | $3.67 |
| 200GB SSD | $0.27 |
| **Total** | **$4.41** |

### Estimated Total Cost

| Scenario | Time | Cost |
|----------|------|------|
| Full training (3 epochs) | 8-12 hours | $35-53 |
| Quick test (1 epoch) | 3-4 hours | $13-18 |
| Development/debugging | 1-2 hours | $4-9 |

## Troubleshooting

### Package Installation Fails

```bash
# Update pip
pip install --upgrade pip

# Install packages one by one
pip install torch==2.1.2
pip install transformers==4.36.2
# ... etc
```

### CUDA Not Available

```bash
# Check CUDA installation
nvidia-smi

# Check PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

### Out of Memory

1. Reduce batch size: `per_device_train_batch_size: 2`
2. Increase gradient accumulation: `gradient_accumulation_steps: 8`
3. Enable gradient checkpointing: `gradient_checkpointing: true`

### GCP Quota Issues

```bash
# Check quotas
gcloud compute regions describe us-central1

# Request quota increase at:
# https://console.cloud.google.com/iam-admin/quotas
```

### wandb Login

```bash
# Create account at https://wandb.ai
wandb login
# Enter your API key
```

## Next Steps

After environment setup:

1. **Review configuration**: `training/configs/mistral_lora_config.yaml`
2. **Run training**: `python -m training.train`
3. **Monitor progress**: TensorBoard or wandb dashboard
4. **Evaluate model**: `python -m training.evaluate`
