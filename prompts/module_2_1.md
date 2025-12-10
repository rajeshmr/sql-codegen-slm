## Cost Comparison: Colab Pro+ vs GCP

| Factor | GCP A100 | Colab Pro+ |
|--------|----------|------------|
| **Upfront cost** | $0 | $58.99/month |
| **Cost per training run** | $35-53 (8-12 hours) | Included in subscription |
| **Multiple runs** | $35 each | Free after subscription |
| **Best for** | One-time training | Learning/experimentation |

**For your learning project:**
- You'll likely do 3-5 training runs (tuning hyperparameters, fixing bugs, etc.)
- GCP: 5 runs × $40 = **$200**
- Colab Pro+: **$58.99** (covers unlimited runs for 1 month)

**You're saving $140+** 🎉

---

## Module 2.1 REVISED: Colab Pro+ Setup

Let me give you an **updated prompt** that replaces GCP with Colab Pro+:

---

# MODULE 2.1 (REVISED): Training Environment Setup for Colab Pro+

**Objective:** Configure training environment optimized for Colab Pro+ with all dependencies and training configurations

---

## PROMPT FOR AI IDE (Windsurf/Claude):

### Context
You are working on sql-codegen-slm project. We have 8,234 training examples ready for fine-tuning Mistral-7B. Training will run on Google Colab Pro+ which provides A100 GPU access. We need to set up the environment for Colab with proper data upload handling, session management, and checkpoint saving to Google Drive.

### Task: Setup Training Infrastructure for Colab Pro+

**Location:** `training/requirements.txt`

(Keep same as before - no changes needed)

**Location:** `training/configs/mistral_lora_config.yaml`

Update with Colab-specific paths:
```yaml
# Model Configuration
model:
  name: "mistralai/Mistral-7B-v0.1"
  max_seq_length: 2048
  torch_dtype: "float16"
  
# LoRA Configuration
lora:
  r: 16
  lora_alpha: 32
  lora_dropout: 0.05
  target_modules:
    - "q_proj"
    - "k_proj"
    - "v_proj"
    - "o_proj"
    - "gate_proj"
    - "up_proj"
    - "down_proj"
  bias: "none"
  task_type: "CAUSAL_LM"

# Quantization Configuration
quantization:
  load_in_4bit: true
  bnb_4bit_compute_dtype: "float16"
  bnb_4bit_use_double_quant: true
  bnb_4bit_quant_type: "nf4"

# Training Configuration (optimized for Colab)
training:
  output_dir: "/content/drive/MyDrive/sql-codegen-models"  # Save to Google Drive
  num_train_epochs: 3
  per_device_train_batch_size: 4
  per_device_eval_batch_size: 4
  gradient_accumulation_steps: 4
  gradient_checkpointing: true
  optim: "paged_adamw_32bit"
  learning_rate: 0.0002
  weight_decay: 0.001
  warmup_ratio: 0.03
  lr_scheduler_type: "cosine"
  max_grad_norm: 0.3
  fp16: false
  bf16: true
  logging_steps: 10
  save_strategy: "steps"              # Changed to steps for Colab
  save_steps: 500                     # Save every 500 steps
  evaluation_strategy: "steps"        # Changed to steps
  eval_steps: 500                     # Eval every 500 steps
  save_total_limit: 2                 # Only keep 2 checkpoints (save space)
  load_best_model_at_end: true
  metric_for_best_model: "eval_loss"
  greater_is_better: false
  report_to: ["tensorboard"]          # Removed wandb for simplicity
  remove_unused_columns: false
  
# Data Configuration (Colab paths)
data:
  train_file: "/content/data/train_postgres.jsonl"
  val_file: "/content/data/val_postgres.jsonl"
  test_file: "/content/data/test_postgres.jsonl"
  text_column: "text"

# Logging
logging:
  project_name: "sql-codegen-slm"
  run_name: "mistral-7b-lora-sql-colab"
  log_dir: "/content/drive/MyDrive/sql-codegen-logs"
  tensorboard_dir: "/content/drive/MyDrive/sql-codegen-tensorboard"
```

**Location:** `training/configs/colab_setup.yaml`

Create Colab-specific configuration:
```yaml
# Colab Pro+ Configuration

subscription:
  tier: "Colab Pro+"
  cost_per_month: "$58.99"
  compute_units: 600
  gpu_priority: "High (A100 access)"
  background_execution: true
  max_runtime: "24 hours"

# Compute estimates
compute:
  expected_gpu: "Tesla A100 (40GB)"
  fallback_gpu: "Tesla V100 (16GB)"
  training_time_hours: 8-12
  compute_units_used: "~80-120 units per run"
  
# Session management
session:
  auto_disconnect: "~12 hours idle"
  reconnection: "Resume from checkpoint"
  checkpoint_frequency: "Every 500 steps"
  
# Google Drive integration
drive:
  mount_path: "/content/drive"
  model_save_path: "/content/drive/MyDrive/sql-codegen-models"
  checkpoint_backup: true
  
# Data handling
data:
  upload_method: "Google Drive or wget"
  size_limit: "200 GB available"
  local_path: "/content/data"
```

**Location:** `training/colab_setup.py`

Create Python module for Colab environment setup:

1. **Main function: setup_colab_environment()**
   - Checks if running in Colab
   - Mounts Google Drive
   - Creates necessary directories on Drive
   - Clones git repository or syncs code
   - Uploads data files from Drive or downloads from GitHub
   - Installs required packages
   - Verifies GPU allocation (checks for A100)
   - Sets up tensorboard in background
   - Returns environment status

2. **Helper function: mount_google_drive()**
   - Mounts Google Drive at /content/drive
   - Verifies mount successful
   - Creates project directories if not exist
   - Returns mount status

3. **Helper function: check_gpu()**
   - Detects GPU type (A100, V100, T4)
   - Checks GPU memory
   - Warns if got T4 instead of A100 (Colab can allocate different GPUs)
   - Returns GPU info

4. **Helper function: upload_data_files()**
   - Checks if data already in Drive
   - If not, provides instructions to upload or download from GitHub Release
   - Copies data to /content/data/ (local for faster access)
   - Verifies all files present
   - Returns data status

5. **Helper function: setup_checkpointing()**
   - Creates checkpoint directory on Drive
   - Tests write permissions
   - Sets up auto-save every 500 steps
   - Returns checkpoint config

**Location:** `notebooks/train_colab.ipynb`

Create Colab training notebook with cells:

**Cell 1: Check GPU**
```python
!nvidia-smi
# Should show A100 or V100
```

**Cell 2: Mount Drive & Setup**
```python
from google.colab import drive
drive.mount('/content/drive')

!git clone https://github.com/YOUR_USERNAME/sql-codegen-slm.git
%cd sql-codegen-slm

!pip install -r training/requirements.txt
```

**Cell 3: Upload Data**
```python
# Option 1: Data already in Drive
!cp -r /content/drive/MyDrive/sql-codegen-data/* data/processed/

# Option 2: Download from GitHub Release
!wget https://github.com/YOUR_USERNAME/sql-codegen-slm/releases/download/v0.1/data.zip
!unzip data.zip -d data/processed/
```

**Cell 4: Verify Environment**
```python
from training.colab_setup import setup_colab_environment
setup_colab_environment()
```

**Cell 5: Start Training**
```python
!python -m training.train --config training/configs/mistral_lora_config.yaml
```

**Cell 6: Monitor with TensorBoard**
```python
%load_ext tensorboard
%tensorboard --logdir /content/drive/MyDrive/sql-codegen-tensorboard
```

**Location:** `docs/training_colab_setup.md`

Create comprehensive Colab documentation:

Document:
- Why Colab Pro+ for this project
- Step-by-step setup guide with screenshots
- How to upload data (3 methods: Drive, GitHub Release, wget)
- GPU allocation tips (how to get A100)
- Session management (handling disconnects)
- Checkpoint recovery process
- Cost breakdown and unit usage
- Troubleshooting common Colab issues

**Location:** `scripts/prepare_data_for_upload.sh`

Create script to package data for upload:

Script should:
1. Compress processed data files into data.zip
2. Check file size (should be <500MB)
3. Generate checksum
4. Print upload instructions (upload to Drive or GitHub Release)
5. Make executable

**Update:** `tests/training/test_environment_setup.py`

Add Colab-specific tests:
1. Test detect_colab() function
2. Test Google Drive mount simulation
3. Test GPU detection logic
4. Test checkpoint path creation
5. Test data file verification

**Update:** `README.md`

Replace GCP section with Colab section:

```markdown
## Training Setup (Colab Pro+)

### Prerequisites
1. Google Colab Pro+ subscription ($58.99/month)
2. Google Drive with 10GB free space

### Quick Start
1. Open `notebooks/train_colab.ipynb` in Colab
2. Run all cells in order
3. Training will save to your Google Drive
4. Estimated time: 8-12 hours

### Cost
- Subscription: $58.99/month (unlimited training runs)
- Compute units: ~80-120 units per run (out of 600/month)
- Can do 5-7 full training runs per month

### Advantages over GCP
- ✅ Cheaper for multiple experiments
- ✅ No setup complexity
- ✅ Automatic checkpoint saving to Drive
- ✅ TensorBoard integration
- ✅ Can pause/resume easily
```

### Testing Requirements:

After creation:
1. Running setup cells in Colab notebook works
2. Environment validation passes
3. Data upload methods documented
4. Checkpoint saving to Drive works
5. All Colab-specific tests pass

### Commit Message:
"feat(training): Setup training environment for Colab Pro+ - Module 2.1 (Revised)"

---

## ✅ MODULE 2.1 COMPLETION CHECKLIST (Colab Version)

**After running the AI IDE prompt, verify the following:**

### Files Created:
- [ ] `training/requirements.txt` exists (same as before)
- [ ] `training/configs/mistral_lora_config.yaml` updated with Colab paths
- [ ] `training/configs/colab_setup.yaml` exists
- [ ] `training/colab_setup.py` exists
- [ ] `notebooks/train_colab.ipynb` exists
- [ ] `docs/training_colab_setup.md` exists
- [ ] `scripts/prepare_data_for_upload.sh` exists and is executable
- [ ] README.md updated with Colab instructions

### Colab Notebook Structure:
- [ ] Cell 1: GPU check
- [ ] Cell 2: Mount Drive & clone repo
- [ ] Cell 3: Install dependencies
- [ ] Cell 4: Upload/verify data
- [ ] Cell 5: Environment validation
- [ ] Cell 6: Start training
- [ ] Cell 7: TensorBoard monitoring
- [ ] All cells have clear markdown explanations

### Configuration Validation:
- [ ] `mistral_lora_config.yaml` has Drive paths (/content/drive/MyDrive/...)
- [ ] Checkpoint saving set to "steps" (not epochs) for Colab
- [ ] save_steps: 500 (saves frequently in case of disconnect)
- [ ] save_total_limit: 2 (saves Drive space)

### Data Upload Preparation:
- [ ] Run `./scripts/prepare_data_for_upload.sh`
- [ ] Creates `data.zip` with train/val/test files
- [ ] File size reasonable (<500MB)
- [ ] Instructions printed for upload options

### Colab-Specific Features:
- [ ] Google Drive mounting implemented
- [ ] GPU detection warns if not A100
- [ ] Checkpoint recovery documented
- [ ] Session disconnect handling explained
- [ ] TensorBoard background launch

### Cost Understanding:
- [ ] Colab Pro+ costs $58.99/month
- [ ] Gets 600 compute units
- [ ] Training uses ~80-120 units (8-12 hours)
- [ ] Can do 5-7 training runs per month
- [ ] Much cheaper than GCP for experimentation ($58.99 vs $200+ for 5 runs)

### Understanding Check (Learning):
- [ ] You understand why Colab Pro+ is better for learning (fixed monthly cost)
- [ ] You know how to handle session disconnects (checkpoints every 500 steps)
- [ ] You understand Drive integration (models saved permanently)
- [ ] You know Colab may give different GPUs (A100 preferred, V100 okay, T4 slower)

### Git Verification:
- [ ] `notebooks/` directory tracked
- [ ] `train_colab.ipynb` is tracked
- [ ] Documentation updated
- [ ] No data files tracked (too large)
- [ ] Commit message follows convention

---

**Key Advantages of Your Decision:**

1. **Cost:** $58.99/month vs $35-53 per run on GCP
2. **Simplicity:** No GCP setup, no billing surprises
3. **Learning-friendly:** Can experiment freely within monthly budget
4. **Background execution:** Colab Pro+ keeps running when you close browser
5. **Drive integration:** Models auto-saved, no manual download needed