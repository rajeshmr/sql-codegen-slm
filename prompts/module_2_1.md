# MODULE 2.1: Training Environment Setup

**Objective:** Configure training environment with all dependencies, create training configuration files, and setup GCP compute for fine-tuning

---

## PROMPT FOR AI IDE (Windsurf/Claude):

### Context
You are working on sql-codegen-slm project. We have 8,234 training examples ready for fine-tuning Mistral-7B. We need to set up the complete training environment including: PyTorch, Transformers, PEFT (for LoRA), training configurations, and GCP compute setup scripts. Training will use 4-bit quantization with LoRA for memory efficiency.

### Task: Setup Training Infrastructure

**Location:** `training/requirements.txt`

**Create training-specific requirements file:**
```
# Core ML libraries
torch==2.1.2
transformers==4.36.2
peft==0.7.1
accelerate==0.25.0
bitsandbytes==0.41.3
trl==0.7.10

# Data handling
datasets==2.15.0
tokenizers==0.15.0

# Training utilities
wandb==0.16.1
tensorboard==2.15.1
scipy==1.11.4

# Evaluation
scikit-learn==1.3.2
rouge-score==0.1.2

# Google Cloud (for GCP deployment)
google-cloud-storage==2.10.0
google-auth==2.25.2
```

**Location:** `training/configs/mistral_lora_config.yaml`

**Create training configuration file:**
```yaml
# Model Configuration
model:
  name: "mistralai/Mistral-7B-v0.1"
  max_seq_length: 2048
  torch_dtype: "float16"
  
# LoRA Configuration
lora:
  r: 16                          # LoRA rank
  lora_alpha: 32                 # LoRA scaling factor
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

# Training Configuration
training:
  output_dir: "./training/models/mistral-sql-postgres"
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
  save_strategy: "epoch"
  evaluation_strategy: "epoch"
  save_total_limit: 3
  load_best_model_at_end: true
  metric_for_best_model: "eval_loss"
  greater_is_better: false
  report_to: ["tensorboard", "wandb"]
  remove_unused_columns: false
  
# Data Configuration
data:
  train_file: "data/processed/train_postgres.jsonl"
  val_file: "data/processed/val_postgres.jsonl"
  test_file: "data/processed/test_postgres.jsonl"
  text_column: "text"
  max_samples_train: null      # null = use all
  max_samples_val: null

# Logging
logging:
  project_name: "sql-codegen-slm"
  run_name: "mistral-7b-lora-sql"
  log_dir: "./training/logs"
  tensorboard_dir: "./training/tensorboard"
```

**Location:** `training/configs/gcp_compute.yaml`

**Create GCP compute configuration:**
```yaml
# GCP Compute Configuration for Training

compute:
  project_id: "YOUR_PROJECT_ID"        # Replace with your GCP project
  zone: "us-central1-a"
  machine_type: "n1-highmem-8"         # 8 vCPUs, 52GB RAM
  gpu_type: "nvidia-tesla-a100"        # A100 40GB
  gpu_count: 1
  boot_disk_size_gb: 200
  boot_disk_type: "pd-ssd"

# Container/VM Configuration  
vm:
  image_project: "ml-images"
  image_family: "common-cu121-ubuntu-2204"  # CUDA 12.1, Ubuntu 22.04
  python_version: "3.10"
  
# Cost estimation (per hour)
cost:
  machine: "$0.47/hour"
  gpu_a100: "$3.67/hour"
  disk: "$0.27/hour"
  total_estimated: "$4.41/hour"
  
# Training estimates
estimates:
  training_time_hours: 8-12
  total_cost: "$35-53"
```

**Location:** `training/environment_setup.py`

**Create Python module for environment validation:**

Create a module that:

1. **Main function: validate_environment()**
   - Checks Python version (3.10)
   - Verifies conda environment is activated
   - Checks CUDA availability and version
   - Validates all required packages installed with correct versions
   - Checks GPU memory available
   - Verifies data files exist
   - Prints comprehensive environment report
   - Returns True if all checks pass, False otherwise

2. **Helper function: check_gpu_specs()**
   - Detects GPU type (A100, V100, T4, etc.)
   - Checks GPU memory (should be 40GB+ for A100)
   - Checks CUDA version (should be 11.8+)
   - Returns GPU specifications dictionary

3. **Helper function: check_data_files()**
   - Verifies train/val/test files exist
   - Checks file sizes are reasonable
   - Counts examples in each file
   - Returns data file statistics

4. **Helper function: estimate_memory_requirements()**
   - Calculates expected memory usage:
     - Base model (4-bit): ~3.5GB
     - LoRA adapters: ~100MB
     - Optimizer states: ~2GB
     - Activations (batch_size=4): ~8GB
     - Gradient accumulation overhead: ~4GB
     - Total estimated: ~18GB
   - Compares with available GPU memory
   - Returns memory report with warnings if tight

**Location:** `scripts/setup_training_env.sh`

**Create bash script:**

Script should:
1. Activate conda environment
2. Install training requirements: `pip install -r training/requirements.txt`
3. Run environment validation: `python -m training.environment_setup`
4. Create necessary directories (logs, models, tensorboard)
5. Check if wandb is configured (provide setup instructions if not)
6. Print summary and next steps
7. Make executable

**Location:** `scripts/gcp/create_training_instance.sh`

**Create GCP instance creation script:**

Script should:
1. Check if gcloud CLI is installed and authenticated
2. Read configuration from `training/configs/gcp_compute.yaml`
3. Create GCP Compute Engine instance with:
   - Specified machine type and GPU
   - Deep Learning VM image (with CUDA pre-installed)
   - Sufficient disk space
   - Appropriate firewall rules
4. Install Python dependencies on instance
5. Clone git repository to instance
6. Setup SSH access
7. Print connection instructions
8. Make executable

**Location:** `scripts/gcp/sync_data_to_gcp.sh`

**Create data sync script:**

Script should:
1. Check if GCP instance exists and is running
2. Use `gcloud compute scp` to copy:
   - Processed data files (train/val/test)
   - Training configurations
   - Code files
3. Exclude raw data (too large)
4. Show progress bar
5. Verify files transferred successfully
6. Make executable

**Location:** `scripts/gcp/connect_to_instance.sh`

**Create SSH connection helper:**

Script should:
1. Check instance status
2. Start instance if stopped
3. Open SSH connection with port forwarding for:
   - TensorBoard (port 6006)
   - Jupyter (port 8888, optional)
4. Print connection details
5. Make executable

**Location:** `scripts/gcp/stop_instance.sh`

**Create instance stop script:**

Script should:
1. Check instance status
2. Stop instance (to save costs when not training)
3. Confirm stop
4. Print cost saved message
5. Make executable

**Create test file:** `tests/training/test_environment_setup.py`

Create pytest tests that:
1. Test validate_environment() runs without errors
2. Test check_gpu_specs() returns proper structure
3. Test check_data_files() finds all required files
4. Test memory estimation calculations
5. Test configuration YAML files are valid
6. Test all required packages are in requirements.txt
7. Test GCP config has required fields

**Create documentation:** `docs/training_setup.md`

Document:
- Hardware requirements (GPU, RAM, disk)
- Installation instructions (local and GCP)
- Configuration options explained
- GCP setup walkthrough with screenshots/commands
- Cost estimation and optimization tips
- Troubleshooting common issues

### Testing Requirements:

After creation:
1. Running `./scripts/setup_training_env.sh` installs all dependencies
2. Running `python -m training.environment_setup` passes all checks
3. All configuration YAML files are valid
4. Running `pytest tests/training/test_environment_setup.py -v` passes
5. GCP scripts are created and executable

### Update README.md:

Add "Training Setup" section:
```markdown
## Training Setup

### Local Setup (for testing)
```bash
./scripts/setup_training_env.sh
python -m training.environment_setup
```

### GCP Setup (for actual training)
```bash
# 1. Create training instance
./scripts/gcp/create_training_instance.sh

# 2. Sync data
./scripts/gcp/sync_data_to_gcp.sh

# 3. Connect to instance
./scripts/gcp/connect_to_instance.sh

# 4. After training, stop instance
./scripts/gcp/stop_instance.sh
```

**Estimated cost:** $35-53 for full training run (8-12 hours)
```

### Commit Message:
"feat(training): Setup training environment and GCP infrastructure - Module 2.1"

---

## ✅ MODULE 2.1 COMPLETION CHECKLIST

**After running the AI IDE prompt, verify the following:**

### Files Created:
- [ ] `training/requirements.txt` exists
- [ ] `training/configs/mistral_lora_config.yaml` exists
- [ ] `training/configs/gcp_compute.yaml` exists
- [ ] `training/environment_setup.py` exists
- [ ] `training/__init__.py` exists
- [ ] `scripts/setup_training_env.sh` exists and is executable
- [ ] `scripts/gcp/create_training_instance.sh` exists and is executable
- [ ] `scripts/gcp/sync_data_to_gcp.sh` exists and is executable
- [ ] `scripts/gcp/connect_to_instance.sh` exists and is executable
- [ ] `scripts/gcp/stop_instance.sh` exists and is executable
- [ ] `tests/training/test_environment_setup.py` exists
- [ ] `tests/training/__init__.py` exists
- [ ] `docs/training_setup.md` exists
- [ ] README.md updated with training setup section

### Directory Structure:
- [ ] `training/configs/` directory created
- [ ] `training/logs/` directory created (or will be created by script)
- [ ] `training/models/` directory created
- [ ] `training/tensorboard/` directory created
- [ ] `scripts/gcp/` directory created

### Requirements Validation:
- [ ] Open `training/requirements.txt`
- [ ] Contains torch, transformers, peft, accelerate
- [ ] Contains bitsandbytes for 4-bit quantization
- [ ] Contains trl for training
- [ ] Contains wandb and tensorboard for logging
- [ ] Contains google-cloud-storage for GCP
- [ ] Version numbers specified

### Configuration Validation:
- [ ] Open `mistral_lora_config.yaml`
- [ ] Has model configuration (Mistral-7B)
- [ ] Has LoRA config (r=16, alpha=32, dropout=0.05)
- [ ] Has quantization config (4-bit settings)
- [ ] Has training hyperparameters (epochs, batch size, learning rate)
- [ ] Has data file paths
- [ ] Has logging configuration

GCP Configuration:
- [ ] Open `gcp_compute.yaml`
- [ ] Has machine type (n1-highmem-8)
- [ ] Has GPU config (A100)
- [ ] Has disk configuration
- [ ] Has cost estimates
- [ ] Has training time estimates

### Local Environment Setup:
- [ ] Run `./scripts/setup_training_env.sh`
- [ ] Script installs dependencies without errors
- [ ] Creates necessary directories
- [ ] Shows completion message

### Environment Validation:
- [ ] Run `python -m training.environment_setup`
- [ ] Checks Python version (should show 3.10)
- [ ] Checks if conda env activated
- [ ] Lists all required packages and versions
- [ ] Checks for CUDA/GPU (will fail on Mac, that's okay for now)
- [ ] Verifies data files exist
- [ ] Shows memory estimates

### GCP Scripts Validation:
- [ ] Check `scripts/gcp/create_training_instance.sh` exists
- [ ] Contains gcloud commands for instance creation
- [ ] Specifies A100 GPU
- [ ] Has error handling
- [ ] Check `scripts/gcp/sync_data_to_gcp.sh` exists
- [ ] Uses gcloud compute scp for file transfer
- [ ] Check `scripts/gcp/connect_to_instance.sh` exists
- [ ] Opens SSH with port forwarding
- [ ] Check `scripts/gcp/stop_instance.sh` exists
- [ ] Stops instance to save costs

### Functional Tests:
- [ ] Run `pytest tests/training/test_environment_setup.py -v`
- [ ] All tests pass
- [ ] Configuration YAML validation works
- [ ] Data file checks work
- [ ] Memory estimation calculates correctly

### Documentation Check:
- [ ] Open `docs/training_setup.md`
- [ ] Has hardware requirements listed
- [ ] Has installation instructions
- [ ] Explains configuration options
- [ ] Has GCP setup walkthrough
- [ ] Has cost breakdown
- [ ] Has troubleshooting section

### Configuration Understanding:

Check `mistral_lora_config.yaml`:
- [ ] Understand LoRA rank (r=16): Lower = less parameters, faster but less expressive
- [ ] Understand alpha/r ratio (32/16=2): Controls adaptation strength
- [ ] Understand batch_size × gradient_accumulation (4×4=16 effective batch)
- [ ] Understand learning rate (2e-4): Standard for LoRA fine-tuning
- [ ] Understand warmup_ratio (0.03): 3% of steps for learning rate warmup

### Cost Awareness:
- [ ] GCP A100 costs ~$4.41/hour
- [ ] Training estimate: 8-12 hours
- [ ] Total cost estimate: $35-53
- [ ] Stopping instance when not training saves money
- [ ] Understand this is cheaper than Mac Studio ($3,000+)

### Understanding Check (Learning):
- [ ] You understand why we use 4-bit quantization (memory efficiency)
- [ ] You know what LoRA is (Low-Rank Adaptation - efficient fine-tuning)
- [ ] You understand gradient accumulation (simulates larger batches)
- [ ] You know why we need A100 GPU (40GB VRAM, good for 7B model)
- [ ] You understand the cost tradeoff (cloud vs local hardware)

### Git Verification:
- [ ] `training/logs/` NOT tracked (.gitignore)
- [ ] `training/models/` NOT tracked (.gitignore)
- [ ] `training/tensorboard/` NOT tracked (.gitignore)
- [ ] Configuration files ARE tracked (YAML files)
- [ ] Code files ARE tracked (Python, bash scripts)
- [ ] Documentation IS tracked
- [ ] Commit message follows convention

### What You Should Have Learned:
1. **LoRA Fine-tuning**: Efficient method to adapt large models with minimal parameters
2. **4-bit Quantization**: Reduces model memory from 14GB to 3.5GB
3. **Training Configuration**: Hyperparameters, batch sizes, learning rates
4. **GCP Compute**: Setting up cloud GPU instances for ML training
5. **Cost Management**: Understanding cloud costs and optimization
6. **Environment Validation**: Importance of checking dependencies before training
7. **Configuration Files**: Using YAML for reproducible experiments

---

**TROUBLESHOOTING (if checks fail):**

❌ **Package installation fails:**
- Check Python version is 3.10
- Try updating pip: `pip install --upgrade pip`
- Install packages one by one to identify issue
- Check CUDA version if GPU-related packages fail

❌ **YAML syntax errors:**
- Validate YAML with: `python -c "import yaml; yaml.safe_load(open('file.yaml'))"`
- Check indentation (use spaces, not tabs)
- Check quotes around special characters

❌ **Environment validation fails:**
- If CUDA check fails on Mac: Expected, skip for now
- If data files not found: Check paths in config match actual locations
- If memory estimates wrong: Review calculation logic

❌ **GCP scripts fail:**
- Check gcloud CLI installed: `gcloud --version`
- Authenticate: `gcloud auth login`
- Set project: `gcloud config set project YOUR_PROJECT_ID`
- Check quotas: Some regions have limited A100 availability

---

**SAMPLE OUTPUT YOU SHOULD SEE:**

When running `./scripts/setup_training_env.sh`:
```
Setting up training environment...

Installing training dependencies...
✅ torch==2.1.2 installed
✅ transformers==4.36.2 installed
✅ peft==0.7.1 installed
✅ accelerate==0.25.0 installed
✅ bitsandbytes==0.41.3 installed
✅ trl==0.7.10 installed
... [more packages]

Creating directories...
✅ training/logs/
✅ training/models/
✅ training/tensorboard/

Validating environment...
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Python version: 3.10.13 ✅
Conda environment: sql-codegen ✅
CUDA available: No (Mac) ⚠️
Data files: All present ✅

Required packages:
✅ torch 2.1.2
✅ transformers 4.36.2
✅ peft 0.7.1
[... more packages]

Memory Estimates (for GCP A100):
Base model (4-bit): 3.5 GB
LoRA adapters: 0.1 GB
Optimizer states: 2.0 GB
Activations: 8.0 GB
Total estimated: ~18 GB / 40 GB available ✅

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ Environment setup complete!

Next steps:
1. Review training configuration: training/configs/mistral_lora_config.yaml
2. Setup GCP instance: ./scripts/gcp/create_training_instance.sh
3. Or proceed with local testing (limited without GPU)
```
