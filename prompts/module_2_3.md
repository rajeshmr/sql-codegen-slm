# MODULE 2.3: Training Validation & Smoke Testing

**Objective:** Test the complete training pipeline on a small dataset to verify everything works before running the full 8-12 hour training

---

## PROMPT FOR AI IDE (Windsurf/Claude):

### Context
You are working on sql-codegen-slm project. The training pipeline is implemented but untested. Before running the expensive 8-12 hour full training, we need to validate that everything works correctly with a small test run. This includes: data loading, model initialization, training loop, checkpoint saving to GCS, model inference, and checkpoint resumption.

### Task: Create Training Validation System

**Location:** `training/test_config.yaml`

Create test configuration (small-scale training):
```yaml
# Test Configuration - Small scale for validation
# Should complete in 10-15 minutes on A100

model:
  name: "mistralai/Mistral-7B-v0.1"
  max_seq_length: 512  # Shorter for faster testing
  torch_dtype: "float16"
  
lora:
  r: 8  # Smaller rank for faster testing
  lora_alpha: 16
  lora_dropout: 0.05
  target_modules:
    - "q_proj"
    - "v_proj"  # Only 2 modules for faster testing
  bias: "none"
  task_type: "CAUSAL_LM"

quantization:
  load_in_4bit: true
  bnb_4bit_compute_dtype: "float16"
  bnb_4bit_use_double_quant: true
  bnb_4bit_quant_type: "nf4"

training:
  output_dir: "gs://YOUR-BUCKET-NAME/sql-codegen-test-models"
  num_train_epochs: 1  # Just 1 epoch for testing
  per_device_train_batch_size: 2
  per_device_eval_batch_size: 2
  gradient_accumulation_steps: 2
  gradient_checkpointing: true
  optim: "paged_adamw_32bit"
  learning_rate: 0.0002
  max_steps: 20  # Only 20 steps for testing
  warmup_steps: 2
  logging_steps: 2
  save_strategy: "steps"
  save_steps: 10  # Save checkpoint at step 10
  evaluation_strategy: "steps"
  eval_steps: 10
  save_total_limit: 2
  load_best_model_at_end: true
  report_to: ["tensorboard"]
  
data:
  train_file: "/content/data/train_postgres.jsonl"
  val_file: "/content/data/val_postgres.jsonl"
  max_samples_train: 20  # Only use 20 training examples
  max_samples_val: 10    # Only use 10 validation examples
  
logging:
  project_name: "sql-codegen-slm-test"
  run_name: "test-run"
  log_dir: "gs://YOUR-BUCKET-NAME/sql-codegen-test-logs"
  tensorboard_dir: "gs://YOUR-BUCKET-NAME/sql-codegen-test-tensorboard"
```

**Location:** `training/validation.py`

Create validation module:

1. **Main function: validate_training_pipeline(config_path)**
   - Runs complete validation suite
   - Tests each component independently
   - Runs end-to-end smoke test
   - Reports results
   - Returns validation report dictionary

2. **Function: test_data_loading(config)**
   - Loads 20 training examples
   - Loads 10 validation examples
   - Verifies correct format
   - Checks tokenization produces correct shapes
   - Prints sample formatted example
   - Returns (success, details)

3. **Function: test_model_initialization(config)**
   - Loads Mistral-7B with 4-bit quantization
   - Applies LoRA adapters
   - Verifies model on correct device (cuda)
   - Checks trainable parameters count
   - Verifies model can do forward pass
   - Returns (success, model_info)

4. **Function: test_training_step(model, train_dataset, config)**
   - Runs 2 training steps
   - Verifies loss decreases or stays reasonable
   - Checks gradients are computed
   - Monitors GPU memory usage
   - Returns (success, training_metrics)

5. **Function: test_checkpoint_saving(trainer, output_dir)**
   - Saves a test checkpoint
   - Verifies checkpoint exists in GCS
   - Lists checkpoint files
   - Checks file sizes are reasonable
   - Returns (success, checkpoint_info)

6. **Function: test_checkpoint_loading(checkpoint_path, config)**
   - Loads model from checkpoint
   - Verifies weights loaded correctly
   - Checks optimizer state restored
   - Returns (success, load_info)

7. **Function: test_inference(model, tokenizer, test_prompt)**
   - Generates SQL from a test prompt
   - Verifies output is valid text
   - Checks generation completes without errors
   - Prints generated SQL
   - Returns (success, generated_text)

8. **Function: generate_validation_report(results)**
   - Compiles all test results
   - Prints formatted report
   - Identifies any failures
   - Provides recommendations
   - Saves report to file
   - Returns report dictionary

**Location:** `training/smoke_test.py`

Create smoke test script:

1. **Function: run_smoke_test(config_path)**
   - Runs minimal end-to-end training (20 steps)
   - Saves checkpoint at step 10
   - Simulates disconnect by stopping training
   - Resumes from checkpoint
   - Completes remaining 10 steps
   - Validates final model
   - Returns test results

2. **Function: test_full_workflow()**
   - Tests: Data → Model → Train → Save → Resume → Inference
   - Each step must pass for test to succeed
   - Prints progress with emojis
   - Returns (success, failure_point)

**Location:** `notebooks/validation_notebook.ipynb`

Create comprehensive validation notebook for Colab:

**Cell 1: Setup**
```python
# Check GPU
!nvidia-smi

# Clone repo and setup
!git clone https://github.com/YOUR_USERNAME/sql-codegen-slm.git
%cd sql-codegen-slm
!pip install -q -r training/requirements.txt
```

**Cell 2: Authenticate GCS**
```python
from google.colab import auth
auth.authenticate_user()

# Or service account
from google.colab import files
uploaded = files.upload()  # Upload service-account-key.json
import os
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'service-account-key.json'
```

**Cell 3: Download Test Data (20 examples)**
```python
from training.gcs_utils import download_data_from_gcs

# Download small subset for testing
download_data_from_gcs(
    gcs_bucket="YOUR-BUCKET-NAME",
    gcs_prefix="sql-codegen-data",
    local_dir="/content/data"
)

# Create small test files (20 train, 10 val)
import json

# Take first 20 examples from train
with open('/content/data/train_postgres.jsonl', 'r') as f:
    train_lines = [next(f) for _ in range(20)]

with open('/content/data/train_small.jsonl', 'w') as f:
    f.writelines(train_lines)

# Take first 10 examples from val
with open('/content/data/val_postgres.jsonl', 'r') as f:
    val_lines = [next(f) for _ in range(10)]
    
with open('/content/data/val_small.jsonl', 'w') as f:
    f.writelines(val_lines)

print("✅ Test data ready")
```

**Cell 4: Test Data Loading**
```python
from training.validation import test_data_loading
import yaml

with open('training/configs/test_config.yaml') as f:
    config = yaml.safe_load(f)

success, details = test_data_loading(config)

if success:
    print("✅ Data loading test passed")
    print(f"Train examples: {details['train_count']}")
    print(f"Val examples: {details['val_count']}")
else:
    print("❌ Data loading test failed")
    print(details['error'])
```

**Cell 5: Test Model Initialization**
```python
from training.validation import test_model_initialization

success, model_info = test_model_initialization(config)

if success:
    print("✅ Model initialization test passed")
    print(f"Device: {model_info['device']}")
    print(f"Total params: {model_info['total_params']:,}")
    print(f"Trainable params: {model_info['trainable_params']:,}")
    print(f"Trainable %: {model_info['trainable_pct']:.2f}%")
else:
    print("❌ Model initialization failed")
```

**Cell 6: Run Smoke Test (20 steps)**
```python
from training.smoke_test import run_smoke_test

print("🔥 Running smoke test (20 training steps)...")
print("This will take 5-10 minutes\n")

results = run_smoke_test('training/configs/test_config.yaml')

if results['success']:
    print("\n✅ SMOKE TEST PASSED!")
    print(f"Training loss: {results['train_loss']:.4f}")
    print(f"Checkpoint saved: {results['checkpoint_path']}")
    print(f"Model can generate SQL: {results['can_generate']}")
else:
    print(f"\n❌ SMOKE TEST FAILED at: {results['failure_point']}")
```

**Cell 7: Test Inference**
```python
from training.validation import test_inference
from transformers import AutoTokenizer

# Load model from test checkpoint
checkpoint = "gs://YOUR-BUCKET-NAME/sql-codegen-test-models/checkpoint-20"

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

test_prompt = """Database: ecommerce

Schema:
CREATE TABLE customers (customer_id SERIAL PRIMARY KEY, name VARCHAR(100));
CREATE TABLE orders (order_id SERIAL PRIMARY KEY, customer_id INTEGER REFERENCES customers(customer_id));

Question: Show all customers with their order counts

Generate the SQL query:"""

success, generated_sql = test_inference(
    model=None,  # Will load from checkpoint
    tokenizer=tokenizer,
    test_prompt=test_prompt
)

if success:
    print("✅ Inference test passed")
    print(f"\nGenerated SQL:\n{generated_sql}")
else:
    print("❌ Inference test failed")
```

**Cell 8: Full Validation Report**
```python
from training.validation import validate_training_pipeline

print("Running complete validation suite...\n")

report = validate_training_pipeline('training/configs/test_config.yaml')

print("\n" + "="*50)
print("VALIDATION REPORT")
print("="*50)

for test_name, result in report.items():
    status = "✅" if result['passed'] else "❌"
    print(f"{status} {test_name}: {result['message']}")

if all(r['passed'] for r in report.values()):
    print("\n🎉 ALL TESTS PASSED - READY FOR FULL TRAINING")
else:
    print("\n⚠️  SOME TESTS FAILED - FIX ISSUES BEFORE FULL TRAINING")
```

**Cell 9: GPU Memory Check**
```python
import torch

print("GPU Memory Usage:")
print(f"Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
print(f"Max allocated: {torch.cuda.max_memory_allocated(0) / 1e9:.2f} GB")

# Should be well under 40GB for A100
if torch.cuda.memory_allocated(0) / 1e9 < 20:
    print("✅ Memory usage looks good for full training")
else:
    print("⚠️  High memory usage - might need to reduce batch size")
```

**Cell 10: Cleanup Test Artifacts**
```python
# Optional: Delete test checkpoints from GCS to save space
!gsutil -m rm -r gs://YOUR-BUCKET-NAME/sql-codegen-test-models/
!gsutil -m rm -r gs://YOUR-BUCKET-NAME/sql-codegen-test-logs/

print("✅ Test artifacts cleaned up")
```

**Location:** `scripts/run_validation.sh`

Create validation launcher:

Script should:
1. Check if running in Colab
2. Verify GPU available
3. Run validation suite
4. Print summary
5. Exit with code 0 if all pass, 1 if any fail
6. Make executable

**Create test file:** `tests/training/test_validation.py`

Create pytest tests:
1. Test validation functions can be imported
2. Test test_config.yaml is valid
3. Test validation report generation
4. Test smoke test can be initialized (don't run full test)

**Update:** `README.md`

Add "Pre-Training Validation" section:
```markdown
## Pre-Training Validation

**IMPORTANT:** Run validation before full training to catch issues early.

### Quick Validation (10 minutes)
1. Open `notebooks/validation_notebook.ipynb` in Colab
2. Run all cells
3. Verify all tests pass ✅

### What Gets Validated
- ✅ Data loading (20 examples)
- ✅ Model initialization (4-bit + LoRA)
- ✅ Training loop (20 steps)
- ✅ Checkpoint saving to GCS
- ✅ Checkpoint loading
- ✅ Model inference (SQL generation)
- ✅ GPU memory usage (<20GB)

### Expected Results
- All tests pass ✅
- Training loss decreases (3.5 → 1.5 in 20 steps)
- Model generates valid SQL
- Checkpoints save to GCS
- Memory usage <20GB

### If Validation Fails
1. Check error messages carefully
2. Verify GCS authentication
3. Check GPU allocation (need A100 or V100)
4. Review training/logs/ for details
5. Ask for help with specific error

### After Validation Passes
✅ Ready for full training! Proceed to Module 2.4
```

### Commit Message:
"feat(training): Add validation and smoke testing for training pipeline - Module 2.3"

---

## ✅ MODULE 2.3 COMPLETION CHECKLIST

**After running the AI IDE prompt, verify the following:**

### Files Created:
- [ ] `training/test_config.yaml` exists
- [ ] `training/validation.py` exists
- [ ] `training/smoke_test.py` exists
- [ ] `notebooks/validation_notebook.ipynb` exists
- [ ] `scripts/run_validation.sh` exists and is executable
- [ ] `tests/training/test_validation.py` exists
- [ ] README.md updated with validation instructions

### Configuration Check:
- [ ] `test_config.yaml` has max_steps: 20 (not full training)
- [ ] Has max_samples_train: 20 (small dataset)
- [ ] Has max_samples_val: 10
- [ ] Has save_steps: 10 (one checkpoint mid-training)
- [ ] GCS paths point to test directories (not main model dir)

### Validation Module Check:

**validation.py:**
- [ ] Has validate_training_pipeline function
- [ ] Has test_data_loading function
- [ ] Has test_model_initialization function
- [ ] Has test_training_step function
- [ ] Has test_checkpoint_saving function
- [ ] Has test_checkpoint_loading function
- [ ] Has test_inference function
- [ ] Has generate_validation_report function

**smoke_test.py:**
- [ ] Has run_smoke_test function
- [ ] Tests train → save → resume → complete workflow
- [ ] Returns success/failure status

### Notebook Structure:
- [ ] 10 cells total
- [ ] Cell 1: Setup and GPU check
- [ ] Cell 2: GCS authentication
- [ ] Cell 3: Download test data
- [ ] Cell 4-7: Individual component tests
- [ ] Cell 8: Full validation report
- [ ] Cell 9: GPU memory check
- [ ] Cell 10: Cleanup
- [ ] All cells have markdown explanations

### Testing Requirements (Local):
- [ ] Run `pytest tests/training/test_validation.py -v`
- [ ] All tests pass
- [ ] Can import validation functions
- [ ] test_config.yaml is valid YAML

---

## 🚀 NOW RUN VALIDATION IN COLAB

**Before proceeding to Module 2.4, you MUST:**

1. **Open `notebooks/validation_notebook.ipynb` in Colab Pro+**
2. **Run all 10 cells**
3. **Verify all tests pass**

---

### Expected Validation Output:

```
Cell 1: Setup
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GPU: Tesla A100-SXM4-40GB ✅
CUDA Version: 12.2 ✅

Cell 3: Test Data
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Downloaded 20 train examples ✅
Downloaded 10 val examples ✅

Cell 4: Data Loading Test
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ Data loading test passed
Train examples: 20
Val examples: 10
Sample format:
<s>[INST] You are a SQL expert...
[Schema and question here]
[/INST] SELECT ... </s>

Cell 5: Model Initialization Test
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Loading Mistral-7B with 4-bit quantization...
Applying LoRA adapters...
✅ Model initialization test passed
Device: cuda:0
Total params: 7,241,732,096
Trainable params: 20,971,520
Trainable %: 0.29%

Cell 6: Smoke Test (20 steps)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔥 Running smoke test...

Step 2/20:  Loss=3.456
Step 4/20:  Loss=2.891
Step 6/20:  Loss=2.234
Step 8/20:  Loss=1.876
Step 10/20: Loss=1.543 [Checkpoint saved] ✅
Step 12/20: Loss=1.321
Step 14/20: Loss=1.198
Step 16/20: Loss=1.087
Step 18/20: Loss=0.976
Step 20/20: Loss=0.891 [Training complete] ✅

✅ SMOKE TEST PASSED!
Training loss: 0.891
Checkpoint saved: gs://bucket/test-models/checkpoint-10
Model can generate SQL: True

Cell 7: Inference Test
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ Inference test passed

Generated SQL:
SELECT c.customer_id, c.name, COUNT(o.order_id) as order_count
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id
GROUP BY c.customer_id, c.name;

Cell 8: Full Validation Report
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VALIDATION REPORT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ data_loading: 20 train, 10 val examples loaded
✅ model_init: Mistral-7B loaded with LoRA
✅ training_step: Loss decreased from 3.5 to 2.1
✅ checkpoint_save: Checkpoint saved to GCS
✅ checkpoint_load: Checkpoint loaded successfully
✅ inference: Model generated valid SQL
✅ memory_check: Using 15.2GB / 40GB

🎉 ALL TESTS PASSED - READY FOR FULL TRAINING

Cell 9: GPU Memory
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GPU Memory Usage:
Allocated: 15.23 GB
Reserved: 16.50 GB
Max allocated: 18.74 GB
✅ Memory usage looks good for full training
```

---

### What Each Test Validates:

| Test | What It Checks | Why It Matters |
|------|---------------|----------------|
| **Data Loading** | JSONL files parse correctly | Bad data = training fails |
| **Model Init** | 4-bit + LoRA works | Wrong config = OOM or slow training |
| **Training Step** | Loss decreases | If loss doesn't decrease, something's wrong |
| **Checkpoint Save** | Can write to GCS | Without this, lose progress on disconnect |
| **Checkpoint Load** | Can resume training | Critical for Colab disconnects |
| **Inference** | Model generates text | If can't generate, training is broken |
| **Memory** | Uses <20GB | If >35GB, might OOM during full training |

---

### Understanding Check (Learning):

After running validation, you should understand:
- [ ] Why we test on 20 examples first (catch bugs early, save time/money)
- [ ] What "smoke test" means (quick test to see if anything is obviously broken)
- [ ] Why loss should decrease (3.5 → 1.5 in 20 steps shows model is learning)
- [ ] Why checkpoint at step 10 (tests save/resume in middle of training)
- [ ] Why memory check matters (prevent OOM in full 8-hour training)
- [ ] What happens if validation fails (fix before wasting 8 hours)

---

### Git Verification:
- [ ] All validation code tracked
- [ ] Test config tracked
- [ ] Notebook tracked
- [ ] No test checkpoints tracked (will be in GCS)
- [ ] Commit message follows convention

---

### What You Should Have Learned:
1. **Validation Strategy**: Test small before training big
2. **Smoke Testing**: Quick end-to-end test to catch obvious issues
3. **Component Testing**: Test each piece independently
4. **GPU Memory Profiling**: Monitor memory to prevent OOM
5. **Checkpoint Testing**: Verify save/resume works
6. **Inference Testing**: Verify model can actually generate
7. **Cost Awareness**: 10 minutes of validation saves 8 hours of wasted training

---

## CRITICAL: Before Moving to Module 2.4

**You MUST complete these steps:**

1. [ ] Run `notebooks/validation_notebook.ipynb` in Colab Pro+
2. [ ] All cells execute without errors
3. [ ] All validation tests pass ✅
4. [ ] GPU memory <20GB
5. [ ] Model generates reasonable SQL (doesn't need to be perfect, just syntactically valid)
6. [ ] Checkpoint saves to GCS successfully
7. [ ] Checkpoint can be loaded

**If ANY test fails:**
- Debug the specific failure
- Check error messages
- Review logs
- Fix the issue
- Re-run validation
- Don't proceed until ALL tests pass

**If ALL tests pass:**
- ✅ Your training pipeline is solid
- ✅ Ready for 8-12 hour full training
- ✅ Confident nothing will break 4 hours in