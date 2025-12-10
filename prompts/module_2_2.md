# MODULE 2.2: Implement Training Pipeline

**Objective:** Create the complete training script with data loading, model initialization, LoRA setup, and training loop

---

## PROMPT FOR AI IDE (Windsurf/Claude):

### Context
You are working on sql-codegen-slm project. Environment is configured for Colab Pro+ with GCS storage. Now we need to implement the actual training pipeline that will fine-tune Mistral-7B using LoRA on the 8,234 PostgreSQL training examples. The script must handle data loading, model initialization with 4-bit quantization, LoRA adapter setup, training loop, evaluation, and checkpoint saving to GCS.

### Task: Implement Complete Training Pipeline

**Location:** `training/data_loader.py`

Create data loading module:

1. **Main function: load_training_data(train_file, val_file, tokenizer, max_seq_length)**
   - Loads JSONL files (train and validation)
   - Tokenizes examples using provided tokenizer
   - Creates PyTorch datasets
   - Handles formatting for instruction tuning (system/user/assistant messages)
   - Returns tuple: (train_dataset, val_dataset)

2. **Helper function: format_instruction_example(example)**
   - Takes example with "messages" field
   - Formats into single text string for training:
     - Combines system, user, assistant messages
     - Adds special tokens (BOS, EOS)
     - Format: `<s>[INST] {system}\n{user} [/INST] {assistant}</s>`
   - Returns formatted text string

3. **Helper function: tokenize_function(examples, tokenizer, max_seq_length)**
   - Tokenizes formatted text
   - Truncates to max_seq_length
   - Adds padding
   - Returns tokenized examples with input_ids, attention_mask, labels

4. **Helper function: create_data_collator(tokenizer)**
   - Creates data collator for batching
   - Handles padding dynamically
   - Sets labels for causal language modeling (shift right by 1)
   - Returns DataCollatorForLanguageModeling object

5. **Class: SQLDataset(torch.utils.data.Dataset)**
   - Custom dataset class for SQL examples
   - Implements __init__, __len__, __getitem__
   - Handles lazy loading for memory efficiency
   - Caches tokenized examples

**Location:** `training/model_utils.py`

Create model initialization module:

1. **Main function: load_model_and_tokenizer(config)**
   - Loads base Mistral-7B model with 4-bit quantization
   - Loads tokenizer
   - Prepares model for k-bit training
   - Returns tuple: (model, tokenizer)

2. **Helper function: setup_quantization_config(config)**
   - Creates BitsAndBytesConfig from config
   - Sets load_in_4bit=True
   - Configures bnb_4bit_compute_dtype
   - Returns quantization config object

3. **Helper function: setup_lora_config(config)**
   - Creates LoraConfig from config parameters
   - Sets r (rank), lora_alpha, lora_dropout
   - Specifies target_modules
   - Returns LoraConfig object

4. **Helper function: apply_lora_to_model(model, lora_config)**
   - Applies LoRA adapters to model
   - Uses get_peft_model from peft library
   - Prints trainable parameters summary
   - Returns PEFT model

5. **Helper function: print_trainable_parameters(model)**
   - Counts total parameters vs trainable parameters
   - Calculates percentage trainable
   - Prints summary in readable format
   - Returns (total_params, trainable_params)

**Location:** `training/train.py`

Create main training script:

1. **Main function: main()**
   - Parses command line arguments (--config path)
   - Loads configuration from YAML
   - Sets up logging
   - Initializes model and tokenizer
   - Loads training data
   - Sets up trainer
   - Runs training
   - Saves final model
   - Runs final evaluation
   - Returns training results

2. **Helper function: parse_args()**
   - Parses command line arguments:
     - --config: path to config YAML
     - --resume_from_checkpoint: optional checkpoint path
     - --output_dir: override output directory
   - Returns parsed arguments

3. **Helper function: load_config(config_path)**
   - Loads YAML configuration file
   - Validates required fields present
   - Replaces YOUR-BUCKET-NAME placeholder with actual bucket
   - Returns config dictionary

4. **Helper function: setup_logging(config)**
   - Configures Python logging
   - Sets up file and console handlers
   - Configures log levels
   - Creates log directory if needed
   - Returns logger object

5. **Helper function: setup_training_args(config)**
   - Creates TrainingArguments from config
   - Sets all hyperparameters
   - Configures output directories (GCS paths)
   - Sets evaluation and save strategies
   - Returns TrainingArguments object

6. **Helper function: create_trainer(model, tokenizer, train_dataset, val_dataset, training_args, data_collator)**
   - Creates Trainer object from transformers
   - Sets up model, datasets, arguments
   - Configures callbacks (checkpoint, early stopping)
   - Returns Trainer object

7. **Helper function: train_model(trainer, resume_from_checkpoint)**
   - Calls trainer.train()
   - Handles checkpoint resumption if provided
   - Catches and logs any errors
   - Returns training output (metrics, checkpoints)

8. **Helper function: evaluate_model(trainer)**
   - Runs final evaluation on validation set
   - Computes metrics (loss, perplexity)
   - Logs results
   - Returns evaluation metrics

9. **Helper function: save_final_model(trainer, output_dir)**
   - Saves final model to output_dir
   - Saves LoRA adapters only (not full model)
   - Saves tokenizer
   - Saves training config
   - Creates README with model card
   - Returns save path

**Location:** `training/callbacks.py`

Create custom training callbacks:

1. **Class: GCSCheckpointCallback(TrainerCallback)**
   - Callback to sync checkpoints to GCS after saving
   - Implements on_save method
   - Uses gcs_utils to upload checkpoint
   - Logs upload status

2. **Class: TrainingProgressCallback(TrainerCallback)**
   - Logs detailed training progress
   - Implements on_log method
   - Prints step, loss, learning rate
   - Estimates time remaining

3. **Class: EarlyStoppingCallback(TrainerCallback)**
   - Stops training if validation loss doesn't improve
   - Implements on_evaluate method
   - Tracks best validation loss
   - Stops if no improvement for N evaluations

**Location:** `training/metrics.py`

Create evaluation metrics module:

1. **Function: compute_metrics(eval_pred)**
   - Computes perplexity from loss
   - Calculates additional metrics if needed
   - Used by Trainer for evaluation
   - Returns dictionary of metrics

2. **Function: evaluate_sql_generation(model, tokenizer, test_examples, num_samples)**
   - Takes random samples from test set
   - Generates SQL for each example
   - Compares generated vs ground truth
   - Calculates exact match accuracy
   - Returns evaluation report

**Location:** `scripts/run_training.sh`

Create training launcher script:

Script should:
1. Check if running in Colab (via environment variables)
2. Verify GCS authentication
3. Check GPU availability
4. Launch training with proper arguments
5. Handle errors gracefully
6. Print start time, estimated end time
7. Make executable

**Create test file:** `tests/training/test_training_pipeline.py`

Create pytest tests:
1. Test data loading with small sample
2. Test model initialization (without actual downloading)
3. Test LoRA config creation
4. Test training args setup
5. Test format_instruction_example produces correct format
6. Test tokenization produces proper shapes
7. Test end-to-end pipeline with 2 examples (smoke test)

**Location:** `notebooks/test_training_setup.ipynb`

Create testing notebook for Colab:

**Cell 1: Test Data Loading**
```python
from training.data_loader import load_training_data
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

# Test with small sample
train_dataset, val_dataset = load_training_data(
    train_file="/content/data/train_postgres.jsonl",
    val_file="/content/data/val_postgres.jsonl",
    tokenizer=tokenizer,
    max_seq_length=2048
)

print(f"Train samples: {len(train_dataset)}")
print(f"Val samples: {len(val_dataset)}")
print(f"Sample input shape: {train_dataset[0]['input_ids'].shape}")
```

**Cell 2: Test Model Loading**
```python
from training.model_utils import load_model_and_tokenizer
import yaml

# Load config
with open('training/configs/mistral_lora_config.yaml') as f:
    config = yaml.safe_load(f)

# Load model (this will take a few minutes)
model, tokenizer = load_model_and_tokenizer(config)

print("Model loaded successfully!")
print(f"Model device: {model.device}")
print(f"Model dtype: {model.dtype}")
```

**Cell 3: Test Training (1 step)**
```python
from training.train import main
import sys

# Override config for quick test
sys.argv = [
    'train.py',
    '--config', 'training/configs/mistral_lora_config.yaml'
]

# Modify config to train for just 1 step (testing)
# Then run: main()
```

**Update:** `README.md`

Add "Training" section:
```markdown
## Training the Model

### Quick Start (Colab)
1. Open `notebooks/train_colab_gcs.ipynb`
2. Run all cells
3. Training takes 8-12 hours
4. Model saves to GCS automatically

### Manual Training
```bash
python -m training.train \
  --config training/configs/mistral_lora_config.yaml
```

### Monitoring Training
- **TensorBoard**: View in Colab or locally
  ```python
  %tensorboard --logdir gs://YOUR-BUCKET-NAME/sql-codegen-tensorboard
  ```
- **Logs**: Check training/logs/ for detailed logs
- **Checkpoints**: Every 500 steps in GCS

### Training Output
- LoRA adapters: ~100MB
- Saved to: gs://YOUR-BUCKET-NAME/sql-codegen-models/
- Final model includes: adapter_model.bin, adapter_config.json, tokenizer

### Expected Results
- Training loss: Should decrease to ~0.5-1.0
- Validation loss: ~1.0-1.5
- Training time: 8-12 hours on A100
```

### Commit Message:
"feat(training): Implement complete training pipeline with LoRA fine-tuning - Module 2.2"

---

## ✅ MODULE 2.2 COMPLETION CHECKLIST

**After running the AI IDE prompt, verify the following:**

### Files Created:
- [ ] `training/data_loader.py` exists
- [ ] `training/model_utils.py` exists
- [ ] `training/train.py` exists
- [ ] `training/callbacks.py` exists
- [ ] `training/metrics.py` exists
- [ ] `scripts/run_training.sh` exists and is executable
- [ ] `tests/training/test_training_pipeline.py` exists
- [ ] `notebooks/test_training_setup.ipynb` exists
- [ ] README.md updated with training instructions

### Code Structure Validation:

**data_loader.py:**
- [ ] Has load_training_data function
- [ ] Has format_instruction_example function
- [ ] Has SQLDataset class
- [ ] Handles JSONL loading
- [ ] Tokenization implemented

**model_utils.py:**
- [ ] Has load_model_and_tokenizer function
- [ ] Has setup_quantization_config function
- [ ] Has setup_lora_config function
- [ ] Has apply_lora_to_model function
- [ ] Has print_trainable_parameters function

**train.py:**
- [ ] Has main() function
- [ ] Loads config from YAML
- [ ] Initializes model with LoRA
- [ ] Loads datasets
- [ ] Creates Trainer
- [ ] Handles training loop
- [ ] Saves model to GCS

**callbacks.py:**
- [ ] Has GCSCheckpointCallback
- [ ] Has TrainingProgressCallback
- [ ] Optional: EarlyStoppingCallback

### Functional Tests (Local - No GPU Needed):
- [ ] Run `pytest tests/training/test_training_pipeline.py -v`
- [ ] Data loading test passes
- [ ] Config loading test passes
- [ ] Formatting test passes
- [ ] Tokenization test passes
- [ ] LoRA config creation test passes

### Code Quality Check:

Pick 3 random functions:
- [ ] Has docstrings explaining purpose
- [ ] Has type hints for parameters
- [ ] Has error handling (try/except where appropriate)
- [ ] Has logging statements
- [ ] Returns meaningful values

### Instruction Format Verification:

Test format_instruction_example manually:
```python
from training.data_loader import format_instruction_example

example = {
    "messages": [
        {"role": "system", "content": "You are a SQL expert."},
        {"role": "user", "content": "Schema: ...\nQuestion: How many users?"},
        {"role": "assistant", "content": "SELECT COUNT(*) FROM users;"}
    ]
}

formatted = format_instruction_example(example)
print(formatted)
```

Should produce:
```
<s>[INST] You are a SQL expert.
Schema: ...
Question: How many users? [/INST] SELECT COUNT(*) FROM users;</s>
```

- [ ] Format includes <s> (BOS token)
- [ ] Format includes [INST] and [/INST] tags
- [ ] Format includes </s> (EOS token)
- [ ] System, user, assistant messages combined correctly

### Configuration Integration:

Check train.py loads config correctly:
- [ ] Reads training/configs/mistral_lora_config.yaml
- [ ] Extracts model name
- [ ] Extracts LoRA parameters (r, alpha, dropout, target_modules)
- [ ] Extracts training hyperparameters
- [ ] Extracts data file paths
- [ ] Creates TrainingArguments with GCS output_dir

### GCS Integration Check:

Verify GCS paths in train.py:
- [ ] output_dir uses gs:// URI
- [ ] logging_dir uses gs:// URI
- [ ] Checkpoints will save to GCS
- [ ] Trainer can write to GCS (Transformers supports this)

### Understanding Check (Learning):

**Data Loading:**
- [ ] You understand why we format as <s>[INST]...[/INST]...</s> (Mistral instruction format)
- [ ] You know what tokenization does (text → token IDs)
- [ ] You understand max_seq_length (truncate long examples)

**Model Setup:**
- [ ] You understand 4-bit quantization (reduces memory 4x)
- [ ] You know LoRA adds adapters to specific layers
- [ ] You understand trainable parameters (~94M) vs total (7B)

**Training Loop:**
- [ ] You know Trainer handles forward/backward pass automatically
- [ ] You understand gradient accumulation (simulates larger batches)
- [ ] You know checkpoints save every 500 steps

**LoRA Architecture:**
- [ ] Base model weights frozen (7B parameters)
- [ ] Only LoRA adapters train (~94M parameters)
- [ ] Final model = base model + adapter weights

### Ready for Training Check:

Before actually training, verify:
- [ ] GCS bucket created and accessible
- [ ] Data uploaded to GCS
- [ ] Service account key ready for Colab
- [ ] Colab Pro+ subscription active
- [ ] All code committed to git

### What You Should Have Learned:
1. **Instruction Tuning Format**: How to format examples for Mistral
2. **Data Loading**: PyTorch datasets and data loaders
3. **LoRA Setup**: Applying adapters to base model
4. **Trainer API**: HuggingFace Trainer for simplified training
5. **Checkpoint Management**: Saving and resuming training
6. **GCS Integration**: Writing directly to cloud storage
7. **Training Callbacks**: Custom hooks for training events

### Git Verification:
- [ ] All training code files tracked
- [ ] Test files tracked
- [ ] Notebooks tracked
- [ ] No model weights tracked (too large)
- [ ] No checkpoints tracked (will be in GCS)
- [ ] Commit message follows convention

---

**SAMPLE CODE STRUCTURE:**

`training/train.py` should look like:
```python
import argparse
import yaml
import logging
from transformers import Trainer, TrainingArguments
from training.data_loader import load_training_data
from training.model_utils import load_model_and_tokenizer, apply_lora_to_model
from training.callbacks import GCSCheckpointCallback, TrainingProgressCallback

def main():
    # Parse arguments
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Setup logging
    logger = setup_logging(config)
    logger.info("Starting training...")
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config)
    
    # Load data
    train_dataset, val_dataset = load_training_data(
        config['data']['train_file'],
        config['data']['val_file'],
        tokenizer,
        config['model']['max_seq_length']
    )
    
    # Setup training arguments
    training_args = setup_training_args(config)
    
    # Create trainer
    trainer = create_trainer(
        model, tokenizer, 
        train_dataset, val_dataset,
        training_args
    )
    
    # Train
    train_result = trainer.train(
        resume_from_checkpoint=args.resume_from_checkpoint
    )
    
    # Save model
    save_final_model(trainer, training_args.output_dir)
    
    # Evaluate
    eval_results = evaluate_model(trainer)
    
    logger.info(f"Training complete! Results: {eval_results}")
    
    return train_result

if __name__ == "__main__":
    main()
```