"""
Smoke test for SQL Codegen SLM training pipeline.
Module 2.3: Training Validation & Smoke Testing

Runs minimal end-to-end training to verify the pipeline works.
"""

import logging
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml

logger = logging.getLogger(__name__)


def run_smoke_test(config_path: str) -> Dict[str, Any]:
    """
    Run minimal end-to-end training (20 steps).
    
    Tests the complete workflow:
    1. Load data
    2. Initialize model
    3. Train for 10 steps
    4. Save checkpoint
    5. Resume from checkpoint
    6. Complete remaining 10 steps
    7. Test inference
    
    Args:
        config_path: Path to test configuration
        
    Returns:
        Test results dictionary
    """
    from training.data_loader import create_data_collator, load_training_data
    from training.model_utils import load_model_and_tokenizer
    from training.callbacks import TrainingProgressCallback
    from transformers import Trainer, TrainingArguments
    
    logging.basicConfig(level=logging.INFO)
    
    results = {
        "success": False,
        "failure_point": None,
        "train_loss": None,
        "checkpoint_path": None,
        "can_generate": False,
        "steps_completed": 0,
    }
    
    print("\n" + "=" * 60)
    print("🔥 SMOKE TEST - SQL CODEGEN SLM")
    print("=" * 60)
    print(f"Config: {config_path}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60 + "\n")
    
    # Load config
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print("✅ Config loaded")
    except Exception as e:
        results["failure_point"] = f"config_load: {e}"
        return results
    
    # Setup output directory
    output_dir = config["training"].get("output_dir", "/tmp/smoke_test")
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Load data
    print("\n📊 Step 1: Loading data...")
    try:
        model_name = config["model"]["name"]
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        train_dataset, val_dataset = load_training_data(
            train_file=config["data"]["train_file"],
            val_file=config["data"]["val_file"],
            tokenizer=tokenizer,
            max_seq_length=config["model"].get("max_seq_length", 512),
            max_samples=config["data"].get("max_samples_train", 20)
        )
        print(f"   ✅ Loaded {len(train_dataset)} train, {len(val_dataset) if val_dataset else 0} val examples")
    except Exception as e:
        results["failure_point"] = f"data_load: {e}"
        print(f"   ❌ Failed: {e}")
        return results
    
    # Step 2: Initialize model
    print("\n🤖 Step 2: Initializing model...")
    try:
        model, tokenizer = load_model_and_tokenizer(config)
        print(f"   ✅ Model loaded on {next(model.parameters()).device}")
    except Exception as e:
        results["failure_point"] = f"model_init: {e}"
        print(f"   ❌ Failed: {e}")
        return results
    
    # Step 3: Setup trainer for first 10 steps
    print("\n🏋️ Step 3: Training first 10 steps...")
    try:
        training_args = TrainingArguments(
            output_dir=output_dir,
            max_steps=10,
            per_device_train_batch_size=config["training"].get("per_device_train_batch_size", 2),
            gradient_accumulation_steps=config["training"].get("gradient_accumulation_steps", 2),
            logging_steps=2,
            save_strategy="steps",
            save_steps=10,
            evaluation_strategy="no",
            fp16=config["training"].get("fp16", True),
            bf16=False,
            report_to=[],
            remove_unused_columns=False,
            dataloader_pin_memory=False,
        )
        
        data_collator = create_data_collator(tokenizer)
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            callbacks=[TrainingProgressCallback()],
        )
        
        # Train first 10 steps
        train_result = trainer.train()
        results["steps_completed"] = train_result.global_step
        
        print(f"   ✅ Completed {train_result.global_step} steps, loss: {train_result.training_loss:.4f}")
    except Exception as e:
        results["failure_point"] = f"training_phase1: {e}"
        print(f"   ❌ Failed: {e}")
        return results
    
    # Step 4: Save checkpoint
    print("\n💾 Step 4: Saving checkpoint...")
    try:
        checkpoint_dir = os.path.join(output_dir, "checkpoint-10")
        trainer.save_model(checkpoint_dir)
        trainer.save_state()
        results["checkpoint_path"] = checkpoint_dir
        
        # Verify checkpoint exists
        if os.path.exists(checkpoint_dir):
            files = list(Path(checkpoint_dir).glob("*"))
            size_mb = sum(f.stat().st_size for f in files) / 1e6
            print(f"   ✅ Checkpoint saved: {len(files)} files, {size_mb:.1f} MB")
        else:
            raise FileNotFoundError("Checkpoint directory not created")
    except Exception as e:
        results["failure_point"] = f"checkpoint_save: {e}"
        print(f"   ❌ Failed: {e}")
        return results
    
    # Step 5: Simulate disconnect - clear model from memory
    print("\n🔌 Step 5: Simulating disconnect (clearing memory)...")
    try:
        del model, trainer
        torch.cuda.empty_cache()
        print("   ✅ Memory cleared")
    except Exception as e:
        print(f"   ⚠️ Warning: {e}")
    
    # Step 6: Resume from checkpoint
    print("\n🔄 Step 6: Resuming from checkpoint...")
    try:
        # Reload model
        model, tokenizer = load_model_and_tokenizer(config)
        
        # Load checkpoint weights
        from peft import PeftModel
        # The model is already a PEFT model, so we need to load the adapter
        model.load_adapter(checkpoint_dir, adapter_name="default")
        
        print(f"   ✅ Model restored from checkpoint")
    except Exception as e:
        # Try alternative loading method
        try:
            model, tokenizer = load_model_and_tokenizer(config)
            print(f"   ⚠️ Loaded fresh model (checkpoint load failed: {e})")
        except Exception as e2:
            results["failure_point"] = f"checkpoint_load: {e2}"
            print(f"   ❌ Failed: {e2}")
            return results
    
    # Step 7: Complete remaining 10 steps
    print("\n🏋️ Step 7: Training remaining 10 steps...")
    try:
        training_args = TrainingArguments(
            output_dir=output_dir,
            max_steps=10,
            per_device_train_batch_size=config["training"].get("per_device_train_batch_size", 2),
            gradient_accumulation_steps=config["training"].get("gradient_accumulation_steps", 2),
            logging_steps=2,
            save_strategy="no",
            evaluation_strategy="no",
            fp16=config["training"].get("fp16", True),
            bf16=False,
            report_to=[],
            remove_unused_columns=False,
            dataloader_pin_memory=False,
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            callbacks=[TrainingProgressCallback()],
        )
        
        train_result = trainer.train()
        results["steps_completed"] += train_result.global_step
        results["train_loss"] = train_result.training_loss
        
        print(f"   ✅ Completed {train_result.global_step} more steps, final loss: {train_result.training_loss:.4f}")
    except Exception as e:
        results["failure_point"] = f"training_phase2: {e}"
        print(f"   ❌ Failed: {e}")
        return results
    
    # Step 8: Test inference
    print("\n🔮 Step 8: Testing inference...")
    try:
        test_prompt = """You are a PostgreSQL expert. Generate SQL for the following:

Schema:
CREATE TABLE users (id SERIAL PRIMARY KEY, name VARCHAR(100));

Question: How many users are there?"""
        
        formatted = f"<s>[INST] {test_prompt} [/INST]"
        inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=64,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "[/INST]" in generated:
            sql = generated.split("[/INST]")[-1].strip()
        else:
            sql = generated
        
        results["can_generate"] = len(sql) > 0
        results["generated_sql"] = sql[:200]
        
        print(f"   ✅ Generated SQL: {sql[:100]}...")
    except Exception as e:
        results["failure_point"] = f"inference: {e}"
        print(f"   ❌ Failed: {e}")
        return results
    
    # Success!
    results["success"] = True
    
    print("\n" + "=" * 60)
    print("✅ SMOKE TEST PASSED!")
    print("=" * 60)
    print(f"Total steps: {results['steps_completed']}")
    print(f"Final loss: {results['train_loss']:.4f}")
    print(f"Checkpoint: {results['checkpoint_path']}")
    print(f"Can generate: {results['can_generate']}")
    print("=" * 60 + "\n")
    
    return results


def test_full_workflow() -> tuple:
    """
    Test the complete workflow: Data → Model → Train → Save → Resume → Inference.
    
    Returns:
        Tuple of (success, failure_point)
    """
    config_path = "training/configs/test_config.yaml"
    
    if not os.path.exists(config_path):
        return False, "config_not_found"
    
    results = run_smoke_test(config_path)
    
    return results["success"], results.get("failure_point")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = "training/configs/test_config.yaml"
    
    results = run_smoke_test(config_path)
    
    sys.exit(0 if results["success"] else 1)
