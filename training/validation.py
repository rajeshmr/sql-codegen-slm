"""
Training validation module for SQL Codegen SLM.
Module 2.3: Training Validation & Smoke Testing

Validates the complete training pipeline before running full training.
"""

import json
import logging
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import yaml

logger = logging.getLogger(__name__)


def test_data_loading(config: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """
    Test data loading with a small subset.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (success, details)
    """
    from training.data_loader import format_instruction_example, load_training_data
    from transformers import AutoTokenizer
    
    details = {}
    
    try:
        # Load tokenizer
        model_name = config["model"]["name"]
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Get data paths
        train_file = config["data"]["train_file"]
        val_file = config["data"]["val_file"]
        max_samples_train = config["data"].get("max_samples_train", 20)
        max_samples_val = config["data"].get("max_samples_val", 10)
        max_seq_length = config["model"].get("max_seq_length", 512)
        
        # Load data
        train_dataset, val_dataset = load_training_data(
            train_file=train_file,
            val_file=val_file,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            max_samples=max_samples_train
        )
        
        details["train_count"] = len(train_dataset)
        details["val_count"] = len(val_dataset) if val_dataset else 0
        
        # Verify format by loading raw example
        with open(train_file, 'r') as f:
            first_line = f.readline()
            example = json.loads(first_line)
        
        formatted = format_instruction_example(example)
        details["sample_format"] = formatted[:500] + "..." if len(formatted) > 500 else formatted
        
        # Check tokenization
        sample = train_dataset[0]
        details["input_ids_shape"] = list(sample["input_ids"].shape)
        details["attention_mask_shape"] = list(sample["attention_mask"].shape)
        
        logger.info(f"Data loading test passed: {details['train_count']} train, {details['val_count']} val")
        
        return True, details
        
    except Exception as e:
        details["error"] = str(e)
        logger.error(f"Data loading test failed: {e}")
        return False, details


def test_model_initialization(config: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """
    Test model initialization with 4-bit quantization and LoRA.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (success, model_info)
    """
    from training.model_utils import load_model_and_tokenizer, print_trainable_parameters
    
    model_info = {}
    
    try:
        logger.info("Loading model with 4-bit quantization...")
        model, tokenizer = load_model_and_tokenizer(config)
        
        # Check device
        device = next(model.parameters()).device
        model_info["device"] = str(device)
        
        # Check parameters
        total_params, trainable_params = print_trainable_parameters(model)
        model_info["total_params"] = total_params
        model_info["trainable_params"] = trainable_params
        model_info["trainable_pct"] = 100 * trainable_params / total_params if total_params > 0 else 0
        
        # Test forward pass
        logger.info("Testing forward pass...")
        test_input = tokenizer("SELECT * FROM users", return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**test_input)
        
        model_info["forward_pass"] = True
        model_info["output_shape"] = list(outputs.logits.shape)
        
        # Memory usage
        if torch.cuda.is_available():
            model_info["gpu_memory_gb"] = torch.cuda.memory_allocated(0) / 1e9
        
        logger.info(f"Model initialization test passed: {trainable_params:,} trainable params")
        
        # Clean up
        del model
        torch.cuda.empty_cache()
        
        return True, model_info
        
    except Exception as e:
        model_info["error"] = str(e)
        logger.error(f"Model initialization test failed: {e}")
        return False, model_info


def test_training_step(
    model,
    train_dataset,
    config: Dict[str, Any]
) -> Tuple[bool, Dict[str, Any]]:
    """
    Test a few training steps.
    
    Args:
        model: Model to train
        train_dataset: Training dataset
        config: Configuration dictionary
        
    Returns:
        Tuple of (success, training_metrics)
    """
    from transformers import Trainer, TrainingArguments
    from training.data_loader import create_data_collator
    from transformers import AutoTokenizer
    
    metrics = {}
    
    try:
        # Get tokenizer
        model_name = config["model"]["name"]
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Create minimal training args
        training_args = TrainingArguments(
            output_dir="/tmp/test_training",
            max_steps=2,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            logging_steps=1,
            save_strategy="no",
            report_to=[],
            fp16=config["training"].get("fp16", True),
            bf16=False,
            remove_unused_columns=False,
        )
        
        # Create trainer
        data_collator = create_data_collator(tokenizer)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
        )
        
        # Run 2 steps
        logger.info("Running 2 training steps...")
        initial_memory = torch.cuda.memory_allocated(0) / 1e9 if torch.cuda.is_available() else 0
        
        result = trainer.train()
        
        metrics["train_loss"] = result.training_loss
        metrics["global_step"] = result.global_step
        
        # Check memory
        if torch.cuda.is_available():
            final_memory = torch.cuda.memory_allocated(0) / 1e9
            metrics["memory_before_gb"] = initial_memory
            metrics["memory_after_gb"] = final_memory
            metrics["memory_increase_gb"] = final_memory - initial_memory
        
        # Check gradients were computed
        has_grad = any(p.grad is not None for p in model.parameters() if p.requires_grad)
        metrics["gradients_computed"] = has_grad
        
        logger.info(f"Training step test passed: loss={metrics['train_loss']:.4f}")
        
        return True, metrics
        
    except Exception as e:
        metrics["error"] = str(e)
        logger.error(f"Training step test failed: {e}")
        return False, metrics


def test_checkpoint_saving(
    trainer,
    output_dir: str
) -> Tuple[bool, Dict[str, Any]]:
    """
    Test checkpoint saving.
    
    Args:
        trainer: Trainer object
        output_dir: Directory to save checkpoint
        
    Returns:
        Tuple of (success, checkpoint_info)
    """
    checkpoint_info = {}
    
    try:
        checkpoint_dir = os.path.join(output_dir, "checkpoint-test")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        logger.info(f"Saving test checkpoint to {checkpoint_dir}...")
        trainer.save_model(checkpoint_dir)
        trainer.save_state()
        
        # List files
        files = list(Path(checkpoint_dir).glob("*"))
        checkpoint_info["files"] = [f.name for f in files]
        checkpoint_info["total_size_mb"] = sum(f.stat().st_size for f in files) / 1e6
        checkpoint_info["checkpoint_path"] = checkpoint_dir
        
        logger.info(f"Checkpoint saved: {len(files)} files, {checkpoint_info['total_size_mb']:.1f} MB")
        
        return True, checkpoint_info
        
    except Exception as e:
        checkpoint_info["error"] = str(e)
        logger.error(f"Checkpoint saving test failed: {e}")
        return False, checkpoint_info


def test_checkpoint_loading(
    checkpoint_path: str,
    config: Dict[str, Any]
) -> Tuple[bool, Dict[str, Any]]:
    """
    Test loading model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint
        config: Configuration dictionary
        
    Returns:
        Tuple of (success, load_info)
    """
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    
    load_info = {}
    
    try:
        logger.info(f"Loading model from checkpoint: {checkpoint_path}")
        
        # Check if checkpoint exists
        if not os.path.exists(checkpoint_path):
            load_info["error"] = f"Checkpoint not found: {checkpoint_path}"
            return False, load_info
        
        # Load base model
        model_name = config["model"]["name"]
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        
        # Load LoRA weights
        model = PeftModel.from_pretrained(base_model, checkpoint_path)
        
        load_info["loaded"] = True
        load_info["device"] = str(next(model.parameters()).device)
        
        # Clean up
        del model, base_model
        torch.cuda.empty_cache()
        
        logger.info("Checkpoint loading test passed")
        
        return True, load_info
        
    except Exception as e:
        load_info["error"] = str(e)
        logger.error(f"Checkpoint loading test failed: {e}")
        return False, load_info


def test_inference(
    model,
    tokenizer,
    test_prompt: str
) -> Tuple[bool, str]:
    """
    Test model inference.
    
    Args:
        model: Model to test
        tokenizer: Tokenizer
        test_prompt: Prompt to generate from
        
    Returns:
        Tuple of (success, generated_text)
    """
    try:
        logger.info("Testing inference...")
        
        # Format as instruction
        formatted_prompt = f"<s>[INST] {test_prompt} [/INST]"
        
        # Tokenize
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Decode
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract response
        if "[/INST]" in generated:
            response = generated.split("[/INST]")[-1].strip()
        else:
            response = generated
        
        logger.info(f"Inference test passed. Generated: {response[:100]}...")
        
        return True, response
        
    except Exception as e:
        logger.error(f"Inference test failed: {e}")
        return False, str(e)


def test_gcs_sync(bucket_name: str, test_dir: str) -> Tuple[bool, Dict[str, Any]]:
    """
    Test GCS sync capability.
    
    Args:
        bucket_name: GCS bucket name
        test_dir: Local directory to sync
        
    Returns:
        Tuple of (success, sync_info)
    """
    sync_info = {}
    
    try:
        # Create test file
        test_file = os.path.join(test_dir, "gcs_test.txt")
        with open(test_file, 'w') as f:
            f.write(f"GCS sync test at {datetime.now()}")
        
        # Try to sync
        gcs_path = f"gs://{bucket_name}/test-sync/"
        
        result = subprocess.run(
            ["gsutil", "cp", test_file, gcs_path],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            sync_info["synced"] = True
            sync_info["gcs_path"] = gcs_path
            
            # Clean up
            subprocess.run(
                ["gsutil", "rm", f"{gcs_path}gcs_test.txt"],
                capture_output=True,
                timeout=30
            )
            
            logger.info("GCS sync test passed")
            return True, sync_info
        else:
            sync_info["error"] = result.stderr
            return False, sync_info
            
    except subprocess.TimeoutExpired:
        sync_info["error"] = "GCS sync timed out"
        return False, sync_info
    except FileNotFoundError:
        sync_info["error"] = "gsutil not found"
        return False, sync_info
    except Exception as e:
        sync_info["error"] = str(e)
        return False, sync_info


def generate_validation_report(results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate a formatted validation report.
    
    Args:
        results: Dictionary of test results
        
    Returns:
        Report dictionary
    """
    report = {}
    all_passed = True
    
    print("\n" + "=" * 60)
    print("VALIDATION REPORT")
    print("=" * 60)
    
    for test_name, result in results.items():
        passed = result.get("passed", False)
        message = result.get("message", "")
        
        status = "PASSED" if passed else "FAILED"
        emoji = "✅" if passed else "❌"
        
        print(f"{emoji} {test_name}: {status}")
        if message:
            print(f"   {message}")
        
        report[test_name] = {
            "passed": passed,
            "message": message,
            "details": result.get("details", {})
        }
        
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED - READY FOR FULL TRAINING")
    else:
        print("\n⚠️  SOME TESTS FAILED - FIX ISSUES BEFORE FULL TRAINING")
    
    report["all_passed"] = all_passed
    report["timestamp"] = datetime.now().isoformat()
    
    return report


def validate_training_pipeline(config_path: str) -> Dict[str, Any]:
    """
    Run complete validation suite.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Validation report dictionary
    """
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "=" * 60)
    print("SQL CODEGEN SLM - TRAINING VALIDATION")
    print("=" * 60 + "\n")
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    results = {}
    
    # Test 1: Data Loading
    print("\n📊 Testing data loading...")
    success, details = test_data_loading(config)
    results["data_loading"] = {
        "passed": success,
        "message": f"{details.get('train_count', 0)} train, {details.get('val_count', 0)} val examples loaded" if success else details.get("error", ""),
        "details": details
    }
    
    if not success:
        return generate_validation_report(results)
    
    # Test 2: Model Initialization
    print("\n🤖 Testing model initialization...")
    success, model_info = test_model_initialization(config)
    results["model_init"] = {
        "passed": success,
        "message": f"Mistral-7B loaded with LoRA ({model_info.get('trainable_pct', 0):.2f}% trainable)" if success else model_info.get("error", ""),
        "details": model_info
    }
    
    if not success:
        return generate_validation_report(results)
    
    # Test 3: GCS Sync (if available)
    print("\n☁️ Testing GCS sync...")
    gcs_config = config.get("gcs", {})
    bucket_name = gcs_config.get("bucket", "sql-codegen-slm-data")
    
    success, sync_info = test_gcs_sync(bucket_name, "/tmp")
    results["gcs_sync"] = {
        "passed": success,
        "message": "GCS sync working" if success else sync_info.get("error", "GCS not available"),
        "details": sync_info
    }
    
    # Test 4: GPU Memory Check
    print("\n💾 Checking GPU memory...")
    if torch.cuda.is_available():
        memory_gb = torch.cuda.memory_allocated(0) / 1e9
        max_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        memory_ok = memory_gb < max_memory_gb * 0.5  # Less than 50% used
        
        results["memory_check"] = {
            "passed": memory_ok,
            "message": f"Using {memory_gb:.1f}GB / {max_memory_gb:.1f}GB",
            "details": {"allocated_gb": memory_gb, "total_gb": max_memory_gb}
        }
    else:
        results["memory_check"] = {
            "passed": False,
            "message": "No GPU available",
            "details": {}
        }
    
    return generate_validation_report(results)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = "training/configs/test_config.yaml"
    
    report = validate_training_pipeline(config_path)
    
    # Exit with appropriate code
    sys.exit(0 if report.get("all_passed", False) else 1)
