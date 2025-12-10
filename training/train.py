"""
Main training script for SQL Codegen SLM.
Module 2.2: Training Pipeline

Fine-tunes Mistral-7B using LoRA for PostgreSQL query generation.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml
from transformers import Trainer, TrainingArguments

from training.callbacks import (
    EarlyStoppingCallback,
    GCSCheckpointCallback,
    TrainingProgressCallback,
)
from training.data_loader import create_data_collator, load_training_data
from training.metrics import compute_metrics, evaluate_sql_generation
from training.model_utils import load_model_and_tokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Default GCS bucket
DEFAULT_BUCKET = "sql-codegen-slm-data"


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Train SQL Codegen SLM with LoRA fine-tuning"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training configuration YAML file"
    )
    
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override output directory from config"
    )
    
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from latest checkpoint in output_dir"
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load and validate configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Configuration dictionary
    """
    path = Path(config_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    
    # Validate required sections
    required_sections = ["model", "lora", "training", "data"]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")
    
    logger.info(f"Loaded configuration from {config_path}")
    
    return config


def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    """
    Configure logging with file and console handlers.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured logger
    """
    log_config = config.get("logging", {})
    log_dir = log_config.get("log_dir", "/content/logs")
    
    # Create log directory
    os.makedirs(log_dir, exist_ok=True)
    
    # Create file handler
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    
    # Add to root logger
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    
    logger.info(f"Logging to file: {log_file}")
    
    return logger


def setup_training_args(
    config: Dict[str, Any],
    output_dir_override: Optional[str] = None
) -> TrainingArguments:
    """
    Create TrainingArguments from configuration.
    
    Args:
        config: Configuration dictionary
        output_dir_override: Optional override for output directory
        
    Returns:
        TrainingArguments object
    """
    train_config = config.get("training", {})
    log_config = config.get("logging", {})
    
    # Determine output directory
    output_dir = output_dir_override or train_config.get("output_dir", "/content/models")
    
    # Determine logging directory
    logging_dir = log_config.get("tensorboard_dir", "/content/tensorboard")
    
    # Create training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=train_config.get("num_train_epochs", 3),
        per_device_train_batch_size=train_config.get("per_device_train_batch_size", 4),
        per_device_eval_batch_size=train_config.get("per_device_eval_batch_size", 4),
        gradient_accumulation_steps=train_config.get("gradient_accumulation_steps", 4),
        gradient_checkpointing=train_config.get("gradient_checkpointing", True),
        optim=train_config.get("optim", "paged_adamw_32bit"),
        learning_rate=train_config.get("learning_rate", 2e-4),
        weight_decay=train_config.get("weight_decay", 0.001),
        warmup_ratio=train_config.get("warmup_ratio", 0.03),
        lr_scheduler_type=train_config.get("lr_scheduler_type", "cosine"),
        max_grad_norm=train_config.get("max_grad_norm", 0.3),
        fp16=train_config.get("fp16", False),
        bf16=train_config.get("bf16", True),
        logging_steps=train_config.get("logging_steps", 10),
        logging_dir=logging_dir,
        save_strategy=train_config.get("save_strategy", "steps"),
        save_steps=train_config.get("save_steps", 500),
        evaluation_strategy=train_config.get("evaluation_strategy", "steps"),
        eval_steps=train_config.get("eval_steps", 500),
        save_total_limit=train_config.get("save_total_limit", 2),
        load_best_model_at_end=train_config.get("load_best_model_at_end", True),
        metric_for_best_model=train_config.get("metric_for_best_model", "eval_loss"),
        greater_is_better=train_config.get("greater_is_better", False),
        report_to=train_config.get("report_to", ["tensorboard"]),
        remove_unused_columns=train_config.get("remove_unused_columns", False),
        dataloader_pin_memory=False,  # For Colab compatibility
    )
    
    logger.info(f"Training arguments configured:")
    logger.info(f"  Output dir: {output_dir}")
    logger.info(f"  Epochs: {training_args.num_train_epochs}")
    logger.info(f"  Batch size: {training_args.per_device_train_batch_size}")
    logger.info(f"  Gradient accumulation: {training_args.gradient_accumulation_steps}")
    logger.info(f"  Learning rate: {training_args.learning_rate}")
    
    return training_args


def create_trainer(
    model,
    tokenizer,
    train_dataset,
    val_dataset,
    training_args: TrainingArguments,
    bucket_name: str = DEFAULT_BUCKET
) -> Trainer:
    """
    Create Trainer with callbacks.
    
    Args:
        model: Model to train
        tokenizer: Tokenizer
        train_dataset: Training dataset
        val_dataset: Validation dataset
        training_args: Training arguments
        bucket_name: GCS bucket for checkpoint syncing
        
    Returns:
        Configured Trainer object
    """
    # Create data collator
    data_collator = create_data_collator(tokenizer)
    
    # Setup callbacks
    callbacks = [
        TrainingProgressCallback(),
        GCSCheckpointCallback(bucket_name=bucket_name, gcs_prefix="models"),
        EarlyStoppingCallback(patience=3, min_delta=0.01),
    ]
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
    )
    
    logger.info("Trainer created with callbacks: "
                "TrainingProgress, GCSCheckpoint, EarlyStopping")
    
    return trainer


def train_model(
    trainer: Trainer,
    resume_from_checkpoint: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run the training loop.
    
    Args:
        trainer: Configured Trainer object
        resume_from_checkpoint: Optional checkpoint path to resume from
        
    Returns:
        Training results dictionary
    """
    logger.info("Starting training...")
    
    try:
        train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        
        # Log metrics
        metrics = train_result.metrics
        logger.info(f"Training completed. Metrics: {metrics}")
        
        # Save metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        
        return {
            "status": "success",
            "metrics": metrics,
            "global_step": train_result.global_step,
        }
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise


def evaluate_model(trainer: Trainer) -> Dict[str, float]:
    """
    Run final evaluation on validation set.
    
    Args:
        trainer: Trained Trainer object
        
    Returns:
        Evaluation metrics dictionary
    """
    logger.info("Running final evaluation...")
    
    eval_results = trainer.evaluate()
    
    # Log metrics
    trainer.log_metrics("eval", eval_results)
    trainer.save_metrics("eval", eval_results)
    
    logger.info(f"Evaluation results: {eval_results}")
    
    return eval_results


def save_final_model(
    trainer: Trainer,
    output_dir: str,
    config: Dict[str, Any]
) -> str:
    """
    Save the final model, tokenizer, and config.
    
    Args:
        trainer: Trained Trainer object
        output_dir: Directory to save model
        config: Training configuration
        
    Returns:
        Path to saved model
    """
    final_dir = os.path.join(output_dir, "final_model")
    os.makedirs(final_dir, exist_ok=True)
    
    logger.info(f"Saving final model to {final_dir}")
    
    # Save model (LoRA adapters)
    trainer.model.save_pretrained(final_dir)
    
    # Save tokenizer
    trainer.tokenizer.save_pretrained(final_dir)
    
    # Save training config
    config_path = os.path.join(final_dir, "training_config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    
    # Create model card
    model_card = f"""# SQL Codegen SLM - Fine-tuned Mistral-7B

## Model Description
Fine-tuned Mistral-7B for PostgreSQL query generation using LoRA.

## Training Details
- Base model: {config['model']['name']}
- LoRA rank: {config['lora']['r']}
- Training epochs: {config['training']['num_train_epochs']}
- Training date: {datetime.now().strftime('%Y-%m-%d')}

## Usage
```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
model = PeftModel.from_pretrained(base_model, "{final_dir}")
tokenizer = AutoTokenizer.from_pretrained("{final_dir}")
```

## License
MIT
"""
    
    readme_path = os.path.join(final_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write(model_card)
    
    logger.info(f"Model saved to {final_dir}")
    
    return final_dir


def main():
    """Main training entry point."""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    setup_logging(config)
    
    logger.info("=" * 60)
    logger.info("SQL Codegen SLM Training")
    logger.info("=" * 60)
    
    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        logger.warning("No GPU available! Training will be very slow.")
    
    # Load model and tokenizer
    logger.info("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(config)
    
    # Load data
    logger.info("Loading training data...")
    train_dataset, val_dataset = load_training_data(
        train_file=config["data"]["train_file"],
        val_file=config["data"]["val_file"],
        tokenizer=tokenizer,
        max_seq_length=config["model"].get("max_seq_length", 2048)
    )
    
    # Setup training arguments
    training_args = setup_training_args(config, args.output_dir)
    
    # Create trainer
    trainer = create_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        training_args=training_args,
        bucket_name=DEFAULT_BUCKET
    )
    
    # Determine checkpoint for resumption
    resume_checkpoint = args.resume_from_checkpoint
    if args.resume and resume_checkpoint is None:
        # Find latest checkpoint
        checkpoints = list(Path(training_args.output_dir).glob("checkpoint-*"))
        if checkpoints:
            resume_checkpoint = str(max(checkpoints, key=lambda x: int(x.name.split("-")[1])))
            logger.info(f"Resuming from latest checkpoint: {resume_checkpoint}")
    
    # Train
    train_result = train_model(trainer, resume_checkpoint)
    
    # Evaluate
    eval_results = evaluate_model(trainer)
    
    # Save final model
    save_final_model(trainer, training_args.output_dir, config)
    
    # Final sync to GCS
    logger.info("Syncing final model to GCS...")
    try:
        import subprocess
        subprocess.run(
            ["gsutil", "-m", "rsync", "-r", 
             training_args.output_dir, 
             f"gs://{DEFAULT_BUCKET}/models/"],
            check=True
        )
        logger.info(f"Model synced to gs://{DEFAULT_BUCKET}/models/")
    except Exception as e:
        logger.warning(f"GCS sync failed: {e}")
    
    logger.info("=" * 60)
    logger.info("Training Complete!")
    logger.info(f"Final eval loss: {eval_results.get('eval_loss', 'N/A')}")
    logger.info("=" * 60)
    
    return train_result


if __name__ == "__main__":
    main()
