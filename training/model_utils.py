"""
Model utilities for SQL Codegen SLM training.
Module 2.2: Training Pipeline

Handles model loading, quantization configuration, and LoRA setup.
"""

import logging
from typing import Any, Dict, Tuple

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)

logger = logging.getLogger(__name__)


def setup_quantization_config(config: Dict[str, Any]) -> BitsAndBytesConfig:
    """
    Create BitsAndBytesConfig for 4-bit quantization.
    
    Args:
        config: Configuration dictionary with quantization settings
        
    Returns:
        BitsAndBytesConfig object
    """
    quant_config = config.get("quantization", {})
    
    # Map string dtype to torch dtype
    compute_dtype_str = quant_config.get("bnb_4bit_compute_dtype", "float16")
    compute_dtype = getattr(torch, compute_dtype_str, torch.float16)
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=quant_config.get("load_in_4bit", True),
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=quant_config.get("bnb_4bit_use_double_quant", True),
        bnb_4bit_quant_type=quant_config.get("bnb_4bit_quant_type", "nf4"),
    )
    
    logger.info(f"Quantization config: 4-bit={bnb_config.load_in_4bit}, "
                f"compute_dtype={compute_dtype}, quant_type={bnb_config.bnb_4bit_quant_type}")
    
    return bnb_config


def setup_lora_config(config: Dict[str, Any]) -> LoraConfig:
    """
    Create LoraConfig from configuration.
    
    Args:
        config: Configuration dictionary with LoRA settings
        
    Returns:
        LoraConfig object
    """
    lora_config = config.get("lora", {})
    
    peft_config = LoraConfig(
        r=lora_config.get("r", 16),
        lora_alpha=lora_config.get("lora_alpha", 32),
        lora_dropout=lora_config.get("lora_dropout", 0.05),
        target_modules=lora_config.get("target_modules", [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]),
        bias=lora_config.get("bias", "none"),
        task_type=lora_config.get("task_type", "CAUSAL_LM"),
    )
    
    logger.info(f"LoRA config: r={peft_config.r}, alpha={peft_config.lora_alpha}, "
                f"dropout={peft_config.lora_dropout}, targets={peft_config.target_modules}")
    
    return peft_config


def print_trainable_parameters(model: PreTrainedModel) -> Tuple[int, int]:
    """
    Print and return trainable vs total parameters.
    
    Args:
        model: The model to analyze
        
    Returns:
        Tuple of (total_params, trainable_params)
    """
    total_params = 0
    trainable_params = 0
    
    for _, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    trainable_percent = 100 * trainable_params / total_params if total_params > 0 else 0
    
    logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} "
                f"({trainable_percent:.2f}%)")
    print(f"\n{'='*50}")
    print(f"Model Parameters Summary")
    print(f"{'='*50}")
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Trainable %:          {trainable_percent:.2f}%")
    print(f"{'='*50}\n")
    
    return total_params, trainable_params


def apply_lora_to_model(
    model: PreTrainedModel,
    lora_config: LoraConfig
) -> PreTrainedModel:
    """
    Apply LoRA adapters to the model.
    
    Args:
        model: Base model to add LoRA to
        lora_config: LoRA configuration
        
    Returns:
        PEFT model with LoRA adapters
    """
    logger.info("Applying LoRA adapters to model...")
    
    # Prepare model for k-bit training (handles gradient checkpointing, etc.)
    model = prepare_model_for_kbit_training(model)
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    
    # Print parameter summary
    print_trainable_parameters(model)
    
    return model


def load_model_and_tokenizer(
    config: Dict[str, Any]
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Load base model with quantization and tokenizer.
    
    Main entry point for model loading. Loads Mistral-7B with 4-bit
    quantization and applies LoRA adapters.
    
    Args:
        config: Configuration dictionary with model, quantization, and LoRA settings
        
    Returns:
        Tuple of (model with LoRA, tokenizer)
    """
    model_config = config.get("model", {})
    model_name = model_config.get("name", "mistralai/Mistral-7B-v0.1")
    
    logger.info(f"Loading model: {model_name}")
    
    # Setup quantization
    bnb_config = setup_quantization_config(config)
    
    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    
    # Set padding token if not set (Mistral doesn't have one by default)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load model with quantization
    logger.info("Loading model with 4-bit quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    
    # Enable gradient checkpointing for memory efficiency
    if config.get("training", {}).get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")
    
    # Setup and apply LoRA
    lora_config = setup_lora_config(config)
    model = apply_lora_to_model(model, lora_config)
    
    logger.info("Model and tokenizer loaded successfully!")
    
    return model, tokenizer


if __name__ == "__main__":
    # Quick test of config creation (doesn't load actual model)
    import yaml
    
    logging.basicConfig(level=logging.INFO)
    
    # Test with sample config
    sample_config = {
        "model": {
            "name": "mistralai/Mistral-7B-v0.1",
            "max_seq_length": 2048
        },
        "quantization": {
            "load_in_4bit": True,
            "bnb_4bit_compute_dtype": "float16",
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_quant_type": "nf4"
        },
        "lora": {
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "bias": "none",
            "task_type": "CAUSAL_LM"
        }
    }
    
    print("Testing config creation...")
    bnb_config = setup_quantization_config(sample_config)
    lora_config = setup_lora_config(sample_config)
    print("Configs created successfully!")
