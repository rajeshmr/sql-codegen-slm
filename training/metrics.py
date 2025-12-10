"""
Evaluation metrics for SQL Codegen SLM.
Module 2.2: Training Pipeline

Provides metrics computation for training evaluation and
SQL generation quality assessment.
"""

import logging
import math
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

logger = logging.getLogger(__name__)


def compute_metrics(eval_pred) -> Dict[str, float]:
    """
    Compute metrics from evaluation predictions.
    
    Used by HuggingFace Trainer during evaluation.
    Computes perplexity from loss.
    
    Args:
        eval_pred: EvalPrediction object with predictions and labels
        
    Returns:
        Dictionary with computed metrics
    """
    # For language modeling, we compute perplexity from loss
    # The Trainer computes loss automatically, so we just need to
    # convert it to perplexity
    
    # Note: eval_pred contains logits and labels
    # But for causal LM, we typically just use the loss
    # which is computed by the model
    
    predictions = eval_pred.predictions
    labels = eval_pred.label_ids
    
    # If predictions are logits, compute cross-entropy loss
    if len(predictions.shape) == 3:  # (batch, seq_len, vocab_size)
        # Shift for causal LM
        shift_logits = predictions[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Flatten
        loss_fct = torch.nn.CrossEntropyLoss(reduction='mean', ignore_index=-100)
        shift_logits = torch.tensor(shift_logits).view(-1, shift_logits.shape[-1])
        shift_labels = torch.tensor(shift_labels).view(-1)
        
        loss = loss_fct(shift_logits, shift_labels).item()
    else:
        # Predictions might already be loss values
        loss = float(np.mean(predictions))
    
    # Compute perplexity
    try:
        perplexity = math.exp(loss)
    except OverflowError:
        perplexity = float('inf')
    
    return {
        "perplexity": perplexity,
    }


def generate_sql(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.1,
    do_sample: bool = True
) -> str:
    """
    Generate SQL from a prompt.
    
    Args:
        model: The fine-tuned model
        tokenizer: The tokenizer
        prompt: Input prompt with schema and question
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        do_sample: Whether to use sampling
        
    Returns:
        Generated SQL string
    """
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract SQL (after [/INST] tag)
    if "[/INST]" in generated:
        sql = generated.split("[/INST]")[-1].strip()
    else:
        sql = generated.strip()
    
    # Clean up
    sql = sql.replace("</s>", "").strip()
    
    return sql


def normalize_sql(sql: str) -> str:
    """
    Normalize SQL for comparison.
    
    Args:
        sql: SQL string to normalize
        
    Returns:
        Normalized SQL string
    """
    # Lowercase
    sql = sql.lower()
    
    # Remove extra whitespace
    sql = " ".join(sql.split())
    
    # Remove trailing semicolon
    sql = sql.rstrip(";").strip()
    
    return sql


def exact_match(generated: str, reference: str) -> bool:
    """
    Check if generated SQL exactly matches reference.
    
    Args:
        generated: Generated SQL
        reference: Reference SQL
        
    Returns:
        True if exact match after normalization
    """
    return normalize_sql(generated) == normalize_sql(reference)


def evaluate_sql_generation(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    test_examples: List[Dict[str, Any]],
    num_samples: int = 50,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Evaluate SQL generation quality on test examples.
    
    Takes random samples from test set, generates SQL for each,
    and compares against ground truth.
    
    Args:
        model: Fine-tuned model
        tokenizer: Tokenizer
        test_examples: List of test examples with messages
        num_samples: Number of samples to evaluate
        seed: Random seed for reproducibility
        
    Returns:
        Evaluation report with metrics and examples
    """
    random.seed(seed)
    
    # Sample examples
    if len(test_examples) > num_samples:
        samples = random.sample(test_examples, num_samples)
    else:
        samples = test_examples
    
    logger.info(f"Evaluating SQL generation on {len(samples)} samples...")
    
    results = []
    exact_matches = 0
    
    for i, example in enumerate(samples):
        messages = example.get("messages", [])
        
        # Extract components
        system_msg = ""
        user_msg = ""
        reference_sql = ""
        
        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            elif msg["role"] == "user":
                user_msg = msg["content"]
            elif msg["role"] == "assistant":
                reference_sql = msg["content"]
        
        # Create prompt
        if system_msg:
            prompt = f"<s>[INST] {system_msg}\n{user_msg} [/INST]"
        else:
            prompt = f"<s>[INST] {user_msg} [/INST]"
        
        # Generate SQL
        try:
            generated_sql = generate_sql(model, tokenizer, prompt)
        except Exception as e:
            logger.warning(f"Error generating SQL for sample {i}: {e}")
            generated_sql = ""
        
        # Check exact match
        is_match = exact_match(generated_sql, reference_sql)
        if is_match:
            exact_matches += 1
        
        results.append({
            "index": i,
            "prompt": user_msg[:200] + "..." if len(user_msg) > 200 else user_msg,
            "reference": reference_sql,
            "generated": generated_sql,
            "exact_match": is_match
        })
        
        # Log progress
        if (i + 1) % 10 == 0:
            logger.info(f"Evaluated {i + 1}/{len(samples)} samples")
    
    # Compute metrics
    accuracy = exact_matches / len(samples) if samples else 0
    
    report = {
        "num_samples": len(samples),
        "exact_match_accuracy": accuracy,
        "exact_matches": exact_matches,
        "results": results[:10],  # Include first 10 for inspection
    }
    
    logger.info(f"SQL Generation Evaluation Complete:")
    logger.info(f"  Samples: {len(samples)}")
    logger.info(f"  Exact Match Accuracy: {accuracy:.2%}")
    
    return report


if __name__ == "__main__":
    # Test normalization
    logging.basicConfig(level=logging.INFO)
    
    sql1 = "SELECT COUNT(*) FROM users;"
    sql2 = "select count(*)  from  users"
    
    print(f"SQL1: {sql1}")
    print(f"SQL2: {sql2}")
    print(f"Normalized SQL1: {normalize_sql(sql1)}")
    print(f"Normalized SQL2: {normalize_sql(sql2)}")
    print(f"Exact match: {exact_match(sql1, sql2)}")
