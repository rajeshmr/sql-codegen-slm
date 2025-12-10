"""
Data loading module for SQL Codegen SLM training.
Module 2.2: Training Pipeline

Handles loading JSONL data, formatting for instruction tuning,
and creating PyTorch datasets for training.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from transformers.data.data_collator import DataCollatorForLanguageModeling

logger = logging.getLogger(__name__)


def format_instruction_example(example: Dict[str, Any]) -> str:
    """
    Format a single example into Mistral instruction format.
    
    Takes example with "messages" field containing system, user, assistant messages
    and formats into: <s>[INST] {system}\n{user} [/INST] {assistant}</s>
    
    Args:
        example: Dictionary with "messages" key containing list of role/content dicts
        
    Returns:
        Formatted text string ready for tokenization
    """
    messages = example.get("messages", [])
    
    system_content = ""
    user_content = ""
    assistant_content = ""
    
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        
        if role == "system":
            system_content = content
        elif role == "user":
            user_content = content
        elif role == "assistant":
            assistant_content = content
    
    # Mistral instruction format
    # <s>[INST] {system}\n{user} [/INST] {assistant}</s>
    if system_content:
        formatted = f"<s>[INST] {system_content}\n{user_content} [/INST] {assistant_content}</s>"
    else:
        formatted = f"<s>[INST] {user_content} [/INST] {assistant_content}</s>"
    
    return formatted


def tokenize_function(
    text: str,
    tokenizer: PreTrainedTokenizer,
    max_seq_length: int
) -> Dict[str, torch.Tensor]:
    """
    Tokenize formatted text for training.
    
    Args:
        text: Formatted instruction text
        tokenizer: HuggingFace tokenizer
        max_seq_length: Maximum sequence length (truncate if longer)
        
    Returns:
        Dictionary with input_ids, attention_mask, and labels
    """
    # Tokenize
    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=max_seq_length,
        padding="max_length",
        return_tensors="pt"
    )
    
    # For causal LM, labels are the same as input_ids
    # The model will shift internally for next-token prediction
    tokenized["labels"] = tokenized["input_ids"].clone()
    
    # Squeeze batch dimension since we're processing one at a time
    return {
        "input_ids": tokenized["input_ids"].squeeze(0),
        "attention_mask": tokenized["attention_mask"].squeeze(0),
        "labels": tokenized["labels"].squeeze(0)
    }


class SQLDataset(Dataset):
    """
    Custom PyTorch Dataset for SQL training examples.
    
    Handles lazy loading and caching of tokenized examples for memory efficiency.
    """
    
    def __init__(
        self,
        data_file: str,
        tokenizer: PreTrainedTokenizer,
        max_seq_length: int = 2048,
        cache_tokenized: bool = True,
        max_samples: Optional[int] = None
    ):
        """
        Initialize the dataset.
        
        Args:
            data_file: Path to JSONL file with training examples
            tokenizer: HuggingFace tokenizer
            max_seq_length: Maximum sequence length
            cache_tokenized: Whether to cache tokenized examples in memory
            max_samples: Optional limit on number of samples to load (for testing)
        """
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.cache_tokenized = cache_tokenized
        self.cache: Dict[int, Dict[str, torch.Tensor]] = {}
        
        # Load examples from JSONL
        self.examples = self._load_jsonl(data_file, max_samples)
        logger.info(f"Loaded {len(self.examples)} examples from {data_file}")
    
    def _load_jsonl(self, file_path: str, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load examples from JSONL file."""
        examples = []
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        examples.append(json.loads(line))
                        # Stop if we've reached max_samples
                        if max_samples and len(examples) >= max_samples:
                            break
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipping invalid JSON at line {line_num}: {e}")
        
        return examples
    
    def __len__(self) -> int:
        """Return number of examples."""
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single tokenized example.
        
        Args:
            idx: Index of example to retrieve
            
        Returns:
            Dictionary with input_ids, attention_mask, labels tensors
        """
        # Check cache first
        if self.cache_tokenized and idx in self.cache:
            return self.cache[idx]
        
        # Format and tokenize
        example = self.examples[idx]
        formatted_text = format_instruction_example(example)
        tokenized = tokenize_function(
            formatted_text,
            self.tokenizer,
            self.max_seq_length
        )
        
        # Cache if enabled
        if self.cache_tokenized:
            self.cache[idx] = tokenized
        
        return tokenized


def create_data_collator(tokenizer: PreTrainedTokenizer) -> DataCollatorForLanguageModeling:
    """
    Create a data collator for batching training examples.
    
    Args:
        tokenizer: HuggingFace tokenizer
        
    Returns:
        DataCollatorForLanguageModeling configured for causal LM
    """
    return DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
        pad_to_multiple_of=8  # Efficient for GPU
    )


def load_training_data(
    train_file: str,
    val_file: str,
    tokenizer: PreTrainedTokenizer,
    max_seq_length: int = 2048,
    max_samples: Optional[int] = None
) -> Tuple[SQLDataset, SQLDataset]:
    """
    Load training and validation datasets.
    
    Main entry point for data loading. Creates SQLDataset objects
    for both training and validation data.
    
    Args:
        train_file: Path to training JSONL file
        val_file: Path to validation JSONL file
        tokenizer: HuggingFace tokenizer
        max_seq_length: Maximum sequence length for tokenization
        max_samples: Optional limit on number of samples to load (for testing)
        
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    logger.info(f"Loading training data from {train_file}")
    train_dataset = SQLDataset(
        data_file=train_file,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        cache_tokenized=True,
        max_samples=max_samples
    )
    
    logger.info(f"Loading validation data from {val_file}")
    val_dataset = SQLDataset(
        data_file=val_file,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        cache_tokenized=True,
        max_samples=max_samples
    )
    
    logger.info(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    return train_dataset, val_dataset


if __name__ == "__main__":
    # Quick test
    from transformers import AutoTokenizer
    
    logging.basicConfig(level=logging.INFO)
    
    # Test format function
    example = {
        "messages": [
            {"role": "system", "content": "You are a SQL expert."},
            {"role": "user", "content": "Schema: CREATE TABLE users (id INT);\nQuestion: How many users?"},
            {"role": "assistant", "content": "SELECT COUNT(*) FROM users;"}
        ]
    }
    
    formatted = format_instruction_example(example)
    print("Formatted example:")
    print(formatted)
    print()
    
    # Test with actual tokenizer if available
    try:
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
        tokenized = tokenize_function(formatted, tokenizer, 512)
        print(f"Tokenized shape: {tokenized['input_ids'].shape}")
    except Exception as e:
        print(f"Tokenizer test skipped: {e}")
