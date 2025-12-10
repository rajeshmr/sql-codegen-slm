"""
Tests for the training pipeline.
Module 2.2: Training Pipeline

Tests data loading, model configuration, and training setup.
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestDataLoader:
    """Tests for data_loader.py"""
    
    def test_format_instruction_example_basic(self):
        """Test basic instruction formatting."""
        from training.data_loader import format_instruction_example
        
        example = {
            "messages": [
                {"role": "system", "content": "You are a SQL expert."},
                {"role": "user", "content": "Question: How many users?"},
                {"role": "assistant", "content": "SELECT COUNT(*) FROM users;"}
            ]
        }
        
        formatted = format_instruction_example(example)
        
        assert "<s>" in formatted
        assert "[INST]" in formatted
        assert "[/INST]" in formatted
        assert "</s>" in formatted
        assert "You are a SQL expert." in formatted
        assert "How many users?" in formatted
        assert "SELECT COUNT(*) FROM users;" in formatted
    
    def test_format_instruction_example_no_system(self):
        """Test formatting without system message."""
        from training.data_loader import format_instruction_example
        
        example = {
            "messages": [
                {"role": "user", "content": "Question: How many users?"},
                {"role": "assistant", "content": "SELECT COUNT(*) FROM users;"}
            ]
        }
        
        formatted = format_instruction_example(example)
        
        assert "<s>[INST]" in formatted
        assert "[/INST]" in formatted
        assert "</s>" in formatted
    
    def test_format_instruction_example_structure(self):
        """Test that format follows Mistral instruction format."""
        from training.data_loader import format_instruction_example
        
        example = {
            "messages": [
                {"role": "system", "content": "System prompt"},
                {"role": "user", "content": "User question"},
                {"role": "assistant", "content": "Assistant response"}
            ]
        }
        
        formatted = format_instruction_example(example)
        
        # Should follow: <s>[INST] {system}\n{user} [/INST] {assistant}</s>
        assert formatted.startswith("<s>[INST]")
        assert formatted.endswith("</s>")
        assert "[/INST]" in formatted
        
        # Check order
        inst_pos = formatted.find("[INST]")
        end_inst_pos = formatted.find("[/INST]")
        assert inst_pos < end_inst_pos
    
    def test_sql_dataset_load_jsonl(self):
        """Test loading JSONL data."""
        from training.data_loader import SQLDataset
        
        # Create temp JSONL file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for i in range(5):
                example = {
                    "messages": [
                        {"role": "system", "content": "System"},
                        {"role": "user", "content": f"Question {i}"},
                        {"role": "assistant", "content": f"SELECT {i}"}
                    ]
                }
                f.write(json.dumps(example) + "\n")
            temp_path = f.name
        
        try:
            # Mock tokenizer
            mock_tokenizer = MagicMock()
            mock_tokenizer.return_value = {
                "input_ids": MagicMock(squeeze=lambda x: MagicMock(shape=(512,))),
                "attention_mask": MagicMock(squeeze=lambda x: MagicMock())
            }
            mock_tokenizer.pad_token = "<pad>"
            
            # Create dataset (without actually tokenizing)
            with patch('training.data_loader.tokenize_function') as mock_tokenize:
                mock_tokenize.return_value = {
                    "input_ids": MagicMock(),
                    "attention_mask": MagicMock(),
                    "labels": MagicMock()
                }
                
                dataset = SQLDataset(
                    data_file=temp_path,
                    tokenizer=mock_tokenizer,
                    max_seq_length=512,
                    cache_tokenized=False
                )
                
                assert len(dataset) == 5
        finally:
            os.unlink(temp_path)
    
    def test_load_training_data_function_exists(self):
        """Test that load_training_data function exists."""
        from training.data_loader import load_training_data
        assert callable(load_training_data)
    
    def test_create_data_collator_function_exists(self):
        """Test that create_data_collator function exists."""
        from training.data_loader import create_data_collator
        assert callable(create_data_collator)


class TestModelUtils:
    """Tests for model_utils.py"""
    
    def test_setup_quantization_config(self):
        """Test quantization config creation."""
        from training.model_utils import setup_quantization_config
        
        config = {
            "quantization": {
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": "float16",
                "bnb_4bit_use_double_quant": True,
                "bnb_4bit_quant_type": "nf4"
            }
        }
        
        bnb_config = setup_quantization_config(config)
        
        assert bnb_config.load_in_4bit == True
        assert bnb_config.bnb_4bit_quant_type == "nf4"
        assert bnb_config.bnb_4bit_use_double_quant == True
    
    def test_setup_lora_config(self):
        """Test LoRA config creation."""
        from training.model_utils import setup_lora_config
        
        config = {
            "lora": {
                "r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.05,
                "target_modules": ["q_proj", "v_proj"],
                "bias": "none",
                "task_type": "CAUSAL_LM"
            }
        }
        
        lora_config = setup_lora_config(config)
        
        assert lora_config.r == 16
        assert lora_config.lora_alpha == 32
        assert lora_config.lora_dropout == 0.05
        assert "q_proj" in lora_config.target_modules
        assert lora_config.task_type == "CAUSAL_LM"
    
    def test_setup_lora_config_defaults(self):
        """Test LoRA config with defaults."""
        from training.model_utils import setup_lora_config
        
        config = {"lora": {}}
        lora_config = setup_lora_config(config)
        
        # Should use defaults
        assert lora_config.r == 16
        assert lora_config.lora_alpha == 32
    
    def test_print_trainable_parameters(self):
        """Test parameter counting."""
        from training.model_utils import print_trainable_parameters
        import torch.nn as nn
        
        # Create simple model
        model = nn.Linear(10, 5)
        
        total, trainable = print_trainable_parameters(model)
        
        assert total > 0
        assert trainable > 0
        assert trainable <= total


class TestTrainScript:
    """Tests for train.py"""
    
    def test_load_config(self):
        """Test configuration loading."""
        from training.train import load_config
        
        # Create temp config
        config = {
            "model": {"name": "test-model"},
            "lora": {"r": 8},
            "training": {"epochs": 1},
            "data": {"train_file": "train.jsonl"}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            temp_path = f.name
        
        try:
            loaded = load_config(temp_path)
            assert loaded["model"]["name"] == "test-model"
            assert loaded["lora"]["r"] == 8
        finally:
            os.unlink(temp_path)
    
    def test_load_config_missing_section(self):
        """Test config validation for missing sections."""
        from training.train import load_config
        
        # Config missing required section
        config = {
            "model": {"name": "test"},
            # Missing: lora, training, data
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match="Missing required config section"):
                load_config(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_setup_training_args(self):
        """Test training arguments setup."""
        from training.train import setup_training_args
        
        config = {
            "training": {
                "output_dir": "/tmp/test_output",
                "num_train_epochs": 2,
                "per_device_train_batch_size": 4,
                "learning_rate": 0.0001,
                "save_steps": 500,
                "eval_steps": 500,
                "bf16": False,  # Disable for CPU testing
                "fp16": False,
            },
            "logging": {
                "tensorboard_dir": "/tmp/tensorboard"
            }
        }
        
        args = setup_training_args(config)
        
        assert args.output_dir == "/tmp/test_output"
        assert args.num_train_epochs == 2
        assert args.per_device_train_batch_size == 4
        assert args.learning_rate == 0.0001
        assert args.save_steps == 500
    
    def test_parse_args_function_exists(self):
        """Test that parse_args function exists."""
        from training.train import parse_args
        assert callable(parse_args)
    
    def test_main_function_exists(self):
        """Test that main function exists."""
        from training.train import main
        assert callable(main)


class TestCallbacks:
    """Tests for callbacks.py"""
    
    def test_gcs_checkpoint_callback_init(self):
        """Test GCS callback initialization."""
        from training.callbacks import GCSCheckpointCallback
        
        callback = GCSCheckpointCallback(
            bucket_name="test-bucket",
            gcs_prefix="models"
        )
        
        assert callback.bucket_name == "test-bucket"
        assert callback.gcs_prefix == "models"
        assert callback.gcs_path == "gs://test-bucket/models"
    
    def test_training_progress_callback_init(self):
        """Test progress callback initialization."""
        from training.callbacks import TrainingProgressCallback
        
        callback = TrainingProgressCallback()
        
        assert callback.start_time is None
        assert callback.total_steps == 0
    
    def test_early_stopping_callback_init(self):
        """Test early stopping callback initialization."""
        from training.callbacks import EarlyStoppingCallback
        
        callback = EarlyStoppingCallback(patience=5, min_delta=0.001)
        
        assert callback.patience == 5
        assert callback.min_delta == 0.001
        assert callback.best_loss is None
        assert callback.wait_count == 0


class TestMetrics:
    """Tests for metrics.py"""
    
    def test_normalize_sql(self):
        """Test SQL normalization."""
        from training.metrics import normalize_sql
        
        sql1 = "SELECT COUNT(*) FROM users;"
        sql2 = "  select  count(*)   from   users  ;  "
        
        assert normalize_sql(sql1) == normalize_sql(sql2)
    
    def test_exact_match(self):
        """Test exact match comparison."""
        from training.metrics import exact_match
        
        assert exact_match(
            "SELECT * FROM users",
            "select * from users"
        ) == True
        
        assert exact_match(
            "SELECT * FROM users",
            "SELECT id FROM users"
        ) == False
    
    def test_compute_metrics_function_exists(self):
        """Test that compute_metrics function exists."""
        from training.metrics import compute_metrics
        assert callable(compute_metrics)
    
    def test_evaluate_sql_generation_function_exists(self):
        """Test that evaluate_sql_generation function exists."""
        from training.metrics import evaluate_sql_generation
        assert callable(evaluate_sql_generation)


class TestConfigIntegration:
    """Test configuration file integration."""
    
    def test_mistral_config_exists(self):
        """Test that Mistral config file exists."""
        config_path = PROJECT_ROOT / "training" / "configs" / "mistral_lora_config.yaml"
        assert config_path.exists()
    
    def test_mistral_config_valid_yaml(self):
        """Test that Mistral config is valid YAML."""
        config_path = PROJECT_ROOT / "training" / "configs" / "mistral_lora_config.yaml"
        
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        assert isinstance(config, dict)
    
    def test_mistral_config_has_required_sections(self):
        """Test that config has all required sections."""
        config_path = PROJECT_ROOT / "training" / "configs" / "mistral_lora_config.yaml"
        
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        assert "model" in config
        assert "lora" in config
        assert "training" in config
        assert "data" in config
    
    def test_mistral_config_model_name(self):
        """Test that config specifies Mistral model."""
        config_path = PROJECT_ROOT / "training" / "configs" / "mistral_lora_config.yaml"
        
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        assert "mistral" in config["model"]["name"].lower()
    
    def test_mistral_config_lora_params(self):
        """Test LoRA parameters in config."""
        config_path = PROJECT_ROOT / "training" / "configs" / "mistral_lora_config.yaml"
        
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        lora = config["lora"]
        assert "r" in lora
        assert "lora_alpha" in lora
        assert "target_modules" in lora
        assert isinstance(lora["target_modules"], list)


class TestEndToEnd:
    """End-to-end smoke tests."""
    
    def test_import_all_modules(self):
        """Test that all training modules can be imported."""
        from training import data_loader
        from training import model_utils
        from training import train
        from training import callbacks
        from training import metrics
        
        assert data_loader is not None
        assert model_utils is not None
        assert train is not None
        assert callbacks is not None
        assert metrics is not None
    
    def test_format_and_check_example(self):
        """Test formatting a complete example."""
        from training.data_loader import format_instruction_example
        
        # Real-world-like example
        example = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert PostgreSQL query generator."
                },
                {
                    "role": "user",
                    "content": "Schema:\nCREATE TABLE users (id INT, name TEXT);\n\nQuestion: How many users are there?"
                },
                {
                    "role": "assistant",
                    "content": "SELECT COUNT(*) FROM users;"
                }
            ]
        }
        
        formatted = format_instruction_example(example)
        
        # Verify structure
        assert formatted.startswith("<s>[INST]")
        assert formatted.endswith("</s>")
        assert "PostgreSQL" in formatted
        assert "CREATE TABLE" in formatted
        assert "SELECT COUNT(*)" in formatted
