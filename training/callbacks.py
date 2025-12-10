"""
Custom training callbacks for SQL Codegen SLM.
Module 2.2: Training Pipeline

Provides callbacks for GCS checkpoint syncing, progress logging,
and early stopping.
"""

import logging
import os
import subprocess
import time
from datetime import datetime, timedelta
from typing import Optional

from transformers import TrainerCallback, TrainerControl, TrainerState
from transformers.training_args import TrainingArguments

logger = logging.getLogger(__name__)


class GCSCheckpointCallback(TrainerCallback):
    """
    Callback to sync checkpoints to Google Cloud Storage after saving.
    
    Automatically uploads checkpoints to GCS bucket after each save,
    ensuring training progress is backed up to cloud storage.
    """
    
    def __init__(self, bucket_name: str, gcs_prefix: str = "models"):
        """
        Initialize the callback.
        
        Args:
            bucket_name: GCS bucket name (without gs:// prefix)
            gcs_prefix: Prefix path in bucket for checkpoints
        """
        self.bucket_name = bucket_name
        self.gcs_prefix = gcs_prefix
        self.gcs_path = f"gs://{bucket_name}/{gcs_prefix}"
        logger.info(f"GCS checkpoint callback initialized: {self.gcs_path}")
    
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        """
        Called after a checkpoint is saved.
        
        Syncs the checkpoint directory to GCS.
        """
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        
        if os.path.exists(checkpoint_dir):
            gcs_checkpoint_path = f"{self.gcs_path}/checkpoint-{state.global_step}"
            
            try:
                logger.info(f"Syncing checkpoint to GCS: {gcs_checkpoint_path}")
                
                # Use gsutil to sync
                result = subprocess.run(
                    ["gsutil", "-m", "rsync", "-r", checkpoint_dir, gcs_checkpoint_path],
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minute timeout
                )
                
                if result.returncode == 0:
                    logger.info(f"✅ Checkpoint synced to {gcs_checkpoint_path}")
                else:
                    logger.warning(f"GCS sync warning: {result.stderr}")
                    
            except subprocess.TimeoutExpired:
                logger.error("GCS sync timed out after 5 minutes")
            except FileNotFoundError:
                logger.warning("gsutil not found - skipping GCS sync")
            except Exception as e:
                logger.error(f"Error syncing to GCS: {e}")
        
        return control


class TrainingProgressCallback(TrainerCallback):
    """
    Callback for detailed training progress logging.
    
    Logs step, loss, learning rate, and estimates time remaining.
    """
    
    def __init__(self):
        """Initialize the callback."""
        self.start_time: Optional[float] = None
        self.last_log_time: Optional[float] = None
        self.total_steps: int = 0
    
    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        """Called at the beginning of training."""
        self.start_time = time.time()
        self.last_log_time = self.start_time
        self.total_steps = state.max_steps
        
        print("\n" + "=" * 60)
        print("🚀 TRAINING STARTED")
        print("=" * 60)
        print(f"Total steps: {self.total_steps}")
        print(f"Epochs: {args.num_train_epochs}")
        print(f"Batch size: {args.per_device_train_batch_size}")
        print(f"Gradient accumulation: {args.gradient_accumulation_steps}")
        print(f"Effective batch size: {args.per_device_train_batch_size * args.gradient_accumulation_steps}")
        print(f"Learning rate: {args.learning_rate}")
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60 + "\n")
        
        return control
    
    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict = None,
        **kwargs
    ):
        """Called when logging metrics."""
        if logs is None:
            return control
        
        current_time = time.time()
        
        # Extract metrics
        step = state.global_step
        loss = logs.get("loss", logs.get("train_loss", "N/A"))
        lr = logs.get("learning_rate", "N/A")
        epoch = logs.get("epoch", state.epoch)
        
        # Calculate progress and ETA
        if self.start_time and self.total_steps > 0:
            elapsed = current_time - self.start_time
            progress = step / self.total_steps
            
            if progress > 0:
                estimated_total = elapsed / progress
                remaining = estimated_total - elapsed
                eta = datetime.now() + timedelta(seconds=remaining)
                eta_str = eta.strftime("%H:%M:%S")
            else:
                eta_str = "Calculating..."
        else:
            eta_str = "N/A"
        
        # Format loss
        if isinstance(loss, float):
            loss_str = f"{loss:.4f}"
        else:
            loss_str = str(loss)
        
        # Format learning rate
        if isinstance(lr, float):
            lr_str = f"{lr:.2e}"
        else:
            lr_str = str(lr)
        
        # Log progress
        progress_pct = (step / self.total_steps * 100) if self.total_steps > 0 else 0
        logger.info(
            f"Step {step}/{self.total_steps} ({progress_pct:.1f}%) | "
            f"Loss: {loss_str} | LR: {lr_str} | Epoch: {epoch:.2f} | ETA: {eta_str}"
        )
        
        return control
    
    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        """Called at the end of training."""
        if self.start_time:
            total_time = time.time() - self.start_time
            hours = int(total_time // 3600)
            minutes = int((total_time % 3600) // 60)
            seconds = int(total_time % 60)
            
            print("\n" + "=" * 60)
            print("✅ TRAINING COMPLETED")
            print("=" * 60)
            print(f"Total steps: {state.global_step}")
            print(f"Total time: {hours}h {minutes}m {seconds}s")
            print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 60 + "\n")
        
        return control


class EarlyStoppingCallback(TrainerCallback):
    """
    Callback for early stopping based on validation loss.
    
    Stops training if validation loss doesn't improve for N evaluations.
    """
    
    def __init__(self, patience: int = 3, min_delta: float = 0.01):
        """
        Initialize the callback.
        
        Args:
            patience: Number of evaluations to wait for improvement
            min_delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss: Optional[float] = None
        self.wait_count = 0
        
        logger.info(f"Early stopping enabled: patience={patience}, min_delta={min_delta}")
    
    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: dict = None,
        **kwargs
    ):
        """Called after evaluation."""
        if metrics is None:
            return control
        
        eval_loss = metrics.get("eval_loss")
        
        if eval_loss is None:
            return control
        
        if self.best_loss is None:
            self.best_loss = eval_loss
            logger.info(f"Initial eval loss: {eval_loss:.4f}")
        elif eval_loss < self.best_loss - self.min_delta:
            # Improvement
            improvement = self.best_loss - eval_loss
            logger.info(f"Eval loss improved: {self.best_loss:.4f} -> {eval_loss:.4f} "
                       f"(Δ={improvement:.4f})")
            self.best_loss = eval_loss
            self.wait_count = 0
        else:
            # No improvement
            self.wait_count += 1
            logger.info(f"No improvement in eval loss. "
                       f"Best: {self.best_loss:.4f}, Current: {eval_loss:.4f}. "
                       f"Patience: {self.wait_count}/{self.patience}")
            
            if self.wait_count >= self.patience:
                logger.warning(f"Early stopping triggered after {self.wait_count} evaluations "
                             f"without improvement")
                control.should_training_stop = True
        
        return control
