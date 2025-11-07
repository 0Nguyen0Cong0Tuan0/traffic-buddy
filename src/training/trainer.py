"""
Training loop for Vietnamese Traffic VQA Model
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from transformers import get_linear_schedule_with_warmup
import logging
from tqdm import tqdm
from pathlib import Path
import json
from typing import Dict, List, Optional
import time
import shutil
from collections import defaultdict

from models.video_llava import extract_answer_letter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Trainer:
    """
    Trainer for Vietnamese Traffic VQA
    """
    
    def __init__(
        self,
        model,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = "cuda",
        output_dir: str = "checkpoints",
        max_grad_norm: float = 1.0,
        logging_steps: int = 10,
        eval_steps: int = 100,
        save_steps: int = 500,
        save_total_limit: int = 3,
        use_amp: bool = True,
        gradient_accumulation_steps: int = 1
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_grad_norm = max_grad_norm
        self.logging_steps = logging_steps
        self.eval_steps = eval_steps
        self.save_steps = save_steps
        self.save_total_limit = save_total_limit
        self.use_amp = use_amp
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # Initialize mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if (use_amp and torch.cuda.is_available()) else None
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_val_accuracy = 0.0
        self.training_history = []
        
        logger.info(f"Trainer initialized. Output dir: {self.output_dir}")
    
    def train(
        self,
        num_epochs: int,
        resume_from_checkpoint: Optional[str] = None
    ):
        """
        Main training loop
        """
        logger.info("="*60)
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info("="*60)

        if resume_from_checkpoint:
            self._load_checkpoint(resume_from_checkpoint)
        
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Train one epoch
            train_metrics = self._train_epoch()
            
            # Evaluate
            val_metrics = self.evaluate()
            
            # Log metrics
            self._log_metrics(train_metrics, val_metrics, epoch)
            
            # Save checkpoint
            self._save_checkpoint(val_metrics['accuracy'])
            
            # Early stopping check
            if self._should_stop_early():
                logger.info("Early stopping triggered!")
                break
        
        logger.info("="*60)
        logger.info("Training completed!")
        logger.info(f"Best validation accuracy: {self.best_val_accuracy:.4f}")
        logger.info("="*60)
        
    def _train_epoch(self) -> Dict[str, float]:
        """
        Train one epoch
        """
        self.model.train()
        
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Training Epoch {self.current_epoch + 1}"
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            loss = self._training_step(batch)
            
            total_loss += loss
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss:.4f}",
                'avg_loss': f"{total_loss / num_batches:.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
            
            # Logging
            if self.global_step % self.logging_steps == 0:
                self._log_training_step(loss)
            
            # Evaluation
            if self.global_step % self.eval_steps == 0:
                val_metrics = self.evaluate()
                self.model.train()  # Back to training mode
                self._save_checkpoint(val_metrics['accuracy'])
            
            self.global_step += 1
        
        avg_loss = total_loss / num_batches
        return {'loss': avg_loss}
    
    def _training_step(self, batch: Dict) -> float:
        """
        Single training step
        """
        # Move batch to device
        frames = batch['frames'].to(self.device)
        prompts = batch['prompts']
        answers = batch['answers']
        
        # Prepare inputs
        inputs = self.model.prepare_inputs(frames, prompts, answers)
        
        # Move inputs to device (only tensors)
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                 for k, v in inputs.items()}
        
        # Forward pass with mixed precision
        if self.use_amp and torch.cuda.is_available():
            with torch.cuda.amp.autocast():
                outputs = self.model(**inputs)
                loss = outputs.loss / self.gradient_accumulation_steps
            
            # Backward pass
            self.scaler.scale(loss).backward()
        else:
            outputs = self.model(**inputs)
            loss = outputs.loss / self.gradient_accumulation_steps
            loss.backward()
        
        # Gradient accumulation
        if (self.global_step + 1) % self.gradient_accumulation_steps == 0:
            if self.use_amp and torch.cuda.is_available():
                # Unscale gradients and clip
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm
                )
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm
                )
                self.optimizer.step()
            
            if self.scheduler is not None:
                self.scheduler.step()
            
            self.optimizer.zero_grad()
        
        return loss.item() * self.gradient_accumulation_steps
    
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate on validation set
        """
        logger.info("\nEvaluating...")
        
        self.model.eval()
        
        correct = 0
        total = 0
        predictions = []
        
        progress_bar = tqdm(self.val_dataloader, desc="Evaluating")
        
        for batch in progress_bar:
            frames = batch['frames'].to(self.device)
            prompts = batch['prompts']
            true_answers = batch['answer_letters']
            choices = batch['choices']
            
            # Generate predictions
            generated_texts = self.model.generate(
                frames,
                prompts,
                max_new_tokens=64,
                temperature=0.1,
                do_sample=False
            )
            
            # Extract answer letters
            from models.video_llava import extract_answer_letter
            
            for gen_text, true_ans, choice_list in zip(
                generated_texts, true_answers, choices
            ):
                pred_ans = extract_answer_letter(gen_text, choice_list)
                predictions.append(pred_ans)
                
                if pred_ans == true_ans:
                    correct += 1
                total += 1
            
            # Update progress
            progress_bar.set_postfix({
                'accuracy': f"{correct / total:.4f}"
            })
        
        accuracy = correct / total if total > 0 else 0.0
        
        logger.info(f"Validation Accuracy: {accuracy:.4f} ({correct}/{total})")
        
        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'predictions': predictions
        }
    
    def _log_training_step(self, loss: float):
        """
        Log training step
        """
        log_data = {
            'step': self.global_step,
            'epoch': self.current_epoch,
            'loss': loss,
            'lr': self.optimizer.param_groups[0]['lr']
        }
        
        # Log to file
        with open(self.output_dir / 'training_log.jsonl', 'a') as f:
            f.write(json.dumps(log_data) + '\n')
    
    def _log_metrics(
        self,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        epoch: int
    ):
        """
        Log epoch metrics
        """
        metrics = {
            'epoch': epoch + 1,
            'train_loss': train_metrics['loss'],
            'val_accuracy': val_metrics['accuracy'],
            'val_correct': val_metrics['correct'],
            'val_total': val_metrics['total']
        }
        
        self.training_history.append(metrics)
        
        logger.info(f"\nEpoch {epoch + 1} Summary:")
        logger.info(f"  Train Loss: {train_metrics['loss']:.4f}")
        logger.info(f"  Val Accuracy: {val_metrics['accuracy']:.4f}")
        
        # Save history
        with open(self.output_dir / 'training_history.json', 'w') as f:
            json.dump(self.training_history, f, indent=2)
    
    def _save_checkpoint(self, val_accuracy: float):
        """
        Save checkpoint
        """
        # Update best model
        if val_accuracy > self.best_val_accuracy:
            self.best_val_accuracy = val_accuracy
            best_model_path = self.output_dir / 'best_model'
            self.model.save_pretrained(str(best_model_path))
            logger.info(f"✓ New best model saved! Accuracy: {val_accuracy:.4f}")
        
        # Save checkpoint
        checkpoint_path = self.output_dir / f'checkpoint-{self.global_step}'
        checkpoint_path.mkdir(exist_ok=True)
        
        self.model.save_pretrained(str(checkpoint_path))
        
        # Save training state
        state = {
            'global_step': self.global_step,
            'epoch': self.current_epoch,
            'best_val_accuracy': self.best_val_accuracy,
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict() if self.scheduler else None
        }
        
        torch.save(state, checkpoint_path / 'training_state.pt')
        
        # Clean up old checkpoints
        self._cleanup_checkpoints()
    
    def _cleanup_checkpoints(self):
        """
        Remove old checkpoints
        """
        checkpoints = sorted(
            [p for p in self.output_dir.iterdir() if p.name.startswith('checkpoint-')],
            key=lambda x: int(x.name.split('-')[1])
        )
        
        if len(checkpoints) > self.save_total_limit:
            for checkpoint in checkpoints[:-self.save_total_limit]:
                logger.info(f"Removing old checkpoint: {checkpoint}")
                import shutil
                shutil.rmtree(checkpoint)
    
    def _load_checkpoint(self, checkpoint_path: str):
        """
        Load checkpoint
        """
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        
        checkpoint_path = Path(checkpoint_path)
        state = torch.load(checkpoint_path / 'training_state.pt')
        
        self.global_step = state['global_step']
        self.current_epoch = state['epoch']
        self.best_val_accuracy = state['best_val_accuracy']
        
        self.optimizer.load_state_dict(state['optimizer_state'])
        if self.scheduler and state['scheduler_state']:
            self.scheduler.load_state_dict(state['scheduler_state'])
        
        logger.info(f"Resumed from step {self.global_step}, epoch {self.current_epoch}")
    
    def _should_stop_early(self, patience: int = 5) -> bool:
        """Check if should stop early"""
        if len(self.training_history) < patience:
            return False
        
        recent_accuracies = [
            h['val_accuracy'] for h in self.training_history[-patience:]
        ]
        
        # Stop if no improvement in last `patience` epochs
        if all(acc <= self.best_val_accuracy for acc in recent_accuracies):
            return True
        
        return False


def create_optimizer(model, learning_rate: float = 2e-4, weight_decay: float = 0.01):
    """
    Create AdamW optimizer
    """
    # Separate parameters for weight decay
    no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']
    
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() 
                      if not any(nd in n for nd in no_decay) and p.requires_grad],
            'weight_decay': weight_decay,
        },
        {
            'params': [p for n, p in model.named_parameters() 
                      if any(nd in n for nd in no_decay) and p.requires_grad],
            'weight_decay': 0.0,
        }
    ]
    
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    return optimizer


def create_scheduler(
    optimizer,
    num_training_steps: int,
    num_warmup_steps: int,
    scheduler_type: str = "cosine"
):
    """
    Create learning rate scheduler
    """
    if scheduler_type == "cosine":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
    elif scheduler_type == "onecycle":
        scheduler = OneCycleLR(
            optimizer,
            max_lr=optimizer.defaults['lr'],
            total_steps=num_training_steps,
            pct_start=0.1
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    return scheduler