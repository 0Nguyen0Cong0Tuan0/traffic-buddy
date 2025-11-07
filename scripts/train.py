"""
Main training script for Vietnamese Traffic VQA

Supports two modes:
- FULL MODE: High accuracy, requires 16GB+ GPU (use training_config.yaml)
- LITE MODE: Good accuracy, requires 8GB+ GPU (use training_config_lite.yaml)

Usage:
    # Full mode (powerful machine)
    python scripts/train.py --config configs/training_config.yaml
    
    # Lite mode (weaker machine)
    python scripts/train.py --config configs/training_config_lite.yaml
    
    # Auto-detect and use appropriate mode
    python scripts/train.py --mode auto
"""
import argparse
import yaml
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data.dataset import VietnameseTrafficDataset, collate_fn
from data.augmentation import get_train_augmentation, get_val_augmentation
from models.video_llava import VietnameseTrafficVQAModel
from training.trainer import Trainer, create_optimizer, create_scheduler

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def detect_gpu_mode() -> str:
    """
    Auto-detect appropriate training mode based on GPU memory
    Returns: 'full' or 'lite'
    """
    if not torch.cuda.is_available():
        logger.warning("No GPU detected! Training will be very slow.")
        return 'lite'
    
    try:
        # Get GPU memory
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9  # GB
        logger.info(f"Detected GPU memory: {gpu_memory:.1f} GB")
        
        if gpu_memory >= 16:
            logger.info("✓ Sufficient memory for FULL mode (7B model)")
            return 'full'
        elif gpu_memory >= 8:
            logger.info("⚠ Limited memory - using LITE mode (2B model)")
            return 'lite'
        else:
            logger.warning("⚠ Very limited GPU memory - using LITE mode with caution")
            return 'lite'
    except Exception as e:
        logger.warning(f"Could not detect GPU memory: {e}")
        logger.info("Defaulting to LITE mode for safety")
        return 'lite'


def create_dataloaders(config: dict):
    """Create train and validation dataloaders"""
    logger.info("Creating dataloaders...")
    
    # Training dataset
    train_dataset = VietnameseTrafficDataset(
        json_path=config['data']['train_json'],
        video_root=config['data']['video_root'],
        num_frames=config['data']['num_frames'],
        image_size=tuple(config['data']['image_size']),
        use_support_frames=config['data']['use_support_frames'],
        transform=get_train_augmentation() if config['training']['use_augmentation'] else None
    )
    
    # Validation dataset
    val_dataset = VietnameseTrafficDataset(
        json_path=config['data']['val_json'],
        video_root=config['data']['video_root'],
        num_frames=config['data']['num_frames'],
        image_size=tuple(config['data']['image_size']),
        use_support_frames=config['data']['use_support_frames'],
        transform=get_val_augmentation()
    )
    
    logger.info(f"Train dataset: {len(train_dataset)} samples")
    logger.info(f"Val dataset: {len(val_dataset)} samples")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['eval_batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser(description='Train Vietnamese Traffic VQA Model')
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to training configuration file'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['full', 'lite', 'auto'],
        default=None,
        help='Training mode: full (16GB+ GPU), lite (8GB+ GPU), auto (detect)'
    )
    parser.add_argument(
        '--resume_from_checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint to resume training'
    )
    
    args = parser.parse_args()
    
    # Auto-detect mode based on GPU memory
    if args.mode == 'auto' or (args.config is None and args.mode is None):
        logger.info("Auto-detecting GPU capabilities...")
        args.mode = detect_gpu_mode()
        logger.info(f"Selected mode: {args.mode.upper()}")
    
    # Select config based on mode
    if args.config is None:
        if args.mode == 'lite':
            args.config = 'configs/training_config_lite.yaml'
            logger.info("Using LITE configuration for weaker machines")
        else:
            args.config = 'configs/training_config.yaml'
            logger.info("Using FULL configuration for powerful machines")
    
    # Load configuration
    config = load_config(args.config)
    logger.info(f"Loaded config from {args.config}")
    
    # Override mode if specified
    if args.mode:
        config['model']['mode'] = args.mode
    
    # Set random seed
    torch.manual_seed(config['training']['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['training']['seed'])
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(config)
    
    # Initialize model
    logger.info("Initializing model...")
    model = VietnameseTrafficVQAModel(
        model_name=config['model']['model_name'],
        mode=config['model'].get('mode', 'full'),
        use_lora=config['model']['use_lora'],
        lora_r=config['model']['lora_r'],
        lora_alpha=config['model']['lora_alpha'],
        lora_dropout=config['model']['lora_dropout'],
        load_in_4bit=config['model']['load_in_4bit'],
        load_in_8bit=config['model']['load_in_8bit'],
        device_map="auto",
        torch_dtype=getattr(torch, config['model']['torch_dtype']),
        cpu_offload=config['model'].get('cpu_offload', False)
    )
    
    # Create optimizer
    optimizer = create_optimizer(
        model,
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Create scheduler
    num_training_steps = len(train_loader) * config['training']['num_epochs']
    num_warmup_steps = int(num_training_steps * config['training']['warmup_ratio'])
    
    scheduler = create_scheduler(
        optimizer,
        num_training_steps=num_training_steps,
        num_warmup_steps=num_warmup_steps,
        scheduler_type=config['training']['scheduler_type']
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device="cuda" if torch.cuda.is_available() else "cpu",
        output_dir=config['training']['output_dir'],
        max_grad_norm=config['training']['max_grad_norm'],
        logging_steps=config['training']['logging_steps'],
        eval_steps=config['training']['eval_steps'],
        save_steps=config['training']['save_steps'],
        save_total_limit=config['training']['save_total_limit'],
        use_amp=config['training']['use_amp'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps']
    )
    
    # Start training
    logger.info("="*60)
    logger.info("Starting Training")
    logger.info("="*60)
    logger.info(f"Mode: {config['model'].get('mode', 'full').upper()}")
    logger.info(f"Model: {config['model']['model_name']}")
    logger.info(f"Train samples: {len(train_loader.dataset)}")
    logger.info(f"Val samples: {len(val_loader.dataset)}")
    logger.info(f"Batch size: {config['training']['batch_size']}")
    logger.info(f"Gradient accumulation: {config['training']['gradient_accumulation_steps']}")
    logger.info(f"Effective batch size: {config['training']['batch_size'] * config['training']['gradient_accumulation_steps']}")
    logger.info(f"Learning rate: {config['training']['learning_rate']}")
    logger.info(f"Num epochs: {config['training']['num_epochs']}")
    logger.info(f"Total steps: {num_training_steps}")
    logger.info(f"Warmup steps: {num_warmup_steps}")
    logger.info(f"Output dir: {config['training']['output_dir']}")
    logger.info("="*60)
    
    trainer.train(
        num_epochs=config['training']['num_epochs'],
        resume_from_checkpoint=args.resume_from_checkpoint
    )

if __name__ == '__main__':
    main()