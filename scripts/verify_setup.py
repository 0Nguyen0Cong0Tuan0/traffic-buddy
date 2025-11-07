"""
Verify training setup and hardware requirements
"""
import sys
import torch
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_gpu():
    """Check GPU availability and specs"""
    logger.info("="*60)
    logger.info("Checking GPU...")
    logger.info("="*60)
    
    if not torch.cuda.is_available():
        logger.error(" CUDA not available!")
        logger.warning(" Training will be VERY slow on CPU")
        logger.info(" Recommendation: Use LITE mode on CPU (not recommended)")
        return False
    
    logger.info(f" CUDA available: {torch.version.cuda}")
    logger.info(f" PyTorch version: {torch.__version__}")
    
    num_gpus = torch.cuda.device_count()
    logger.info(f" Number of GPUs: {num_gpus}")
    
    for i in range(num_gpus):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
        logger.info(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        
        # Recommend mode based on memory
        if gpu_memory >= 16:
            logger.info(f"   Sufficient for FULL mode (Qwen2-VL-7B)")
        elif gpu_memory >= 8:
            logger.info(f"   Recommended: LITE mode (Qwen2-VL-2B)")
        else:
            logger.warning(f"   Limited memory. Use LITE mode with caution")
    
    return True


def check_dependencies():
    """Check required dependencies"""
    logger.info("\n" + "="*60)
    logger.info("Checking dependencies...")
    logger.info("="*60)
    
    required_packages = {
        'torch': 'PyTorch',
        'transformers': 'Transformers',
        'accelerate': 'Accelerate',
        'peft': 'PEFT (LoRA)',
        'bitsandbytes': 'BitsAndBytes',
        'cv2': 'OpenCV',
        'pandas': 'Pandas',
        'tqdm': 'tqdm',
        'yaml': 'PyYAML'
    }
    
    missing = []
    
    for package, name in required_packages.items():
        try:
            if package == 'cv2':
                import cv2
            else:
                __import__(package)
            logger.info(f" {name}")
        except ImportError:
            logger.error(f" {name} not installed")
            missing.append(package)
    
    if missing:
        logger.error(f"\nMissing packages: {', '.join(missing)}")
        logger.error("Run: pip install -r requirements.txt")
        return False
    
    return True


def check_data():
    """Check data availability"""
    logger.info("\n" + "="*60)
    logger.info("Checking data...")
    logger.info("="*60)
    
    data_paths = {
        'Train split': 'data/processed/train_split.json',
        'Val split': 'data/processed/val_split.json',
        'Video directory': 'data/train/videos',
    }
    
    all_exist = True
    
    for name, path in data_paths.items():
        path = Path(path)
        if path.exists():
            if path.is_file():
                size = path.stat().st_size / 1024
                logger.info(f" {name}: {path} ({size:.1f} KB)")
            else:
                num_files = len(list(path.glob('*')))
                logger.info(f" {name}: {path} ({num_files} files)")
        else:
            logger.error(f" {name}: {path} not found")
            all_exist = False
    
    if not all_exist:
        logger.error("\nRun data preparation first:")
        logger.error("python scripts/prepare_data.py --train_json data/train/train.json --video_root data/ --output_dir data/processed")
        return False
    
    return True


def check_model_access():
    """Check model access"""
    logger.info("\n" + "="*60)
    logger.info("Checking model access...")
    logger.info("="*60)
    
    try:
        from transformers import AutoTokenizer
        
        model_name = "Qwen/Qwen2-VL-7B-Instruct"
        logger.info(f"Testing access to {model_name}...")
        
        # Try to load tokenizer (lightweight test)
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        logger.info(f" Model access verified")
        logger.info(f" Model will be downloaded on first training run (~14GB)")
        
        return True
        
    except Exception as e:
        logger.error(f" Cannot access model: {e}")
        logger.error("\nMake sure you:")
        logger.error("1. Have internet connection")
        logger.error("2. Have Hugging Face account")
        logger.error("3. Accepted model license (if required)")
        return False


def estimate_training_time():
    """Estimate training time"""
    logger.info("\n" + "="*60)
    logger.info("Training time estimation...")
    logger.info("="*60)
    
    # Load config
    import json
    try:
        with open('data/processed/train_split.json', 'r') as f:
            data = json.load(f)
            num_train = data['__count__']
        
        with open('data/processed/val_split.json', 'r') as f:
            data = json.load(f)
            num_val = data['__count__']
    except:
        logger.warning("Cannot load data splits")
        num_train, num_val = 1190, 300
    
    # Assumptions
    batch_size = 2
    grad_accum = 8
    num_epochs = 5
    
    effective_batch = batch_size * grad_accum
    steps_per_epoch = num_train // effective_batch
    total_steps = steps_per_epoch * num_epochs
    
    # Timing estimates (seconds)
    time_per_step_train = 4.0  # Conservative estimate
    time_per_step_eval = 2.0
    
    train_time_per_epoch = steps_per_epoch * time_per_step_train / 60
    eval_time_per_epoch = (num_val // batch_size) * time_per_step_eval / 60
    
    total_time = (train_time_per_epoch + eval_time_per_epoch) * num_epochs / 60
    
    logger.info(f"Dataset: {num_train} train, {num_val} val samples")
    logger.info(f"Config: batch_size={batch_size}, grad_accum={grad_accum}")
    logger.info(f"Effective batch size: {effective_batch}")
    logger.info(f"Steps per epoch: {steps_per_epoch}")
    logger.info(f"Total steps: {total_steps}")
    logger.info(f"\nEstimated time:")
    logger.info(f"  Per epoch: {train_time_per_epoch + eval_time_per_epoch:.1f} minutes")
    logger.info(f"  Total (5 epochs): {total_time:.1f} hours")
    logger.info(f"\n Actual time may vary based on GPU and data loading speed")


def main():
    logger.info("\n" + "="*80)
    logger.info(" "*20 + "TRAINING SETUP VERIFICATION")
    logger.info("="*80)
    
    checks = [
        ("GPU", check_gpu),
        ("Dependencies", check_dependencies),
        ("Data", check_data),
        ("Model Access", check_model_access),
    ]
    
    results = []
    
    for name, check_func in checks:
        try:
            result = check_func()
            results.append(result)
        except Exception as e:
            logger.error(f"Error in {name} check: {e}")
            results.append(False)
    
    # Estimate training time (non-critical)
    try:
        estimate_training_time()
    except Exception as e:
        logger.warning(f"Could not estimate training time: {e}")
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("SUMMARY")
    logger.info("="*80)
    
    if all(results):
        logger.info(" All checks passed! Ready to train.")
        
        # Recommend mode based on GPU
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info("\n" + "="*80)
            logger.info("RECOMMENDED MODE")
            logger.info("="*80)
            
            if gpu_memory >= 16:
                logger.info(" FULL MODE (Best Accuracy)")
                logger.info("  python scripts/train.py --mode full")
                logger.info("  # or")
                logger.info("  python scripts/train.py --config configs/training_config.yaml")
            elif gpu_memory >= 8:
                logger.info(" LITE MODE (Good Accuracy, Fits Your GPU)")
                logger.info("  python scripts/train.py --mode lite")
                logger.info("  # or")
                logger.info("  python scripts/train.py --config configs/training_config_lite.yaml")
            else:
                logger.warning(" LITE MODE (Minimal Setup)")
                logger.info("  python scripts/train.py --mode lite")
            
            logger.info("\n Or let the system auto-detect:")
            logger.info("  python scripts/train.py --mode auto")
        else:
            logger.warning("\nNo GPU detected - training will be very slow")
            logger.info("  python scripts/train.py --mode lite")
    else:
        logger.error(" Some checks failed. Fix the issues above before training.")
        sys.exit(1)


if __name__ == '__main__':
    main()