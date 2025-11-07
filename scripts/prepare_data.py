
"""
Main script for data preparation phase
Usage:
    python scripts/prepare_data.py --train_json data/train/train.json \
                                    --video_root data/ \
                                    --output_dir data/processed \
                                    --val_ratio 0.2
"""
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data.preprocessing import prepare_data_for_training
from data.dataset import VietnameseTrafficDataset, collate_fn
from torch.utils.data import DataLoader
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_dataloader(
    split_json: str,
    video_root: str,
    num_samples: int = 5,
):
    """
    Test the DataLoader with a few samples
    """
    logger.info(f"\n{'='*60}")
    logger.info("Testing DataLoader")
    logger.info(f"{'='*60}")

    # Create dataset
    dataset = VietnameseTrafficDataset(
        json_path=split_json,
        video_root=video_root,
        num_frames=8,
        image_size=(224, 224),
        use_support_frames=True,
        max_samples=num_samples
    )

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    # Test one batch
    logger.info(f"\nLoading first batch...")
    batch = next(iter(dataloader))
    
    logger.info(f"\nBatch contents:")
    logger.info(f"  IDs: {batch['ids']}")
    logger.info(f"  Frames shape: {batch['frames'].shape}")
    logger.info(f"  Frames dtype: {batch['frames'].dtype}")
    logger.info(f"\nSample 1:")
    logger.info(f"  Question: {batch['questions'][0]}")
    logger.info(f"  Choices: {batch['choices'][0]}")
    logger.info(f"  Answer: {batch['answers'][0]}")
    logger.info(f"  Answer letter: {batch['answer_letters'][0]}")
    logger.info(f"\nPrompt:\n{batch['prompts'][0]}")

    logger.info(f"\n{'='*60}")
    logger.info("DataLoader test completed successfully!")
    logger.info(f"{'='*60}\n")

def main():
    parser = argparse.ArgumentParser(description="Prepare data for Vietnamese Traffic Video QA")
    parser.add_argument(
        '--train_json',
        type=str,
        default='data/train/train.json',
        help='Path to training JSON file (train.json)'
    )

    parser.add_argument(
        '--video_root',
        type=str,
        default='data/',
        help='Root directory containing training videos'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/processed',
        help='Directory to save processed data'
    )

    parser.add_argument(
        '--val_ratio',
        type=float,
        default=0.2,
        help='Ratio of validation set from training data'
    )

    parser.add_argument(
        '--random_seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )

    parser.add_argument(
        '--test_dataloader',
        action='store_true',
        help='Test dataloader after preparation'
    )

    args = parser.parse_args()

    # Run data preparation
    result = prepare_data_for_training(
        train_json=args.train_json,
        video_root=args.video_root,
        output_dir=args.output_dir,
        val_ratio=args.val_ratio,
        random_seed=args.random_seed
    )

    # Test dataloader if requested
    if args.test_dataloader:
        train_split_json = Path(args.output_dir) / 'train_split.json'
        test_dataloader(
            split_json=str(train_split_json),
            video_root=args.video_root,
            num_samples=5
        )
    
    return result

if __name__ == '__main__':
    main()