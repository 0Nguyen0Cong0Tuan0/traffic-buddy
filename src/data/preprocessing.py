import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import logging
from collections import Counter
import pandas as pd
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    Preprocessing utilities for Vietnamese Traffic Dataset
    """

    def __init__(
        self,
        train_json_path: str,
        video_root: str,    
    ):
        self.train_json_path = train_json_path
        self.video_root = Path(video_root)

        with open(self.train_json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.samples = self.data['data']
        logger.info(f"Loaded {len(self.samples)} samples from {train_json_path}")
    
    def analyze_dataset(self) -> Dict:
        """
        Analyze dataset statistics
        """
        logger.info("Analyzing dataset statistics...")

        stats = {
            'total_samples': len(self.samples),
            'unique_videos': len(set([s['video_path'] for s in self.samples])),
            'num_choices': Counter(),
            'answer_distribution': Counter(),
            'question_types': Counter(),
            'support_frame_stats': {
                'min': float('inf'),
                'max': 0,
                'avg': 0,
                'total': 0,
            },
        }

        support_frames_counts = []

        for sample in self.samples:
            # Count number of choices
            num_choices = len(sample.get('choices', []))
            stats['num_choices'][num_choices] += 1

            # Answer distribution
            answer = sample.get('answer', '')
            answer_letter = answer[0] if answer and answer[0] in ['A', 'B', 'C', 'D'] else 'Unknown'
            stats['answer_distribution'][answer_letter] += 1

            # Question type (rough categorization)
            question = sample['question'].lower()
            if 'biển' in question:
                stats['question_types']['traffic_sign'] += 1
            elif 'rẽ' in question:
                stats['question_types']['turn_direction'] += 1
            elif 'làn' in question:
                stats['question_types']['lane'] += 1
            elif 'tốc độ' in question:
                stats['question_types']['speed'] += 1
            else:
                stats['question_types']['other'] += 1

            # Support frame statistics
            support_frames = sample.get('support_frames', [])
            if support_frames:
                support_frames_counts.append(len(support_frames))
        
        if support_frames_counts:
            stats['support_frame_stats']['min'] = min(support_frames_counts)
            stats['support_frame_stats']['max'] = max(support_frames_counts)
            stats['support_frame_stats']['avg'] = sum(support_frames_counts) / len(support_frames_counts)
            stats['support_frame_stats']['total'] = len([s for s in self.samples if s.get('support_frames')])

        # Print statistics
        logger.info(f"\n{'='*50}")
        logger.info(f"Dataset Statistics:")
        logger.info(f"{'='*50}")
        logger.info(f"Total samples: {stats['total_samples']}")
        logger.info(f"Unique videos: {stats['unique_videos']}")
        logger.info(f"\nNumber of choices distribution:")
        for num, count in sorted(stats['num_choices'].items()):
            logger.info(f"  {num} choices: {count} samples")
        logger.info(f"\nAnswer distribution:")
        for answer, count in sorted(stats['answer_distribution'].items()):
            logger.info(f"  {answer}: {count} samples ({count/stats['total_samples']*100:.1f}%)")
        logger.info(f"\nQuestion types:")
        for qtype, count in sorted(stats['question_types'].items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {qtype}: {count} samples ({count/stats['total_samples']*100:.1f}%)")
        logger.info(f"\nSupport frames statistics:")
        logger.info(f"  Samples with support frames: {stats['support_frame_stats']['total']}")
        if stats['support_frame_stats']['total'] > 0:
            logger.info(f"  Min frames per sample: {stats['support_frame_stats']['min']}")
            logger.info(f"  Max frames per sample: {stats['support_frame_stats']['max']}")
            logger.info(f"  Avg frames per sample: {stats['support_frame_stats']['avg']:.2f}")
        
        return stats

    def verify_video_files(self) -> Tuple[List[str], List[str]]:
        """
        Verify that all video files exist
        Returns:
            valid_samples: List of sample IDs with existing videos
            missing_videos: List of missing video paths
        """
        logger.info("Verifying video files...")

        valid_samples = []
        missing_videos = []

        for sample in self.samples:
            video_path = self.video_root / sample['video_path']
            if video_path.exists():
                valid_samples.append(sample['id'])
            else:
                missing_videos.append(sample['video_path'])
        
        logger.info(f"Valid samples: {len(valid_samples)}/{len(self.samples)}")
        if missing_videos:
            logger.warning(f"Missing videos: {len(missing_videos)}")
            for vid in missing_videos[:5]:
                logger.warning(f"  {vid}")
            if len(missing_videos) > 5:
                logger.warning(f"  ... and {len(missing_videos) - 5} more")
        
        return valid_samples, missing_videos
    
    def create_train_val_split(
        self,
        val_ratio: float = 0.2,
        random_seed: int = 42,
        stratify_by_answer: bool = True,
        output_dir: str = None
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Create train/validation split
        Args:
            val_ratio: Ratio of validation set
            random_seed: Random seed for reproducibility
            stratify_by_answer: Whether to stratify by answer distribution
            output_dir: Directory to save split JSON files
        Returns:
            train_samples, val_samples
        """
        random.seed(random_seed)

        if stratify_by_answer:
            # Group by answer
            answer_groups = {}

            for sample in self.samples:
                answer = sample.get('answer', '')
                answer_letter = answer[0] if answer and answer[0] in ['A', 'B', 'C', 'D'] else 'Unknown'
                if answer_letter not in answer_groups:
                    answer_groups[answer_letter] = []
                answer_groups[answer_letter].append(sample)
            
            train_samples = []
            val_samples = []

            for answer, samples in answer_groups.items():
                random.shuffle(samples)
                split_idx = int(len(samples) * (1 - val_ratio))
                train_samples.extend(samples[:split_idx])
                val_samples.extend(samples[split_idx:])
            
            random.shuffle(train_samples)
            random.shuffle(val_samples)
        else:
            # Simple random split
            samples_copy = self.samples.copy()
            random.shuffle(samples_copy)
            split_idx = int(len(samples_copy) * (1 - val_ratio))
            train_samples = samples_copy[:split_idx]
            val_samples = samples_copy[split_idx:]
        
        logger.info(f"Train samples: {len(train_samples)}")
        logger.info(f"Validation samples: {len(val_samples)}")

        # Save to files if output_dir is provided
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            train_data = {'__count__':len(train_samples), 'data': train_samples}
            val_data = {'__count__':len(val_samples), 'data': val_samples}

            with open(output_path / 'train_split.json', 'w', encoding='utf-8') as f:
                json.dump(train_data, f, ensure_ascii=False, indent=2)
            
            with open(output_path / 'val_split.json', 'w', encoding='utf-8') as f:
                json.dump(val_data, f, ensure_ascii=False, indent=2)

            logger.info(f"Save splits to {output_dir}")
        
        return train_samples, val_samples
    
    def export_to_csv(
        self,
        output_path: str
    ):
        """
        Export dataset to CSV file
        """
        logger.info(f"Exporting dataset to CSV at {output_path}...")

        rows = []
        for sample in self.samples:
            row = {
                'id': sample['id'],
                'question': sample['question'],
                'answer': sample.get('answer', ''),
                'video_path': sample['video_path'],
                'num_choices': len(sample.get('choices', [])),
                'num_support_frames': len(sample.get('support_frames', []))
            }

            # Add individual choices
            for i, choice in enumerate(sample.get('choices', [])):
                row[f'choice_{chr(65+i)}'] = choice
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        logger.info(f"Exported {len(rows)} samples to {output_path}")


def prepare_data_for_training(
    train_json: str,
    video_root: str,
    output_dir: str,
    val_ratio: float = 0.2,
    random_seed: int = 42
):
    """
    Main function to prepare data for training
    """
    logger.info('='*60)
    logger.info("Preparing data for training...")
    logger.info('='*60)

    # Initialize preprocessor
    preprocessor = DataPreprocessor(
        train_json_path=train_json,
        video_root=video_root
    )

    # Analyze dataset
    logger.info("\nAnalyzing dataset...")
    stats = preprocessor.analyze_dataset()
    
    # Verify video files
    logger.info("\nVerifying video files...")
    valid_samples, missing_videos = preprocessor.verify_video_files()
    
    # Create train/val split
    logger.info("\nCreating train/val split...")
    train_samples, val_samples = preprocessor.create_train_val_split(
        val_ratio=val_ratio,
        random_seed=random_seed,
        stratify_by_answer=True,
        output_dir=output_dir
    )

    # Export to CSV for analysis
    logger.info("\nExporting to CSV...")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    preprocessor.export_to_csv(str(output_path / 'dataset_analysis.csv'))
    
    # Save statistics
    with open(output_path / 'dataset_stats.json', 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2, default=str)
    
    logger.info("\n" + "="*60)
    logger.info("Data Preparation Complete!")
    logger.info("="*60)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Files created:")
    logger.info(f"  - train_split.json ({len(train_samples)} samples)")
    logger.info(f"  - val_split.json ({len(val_samples)} samples)")
    logger.info(f"  - dataset_analysis.csv")
    logger.info(f"  - dataset_stats.json")
    
    return {
        'train_samples': train_samples,
        'val_samples': val_samples,
        'stats': stats,
        'valid_samples': valid_samples,
        'missing_videos': missing_videos
    }