"""
Test data loading and video processing
"""

import sys
from pathlib import Path
import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data.dataset import VietnameseTrafficDataset, collate_fn
from torch.utils.data import DataLoader
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_dataset():
    """Test dataset loading"""
    logger.info("="*60)
    logger.info("Testing Dataset")
    logger.info("="*60)
    
    dataset = VietnameseTrafficDataset(
        json_path='data/processed/train_split.json',
        video_root='data/',
        num_frames=4,
        image_size=(112, 112),
        use_support_frames=True,
        max_samples=3  # Test with just 3 samples
    )
    
    logger.info(f"Dataset size: {len(dataset)}")
    
    # Test loading one sample
    logger.info("\nLoading sample 0...")
    sample = dataset[0]
    
    logger.info(f"Keys: {sample.keys()}")
    logger.info(f"ID: {sample['id']}")
    logger.info(f"Frames shape: {sample['frames'].shape}")
    logger.info(f"Frames dtype: {sample['frames'].dtype}")
    logger.info(f"Question: {sample['question'][:100]}...")
    logger.info(f"Answer: {sample['answer']}")
    
    return dataset


def test_dataloader(dataset):
    """Test dataloader"""
    logger.info("\n" + "="*60)
    logger.info("Testing DataLoader")
    logger.info("="*60)
    
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,  # Use 0 for testing
        collate_fn=collate_fn
    )
    
    logger.info("Loading first batch...")
    batch = next(iter(dataloader))
    
    logger.info(f"\nBatch keys: {batch.keys()}")
    logger.info(f"Batch size: {len(batch['ids'])}")
    logger.info(f"Frames shape: {batch['frames'].shape}")
    logger.info(f"Frames dtype: {batch['frames'].dtype}")
    
    logger.info(f"\nSample 1:")
    logger.info(f"  ID: {batch['ids'][0]}")
    logger.info(f"  Question: {batch['questions'][0][:80]}...")
    logger.info(f"  Answer: {batch['answers'][0]}")
    logger.info(f"  Answer letter: {batch['answer_letters'][0]}")
    
    return batch


def test_video_format_conversion(batch):
    """Test converting batch format for model"""
    logger.info("\n" + "="*60)
    logger.info("Testing Video Format Conversion")
    logger.info("="*60)
    
    frames = batch['frames']  # (B, T, C, H, W)
    logger.info(f"Original frames shape: {frames.shape}")
    
    batch_size = frames.shape[0]
    videos_list = []
    
    for i in range(batch_size):
        video = frames[i]  # (T, C, H, W)
        logger.info(f"\nVideo {i} before conversion:")
        logger.info(f"  Shape: {video.shape}")
        
        # Convert to (T, H, W, C)
        video = video.permute(0, 2, 3, 1)
        logger.info(f"  After permute: {video.shape}")
        
        # Convert to numpy
        video_np = video.cpu().numpy()
        logger.info(f"  Numpy shape: {video_np.shape}")
        
        # Scale to [0, 255] if needed
        if video_np.max() <= 1.0:
            video_np = (video_np * 255).astype('uint8')
        
        logger.info(f"  Final dtype: {video_np.dtype}")
        
        videos_list.append(video_np)
    
    logger.info(f"\nConverted {len(videos_list)} videos")
    logger.info(f"Each video shape: {videos_list[0].shape}")
    
    return videos_list


def test_model_input_preparation():
    """Test preparing inputs for model"""
    logger.info("\n" + "="*60)
    logger.info("Testing Model Input Preparation")
    logger.info("="*60)
    
    try:
        from transformers import AutoProcessor
        
        logger.info("Loading processor...")
        processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct",
            trust_remote_code=True
        )
        logger.info(" Processor loaded")
        
        # Create dummy video
        dummy_video = np.random.randint(0, 255, (4, 112, 112, 3), dtype=np.uint8)
        dummy_text = ["<|im_start|>user\n<video>\nTest question?<|im_end|>\n<|im_start|>assistant\n"]
        
        logger.info(f"\nDummy video shape: {dummy_video.shape}")
        logger.info(f"Dummy video dtype: {dummy_video.dtype}")
        logger.info(f"Dummy text: {dummy_text[0][:50]}...")
        
        logger.info("\nProcessing inputs...")
        inputs = processor(
            text=dummy_text,
            videos=[dummy_video],
            padding=True,
            return_tensors="pt"
        )
        
        logger.info(" Inputs processed successfully!")
        logger.info(f"Input keys: {inputs.keys()}")
        for key, value in inputs.items():
            if hasattr(value, 'shape'):
                logger.info(f"  {key}: {value.shape}")
        
    except Exception as e:
        logger.error(f" Error: {e}")
        import traceback
        traceback.print_exc()


def main():
    logger.info("="*60)
    logger.info("Data Pipeline Test")
    logger.info("="*60)
    
    try:
        # Test 1: Dataset
        dataset = test_dataset()
        
        # Test 2: DataLoader
        batch = test_dataloader(dataset)
        
        # Test 3: Video format conversion
        videos_list = test_video_format_conversion(batch)
        
        logger.info("\n" + "="*60)
        logger.info(" All tests passed!")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"\n Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()