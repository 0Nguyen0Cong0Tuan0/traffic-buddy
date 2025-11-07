"""
Comprehensive test of the entire training pipeline
This tests everything except actual model training
"""

import sys
from pathlib import Path
import torch
import numpy as np
import logging

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_1_data_loading():
    """Test 1: Data loading"""
    logger.info("\n" + "="*60)
    logger.info("TEST 1: Data Loading")
    logger.info("="*60)
    
    try:
        from data.dataset import VietnameseTrafficDataset, collate_fn
        from torch.utils.data import DataLoader
        
        # Create dataset
        dataset = VietnameseTrafficDataset(
            json_path='data/processed/train_split.json',
            video_root='data/',
            num_frames=4,
            image_size=(112, 112),
            use_support_frames=True,
            max_samples=2
        )
        
        logger.info(f" Dataset created: {len(dataset)} samples")
        
        # Load one sample
        sample = dataset[0]
        logger.info(f" Sample loaded")
        logger.info(f"  Frames shape: {sample['frames'].shape}")
        logger.info(f"  Question: {sample['question'][:50]}...")
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn
        )
        
        batch = next(iter(dataloader))
        logger.info(f" Batch created")
        logger.info(f"  Batch frames shape: {batch['frames'].shape}")
        logger.info(f"  Expected: (2, 4, 3, 112, 112)")
        
        assert batch['frames'].shape == (2, 4, 3, 112, 112), "Wrong batch shape!"
        
        logger.info(" TEST 1 PASSED: Data loading works!")
        return True
        
    except Exception as e:
        logger.error(f" TEST 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_2_video_format_conversion():
    """Test 2: Video format conversion"""
    logger.info("\n" + "="*60)
    logger.info("TEST 2: Video Format Conversion")
    logger.info("="*60)
    
    try:
        # Create dummy batch
        frames = torch.randn(2, 4, 3, 112, 112)  # (B, T, C, H, W)
        logger.info(f"Input frames shape: {frames.shape}")
        
        # Convert to list of numpy arrays
        videos_list = []
        for i in range(frames.shape[0]):
            video = frames[i]  # (T, C, H, W)
            video = video.permute(0, 2, 3, 1)  # → (T, H, W, C)
            video = video.numpy()
            if video.max() <= 1.0:
                video = (video * 255).astype('uint8')
            videos_list.append(video)
        
        logger.info(f" Converted to list of {len(videos_list)} videos")
        logger.info(f"  Each video shape: {videos_list[0].shape}")
        logger.info(f"  Expected: (4, 112, 112, 3)")
        
        assert videos_list[0].shape == (4, 112, 112, 3), "Wrong video shape!"
        assert len(videos_list) == 2, "Wrong number of videos!"
        
        logger.info(" TEST 2 PASSED: Video format conversion works!")
        return True
        
    except Exception as e:
        logger.error(f" TEST 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_3_model_initialization():
    """Test 3: Model initialization"""
    logger.info("\n" + "="*60)
    logger.info("TEST 3: Model Initialization")
    logger.info("="*60)
    
    try:
        from models.video_llava import VietnameseTrafficVQAModel
        
        logger.info("Initializing LITE mode model...")
        logger.info("(This will download ~6GB if not cached)")
        
        model = VietnameseTrafficVQAModel(
            model_name="Qwen/Qwen2-VL-2B-Instruct",
            mode="lite",
            use_lora=True,
            lora_r=8,
            lora_alpha=16,
            load_in_8bit=True,
            cpu_offload=True,
            device_map="auto"
        )
        
        logger.info(" Model initialized")
        logger.info(" TEST 3 PASSED: Model initialization works!")
        return True, model
        
    except Exception as e:
        logger.error(f" TEST 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_4_prepare_inputs(model):
    """Test 4: Prepare inputs"""
    logger.info("\n" + "="*60)
    logger.info("TEST 4: Prepare Inputs")
    logger.info("="*60)
    
    if model is None:
        logger.warning(" Skipping: Model not available")
        return False
    
    try:
        # Create dummy data
        frames = torch.randn(2, 4, 3, 112, 112) * 0.5 + 0.5  # [0, 1]
        prompts = [
            "Câu hỏi: Test 1?\nLựa chọn:\nA. Yes\nB. No\nTrả lời:",
            "Câu hỏi: Test 2?\nLựa chọn:\nA. Yes\nB. No\nTrả lời:"
        ]
        answers = ["A. Yes", "B. No"]
        
        logger.info("Preparing inputs...")
        inputs = model.prepare_inputs(frames, prompts, answers)
        
        logger.info(" Inputs prepared")
        logger.info(f"  Keys: {list(inputs.keys())}")
        
        # Check required keys
        required_keys = ['input_ids', 'attention_mask']
        for key in required_keys:
            assert key in inputs, f"Missing key: {key}"
        
        # Check labels were created
        assert 'labels' in inputs, "Labels not created!"
        logger.info(f"   Labels created: {inputs['labels'].shape}")
        
        logger.info(" TEST 4 PASSED: Prepare inputs works!")
        return True
        
    except Exception as e:
        logger.error(f" TEST 4 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_5_forward_pass(model):
    """Test 5: Forward pass"""
    logger.info("\n" + "="*60)
    logger.info("TEST 5: Forward Pass")
    logger.info("="*60)
    
    if model is None:
        logger.warning(" Skipping: Model not available")
        return False
    
    try:
        # Create dummy data
        frames = torch.randn(1, 4, 3, 112, 112) * 0.5 + 0.5
        prompts = ["Câu hỏi: Test?\nLựa chọn:\nA. Yes\nB. No\nTrả lời:"]
        answers = ["A. Yes"]
        
        logger.info("Preparing inputs...")
        inputs = model.prepare_inputs(frames, prompts, answers)
        
        logger.info("Running forward pass...")
        with torch.no_grad():
            outputs = model(**inputs)
        
        logger.info(" Forward pass completed")
        logger.info(f"  Output type: {type(outputs)}")
        
        # Check loss exists
        assert hasattr(outputs, 'loss'), "No loss in outputs!"
        logger.info(f"   Loss: {outputs.loss.item():.4f}")
        
        logger.info(" TEST 5 PASSED: Forward pass works!")
        return True
        
    except Exception as e:
        logger.error(f" TEST 5 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    logger.info("="*60)
    logger.info("COMPREHENSIVE PIPELINE TEST")
    logger.info("="*60)
    
    results = {}
    
    # Test 1: Data loading (always runs)
    results['data_loading'] = test_1_data_loading()
    
    # Test 2: Video format (always runs)
    results['video_format'] = test_2_video_format_conversion()
    
    # Test 3-5: Model tests (require download)
    logger.info("\n" + "="*60)
    logger.info("Model tests require downloading ~6GB")
    logger.info("Skip if you don't have space/bandwidth")
    logger.info("="*60)
    
    import sys
    response = input("Run model tests? (y/n): ").lower().strip()
    
    if response == 'y':
        success, model = test_3_model_initialization()
        results['model_init'] = success
        
        if success:
            results['prepare_inputs'] = test_4_prepare_inputs(model)
            results['forward_pass'] = test_5_forward_pass(model)
        else:
            results['prepare_inputs'] = False
            results['forward_pass'] = False
    else:
        logger.info(" Skipping model tests")
        results['model_init'] = None
        results['prepare_inputs'] = None
        results['forward_pass'] = None
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("TEST SUMMARY")
    logger.info("="*60)
    
    for test_name, result in results.items():
        if result is None:
            status = " SKIPPED"
        elif result:
            status = " PASSED"
        else:
            status = " FAILED"
        logger.info(f"{test_name:20s}: {status}")
    
    # Final verdict
    logger.info("\n" + "="*60)
    
    essential_tests = ['data_loading', 'video_format']
    essential_passed = all(results[t] for t in essential_tests)
    
    if essential_passed:
        logger.info(" ESSENTIAL TESTS PASSED!")
        logger.info("\nYour code is ready for submission.")
        logger.info("Model tests are optional but recommended.")
    else:
        logger.error(" SOME ESSENTIAL TESTS FAILED!")
        logger.error("\nPlease fix the issues before proceeding.")
        sys.exit(1)


if __name__ == '__main__':
    main()