import os
import json
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VietnameseTrafficDataset(Dataset):
    """
    Dataset for Vietnamese Traffic Video Question Answering
    """
    def __init__(
        self,
        json_path: str,
        video_root: str,
        num_frames: int = 8,
        use_support_frames: bool = True,
        transform=None,
        max_samples: Optional[int] = None
    ):
        """
        Args:
            json_path: Path to train.json or test.json
            video_root: Root directory containing videos
            num_frames: Number of frames to sample from video
            use_support_frames: Whether to prioritize support_frames
            transform: Optional transform to apply to frames
            max_samples: Maximum number of samples (for debugging)
        """
        self.json_path = json_path
        self.video_root = Path(video_root)
        self.num_frames = num_frames
        self.use_support_frames = use_support_frames
        self.transform = transform
        
        # Load data
        with open(self.json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.samples = data['data']
        if max_samples:
            self.samples = self.samples[:max_samples]
        
        logger.info(f"Loaded {len(self.samples)} samples from {json_path}")

    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]

        # Get video path
        video_path = self.video_root / sample['video_path']

        # Extract frames
        frames = self._extract_frames(
            str(video_path),
            sample.get('support_frames', [] if self.use_support_frames else None)
        )

        # Apply transforms if provided
        if self.transform:
            frames = self.transform(frames)

        # Prepare question and choices
        question = sample['question']
        choices = sample['choices']

        # Format prompt
        prompt = self._format_prompt(question, choices)

        # Get answer if available (for training)
        answer = sample.get('answer', None)
        answer_letter = self._extract_answer_letter(answer) if answer else None

        return {
            'id': sample['id'],
            'frames': frames,  # Shape: (num_frames, H, W, C) in [0, 255] uint8
            'question': question,
            'choices': choices,
            'prompt': prompt,
            'answer': answer,
            'answer_letter': answer_letter,
            'video_path': str(video_path)
        }

    def _extract_frames(
        self,
        video_path: str,
        support_frames: Optional[List[float]]
    ) -> np.ndarray:
        """
        Extract frames from video at original resolution
        """
        # Debug: Check if video path exists
        logger.info(f"Attempting to open video: {video_path}")
        if not os.path.exists(video_path):
            logger.error(f"Video file does not exist: {video_path}")
            # Return black frames as fallback
            return np.zeros((self.num_frames, 224, 224, 3), dtype=np.uint8)
        
        # Debug: Check file permissions and size
        try:
            file_size = os.path.getsize(video_path)
            logger.info(f"Video file size: {file_size} bytes")
        except OSError as e:
            logger.error(f"Error accessing video file: {e}")
            return np.zeros((self.num_frames, 224, 224, 3), dtype=np.uint8)

        # Initialize video capture
        cap = cv2.VideoCapture(video_path)
        
        # Debug: Check if video capture opened successfully
        if not cap.isOpened():
            logger.error(f"Failed to open video with OpenCV: {video_path}")
            logger.info(f"OpenCV version: {cv2.__version__}")
            logger.info(f"Available backends: {cv2.videoio_registry.getBackends()}")
            return np.zeros((self.num_frames, 224, 224, 3), dtype=np.uint8)
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        # Get original video dimensions
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if total_frames == 0 or fps == 0:
            logger.error(f"Invalid video properties: frames={total_frames}, fps={fps}")
            cap.release()
            return np.zeros((self.num_frames, 224, 224, 3), dtype=np.uint8)

        duration = total_frames / fps if fps > 0 else 0
        logger.info(f"Video info - Total frames: {total_frames}, FPS: {fps}, Duration: {duration:.2f}s, "
                    f"Resolution: {frame_width}x{frame_height}")

        # Determine which frames to extract
        if self.use_support_frames and support_frames:
            frame_indices = self._get_support_frame_indices(support_frames, fps, total_frames)
        else:
            frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)

        # Debug: Log frame indices
        logger.info(f"Frame indices to extract: {frame_indices.tolist()}")

        # Extract frames
        frames = []
        for frame_idx in frame_indices:
            logger.debug(f"Attempting to read frame {frame_idx}")
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                logger.debug(f"Successfully read frame {frame_idx}")
                # Convert BGR to RGB, keep original size
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            else:
                logger.warning(f"Failed to read frame {frame_idx}")
                # Use last valid frame or black frame
                if frames:
                    logger.debug(f"Using last valid frame for index {frame_idx}")
                    frames.append(frames[-1].copy())
                else:
                    logger.debug(f"Using black frame for index {frame_idx}")
                    frames.append(np.zeros((frame_height, frame_width, 3), dtype=np.uint8))
        
        # Debug: Log number of successfully extracted frames
        logger.info(f"Extracted {len(frames)} frames")

        cap.release()

        # Ensure we have exactly num_frames
        while len(frames) < self.num_frames:
            logger.debug(f"Padding with last frame or black frame to reach {self.num_frames} frames")
            frames.append(frames[-1].copy() if frames else 
                         np.zeros((frame_height, frame_width, 3), dtype=np.uint8))

        frames = np.stack(frames[:self.num_frames])
        
        # Save frames to 'frame' folder
        self._save_frames(frames, video_path)
        
        # Debug: Check if frames are all zeros
        if np.all(frames == 0):
            logger.warning(f"All extracted frames are black for video: {video_path}")
        
        return frames

    def _save_frames(self, frames: np.ndarray, video_path: str) -> None:
        """
        Save frames to 'frame' folder with names derived from video_path and frame index
        """
        # Create 'frame' directory if it doesn't exist
        output_dir = "frame"
        os.makedirs(output_dir, exist_ok=True)
        
        # Get base name of video file (without extension)
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        # Save each frame
        for idx, frame in enumerate(frames):
            # Construct filename: {video_name}_{idx}.png
            filename = f"{video_name}_{idx}.png"
            output_path = os.path.join(output_dir, filename)
            
            # Convert RGB back to BGR for OpenCV saving
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, frame_bgr)
            logger.info(f"Saved frame to {output_path} with shape {frame.shape}")

    def _get_support_frame_indices(
        self,
        support_frames: List[float],
        fps: float,
        total_frames: int
    ) -> np.ndarray:
        """
        Get frame indices based on support_frames timestamps
        Also includes uniformly sampled frames for context
        """
        # Convert timestamps to frame indices
        support_indices = [int(timestamp * fps) for timestamp in support_frames]
        support_indices = [min(idx, total_frames - 1) for idx in support_indices]

        # Add uniformly sampled frames for context
        num_uniform = max(1, self.num_frames - len(support_indices))
        uniform_indices = np.linspace(0, total_frames - 1, num_uniform, dtype=int)

        # Combine and sort
        all_indices = sorted(set(list(support_indices) + list(uniform_indices)))

        # Sample to get exactly num_frames
        if len(all_indices) > self.num_frames:
            # Prioritize support frames
            indices = support_indices[:self.num_frames]
            if len(indices) < self.num_frames:
                remaining = self.num_frames - len(indices)
                uniform_subset = [idx for idx in uniform_indices if idx not in indices][:remaining]
                indices.extend(uniform_subset)
        else:
            indices = all_indices
            # Fill with more uniform samples if needed
            while len(indices) < self.num_frames:
                fill_idx = np.random.randint(0, total_frames)
                if fill_idx not in indices:
                    indices.append(fill_idx)
        
        return np.array(sorted(indices[:self.num_frames]))

    def _format_prompt(
        self,
        question: str,
        choices: List[str]
    ) -> str:
        """
        Format question and choices into a prompt
        """
        prompt = f"Câu hỏi: {question}\n"

        if choices:
            prompt += "Các lựa chọn:\n"
            for choice in choices:
                prompt += f"{choice}\n"
            prompt += "Hãy chọn đáp án đúng (A, B, C, hoặc D):"
        
        return prompt.strip()

    def _extract_answer_letter(
        self,
        answer: str
    ) -> Optional[str]:
        """Extract answer letter from answer string"""
        if not answer:
            return None

        # Answer format: "A. Some answer text" or just "A"
        answer = answer.strip()
        if answer and answer[0] in ['A', 'B', 'C', 'D']:
            return answer[0]

        return None
    
def collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate function for DataLoader
    Converts numpy frames to torch tensors in the correct format
    """
    # Stack frames and convert to torch tensor
    # Input: List of numpy arrays (num_frames, H, W, C) in [0, 255] uint8
    # Output: torch tensor (B, T, C, H, W) in [0, 1] float32
    frames_list = [item['frames'] for item in batch]
    
    # Convert to torch and normalize
    frames = torch.from_numpy(np.stack(frames_list))  # (B, T, H, W, C)
    frames = frames.permute(0, 1, 4, 2, 3)  # (B, T, C, H, W)
    frames = frames.float() / 255.0  # Normalize to [0, 1]

    return {
        'ids': [item['id'] for item in batch],
        'frames': frames,  # (B, T, C, H, W) in [0, 1]
        'questions': [item['question'] for item in batch],
        'choices': [item['choices'] for item in batch],
        'prompts': [item['prompt'] for item in batch],
        'answers': [item['answer'] for item in batch],
        'answer_letters': [item['answer_letter'] for item in batch],
        'video_paths': [item['video_path'] for item in batch]
    }