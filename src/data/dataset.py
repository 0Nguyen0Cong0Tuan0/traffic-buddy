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
        image_size: Tuple[int, int] = (224, 224),
        use_support_frames: bool = True,
        transform=None,
        max_samples: Optional[int] = None
    ):
        """
        Args:
            json_path: Path to train.json or test.json
            video_root: Root directory containing videos
            num_frames: Number of frames to sample from video
            image_size: Target image size (height, width)
            use_support_frames: Whether to prioritize support_frames
            transform: Optional transform to apply to frames
            max_samples: Maximum number of samples (for debugging)
        """
        self.json_path = json_path
        self.video_root = Path(video_root)
        self.num_frames = num_frames
        self.image_size = image_size
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
            'frames': frames, # Shape: (num_frames, H, w, C)
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
        support_frames: List[float]
    ) -> np.ndarray:
        """
        Extract frames from video
        Args:
            video_path: Path to video file
            support_frames: List of timestamps (in seconds) of important frames
        Returns:
            frames: numpy array of shape (num_frames, H, W, C)
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            # Return black frames if video cannot be opened
            return np.zeros((self.num_frames, *self.image_size, 3), dtype=np.uint8)
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0

        # Determine which frames to extract
        if self.use_support_frames and support_frames:
            frame_indices = self._get_support_frame_indices(support_frames, fps, total_frames)
        else:
            # Uniform sampling
            frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        
        # Extract frames
        frames = []
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                # Resize frame
                frame = cv2.resize(frame, self.image_size)
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                # Use last valid frame or black frame
                if frames:
                    frames.append(frames[-1])
                else:
                    frames.append(np.zeros((*self.image_size, 3), dtype=np.uint8))

        cap.release()

        # Ensure we have exactly num_frames
        while len(frames) < self.num_frames:
            frames.append(frames[-1] if frames else np.zeros((*self.image_size, 3), dtype=np.uint8))

        frames = np.stack(frames[:self.num_frames])

        return frames

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
            prompt += "Đáp án:"
        
        return prompt.strip()

    def _extract_answer_letter(
        self,
        answer: str
    ) -> str:
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
    """
    # Stack frames
    frames = torch.from_numpy(np.stack([item['frames'] for item in batch]))
    frames = frames.permute(0, 1, 4, 2, 3)  # (B, T, C, H, W)
    frames = frames.float() / 255.0  # Normalize to [0, 1]

    return {
        'ids': [item['id'] for item in batch],
        'frames': frames,
        'questions': [item['question'] for item in batch],
        'choices': [item['choices'] for item in batch],
        'prompts': [item['prompt'] for item in batch],
        'answers': [item['answer'] for item in batch],
        'answer_letters': [item['answer_letter'] for item in batch],
        'video_paths': [item['video_path'] for item in batch]
    }