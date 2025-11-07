"""
Data augmentation for video frames
"""

import numpy as np
import cv2
from typing import List, Tuple
import random

class VideoAugmentation:
    """
    Augmentation pipeline for video frames
    Suitable for dashcam traffic videos
    """

    def __init__(
        self,
        brightness_range: Tuple[float, float] = (0.8, 1.2),
        contrast_range: Tuple[float, float] = (0.8, 1.2),
        saturation_range: Tuple[float, float] = (0.8, 1.2),
        horizontal_flip_prob: float = 0.0,  # Don't flip for traffic (changes meaning)
        gaussian_noise_prob: float = 0.3,
        gaussian_noise_std: float = 5.0,
        motion_blur_prob: float = 0.2,
        motion_blur_size: int = 5,
        random_crop_prob: float = 0.3,
        crop_scale_range: Tuple[float, float] = (0.9, 1.0),
        train: bool = True
    ):
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range
        self.horizontal_flip_prob = horizontal_flip_prob
        self.gaussian_noise_prob = gaussian_noise_prob
        self.gaussian_noise_std = gaussian_noise_std
        self.motion_blur_prob = motion_blur_prob
        self.motion_blur_size = motion_blur_size
        self.random_crop_prob = random_crop_prob
        self.crop_scale_range = crop_scale_range
        self.train = train
    
    def __call__(
        self,
        frames: np.ndarray
    ) -> np.ndarray:
        """
        Apply augmentation to video frames
        
        Args:
            frames (np.ndarray): Video frames of shape (T, H, W, C)
        
        Returns:
            np.ndarray: Augmented video frames of shape (T, H, W, C)
        """
        if not self.train:
            return frames
        
        # Apply same augmentation to all frames for consistency
        augmented_frames = []

        # Sample augmentation parameters onec for all frames
        brightness_factor = random.uniform(*self.brightness_range)
        contrast_factor = random.uniform(*self.contrast_range)
        saturation_factor = random.uniform(*self.saturation_range)
        do_flip = random.random() < self.horizontal_flip_prob
        do_noise = random.random() < self.gaussian_noise_prob
        do_blur = random.random() < self.motion_blur_prob
        do_crop = random.random() < self.random_crop_prob

        # Random crop parameters
        if do_crop:
            h, w = frames.shape[1:3]
            crop_scale = random.uniform(*self.crop_scale_range)
            new_h, new_w = int(h * crop_scale), int(w * crop_scale)
            top = random.randint(0, h - new_h)
            left = random.randint(0, w - new_w)
        
        for frame in frames:
            # Color augmentation
            frame = self._adjust_brightness(frame, brightness_factor)
            frame = self._adjust_contrast(frame, contrast_factor)
            frame = self._adjust_saturation(frame, saturation_factor)

            # Horizontal flip
            if do_flip:
                frame = cv2.flip(frame, 1)
            
            # Gaussian noise (simulate camera noise)
            if do_noise:
                frame = self._add_gaussian_noise(frame, self.gaussian_noise_std)
            
            # Motion blur (simulate camera shake)
            if do_blur:
                frame = self._add_motion_blur(frame, self.motion_blur_size)
            
            # Random crop
            if do_crop:
                frame = frame[top:top+new_h, left:left+new_w]
                frame = cv2.resize(frame, (w, h))
            
            augmented_frames.append(frame)
        
        return np.stack(augmented_frames)

    def _adjust_brightness(
        self,
        frame: np.ndarray,
        factor: float
    ) -> np.ndarray:
        """
        Adjust brightness
        """
        frame = frame.astype(np.float32)
        frame = np.clip(frame * factor, 0, 255)
        return frame.astype(np.uint8)

    def _adjust_contrast(
        self,
        frame: np.ndarray,
        factor: float
    ) -> np.ndarray:
        """
        Adjust contrast
        """
        frame = frame.astype(np.float32)
        mean = frame.mean()
        frame = np.clip((frame - mean) * factor + mean, 0, 255)
        return frame.astype(np.uint8)

    def _adjust_saturation(
        self,
        frame: np.ndarray,
        factor: float
    ) -> np.ndarray:
        """
        Adjust saturation
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
        frame = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        return frame

    def _add_gaussian_noise(
        self,
        frame: np.ndarray,
        std: float
    ) -> np.ndarray:
        """
        Add Gaussian noise
        """
        noise = np.random.normal(0, std, frame.shape)
        frame = frame.astype(np.float32) + noise
        frame = np.clip(frame, 0, 255)
        return frame.astype(np.uint8)

    def _add_motion_blur(
        self,
        frame: np.ndarray,
        size: int
    ) -> np.ndarray:
        """
        Add motion blur
        """
        kernel = np.zeros((size, size))
        kernel[int((size - 1)/2), :] = np.ones(size)
        kernel = kernel / size

        blurred = cv2.filter2D(frame, -1, kernel)
        return blurred

class TemporalAugmentation:
    """
    Temporal augmentation for video frames
    """

    def __init__(
        self,
        temporal_crop_prob: float = 0.3,
        temporal_reverse_prob: float = 0.0,
        frame_dropout_prob: float = 0.1,
        train: bool = True
    ):
        self.temporal_crop_prob = temporal_crop_prob
        self.temporal_reverse_prob = temporal_reverse_prob
        self.frame_dropout_prob = frame_dropout_prob
        self.train = train
    
    def __call__(
        self,
        frames: np.ndarray
    ) -> np.ndarray:
        """
        Apply temporal augmentation to video frames
        """
        if not self.train:
            return frames
        
        num_frames = frames.shape[0]

        # Temporal crop (sample subset of frames)
        if random.random() < self.temporal_crop_prob and num_frames > 4:
            crop_length = random.randint(num_frames // 2, num_frames)
            start_idx = random.randint(0, num_frames - crop_length)
            frames = frames[start_idx:start_idx + crop_length]

            # Resize back to original length by sampling
            indices = np.linspace(0, len(frames) - 1, num_frames, dtype=int)
            frames = frames[indices]
        
        # Temporal reverse
        if random.random() < self.temporal_reverse_prob:
            frames = frames[::-1]

        # Frame dropout (simulate missing frames)
        if random.random() < self.frame_dropout_prob:
            dropout_indices = random.sample(
                range(1, num_frames - 1), 
                k=min(2, num_frames // 4)
            )

            for idx in dropout_indices:
                # Replace with interpolated of neighboring frames
                frames[idx] = (frames[idx - 1].astype(np.float32) + 
                              frames[idx + 1].astype(np.float32)) / 2
                frames[idx] = frames[idx].astype(np.uint8)
        
        return frames

def get_train_augmentation():
    """
    Get training augmentation pipeline
    """
    spatial_aug = VideoAugmentation(
        brightness_range=(0.7, 1.3),
        contrast_range=(0.7, 1.3),
        saturation_range=(0.7, 1.3),
        horizontal_flip_prob=0.0,
        gaussian_noise_prob=0.3,
        gaussian_noise_std=5.0,
        motion_blur_prob=0.2,
        motion_blur_size=5,
        random_crop_prob=0.2,
        crop_scale_range=(0.8, 1.0),
        train=True
    )

    temporal_aug = TemporalAugmentation(
        temporal_crop_prob=0.2,
        temporal_reverse_prob=0.0,
        frame_dropout_prob=0.1,
        train=True
    )

    def augment(frames):
        frames = spatial_aug(frames)
        frames = temporal_aug(frames)
        return frames

    return augment

def get_val_augmentation():
    """Get validation augmentation (no augmentation)"""
    return lambda frames: frames