import cv2
import numpy as np
from typing import List
from PIL import Image
from ..config import settings

def sample_frames(video_path: str, n_frames: int = None) -> List[Image.Image]:
    """
    Sample n_frames uniformly from the video located at video_path.
    
    Args:
        video_path (str): Path to the video file.
        n_frames (int, optional): Number of frames to sample. Defaults to settings.NUM_FRAMES.
        
    Returns:
        List[Image.Image]: List of sampled frames as PIL Images.
    """
    n_frames = n_frames or settings.NUM_FRAMES
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total - 1, n_frames, dtype=int)
    frames = []
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, settings.FRAME_SIZE)
            frames.append(Image.fromarray(frame))
    cap.release()
    return frames