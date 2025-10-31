import os
from pathlib import Path

class Paths:
    ROOT = Path(__file__).parent.parent.parent
    DATA_ROOT = Path("./data/raw")
    MODEL_CACHE = Path("./models/pretrained")
    TRAIN_JSON = DATA_ROOT / "train/train.json"
    PUBLIC_JSON = DATA_ROOT / "public/test.json"
    VIDEO_ROOT = DATA_ROOT / "train/videos"

    @staticmethod
    def video_path(rel_path: str) -> Path:
        return Paths.VIDEO_ROOT / rel_path