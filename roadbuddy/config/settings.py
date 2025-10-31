from pathlib import Path
from dataclasses import dataclass

@dataclass
class Settings:
    DATA_ROOT = Path("../data")
    TRAIN_JSON = DATA_ROOT / "train/train.json"
    PUBLIC_JSON = DATA_ROOT / "public/test.json"
    MODEL_DIR = Path("../models")
    OUTPUT_DIR = Path("../outputs")
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Inference
    NUM_FRAMES = 8
    FRAME_SIZE = (640, 640)
    CONF_THRESHOLD = 0.25
    MAX_TOKENS = 10

    # Laws
    LAWS_PATH = Path(__file__).parent.parent / "laws/vietnam_laws.json"

settings = Settings()