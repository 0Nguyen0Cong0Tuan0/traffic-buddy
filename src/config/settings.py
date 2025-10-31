from dataclasses import dataclass

@dataclass
class InferenceConfig:
    num_frames: int = 8
    frame_strategy: str = "uniform"
    enhance_night: bool = True
    use_ocr: bool = True
    use_rules: bool = True
    max_tokens: int = 10
    temperature: float = 0.0