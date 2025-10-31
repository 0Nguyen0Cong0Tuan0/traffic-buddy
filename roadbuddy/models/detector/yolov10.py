from ultralytics import YOLO
from typing import List, Tuple
import torch

class YOLODetector:
    def __init__(self, model_path: str, conf: float = 0.25):
        self.model = YOLO(model_path)
        self.conf = conf

    def detect(self, frame) -> List[Tuple[str, float, Tuple[int,int,int,int]]]:
        results = self.model(frame, conf=self.conf, verbose=False)[0]
        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            label = results.names[cls_id]
            detections.append((label, conf, (x1, y1, x2, y2)))
        return detections