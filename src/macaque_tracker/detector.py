import cv2
import torch
from ultralytics import YOLO
import numpy as np


class MacaqueDetector:
    def __init__(self, model_path: str = 'yolo11n.pt', confidence: float = 0.5):
        self.model = YOLO(model_path)
        self.confidence = confidence
        
    def detect_primates(self, frame: np.ndarray) -> list[dict[str, any]]:
        results = self.model(frame, conf=self.confidence)
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    if int(box.cls) in [16, 17, 18, 19, 20]:  # Various animal classes that might include primates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0])
                        
                        detection = {
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': conf,
                            'class_id': int(box.cls),
                            'center': [(x1 + x2) / 2, (y1 + y2) / 2]
                        }
                        detections.append(detection)
        
        return detections
    
    def extract_features(self, frame: np.ndarray, bbox: list[int]) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        cropped = frame[y1:y2, x1:x2]
        
        if cropped.size == 0:
            return np.zeros(512)
        
        resized = cv2.resize(cropped, (224, 224))
        
        # Simple feature extraction using histogram and basic statistics
        hist_b = cv2.calcHist([resized], [0], None, [32], [0, 256])
        hist_g = cv2.calcHist([resized], [1], None, [32], [0, 256])
        hist_r = cv2.calcHist([resized], [2], None, [32], [0, 256])
        
        # Additional features
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_ratio = np.sum(edges > 0) / edges.size
        
        # Texture features
        mean_intensity = np.mean(gray)
        std_intensity = np.std(gray)
        
        features = np.concatenate([
            hist_b.flatten(),
            hist_g.flatten(),
            hist_r.flatten(),
            [edge_ratio, mean_intensity, std_intensity]
        ])
        
        return features[:512]  # Ensure consistent feature size