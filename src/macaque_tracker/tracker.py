import numpy as np
from typing import List, Dict, Any, Tuple
from collections import defaultdict
import cv2


class SimpleTracker:
    def __init__(self, max_disappeared: int = 30, max_distance: float = 100):
        self.next_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        
    def register(self, detection: Dict[str, Any]) -> int:
        self.objects[self.next_id] = {
            'centroid': detection['center'],
            'bbox': detection['bbox'],
            'features': detection.get('features', []),
            'confidence': detection['confidence']
        }
        self.disappeared[self.next_id] = 0
        self.next_id += 1
        return self.next_id - 1
        
    def deregister(self, object_id: int):
        del self.objects[object_id]
        del self.disappeared[object_id]
        
    def update(self, detections: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects
            
        if len(self.objects) == 0:
            for detection in detections:
                detection['track_id'] = self.register(detection)
            return self.objects
            
        # Compute distance matrix
        object_centroids = np.array([obj['centroid'] for obj in self.objects.values()])
        detection_centroids = np.array([det['center'] for det in detections])
        
        D = np.linalg.norm(object_centroids[:, np.newaxis] - detection_centroids, axis=2)
        
        # Find minimum distances
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]
        
        used_row_idxs = set()
        used_col_idxs = set()
        
        object_ids = list(self.objects.keys())
        
        for (row, col) in zip(rows, cols):
            if row in used_row_idxs or col in used_col_idxs:
                continue
                
            if D[row, col] > self.max_distance:
                continue
                
            object_id = object_ids[row]
            self.objects[object_id] = {
                'centroid': detections[col]['center'],
                'bbox': detections[col]['bbox'],
                'features': detections[col].get('features', []),
                'confidence': detections[col]['confidence']
            }
            self.disappeared[object_id] = 0
            detections[col]['track_id'] = object_id
            
            used_row_idxs.add(row)
            used_col_idxs.add(col)
            
        unused_row_idxs = set(range(0, D.shape[0])).difference(used_row_idxs)
        unused_col_idxs = set(range(0, D.shape[1])).difference(used_col_idxs)
        
        if D.shape[0] >= D.shape[1]:
            for row in unused_row_idxs:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
        else:
            for col in unused_col_idxs:
                detections[col]['track_id'] = self.register(detections[col])
                
        return self.objects


class TrackletManager:
    def __init__(self):
        self.tracklets = defaultdict(list)
        
    def add_detection(self, track_id: int, frame_num: int, detection: Dict[str, Any]):
        self.tracklets[track_id].append({
            'frame': frame_num,
            'bbox': detection['bbox'],
            'confidence': detection['confidence'],
            'features': detection.get('features', [])
        })
        
    def get_tracklet_features(self, track_id: int) -> np.ndarray:
        if track_id not in self.tracklets:
            return np.array([])
            
        features = []
        for detection in self.tracklets[track_id]:
            if 'features' in detection and len(detection['features']) > 0:
                features.append(detection['features'])
                
        if len(features) == 0:
            return np.array([])
            
        return np.mean(features, axis=0)
        
    def get_all_tracklet_features(self) -> Dict[int, np.ndarray]:
        return {track_id: self.get_tracklet_features(track_id) 
                for track_id in self.tracklets.keys()}