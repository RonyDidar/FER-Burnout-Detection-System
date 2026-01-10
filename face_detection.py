### ========================================================================================================================================
## Name:        Saputra Rizky Johan (사푸트라)
## Institution: Seoul National University, South Korea
## Purpose:     Undergraduate Thesis
## Department:  Computer Science and Engineering
## College:     Engineering
## Advisor:     Prof. Lee Changun (이창건 교수)
## Title:       The Application of Facial Emotion Recognition (FER) in the Detection and Measurement of Burnout and Prolonged Stress Levels
## Module:      face_detection.py (See specifications and thesis section 4.1. functional modules for further details)
### ========================================================================================================================================

# =============================================================================================
# SPECIFICATIONS
# =============================================================================================
# Face detection module using MTCNN or OpenCV
"""
face_detection.py
Face detection and landmark extraction using MTCNN.
Provides:
 - FaceDetector class with detect_faces(), extract_features(), draw_faces()
"""

# =============================================================================================
# SETUP
# =============================================================================================
import cv2
import numpy as np
from mtcnn import MTCNN
from typing import List, Dict, Tuple

# =============================================================================================
# IMPLEMENTATION
# =============================================================================================

class FaceDetector:
    def __init__(self, min_face_size: int = 40):
        """
        Initialize MTCNN face detector.
        :param min_face_size: ignore detections smaller than this width/height
        """
        self.detector = MTCNN()
        self.min_face_size = min_face_size

    def detect_faces(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect faces in an image.
        :param frame: BGR image (numpy array)
        :return: list of face dicts { 'bbox':(x,y,w,h), 'keypoints':{...} }
        """
        # MTCNN expects RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.detector.detect_faces(rgb)
        faces = []
        for res in results:
            x, y, w, h = res['box']  # may contain negative numbers
            x, y = max(0, x), max(0, y)
            if w < self.min_face_size or h < self.min_face_size:
                continue
            keypoints = res.get('keypoints', {})
            faces.append({'bbox': (x, y, w, h), 'keypoints': keypoints})
        return faces

    def extract_features(self, keypoints: Dict) -> List[float]:
        """
        Compute geometry-based landmark features from MTCNN keypoints.
        :param keypoints: dict with left_eye, right_eye, nose, mouth_left, mouth_right
        :return: list of numeric features [eye_distance, mouth_width, nose_eye_ratio]
        """
        # Default safe vectors
        try:
            left_eye = np.array(keypoints['left_eye'])
            right_eye = np.array(keypoints['right_eye'])
            nose = np.array(keypoints['nose'])
            mouth_left = np.array(keypoints['mouth_left'])
            mouth_right = np.array(keypoints['mouth_right'])
        except Exception:
            # If missing keypoints, return zeros
            return [0.0, 0.0, 0.0]

        eye_distance = float(np.linalg.norm(left_eye - right_eye))
        mouth_width = float(np.linalg.norm(mouth_left - mouth_right))
        nose_eye_ratio = float((np.linalg.norm(left_eye - nose) + np.linalg.norm(right_eye - nose)) / 2.0 + 1e-6)
        return [eye_distance, mouth_width, nose_eye_ratio]

    def draw_faces(self, frame: np.ndarray, faces: List[Dict]) -> np.ndarray:
        """
        Draw bounding boxes and keypoints on the frame.
        :param frame: BGR image
        :param faces: list from detect_faces()
        :return: annotated frame
        """
        for face in faces:
            x, y, w, h = face['bbox']
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            keypoints = face.get('keypoints', {})
            for k, pt in keypoints.items():
                if isinstance(pt, (tuple, list)) and len(pt) >= 2:
                    cv2.circle(frame, tuple(pt), 2, (0, 0, 255), -1)
        return frame

### ===================================================================================================================================
## END: Add implementations if necessary
### ===================================================================================================================================