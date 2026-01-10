### ========================================================================================================================================
## Name:        Saputra Rizky Johan (사푸트라)
## Institution: Seoul National University, South Korea
## Purpose:     Undergraduate Thesis
## Department:  Computer Science and Engineering
## College:     Engineering
## Advisor:     Prof. Lee Changun (이창건 교수)
## Title:       The Application of Facial Emotion Recognition (FER) in the Detection and Measurement of Burnout and Prolonged Stress Levels
## Module:      burnout_scoring.py (See specifications and thesis section 4.1. functional modules for further details)
### ========================================================================================================================================

# =============================================================================================
# SPECIFICATIONS
# =============================================================================================
# Burnout scoring logic based on emotion probabilities
"""
burnout_scoring.py
Temporal aggregation and burnout index calculation.
"""

# =============================================================================================
# SETUP
# =============================================================================================
from collections import deque
from typing import List, Tuple
import numpy as np

# =============================================================================================
# IMPLEMENTATION
# =============================================================================================

class BurnoutScorer:
    def __init__(self, window_size: int = 30):
        """
        Maintain a sliding window of emotion probability vectors and optional landmark features.
        :param window_size: number of frames to average over
        """
        self.window_size = window_size
        self.emotion_history = deque(maxlen=window_size)  # each entry: np.array len 7
        self.landmark_history = deque(maxlen=window_size)  # each entry: list of landmarks
        self.weights = {'Angry':3,'Disgust':2,'Fear':2,'Happy':-2,'Neutral':1,'Sad':3,'Surprise':-1}
        self.order = ['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']

    def update(self, emotion_probs: np.ndarray, landmark_features: List[float] = None):
        """
        Append new frame data to history.
        :param emotion_probs: numpy array shape (7,)
        :param landmark_features: list of floats
        """
        if emotion_probs is None:
            return
        self.emotion_history.append(np.array(emotion_probs, dtype=float))
        if landmark_features is not None:
            self.landmark_history.append(np.array(landmark_features, dtype=float))

    def compute_score(self) -> int:
        """
        Compute the normalized burnout index from current window.
        :return: int 0-100
        """
        if not self.emotion_history:
            return 0
        avg = np.mean(np.array(self.emotion_history), axis=0)  # (7,)
        weighted = 0.0
        for i, label in enumerate(self.order):
            weighted += float(avg[i] * self.weights[label])
        # Normalize: choose a mapping that centers typical ranges to 0-100
        # Weighted range roughly [-2..3]*1 -> we'll shift and scale heuristically
        norm = (weighted + 3.0) / 6.0 * 100.0
        norm = float(np.clip(norm, 0.0, 100.0))
        return int(round(norm))

    def compute_alert(self, threshold: int = 80) -> Tuple[bool, str]:
        """
        Return alert boolean and message if current score exceeds threshold.
        """
        score = self.compute_score()
        if score >= threshold:
            return True, f"ALERT: High burnout ({score}%)"
        return False, ""
    
### ===================================================================================================================================
## END: Add implementations if necessary
### ===================================================================================================================================