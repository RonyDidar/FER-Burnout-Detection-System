### ========================================================================================================================================
## Name:        Saputra Rizky Johan (사푸트라)
## Institution: Seoul National University, South Korea
## Purpose:     Undergraduate Thesis
## Department:  Computer Science and Engineering
## College:     Engineering
## Advisor:     Prof. Lee Changun (이창건 교수)
## Title:       The Application of Facial Emotion Recognition (FER) in the Detection and Measurement of Burnout and Prolonged Stress Levels
## Module:      explainability.py (See specifications and thesis section 4.1. functional modules for further details)
### ========================================================================================================================================

# =============================================================================================
# SPECIFICATIONS
# =============================================================================================
# Landmark heatmap visualization for explainability
"""
explainability.py
Generate simple landmark heatmaps / contribution plots for interpretability.
"""

# =============================================================================================
# SETUP
# =============================================================================================
import cv2
import matplotlib.pyplot as plt
import numpy as np
from typing import List

# =============================================================================================
# IMPLEMENTATION
# =============================================================================================

class BurnoutExplainability:
    def __init__(self):
        pass

    def plot_landmark_heatmap(self, landmarks_list: List[List[float]], burnout_scores: List[float], user_id: str = "user"):
        """
        landmarks_list: list of [eye_distance, mouth_width, nose_eye_ratio] per frame
        burnout_scores: corresponding list of burnout values per frame
        Produces a bar plot of weighted contributions.
        """
        if not landmarks_list or not burnout_scores:
            print("[Explainability] No data to plot.")
            return
        landmarks_arr = np.array(landmarks_list)  # shape (N,3)
        burnout_arr = np.array(burnout_scores).reshape(-1)  # (N,)
        # compute contribution via (landmarks^T * burnout), normalized
        weighted = (landmarks_arr.T @ burnout_arr)  # length 3
        # scale for visualization
        weighted_norm = weighted / (np.max(np.abs(weighted)) + 1e-6)
        labels = ['EyeDist', 'MouthWidth', 'NoseEyeRatio']
        plt.figure(figsize=(6,4))
        plt.bar(labels, weighted_norm)
        plt.title(f'Landmark Contribution to Burnout - {user_id}')
        plt.ylabel('Normalized Weighted Contribution')
        plt.grid(True)
        plt.show()

    def show_keypoint_overlay_heatmap(self, frame, faces, contributions_per_face):
        """
        Optional: overlay simple colored dots on face keypoints based on contributions.
        contributions_per_face: list of dicts mapping keypoint name to scalar weight
        (This is a simple visualization and not a dense heatmap.)
        """
        vis = frame.copy()
        for face, contrib in zip(faces, contributions_per_face):
            keypoints = face.get('keypoints', {})
            for k, pt in keypoints.items():
                weight = contrib.get(k, 0.0)
                # map weight to color intensity
                intensity = int(255 * min(1.0, max(0.0, abs(weight))))
                color = (0, intensity, 255-intensity)  # blue->red gradient
                cv2.circle(vis, tuple(pt), 4, color, -1)
        plt.figure(figsize=(6,4))
        plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

### ===================================================================================================================================
## END: Add implementations if necessary
### ===================================================================================================================================