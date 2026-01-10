### ========================================================================================================================================
## Name:        Saputra Rizky Johan (사푸트라)
## Institution: Seoul National University, South Korea
## Purpose:     Undergraduate Thesis
## Department:  Computer Science and Engineering
## College:     Engineering
## Advisor:     Prof. Lee Changun (이창건 교수)
## Title:       The Application of Facial Emotion Recognition (FER) in the Detection and Measurement of Burnout and Prolonged Stress Levels
## Module:      dashboard.py (See specifications and thesis section 4.1. functional modules for further details)
### ========================================================================================================================================

# =============================================================================================
# SPECIFICATIONS
# =============================================================================================
# Dashboard for plotting real-time and session trends
"""
dashboard.py
Real-time visualization and session data collection.
Keeps a session_data list for logging to CSV later.
"""

# =============================================================================================
# SETUP
# =============================================================================================
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
from typing import List, Dict, Any
from datetime import datetime

# =============================================================================================
# IMPLEMENTATION
# =============================================================================================
class BurnoutDashboard:
    def __init__(self, plot_maxlen: int = 300):
        self.burnout_history = deque(maxlen=plot_maxlen)
        self.session_data: List[Dict[str, Any]] = []  # store per-frame/per-face records
        # Prepare matplotlib figure for live plotting
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(6,2.5))
        self.ax.set_ylim(0, 100)
        self.line, = self.ax.plot([], [], color='red')
        self.ax.set_title('Burnout Index (recent)')
        self.ax.set_xlabel('Frames')
        self.ax.set_ylabel('Burnout %')

    def update(self, frame: np.ndarray, faces: List[Dict], emotions: List, burnout_score: int, landmark_feats: List) -> np.ndarray:
        """
        Draw overlays and record session data.
        :param frame: BGR frame (will be updated in-place)
        :param faces: list of face dicts from detector
        :param emotions: list of tuples (label, probs) corresponding to faces
        :param burnout_score: integer 0-100 (current aggregated score)
        :param landmark_feats: list of lists corresponding to faces
        :return: annotated frame
        """
        # Draw faces and overlays
        for i, face in enumerate(faces):
            x, y, w, h = face['bbox']
            # safe bounds
            x2, y2 = x + w, y + h
            x2 = min(frame.shape[1]-1, x2)
            y2 = min(frame.shape[0]-1, y2)
            cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
            keypoints = face.get('keypoints', {})
            for pt in keypoints.values():
                if isinstance(pt, (tuple, list)):
                    cv2.circle(frame, tuple(pt), 2, (0, 0, 255), -1)

            # write emotion + burnout
            if i < len(emotions):
                label, probs = emotions[i]
                prob_max = float(np.max(probs)) if probs is not None else 0.0
                cv2.putText(frame, f"{label} ({prob_max*100:.0f}%)", (x, max(15, y-20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (36,255,12), 2)
            # global burnout
            cv2.putText(frame, f"Burnout: {burnout_score}%", (10, frame.shape[0]-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        # Append to history and session_data
        self.burnout_history.append(burnout_score)
        # For recording, create per-frame aggregated record (one record per frame)
        record = {
            'Timestamp': datetime.now().isoformat(),
            'Burnout': burnout_score,
            'NumFaces': len(faces),
            'Faces': []
        }
        for i, face in enumerate(faces):
            label, probs = emotions[i] if i < len(emotions) else ("", [0]*7)
            landmarks = face.get('keypoints', {})
            landmark_feat = landmark_feats[i] if i < len(landmark_feats) else []
            face_record = {
                'bbox': face['bbox'],
                'label': label,
                'probs': [float(p) for p in (probs if probs is not None else [0]*7)],
                'landmark_features': [float(x) for x in (landmark_feat if landmark_feat is not None else [])],
                'keypoints': landmarks
            }
            record['Faces'].append(face_record)
            # Also append a per-face row to session_data (flat)
            self.session_data.append({
                'Timestamp': record['Timestamp'],
                'User_FaceIndex': i,
                'BBox': face['bbox'],
                'Emotion': label,
                'Probs': face_record['probs'],
                'LandmarkFeatures': face_record['landmark_features'],
                'Burnout': burnout_score
            })

        # Update the live plot (non-blocking)
        try:
            self.line.set_xdata(range(len(self.burnout_history)))
            self.line.set_ydata(list(self.burnout_history))
            self.ax.relim()
            self.ax.autoscale_view(True,True,True)
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
        except Exception:
            pass

        return frame

    def plot_burnout_trend(self, burnout_seq: List[int], user_id: str = "user"):
        """
        Plot session-specific burnout sequence in a blocking matplotlib window (for saving as figure).
        """
        plt.figure(figsize=(10,4))
        plt.plot(burnout_seq, marker='o', linestyle='-')
        plt.title(f"Burnout Trend - {user_id}")
        plt.xlabel('Frame index')
        plt.ylabel('Burnout (%)')
        plt.grid(True)
        plt.show()

    def save_session_csv(self, filename: str):
        """
        Save the flattened session_data list (per-face rows) to CSV.
        """
        if not self.session_data:
            print("[Dashboard] No session data to save.")
            return
        # Expand lists (Probs, LandmarkFeatures) to strings so CSV is simple
        df = pd.DataFrame(self.session_data)
        # Convert lists to JSON-like strings
        df['Probs'] = df['Probs'].apply(lambda p: ','.join([f"{x:.4f}" for x in p]) if isinstance(p, list) else "")
        df['LandmarkFeatures'] = df['LandmarkFeatures'].apply(lambda p: ','.join([f"{x:.4f}" for x in p]) if isinstance(p, list) else "")
        df.to_csv(filename, index=False)
        print(f"[Dashboard] Session CSV saved to: {filename}")

### ===================================================================================================================================
## END: Add implementations if necessary
### ===================================================================================================================================