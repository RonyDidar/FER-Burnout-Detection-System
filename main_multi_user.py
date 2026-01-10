### ========================================================================================================================================
## Name:        Saputra Rizky Johan (사푸트라)
## Institution: Seoul National University, South Korea
## Purpose:     Undergraduate Thesis
## Department:  Computer Science and Engineering
## College:     Engineering
## Advisor:     Prof. Lee Changun (이창건 교수)
## Title:       The Application of Facial Emotion Recognition (FER) in the Detection and Measurement of Burnout and Prolonged Stress Levels
## Module:      main_multi_user.py (See specifications and thesis section 4.1. functional modules for further details)
### ========================================================================================================================================

# =============================================================================================
# SPECIFICATIONS
# =============================================================================================
# Multi-user FER burnout detection without recording
"""
main_multi_user.py
Multi-user processing without video recording. Each user has separate modules and dashboards.
Simple demo for simultaneous tracking.
"""

# =============================================================================================
# SETUP
# =============================================================================================
import cv2
from face_detection import FaceDetector
from emotion_recognition import EmotionRecognizer
from burnout_scoring import BurnoutScorer
from dashboard import BurnoutDashboard
from session_manager import SessionManager
from survey_integration import SurveyCollector
from lstm_burnout_predictor import LSTMBurnoutPredictor
from explainability import BurnoutExplainability
from analysis import compute_correlation
from longitudinal_analysis import LongitudinalAnalyzer

# =============================================================================================
# IMPLEMENTATION
# =============================================================================================

def safe_crop(frame, bbox):
    h, w = frame.shape[:2]
    x,y,ww,hh = bbox
    x = max(0, x); y = max(0, y)
    x2 = min(w, x + ww); y2 = min(h, y + hh)
    if x2 <= x or y2 <= y:
        return None
    return frame[y:y2, x:x2]


def run_multi_user(user_ids):
    # initialize per-user modules
    users = {}
    for uid in user_ids:
        users[uid] = {
            'detector': FaceDetector(),
            'emotion': EmotionRecognizer(model_path='emo0.1.h5'),
            'scorer': BurnoutScorer(window_size=30),
            'dashboard': BurnoutDashboard(),
            'session': SessionManager(uid),
            'survey': SurveyCollector(uid),
            'lstm': LSTMBurnoutPredictor(seq_len=30, input_dim=10),
            'explain': BurnoutExplainability(),
            'emotion_seq': [],
            'landmark_seq': [],
            'burnout_seq': []
        }

    cap = cv2.VideoCapture(0)
    print("[Main Multi] Press 'q' to stop.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # process same frame for all users (for simple demo)
        for uid, data in users.items():
            detector = data['detector']
            er = data['emotion']
            scorer = data['scorer']
            dashboard = data['dashboard']
            # detect faces and update
            faces = detector.detect_faces(frame)
            emotions = []
            landmark_feats = []
            for f in faces:
                x,y,w,h = f['bbox']
                x2, y2 = x+w, y+h
                
                face_img = safe_crop(frame, f['bbox'])
                label, probs = er.predict_emotion(face_img)

                emotions.append((label, probs))
                lf = detector.extract_features(f.get('keypoints', {}))
                landmark_feats.append(lf)
                scorer.update(probs, lf)
                data['emotion_seq'].append(probs)
                data['landmark_seq'].append(lf)
                data['burnout_seq'].append(scorer.compute_score())
                if len(data['emotion_seq']) > data['lstm'].seq_len:
                    data['emotion_seq'].pop(0); data['landmark_seq'].pop(0); data['burnout_seq'].pop(0)
            # predict and update dashboard
            predicted = None
            if len(data['emotion_seq']) >= data['lstm'].seq_len:
                predicted = data['lstm'].predict_next(data['emotion_seq'], data['landmark_seq'])
            frame = dashboard.update(frame, faces, emotions, scorer.compute_score(), landmark_feats)
            if predicted is not None:
                cv2.putText(frame, f"{uid} Pred: {predicted:.1f}%", (10, 30+30*user_ids.index(uid)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

        cv2.imshow("Multi-User Burnout System", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # post-session operations for each user
    for uid, data in users.items():
        s_entry = data['survey'].collect_survey()
        data['survey'].save_survey(s_entry)
        data['session'].save_session(data['dashboard'])
        compute_correlation(uid)
        analyzer = LongitudinalAnalyzer()
        analyzer.plot_longitudinal_trend(uid, freq='D')

# =============================================================================================
# DEMO TESTING
# =============================================================================================
if __name__ == "__main__":
    run_multi_user(['user001','user002'])

### ===================================================================================================================================
## END: Add implementations if necessary
### ===================================================================================================================================