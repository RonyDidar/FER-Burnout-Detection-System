### ========================================================================================================================================
## Name:        Saputra Rizky Johan (사푸트라)
## Institution: Seoul National University, South Korea
## Purpose:     Undergraduate Thesis
## Department:  Computer Science and Engineering
## College:     Engineering
## Advisor:     Prof. Lee Changun (이창건 교수)
## Title:       The Application of Facial Emotion Recognition (FER) in the Detection and Measurement of Burnout and Prolonged Stress Levels
## Module:      main_multi_user_record.py (See specifications and thesis section 4.1. functional modules for further details)
### ========================================================================================================================================

# =============================================================================================
# SPECIFICATIONS
# =============================================================================================
# Multi-user FER burnout detection with recording
"""
main_multi_user_record.py
Full multi-user real-time system with video recording and all post-session analysis.
"""

# =============================================================================================
# SETUP
# =============================================================================================
import cv2
import os
from datetime import datetime
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

def run_multi_user_record(user_ids):
    # prepare video recording folder and filename
    os.makedirs('recorded_videos', exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    video_path = os.path.join('recorded_videos', f"multi_user_{stamp}.avi")
    frame_width, frame_height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_path, fourcc, 20.0, (frame_width, frame_height))

    # initialize per-user
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
    cap.set(3, frame_width); cap.set(4, frame_height)
    print("[MultiRecord] Recording started. Press 'q' to stop.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # process
        for uid, data in users.items():
            detector = data['detector']
            er = data['emotion']
            scorer = data['scorer']
            dashboard = data['dashboard']
            faces = detector.detect_faces(frame)
            emotions = []
            landmark_feats = []
            for f in faces:
                x,y,w,h = f['bbox']
                x2, y2 = x+w, y+h
                face_img = frame[y:y2, x:x2].copy()
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
            predicted = None
            if len(data['emotion_seq']) >= data['lstm'].seq_len:
                predicted = data['lstm'].predict_next(data['emotion_seq'], data['landmark_seq'])
            frame = dashboard.update(frame, faces, emotions, scorer.compute_score(), landmark_feats)
            if predicted is not None:
                cv2.putText(frame, f"{uid} Pred: {predicted:.1f}%", (10, 30+30*user_ids.index(uid)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

        # write & show
        out.write(frame)
        cv2.imshow("Multi-User Recording Burnout System", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"[MultiRecord] Video saved to {video_path}")

    # post-session ops
    for uid, data in users.items():
        s_entry = data['survey'].collect_survey()
        data['survey'].save_survey(s_entry)
        data['session'].save_session(data['dashboard'])
        compute_correlation(uid)
        analyzer = LongitudinalAnalyzer()
        analyzer.plot_longitudinal_trend(uid, freq='D')
        # explainability visualization
        data['explain'].plot_landmark_heatmap(data['landmark_seq'], data['burnout_seq'], user_id=uid)

# =============================================================================================
# DEMO TESTING
# =============================================================================================
if __name__ == "__main__":
    run_multi_user_record(['user001','user002'])

### ===================================================================================================================================
## END: Add implementations if necessary
### ===================================================================================================================================