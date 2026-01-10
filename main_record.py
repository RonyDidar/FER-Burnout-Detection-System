### ========================================================================================================================================
## Name:        Saputra Rizky Johan (사푸트라)
## Institution: Seoul National University, South Korea
## Purpose:     Undergraduate Thesis
## Department:  Computer Science and Engineering
## College:     Engineering
## Advisor:     Prof. Lee Changun (이창건 교수)
## Title:       The Application of Facial Emotion Recognition (FER) in the Detection and Measurement of Burnout and Prolonged Stress Levels
## Module:      main_record.py (See specifications and thesis section 4.1. functional modules for further details)
### ========================================================================================================================================

# =============================================================================================
# SPECIFICATIONS
# =============================================================================================

# =============================================================================================
# SETUP
# =============================================================================================
import os
import cv2
import csv
from datetime import datetime
from face_detection import FaceDetector
from emotion_recognition import EmotionRecognizer
from burnout_scoring import BurnoutScorer
from dashboard import BurnoutDashboard
from session_manager import SessionManager
from survey_integration import SurveyCollector
from lstm_burnout_predictor import LSTMBurnoutPredictor
from explainability import BurnoutExplainability
from longitudinal_analysis import LongitudinalAnalyzer
from analysis import compute_correlation

# =============================================================================================
# IMPLEMENTATION
# =============================================================================================
# Declare a function to securely crop the detected frames through a bounding box
def safe_crop(frame, bbox):
    h, w = frame.shape[:2]
    x, y, ww, hh = bbox
    x = max(0, int(x)); y = max(0, int(y))
    x2 = min(w, x + int(ww)); y2 = min(h, y + int(hh))
    if x2 <= x or y2 <= y:
        return None
    return frame[y:y2, x:x2]

# Declare a function to run and execute a real time facial detection session for a single user
def run_single_user_record(user_id='user001', record_video=True, frame_width=640, frame_height=480, fps=20.0):
    # prepare folders
    os.makedirs('recorded_videos', exist_ok=True)
    os.makedirs('data_sessions', exist_ok=True)

    # timestamped filenames
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    video_filename = os.path.join('recorded_videos', f"{user_id}_{stamp}.avi")
    frame_log_filename = os.path.join('data_sessions', f"{user_id}_{stamp}_frames.csv")

    # Initialize modules
    fd = FaceDetector()
    er = EmotionRecognizer(model_path='emo0.1.h5')
    scorer = BurnoutScorer(window_size=30)
    dashboard = BurnoutDashboard()
    session = SessionManager(user_id)
    survey = SurveyCollector(user_id)
    lstm = LSTMBurnoutPredictor(seq_len=30, input_dim=10)
    explain = BurnoutExplainability()

        # Video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0, cv2.CAP_QT)  # fallback for macOS
    if not cap.isOpened():
        print("Could not open webcam with any backend")
        return
    print("Camera backend:", cap.getBackendName())

    # Warm up camera: grab a few frames before recording
    warmup_ok = False
    for _ in range(5):
        ret, frame = cap.read()
        if ret and frame is not None:
            warmup_ok = True
            break
    if not warmup_ok:
        print("Could not grab initial frame, exiting.")
        cap.release()
        return

    # Video writer (use mp4 for macOS)
    out = None
    if record_video:
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(video_filename, fourcc, fps, (frame_width, frame_height))
        print(f"[Record] Writing video to: {video_filename}")


    # CSV per-frame log
    csv_file = open(frame_log_filename, mode='w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Timestamp', 'FrameIndex', 'Burnout', 'NumFaces', 'FaceIndex', 'Label', 'Probs'])

    emotion_seq, landmark_seq, burnout_seq = [], [], []
    frame_index = 0

    print("[MainRecord] Press 'q' in the video window to stop. Ctrl+C also works.")

    cv2.namedWindow("Single-User Burnout System", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Single-User Burnout System",
                      cv2.WND_PROP_FULLSCREEN,
                      cv2.WINDOW_NORMAL)

    idle_frames = 0  # fallback counter for frozen webcam

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                idle_frames += 1
                if idle_frames > 50:
                    print("[MainRecord] No frames detected, exiting.")
                    break
                continue
            idle_frames = 0

            # Resize to standard size for speed (if camera gives larger)
            frame = cv2.resize(frame, (frame_width, frame_height))

            # Detect faces
            faces = fd.detect_faces(frame)
            emotions = []
            landmark_feats = []

            for i, f in enumerate(faces):
                crop = safe_crop(frame, f['bbox'])
                if crop is None:
                    continue
                label, probs = er.predict_emotion(crop)
                emotions.append((label, probs))
                lf = fd.extract_features(f.get('keypoints', {}))
                landmark_feats.append(lf)

                # Update scorer and sequences
                scorer.update(probs, lf)
                emotion_seq.append(probs)
                landmark_seq.append(lf)
                burnout_seq.append(scorer.compute_score())

                # keep only last seq_len
                if len(emotion_seq) > lstm.seq_len:
                    emotion_seq.pop(0); landmark_seq.pop(0); burnout_seq.pop(0)

                # write CSV per-face per-frame (flatten probs to string)
                probs_str = ",".join([f"{p:.4f}" for p in (probs if probs is not None else [0]*7)])
                csv_writer.writerow([datetime.now().isoformat(), frame_index, scorer.compute_score(), len(faces), i, label, probs_str])

            # LSTM prediction (if enough history)
            predicted = None
            if len(emotion_seq) >= lstm.seq_len:
                try:
                    predicted = lstm.predict_next(emotion_seq, landmark_seq)
                except Exception as e:
                    print("[LSTM] predict error:", e)
                    predicted = None

            # Overlay and show
            annotated = dashboard.update(frame.copy(), faces, emotions, scorer.compute_score(), landmark_feats)
            if predicted is not None:
                cv2.putText(annotated, f"Predicted Next Burnout: {predicted:.1f}%", (10,30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

            # display
            cv2.imshow("Single-User Burnout System", annotated)

            # write to video file
            if record_video and out is not None:
                # ensure frame is same size as writer expects
                try:
                    out.write(annotated)
                except Exception as e:
                    print("[Record] Failed to write frame:", e)

            frame_index += 1

            # quit key or window closed
            key = cv2.waitKey(10)
            if key in [ord('q'), 27]:  # q or ESC
                print("[MainRecord] Exit key pressed, stopping.")
                break
            if cv2.getWindowProperty("Single-User Burnout System", cv2.WND_PROP_VISIBLE) < 1:
                print("[MainRecord] Window closed, stopping.")
                break


    except KeyboardInterrupt:
        print("[MainRecord] KeyboardInterrupt received, stopping...")

    finally:
        # cleanup
        cap.release()
        if out is not None:
            out.release()
            print(f"[Record] Video saved to: {video_filename}")
        csv_file.close()

        cv2.destroyAllWindows()

        # Post-session ops
        try:
            # save dashboard session data and metrics using SessionManager
            session.save_session(dashboard)
            s_entry = survey.collect_survey()
            survey.save_survey(s_entry)
            compute_correlation(user_id)
            analyzer = LongitudinalAnalyzer()
            analyzer.plot_longitudinal_trend(user_id, freq='D')
            # explainability visualization (optional)
            explain.plot_landmark_heatmap(landmark_seq, burnout_seq, user_id=user_id)
        except Exception as e:
            print("[PostSession] Error during post-session operations:", e)

        print("[MainRecord] Session completed.")

# =============================================================================================
# DEMO TESTING
# =============================================================================================
if __name__ == "__main__":
    run_single_user_record(user_id="user001", record_video=True)

### ===================================================================================================================================
## END: Add implementations if necessary
### ===================================================================================================================================