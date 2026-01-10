### ========================================================================================================================================
## Name:        Saputra Rizky Johan (사푸트라)
## Institution: Seoul National University, South Korea
## Purpose:     Undergraduate Thesis
## Department:  Computer Science and Engineering
## College:     Engineering
## Advisor:     Prof. Lee Changun (이창건 교수)
## Title:       The Application of Facial Emotion Recognition (FER) in the Detection and Measurement of Burnout and Prolonged Stress Levels
## Module:      main.py (See specifications and thesis section 4.1. functional modules for further details)
### ========================================================================================================================================

# =============================================================================================
# SPECIFICATIONS
# =============================================================================================
"""
    # Module:   main.py
    # Purpose:  Single-user real-time entry point to run the FER-based burnout detection system.
    # Functions:
    - safe_crop(): Safely crops detected face regions within frame bounds.
    - run_single_user(): Runs end-to-end real-time pipelines (refer to chapter 4 of the thesis).
    # Inputs:
    - Live webcam frames (BGR).
    - CNN model weights ('emo0_1.h5').
    - User ID and video-recording flag.
    # Outputs:
    - Real time annotated video window and its stored AVI recordings (recorded_videos directory).
    - Burnout score CSVs, survey data, and longitudinal plots per user (data_sessions directory).
    # Core Dependencies:
    - OpenCV (cv2) for video I/O and drawing overlays.
    - FaceDetector           : face_detection.py
    - EmotionRecognizer      : emotion_recognition.py
    - BurnoutScorer          : burnout_scoring.py
    - BurnoutDashboard       : dashboard.py
    - SessionManager         : session_manager.py
    - SurveyCollector        : survey_integration.py
    - LSTMBurnoutPredictor   : lstm_burnout_predictor.py
    - BurnoutExplainability  : explainability.py
    - compute_correlation    : analysis.py
    - LongitudinalAnalyzer   : longitudinal_analysis.py
    Notes:
    - Default model path is 'emo0_1.h5'. (Set this model to avoid functionality mismatch)
    - Scorer returns a burnout proxy to ensure consistency.
    - Select 'q' to end session. Otherwise, '^c' in case of faulty.
    - LSTM expects a fixed sequence length.
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
# Declare a function to securely crop the detected frames through a bounding box
def safe_crop(frame, bbox):
    h, w = frame.shape[:2]
    x,y,ww,hh = bbox
    x = max(0, x); y = max(0, y)
    x2 = min(w, x + ww); y2 = min(h, y + hh)
    if x2 <= x or y2 <= y:
        return None
    return frame[y:y2, x:x2]

# Declare a function to run and execute a real time facial detection session for a single user
def run_single_user(user_id='user001', record_video=False):
    # Instantiate all the components of the pipeline
    fd = FaceDetector()                                             # Facial detection
    er = EmotionRecognizer(model_path = 'emo0.1.h5')         # CNN classifier
    scorer = BurnoutScorer(window_size = 30)                        # Burnout scoring system
    dashboard = BurnoutDashboard()                                  # Real time dashboard
    session = SessionManager(user_id)                               # Persistent session manager
    survey = SurveyCollector(user_id)                               # Post-session MBI survey
    lstm = LSTMBurnoutPredictor(seq_len = 30, input_dim = 10)       # LSTM Model
    explain = BurnoutExplainability()                               # Grad-CAM and saliency

    # Call the default camera (set the index 0 as the primary camera for this project)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Unable to open camera (index 0). Check permissions or device.")
        return

    # Prepare the annotated video recorder with the specified dimensions
    if record_video:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(f"{user_id}_record.avi", fourcc, 20.0, (640,480))

    # Initialize the lists of the sequences for temporal inputs
    emotion_seq, landmark_seq, burnout_seq = [], [], []
    print("[Main] Press 'q' on video window to stop session.")
    while True:
        ret, frame = cap.read()
        if not ret:
            # Terminate the loop if camrea is not read.
            break

        # Declare the facial detection for each frame
        faces = fd.detect_faces(frame)
        
        # Initialize the lists of emotion and feature vectors with the detected faces.
        emotions = []
        landmark_feats = []

        # Declare a handling process for each detected face
        for f in faces:
            # Crop the facial region with its bounds and predict the emotion label and probability
            face_img = safe_crop(frame, f['bbox'])
            label, probs = er.predict_emotion(face_img)

            # Update the downstream buffers strictly when the predictions are valid
            if label is not None and probs is not None:
                emotions.append((label, probs))

                 # Extract the landmark features for updating the LSTM features and scores
                lf = fd.extract_features(f.get('keypoints', {}))
                landmark_feats.append(lf)

                # Update scoring state with the its recent probability and landmark values
                scorer.update(probs, lf)

                # Update the LSTM input buffers with the recent emotion and landmark features
                emotion_seq.append(probs)
                landmark_seq.append(lf)
                burnout_seq.append(scorer.compute_score())
                
                # Ensure a fixed length time window with 'seq_len' by discarding oldest frames
                if len(emotion_seq) > lstm.seq_len:
                    emotion_seq.pop(0); landmark_seq.pop(0); burnout_seq.pop(0)

        # LSTM prediction of the subsequent burnout trend
        predicted = None
        if len(emotion_seq) >= lstm.seq_len:
            predicted = lstm.predict_next(emotion_seq, landmark_seq)

        # Render the real time overlays with boxes, labels, graphs and scores
        annotated = dashboard.update(frame, faces, emotions, scorer.compute_score(), landmark_feats)

        # Declare an overlay for every predicted subsequent value with cv2
        if predicted is not None:
            cv2.putText(annotated, f"Predicted Next Burnout: {predicted:.1f}%", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

        # Display the live window with cv2
        cv2.imshow("Single-User Burnout System", annotated)

        # Write the annotated frames to the video session
        if record_video:
            out.write(annotated)
        
        # Include an exit session with a 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup the camera recordings and its corresponding windows 
    cap.release()
    if record_video:
        out.release()
    cv2.destroyAllWindows()

    # Post session clean up (collect survey, persist session, and run analysis)
    s_entry = survey.collect_survey()
    survey.save_survey(s_entry)
    session.save_session(dashboard)
    compute_correlation(user_id)
    analyzer = LongitudinalAnalyzer()
    analyzer.plot_longitudinal_trend(user_id, freq = 'D')
    print("[Main] Single-user session completed.")

# =============================================================================================
# DEMO TESTING
# =============================================================================================
# Declare a demo test by running a default single-user session
if __name__ == "__main__":
    run_single_user(user_id = "user001", record_video = False)

### ===================================================================================================================================
## END: Add implementations if necessary
### ===================================================================================================================================