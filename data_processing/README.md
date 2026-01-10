# Burnout Detection from Facial Emotion and Landmarks

This project implements a pipeline for **burnout detection** using **facial emotion recognition**, **facial landmarks**, and a **sequence model (LSTM)**.  
It combines computer vision (MTCNN, OpenCV), deep learning (Keras/TensorFlow), and time-series modeling to classify **burnout risk** from webcam video streams.

---

## Theoretical Background

1. **Emotion Recognition (FER)**  
   - Detects 7 basic emotions: *Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise*.  
   - Uses a pre-trained deep CNN model (`fer2013_model.h5`, originally `Emo0.1.h5`).  
   - Input: cropped face (48×48, RGB).  
   - Output: 7-dimensional probability distribution over emotions.

2. **Facial Landmarks**  
   - Extracts geometric features such as **eye aspect ratio**, **mouth openness**, etc.  
   - Provides additional context about fatigue and stress.

3. **Temporal Modeling (LSTM)**  
   - Burnout develops gradually, so features are tracked over time.  
   - Each time step: `[emotion_probs (7) + landmark_features (3)] = 10 features`.  
   - Sequences of length **30 frames** are fed into an LSTM to predict burnout risk.

---

## Project Structure

- **`main.py`**  
  Entry point. Captures video frames, runs face detection, emotion recognition, feature extraction, and passes sequences to the LSTM burnout predictor.

- **`emotion_recognition.py`**  
  Loads the FER model (`fer2013_model.h5`) and predicts emotions from cropped faces.  
  Handles preprocessing (RGB vs grayscale, resizing, normalization).

- **`face_detection.py`**  
  Uses **MTCNN** to detect faces and extract bounding boxes + facial landmarks.  
  Provides safe cropping and feature extraction functions.

- **`burnout_prediction.py`**  
  Defines and loads the LSTM model.  
  Predicts burnout probability given a sequence of feature vectors.

- **`utils.py`** *(if present)*  
  Helper functions for data formatting, visualization, etc.

- **`fer2013_model.h5`**  
  Pretrained **Facial Emotion Recognition CNN** (based on FER2013 dataset).  
  Downloaded from [Hugging Face: `shivamprasad1001/Emo0.1`](https://huggingface.co/shivamprasad1001/Emo0.1).  
  Originally named `Emo0.1.h5`, renamed in this project for compatibility.

---

## Environment Setup

### Python Interpreter
The project uses a local virtual environment:

```bash
./burnout_env/bin/python main.py
```

Interpreter: Python 3.9.19

### Installing Dependencies
```bash
./burnout_env/bin/pip install -r requirements.txt
```

### Key Dependencies & Versions
- Python 3.9
- Key Dependencies & Versions
- OpenCV 4.9
- MTCNN 0.1.1
- NumPy 1.23
- urllib3 < 2.0 (due to LibreSSL warning)

## Running the Program