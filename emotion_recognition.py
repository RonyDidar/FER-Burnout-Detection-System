### ========================================================================================================================================
## Name:        Saputra Rizky Johan (사푸트라)
## Institution: Seoul National University, South Korea
## Purpose:     Undergraduate Thesis
## Department:  Computer Science and Engineering
## College:     Engineering
## Advisor:     Prof. Lee Changun (이창건 교수)
## Title:       The Application of Facial Emotion Recognition (FER) in the Detection and Measurement of Burnout and Prolonged Stress Levels
## Module:      emotion_recognition.py (See specifications and thesis section 4.1. functional modules for further details)
### ========================================================================================================================================

# =============================================================================================
# SPECIFICATIONS
# =============================================================================================
# emotion_recognition.py

# =============================================================================================
# SETUP
# =============================================================================================
import cv2
import numpy as np
from keras.models import load_model
from typing import Tuple, List

# =============================================================================================
# IMPLEMENTATION
# =============================================================================================

def _safe_rgb(img: np.ndarray) -> np.ndarray:
    # Ensure 3 channels RGB no matter what comes in
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

class EmotionRecognizer:
    def __init__(self, model_path: str = "emo0.1.h5"):
        self.model = load_model(model_path)
        # Typical FER-2013 order (your weights may differ in ordering but class count should be 7)
        self.labels: List[str] = ['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']

        # Infer input spec
        in_shape = self.model.input_shape  # e.g. (None, 48, 48, 1) or (None, 48, 48, 3) or (None, 1, 48, 48)
        if isinstance(in_shape, list):  # some models report list for multi-input; assume first
            in_shape = in_shape[0]
        self.channels_first = False
        if len(in_shape) != 4:
            raise ValueError(f"Unexpected model.input_shape: {in_shape}")

        # Parse dims
        _, d1, d2, d3 = in_shape  # default assumption channels_last
        if d1 in (1,3):  # this implies channels_first (C, H, W)
            self.channels_first = True
            self.in_c, self.in_h, self.in_w = d1, d2, d3
        else:
            # channels_last (H, W, C)
            self.in_h, self.in_w, self.in_c = d1, d2, d3

    def _preprocess(self, face_img: np.ndarray) -> np.ndarray:
        # make sure crop is valid
        if face_img is None or face_img.size == 0:
            # create a dummy 1xHxWxC to avoid hard crash; model will output garbage for this frame
            c = self.in_c
            if c == 1:
                dummy = np.zeros((self.in_h, self.in_w, 1), dtype=np.float32)
            else:
                dummy = np.zeros((self.in_h, self.in_w, c), dtype=np.float32)
            return np.expand_dims(dummy, 0)

        # Prepare channels
        if self.in_c == 1:
            # model expects grayscale
            try:
                gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            except Exception:
                gray = face_img if face_img.ndim == 2 else cv2.cvtColor(face_img, cv2.COLOR_RGB2GRAY)
            resized = cv2.resize(gray, (self.in_w, self.in_h), interpolation=cv2.INTER_AREA)
            x = resized.astype('float32') / 255.0
            if self.channels_first:
                x = x[np.newaxis, np.newaxis, :, :]  # (1,1,H,W)
            else:
                x = x[:, :, np.newaxis]              # (H,W,1)
                x = np.expand_dims(x, axis=0)        # (1,H,W,1)
        else:
            # model expects RGB 3-channel
            rgb = _safe_rgb(face_img)
            resized = cv2.resize(rgb, (self.in_w, self.in_h), interpolation=cv2.INTER_AREA)
            x = resized.astype('float32') / 255.0
            if self.channels_first:
                x = np.transpose(x, (2,0,1))        # (C,H,W)
                x = np.expand_dims(x, axis=0)       # (1,C,H,W)
            else:
                x = np.expand_dims(x, axis=0)       # (1,H,W,C)
        return x

    def predict_emotion(self, face_img: np.ndarray) -> Tuple[str, np.ndarray]:
        x = self._preprocess(face_img)
        preds = self.model.predict(x, verbose=0)
        preds = np.array(preds).reshape(-1)  # flatten to (num_classes,)
        # Safety: if model has unexpected #classes, just map best we can
        if len(preds) != len(self.labels):
            # pad or trim to 7 so downstream stays consistent
            if len(preds) < len(self.labels):
                preds = np.pad(preds, (0, len(self.labels)-len(preds)), mode='constant')
            else:
                preds = preds[:len(self.labels)]
        label = self.labels[int(np.argmax(preds))]
        return label, preds

### ===================================================================================================================================
## END: Add implementations if necessary
### ===================================================================================================================================