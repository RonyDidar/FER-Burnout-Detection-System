### ========================================================================================================================================
## Name:        Saputra Rizky Johan (사푸트라)
## Institution: Seoul National University, South Korea
## Purpose:     Undergraduate Thesis
## Department:  Computer Science and Engineering
## College:     Engineering
## Advisor:     Prof. Lee Changun (이창건 교수)
## Title:       The Application of Facial Emotion Recognition (FER) in the Detection and Measurement of Burnout and Prolonged Stress Levels
## Module:      lstm_burnout_predictor.py (See specifications and thesis section 4.1. functional modules for further details)
### ========================================================================================================================================

# =============================================================================================
# SPECIFICATIONS
# =============================================================================================
# LSTM model for temporal burnout prediction
"""
lstm_burnout_predictor.py
A simple LSTM predictor for next burnout score based on sequences.
Note: training is optional. If you do not load weights, predictions will be invalid.
"""

# =============================================================================================
# SETUP
# =============================================================================================
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
from typing import List

# =============================================================================================
# IMPLEMENTATION
# =============================================================================================

class LSTMBurnoutPredictor:
    def __init__(self, seq_len: int = 30, input_dim: int = 10):
        """
        seq_len: number of timesteps in sequence
        input_dim: features per timestep (7 emotion probs + N landmark features)
        """
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(LSTM(64, input_shape=(self.seq_len, self.input_dim), return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(32))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='linear'))  # predict next burnout scalar
        model.compile(optimizer=Adam(1e-3), loss='mse')
        return model

    def prepare_input(self, emotion_seq: List[np.ndarray], landmark_seq: List[List[float]]):
        """
        Combine emotion_prob seq (list of arrays shape (7,)) and landmark_seq list.
        Returns array shaped (1, seq_len, input_dim)
        """
        # combine per timestep
        seq = []
        for e, l in zip(emotion_seq, landmark_seq):
            e_arr = np.array(e).reshape(-1)  # length 7
            l_arr = np.array(l).reshape(-1)  # length input_dim -7
            combined = np.concatenate([e_arr, l_arr])
            seq.append(combined)
        seq = np.array(seq)
        # pad if shorter than seq_len
        if seq.shape[0] < self.seq_len:
            pad = np.zeros((self.seq_len - seq.shape[0], seq.shape[1]))
            seq = np.vstack([pad, seq])
        seq = seq.reshape(1, self.seq_len, self.input_dim)
        return seq

    def predict_next(self, emotion_seq: List[np.ndarray], landmark_seq: List[List[float]]):
        """
        Predict next burnout value. If model weights are random (untrained), this output is unreliable.
        """
        if len(emotion_seq) == 0 or len(landmark_seq) == 0:
            return None
        X = self.prepare_input(emotion_seq[-self.seq_len:], landmark_seq[-self.seq_len:])
        pred = self.model.predict(X, verbose=0)
        return float(pred.flatten()[0])

    def train(self, X_train, y_train, epochs=20, batch_size=8, validation_data=None):
        """
        Train the LSTM model. X_train shape should be (n_samples, seq_len, input_dim)
        """
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=validation_data)

### ===================================================================================================================================
## END: Add implementations if necessary
### ===================================================================================================================================