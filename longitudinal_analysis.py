### ========================================================================================================================================
## Name:        Saputra Rizky Johan (사푸트라)
## Institution: Seoul National University, South Korea
## Purpose:     Undergraduate Thesis
## Department:  Computer Science and Engineering
## College:     Engineering
## Advisor:     Prof. Lee Changun (이창건 교수)
## Title:       The Application of Facial Emotion Recognition (FER) in the Detection and Measurement of Burnout and Prolonged Stress Levels
## Module:      longitudinal_analysis.py (See specifications and thesis section 4.1. functional modules for further details)
### ========================================================================================================================================

# =============================================================================================
# SPECIFICATIONS
# =============================================================================================
# Longitudinal multi-session analysis
"""
longitudinal_analysis.py
Aggregate multiple session CSV files for a given user and plot resampled trends (daily/weekly/monthly).
"""

# =============================================================================================
# SETUP
# =============================================================================================
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
from datetime import datetime

# =============================================================================================
# IMPLEMENTATION
# =============================================================================================

class LongitudinalAnalyzer:
    def __init__(self, data_folder: str = "data_sessions"):
        self.data_folder = data_folder

    def aggregate_user_sessions(self, user_id: str) -> pd.DataFrame:
        """
        Combine all session CSV files for user into a single dataframe.
        Session CSV filenames expected to be user_{timestamp}_session.csv
        """
        pattern = os.path.join(self.data_folder, f"{user_id}_*_session.csv")
        files = sorted(glob.glob(pattern))
        dfs = []
        for f in files:
            try:
                df = pd.read_csv(f)
                # ensure timestamp column exists and is parsed
                if 'Timestamp' in df.columns:
                    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
                else:
                    # fall back to file modified time
                    mt = datetime.fromtimestamp(os.path.getmtime(f))
                    df['Timestamp'] = mt
                df['SessionFile'] = os.path.basename(f)
                dfs.append(df)
            except Exception as e:
                print(f"[Longitudinal] Failed reading {f}: {e}")
        if not dfs:
            print(f"[Longitudinal] No session CSVs found for {user_id}.")
            return pd.DataFrame()
        big = pd.concat(dfs, ignore_index=True)
        return big

    def plot_longitudinal_trend(self, user_id: str, freq: str = 'D'):
        """
        Plot average burnout over time aggregated by freq:
        'D' = daily, 'W' = weekly, 'M' = monthly
        """
        df = self.aggregate_user_sessions(user_id)
        if df.empty:
            return
        df = df.dropna(subset=['Timestamp'])
        df.set_index('Timestamp', inplace=True)
        # resample and compute mean per period
        trend = df['Burnout'].resample(freq).mean()
        plt.figure(figsize=(10,4))
        trend.plot(marker='o', linestyle='-')
        plt.title(f'{user_id} - Average Burnout (freq={freq})')
        plt.xlabel('Time')
        plt.ylabel('Burnout (%)')
        plt.grid(True)
        plt.show()

### ===================================================================================================================================
## END: Add implementations if necessary
### ===================================================================================================================================