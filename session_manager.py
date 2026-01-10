### ========================================================================================================================================
## Name:        Saputra Rizky Johan (사푸트라)
## Institution: Seoul National University, South Korea
## Purpose:     Undergraduate Thesis
## Department:  Computer Science and Engineering
## College:     Engineering
## Advisor:     Prof. Lee Changun (이창건 교수)
## Title:       The Application of Facial Emotion Recognition (FER) in the Detection and Measurement of Burnout and Prolonged Stress Levels
## Module:      session_manager.py (See specifications and thesis section 4.1. functional modules for further details)
### ========================================================================================================================================

# =============================================================================================
# SPECIFICATIONS
# =============================================================================================
"""
session_manager.py
Manages per-user sessions and writes CSV metrics and session logs.
"""

# =============================================================================================
# SETUP
# =============================================================================================
import os
from datetime import datetime
import pandas as pd
from typing import Optional
from dashboard import BurnoutDashboard

# =============================================================================================
# IMPLEMENTATION
# =============================================================================================
#
class SessionManager:
    #
    def __init__(self, user_id: str, data_folder: str = "data_sessions"):
        self.user_id = user_id
        self.data_folder = data_folder
        os.makedirs(self.data_folder, exist_ok=True)
        self.session_start = datetime.now()
        # filename timestamped
        stamp = self.session_start.strftime("%Y-%m-%d_%H-%M-%S")
        self.session_filename = os.path.join(self.data_folder, f"{self.user_id}_{stamp}_session.csv")
        self.metrics_filename = os.path.join(self.data_folder, f"{self.user_id}_{stamp}_metrics.csv")

    #
    def save_session(self, dashboard: BurnoutDashboard) -> Optional[str]:
        """
        Save dashboard.session_data and produce metrics (max/min/avg burnout)
        """
        # Save flattened session CSV via dashboard
        dashboard.save_session_csv(self.session_filename)

        # Compute metrics from saved CSV
        try:
            df = pd.read_csv(self.session_filename)
            if 'Burnout' in df.columns:
                max_b = df['Burnout'].max()
                min_b = df['Burnout'].min()
                avg_b = df['Burnout'].mean()
            else:
                max_b = min_b = avg_b = 0.0
            metrics = pd.DataFrame([{
                'User': self.user_id,
                'SessionStart': self.session_start.isoformat(),
                'MaxBurnout': float(max_b),
                'MinBurnout': float(min_b),
                'AvgBurnout': float(avg_b),
                'NumRecords': len(df)
            }])
            metrics.to_csv(self.metrics_filename, index=False)
            print(f"[SessionManager] Metrics saved to: {self.metrics_filename}")
            return self.session_filename
        except Exception as e:
            print(f"[SessionManager] Error saving session metrics: {e}")
            return None
        
### ===================================================================================================================================
## END: Add implementations if necessary
### ===================================================================================================================================