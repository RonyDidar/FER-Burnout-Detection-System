### ========================================================================================================================================
## Name:        Saputra Rizky Johan (사푸트라)
## Institution: Seoul National University, South Korea
## Purpose:     Undergraduate Thesis
## Department:  Computer Science and Engineering
## College:     Engineering
## Advisor:     Prof. Lee Changun (이창건 교수)
## Title:       The Application of Facial Emotion Recognition (FER) in the Detection and Measurement of Burnout and Prolonged Stress Levels
## Module:      survey_integration.py (See specifications and thesis section 4.1. functional modules for further details)
### ========================================================================================================================================

# =============================================================================================
# SPECIFICATIONS
# =============================================================================================
"""
survey_integration.py
Collects simple self-report surveys from users (terminal input) and saves them to CSV.
"""

# =============================================================================================
# SETUP
# =============================================================================================
import os
import pandas as pd
from datetime import datetime
from typing import Dict

# =============================================================================================
# IMPLEMENTATION
# =============================================================================================
class SurveyCollector:
    def __init__(self, user_id: str, data_folder: str = "data_sessions"):
        self.user_id = user_id
        self.data_folder = data_folder
        os.makedirs(self.data_folder, exist_ok=True)
        self.survey_filename = os.path.join(self.data_folder, f"{self.user_id}_survey.csv")

    def collect_survey(self) -> Dict:
        """
        Prompt the user in terminal to input short survey responses.
        Returns dict containing answers + timestamp.
        """
        print(f"\nSurvey for user {self.user_id}")
        try:
            stress = int(input("Rate your current stress (1-10): ").strip())
        except Exception:
            stress = 5
        try:
            fatigue = int(input("Rate your fatigue (1-10): ").strip())
        except Exception:
            fatigue = 5
        try:
            motivation = int(input("Rate your motivation (1-10): ").strip())
        except Exception:
            motivation = 5
        entry = {
            'User': self.user_id,
            'Timestamp': datetime.now().isoformat(),
            'Stress': int(max(1, min(10, stress))),
            'Fatigue': int(max(1, min(10, fatigue))),
            'Motivation': int(max(1, min(10, motivation)))
        }
        return entry

    def save_survey(self, entry: Dict):
        """
        Append or create survey CSV for this user.
        """
        df = pd.DataFrame([entry])
        if os.path.exists(self.survey_filename):
            df.to_csv(self.survey_filename, mode='a', header=False, index=False)
        else:
            df.to_csv(self.survey_filename, index=False)
        print(f"[SurveyCollector] Survey saved to: {self.survey_filename}")

### ===================================================================================================================================
## END: Add implementations if necessary
### ===================================================================================================================================