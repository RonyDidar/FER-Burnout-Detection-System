### ========================================================================================================================================
## Name:        Saputra Rizky Johan (사푸트라)
## Institution: Seoul National University, South Korea
## Purpose:     Undergraduate Thesis
## Department:  Computer Science and Engineering
## College:     Engineering
## Advisor:     Prof. Lee Changun (이창건 교수)
## Title:       The Application of Facial Emotion Recognition (FER) in the Detection and Measurement of Burnout and Prolonged Stress Levels
## Module:      statistical_analytics.py (Additional module for overall data statistical analysis and thesis documentation purposes)
### ========================================================================================================================================

# =============================================================================================
# SETUP
# =============================================================================================
import os
import pandas as pd
from scipy.stats import pearsonr

# Directory containing your session metrics CSVs
DATA_DIR = "data_sessions"

# =============================================================================================
# IMPLEMENTATION
# =============================================================================================
# List to collect all participant stats
session_stats = []

for file in os.listdir(DATA_DIR):
    # Provess the frames' CSVs
    if file.endswith(".csv") and "frames" in file:
        file_path = os.path.join(DATA_DIR, file)
        df = pd.read_csv(file_path)

        # Ensure the Burnout column exists
        if "Burnout" in df.columns:
            participant_id = file.split("_")[0]  # e.g., "user001"
            
            avg_burnout = df["Burnout"].mean()
            std_burnout = df["Burnout"].std()
            min_burnout = df["Burnout"].min()
            max_burnout = df["Burnout"].max()

            # Normalize the values with a 0–1 scale for scientific notation
            avg_norm = avg_burnout / 100
            std_norm = std_burnout / 100
            min_norm = min_burnout / 100
            max_norm = max_burnout / 100

            # Append all the results for readability
            session_stats.append({
                "Participant": participant_id,
                "AvgSystem": round(avg_norm, 3),
                "Min": round(min_norm, 2),
                "Max": round(max_norm, 2),
                "StdDev": round(std_norm, 3)
            })

# Make dataframe of all results
summary_df = pd.DataFrame(session_stats)

# Sort by participant for clarity (Optional)
summary_df = summary_df.sort_values("Participant")

# Display the statistical results
print("• Summary Results:")
print(summary_df)

# Display the Pearson's Correlation Coefficient value (Adjust values if necessary)
system_scores = [0.400, 0.372, 0.717, 0.714, 0.706]
survey_scores = [0.600, 0.600, 0.733, 0.867, 0.867]
r, p = pearsonr(system_scores, survey_scores)
print("• r = ", r)

### ===================================================================================================================================
## END: Add implementations if necessary
### ===================================================================================================================================