### ========================================================================================================================================
## Name:        Saputra Rizky Johan (사푸트라)
## Institution: Seoul National University, South Korea
## Purpose:     Undergraduate Thesis
## Department:  Computer Science and Engineering
## College:     Engineering
## Advisor:     Prof. Lee Changun (이창건 교수)
## Title:       The Application of Facial Emotion Recognition (FER) in the Detection and Measurement of Burnout and Prolonged Stress Levels
## Module:      analysis.py (See specifications and thesis section 4.1. functional modules for further details)
### ========================================================================================================================================

# =============================================================================================
# SPECIFICATIONS
# =============================================================================================
# Correlation analysis between FER burnout and survey results
"""
analysis.py
Compute correlation metrics between per-session average burnout and survey responses.
"""

# =============================================================================================
# SETUP
# =============================================================================================
import glob
import os
import pandas as pd
from scipy.stats import pearsonr, spearmanr

# =============================================================================================
# IMPLEMENTATION
# =============================================================================================

def compute_correlation(user_id: str, data_folder: str = "data_sessions"):
    """
    Compute correlation between session-level burnout averages and survey "Stress" entries.
    This pairs session files in chronological order with survey rows in chronological order.
    """
    # find session metric files for the user (we saved metrics per session)
    pattern = os.path.join(data_folder, f"{user_id}_*_metrics.csv")
    metric_files = sorted(glob.glob(pattern))
    if not metric_files:
        print(f"[Analysis] No metric files found for user {user_id}.")
        return None

    session_metrics = []
    for mf in metric_files:
        try:
            dfm = pd.read_csv(mf)
            # Expect one-row metrics file
            session_metrics.append(float(dfm['AvgBurnout'].values[0]))
        except Exception:
            continue

    survey_file = os.path.join(data_folder, f"{user_id}_survey.csv")
    if not os.path.exists(survey_file):
        print(f"[Analysis] No survey file found for user {user_id}.")
        return None

    survey_df = pd.read_csv(survey_file)
    # Use 'Stress' column from survey as ground truth measure
    survey_scores = survey_df['Stress'].astype(float).tolist()

    # Pair entries in order. If lengths differ, use min length.
    n = min(len(session_metrics), len(survey_scores))
    if n == 0:
        print(f"[Analysis] Not enough data to compute correlation for {user_id}.")
        return None

    x = session_metrics[:n]
    y = survey_scores[:n]

    pearson_corr, pearson_p = pearsonr(x, y)
    spearman_corr, spearman_p = spearmanr(x, y)
    results = {
        'user': user_id,
        'pearson_corr': float(pearson_corr),
        'pearson_p': float(pearson_p),
        'spearman_corr': float(spearman_corr),
        'spearman_p': float(spearman_p),
        'n_pairs': n
    }
    print(f"[Analysis] User {user_id}: Pearson {pearson_corr:.3f} (p={pearson_p:.3f}), Spearman {spearman_corr:.3f} (p={spearman_p:.3f})")
    return results

### ===================================================================================================================================
## END: Add implementations if necessary
### ===================================================================================================================================