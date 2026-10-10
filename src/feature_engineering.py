"""Engineered features from the project spec."""
import numpy as np
import pandas as pd

from .utils import TOTAL_ASSIGNMENTS


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Assessment_Avg"] = (df["Assignment_Avg"] + df["Quiz_Avg"] + df["Midterm_Score"] + df["Practical_Score"]) / 4

    # 0 = Normal (>=80), 1 = Warning (65-79), 2 = Critical (<65)
    df["Attendance_Status"] = np.select(
        [df["Attendance_Percentage"] >= 80, df["Attendance_Percentage"] >= 65], [0, 1], default=2
    )

    submission_rate = 100 * (1 - df["Assignments_Missed"] / TOTAL_ASSIGNMENTS).clip(0, 1)
    study_score = 100 * (df["Study_Hours"] / 6).clip(0, 1)
    df["Engagement_Score"] = 0.4 * df["Participation_Score"] + 0.3 * submission_rate + 0.3 * study_score

    df["Academic_Performance_Index"] = (
        0.35 * (df["Previous_GPA"] * 10) + 0.40 * df["Assessment_Avg"] + 0.25 * df["Attendance_Percentage"]
    )
    for c in ["Assessment_Avg", "Engagement_Score", "Academic_Performance_Index"]:
        df[c] = df[c].round(2)
    return df
