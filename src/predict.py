"""Load the trained model and score students (single record or DataFrame)."""
import joblib
import pandas as pd

from .feature_engineering import add_features
from .utils import (ALL_FEATURES, ID_TO_LABEL, INTERVENTIONS, MODEL_DIR, RISK_LABELS, RISK_MESSAGES,
                    explain_risk)

DEFAULTS = {"Age": 20, "Semester": 4, "Gender": "Male", "Department": "Computer Science",
            "Classes_Missed": None, "Late_Submissions": 1, "Previous_Backlogs": 0,
            "Practical_Score": None, "Participation_Score": 60}


class RiskPredictor:
    def __init__(self):
        self.pre = joblib.load(MODEL_DIR / "preprocessor.pkl")
        self.model = joblib.load(MODEL_DIR / "risk_model.pkl")

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col, default in DEFAULTS.items():
            if col not in df:
                if col == "Classes_Missed":
                    df[col] = ((100 - df["Attendance_Percentage"]) * 0.62).round()
                elif col == "Practical_Score":
                    df[col] = df["Assignment_Avg"]
                else:
                    df[col] = default
        return add_features(df)

    def predict_df(self, df: pd.DataFrame) -> pd.DataFrame:
        prep = self._prepare(df)
        proba = self.model.predict_proba(self.pre.transform(prep[ALL_FEATURES]))
        out = prep.copy()
        for i, label in enumerate(RISK_LABELS):
            out[f"Prob_{label}"] = proba[:, i]
        out["Predicted_Risk"] = [ID_TO_LABEL[i] for i in proba.argmax(1)]
        return out

    def predict_one(self, record: dict) -> dict:
        row = self.predict_df(pd.DataFrame([record])).iloc[0]
        label = row["Predicted_Risk"]
        return {
            "risk": label,
            "message": RISK_MESSAGES[label],
            "probabilities": {l: float(row[f"Prob_{l}"]) for l in RISK_LABELS},
            "factors": explain_risk(row),
            "interventions": INTERVENTIONS[label],
        }


if __name__ == "__main__":
    demo = {"Attendance_Percentage": 58, "Previous_GPA": 5.4, "Assignment_Avg": 46, "Quiz_Avg": 51,
            "Midterm_Score": 43, "Study_Hours": 1.5, "Assignments_Missed": 4, "Participation_Score": 42}
    import json
    print(json.dumps(RiskPredictor().predict_one(demo), indent=2))
