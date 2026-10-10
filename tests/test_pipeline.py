import pandas as pd
import pytest

from src.predict import RiskPredictor
from src.utils import ALL_FEATURES, OUTCOME_COL, TARGET, PROCESSED_PATH
from src.data_generation import derive_risk


def test_no_target_leakage():
    assert OUTCOME_COL not in ALL_FEATURES and TARGET not in ALL_FEATURES


def test_processed_data_is_clean():
    df = pd.read_csv(PROCESSED_PATH)
    assert df.isna().sum().sum() == 0
    assert df["Student_ID"].is_unique
    assert df["Attendance_Percentage"].between(0, 100).all()


def test_risk_derivation_thresholds():
    s = derive_risk(pd.Series([10, 49.9, 50, 64.9, 65, 99]))
    assert s.tolist() == ["High", "High", "Medium", "Medium", "Low", "Low"]


@pytest.fixture(scope="module")
def predictor():
    return RiskPredictor()


def test_probabilities_sum_to_one(predictor):
    r = predictor.predict_one({"Attendance_Percentage": 70, "Previous_GPA": 6.5, "Assignment_Avg": 60,
                               "Quiz_Avg": 60, "Midterm_Score": 60, "Study_Hours": 3, "Assignments_Missed": 2})
    assert abs(sum(r["probabilities"].values()) - 1) < 1e-6


def test_weak_and_strong_profiles(predictor):
    weak = predictor.predict_one({"Attendance_Percentage": 55, "Previous_GPA": 5.0, "Assignment_Avg": 40,
                                  "Quiz_Avg": 45, "Midterm_Score": 40, "Study_Hours": 1, "Assignments_Missed": 6})
    strong = predictor.predict_one({"Attendance_Percentage": 95, "Previous_GPA": 8.9, "Assignment_Avg": 90,
                                    "Quiz_Avg": 88, "Midterm_Score": 87, "Study_Hours": 5, "Assignments_Missed": 0})
    assert weak["risk"] == "High" and weak["factors"]
    assert strong["risk"] == "Low" and not strong["factors"]
