"""Shared constants, paths and rule-based explanation / intervention logic."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dataset"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"

RAW_PATH = DATA_DIR / "students_raw.csv"
PROCESSED_PATH = DATA_DIR / "students_processed.csv"
SCORED_PATH = DATA_DIR / "students_scored.csv"
CLUSTERED_PATH = DATA_DIR / "students_clustered.csv"

RISK_LABELS = ["Low", "Medium", "High"]           # order == class ids 0, 1, 2
LABEL_TO_ID = {label: i for i, label in enumerate(RISK_LABELS)}
ID_TO_LABEL = {i: label for label, i in LABEL_TO_ID.items()}

TARGET = "Risk_Level"
OUTCOME_COL = "Final_Score"       # later outcome -> used ONLY to derive the target, never as a feature
TOTAL_ASSIGNMENTS = 10            # assignments per course used to derive submission rate

CATEGORICAL_FEATURES = ["Gender", "Department"]
BASE_NUMERIC = [
    "Age", "Semester", "Previous_GPA", "Attendance_Percentage", "Assignment_Avg",
    "Quiz_Avg", "Midterm_Score", "Practical_Score", "Study_Hours", "Classes_Missed",
    "Assignments_Missed", "Late_Submissions", "Participation_Score", "Previous_Backlogs",
]
ENGINEERED = ["Assessment_Avg", "Attendance_Status", "Engagement_Score", "Academic_Performance_Index"]
NUMERIC_FEATURES = BASE_NUMERIC + ENGINEERED
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

RISK_MESSAGES = {
    "High": "High Risk - Immediate Academic Support Recommended",
    "Medium": "Medium Risk - Monitor Student",
    "Low": "Low Risk - Normal Progress",
}

# Support suggestions - NOT judgements about a student's ability.
INTERVENTIONS = {
    "High": [
        "Faculty mentoring",
        "Weekly academic monitoring",
        "Additional classes / remedial sessions",
        "Assignment completion tracking",
        "Attendance monitoring",
    ],
    "Medium": [
        "Monitor attendance",
        "Encourage additional practice",
        "Faculty check-in",
        "Track upcoming assessments",
    ],
    "Low": [
        "Continue normal academic progress",
        "Encourage consistent performance",
    ],
}

# (feature, label, critical-test, warning-test, formatter)
_RULES = [
    ("Attendance_Percentage", "Low attendance", lambda v: v < 65, lambda v: v < 80, lambda v: f"{v:.0f}%"),
    ("Assessment_Avg", "Low assessment performance", lambda v: v < 50, lambda v: v < 70, lambda v: f"{v:.0f}%"),
    ("Previous_GPA", "Low previous GPA", lambda v: v < 5.5, lambda v: v < 7.0, lambda v: f"{v:.1f}"),
    ("Assignments_Missed", "Multiple missed assignments", lambda v: v >= 4, lambda v: v >= 2, lambda v: f"{v:.0f}"),
    ("Study_Hours", "Low daily study hours", lambda v: v < 2, lambda v: v < 3, lambda v: f"{v:.1f}/day"),
    ("Participation_Score", "Low class participation", lambda v: v < 45, lambda v: v < 60, lambda v: f"{v:.0f}%"),
    ("Previous_Backlogs", "Previous backlogs", lambda v: v >= 2, lambda v: v >= 1, lambda v: f"{v:.0f}"),
    ("Late_Submissions", "Frequent late submissions", lambda v: v >= 6, lambda v: v >= 3, lambda v: f"{v:.0f}"),
]


def explain_risk(row) -> list:
    """Rule-based list of warning signals for one student (dict / Series with engineered features)."""
    factors = []
    for feat, label, is_critical, is_warning, fmt in _RULES:
        if feat not in row:
            continue
        v = float(row[feat])
        if is_critical(v):
            factors.append({"factor": label, "value": fmt(v), "severity": "critical"})
        elif is_warning(v):
            factors.append({"factor": label, "value": fmt(v), "severity": "warning"})
    factors.sort(key=lambda f: f["severity"] != "critical")
    return factors


def ensure_dirs():
    for d in (DATA_DIR, MODEL_DIR, REPORT_DIR, FIG_DIR):
        d.mkdir(parents=True, exist_ok=True)
