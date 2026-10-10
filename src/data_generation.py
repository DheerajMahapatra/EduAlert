"""Synthetic student dataset generator (clearly SYNTHETIC - not real student records).

Design (avoids target leakage):
  * Predictors describe an EARLIER period: previous GPA, attendance so far, internal
    assessments, study behaviour, engagement.
  * The target Risk_Level is derived from a LATER outcome (Final_Score), which is never
    used as a model feature.
  * Latent 'ability' and 'motivation' drive both predictors and outcome, plus noise,
    so the problem is learnable but not trivially separable.
"""
import numpy as np
import pandas as pd

from .utils import RAW_PATH, ensure_dirs

DEPARTMENTS = ["Computer Science", "Electronics", "Mechanical", "Civil", "Electrical", "Management"]


def derive_risk(final_score: pd.Series) -> pd.Series:
    """High: final < 50 | Medium: 50-<65 | Low: >= 65."""
    return pd.Series(
        np.select([final_score < 50, final_score < 65], ["High", "Medium"], default="Low"),
        index=final_score.index,
    )


def generate(n: int = 6000, seed: int = 42, add_noise_issues: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    clip = np.clip

    ability = rng.normal(0, 1, n)
    motivation = 0.3 * ability + np.sqrt(1 - 0.09) * rng.normal(0, 1, n)

    semester = rng.integers(2, 9, n)
    age = 18 + (semester - 1) // 2 + rng.integers(0, 2, n)
    gender = rng.choice(["Male", "Female", "Other"], n, p=[0.52, 0.46, 0.02])
    dept = rng.choice(DEPARTMENTS, n, p=[0.28, 0.18, 0.16, 0.12, 0.12, 0.14])

    prev_gpa = clip(6.9 + 0.95 * ability + 0.35 * motivation + rng.normal(0, 0.45, n), 2.5, 10)
    attendance = clip(77 + 9.5 * motivation + 2.5 * ability + rng.normal(0, 6, n), 25, 100)
    assignment = clip(66 + 7 * motivation + 6 * ability + rng.normal(0, 7, n), 0, 100)
    quiz = clip(62 + 5 * motivation + 9 * ability + rng.normal(0, 8, n), 0, 100)
    midterm = clip(60 + 4 * motivation + 11 * ability + rng.normal(0, 8, n), 0, 100)
    practical = clip(68 + 6 * motivation + 6 * ability + rng.normal(0, 7, n), 0, 100)
    study = clip(3.0 + 0.9 * motivation + 0.25 * ability + rng.normal(0, 0.7, n), 0.2, 9)

    total_classes = rng.integers(55, 70, n)
    classes_missed = np.round(total_classes * (100 - attendance) / 100).astype(int)
    p_miss = clip(0.12 - 0.05 * motivation - 0.02 * ability, 0.01, 0.7)
    assignments_missed = rng.binomial(10, p_miss)
    p_late = clip(0.20 - 0.06 * motivation, 0.02, 0.8)
    late = rng.binomial(10, p_late)
    participation = clip(60 + 10 * motivation + 3 * ability + rng.normal(0, 9, n), 0, 100)
    backlogs = np.minimum(rng.poisson(np.exp(-1.1 - 0.7 * ability - 0.3 * motivation)), 6)

    # Later outcome (not a feature)
    final = clip(
        67.5 + 9 * ability + 6 * motivation + 0.12 * (attendance - 77) + 1.2 * (study - 3)
        - 0.8 * assignments_missed - 1.0 * backlogs + rng.normal(0, 6.5, n),
        0, 100,
    )

    df = pd.DataFrame({
        "Student_ID": [f"STU{i:05d}" for i in range(1, n + 1)],
        "Age": age, "Gender": gender, "Department": dept, "Semester": semester,
        "Previous_GPA": prev_gpa.round(2),
        "Attendance_Percentage": attendance.round(1),
        "Assignment_Avg": assignment.round(1),
        "Quiz_Avg": quiz.round(1),
        "Midterm_Score": midterm.round(1),
        "Practical_Score": practical.round(1),
        "Study_Hours": study.round(1),
        "Classes_Missed": classes_missed,
        "Assignments_Missed": assignments_missed,
        "Late_Submissions": late,
        "Participation_Score": participation.round(1),
        "Previous_Backlogs": backlogs,
        "Final_Score": final.round(1),
    })
    df["Risk_Level"] = derive_risk(df["Final_Score"])

    if add_noise_issues:  # realistic data-quality problems so preprocessing has real work to do
        for col in ["Attendance_Percentage", "Quiz_Avg", "Study_Hours", "Participation_Score", "Previous_GPA"]:
            df.loc[rng.choice(n, int(0.015 * n), replace=False), col] = np.nan
        df.loc[rng.choice(n, 40, replace=False), "Gender"] = np.nan
        df.loc[rng.choice(n, 8, replace=False), "Attendance_Percentage"] = 130.0   # impossible values
        df.loc[rng.choice(n, 8, replace=False), "Study_Hours"] = 20.0              # implausible outliers
        df = pd.concat([df, df.sample(40, random_state=seed)], ignore_index=True)  # duplicate rows
    return df


def main(n: int = 6000, seed: int = 42):
    ensure_dirs()
    df = generate(n, seed)
    df.to_csv(RAW_PATH, index=False)
    print(f"[data_generation] wrote {len(df)} rows (SYNTHETIC) -> {RAW_PATH}")
    print(df["Risk_Level"].value_counts(normalize=True).round(3).to_dict())
    return df


if __name__ == "__main__":
    main()
