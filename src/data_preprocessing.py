"""Cleaning: duplicates, invalid values, missing values, outliers."""
import pandas as pd

from .utils import RAW_PATH, PROCESSED_PATH, BASE_NUMERIC, CATEGORICAL_FEATURES, ensure_dirs

VALID_RANGES = {
    "Age": (15, 40), "Semester": (1, 12), "Previous_GPA": (0, 10),
    "Attendance_Percentage": (0, 100), "Assignment_Avg": (0, 100), "Quiz_Avg": (0, 100),
    "Midterm_Score": (0, 100), "Practical_Score": (0, 100), "Study_Hours": (0, 12),
    "Classes_Missed": (0, 200), "Assignments_Missed": (0, 50), "Late_Submissions": (0, 50),
    "Participation_Score": (0, 100), "Previous_Backlogs": (0, 30), "Final_Score": (0, 100),
}


def clean(df: pd.DataFrame):
    """Return (clean_df, summary dict)."""
    summary = {"rows_in": len(df)}
    df = df.drop_duplicates().drop_duplicates(subset="Student_ID").copy()
    summary["duplicates_removed"] = summary["rows_in"] - len(df)

    df = df.dropna(subset=["Risk_Level"])          # cannot learn from unlabeled rows

    invalid = 0
    for col, (lo, hi) in VALID_RANGES.items():
        if col in df:
            bad = df[col].notna() & ~df[col].between(lo, hi)
            invalid += int(bad.sum())
            df.loc[bad, col] = float("nan")        # treat impossible values as missing
    summary["invalid_values_nulled"] = invalid

    summary["missing_before_imputation"] = int(df[BASE_NUMERIC + CATEGORICAL_FEATURES].isna().sum().sum())
    for col in BASE_NUMERIC:                       # median is robust to skew/outliers
        df[col] = df[col].fillna(df[col].median())
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].fillna(df[col].mode().iloc[0])

    int_cols = ["Age", "Semester", "Classes_Missed", "Assignments_Missed", "Late_Submissions", "Previous_Backlogs"]
    df[int_cols] = df[int_cols].round().astype(int)

    # IQR winsorising of remaining extreme (but valid) values on continuous columns
    capped = 0
    for col in ["Study_Hours", "Previous_GPA", "Participation_Score"]:
        q1, q3 = df[col].quantile([0.25, 0.75])
        lo, hi = q1 - 3 * (q3 - q1), q3 + 3 * (q3 - q1)
        capped += int(((df[col] < lo) | (df[col] > hi)).sum())
        df[col] = df[col].clip(lo, hi)
    summary["outliers_capped"] = capped
    summary["rows_out"] = len(df)
    return df.reset_index(drop=True), summary


def main():
    from .feature_engineering import add_features
    ensure_dirs()
    raw = pd.read_csv(RAW_PATH)
    df, summary = clean(raw)
    df = add_features(df)
    df.to_csv(PROCESSED_PATH, index=False)
    print(f"[preprocessing] {summary}")
    print(f"[preprocessing] wrote {PROCESSED_PATH}")
    return df, summary


if __name__ == "__main__":
    main()
