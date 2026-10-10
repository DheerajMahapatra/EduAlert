"""Association-rule mining (Apriori): which warning-sign combinations co-occur with High risk?"""
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

from .utils import PROCESSED_PATH, REPORT_DIR, ensure_dirs


def binarise(df: pd.DataFrame) -> pd.DataFrame:
    items = pd.DataFrame({
        "Low attendance (<65%)": df["Attendance_Percentage"] < 65,
        "Average attendance (65-80%)": df["Attendance_Percentage"].between(65, 80, inclusive="left"),
        "Low assessment (<50)": df["Assessment_Avg"] < 50,
        "Low previous GPA (<5.5)": df["Previous_GPA"] < 5.5,
        "Missed >=4 assignments": df["Assignments_Missed"] >= 4,
        "Low study hours (<2h)": df["Study_Hours"] < 2,
        "Has backlogs": df["Previous_Backlogs"] >= 1,
        "Low participation (<45)": df["Participation_Score"] < 45,
        "Risk = High": df["Risk_Level"] == "High",
        "Risk = Low": df["Risk_Level"] == "Low",
    })
    return items.astype(bool)


def main(min_support: float = 0.03, min_conf: float = 0.5):
    ensure_dirs()
    df = pd.read_csv(PROCESSED_PATH)
    freq = apriori(binarise(df), min_support=min_support, use_colnames=True)
    rules = association_rules(freq, metric="confidence", min_threshold=min_conf)
    rules = rules[rules["consequents"].apply(lambda s: s == frozenset({"Risk = High"}))]
    rules = rules[(rules["lift"] > 1.2) & (rules["antecedents"].apply(len) <= 3)]   # keep rules readable
    rules = rules.sort_values(["lift", "support"], ascending=False)
    rules["antecedents"] = rules["antecedents"].apply(lambda s: " + ".join(sorted(s)))
    rules["consequents"] = rules["consequents"].apply(lambda s: " + ".join(sorted(s)))
    keep = rules[["antecedents", "consequents", "support", "confidence", "lift"]].round(3).head(25)
    keep.to_csv(REPORT_DIR / "association_rules.csv", index=False)
    print(f"[association] {len(rules)} rules -> reports/association_rules.csv")
    print(keep.head(8).to_string(index=False))
    return keep


if __name__ == "__main__":
    main()
