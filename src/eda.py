"""Exploratory data analysis - answers the four EDA questions from the spec."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .utils import PROCESSED_PATH, FIG_DIR, REPORT_DIR, RISK_LABELS, ensure_dirs

PALETTE = {"Low": "#2e9e5b", "Medium": "#f0a30a", "High": "#d64545"}


def run(df: pd.DataFrame = None):
    ensure_dirs()
    sns.set_theme(style="whitegrid")
    df = pd.read_csv(PROCESSED_PATH) if df is None else df
    out = {}

    # risk distribution
    fig, ax = plt.subplots(figsize=(5.5, 4))
    counts = df["Risk_Level"].value_counts().reindex(RISK_LABELS)
    ax.bar(counts.index, counts.values, color=[PALETTE[k] for k in counts.index])
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v} ({v / len(df):.0%})", ha="center", va="bottom")
    ax.set_title("Risk level distribution"); ax.set_ylabel("Students")
    fig.tight_layout(); fig.savefig(FIG_DIR / "eda_risk_distribution.png", dpi=140); plt.close(fig)

    # Q1 attendance -> final score
    s = df.sample(1500, random_state=1)
    fig, ax = plt.subplots(figsize=(6, 4.2))
    sns.scatterplot(data=s, x="Attendance_Percentage", y="Final_Score", hue="Risk_Level",
                    hue_order=RISK_LABELS, palette=PALETTE, alpha=0.6, s=18, ax=ax)
    sns.regplot(data=s, x="Attendance_Percentage", y="Final_Score", scatter=False, color="black", ax=ax)
    ax.set_title("Q1: Attendance vs final score")
    fig.tight_layout(); fig.savefig(FIG_DIR / "eda_attendance_vs_final.png", dpi=140); plt.close(fig)
    out["corr_attendance_final"] = round(df["Attendance_Percentage"].corr(df["Final_Score"]), 3)

    # Q2 previous GPA -> final score
    fig, ax = plt.subplots(figsize=(6, 4.2))
    sns.scatterplot(data=s, x="Previous_GPA", y="Final_Score", hue="Risk_Level",
                    hue_order=RISK_LABELS, palette=PALETTE, alpha=0.6, s=18, ax=ax)
    sns.regplot(data=s, x="Previous_GPA", y="Final_Score", scatter=False, color="black", ax=ax)
    ax.set_title("Q2: Previous GPA vs final score")
    fig.tight_layout(); fig.savefig(FIG_DIR / "eda_gpa_vs_final.png", dpi=140); plt.close(fig)
    out["corr_gpa_final"] = round(df["Previous_GPA"].corr(df["Final_Score"]), 3)

    # Q3 missed assignments -> risk
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ct = pd.crosstab(df["Assignments_Missed"].clip(upper=6), df["Risk_Level"], normalize="index")[RISK_LABELS]
    ct.plot(kind="bar", stacked=True, color=[PALETTE[k] for k in RISK_LABELS], ax=ax, width=0.8)
    ax.set_xlabel("Assignments missed (6 = 6+)"); ax.set_ylabel("Share of students")
    ax.set_title("Q3: Missed assignments vs risk level"); ax.legend(title="Risk")
    fig.tight_layout(); fig.savefig(FIG_DIR / "eda_missed_assignments_vs_risk.png", dpi=140); plt.close(fig)
    out["high_risk_share_by_missed"] = (
        df.groupby(df["Assignments_Missed"].clip(upper=6))["Risk_Level"].apply(lambda x: round((x == "High").mean(), 3)).to_dict()
    )

    # correlation heatmap (predictors + outcome)
    num = df.select_dtypes("number").drop(columns=["Attendance_Status"], errors="ignore")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(num.corr(), cmap="RdBu_r", center=0, annot=False, ax=ax)
    ax.set_title("Correlation heatmap")
    fig.tight_layout(); fig.savefig(FIG_DIR / "eda_correlation_heatmap.png", dpi=140); plt.close(fig)

    df.describe().T.round(2).to_csv(REPORT_DIR / "summary_statistics.csv")
    print(f"[eda] figures saved to {FIG_DIR}; {out}")
    return out


if __name__ == "__main__":
    run()
