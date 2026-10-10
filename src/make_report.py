"""Generate reports/model_report.pdf from saved metrics and figures."""
import json

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from .utils import FIG_DIR, MODEL_DIR, PROCESSED_PATH, REPORT_DIR, RISK_LABELS


def _table(df, col_widths=None):
    data = [list(df.columns)] + df.astype(str).values.tolist()
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3a68")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 7), ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f5fa")]),
    ]))
    return t


def _img(name, width=13 * cm):
    from PIL import Image as PILImage
    p = FIG_DIR / name
    w, h = PILImage.open(p).size
    return Image(str(p), width=width, height=width * h / w)


def main():
    ss = getSampleStyleSheet()
    df = pd.read_csv(PROCESSED_PATH)
    meta = json.loads((MODEL_DIR / "metadata.json").read_text())
    comp = pd.read_csv(REPORT_DIR / "model_comparison.csv")
    dist = df["Risk_Level"].value_counts().reindex(RISK_LABELS)
    fi = pd.read_csv(REPORT_DIR / "feature_importance.csv").head(6)

    S = []
    P = lambda t, s="BodyText": S.append(Paragraph(t, ss[s]))
    P("EduAlert: Early Identification of At-Risk Students Using Data Mining", "Title")
    P("Model report", "Heading2")
    P("1. Problem and approach", "Heading2")
    P("Struggling students are usually identified only after final results. EduAlert uses earlier-period "
      "academic, attendance, assessment and engagement data to flag students at Low / Medium / High risk "
      "early enough for support to help, and explains each flag with its main warning signals.")
    P("2. Data", "Heading2")
    P(f"<b>Synthetic, anonymised data</b> (not real student records): {len(df):,} students, "
      f"{len(meta['features'])} model features. Risk_Level is derived from a <i>later</i> outcome "
      f"(final score: High &lt;50, Medium 50-65, Low &ge;65); the final score itself is never a model feature, "
      f"which avoids target leakage. Class balance: "
      + ", ".join(f"{k} {v} ({v / len(df):.0%})" for k, v in dist.items()) + ".")
    S.append(_img("eda_risk_distribution.png", 8 * cm))
    P("3. Pipeline", "Heading2")
    P("Remove duplicates, null impossible values, median/mode imputation, outlier capping, feature engineering "
      "(Assessment_Avg, Attendance_Status, Engagement_Score, Academic_Performance_Index), one-hot encoding, "
      "standard scaling, stratified 80/20 split, Random Forest tuned by 3-fold grid search "
      f"(best: {meta['rf_best_params']}).")
    P("4. Results (held-out test set)", "Heading2")
    S.append(_table(comp))
    S.append(Spacer(1, 8))
    P("For an early-warning system the costly error is a truly high-risk student predicted as Low, so "
      "high-risk recall is reported alongside precision and F1. Note that simpler models can achieve a higher "
      "high-risk recall than the Random Forest; the decision threshold can be lowered to trade precision "
      "for recall depending on how much follow-up capacity the institution has.")
    S.append(_img("confusion_matrices.png", 12 * cm))
    S.append(PageBreak())
    P("5. Main drivers of risk", "Heading2")
    S.append(_table(fi))
    S.append(_img("feature_importance.png", 11 * cm))
    P("6. Data mining insights", "Heading2")
    S.append(_table(pd.read_csv(REPORT_DIR / "cluster_vs_risk.csv")))
    S.append(_img("cluster_pca.png", 10 * cm))
    P("7. Limitations and responsible use", "Heading2")
    P("Results come from synthetic data, so absolute scores do not transfer to a real institution; retrain on "
      "real, consented, anonymised data before any use. Predictions are probabilistic support signals, not "
      "judgements of a student's ability, and should be reviewed by faculty. Check for group-level bias "
      "(e.g. by gender or department) before deployment.")
    SimpleDocTemplate(str(REPORT_DIR / "model_report.pdf"), pagesize=A4, topMargin=1.5 * cm,
                      bottomMargin=1.5 * cm, leftMargin=1.8 * cm, rightMargin=1.8 * cm).build(S)
    print("[report] wrote reports/model_report.pdf")


if __name__ == "__main__":
    main()
