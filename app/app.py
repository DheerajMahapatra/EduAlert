"""EduAlert - Student Early Warning System (Streamlit).  Run:  streamlit run app/app.py"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.express as px
import streamlit as st

from src.predict import RiskPredictor
from src.utils import (CLUSTERED_PATH, FIG_DIR, INTERVENTIONS, MODEL_DIR, REPORT_DIR, RISK_LABELS,
                       SCORED_PATH)

COLORS = {"Low": "#2e9e5b", "Medium": "#f0a30a", "High": "#d64545"}
ICON = {"Low": "🟢", "Medium": "🟠", "High": "🔴"}

st.set_page_config(page_title="EduAlert", page_icon="🎓", layout="wide")


@st.cache_resource
def get_predictor():
    return RiskPredictor()


@st.cache_data
def load_scored():
    return pd.read_csv(SCORED_PATH)


def artefacts_ready() -> bool:
    return (MODEL_DIR / "risk_model.pkl").exists() and SCORED_PATH.exists()


st.sidebar.title("🎓 EduAlert")
st.sidebar.caption("Early Identification of At-Risk Students Using Data Mining")
page = st.sidebar.radio("Navigate", ["Dashboard", "Student Prediction", "Batch Prediction",
                                     "Data Mining Insights", "Model Performance"])
st.sidebar.info("Demo uses **synthetic** student data. Recommendations are support suggestions, "
                "not judgements about a student's ability.")

if not artefacts_ready():
    st.error("Model artefacts not found. Run `python run_pipeline.py` first.")
    st.stop()

# ------------------------------------------------------------------ Dashboard
if page == "Dashboard":
    st.title("STUDENT EARLY WARNING SYSTEM")
    df = load_scored()
    c1, c2, c3 = st.columns(3)
    dept = c1.multiselect("Department", sorted(df["Department"].unique()))
    sem = c2.multiselect("Semester", sorted(df["Semester"].unique()))
    mask = pd.Series(True, index=df.index)
    if dept:
        mask &= df["Department"].isin(dept)
    if sem:
        mask &= df["Semester"].isin(sem)
    view = df[mask]

    counts = view["Predicted_Risk"].value_counts().reindex(RISK_LABELS).fillna(0).astype(int)
    m = st.columns(4)
    m[0].metric("Total students", f"{len(view):,}")
    for col, lab in zip(m[1:], RISK_LABELS):
        col.metric(f"{ICON[lab]} {lab} risk", f"{counts[lab]:,}", f"{counts[lab] / max(len(view), 1):.0%}", delta_color="off")

    a, b = st.columns(2)
    a.plotly_chart(px.pie(values=counts.values, names=counts.index, color=counts.index, hole=0.45,
                          color_discrete_map=COLORS, title="Risk distribution"), width="stretch")
    b.plotly_chart(px.scatter(view.sample(min(1500, len(view)), random_state=1), x="Attendance_Percentage",
                              y="Assessment_Avg", color="Predicted_Risk", color_discrete_map=COLORS,
                              category_orders={"Predicted_Risk": RISK_LABELS}, opacity=0.6,
                              title="Attendance vs assessment average"), width="stretch")

    st.subheader("Students requiring attention")
    attn = view[view["Predicted_Risk"].isin(["High", "Medium"])].sort_values("Prob_High", ascending=False)
    st.dataframe(
        attn[["Student_ID", "Department", "Semester", "Attendance_Percentage", "Assessment_Avg",
              "Previous_GPA", "Assignments_Missed", "Predicted_Risk", "Prob_High"]].head(100)
        .rename(columns={"Attendance_Percentage": "Attendance %", "Assessment_Avg": "Assessment avg",
                         "Prob_High": "P(High)"}),
        width="stretch", hide_index=True)
    st.caption("Risk here is an out-of-fold model prediction for each student (not in-sample).")

# ------------------------------------------------------------------ Single prediction
elif page == "Student Prediction":
    st.title("STUDENT RISK PREDICTION")
    with st.form("student"):
        c1, c2 = st.columns(2)
        sid = c1.text_input("Student ID", "STU102")
        att = c1.slider("Attendance %", 0, 100, 58)
        gpa = c1.slider("Previous GPA", 0.0, 10.0, 5.4, 0.1)
        asg = c1.slider("Assignment average %", 0, 100, 46)
        quiz = c1.slider("Quiz average %", 0, 100, 51)
        mid = c2.slider("Midterm score %", 0, 100, 43)
        prac = c2.slider("Practical score %", 0, 100, 50)
        study = c2.slider("Study hours / day", 0.0, 9.0, 1.5, 0.5)
        missed = c2.slider("Assignments missed (of 10)", 0, 10, 4)
        part = c2.slider("Participation %", 0, 100, 42)
        backlogs = c2.number_input("Previous backlogs", 0, 10, 0)
        go = st.form_submit_button("PREDICT RISK", type="primary")
    if go:
        res = get_predictor().predict_one({
            "Attendance_Percentage": att, "Previous_GPA": gpa, "Assignment_Avg": asg, "Quiz_Avg": quiz,
            "Midterm_Score": mid, "Practical_Score": prac, "Study_Hours": study, "Assignments_Missed": missed,
            "Participation_Score": part, "Previous_Backlogs": backlogs})
        st.subheader(f"{ICON[res['risk']]} {sid}: {res['message']}")
        c1, c2 = st.columns(2)
        pr = pd.DataFrame({"Risk": RISK_LABELS[::-1], "Probability": [res["probabilities"][k] for k in RISK_LABELS[::-1]]})
        fig = px.bar(pr, x="Probability", y="Risk", orientation="h", color="Risk", color_discrete_map=COLORS,
                     range_x=[0, 1], text=pr["Probability"].map("{:.0%}".format), title="Risk probability")
        fig.update_layout(showlegend=False)
        c1.plotly_chart(fig, width="stretch")
        with c2:
            st.markdown("**Main risk factors**")
            if res["factors"]:
                for f in res["factors"]:
                    st.write(f"{'🔴' if f['severity'] == 'critical' else '⚠️'} {f['factor']} — {f['value']}")
            else:
                st.write("✅ No warning signals detected")
            st.markdown("**Recommended support**")
            for a in res["interventions"]:
                st.write(f"• {a}")

# ------------------------------------------------------------------ Batch
elif page == "Batch Prediction":
    st.title("Batch prediction")
    st.write("Upload a CSV with at least: `Attendance_Percentage, Previous_GPA, Assignment_Avg, Quiz_Avg, "
             "Midterm_Score, Study_Hours, Assignments_Missed`. Optional columns are filled with defaults.")
    up = st.file_uploader("CSV file", type="csv")
    if up:
        data = pd.read_csv(up)
        need = ["Attendance_Percentage", "Previous_GPA", "Assignment_Avg", "Quiz_Avg", "Midterm_Score",
                "Study_Hours", "Assignments_Missed"]
        miss = [c for c in need if c not in data.columns]
        if miss:
            st.error(f"Missing columns: {miss}")
        else:
            out = get_predictor().predict_df(data)
            show = [c for c in ["Student_ID"] if c in out] + need + ["Prob_Low", "Prob_Medium", "Prob_High", "Predicted_Risk"]
            st.dataframe(out[show], width="stretch", hide_index=True)
            st.download_button("Download predictions", out[show].to_csv(index=False), "predictions.csv", "text/csv")

# ------------------------------------------------------------------ Insights
elif page == "Data Mining Insights":
    st.title("Data mining insights")
    t1, t2, t3 = st.tabs(["EDA", "Clustering", "Association rules"])
    with t1:
        for f in ["eda_attendance_vs_final", "eda_gpa_vs_final", "eda_missed_assignments_vs_risk", "eda_correlation_heatmap"]:
            if (FIG_DIR / f"{f}.png").exists():
                st.image(str(FIG_DIR / f"{f}.png"))
    with t2:
        st.image(str(FIG_DIR / "cluster_pca.png"))
        st.markdown("**Cluster profiles**")
        st.dataframe(pd.read_csv(REPORT_DIR / "cluster_profiles.csv"), hide_index=True, width="stretch")
        st.markdown("**Share of each risk level inside each cluster**")
        st.dataframe(pd.read_csv(REPORT_DIR / "cluster_vs_risk.csv"), hide_index=True, width="stretch")
    with t3:
        st.write("Warning-sign combinations that co-occur with **High risk** (Apriori).")
        st.dataframe(pd.read_csv(REPORT_DIR / "association_rules.csv"), hide_index=True, width="stretch")

# ------------------------------------------------------------------ Performance
else:
    st.title("Model performance")
    st.caption("Held-out test set (20%). For an early-warning system, high-risk recall matters as much as accuracy: "
               "missing a student who needs help is the costly error.")
    st.dataframe(pd.read_csv(REPORT_DIR / "model_comparison.csv"), hide_index=True, width="stretch")
    a, b = st.columns(2)
    a.image(str(FIG_DIR / "confusion_matrices.png"))
    b.image(str(FIG_DIR / "feature_importance.png"))
    meta = json.loads((MODEL_DIR / "metadata.json").read_text())
    st.caption(f"Random Forest params: {meta['rf_best_params']}")
