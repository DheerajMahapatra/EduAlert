"""Train + compare 4 models, tune Random Forest (primary), save artefacts.

Usage:  python -m src.train [--mlflow]
"""
import argparse
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from .utils import (ALL_FEATURES, CATEGORICAL_FEATURES, FIG_DIR, ID_TO_LABEL, LABEL_TO_ID, MODEL_DIR,
                    NUMERIC_FEATURES, OUTCOME_COL, PROCESSED_PATH, REPORT_DIR, RISK_LABELS, SCORED_PATH,
                    TARGET, ensure_dirs)

SEED = 42
HIGH = LABEL_TO_ID["High"]


def build_preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
    ])


def evaluate(name, model, X_te, y_te):
    pred = model.predict(X_te)
    proba = model.predict_proba(X_te)
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_te, pred),
        "Precision (macro)": precision_score(y_te, pred, average="macro"),
        "Recall (macro)": recall_score(y_te, pred, average="macro"),
        "F1 (macro)": f1_score(y_te, pred, average="macro"),
        "High-risk Precision": precision_score(y_te, pred, labels=[HIGH], average=None)[0],
        "High-risk Recall": recall_score(y_te, pred, labels=[HIGH], average=None)[0],
        "High-risk F1": f1_score(y_te, pred, labels=[HIGH], average=None)[0],
        "ROC-AUC (OvR)": roc_auc_score(y_te, proba, multi_class="ovr", average="macro"),
        "_pred": pred,
    }


def feature_names(pre):
    names = pre.get_feature_names_out()
    return [n.split("__", 1)[1] for n in names]


def main(use_mlflow: bool = False):
    ensure_dirs()
    df = pd.read_csv(PROCESSED_PATH)
    assert OUTCOME_COL not in ALL_FEATURES and TARGET not in ALL_FEATURES, "target leakage!"

    X = df[ALL_FEATURES]
    y = df[TARGET].map(LABEL_TO_ID)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

    pre = build_preprocessor().fit(X_tr)          # fit on train only
    Xtr, Xte = pre.transform(X_tr), pre.transform(X_te)

    # --- Random Forest (primary) with hyper-parameter tuning
    grid = GridSearchCV(
        RandomForestClassifier(class_weight="balanced_subsample", random_state=SEED, n_jobs=-1),
        {"n_estimators": [200, 400], "max_depth": [8, 12, None], "min_samples_leaf": [1, 3]},
        scoring="f1_macro", cv=3, n_jobs=-1,
    ).fit(Xtr, y_tr)
    rf = grid.best_estimator_
    print(f"[train] RF best params: {grid.best_params_}  (cv f1_macro={grid.best_score_:.3f})")

    models = {
        "Logistic Regression (baseline)": LogisticRegression(max_iter=3000, class_weight="balanced").fit(Xtr, y_tr),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=20, class_weight="balanced",
                                                random_state=SEED).fit(Xtr, y_tr),
        "Random Forest (primary)": rf,
        "XGBoost": XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.06, subsample=0.9,
                                 colsample_bytree=0.9, eval_metric="mlogloss", random_state=SEED, n_jobs=-1
                                 ).fit(Xtr, y_tr, sample_weight=compute_sample_weight("balanced", y_tr)),
    }
    results = [evaluate(n, m, Xte, y_te) for n, m in models.items()]
    table = pd.DataFrame([{k: v for k, v in r.items() if k != "_pred"} for r in results]).round(4)
    table.to_csv(REPORT_DIR / "model_comparison.csv", index=False)
    print(table.to_string(index=False))

    # --- confusion matrices
    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    for ax, r in zip(axes.ravel(), results):
        cm = confusion_matrix(y_te, r["_pred"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=RISK_LABELS, yticklabels=RISK_LABELS)
        ax.set_title(f"{r['Model']}\nhigh-risk recall={r['High-risk Recall']:.2f}", fontsize=10)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    fig.tight_layout(); fig.savefig(FIG_DIR / "confusion_matrices.png", dpi=140); plt.close(fig)

    # --- feature importance (RF)
    names = feature_names(pre)
    imp = pd.Series(rf.feature_importances_, index=names).sort_values().tail(15)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.barh(imp.index, imp.values, color="#3b6fb6")
    ax.set_title("Random Forest feature importance (top 15)"); ax.set_xlabel("Importance")
    fig.tight_layout(); fig.savefig(FIG_DIR / "feature_importance.png", dpi=140); plt.close(fig)
    pd.Series(rf.feature_importances_, index=names).sort_values(ascending=False).round(4).to_csv(
        REPORT_DIR / "feature_importance.csv", header=["importance"], index_label="feature")

    # --- save artefacts (fit final preprocessor/model on train split; metrics above are held-out)
    joblib.dump(pre, MODEL_DIR / "preprocessor.pkl")
    joblib.dump(rf, MODEL_DIR / "risk_model.pkl")
    meta = {
        "classes": RISK_LABELS, "features": ALL_FEATURES, "rf_best_params": grid.best_params_,
        "rf_cv_f1_macro": round(grid.best_score_, 4), "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
        "metrics": table.to_dict(orient="records"),
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(meta, indent=2))

    # --- score whole cohort with OUT-OF-FOLD predictions (honest, not in-sample) for the dashboard
    from sklearn.pipeline import make_pipeline
    oof_pipe = make_pipeline(build_preprocessor(), RandomForestClassifier(
        **{**grid.best_params_, "class_weight": "balanced_subsample", "random_state": SEED, "n_jobs": -1}))
    oof = cross_val_predict(oof_pipe, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                            method="predict_proba")
    scored = df.copy()
    scored["Prob_Low"], scored["Prob_Medium"], scored["Prob_High"] = oof[:, 0], oof[:, 1], oof[:, 2]
    scored["Predicted_Risk"] = [ID_TO_LABEL[i] for i in oof.argmax(1)]
    scored.to_csv(SCORED_PATH, index=False)
    print(f"[train] artefacts -> {MODEL_DIR}; scored cohort -> {SCORED_PATH}")

    if use_mlflow:
        try:
            import mlflow
            mlflow.set_experiment("EduAlert")
            for r in results:
                with mlflow.start_run(run_name=r["Model"]):
                    mlflow.log_metrics({k.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_"): float(v)
                                        for k, v in r.items() if k not in ("Model", "_pred")})
            with mlflow.start_run(run_name="RF final"):
                mlflow.log_params(grid.best_params_)
                mlflow.sklearn.log_model(rf, "risk_model")
            print("[train] logged to MLflow (run `mlflow ui`)")
        except ImportError:
            print("[train] mlflow not installed - skipped (pip install mlflow)")
    return table


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlflow", action="store_true")
    main(ap.parse_args().mlflow)
