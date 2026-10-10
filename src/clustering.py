"""K-Means clustering of students (unsupervised - no risk label needed)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from .utils import CLUSTERED_PATH, FIG_DIR, PROCESSED_PATH, REPORT_DIR, RISK_LABELS, ensure_dirs

FEATURES = ["Attendance_Percentage", "Assessment_Avg", "Previous_GPA", "Study_Hours", "Engagement_Score"]
NAMES = ["Struggling & disengaged", "Average / inconsistent", "High attendance + high marks"]


def main(k: int = 3):
    ensure_dirs()
    df = pd.read_csv(PROCESSED_PATH)
    Z = StandardScaler().fit_transform(df[FEATURES])

    ks = range(2, 8)
    inertia, sil = [], []
    for kk in ks:
        km = KMeans(kk, n_init=10, random_state=42).fit(Z)
        inertia.append(km.inertia_); sil.append(silhouette_score(Z, km.labels_, sample_size=3000, random_state=42))
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    ax[0].plot(list(ks), inertia, "o-"); ax[0].set_title("Elbow (inertia)"); ax[0].set_xlabel("k")
    ax[1].plot(list(ks), sil, "o-", color="darkorange"); ax[1].set_title("Silhouette score"); ax[1].set_xlabel("k")
    fig.tight_layout(); fig.savefig(FIG_DIR / "cluster_k_selection.png", dpi=140); plt.close(fig)

    km = KMeans(k, n_init=20, random_state=42).fit(Z)
    df["_c"] = km.labels_
    order = df.groupby("_c")["Assessment_Avg"].mean().sort_values().index      # low -> high performance
    remap = {old: new for new, old in enumerate(order)}
    df["Cluster"] = df["_c"].map(remap)
    df["Cluster_Name"] = df["Cluster"].map(dict(enumerate(NAMES)))

    profile = df.groupby("Cluster_Name")[FEATURES + ["Final_Score"]].mean().round(1)
    profile["Students"] = df["Cluster_Name"].value_counts()
    profile = profile.loc[NAMES]
    profile.to_csv(REPORT_DIR / "cluster_profiles.csv")
    xt = pd.crosstab(df["Cluster_Name"], df["Risk_Level"], normalize="index")[RISK_LABELS].loc[NAMES].round(3)
    xt.to_csv(REPORT_DIR / "cluster_vs_risk.csv")

    p = PCA(2, random_state=42).fit_transform(Z)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for c, name in enumerate(NAMES):
        m = df["Cluster"] == c
        ax.scatter(p[m, 0][:1500], p[m, 1][:1500], s=10, alpha=0.5, label=name)
    ax.set_title("K-Means student groups (PCA projection)"); ax.legend(markerscale=2, fontsize=8)
    fig.tight_layout(); fig.savefig(FIG_DIR / "cluster_pca.png", dpi=140); plt.close(fig)

    df[["Student_ID", "Cluster", "Cluster_Name"]].to_csv(CLUSTERED_PATH, index=False)
    print(f"[clustering] silhouette(k={k})={sil[k - 2]:.3f}\n{profile}\n{xt}")
    return profile, xt


if __name__ == "__main__":
    main()
