#!/usr/bin/env python3
"""
generate_outputs.py
===================
Generates ALL output images for the Space Debris Detection project:

    EDA Plots:
        outputs/eda_inc_ecc.png
        outputs/eda_altitude.png
        outputs/eda_velocity.png

    3D Orbit Visualisation:
        outputs/3d_orbit_visualization.png

    ML Model Results (Random Forest + Gradient Boosting):
        outputs/confusion_matrix_rf.png
        outputs/confusion_matrix_gb.png
        outputs/feature_importance.png
        outputs/classification_report.png
        outputs/model_comparison.png

    Cross-Year Validation (2024 → 2026):
        outputs/cross_year_accuracy.png
        outputs/cross_year_per_group.png
        outputs/cross_year_confusion_matrix.png
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for scripts
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")

from datetime import datetime, timezone
from sgp4.api import Satrec, jday
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
)
from imblearn.over_sampling import SMOTE
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401

# Output directory
os.makedirs("outputs", exist_ok=True)
sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 100

# Constants
EARTH_RADIUS = 6371.0        # km
MU           = 398600.4418   # km^3/s^2

# 14 orbital features
FEATURE_COLS = [
    "eccentricity", "inclination", "raan", "arg_perigee",
    "mean_anomaly", "mean_motion", "semi_major_axis",
    "altitude", "perigee", "apogee", "orbital_period",
    "distance_from_center", "velocity_magnitude", "specific_orbital_energy",
]

# Group definitions
POSITIVE_GROUPS = ["cosmos_1408", "cosmos_2251", "fengyun_1c", "iridium_33"]
NEGATIVE_GROUPS = ["active_satellites"]

GROUP_COLORS = {
    "cosmos_1408"       : "#E53935",
    "cosmos_2251"       : "#FB8C00",
    "fengyun_1c"        : "#8E24AA",
    "iridium_33"        : "#1E88E5",
    "active_satellites" : "#43A047",
}
GROUP_LABELS = {
    "cosmos_1408"       : "Cosmos 1408 Debris (2021)",
    "cosmos_2251"       : "Cosmos 2251 Debris (2009)",
    "fengyun_1c"        : "FengYun 1C Debris (2007)",
    "iridium_33"        : "Iridium 33 Debris (2009)",
    "active_satellites" : "Active Satellites",
}

# Data directories
TLE_DIR_2024 = "OLD_dataset"
TLE_DIR_2026 = "26_Data set"

# Helper: parse a TLE file
def parse_tle_file(filepath: str, label: int, group_key: str, eval_time: datetime = None) -> list:
    if not os.path.exists(filepath):
        print(f"  [WARN] File not found, skipping: {filepath}")
        return []

    if eval_time is None:
        eval_time = datetime.now(timezone.utc)

    jd_now, fr_now = jday(
        eval_time.year, eval_time.month, eval_time.day,
        eval_time.hour, eval_time.minute, eval_time.second,
    )

    with open(filepath, "r", encoding="utf-8", errors="ignore") as fh:
        lines = [l.rstrip() for l in fh if l.strip()]

    records = []
    for i in range(0, len(lines) - 2, 3):
        name  = lines[i].strip()
        line1 = lines[i + 1].strip()
        line2 = lines[i + 2].strip()
        if not (line1.startswith("1 ") and line2.startswith("2 ")):
            continue
        try:
            sat  = Satrec.twoline2rv(line1, line2)
            ecc  = sat.ecco
            inc  = sat.inclo
            raan = sat.nodeo
            argp = sat.argpo
            ma   = sat.mo
            mm   = sat.no_kozai          # rev/day

            mm_rad_s   = mm * 2 * np.pi / 86400.0
            sma        = (MU / mm_rad_s ** 2) ** (1 / 3)
            altitude   = sma - EARTH_RADIUS
            perigee    = sma * (1 - ecc)
            apogee     = sma * (1 + ecc)
            orb_period = 86400.0 / mm

            e, r, v = sat.sgp4(jd_now, fr_now)
            if e != 0:
                continue

            dist_c  = np.sqrt(r[0]**2 + r[1]**2 + r[2]**2)
            vel_mag = np.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
            soe     = 0.5 * vel_mag**2 - MU / dist_c

            records.append({
                "sat_name"               : name,
                "eccentricity"           : ecc,
                "inclination"            : inc,
                "raan"                   : raan,
                "arg_perigee"            : argp,
                "mean_anomaly"           : ma,
                "mean_motion"            : mm,
                "semi_major_axis"        : sma,
                "altitude"               : altitude,
                "perigee"                : perigee,
                "apogee"                 : apogee,
                "orbital_period"         : orb_period,
                "distance_from_center"   : dist_c,
                "velocity_magnitude"     : vel_mag,
                "specific_orbital_energy": soe,
                "label"                  : label,
                "source_file"            : group_key,
                "x": r[0], "y": r[1], "z": r[2],
            })
        except Exception:
            continue

    return records

if __name__ == "__main__":
    # LOAD 2024 DATA
    print("Loading 2024 TLE data from OLD_dataset/ ...")

    FILE_MAP_2024 = {
        "cosmos_1408"       : os.path.join(TLE_DIR_2024, "cosmos-1408-debris.txt"),
        "cosmos_2251"       : os.path.join(TLE_DIR_2024, "cosmos-2251-debris.txt"),
        "fengyun_1c"        : os.path.join(TLE_DIR_2024, "fengyun-1c-debris.txt"),
        "iridium_33"        : os.path.join(TLE_DIR_2024, "iridium-33-debris.txt"),
        "active_satellites" : os.path.join(TLE_DIR_2024, "active-sate.txt"),
    }

    EVAL_2024 = datetime(2024, 11, 17, 18, 0, 0, tzinfo=timezone.utc)

    all_2024 = []
    for grp in POSITIVE_GROUPS + NEGATIVE_GROUPS:
        lbl  = 1 if grp in POSITIVE_GROUPS else 0
        recs = parse_tle_file(FILE_MAP_2024[grp], lbl, grp, eval_time=EVAL_2024)
        all_2024.extend(recs)
        print(f"  {grp:<25}  {len(recs):>5} objects")

    df = pd.DataFrame(all_2024)
    print(f"\nTotal 2024: {len(df)}  |  Debris: {(df['label']==1).sum()}  |  Active: {(df['label']==0).sum()}")

    # EDA PLOTS
    print("\nGenerating EDA plots ...")

    # EDA 1: Inclination vs Eccentricity
    fig, ax = plt.subplots(figsize=(11, 7))
    for grp in POSITIVE_GROUPS + NEGATIVE_GROUPS:
        subset = df[df["source_file"] == grp]
        ax.scatter(
            np.degrees(subset["inclination"]), subset["eccentricity"],
            s=5, alpha=0.4, label=GROUP_LABELS[grp], color=GROUP_COLORS[grp]
        )
    ax.set_xlabel("Inclination (degrees)", fontsize=12)
    ax.set_ylabel("Eccentricity", fontsize=12)
    ax.set_title("Inclination vs Eccentricity — Debris vs Active Satellites",
                 fontsize=14, fontweight="bold")
    ax.legend(markerscale=4, fontsize=9)
    plt.tight_layout()
    plt.savefig("outputs/eda_inc_ecc.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: outputs/eda_inc_ecc.png")

    # EDA 2: Altitude Distribution
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.histplot(data=df, x="altitude", hue="label", bins=60, alpha=0.6,
                 kde=True, ax=ax, palette={0: "#43A047", 1: "#E53935"})
    ax.set_xlabel("Altitude above Earth (km)", fontsize=12)
    ax.set_title("Altitude Distribution: Active Satellites vs Debris",
                 fontsize=14, fontweight="bold")
    ax.get_legend().set_title("Class (0=Active, 1=Debris)")
    plt.tight_layout()
    plt.savefig("outputs/eda_altitude.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: outputs/eda_altitude.png")

    # EDA 3: Velocity Distribution
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.histplot(data=df, x="velocity_magnitude", hue="label", bins=60, alpha=0.6,
                 kde=True, ax=ax, palette={0: "#43A047", 1: "#E53935"})
    ax.set_xlabel("Velocity Magnitude (km/s)", fontsize=12)
    ax.set_title("Velocity Magnitude: Active Satellites vs Debris",
                 fontsize=14, fontweight="bold")
    ax.get_legend().set_title("Class (0=Active, 1=Debris)")
    plt.tight_layout()
    plt.savefig("outputs/eda_velocity.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: outputs/eda_velocity.png")

    # 3D ORBITAL VISUALISATION
    print("\nGenerating 3D orbital visualisation ...")
    fig  = plt.figure(figsize=(14, 10))
    ax3d = fig.add_subplot(111, projection="3d")

    for grp in POSITIVE_GROUPS + NEGATIVE_GROUPS:
        subset = df[df["source_file"] == grp]
        sample = subset.sample(min(250, len(subset)), random_state=42)
        ax3d.scatter(sample["x"], sample["y"], sample["z"],
                     s=2, alpha=0.35,
                     color=GROUP_COLORS[grp], label=GROUP_LABELS[grp])

    # Draw translucent Earth sphere
    u  = np.linspace(0, 2 * np.pi, 40)
    v  = np.linspace(0, np.pi, 40)
    xe = EARTH_RADIUS * np.outer(np.cos(u), np.sin(v))
    ye = EARTH_RADIUS * np.outer(np.sin(u), np.sin(v))
    ze = EARTH_RADIUS * np.outer(np.ones_like(u), np.cos(v))
    ax3d.plot_surface(xe, ye, ze, color="royalblue", alpha=0.18, linewidth=0)

    ax3d.set_xlabel("X (km)")
    ax3d.set_ylabel("Y (km)")
    ax3d.set_zlabel("Z (km)")
    ax3d.set_title("3D Orbital Positions: Space Debris vs Active Satellites",
                   pad=15, fontsize=13, fontweight="bold")
    ax3d.legend(loc="upper left", markerscale=6, fontsize=8)
    plt.tight_layout()
    plt.savefig("outputs/3d_orbit_visualization.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: outputs/3d_orbit_visualization.png")

    # TRAIN / TEST SPLIT — 2024 DATA
    X = df[FEATURE_COLS].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain: {len(X_train)}  |  Test: {len(X_test)}")

    # RANDOM FOREST — UNBALANCED DATA
    print("\nTraining Random Forest (unbalanced data) ...")
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=None,
        n_jobs=-1, random_state=42, oob_score=True,
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_acc  = accuracy_score(y_test, rf_pred)
    print(f"  RF Test Accuracy : {rf_acc * 100:.4f}%  |  OOB: {rf.oob_score_ * 100:.4f}%")

    # Confusion Matrix — RF
    cm = confusion_matrix(y_test, rf_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Active Satellite", "Debris"],
                yticklabels=["Active Satellite", "Debris"],
                annot_kws={"size": 16}, ax=ax)
    ax.set_xlabel("Predicted Label", fontsize=13)
    ax.set_ylabel("Actual Label", fontsize=13)
    ax.set_title(f"Confusion Matrix — Random Forest\nAccuracy = {rf_acc * 100:.2f}%",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("outputs/confusion_matrix_rf.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: outputs/confusion_matrix_rf.png")

    # Feature Importance
    importances = pd.Series(rf.feature_importances_, index=FEATURE_COLS).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(10, 7))
    colors  = ["#1E88E5" if imp > importances.median() else "#90CAF9" for imp in importances]
    bars    = ax.barh(importances.index, importances.values, color=colors)
    ax.set_xlabel("Feature Importance (Gini)", fontsize=12)
    ax.set_title("Random Forest — Which Orbital Characteristics Matter Most?",
                 fontsize=13, fontweight="bold")
    ax.axvline(importances.median(), color="red", linestyle="--", alpha=0.6, label="Median")
    ax.legend(fontsize=10)
    for bar, val in zip(bars, importances.values):
        ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig("outputs/feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: outputs/feature_importance.png")

    # Classification Report Plot — RF
    report    = classification_report(y_test, rf_pred,
                                      target_names=["Active Satellite", "Debris"],
                                      output_dict=True)
    report_df = pd.DataFrame(report).transpose().drop(["accuracy", "macro avg", "weighted avg"])
    fig, ax = plt.subplots(figsize=(8, 4))
    report_df[["precision", "recall", "f1-score"]].plot(
        kind="bar", ax=ax,
        color=["#1E88E5", "#43A047", "#E53935"], edgecolor="white", width=0.6
    )
    ax.set_title("Classification Report — Random Forest", fontsize=13, fontweight="bold")
    ax.set_ylim(0.88, 1.02)
    ax.set_ylabel("Score")
    ax.set_xticklabels(["Active Satellite", "Debris"], rotation=0, fontsize=11)
    ax.legend(["Precision", "Recall", "F1-Score"], fontsize=10)
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.3f}",
                    (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig("outputs/classification_report.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: outputs/classification_report.png")

    # RANDOM FOREST — SMOTE BALANCED DATA
    print("\nTraining Random Forest (SMOTE balanced data) ...")
    smote          = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
    print(f"  After SMOTE — Train: {len(X_train_sm)}"
          f"  |  Debris: {y_train_sm.sum()}  |  Active: {(y_train_sm == 0).sum()}")

    rf_smote = RandomForestClassifier(
        n_estimators=200, max_depth=None,
        n_jobs=-1, random_state=42, oob_score=True,
    )
    rf_smote.fit(X_train_sm, y_train_sm)
    rf_sm_pred = rf_smote.predict(X_test)
    rf_sm_acc  = accuracy_score(y_test, rf_sm_pred)
    print(f"  RF (SMOTE) Test Accuracy : {rf_sm_acc * 100:.4f}%  |  OOB: {rf_smote.oob_score_ * 100:.4f}%")

    # GRADIENT BOOSTING
    print("\nTraining Gradient Boosting (this may take a minute) ...")
    gb = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.1,
        max_depth=5, random_state=42,
    )
    gb.fit(X_train, y_train)
    gb_pred = gb.predict(X_test)
    gb_acc  = accuracy_score(y_test, gb_pred)
    print(f"  GB Test Accuracy : {gb_acc * 100:.4f}%")

    # Confusion Matrix — GB
    cm_gb = confusion_matrix(y_test, gb_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm_gb, annot=True, fmt="d", cmap="Greens",
                xticklabels=["Active Satellite", "Debris"],
                yticklabels=["Active Satellite", "Debris"],
                annot_kws={"size": 16}, ax=ax)
    ax.set_xlabel("Predicted Label", fontsize=13)
    ax.set_ylabel("Actual Label", fontsize=13)
    ax.set_title(f"Confusion Matrix — Gradient Boosting\nAccuracy = {gb_acc * 100:.2f}%",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("outputs/confusion_matrix_gb.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: outputs/confusion_matrix_gb.png")

    # MODEL COMPARISON PLOT
    print("\nGenerating model comparison plot ...")
    model_names = [
        "RF\n(Unbalanced)", "RF\n(SMOTE)", "Gradient\nBoosting"
    ]
    accs   = [rf_acc * 100, rf_sm_acc * 100, gb_acc * 100]
    colors = ["#1E88E5", "#43A047", "#FB8C00"]

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.bar(model_names, accs, color=colors, edgecolor="white", width=0.45)
    ax.set_ylim(min(accs) - 2, 101)
    ax.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax.set_title("Model Comparison: RF vs RF-SMOTE vs Gradient Boosting\n(Trained & Tested on 2024 Data)",
                 fontsize=13, fontweight="bold")
    ax.axhline(99, color="black", linestyle="--", alpha=0.4, label="99% threshold")
    ax.legend(fontsize=10)
    for bar, val in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{val:.4f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig("outputs/model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: outputs/model_comparison.png")

    # CROSS-YEAR PREDICTION (2026 DATA)
    print("\nLoading 2026 TLE data from 26_Data set/ ...")

    FILE_MAP_2026 = {
        "cosmos_1408"       : os.path.join(TLE_DIR_2026, "cosmos_1408_2026.txt"),
        "cosmos_2251"       : os.path.join(TLE_DIR_2026, "cosmos_2251_2026.txt"),
        "fengyun_1c"        : os.path.join(TLE_DIR_2026, "fengyun_1c_2026.txt"),
        "iridium_33"        : os.path.join(TLE_DIR_2026, "iridium_33_2026.txt"),
        "active_satellites" : os.path.join(TLE_DIR_2026, "active_satellites_2026.txt"),
    }

    EVAL_2026 = datetime(2026, 5, 28, 0, 0, 0, tzinfo=timezone.utc)

    all_2026 = []
    for grp in POSITIVE_GROUPS + NEGATIVE_GROUPS:
        lbl  = 1 if grp in POSITIVE_GROUPS else 0
        recs = parse_tle_file(FILE_MAP_2026[grp], lbl, grp, eval_time=EVAL_2026)
        all_2026.extend(recs)
        print(f"  {grp:<25}  {len(recs):>5} objects (2026)")

    df_2026 = pd.DataFrame(all_2026)
    X_2026  = df_2026[FEATURE_COLS].values
    y_2026  = df_2026["label"].values

    rf_cross = accuracy_score(y_2026, rf.predict(X_2026))
    rf_cross_auc = roc_auc_score(y_2026, rf.predict_proba(X_2026)[:, 1])
    gb_cross = accuracy_score(y_2026, gb.predict(X_2026))
    gb_cross_auc = roc_auc_score(y_2026, gb.predict_proba(X_2026)[:, 1])
    print(f"\n  Cross-Year Accuracy — RF : {rf_cross * 100:.4f}%  |  AUC : {rf_cross_auc * 100:.4f}%")
    print(f"  Cross-Year Accuracy — GB : {gb_cross * 100:.4f}%  |  AUC : {gb_cross_auc * 100:.4f}%")

    # Cross-Year Accuracy Bar Chart
    fig, ax = plt.subplots(figsize=(10, 6))
    lbls    = ["RF 2024 Val", "GB 2024 Val", "RF 2026 Cross", "GB 2026 Cross"]
    vals    = [rf_acc * 100, gb_acc * 100, rf_cross * 100, gb_cross * 100]
    cols    = ["#1E88E5", "#43A047", "#E53935", "#FB8C00"]
    bars2   = ax.bar(lbls, vals, color=cols, edgecolor="white", width=0.5)
    ax.set_ylim(min(vals) - 2, 101)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("2024 Validation vs 2026 Cross-Year Accuracy",
                 fontsize=13, fontweight="bold")
    ax.axhline(99, color="black", linestyle="--", alpha=0.4, label="99% line")
    ax.legend(fontsize=10)
    for bar, val in zip(bars2, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{val:.2f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig("outputs/cross_year_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: outputs/cross_year_accuracy.png")

    # Cross-Year Confusion Matrix (RF)
    rf_pred_2026 = rf.predict(X_2026)
    cm_cy = confusion_matrix(y_2026, rf_pred_2026)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm_cy, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Active Sat.", "Debris"],
                yticklabels=["Active Sat.", "Debris"],
                annot_kws={"size": 16}, ax=ax)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("Actual Label (2026 Ground Truth)", fontsize=12)
    ax.set_title(f"RF Cross-Year Confusion Matrix\nTrained 2024 → Tested 2026  |  Acc = {rf_cross * 100:.2f}%",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig("outputs/cross_year_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: outputs/cross_year_confusion_matrix.png")

    # Per-Group Analysis (2026)
    group_rows = []
    for grp in POSITIVE_GROUPS + NEGATIVE_GROUPS:
        mask      = df_2026["source_file"] == grp
        subset_X  = df_2026.loc[mask, FEATURE_COLS].values
        if len(subset_X) == 0:
            continue
        true_lbl  = "DEBRIS" if grp in POSITIVE_GROUPS else "ACTIVE"
        rf_d_pct  = rf.predict(subset_X).mean() * 100
        gb_d_pct  = gb.predict(subset_X).mean() * 100
        group_rows.append({"group": grp, "true": true_lbl,
                            "rf": rf_d_pct, "gb": gb_d_pct})

    gdf = pd.DataFrame(group_rows)
    fig, ax = plt.subplots(figsize=(12, 6))
    x     = np.arange(len(gdf))
    w     = 0.35
    b1    = ax.bar(x - w / 2, gdf["rf"], w, label="Random Forest", color="#1E88E5", edgecolor="white")
    b2    = ax.bar(x + w / 2, gdf["gb"], w, label="Gradient Boosting", color="#43A047", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['group']}\n({r['true']})" for _, r in gdf.iterrows()], fontsize=9)
    ax.set_ylabel("% Objects Predicted as Debris", fontsize=11)
    ax.set_title("Per-Group: % Predicted as Debris on 2026 Data\n(Debris groups → ~100%,  Active groups → ~0%)",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(0, 115)
    ax.legend(fontsize=10)
    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig("outputs/cross_year_per_group.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: outputs/cross_year_per_group.png")

    # ROC Curve Plot
    print("\nGenerating ROC curve comparison plot ...")
    rf_fpr, rf_tpr, _ = roc_curve(y_2026, rf.predict_proba(X_2026)[:, 1])
    gb_fpr, gb_tpr, _ = roc_curve(y_2026, gb.predict_proba(X_2026)[:, 1])

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(rf_fpr, rf_tpr, label=f"Random Forest (AUC = {rf_cross_auc:.4f})", color="#1E88E5", lw=2)
    ax.plot(gb_fpr, gb_tpr, label=f"Gradient Boosting (AUC = {gb_cross_auc:.4f})", color="#43A047", lw=2)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve - 2026 Cross-Year Prediction", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    plt.savefig("outputs/cross_year_roc_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: outputs/cross_year_roc_curve.png")

    # FINAL SUMMARY
    print("\n" + "=" * 55)
    print("  ALL OUTPUTS SAVED IN: outputs/")
    print("=" * 55)
    for f in sorted(os.listdir("outputs")):
        size = os.path.getsize(os.path.join("outputs", f))
        print(f"  {f:<45} ({size // 1024} KB)")
    print("=" * 55)
