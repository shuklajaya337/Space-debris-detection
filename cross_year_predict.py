#!/usr/bin/env python3
"""
cross_year_predict.py
=====================
Cross-Year Temporal Validation:
    Train RF + GB on 2024 TLE data (OLD_dataset) -> Predict on 2026 TLE data (26_Data set)

This validates whether debris detected in 2024 is STILL detected
as debris in 2026, proving the model generalises across time.
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")

from datetime import datetime, timezone
from sgp4.api import Satrec, jday
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Constants
EARTH_RADIUS = 6371.0
MU           = 398600.4418   # km^3/s^2

# 14 orbital features
FEATURE_COLS = [
    "eccentricity", "inclination", "raan", "arg_perigee",
    "mean_anomaly", "mean_motion", "semi_major_axis",
    "altitude", "perigee", "apogee", "orbital_period",
    "distance_from_center", "velocity_magnitude", "specific_orbital_energy",
]

POSITIVE_GROUPS = ["cosmos_1408", "cosmos_2251", "fengyun_1c", "iridium_33"]
NEGATIVE_GROUPS = ["active_satellites"]

# Folders
TLE_DIR_2024 = "OLD_dataset"
TLE_DIR_2026 = "26_Data set"

os.makedirs("outputs", exist_ok=True)
sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 100

# Helper: Parse TLE File
def parse_tle_file(filepath: str, label: int, group_key: str, eval_time: datetime = None) -> list:
    if not os.path.exists(filepath):
        print(f"  [WARN] Not found, skipping: {filepath}")
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
    # STEP 1: LOAD 2024 TRAINING DATA
    print("=" * 65)
    print("  STEP 1 - Loading 2024 Training Data  (OLD_dataset/)")
    print("=" * 65)

    FILE_MAP_2024 = {
        "cosmos_1408"       : os.path.join(TLE_DIR_2024, "cosmos-1408-debris.txt"),
        "cosmos_2251"       : os.path.join(TLE_DIR_2024, "cosmos-2251-debris.txt"),
        "fengyun_1c"        : os.path.join(TLE_DIR_2024, "fengyun-1c-debris.txt"),
        "iridium_33"        : os.path.join(TLE_DIR_2024, "iridium-33-debris.txt"),
        "active_satellites" : os.path.join(TLE_DIR_2024, "active-sate.txt"),
    }

    EVAL_TIME_2024 = datetime(2024, 11, 17, 18, 0, 0, tzinfo=timezone.utc)

    records_2024 = []
    for grp in POSITIVE_GROUPS + NEGATIVE_GROUPS:
        lbl  = 1 if grp in POSITIVE_GROUPS else 0
        recs = parse_tle_file(FILE_MAP_2024[grp], lbl, grp, eval_time=EVAL_TIME_2024)
        records_2024.extend(recs)
        tag = "DEBRIS" if lbl == 1 else "ACTIVE SAT"
        print(f"  {grp:<25}  {len(recs):>5} objects  [{tag}]")

    df_2024 = pd.DataFrame(records_2024)
    print(f"\n  Total 2024 records : {len(df_2024)}")

    # STEP 2: LOAD 2026 TEST DATA
    print("\n" + "=" * 65)
    print("  STEP 2 - Loading 2026 Test Data  (26_Data set/)")
    print("=" * 65)

    FILE_MAP_2026 = {
        "cosmos_1408"       : os.path.join(TLE_DIR_2026, "cosmos_1408_2026.txt"),
        "cosmos_2251"       : os.path.join(TLE_DIR_2026, "cosmos_2251_2026.txt"),
        "fengyun_1c"        : os.path.join(TLE_DIR_2026, "fengyun_1c_2026.txt"),
        "iridium_33"        : os.path.join(TLE_DIR_2026, "iridium_33_2026.txt"),
        "active_satellites" : os.path.join(TLE_DIR_2026, "active_satellites_2026.txt"),
    }

    EVAL_TIME_2026 = datetime(2026, 5, 28, 0, 0, 0, tzinfo=timezone.utc)

    records_2026 = []
    for grp in POSITIVE_GROUPS + NEGATIVE_GROUPS:
        lbl  = 1 if grp in POSITIVE_GROUPS else 0
        recs = parse_tle_file(FILE_MAP_2026[grp], lbl, grp, eval_time=EVAL_TIME_2026)
        records_2026.extend(recs)
        tag = "DEBRIS" if lbl == 1 else "ACTIVE SAT"
        print(f"  {grp:<25}  {len(recs):>5} objects  [{tag}]")

    df_2026 = pd.DataFrame(records_2026)
    print(f"\n  Total 2026 records : {len(df_2026)}")

    # STEP 3: TRAIN MODELS ON 2024 DATA
    print("\n" + "=" * 65)
    print("  STEP 3 - Training Models on 2024 Data")
    print("=" * 65)

    X_2024 = df_2024[FEATURE_COLS].values
    y_2024 = df_2024["label"].values

    X_train, X_val, y_train, y_val = train_test_split(
        X_2024, y_2024, test_size=0.2, random_state=42, stratify=y_2024
    )

    print("  Training Random Forest (200 trees)...")
    rf = RandomForestClassifier(n_estimators=200, max_depth=None, n_jobs=-1, random_state=42, oob_score=True)
    rf.fit(X_train, y_train)
    rf_val_acc = accuracy_score(y_val, rf.predict(X_val))
    print(f"    2024 validation accuracy : {rf_val_acc * 100:.4f}%  |  OOB: {rf.oob_score_ * 100:.4f}%")

    print("  Training Gradient Boosting (200 estimators)...")
    gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
    gb.fit(X_train, y_train)
    gb_val_acc = accuracy_score(y_val, gb.predict(X_val))
    print(f"    2024 validation accuracy : {gb_val_acc * 100:.4f}%")

    # Save trained models
    joblib.dump(rf, "outputs/rf_debris_classifier_2024.pkl")
    joblib.dump(gb, "outputs/gb_debris_classifier_2024.pkl")
    print("\n  Models saved in outputs/")

    # STEP 4: CROSS-YEAR PREDICTION
    print("\n" + "=" * 65)
    print("  STEP 4 - Cross-Year Prediction: 2024 Model -> 2026 Data")
    print("=" * 65)

    X_2026 = df_2026[FEATURE_COLS].values
    y_2026 = df_2026["label"].values

    rf_pred_2026 = rf.predict(X_2026)
    rf_cross_acc = accuracy_score(y_2026, rf_pred_2026)

    gb_pred_2026 = gb.predict(X_2026)
    gb_cross_acc = accuracy_score(y_2026, gb_pred_2026)

    print(f"\n  Random Forest   - 2026 Cross-Year Accuracy : {rf_cross_acc * 100:.4f}%")
    print(f"  Gradient Boost  - 2026 Cross-Year Accuracy : {gb_cross_acc * 100:.4f}%")

    # STEP 5: PER-GROUP ANALYSIS
    print("\n" + "=" * 65)
    print("  STEP 5 - Per-Group Prediction Analysis (2026 Data)")
    print("=" * 65)
    print(f"\n  {'Group':<25}  {'True Label':<12}  {'RF Debris%':>11}  {'GB Debris%':>11}")
    print("  " + "-" * 65)

    group_results = []
    for grp in POSITIVE_GROUPS + NEGATIVE_GROUPS:
        mask       = df_2026["source_file"] == grp
        true_lbl   = "DEBRIS" if grp in POSITIVE_GROUPS else "ACTIVE SAT"
        subset_X   = df_2026.loc[mask, FEATURE_COLS].values
        if len(subset_X) == 0:
            continue
        rf_debris_pct = rf.predict(subset_X).mean() * 100
        gb_debris_pct = gb.predict(subset_X).mean() * 100
        print(f"  {grp:<25}  {true_lbl:<12}  {rf_debris_pct:>10.2f}%  {gb_debris_pct:>10.2f}%")
        group_results.append({
            "group": grp, "true_label": true_lbl,
            "rf_debris_pct": rf_debris_pct, "gb_debris_pct": gb_debris_pct,
            "n_objects": mask.sum(),
        })

    group_df = pd.DataFrame(group_results)

    # STEP 6: VISUALISATIONS
    print("\n  Generating cross-year plots...")

    # Plot 1: Accuracy Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    models  = ["RF 2024 Val", "GB 2024 Val", "RF 2026 Cross", "GB 2026 Cross"]
    accs    = [rf_val_acc * 100, gb_val_acc * 100, rf_cross_acc * 100, gb_cross_acc * 100]
    colors  = ["#1E88E5", "#43A047", "#E53935", "#FB8C00"]
    bars    = ax.bar(models, accs, color=colors, edgecolor="white", width=0.5)
    ax.set_ylim(min(accs) - 2, 101)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("2024 Validation vs 2026 Cross-Year Accuracy", fontweight="bold")
    for bar, val in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{val:.2f}%", ha="center", va="bottom", fontweight="bold")
    plt.tight_layout()
    plt.savefig("outputs/cross_year_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Plot 2: Per-group classification
    fig, ax = plt.subplots(figsize=(12, 6))
    x      = np.arange(len(group_df))
    width  = 0.35
    bars1  = ax.bar(x - width / 2, group_df["rf_debris_pct"], width, label="Random Forest", color="#1E88E5")
    bars2  = ax.bar(x + width / 2, group_df["gb_debris_pct"], width, label="Gradient Boosting", color="#43A047")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['group']}\n({r['true_label']})" for _, r in group_df.iterrows()], rotation=15)
    ax.set_ylabel("% Predicted as Debris")
    ax.set_title("Per-Group Prediction on 2026 Data", fontweight="bold")
    ax.set_ylim(0, 110)
    ax.legend()
    plt.tight_layout()
    plt.savefig("outputs/cross_year_per_group.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("  Cross-year plots saved in outputs/")
