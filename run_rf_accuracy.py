#!/usr/bin/env python3
"""
run_rf_accuracy.py
==================
Standalone script to train and evaluate both Random Forest and
Gradient Boosting classifiers on the 2024 TLE data (OLD_dataset),
then print a side-by-side accuracy comparison.
"""

import os
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from datetime import datetime, timezone
from sgp4.api import Satrec, jday
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Constants
EARTH_RADIUS = 6371.0        # km
MU           = 398600.4418   # km^3/s^2 (Earth gravitational parameter)
TLE_DIR      = "OLD_dataset"

FEATURE_COLS = [
    "eccentricity", "inclination", "raan", "arg_perigee",
    "mean_anomaly", "mean_motion", "semi_major_axis",
    "altitude", "perigee", "apogee", "orbital_period",
    "distance_from_center", "velocity_magnitude", "specific_orbital_energy",
]

POSITIVE_GROUPS = ["cosmos_1408", "cosmos_2251", "fengyun_1c", "iridium_33"]
NEGATIVE_GROUPS = ["active_satellites"]

FILE_MAP = {
    "cosmos_1408"       : "cosmos-1408-debris.txt",
    "cosmos_2251"       : "cosmos-2251-debris.txt",
    "fengyun_1c"        : "fengyun-1c-debris.txt",
    "iridium_33"        : "iridium-33-debris.txt",
    "active_satellites" : "active-sate.txt",
}

def parse_tle_file(filepath: str, label: int, group_key: str, eval_time: datetime) -> list:
    """Parse a TLE file and extract 14 orbital features per object using SGP4."""
    if not os.path.exists(filepath):
        print(f"  [WARN] File not found, skipping: {filepath}")
        return []

    records = []
    jd_now, fr_now = jday(
        eval_time.year, eval_time.month, eval_time.day,
        eval_time.hour, eval_time.minute, eval_time.second,
    )

    with open(filepath, "r", encoding="utf-8", errors="ignore") as fh:
        lines = [l.rstrip() for l in fh if l.strip()]

    for i in range(0, len(lines) - 2, 3):
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

            # Derived Keplerian / physical features
            mm_rad_s   = mm * 2 * np.pi / 86400.0
            sma        = (MU / mm_rad_s ** 2) ** (1 / 3)   # km
            altitude   = sma - EARTH_RADIUS                 # km
            perigee    = sma * (1 - ecc)                    # km
            apogee     = sma * (1 + ecc)                    # km
            orb_period = 86400.0 / mm                       # seconds

            e, r, v = sat.sgp4(jd_now, fr_now)
            if e != 0:
                continue  # SGP4 propagation error — skip object

            dist_c  = np.sqrt(r[0]**2 + r[1]**2 + r[2]**2)   # km
            vel_mag = np.sqrt(v[0]**2 + v[1]**2 + v[2]**2)   # km/s
            soe     = 0.5 * vel_mag**2 - MU / dist_c          # specific orbital energy

            records.append({
                "eccentricity"          : ecc,
                "inclination"           : inc,
                "raan"                  : raan,
                "arg_perigee"           : argp,
                "mean_anomaly"          : ma,
                "mean_motion"           : mm,
                "semi_major_axis"       : sma,
                "altitude"              : altitude,
                "perigee"               : perigee,
                "apogee"                : apogee,
                "orbital_period"        : orb_period,
                "distance_from_center"  : dist_c,
                "velocity_magnitude"    : vel_mag,
                "specific_orbital_energy": soe,
                "label"                 : label,
                "source_file"           : group_key,
            })
        except Exception:
            continue

    return records

if __name__ == "__main__":
    print("=" * 60)
    print("  LOADING 2024 DATASET  (OLD_dataset/)")
    print("=" * 60)

    eval_time = datetime(2024, 11, 17, 18, 0, 0, tzinfo=timezone.utc)
    all_records = []
    
    for grp in POSITIVE_GROUPS + NEGATIVE_GROUPS:
        lbl  = 1 if grp in POSITIVE_GROUPS else 0
        tag  = "DEBRIS" if lbl == 1 else "ACTIVE SAT"
        path = os.path.join(TLE_DIR, FILE_MAP[grp])
        recs = parse_tle_file(path, lbl, grp, eval_time)
        all_records.extend(recs)
        print(f"  {grp:<25}  {len(recs):>5} objects  [{tag}]  ({FILE_MAP[grp]})")

    df = pd.DataFrame(all_records)
    print(f"\n  Total   : {len(df)}")
    print(f"  Debris  : {(df['label'] == 1).sum()}")
    print(f"  Active  : {(df['label'] == 0).sum()}")

    # Train / Test Split
    X = df[FEATURE_COLS].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n  Train samples : {len(X_train)}")
    print(f"  Test  samples : {len(X_test)}")

    # Random Forest
    print("\n" + "=" * 60)
    print("  TRAINING: Random Forest Classifier")
    print("=" * 60)
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=None,
        n_jobs=-1, random_state=42, oob_score=True,
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_acc  = accuracy_score(y_test, rf_pred)
    rf_oob  = rf.oob_score_

    print(f"  Test Accuracy : {rf_acc:.6f}  ({rf_acc * 100:.4f}%)")
    print(f"  OOB Score     : {rf_oob:.6f}  ({rf_oob * 100:.4f}%)")
    print()
    print(classification_report(y_test, rf_pred, target_names=["Active Satellite", "Debris"]))

    # Gradient Boosting
    print("=" * 60)
    print("  TRAINING: Gradient Boosting Classifier")
    print("  (This may take some time...)")
    print("=" * 60)
    gb = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.1,
        max_depth=5, random_state=42,
    )
    gb.fit(X_train, y_train)
    gb_pred = gb.predict(X_test)
    gb_acc  = accuracy_score(y_test, gb_pred)

    print(f"  Test Accuracy : {gb_acc:.6f}  ({gb_acc * 100:.4f}%)")
    print()
    print(classification_report(y_test, gb_pred, target_names=["Active Satellite", "Debris"]))

    # Side-by-Side Comparison
    print("=" * 60)
    print("  MODEL COMPARISON — 2024 DATASET")
    print("=" * 60)
    print(f"  {'Model':<30}  {'Accuracy':>10}  {'OOB':>10}")
    print("  " + "-" * 55)
    print(f"  {'Random Forest (200 trees)':<30}  {rf_acc * 100:>9.4f}%  {rf_oob * 100:>9.4f}%")
    print(f"  {'Gradient Boosting (200 est.)':<30}  {gb_acc * 100:>9.4f}%  {'N/A':>10}")
    print("=" * 60)

    winner = "Random Forest" if rf_acc >= gb_acc else "Gradient Boosting"
    print(f"\n  Best model on 2024 test data: {winner}\n")
