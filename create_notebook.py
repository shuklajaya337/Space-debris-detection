#!/usr/bin/env python3
"""
create_notebook.py
==================
Generates the complete Jupyter Notebook for the Space Debris Detection project:
Writes to both detection.ipynb and Space_Debris_Complete.ipynb in the current directory.
"""

import json
import os

_cell_id_counter = [0]

def _new_id():
    _cell_id_counter[0] += 1
    return f'cell_{_cell_id_counter[0]:04d}'

def code_cell(src):
    lines = src.strip('\n').split('\n')
    source = [l + '\n' for l in lines[:-1]] + [lines[-1]]
    return {
        'cell_type': 'code',
        'execution_count': None,
        'id': _new_id(),
        'metadata': {},
        'outputs': [],
        'source': source
    }

def md_cell(src):
    lines = src.strip('\n').split('\n')
    source = [l + '\n' for l in lines[:-1]] + [lines[-1]]
    return {
        'cell_type': 'markdown',
        'id': _new_id(),
        'metadata': {},
        'source': source
    }

cells = []

# ==============================================================================
# SECTION 0: TITLE & OVERVIEW
# ==============================================================================
cells.append(md_cell(r"""# 🛰️ Space Debris Detection & Cross-Year Validation
An end-to-end Machine Learning pipeline for **Space Debris Classification and Temporal Validation**.

Developed with high precision to classify orbital objects as active satellites or space debris using raw Two-Line Element (TLE) datasets from **2024 (OLD_dataset)** and **2026 (26_Data set)**.

---

## 📌 Project Architecture
1. **Physics-based Feature Extraction**: Leveraging SGP4 propagation to extract 14 orbital parameters from raw TLE files.
2. **Exploratory Data Analysis**: Standard visualisations (altitude, velocity, inclination-eccentricity) and a **3D Interactive Geocentric Orbital Plot**.
3. **Imbalance Mitigation**: Applying SMOTE (Synthetic Minority Over-sampling Technique) to address class imbalance.
4. **Machine Learning Classifiers**: Comparing Random Forest (unbalanced/balanced) and Gradient Boosting.
5. **Temporal Cross-Year Validation**: Training models on 2024 data and testing them on 2026 data to evaluate real-world generalization.
6. **Per-Group Analysis**: Investigating classification performance across debris clouds (Cosmos 1408, Cosmos 2251, FengYun 1C, Iridium 33) and active satellites.
"""))

# ==============================================================================
# SECTION 1: IMPORTS
# ==============================================================================
cells.append(md_cell(r"""## Section 1: Setup & Libraries"""))
cells.append(code_cell(r"""import os
import warnings
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timezone
from sgp4.api import Satrec, jday
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from mpl_toolkits.mplot3d import Axes3D

warnings.filterwarnings('ignore')
plt.rcParams['figure.dpi'] = 100
sns.set_theme(style='whitegrid')
print("All libraries imported successfully!")"""))

# ==============================================================================
# SECTION 2: DATA LOCATIONS
# ==============================================================================
cells.append(md_cell(r"""## Section 2: Data Locations & Constants
Setting up directory paths and file name mappings for 2024 and 2026 TLE files."""))
cells.append(code_cell(r"""# Directory paths
DIR_2024 = "OLD_dataset"
DIR_2026 = "26_Data set"

# File mappings (2024 dataset has different file names)
FILES_2024 = {
    'cosmos_1408'      : 'cosmos-1408-debris.txt',
    'cosmos_2251'      : 'cosmos-2251-debris.txt',
    'fengyun_1c'       : 'fengyun-1c-debris.txt',
    'iridium_33'       : 'iridium-33-debris.txt',
    'active_satellites': 'active-sate.txt',
}

# 2026 dataset names
FILES_2026 = {
    'cosmos_1408'      : 'cosmos_1408_2026.txt',
    'cosmos_2251'      : 'cosmos_2251_2026.txt',
    'fengyun_1c'       : 'fengyun_1c_2026.txt',
    'iridium_33'       : 'iridium_33_2026.txt',
    'active_satellites': 'active_satellites_2026.txt',
}

POSITIVE_GROUPS = ['cosmos_1408', 'cosmos_2251', 'fengyun_1c', 'iridium_33']
NEGATIVE_GROUPS = ['active_satellites']

GROUP_COLORS = {
    'cosmos_1408'       : '#E53935',
    'cosmos_2251'       : '#FB8C00',
    'fengyun_1c'        : '#8E24AA',
    'iridium_33'        : '#1E88E5',
    'active_satellites' : '#43A047',
}

GROUP_LABELS = {
    'cosmos_1408'       : 'Cosmos 1408 Debris',
    'cosmos_2251'       : 'Cosmos 2251 Debris',
    'fengyun_1c'        : 'FengYun 1C Debris',
    'iridium_33'        : 'Iridium 33 Debris',
    'active_satellites' : 'Active Satellites',
}

EARTH_RADIUS = 6371.0  # km
MU = 398600.4418       # km^3/s^2

# 14 orbital features used in classifiers
FEATURE_COLS = [
    'eccentricity', 'inclination', 'raan', 'arg_perigee',
    'mean_anomaly', 'mean_motion', 'semi_major_axis',
    'altitude', 'perigee', 'apogee', 'orbital_period',
    'distance_from_center', 'velocity_magnitude', 'specific_orbital_energy'
]
print("Configurations and metadata established.")"""))

# ==============================================================================
# SECTION 3: TLE PARSING
# ==============================================================================
cells.append(md_cell(r"""## Section 3: SGP4 Feature Extraction Engine
A physics-driven engine using the SGP4 orbital propagator to parse two-line element sets (TLE) and calculate exact positions, velocities, and derived orbital characteristics."""))
cells.append(code_cell(r"""def parse_tle_file(filepath, label, group_key, eval_time):
    if not os.path.exists(filepath):
        print(f"  [WARN] Path '{filepath}' not found, skipping.")
        return []

    jd_now, fr_now = jday(eval_time.year, eval_time.month, eval_time.day,
                          eval_time.hour, eval_time.minute, eval_time.second)

    with open(filepath, 'r', encoding='utf-8', errors='ignore') as fh:
        lines = [l.rstrip() for l in fh if l.strip()]

    records = []
    for i in range(0, len(lines) - 2, 3):
        name  = lines[i].strip()
        line1 = lines[i + 1].strip()
        line2 = lines[i + 2].strip()
        if not (line1.startswith('1 ') and line2.startswith('2 ')):
            continue
        try:
            sat = Satrec.twoline2rv(line1, line2)
            ecc = sat.ecco
            inc = sat.inclo
            raan = sat.nodeo
            argp = sat.argpo
            ma = sat.mo
            mm = sat.no_kozai          # rev/day

            # Physics calculations
            mm_rad_s = mm * 2 * np.pi / 86400.0
            sma = (MU / mm_rad_s**2)**(1/3)
            altitude = sma - EARTH_RADIUS
            perigee = sma * (1 - ecc)
            apogee = sma * (1 + ecc)
            orb_period = 86400.0 / mm

            e, r, v = sat.sgp4(jd_now, fr_now)
            if e != 0:
                continue  # Skip objects with propagation error

            dist_c = np.sqrt(r[0]**2 + r[1]**2 + r[2]**2)
            vel_mag = np.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
            soe = 0.5 * vel_mag**2 - MU / dist_c

            records.append({
                'sat_name': name,
                'eccentricity': ecc,
                'inclination': inc,
                'raan': raan,
                'arg_perigee': argp,
                'mean_anomaly': ma,
                'mean_motion': mm,
                'semi_major_axis': sma,
                'altitude': altitude,
                'perigee': perigee,
                'apogee': apogee,
                'orbital_period': orb_period,
                'distance_from_center': dist_c,
                'velocity_magnitude': vel_mag,
                'specific_orbital_energy': soe,
                'label': label,
                'source_file': group_key,
                'x': r[0], 'y': r[1], 'z': r[2]
            })
        except Exception:
            continue

    return records
print("Physics parser defined.")"""))

# ==============================================================================
# SECTION 4: INGESTION OF 2024 DATA
# ==============================================================================
cells.append(md_cell(r"""## Section 4: Load & Parse 2024 Dataset
Extracting 14 orbital features from raw files in `OLD_dataset` evaluated at a snapshot date (17-Nov-2024)."""))
cells.append(code_cell(r"""# Fixed evaluation timestamp for 2024 dataset
EVAL_2024 = datetime(2024, 11, 17, 18, 0, 0, tzinfo=timezone.utc)

records_2024 = []
for grp in POSITIVE_GROUPS + NEGATIVE_GROUPS:
    lbl = 1 if grp in POSITIVE_GROUPS else 0
    path = os.path.join(DIR_2024, FILES_2024[grp])
    recs = parse_tle_file(path, lbl, grp, EVAL_2024)
    records_2024.extend(recs)
    tag = "DEBRIS" if lbl == 1 else "ACTIVE SAT"
    print(f"  Loaded {grp:<25s} -> {len(recs):>5d} objects [{tag}]")

df_2024 = pd.DataFrame(records_2024)
print(f"\nTotal 2024 Records: {len(df_2024)}")
print(f"Debris (label=1)  : {(df_2024['label']==1).sum()}")
print(f"Active (label=0)  : {(df_2024['label']==0).sum()}")"""))

# ==============================================================================
# SECTION 5: EDA
# ==============================================================================
cells.append(md_cell(r"""## Section 5: Exploratory Data Analysis (EDA)
Visualising how orbital inclination, altitude, and velocity differ between active satellites and debris."""))
cells.append(code_cell(r"""# Plot 1: Inclination vs Eccentricity
fig, ax = plt.subplots(figsize=(10, 6))
for grp in POSITIVE_GROUPS + NEGATIVE_GROUPS:
    subset = df_2024[df_2024['source_file'] == grp]
    ax.scatter(np.degrees(subset['inclination']), subset['eccentricity'],
               s=5, alpha=0.4, label=GROUP_LABELS[grp], color=GROUP_COLORS[grp])
ax.set_xlabel('Inclination (degrees)', fontsize=12)
ax.set_ylabel('Eccentricity', fontsize=12)
ax.set_title('Inclination vs Eccentricity (2024 Dataset)', fontsize=14, fontweight='bold')
ax.legend(markerscale=4, fontsize=9)
plt.tight_layout()
plt.show()

# Plot 2: Altitude Distribution
fig, ax = plt.subplots(figsize=(10, 5))
sns.histplot(data=df_2024, x='altitude', hue='label', bins=60, alpha=0.6, kde=True, ax=ax,
             palette={0:'#43A047', 1:'#E53935'})
ax.set_xlabel('Altitude (km)')
ax.set_title('Altitude Distribution: Active vs Debris', fontsize=14, fontweight='bold')
ax.get_legend().set_title('Class (0=Active, 1=Debris)')
plt.tight_layout()
plt.show()

# Plot 3: Velocity Magnitude
fig, ax = plt.subplots(figsize=(10, 5))
sns.histplot(data=df_2024, x='velocity_magnitude', hue='label', bins=60, alpha=0.6, kde=True, ax=ax,
             palette={0:'#43A047', 1:'#E53935'})
ax.set_xlabel('Velocity Magnitude (km/s)')
ax.set_title('Velocity Magnitude: Active vs Debris', fontsize=14, fontweight='bold')
ax.get_legend().set_title('Class (0=Active, 1=Debris)')
plt.tight_layout()
plt.show()"""))

# ==============================================================================
# SECTION 6: 3D ORBITS
# ==============================================================================
cells.append(md_cell(r"""## Section 6: 3D Orbital Visualisation
Mapping positional coordinates (X, Y, Z) of space debris clouds and active satellites around Earth."""))
cells.append(code_cell(r"""fig = plt.figure(figsize=(12, 9))
ax3d = fig.add_subplot(111, projection='3d')

for grp in POSITIVE_GROUPS + NEGATIVE_GROUPS:
    subset = df_2024[df_2024['source_file'] == grp]
    sample = subset.sample(min(250, len(subset)), random_state=42)
    ax3d.scatter(sample['x'], sample['y'], sample['z'],
                 s=2, alpha=0.35, color=GROUP_COLORS[grp], label=GROUP_LABELS[grp])

# Translucent Earth Sphere
u = np.linspace(0, 2 * np.pi, 40)
v = np.linspace(0, np.pi, 40)
xe = EARTH_RADIUS * np.outer(np.cos(u), np.sin(v))
ye = EARTH_RADIUS * np.outer(np.sin(u), np.sin(v))
ze = EARTH_RADIUS * np.outer(np.ones_like(u), np.cos(v))
ax3d.plot_surface(xe, ye, ze, color='royalblue', alpha=0.15, linewidth=0)

ax3d.set_xlabel('X (km)')
ax3d.set_ylabel('Y (km)')
ax3d.set_zlabel('Z (km)')
ax3d.set_title('3D Orbital Mapping of Space Debris and Satellites', pad=15, fontweight='bold')
ax3d.legend(loc='upper left', markerscale=6, fontsize=8)
plt.tight_layout()
plt.show()"""))

# ==============================================================================
# SECTION 7: TRAINING INITIAL MODELS
# ==============================================================================
cells.append(md_cell(r"""## Section 7: Train/Test Split (2024 Data)
Dividing features and labels into stratified train (80%) and validation (20%) sets."""))
cells.append(code_cell(r"""X = df_2024[FEATURE_COLS].values
y = df_2024['label'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training subset shape  : {X_train.shape}")
print(f"Validation subset shape: {X_test.shape}")"""))

# ==============================================================================
# SECTION 8: RANDOM FOREST
# ==============================================================================
cells.append(md_cell(r"""## Section 8: Random Forest (Unbalanced Data)
Training a Random Forest Classifier on raw class distributions and evaluating accuracy."""))
cells.append(code_cell(r"""rf = RandomForestClassifier(n_estimators=200, random_state=42, oob_score=True, n_jobs=-1)
rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)
print(f"Random Forest validation accuracy: {rf_acc*100:.4f}%")
print(f"Out-of-Bag (OOB) Score           : {rf.oob_score_*100:.4f}%")
print("\n=== Classification Report ===")
print(classification_report(y_test, rf_pred, target_names=["Active Satellite", "Debris"]))"""))

# ==============================================================================
# SECTION 9: SMOTE
# ==============================================================================
cells.append(md_cell(r"""## Section 9: Handling Class Imbalance via SMOTE
Balancing the minority class (debris) by synthetically over-sampling to build an unbiased model."""))
cells.append(code_cell(r"""smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
print(f"Before SMOTE - Debris: {y_train.sum()}, Active: {len(y_train) - y_train.sum()}")
print(f"After SMOTE  - Debris: {y_train_sm.sum()}, Active: {len(y_train_sm) - y_train_sm.sum()}")

rf_smote = RandomForestClassifier(n_estimators=200, random_state=42, oob_score=True, n_jobs=-1)
rf_smote.fit(X_train_sm, y_train_sm)

rf_sm_pred = rf_smote.predict(X_test)
rf_sm_acc = accuracy_score(y_test, rf_sm_pred)
print(f"RF (SMOTE Balanced) validation accuracy: {rf_sm_acc*100:.4f}%")
print(f"Balanced OOB Score                      : {rf_smote.oob_score_*100:.4f}%")"""))

# ==============================================================================
# SECTION 10: GRADIENT BOOSTING
# ==============================================================================
cells.append(md_cell(r"""## Section 10: Gradient Boosting Classifier
Training an alternative ensemble model using Gradient Boosting for comparison."""))
cells.append(code_cell(r"""gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
gb.fit(X_train, y_train)

gb_pred = gb.predict(X_test)
gb_acc = accuracy_score(y_test, gb_pred)
print(f"Gradient Boosting validation accuracy: {gb_acc*100:.4f}%")
print("\n=== Classification Report ===")
print(classification_report(y_test, gb_pred, target_names=["Active Satellite", "Debris"]))"""))

# ==============================================================================
# SECTION 11: MODEL COMPARISON
# ==============================================================================
cells.append(md_cell(r"""## Section 11: Model Comparison (2024 Data)
Comparing the performance of Random Forest (Unbalanced), Random Forest (SMOTE), and Gradient Boosting."""))
cells.append(code_cell(r"""models = ['RF (Unbalanced)', 'RF (SMOTE)', 'Gradient Boosting']
accuracies = [rf_acc * 100, rf_sm_acc * 100, gb_acc * 100]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(models, accuracies, color=['#1E88E5', '#43A047', '#FB8C00'], edgecolor='white', width=0.4)
ax.set_ylim(min(accuracies) - 2, 101)
ax.set_ylabel('Validation Accuracy (%)')
ax.set_title('Classifier Performance Comparison (2024 Validation)', fontweight='bold')
for bar, val in zip(bars, accuracies):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            f"{val:.4f}%", ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.show()"""))

# ==============================================================================
# SECTION 12: LOADING 2026 DATA
# ==============================================================================
cells.append(md_cell(r"""## Section 12: Load & Parse 2026 Test Dataset
Ingesting 2026 TLE files from `26_Data set` evaluated at the current epoch to enable cross-year prediction validation."""))
cells.append(code_cell(r"""EVAL_2026 = datetime(2026, 5, 28, 0, 0, 0, tzinfo=timezone.utc)

records_2026 = []
for grp in POSITIVE_GROUPS + NEGATIVE_GROUPS:
    lbl = 1 if grp in POSITIVE_GROUPS else 0
    path = os.path.join(DIR_2026, FILES_2026[grp])
    recs = parse_tle_file(path, lbl, grp, EVAL_2026)
    records_2026.extend(recs)
    tag = "DEBRIS" if lbl == 1 else "ACTIVE SAT"
    print(f"  Loaded 2026 {grp:<25s} -> {len(recs):>5d} objects [{tag}]")

df_2026 = pd.DataFrame(records_2026)
print(f"\nTotal 2026 Records: {len(df_2026)}")
print(f"Debris (label=1)  : {(df_2026['label']==1).sum()}")
print(f"Active (label=0)  : {(df_2026['label']==0).sum()}")"""))

# ==============================================================================
# SECTION 13: CROSS-YEAR VALIDATION
# ==============================================================================
cells.append(md_cell(r"""## Section 13: Cross-Year Prediction
Validating whether models trained on 2024 features generalise to class predictions on 2026 features."""))
cells.append(code_cell(r"""X_2026 = df_2026[FEATURE_COLS].values
y_2026 = df_2026['label'].values

rf_pred_26 = rf.predict(X_2026)
rf_cross_acc = accuracy_score(y_2026, rf_pred_26)

gb_pred_26 = gb.predict(X_2026)
gb_cross_acc = accuracy_score(y_2026, gb_pred_26)

print(f"Random Forest 2026 Cross-Year Accuracy       : {rf_cross_acc*100:.4f}%")
print(f"Gradient Boosting 2026 Cross-Year Accuracy   : {gb_cross_acc*100:.4f}%")

print("\n=== Random Forest Classification Report on 2026 ===")
print(classification_report(y_2026, rf_pred_26, target_names=["Active Satellite", "Debris"]))

print("=== Gradient Boosting Classification Report on 2026 ===")
print(classification_report(y_2026, gb_pred_26, target_names=["Active Satellite", "Debris"]))"""))

# ==============================================================================
# SECTION 14: PER-GROUP VALIDATION
# ==============================================================================
cells.append(md_cell(r"""## Section 14: Per-Group Accuracy & Tracking
Determining the exact percentage of objects in each target category (debris groups and active satellites) predicted as debris in 2026."""))
cells.append(code_cell(r"""group_results = []
print(f"{'Group Name':<25s}  {'Ground Truth':<12s}  {'RF Debris%':>11s}  {'GB Debris%':>11s}")
print("-" * 65)

for grp in POSITIVE_GROUPS + NEGATIVE_GROUPS:
    mask = df_2026['source_file'] == grp
    true_label = "DEBRIS" if grp in POSITIVE_GROUPS else "ACTIVE"
    subset_X = df_2026.loc[mask, FEATURE_COLS].values
    if len(subset_X) == 0:
        continue
    rf_pct = rf.predict(subset_X).mean() * 100
    gb_pct = gb.predict(subset_X).mean() * 100
    print(f"{grp:<25s}  {true_label:<12s}  {rf_pct:>10.2f}%  {gb_pct:>10.2f}%")
    group_results.append({
        'group': grp, 'true': true_label, 'rf': rf_pct, 'gb': gb_pct
    })
group_df = pd.DataFrame(group_results)"""))

# ==============================================================================
# SECTION 15: VISUALISATIONS
# ==============================================================================
cells.append(md_cell(r"""## Section 15: Temporal Validation Plots
Plotting cross-year validation performance metrics."""))
cells.append(code_cell(r"""# Plot 1: 2024 Val vs 2026 Cross-Year Accuracy
fig, ax = plt.subplots(figsize=(10, 6))
labels_bar = ["RF 2024 Val", "GB 2024 Val", "RF 2026 Cross", "GB 2026 Cross"]
vals_bar = [rf_acc * 100, gb_acc * 100, rf_cross_acc * 100, gb_cross_acc * 100]
colors_bar = ["#1E88E5", "#43A047", "#E53935", "#FB8C00"]
bars = ax.bar(labels_bar, vals_bar, color=colors_bar, edgecolor="white", width=0.5)
ax.set_ylim(min(vals_bar) - 2, 101)
ax.set_ylabel("Accuracy (%)")
ax.set_title("2024 Validation vs 2026 Cross-Year Accuracy", fontweight="bold")
for bar, val in zip(bars, vals_bar):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            f"{val:.2f}%", ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.show()

# Plot 2: Per-Group Predictions
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(group_df))
width = 0.35
b1 = ax.bar(x - width/2, group_df['rf'], width, label="Random Forest", color="#1E88E5")
b2 = ax.bar(x + width/2, group_df['gb'], width, label="Gradient Boosting", color="#43A047")
ax.set_xticks(x)
ax.set_xticklabels([f"{r['group']}\n({r['true']})" for _, r in group_df.iterrows()], fontsize=9)
ax.set_ylabel("% Predicted as Debris")
ax.set_title("Per-Group Prediction Rates on 2026 Data", fontweight="bold")
ax.set_ylim(0, 115)
ax.axhline(50, color="red", linestyle="--", alpha=0.4, label="Decision Threshold")
ax.legend()
for bar in list(b1) + list(b2):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f"{bar.get_height():.1f}%", ha='center', va='bottom', fontsize=8)
plt.tight_layout()
plt.show()"""))

# ==============================================================================
# SECTION 16: CONCLUSION
# ==============================================================================
cells.append(md_cell(r"""## Section 16: Model Persistence & Key Findings
Saving classifiers to disk and summarizing cross-year performance stats."""))
cells.append(code_cell(r"""# Save models
os.makedirs('outputs', exist_ok=True)
joblib.dump(rf, 'outputs/rf_model_complete.pkl')
joblib.dump(gb, 'outputs/gb_model_complete.pkl')

print("Models persisted successfully:")
print("  - outputs/rf_model_complete.pkl")
print("  - outputs/gb_model_complete.pkl")

print("\n=== FINAL EXECUTIVE SUMMARY ===")
print(f"2024 Dataset Size      : {len(df_2024)} objects")
print(f"2026 Dataset Size      : {len(df_2026)} objects")
print(f"RF 2024 Val Accuracy   : {rf_acc*100:.3f}%")
print(f"GB 2024 Val Accuracy   : {gb_acc*100:.3f}%")
print(f"RF 2026 Cross Accuracy : {rf_cross_acc*100:.3f}%")
print(f"GB 2026 Cross Accuracy : {gb_cross_acc*100:.3f}%")"""))

# Construct notebook JSON
notebook = {
    'cells': cells,
    'metadata': {
        'kernelspec': {
            'display_name': 'Python 3',
            'language': 'python',
            'name': 'python3',
        },
        'language_info': {
            'name': 'python',
            'version': '3.10.0',
        },
    },
    'nbformat': 4,
    'nbformat_minor': 5,
}

# Write to both target filenames in current directory
for output_name in ['detection.ipynb', 'Space_Debris_Complete.ipynb']:
    with open(output_name, 'w', encoding='utf-8') as fh:
        json.dump(notebook, fh, indent=1, ensure_ascii=False)
    print(f"Generated notebook: {output_name}")
