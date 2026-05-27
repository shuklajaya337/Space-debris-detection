# Space Debris Detection: Project Workflow & Explanation

This document provides a step-by-step explanation of how the Space Debris Detection Machine Learning pipeline is structured, the logic behind the code, and how data flows through the system. You can use this guide to explain the project to supervisors, examiners, or recruiters.

---

## 1. Data Preparation & Ingestion
**Goal:** Gather the raw data needed to train the model.
* **The Process:** We use raw **Two-Line Element (TLE)** data from CelesTrak. This data is divided into two separate directories:
  - `OLD_dataset/` containing 2024 snapshots (used for model training and internal validation).
  - `26_Data set/` containing 2026 snapshots (used for temporal cross-year validation).
* **Why we did it this way:** Maintaining separate folders for different epochs ensures there is zero temporal leakage between the training data and the validation test sets.

---

## 2. Feature Extraction (The Physics Engine)
**Goal:** Convert raw lines of text (TLEs) into meaningful mathematical numbers for the ML model.
* **The Process:** We utilize the **SGP4 (Simplified General Perturbations)** orbital propagator. TLE data only contains orbital state snapshots; SGP4 physics equations use these parameters to calculate exact physical positions ($X, Y, Z$) and velocity vectors at specific epochs.
* **Extracted Features (14 total):** 
  * Keplerian elements: *Eccentricity, Inclination, RAAN, Argument of Perigee, Mean Anomaly, Mean Motion.*
  * Derived properties: *Semi-major axis, Altitude, Perigee, Apogee, Orbital Period, Distance from Earth's center, Velocity Magnitude, and Specific Orbital Energy.*
* **Why we did it this way:** Machine learning models cannot understand raw text strings. By converting TLEs into physics-based features, the models discover actual physical patterns (e.g., debris clouds reside at specific altitude bands and have higher eccentricities compared to active satellites).

---

## 3. Exploratory Data Analysis (EDA)
**Goal:** Verify visually that the extracted features clearly separate debris from active satellites.
* **Plots Generated:**
  * **Inclination vs Eccentricity (`eda_inc_ecc.png`)**: Shows distinct clusters where active satellites follow tight, controlled orbits, while debris is scattered randomly.
  * **Altitude (`eda_altitude.png`)**: Demonstrates the typical physical zones where debris clouds linger (e.g., LEO altitudes) versus where operational satellites are maintained.
  * **Velocity (`eda_velocity.png`)**: Illustrates the difference in velocities between operational satellites and fragmented debris.

---

## 4. 3D Orbital Visualization
**Goal:** Map the data into a real-world, visual perspective.
* **The Process (`3d_orbit_visualization.png`):** We plot the X, Y, and Z geocentric coordinates (propagated using SGP4) in a 3D coordinate system around a translucent sphere representing Earth.
* **Why we did it this way:** It provides a highly impactful visual demonstration of the "Kessler Syndrome" effect, allowing viewers to see exactly where the debris clouds physically sit relative to active satellite constellations.

---

## 5. Machine Learning Classification
**Goal:** Build an AI model capable of instantly categorizing a space object as "Debris" or "Active" based purely on its orbital characteristics.
* **Model Comparison:** 
  1. **Random Forest Classifier**: Robust, non-linear ensemble tree model.
  2. **SMOTE Balanced Random Forest**: Address class imbalance (more active satellites than debris) using Synthetic Minority Over-sampling Technique (SMOTE).
  3. **Gradient Boosting Classifier**: Sequentially trains weak learners to achieve higher prediction accuracy.
* **2024 Internal Validation Results:**
  - Random Forest: **98.79%** accuracy
  - Gradient Boosting: **99.32%** accuracy

---

## 6. Temporal Cross-Year Validation (2024 Model → 2026 Data)
**Goal:** Evaluate the model's ability to generalize over time.
* **The Process (`cross_year_accuracy.png` / `cross_year_per_group.png`):**
  - Models are trained on **2024 data** (`OLD_dataset/`).
  - Predictions are made on **2026 data** (`26_Data set/`).
* **Cross-Year Accuracy Results:**
  - Random Forest: **99.11%** accuracy
  - Gradient Boosting: **99.44%** accuracy
* **Per-Group Analysis Findings:**
  - Debris clouds are successfully identified as debris in 2026 (Cosmos 2251 and Cosmos 1408 debris are classified with **94%-100% accuracy**).
  - Cosmos 1408 fragments decayed in count from 13 (in 2024) to only 4 (in 2026) due to atmospheric re-entry, demonstrating real-world debris decay.
  - Operational active satellites are classified as active with **>99.4% accuracy**, verifying low false-alarm rates.

---

## Summary of the Pipeline Flow
`Raw TLE Files (2024 & 2026)` ➔ `SGP4 Physics Engine` ➔ `14 Orbital Features` ➔ `Data Visualization` ➔ `Model Training` ➔ `Temporal Cross-Year Prediction` ➔ `Accuracy Reports & Plots`
