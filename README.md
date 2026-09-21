# Phishing Website Detection — XAI Code Repository

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-%23139CFF.svg)](https://xgboost.ai/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainable%20AI-success.svg)](https://shap.readthedocs.io/en/latest/)
[![LIME](https://img.shields.io/badge/LIME-Local%20Explanations-orange.svg)](https://github.com/marcotcr/lime)

---

## 📦 Requirements

Install all dependencies using:

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install pandas numpy scikit-learn xgboost shap lime statsmodels joblib
```

> **Python 3.8 or above** is required.

---

## 📁 Scripts Overview

Run scripts in the following order:

| # | Script | What It Does |
|:--|:-------|:-------------|
| 1 | `phishing_detection_full_pipeline.py` | Main training script. Trains Random Forest & XGBoost with 5-fold cross-validated grid search. Outputs evaluation metrics and saves `rf_model.pkl` and `xgb_model.pkl`. |
| 2 | `01_full_shap_full_test_set.py` | Loads saved models and runs SHAP analysis on the full test set. Requires `.pkl` files from Script 1. |
| 3 | `02_misclassified_case_analysis.py` | Runs LIME on misclassified instances. Reports feature values alongside LIME weights. |
| 4 | `03_lime_correct_and_misclassified_examples.py` | Generates 8-feature LIME explanations for selected correct and misclassified examples. |
| 5 | `04_cross_dataset_check_phiusiil.py` | Trains a restricted 8-feature model and evaluates it on an external dataset sample. |

---

## 🗂️ Data

The scripts expect the following CSV files to be placed in the **root directory** of the project:

- `dataset_B_05_2020.csv` — Used by Scripts 1–4
- `PhiUSIIL_Phishing_URL_Dataset.csv` — Used by Script 5 only

> ⚠️ Dataset files are **not included** in this repository. Please obtain them separately and place them in the project root before running any scripts.

---

## ▶️ How to Run

```bash
# Step 1 — Train models and generate base outputs
python phishing_detection_full_pipeline.py

# Step 2 — Full SHAP analysis on test set
python 01_full_shap_full_test_set.py

# Step 3 — LIME on misclassified cases
python 02_misclassified_case_analysis.py

# Step 4 — LIME explanations for selected examples
python 03_lime_correct_and_misclassified_examples.py

# Step 5 — Cross-dataset evaluation
python 04_cross_dataset_check_phiusiil.py
```

---

## 🔁 Reproducibility

- All scripts use `random_state=42`.
- LIME uses random perturbation sampling — minor numerical variation in weights may occur across environments even with a fixed seed.
- Model files (`rf_model.pkl`, `xgb_model.pkl`) are produced by Script 1 and used by Scripts 2–4.

---

## ⚖️ Copyright

**© 2026 Uneeb Zulfiqar. All Rights Reserved.**

*This code is made available for review purposes only. Reproduction, redistribution, or reuse in any form without explicit written permission from the author is strictly prohibited.*
