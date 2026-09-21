# 🛡️ AI-Based Phishing Website Detection Using Explainable Machine Learning (XAI)

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-%23139CFF.svg)](https://xgboost.ai/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainable%20AI-success.svg)](https://shap.readthedocs.io/en/latest/)

> **Code accompanying the research paper on AI-Based Phishing Detection.** This repository implements Random Forest and XGBoost models on the Hannousse & Yahiouche dataset, utilizing SHAP (global) and LIME (local) to demystify and explain model predictions.

---

## 📑 Table of Contents
- [Overview](#-overview)
- [Project Setup](#-project-setup)
- [Data Requirements](#-data-requirements)
- [Pipeline & Scripts](#-pipeline--scripts)
- [Important Notes](#-important-notes)

---

## 🔍 Overview
This project focuses on the intersection of cybersecurity and Explainable AI (XAI). We train state-of-the-art machine learning models (Random Forest and XGBoost) to detect phishing URLs and subsequently apply **SHAP** and **LIME** to ensure these models remain transparent and interpretable.

---

## ⚙️ Project Setup

Clone the repository and install the required dependencies:

```bash
# Clone the repository
git clone https://github.com/uneebzulfiqar45-cpu/phishing-website-detection-xai.git
cd phishing-website-detection-xai

# Install dependencies
pip install -r requirements.txt
```

*(Alternatively, you can install the packages directly: `pip install pandas numpy scikit-learn xgboost shap lime statsmodels joblib`)*

---

## 📊 Data Requirements

The models rely on the following datasets, which are not included in this repository due to size constraints. Please download them and place them in the project's root directory:

1. **Hannousse & Yahiouche Phishing Website Dataset**  
   - **File:** `dataset_B_05_2020.csv`  
   - **Source:** [Mendeley Data (DOI: 10.17632/c2gw7fy2j4.3)](https://data.mendeley.com/datasets/c2gw7fy2j4/3)
2. **PhiUSIIL Phishing URL Dataset** *(Only needed for Script 04)*  
   - **File:** `PhiUSIIL_Phishing_URL_Dataset.csv`  
   - **Source:** [UCI Machine Learning Repository (Dataset ID: 967)](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset)

---

## 🚀 Pipeline & Scripts

The scripts are designed to be run sequentially. Each is self-contained provided the datasets are present.

| Step | Script | Description |
| :--- | :--- | :--- |
| **01** | `phishing_detection_full_pipeline.py` | Loads and preprocesses data, trains RF & XGBoost (5-fold grid search), evaluates, runs McNemar's test, and generates initial SHAP/LIME figures. |
| **02** | `01_full_shap_full_test_set.py` | Recomputes SHAP importance on the complete 2,286-instance test set. Requires `rf_model.pkl` and `xgb_model.pkl` generated in Step 1. |
| **03** | `02_misclassified_case_analysis.py` | Runs LIME on individual misclassified instances, analyzing the actual feature values driving the incorrect predictions. |
| **04** | `03_lime_correct_and_misclassified_examples.py` | Regenerates the 8-feature LIME explanations for illustrative correct and misclassified cases. |
| **05** | `04_cross_dataset_check_phiusiil.py` | Performs a partial cross-dataset check. Trains a restricted 8-feature model on Hannousse data and evaluates it against a PhiUSIIL sample to compare accuracy and SHAP ranking. |

---

## 💡 Important Notes

- **Reproducibility:** All scripts utilize `random_state=42` throughout the codebase.
- **LIME Stability:** LIME's local surrogate employs random perturbation sampling. Even with a fixed seed, slight shifts in weight values may occur across different environments or execution orders.
- **Model Artifacts:** Pre-trained model files and specific train/test indices are excluded from the repository to save space. Running Script 1 from start to finish will regenerate them.

---
*Created for robust and interpretable phishing website detection research.*
