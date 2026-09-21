# 🛡️ AI-Based Phishing Website Detection Using Explainable Machine Learning (XAI)

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-%23139CFF.svg)](https://xgboost.ai/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainable%20AI-success.svg)](https://shap.readthedocs.io/en/latest/)

> **Official Code Repository** accompanying the research paper: *"AI-Based Phishing Website Detection Using Explainable Machine Learning"*. 
> This repository provides the complete implementation of the methodology, including data preprocessing, model training, evaluation, and the generation of explainability artifacts using SHAP and LIME.

---

## 📑 Table of Contents
1. [Abstract & Methodology](#-abstract--methodology)
2. [Repository Structure](#-repository-structure)
3. [Environment Setup](#-environment-setup)
4. [Datasets & Preparation](#-datasets--preparation)
5. [Reproducing Results](#-reproducing-results)
6. [Citation & Contact](#-citation--contact)

---

## 📖 Abstract & Methodology

As phishing attacks grow in sophistication, robust and **interpretable** detection mechanisms are essential. This research bridges the gap between high-accuracy machine learning and model transparency. 

We train state-of-the-art predictive models (**Random Forest** and **XGBoost**) on comprehensive phishing URL datasets. To overcome the "black-box" nature of these algorithms, we apply **Explainable AI (XAI)** frameworks:
- **SHAP (SHapley Additive exPlanations):** Used for *global interpretability* to rank the overall importance of URL and website features.
- **LIME (Local Interpretable Model-agnostic Explanations):** Used for *local interpretability* to explain the decision-making process for individual instances, particularly analyzing misclassified cases.

---

## 📂 Repository Structure

The scripts are highly modular and designed to be executed sequentially to reproduce the findings presented in the paper.

| Script / File | Purpose & Output |
| :--- | :--- |
| `phishing_detection_full_pipeline.py` | **Main Pipeline:** Loads data, applies 5-fold grid search for RF and XGBoost, evaluates performance (Accuracy, F1, McNemar's test), and generates base SHAP/LIME figures. |
| `01_full_shap_full_test_set.py` | **Global Interpretability:** Recomputes SHAP importance across the *complete 2,286-instance test set* using the trained models. |
| `02_misclassified_case_analysis.py` | **Error Analysis:** Applies LIME on individually misclassified cases to isolate the exact feature values driving incorrect predictions. |
| `03_lime_correct_and_misclassified_examples.py`| **Local Interpretability:** Regenerates precise 8-feature LIME explanations for illustrative cases highlighted in the paper's figures. |
| `04_cross_dataset_check_phiusiil.py` | **Cross-Dataset Validation:** Trains a restricted 8-feature model on Hannousse data and evaluates its generalizability on a PhiUSIIL sample (Accuracy & SHAP ranking). |

---

## ⚙️ Environment Setup

To guarantee reproducibility, we recommend running this codebase within a virtual environment. 

```bash
# Clone the repository
git clone https://github.com/uneebzulfiqar45-cpu/phishing-website-detection-xai.git
cd phishing-website-detection-xai

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 📊 Datasets & Preparation

Due to size constraints and licensing, the raw datasets are **not included** directly in this repository. To run the pipeline, please download the following datasets and place them in the root directory:

1. **Hannousse & Yahiouche Phishing Website Dataset** (Primary Dataset)
   - **Filename:** `dataset_B_05_2020.csv`  
   - **Source:** [Mendeley Data (DOI: 10.17632/c2gw7fy2j4.3)](https://data.mendeley.com/datasets/c2gw7fy2j4/3)

2. **PhiUSIIL Phishing URL Dataset** (Used for Cross-Validation in Script 04)
   - **Filename:** `PhiUSIIL_Phishing_URL_Dataset.csv`  
   - **Source:** [UCI Machine Learning Repository (Dataset ID: 967)](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset)

---

## 🚀 Reproducing Results

Execute the scripts in the following order. Each script is self-contained (provided the datasets and previous model artifacts exist).

```bash
# 1. Run the primary training and evaluation pipeline
python phishing_detection_full_pipeline.py

# 2. Generate comprehensive SHAP analysis on the full test set
python 01_full_shap_full_test_set.py

# 3. Analyze misclassified edge cases
python 02_misclassified_case_analysis.py

# 4. Generate local LIME explanations for specific examples
python 03_lime_correct_and_misclassified_examples.py

# 5. Validate the model against an independent cross-dataset
python 04_cross_dataset_check_phiusiil.py
```

### 📌 Important Reproducibility Notes
- **Seed States:** All scripts utilize `random_state=42` to ensure consistent data splitting and model initialization.
- **LIME Stability:** Because LIME's surrogate model uses random perturbation sampling, slight numerical shifts in feature weights may occur across different runs or environments, despite fixed seeds. This behavior is documented in the paper's limitations.
- **Artifacts:** Trained `.pkl` model files are generated by the first script and utilized by subsequent scripts.

---

## 📝 Citation & Contact

If you utilize this code or methodology in your research, please consider citing our paper:

```bibtex
@article{zulfiqar2023phishingxai,
  title={AI-Based Phishing Website Detection Using Explainable Machine Learning},
  author={Zulfiqar, Uneeb and [Co-Authors]},
  journal={[Journal/Conference Name]},
  year={[Year]},
  doi={[DOI if available]}
}
```

### ⚖️ Copyright & License

**© 2026 Uneeb Zulfiqar. All Rights Reserved.**

*This code is provided exclusively for the peer-review process of the associated research paper. It may not be copied, modified, distributed, or used in any other research or commercial projects without explicit written permission from the author until the paper is officially published and a formal open-source license is granted.*

*For questions, discussions, or issues related to the code, please open an issue in this repository.*
