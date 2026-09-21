"""
AI-Based Phishing Website Detection Using Explainable Machine Learning
Full pipeline: data loading -> preprocessing -> Random Forest & XGBoost training
-> evaluation -> SHAP (global) -> LIME (local) -> McNemar's test -> charts

Dataset: Hannousse & Yahiouche phishing website dataset (Mendeley Data)
DOI: 10.17632/c2gw7fy2j4.3
11,430 URLs, 87 features, 50/50 phishing/legitimate balance
"""

import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import xgboost as xgb
import shap
from lime.lime_tabular import LimeTabularExplainer
from statsmodels.stats.contingency_tables import mcnemar
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ============================================================
# STEP 1: LOAD AND PREPROCESS THE DATASET
# ============================================================

# Download dataset_B_05_2020.csv from:
# https://data.mendeley.com/datasets/c2gw7fy2j4/3
df = pd.read_csv('dataset_B_05_2020.csv')

# Drop the URL column (not a numeric feature), encode the label
df = df.drop(columns=['url'])
df['status'] = df['status'].map({'legitimate': 0, 'phishing': 1})

X = df.drop(columns=['status'])
y = df['status']

print("Dataset shape:", X.shape)
print("Class balance:", y.value_counts().to_dict())

# Stratified 80/20 split, keeps the 50/50 class balance intact in both splits
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print("Train shape:", X_train.shape, "Test shape:", X_test.shape)


# ============================================================
# STEP 2: TRAIN RANDOM FOREST (5-fold CV grid search)
# ============================================================

rf_param_grid = {
    'n_estimators': [200, 400],
    'max_depth': [None, 20],
    'min_samples_split': [2, 5]
}
rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    rf_param_grid, cv=5, scoring='f1', n_jobs=-1
)
rf_grid.fit(X_train, y_train)
rf_best = rf_grid.best_estimator_
print("Random Forest best params:", rf_grid.best_params_)


# ============================================================
# STEP 3: TRAIN XGBOOST (5-fold CV grid search)
# ============================================================

xgb_param_grid = {
    'n_estimators': [200, 400],
    'max_depth': [4, 6],
    'learning_rate': [0.05, 0.1]
}
xgb_grid = GridSearchCV(
    xgb.XGBClassifier(random_state=42, eval_metric='logloss', n_jobs=-1),
    xgb_param_grid, cv=5, scoring='f1', n_jobs=-1
)
xgb_grid.fit(X_train, y_train)
xgb_best = xgb_grid.best_estimator_
print("XGBoost best params:", xgb_grid.best_params_)


# ============================================================
# STEP 4: EVALUATE BOTH MODELS
# ============================================================

results = {}
for name, model in [('random_forest', rf_best), ('xgboost', xgb_best)]:
    pred = model.predict(X_test)
    results[name] = {
        'accuracy': accuracy_score(y_test, pred),
        'precision': precision_score(y_test, pred),
        'recall': recall_score(y_test, pred),
        'f1': f1_score(y_test, pred),
        'confusion_matrix': confusion_matrix(y_test, pred).tolist()
    }
    print(name, results[name])

with open('results.json', 'w') as f:
    json.dump(results, f, indent=2)

joblib.dump(rf_best, 'rf_model.pkl')
joblib.dump(xgb_best, 'xgb_model.pkl')


# ============================================================
# STEP 5: McNEMAR'S TEST (statistical comparison of the two models)
# ============================================================

rf_pred = rf_best.predict(X_test)
xgb_pred = xgb_best.predict(X_test)

rf_correct = (rf_pred == y_test.values)
xgb_correct = (xgb_pred == y_test.values)

both_correct = int(((rf_correct) & (xgb_correct)).sum())
rf_only = int(((rf_correct) & (~xgb_correct)).sum())      # RF right, XGBoost wrong
xgb_only = int(((~rf_correct) & (xgb_correct)).sum())      # XGBoost right, RF wrong
both_wrong = int(((~rf_correct) & (~xgb_correct)).sum())

table = [[both_correct, rf_only], [xgb_only, both_wrong]]
mcnemar_result = mcnemar(table, exact=False, correction=True)

print("McNemar contingency table:", table)
print("McNemar statistic:", mcnemar_result.statistic)
print("McNemar p-value:", mcnemar_result.pvalue)
# Result used in the paper: chi2(1) = 5.92, p = 0.015


# ============================================================
# STEP 6: SHAP - GLOBAL EXPLANATION
# ============================================================

# Sample 300 of the 2,286 test instances (fixed seed) to keep SHAP
# computation tractable while remaining representative
sample_idx = X_test.sample(n=300, random_state=42).index
X_sample = X_test.loc[sample_idx]

shap_results = {}
for name, model in [('random_forest', rf_best), ('xgboost', xgb_best)]:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    if isinstance(shap_values, list):
        sv = shap_values[1]
    elif np.array(shap_values).ndim == 3:
        sv = shap_values[:, :, 1]
    else:
        sv = shap_values
    mean_abs_shap = np.abs(sv).mean(axis=0)
    importance = pd.Series(mean_abs_shap, index=X_sample.columns).sort_values(ascending=False)
    shap_results[name] = importance.head(15).to_dict()
    importance.to_csv(f'shap_importance_{name}.csv')

with open('shap_top_features.json', 'w') as f:
    json.dump(shap_results, f, indent=2)


# ============================================================
# STEP 7: LIME - LOCAL EXPLANATION (individual predictions)
# ============================================================

explainer = LimeTabularExplainer(
    X_train.values,
    feature_names=X_train.columns.tolist(),
    class_names=['legitimate', 'phishing'],
    mode='classification',
    random_state=42
)

# Pick one correctly classified phishing example and one legitimate example
rf_test_pred = rf_best.predict(X_test)
phishing_idx = X_test[(y_test.values == 1) & (rf_test_pred == 1)].index[0]
legit_idx = X_test[(y_test.values == 0) & (rf_test_pred == 0)].index[0]

lime_results = {}
for label, idx in [('phishing_example', phishing_idx), ('legitimate_example', legit_idx)]:
    row = X_test.loc[idx]
    # labels=(1,) explicitly: both explanations describe the phishing-class (class 1)
    # probability, so a positive weight always means "pushes toward phishing" and a
    # negative weight always means "pushes toward legitimate" in both charts.
    exp = explainer.explain_instance(row.values, rf_best.predict_proba, labels=(1,), num_features=8)
    lime_results[label] = {
        'row_index': int(idx),
        'true_label': int(y_test.loc[idx]),
        'rf_prediction_proba': rf_best.predict_proba([row.values])[0].tolist(),
        'lime_explanation': exp.as_list(label=1)
    }

with open('lime_examples.json', 'w') as f:
    json.dump(lime_results, f, indent=2)


# ============================================================
# STEP 8: CHARTS
# ============================================================

# Chart 1: model comparison bar chart
metrics = ['accuracy', 'precision', 'recall', 'f1']
rf_vals = [results['random_forest'][m] for m in metrics]
xgb_vals = [results['xgboost'][m] for m in metrics]

x = np.arange(len(metrics))
width = 0.35
fig, ax = plt.subplots(figsize=(7, 5))
b1 = ax.bar(x - width/2, rf_vals, width, label='Random Forest', color='#4C72B0')
b2 = ax.bar(x + width/2, xgb_vals, width, label='XGBoost', color='#DD8452')
ax.set_ylim(0.9, 1.0)
ax.set_ylabel('Score')
ax.set_title('Model Performance Comparison')
ax.set_xticks(x)
ax.set_xticklabels(['Accuracy', 'Precision', 'Recall', 'F1-score'])
ax.legend()
for bars in [b1, b2]:
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f'{h:.3f}', xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 3),
                    textcoords='offset points', ha='center', fontsize=9)
plt.tight_layout()
plt.savefig('model_comparison_chart.png', dpi=200)
plt.close()

# Chart 2: SHAP global feature importance (XGBoost, top 15)
xgb_feat = pd.Series(shap_results['xgboost']).sort_values()
fig, ax = plt.subplots(figsize=(7, 6))
ax.barh(xgb_feat.index, xgb_feat.values, color='#55A868')
ax.set_xlabel('Mean |SHAP value|')
ax.set_title('SHAP Global Feature Importance - XGBoost (Top 15)')
plt.tight_layout()
plt.savefig('shap_importance_xgboost.png', dpi=200)
plt.close()

# Charts 3 & 4: LIME explanations (phishing and legitimate examples)
for key, title, fname in [
    ('phishing_example', 'LIME Explanation - Phishing URL', 'lime_phishing_example.png'),
    ('legitimate_example', 'LIME Explanation - Legitimate URL', 'lime_legitimate_example.png')
]:
    exp = lime_results[key]['lime_explanation']
    labels = [e[0] for e in exp][::-1]
    values = [e[1] for e in exp][::-1]
    colors = ['#DD8452' if v > 0 else '#4C72B0' for v in values]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(labels, values, color=colors)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('LIME weight (contribution to prediction)')
    ax.set_title(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(fname, dpi=200)
    plt.close()

print("Pipeline complete. All results, models, and charts saved.")
