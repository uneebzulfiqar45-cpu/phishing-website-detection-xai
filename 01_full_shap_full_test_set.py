import pandas as pd
import numpy as np
import joblib
import json

df = pd.read_csv('dataset_B_05_2020.csv')
df = df.drop(columns=['url'])
df['status'] = df['status'].map({'legitimate': 0, 'phishing': 1})
X = df.drop(columns=['status'])
y = df['status']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print("Test shape:", X_test.shape)

rf_best = joblib.load('rf_model.pkl')
xgb_best = joblib.load('xgb_model.pkl')

import shap

full_shap_results = {}
for name, model in [('random_forest', rf_best), ('xgboost', xgb_best)]:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    if isinstance(shap_values, list):
        sv = shap_values[1]
    elif np.array(shap_values).ndim == 3:
        sv = shap_values[:, :, 1]
    else:
        sv = shap_values
    mean_abs_shap = np.abs(sv).mean(axis=0)
    importance = pd.Series(mean_abs_shap, index=X_test.columns).sort_values(ascending=False)
    full_shap_results[name] = importance.head(15).to_dict()
    importance.to_csv(f'full_shap_importance_{name}.csv')
    print(f"--- {name} top 15 (full {X_test.shape[0]} test instances) ---")
    print(importance.head(15))

with open('full_shap_top_features.json', 'w') as f:
    json.dump(full_shap_results, f, indent=2)

print("Done.")
