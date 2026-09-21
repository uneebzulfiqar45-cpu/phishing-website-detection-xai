import pandas as pd
import numpy as np
import joblib
import json

df = pd.read_csv('dataset_B_05_2020.csv')
df2 = df.drop(columns=['url'])
df2['status'] = df2['status'].map({'legitimate': 0, 'phishing': 1})
X = df2.drop(columns=['status'])
y = df2['status']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

rf_best = joblib.load('rf_model.pkl')

from lime.lime_tabular import LimeTabularExplainer
explainer = LimeTabularExplainer(
    X_train.values,
    feature_names=X_train.columns.tolist(),
    class_names=['legitimate', 'phishing'],
    mode='classification',
    random_state=42
)

rf_test_pred = rf_best.predict(X_test)

# correctly classified examples (same selection logic as original script)
phishing_idx = X_test[(y_test.values == 1) & (rf_test_pred == 1)].index[0]
legit_idx = X_test[(y_test.values == 0) & (rf_test_pred == 0)].index[0]

# misclassified examples (same as before: first phishing-as-legit, first legit-as-phishing)
mis_mask = (rf_test_pred != y_test.values)
mis_idx_all = X_test.index[mis_mask]
phishing_as_legit = [i for i in mis_idx_all if y_test.loc[i] == 1]
legit_as_phishing = [i for i in mis_idx_all if y_test.loc[i] == 0]

cases = {
    'fig7_phishing_correct': phishing_idx,
    'fig8_legit_correct': legit_idx,
    'fig9_misclassified_phishing': phishing_as_legit[0],
    'fig10_misclassified_legit': legit_as_phishing[0],
}

results = {}
for label, idx in cases.items():
    row = X_test.loc[idx]
    proba = rf_best.predict_proba([row.values])[0]
    exp = explainer.explain_instance(row.values, rf_best.predict_proba, labels=(1,), num_features=8)
    lime_list = exp.as_list(label=1)
    results[label] = {
        'row_index': int(idx),
        'true_label': int(y_test.loc[idx]),
        'proba_phishing': float(proba[1]),
        'lime_list': lime_list
    }
    print(label, 'P(phishing)=', round(proba[1]*100,1))
    for cond, w in lime_list:
        print(f'   {cond:35s} {w:+.4f}')

with open('lime_full_8_results.json', 'w') as f:
    json.dump(results, f, indent=2)
