import pandas as pd
import numpy as np
import joblib
import json

df = pd.read_csv('dataset_B_05_2020.csv')
urls = df['url']
df2 = df.drop(columns=['url'])
df2['status'] = df2['status'].map({'legitimate': 0, 'phishing': 1})
X = df2.drop(columns=['status'])
y = df2['status']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

rf_best = joblib.load('rf_model.pkl')

rf_pred = rf_best.predict(X_test)
rf_proba = rf_best.predict_proba(X_test)[:, 1]

mis_mask = (rf_pred != y_test.values)
mis_idx_all = X_test.index[mis_mask]
print("Total RF misclassified:", len(mis_idx_all))

# split into phishing-labeled-legit and legit-labeled-phishing
phishing_as_legit = [i for i in mis_idx_all if y_test.loc[i] == 1]
legit_as_phishing = [i for i in mis_idx_all if y_test.loc[i] == 0]
print("Phishing predicted as legit:", len(phishing_as_legit))
print("Legit predicted as phishing:", len(legit_as_phishing))

from lime.lime_tabular import LimeTabularExplainer
explainer = LimeTabularExplainer(
    X_train.values,
    feature_names=X_train.columns.tolist(),
    class_names=['legitimate', 'phishing'],
    mode='classification',
    random_state=42
)

def explain_case(idx):
    row = X_test.loc[idx]
    proba = rf_best.predict_proba([row.values])[0]
    exp = explainer.explain_instance(row.values, rf_best.predict_proba, labels=(1,), num_features=8)
    lime_list = exp.as_list(label=1)
    # get actual feature values for the features that appear in the explanation
    feat_vals = []
    for feat_desc, weight in lime_list:
        # feat_desc like "google_index <= 0.00" - extract base feature name
        base_feat = None
        for col in X.columns:
            if col in feat_desc:
                if base_feat is None or len(col) > len(base_feat):
                    base_feat = col
        actual_val = row[base_feat] if base_feat else None
        feat_vals.append({'condition': feat_desc, 'feature': base_feat, 'actual_value': float(actual_val) if actual_val is not None else None, 'lime_weight': weight})
    return {
        'row_index': int(idx),
        'url': urls.loc[idx],
        'true_label': int(y_test.loc[idx]),
        'rf_pred_proba_phishing': float(proba[1]),
        'features': feat_vals
    }

# pick first two examples of each kind for illustration
results = {}
results['misclassified_phishing_1'] = explain_case(phishing_as_legit[0])
results['misclassified_phishing_2'] = explain_case(phishing_as_legit[1])
results['misclassified_legit_1'] = explain_case(legit_as_phishing[0])

with open('misclassified_lime_details.json', 'w') as f:
    json.dump(results, f, indent=2)

for k, v in results.items():
    print("====", k, "====")
    print("URL:", v['url'])
    print("True label:", v['true_label'], "Pred P(phishing):", round(v['rf_pred_proba_phishing']*100,1), "%")
    for f in v['features']:
        print(f"  {f['condition']:35s} actual={f['actual_value']:<10} weight={f['lime_weight']:+.4f}")
