"""
Partial cross-dataset check: Hannousse & Yahiouche -> PhiUSIIL
================================================================
google_index and page_rank (this paper's top two SHAP features) have no
equivalent column in PhiUSIIL, so they cannot be tested directly. This
script instead trains a Random Forest on the 8 features that map cleanly
between the two schemas, then evaluates the SAME trained model on a
PhiUSIIL sample, comparing both accuracy and SHAP feature ranking.

Requires:
  - dataset_B_05_2020.csv   (Hannousse & Yahiouche, Mendeley DOI 10.17632/c2gw7fy2j4.3)
  - PhiUSIIL_Phishing_URL_Dataset.csv (UCI ML Repository, id=967)
"""

import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import shap

# Feature name mapping: Hannousse column -> PhiUSIIL column
# Only features with an unambiguous, direct conceptual match are included.
OVERLAP_MAP = {
    'length_url': 'URLLength',
    'length_hostname': 'DomainLength',
    'ip': 'IsDomainIP',
    'nb_subdomains': 'NoOfSubDomain',
    'ratio_digits_url': 'DegitRatioInURL',
    'nb_qm': 'NoOfQMarkInURL',
    'nb_and': 'NoOfAmpersandInURL',
    'nb_eq': 'NoOfEqualsInURL',
}
H_COLS = list(OVERLAP_MAP.keys())


def train_on_hannousse():
    df = pd.read_csv('dataset_B_05_2020.csv')
    df['status'] = df['status'].map({'legitimate': 0, 'phishing': 1})
    X, y = df[H_COLS], df['status']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    rf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    pred = rf.predict(X_test)
    metrics = {
        'accuracy': accuracy_score(y_test, pred),
        'precision': precision_score(y_test, pred),
        'recall': recall_score(y_test, pred),
        'f1': f1_score(y_test, pred),
    }
    return rf, X_test, metrics


def evaluate_on_phiusiil(rf, sample_size=8000):
    dfp = pd.read_csv('PhiUSIIL_Phishing_URL_Dataset.csv')
    dfp['status'] = 1 - dfp['label']  # PhiUSIIL: 1=legitimate; flip to match Hannousse (1=phishing)
    Xp = pd.DataFrame({h: dfp[p] for h, p in OVERLAP_MAP.items()})
    yp = dfp['status']
    Xp_s, _, yp_s, _ = train_test_split(Xp, yp, train_size=sample_size, stratify=yp, random_state=42)
    pred = rf.predict(Xp_s)
    metrics = {
        'accuracy': accuracy_score(yp_s, pred),
        'precision': precision_score(yp_s, pred),
        'recall': recall_score(yp_s, pred),
        'f1': f1_score(yp_s, pred),
    }
    return Xp_s, metrics


def compare_shap_ranking(rf, X_test, Xp_sample, n=300, seed=42):
    explainer = shap.TreeExplainer(rf)

    def rank(X):
        sv = explainer.shap_values(X)
        sv = sv[:, :, 1] if np.array(sv).ndim == 3 else (sv[1] if isinstance(sv, list) else sv)
        return pd.Series(np.abs(sv).mean(axis=0), index=H_COLS).sort_values(ascending=False)

    rank_h = rank(X_test.sample(n=n, random_state=seed))
    rank_p = rank(Xp_sample.sample(n=n, random_state=seed))
    return rank_h, rank_p


if __name__ == '__main__':
    rf, X_test, h_metrics = train_on_hannousse()
    print("Hannousse in-domain (8-feature model):", h_metrics)

    Xp_sample, p_metrics = evaluate_on_phiusiil(rf)
    print("PhiUSIIL cross-dataset (same model):", p_metrics)

    rank_h, rank_p = compare_shap_ranking(rf, X_test, Xp_sample)
    print("\nSHAP ranking - Hannousse:\n", rank_h)
    print("\nSHAP ranking - PhiUSIIL:\n", rank_p)

    with open('cross_dataset_results.json', 'w') as f:
        json.dump({
            'hannousse_metrics': h_metrics,
            'phiusiil_metrics': p_metrics,
            'hannousse_shap_rank': rank_h.to_dict(),
            'phiusiil_shap_rank': rank_p.to_dict(),
        }, f, indent=2)
