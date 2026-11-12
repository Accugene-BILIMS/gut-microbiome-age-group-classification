"""Comparison of five classifiers on the training set.

Input
  S1_data_processing/results/discovery_genus_relab_cut0.01.csv   genus relative abundance (%)
  S1_data_processing/results/split_sample_ids.csv                training set (n = 420)
Output (S2_algorithm_comparison/results/)
  classifier_cv_split_metrics.csv   metrics of each classifier in each of the 50 cross-validation splits
  TableS1_classifier_comparison.csv mean and SD of each metric across the 50 splits (Supplementary Table S1)

Process
  1) Features: all 783 genera; default hyperparameters; no feature selection.
  2) Classifiers: SVM (RBF kernel, balanced class weights), KNN, decision tree (balanced class weights),
     random forest (balanced class weights) and XGBoost (multi:softprob); random_state 42.
     SVM and KNN use standardization fitted on the training fold only; tree models use relative abundances.
  3) Repeated stratified five-fold cross-validation, 10 repeats (50 splits); each classifier is fitted on the
     training fold and evaluated on the validation fold (macro and per-group one-vs-rest AUC, accuracy,
     macro sensitivity, specificity and F1-score).

Usage
  python S2_1_algorithm_comparison.py
"""
import os
import sys
import time

import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

PKG = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, os.path.join(PKG, 'common'))
from analysis_common import CLASSES, SEED, classification_metrics, cv_splitter, load_split, summarize  # noqa: E402

OUT = os.path.join(PKG, 'S2_algorithm_comparison', 'results')

MODELS = {
    'SVM': make_pipeline(StandardScaler(), SVC(probability=True, class_weight='balanced', random_state=SEED)),
    'KNN': make_pipeline(StandardScaler(), KNeighborsClassifier()),
    'DT': DecisionTreeClassifier(class_weight='balanced', random_state=SEED),
    'RF': RandomForestClassifier(class_weight='balanced', n_jobs=-1, random_state=SEED),
    'XGB': XGBClassifier(objective='multi:softprob', num_class=len(CLASSES), eval_metric='mlogloss',
                         random_state=SEED, verbosity=0, n_jobs=-1),
}
METRICS = ['auc_macro', 'accuracy', 'sensitivity_macro', 'specificity_macro', 'f1_macro'] + [f'auc_{c}' for c in CLASSES]


def main():
    X, y, _ = load_split('train')
    print(f'training set: {X.shape[0]} participants x {X.shape[1]} genera', flush=True)
    rows = []
    t0 = time.time()
    for split_i, (tr, va) in enumerate(cv_splitter().split(X, y), start=1):
        for name, base in MODELS.items():
            model = clone(base).fit(X.iloc[tr], y[tr])
            m = classification_metrics(y[va], model.predict_proba(X.iloc[va]))
            rows.append({'split': split_i, 'repeat': (split_i - 1) // 5 + 1, 'model': name, **m})
        if split_i % 10 == 0:
            print(f'split {split_i}/50 ({time.time() - t0:.0f} s)', flush=True)

    os.makedirs(OUT, exist_ok=True)
    fold = pd.DataFrame(rows)
    fold.to_csv(os.path.join(OUT, 'classifier_cv_split_metrics.csv'), index=False)
    summary = summarize(fold, ['model'], METRICS).sort_values('auc_macro_mean', ascending=False)
    summary.to_csv(os.path.join(OUT, 'TableS1_classifier_comparison.csv'), index=False)

    for _, r in summary.iterrows():
        print(f"{r['model']:>3}  macro-AUC {r['auc_macro_mean']:.3f} ± {r['auc_macro_sd']:.3f}  "
              f"[10s {r['auc_10s_mean']:.3f} | 20s_40s {r['auc_20s_40s_mean']:.3f} | 50s_plus {r['auc_50s_plus_mean']:.3f}]")


if __name__ == '__main__':
    main()
