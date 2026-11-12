"""Classification performance by number of top-ranked genera (panel size), nested cross-validation.

Input
  S1_data_processing/results/discovery_genus_relab_cut0.01.csv   genus relative abundance (%)
  S1_data_processing/results/split_sample_ids.csv                training set (n = 420)
Output (S4_panel_size/results/)
  panel_size_cv_split_metrics.csv   metrics per split and panel size k

Process (repeated stratified five-fold cross-validation, 10 repeats = 50 splits)
  1) In each training fold (n = 336), a random forest (default hyperparameters, balanced class weights,
     random_state 42) is fitted on all 783 genera and the genera are ranked by impurity-based importance.
  2) For each k in 1-15, 20, 25, 30, 40, 50, 60, 75, 100, 125, 150, 200, 300 and 783, a random forest with the
     same settings is fitted on the top-k genera of that training fold and evaluated on the validation fold (n = 84).

Usage
  python S4_1_panel_size_nested_cv.py --workers 4
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier

PKG = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, os.path.join(PKG, 'common'))
from analysis_common import SEED, classification_metrics, cv_splitter, load_split  # noqa: E402

OUT = os.path.join(PKG, 'S4_panel_size', 'results')
K_GRID = list(range(1, 16)) + [20, 25, 30, 40, 50, 60, 75, 100, 125, 150, 200, 300, 783]
RF = RandomForestClassifier(class_weight='balanced', n_jobs=1, random_state=SEED)


def run_split(split_i, tr, va, X, y):
    X_tr, y_tr, X_va, y_va = X.iloc[tr], y[tr], X.iloc[va], y[va]
    ranking = list(X.columns[np.argsort(clone(RF).fit(X_tr, y_tr).feature_importances_)[::-1]])
    rows = []
    for k in K_GRID:
        feats = ranking[:k]
        m = classification_metrics(y_va, clone(RF).fit(X_tr[feats], y_tr).predict_proba(X_va[feats]))
        rows.append({'split': split_i, 'repeat': (split_i - 1) // 5 + 1, 'k': k, **m})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()
    X, y, _ = load_split('train')
    splits = list(cv_splitter().split(X, y))
    t0 = time.time()
    out = Parallel(n_jobs=args.workers)(delayed(run_split)(i, tr, va, X, y) for i, (tr, va) in enumerate(splits, start=1))
    os.makedirs(OUT, exist_ok=True)
    pd.DataFrame([r for o in out for r in o]).to_csv(os.path.join(OUT, 'panel_size_cv_split_metrics.csv'), index=False)
    print(f'{len(splits)} splits x {len(K_GRID)} panel sizes in {time.time() - t0:.0f} s')


if __name__ == '__main__':
    main()
