"""Stability of the panel genera across the training folds of the nested cross-validation.

Input
  S1_data_processing/results/discovery_genus_relab_cut0.01.csv   genus relative abundance (%)
  S1_data_processing/results/split_sample_ids.csv                training set (n = 420)
  S6_final_model_holdout_markers/results/final_panel.csv
Output (S6_final_model_holdout_markers/results/)
  panel_stability_fold_top10.csv   per genus: number of the 50 training folds in which it ranked in the top 10

Process
  In each of the 50 training folds (repeated stratified five-fold cross-validation, 10 repeats, random_state 42;
  n = 336), a random forest (default hyperparameters, balanced class weights, random_state 42) is fitted on all
  783 genera and the 10 genera with the highest impurity-based importance are recorded.

Usage
  python S6_2_panel_stability.py
"""
import os
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier

PKG = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, os.path.join(PKG, 'common'))
from analysis_common import SEED, cv_splitter, load_split  # noqa: E402

RES = os.path.join(PKG, 'S6_final_model_holdout_markers', 'results')
K = 10


def top_k(X, y, tr):
    rf = RandomForestClassifier(class_weight='balanced', n_jobs=1, random_state=SEED).fit(X.iloc[tr], y[tr])
    return list(X.columns[np.argsort(rf.feature_importances_)[::-1][:K]])


def main():
    X, y, _ = load_split('train')
    tops = Parallel(n_jobs=6)(delayed(top_k)(X, y, tr) for tr, _ in cv_splitter().split(X, y))
    counts = pd.Series([f for t in tops for f in t]).value_counts()
    panel = set(pd.read_csv(os.path.join(RES, 'final_panel.csv'))['taxonomy'])
    out = pd.DataFrame({'genus': [f.split(';')[-1] for f in counts.index], 'fold_top10_of_50': counts.values,
                        'in_final_panel': [f in panel for f in counts.index]})
    out.to_csv(os.path.join(RES, 'panel_stability_fold_top10.csv'), index=False)
    print(out.head(15).to_string(index=False))


if __name__ == '__main__':
    main()
