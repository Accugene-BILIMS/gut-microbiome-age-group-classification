"""Nested cross-validation with genus selection and hyperparameter tuning inside each training fold.

Input
  S1_data_processing/results/discovery_genus_relab_cut0.01.csv   genus relative abundance (%)
  S1_data_processing/results/split_sample_ids.csv                training set (n = 420)
Output (S5_nested_cv_tuning/results/)
  tuning_cv_split_metrics.csv      metrics per split, panel size k and model (default or tuned random forest)
  tuning_best_params.csv           tuned hyperparameters and inner cross-validation AUC per split and k
  tuning_oof_predictions_k10.csv   validation-fold probabilities of the tuned 10-genus random forest

Process (outer loop: repeated stratified five-fold cross-validation, 10 repeats = 50 splits)
  Within each training fold (n = 336) only:
  1) Random forest (default hyperparameters, balanced class weights, random_state 42) on all 783 genera;
     genera ranked by impurity-based importance; top k retained (k = 10, 15, 100, 783).
  2) Optuna (TPE sampler, seed 42 + split number, 30 trials) tunes a random forest (300 trees, balanced class
     weights) on the top-k genera, maximizing the mean macro-AUC of an inner stratified five-fold
     cross-validation (shuffle, random_state 42 + split number). Search space: max_depth 3-20,
     min_samples_split 2-40, min_samples_leaf 1-20, max_leaf_nodes 8-128 (log scale), max_features
     {sqrt, log2, 0.5}.
  3) The random forest with the best hyperparameters is refitted on the training fold and evaluated once on the
     validation fold (n = 84). A default random forest on the same top-k genera is evaluated as a reference.
  The validation fold is used in none of steps 1-3.

Usage
  python S5_1_nested_cv_tuning.py --workers 6
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import optuna
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

PKG = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, os.path.join(PKG, 'common'))
from analysis_common import SEED, classification_metrics, cv_splitter, load_split  # noqa: E402

OUT = os.path.join(PKG, 'S5_nested_cv_tuning', 'results')
K_LIST = [10, 15, 100, 783]
OOF_K = 10
N_TRIALS = 30
N_ESTIMATORS = 300
DEFAULT_RF = RandomForestClassifier(class_weight='balanced', n_jobs=1, random_state=SEED)


def suggest_params(trial):
    return {
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 40),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
        'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 8, 128, log=True),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.5]),
    }


def tuned_rf(params):
    return RandomForestClassifier(n_estimators=N_ESTIMATORS, class_weight='balanced', n_jobs=1,
                                  random_state=SEED, **params)


def tune(X, y, split_i):
    inner = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED + split_i).split(X, y))

    def objective(trial):
        model = tuned_rf(suggest_params(trial))
        scores = []
        for tr, va in inner:
            prob = clone(model).fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[va])
            scores.append(roc_auc_score(y[va], prob, multi_class='ovr', average='macro'))
        return float(np.mean(scores))

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED + split_i))
    study.optimize(objective, n_trials=N_TRIALS)
    return study.best_params, study.best_value


def run_split(split_i, tr, va, X, y):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    t0 = time.time()
    X_tr, y_tr, X_va, y_va = X.iloc[tr], y[tr], X.iloc[va], y[va]
    ranking = list(X.columns[np.argsort(clone(DEFAULT_RF).fit(X_tr, y_tr).feature_importances_)[::-1]])
    rows, params, preds = [], [], []
    for k in K_LIST:
        feats = ranking[:k]
        base = {'split': split_i, 'repeat': (split_i - 1) // 5 + 1, 'k': k}
        prob = clone(DEFAULT_RF).fit(X_tr[feats], y_tr).predict_proba(X_va[feats])
        rows.append({**base, 'model': 'RF_default', **classification_metrics(y_va, prob)})
        best, inner_auc = tune(X_tr[feats], y_tr, split_i)
        prob = tuned_rf(best).fit(X_tr[feats], y_tr).predict_proba(X_va[feats])
        rows.append({**base, 'model': 'RF_tuned', 'inner_cv_auc': inner_auc, **classification_metrics(y_va, prob)})
        params.append({**base, 'inner_cv_auc': inner_auc, 'params': json.dumps(best)})
        if k == OOF_K:
            preds += [{**base, 'Sample_name': s, 'true': int(t), **{f'p_{c}': p for c, p in enumerate(pr)}}
                      for s, t, pr in zip(X_va.index, y_va, prob)]
    print(f'split {split_i:>2} done in {time.time() - t0:.0f} s', flush=True)
    return rows, params, preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=6)
    args = ap.parse_args()

    X, y, _ = load_split('train')
    splits = list(cv_splitter().split(X, y))
    t0 = time.time()
    out = Parallel(n_jobs=args.workers)(delayed(run_split)(i, tr, va, X, y) for i, (tr, va) in enumerate(splits, start=1))

    os.makedirs(OUT, exist_ok=True)
    fold = pd.DataFrame([r for o in out for r in o[0]])
    fold.to_csv(os.path.join(OUT, 'tuning_cv_split_metrics.csv'), index=False)
    pd.DataFrame([p for o in out for p in o[1]]).to_csv(os.path.join(OUT, 'tuning_best_params.csv'), index=False)
    pd.DataFrame([p for o in out for p in o[2]]).to_csv(os.path.join(OUT, 'tuning_oof_predictions_k10.csv'), index=False)
    print(f'finished {len(out)} splits in {time.time() - t0:.0f} s')
    print(fold.groupby(['k', 'model'])['auc_macro'].agg(['mean', 'std']).round(3).to_string())


if __name__ == '__main__':
    main()
