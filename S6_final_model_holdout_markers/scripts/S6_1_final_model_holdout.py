"""Final 10-genus model on the training set and single evaluation on the hold-out set.

Input
  S1_data_processing/results/discovery_genus_relab_cut0.01.csv   genus relative abundance (%)
  S1_data_processing/results/split_sample_ids.csv                training set (n = 420), hold-out set (n = 60)
Output
  S6_final_model_holdout_markers/results/
    final_genus_ranking.csv      importance ranking of all 783 genera on the training set
    final_panel.csv              the 10 genera of the panel
    final_model_params.json      tuned hyperparameters, 5-fold CV AUC of the tuning, panel genera
    holdout_predictions.csv      predicted probabilities per hold-out participant
    holdout_metrics.csv          point estimates with stratified bootstrap 95% CI (2,000 resamples)
    holdout_confusion.csv        confusion matrix
  S6_final_model_holdout_markers/model/final_rf.joblib   final random forest (model, features, classes)

Process
  1) Random forest (default hyperparameters, balanced class weights, random_state 42) on all 783 genera of the
     training set; genera ranked by impurity-based importance; the top 10 form the panel.
  2) Optuna (TPE sampler, seed 42, 30 trials) tunes a random forest (300 trees, balanced class weights) on the
     panel, maximizing the mean macro-AUC of a stratified five-fold cross-validation of the training set
     (shuffle, random_state 42). Search space: max_depth 3-20, min_samples_split 2-40, min_samples_leaf 1-20,
     max_leaf_nodes 8-128 (log scale), max_features {sqrt, log2, 0.5}.
  3) The random forest with the best hyperparameters is fitted on the whole training set.
  4) The hold-out set, not used before, is evaluated once; 95% CIs by stratified bootstrap (resampling within
     each age group, numpy default_rng seed 42).

Usage
  python S6_1_final_model_holdout.py
"""
import json
import os
import sys

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold

PKG = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, os.path.join(PKG, 'common'))
from analysis_common import CLASSES, SEED, classification_metrics, load_split  # noqa: E402

STEP = os.path.join(PKG, 'S6_final_model_holdout_markers')
RES, MODEL_DIR = os.path.join(STEP, 'results'), os.path.join(STEP, 'model')
K = 10
N_TRIALS = 30
N_ESTIMATORS = 300
N_BOOT = 2000
DEFAULT_RF = RandomForestClassifier(class_weight='balanced', n_jobs=-1, random_state=SEED)


def short(name):
    parts = name.split(';')
    return parts[-1] if parts[-1] != '__' else parts[-2] + ';__'


def tune(X, y):
    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(X, y))

    def objective(trial):
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 40),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
            'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 8, 128, log=True),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.5]),
        }
        model = RandomForestClassifier(n_estimators=N_ESTIMATORS, class_weight='balanced', n_jobs=-1,
                                       random_state=SEED, **params)
        scores = [roc_auc_score(y[va], clone(model).fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[va]),
                                multi_class='ovr', average='macro') for tr, va in folds]
        return float(np.mean(scores))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=N_TRIALS)
    return study.best_params, study.best_value


def bootstrap_ci(y, prob, n_boot=N_BOOT):
    rng = np.random.default_rng(SEED)
    idx_by_class = [np.where(y == c)[0] for c in range(len(CLASSES))]
    reps = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(ix, size=len(ix), replace=True) for ix in idx_by_class])
        reps.append(classification_metrics(y[idx], prob[idx]))
    return pd.DataFrame(reps)


def main():
    os.makedirs(RES, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    X, y, _ = load_split('train')
    X_ho, y_ho, ho_ids = load_split('holdout')

    # 1) panel
    imp = clone(DEFAULT_RF).fit(X, y).feature_importances_
    ranking = pd.DataFrame({'taxonomy': X.columns, 'genus': [short(c) for c in X.columns], 'importance': imp})
    ranking = ranking.sort_values('importance', ascending=False).reset_index(drop=True)
    ranking.insert(0, 'rank', ranking.index + 1)
    ranking.to_csv(os.path.join(RES, 'final_genus_ranking.csv'), index=False)
    panel = ranking.head(K).copy()
    panel.to_csv(os.path.join(RES, 'final_panel.csv'), index=False)
    feats = panel['taxonomy'].tolist()
    print(panel[['rank', 'genus', 'importance']].round(4).to_string(index=False))

    # 2-3) tuning and final fit
    best, cv_auc = tune(X[feats], y)
    with open(os.path.join(RES, 'final_model_params.json'), 'w') as fh:
        json.dump({'params': best, 'n_estimators': N_ESTIMATORS, 'cv5_auc_on_train': cv_auc, 'panel': feats}, fh, indent=2)
    model = RandomForestClassifier(n_estimators=N_ESTIMATORS, class_weight='balanced', n_jobs=-1,
                                   random_state=SEED, **best).fit(X[feats], y)
    joblib.dump({'model': model, 'features': feats, 'classes': CLASSES}, os.path.join(MODEL_DIR, 'final_rf.joblib'))
    print(f'best hyperparameters: {best}')

    # 4) hold-out, evaluated once
    prob = model.predict_proba(X_ho[feats])
    point = classification_metrics(y_ho, prob)
    boot = bootstrap_ci(y_ho, prob)
    table = pd.DataFrame({'metric': list(point), 'estimate': list(point.values()),
                          'ci_low': [boot[m].quantile(0.025) for m in point],
                          'ci_high': [boot[m].quantile(0.975) for m in point]}).round(4)
    table.to_csv(os.path.join(RES, 'holdout_metrics.csv'), index=False)
    pd.DataFrame(prob, columns=[f'p_{c}' for c in CLASSES]).assign(
        Sample_name=ho_ids, true=[CLASSES[i] for i in y_ho], pred=[CLASSES[i] for i in prob.argmax(1)]
    ).to_csv(os.path.join(RES, 'holdout_predictions.csv'), index=False)
    cm = pd.DataFrame(confusion_matrix(y_ho, prob.argmax(1)), index=[f'true_{c}' for c in CLASSES],
                      columns=[f'pred_{c}' for c in CLASSES])
    cm.to_csv(os.path.join(RES, 'holdout_confusion.csv'))
    for _, r in table.iterrows():
        print(f"{r['metric']:<18} {r['estimate']:.3f}  [{r['ci_low']:.3f}, {r['ci_high']:.3f}]")


if __name__ == '__main__':
    main()
