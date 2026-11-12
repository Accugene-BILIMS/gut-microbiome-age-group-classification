"""External validation of the final 10-genus random forest in the independent cohort.

Input
  S6_final_model_holdout_markers/model/final_rf.joblib            final random forest and its 10 genera
  S1_data_processing/results/external_genus_relab_cut0.01.csv    external cohort genus relative abundance (%)
  metadata/metadata_external.csv                                 age group of the external cohort
  S1_data_processing/results/discovery_genus_relab_cut0.01.csv, split_sample_ids.csv   training set (n = 420)
Output (S7_external_validation/results/)
  external_predictions.csv           predicted probabilities per participant
  external_metrics.csv               macro-AUC, per-group AUC, accuracy, balanced accuracy (macro sensitivity),
                                     macro specificity and F1-score with stratified bootstrap 95% CI
  external_confusion.csv             confusion matrix
  external_balanced_subsampling.csv  metrics in 30 class-balanced subsamples (15 per age group)
  panel_detection.csv                detection rate and median relative abundance of each panel genus,
                                     external cohort vs training set

Process
  1) The final model is applied without retraining, tuning or recalibration; genera not detected in the external
     cohort are 0.
  2) 95% CIs from 2,000 stratified bootstrap resamples (resampling within each age group, numpy default_rng seed 42).
  3) Supplementary: 30 subsamples of 15 participants per age group drawn without replacement (numpy default_rng
     seed 42); the subsamples reuse the same participants.

Usage
  python S7_1_external_validation.py
"""
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

PKG = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, os.path.join(PKG, 'common'))
from analysis_common import CLASSES, SEED, classification_metrics, load_external, load_split  # noqa: E402

MODEL = os.path.join(PKG, 'S6_final_model_holdout_markers', 'model', 'final_rf.joblib')
RES = os.path.join(PKG, 'S7_external_validation', 'results')
N_BOOT = 2000
N_SUB, PER_CLASS = 30, 15


def short(name):
    parts = name.split(';')
    return parts[-1] if parts[-1] != '__' else parts[-2] + ';__'


def stratified_bootstrap(y, prob, rng):
    idx_by_class = [np.where(y == c)[0] for c in range(len(CLASSES))]
    reps = []
    for _ in range(N_BOOT):
        idx = np.concatenate([rng.choice(ix, size=len(ix), replace=True) for ix in idx_by_class])
        reps.append(classification_metrics(y[idx], prob[idx]))
    return pd.DataFrame(reps)


def main():
    os.makedirs(RES, exist_ok=True)
    bundle = joblib.load(MODEL)
    model, feats = bundle['model'], bundle['features']
    X_tr, _, _ = load_split('train')
    X_ext_all, y = load_external(list(X_tr.columns))
    X = X_ext_all[feats]
    print(f'external cohort n = {len(y)}, age groups {dict(zip(CLASSES, np.bincount(y)))}')

    det = pd.DataFrame({
        'genus': [short(f) for f in feats],
        'detected_external_%': [(X[f] > 0).mean() * 100 for f in feats],
        'detected_train_%': [(X_tr[f] > 0).mean() * 100 for f in feats],
        'median_external_%abund': [X[f].median() for f in feats],
        'median_train_%abund': [X_tr[f].median() for f in feats],
    }).round(3)
    det.to_csv(os.path.join(RES, 'panel_detection.csv'), index=False)

    prob = model.predict_proba(X)
    pred = prob.argmax(1)
    point = classification_metrics(y, prob)
    boot = stratified_bootstrap(y, prob, np.random.default_rng(SEED))
    table = pd.DataFrame({'metric': list(point), 'estimate': list(point.values()),
                          'ci_low': [boot[m].quantile(0.025) for m in point],
                          'ci_high': [boot[m].quantile(0.975) for m in point]}).round(4)
    table['metric'] = table['metric'].replace({'sensitivity_macro': 'balanced_accuracy (macro sensitivity)'})
    table.to_csv(os.path.join(RES, 'external_metrics.csv'), index=False)
    cm = confusion_matrix(y, pred, labels=range(len(CLASSES)))
    pd.DataFrame(cm, index=[f'true_{c}' for c in CLASSES], columns=[f'pred_{c}' for c in CLASSES]) \
        .to_csv(os.path.join(RES, 'external_confusion.csv'))
    pd.DataFrame(prob, columns=[f'p_{c}' for c in CLASSES], index=X.index).assign(
        true=[CLASSES[i] for i in y], pred=[CLASSES[i] for i in pred]).to_csv(os.path.join(RES, 'external_predictions.csv'))
    for _, r in table.iterrows():
        print(f"{r['metric']:<38} {r['estimate']:.3f}  [{r['ci_low']:.3f}, {r['ci_high']:.3f}]")

    rng = np.random.default_rng(SEED)
    sub = []
    for i in range(N_SUB):
        idx = np.concatenate([rng.choice(np.where(y == c)[0], size=PER_CLASS, replace=False) for c in range(len(CLASSES))])
        sub.append({'draw': i + 1, **classification_metrics(y[idx], prob[idx])})
    pd.DataFrame(sub).to_csv(os.path.join(RES, 'external_balanced_subsampling.csv'), index=False)


if __name__ == '__main__':
    main()
