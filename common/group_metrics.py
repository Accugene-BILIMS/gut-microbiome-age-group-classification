"""Per-group performance tables with stratified bootstrap confidence intervals.

Process
  - per_class(): precision, sensitivity, specificity, F1-score and one-vs-rest AUC for each age group, their macro
    averages (macro-AUC from the multiclass one-vs-rest AUC) and accuracy; the predicted class is the argmax.
  - boot_table(): point estimates with 95% CIs from 2,000 stratified bootstrap resamples (resampling within each
    true age group, numpy default_rng seed 42).
  - pivot(): one row per set and age group, one column per metric.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

GROUPS = ['10s', '20s_40s', '50s_plus']
GROUP_LABEL = {'10s': '10s', '20s_40s': '20s–40s', '50s_plus': '50s+'}
SEED, N_BOOT = 42, 2000


def per_class(y, prob):
    pred = prob.argmax(1)
    cm = confusion_matrix(y, pred, labels=range(3))
    out = {}
    for c, g in enumerate(GROUPS):
        tn = cm.sum() - cm[c].sum() - cm[:, c].sum() + cm[c, c]
        out[g] = {'precision': precision_score(y == c, pred == c, zero_division=0),
                  'sensitivity': recall_score(y == c, pred == c, zero_division=0),
                  'specificity': tn / (cm.sum() - cm[c].sum()),
                  'f1': f1_score(y == c, pred == c, zero_division=0),
                  'auc': roc_auc_score(y == c, prob[:, c])}
    out['macro'] = {m: np.mean([out[g][m] for g in GROUPS]) for m in out['10s']}
    out['macro']['auc'] = roc_auc_score(y, prob, multi_class='ovr', average='macro')
    out['macro']['accuracy'] = (pred == y).mean()
    return out


def flat(d):
    return {(k, m): v for k, mm in d.items() for m, v in mm.items()}


def boot_table(y, prob, label):
    point = flat(per_class(y, prob))
    rng = np.random.default_rng(SEED)
    idx_by = [np.where(y == c)[0] for c in range(3)]
    reps = []
    for _ in range(N_BOOT):
        idx = np.concatenate([rng.choice(ix, len(ix), replace=True) for ix in idx_by])
        reps.append(flat(per_class(y[idx], prob[idx])))
    reps = pd.DataFrame(reps)
    rows = []
    for (grp, m), v in point.items():
        lo, hi = reps[(grp, m)].quantile([0.025, 0.975])
        rows.append({'set': label, 'group': grp, 'metric': m, 'value': f'{v:.3f} ({lo:.3f}–{hi:.3f})'})
    return pd.DataFrame(rows)


def pivot(long):
    long = long.copy()
    long['group'] = long['group'].map({**GROUP_LABEL, 'macro': 'Macro average'})
    order = ['precision', 'sensitivity', 'specificity', 'f1', 'auc', 'accuracy']
    t = long.pivot_table(index=['set', 'group'], columns='metric', values='value', aggfunc='first')
    t = t[[c for c in order if c in t.columns]].rename(columns={'f1': 'F1-score', 'auc': 'AUC', 'precision': 'Precision',
                                                               'sensitivity': 'Sensitivity', 'specificity': 'Specificity',
                                                               'accuracy': 'Accuracy'})
    sets = list(dict.fromkeys(long['set']))
    groups = ['10s', '20s–40s', '50s+', 'Macro average']
    return t.reindex(pd.MultiIndex.from_product([sets, groups])).reset_index().rename(columns={'level_0': 'Set', 'level_1': 'Group'}).fillna('')
