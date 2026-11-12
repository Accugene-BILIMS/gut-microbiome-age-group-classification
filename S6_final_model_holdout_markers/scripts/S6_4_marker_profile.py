"""Profile of each panel genus: stability, covariate-adjusted age association and group-specific contribution.

Input
  S1_data_processing/results/discovery_genus_relab_cut0.01.csv   genus relative abundance (%)
  S1_data_processing/results/split_sample_ids.csv                training set (n = 420)
  S6_final_model_holdout_markers/results/final_panel.csv, final_model_params.json, panel_stability_fold_top10.csv,
    panel_associations.csv, panel_associations_independent.csv
Output (S6_final_model_holdout_markers/results/)
  permutation_importance_cv_folds.csv   per split and genus: mean drop in one-vs-rest AUC per age group when permuted
  panel_marker_profile.csv              one row per panel genus (values of Table 3)

Process
  1) Stability: number of the 50 training folds in which the genus ranked in the top 10.
  2) Association type from the MaAsLin2 contrasts in the discovery cohort excluding the hold-out set (q < 0.05):
       progressive             all three contrasts significant with the same direction
       adolescent-associated   10s differs from both adult groups; 20s_40s vs 50s_plus not significant
       older-adult-associated  50s_plus differs from 20s_40s; 10s vs 20s_40s not significant
     with the contrasts reaching q < 0.05 and the direction of each contrast; the 20s_40s vs 50s_plus contrast in
     the participants not used for model development is added.
  3) Group-specific contribution: in repeated stratified five-fold cross-validation of the training set
     (10 repeats, random_state 42), a random forest with the final hyperparameters is fitted on each training fold;
     on the validation fold each genus is permuted 10 times (numpy default_rng seed 42 + split number) and the
     mean drop in one-vs-rest AUC per age group is recorded, then averaged over the 50 splits.
  The hold-out set is not used.

Usage
  python S6_4_marker_profile.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

PKG = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, os.path.join(PKG, 'common'))
from analysis_common import CLASSES, SEED, cv_splitter, load_split  # noqa: E402

RES = os.path.join(PKG, 'S6_final_model_holdout_markers', 'results')
N_PERM = 10
Q = 0.05
LABEL = {'10s_vs_20s_40s': '10s→20s_40s', '10s_vs_50s_plus': '10s→50s+', '20s_40s_vs_50s_plus': '20s_40s→50s+'}


def class_aucs(y, prob):
    return [roc_auc_score(y == c, prob[:, c]) for c in range(len(CLASSES))]


def fold_importance(split_i, tr, va, X, y, feats, params):
    rng = np.random.default_rng(SEED + split_i)
    model = RandomForestClassifier(class_weight='balanced', n_jobs=1, random_state=SEED, **params).fit(X.iloc[tr][feats], y[tr])
    Xv = X.iloc[va][feats].copy()
    base = class_aucs(y[va], model.predict_proba(Xv))
    rows = []
    for f in feats:
        drops = []
        for _ in range(N_PERM):
            Xp = Xv.copy()
            Xp[f] = rng.permutation(Xp[f].values)
            drops.append(np.subtract(base, class_aucs(y[va], model.predict_proba(Xp))))
        d = np.mean(drops, axis=0)
        rows.append({'split': split_i, 'feature': f, **{f'drop_auc_{c}': v for c, v in zip(CLASSES, d)},
                     'drop_auc_macro': float(np.mean(d))})
    return rows


def assoc_type(r):
    sig = {k: r[f'q_{k}'] < Q for k in LABEL}
    sign = {k: np.sign(r[f'coef_{k}']) for k in LABEL}
    if all(sig.values()) and len(set(sign.values())) == 1:
        return 'progressive'
    if sig['10s_vs_20s_40s'] and sig['10s_vs_50s_plus'] and not sig['20s_40s_vs_50s_plus']:
        return 'adolescent-associated'
    if sig['20s_40s_vs_50s_plus'] and not sig['10s_vs_20s_40s']:
        return 'older-adult-associated'
    return 'other'


SHORT = {'10s_vs_20s_40s': '10s vs 20s–40s', '10s_vs_50s_plus': '10s vs 50s+', '20s_40s_vs_50s_plus': '20s–40s vs 50s+'}


def significant_contrasts(r):
    """Which of the three pairwise contrasts reach q < 0.05, e.g. '2/3: 10s vs 20s–40s, 10s vs 50s+'."""
    hits = [SHORT[k] for k in LABEL if r[f'q_{k}'] < Q]
    return f"{len(hits)}/3" + (f": {', '.join(hits)}" if hits else '')


def direction(r):
    # MaAsLin2 coef = level vs reference; positive = higher in the older group of the contrast
    arrow = lambda k: ('↑' if r[f'coef_{k}'] > 0 else '↓') + ('' if r[f'q_{k}'] < Q else ' (ns)')
    return '; '.join(f'{LABEL[k]} {arrow(k)}' for k in LABEL)


def main():
    X, y, _ = load_split('train')
    panel = pd.read_csv(os.path.join(RES, 'final_panel.csv'))
    feats = panel['taxonomy'].tolist()
    params = json.load(open(os.path.join(RES, 'final_model_params.json')))
    rf_params = {**params['params'], 'n_estimators': params['n_estimators']}

    out = Parallel(n_jobs=6)(delayed(fold_importance)(i, tr, va, X, y, feats, rf_params)
                             for i, (tr, va) in enumerate(cv_splitter().split(X, y), start=1))
    perm = pd.DataFrame([r for o in out for r in o])
    perm.to_csv(os.path.join(RES, 'permutation_importance_cv_folds.csv'), index=False)
    imp = perm.groupby('feature')[[f'drop_auc_{c}' for c in CLASSES] + ['drop_auc_macro']].mean()

    assoc = pd.read_csv(os.path.join(RES, 'panel_associations.csv')).set_index('genus')
    indep = pd.read_csv(os.path.join(RES, 'panel_associations_independent.csv')).set_index('genus')
    freq = pd.read_csv(os.path.join(RES, 'panel_stability_fold_top10.csv')).set_index('genus')

    rows = []
    for _, p in panel.iterrows():
        g, a = p['genus'], assoc.loc[p['genus']]
        rows.append({
            'rank': p['rank'], 'genus': g,
            'stability_fold_top10_of_50': int(freq['fold_top10_of_50'].get(g, 0)),
            'association_type': assoc_type(a), 'significant_contrasts': significant_contrasts(a),
            'direction': direction(a),
            **{f'q_{k}': a[f'q_{k}'] for k in LABEL},
            'q_20s_40s_vs_50s_plus_independent': indep.loc[g, 'q_20s_40s_vs_50s_plus'],
            'replicated_independent': bool(indep.loc[g, 'q_20s_40s_vs_50s_plus'] < Q
                                           and np.sign(indep.loc[g, 'coef_20s_40s_vs_50s_plus']) == np.sign(a['coef_20s_40s_vs_50s_plus'])),
            **{f'perm_drop_auc_{c}': imp.loc[p['taxonomy'], f'drop_auc_{c}'] for c in CLASSES},
            'perm_drop_auc_macro': imp.loc[p['taxonomy'], 'drop_auc_macro'],
        })
    prof = pd.DataFrame(rows)
    prof.to_csv(os.path.join(RES, 'panel_marker_profile.csv'), index=False)
    print(prof[['rank', 'genus', 'stability_fold_top10_of_50', 'association_type', 'significant_contrasts']].to_string(index=False))


if __name__ == '__main__':
    main()
