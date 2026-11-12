"""Genus selection by MaAsLin2 inside the training folds of the nested cross-validation.

Input
  S1_data_processing/results/discovery_genus_relab_cut0.01.csv   genus relative abundance (%)
  S1_data_processing/results/discovery_covariates.csv            Sex, Diet, Drug, Disease_pr
  S1_data_processing/results/split_sample_ids.csv                training set (n = 420)
  common/run_maaslin2.R                                           MaAsLin2 call
Output (S3_genus_selection_maaslin2/results/)
  genus_selection_cv_split_metrics.csv   metrics of RF and XGBoost per split and genus set
  genus_selection_split_genera.csv       per split: top-30 genera, genera removed by the prevalence filter,
                                         genera retained by MaAsLin2
  genus_selection_maaslin2_contrasts.csv per split: MaAsLin2 age-group contrasts of the top-30 genera

Process (repeated stratified five-fold cross-validation, 10 repeats = 50 splits; within each training fold, n = 336)
  1) Random forest (default hyperparameters) on all 783 genera; genera ranked by impurity-based importance; top 30.
  2) MaAsLin2 on the top 30 with covariates Sex, Diet, Drug and Disease_pr, run with age reference 10s and
     20s_40s to obtain the three pairwise contrasts (10s vs 20s_40s, 10s vs 50s_plus, 20s_40s vs 50s_plus).
  3) Genera with q < 0.05 in at least two of the three contrasts are retained.
  4) RF and XGBoost (default hyperparameters) are fitted on all 783 genera, on the top 30, and on the
     MaAsLin2-retained genera, and evaluated on the validation fold (n = 84), which is not used in steps 1-3.

Usage
  python S3_1_genus_selection_nested_cv.py --workers 4
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

PKG = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, os.path.join(PKG, 'common'))
from analysis_common import CLASSES, SEED, TARGET, classification_metrics, cv_splitter, load_covariates, load_split  # noqa: E402

OUT = os.path.join(PKG, 'S3_genus_selection_maaslin2', 'results')
R_SCRIPT = os.path.join(PKG, 'common', 'run_maaslin2.R')

N_TOP = 30
Q_CUT, MIN_CONTRASTS = 0.05, 2
MODELS = {
    'RF': RandomForestClassifier(class_weight='balanced', n_jobs=1, random_state=SEED),
    'XGB': XGBClassifier(objective='multi:softprob', num_class=len(CLASSES), eval_metric='mlogloss',
                         random_state=SEED, verbosity=0, n_jobs=1),
}
# (age reference of the run, MaAsLin2 'value' level, contrast label)
CONTRASTS = [('10s', '20s_40s', '10s_vs_20s_40s'),
             ('10s', '50s_plus', '10s_vs_50s_plus'),
             ('20s_40s', '50s_plus', '20s_40s_vs_50s_plus')]


def short(name):
    parts = name.split(';')
    return parts[-1] if parts[-1] != '__' else parts[-2] + ';__'


def run_maaslin(X, meta, split_i):
    """MaAsLin2 on X (participants x genera). Returns the age-group contrasts and the genera that were tested."""
    alias = {f: f'F{i:03d}' for i, f in enumerate(X.columns)}  # MaAsLin2 renames names that contain ';'
    back = {v: k for k, v in alias.items()}
    tmp = tempfile.mkdtemp(prefix=f'maaslin_s{split_i:02d}_')
    try:
        X.rename(columns=alias).to_csv(os.path.join(tmp, 'features.tsv'), sep='\t')
        meta.to_csv(os.path.join(tmp, 'metadata.tsv'), sep='\t')
        rows, tested = [], set()
        for ref in ['10s', '20s_40s']:
            out = os.path.join(tmp, f'out_{ref}')
            subprocess.run(['Rscript', R_SCRIPT, os.path.join(tmp, 'features.tsv'),
                            os.path.join(tmp, 'metadata.tsv'), out, ref],
                           check=True, capture_output=True, text=True)
            res = pd.read_csv(os.path.join(out, 'all_results.tsv'), sep='\t')
            res['feature'] = res['feature'].map(back)
            tested |= set(res['feature'])
            for r, value, label in CONTRASTS:
                if r != ref:
                    continue
                sub = res[(res['metadata'] == TARGET) & (res['value'] == value)]
                rows += [{'feature': f, 'contrast': label, 'coef': c, 'pval': p, 'qval': q}
                         for f, c, p, q in zip(sub['feature'], sub['coef'], sub['pval'], sub['qval'])]
        return pd.DataFrame(rows), tested
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def select_genera(contrasts, top30):
    """Genera (in top-30 order) with q < 0.05 in at least two of the three contrasts.
    Genera removed by the MaAsLin2 prevalence filter have no results and are not retained."""
    n_sig = (contrasts.assign(sig=contrasts['qval'] < Q_CUT)
             .groupby('feature')['sig'].sum().reindex(top30, fill_value=0))
    return [f for f in top30 if n_sig[f] >= MIN_CONTRASTS]


def run_split(split_i, tr, va, X, y, meta):
    X_tr, y_tr, X_va, y_va = X.iloc[tr], y[tr], X.iloc[va], y[va]

    selector = clone(MODELS['RF']).set_params(n_jobs=-1).fit(X_tr, y_tr)
    top30 = list(X.columns[np.argsort(selector.feature_importances_)[::-1][:N_TOP]])

    age = pd.DataFrame({TARGET: [CLASSES[i] for i in y_tr]}, index=X_tr.index)
    contrasts, tested = run_maaslin(X_tr[top30], pd.concat([age, meta.loc[X_tr.index]], axis=1), split_i)
    retained = select_genera(contrasts, top30)

    sets = {'all783': list(X.columns), 'top30': top30, 'top30_MaAsLin2': retained}
    metrics = []
    for set_name, feats in sets.items():
        if not feats:
            continue
        for model_name, base in MODELS.items():
            model = clone(base).fit(X_tr[feats], y_tr)
            m = classification_metrics(y_va, model.predict_proba(X_va[feats]))
            metrics.append({'split': split_i, 'repeat': (split_i - 1) // 5 + 1, 'model': model_name,
                            'set': set_name, 'n_features': len(feats), **m})
    genera = {'split': split_i, 'n_top30': len(top30), 'n_tested_after_prevalence': len(tested),
              'prevalence_filtered': ';'.join(short(f) for f in top30 if f not in tested),
              'n_top30_MaAsLin2': len(retained), 'top30': '|'.join(top30), 'top30_MaAsLin2': '|'.join(retained)}
    return metrics, genera, contrasts.assign(split=split_i)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=1)
    args = ap.parse_args()

    X, y, ids = load_split('train')
    meta = load_covariates(ids)
    splits = list(cv_splitter().split(X, y))
    print(f'training set {X.shape}, {len(splits)} splits, {args.workers} worker(s)', flush=True)

    t0 = time.time()
    out = Parallel(n_jobs=args.workers, verbose=5)(
        delayed(run_split)(i, tr, va, X, y, meta) for i, (tr, va) in enumerate(splits, start=1))

    os.makedirs(OUT, exist_ok=True)
    pd.DataFrame([m for o in out for m in o[0]]).to_csv(os.path.join(OUT, 'genus_selection_cv_split_metrics.csv'), index=False)
    pd.DataFrame([o[1] for o in out]).to_csv(os.path.join(OUT, 'genus_selection_split_genera.csv'), index=False)
    pd.concat([o[2] for o in out]).to_csv(os.path.join(OUT, 'genus_selection_maaslin2_contrasts.csv'), index=False)
    print(f'finished in {time.time() - t0:.0f} s')


if __name__ == '__main__':
    main()
