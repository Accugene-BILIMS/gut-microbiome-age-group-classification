"""Covariate-adjusted age-group associations of the genera (MaAsLin2) for interpretation of the panel.

Input
  S1_data_processing/results/discovery_genus_relab_cut0.01.csv   genus relative abundance (%)
  S1_data_processing/results/discovery_covariates.csv            Sex, Diet, Drug, Disease_pr
  S1_data_processing/results/split_sample_ids.csv                training and hold-out sets
  S6_final_model_holdout_markers/results/final_panel.csv
  common/run_maaslin2.R
Output (S6_final_model_holdout_markers/results/)
  --cohort discovery    maaslin2_age_contrasts.csv, panel_associations.csv
  --cohort independent  maaslin2_age_contrasts_independent.csv, panel_associations_independent.csv
    maaslin2_age_contrasts*.csv   age-group contrasts (coefficient, P, q) for every genus tested
    panel_associations*.csv       the same for the 10 panel genera, with the number of contrasts with q < 0.05

Process
  discovery:   all participants except the hold-out set (n = 3,599); MaAsLin2 on all 783 genera with
               covariates Sex, Diet, Drug and Disease_pr, run with age reference 10s and 20s_40s for the three
               pairwise contrasts.
  independent: participants used neither for model development nor for the hold-out set (n = 3,179). All 10s
               participants belong to the training or hold-out set, so only 20s_40s vs 50s_plus is tested.
  The hold-out set is not used. This analysis is not used to build the classifier.

Usage
  python S6_3_maaslin2_associations.py --cohort discovery
  python S6_3_maaslin2_associations.py --cohort independent
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile

import pandas as pd

PKG = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, os.path.join(PKG, 'common'))
from analysis_common import SPLIT_IDS, TARGET, load_covariates, load_genus  # noqa: E402

RES = os.path.join(PKG, 'S6_final_model_holdout_markers', 'results')
R_SCRIPT = os.path.join(PKG, 'common', 'run_maaslin2.R')
CONTRASTS = [('10s', '20s_40s', '10s_vs_20s_40s'), ('10s', '50s_plus', '10s_vs_50s_plus'),
             ('20s_40s', '50s_plus', '20s_40s_vs_50s_plus')]


def short(name):
    parts = name.split(';')
    return parts[-1] if parts[-1] != '__' else parts[-2] + ';__'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cohort', choices=['discovery', 'independent'], default='discovery')
    cohort = ap.parse_args().cohort
    tag = '' if cohort == 'discovery' else '_independent'

    df, features = load_genus()
    split = pd.read_csv(SPLIT_IDS)
    excluded = set(split['Sample_name']) if cohort == 'independent' else set(split.query("split == 'holdout'")['Sample_name'])
    disc = df.loc[[s for s in df.index if s not in excluded]]
    meta = pd.concat([disc[[TARGET]], load_covariates(disc.index)], axis=1)
    print(f'{cohort} cohort n = {len(disc)}, groups {disc[TARGET].value_counts().to_dict()}')

    alias = {f: f'F{i:03d}' for i, f in enumerate(features)}  # MaAsLin2 renames names that contain ';'
    back = {v: k for k, v in alias.items()}
    tmp = tempfile.mkdtemp(prefix='maaslin_')
    rows = []
    try:
        disc[features].rename(columns=alias).to_csv(os.path.join(tmp, 'features.tsv'), sep='\t')
        meta.to_csv(os.path.join(tmp, 'metadata.tsv'), sep='\t')
        refs = ['10s', '20s_40s'] if cohort == 'discovery' else ['20s_40s']
        for ref in refs:
            out = os.path.join(tmp, f'out_{ref}')
            subprocess.run(['Rscript', R_SCRIPT, os.path.join(tmp, 'features.tsv'), os.path.join(tmp, 'metadata.tsv'),
                            out, ref], check=True, capture_output=True, text=True)
            res = pd.read_csv(os.path.join(out, 'all_results.tsv'), sep='\t')
            res['feature'] = res['feature'].map(back)
            for r, value, label in CONTRASTS:
                if r == ref:
                    sub = res[(res['metadata'] == TARGET) & (res['value'] == value)]
                    rows += [{'taxonomy': f, 'genus': short(f), 'contrast': label, 'coef': c, 'pval': p, 'qval': q}
                             for f, c, p, q in zip(sub['feature'], sub['coef'], sub['pval'], sub['qval'])]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    con = pd.DataFrame(rows)
    con.to_csv(os.path.join(RES, f'maaslin2_age_contrasts{tag}.csv'), index=False)

    panel = pd.read_csv(os.path.join(RES, 'final_panel.csv'))
    out = []
    for _, r in panel.iterrows():
        c = con[con['taxonomy'] == r['taxonomy']].set_index('contrast')
        row = {'rank': r['rank'], 'genus': r['genus'], 'tested': not c.empty}
        for _, _, label in CONTRASTS:
            row[f'coef_{label}'] = c.loc[label, 'coef'] if label in c.index else None
            row[f'q_{label}'] = c.loc[label, 'qval'] if label in c.index else None
        row['n_contrasts_q<0.05'] = int((c['qval'] < 0.05).sum()) if not c.empty else 0
        out.append(row)
    out = pd.DataFrame(out)
    out.to_csv(os.path.join(RES, f'panel_associations{tag}.csv'), index=False)
    print(f"MaAsLin2 tested {con['taxonomy'].nunique()} genera after the 10% prevalence filter")
    print(out[['rank', 'genus', 'n_contrasts_q<0.05']].to_string(index=False))


if __name__ == '__main__':
    main()
