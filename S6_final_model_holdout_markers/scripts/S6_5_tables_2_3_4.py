"""Table 2 (internal validation per age group), Table 3 (panel profile) and Table 4 (MaAsLin2 of the panel).

Input
  S5_nested_cv_tuning/results/tuning_oof_predictions_k10.csv   validation-fold probabilities of the tuned 10-genus RF
  S6_final_model_holdout_markers/results/holdout_predictions.csv, panel_marker_profile.csv, final_panel.csv,
    panel_associations.csv, panel_associations_independent.csv
Output (S6_final_model_holdout_markers/results/)
  Table2_internal_per_group.csv   nested CV (mean ± SD over 50 splits) and hold-out (estimate with stratified
                                  bootstrap 95% CI) per age group and macro average
  Table3_panel_profile.csv        the 10 panel genera: importance, stability, association type, contrasts with
                                  q < 0.05, direction, group-specific contribution, full taxonomy
  Table4_MaAsLin2.csv             MaAsLin2 coefficients and q values of the panel genera (discovery cohort excluding
                                  the hold-out set; participants not used for model development)

Process
  Precision, sensitivity, specificity, F1-score and AUC per age group, macro averages and accuracy; nested CV values
  are computed per split and summarized as mean ± SD; hold-out CIs from 2,000 stratified bootstrap resamples.

Usage
  python S6_5_tables_2_3_4.py
"""
import os
import sys

import pandas as pd

PKG = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, os.path.join(PKG, 'common'))
from group_metrics import GROUPS, boot_table, flat, per_class, pivot  # noqa: E402

RES = os.path.join(PKG, 'S6_final_model_holdout_markers', 'results')
OOF = os.path.join(PKG, 'S5_nested_cv_tuning', 'results', 'tuning_oof_predictions_k10.csv')


def internal_per_group_table():
    oof = pd.read_csv(OOF)
    per_split = []
    for s, d in oof.groupby('split'):
        per_split.append(flat(per_class(d['true'].values, d[[f'p_{c}' for c in range(3)]].values)))
    ps = pd.DataFrame(per_split)
    cv = pd.DataFrame([{'set': 'Nested CV (n = 420, 50 splits)', 'group': g, 'metric': m,
                        'value': f'{ps[(g, m)].mean():.3f} ± {ps[(g, m)].std():.3f}'} for (g, m) in ps.columns])
    hold = pd.read_csv(os.path.join(RES, 'holdout_predictions.csv'))
    y = hold['true'].map({g: i for i, g in enumerate(GROUPS)}).values
    ho = boot_table(y, hold[[f'p_{g}' for g in GROUPS]].values, 'Hold-out (n = 60)')
    return pivot(pd.concat([cv, ho]))


def panel_profile_table():
    p = pd.read_csv(os.path.join(RES, 'panel_marker_profile.csv'))
    full = pd.read_csv(os.path.join(RES, 'final_panel.csv'))[['genus', 'taxonomy', 'importance']]
    p = p.merge(full, on='genus')
    direction = (p['direction'].str.replace('20s_40s', '20s–40s', regex=False)
                 .str.replace('(ns)', '(q ≥ 0.05)', regex=False))
    drop = {g: p[f'perm_drop_auc_{g}'].map('{:.3f}'.format) for g in GROUPS}
    return pd.DataFrame({
        'Rank': p['rank'], 'Genus': p['genus'].str.replace('g__', ''),
        'RF importance': p['importance'].map('{:.4f}'.format),
        'Stability: top-10 in nested-CV training folds (of 50)': p['stability_fold_top10_of_50'],
        'Association type': p['association_type'], 'Contrasts with q < 0.05': p['significant_contrasts'],
        'Direction across age groups': direction,
        'Group-specific contribution (AUC drop): 10s': drop['10s'],
        'Group-specific contribution (AUC drop): 20s–40s': drop['20s_40s'],
        'Group-specific contribution (AUC drop): 50s+': drop['50s_plus'],
        'Full taxonomy (GTDB)': p['taxonomy']})


def maaslin_table():
    d = pd.read_csv(os.path.join(RES, 'panel_associations.csv'))
    i = pd.read_csv(os.path.join(RES, 'panel_associations_independent.csv')).set_index('genus')
    out = pd.DataFrame({'Rank': d['rank'], 'Genus': d['genus'].str.replace('g__', '')})
    for k, lab in [('10s_vs_20s_40s', '10s vs 20s–40s'), ('10s_vs_50s_plus', '10s vs 50s+'), ('20s_40s_vs_50s_plus', '20s–40s vs 50s+')]:
        out[f'{lab}: coef'] = d[f'coef_{k}'].round(3)
        out[f'{lab}: q'] = d[f'q_{k}'].map(lambda v: f'{v:.2e}')
    out['Contrasts with q<0.05 (of 3)'] = d['n_contrasts_q<0.05']
    out['Not in model development (n = 3,179), 20s–40s vs 50s+: coef'] = d['genus'].map(i['coef_20s_40s_vs_50s_plus']).round(3)
    out['Not in model development: q'] = d['genus'].map(i['q_20s_40s_vs_50s_plus']).map(lambda v: f'{v:.2e}')
    return out


def main():
    for name, table in [('Table2_internal_per_group', internal_per_group_table()),
                        ('Table3_panel_profile', panel_profile_table()), ('Table4_MaAsLin2', maaslin_table())]:
        table.to_csv(os.path.join(RES, f'{name}.csv'), index=False)
        print(f'{name}: {table.shape}')


if __name__ == '__main__':
    main()
