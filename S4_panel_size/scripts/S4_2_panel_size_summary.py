"""Summary of performance by panel size (values of Supplementary Table S3 without the tuned columns).

Input
  S4_panel_size/results/panel_size_cv_split_metrics.csv
Output (S4_panel_size/results/)
  panel_size_summary.csv   one row per panel size k

Process
  Per k across the 50 splits: mean and SD of macro-AUC; mean accuracy and per-group AUC; paired change in
  macro-AUC relative to all 783 genera and the number of splits below it; paired change from the previous k in
  the grid (total and per added genus) and the number of splits in which it increased.

Usage
  python S4_2_panel_size_summary.py
"""
import os

import pandas as pd

PKG = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
RES = os.path.join(PKG, 'S4_panel_size', 'results')


def main():
    fold = pd.read_csv(os.path.join(RES, 'panel_size_cv_split_metrics.csv'))
    piv = fold.pivot(index='split', columns='k', values='auc_macro')
    ks = sorted(piv.columns)
    full = piv[783]
    rows = []
    for i, k in enumerate(ks):
        g = fold[fold['k'] == k]
        step = piv[k] - piv[ks[i - 1]] if i else None
        rows.append({
            'k': k,
            'auc_mean': g['auc_macro'].mean(), 'auc_sd': g['auc_macro'].std(),
            'delta_vs_all783': (piv[k] - full).mean(),
            'splits_below_all783': int(((piv[k] - full) < 0).sum()) if k != 783 else 0,
            'accuracy_mean': g['accuracy'].mean(),
            'auc_10s': g['auc_10s'].mean(), 'auc_20s_40s': g['auc_20s_40s'].mean(), 'auc_50s_plus': g['auc_50s_plus'].mean(),
            'prev_k': ks[i - 1] if i else None,
            'step_gain_vs_prev_k': step.mean() if i else None,
            'step_gain_per_added_genus': step.mean() / (k - ks[i - 1]) if i else None,
            'splits_improved_vs_prev_k': int((step > 0).sum()) if i else None,
        })
    table = pd.DataFrame(rows).round(4)
    table.to_csv(os.path.join(RES, 'panel_size_summary.csv'), index=False)
    print(table[['k', 'auc_mean', 'auc_sd', 'delta_vs_all783', 'step_gain_vs_prev_k']].to_string(index=False))


if __name__ == '__main__':
    main()
