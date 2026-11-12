"""Supplementary Table S2. Genus selection inside the training folds.

Input
  S3_genus_selection_maaslin2/results/genus_selection_cv_split_metrics.csv
Output (S3_genus_selection_maaslin2/results/)
  TableS2_genus_selection.csv   mean and SD across the 50 splits per model and genus set

Process
  Mean and SD of the number of genera and of each metric across the 50 splits, for RF and XGBoost with all 783
  genera, the top 30 genera, and the genera retained by MaAsLin2 (q < 0.05 in at least two of three contrasts).

Usage
  python S3_2_table_s2.py
"""
import os
import sys

import pandas as pd

PKG = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, os.path.join(PKG, 'common'))
from analysis_common import CLASSES, summarize  # noqa: E402

RES = os.path.join(PKG, 'S3_genus_selection_maaslin2', 'results')
SETS = ['all783', 'top30', 'top30_MaAsLin2']
METRICS = ['n_features', 'auc_macro', 'accuracy', 'sensitivity_macro', 'specificity_macro', 'f1_macro'] + \
          [f'auc_{c}' for c in CLASSES]


def main():
    fold = pd.read_csv(os.path.join(RES, 'genus_selection_cv_split_metrics.csv'))
    summary = summarize(fold, ['model', 'set'], METRICS)
    summary['n_splits_evaluated'] = fold.groupby(['model', 'set']).size().values
    summary['set'] = pd.Categorical(summary['set'], SETS, ordered=True)
    summary = summary.sort_values(['model', 'set'])
    summary.to_csv(os.path.join(RES, 'TableS2_genus_selection.csv'), index=False)
    for _, r in summary.iterrows():
        print(f"{r['model']:>3} {r['set']:<15} genera {r['n_features_mean']:6.2f}  "
              f"macro-AUC {r['auc_macro_mean']:.3f} ± {r['auc_macro_sd']:.3f}")


if __name__ == '__main__':
    main()
