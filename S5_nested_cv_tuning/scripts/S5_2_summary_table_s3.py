"""Summary of the nested cross-validation with tuning, and Supplementary Table S3.

Input
  S5_nested_cv_tuning/results/tuning_cv_split_metrics.csv
  S4_panel_size/results/panel_size_summary.csv            performance by panel size (default random forest)
Output (S5_nested_cv_tuning/results/)
  tuning_summary.csv              mean and SD per panel size k and model across the 50 splits
  tuning_paired_comparisons.csv   paired differences in macro-AUC across splits (tuned vs default, k vs 783,
                                  15 vs 10)
  TableS3_panel_size.csv          performance by panel size with the macro-AUC after tuning (k = 10, 15, 100, 783)

Process
  Mean and SD of each metric across the 50 splits; paired differences computed split by split. The panel-size
  table and this analysis use the same 50 splits, so their default random-forest macro-AUCs agree; this is
  checked before the tuned values are added.

Usage
  python S5_2_summary_table_s3.py
"""
import os
import sys

import pandas as pd

PKG = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, os.path.join(PKG, 'common'))
from analysis_common import CLASSES, summarize  # noqa: E402

RES = os.path.join(PKG, 'S5_nested_cv_tuning', 'results')
PANEL_SIZE = os.path.join(PKG, 'S4_panel_size', 'results', 'panel_size_summary.csv')
METRICS = ['auc_macro', 'accuracy', 'sensitivity_macro', 'specificity_macro', 'f1_macro'] + [f'auc_{c}' for c in CLASSES]


def main():
    fold = pd.read_csv(os.path.join(RES, 'tuning_cv_split_metrics.csv'))
    summary = summarize(fold, ['k', 'model'], METRICS)
    summary.to_csv(os.path.join(RES, 'tuning_summary.csv'), index=False)

    p = fold.pivot_table(index='split', columns=['k', 'model'], values='auc_macro')
    pairs = [(f'k={k}: tuned − default', p[(k, 'RF_tuned')] - p[(k, 'RF_default')]) for k in [10, 15, 100, 783]]
    pairs += [(f'tuned: k={k} − k=783', p[(k, 'RF_tuned')] - p[(783, 'RF_tuned')]) for k in [10, 15, 100]]
    pairs += [('tuned: k=15 − k=10', p[(15, 'RF_tuned')] - p[(10, 'RF_tuned')]),
              ('default: k=15 − k=10', p[(15, 'RF_default')] - p[(10, 'RF_default')])]
    comp = pd.DataFrame([{'comparison': name, 'mean_diff': d.mean(), 'sd_diff': d.std(),
                          'splits_positive': int((d > 0).sum()), 'n_splits': len(d)} for name, d in pairs]).round(4)
    comp.to_csv(os.path.join(RES, 'tuning_paired_comparisons.csv'), index=False)

    s3 = pd.read_csv(PANEL_SIZE)
    default = summary[summary.model == 'RF_default'].set_index('k')['auc_macro_mean']
    assert (abs(s3.set_index('k').loc[default.index, 'auc_mean'] - default) < 1e-3).all()
    tuned = summary[summary.model == 'RF_tuned'].set_index('k')
    s3['auc_tuned_mean'] = s3['k'].map(tuned['auc_macro_mean']).round(4)
    s3['auc_tuned_sd'] = s3['k'].map(tuned['auc_macro_sd']).round(4)
    s3.to_csv(os.path.join(RES, 'TableS3_panel_size.csv'), index=False)

    print(summary[['k', 'model', 'auc_macro_mean', 'auc_macro_sd', 'accuracy_mean']].to_string(index=False))
    print()
    print(comp.to_string(index=False))


if __name__ == '__main__':
    main()
