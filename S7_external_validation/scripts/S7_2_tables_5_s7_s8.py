"""Table 5 (external validation per age group), Supplementary Tables S7 and S8.

Input (S7_external_validation/results/)
  external_predictions.csv, external_balanced_subsampling.csv, panel_detection.csv
Output (S7_external_validation/results/)
  Table5_external_per_group.csv   precision, sensitivity, specificity, F1-score and AUC per age group, macro
                                  averages and accuracy, with stratified bootstrap 95% CI (2,000 resamples)
  TableS7_external_balanced.csv   mean and SD of each metric across the 30 class-balanced subsamples
  TableS8_panel_detection.csv     detection and median abundance of the panel genera, external vs training set

Usage
  python S7_2_tables_5_s7_s8.py
"""
import os
import sys

import pandas as pd

PKG = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, os.path.join(PKG, 'common'))
from group_metrics import GROUPS, boot_table, pivot  # noqa: E402

RES = os.path.join(PKG, 'S7_external_validation', 'results')


def main():
    pred = pd.read_csv(os.path.join(RES, 'external_predictions.csv'), index_col=0)
    y = pred['true'].map({g: i for i, g in enumerate(GROUPS)}).values
    t5 = pivot(boot_table(y, pred[[f'p_{g}' for g in GROUPS]].values, 'External cohort (n = 108)'))
    t5.to_csv(os.path.join(RES, 'Table5_external_per_group.csv'), index=False)

    sub = pd.read_csv(os.path.join(RES, 'external_balanced_subsampling.csv')).drop(columns='draw')
    s7 = pd.DataFrame({'mean': sub.mean(), 'sd': sub.std()}).round(3).reset_index().rename(columns={'index': 'metric'})
    s7.to_csv(os.path.join(RES, 'TableS7_external_balanced.csv'), index=False)

    s8 = pd.read_csv(os.path.join(RES, 'panel_detection.csv'))
    s8['genus'] = s8['genus'].str.replace('g__', '', regex=False)
    s8.to_csv(os.path.join(RES, 'TableS8_panel_detection.csv'), index=False)
    print(t5.to_string(index=False))


if __name__ == '__main__':
    main()
