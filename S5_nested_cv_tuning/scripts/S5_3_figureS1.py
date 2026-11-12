"""Supplementary Figure S1. Number of cross-validation repeats.

Input
  S5_nested_cv_tuning/results/tuning_cv_split_metrics.csv
Output (S5_nested_cv_tuning/figures/)
  FigureS1_cv_repeats_convergence.tiff / .png

Process
  Cumulative mean macro-AUC after each repeat of the repeated stratified five-fold cross-validation (five splits
  per repeat) for the tuned random forest with the 10-genus panel and with all 783 genera; the legend gives the
  largest change over the last three repeats.

Usage
  python S5_3_figureS1.py
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PKG = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, os.path.join(PKG, 'common'))
from figure_style import GROUP_COLOR, save, ygrid  # noqa: E402

STEP = os.path.join(PKG, 'S5_nested_cv_tuning')


def cumulative(df):
    per_repeat = df.groupby('repeat')['auc_macro'].mean().sort_index()
    return per_repeat.expanding().mean()


def main():
    m2 = pd.read_csv(os.path.join(STEP, 'results', 'tuning_cv_split_metrics.csv'))
    series = [('10-genus panel (final procedure)', m2[(m2.k == 10) & (m2.model == 'RF_tuned')], GROUP_COLOR['10s']),
              ('All 783 genera', m2[(m2.k == 783) & (m2.model == 'RF_tuned')], GROUP_COLOR['20s_40s'])]
    fig, ax = plt.subplots(figsize=(3.2, 3.0))
    for label, df, color in series:
        c = cumulative(df)
        change = np.abs(np.diff(c.values[-4:])).max()
        ax.plot(c.index, c.values, color=color, linewidth=2, marker='o', markersize=4,
                label=f'{label}\nfinal {c.values[-1]:.3f}; change over last 3 repeats ≤ {change:.3f}')
    ax.legend(loc='upper center', bbox_to_anchor=(0.45, -0.25), frameon=False, handlelength=1.2)
    ax.set_xticks(range(1, 11))
    ax.set_xlim(0.6, 10.4)
    ax.set_xlabel('Number of repeats (five folds each)')
    ax.set_ylabel('Cumulative mean macro-AUC')
    ax.spines['bottom'].set_bounds(1, 10)
    ygrid(ax)
    fig.tight_layout()
    save(fig, 'FigureS1_cv_repeats_convergence', os.path.join(STEP, 'figures'), width='single')


if __name__ == '__main__':
    main()
