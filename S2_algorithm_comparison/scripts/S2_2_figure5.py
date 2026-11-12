"""Figure 5. Classifier comparison.

Input
  S2_algorithm_comparison/results/classifier_cv_split_metrics.csv
Output (S2_algorithm_comparison/figures/)
  Figure5_classifier_comparison.tiff / .png

Process
  (A) Macro-AUC of each classifier across the 50 cross-validation splits (box plot; numbers are means).
  (B) Mean one-vs-rest AUC per age group.

Usage
  python S2_2_figure5.py
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PKG = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, os.path.join(PKG, 'common'))
from figure_style import GROUP_COLOR, GROUP_LABEL, INK, MODEL_COLOR, MUTED, panel_label, save, ygrid  # noqa: E402

STEP = os.path.join(PKG, 'S2_algorithm_comparison')
NAMES = {'RF': 'Random forest', 'XGB': 'XGBoost', 'SVM': 'SVM', 'KNN': 'KNN', 'DT': 'Decision tree'}


def main():
    fold = pd.read_csv(os.path.join(STEP, 'results', 'classifier_cv_split_metrics.csv'))
    order = fold.groupby('model')['auc_macro'].mean().sort_values(ascending=False).index.tolist()

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.0, 3.0), gridspec_kw={'width_ratios': [1.1, 1]})
    data = [fold.loc[fold.model == m, 'auc_macro'] for m in order]
    bp = ax.boxplot(data, widths=0.55, patch_artist=True, showfliers=False,
                    medianprops={'color': 'white', 'linewidth': 1.5}, whiskerprops={'color': MUTED}, capprops={'color': MUTED})
    for b, m in zip(bp['boxes'], order):
        b.set(facecolor=MODEL_COLOR[m], edgecolor=MODEL_COLOR[m])
    for i, d in enumerate(data, start=1):
        q1, q3 = np.percentile(d, [25, 75])
        whisker_top = d[d <= q3 + 1.5 * (q3 - q1)].max()
        ax.text(i, whisker_top + 0.012, f'{d.mean():.3f}', ha='center', fontsize=7, color=INK)
    ax.axhline(0.5, color=MUTED, linewidth=0.8)
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels([NAMES[m] for m in order], rotation=20, ha='right')
    ax.set_ylabel('Macro-AUC (50 CV splits)')
    ax.set_ylim(0.45, 0.92)
    ygrid(ax)
    panel_label(ax, 'A')

    groups = ['10s', '20s_40s', '50s_plus']
    means = fold.groupby('model')[[f'auc_{g}' for g in groups]].mean().loc[order]
    x = np.arange(len(order))
    w = 0.26
    for j, g in enumerate(groups):
        bx.bar(x + (j - 1) * w, means[f'auc_{g}'], width=w * 0.92, color=GROUP_COLOR[g], label=GROUP_LABEL[g])
    bx.axhline(0.5, color=MUTED, linewidth=0.8)
    bx.set_xticks(x)
    bx.set_xticklabels([NAMES[m] for m in order], rotation=20, ha='right')
    bx.set_ylabel('Mean one-vs-rest AUC')
    bx.set_ylim(0.45, 0.9)
    bx.legend(frameon=False, ncol=3, loc='upper right')
    ygrid(bx)
    panel_label(bx, 'B')
    fig.tight_layout()
    save(fig, 'Figure5_classifier_comparison', os.path.join(STEP, 'figures'))


if __name__ == '__main__':
    main()
