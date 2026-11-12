"""Figure 9. External validation.

Input (S7_external_validation/results/)
  external_predictions.csv, external_metrics.csv
Output (S7_external_validation/figures/)
  Figure9_external_validation.tiff / .png

Process
  (A) Age-group distribution of the external cohort (n = 108).
  (B) ROC curve per age group of the final model applied without retraining, with stratified bootstrap 95% CIs.
  (C) Confusion matrix (counts and row percentages) with balanced accuracy.

Usage
  python S7_3_figure9.py
"""
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

PKG = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, os.path.join(PKG, 'common'))
from figure_style import GROUP_COLOR, GROUP_LABEL, INK, confusion_panel, panel_label, roc_panel, save, ygrid  # noqa: E402

STEP = os.path.join(PKG, 'S7_external_validation')
GROUPS = ['10s', '20s_40s', '50s_plus']


def main():
    pred = pd.read_csv(os.path.join(STEP, 'results', 'external_predictions.csv'), index_col=0)
    ci = pd.read_csv(os.path.join(STEP, 'results', 'external_metrics.csv')).set_index('metric')
    y = pred['true'].map({g: i for i, g in enumerate(GROUPS)}).values
    p = pred[[f'p_{g}' for g in GROUPS]].values

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 3.6), gridspec_kw={'width_ratios': [0.7, 1, 0.9]})
    ax = axes[0]
    n = [int((y == c).sum()) for c in range(3)]
    ax.bar(range(3), n, color=[GROUP_COLOR[g] for g in GROUPS], width=0.65)
    for i, v in enumerate(n):
        ax.text(i, v + 1.5, f'{v}\n({v / len(y):.0%})', ha='center', fontsize=6.8, color=INK)
    ax.set_xticks(range(3))
    ax.set_xticklabels([GROUP_LABEL[g] for g in GROUPS])
    ax.set_ylabel('Participants (n)')
    ax.set_ylim(0, 95)
    ygrid(ax)
    ax.set_title('External cohort, n=108', fontsize=7.8, color=INK)
    panel_label(ax, 'A', x=-0.35)

    r = ci.loc['auc_macro']
    ba = ci.loc['balanced_accuracy (macro sensitivity)']
    roc_panel(axes[1], y, p, ci, f'External validation (no retraining)\nmacro-AUC {r.estimate:.3f} [{r.ci_low:.3f}–{r.ci_high:.3f}]')
    panel_label(axes[1], 'B', x=-0.3)
    confusion_panel(axes[2], y, p.argmax(1), f'Confusion matrix\nbalanced accuracy {ba.estimate:.3f} [{ba.ci_low:.2f}–{ba.ci_high:.2f}]')
    panel_label(axes[2], 'C', x=-0.45)
    fig.tight_layout()
    save(fig, 'Figure9_external_validation', os.path.join(STEP, 'figures'))


if __name__ == '__main__':
    main()
