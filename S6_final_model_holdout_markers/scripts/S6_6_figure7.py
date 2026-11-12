"""Figure 7. Internal validation of the 10-genus random forest.

Input
  S5_nested_cv_tuning/results/tuning_oof_predictions_k10.csv, tuning_summary.csv
  S6_final_model_holdout_markers/results/holdout_predictions.csv, holdout_metrics.csv
Output (S6_final_model_holdout_markers/figures/)
  Figure7_internal_validation.tiff / .png

Process
  (A) Nested cross-validation: mean one-vs-rest ROC curve per age group across the 50 splits (± 1 SD band).
  (B) Hold-out set (n = 60, evaluated once): ROC curve per age group with stratified bootstrap 95% CI.
  (C) Hold-out confusion matrix (counts and row percentages).

Usage
  python S6_6_figure7.py
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

PKG = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, os.path.join(PKG, 'common'))
from figure_style import GROUP_COLOR, GROUP_LABEL, MUTED, confusion_panel, panel_label, roc_panel, save, style_roc  # noqa: E402

STEP = os.path.join(PKG, 'S6_final_model_holdout_markers')
OOF = os.path.join(PKG, 'S5_nested_cv_tuning', 'results', 'tuning_oof_predictions_k10.csv')
SUMMARY = os.path.join(PKG, 'S5_nested_cv_tuning', 'results', 'tuning_summary.csv')
GROUPS = ['10s', '20s_40s', '50s_plus']
GRID = np.linspace(0, 1, 101)


def main():
    oof = pd.read_csv(OOF)
    summ = pd.read_csv(SUMMARY)
    s10 = summ[(summ.k == 10) & (summ.model == 'RF_tuned')].iloc[0]
    hold = pd.read_csv(os.path.join(STEP, 'results', 'holdout_predictions.csv'))
    hci = pd.read_csv(os.path.join(STEP, 'results', 'holdout_metrics.csv')).set_index('metric')

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 3.6), gridspec_kw={'width_ratios': [1, 1, 0.9]})
    ax = axes[0]
    for c, g in enumerate(GROUPS):
        curves, aucs = [], []
        for _, d in oof.groupby('split'):
            y = (d['true'] == c).values
            fpr, tpr, _ = roc_curve(y, d[f'p_{c}'])
            t = np.interp(GRID, fpr, tpr)
            t[0] = 0
            curves.append(t)
            aucs.append(roc_auc_score(y, d[f'p_{c}']))
        m, s = np.mean(curves, axis=0), np.std(curves, axis=0)
        ax.fill_between(GRID, np.clip(m - s, 0, 1), np.clip(m + s, 0, 1), color=GROUP_COLOR[g], alpha=0.12, linewidth=0)
        ax.plot(GRID, m, color=GROUP_COLOR[g], linewidth=2, label=f'{GROUP_LABEL[g]}  {np.mean(aucs):.2f} ± {np.std(aucs, ddof=1):.2f}')
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=0.8)
    style_roc(ax, f"Nested CV, train n=420 (50 splits)\nmacro-AUC {s10['auc_macro_mean']:.3f} ± {s10['auc_macro_sd']:.3f}")
    panel_label(ax, 'A', x=-0.3)

    y_h = hold['true'].map({g: i for i, g in enumerate(GROUPS)}).values
    p_h = hold[[f'p_{g}' for g in GROUPS]].values
    r = hci.loc['auc_macro']
    roc_panel(axes[1], y_h, p_h, hci, f'Hold-out, n=60 (evaluated once)\nmacro-AUC {r.estimate:.3f} [{r.ci_low:.3f}–{r.ci_high:.3f}]')
    panel_label(axes[1], 'B', x=-0.3)
    confusion_panel(axes[2], y_h, p_h.argmax(1), f"Hold-out confusion matrix\naccuracy {hci.loc['accuracy'].estimate:.3f}")
    panel_label(axes[2], 'C', x=-0.45)
    fig.tight_layout()
    save(fig, 'Figure7_internal_validation', os.path.join(STEP, 'figures'))


if __name__ == '__main__':
    main()
