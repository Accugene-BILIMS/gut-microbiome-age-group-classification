"""Figure 2. Study design.

Input
  none (diagram; the numbers shown are those reported in the results of the analysis steps)
Output (S9_cohort_overview/figures/)
  Figure2_study_design.tiff / .png

Process
  Flow diagram of the class-balanced split, nested cross-validation, final model, hold-out and external evaluation,
  and the association analyses.

Usage
  python S9_3_figure2.py
"""
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

PKG = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, os.path.join(PKG, 'common'))
from figure_style import INK, MUTED, save  # noqa: E402

STEP = os.path.join(PKG, 'S9_cohort_overview')
FILL = {'data': '#eef3fa', 'cv': '#fff4e8', 'model': '#e8f6f0', 'eval': '#ffffff', 'assoc': '#f4f1fb'}
EDGE = {'data': '#2a78d6', 'cv': '#eb6834', 'model': '#1baf7a', 'eval': INK, 'assoc': '#6250d6'}


def box(ax, x, y, w, h, kind, title, body, title_size=7.4, body_size=6.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.008,rounding_size=0.012',
                                facecolor=FILL[kind], edgecolor=EDGE[kind], linewidth=1.1))
    ax.text(x + w / 2, y + h - 0.012, title, ha='center', va='top', fontsize=title_size, fontweight='bold', color=INK)
    ax.text(x + w / 2, y + h - 0.045, body, ha='center', va='top', fontsize=body_size, color=INK, linespacing=1.35)


def arrow(ax, x1, y1, x2, y2, text=None):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=9, color=MUTED, linewidth=1))
    if text:
        ax.text((x1 + x2) / 2 + 0.008, (y1 + y2) / 2, text, fontsize=6, color=MUTED, va='center')


def main():
    fig = plt.figure(figsize=(7.0, 7.2))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    box(ax, 0.14, 0.875, 0.72, 0.11, 'data', 'Korean discovery cohort (16S rRNA, V4)',
        'n = 3,659 participants (aged 10–86 years)\n'
        'Genus relative abundance; values ≤0.01% set to 0; 783 genera; no renormalization')
    arrow(ax, 0.40, 0.875, 0.27, 0.80)
    arrow(ax, 0.60, 0.875, 0.78, 0.80)
    box(ax, 0.04, 0.70, 0.46, 0.10, 'data', 'Class-balanced subset, n = 480',
        '160 per age group (10s / 20s–40s / 50s+)\nhold-out separated before any selection or fitting')
    box(ax, 0.58, 0.70, 0.38, 0.10, 'assoc', 'Remaining participants, n = 3,179',
        'not used for model development\n(20s–40s 2,887; 50s+ 292; no 10s)')

    # hold-out on the far left so its arrow runs down the margin to the hold-out test
    arrow(ax, 0.11, 0.70, 0.11, 0.655)
    arrow(ax, 0.37, 0.70, 0.37, 0.655)
    box(ax, 0.02, 0.585, 0.18, 0.07, 'eval', 'Hold-out set, n = 60', '20 per age group')
    box(ax, 0.22, 0.585, 0.30, 0.07, 'data', 'Training set, n = 420', '140 per age group')

    arrow(ax, 0.37, 0.585, 0.37, 0.545)
    box(ax, 0.07, 0.30, 0.45, 0.245, 'cv', 'Nested CV: repeated stratified 5-fold × 10 (50 splits)',
        'In each training fold (n = 336) only:\n'
        '1  rank 783 genera by random-forest Gini importance\n'
        '2  keep the top k genera\n'
        '3  tune RF with Optuna (inner 5-fold CV, 30 trials)\n'
        'Evaluate once on the validation fold (n = 84)\n\n'
        'Classifier comparison (5 algorithms) → random forest\n'
        'Panel-size curve (k = 1…783) → plateau 10–15 → k = 10\n'
        'Internal estimate, 10 genera: macro-AUC 0.703 ± 0.040', body_size=6.2)

    arrow(ax, 0.295, 0.30, 0.295, 0.255)
    box(ax, 0.07, 0.165, 0.45, 0.09, 'model', 'Final model (all 420 training participants)',
        'Top-10 genera by RF importance → Optuna tuning (5-fold CV)\n→ random forest refit on n = 420', body_size=6.2)

    arrow(ax, 0.035, 0.585, 0.035, 0.105)
    arrow(ax, 0.20, 0.165, 0.20, 0.105)
    arrow(ax, 0.40, 0.165, 0.45, 0.105)
    box(ax, 0.02, 0.02, 0.29, 0.085, 'eval', 'Hold-out test (once)', 'n = 60: macro-AUC 0.776\n(95% CI 0.675–0.864)')
    box(ax, 0.33, 0.02, 0.30, 0.085, 'eval', 'External validation (no retraining)',
        'n = 108, V3–V4\nmacro-AUC 0.757 (0.686–0.829)')

    box(ax, 0.58, 0.40, 0.38, 0.27, 'assoc', 'Association analysis (interpretation only)',
        'MaAsLin2, adjusted for sex, diet,\nmedication and disease status\n\n'
        'Discovery cohort without hold-out\n(n = 3,599): three pairwise contrasts\n\n'
        'Participants not used for model development\n(n = 3,179): 20s–40s vs 50s+')
    arrow(ax, 0.77, 0.70, 0.77, 0.67)
    arrow(ax, 0.52, 0.23, 0.62, 0.40)
    box(ax, 0.66, 0.16, 0.30, 0.17, 'model', 'Marker profile of the 10-genus panel',
        'stability across the 50 splits\nassociation type (adolescent-associated,\nolder-adult-associated, progressive)\n'
        'group-specific contribution\nto classification')
    arrow(ax, 0.81, 0.40, 0.81, 0.33)

    for k, (label, kind) in enumerate([('data / split', 'data'), ('nested CV', 'cv'), ('final model', 'model'),
                                       ('association', 'assoc'), ('evaluation', 'eval')]):
        x = 0.68 + (k % 2) * 0.15
        y = 0.105 - (k // 2) * 0.035
        ax.add_patch(FancyBboxPatch((x, y), 0.018, 0.018, boxstyle='round,pad=0.002', facecolor=FILL[kind], edgecolor=EDGE[kind]))
        ax.text(x + 0.025, y + 0.009, label, fontsize=6, color=INK, va='center')
    save(fig, 'Figure2_study_design', os.path.join(STEP, 'figures'))


if __name__ == '__main__':
    main()
