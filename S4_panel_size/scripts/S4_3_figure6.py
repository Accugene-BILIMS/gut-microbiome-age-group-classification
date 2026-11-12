"""Figure 6. Choice of panel size.

Input
  S4_panel_size/results/panel_size_summary.csv
Output (S4_panel_size/figures/)
  Figure6_panel_size.tiff / .png

Process
  (A) Macro-AUC (mean ± SD across 50 splits) by number of top-ranked genera; plateau 10-15 shaded; 10 genera marked.
  (B) Paired change in macro-AUC from adding the k-th genus, k = 2-15.

Usage
  python S4_3_figure6.py
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PKG = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, os.path.join(PKG, 'common'))
from figure_style import GROUP_COLOR, INK, MUTED, panel_label, save, ygrid  # noqa: E402

STEP = os.path.join(PKG, 'S4_panel_size')
BLUE = GROUP_COLOR['10s']
CHOSEN, PLATEAU = 10, (10, 15)


def main():
    t = pd.read_csv(os.path.join(STEP, 'results', 'panel_size_summary.csv')).set_index('k')
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.0, 3.0), gridspec_kw={'width_ratios': [1.4, 1]})

    ax.axvspan(*PLATEAU, color=BLUE, alpha=0.08, linewidth=0)
    ax.fill_between(t.index, t['auc_mean'] - t['auc_sd'], t['auc_mean'] + t['auc_sd'], color=BLUE, alpha=0.12, linewidth=0)
    ax.plot(t.index, t['auc_mean'], color=BLUE, linewidth=2, marker='o', markersize=3, zorder=3)
    full = t.loc[783, 'auc_mean']
    ax.axhline(full, color=MUTED, linewidth=0.8)
    ax.text(1.0, full + 0.006, f'all 783 genera: {full:.3f}', fontsize=7, color=INK)
    ax.scatter([CHOSEN], [t.loc[CHOSEN, 'auc_mean']], s=55, facecolor='white', edgecolor=INK, linewidth=1.4, zorder=4)
    ax.annotate(f"10 genera: {t.loc[CHOSEN, 'auc_mean']:.3f}", (CHOSEN, t.loc[CHOSEN, 'auc_mean']),
                xytext=(22, 0.645), fontsize=7.5, color=INK, arrowprops={'arrowstyle': '-', 'color': MUTED, 'linewidth': 0.8})
    ax.text(np.sqrt(PLATEAU[0] * PLATEAU[1]), 0.515, 'plateau\n10–15', ha='center', fontsize=7, color=BLUE)
    ticks = [1, 2, 3, 5, 10, 15, 20, 30, 50, 100, 200, 783]
    ax.set_xscale('log')
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(x) for x in ticks])
    ax.minorticks_off()
    ax.set_xlim(0.85, 950)
    ax.set_ylim(0.5, 0.81)
    ax.set_xlabel('Number of top-ranked genera (log scale)')
    ax.set_ylabel('Macro-AUC (mean ± SD, 50 splits)')
    ygrid(ax)
    panel_label(ax, 'A', x=-0.1)

    sub = t.loc[2:15]
    bx.bar(sub.index, sub['step_gain_vs_prev_k'], width=0.7, color=[BLUE if k <= 9 else '#9cc3ef' for k in sub.index])
    bx.axhline(0, color=MUTED, linewidth=0.8)
    bx.axvline(9.5, color=INK, linewidth=0.8)
    bx.text(9.7, sub['step_gain_vs_prev_k'].max() * 0.97, 'k ≥ 10:\ngain ≤ 0.003\nper genus', fontsize=7, color=INK, va='top')
    bx.set_xticks(range(2, 16))
    bx.set_xlabel('Panel size k')
    bx.set_ylabel('Change in macro-AUC (k−1 → k)')
    ygrid(bx)
    panel_label(bx, 'B', x=-0.16)
    fig.tight_layout()
    save(fig, 'Figure6_panel_size', os.path.join(STEP, 'figures'))


if __name__ == '__main__':
    main()
