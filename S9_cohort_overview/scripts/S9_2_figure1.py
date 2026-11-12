"""Figure 1. Age distribution of the discovery cohort.

Input
  S1_data_processing/results/discovery_covariates.csv   Age and Age_group3 of the 3,659 participants
Output (S9_cohort_overview/figures/)
  Figure1_cohort_age_distribution.tiff / .png

Process
  Number of participants per age decade, coloured by the three age groups (10s, 20s-40s, 50s and older).

Usage
  python S9_2_figure1.py
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PKG = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, os.path.join(PKG, 'common'))
from figure_style import GROUP_COLOR, GROUP_LABEL, INK, save, ygrid  # noqa: E402

STEP = os.path.join(PKG, 'S9_cohort_overview')
COVARIATES = os.path.join(PKG, 'S1_data_processing', 'results', 'discovery_covariates.csv')


def main():
    meta = pd.read_csv(COVARIATES)
    decade = (meta['Age'] // 10 * 10).astype(int)
    counts = decade.value_counts().sort_index()
    group_of = {d: ('10s' if d < 20 else '20s_40s' if d < 50 else '50s_plus') for d in counts.index}

    fig, ax = plt.subplots(figsize=(3.3, 2.6))
    x = np.arange(len(counts))
    ax.bar(x, counts.values, width=0.7, color=[GROUP_COLOR[group_of[d]] for d in counts.index])
    for xi, v in zip(x, counts.values):
        ax.text(xi, v + 20, f'{v:,}', ha='center', fontsize=7, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{d}s' for d in counts.index])
    ax.set_xlabel('Age decade')
    ax.set_ylabel('Participants (n)')
    ygrid(ax)
    totals = meta['Age_group3'].value_counts()
    handles = [plt.Rectangle((0, 0), 1, 1, color=GROUP_COLOR[g]) for g in GROUP_COLOR]
    ax.legend(handles, [f'{GROUP_LABEL[g]} (n = {totals[g]:,}, {totals[g] / len(meta):.1%})' for g in GROUP_COLOR],
              frameon=False, loc='upper right')
    ax.set_ylim(0, 2350)   # room for the legend above the tallest bar
    save(fig, 'Figure1_cohort_age_distribution', os.path.join(STEP, 'figures'), width='single')


if __name__ == '__main__':
    main()
