"""Figure 3. Phylum-level composition and age correlations.

Input
  S0_upstream_QIIME2/results/discovery_V4/normalized_phylum_table.csv
  S1_data_processing/results/discovery_covariates.csv               Age_group3 and Age
  S8_diversity_phylum_statistics/results/TableS4_phylum_age_correlation.csv
Output (S8_diversity_phylum_statistics/figures/)
  Figure3_phylum_composition_correlation.tiff / .png

Process
  (A) Mean relative abundance of Firmicutes (sum of Firmicutes_* phyla), Bacteroidota and Proteobacteria by decade.
  (B) Spearman rho between age and phylum abundance within each age group (Benjamini-Hochberg across nine tests;
      * q < 0.05, ** q < 0.01, *** q < 0.001); cells with q < 0.05 and |rho| >= 0.1 are outlined.

Usage
  python S8_5_figure3.py
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

PKG = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, os.path.join(PKG, 'common'))
from figure_style import GROUP_LABEL, INK, MUTED, PHYLUM_COLOR, panel_label, save, taxon, ygrid  # noqa: E402

STEP = os.path.join(PKG, 'S8_diversity_phylum_statistics')
PHYLUM_TABLE = os.path.join(PKG, 'S0_upstream_QIIME2', 'results', 'discovery_V4', 'normalized_phylum_table.csv')
COVARIATES = os.path.join(PKG, 'S1_data_processing', 'results', 'discovery_covariates.csv')
PHYLA = ['Firmicutes', 'Bacteroidota', 'Proteobacteria']
DIVERGING = LinearSegmentedColormap.from_list('div', ['#2a78d6', '#f3f4f6', '#e34948'])


def stars(q):
    return '***' if q < 0.001 else '**' if q < 0.01 else '*' if q < 0.05 else ''


def main():
    meta = pd.read_csv(COVARIATES, index_col='Sample_name')
    ph = pd.read_csv(PHYLUM_TABLE, index_col=0).T.loc[meta.index] * 100
    ph.columns = [c.split(';')[-1] for c in ph.columns]
    rel = pd.DataFrame({'Firmicutes': ph[[c for c in ph.columns if c.startswith('p__Firmicutes')]].sum(axis=1),
                        'Bacteroidota': ph['p__Bacteroidota'], 'Proteobacteria': ph['p__Proteobacteria']})
    decade = (meta['Age'] // 10 * 10).astype(int)
    means = rel.groupby(decade).mean()

    cor = pd.read_csv(os.path.join(STEP, 'results', 'TableS4_phylum_age_correlation.csv'))

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.0, 3.0), gridspec_kw={'width_ratios': [1.35, 1]})
    x = np.arange(len(means))
    for p in PHYLA:
        ax.plot(x, means[p], color=PHYLUM_COLOR[p], linewidth=2, marker='o', markersize=4,
                markeredgecolor='white', markeredgewidth=1)
        ax.text(x[-1] + 0.15, means[p].iloc[-1], taxon(p), color=INK, fontsize=7.5, va='center')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{d}s' for d in means.index])
    ax.set_xlim(-0.3, len(x) + 1.3)
    ax.set_ylabel('Mean relative abundance (%)')
    ax.set_xlabel('Age decade')
    ygrid(ax)
    panel_label(ax, 'A')

    groups = ['10s', '20s_40s', '50s_plus']
    rho = cor.pivot(index='phylum', columns='age_group', values='spearman_rho').loc[PHYLA, groups]
    q = cor.pivot(index='phylum', columns='age_group', values='spearman_q').loc[PHYLA, groups]
    im = bx.imshow(rho.values, cmap=DIVERGING, vmin=-0.3, vmax=0.3, aspect='auto')
    for i in range(len(PHYLA)):
        for j in range(len(groups)):
            r, qq = rho.iloc[i, j], q.iloc[i, j]
            bx.text(j, i, f'{r:+.2f}{stars(qq)}', ha='center', va='center', fontsize=7.5, color=INK)
            if qq < 0.05 and abs(r) >= 0.1:
                bx.add_patch(plt.Rectangle((j - 0.48, i - 0.48), 0.96, 0.96, fill=False, edgecolor=INK, linewidth=1.2))
    bx.set_xticks(range(len(groups)))
    bx.set_xticklabels([f'{GROUP_LABEL[g]}\n(n={(meta.Age_group3 == g).sum():,})' for g in groups], fontsize=7)
    bx.set_yticks(range(len(PHYLA)))
    bx.set_yticklabels([taxon(p) for p in PHYLA])
    bx.tick_params(length=0)
    for s in bx.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=bx, fraction=0.05, pad=0.03)
    cb.set_label('Spearman ρ (age vs abundance)', fontsize=7)
    cb.outline.set_visible(False)
    bx.set_title('Within-group correlation with age', fontsize=8.5, color=INK)
    panel_label(bx, 'B', x=-0.32)
    fig.text(0.01, -0.04, 'B: BH-FDR across 9 tests; *q<0.05, **q<0.01, ***q<0.001; outlined: q<0.05 and |ρ| ≥ 0.1.',
             fontsize=6.5, color=MUTED)
    fig.tight_layout()
    save(fig, 'Figure3_phylum_composition_correlation', os.path.join(STEP, 'figures'))


if __name__ == '__main__':
    main()
