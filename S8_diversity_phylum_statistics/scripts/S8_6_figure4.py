"""Figure 4. Alpha and beta diversity.

Input
  S0_upstream_QIIME2/results/discovery_V4/alpha_diversity_shannon.tsv
  S1_data_processing/results/discovery_covariates.csv               Age
  S8_diversity_phylum_statistics/results/TableS6_shannon_pairwise.csv, pcoa_scores.csv, betadisper_distances.csv,
    permanova.csv, permanova_covariates.csv, permdisp.csv
Output (S8_diversity_phylum_statistics/figures/)
  Figure4_alpha_beta_diversity.tiff / .png

Process
  (A) Shannon entropy by age decade; brackets mark comparisons with the 10s with q < 0.05 (Benjamini-Hochberg
      across all 28 pairwise Wilcoxon tests).
  (B) PCoA of Bray-Curtis dissimilarities with 95% normal ellipses and group centroids; PERMANOVA without and with
      covariate adjustment.
  (C) Distance to the age-group centroid (PERMDISP).

Usage
  python S8_6_figure4.py
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse

PKG = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, os.path.join(PKG, 'common'))
from figure_style import GROUP_COLOR, GROUP_LABEL, INK, MUTED, panel_label, save, ygrid  # noqa: E402

STEP = os.path.join(PKG, 'S8_diversity_phylum_statistics')
SHANNON = os.path.join(PKG, 'S0_upstream_QIIME2', 'results', 'discovery_V4', 'alpha_diversity_shannon.tsv')
COVARIATES = os.path.join(PKG, 'S1_data_processing', 'results', 'discovery_covariates.csv')
GROUPS = ['10s', '20s_40s', '50s_plus']


def stars(q):
    return '***' if q < 0.001 else '**' if q < 0.01 else '*' if q < 0.05 else 'ns'


def ellipse(ax, x, y, color):
    cov = np.cov(x, y)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    w, h = 2 * np.sqrt(vals * 5.991)  # 95% for 2 d.f.
    ax.add_patch(Ellipse((x.mean(), y.mean()), w, h, angle=angle, fill=False, edgecolor=color, linewidth=1.6))


def main():
    meta = pd.read_csv(COVARIATES, index_col='Sample_name')
    sh = pd.read_csv(SHANNON, sep='\t', index_col=0)['shannon_entropy']
    meta['shannon'] = sh.loc[meta.index]
    meta['decade'] = (meta['Age'] // 10 * 10).astype(int)
    decades = sorted(meta['decade'].unique())

    pw = pd.read_csv(os.path.join(STEP, 'results', 'TableS6_shannon_pairwise.csv'))

    pcoa = pd.read_csv(os.path.join(STEP, 'results', 'pcoa_scores.csv'))
    disp = pd.read_csv(os.path.join(STEP, 'results', 'betadisper_distances.csv'))
    perm = pd.read_csv(os.path.join(STEP, 'results', 'permanova.csv')).iloc[0]
    permc = pd.read_csv(os.path.join(STEP, 'results', 'permanova_covariates.csv')).set_index('term').loc['Age_group3']
    pdisp = pd.read_csv(os.path.join(STEP, 'results', 'permdisp.csv')).iloc[0]

    fig = plt.figure(figsize=(7.0, 5.8))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.05], width_ratios=[1.25, 1.25, 0.9], hspace=0.45, wspace=0.42)
    ax = fig.add_subplot(gs[0, :])
    bx = fig.add_subplot(gs[1, :2])
    cx = fig.add_subplot(gs[1, 2])

    # A
    data = [meta.loc[meta.decade == d, 'shannon'] for d in decades]
    colors = [GROUP_COLOR['10s' if d < 20 else '20s_40s' if d < 50 else '50s_plus'] for d in decades]
    bp = ax.boxplot(data, widths=0.55, patch_artist=True, showfliers=False,
                    medianprops={'color': 'white', 'linewidth': 1.5}, whiskerprops={'color': MUTED},
                    capprops={'color': MUTED})
    for b, c in zip(bp['boxes'], colors):
        b.set(facecolor=c, edgecolor=c)
    ax.set_xticks(range(1, len(decades) + 1))
    ax.set_xticklabels([f'{d}s\n(n={len(x):,})' for d, x in zip(decades, data)], fontsize=6.8)
    ax.set_ylabel('Shannon entropy')
    ygrid(ax)
    sig = pw[(pw.decade_a == '10s') & (pw.q_bh < 0.05)]
    top = max(np.percentile(x, 99) for x in data)
    step = 0.38
    for i, (_, r) in enumerate(sig.iterrows()):
        j = decades.index(int(r['decade_b'][:-1])) + 1
        yy = top + 0.2 + i * step
        ax.plot([1, 1, j, j], [yy - 0.06, yy, yy, yy - 0.06], color=INK, linewidth=0.7)
        ax.text(j + 0.08, yy, stars(r['q_bh']), ha='left', va='center', fontsize=7, color=INK)
    ax.set_ylim(top=top + 0.3 + len(sig) * step)
    ax.set_title('Shannon entropy by age decade (pairwise Wilcoxon, BH-FDR across 28 comparisons)', fontsize=8, color=INK)
    panel_label(ax, 'A', x=-0.06)

    # B
    for g in GROUPS[::-1]:
        s = pcoa[pcoa.Age_group3 == g]
        bx.scatter(s.PCo1, s.PCo2, s=3, color=GROUP_COLOR[g], alpha=0.25 if g == '20s_40s' else 0.55, linewidths=0)
    for g in GROUPS:
        s = pcoa[pcoa.Age_group3 == g]
        ellipse(bx, s.PCo1.values, s.PCo2.values, GROUP_COLOR[g])
        bx.scatter(s.PCo1.mean(), s.PCo2.mean(), s=46, color=GROUP_COLOR[g], edgecolor='white', linewidth=1.5, zorder=5,
                   label=f'{GROUP_LABEL[g]} (n={len(s):,})')
    bx.set_xlabel(f'PCo1 ({pcoa.pct_PCo1.iloc[0]:.1f}%)')
    bx.set_ylabel(f'PCo2 ({pcoa.pct_PCo2.iloc[0]:.1f}%)')
    bx.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.17), ncol=3, markerscale=0.8)
    bx.set_title(f"Bray–Curtis PCoA (genus level)\nPERMANOVA R² = {perm['R2']:.4f}, P = {perm['Pr(>F)']:.3f}; "
                 f"covariate-adjusted R² = {permc['R2']:.4f}, P = {permc['Pr(>F)']:.3f}", fontsize=7.5, color=INK)
    panel_label(bx, 'B', x=-0.1)

    # C
    dd = [disp.loc[disp.Age_group3 == g, 'distance_to_centroid'] for g in GROUPS]
    bp = cx.boxplot(dd, widths=0.55, patch_artist=True, showfliers=False,
                    medianprops={'color': 'white', 'linewidth': 1.5}, whiskerprops={'color': MUTED}, capprops={'color': MUTED})
    for b, g in zip(bp['boxes'], GROUPS):
        b.set(facecolor=GROUP_COLOR[g], edgecolor=GROUP_COLOR[g])
    cx.set_xticks([1, 2, 3])
    cx.set_xticklabels([GROUP_LABEL[g] for g in GROUPS])
    cx.set_ylabel('Distance to group centroid')
    ygrid(cx)
    cx.set_title(f"PERMDISP F = {pdisp['permutest_F']:.1f}, P = {pdisp['permutest_p']:.3f}", fontsize=8, color=INK)
    panel_label(cx, 'C', x=-0.32)
    save(fig, 'Figure4_alpha_beta_diversity', os.path.join(STEP, 'figures'))


if __name__ == '__main__':
    main()
