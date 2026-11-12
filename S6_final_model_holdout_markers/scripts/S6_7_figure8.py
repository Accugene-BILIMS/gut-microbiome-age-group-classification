"""Figure 8. Age associations of the 10-genus signature.

Input
  S1_data_processing/results/discovery_genus_relab_cut0.01.csv, split_sample_ids.csv
  S6_final_model_holdout_markers/results/panel_marker_profile.csv, final_panel.csv, panel_associations.csv
Output (S6_final_model_holdout_markers/figures/)
  Figure8_panel_markers.tiff / .png

Process
  (A) Discovery cohort excluding the hold-out set (n = 3,599): boxes show log10 relative abundance (%) among
      participants in which the genus was detected; percentages give the detection rate per age group; genera
      grouped by association type; stars give the MaAsLin2 q of each pairwise contrast (*** < 0.001, ** < 0.01,
      * < 0.05).
  (B) Group-specific contribution in the final model: drop in one-vs-rest AUC when the genus is permuted.

Usage
  python S6_7_figure8.py
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

PKG = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, os.path.join(PKG, 'common'))
from analysis_common import GENUS_TABLE, SPLIT_IDS  # noqa: E402
from figure_style import (GROUP_COLOR, GROUP_LABEL, INK, MUTED, TYPE_LABEL, panel_label, save,  # noqa: E402
                          short_genus, taxon)

STEP = os.path.join(PKG, 'S6_final_model_holdout_markers')
GROUPS = ['10s', '20s_40s', '50s_plus']
TYPE_ORDER = ['adolescent-associated', 'older-adult-associated', 'progressive']
CONTRASTS = [('10s_vs_20s_40s', 0, 1), ('10s_vs_50s_plus', 0, 2), ('20s_40s_vs_50s_plus', 1, 2)]


def stars(q):
    return '***' if q < 0.001 else '**' if q < 0.01 else '*' if q < 0.05 else None


def main():
    genus = pd.read_csv(GENUS_TABLE, index_col='Sample_name')
    holdout = set(pd.read_csv(SPLIT_IDS).query("split == 'holdout'")['Sample_name'])
    disc = genus.loc[[s for s in genus.index if s not in holdout]]
    profile = pd.read_csv(os.path.join(STEP, 'results', 'panel_marker_profile.csv'))
    panel = pd.read_csv(os.path.join(STEP, 'results', 'final_panel.csv')).set_index('genus')
    assoc = pd.read_csv(os.path.join(STEP, 'results', 'panel_associations.csv')).set_index('genus')
    profile['type_order'] = profile['association_type'].map({t: i for i, t in enumerate(TYPE_ORDER)})
    profile = profile.sort_values(['type_order', 'rank'])

    fig = plt.figure(figsize=(7.0, 7.6))
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 0.75], hspace=0.55)
    gsa = gs[0:2].subgridspec(2, 5, hspace=0.6, wspace=0.45)
    axes = [fig.add_subplot(gsa[i // 5, i % 5]) for i in range(10)]

    for ax, (_, r) in zip(axes, profile.iterrows()):
        tax = panel.loc[r['genus'], 'taxonomy']
        vals = [np.log10(disc.loc[(disc.Age_group3 == g) & (disc[tax] > 0), tax]) for g in GROUPS]
        bp = ax.boxplot(vals, widths=0.6, patch_artist=True, showfliers=False,
                        medianprops={'color': 'white', 'linewidth': 1.2}, whiskerprops={'color': MUTED, 'linewidth': 0.7},
                        capprops={'color': MUTED, 'linewidth': 0.7})
        for b, g in zip(bp['boxes'], GROUPS):
            b.set(facecolor=GROUP_COLOR[g], edgecolor=GROUP_COLOR[g])
        top = max(np.percentile(v, 75) + 1.5 * (np.percentile(v, 75) - np.percentile(v, 25)) for v in vals)
        top = min(top, max(v.max() for v in vals))
        level = 0
        for label, i, j in CONTRASTS:
            s = stars(assoc.loc[r['genus'], f'q_{label}'])
            if s is None:
                continue
            yy = top + 0.25 + level * 0.6
            ax.plot([i + 1, i + 1, j + 1, j + 1], [yy - 0.08, yy, yy, yy - 0.08], color=INK, linewidth=0.6)
            ax.text((i + j) / 2 + 1, yy - 0.02, s, ha='center', va='bottom', fontsize=6, color=INK)
            level += 1
        ax.set_ylim(top=top + 0.5 + level * 0.6)
        # detection rate under each box (the boxes show detected samples only)
        det = [(disc.loc[disc.Age_group3 == g, tax] > 0).mean() * 100 for g in GROUPS]
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels([f'{d:.0f}%' for d in det])   # group identity by color (legend above the panels)
        ax.tick_params(axis='y', labelsize=6)
        n_sig = r['significant_contrasts'].split(':')[0]  # e.g. '2/3' = contrasts with q < 0.05
        ax.set_title(f"{taxon(short_genus(tax))}\n{TYPE_LABEL[r['association_type']]} ({n_sig})", fontsize=6.8, color=INK)
    axes[0].set_ylabel('log10 relative abundance (%)\nwhen detected', fontsize=6.5)
    axes[5].set_ylabel('log10 relative abundance (%)\nwhen detected', fontsize=6.5)
    panel_label(axes[0], 'A', x=-0.55, y=1.25)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(facecolor=GROUP_COLOR[g], edgecolor=GROUP_COLOR[g], label=GROUP_LABEL[g]) for g in GROUPS],
               loc='lower center', bbox_to_anchor=(0.5, 0.95), ncol=3, frameon=False, handlelength=1.2, columnspacing=1.5,
               title='Age group (percentages below the boxes give the detection rate)')

    bx = fig.add_subplot(gs[2])
    mat = profile[[f'perm_drop_auc_{g}' for g in GROUPS]].T.values
    cmap = LinearSegmentedColormap.from_list('seq', ['#f3f4f6', '#9ec5f4', '#2a78d6', '#104281'])
    im = bx.imshow(mat, cmap=cmap, vmin=0, vmax=0.12, aspect='auto')
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            bx.text(j, i, f'{v:.3f}', ha='center', va='center', fontsize=6.3, color='white' if v > 0.06 else INK)
    bx.set_yticks(range(3))
    bx.set_yticklabels([GROUP_LABEL[g] for g in GROUPS])
    bx.set_xticks(range(len(profile)))
    bx.set_xticklabels([taxon(g.replace('g__', '')) for g in profile['genus']], rotation=30, ha='right', fontsize=6.5)
    bx.tick_params(length=0)
    for s in bx.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=bx, fraction=0.025, pad=0.01)
    cb.set_label('AUC drop when permuted', fontsize=6.5)
    cb.outline.set_visible(False)
    bx.set_title('Contribution of each genus to identifying each age group (final model, training-set CV)', fontsize=7.5, color=INK)
    panel_label(bx, 'B', x=-0.08, y=1.12)
    save(fig, 'Figure8_panel_markers', os.path.join(STEP, 'figures'))


if __name__ == '__main__':
    main()
