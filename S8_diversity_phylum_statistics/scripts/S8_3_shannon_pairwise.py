"""Pairwise comparison of Shannon entropy between age decades (Supplementary Table S6).

Input
  S0_upstream_QIIME2/results/discovery_V4/alpha_diversity_shannon.tsv   Shannon entropy at 20,000 reads
  S1_data_processing/results/discovery_covariates.csv                   Age of the 3,659 participants
Output (S8_diversity_phylum_statistics/results/)
  TableS6_shannon_pairwise.csv   all 28 pairs of age decades: n, medians, two-sided Wilcoxon rank-sum P,
                                 Benjamini-Hochberg q across the 28 tests

Usage
  python S8_3_shannon_pairwise.py
"""
import itertools
import os

import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

PKG = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
SHANNON = os.path.join(PKG, 'S0_upstream_QIIME2', 'results', 'discovery_V4', 'alpha_diversity_shannon.tsv')
COVARIATES = os.path.join(PKG, 'S1_data_processing', 'results', 'discovery_covariates.csv')
RES = os.path.join(PKG, 'S8_diversity_phylum_statistics', 'results')


def main():
    meta = pd.read_csv(COVARIATES, index_col='Sample_name')
    sh = pd.read_csv(SHANNON, sep='\t', index_col=0)['shannon_entropy']
    meta['shannon'] = sh.loc[meta.index]
    meta['decade'] = (meta['Age'] // 10 * 10).astype(int)
    decades = sorted(meta['decade'].unique())
    rows = []
    for a, b in itertools.combinations(decades, 2):
        xa, xb = meta.loc[meta.decade == a, 'shannon'], meta.loc[meta.decade == b, 'shannon']
        rows.append({'decade_a': f'{a}s', 'decade_b': f'{b}s', 'n_a': len(xa), 'n_b': len(xb),
                     'median_a': xa.median(), 'median_b': xb.median(),
                     'p_wilcoxon': mannwhitneyu(xa, xb, alternative='two-sided').pvalue})
    pw = pd.DataFrame(rows)
    pw['q_bh'] = multipletests(pw['p_wilcoxon'], method='fdr_bh')[1]
    os.makedirs(RES, exist_ok=True)
    pw.round(5).to_csv(os.path.join(RES, 'TableS6_shannon_pairwise.csv'), index=False)
    print(pw[pw.decade_a == '10s'].round(4).to_string(index=False))


if __name__ == '__main__':
    main()
