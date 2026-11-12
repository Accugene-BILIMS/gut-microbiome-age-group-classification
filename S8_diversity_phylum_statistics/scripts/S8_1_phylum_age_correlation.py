"""Correlation between age and phylum abundance within each age group (Supplementary Table S4).

Input
  S0_upstream_QIIME2/results/discovery_V4/normalized_phylum_table.csv   phylum relative abundance (taxa x samples)
  S1_data_processing/results/discovery_covariates.csv                   Age_group3 and Age of the 3,659 participants
Output (S8_diversity_phylum_statistics/results/)
  TableS4_phylum_age_correlation.csv   per age group and phylum: n, mean relative abundance, Spearman rho, P and q,
                                       q < 0.05 and |rho| >= 0.1, Pearson r, P and q, and CLR Spearman rho and q

Process
  1) Relative abundance (%) of Firmicutes (sum of all Firmicutes_* phyla), Bacteroidota and Proteobacteria.
  2) Within each age group: Spearman (and Pearson) correlation of age with each phylum; Benjamini-Hochberg
     correction across the nine tests.
  3) Sensitivity analysis: centred log-ratio over all phyla (Firmicutes_* combined; pseudocount = half the smallest
     non-zero value), Spearman correlation with age, Benjamini-Hochberg across the nine tests.

Usage
  python S8_1_phylum_age_correlation.py
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from statsmodels.stats.multitest import multipletests

PKG = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
PHYLUM_TABLE = os.path.join(PKG, 'S0_upstream_QIIME2', 'results', 'discovery_V4', 'normalized_phylum_table.csv')
COVARIATES = os.path.join(PKG, 'S1_data_processing', 'results', 'discovery_covariates.csv')
RES = os.path.join(PKG, 'S8_diversity_phylum_statistics', 'results')
GROUPS = ['10s', '20s_40s', '50s_plus']


def main():
    meta = pd.read_csv(COVARIATES, index_col='Sample_name')
    ph = pd.read_csv(PHYLUM_TABLE, index_col=0).T.loc[meta.index] * 100
    ph.columns = [c.split(';')[-1] for c in ph.columns]
    firm = [c for c in ph.columns if c.startswith('p__Firmicutes')]
    rel = pd.DataFrame({'Firmicutes': ph[firm].sum(axis=1), 'Bacteroidota': ph['p__Bacteroidota'],
                        'Proteobacteria': ph['p__Proteobacteria']})

    full = pd.concat([rel['Firmicutes'], ph.drop(columns=firm)], axis=1)
    pseudo = full[full > 0].min().min() / 2
    logx = np.log(full + pseudo)
    clr = logx.sub(logx.mean(axis=1), axis=0).rename(columns={'p__Bacteroidota': 'Bacteroidota',
                                                               'p__Proteobacteria': 'Proteobacteria'})

    rows = []
    for g in GROUPS:
        m = meta['Age_group3'] == g
        age = meta.loc[m, 'Age']
        for phylum in rel.columns:
            rho, p_s = spearmanr(age, rel.loc[m, phylum])
            r, p_p = pearsonr(age, rel.loc[m, phylum])
            rho_c, p_c = spearmanr(age, clr.loc[m, phylum])
            rows.append({'age_group': g, 'n': int(m.sum()), 'phylum': phylum,
                         'mean_rel_abundance_%': rel.loc[m, phylum].mean(),
                         'spearman_rho': rho, 'spearman_p': p_s, 'pearson_r': r, 'pearson_p': p_p,
                         'clr_spearman_rho': rho_c, 'clr_spearman_p': p_c})
    out = pd.DataFrame(rows)
    out['spearman_q'] = multipletests(out['spearman_p'], method='fdr_bh')[1]
    out['pearson_q'] = multipletests(out['pearson_p'], method='fdr_bh')[1]
    out['clr_spearman_q'] = multipletests(out['clr_spearman_p'], method='fdr_bh')[1]
    out['meaningful_spearman'] = (out['spearman_q'] < 0.05) & (out['spearman_rho'].abs() >= 0.1)
    cols = ['age_group', 'n', 'phylum', 'mean_rel_abundance_%', 'spearman_rho', 'spearman_p', 'spearman_q',
            'meaningful_spearman', 'pearson_r', 'pearson_p', 'pearson_q', 'clr_spearman_rho', 'clr_spearman_q']
    os.makedirs(RES, exist_ok=True)
    out[cols].to_csv(os.path.join(RES, 'TableS4_phylum_age_correlation.csv'), index=False)
    print(out[['age_group', 'phylum', 'spearman_rho', 'spearman_q', 'meaningful_spearman']].round(3).to_string(index=False))


if __name__ == '__main__':
    main()
