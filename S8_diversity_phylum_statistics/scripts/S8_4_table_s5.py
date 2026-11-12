"""Supplementary Table S5. PERMANOVA and PERMDISP.

Input (S8_diversity_phylum_statistics/results/)
  permanova.csv, permanova_covariates.csv, permdisp.csv, balanced_subsamples.csv
Output (S8_diversity_phylum_statistics/results/)
  TableS5_PERMANOVA_PERMDISP.csv   PERMANOVA (age group only; with covariates, marginal tests), PERMDISP, and the
                                   minimum, median and maximum across the 30 class-balanced subsamples

Usage
  python S8_4_table_s5.py
"""
import os

import pandas as pd

PKG = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
RES = os.path.join(PKG, 'S8_diversity_phylum_statistics', 'results')


def balanced_summary():
    """PERMANOVA and PERMDISP in 30 class-balanced subsamples (160 per group): min, median, max across draws."""
    b = pd.read_csv(os.path.join(RES, 'balanced_subsamples.csv')).drop(columns='draw')
    names = {'permanova_R2': 'PERMANOVA R2', 'permanova_p': 'PERMANOVA P', 'permdisp_F': 'PERMDISP F', 'permdisp_p': 'PERMDISP P',
             'X10s': 'distance to centroid, 10s', 'X20s_40s': 'distance to centroid, 20s–40s',
             'X50s_plus': 'distance to centroid, 50s+'}
    s = b.agg(['min', 'median', 'max']).T.round(4).rename(index=names)
    s = s.rename(columns={'min': 'subsamples_min', 'median': 'subsamples_median', 'max': 'subsamples_max'})
    return s.rename_axis('term').reset_index().assign(model='balanced_subsamples_30x160')


def main():
    t = pd.concat([pd.read_csv(os.path.join(RES, 'permanova.csv')),
                   pd.read_csv(os.path.join(RES, 'permanova_covariates.csv')),
                   pd.read_csv(os.path.join(RES, 'permdisp.csv')).assign(model='PERMDISP'),
                   balanced_summary()], ignore_index=True)
    t.to_csv(os.path.join(RES, 'TableS5_PERMANOVA_PERMDISP.csv'), index=False)
    print(t.to_string(index=False))


if __name__ == '__main__':
    main()
