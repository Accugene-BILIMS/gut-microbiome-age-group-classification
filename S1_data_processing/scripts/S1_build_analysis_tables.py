"""Construction of the analysis tables and the train / hold-out split.

Input
  S0_upstream_QIIME2/results/discovery_V4/normalized_genus_table.csv
      genus-level relative abundance of the discovery cohort (taxa x samples, proportions)
  S0_upstream_QIIME2/results/external_V3V4/normalized_genus_table.csv
      genus-level relative abundance of the external cohort (taxa x samples, proportions)
  metadata/metadata_discovery.csv
      Sample_name, Age, Sex, Diet, Drug, Disease_pr of the 3,659 participants
Output (S1_data_processing/results/)
  discovery_genus_relab_cut0.01.csv   Age_group3, Age and genus relative abundance (%) per participant
  discovery_covariates.csv            Age_group3, Age, Diet, Drug, Sex, Disease_pr per participant
  split_sample_ids.csv                class-balanced training set (n = 420) and hold-out set (n = 60)
  external_genus_relab_cut0.01.csv    external cohort genus relative abundance (%) on the discovery genera

Process
  1) Age groups: 10s (10-19 years), 20s_40s (20-49 years), 50s_plus (50 years and older).
  2) Relative abundances are expressed in %; values of 0.01% or lower are set to 0 without renormalization,
     and genera that are 0 in every participant are removed.
  3) Class-balanced split: 160 participants per age group are drawn (random_state 42), and 20 per group are
     set aside as the hold-out set (random_state 42); the remaining 140 per group form the training set.
     Sampling follows the row order of metadata_discovery.csv.
  4) External cohort: the same genus columns as the discovery cohort (genera not detected = 0) and the same
     0.01% rule.

Usage
  python S1_build_analysis_tables.py
"""
import os

import numpy as np
import pandas as pd

PKG = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
DISC_GENUS = os.path.join(PKG, 'S0_upstream_QIIME2', 'results', 'discovery_V4', 'normalized_genus_table.csv')
EXT_GENUS = os.path.join(PKG, 'S0_upstream_QIIME2', 'results', 'external_V3V4', 'normalized_genus_table.csv')
META = os.path.join(PKG, 'metadata', 'metadata_discovery.csv')
OUT = os.path.join(PKG, 'S1_data_processing', 'results')

CUTOFF = 0.01  # percent
CLASSES = ['10s', '20s_40s', '50s_plus']


def age_group3(age):
    return np.where(age < 20, '10s', np.where(age < 50, '20s_40s', '50s_plus'))


def main():
    meta = pd.read_csv(META, index_col='Sample_name')
    meta['Age_group3'] = age_group3(meta['Age'].values)

    # 1-2) discovery genus table
    raw = pd.read_csv(DISC_GENUS, index_col=0).T * 100                 # participants x genera, %
    R = raw.loc[meta.index]
    F = R.where(R > CUTOFF, 0)
    F = F.loc[:, (F > 0).any()]
    genus = pd.concat([meta[['Age_group3', 'Age']], F], axis=1)
    genus.index.name = 'Sample_name'

    cov = meta[['Age_group3', 'Age', 'Diet', 'Drug', 'Sex', 'Disease_pr']].copy()

    # 3) class-balanced train / hold-out split
    base = genus[['Age_group3']].reset_index()
    bal = pd.concat([base[base['Age_group3'] == c].sample(n=160, random_state=42) for c in CLASSES]
                    ).sample(frac=1, random_state=42).reset_index(drop=True)
    bal = bal.sort_values('Age_group3', ascending=False)
    hold = bal.groupby('Age_group3', group_keys=False).sample(n=20, random_state=42).sort_values('Age_group3')
    train = bal.drop(index=hold.index).sort_values('Age_group3')
    split = pd.concat([train.assign(split='train'), hold.assign(split='holdout')])[['Sample_name', 'Age_group3', 'split']]
    split.insert(2, 'Age', split['Sample_name'].map(genus['Age']).values)

    # 4) external cohort on the discovery genera
    ext = pd.read_csv(EXT_GENUS, index_col=0).T * 100                  # participants x genera, %
    ext = ext.reindex(columns=F.columns, fill_value=0)
    ext = ext.where(ext > CUTOFF, 0)
    ext.index.name = 'Sample_name'

    os.makedirs(OUT, exist_ok=True)
    genus.to_csv(os.path.join(OUT, 'discovery_genus_relab_cut0.01.csv'))
    cov.to_csv(os.path.join(OUT, 'discovery_covariates.csv'))
    split.to_csv(os.path.join(OUT, 'split_sample_ids.csv'), index=False)
    ext.to_csv(os.path.join(OUT, 'external_genus_relab_cut0.01.csv'))
    print(f'discovery {genus.shape}, covariates {cov.shape}, split {split["split"].value_counts().to_dict()}, '
          f'external {ext.shape}')


if __name__ == '__main__':
    main()
