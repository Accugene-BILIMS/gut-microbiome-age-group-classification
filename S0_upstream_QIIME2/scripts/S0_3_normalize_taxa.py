"""Conversion of QIIME 2 taxon count tables to relative abundance.

Input
  <exported-taxa-bar-plots>/level-1.csv ... level-7.csv   taxon counts per sample (samples x taxa,
                                                          followed by sample metadata columns)
Output
  <out_dir>/normalized_<rank>_table.csv                   relative abundance (taxa x samples);
                                                          rank = domain, phylum, class, order,
                                                          family, genus, species

Process
  The sample metadata columns are removed and each sample's counts are divided by its total count,
  so that the abundances of each sample sum to 1.

Usage
  python S0_3_normalize_taxa.py <exported-taxa-bar-plots> <out_dir>
"""
import os
import sys
import pandas as pd

RANKS = ['domain', 'phylum', 'class', 'order', 'family', 'genus', 'species']


def main(data_path, out_path):
    os.makedirs(out_path, exist_ok=True)
    for i, rank in enumerate(RANKS):
        table = pd.read_csv(os.path.join(data_path, f'level-{i + 1}.csv'), index_col=0)
        counts = table.loc[:, :'Description'].drop(columns='Description')
        counts.index.name = '#OTU ID'
        rel = counts.div(counts.sum(axis=1), axis=0)
        rel.T.to_csv(os.path.join(out_path, f'normalized_{rank}_table.csv'))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
