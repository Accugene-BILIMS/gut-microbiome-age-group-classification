#!/usr/bin/env bash
# 16S rRNA V3-V4 amplicon processing of the external cohort (QIIME 2 2022.8)
#
# Input
#   sample-mapping.tsv                       QIIME 2 paired-end FASTQ manifest (PairedEndFastqManifestPhred33V2)
#   sample-metadata.tsv                      sample metadata (required by taxa barplot)
#   2022.10.backbone.full-length.nb.qza      Greengenes2 2022.10 naive Bayes classifier, full length
# Output
#   normalized_taxa/normalized_genus_table.csv   genus-level relative abundance (taxa x samples)
set -euo pipefail

# 1. Import paired-end reads
qiime tools import --type 'SampleData[PairedEndSequencesWithQuality]' \
  --input-path sample-mapping.tsv --input-format PairedEndFastqManifestPhred33V2 \
  --output-path demux.qza

# 2. Denoising, quality filtering, read merging and chimera removal (DADA2)
#    trimming 19 nt (forward) and 20 nt (reverse); truncation at 260/240 nt;
#    other parameters at defaults (max-ee 2/2, trunc-q 2, min-overlap 12)
qiime dada2 denoise-paired --i-demultiplexed-seqs demux.qza \
  --p-trim-left-f 19 --p-trim-left-r 20 --p-trunc-len-f 260 --p-trunc-len-r 240 \
  --p-chimera-method consensus --p-n-threads 64 \
  --o-representative-sequences rep-seqs.qza --o-table table.qza --o-denoising-stats stats-dada2.qza

# 3. Taxonomic classification of ASVs (confidence 0.7, default)
qiime feature-classifier classify-sklearn --i-classifier 2022.10.backbone.full-length.nb.qza \
  --i-reads rep-seqs.qza --o-classification taxonomy.qza

# 4. Taxon count tables at each rank, converted to relative abundance
qiime taxa barplot --i-table table.qza --i-taxonomy taxonomy.qza \
  --m-metadata-file sample-metadata.tsv --o-visualization taxa-bar-plots.qzv
qiime tools export --input-path taxa-bar-plots.qzv --output-path exported-taxa-bar-plots
python S0_3_normalize_taxa.py exported-taxa-bar-plots normalized_taxa
