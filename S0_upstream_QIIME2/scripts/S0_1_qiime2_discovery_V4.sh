#!/usr/bin/env bash
# 16S rRNA V4 amplicon processing of the discovery cohort (QIIME 2 2022.8)
#
# Input
#   sample-mapping.tsv                 QIIME 2 paired-end FASTQ manifest (PairedEndFastqManifestPhred33V2)
#   sample-metadata.tsv                sample metadata (required by taxa barplot and core-metrics)
#   2022.10.backbone.v4.nb.qza         Greengenes2 2022.10 naive Bayes classifier, V4 (515F/806R)
# Output
#   normalized_taxa/normalized_genus_table.csv    genus-level relative abundance (taxa x samples)
#   normalized_taxa/normalized_phylum_table.csv   phylum-level relative abundance (taxa x samples)
#   Alpha-diversity/shannon/alpha-diversity.tsv   Shannon index at 20,000 reads per sample
set -euo pipefail

# 1. Import paired-end reads
qiime tools import --type 'SampleData[PairedEndSequencesWithQuality]' \
  --input-path sample-mapping.tsv --input-format PairedEndFastqManifestPhred33V2 \
  --output-path demux.qza

# 2. Denoising, quality filtering, read merging and chimera removal (DADA2)
#    primer removal by trimming 19 nt (forward) and 20 nt (reverse); truncation at 240/160 nt;
#    other parameters at defaults (max-ee 2/2, trunc-q 2, min-overlap 12)
qiime dada2 denoise-paired --i-demultiplexed-seqs demux.qza \
  --p-trim-left-f 19 --p-trim-left-r 20 --p-trunc-len-f 240 --p-trunc-len-r 160 \
  --p-chimera-method pooled --p-n-threads 64 \
  --o-representative-sequences rep-seqs.qza --o-table table.qza --o-denoising-stats stats-dada2.qza

# 3. Taxonomic classification of ASVs (confidence 0.7, default)
qiime feature-classifier classify-sklearn --i-classifier 2022.10.backbone.v4.nb.qza \
  --i-reads rep-seqs.qza --o-classification taxonomy.qza

# 4. Taxon count tables at each rank from the non-rarefied ASV table, converted to relative abundance
qiime taxa barplot --i-table table.qza --i-taxonomy taxonomy.qza \
  --m-metadata-file sample-metadata.tsv --o-visualization taxa-bar-plots.qzv
qiime tools export --input-path taxa-bar-plots.qzv --output-path exported-taxa-bar-plots
python S0_3_normalize_taxa.py exported-taxa-bar-plots normalized_taxa

# 5. Shannon index after rarefaction to 20,000 reads (every sample had at least 20,002 non-chimeric reads)
qiime phylogeny align-to-tree-mafft-fasttree --i-sequences rep-seqs.qza --p-n-threads 64 \
  --o-alignment aligned-rep-seqs.qza --o-masked-alignment masked-aligned-rep-seqs.qza \
  --o-tree unrooted-tree.qza --o-rooted-tree rooted-tree.qza
qiime diversity core-metrics-phylogenetic --i-phylogeny rooted-tree.qza --i-table table.qza \
  --p-sampling-depth 20000 --m-metadata-file sample-metadata.tsv --output-dir core-metrics-result
qiime tools export --input-path core-metrics-result/shannon_vector.qza --output-path Alpha-diversity/shannon
