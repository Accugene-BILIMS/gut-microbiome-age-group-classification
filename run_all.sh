#!/usr/bin/env bash
# Runs every analysis step from the QIIME 2 output tables (S0 results) to the figures and tables of the manuscript.
#
# Input
#   S0_upstream_QIIME2/results/   genus and phylum relative-abundance tables and Shannon entropy
#   metadata/                      participant metadata
# Output
#   results/, figures/ and model/ of steps S1-S9 (existing files are overwritten)
# Process
#   Steps run in order; each step reads only the outputs of earlier steps. Approximate run time on a 6-core laptop:
#   S5 about 30 min, S8 about 35 min, every other step a few minutes or less (total about 75 min).
#
# Usage
#   bash run_all.sh            (python and Rscript on PATH; see README.md for versions)
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python}
step() { echo; echo "=== $1 ($(date +%T))"; }

step S1; (cd S1_data_processing/scripts && $PY S1_build_analysis_tables.py)
step S2; (cd S2_algorithm_comparison/scripts && $PY S2_1_algorithm_comparison.py && $PY S2_2_figure5.py)
step S3; (cd S3_genus_selection_maaslin2/scripts && $PY S3_1_genus_selection_nested_cv.py --workers 4 && $PY S3_2_table_s2.py)
step S4; (cd S4_panel_size/scripts && $PY S4_1_panel_size_nested_cv.py --workers 4 && $PY S4_2_panel_size_summary.py && $PY S4_3_figure6.py)
step S5; (cd S5_nested_cv_tuning/scripts && $PY S5_1_nested_cv_tuning.py --workers 6 && $PY S5_2_summary_table_s3.py && $PY S5_3_figureS1.py)
step S6; (cd S6_final_model_holdout_markers/scripts && $PY S6_1_final_model_holdout.py && $PY S6_2_panel_stability.py \
          && $PY S6_3_maaslin2_associations.py --cohort discovery && $PY S6_3_maaslin2_associations.py --cohort independent \
          && $PY S6_4_marker_profile.py && $PY S6_5_tables_2_3_4.py && $PY S6_6_figure7.py && $PY S6_7_figure8.py)
step S7; (cd S7_external_validation/scripts && $PY S7_1_external_validation.py && $PY S7_2_tables_5_s7_s8.py && $PY S7_3_figure9.py)
step S8; (cd S8_diversity_phylum_statistics/scripts && $PY S8_1_phylum_age_correlation.py && Rscript S8_2_beta_diversity.R \
          && $PY S8_3_shannon_pairwise.py && $PY S8_4_table_s5.py && $PY S8_5_figure3.py && $PY S8_6_figure4.py)
step S9; (cd S9_cohort_overview/scripts && $PY S9_1_table1.py && $PY S9_2_figure1.py && $PY S9_3_figure2.py)
echo; echo "=== done ($(date +%T))"
