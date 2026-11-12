# Code and data for "Machine Learning-Based Identification of a Gut Microbiome Signature for Age-Group Classification in a Large Korean Cohort"

This package contains the analysis code and the processed data behind every figure and table of the manuscript, from the
QIIME 2 output tables to the final figures and tables. Running the package from the provided input files reproduces the
reported results (see [Reproducibility](#reproducibility)).

The scripts are those used for the analyses reported in the manuscript. To make the package self-contained and easy to
run, input and output paths were set relative to the package folder, output files were renamed after the figures and
tables they support, and outputs not used in the manuscript were removed. The analysis code, parameters and random seeds
were not changed.

## Data

| Data | Location |
|---|---|
| Raw 16S rRNA amplicon reads, discovery cohort (V4, n = 3,659) | NCBI SRA, BioProject PRJNA1559924 |
| Raw 16S rRNA amplicon reads, external cohort (V3–V4, n = 108) | NCBI SRA, BioProject PRJNA1559905 |
| Genus- and phylum-level relative abundance, Shannon entropy | `S0_upstream_QIIME2/results/` |
| Participant metadata (age group) | `metadata/` |

Participants are identified by pseudonyms (discovery S0001–S3659, external E001–E108), which are the sample names used
in the SRA submissions. The NCBI BioSample records give the age group of each participant.

**Public version.** `metadata/metadata_discovery.csv` and `metadata/metadata_external.csv` give the age group of each
participant only. To protect participant privacy, participant-level covariates (exact age, sex, diet, medication use and
disease status) are not public; they are available to qualified researchers on request to the corresponding author
(see [Contact](#contact) and the Data Availability Statement of the article). Accordingly,
`S1_data_processing/results/discovery_covariates.csv` is not included and the `Age` column is removed from
`S1_data_processing/results/discovery_genus_relab_cut0.01.csv` and `split_sample_ids.csv`. The code is the same as in the
full analysis, and all results, figures and tables of every step are included.

## Folder structure

```
README.md, requirements.txt, R_packages.txt, run_all.sh
metadata/                          participant metadata (pseudonymized)
common/                            code shared by several steps
  analysis_common.py                 data loaders, cross-validation splitter (repeated stratified 5-fold x 10, seed 42), metrics
  group_metrics.py                   per-group metrics with stratified bootstrap 95% CIs (Tables 2 and 5)
  figure_style.py                    figure colours and export (TIFF, 300 dpi, 85/180 mm), ROC and confusion-matrix panels
  run_maaslin2.R                     MaAsLin2 settings (Supplementary Table S2, Table 4, Figure 8)
S0_upstream_QIIME2/ ... S9_cohort_overview/
  scripts/                           run in the order of their number (S2_1, S2_2, ...)
  results/                           tables (csv/json); files named Table*/TableS* are the manuscript tables
  figures/                           manuscript figures (TIFF as submitted, PNG preview)
  model/                             (S6 only) the final random forest
```

Every script starts with a header that lists its inputs, outputs and processing steps.

## Analysis steps

| Step | Content | Manuscript items | Run time* |
|---|---|---|---|
| S0 `S0_upstream_QIIME2` | QIIME 2 processing of the raw reads (DADA2, Greengenes2 classification), relative abundance, Shannon entropy at 20,000 reads | input of all steps | hours (server) |
| S1 `S1_data_processing` | analysis tables: relative abundance in %, values ≤ 0.01% set to 0 (no renormalization), 783 genera; class-balanced training (n = 420) and hold-out (n = 60) sets; external cohort on the same genera | Section 2.3, 2.5 | < 1 min |
| S2 `S2_algorithm_comparison` | five classifiers, all 783 genera, 50 cross-validation splits | Figure 5, Table S1 | 1.5 min |
| S3 `S3_genus_selection_maaslin2` | genus selection by MaAsLin2 inside the training folds | Table S2 | 1.5 min |
| S4 `S4_panel_size` | performance by number of top-ranked genera (nested cross-validation) | Figure 6 | 1 min |
| S5 `S5_nested_cv_tuning` | nested cross-validation with genus selection and Optuna tuning inside each training fold | Table S3, Figure S1, Section 3.4 | 30 min |
| S6 `S6_final_model_holdout_markers` | final 10-genus random forest, single hold-out evaluation, panel stability, MaAsLin2 associations, permutation importance | Figures 7–8, Tables 2–4 | 3 min |
| S7 `S7_external_validation` | final model applied to the external cohort without retraining | Figure 9, Table 5, Tables S7–S8 | 1 min |
| S8 `S8_diversity_phylum_statistics` | phylum–age correlations, PERMANOVA, PERMDISP, Shannon entropy tests | Figures 3–4, Tables S4–S6 | 35 min |
| S9 `S9_cohort_overview` | cohort characteristics, age distribution, study design | Figures 1–2, Table 1 | < 1 min |

\* On a 6-core laptop (macOS, Intel).

### Figures and tables

| Item | File | Script |
|---|---|---|
| Figure 1 | `S9_cohort_overview/figures/Figure1_cohort_age_distribution.tiff` | `S9_2_figure1.py` |
| Figure 2 | `S9_cohort_overview/figures/Figure2_study_design.tiff` | `S9_3_figure2.py` |
| Figure 3 | `S8_diversity_phylum_statistics/figures/Figure3_phylum_composition_correlation.tiff` | `S8_5_figure3.py` |
| Figure 4 | `S8_diversity_phylum_statistics/figures/Figure4_alpha_beta_diversity.tiff` | `S8_6_figure4.py` |
| Figure 5 | `S2_algorithm_comparison/figures/Figure5_classifier_comparison.tiff` | `S2_2_figure5.py` |
| Figure 6 | `S4_panel_size/figures/Figure6_panel_size.tiff` | `S4_3_figure6.py` |
| Figure 7 | `S6_final_model_holdout_markers/figures/Figure7_internal_validation.tiff` | `S6_6_figure7.py` |
| Figure 8 | `S6_final_model_holdout_markers/figures/Figure8_panel_markers.tiff` | `S6_7_figure8.py` |
| Figure 9 | `S7_external_validation/figures/Figure9_external_validation.tiff` | `S7_3_figure9.py` |
| Supplementary Figure S1 | `S5_nested_cv_tuning/figures/FigureS1_cv_repeats_convergence.tiff` | `S5_3_figureS1.py` |
| Table 1 | `S9_cohort_overview/results/Table1_cohort_characteristics.csv` | `S9_1_table1.py` |
| Table 2 | `S6_final_model_holdout_markers/results/Table2_internal_per_group.csv` | `S6_5_tables_2_3_4.py` |
| Table 3 | `S6_final_model_holdout_markers/results/Table3_panel_profile.csv` | `S6_5_tables_2_3_4.py` |
| Table 4 | `S6_final_model_holdout_markers/results/Table4_MaAsLin2.csv` | `S6_5_tables_2_3_4.py` |
| Table 5 | `S7_external_validation/results/Table5_external_per_group.csv` | `S7_2_tables_5_s7_s8.py` |
| Table S1 | `S2_algorithm_comparison/results/TableS1_classifier_comparison.csv` | `S2_1_algorithm_comparison.py` |
| Table S2 | `S3_genus_selection_maaslin2/results/TableS2_genus_selection.csv` | `S3_2_table_s2.py` |
| Table S3 | `S5_nested_cv_tuning/results/TableS3_panel_size.csv` | `S5_2_summary_table_s3.py` |
| Table S4 | `S8_diversity_phylum_statistics/results/TableS4_phylum_age_correlation.csv` | `S8_1_phylum_age_correlation.py` |
| Table S5 | `S8_diversity_phylum_statistics/results/TableS5_PERMANOVA_PERMDISP.csv` | `S8_4_table_s5.py` |
| Table S6 | `S8_diversity_phylum_statistics/results/TableS6_shannon_pairwise.csv` | `S8_3_shannon_pairwise.py` |
| Table S7 | `S7_external_validation/results/TableS7_external_balanced.csv` | `S7_2_tables_5_s7_s8.py` |
| Table S8 | `S7_external_validation/results/TableS8_panel_detection.csv` | `S7_2_tables_5_s7_s8.py` |

The csv tables keep the variable names of the code (for example `auc_macro`, `sensitivity_macro`, `10s`, `20s_40s`,
`50s_plus`); the submitted tables show the same values with reader-facing labels (Macro-AUC, Sensitivity (macro), 10s,
20s–40s, 50s+), and the supplementary tables are rounded to four decimals with P < 0.001 in scientific notation.
Figure 2 is a diagram; the numbers in it are those reported in the results of S5–S7.

## Software

**Upstream processing (S0)**: QIIME 2 2022.8 (framework 2022.8.3, Python 3.8.13) with DADA2 1.22.0 and q2-feature-classifier
(scikit-learn 0.24.1); Greengenes2 2022.10 naive Bayes classifiers (`2022.10.backbone.v4.nb.qza` for V4,
`2022.10.backbone.full-length.nb.qza` for V3–V4), confidence 0.7. DADA2 settings: discovery cohort trim 19/20 nt,
truncation 240/160 nt, pooled chimera removal; external cohort trim 19/20 nt, truncation 260/240 nt, consensus chimera
removal; other parameters at their defaults. Full commands: `S0_upstream_QIIME2/scripts/`.

**Analysis (S1–S9)**: Python 3.9.7 and R 4.3.2 (macOS).

| Python | Version | R | Version |
|---|---|---|---|
| numpy | 1.20.3 | Maaslin2 | 1.16.0 |
| pandas | 1.3.4 | vegan | 2.7.1 |
| scipy | 1.7.1 | permute | 0.9.8 |
| scikit-learn | 0.24.2 | | |
| xgboost | 2.1.4 | | |
| optuna | 4.9.0 | | |
| statsmodels | 0.12.2 | | |
| matplotlib | 3.4.3 | | |
| joblib | 1.5.3 | | |
| Pillow | 11.3.0 | | |

Figures use the Arial font. `requirements.txt` and `R_packages.txt` list the same versions.

## How to reproduce

1. **Upstream (optional)**: download the reads from the BioProjects above and run `S0_upstream_QIIME2/scripts/S0_1_qiime2_discovery_V4.sh`
   and `S0_2_qiime2_external_V3V4.sh` in a QIIME 2 2022.8 environment with the Greengenes2 2022.10 classifiers.
   The resulting tables are already provided in `S0_upstream_QIIME2/results/`.
2. **Analysis**: install the Python and R packages above, then from the package folder run

   ```
   bash run_all.sh
   ```

   This runs S1 to S9 in order (about 75 min) and overwrites the files in each `results/`, `figures/` and `model/`
   folder. Each step can also be run on its own (`cd S<n>_*/scripts` and run the scripts in numerical order); a step
   needs only the outputs of earlier steps.

   With the public data, the classification analyses can be rerun step by step: S2, S4, S5, S6 (all scripts except
   `S6_3_maaslin2_associations.py`), S7 and `S8_4_table_s5.py`. The other scripts (S1, S3, `S6_3`, S8 except `S8_4`, S9)
   and `run_all.sh` need the participant-level covariates; their outputs are included in the package. Researchers who
   obtain the covariates on request can place them in `metadata/metadata_discovery.csv` and
   `S1_data_processing/results/discovery_covariates.csv` and run `run_all.sh`.

## Reproducibility

Running `run_all.sh` on the provided inputs reproduces every number reported in the manuscript and every figure (TIFF
files identical to the submitted figures) and table. Differences that can appear in the last digits:

- **S1**: the genus tables can differ from the values used in the study by up to 4 × 10⁻¹⁵ percentage points (last digit
  of a 17-digit number), because the files used in the study had been saved and re-read once more. The zero pattern is
  identical.
- **S2**: with this input, the SVM AUCs can differ in the fourth decimal in some splits (largest difference 0.0026 for a
  per-group AUC); LibSVM probability calibration is sensitive to the last digits of the input. Predicted classes are
  identical, and the values reported in the manuscript (three decimals) are unchanged.
- **S6 and S7**: random-forest probabilities computed with several threads (`n_jobs=-1`) can differ by about 2 × 10⁻¹⁶,
  because the order in which the trees are summed varies between runs. The fitted trees are identical.
- **S3, S6, S8**: MaAsLin2 and Bray–Curtis-based statistics can differ in the last digits (≤ 4 × 10⁻¹²) for the same
  reason as S1; P values and all reported values are unchanged.

## License

The code is released under the MIT License (`LICENSE`). The data files in this repository (processed tables, results,
figures and the age-group metadata) are released under the Creative Commons Attribution 4.0 International License
(CC BY 4.0, https://creativecommons.org/licenses/by/4.0/). Please cite the article when using the code or data.

## Contact

Sun Jae Kwon, sjaes@accugenelab.com
