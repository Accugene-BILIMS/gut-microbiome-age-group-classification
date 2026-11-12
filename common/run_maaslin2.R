# One MaAsLin2 run with the covariates of the study (called from the Python analysis scripts).
#
# Input
#   features.tsv   participants x genera, relative abundance (%) (values <= 0.01% already set to 0)
#   metadata.tsv   participants x (Age_group3, Sex, Diet, Drug, Disease_pr)
#   age_reference  10s or 20s_40s (two runs give all three pairwise age-group contrasts)
# Output
#   <output_dir>/all_results.tsv   MaAsLin2 results for every genus x coefficient
# Process
#   normalization NONE (input is already relative abundance of all reads), log transform, linear model,
#   minimum prevalence 10%, Benjamini-Hochberg correction over all coefficients, standardized continuous
#   variables; fixed effects Age_group3, Sex, Diet, Drug, Disease_pr; reference levels: Diet 3_Balanced,
#   Drug 1_No_Drug, Age_group3 = age_reference.
#
# Usage
#   Rscript run_maaslin2.R <features.tsv> <metadata.tsv> <output_dir> <age_reference>

suppressPackageStartupMessages(library(Maaslin2))

args <- commandArgs(trailingOnly = TRUE)
features <- read.table(args[1], header = TRUE, sep = "\t", row.names = 1, check.names = FALSE)
metadata <- read.table(args[2], header = TRUE, sep = "\t", row.names = 1, check.names = FALSE,
                       stringsAsFactors = FALSE)
age_ref <- args[4]

invisible(capture.output(
  fit <- Maaslin2(
    input_data = features,
    input_metadata = metadata,
    output = args[3],
    min_abundance = 0,
    min_prevalence = 0.1,
    normalization = "NONE",
    transform = "LOG",
    analysis_method = "LM",
    correction = "BH",
    standardize = TRUE,
    fixed_effects = c("Age_group3", "Sex", "Diet", "Drug", "Disease_pr"),
    reference = c(paste0("Age_group3,", age_ref), "Diet,3_Balanced", "Drug,1_No_Drug"),
    plot_heatmap = FALSE,
    plot_scatter = FALSE,
    cores = 1
  )
))
