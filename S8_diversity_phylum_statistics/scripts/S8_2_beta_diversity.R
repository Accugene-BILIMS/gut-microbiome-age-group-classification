# Beta diversity across the three age groups: PERMANOVA, PERMDISP and PCoA (Bray-Curtis, genus level).
#
# Input
#   S1_data_processing/results/discovery_genus_relab_cut0.01.csv   genus relative abundance (%) of 3,659 participants
#   S1_data_processing/results/discovery_covariates.csv            Sex, Diet, Drug, Disease_pr
# Output (S8_diversity_phylum_statistics/results/)
#   permanova.csv              PERMANOVA, age group only
#   permanova_covariates.csv   PERMANOVA with Sex, Diet, Drug, Disease_pr and age group, marginal tests
#   permdisp.csv               mean distance to the age-group centroid and the permutation test of dispersion
#   betadisper_distances.csv   distance to the age-group centroid per participant
#   pcoa_scores.csv            first two principal coordinates per participant and % variance explained
#   balanced_subsamples.csv    PERMANOVA and PERMDISP in 30 class-balanced subsamples (160 per age group)
# Process
#   Bray-Curtis dissimilarity of the 783 genera (vegan). PERMANOVA with adonis2 (999 permutations); covariate model
#   with by = "margin", so the age-group R2 is the variance explained after the covariates. PERMDISP with betadisper
#   and permutest (999 permutations). PCoA with cmdscale. Balanced subsamples: 160 participants per age group drawn
#   without replacement. set.seed(42).
#
# Usage
#   Rscript S8_2_beta_diversity.R

suppressPackageStartupMessages(library(vegan))
set.seed(42)

args <- commandArgs(trailingOnly = FALSE)
script <- normalizePath(sub("^--file=", "", args[grep("^--file=", args)]))
pkg <- dirname(dirname(dirname(script)))
res_dir <- file.path(pkg, "S8_diversity_phylum_statistics", "results")
dir.create(res_dir, showWarnings = FALSE, recursive = TRUE)

genus <- read.csv(file.path(pkg, "S1_data_processing", "results", "discovery_genus_relab_cut0.01.csv"),
                  row.names = 1, check.names = FALSE)
meta <- read.csv(file.path(pkg, "S1_data_processing", "results", "discovery_covariates.csv"),
                 row.names = 1, stringsAsFactors = TRUE)
meta <- meta[rownames(genus), ]
X <- as.matrix(genus[, startsWith(colnames(genus), "d__")])
meta$Age_group3 <- factor(meta$Age_group3, levels = c("10s", "20s_40s", "50s_plus"))
cat("participants", nrow(X), "genera", ncol(X), "\n")

d <- vegdist(X, method = "bray")

tidy_adonis <- function(fit, label) {
  out <- as.data.frame(fit)
  out$term <- rownames(out)
  out$model <- label
  out[, c("model", "term", "Df", "SumOfSqs", "R2", "F", "Pr(>F)")]
}

# PERMANOVA, age group only
a <- adonis2(d ~ Age_group3, data = meta, permutations = 999, parallel = 4)
write.csv(tidy_adonis(a, "age_group_only"), file.path(res_dir, "permanova.csv"), row.names = FALSE)
print(a)

# PERMDISP
bd <- betadisper(d, meta$Age_group3)
pt <- permutest(bd, permutations = 999)
disp <- data.frame(group = names(tapply(bd$distances, meta$Age_group3, mean)),
                   n = as.vector(table(meta$Age_group3)),
                   mean_distance_to_centroid = as.vector(tapply(bd$distances, meta$Age_group3, mean)),
                   sd = as.vector(tapply(bd$distances, meta$Age_group3, sd)))
disp$permutest_F <- pt$tab["Groups", "F"]
disp$permutest_p <- pt$tab["Groups", "Pr(>F)"]
write.csv(disp, file.path(res_dir, "permdisp.csv"), row.names = FALSE)
write.csv(data.frame(Sample_name = rownames(X), Age_group3 = meta$Age_group3, distance_to_centroid = bd$distances),
          file.path(res_dir, "betadisper_distances.csv"), row.names = FALSE)
print(disp)

# PCoA
pc <- cmdscale(d, k = 2, eig = TRUE)
eig_pos <- pc$eig[pc$eig > 0]
pct <- round(100 * pc$eig[1:2] / sum(eig_pos), 2)
write.csv(data.frame(Sample_name = rownames(X), PCo1 = pc$points[, 1], PCo2 = pc$points[, 2],
                     Age_group3 = meta$Age_group3, pct_PCo1 = pct[1], pct_PCo2 = pct[2]),
          file.path(res_dir, "pcoa_scores.csv"), row.names = FALSE)

# PERMANOVA with covariates (marginal tests)
ac <- adonis2(d ~ Sex + Diet + Drug + Disease_pr + Age_group3, data = meta, permutations = 999,
              by = "margin", parallel = 4)
write.csv(tidy_adonis(ac, "covariate_adjusted_marginal"), file.path(res_dir, "permanova_covariates.csv"), row.names = FALSE)
print(ac)

# Class-balanced subsamples
idx_by <- split(seq_len(nrow(meta)), meta$Age_group3)
sub <- do.call(rbind, lapply(1:30, function(i) {
  idx <- unlist(lapply(idx_by, function(ix) if (length(ix) > 160) sample(ix, 160) else ix))
  ds <- vegdist(X[idx, ], method = "bray")
  fit <- adonis2(ds ~ Age_group3, data = meta[idx, , drop = FALSE], permutations = 999)
  bds <- betadisper(ds, meta$Age_group3[idx])
  pts <- permutest(bds, permutations = 999)
  data.frame(draw = i, permanova_R2 = fit$R2[1], permanova_p = fit$`Pr(>F)`[1],
             permdisp_F = pts$tab["Groups", "F"], permdisp_p = pts$tab["Groups", "Pr(>F)"],
             t(tapply(bds$distances, meta$Age_group3[idx], mean)))
}))
write.csv(sub, file.path(res_dir, "balanced_subsamples.csv"), row.names = FALSE)
print(summary(sub[, c("permanova_R2", "permanova_p", "permdisp_F", "permdisp_p")]))
