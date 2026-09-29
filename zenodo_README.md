# Zenodo data README

This dataset contains data files and experimental results associated with the manuscript "Uncertainty-aware benchmarking reveals ambiguous transcripts in mRNA–lncRNA classification".


**Related Code Repository**: [https://github.com/cbib/uncertainty_aware_lnclassifier](https://github.com/cbib/uncertainty_aware_lnclassifier)


**Data Archive DOI**: `10.5281/zenodo.19551649`


**Manuscript**: [Citation when available]


## File organization


The attached `data.zip` contains the following sections and files:


**Features**



- `gencode.v47.common.cdhit.cv_full_table.tsv` — Complete feature matrix: all raw scores and computed features from every classifier, per transcript

- `gencode.v47.common.cdhit.cv_binary_class_table.tsv` — Binary classification matrix: classification assigned by each of the tools, per transcript. (lncRNA = False/protein-coding = True).

- `gencode.v47.common.cdhit.cv_probs_table.tsv` — Coding-probability scores (0–1) from each classifier per transcript, used as input for ensemble/entropy analysis


**Entropy**



- `gencode.v47.common.cdhit.cv_entropy_groups.tsv` — Low / middle / high entropy group assignment for each transcript (two-column: seq_ID, entropy_group)

- `gencode.v47.common.cdhit.cv_uncertainty_analysis.tsv` — Per-transcript entropy metrics (H_pred, I_bald) plus per-tool entropies, true class, and biotype


**Stats**



- `high_entropy_pc_v_lnc_cat_freq.tsv` — Categorical feature frequencies (group1 vs group2) for high-entropy coding vs lncRNA transcripts

- `high_entropy_pc_v_lnc_chi2.csv` — Chi-squared + Cramér's V + odds-ratio results for categorical features, high-entropy coding vs lncRNA

- `high_entropy_pc_v_lnc_mannwhitney.csv` — Mann-Whitney U + VDA results for continuous features, high-entropy coding vs lncRNA

- `low_entropy_pc_v_lnc_cat_freq.tsv` — Categorical feature frequencies for low-entropy coding vs lncRNA transcripts

- `low_entropy_pc_v_lnc_chi2.csv` — Chi-squared + Cramér's V + odds-ratio results for categorical features, low-entropy coding vs lncRNA

- `low_entropy_pc_v_lnc_mannwhitney.csv` — Mann-Whitney U + VDA results for continuous features, low-entropy coding vs lncRNA

- `low_vs_high_entropy_cat_freq.tsv` — Categorical feature frequencies comparing all low-entropy vs high-entropy transcripts

- `low_vs_high_entropy_chi2.csv` — Chi-squared + Cramér's V + odds-ratio results for low vs high entropy (all transcripts)

- `low_vs_high_entropy_mannwhitney.csv` — Mann-Whitney U + VDA results for low vs high entropy (all transcripts)

- `supp_coding_low_vs_high_entropy_cat_freq.tsv` — Categorical feature frequencies for low vs high entropy within coding transcripts only

- `supp_coding_low_vs_high_entropy_chi2.csv` — Chi-squared results for low vs high entropy, coding-only

- `supp_coding_low_vs_high_entropy_mannwhitney.csv` — Mann-Whitney U + VDA results for low vs high entropy, coding-only

- `supp_lncrna_low_vs_high_entropy_cat_freq.tsv` — Categorical feature frequencies for low vs high entropy within lncRNA transcripts only

- `supp_lncrna_low_vs_high_entropy_chi2.csv` — Chi-squared results for low vs high entropy, lncRNA-only

- `supp_lncrna_low_vs_high_entropy_mannwhitney.csv` — Mann-Whitney U + VDA results for low vs high entropy, lncRNA-only

- `top_features_from_clusters.csv` — Top discriminative features selected after FDR correction and cluster-based deduplication, with VDA and test statistics


**Clustering**



- `feature_correlation_matrix.csv` — Full pairwise Spearman correlation matrix across all features

- `feature_clusters_at_distances.csv` — Cluster ID assigned to each feature at every correlation-distance threshold (0.05–1.45 step 0.05)

- `silhouette_scores.csv` — Silhouette score and number of clusters at each distance threshold

- `optimal_threshold.txt` — Distance threshold that maximises silhouette score

- `feature_correlation_dendrogram.pdf` — Hierarchical clustering dendrogram of pairwise Spearman feature correlations


**SHAP**



- `performance_summary.csv` — Precision / recall / F1 / accuracy summary across CV folds for the random-forest model

- `all_predictions.csv` — Per-transcript, per-fold random-forest classification predictions (true label, predicted label, class probabilities)

- `shap_aggregated.csv` — Mean ± SD absolute SHAP value per feature, aggregated across all transcripts and folds

- `shap_all_transcripts.csv` — Raw SHAP values for every transcript × feature × fold

- `shap_per_fold_mean_abs.csv` — Mean absolute SHAP value per feature per CV fold


**REP features (RNA)**



- `all_transcripts.out` — Raw RepeatMasker output for all GENCODE v47 transcripts (Smith-Waterman scores, divergence, repeat family/class)

- `all_transcripts_te_features.csv` — Repetitive element summary features per transcript derived from RepeatMasker (counts, scores, coverage by TE class etc.)


**REP features (DNA)**

- `all_transcripts_te_features.csv` — RepeatMasker summary features from unspliced genomic sequences. The `REP features (RNA)` section contains the separate spliced RNA results.

**NBD features**



- `features_nonb_features.csv` — Non-B DNA structural features per transcript (A-phased repeats, G-quadruplexes, Z-DNA, etc.)


**ScanFold**

- `scanfold_final_partners_stats.tsv` — RNA secondary-structure summary per transcript.

**rG4**

- `peak_counts.csv` — Predicted RNA G-quadruplex peak counts per transcript.

**Configuration**

- `config.yaml`, `feature_analysis_config.yaml`, `shap_config.yaml`, `figures_config.yaml`, `shap_cherry_picks.json` — Analysis, feature, SHAP and figure settings.

**Snakevision pipeline graphs**

- `nonb-pipeline-dag.dot`, `nonb-pipeline-dag.svg`, `revision-wt-dag.dot`, `revision-wt-dag.svg`, `rulegraph-dashboard.svg`, `secondrna-dag.dot`, `secondrna-dag.svg`, `te-pipeline-dag.dot`, `te-pipeline-dag.svg` — Pipeline diagrams in DOT and SVG formats.

## Archive layout and restoration

`manifest.json` maps archive members to pipeline-relative paths. Sections keep files with the same basename separate, including RNA and DNA repeat features and full and clustered SHAP results. To restore the archive into a pipeline checkout, use the companion script:

```bash
python unzip_zenodo.py --zip /path/to/data.zip --paper-dir /path/to/checkout --dry-run
python unzip_zenodo.py --zip /path/to/data.zip --paper-dir /path/to/checkout
```

The second command writes files and replaces existing files at mapped paths.
