# mRNA–lncRNA classification benchmark

Snakemake workflows for the study *Uncertainty-aware benchmarking reveals ambiguous transcripts in mRNA-lncRNA classification*. Model training, inference and downstream data analyses have separate targets or entry points described below.

## Inputs

Run from the repository root (`revision-wt/`). The default experiment is `gencode.v47.common.cdhit.cv` (`config/config.yaml`). Supply GENCODE GRCh38 transcript FASTA files under `resources/`:

| Path | Use |
|---|---|
| `resources/gencode.v47.transcripts.fa`, `resources/gencode.v47.pc_transcripts.fa`, `resources/gencode.v47.lncRNA_transcripts.fa` | v47 source and coding/lncRNA sequences |
| `resources/gencode.v46.transcripts.fa`, `resources/gencode.v46.pc_transcripts.fa`, `resources/gencode.v46.lncRNA_transcripts.fa` | v46 comparison used when selecting transcripts common to versions |
| `resources/gencode.v47.pc_transcripts.fa.gz`, `resources/gencode.v47.lncRNA_transcripts.fa.gz` | reference sequences read by fold-table processing |
| `resources/Homo_sapiens.GRCh38.cds.all.fa.gz` | Ensembl GRCh38 CDS reference configured for training |

These references come from GENCODE and Ensembl and are **not supplied by a fresh source checkout**. Preserve the exact filenames and FASTA identifiers. The CV preparation rules create the fold FASTAs under `results/{expt}/datasets/fold{1..5}/`; do not place input files there. For an analysis-only run, supply the existing merged CV tables and the configured supplementary feature files at their paths below. Existing local results or symlinks can conceal missing upstream rules, so inspect the dry-run job list before trusting them.

Feature analysis reads TE tables from `te_pipeline/`, genomic transcript-span Non-B DNA features from `nonb-pipeline/`, and ScanFold/RG4 tables from `secondrna/`. Initialize the registered submodules with `git submodule update --init --recursive`, run their own documented workflows, and check the configured output paths.

## Requirements

Install Snakemake 9 and Conda/Mamba. The rules use `workflow/envs/`, external tool installations, and a versioned DIAMOND wrapper; fetching packages, wrappers, reference data, and some tool sources requires network access or a prepared local cache. lncRNA-BERT and RNAsamba training require GPUs. The default profile, `profiles/default/config.yaml`, selects SLURM with up to 30 jobs; the explicit local commands below override that executor. The Conda environment files and wrapper URL are recorded in the workflow, but a clean environment solve and full execution have not been verified.

## Run

From `revision-wt/`, with the references and required tool inputs in place:

```bash
snakemake --executor local --cores 8 --use-conda all_cv
```

`all_cv` is training only. Its output is `results/gencode.v47.common.cdhit.cv/training/cv_training.done`, a completion marker rather than a model. For stages beyond training, request a concrete file target:

```bash
# Prepare fold FASTAs
snakemake --executor local --cores 8 --use-conda results/gencode.v47.common.cdhit.cv/datasets/cv_split.done
# Train one fold or one tool across folds
snakemake --executor local --cores 8 --use-conda results/gencode.v47.common.cdhit.cv/training/fold1/training.done
snakemake --executor local --cores 8 --use-conda results/gencode.v47.common.cdhit.cv/training/cpat.done
# Test one fold or one tool across folds
snakemake --executor local --cores 8 --use-conda results/gencode.v47.common.cdhit.cv/testing/fold1/testing.done
snakemake --executor local --cores 8 --use-conda results/gencode.v47.common.cdhit.cv/testing/cpat.done
# Process and merge fold predictions
snakemake --executor local --cores 8 --use-conda results/gencode.v47.common.cdhit.cv/tables/gencode.v47.common.cdhit.cv_full_table.tsv
```

Valid training-tool target names are `cpat`, `lncfinder`, `plncpro`, `lncDC`, `lncDC_ss`, `mRNN`, `lncrnabert`, and `rnasamba`; testing also accepts `FEELnc`. `fold1` through `fold5` are configured. Fold processing requires the selected prediction files, including the CPAT training cutoff and PLncPRO feature file. The fold and tool `.done` files only mark completed collections of results.

Run the later entry points after their upstream tables and supplementary inputs exist:

```bash
snakemake -s workflow/rules/feature_analysis.smk --executor local --cores 8 --use-conda feature_analysis_all
snakemake -s workflow/rules/figures.smk --executor local --cores 8 --use-conda all_figures
```

`feature_analysis_all` requests statistical tests, embeddings, and configured SHAP modes. Smaller checks include `entropy_all`, `clustering_all`, `statistical_tests_all`, `embeddings_all`, and `shap_testing` (100 transcripts per fold). `all_figures` requests performance, entropy, UpSet, t-SNE, and clustered SHAP figures; the GENCODE comparison and timeline targets are separate.

For a DAG check without executing jobs, append `--dry-run` to a command. Use `--executor slurm --jobs 30` on a configured cluster. A dry run with already-present outputs can show nothing to do; this is not evidence that a fresh checkout can produce them.

## Configuration

The CV entry point loads `config/config.yaml`; feature analysis loads `config/feature_analysis_config.yaml` and `config/shap_config.yaml`; figures also loads `config/figures_config.yaml`. Edit a local copy or use Snakemake `--configfile path/to/override.yaml` with the relevant entry point. CLI overrides of nested settings should be checked with a dry run because each entry point also declares its config files.

### Path parameters

| Parameter | Default | Meaning |
|---|---|---|
| `experiments.*.{fasta,pc_fasta,lnc_fasta}` | `resources/gencode.v47.*.fa` | Read-only source FASTAs; rules create fold data under `results/{expt}/datasets/`. |
| `experiments.*.reference_cds` | `resources/Homo_sapiens.GRCh38.cds.all.fa.gz` | Read-only CDS reference. |
| `feature_analysis.*.{te_features_rna,te_features_dna,nbd_features}` | `te_pipeline/results/gencode.v47/features/all_transcripts_te_features.csv`; `te_pipeline/results/gencode.v47.bis_unspliced/features/all_transcripts_te_features.csv`; `nonb-pipeline/results/gencode.v47.transcripts/extended_analysis/features_nonb_features.csv` | Read-only supplementary tables. The Non-B path uses unspliced genomic transcript spans. |
| `feature_analysis.*.{scanfold_features,rg4_features}` | `secondrna/results/gencode.v47/scanfold_final_partners_stats.tsv`; `secondrna/results/gencode.v47/rg4detector/peak_counts.csv` | Read-only secondary-structure tables. Set a path to null to omit that source. |
| `feature_analysis.*.cluster_file`, `shap.*.cluster_file` | `results/{expt}/features/clustering/feature_clusters_at_distances.csv` | Produced by clustering; read by filtered SHAP. |
| Output paths | `results/{expt}/...` | Created by rules; there is no configurable result-root parameter. |

The CV source FASTAs are sequences; supplementary paths are feature tables matched by transcript ID. The Non-B `transcripts` input describes genomic transcript spans, including introns, while TE RNA and DNA paths represent distinct feature sources.

### Functioning parameters

| Parameter | Default | Meaning |
|---|---|---|
| `experiments.*.n_folds` | `5` | CV fold count. |
| `experiments.*.preprocessing.{common_with,redundancy}` | `gencode.v46`, `cdhit` | Compare with v46, then reduce redundancy; CD-HIT identity `0.9` is hard-coded in the rule. |
| `experiments.*.tools` | 11 prediction names in `config/config.yaml` | Prediction columns requested for fold processing. These names differ from aggregate target names. |
| `training.*.{train_split,mRNN_validation_split,lncrnabert_validation_split}` | `80`, `20`, `20` | Percent splits used by training rules. |
| `feature_analysis.*.clustering.{corr_method,distance_min,distance_max,distance_step}` | `spearman`, `0.05`, `1.50`, `0.05` | Correlation and distance grid for clustering. |
| `feature_analysis.*.cluster_threshold` | `null` | Use `optimal_threshold.txt` when unset. |
| `feature_analysis.*.statistical_analysis.{entropy_grouping_mode,low_threshold,high_threshold,high_threshold_sec}` | `class_separated`, `10`, `90`, `90` | Grouping mode and percentile cutoffs. |
| `feature_analysis.*.statistical_analysis.{fdr_method,fdr_alpha}` | `fdr_bh`, `0.01` | Multiple-testing correction and significance level. |
| `shap_run_experiments` | `[gencode.v47.common.cdhit.cv]` | Experiments included in SHAP convenience targets. |
| `shap.*.{n_folds,background_sample,force_rerun}` | `5`, `500`, `true` | Fold count, SHAP background rows, and whether cached RF/SHAP data are ignored for the active v47 block. |
| `shap.*.modes` | `full`, `clustered`, `testing` | Full features, correlation-filtered features, or filtered features capped at 100 transcripts per fold. |

For a one-off small SHAP run, use `shap_testing`; it is already configured with the 100-transcript cap. The `training.*.train_pc` and `train_lnc` paths remain in the config but are not used by the current CV orchestrator.

### Visualization customization

| Parameter | Default | Meaning |
|---|---|---|
| `figures.run_figures` | `[gencode.v47.common.cdhit.cv]` | Experiments selected for `all_figures`. |
| `shap.*.top_n_features` | `20` | SHAP analysis plotting count; publication figure rule separately fixes `--shap-top-n 20`. |
| `shap.*.cherry_picks_file` | `config/shap_cherry_picks.json` | Transcript selection for SHAP waterfall plots. |
| Figure paths and format | `results/{expt}/figures/**/*.pdf` | Fixed by rules; no output format or result-root config. |

### Environment

| Parameter | Default | Meaning |
|---|---|---|
| `profiles/default/config.yaml: executor` | `slurm` | Cluster executor; override with `--executor local` for local runs. |
| `profiles/default/config.yaml: jobs` | `30` | Maximum submitted cluster jobs. |
| `profiles/default/config.yaml: use-conda` | `true` | Create/use rule environments; local examples state this explicitly. |
| `--cores` | `8` in examples | Local CPU budget; choose according to available resources. |

## Outputs

For the configured experiment, read results under `results/gencode.v47.common.cdhit.cv/`:

| Path | Meaning |
|---|---|
| `datasets/fold{1..5}/{train_pc,train_lnc,test_all,test_pc,test_lnc}.fa` | Prepared fold sequences. `datasets/cv_split.done` is a marker. |
| `training/fold{1..5}/` | Trained models and cutoffs; `training/cv_training.done` is a marker. |
| `testing/fold{1..5}/` | Per-tool predictions; `testing/{fold}/testing.done` is a marker. |
| `testing/{fold}/tables/{fold}_full_table.tsv` | One fold's joined predictions and features. |
| `tables/{expt}_full_table.tsv`, `tables/{expt}_binary_class_table.tsv` | Merged transcript-level predictions and binary classes across folds. |
| `tables/{expt}_dropout_report.tsv` | Reports transcripts lost during fold-table merging; inspect before interpreting counts. |
| `features/supplementary_features.tsv`, `features/excluded_transcripts.tsv` | Joined feature matrix and exclusions for missing or invalid supplementary data. |
| `features/entropy/{expt}_uncertainty_analysis.tsv`, `features/entropy/{expt}_entropy_groups.tsv` | Uncertainty metrics and percentile groups. |
| `features/clustering/feature_clusters_at_distances.csv`, `features/statistical_analysis/*_mannwhitney.csv`, `features/statistical_analysis/*_chi2.csv` | Feature clusters and corrected statistical tests. `statistical_tests.flag` is a marker. |
| `features/shap_{mode}/shap_aggregated.csv` | Cross-fold SHAP values for the requested mode. |
| `figures/performance/performance_CV.pdf`, `figures/entropy/entropy_bald_scatter.pdf`, `figures/upset/main_upset.pdf`, `figures/embeddings/tsne_three_panels.pdf`, `figures/shap_clustered/shap_importance_mean_std.pdf` | Main figure files; `all_figures` does not include the separate GENCODE and timeline figures. |

## Reproducibility

Rule Conda files live under `workflow/envs/`; some external wrappers and tool downloads still require a prepared network/cache, and dependency resolution has not been verified in a clean checkout.
