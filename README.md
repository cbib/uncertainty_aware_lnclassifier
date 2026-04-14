# Uncertainty-aware benchmarking reveals ambiguous transcripts in mRNA-lncRNA classification (Snakemake pipeline)

This repository contains source code of the Snakemake pipeline associated with the publication:

`Uncertainty-aware benchmarking reveals ambiguous transcripts in mRNA-lncRNA classification`

It consists of a benchmark comparing lncRNA/mRNA classifiers across N-fold cross-validation
on GENCODE data, followed by feature analysis and figure production.

## Requirements

- Conda / Mamba
- Snakemake ≥ 9
- GPU node (optional — required only for lncRNA-BERT and RNAsamba training)

## Pipeline overview

The pipeline is split into three independent entry points that run sequentially:

```
Snakefile                      →  CV benchmark (dataset prep, training, testing)
workflow/rules/feature_analysis.smk  →  SHAP, clustering, entropy, embeddings, statistical tests
workflow/rules/figures.smk           →  Publication figures
```

---

## 1. CV benchmark

**Entry point**: `Snakefile`
**Config**: `config/config.yaml`
**Target rule**: `all` (delegates to `all_cv`)

Runs end-to-end N-fold cross-validation: dataset preparation, CD-HIT redundancy
reduction, fold splitting, per-tool training and inference, and aggregation of
per-fold tables.

```bash
snakemake --use-conda -j N          # full CV pipeline (default target: all)
snakemake --use-conda -j N all_cv   # same, explicit target
```

### Training targets

Train all tools across all folds (full training stage):

```bash
snakemake --use-conda -j N \
  results/gencode.v47.common.cdhit.cv/training/cv_training.done
```

Train all folds for a **single tool** (`{tool}` = one of the values in the table below):

```bash
snakemake --use-conda -j N \
  "results/gencode.v47.common.cdhit.cv/training/{tool}.done"
```

Train **all tools for one fold** (`{fold}` = `fold1`, `fold2`, …):

```bash
snakemake --use-conda -j N \
  "results/gencode.v47.common.cdhit.cv/training/{fold}/training.done"
```

### Testing targets

Test all folds for a **single tool**:

```bash
snakemake --use-conda -j N \
  "results/gencode.v47.common.cdhit.cv/testing/{tool}.done"
```

Test **all tools for one fold**:

```bash
snakemake --use-conda -j N \
  "results/gencode.v47.common.cdhit.cv/testing/{fold}/testing.done"
```

### Tool names

| Tool name (wildcard) | Description | Train | Test |
|----------------------|-------------|:-----:|:----:|
| `cpat` | CPAT (coding potential + hexamer) | ✓ | ✓ |
| `lncfinder` | LNCFinder (no-ss and ss modes) | ✓ | ✓ |
| `plncpro` | PLncPRO (DIAMOND + SVM) | ✓ | ✓ |
| `lncDC` | LncDC (no secondary structure) | ✓ | ✓ |
| `lncDC_ss` | LncDC (with secondary structure) | ✓ | ✓ |
| `mRNN` | mRNN 5-model ensemble | ✓ | ✓ |
| `lncrnabert` | lncRNA-BERT (GPU) | ✓ | ✓ |
| `rnasamba` | RNAsamba (GPU) | ✓ | ✓ |
| `FEELnc` | FEELnc CodPot | x | ✓ |

### Standalone tool targets

These tools cannot be directly retrained, so they are not part of the benchmark nor the CV fold pipeline. However, they through their own `.smk` rule files:

| Tool name (wildcard) | Description | Train | Test |
|----------------------|-------------|:-----:|:----:|
| `lncrnanet` | LncRNAnet | x | ✓ |
| `lncaDeep` | LncADeep | x | ✓ |


### Key outputs

All outputs are under `results/gencode.v47.common.cdhit.cv/`:

| Path | Content |
|------|---------|
| `datasets/cv_split.done` | Fold split sentinel |
| `training/{fold}/{tool}/` | Trained models per fold |
| `testing/{fold}/{tool}/` | Raw predictions per fold |
| `testing/{fold}/tables/{fold}_full_table.tsv` | Per-fold merged predictions |
| `training/cv_training.done` | All-training sentinel |
| `tables/{expt}_full_table.tsv` | Merged predictions across all folds |
| `tables/{expt}_binary_class_table.tsv` | Binary classification summary |

---

## 2. Feature analysis

**Entry point**: `workflow/rules/feature_analysis.smk`
**Configs**: `config/feature_analysis_config.yaml` + `config/shap_config.yaml`
**Target rule**: `feature_analysis_all`

Runs all post-CV feature analysis stages. Requires the merged CV tables from step 1.

```bash
snakemake -s workflow/rules/feature_analysis.smk \
          --use-conda -j N \
          feature_analysis_all
```

Stage targets (can be run individually):

```bash
# Entropy / uncertainty
snakemake -s workflow/rules/feature_analysis.smk --use-conda -j N entropy_all

# Feature clustering (Spearman + Ward linkage at multiple distance thresholds)
snakemake -s workflow/rules/feature_analysis.smk --use-conda -j N clustering_all

# Statistical tests: Mann-Whitney U + chi² per entropy group, FDR correction
snakemake -s workflow/rules/feature_analysis.smk --use-conda -j N statistical_tests_all

# Embeddings: UMAP, t-SNE, PCA
snakemake -s workflow/rules/feature_analysis.smk --use-conda -j N embeddings_all

# SHAP — smoke-test (100 transcripts/fold, fast end-to-end check)
snakemake -s workflow/rules/feature_analysis.smk --use-conda -j N shap_testing

# SHAP — full feature set (all transcripts, all folds)
snakemake -s workflow/rules/feature_analysis.smk --use-conda -j N shap_full_all

# SHAP — correlation-filtered feature set (cluster_0.25 de-duplicated)
snakemake -s workflow/rules/feature_analysis.smk --use-conda -j N shap_clustered_all

```

Key outputs under `results/{expt}/features/`:

| Path | Content |
|------|---------|
| `entropy/{expt}_uncertainty_analysis.tsv` | H_pred, BALD per transcript |
| `entropy/{expt}_entropy_groups.tsv` | Low / mid / high entropy group assignments |
| `clustering/feature_clusters_at_distances.csv` | Spearman-Ward cluster assignments |
| `statistical_analysis/all_statistical_tests.csv` | MWU + chi² results with FDR |
| `embeddings/embeddings_complete.flag` | UMAP / t-SNE / PCA cache sentinel |
| `shap_{mode}/shap_aggregated.csv` | Aggregated SHAP values across folds |

---

## 3. Publication figures

**Entry point**: `workflow/rules/figures.smk`
**Configs**: `config/feature_analysis_config.yaml` + `config/shap_config.yaml` + `config/figures_config.yaml`
**Target rule**: `all_figures`

Produces all main paper figures. Requires outputs from both steps 1 and 2.

```bash
snakemake -s workflow/rules/figures.smk \
          --use-conda -j N \
          all_figures
```

Individual figure targets:

```bash
snakemake -s workflow/rules/figures.smk --use-conda -j N performance_figures    # per-tool CV performance
snakemake -s workflow/rules/figures.smk --use-conda -j N entropy_main_figures   # entropy scatter + stratification
snakemake -s workflow/rules/figures.smk --use-conda -j N upset_figure           # tool-agreement UpSet plot
snakemake -s workflow/rules/figures.smk --use-conda -j N tsne_figure            # t-SNE embedding panels
snakemake -s workflow/rules/figures.smk --use-conda -j N shap_figures           # SHAP importance plots
snakemake -s workflow/rules/figures.smk --use-conda -j N gencode_versions_figure # GENCODE version timeline
snakemake -s workflow/rules/figures.smk --use-conda -j N generate_timeline      # tool publication timeline
```

Key outputs under `results/{dataset}/figures/`:

| Path | Content |
|------|---------|
| `performance/performance_CV.pdf` | Per-tool CV performance |
| `entropy/entropy_bald_scatter.pdf` | H_pred vs BALD scatter |
| `upset/main_upset.pdf` | Tool agreement UpSet plot |
| `embeddings/tsne_three_panels.pdf` | t-SNE coloured by class / entropy / tool |
| `shap_{mode}/shap_importance_mean_std.pdf` | SHAP feature importance |

---

## Dataset

Main dataset: `gencode.v47.common.cdhit.cv` (GENCODE v47, CD-HIT 90% redundancy reduction)

Resources are expected under `resources/`:

```
resources/gencode.v47.transcripts.fa
resources/gencode.v47.pc_transcripts.fa
resources/gencode.v47.lncRNA_transcripts.fa
```

## Supplementary feature inputs

The feature analysis pipeline can incorporate external feature tables (configured
in `config/feature_analysis_config.yaml`):

- **TE features**: produced by `te_pipeline/` (RepeatMasker-based TE annotation)
- **Non-B DNA features**: produced by `nonb-pipeline/`

These pipelines are managed as **git submodules**. To add them to a fresh clone:

```bash
# Add the submodules (first time only, after cloning the repo)
git submodule add https://github.com/cbib/rep_extraction_pipeline te_pipeline
git submodule add https://github.com/cbib/nbd_extraction_pipeline nonb-pipeline

# Commit the .gitmodules file and submodule pointers
git add .gitmodules te_pipeline nonb-pipeline
git commit -m "Add te_pipeline and nonb-pipeline as submodules"
```

If you clone a repository that already has these submodules registered, initialise
them with:

```bash
git clone --recurse-submodules <repo_url>
# — or, after a plain clone —
git submodule update --init --recursive
```

To update a submodule to its latest upstream commit:

```bash
git submodule update --remote te_pipeline
git submodule update --remote nonb-pipeline
git add te_pipeline nonb-pipeline
git commit -m "Update te_pipeline and nonb-pipeline submodules"
```

## Configs

| File | Used by |
|------|---------|
| `config/config.yaml` | CV benchmark (Snakefile) |
| `config/feature_analysis_config.yaml` | Feature analysis + figures |
| `config/shap_config.yaml` | SHAP pipeline + figures |
| `config/figures_config.yaml` | Figures only |
