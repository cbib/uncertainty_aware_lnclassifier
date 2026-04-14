# ── Standalone config (no-op when included from feature_analysis.smk) ────────
configfile: "config/feature_analysis_config.yaml"
configfile: "config/shap_config.yaml"
configfile: "config/figures_config.yaml"

# ── Dataset constants ─────────────────────────────────────────────────────────
_FIG_NFOLDS = 5


# ── Config helpers ────────────────────────────────────────────────────────────
def _fig_opt_path(path):
    """Return [] for null/empty paths so Snakemake treats them as optional."""
    return [path] if path else []


def _fig_opt_arg(flag, path):
    """Build optional CLI arg string for nullable config paths."""
    return f"{flag} '{path}'" if path else ""


def _fig_run_expt(wc):
    """Helper to get which dataset to create figures for"""
    return config["figures"].get("run_figures", "gencode.v47.common.cdhit.cv")


# ── Performance figures (script: plot_performance_figures.py) ─────────────────
rule performance_figures:
    """CV performance and classification consensus figures.

    Input (upstream pipeline outputs):
      - per-fold binary class tables from CV testing

    Output (in results/{expt}/figures/):
      - performance_CV.pdf
      - correct_classification.pdf
    """
    input:
        fold_tables = expand(
            "results/{expt}/testing/{fold}/tables/{fold}_binary_class_table.tsv",
            expt="{expt}",
            fold=[f"fold{f}" for f in range(1, _FIG_NFOLDS + 1)],
        ),
    output:
        performance_cv = "results/{expt}/figures/performance/performance_CV.pdf",
        correct_cls    = "results/{expt}/figures/performance/correct_classification.pdf",
    params:
        output_dir = "results/{expt}/figures/performance",
    log:
        "logs/{expt}/figures/performance_figures.log",
    benchmark:
        "benchmarks/{expt}/figures/performance_figures.txt",
    resources:
        mem_mb  = 4000,
        runtime = 30,
    conda:
        "lnc-datasets"
    shell:
        """
        python -u workflow/scripts/plot_performance_figures.py \
            --dataset      {wildcards.expt} \
            --results-dir  results \
            --output-dir   {params.output_dir} \
            --n-folds      {_FIG_NFOLDS} \
        2>&1 | tee {log}
        """


# ── Entropy input helpers ─────────────────────────────────────────────────────

def _entropy_main_inputs(wc):
    """Base inputs required for all entropy figure modes."""
    sa = f"results/{wc.expt}/features/statistical_analysis"
    return {
        "entropy_tsv":        f"results/{wc.expt}/features/entropy/{wc.expt}_uncertainty_analysis.tsv",
        "groups_tsv":         f"results/{wc.expt}/features/entropy/{wc.expt}_entropy_groups.tsv",
        "cluster_file":       f"results/{wc.expt}/features/clustering/feature_clusters_at_distances.csv",
        "optimal_threshold":  f"results/{wc.expt}/features/clustering/optimal_threshold.txt",
        "high_entropy_mwu":   f"{sa}/high_entropy_pc_v_lnc_mannwhitney.csv",
        "high_entropy_chi2":  f"{sa}/high_entropy_pc_v_lnc_chi2.csv",
        "high_entropy_freq":  f"{sa}/high_entropy_pc_v_lnc_cat_freq.tsv",
        "low_entropy_mwu":    f"{sa}/low_entropy_pc_v_lnc_mannwhitney.csv",
        "low_entropy_chi2":   f"{sa}/low_entropy_pc_v_lnc_chi2.csv",
        "low_entropy_freq":   f"{sa}/low_entropy_pc_v_lnc_cat_freq.tsv",
        "low_vs_high_mwu":    f"{sa}/low_vs_high_entropy_mannwhitney.csv",
        "low_vs_high_chi2":   f"{sa}/low_vs_high_entropy_chi2.csv",
        "low_vs_high_freq":   f"{sa}/low_vs_high_entropy_cat_freq.tsv",
    }


def _entropy_supp_inputs(wc):
    """Extends _entropy_main_inputs with within-class low-vs-high stat files."""
    sa = f"results/{wc.expt}/features/statistical_analysis"
    inputs = _entropy_main_inputs(wc)
    inputs.update({
        "supp_cod_mwu":  f"{sa}/supp_coding_low_vs_high_entropy_mannwhitney.csv",
        "supp_cod_chi2": f"{sa}/supp_coding_low_vs_high_entropy_chi2.csv",
        "supp_cod_freq": f"{sa}/supp_coding_low_vs_high_entropy_cat_freq.tsv",
        "supp_lnc_mwu":  f"{sa}/supp_lncrna_low_vs_high_entropy_mannwhitney.csv",
        "supp_lnc_chi2": f"{sa}/supp_lncrna_low_vs_high_entropy_chi2.csv",
        "supp_lnc_freq": f"{sa}/supp_lncrna_low_vs_high_entropy_cat_freq.tsv",
    })
    return inputs


def _entropy_all_inputs(wc):
    """Return inputs for the configured stratification mode."""
    mode = (config["feature_analysis"][wc.expt]
                  ["statistical_analysis"]
                  .get("entropy_grouping_mode", "overall"))
    return _entropy_supp_inputs(wc) if mode == "class_separated" else _entropy_main_inputs(wc)


def _entropy_supp_flag(wc):
    """Return --with-supplementary flag string when using class-separated mode."""
    mode = (config["feature_analysis"][wc.expt]
                  ["statistical_analysis"]
                  .get("entropy_grouping_mode", "overall"))
    return "--with-supplementary" if mode == "class_separated" else ""


# ── Entropy scatter + statistical test figures (script: plot_entropy_main_figures.py) ──
rule entropy_main_figures:
    """Entropy scatter and statistical test figures.

    Inputs adapt to the configured stratification mode (overall / class_separated):
      - overall:          base statistical result CSVs only
      - class_separated:  base CSVs + within-class supp_coding_* / supp_lncrna_* CSVs

    Outputs supp_cod / supp_lnc are real PDFs when mode is class_separated;
    empty placeholder files otherwise (Snakemake output-existence guarantee).

    Output (in results/{expt}/figures/entropy/):
      - entropy_bald_scatter.pdf
      - high_entropy_pc_v_lnc_statistical_tests_results.pdf
      - low_entropy_pc_v_lnc_statistical_tests_results.pdf
      - low_vs_high_entropy_statistical_tests_results.pdf
      - supp_coding_low_vs_high_entropy_statistical_tests_results.pdf  [real / placeholder]
      - supp_lncrna_low_vs_high_entropy_statistical_tests_results.pdf  [real / placeholder]
    """
    input:
        unpack(_entropy_all_inputs),
    output:
        entropy_scatter = "results/{expt}/figures/entropy/entropy_bald_scatter.pdf",
        stat_he         = "results/{expt}/figures/entropy/high_entropy_pc_v_lnc_statistical_tests_results.pdf",
        stat_le         = "results/{expt}/figures/entropy/low_entropy_pc_v_lnc_statistical_tests_results.pdf",
        stat_lvh        = "results/{expt}/figures/entropy/low_vs_high_entropy_statistical_tests_results.pdf",
        supp_cod        = "results/{expt}/figures/entropy/supp_coding_low_vs_high_entropy_statistical_tests_results.pdf",
        supp_lnc        = "results/{expt}/figures/entropy/supp_lncrna_low_vs_high_entropy_statistical_tests_results.pdf",
    params:
        output_dir = "results/{expt}/figures/entropy",
        cluster_threshold = lambda wc, input: (
            config["feature_analysis"][wc.expt].get("cluster_threshold")
            or open(input.optimal_threshold).read().strip()
        ),
        with_supp_flag = _entropy_supp_flag,
    log:
        "logs/{expt}/figures/entropy_main_figures.log",
    benchmark:
        "benchmarks/{expt}/figures/entropy_main_figures.txt",
    resources:
        mem_mb  = 4000,
        runtime = 60,
    conda:
        "lnc-datasets"
    shell:
        """
        python -u workflow/scripts/plot_entropy_main_figures.py \
            --dataset            {wildcards.expt} \
            --output-dir         {params.output_dir} \
            --results-dir        results \
            --cluster-threshold  {params.cluster_threshold} \
            {params.with_supp_flag} \
        2>&1 | tee {log}

        # Produce empty supps if unavailable (dirty fix)
        [ -f {output.supp_cod} ] || touch {output.supp_cod}
        [ -f {output.supp_lnc} ] || touch {output.supp_lnc}
        """


# ── UpSet figure (script: plot_upset_figure.py) ────────────────────────────────
rule upset_figure:
    """UpSet plot of tool label combinations.

    Output (in results/{expt}/figures/):
      - main_upset.pdf
    """
    output:
        upset = "results/{expt}/figures/upset/main_upset.pdf",
    params:
        output_dir = subpath(output[0], parent=True),
    log:
        "logs/{expt}/figures/upset_figure.log",
    benchmark:
        "benchmarks/{expt}/figures/upset_figure.txt",
    resources:
        mem_mb  = 4000,
        runtime = 20,
    conda:
        "lnc-datasets"
    shell:
        """
        python -u workflow/scripts/plot_upset_figure.py \
            --dataset    {wildcards.expt} \
            --output-dir {params.output_dir} \
        2>&1 | tee {log}
        """


# ── t-SNE figure (script: plot_tsne_figure.py) ────────────────────────────────
rule tsne_figure:
    """Three-panel t-SNE figure (coding class, H_pred, entropy group).

    Input (upstream pipeline outputs):
      - entropy TSV from compute_entropy_metrics
      - embeddings_complete flag from compute_embeddings

    Output (in results/{expt}/figures/):
      - tsne_three_panels.pdf
    """
    input:
        entropy_tsv = "results/{expt}/features/entropy/{expt}_uncertainty_analysis.tsv",
        groups_tsv= "results/{expt}/features/entropy/{expt}_entropy_groups.tsv",
        embeddings  = "results/{expt}/features/embeddings/embeddings_complete.flag",
    output:
        tsne = "results/{expt}/figures/embeddings/tsne_three_panels.pdf",
    params:
        output_dir = "results/{expt}/figures/embeddings",
    log:
        "logs/{expt}/figures/tsne_figure.log",
    benchmark:
        "benchmarks/{expt}/figures/tsne_figure.txt",
    resources:
        mem_mb  = 4000,
        runtime = 60,
    conda:
        "lnc-datasets"
    shell:
        """
        python -u workflow/scripts/plot_tsne_figure.py \
            --dataset      {wildcards.expt} \
            --results-dir  results \
            --output-dir   {params.output_dir} \
            --entropy-tsv  {input.entropy_tsv} \
            --groups-tsv   {input.groups_tsv} \
        2>&1 | tee {log}
        """


# ── SHAP figures (script: plot_shap_figures.py) ────────────────────────────────
rule shap_figures:
    """All SHAP publication figures.

    Assumptions:
      - SHAP is run in 'clustered' mode (shap_clustered) with TOP_N=20 features.
      - N_STEPS = min(30, len(shap_agg)) evaluates to 30 for the clustered
        feature set (which always contains >30 features after filtering).
      - Cherry-picked waterfall transcripts are EGF, MALAT1, SWI5, MEG3.

    Input (upstream pipeline outputs):
      - shap_clustered aggregation outputs

    Output (in results/{expt}/figures/shap_{mode}):
      shap_importance_mean_std.pdf, shap_cumulative_importance_top30.pdf,
      shap_fold_heatmap.pdf, shap_beeswarm_all_transcripts.pdf,
      shap_prediction_probability_distribution.pdf, and four waterfall PDFs.
    """
    input:
        shap_agg   = "results/{expt}/features/shap_{mode}/shap_aggregated.csv",
        shap_pfold = "results/{expt}/features/shap_{mode}/shap_per_fold_mean_abs.csv",
        all_preds  = "results/{expt}/features/shap_{mode}/all_predictions.csv",
        perf_summ  = "results/{expt}/features/shap_{mode}/performance_summary.csv",
    output:
        shap_imp       = "results/{expt}/figures/shap_{mode}/shap_importance_mean_std.pdf",
        shap_cum       = "results/{expt}/figures/shap_{mode}/shap_cumulative_importance_top30.pdf",
        shap_heatmap   = "results/{expt}/figures/shap_{mode}/shap_fold_heatmap.pdf",
        shap_beeswarm  = "results/{expt}/figures/shap_{mode}/shap_beeswarm_all_transcripts.pdf",
        shap_prob_dist = "results/{expt}/figures/shap_{mode}/shap_prediction_probability_distribution.pdf",
        wf_le_cod      = "results/{expt}/figures/shap_{mode}/shap_waterfall_low_entropy_coding_EGF.pdf",
        wf_le_lnc      = "results/{expt}/figures/shap_{mode}/shap_waterfall_low_entropy_lncrna_MALAT1.pdf",
        wf_he_cod      = "results/{expt}/figures/shap_{mode}/shap_waterfall_high_entropy_coding_SWI5.pdf",
        wf_he_lnc      = "results/{expt}/figures/shap_{mode}/shap_waterfall_high_entropy_lncrna_MEG3.pdf",
    params:
        output_dir = "results/{expt}/figures/shap_{mode}",
        shap_dir   = "results/{expt}/features/shap_{mode}",
    log:
        "logs/{expt}/figures/shap_{mode}_figures.log",
    benchmark:
        "benchmarks/{expt}/figures/shap_{mode}_figures.txt",
    resources:
        mem_mb  = 4000,
        runtime = 60,
    conda:
        "lnc-datasets"
    shell:
        """
        python -u workflow/scripts/plot_shap_figures.py \
            --output-dir {params.output_dir} \
            --shap-dir   {params.shap_dir} \
            --n-folds    {_FIG_NFOLDS} \
            --shap-top-n 20 \
        2>&1 | tee {log}
        """


# ── Aggregate targets ──────────────────────────────────────────────────────────
rule all_figures:
    """All main paper figures (performance, entropy, upset, t-SNE, SHAP).

    Supplementary entropy figures (supp_cod / supp_lnc) are included automatically
    when entropy_grouping_mode == 'class_separated' in feature_analysis_config.yaml.
    """
    input:
        expand(rules.performance_figures.output, expt=_fig_run_expt),
        expand(rules.entropy_main_figures.output, expt=_fig_run_expt),
        expand(rules.upset_figure.output, expt=_fig_run_expt),
        expand(rules.tsne_figure.output, expt=_fig_run_expt),
        expand(rules.shap_figures.output, expt=_fig_run_expt, mode="clustered"),

rule cleanup_figures:
    """Remove all generated figure files."""
    shell:
        "rm -rf results/gencode.v47.common.cdhit.cv/figures"


rule gencode_versions_figure:
    input:
        tsv="results/gencode_comparison/v46_vs_v47/gencode.v47.comparison.tsv"
    output:
        figure="results/figures/001_gencode_versions.png"
    conda:
        "lnc-datasets"
    notebook:
        "workflow/notebooks/figures/001_gencode_versions.ipynb"


# Timeline visualization
rule generate_timeline:
    input:
        tools="config/timeline_tools.csv",
        gencode="config/timeline_gencode.csv"
    output:
        pdf="results/figures/timeline.pdf",
        png="results/figures/timeline.png",
        html="results/figures/timeline.html"
    conda:
        "lnc-datasets"
    log:
        "logs/generate_timeline.log"
    benchmark:
        "benchmarks/generate_timeline.txt"
    shell:
        """
        python workflow/scripts/generate_timeline.py \
            {input.tools} \
            {input.gencode} \
            results/figures/timeline \
        > {log} 2>&1
        """
