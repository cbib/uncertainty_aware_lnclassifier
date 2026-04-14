rule figures_all:
    """Produce all figures: entropy-analysis + main paper figures."""
    input:
        rules.entropy_analysis_figures.output,

# ── Entropy analysis figures (script: plot_entropy_analysis.py) ───────────────
rule entropy_analysis_figures:
    """Entropy stratification analysis figures (replaces 014_entropy_analysis.ipynb).

    Compares overall vs class-separated entropy thresholds and produces scatter,
    accuracy-vs-threshold, and top-10 univariate test figures for each strategy.

    Input (upstream pipeline outputs):
      - entropy TSV from compute_entropy_metrics
      - cluster CSV from compute_feature_clustering

    Output (in results/{dataset}/features/figures/entropy_analysis_tests/):
      - hist_H_pred_by_class.pdf
      - s1_scatter.pdf, s1_accuracy_vs_threshold.pdf
      - s1_low_vs_high.pdf, s1_low_pc_vs_lnc.pdf, s1_high_pc_vs_lnc.pdf
      - s2_scatter.pdf, s2_accuracy_vs_threshold.pdf
      - s2_low_vs_high_combined.pdf, s2_coding_low_vs_high.pdf,
        s2_lncrna_low_vs_high.pdf, s2_low_pc_vs_lnc.pdf, s2_high_pc_vs_lnc.pdf
    (Each PDF is accompanied by a matching PNG produced as a side-effect.)
    """
    input:
        entropy_tsv  = "results/{dataset}/features/entropy/{dataset}_uncertainty_analysis.tsv",
        cluster_file = "results/{dataset}/features/clustering/feature_clusters_at_distances.csv",
        te           = lambda wc: _fig_opt_path(config["feature_analysis"][wc.dataset].get("te_features")),
        nbd          = lambda wc: _fig_opt_path(config["feature_analysis"][wc.dataset].get("nbd_features")),
    output:
        hist        = "results/{dataset}/features/figures/entropy_analysis_tests/hist_H_pred_by_class.pdf",
        s1_scatter  = "results/{dataset}/features/figures/entropy_analysis_tests/s1_scatter.pdf",
        s1_acc      = "results/{dataset}/features/figures/entropy_analysis_tests/s1_accuracy_vs_threshold.pdf",
        s1_lh       = "results/{dataset}/features/figures/entropy_analysis_tests/s1_low_vs_high.pdf",
        s1_low_cls  = "results/{dataset}/features/figures/entropy_analysis_tests/s1_low_pc_vs_lnc.pdf",
        s1_high_cls = "results/{dataset}/features/figures/entropy_analysis_tests/s1_high_pc_vs_lnc.pdf",
        s2_scatter  = "results/{dataset}/features/figures/entropy_analysis_tests/s2_scatter.pdf",
        s2_acc      = "results/{dataset}/features/figures/entropy_analysis_tests/s2_accuracy_vs_threshold.pdf",
        s2_comb     = "results/{dataset}/features/figures/entropy_analysis_tests/s2_low_vs_high_combined.pdf",
        s2_cod      = "results/{dataset}/features/figures/entropy_analysis_tests/s2_coding_low_vs_high.pdf",
        s2_lnc      = "results/{dataset}/features/figures/entropy_analysis_tests/s2_lncrna_low_vs_high.pdf",
        s2_low_cls  = "results/{dataset}/features/figures/entropy_analysis_tests/s2_low_pc_vs_lnc.pdf",
        s2_high_cls = "results/{dataset}/features/figures/entropy_analysis_tests/s2_high_pc_vs_lnc.pdf",
    params:
        output_dir = "results/{dataset}/features/figures/entropy_analysis_tests",
        te_arg  = lambda wc: _fig_opt_arg("--te-features",  config["feature_analysis"][wc.dataset].get("te_features")),
        nbd_arg = lambda wc: _fig_opt_arg("--nbd-features", config["feature_analysis"][wc.dataset].get("nbd_features")),
    log:
        "logs/{dataset}/figures/entropy_analysis_figures.log",
    benchmark:
        "benchmarks/{dataset}/figures/entropy_analysis_figures.txt",
    resources:
        mem_mb  = 4000,
        runtime  = 120,
    conda:
        "lnc-datasets"
    shell:
        """
        python -u workflow/scripts/plot_entropy_analysis.py \
            --dataset      {wildcards.dataset} \
            --entropy-tsv  {input.entropy_tsv} \
            --cluster-file {input.cluster_file} \
            --output-dir   {params.output_dir} \
            {params.te_arg} \
            {params.nbd_arg} \
        2>&1 | tee {log}
        """
