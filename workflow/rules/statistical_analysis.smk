configfile: "config/feature_analysis_config.yaml"

# Include dependencies
include: "entropy.smk"
include: "clustering.smk"

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  statistical_analysis.smk — Statistical testing pipeline                 ║
# ║                                                                          ║
# ║  Performs univariate statistical tests on transcript features based on   ║
# ║  entropy groups:                                                         ║
# ║    - Mann-Whitney U tests for continuous variables                       ║
# ║    - Chi-squared tests for categorical variables                         ║
# ║    - Effect size calculations (VDA, Cramér's V, Odds Ratio)              ║
# ║    - FDR correction                                                      ║
# ║    - Feature clustering-based selection                                  ║
# ║    - Visualization of top features                                       ║
# ║                                                                          ║
# ║  Wildcards                                                               ║
# ║    {expt} — experiment name (key in config["feature_analysis"])          ║
# ║                                                                          ║
# ║  Usage examples                                                          ║
# ║    snakemake statistical_tests -j 1 --use-conda                          ║
# ║    snakemake results/.../features/statistical_analysis/statistical_tests.flag ║
# ╚══════════════════════════════════════════════════════════════════════════╝


# ── Config helpers ────────────────────────────────────────────────────────────

def _stat_cfg(wc):
    """Return the statistical_analysis block for the experiment."""
    return config["feature_analysis"][wc.expt].get("statistical_analysis", {})


def _opt_path(path):
    """Return [] for null/empty paths so Snakemake does not require missing files."""
    return [path] if path else []


def _opt_arg(flag, path):
    """Build optional CLI args for nullable config paths."""
    return f"{flag} '{path}'" if path else ""


# ── Rule: statistical_tests ───────────────────────────────────────────────────

rule statistical_tests:
    """
    Perform univariate statistical tests on features grouped by entropy levels.

    Performs Mann-Whitney U tests on continuous variables, Chi-squared tests on
    categorical variables, and calculates effect sizes (VDA, Cramér's V, odds ratio).
    Applies FDR correction and generates visualizations of top features.

        Outputs:
            - *_mannwhitney.csv / *_chi2.csv     : Pairwise result tables for entropy figures
    """
    input:
        full_table="results/{expt}/tables/{expt}_full_table.tsv",
        binary="results/{expt}/tables/{expt}_binary_class_table.tsv",
        entropy_tsv="results/{expt}/features/entropy/{expt}_uncertainty_analysis.tsv",
        groups_tsv="results/{expt}/features/entropy/{expt}_entropy_groups.tsv",
        cluster_file="results/{expt}/features/clustering/feature_clusters_at_distances.csv",
        optimal_threshold="results/{expt}/features/clustering/optimal_threshold.txt",
    output:
        high_entropy_pc_v_lnc_mwu="results/{expt}/features/statistical_analysis/high_entropy_pc_v_lnc_mannwhitney.csv",
        high_entropy_pc_v_lnc_chi2="results/{expt}/features/statistical_analysis/high_entropy_pc_v_lnc_chi2.csv",
        high_entropy_pc_v_lnc_freq="results/{expt}/features/statistical_analysis/high_entropy_pc_v_lnc_cat_freq.tsv",
        low_entropy_pc_v_lnc_mwu="results/{expt}/features/statistical_analysis/low_entropy_pc_v_lnc_mannwhitney.csv",
        low_entropy_pc_v_lnc_chi2="results/{expt}/features/statistical_analysis/low_entropy_pc_v_lnc_chi2.csv",
        low_entropy_pc_v_lnc_freq="results/{expt}/features/statistical_analysis/low_entropy_pc_v_lnc_cat_freq.tsv",
        low_vs_high_entropy_mwu="results/{expt}/features/statistical_analysis/low_vs_high_entropy_mannwhitney.csv",
        low_vs_high_entropy_chi2="results/{expt}/features/statistical_analysis/low_vs_high_entropy_chi2.csv",
        low_vs_high_entropy_freq="results/{expt}/features/statistical_analysis/low_vs_high_entropy_cat_freq.tsv",
        flag="results/{expt}/features/statistical_analysis/statistical_tests.flag",
    params:
        output_dir="results/{expt}/features/statistical_analysis",
        fdr_method=lambda wc: _stat_cfg(wc).get("fdr_method", "fdr_bh"),
        fdr_alpha=lambda wc: _stat_cfg(wc).get("fdr_alpha", 0.01),
        te_arg=lambda wc: _opt_arg("--te-features", config["feature_analysis"][wc.expt].get("te_features")),
        nbd_arg=lambda wc: _opt_arg("--nbd-features", config["feature_analysis"][wc.expt].get("nbd_features")),
        cluster_arg=lambda wc: _opt_arg("--cluster-file", config["feature_analysis"][wc.expt].get("cluster_file")),
        cluster_threshold=lambda wc, input: (
            config["feature_analysis"][wc.expt].get("cluster_threshold")
            or open(input.optimal_threshold).read().strip()
        ),
    log:
        "logs/{expt}/features/statistical_analysis/statistical_tests.log",
    threads: 4
    resources:
        mem_mb=16000,
        runtime=60,
    conda:
        "lnc-datasets"
    shell:
        """
        python -u workflow/scripts/statistical_tests.py \
            --dataset             {wildcards.expt}    \
            --output-dir          {params.output_dir} \
            --entropy-tsv         {input.entropy_tsv} \
            --groups-tsv          {input.groups_tsv} \
            {params.te_arg} \
            {params.nbd_arg} \
            {params.cluster_arg} \
            --cluster-threshold   {params.cluster_threshold} \
            --fdr-method          {params.fdr_method} \
            --fdr-alpha           {params.fdr_alpha} \
            --verbose 2>&1 | tee {log} && \
        touch {output.flag}
        """


# ── Convenience target ────────────────────────────────────────────────────

rule statistical_tests_all:
    """Run statistical testing for all configured experiments."""
    input:
        expand(
            "results/{expt}/features/statistical_analysis/statistical_tests.flag",
            expt=config.get("feature_analysis", {}).keys(),
        ),
