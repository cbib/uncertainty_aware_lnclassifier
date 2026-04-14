configfile: "config/feature_analysis_config.yaml"

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  entropy.smk — Uncertainty-metric computation                           ║
# ║                                                                          ║
# ║  Single rule:                                                            ║
# ║    compute_entropy_metrics — H_pred / H_exp / I_bald from tool probs    ║
# ║                                                                          ║
# ║  Wildcards                                                               ║
# ║    {expt}  experiment name (config feature_analysis key)                ║
# ║                                                                          ║
# ║  Usage examples                                                          ║
# ║    snakemake compute_entropy_metrics -j 4 --use-conda                   ║
# ║    snakemake entropy_all             -j 4 --use-conda                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝


# ── Rules ─────────────────────────────────────────────────────────────────────

rule compute_entropy_metrics:
    """Compute per-transcript uncertainty metrics (H_pred, H_exp, I_bald) from tool probabilities.

    Reads the full feature table and the binary class table, extracts tool
    probabilities via get_probabilities(), and writes a TSV with one row per
    transcript containing all entropy / mutual-information metrics plus ground-
    truth coding_class and biotype columns.
    """
    input:
        full_table = "results/{expt}/tables/{expt}_full_table.tsv",
        binary     = "results/{expt}/tables/{expt}_binary_class_table.tsv",
    output:
        tsv = "results/{expt}/features/entropy/{expt}_uncertainty_analysis.tsv",
    params:
        output_dir = "results/{expt}/features/entropy",
    log:
        "logs/{expt}/features/entropy/compute_entropy_metrics.log",
    resources:
        mem_mb  = 8000,
        runtime = 30,
    conda:
        "lnc-datasets"
    shell:
        """
        python -u workflow/scripts/compute_entropy.py \
            --dataset     {wildcards.expt} \
            --output-dir  {params.output_dir} \
            2>&1 | tee {log}
        """

# ── Entropy group assignment ──────────────────────────────────────────────────

rule compute_entropy_groups:
    """Assign each transcript to an entropy group (low / middle / high).

    Uses the configured thresholding strategy and percentile thresholds from
    feature_analysis_config.yaml to produce a traceable group-assignment TSV
    that is consumed by downstream figure rules (entropy_main_figures).

    Strategies
    ----------
    overall        — thresholds across all transcripts (Strategy 1)
    class_separated — thresholds within coding / lncRNA separately (Strategy 2)

    Output columns
    --------------
    entropy_group : low | high | middle
                    low_lncRNA | high_lncRNA | low_coding | high_coding | middle
    """
    input:
        entropy_tsv = "results/{expt}/features/entropy/{expt}_uncertainty_analysis.tsv",
    output:
        tsv = "results/{expt}/features/entropy/{expt}_entropy_groups.tsv",
    params:
        mode    = lambda wc: config["feature_analysis"][wc.expt]["statistical_analysis"]["entropy_grouping_mode"],
        low_th  = lambda wc: config["feature_analysis"][wc.expt]["statistical_analysis"].get("low_threshold",  10),
        high_th = lambda wc: config["feature_analysis"][wc.expt]["statistical_analysis"].get("high_threshold", 90),
    log:
        "logs/{expt}/features/entropy/compute_entropy_groups.log",
    resources:
        mem_mb  = 2000,
        runtime = 10,
    conda:
        "lnc-datasets"
    shell:
        """
        python -u workflow/scripts/compute_entropy_groups.py \
            --entropy-tsv {input.entropy_tsv} \
            --output-tsv  {output.tsv} \
            --mode        {params.mode} \
            --low-th      {params.low_th} \
            --high-th     {params.high_th} \
        2>&1 | tee {log}
        """


# ── Convenience aggregate target ─────────────────────────────────────────────

rule entropy_all:
    """Run entropy metric computation for all configured experiments."""
    input:
        expand(
            "results/{expt}/features/entropy/{expt}_uncertainty_analysis.tsv",
            expt=list(config["feature_analysis"].keys()),
        ),
