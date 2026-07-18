configfile: "config/feature_analysis_config.yaml"

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  supplementary.smk — Upstream supplementary feature merge                ║
# ║                                                                          ║
# ║  Loads all 5 supplementary pipeline feature files once per experiment,   ║
# ║  applies per-pipeline cleaning, checks coverage against the main         ║
# ║  analysis transcript set, and writes a single merged feature TSV         ║
# ║  consumed by every downstream feature-analysis rule.                     ║
# ║                                                                          ║
# ║  Wildcard                                                                ║
# ║    {expt} — experiment name (key in config["feature_analysis"])          ║
# ╚══════════════════════════════════════════════════════════════════════════╝


def _opt_path(path):
    """Return [] for null/empty paths so Snakemake does not treat them as required inputs."""
    return [path] if path else []


def _opt_arg(flag, path):
    """Build optional CLI args for nullable config paths."""
    return f"{flag} '{path}'" if path else ""


rule merge_supplementary_features:
    """
    Load all supplementary pipeline feature files, apply per-pipeline index and
    column cleaning, check coverage against the main analysis transcript set, and
    write a single merged feature TSV consumed by all downstream feature-analysis rules.

    Produces:
      supplementary_features.tsv     — cleaned, merged feature matrix (transcripts × features)
                                        indexed by seq_ID; only transcripts present with valid
                                        data in every non-empty pipeline are included
      supplementary_traceability.tsv — long-format report (may be empty), one row per
                                        (transcript, issue); columns: category, pipeline, detail.
                                        category ∈ {missing_from_pipeline, not_in_main_index,
                                        invalid_data}
    """
    input:
        binary   = "results/{expt}/tables/{expt}_full_table.tsv",
        te_rna   = lambda wc: _opt_path(config["feature_analysis"][wc.expt].get("te_features_rna")),
        te_dna   = lambda wc: _opt_path(config["feature_analysis"][wc.expt].get("te_features_dna")),
        nbd      = lambda wc: _opt_path(config["feature_analysis"][wc.expt].get("nbd_features")),
        scanfold = lambda wc: _opt_path(config["feature_analysis"][wc.expt].get("scanfold_features")),
        rg4      = lambda wc: _opt_path(config["feature_analysis"][wc.expt].get("rg4_features")),
    output:
        features     = "results/{expt}/features/supplementary_features.tsv",
        traceability = "results/{expt}/features/supplementary_traceability.tsv",
    params:
        te_rna_arg   = lambda wc: _opt_arg("--te-features-rna",   config["feature_analysis"][wc.expt].get("te_features_rna")),
        te_dna_arg   = lambda wc: _opt_arg("--te-features-dna",   config["feature_analysis"][wc.expt].get("te_features_dna")),
        nbd_arg      = lambda wc: _opt_arg("--nbd-features",      config["feature_analysis"][wc.expt].get("nbd_features")),
        scanfold_arg = lambda wc: _opt_arg("--scanfold-features", config["feature_analysis"][wc.expt].get("scanfold_features")),
        rg4_arg      = lambda wc: _opt_arg("--rg4-features",      config["feature_analysis"][wc.expt].get("rg4_features")),
    log:
        "logs/{expt}/features/merge_supplementary_features.log",
    threads: 1
    resources:
        mem_mb  = 8000,
        runtime = 30,
    conda:
        "lnc-datasets"
    shell:
        """
        python -u workflow/scripts/merge_supplementary_features.py \
            --main-index   {input.binary}   \
            {params.te_rna_arg}             \
            {params.te_dna_arg}             \
            {params.nbd_arg}                \
            {params.scanfold_arg}           \
            {params.rg4_arg}                \
            --output       {output.features} \
            --trace-output {output.traceability} \
        2>&1 | tee {log}
        """
