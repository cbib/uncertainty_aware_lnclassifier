# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  feature_analysis.smk — Standalone entry-point for the feature           ║
# ║  analysis pipeline                                                        ║
# ║                                                                           ║
# ║  This file stitches together all feature-analysis rule modules so the    ║
# ║  entire pipeline (or any individual stage) can be invoked directly       ║
# ║  without going through the main Snakefile:                               ║
# ║                                                                           ║
# ║    snakemake -s workflow/rules/feature_analysis.smk <target> -j N        ║
# ║                                                                           ║
# ║  Module load order (dependencies respected):                             ║
# ║    statistical_analysis.smk  ← already includes entropy + clustering     ║
# ║    embeddings.smk                                                         ║
# ║    shap.smk                                                               ║
# ║                                                                           ║
# ║  Configfiles                                                              ║
# ║    config/feature_analysis_config.yaml  (entropy / clustering / stat /   ║
# ║                                          univariate / embeddings)         ║
# ║    config/shap_config.yaml              (RF + SHAP pipeline)             ║
# ║                                                                           ║
# ║  Convenience targets (all run from paper/)                               ║
# ║  ───────────────────────────────────────────────────────────────         ║
# ║  Stage targets                                                            ║
# ║    entropy_all            — compute entropy metrics                        ║
# ║    clustering_all         — feature-correlation clustering                ║
# ║    statistical_tests_all  — Mann-Whitney / chi² / FDR tests              ║
# ║    embeddings_all         — UMAP / t-SNE / PCA                           ║
# ║    shap_testing           — SHAP smoke-test (100 transcripts/fold)       ║
# ║    shap_full_all          — full-feature SHAP pipeline                   ║
# ║    shap_190_all           — correlation-filtered SHAP pipeline           ║
# ║    rfecv_shap_test        — RFECV → SHAP end-to-end test                 ║
# ║    rfecv_shap_all         — RFECV → SHAP full run                        ║
# ║    feature_analysis_all   — everything above                             ║
# ║                                                                           ║
# ║  Example invocations                                                     ║
# ║    # Dry-run the full pipeline                                            ║
# ║    snakemake -s workflow/rules/feature_analysis.smk feature_analysis_all ║
# ║              -n -j 1 --use-conda                                         ║
# ║                                                                           ║
# ║    # Run up to (and including) clustering, 4 cores                       ║
# ║    snakemake -s workflow/rules/feature_analysis.smk clustering_all       ║
# ║              -j 4 --use-conda                                            ║
# ║                                                                           ║
# ║    # SHAP smoke-test (fastest meaningful end-to-end run)                 ║
# ║    snakemake -s workflow/rules/feature_analysis.smk shap_testing         ║
# ║              -j 6 --use-conda                                            ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# ── Config ────────────────────────────────────────────────────────────────────
# Declare both configfiles here so they are loaded when this file is the
# Snakemake entry point.  Included modules also declare them; duplicate
# configfile directives are no-ops in Snakemake (values are merged).
configfile: "config/feature_analysis_config.yaml"
configfile: "config/shap_config.yaml"

# ── Module includes ───────────────────────────────────────────────────────────
# statistical_analysis.smk already `include:`s entropy.smk and clustering.smk,
# so those must NOT be included separately here — doing so would produce
# duplicate rule definitions.
# univariate.smk is DEPRECATED — do not re-add it here.
include: "statistical_analysis.smk"   # → entropy + clustering + statistical_tests
include: "embeddings.smk"
include: "shap.smk"

# ── Top-level aggregate target ────────────────────────────────────────────────

rule feature_analysis_all:
    """Run every stage of the feature analysis pipeline."""
    input:
        # entropy + clustering (via statistical_tests_all dependency chain)
        rules.statistical_tests_all.input,
        rules.embeddings_all.input,
        rules.shap_all.input,
