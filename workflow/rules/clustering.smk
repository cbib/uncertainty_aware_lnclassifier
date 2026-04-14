configfile: "config/feature_analysis_config.yaml"

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  clustering.smk — Feature correlation clustering pipeline                ║
# ║                                                                          ║
# ║  Produces the feature-cluster assignment table that is consumed as       ║
# ║  input by univariate_analysis (residualization) and shap/rfecv rules.    ║
# ║                                                                          ║
# ║  The optimal clustering threshold is chosen automatically by maximising  ║
# ║  the silhouette score on the precomputed feature distance matrix.        ║
# ║                                                                          ║
# ║  Outputs (per experiment)                                                ║
# ║    results/{expt}/features/clustering/feature_correlation_matrix.csv     ║
# ║    results/{expt}/features/clustering/feature_correlation_dendrogram.pdf ║
# ║    results/{expt}/features/clustering/feature_clusters_at_distances.csv  ║
# ║    results/{expt}/features/clustering/silhouette_scores.csv              ║
# ║    results/{expt}/features/clustering/silhouette_scores.pdf              ║
# ║    results/{expt}/features/clustering/optimal_threshold.txt              ║
# ║                                                                          ║
# ║  Wildcard                                                                ║
# ║    {expt} — experiment name (key in config["feature_analysis"])          ║
# ║                                                                          ║
# ║  Usage                                                                   ║
# ║    snakemake clustering_all -j 4 --use-conda                             ║
# ║    snakemake results/<expt>/features/clustering/feature_clusters_at_distances.csv ║
# ╚══════════════════════════════════════════════════════════════════════════╝


# ── Config helpers ─────────────────────────────────────────────────────────────

def _clust_cfg(wc):
    """Return the top-level feature_analysis block for the experiment."""
    return config["feature_analysis"][wc.expt]


def _clust_sub(wc):
    """Return the clustering sub-config (may be empty / absent)."""
    return config["feature_analysis"][wc.expt].get("clustering", {})


def _opt_path(path):
    """Return [] for null/empty paths so Snakemake does not treat them as required inputs."""
    return [path] if path else []


def _opt_arg(flag, path):
    """Build optional CLI args for nullable config paths."""
    return f"{flag} '{path}'" if path else ""


# ── Aggregate targets ──────────────────────────────────────────────────────────

_clustering_all_targets = [
    f"results/{expt}/features/clustering/feature_clusters_at_distances.csv"
    for expt in config["feature_analysis"]
]


# ── Rule: feature_clustering ───────────────────────────────────────────────────

rule feature_clustering:
    """
    Compute feature Spearman correlations and hierarchical (Ward) clustering.
    The optimal distance threshold is selected as the one maximising the
    silhouette score evaluated on the precomputed feature distance matrix.

    Outputs:
      - feature_correlation_matrix.csv      : symmetrised Spearman corr of continuous features
      - feature_correlation_dendrogram.pdf  : Ward dendrogram with optimal threshold line
      - feature_clusters_at_distances.csv   : flat cluster IDs across distance grid 0.05–1.55
      - silhouette_scores.csv               : silhouette score + n_clusters at each threshold
      - silhouette_scores.pdf               : silhouette score curve with optimal threshold marked
      - optimal_threshold.txt               : single-line file with the optimal distance value
    """
    input:
        full_table  = "results/{expt}/tables/{expt}_full_table.tsv",
        binary      = "results/{expt}/tables/{expt}_binary_class_table.tsv",
        te          = lambda wc: _opt_path(_clust_cfg(wc).get("te_features")),
        nbd         = lambda wc: _opt_path(_clust_cfg(wc).get("nbd_features")),
    output:
        corr_matrix       = "results/{expt}/features/clustering/feature_correlation_matrix.csv",
        dendrogram        = "results/{expt}/features/clustering/feature_correlation_dendrogram.pdf",
        clusters          = "results/{expt}/features/clustering/feature_clusters_at_distances.csv",
        silhouette_csv    = "results/{expt}/features/clustering/silhouette_scores.csv",
        silhouette_pdf    = "results/{expt}/features/clustering/silhouette_scores.pdf",
        optimal_threshold = "results/{expt}/features/clustering/optimal_threshold.txt",
    params:
        output_dir   = lambda wc: f"results/{wc.expt}/features/clustering",
        te_features  = lambda wc: _clust_cfg(wc)["te_features"],
        nbd_features = lambda wc: _clust_cfg(wc)["nbd_features"],
        te_arg       = lambda wc: _opt_arg("--te-features", _clust_cfg(wc).get("te_features")),
        nbd_arg      = lambda wc: _opt_arg("--nbd-features", _clust_cfg(wc).get("nbd_features")),
        corr_method  = lambda wc: _clust_sub(wc).get("corr_method", "spearman"),
        distance_min  = lambda wc: _clust_sub(wc).get("distance_min", 0.05),
        distance_max  = lambda wc: _clust_sub(wc).get("distance_max", 1.60),
        distance_step = lambda wc: _clust_sub(wc).get("distance_step", 0.05),
    log:
        "logs/{expt}/features/clustering/feature_clustering.log",
    threads: 4
    resources:
        mem_mb  = 32000,
        runtime = 120,
    conda:
        "lnc-datasets"
    shell:
        """
        python -u workflow/scripts/compute_feature_clustering.py  \
            --dataset             {wildcards.expt}                  \
            --output-dir          {params.output_dir}               \
            {params.te_arg}                                       \
            {params.nbd_arg}                                      \
            --corr-method         {params.corr_method}              \
            --distance-min        {params.distance_min}             \
            --distance-max        {params.distance_max}             \
            --distance-step       {params.distance_step}            \
        > {log} 2>&1
        """


# ── Rule: clustering_all ───────────────────────────────────────────────────────

rule clustering_all:
    """Convenience target: run feature clustering for all configured experiments."""
    input:
        _clustering_all_targets,
