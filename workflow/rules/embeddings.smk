configfile: "config/feature_analysis_config.yaml"

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  embeddings.smk — Dimensionality-reduction embedding pipeline            ║
# ║                                                                          ║
# ║  Computes UMAP, t-SNE, and PCA embeddings from lncRNA classifier         ║
# ║  features and writes results to results/{expt}/features/embeddings/.     ║
# ║                                                                          ║
# ║  Wildcards                                                               ║
# ║    {expt}  experiment name  (key under config["feature_analysis"])       ║
# ║                                                                          ║
# ║  Usage examples                                                          ║
# ║    snakemake compute_embeddings --use-conda -j 1 \                       ║
# ║        --config expt=gencode.v47.common.cdhit.cv                         ║
# ║    snakemake embeddings_all     --use-conda -j 4                         ║
# ╚══════════════════════════════════════════════════════════════════════════╝


# ── Config helpers ────────────────────────────────────────────────────────────

def _embed_cfg(wc):
    """Return the embeddings sub-config block for the given experiment."""
    return config["feature_analysis"][wc.expt]["embeddings"]


# ── Rules ─────────────────────────────────────────────────────────────────────

rule compute_embeddings:
    """Compute UMAP / t-SNE / PCA embeddings for a single experiment.

    Reads the full feature table and (optionally) TE / NBD pipeline feature
    files, scales them, and writes all embeddings to the cache directory.
    A sentinel file is written upon successful completion so the rule can be
    skipped on subsequent runs without --force-rerun.
    """
    input:
        full_table    = "results/{expt}/tables/{expt}_full_table.tsv",
        binary        = "results/{expt}/tables/{expt}_binary_class_table.tsv",
        supplementary = "results/{expt}/features/supplementary_features.tsv",
    output:
        flag = "results/{expt}/features/embeddings/embeddings_complete.flag",
    params:
        output_dir      = "results/{expt}/features/embeddings",
        nbd_features    = lambda wc: config["feature_analysis"][wc.expt]["nbd_features"],
        methods         = lambda wc: ",".join(_embed_cfg(wc)["methods"].keys()),
        umap_n          = lambda wc: list(_embed_cfg(wc)["methods"]["umap"]["n_neighbors"])[0],
        umap_min_dist   = lambda wc: list(_embed_cfg(wc)["methods"]["umap"]["min_dist"])[0],
        tsne_perplexity = lambda wc: list(_embed_cfg(wc)["methods"]["tsne"]["perplexity"])[0],
        tsne_n_iter     = lambda wc: list(_embed_cfg(wc)["methods"]["tsne"]["n_iter"])[0],
        preprocess      = lambda wc: ("--preprocess " + str(_embed_cfg(wc)["preprocess"])) if _embed_cfg(wc).get("preprocess") else "",
    log:
        "logs/{expt}/features/embeddings/compute_embeddings.log",
    resources:
        mem_mb   = 32000,
        runtime  = 120,
    conda:
        "lnc-datasets"
    shell:
        """
        python -u workflow/scripts/compute_embeddings.py \
            --dataset           {wildcards.expt} \
            --results-dir       results \
            --output-dir        {params.output_dir} \
            --supplementary-features {input.supplementary} \
            --methods           {params.methods} \
            --umap-neighbors    {params.umap_n} \
            --umap-min-dist     {params.umap_min_dist} \
            --tsne-perplexity   {params.tsne_perplexity} \
            --tsne-n-iter       {params.tsne_n_iter} \
            {params.preprocess} \
        2>&1 | tee {log}
        """


rule embeddings_all:
    """Convenience target: compute embeddings for all configured experiments."""
    input:
        expand(
            "results/{expt}/features/embeddings/embeddings_complete.flag",
            expt=config["feature_analysis"].keys(),
        ),
