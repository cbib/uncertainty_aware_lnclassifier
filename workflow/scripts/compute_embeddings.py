#!/usr/bin/env python
"""
compute_embeddings.py — CLI script for computing dimensionality reduction embeddings.

Computes UMAP, t-SNE, and/or PCA embeddings from lncRNA classifier features,
saving results to a cache directory for downstream visualisation.

Usage (from paper/):
    python -u workflow/scripts/compute_embeddings.py \
        --dataset           gencode.v47.common.cdhit.cv  \
        --output-dir        results/gencode.v47.common.cdhit.cv/embeddings \
        --te-features       te_pipeline/results/te_analysis_flexible/features/all_transcripts_te_features.csv \
        --nbd-features      nonb-pipeline/results/gencode.v47/extended_analysis/features_nonb_features.csv \
        --methods           umap,tsne,pca \
        --umap-neighbors    30 \
        --umap-min-dist     0.1 \
        --tsne-perplexity   30 \
        --tsne-n-iter       1000
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add workflow/ to sys.path so that utils.* can be imported
sys.path.insert(0, str(Path(__file__).parents[1]))

from utils.embeddings import EmbeddingPipeline
from utils.entropy import load_dataset
from utils.features import custom_feature_scaling, filter_feature_columns

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute dimensionality reduction embeddings (UMAP / t-SNE / PCA) "
        "for lncRNA classifier features.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Core dataset / paths ────────────────────────────────────────────────
    parser.add_argument(
        "--dataset",
        required=True,
        help="Experiment name, e.g. 'gencode.v47.common.cdhit.cv'.",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Base results directory (relative to CWD = paper/).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Where to write embedding cache, features, and labels.",
    )

    # ── Optional supplementary feature files ───────────────────────────────
    parser.add_argument(
        "--te-features",
        default="",
        help="Path to TE pipeline features CSV.  Empty = not used.",
    )
    parser.add_argument(
        "--nbd-features",
        default="",
        help="Path to NBD pipeline features CSV.  Empty = not used.",
    )

    # ── Method selection ────────────────────────────────────────────────────
    parser.add_argument(
        "--methods",
        default="umap,tsne,pca",
        help="Comma-separated list of methods to run.",
    )

    # ── UMAP hyperparameters ────────────────────────────────────────────────
    parser.add_argument(
        "--umap-neighbors",
        type=int,
        default=30,
        help="n_neighbors for UMAP.",
    )
    parser.add_argument(
        "--umap-min-dist",
        type=float,
        default=0.1,
        help="min_dist for UMAP.",
    )

    # ── t-SNE hyperparameters ───────────────────────────────────────────────
    parser.add_argument(
        "--tsne-perplexity",
        type=int,
        default=30,
        help="Perplexity for t-SNE.",
    )
    parser.add_argument(
        "--tsne-n-iter",
        type=int,
        default=1000,
        help="Number of iterations for t-SNE.",
    )

    # ── Preprocessing ───────────────────────────────────────────────────────
    parser.add_argument(
        "--preprocess",
        default="",
        help="Optional preprocessing before embedding, e.g. 'pca_50'.  "
        "Empty / omitted = use scaled features directly.",
    )

    # ── Caching control ─────────────────────────────────────────────────────
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Recompute embeddings even if the sentinel file already exists.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    sentinel = output_dir / "embeddings_complete.flag"

    # ── Early exit if already complete and not forced ──────────────────────
    if sentinel.exists() and not args.force_rerun:
        print(
            f"Sentinel file found at {sentinel}; skipping computation. "
            f"Pass --force-rerun to override."
        )
        sys.exit(0)

    # ── 1. Load main dataset ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Loading dataset: {args.dataset}")
    data = load_dataset(args.dataset)

    features_df = data["features"]
    labels = data["labels"]

    print(
        f"  Transcripts: {features_df.shape[0]}, "
        f"raw columns: {features_df.shape[1]}"
    )

    # ── 2. Merge optional supplementary feature files ──────────────────────
    for path_str, tag in [
        (args.te_features, "TE"),
        (args.nbd_features, "NBD"),
    ]:
        if not path_str:
            continue
        p = Path(path_str)
        if not p.exists():
            print(f"Warning: {tag} features file not found: {path_str} — skipping.")
            continue
        print(f"Loading {tag} features from {path_str} ...")
        extra_df = pd.read_csv(
            p,
            sep="\t" if p.suffix == ".tsv" else ",",
            index_col=0,
        )
        features_df = features_df.join(extra_df, how="left", rsuffix=f"_{tag.lower()}")
        print(f"  Columns after merging {tag}: {features_df.shape[1]}")

    # ── 3. Filter, select numeric columns, drop constants ─────────────────
    print(f"\n{'='*60}")
    print("Filtering and scaling features ...")

    feature_cols = filter_feature_columns(features_df)
    selected_df = features_df[feature_cols].select_dtypes(include="number")
    print(
        f"  Numeric feature columns after filter_feature_columns: {selected_df.shape[1]}"
    )

    nunique = selected_df.nunique()
    constant_cols = nunique[nunique <= 1].index.tolist()
    if constant_cols:
        print(f"  Removing {len(constant_cols)} constant features: {constant_cols}")
        selected_df = selected_df.drop(columns=constant_cols)

    # ── 4. Scale features ──────────────────────────────────────────────────
    scaled_values = custom_feature_scaling(selected_df)
    scaled_df = pd.DataFrame(
        scaled_values,
        index=selected_df.index,
        columns=selected_df.columns,
    )
    print(f"  Final shape for embedding: {scaled_df.shape}")

    # ── 5. Initialise pipeline and persist features / labels ──────────────
    subset_id = "all"
    pipeline = EmbeddingPipeline(
        embedding_dir=output_dir,
        subset_id=subset_id,
        resource_guard=None,
    )

    pipeline.save_features(scaled_df, name="scaled_features")

    labels_aligned = labels.loc[labels.index.isin(scaled_df.index)]
    pipeline.save_labels(labels_aligned, name="ground_truth")

    # ── 6. Compute embeddings ──────────────────────────────────────────────
    X = scaled_df.values
    index = scaled_df.index
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    preprocess = args.preprocess.strip() or None

    for method in methods:
        print(f"\n{'='*60}")
        print(f"Computing {method.upper()} embedding ...")

        if method == "umap":
            kwargs = {
                "n_neighbors": args.umap_neighbors,
                "min_dist": args.umap_min_dist,
            }
        elif method == "tsne":
            kwargs = {
                "perplexity": args.tsne_perplexity,
                "max_iter": args.tsne_n_iter,
            }
        elif method == "pca":
            kwargs = {"n_components": 2}
        else:
            print(f"  Unknown method '{method}' — skipping.")
            continue

        pipeline.compute_or_load_embedding(
            X,
            index,
            method=method,
            preprocess=preprocess,
            force_recompute=args.force_rerun,
            **kwargs,
        )

    # ── 7. Write sentinel ──────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    sentinel.write_text("done\n")
    print(f"\n{'='*60}")
    print(f"All embeddings complete.  Sentinel written: {sentinel}")


if __name__ == "__main__":
    main()
