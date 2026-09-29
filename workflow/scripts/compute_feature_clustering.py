"""
compute_feature_clustering.py — Build feature correlation / clustering artefacts.

Produces five files:
  <output-dir>/feature_correlation_matrix.csv      — Spearman correlation matrix (continuous features)
  <output-dir>/feature_correlation_dendrogram.pdf  — Ward dendrogram with optimal threshold line
  <output-dir>/feature_clusters_at_distances.csv   — Flat cluster IDs across distance grid
  <output-dir>/silhouette_scores.csv               — Silhouette score at each distance threshold
  <output-dir>/optimal_threshold.txt               — Single-line file containing the optimal distance

The optimal distance threshold is the one that maximises the silhouette score computed on
the precomputed feature distance matrix (mirrors the manual analysis in 021_feature_clustering.ipynb).

The CSV of cluster assignments (feature_clusters_at_distances.csv) is the critical
upstream artefact consumed by univariate_analysis.py and the SHAP/RFECV rules.

Usage:
    python compute_feature_clustering.py \\
        --dataset gencode.v47.common.cdhit.cv   \\
        --output-dir results/gencode.v47.common.cdhit.cv/clustering \\
        --supplementary-features results/gencode.v47.common.cdhit.cv/features/supplementary_features.tsv
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend — must come before pyplot import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score

# ── Path setup ─────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parents[1]))

from utils.entropy import load_dataset  # noqa: E402
from utils.feature_analysis import cluster_features  # noqa: E402
from utils.features import (  # noqa: E402
    filter_feature_columns,
    get_categorical_and_continuous_columns,
)
from utils.plotting import color_feature_ticklabels, feature_label  # noqa: E402

# ── CLI ────────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(
        description="Compute feature correlation matrix and hierarchical clustering."
    )
    p.add_argument(
        "--dataset",
        required=True,
        metavar="EXPT",
        help="Experiment / dataset name (e.g. gencode.v47.common.cdhit.cv)",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        metavar="DIR",
        help="Directory where the three output files will be written.",
    )
    p.add_argument(
        "--supplementary-features",
        required=True,
        metavar="TSV",
        help="Path to merged supplementary features TSV (output of merge_supplementary_features.py).",
    )
    p.add_argument(
        "--distance-min",
        type=float,
        default=0.05,
        metavar="FLOAT",
        help="Smallest distance threshold for flat clustering (default: 0.05).",
    )
    p.add_argument(
        "--distance-max",
        type=float,
        default=1.60,
        metavar="FLOAT",
        help="Exclusive upper bound for distance grid (default: 1.60).",
    )
    p.add_argument(
        "--distance-step",
        type=float,
        default=0.05,
        metavar="FLOAT",
        help="Step size for distance grid (default: 0.05).",
    )
    p.add_argument(
        "--dendrogram-threshold",
        type=float,
        default=None,
        metavar="FLOAT",
        help=(
            "Distance threshold highlighted on the dendrogram. "
            "Defaults to the optimal threshold derived from the maximum silhouette score."
        ),
    )
    p.add_argument(
        "--corr-method",
        default="spearman",
        choices=["spearman", "pearson", "kendall"],
        help="Correlation method (default: spearman).",
    )
    p.add_argument(
        "--force-rerun",
        action="store_true",
        default=False,
        help="Recompute even if output files already exist.",
    )
    return p.parse_args()


# ── Feature assembly ────────────────────────────────────────────────────────────


def build_full_feature_set(
    features: pd.DataFrame,
    supplementary: pd.DataFrame,
    index: pd.Index,
) -> pd.DataFrame:
    """
    Concatenate main features with the merged supplementary features.

    Steps:
        1. Reindex both DataFrames to the common transcript index.
        2. Concatenate along columns.
        3. Drop duplicated column names (keep first occurrence).
        4. Fill NaNs with 0 (expected for absent feature indicators).
        5. Convert everything to numeric (bool/object → 0/1).
        6. Drop constant columns (nunique ≤ 1).
    """
    supplementary_aligned = supplementary.reindex(index).fillna(0)
    # Align with notebook rename
    if "motif_types_present" in supplementary_aligned.columns:
        supplementary_aligned = supplementary_aligned.rename(
            columns={"motif_types_present": "n_motif_types"}
        )

    full = pd.concat([features.loc[index], supplementary_aligned], axis=1)
    full = full.loc[:, ~full.columns.duplicated(keep="first")]
    full.fillna(0, inplace=True)
    full = full.apply(pd.to_numeric, errors="coerce")

    numeric_cols = full.select_dtypes(include=[np.number]).columns
    print(
        f"  Keeping {len(numeric_cols)} numeric features "
        f"(out of {full.shape[1]} total after concatenation)"
    )
    full = full[numeric_cols]

    nunique = full.nunique()
    constant = nunique[nunique <= 1].index.tolist()
    if constant:
        print(
            f"  Removing {len(constant)} constant features: {constant[:8]}{'…' if len(constant) > 8 else ''}"
        )
        full = full.drop(columns=constant)

    return full


# ── Correlation & linkage ────────────────────────────────────────────────────────


def compute_correlation_and_linkage(
    continuous_df: pd.DataFrame, method: str = "spearman"
):
    """Symmetrised correlation matrix + Ward linkage on |1 - |corr||."""
    corr = continuous_df.corr(method=method)
    corr = (corr + corr.T) / 2  # enforce symmetry

    dist = 1.0 - np.abs(corr.values)
    dist = np.where(np.isfinite(dist), dist, 0.0)
    np.fill_diagonal(dist, 0.0)

    linkage = hierarchy.ward(squareform(dist))
    return corr, dist, linkage


# ── Silhouette score ──────────────────────────────────────────────────────────────


def compute_silhouette_scores(
    dist_matrix: np.ndarray,
    dist_linkage: np.ndarray,
    distances: np.ndarray,
) -> pd.DataFrame:
    """
    Compute silhouette score at each distance threshold using the precomputed
    feature distance matrix (mirrors the notebook cell that produces silhouette_df).

    Thresholds that produce only one cluster or N clusters equal to the number
    of features receive NaN (silhouette undefined).

    Returns
    -------
    pd.DataFrame with columns [distance_threshold, silhouette_score, n_clusters]
    indexed from 0.
    """
    n_features = dist_matrix.shape[0]
    records = []
    for dist in distances:
        labels = hierarchy.fcluster(dist_linkage, dist, criterion="distance")
        n_clusters = np.unique(labels).size
        if 1 < n_clusters < n_features:
            score = silhouette_score(dist_matrix, labels, metric="precomputed")
        else:
            score = np.nan
        records.append(
            {
                "distance_threshold": round(float(dist), 4),
                "silhouette_score": score,
                "n_clusters": int(n_clusters),
            }
        )
    return pd.DataFrame(records)


# ── Main ────────────────────────────────────────────────────────────────────────


def main():
    args = parse_args()

    out_dir = Path(args.output_dir)
    matrix_csv = out_dir / "feature_correlation_matrix.csv"
    dendrogram_pdf = out_dir / "feature_correlation_dendrogram.pdf"
    clusters_csv = out_dir / "feature_clusters_at_distances.csv"
    silhouette_csv = out_dir / "silhouette_scores.csv"
    optimal_threshold = out_dir / "optimal_threshold.txt"

    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load primary dataset ────────────────────────────────────────────────
    print(f"\n── Loading dataset: {args.dataset} ──")
    data = load_dataset(args.dataset)
    probs = data["probs"]
    features = data["features"]
    index = probs.index  # common transcript index after dropna

    features_filtered = features[filter_feature_columns(features)]
    print(f"  Transcripts : {len(index):,}")
    print(f"  Core features: {features_filtered.shape[1]}")

    # ── 2. Load merged supplementary features ─────────────────────────────────
    print("\n── Loading supplementary features ──")
    supplementary = pd.read_csv(args.supplementary_features, sep="\t", index_col=0)
    index = index.intersection(supplementary.index)

    # ── 3. Assemble and clean feature matrix ──────────────────────────────────
    print("\n── Assembling full feature set ──")
    full = build_full_feature_set(features_filtered, supplementary, index)

    # ── 5. Separate continuous vs categorical ─────────────────────────────────
    print("\n── Separating continuous / categorical ──")
    cat_cols, cont_cols = get_categorical_and_continuous_columns(full)
    continuous_df = full[cont_cols]
    categorical_df = full[cat_cols]

    # ── 6. Correlation matrix ─────────────────────────────────────────────────
    print(
        f"\n── Computing {args.corr_method} correlation matrix "
        f"({continuous_df.shape[1]} continuous features) ──"
    )
    corr_matrix, dist_matrix, dist_linkage = compute_correlation_and_linkage(
        continuous_df, method=args.corr_method
    )
    corr_matrix.index.name = "feature"
    corr_matrix.to_csv(matrix_csv, index_label="feature")
    print(f"✓ Saved: {matrix_csv}")

    # ── 6. Cluster assignments across distance grid ────────────────────────────
    distances = np.arange(args.distance_min, args.distance_max, args.distance_step)
    print(f"\n── Computing flat clusters for {len(distances)} distance thresholds ──")

    cluster_df = pd.DataFrame(index=corr_matrix.columns)
    for dist in distances:
        i_df = cluster_features(continuous_df, dist_linkage, dist).set_index(
            "feature_name"
        )
        cluster_df[f"cluster_{dist:.2f}"] = i_df["cluster_id"]

    cluster_df.index.name = "feature"
    cluster_df.to_csv(clusters_csv, index_label="feature")
    print(
        f"✓ Saved: {clusters_csv} ({cluster_df.shape[0]} features × {cluster_df.shape[1]} thresholds)"
    )

    # ── 7. Silhouette scores → optimal threshold ───────────────────────────────
    print(f"\n── Computing silhouette scores ({len(distances)} thresholds) ──")
    sil_df = compute_silhouette_scores(dist_matrix, dist_linkage, distances)
    sil_df.to_csv(silhouette_csv, index=False)
    print(f"✓ Saved: {silhouette_csv}")

    # Best threshold = distance with maximum silhouette score (ignoring NaN)
    valid = sil_df.dropna(subset=["silhouette_score"])
    if valid.empty:
        print(
            "⚠  No valid silhouette scores — falling back to --dendrogram-threshold or 0.25"
        )
        best_dist = (
            args.dendrogram_threshold if args.dendrogram_threshold is not None else 0.25
        )
    else:
        best_row = valid.loc[valid["silhouette_score"].idxmax()]
        best_dist = float(best_row["distance_threshold"])
        best_score = float(best_row["silhouette_score"])
        best_n = int(best_row["n_clusters"])
        print(
            f"✓ Optimal threshold: {best_dist:.2f}  "
            f"(silhouette={best_score:.4f}, n_clusters={best_n})"
        )

    # Override with explicit flag if provided
    dendrogram_thresh = (
        args.dendrogram_threshold
        if args.dendrogram_threshold is not None
        else best_dist
    )

    optimal_threshold.write_text(f"{best_dist:.4f}\n")
    print(f"✓ Saved: {optimal_threshold}")

    # ── 8. Cluster membership table at optimal threshold ──────────────────────
    membership_csv = out_dir / "feature_cluster_membership.csv"
    opt_col = f"cluster_{best_dist:.2f}"
    membership = cluster_df[[opt_col]].rename(columns={opt_col: "cluster_id"}).copy()
    rep_map = membership.reset_index().groupby("cluster_id")["feature"].first()
    membership["cluster_representative"] = membership["cluster_id"].map(rep_map)
    membership.to_csv(membership_csv, index_label="feature")
    print(f"✓ Saved: {membership_csv}")

    # ── 9. Silhouette score plot ───────────────────────────────────────────────
    silhouette_plot = out_dir / "silhouette_scores.pdf"
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        sil_df["distance_threshold"],
        sil_df["silhouette_score"],
        marker="o",
        linewidth=2,
        markersize=8,
    )
    ax.axvline(
        best_dist,
        color="red",
        linestyle="--",
        alpha=0.5,
        label=f"Selected threshold: {best_dist:.2f}",
    )
    ax.set_xlabel("Distance Threshold", fontsize=12)
    ax.set_ylabel("Silhouette Score", fontsize=12)
    ax.set_title("Silhouette Score vs Distance Threshold", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.set_dpi(300)
    plt.savefig(silhouette_plot, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved: {silhouette_plot}")

    # ── 10. Dendrogram (using optimal or explicit threshold) ───────────────────
    print(f"\n── Plotting dendrogram (threshold={dendrogram_thresh:.2f}) ──")
    fig, ax = plt.subplots(figsize=(5, 40))
    dn = hierarchy.dendrogram(
        dist_linkage,
        labels=corr_matrix.columns.tolist(),
        ax=ax,
        orientation="left",
        color_threshold=dendrogram_thresh,
    )
    # Readable, prefix-aware leaf labels coloured by rna/dna group (dn["ivl"] is
    # the drawn leaf order, matching the y tick labels bottom-to-top).
    ax.set_yticklabels([feature_label(n)[0] for n in dn["ivl"]])
    color_feature_ticklabels(ax, dn["ivl"], axis="y")
    ax.axvline(
        dendrogram_thresh,
        color="red",
        linestyle="--",
        linewidth=1,
        label=f"threshold = {dendrogram_thresh:.2f} (optimal silhouette)",
    )
    ax.legend(fontsize=7)
    plt.tight_layout()
    fig.set_dpi(300)
    plt.savefig(dendrogram_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved: {dendrogram_pdf}")


if __name__ == "__main__":
    main()
