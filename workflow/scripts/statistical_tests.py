#!/usr/bin/env python3
"""
Statistical Testing Pipeline for Feature Analysis

Performs statistical tests on transcript features based on entropy groups,
calculating effect sizes and identifying top discriminative features.

Outputs (per pairwise comparison):
  - *_mannwhitney.csv  : Mann-Whitney U results + VDA + FDR (continuous features)
  - *_chi2.csv         : Chi-squared results + Cramér's V + OR + FDR (categorical)
  - *_cat_freq.tsv     : % of transcripts per group with each categorical feature
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add workflow utilities to path
_WORKFLOW_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(_WORKFLOW_DIR))

import warnings

from utils.entropy import (
    load_additional_features,
    load_dataset,
    load_entropy_groups,
    split_entropy_group_indices,
)
from utils.features import filter_feature_columns, remove_constant_features
from utils.stats import compute_pairwise_stats

warnings.filterwarnings("ignore")


def setup_logging(verbose=True):
    """Configure console output."""
    if verbose:
        print("✓ Imports successful", file=sys.stderr)


def prepare_features(features, te_features, nbd_features, verbose=True):
    """Combine and preprocess all feature sets."""
    combined = pd.concat([features, te_features, nbd_features], axis=1)
    combined = combined.loc[:, ~combined.columns.duplicated(keep="first")]
    combined.fillna(0, inplace=True)
    combined = combined.apply(pd.to_numeric, errors="coerce")

    # Keep numeric features and remove constants
    numeric_cols = combined.select_dtypes(include=[np.number]).columns
    numeric_df = combined[numeric_cols]
    full_feature_set = remove_constant_features(numeric_df, name="Full feature set")

    cat_cols, continuous_cols = get_categorical_and_continuous_columns(full_feature_set)

    categorical_features = full_feature_set[cat_cols] if cat_cols else pd.DataFrame()
    scalar_features = full_feature_set[scalar_cols] if scalar_cols else pd.DataFrame()

    if verbose:
        print(f"  Continuous features: {len(continuous_cols)}", file=sys.stderr)
        print(f"  Categorical features: {len(cat_cols)}", file=sys.stderr)

    return categorical_features, continuous_features


def save_pairwise_results(
    output_dir: Path,
    stem: str,
    mannu_df: pd.DataFrame,
    chi2_df: pd.DataFrame,
    grp1: pd.Index,
    grp2: pd.Index,
    categorical_features: pd.DataFrame,
) -> None:
    """Persist pairwise test tables and categorical-feature frequencies."""
    mannu_df.to_csv(output_dir / f"{stem}_mannwhitney.csv")
    chi2_df.to_csv(output_dir / f"{stem}_chi2.csv")
    cat_freq = pd.DataFrame(
        {
            "group1": categorical_features.loc[grp1].mean() * 100,
            "group2": categorical_features.loc[grp2].mean() * 100,
        }
    )
    cat_freq.to_csv(output_dir / f"{stem}_cat_freq.tsv", sep="\t")
    print(
        f"✓ Saved pairwise results: {stem}_mannwhitney.csv, {stem}_chi2.csv, "
        f"{stem}_cat_freq.tsv",
        file=sys.stderr,
    )


def load_cluster_assignments(
    cluster_file: str | None,
    cluster_threshold: float,
) -> pd.DataFrame | None:
    """Load the selected cluster assignment column and rename it to ``cluster``."""
    if not cluster_file or not os.path.exists(cluster_file):
        return None

    cluster_df = pd.read_csv(cluster_file, index_col=0)
    cluster_colname = f"cluster_{cluster_threshold:.2f}"
    if cluster_colname not in cluster_df.columns:
        available_thresholds = [
            float(col.split("_")[1])
            for col in cluster_df.columns
            if col.startswith("cluster_")
        ]
        closest_threshold = min(
            available_thresholds,
            key=lambda value: abs(value - cluster_threshold),
        )
        cluster_colname = f"cluster_{closest_threshold:.2f}"
        print(
            f"⚠ Exact threshold {cluster_threshold:.2f} not found, using closest: "
            f"{closest_threshold:.2f}",
            file=sys.stderr,
        )

    cluster_df_subset = cluster_df[[cluster_colname]].copy()
    cluster_df_subset.columns = ["cluster"]
    print(
        f"Loaded cluster assignments from {cluster_colname}: "
        f"{cluster_df_subset['cluster'].nunique(dropna=True)} clusters",
        file=sys.stderr,
    )
    return cluster_df_subset


def annotate_results_with_clusters(
    result_df: pd.DataFrame,
    cluster_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """Append cluster IDs to a result table when cluster assignments exist."""
    if cluster_df is None:
        return result_df
    annotated_df = result_df.join(cluster_df, how="left")
    columns = ["cluster"] + [col for col in annotated_df.columns if col != "cluster"]
    return annotated_df[columns]


def summarize_cluster_counts(
    mannu_df: pd.DataFrame,
    chi2_df: pd.DataFrame,
) -> tuple[int, int]:
    """Return significant and total cluster counts across both result tables."""
    if "cluster" not in mannu_df.columns and "cluster" not in chi2_df.columns:
        return 0, 0

    all_clusters = pd.concat(
        [
            (
                mannu_df["cluster"]
                if "cluster" in mannu_df.columns
                else pd.Series(dtype=object)
            ),
            (
                chi2_df["cluster"]
                if "cluster" in chi2_df.columns
                else pd.Series(dtype=object)
            ),
        ]
    ).dropna()

    significant_clusters = pd.concat(
        [
            (
                mannu_df.loc[mannu_df["significant"], "cluster"]
                if "cluster" in mannu_df.columns
                else pd.Series(dtype=object)
            ),
            (
                chi2_df.loc[chi2_df["significant"], "cluster"]
                if "cluster" in chi2_df.columns
                else pd.Series(dtype=object)
            ),
        ]
    ).dropna()
    return int(significant_clusters.nunique()), int(all_clusters.nunique())


def build_entropy_main_comparisons(
    groups: pd.Series,
    entropy_df: pd.DataFrame,
) -> dict[str, tuple[pd.Index, pd.Index]]:
    """
    Return transcript indices all of the pairwise comparisons.
        - high_entropy_pc_v_lnc: high entropy coding vs lncRNA
        - low_entropy_pc_v_lnc: low entropy coding vs lncRNA
        - low_vs_high_entropy: low vs high entropy (all transcripts)
    If class-separated groups are detected, also include:
        - supp_coding_low_vs_high_entropy: low vs high entropy within coding transcripts
        - supp_lncrna_low_vs_high_entropy: low vs high entropy within lncRNA
    param groups: Series mapping transcript IDs to entropy group labels (e.g., "high", "low", "high_coding", "low_lncRNA")
    param entropy_df: DataFrame containing transcript metadata, including "coding_class" column (0 for lncRNA, 1 for coding)
    return: Dictionary mapping comparison names to tuples of (group1_indices, group2_indices)
    """
    grp = groups.reindex(entropy_df.index)
    high_idx = grp[grp.str.startswith("high")].index
    low_idx = grp[grp.str.startswith("low")].index

    lnc_idx = entropy_df[entropy_df["coding_class"] == 0].index
    coding_idx = entropy_df[entropy_df["coding_class"] == 1].index

    comparisons: dict[str, tuple[pd.Index, pd.Index]] = {
        "high_entropy_pc_v_lnc": (
            high_idx.intersection(lnc_idx),
            high_idx.intersection(coding_idx),
        ),
        "low_entropy_pc_v_lnc": (
            low_idx.intersection(lnc_idx),
            low_idx.intersection(coding_idx),
        ),
        "low_vs_high_entropy": (low_idx, high_idx),
    }

    is_class_separated = any(
        label.endswith("_coding") or label.endswith("_lncRNA")
        for label in grp.dropna().astype(str).unique()
    )
    if is_class_separated:
        comparisons["supp_coding_low_vs_high_entropy"] = (
            low_idx.intersection(coding_idx),
            high_idx.intersection(coding_idx),
        )
        comparisons["supp_lncrna_low_vs_high_entropy"] = (
            low_idx.intersection(lnc_idx),
            high_idx.intersection(lnc_idx),
        )

    return comparisons


def log_entropy_group_summary(
    groups: pd.Series,
    entropy_df: pd.DataFrame,
    comparisons: dict[str, tuple[pd.Index, pd.Index]],
) -> None:
    """Log entropy-group counts and derived pairwise comparison sizes."""
    grp = groups.reindex(entropy_df.index).dropna().astype(str)
    group_counts = grp.value_counts().sort_index()
    strategy = (
        "class_separated"
        if any(
            label.endswith("_coding") or label.endswith("_lncRNA")
            for label in group_counts.index
        )
        else "overall"
    )

    print(f"Entropy grouping strategy: {strategy}", file=sys.stderr)
    print("Entropy group counts:", file=sys.stderr)
    for label, count in group_counts.items():
        print(f"  {label}: n={count}", file=sys.stderr)

    print("Pairwise comparisons:", file=sys.stderr)
    for comparison_name, (grp1, grp2) in comparisons.items():
        print(
            f"  {comparison_name}: group1 n={len(grp1)}, group2 n={len(grp2)}",
            file=sys.stderr,
        )


def parse_arguments():
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Statistical testing pipeline for feature analysis"
    )
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument(
        "--output-dir", required=True, help="Output directory for results"
    )
    parser.add_argument(
        "--entropy-tsv", required=True, help="Path to entropy metrics TSV"
    )
    parser.add_argument(
        "--groups-tsv",
        required=True,
        help="Path to persisted entropy groups TSV (output of compute_entropy_groups.py)",
    )
    parser.add_argument(
        "--te-features",
        default="",
        help="Path to TE features (optional)",
    )
    parser.add_argument(
        "--nbd-features",
        default="",
        help="Path to NBD features (optional)",
    )
    parser.add_argument(
        "--cluster-file",
        default=None,
        help="Path to feature cluster assignments",
    )
    parser.add_argument(
        "--cluster-threshold",
        type=float,
        default=0.25,
        help="Cluster threshold for feature selection",
    )
    parser.add_argument(
        "--fdr-method",
        default="fdr_bh",
        help="FDR correction method (default: fdr_bh)",
    )
    parser.add_argument(
        "--fdr-alpha",
        type=float,
        default=0.01,
        help="FDR alpha threshold (default: 0.01)",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    return parser.parse_args()


def main():
    args = parse_arguments()

    setup_logging(args.verbose)
    os.makedirs(args.output_dir, exist_ok=True)

    basedir = Path.cwd()
    dataset_name = args.dataset

    # Load main dataset
    print(f"Loading dataset: {dataset_name}", file=sys.stderr)
    dataset = load_dataset(dataset_name)
    pipelines = {
        "te_pipeline": args.te_features or None,
        "nbd_pipeline": args.nbd_features or None,
    }
    dataset.update(
        load_additional_features(
            dataset_name,
            basedir,
            pipelines=pipelines,
        )
    )

    probs = dataset["probs"]
    labels = dataset["labels"]
    features = dataset["features"]
    features_to_keep = filter_feature_columns(features)
    features = features[features_to_keep]
    te_features = dataset.get("te_pipeline", pd.DataFrame()).fillna(0)
    nbd_features = dataset.get("nbd_pipeline", pd.DataFrame()).fillna(0)

    if te_features.empty or nbd_features.empty:
        print("⚠ Some feature sets not loaded", file=sys.stderr)

    # Load entropy metrics
    entropy_df = pd.read_csv(args.entropy_tsv, sep="\t", index_col=0)
    entropy_df = entropy_df.loc[probs.index].copy()
    entropy_df["biotype"] = labels.loc[entropy_df.index, "biotype"]
    entropy_df["coding_class"] = labels.loc[entropy_df.index, "coding_class"]

    # Load persisted entropy groups (produced by compute_entropy_groups.py)
    groups = load_entropy_groups(args.groups_tsv, transcript_index=entropy_df.index)
    grp_low, grp_high = split_entropy_group_indices(groups)
    print(
        f"✓ Loaded persisted entropy groups: {len(grp_low)} low, {len(grp_high)} high",
        file=sys.stderr,
    )

    # Prepare features
    print("Preparing features...", file=sys.stderr)
    categorical_features, scalar_features = prepare_features(
        features, te_features, nbd_features, verbose=args.verbose
    )

    cluster_df_subset = load_cluster_assignments(
        args.cluster_file,
        args.cluster_threshold,
    )

    comparison_output_dir = Path(args.output_dir)
    comparison_results = build_entropy_main_comparisons(groups, entropy_df)
    log_entropy_group_summary(groups, entropy_df, comparison_results)

    for comparison_name, (grp1, grp2) in comparison_results.items():
        if len(grp1) == 0 or len(grp2) == 0:
            print(
                f"⚠ Skipping pairwise results for {comparison_name}: empty group(s) "
                f"({len(grp1)}, {len(grp2)})",
                file=sys.stderr,
            )
            continue

        combined_grp = grp1.union(grp2)
        scalar_features_comp = remove_constant_features(
            scalar_features.loc[combined_grp],
            name=f"Scalar features ({comparison_name})",
        )
        categorical_features_comp = remove_constant_features(
            categorical_features.loc[combined_grp],
            name=f"Categorical features ({comparison_name})",
        )

        # NOTE: continuous comparisons ask always if grp2 > grp1
        pair_mannu_df, pair_chi2_df = compute_pairwise_stats(
            grp1,
            grp2,
            scalar_features_comp,
            categorical_features_comp,
            fdr_method=args.fdr_method,
            fdr_alpha=args.fdr_alpha,
        )
        pair_mannu_df = annotate_results_with_clusters(pair_mannu_df, cluster_df_subset)

        # TODO: Check if I want to remove empty cluster column from categorical features
        pair_chi2_df = annotate_results_with_clusters(pair_chi2_df, cluster_df_subset)
        n_sign = int(
            pair_mannu_df["significant"].sum() + pair_chi2_df["significant"].sum()
        )
        n_features = len(pair_mannu_df) + len(pair_chi2_df)
        n_sig_clusters, n_clusters = summarize_cluster_counts(
            pair_mannu_df,
            pair_chi2_df,
        )
        print(
            f"✓ {comparison_name}: {n_sign}/{n_features} significant features, "
            f"{n_sig_clusters}/{n_clusters} clusters with at least one significant "
            f"feature (FDR < {args.fdr_alpha})",
            file=sys.stderr,
        )
        save_pairwise_results(
            comparison_output_dir,
            comparison_name,
            pair_mannu_df,
            pair_chi2_df,
            grp1=grp1,
            grp2=grp2,
            categorical_features=categorical_features_comp,
        )
    print("✓ Statistical analysis pipeline complete!", file=sys.stderr)


if __name__ == "__main__":
    main()
