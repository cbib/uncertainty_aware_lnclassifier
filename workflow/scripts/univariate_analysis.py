#!/usr/bin/env python3
"""
univariate_analysis.py — CLI script for univariate statistical tests on
transcript features grouped by entropy level, biotype, or classification
correctness.

Usage
-----
python workflow/scripts/univariate_analysis.py \
    --dataset gencode.v47.common.cdhit.cv \
    --output-dir results/gencode.v47.common.cdhit.cv/simple_analysis/H_pred_th10-90 \
    --entropy-tsv results/.../uncertainty_analysis.tsv \
    --entropy-groups-csv results/.../entropy_groups.csv \
    --grouping-method entropy \
    --entropy-column H_pred \
    --low-threshold 10 \
    --high-threshold 90 \
    --residualize \
    --tests mann_whitney,f_test,mutual_info
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make workflow/utils importable regardless of working directory
sys.path.insert(0, str(Path(__file__).parents[1]))

from utils.entropy import load_additional_features, load_dataset  # noqa: E402
from utils.feature_analysis import (  # noqa: E402
    perform_f_tests,
    perform_mann_whitney_tests,
    perform_mutual_info_tests,
    rank_features_by_composite,
    residualize_features,
)
from utils.features import custom_feature_scaling, filter_feature_columns  # noqa: E402

# ────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ────────────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(
        description="Run univariate statistical tests on features grouped by "
        "entropy level, biotype, or classification correctness."
    )
    p.add_argument(
        "--dataset",
        required=True,
        help="Experiment name, e.g. gencode.v47.common.cdhit.cv",
    )
    p.add_argument(
        "--results-dir",
        default="results",
        help="Base results directory (default: results)",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write output CSVs and the completion sentinel",
    )
    p.add_argument(
        "--te-features",
        default="",
        help="Path to TE pipeline features CSV (optional)",
    )
    p.add_argument(
        "--nbd-features",
        default="",
        help="Path to NBD pipeline features CSV (optional)",
    )
    p.add_argument(
        "--entropy-tsv",
        required=True,
        help="Path to uncertainty_analysis TSV",
    )
    p.add_argument(
        "--entropy-groups-csv",
        required=True,
        help="Path to entropy_groups CSV produced by define_entropy_groups.py",
    )
    p.add_argument(
        "--grouping-method",
        required=True,
        choices=["entropy", "biotype", "correctness"],
        help="How to partition transcripts into groups",
    )
    p.add_argument(
        "--entropy-column",
        default="H_pred",
        help="Which entropy column to use for entropy-based grouping (default: H_pred)",
    )
    p.add_argument(
        "--low-threshold",
        type=float,
        default=10.0,
        help="Percentile for low-entropy boundary (default: 10)",
    )
    p.add_argument(
        "--high-threshold",
        type=float,
        default=90.0,
        help="Percentile for high-entropy boundary (default: 90)",
    )
    p.add_argument(
        "--cluster-file",
        default="",
        help="Path to feature_clusters_at_distances.csv for correlation filtering (optional)",
    )
    p.add_argument(
        "--cluster-threshold",
        type=float,
        default=0.25,
        help="Cluster distance threshold for feature filtering (default: 0.25)",
    )
    p.add_argument(
        "--residualize",
        dest="residualize",
        action="store_true",
        default=True,
        help="Residualize features against transcript length (default: True)",
    )
    p.add_argument(
        "--no-residualize",
        dest="residualize",
        action="store_false",
        help="Skip length residualization",
    )
    p.add_argument(
        "--tests",
        default="mann_whitney,f_test,mutual_info",
        help="Comma-separated list of tests to run (default: mann_whitney,f_test,mutual_info)",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance threshold (default: 0.05)",
    )
    p.add_argument(
        "--correction-method",
        default="fdr_bh",
        help="Multiple-testing correction method (default: fdr_bh)",
    )
    p.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    p.add_argument(
        "--force-rerun",
        action="store_true",
        help="Ignore existing completion sentinel and re-run",
    )
    return p.parse_args()


# ────────────────────────────────────────────────────────────────────────────
# Group-label construction
# ────────────────────────────────────────────────────────────────────────────


def _build_group_df(args, data, entropy_groups_df):
    """Return a DataFrame with a single 'group' column, indexed by seq_ID."""
    method = args.grouping_method

    if method == "entropy":
        # Prefer {entropy_column}_group; fall back to generic 'group'
        preferred_col = f"{args.entropy_column}_group"
        if preferred_col in entropy_groups_df.columns:
            col_name = preferred_col
        elif "group" in entropy_groups_df.columns:
            col_name = "group"
        else:
            raise ValueError(
                f"Cannot find group column '{preferred_col}' or 'group' in entropy "
                f"groups CSV. Available columns: {list(entropy_groups_df.columns)}"
            )
        return entropy_groups_df[[col_name]].rename(columns={col_name: "group"})

    elif method == "biotype":
        labels = data["labels"]
        group_df = labels[["coding_class"]].copy()
        group_df["group"] = group_df["coding_class"].map({1: "coding", 0: "lncRNA"})
        return group_df[["group"]]

    elif method == "correctness":
        binary = data["binary"]
        probs = data["probs"]

        # Ensemble vote: mean probability > 0.5 → predicted coding
        ensemble_pred = (probs.mean(axis=1) > 0.5).astype(int)
        real_class = binary["real"].astype(int)

        common_idx = ensemble_pred.index.intersection(real_class.index)
        correct = ensemble_pred.loc[common_idx] == real_class.loc[common_idx]

        return pd.DataFrame(
            {"group": correct.map({True: "correct", False: "incorrect"})},
            index=common_idx,
        )

    else:
        raise ValueError(f"Unknown grouping method: {method}")


# ────────────────────────────────────────────────────────────────────────────
# Top-features summary helpers
# ────────────────────────────────────────────────────────────────────────────


def _rank_and_extract(results_df, score_columns, prefix):
    """
    Call rank_features_by_composite with given score columns and return a
    two-column DataFrame (feature, {prefix}_rank_score).
    Returns None if required columns are absent or results are empty.
    """
    if results_df.empty:
        return None
    missing = [c for c in score_columns if c not in results_df.columns]
    if missing:
        print(f"  ⚠ Skipping composite rank for {prefix}: missing {missing}")
        return None
    ranked = rank_features_by_composite(results_df, score_columns=score_columns)
    return (
        ranked[["feature", "rank_score", "rank_percentile"]]
        .rename(
            columns={
                "rank_score": f"{prefix}_rank_score",
                "rank_percentile": f"{prefix}_rank_pct",
            }
        )
        .reset_index(drop=True)
    )


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sentinel = output_dir / "univariate_complete.flag"

    if sentinel.exists() and not args.force_rerun:
        print(
            f"Completion sentinel already exists at {sentinel}. "
            "Exiting early (pass --force-rerun to override)."
        )
        sys.exit(0)

    sep = "=" * 62
    print(f"\n{sep}")
    print(f"  Univariate Analysis: {args.dataset}")
    print(f"  Grouping method   : {args.grouping_method}")
    print(f"  Output directory  : {output_dir}")
    print(f"{sep}\n")

    # ── 1. Load main dataset ─────────────────────────────────────────────────
    print("Step 1: Loading dataset …")
    data = load_dataset(args.dataset)
    features_df = data["features"].copy()
    print(
        f"  {features_df.shape[0]} transcripts × {features_df.shape[1]} columns loaded."
    )

    # ── 2. Merge optional TE / NBD features ─────────────────────────────────
    print("\nStep 2: Merging supplementary feature files …")
    for label, path_str in [("TE", args.te_features), ("NBD", args.nbd_features)]:
        if not path_str:
            print(f"  {label}: no path provided, skipping.")
            continue
        p = Path(path_str)
        if not p.exists():
            print(f"  ⚠ {label} features file not found: {path_str}")
            continue
        sep_char = "\t" if p.suffix == ".tsv" else ","
        extra = pd.read_csv(p, sep=sep_char, index_col=0)
        features_df = features_df.join(extra, how="left", rsuffix=f"_{label.lower()}")
        print(f"  {label}: merged {extra.shape[1]} columns.")

    # ── 3. Determine feature columns ─────────────────────────────────────────
    print("\nStep 3: Filtering feature columns …")
    feature_cols = filter_feature_columns(features_df)
    if not feature_cols:
        print("  ⚠ No valid feature columns found after filtering.")

    # ── 4. Load entropy TSV and entropy groups CSV ───────────────────────────
    print("\nStep 4: Loading entropy data …")
    entropy_df = pd.read_csv(args.entropy_tsv, sep="\t", index_col=0)
    print(f"  Entropy TSV : {entropy_df.shape[0]} rows × {entropy_df.shape[1]} cols")

    entropy_groups_df = pd.read_csv(args.entropy_groups_csv, index_col=0)
    print(
        f"  Entropy groups: {entropy_groups_df.shape[0]} rows × {entropy_groups_df.shape[1]} cols"
    )

    # ── 5. Build group labels ─────────────────────────────────────────────────
    print(f"\nStep 5: Building group labels ({args.grouping_method}) …")
    group_df = _build_group_df(args, data, entropy_groups_df)
    print(f"  Group distribution: {group_df['group'].value_counts().to_dict()}")

    # ── 6. Align analysis DataFrame to common index ──────────────────────────
    print("\nStep 6: Aligning indices …")
    analysis_df = features_df[feature_cols].copy()
    common_idx = analysis_df.index.intersection(group_df.index)
    analysis_df = analysis_df.loc[common_idx]
    group_df = group_df.loc[common_idx]
    print(f"  Common transcripts: {len(common_idx)}")

    if len(common_idx) == 0:
        print("  ✗ No common transcripts — cannot proceed.")
        sys.exit(1)

    # ── 7. Residualize if requested ──────────────────────────────────────────
    if args.residualize:
        print("\nStep 7: Residualizing features against transcript length …")
        analysis_df = residualize_features(analysis_df)
        # Retain only numeric columns post-residualization
        analysis_df = analysis_df.select_dtypes(include=[np.number])
    else:
        print("\nStep 7: Skipping residualization (--no-residualize).")

    # ── 8. Statistical tests ─────────────────────────────────────────────────
    requested_tests = [t.strip() for t in args.tests.split(",") if t.strip()]
    result_dfs: dict[str, pd.DataFrame] = {}

    print(f"\nStep 8: Running tests: {requested_tests}")

    if "mann_whitney" in requested_tests:
        print("\n  ── Mann-Whitney U Tests ──")
        mw_results = perform_mann_whitney_tests(
            features_df=analysis_df,
            group_df=group_df,
            group_col="group",
        )
        if not mw_results.empty:
            out_path = output_dir / f"{args.dataset}_mann_whitney_results.csv"
            mw_results.to_csv(out_path, index=False)
            print(f"  Saved → {out_path.name}")
            result_dfs["mann_whitney"] = mw_results
        else:
            print("  ⚠ Mann-Whitney returned empty results.")

    if "f_test" in requested_tests:
        print("\n  ── F-Tests ──")
        ft_results = perform_f_tests(
            features_df=analysis_df,
            group_df=group_df,
            group_col="group",
            fdr_method=args.correction_method,
            fdr_alpha=args.alpha,
        )
        if not ft_results.empty:
            out_path = output_dir / f"{args.dataset}_f_test_results.csv"
            ft_results.to_csv(out_path, index=False)
            print(f"  Saved → {out_path.name}")
            result_dfs["f_test"] = ft_results
        else:
            print("  ⚠ F-test returned empty results.")

    if "mutual_info" in requested_tests:
        print("\n  ── Mutual Information ──")
        mi_results = perform_mutual_info_tests(
            features_df=analysis_df,
            group_df=group_df,
            group_col="group",
            fdr_method=args.correction_method,
            fdr_alpha=args.alpha,
        )
        if not mi_results.empty:
            out_path = output_dir / f"{args.dataset}_mutual_info_results.csv"
            mi_results.to_csv(out_path, index=False)
            print(f"  Saved → {out_path.name}")
            result_dfs["mutual_info"] = mi_results
        else:
            print("  ⚠ Mutual-Info returned empty results.")

    # ── 9. Composite ranking → top features summary ──────────────────────────
    print("\nStep 9: Building top-features summary …")
    summary_parts = []

    # Mann-Whitney: pick the first contrast (alphabetical) to rank on
    if "mann_whitney" in result_dfs:
        mw_df = result_dfs["mann_whitney"]
        if "contrast" in mw_df.columns:
            first_contrast = sorted(mw_df["contrast"].unique())[0]
            mw_sub = mw_df[mw_df["contrast"] == first_contrast].copy()
        else:
            mw_sub = mw_df
        part = _rank_and_extract(
            mw_sub,
            score_columns=["effect_size", "vda_deviation", "norm_diff"],
            prefix="mw",
        )
        if part is not None:
            summary_parts.append(part)

    # F-test
    if "f_test" in result_dfs:
        part = _rank_and_extract(
            result_dfs["f_test"],
            score_columns=["f_score"],
            prefix="ft",
        )
        if part is not None:
            summary_parts.append(part)

    # Mutual Information
    if "mutual_info" in result_dfs:
        part = _rank_and_extract(
            result_dfs["mutual_info"],
            score_columns=["mi_score"],
            prefix="mi",
        )
        if part is not None:
            summary_parts.append(part)

    if summary_parts:
        summary = summary_parts[0]
        for part in summary_parts[1:]:
            summary = summary.merge(part, on="feature", how="outer")

        # Sort by the first rank-score column present
        rank_score_cols = [c for c in summary.columns if c.endswith("_rank_score")]
        if rank_score_cols:
            summary = summary.sort_values(rank_score_cols[0], ascending=False)

        summary_path = output_dir / f"{args.dataset}_top_features_summary.csv"
        summary.to_csv(summary_path, index=False)
        print(f"  Saved → {summary_path.name}  ({len(summary)} features)")
    else:
        print("  ⚠ No ranked results available — summary not written.")

    # ── 10. Write sentinel ───────────────────────────────────────────────────
    sentinel.write_text("done\n")
    print(f"\n✓ Wrote completion sentinel: {sentinel}")
    print("All done.\n")


if __name__ == "__main__":
    main()
