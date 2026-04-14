#!/usr/bin/env python3
"""Aggregate per-fold RFECV selections into a consensus feature set.

Reads rfecv_feature_selection.csv from each fold under
  <input-dir>/fold{1..n_folds}/rfecv_feature_selection.csv

counts how many folds selected each feature, and applies a majority-vote
threshold (--min-folds, default 3/5).  The output consensus_features.json
is a plain JSON list of feature names and can be passed directly to
shap_train_fold.py via --selected-features.

Outputs
-------
  consensus_features.json  — list[str] of consensus-selected feature names
  consensus_summary.csv    — per-feature vote_count, mean_rank, selected flag
  consensus_votes.png      — horizontal bar chart sorted by vote count

Usage
-----
  python shap_rfecv_consensus.py \\
      --input-dir  results/gencode.v47.common.cdhit.cv/rfecv \\
      --output-dir results/gencode.v47.common.cdhit.cv/rfecv \\
      --n-folds 5 --min-folds 3
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(
        description="Consensus RFECV feature selection across folds"
    )
    p.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing fold{i}/ subdirs with "
        "rfecv_feature_selection.csv files",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        help="Where to write consensus outputs (may equal --input-dir)",
    )
    p.add_argument(
        "--n-folds",
        type=int,
        required=True,
        help="Number of folds (used to enumerate fold1..foldN)",
    )
    p.add_argument(
        "--min-folds",
        type=int,
        default=3,
        help="Minimum folds a feature must be selected in to be kept "
        "(default: 3, i.e. strict majority of 5)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── load per-fold selection tables ────────────────────────────────────────
    fold_dfs: dict[int, pd.DataFrame] = {}
    for fold in range(1, args.n_folds + 1):
        csv = in_dir / f"fold{fold}" / "rfecv_feature_selection.csv"
        if not csv.exists():
            raise FileNotFoundError(
                f"Missing RFECV selection file: {csv}\n"
                f"Run rfecv_select for all {args.n_folds} folds first."
            )
        fold_dfs[fold] = pd.read_csv(csv, index_col=0)
        n_sel = fold_dfs[fold]["selected"].sum()
        print(f"  fold {fold}: {n_sel} features selected")

    # ── union of all features ever seen ──────────────────────────────────────
    all_features = sorted(set().union(*(df.index for df in fold_dfs.values())))
    print(f"\n[consensus] {len(all_features)} unique features across all folds")

    # ── vote counting ─────────────────────────────────────────────────────────
    records = []
    for feat in all_features:
        votes, ranks = [], []
        for fold, df in fold_dfs.items():
            if feat in df.index:
                votes.append(bool(df.loc[feat, "selected"]))
                ranks.append(int(df.loc[feat, "rfecv_rank"]))
            else:
                # feature absent from this fold's table → not selected there
                votes.append(False)
                ranks.append(999)
        vote_count = sum(votes)
        records.append(
            {
                "feature": feat,
                "vote_count": vote_count,
                "selected": vote_count >= args.min_folds,
                "rfecv_rank_mean": float(np.mean(ranks)),
                "rfecv_rank_std": float(np.std(ranks)),
                "selected_in_folds": ",".join(
                    str(f) for f, v in zip(range(1, args.n_folds + 1), votes) if v
                ),
            }
        )

    summary_df = (
        pd.DataFrame(records)
        .sort_values(["vote_count", "rfecv_rank_mean"], ascending=[False, True])
        .reset_index(drop=True)
    )
    summary_df.to_csv(out_dir / "consensus_summary.csv", index=False)

    # ── write JSON ────────────────────────────────────────────────────────────
    selected_features = summary_df[summary_df["selected"]]["feature"].tolist()
    (out_dir / "consensus_features.json").write_text(
        json.dumps(selected_features, indent=2)
    )

    # ── plot ──────────────────────────────────────────────────────────────────
    top_n = min(60, len(summary_df))
    top = summary_df.head(top_n).iloc[::-1]
    colors = ["steelblue" if s else "lightgray" for s in top["selected"]]

    fig, ax = plt.subplots(figsize=(8, top_n * 0.30 + 2.5))
    ax.barh(top["feature"], top["vote_count"], color=colors, height=0.72, alpha=0.9)
    ax.axvline(
        args.min_folds,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"Threshold: {args.min_folds}/{args.n_folds} folds",
    )
    ax.set_xlabel("Number of folds where feature was selected", fontsize=11)
    ax.set_title(
        f"RFECV consensus: vote counts per feature\n"
        f"(blue = consensus-selected [{len(selected_features)} features], "
        f"top {top_n} shown)",
        fontsize=11,
    )
    ax.set_xlim(0, args.n_folds + 0.5)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.legend(fontsize=10)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()
    fig.savefig(out_dir / "consensus_votes.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── summary ───────────────────────────────────────────────────────────────
    print(f"\n[consensus] ── Results ──────────────────────────────────────────")
    print(f"  Folds processed          : {args.n_folds}")
    print(f"  Unique features evaluated: {len(all_features)}")
    print(f"  Consensus threshold      : >= {args.min_folds}/{args.n_folds} folds")
    print(f"  Features selected        : {len(selected_features)}")
    print()
    for n in range(args.n_folds, 0, -1):
        c = int((summary_df["vote_count"] == n).sum())
        marker = "  <-- consensus selected" if n >= args.min_folds else ""
        print(f"    selected in {n}/{args.n_folds} folds: {c:4d} features{marker}")
    print(f"\n  Outputs written to: {out_dir}")
    print(
        f"  → consensus_features.json  (pass to shap_train_fold.py --selected-features)"
    )


if __name__ == "__main__":
    main()
