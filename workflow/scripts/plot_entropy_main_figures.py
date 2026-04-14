"""plot_entropy_main_figures.py — Entropy scatter and statistical test figures.

Produces:
  entropy_bald_scatter.pdf
  high_entropy_pc_v_lnc_statistical_tests_results.pdf
  low_entropy_pc_v_lnc_statistical_tests_results.pdf
  low_vs_high_entropy_statistical_tests_results.pdf

Usage
-----
python workflow/scripts/plot_entropy_main_figures.py \\
    --dataset     gencode.v47.common.cdhit.cv \\
    --output-dir  results/gencode.v47.common.cdhit.cv/figures/entropy \\
    --results-dir results
"""

import argparse
import logging
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.getLogger("matplotlib").setLevel(logging.ERROR)
logging.getLogger("fontTools").setLevel(logging.ERROR)

sys.path.insert(0, str(Path(__file__).parents[1]))

import utils.plotting  # noqa: E402
from utils.entropy import load_entropy_groups, split_entropy_group_indices  # noqa: E402
from utils.entropy_figures import (  # noqa: E402
    plot_entropy_scatter,
    plot_stat_test_figure,
)

# (comparison name, grp1 label, grp2 label, xlim, annotation, entropy level pair)
COMPARISONS = [
    (
        "high_entropy_pc_v_lnc",
        "lncRNA",
        "Coding",
        (-1.3, 1.3),
        ("coding →", "← lncRNA"),
        ("high", "lnc"),
        ("high", "cod"),
    ),
    (
        "low_entropy_pc_v_lnc",
        "lncRNA",
        "Coding",
        (-1.6, 1.6),
        ("coding →", "← lncRNA"),
        ("low", "lnc"),
        ("low", "cod"),
    ),
    (
        "low_vs_high_entropy",
        "Low entropy",
        "High entropy",
        (-1.05, 0.05),
        ("high →\nentropy ", "← low\nentropy"),
        ("low", None),
        ("high", None),
    ),
]

# Supplementary comparisons — within-class low vs high entropy.
# Only produced by statistical_tests.py when class-separated groups are used.
SUPPLEMENTARY_COMPARISONS = [
    (
        "supp_coding_low_vs_high_entropy",
        "Low coding",
        "High coding",
        (-1.65, 0.05),
        ("high →\nentropy ", "← low\nentropy"),
        ("low", "cod"),
        ("high", "cod"),
    ),
    (
        "supp_lncrna_low_vs_high_entropy",
        "Low lncRNA",
        "High lncRNA",
        (-1.05, 1.05),
        ("high →\nentropy ", "← low\nentropy"),
        ("low", "lnc"),
        ("high", "lnc"),
    ),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Entropy scatter and statistical test figures"
    )
    p.add_argument("--dataset", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--results-dir", default="results")
    p.add_argument("--cluster-threshold", default="0.25", type=float)
    p.add_argument(
        "--with-supplementary",
        action="store_true",
        help="Also produce within-class low-vs-high supplementary figures "
        "(requires class-separated stat files to exist).",
    )
    return p.parse_args()


def main() -> None:
    plt.rcParams["pdf.fonttype"] = 42
    np.random.seed(42)

    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base = Path(args.results_dir) / args.dataset
    stats_dir = base / "features/statistical_analysis"

    # ── Load entropy data ─────────────────────────────────────────────────────
    entropy_df = pd.read_csv(
        base / "features/entropy" / f"{args.dataset}_uncertainty_analysis.tsv",
        sep="\t",
        index_col=0,
    )
    cluster_df = pd.read_csv(
        base / "features/clustering/feature_clusters_at_distances.csv", index_col=0
    )
    groups = load_entropy_groups(
        base / "features/entropy" / f"{args.dataset}_entropy_groups.tsv",
        transcript_index=entropy_df.index,
    )

    cluster_threshold = args.cluster_threshold
    cluster_col = f"cluster_{cluster_threshold:.2f}"
    if cluster_col not in cluster_df.columns:
        available_thresholds = [
            float(col.split("_")[1])
            for col in cluster_df.columns
            if col.startswith("cluster_")
        ]
        closest_threshold = min(
            available_thresholds,
            key=lambda value: abs(value - cluster_threshold),
        )
        cluster_col = f"cluster_{closest_threshold:.2f}"
        print(
            f"⚠ Exact threshold {cluster_threshold:.2f} not found, using closest: "
            f"{closest_threshold:.2f}",
            file=sys.stderr,
        )

    # ── Entropy scatter ───────────────────────────────────────────────────────
    aligned = groups.reindex(entropy_df.index)
    unique_groups = set(aligned.dropna().astype(str).unique())
    edf = entropy_df.copy()
    if any(g.endswith("_coding") or g.endswith("_lncRNA") for g in unique_groups):
        edf["entropy_class"] = aligned.fillna("middle")
        color_map = utils.plotting.COLORS["entropy_class_separated"]
    else:
        edf["entropy_class"] = aligned.str.extract(r"^(low|high)", expand=False).fillna(
            "other"
        )
        color_map = utils.plotting.COLORS["entropy_class"]

    plot_entropy_scatter(
        edf,
        group_col="entropy_class",
        color_map=color_map,
        save_path=output_dir / "entropy_bald_scatter",
    )

    # ── Statistical test figures ──────────────────────────────────────────────
    grp = groups.reindex(entropy_df.index)
    low_idx, high_idx = split_entropy_group_indices(grp)
    lnc_idx = entropy_df[entropy_df["coding_class"] == 0].index
    cod_idx = entropy_df[entropy_df["coding_class"] == 1].index
    idx_map = {"low": low_idx, "high": high_idx, "lnc": lnc_idx, "cod": cod_idx}

    for name, lbl1, lbl2, xlim, annot, (l1, b1), (l2, b2) in COMPARISONS:
        grp1 = idx_map[l1].intersection(idx_map[b1]) if b1 else idx_map[l1]
        grp2 = idx_map[l2].intersection(idx_map[b2]) if b2 else idx_map[l2]
        mannu_df = pd.read_csv(stats_dir / f"{name}_mannwhitney.csv", index_col=0)
        chi2_df = pd.read_csv(stats_dir / f"{name}_chi2.csv", index_col=0)
        cat_freq_df = pd.read_csv(
            stats_dir / f"{name}_cat_freq.tsv", sep="\t", index_col=0
        )
        top_cont, top_cat = plot_stat_test_figure(
            mannu_df,
            chi2_df,
            grp1_label=lbl1,
            grp2_label=lbl2,
            cat_freq_df=cat_freq_df,
            cluster_df=cluster_df,
            save_path=output_dir / f"{name}_statistical_tests_results",
            xlim_chi2=xlim,
            annotation=annot,
            cluster_col=cluster_col,
        )
        print(f"\n[{name}] continuous features (panel 1):\n{top_cont.to_string()}")
        print(f"\n[{name}] categorical features (panels 2-3):\n{top_cat.to_string()}")

    # ── Supplementary: within-class low vs high entropy ───────────────────────
    if not args.with_supplementary:
        print(
            "  Skipping supplementary comparisons (--with-supplementary not set).",
            file=sys.stderr,
        )
    else:
        for (
            name,
            lbl1,
            lbl2,
            xlim,
            annot,
            (l1, b1),
            (l2, b2),
        ) in SUPPLEMENTARY_COMPARISONS:
            grp1 = idx_map[l1].intersection(idx_map[b1]) if b1 else idx_map[l1]
            grp2 = idx_map[l2].intersection(idx_map[b2]) if b2 else idx_map[l2]
            mannu_df = pd.read_csv(stats_dir / f"{name}_mannwhitney.csv", index_col=0)
            chi2_df = pd.read_csv(stats_dir / f"{name}_chi2.csv", index_col=0)
            cat_freq_df = pd.read_csv(
                stats_dir / f"{name}_cat_freq.tsv", sep="\t", index_col=0
            )
            top_cont, top_cat = plot_stat_test_figure(
                mannu_df,
                chi2_df,
                grp1_label=lbl1,
                grp2_label=lbl2,
                cat_freq_df=cat_freq_df,
                cluster_df=cluster_df,
                save_path=output_dir / f"{name}_statistical_tests_results",
                xlim_chi2=xlim,
                annotation=annot,
                cluster_col=cluster_col,
            )
            print(f"\n[{name}] continuous features (panel 1):\n{top_cont.to_string()}")
            print(
                f"\n[{name}] categorical features (panels 2-3):\n{top_cat.to_string()}"
            )

    print("✓ Entropy main figures complete.")


if __name__ == "__main__":
    main()
