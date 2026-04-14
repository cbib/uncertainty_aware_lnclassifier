"""plot_performance_figures.py — CV performance and classification consensus figures.

Replaces the performance-related cells of 000_all_figures.ipynb.

Produces:
  1. performance_CV.pdf      — per-fold CV performance bar chart
  2. correct_classification.pdf — two-panel consensus classification breakdown

Usage
-----
python workflow/scripts/plot_performance_figures.py \\
    --dataset      gencode.v47.common.cdhit.cv \\
    --results-dir  results \\
    --output-dir   results/gencode.v47.common.cdhit.cv/figures \\
    [--n-folds 5]
"""

import argparse
import logging
import sys
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.getLogger("matplotlib").setLevel(logging.ERROR)
logging.getLogger("fontTools").setLevel(logging.ERROR)

# ── Local imports ──────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parents[1]))

from utils import plotting
from utils.parsing import load_tables  # noqa: E402
from utils.process_tools import get_classification_scores  # noqa: E402


# ── CLI ────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CV performance and classification consensus figures"
    )
    p.add_argument("--dataset", required=True, metavar="EXPT")
    p.add_argument(
        "--results-dir",
        required=True,
        metavar="DIR",
        help="Root results directory (e.g. 'results')",
    )
    p.add_argument(
        "--output-dir", required=True, metavar="DIR", help="Directory to write figures"
    )
    p.add_argument(
        "--n-folds",
        type=int,
        default=5,
        metavar="N",
        help="Number of CV folds (default: 5)",
    )
    return p.parse_args()


# ── Helpers ────────────────────────────────────────────────────────────────────
def _save(fig_or_path, path: Path) -> None:
    """Save current figure as PDF + PNG, then close."""
    plt.savefig(path.with_suffix(".pdf"), dpi=300, format="pdf", bbox_inches="tight")
    plt.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path.stem}.[pdf/png]")


# ── Data loading ───────────────────────────────────────────────────────────────
def load_fold_binary_tables(results_dir: Path, dataset: str, n_folds: int) -> dict:
    """Load per-fold binary class tables for CV performance computation."""
    cv_base = results_dir / dataset / "testing"
    cv_dfs = {}
    for fold in range(1, n_folds + 1):
        f = cv_base / f"fold{fold}" / "tables" / f"fold{fold}_binary_class_table.tsv"
        if f.exists():
            df = pd.read_csv(f, sep="\t")
            if "seq_ID" in df.columns:
                df = df.set_index("seq_ID")
            cv_dfs[f"fold{fold}"] = df
            print(f"  Loaded fold{fold}: {len(df)} transcripts")
        else:
            print(f"  WARNING: {f} not found")
    print(f"Total folds loaded: {len(cv_dfs)}")
    return cv_dfs


# ── Figure 1: CV performance ───────────────────────────────────────────────────
def plot_performance_cv(cv_dfs: dict, output_dir: Path) -> None:
    """Bar chart of aggregated CV performance metrics per tool."""
    tool_names = {
        "lncrnabert": "lncRNABERT",
        "mrnn": "mRNN",
        "feelnc": "FEELnc",
        "rnasamba": "RNAsamba",
        "ss_lncfinder": "LncFinder",
        "ss_lncDC": "LncDC",
        "plncpro": "PlncPro",
        "l_cpat": "CPAT",
    }
    cv_scores = {
        fold_name: get_classification_scores(df, average="macro")
        for fold_name, df in cv_dfs.items()
    }
    all_fold_scores = pd.concat(cv_scores, names=["fold", "tool"])
    metric_cols = ["accuracy", "balanced_accuracy", "precision", "recall", "f1_score"]
    cv_agg = all_fold_scores.groupby(level="tool")[metric_cols].agg(["mean", "std"])
    cv_agg.columns = ["_".join(c) for c in cv_agg.columns]
    cv_agg = cv_agg.sort_values("f1_score_mean", ascending=False)

    has_underscore = cv_agg.index.str.contains("_")
    cv_agg["tool_name"] = cv_agg.index.str.split("_").str[-1]
    cv_agg = (
        cv_agg.sort_values("f1_score_mean", ascending=False)
        .reset_index()
        .groupby("tool_name")
        .first()
        .sort_values("f1_score_mean", ascending=False)
        .set_index("tool")
    )
    cv_agg.index = cv_agg.index.map(lambda x: tool_names.get(x, x))

    layout = [["balanced_accuracy", "precision", "recall", "f1_score"]]
    fig_h_cm = 5
    fig_w_cm = 17
    n_tools = len(cv_agg)
    colors = plt.cm.Set3(np.linspace(0, 1, n_tools))

    fig, axes_map = plt.subplot_mosaic(
        layout,
        figsize=(fig_w_cm / 2.54, fig_h_cm / 2.54),
        dpi=300,
        layout="constrained",
    )
    flat_layout = [item for row in layout for item in row]
    for metric in flat_layout:
        ax = axes_map[metric]
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"
        x_pos = np.arange(n_tools)
        bars = ax.bar(
            x_pos,
            cv_agg[mean_col],
            yerr=cv_agg[std_col],
            capsize=3,
            alpha=1,
            color=colors,
            edgecolor="black",
            linewidth=0.5,
        )
        ax.set_xticks(x_pos)
        ax.set_xticklabels(cv_agg.index, rotation=60, ha="right", fontsize=6)
        dx = 0.07
        offset = transforms.ScaledTranslation(dx, 0, fig.dpi_scale_trans)
        for label in ax.get_xticklabels():
            label.set_transform(label.get_transform() + offset)
        ax.set_ylim([0, 1.05])
        ax.set_ylabel("Score", fontsize=7)
        ax.tick_params(axis="y", labelsize=7)
        ax.label_outer()
        ax.set_title(metric.replace("_", " ").title(), fontsize=8)
        top_idx = np.argmax(cv_agg[mean_col].to_numpy())
        top_bar = bars[top_idx]
        center_x = top_bar.get_x() + top_bar.get_width() / 2
        yerr_val = float(cv_agg[std_col].iloc[top_idx])
        top_y = top_bar.get_height() + yerr_val
        ax.annotate(
            "",
            xy=(center_x, top_y),
            xytext=(center_x, top_y + 0.06),
            arrowprops=dict(arrowstyle="->", color="red", lw=1),
            clip_on=False,
        )

    _save(fig, output_dir / "performance_CV")


# ── Figure 2: Classification consensus ────────────────────────────────────────
def plot_correct_classification(binary_raw: pd.DataFrame, output_dir: Path) -> None:
    """Two-panel bar chart: total + per-class consensus classification breakdown."""
    label_cols = [
        "rnasamba",
        "feelnc",
        "l_cpat",
        "ss_lncDC",
        "mrnn",
        "lncrnabert",
        "plncpro",
        "ss_lncfinder",
    ]

    correct = pd.DataFrame(index=binary_raw.index)
    for c in label_cols:
        if c in binary_raw.columns:
            correct[c] = binary_raw[c] == binary_raw["real"]
    correct["n_correct"] = correct.sum(axis=1)
    correct["real"] = binary_raw["real"]

    correct_by_class = (
        correct.groupby("n_correct")["real"]
        .value_counts()
        .unstack()
        .sort_index(ascending=False)
        .rename(columns={True: "coding", False: "lncRNA"})
    )
    all_or_none = pd.DataFrame(columns=correct_by_class.columns)
    all_or_none.loc["None"] = correct_by_class.loc[0]
    all_or_none.loc["All"] = correct_by_class.loc[correct_by_class.index.max()]
    ambiguous_rows = correct_by_class.loc[
        correct_by_class.index.difference([0, correct_by_class.index.max()])
    ]
    all_or_none.loc["Ambiguous"] = ambiguous_rows.sum()
    all_or_none["order"] = [2, 0, 1]
    all_or_none = all_or_none.sort_values("order").drop(columns="order")

    fig_w_cm, fig_h_cm = 4, 8.5
    colors_cls = ["#9467bd", "#d95f02"]
    viridis = plt.get_cmap("viridis", 8)
    vir_colors = [viridis(6), viridis(4), viridis(2)]

    plt.rcParams.update(
        {
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "axes.titlesize": 9,
            "axes.labelsize": 7,
            "lines.linewidth": 0.5,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "font.size": 6,
            "legend.fontsize": 6,
        }
    )

    fig, axes = plt.subplots(
        2, 1, figsize=(fig_w_cm / 2.54, fig_h_cm / 2.54), dpi=300, sharex=True
    )
    total = all_or_none.sum(axis=1) / 1000
    total.plot.bar(
        ax=axes[0],
        color=vir_colors,
        width=0.8,
        alpha=0.9,
        edgecolor="black",
        linewidth=0.5,
    )
    axes[0].set_ylabel("Number of Transcripts (×10³)")
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha="right")
    axes[0].set_ylim(0, 80)

    total_k = all_or_none.sum().sum() / 1000
    for container in axes[0].containers:
        for bar in container:
            h = bar.get_height()
            if np.isfinite(h) and h > 0:
                axes[0].annotate(
                    f"{h / total_k * 100:.1f}%",
                    (bar.get_x() + bar.get_width() / 2, h),
                    ha="center",
                    va="bottom",
                    xytext=(0, 2),
                    textcoords="offset points",
                )

    pct = all_or_none.div(all_or_none.sum(axis=0), axis=1) * 100
    pct.plot.bar(
        stacked=False,
        ax=axes[1],
        color=colors_cls,
        width=0.8,
        alpha=0.9,
        edgecolor="black",
        linewidth=0.5,
    )
    axes[1].set_ylabel("Transcripts in class (%)")
    axes[1].legend(title="")
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha="right")
    axes[1].set_ylim(0, 80)

    for ax in axes:
        ax.tick_params(width=0.5, length=2)
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)

    _save(fig, output_dir / "correct_classification")
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    plt.rcParams["pdf.fonttype"] = 42


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    np.random.seed(42)

    args = parse_args()
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset    : {args.dataset}")
    print(f"Output dir : {output_dir}")

    print("\n── Loading binary table ─────────────────────────────────────────────")
    tables = load_tables(args.dataset)
    binary_raw = tables["binary"]

    print("\n── Loading CV fold tables ────────────────────────────────────────────")
    cv_dfs = load_fold_binary_tables(results_dir, args.dataset, args.n_folds)

    print("\n── Figure 1: CV performance ─────────────────────────────────────────")
    plot_performance_cv(cv_dfs, output_dir)

    print("\n── Figure 2: Classification consensus ──────────────────────────────")
    plot_correct_classification(binary_raw, output_dir)

    print("\n✓ Performance figures complete.")


if __name__ == "__main__":
    main()
