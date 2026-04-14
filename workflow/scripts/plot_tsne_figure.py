"""plot_tsne_figure.py — Three-panel t-SNE figure.

Replaces the t-SNE cell of 000_all_figures.ipynb.

Produces:
  8. tsne_three_panels.pdf — coding class | H_pred continuous | entropy group

Usage
-----
python workflow/scripts/plot_tsne_figure.py \\
    --dataset      gencode.v47.common.cdhit.cv \\
    --results-dir  results \\
    --output-dir   results/gencode.v47.common.cdhit.cv/figures \\
    --entropy-tsv  results/gencode.v47.common.cdhit.cv/features/entropy/gencode.v47.common.cdhit.cv_uncertainty_analysis.tsv
"""

import argparse
import logging
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")
logging.getLogger("matplotlib").setLevel(logging.ERROR)
logging.getLogger("fontTools").setLevel(logging.ERROR)

# ── Local imports ──────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parents[1]))

import utils.plotting  # noqa: E402
from utils.embeddings import EmbeddingPipeline  # noqa: E402
from utils.entropy import load_dataset  # noqa: E402
from utils.features import custom_feature_scaling, filter_feature_columns  # noqa: E402


# ── CLI ────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Three-panel t-SNE figure")
    p.add_argument("--dataset", required=True, metavar="EXPT")
    p.add_argument(
        "--results-dir",
        required=True,
        metavar="DIR",
        help="Root results directory (e.g. 'results')",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        metavar="DIR",
        help="Directory to write the figure",
    )
    p.add_argument(
        "--entropy-tsv",
        required=True,
        metavar="FILE",
        help="Uncertainty analysis TSV (output of compute_entropy_metrics)",
    )
    p.add_argument(
        "--groups-tsv",
        required=True,
        metavar="FILE",
        help="Entropy groups TSV with columns seq_ID and entropy_group",
    )
    return p.parse_args()


# ── Helpers ────────────────────────────────────────────────────────────────────
def _save(fig_or_path, path: Path) -> None:
    """Save current figure as PDF + PNG, then close."""
    plt.savefig(path.with_suffix(".pdf"), dpi=300, format="pdf", bbox_inches="tight")
    plt.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path.stem}.[pdf/png]")


# ── Figure 8: t-SNE three panels ──────────────────────────────────────────────
def plot_tsne_three_panels(
    dataset: str,
    entropy_df: pd.DataFrame,
    groups_df: pd.DataFrame,
    embedding_dir: Path,
    output_dir: Path,
) -> None:
    """Three-panel t-SNE: coding class | H_pred continuous | entropy group."""
    dataset_emb = load_dataset(dataset)
    feats = filter_feature_columns(dataset_emb["features"])
    feats_df = dataset_emb["features"][feats]
    numeric_c = feats_df.select_dtypes(include=[np.number]).columns
    constant_c = feats_df[numeric_c].nunique()
    selected_df = feats_df[numeric_c].drop(columns=constant_c[constant_c <= 1].index)
    scaled_vals = custom_feature_scaling(selected_df)
    scaled_df = pd.DataFrame(
        scaled_vals, index=selected_df.index, columns=selected_df.columns
    )

    pipeline = EmbeddingPipeline(
        embedding_dir=embedding_dir, subset_id="all", resource_guard=None
    )
    tsne_emb, _ = pipeline.compute_or_load_embedding(
        scaled_df,
        scaled_df.index,
        method="tsne",
        perplexity=30,
        n_iter=1000,
        random_state=42,
        force_recompute=False,
    )

    labels_emb = dataset_emb["labels"]
    plot_df = pd.DataFrame(
        tsne_emb, index=scaled_df.index, columns=["tsne_1", "tsne_2"]
    )
    plot_df["coding_class"] = labels_emb.loc[plot_df.index, "coding_class"].map(
        {1: "coding", 0: "lncRNA"}
    )
    plot_df = plot_df.join(entropy_df[["H_pred", "I_bald"]], how="left")
    plot_df = plot_df.join(groups_df[["entropy_group"]], how="left")
    plot_df["entropy_class"] = plot_df["entropy_group"].apply(
        lambda v: (
            "low"
            if isinstance(v, str) and v.startswith("low")
            else ("high" if isinstance(v, str) and v.startswith("high") else "middle")
        )
    )
    plot_df = plot_df.sample(frac=1, random_state=42)

    DOT_SIZE, DOT_ALPHA = 0.4, 0.3
    COLORS = utils.plotting.COLORS

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(17 / 2.54, 6 / 2.54),
        sharey=True,
        constrained_layout=True,
        dpi=300,
    )

    # Panel 1: coding class
    sns.scatterplot(
        data=plot_df,
        x="tsne_1",
        y="tsne_2",
        hue="coding_class",
        palette={"coding": COLORS["pc"], "lncRNA": COLORS["lnc"]},
        s=DOT_SIZE,
        alpha=DOT_ALPHA,
        linewidth=0,
        legend=False,
        ax=axes[0],
        rasterized=True,
    )
    axes[0].legend(
        handles=[
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="coding",
                markerfacecolor=COLORS["pc"],
                markersize=5,
            ),
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="lncRNA",
                markerfacecolor=COLORS["lnc"],
                markersize=5,
            ),
        ],
        title="",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        ncol=2,
        frameon=False,
        fontsize=7,
    )

    # Panel 2: H_pred continuous
    axes[1].scatter(
        plot_df["tsne_1"],
        plot_df["tsne_2"],
        c=plot_df["H_pred"],
        cmap="viridis",
        s=DOT_SIZE,
        alpha=DOT_ALPHA,
        linewidths=0,
        rasterized=True,
    )

    # Panel 3: entropy groups
    for grp, color, alpha in [
        ("middle", "#95a5a6", DOT_ALPHA * 0.4),
        ("high", "#e74c3c", DOT_ALPHA),
        ("low", "#2ecc71", DOT_ALPHA),
    ]:
        mask = plot_df["entropy_class"] == grp
        axes[2].scatter(
            plot_df.loc[mask, "tsne_1"],
            plot_df.loc[mask, "tsne_2"],
            c=color,
            s=DOT_SIZE,
            alpha=alpha,
            linewidths=0,
            rasterized=True,
        )
    axes[2].legend(
        handles=[
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="low",
                markerfacecolor="#2ecc71",
                markersize=5,
            ),
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="high",
                markerfacecolor="#e74c3c",
                markersize=5,
            ),
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="other",
                markerfacecolor="#95a5a6",
                markersize=5,
            ),
        ],
        title="",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        ncol=3,
        frameon=False,
        fontsize=7,
    )

    for ax in axes:
        ax.set_xlabel("t-SNE 1", fontsize=7)
        ax.set_aspect("equal")
        ax.xaxis.set_major_locator(plt.MultipleLocator(50))
        ax.yaxis.set_major_locator(plt.MultipleLocator(50))
        ax.tick_params(width=0.5, length=2, labelsize=6)
        ax.xaxis.set_minor_locator(plt.MultipleLocator(25))
        ax.yaxis.set_minor_locator(plt.MultipleLocator(25))
        ax.tick_params(
            axis="both",
            which="minor",
            labelbottom=False,
            labelleft=False,
            length=1.5,
            width=0.4,
        )
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
    axes[0].set_ylabel("t-SNE 2", fontsize=7)
    fig.get_layout_engine().set(wspace=0.07)
    fig.set_dpi(300)

    _save(fig, output_dir / "tsne_three_panels")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    plt.rcParams["pdf.fonttype"] = 42
    np.random.seed(42)

    args = parse_args()
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset    : {args.dataset}")
    print(f"Output dir : {output_dir}")

    print("\n── Loading entropy TSV ──────────────────────────────────────────────")
    entropy_df = pd.read_csv(args.entropy_tsv, sep="\t", index_col=0)
    print(f"  entropy: {entropy_df.shape}")

    embedding_dir = results_dir / args.dataset / "features" / "embeddings"

    groups_df = pd.read_csv(args.groups_tsv, sep="\t", index_col=0)

    print("\n── Figure 8: t-SNE ──────────────────────────────────────────────────")
    plot_tsne_three_panels(
        args.dataset, entropy_df, groups_df, embedding_dir, output_dir
    )

    print("\n✓ t-SNE figure complete.")


if __name__ == "__main__":
    main()
