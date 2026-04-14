"""plot_upset_figure.py — UpSet plot of tool label combinations.

Replaces the UpSet cell of 000_all_figures.ipynb.

Produces:
  7. main_upset.pdf — UpSet plot of tool label combinations (all transcripts)

Usage
-----
python workflow/scripts/plot_upset_figure.py \\
    --dataset    gencode.v47.common.cdhit.cv \\
    --output-dir results/gencode.v47.common.cdhit.cv/figures
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

# ── Local imports ──────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parents[1]))

from utils.parsing import load_tables  # noqa: E402


# ── CLI ────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="UpSet plot of tool label combinations")
    p.add_argument("--dataset", required=True, metavar="EXPT")
    p.add_argument(
        "--output-dir",
        required=True,
        metavar="DIR",
        help="Directory to write the figure",
    )
    return p.parse_args()


# ── Helpers ────────────────────────────────────────────────────────────────────
def _save(fig_or_path, path: Path) -> None:
    """Save current figure as PDF + PNG, then close."""
    plt.savefig(path.with_suffix(".pdf"), dpi=300, format="pdf", bbox_inches="tight")
    plt.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path.stem}.[pdf/png]")


# ── Figure 7: UpSet plot ───────────────────────────────────────────────────────
def plot_upset(binary_raw: pd.DataFrame, output_dir: Path) -> None:
    """UpSet plot of tool label combinations (all transcripts)."""
    import upsetplot
    from matplotlib.lines import Line2D
    from upsetplot import from_indicators

    kept_tools = [
        "mrnn",
        "ss_lncDC",
        "rnasamba",
        "ss_lncfinder",
        "feelnc",
        "l_cpat",
        "plncpro",
        "lncrnabert",
    ]
    available = [t for t in kept_tools if t in binary_raw.columns]
    kept_real = available + ["real"]

    test_df = binary_raw.copy()
    full_upset_raw = from_indicators(kept_real, data=test_df)

    group_count = (
        full_upset_raw.groupby(level=list(range(len(full_upset_raw.index.names))))
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    cols = [c for c in group_count.columns if c not in ["real", "count"]]
    counts = [group_count[group_count[c] == True]["count"].sum() for c in cols]
    sorted_tools = (
        pd.DataFrame({"tool": cols, "count": counts})
        .sort_values("count", ascending=False)["tool"]
        .tolist()
    )
    full_upset = full_upset_raw.reorder_levels(["real"] + sorted_tools)

    up = upsetplot.UpSet(
        full_upset,
        show_percentages="{:.1%}",
        totals_plot_elements=4,
        sort_by="cardinality",
        sort_categories_by="input",
        max_subset_rank=20,
        facecolor="#414141",
        other_dots_color=0.3,
        element_size=32,
    )
    fig = plt.figure(figsize=(6, 6), dpi=300)
    axes_dict = up.plot(fig=fig)

    matrix_ax = axes_dict["matrix"]
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#414141",
            markersize=13,
            label="protein-coding",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#414141",
            markersize=11,
            alpha=0.3,
            label="lncRNA",
        ),
    ]
    matrix_ax.legend(
        handles=legend_elements,
        loc="lower center",
        fontsize=14,
        frameon=False,
        ncol=2,
        bbox_to_anchor=(0.5, -0.15),
    )
    axes_dict["totals"].set_title(
        "Number of transcripts\nlabelled 'coding'", fontsize=14, linespacing=1.5
    )
    for text_obj in axes_dict["intersections"].texts:
        text_obj.set_rotation(45)
        text_obj.set_rotation_mode("anchor")
        text_obj.set_ha("left")

    _save(fig, output_dir / "main_upset")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.weight"] = "normal"
    plt.rcParams["axes.labelweight"] = "normal"
    np.random.seed(42)

    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset    : {args.dataset}")
    print(f"Output dir : {output_dir}")

    print("\n── Loading binary table ─────────────────────────────────────────────")
    tables = load_tables(args.dataset)
    binary_raw = tables["binary"]

    print("\n── Figure 7: UpSet ──────────────────────────────────────────────────")
    plot_upset(binary_raw, output_dir)

    print("\n✓ UpSet figure complete.")


if __name__ == "__main__":
    main()
