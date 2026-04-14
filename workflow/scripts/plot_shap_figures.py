"""plot_shap_figures.py — All SHAP-related publication figures.

Replaces the SHAP cells of 000_all_figures.ipynb.

Produces:
  9.  shap_importance_mean_std.pdf
  10. shap_cumulative_importance_top{N}.pdf
  11. shap_fold_heatmap.pdf
  12. shap_beeswarm_all_transcripts.pdf
  13. shap_prediction_probability_distribution.pdf
  14. shap_waterfall_low_entropy_coding_EGF.pdf
  15. shap_waterfall_low_entropy_lncrna_MALAT1.pdf
  16. shap_waterfall_high_entropy_coding_SWI5.pdf
  17. shap_waterfall_high_entropy_lncrna_MEG3.pdf

Usage
-----
python workflow/scripts/plot_shap_figures.py \\
    --output-dir results/gencode.v47.common.cdhit.cv/figures \\
    --shap-dir   results/gencode.v47.common.cdhit.cv/features/shap_clustered \\
    [--n-folds 5] [--shap-top-n 20]
"""

import argparse
import logging
import sys
import warnings
from pathlib import Path
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")  # non-interactive backend — must be before pyplot import
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import shap
from matplotlib import font_manager

warnings.filterwarnings("ignore")
logging.getLogger("matplotlib").setLevel(logging.ERROR)
logging.getLogger("fontTools").setLevel(logging.ERROR)

# ── Font / style setup ────────────────────────────────────────────────────────
_msfonts_dir = Path("/mnt/cbib/LNClassifier/msfonts")
if _msfonts_dir.exists():
    for _fp in _msfonts_dir.glob("*.ttf"):
        font_manager.fontManager.addfont(str(_fp))
    matplotlib.rcParams["font.family"] = "Arial"

matplotlib.rcParams["pdf.fonttype"] = 42  # TrueType fonts in PDF
matplotlib.rcParams["lines.linewidth"] = 0.5
matplotlib.rcParams["lines.markersize"] = 5
matplotlib.rcParams["xtick.major.width"] = 0.5
matplotlib.rcParams["ytick.major.width"] = 0.5
matplotlib.rcParams["axes.linewidth"] = 0.5

# ── Local imports ──────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parents[1]))

from utils.entropy_figures import (  # noqa: E402
    FEATURE_LABEL_DICT,
    FEATURE_LABEL_DICT_SHAP,
)

FEATURE_LABEL_DICT.update(FEATURE_LABEL_DICT_SHAP)


# ── CLI ────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SHAP publication figures")
    p.add_argument(
        "--output-dir", required=True, metavar="DIR", help="Directory to write figures"
    )
    p.add_argument(
        "--shap-dir",
        required=True,
        metavar="DIR",
        help="SHAP artefacts directory (shap_aggregated.csv, fold*/ subdirs)",
    )
    p.add_argument(
        "--n-folds",
        type=int,
        default=5,
        metavar="N",
        help="Number of CV folds (default: 5)",
    )
    p.add_argument(
        "--shap-top-n",
        type=int,
        default=20,
        metavar="N",
        help="Number of top SHAP features to show (default: 20)",
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
def load_shap_data(shap_dir: Path, n_folds: int) -> dict:
    """Load aggregated SHAP CSVs and per-fold artefacts."""
    shap_agg = pd.read_csv(shap_dir / "shap_aggregated.csv", index_col=0)
    per_fold_abs = pd.read_csv(shap_dir / "shap_per_fold_mean_abs.csv", index_col=0)
    all_preds = pd.read_csv(shap_dir / "all_predictions.csv", index_col=0)

    fold_data = {}
    for fold_i in range(1, n_folds + 1):
        fold_dir = shap_dir / f"fold{fold_i}"
        required = ["shap_values.csv", "X_test.csv", "y_pred.csv", "base_val.txt"]
        if not all((fold_dir / f).exists() for f in required):
            print(f"  WARNING: fold {fold_i} missing artefacts — skipping")
            continue
        fold_data[fold_i] = {
            "shap_df": pd.read_csv(fold_dir / "shap_values.csv", index_col=0),
            "X_test": pd.read_csv(fold_dir / "X_test.csv", index_col=0),
            "y_pred": pd.read_csv(fold_dir / "y_pred.csv", index_col=0),
            "base_val": float((fold_dir / "base_val.txt").read_text().strip()),
        }
    print(f"SHAP: loaded {len(fold_data)} folds, {len(shap_agg)} features")
    return {
        "shap_agg": shap_agg,
        "per_fold_abs": per_fold_abs,
        "all_preds": all_preds,
        "fold_data": fold_data,
    }


# ── Figure 9: SHAP importance ──────────────────────────────────────────────────
def plot_shap_importance(shap_agg: pd.DataFrame, top_n: int, output_dir: Path) -> None:
    """Bar chart of mean |SHAP value| ± std across folds."""
    top = shap_agg.head(top_n).iloc[::-1].copy()
    top["label"] = top.index.map(lambda x: FEATURE_LABEL_DICT.get(x, x))

    w_cm = 14
    h_cm = top_n * 0.4 + 1.5
    fig, ax = plt.subplots(figsize=(w_cm / 2.54, h_cm / 2.54), dpi=300)
    ax.barh(
        top["label"],
        top["mean_abs_shap"],
        xerr=top["std_abs_shap"],
        color="steelblue",
        ecolor="#444",
        alpha=0.85,
        height=0.7,
        error_kw={"elinewidth": 0.5, "capthick": 0.5, "capsize": 3},
    )
    ax.set_xlabel("Mean |SHAP value| (across folds)", fontsize=7)
    ax.tick_params(axis="y", labelsize=6)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    max_x = (top["mean_abs_shap"] + top["std_abs_shap"]).max()
    xmax = np.ceil(max_x / 0.01) * 0.01
    ax.set_xlim(0, xmax)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(0.01))
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.tick_params(axis="x", labelsize=6)
    plt.tight_layout()
    _save(fig, output_dir / "shap_importance_mean_std")


# ── Figure 10: SHAP cumulative importance ─────────────────────────────────────
def plot_shap_cumulative(
    shap_agg: pd.DataFrame,
    per_fold_abs: pd.DataFrame,
    top_n: int,
    output_dir: Path,
) -> None:
    """Cumulative feature importance bar+line plot (incremental + cumulative %)."""
    cum = (
        shap_agg["mean_abs_shap"]
        .sort_values(ascending=False)
        .rename_axis("feature")
        .reset_index(name="mean_abs_shap")
    )
    cum["feature_label"] = cum["feature"].map(lambda f: FEATURE_LABEL_DICT.get(f, f))
    cum["step"] = np.arange(1, len(cum) + 1)
    cum["cum_abs_shap"] = cum["mean_abs_shap"].cumsum()
    cum["cum_pct"] = 100 * cum["cum_abs_shap"] / cum["mean_abs_shap"].sum()

    n_steps = min(top_n + 10, len(cum))
    sub = cum.head(n_steps).copy()

    ordered = [f for f in cum["feature"] if f in per_fold_abs.columns]
    fold_curves = []
    for _, row in per_fold_abs[ordered].iterrows():
        vals = row.to_numpy(dtype=float)
        total = vals.sum()
        if total > 0:
            fold_curves.append(np.cumsum(vals) / total * 100.0)
    if fold_curves:
        fold_curves = np.vstack(fold_curves)
        n_cv = fold_curves.shape[0]
        mean_c = fold_curves.mean(axis=0)
        se_c = (
            fold_curves.std(axis=0, ddof=1) / np.sqrt(n_cv)
            if n_cv > 1
            else np.zeros_like(mean_c)
        )
        best_idx = int(np.nanargmax(mean_c))
        threshold = mean_c[best_idx] - se_c[best_idx]
        one_se_step = int(np.argmax(mean_c >= threshold) + 1)
        print(
            f"  1-SE rule: {one_se_step} features ({mean_c[best_idx]:.2f}% importance)"
        )

    fig, ax1 = plt.subplots(figsize=(max(9, n_steps * 0.55), 4.8), dpi=300)
    ax1.bar(sub["step"], sub["mean_abs_shap"], color="steelblue", alpha=0.85)
    ax1.set_xlabel("Cumulative step (new feature added)", fontsize=8)
    ax1.set_ylabel("Incremental mean |SHAP|", fontsize=8)
    ax1.set_xticks(sub["step"])
    ax1.set_xticklabels(sub["feature_label"], rotation=65, ha="right", fontsize=6)
    ax1.tick_params(axis="y", labelsize=7)
    ax1.grid(axis="y", linestyle="--", alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(sub["step"], sub["cum_pct"], color="darkred", marker="o", lw=1.2)
    ax2.set_ylabel(
        "Cumulative importance (% total mean |SHAP|)", color="darkred", fontsize=8
    )
    ax2.tick_params(axis="y", labelcolor="darkred", labelsize=7)
    ax2.set_ylim(0, min(100, sub["cum_pct"].max() * 1.08))
    ax2.axvline(x=top_n, color="gray", linestyle="--", lw=0.8)
    for x, y in zip(sub["step"], sub["cum_pct"]):
        ax2.text(
            x, y, f"{y:.1f}%", fontsize=6, color="darkred", ha="center", va="bottom"
        )
    plt.tight_layout()
    _save(fig, output_dir / f"shap_cumulative_importance_top{n_steps}")


# ── Figure 11: SHAP fold heatmap ──────────────────────────────────────────────
def plot_shap_fold_heatmap(
    shap_agg: pd.DataFrame,
    per_fold_abs: pd.DataFrame,
    top_n: int,
    output_dir: Path,
) -> None:
    """Per-fold feature importance heatmap (features × folds)."""
    top_feats = shap_agg.head(top_n).index
    data = per_fold_abs[top_feats].T
    n_folds = per_fold_abs.shape[0]

    fig, ax = plt.subplots(figsize=(n_folds * 1.4 + 2, top_n * 0.4 + 1.5), dpi=300)
    im = ax.imshow(data.values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(n_folds))
    ax.set_xticklabels(per_fold_abs.index, fontsize=9)
    ax.set_yticks(range(len(top_feats)))
    ax.set_yticklabels([FEATURE_LABEL_DICT.get(f, f) for f in top_feats], fontsize=9)
    plt.colorbar(im, ax=ax, label="Mean |SHAP|")
    plt.tight_layout()
    _save(fig, output_dir / "shap_fold_heatmap")


# ── Figure 12: SHAP beeswarm ──────────────────────────────────────────────────
def plot_shap_beeswarm(
    fold_data: dict,
    shap_agg: pd.DataFrame,
    top_n: int,
    output_dir: Path,
) -> None:
    """Beeswarm plot of SHAP values across all test transcripts."""
    top_feats = shap_agg.head(top_n).index
    all_sv, all_xv = [], []
    for res in fold_data.values():
        sv = res["shap_df"].reindex(columns=top_feats, fill_value=0).values
        xv = res["X_test"].reindex(columns=top_feats, fill_value=0).values
        all_sv.append(sv)
        all_xv.append(xv)

    sv_mat = np.vstack(all_sv)
    x_mat = np.vstack(all_xv)
    expl = shap.Explanation(
        values=sv_mat,
        data=x_mat,
        feature_names=[FEATURE_LABEL_DICT.get(f, f) for f in top_feats],
    )
    fig = plt.figure(figsize=(10, 6), dpi=300)
    shap.plots.beeswarm(expl, show=False)
    plt.tight_layout()
    _save(fig, output_dir / "shap_beeswarm_all_transcripts")


# ── Figure 13: SHAP probability distribution ──────────────────────────────────
def plot_shap_prob_distribution(all_preds: pd.DataFrame, output_dir: Path) -> None:
    """Predicted probability distribution histogram by true class."""
    prob_col = "y_pred_proba_class_1"
    fig, ax = plt.subplots(figsize=(7, 4), dpi=300)
    for label, grp in all_preds.groupby("y_true"):
        clsname = "coding (1)" if label == 1 else "lncRNA (0)"
        ax.hist(grp[prob_col], bins=50, alpha=0.6, label=clsname, density=True)
    ax.set_xlabel("Predicted probability — class 1 (coding)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    _save(fig, output_dir / "shap_prediction_probability_distribution")


# ── Figures 14–17: SHAP waterfall ─────────────────────────────────────────────
_CHERRY_PICK = {
    "low_entropy_coding": ("ENST00000265171", "EGF"),
    "low_entropy_lncrna": ("ENST00000710946", "MALAT1"),
    "high_entropy_coding": ("ENST00000652598", "SWI5"),
    "high_entropy_lncrna": ("ENST00000648123", "MEG3"),
}

_GROUP_TITLES = {
    "low_entropy_coding": "Low-entropy coding",
    "low_entropy_lncrna": "Low-entropy lncRNA",
    "high_entropy_coding": "High-entropy coding",
    "high_entropy_lncrna": "High-entropy lncRNA",
}


def _build_transcript_index(fold_data: dict) -> dict:
    index = {}
    for fold_i, res in fold_data.items():
        for full_id in res["shap_df"].index:
            index[full_id.split(".")[0]] = (fold_i, full_id)
    return index


def plot_shap_waterfalls(
    fold_data: dict, shap_agg: pd.DataFrame, output_dir: Path
) -> None:
    """Waterfall SHAP plots for the four cherry-picked transcripts."""
    transcript_index = _build_transcript_index(fold_data)

    WATERFALL_MAX_DISPLAY = 11
    W_CM, H_CM = 8.0, 4.0
    TARGET_ROW_HEIGHT = 0.3
    TARGET_BAR_WIDTH = 0.6

    for group, (tid, gene_name) in _CHERRY_PICK.items():
        short = tid.split(".")[0]
        if short not in transcript_index:
            print(f"  SKIP {tid} ({gene_name}) — not in any test fold")
            continue
        fold_i, full_id = transcript_index[short]
        res = fold_data[fold_i]
        shap_s = res["shap_df"].loc[full_id]
        x_row = res["X_test"].loc[full_id]

        readable = [FEATURE_LABEL_DICT.get(f, f) for f in shap_s.index]
        expl = shap.Explanation(
            values=shap_s.values,
            base_values=res["base_val"],
            data=x_row.values,
            feature_names=readable,
        )

        _orig_arrow = plt.arrow

        # Define patched arrow function to adjust bar properties
        def _patched_arrow(*args, **kwargs):
            kwargs["width"] = TARGET_BAR_WIDTH
            kwargs["head_width"] = TARGET_BAR_WIDTH
            return _orig_arrow(*args, **kwargs)

        fig = plt.figure(figsize=(W_CM / 2.54, H_CM / 2.54), dpi=300)
        with patch.object(plt, "arrow", _patched_arrow):
            shap.plots.waterfall(expl, max_display=WATERFALL_MAX_DISPLAY, show=False)
        fig.suptitle(
            f"{gene_name} — {_GROUP_TITLES[group]} [fold {fold_i}]", fontsize=7, y=0.9
        )

        n = min(WATERFALL_MAX_DISPLAY, len(shap_s.values))
        # TODO: Make fig width configurable instead of 5
        plt.gcf().set_size_inches(5, n * TARGET_ROW_HEIGHT)
        # Walk every Text object across every axis in the figure
        for ax in fig.axes:
            # Tick labels (feature names, x-axis values)
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontsize(7)
            # Free-floating text (bar value annotations)
            for txt in ax.texts:
                txt.set_fontsize(7)
            # Axis labels
            ax.xaxis.label.set_size(7)
            ax.yaxis.label.set_size(7)
        plt.tight_layout()

        fname = f"shap_waterfall_{group}_{gene_name}"
        _save(fig, output_dir / fname)


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    np.random.seed(42)

    args = parse_args()
    shap_dir = Path(args.shap_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"SHAP dir   : {shap_dir}")
    print(f"Output dir : {output_dir}")

    print("\n── Loading SHAP artefacts ───────────────────────────────────────────")
    shap_data = load_shap_data(shap_dir, args.n_folds)

    print("\n── Figure 9: SHAP importance ─────────────────────────────────────────")
    plot_shap_importance(shap_data["shap_agg"], args.shap_top_n, output_dir)

    print("\n── Figure 10: SHAP cumulative ────────────────────────────────────────")
    plot_shap_cumulative(
        shap_data["shap_agg"], shap_data["per_fold_abs"], args.shap_top_n, output_dir
    )

    print("\n── Figure 11: SHAP fold heatmap ─────────────────────────────────────")
    plot_shap_fold_heatmap(
        shap_data["shap_agg"], shap_data["per_fold_abs"], args.shap_top_n, output_dir
    )

    print("\n── Figure 12: SHAP beeswarm ──────────────────────────────────────────")
    plot_shap_beeswarm(
        shap_data["fold_data"], shap_data["shap_agg"], args.shap_top_n, output_dir
    )

    print("\n── Figure 13: SHAP probability distribution ─────────────────────────")
    plot_shap_prob_distribution(shap_data["all_preds"], output_dir)

    print("\n── Figures 14–17: SHAP waterfall ────────────────────────────────────")
    plot_shap_waterfalls(shap_data["fold_data"], shap_data["shap_agg"], output_dir)

    print("\n✓ SHAP figures complete.")


if __name__ == "__main__":
    main()
