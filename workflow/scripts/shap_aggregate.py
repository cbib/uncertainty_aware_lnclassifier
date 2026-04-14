#!/usr/bin/env python3
"""Aggregate per-fold SHAP results and generate global plots + cherry-pick analysis.

Reads all fold artefacts produced by shap_train_fold.py and writes:
  shap_aggregated.csv           mean/std/CV of |SHAP| per feature across folds
  shap_per_fold_mean_abs.csv    per-fold mean |SHAP| matrix (folds × features)
  shap_all_transcripts.csv      full SHAP table with fold + true_label columns
  all_predictions.csv           combined y_pred from all folds (+ probabilities)
  performance_summary.csv       per-fold + aggregate macro-avg classification metrics
  cherry_pick_summary.csv       top-N SHAP features for cherry-picked transcripts
  shap_importance_mean_std.png  feature importance bar chart
  shap_fold_heatmap.png         per-fold|SHAP| heatmap
  beeswarm_all_transcripts.png  beeswarm over all evaluated transcripts
  waterfall_plots/              waterfall per cherry-picked transcript
  beeswarm_plots/               beeswarm per cherry-pick group
"""

import argparse
import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend — must be before pyplot import
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import shap
from matplotlib import font_manager

warnings.filterwarnings("ignore")

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


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Aggregate cross-fold SHAP results")
    p.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing fold1/, fold2/, … subdirectories",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        help="Directory for aggregated outputs (may be same as input-dir)",
    )
    p.add_argument("--n-folds", type=int, required=True)
    p.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of top features to show in plots (default: 20)",
    )
    p.add_argument("--dataset", default="", help="Dataset label for plot titles")
    p.add_argument(
        "--cherry-picks",
        default="",
        help="Path to JSON file with cherry-pick transcript groups "
        "(dict of group_name → [transcript_ids])",
    )
    return p.parse_args()


# ── data loading helpers ──────────────────────────────────────────────────────
def load_fold_artefacts(input_dir: Path, n_folds: int):
    """Load shap_values.csv, X_test.csv, y_pred.csv, base_val.txt for all folds.

    Returns a dict: fold_i (1-based) → dict of dataframes and scalars.
    Skips folds whose artefacts are missing (with a warning).
    """
    fold_data = {}
    for fold_i in range(1, n_folds + 1):
        fold_dir = input_dir / f"fold{fold_i}"
        required = ["shap_values.csv", "X_test.csv", "y_pred.csv", "base_val.txt"]
        if not all((fold_dir / f).exists() for f in required):
            print(
                f"[aggregate] WARNING: fold {fold_i} missing outputs — "
                f"skipping (expected in {fold_dir})"
            )
            continue
        shap_df = pd.read_csv(fold_dir / "shap_values.csv", index_col=0)
        X_test = pd.read_csv(fold_dir / "X_test.csv", index_col=0)
        y_pred = pd.read_csv(fold_dir / "y_pred.csv", index_col=0)
        base_val = float((fold_dir / "base_val.txt").read_text().strip())
        fold_data[fold_i] = dict(
            shap_df=shap_df,
            X_test=X_test,
            y_pred=y_pred,
            base_val=base_val,
        )
    return fold_data


def load_classification_reports(input_dir: Path, n_folds: int):
    reports = []
    for fold_i in range(1, n_folds + 1):
        path = input_dir / f"fold{fold_i}" / "classification_report.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        # The CSV stores a dict-of-dicts; macro avg is stored as a serialised dict
        row = {"fold": fold_i}
        try:
            macro = df["macro avg"].apply(lambda x: eval(x))[0]
            row.update(macro)
        except Exception:
            pass
        try:
            row["accuracy"] = float(df["accuracy"][0])
        except Exception:
            pass
        reports.append(row)
    return pd.DataFrame(reports).set_index("fold") if reports else pd.DataFrame()


# ── aggregation ───────────────────────────────────────────────────────────────
def aggregate_shap(fold_data: dict):
    """Return (shap_agg, per_fold_mean_abs, combined_shap, combined_preds)."""
    # ── per-fold mean |SHAP| ─────────────────────────────────────────────────
    per_fold_series = {}
    for fold_i, res in fold_data.items():
        per_fold_series[f"fold{fold_i}"] = res["shap_df"].abs().mean()

    per_fold_mean_abs = pd.DataFrame(per_fold_series).T.fillna(0)

    shap_agg = pd.DataFrame(
        {
            "mean_abs_shap": per_fold_mean_abs.mean(),
            "std_abs_shap": per_fold_mean_abs.std(),
            "cv_abs_shap": per_fold_mean_abs.std() / (per_fold_mean_abs.mean() + 1e-12),
        }
    ).sort_values("mean_abs_shap", ascending=False)

    # ── combined SHAP table ───────────────────────────────────────────────────
    parts = []
    for fold_i, res in fold_data.items():
        df = res["shap_df"].copy()
        df.insert(0, "fold", fold_i)
        df.insert(1, "true_label", res["y_pred"]["y_true"])
        parts.append(df)
    combined_shap = pd.concat(parts).sort_index()

    # ── combined predictions ──────────────────────────────────────────────────
    pred_parts = []
    for fold_i, res in fold_data.items():
        pf = res["y_pred"].copy()
        pf.insert(0, "fold", fold_i)
        pred_parts.append(pf)
    combined_preds = pd.concat(pred_parts).sort_index()

    return shap_agg, per_fold_mean_abs, combined_shap, combined_preds


# ── plots ─────────────────────────────────────────────────────────────────────
def _save_fig(fig, save_path: Path, dpi: int = 300):
    """Save figure as both PNG and PDF."""
    for ext in ("png", "pdf"):
        p = Path(save_path).with_suffix(f".{ext}")
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        print(f"  Saved: {p}")


def plot_importance(shap_agg, top_n, save_path):
    top = shap_agg.head(top_n).iloc[::-1].copy()
    w_cm, h_cm = 14, top_n * 0.4 + 1.5
    fig, ax = plt.subplots(figsize=(w_cm / 2.54, h_cm / 2.54))
    ax.barh(
        top.index,
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
    ax.tick_params(axis="x", labelsize=6)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    # Auto-scale x-axis: ceil to nearest 0.01 step
    max_x = (top["mean_abs_shap"] + top["std_abs_shap"]).max()
    xmax = np.ceil(max_x / 0.01) * 0.01
    ax.set_xlim(0, xmax)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(0.01))
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    plt.tight_layout()
    _save_fig(fig, save_path)
    plt.close(fig)


def plot_fold_heatmap(per_fold_mean_abs, shap_agg, top_n, save_path):
    top_feats = shap_agg.head(top_n).index
    data = per_fold_mean_abs[top_feats].T  # features × folds
    n_folds = per_fold_mean_abs.shape[0]
    fig, ax = plt.subplots(figsize=(n_folds * 1.4 + 2, top_n * 0.4 + 1.5))
    im = ax.imshow(data.values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(n_folds))
    ax.set_xticklabels(per_fold_mean_abs.index, fontsize=9)
    ax.set_yticks(range(len(top_feats)))
    ax.set_yticklabels(top_feats, fontsize=9)
    plt.colorbar(im, ax=ax, label="Mean |SHAP|")
    ax.set_title(f"Per-fold mean |SHAP| — top {top_n} features", fontsize=9)
    plt.tight_layout()
    _save_fig(fig, save_path)
    plt.close(fig)


def plot_beeswarm_all(fold_data, top_feats, save_path):
    all_sv, all_xv = [], []
    for res in fold_data.values():
        sv = res["shap_df"].reindex(columns=top_feats, fill_value=0).values
        xv = res["X_test"].reindex(columns=top_feats, fill_value=0).values
        all_sv.append(sv)
        all_xv.append(xv)
    sv_mat = np.vstack(all_sv)
    x_mat = np.vstack(all_xv)
    expl = shap.Explanation(values=sv_mat, data=x_mat, feature_names=list(top_feats))
    fig = plt.figure(figsize=(10, 6))
    shap.plots.beeswarm(expl, show=False)
    n = sv_mat.shape[0]
    plt.title(f"All evaluated transcripts (n={n})", fontsize=9)
    _save_fig(fig, save_path)
    plt.close(fig)


# ── cherry-pick helpers ───────────────────────────────────────────────────────
def build_transcript_index(fold_data):
    """Return dict: short_id (no version) → (fold_i, full_id)."""
    index = {}
    for fold_i, res in fold_data.items():
        for full_id in res["shap_df"].index:
            short = full_id.split(".")[0]
            index[short] = (fold_i, full_id)
    return index


def get_shap(tid, transcript_index, fold_data):
    short = tid.split(".")[0]
    if short not in transcript_index:
        return None, None, None
    fold_i, full_id = transcript_index[short]
    return fold_data[fold_i]["shap_df"].loc[full_id], fold_i, full_id


def cherry_summary_df(cherry_pick, transcript_index, fold_data, top_n=5):
    rows = []
    for group, tids in cherry_pick.items():
        for tid in tids:
            shap_s, fold_i, full_id = get_shap(tid, transcript_index, fold_data)
            if shap_s is None:
                print(f"  WARNING: {tid} not found in any test fold — skipped")
                continue
            top = shap_s.abs().nlargest(top_n)
            row = {
                "group": group,
                "transcript": tid,
                "full_id": full_id,
                "fold": fold_i,
            }
            for k, feat in enumerate(top.index):
                row[f"top{k+1}_feat"] = feat
                row[f"top{k+1}_shap"] = float(shap_s[feat])
            rows.append(row)
    return pd.DataFrame(rows)


_WATERFALL_W_CM: float = 8.0
_WATERFALL_TARGET_ROW_HEIGHT: float = 0.3  # inches per displayed feature row
_WATERFALL_MAX_DISPLAY: int = 11
_WATERFALL_BAR_WIDTH: float = 0.6  # patch for SHAP arrow width (default 0.8)
_WATERFALL_FONT_PT: float = 7.0


def waterfall_plots(cherry_pick, transcript_index, fold_data, save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    all_tids = [tid for lst in cherry_pick.values() for tid in lst]
    for tid in all_tids:
        shap_s, fold_i, full_id = get_shap(tid, transcript_index, fold_data)
        if shap_s is None:
            continue
        res = fold_data[fold_i]
        x_row = res["X_test"].loc[full_id]
        expl = shap.Explanation(
            values=shap_s.values,
            base_values=res["base_val"],
            data=x_row.values,
            feature_names=shap_s.index.tolist(),
        )
        # Patch plt.arrow so SHAP uses thinner bars
        _orig_arrow = plt.arrow

        def _patched_arrow(*args, **kwargs):
            kwargs["width"] = _WATERFALL_BAR_WIDTH
            kwargs["head_width"] = _WATERFALL_BAR_WIDTH
            return _orig_arrow(*args, **kwargs)

        plt.arrow = _patched_arrow
        n_rows = min(_WATERFALL_MAX_DISPLAY, len(shap_s.values))
        fig = plt.figure(
            figsize=(_WATERFALL_W_CM / 2.54, n_rows * _WATERFALL_TARGET_ROW_HEIGHT),
            dpi=300,
        )
        shap.plots.waterfall(expl, max_display=_WATERFALL_MAX_DISPLAY, show=False)
        plt.arrow = _orig_arrow  # restore immediately
        fig.suptitle(f"SHAP waterfall — {tid} (fold {fold_i})", fontsize=7, y=0.9)
        # Unify all text to 7 pt
        for ax in fig.axes:
            for lbl in ax.get_xticklabels() + ax.get_yticklabels():
                lbl.set_fontsize(_WATERFALL_FONT_PT)
            for txt in ax.texts:
                txt.set_fontsize(_WATERFALL_FONT_PT)
            ax.xaxis.label.set_size(_WATERFALL_FONT_PT)
            ax.yaxis.label.set_size(_WATERFALL_FONT_PT)
        plt.tight_layout()
        path = save_dir / f"waterfall_{tid}.png"
        _save_fig(fig, path)
        plt.close(fig)


def beeswarm_plots(cherry_pick, transcript_index, fold_data, top_feats, save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    for group, tids in cherry_pick.items():
        sv_rows, x_rows = [], []
        for tid in tids:
            shap_s, fold_i, full_id = get_shap(tid, transcript_index, fold_data)
            if shap_s is None:
                continue
            sv_rows.append(shap_s.reindex(top_feats, fill_value=0).values)
            x_rows.append(
                fold_data[fold_i]["X_test"]
                .loc[full_id]
                .reindex(top_feats, fill_value=0)
                .values
            )
        if not sv_rows:
            continue
        expl = shap.Explanation(
            values=np.array(sv_rows),
            data=np.array(x_rows),
            feature_names=list(top_feats),
        )
        fig = plt.figure(figsize=(10, 5))
        shap.plots.beeswarm(expl, show=False)
        plt.title(f"{group} (n={len(sv_rows)})", fontsize=9)
        path = save_dir / f"beeswarm_{group}.png"
        _save_fig(fig, path)
        plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[aggregate] Loading fold artefacts from {in_dir}  (n_folds={args.n_folds})")
    fold_data = load_fold_artefacts(in_dir, args.n_folds)
    if not fold_data:
        raise RuntimeError("[aggregate] No fold data loaded — check input directory")
    print(f"[aggregate] Loaded {len(fold_data)} folds: {sorted(fold_data.keys())}")

    # ── SHAP aggregation ──────────────────────────────────────────────────────
    print("[aggregate] Aggregating SHAP values…")
    shap_agg, per_fold_mean_abs, combined_shap, combined_preds = aggregate_shap(
        fold_data
    )
    shap_agg.to_csv(out_dir / "shap_aggregated.csv")
    per_fold_mean_abs.to_csv(out_dir / "shap_per_fold_mean_abs.csv")
    combined_shap.to_csv(out_dir / "shap_all_transcripts.csv")
    combined_preds.to_csv(out_dir / "all_predictions.csv")
    print(
        f"[aggregate] Aggregated {len(shap_agg)} features across {len(fold_data)} folds"
    )

    # ── performance summary ───────────────────────────────────────────────────
    print("[aggregate] Building performance summary…")
    perf_df = load_classification_reports(in_dir, args.n_folds)
    if not perf_df.empty:
        mean_row = perf_df.mean().rename("mean")
        std_row = perf_df.std().rename("std")
        perf_df = pd.concat([perf_df, mean_row.to_frame().T, std_row.to_frame().T])
        perf_df.index = list(perf_df.index[:-2]) + ["mean", "std"]
        perf_df.to_csv(out_dir / "performance_summary.csv")
        print("[aggregate] Performance summary:")
        print(perf_df.to_string())

    # ── plots ─────────────────────────────────────────────────────────────────
    top_n = args.top_n
    top_feats = shap_agg.head(top_n).index

    print("[aggregate] Generating plots…")
    plot_importance(
        shap_agg,
        top_n,
        save_path=out_dir / "shap_importance_mean_std.png",
    )
    plot_fold_heatmap(
        per_fold_mean_abs,
        shap_agg,
        top_n,
        save_path=out_dir / "shap_fold_heatmap.png",
    )
    plot_beeswarm_all(
        fold_data,
        top_feats,
        save_path=out_dir / "beeswarm_all_transcripts.png",
    )

    # ── cherry picks ──────────────────────────────────────────────────────────
    cherry_pick = {}
    if args.cherry_picks and Path(args.cherry_picks).exists():
        with open(args.cherry_picks) as fh:
            cherry_pick = {
                k: v for k, v in json.load(fh).items() if isinstance(v, list)
            }
        print(
            f"[aggregate] Loaded cherry picks: "
            f"{sum(len(v) for v in cherry_pick.values())} transcripts across "
            f"{len(cherry_pick)} groups"
        )
    elif args.cherry_picks:
        print(f"[aggregate] WARNING: cherry-picks file not found: {args.cherry_picks}")

    if cherry_pick:
        transcript_index = build_transcript_index(fold_data)
        print("[aggregate] Building cherry-pick summary…")
        cs = cherry_summary_df(cherry_pick, transcript_index, fold_data, top_n=5)
        cs.to_csv(out_dir / "cherry_pick_summary.csv", index=False)
        print("[aggregate] Generating waterfall plots…")
        waterfall_plots(
            cherry_pick,
            transcript_index,
            fold_data,
            save_dir=out_dir / "waterfall_plots",
        )
        print("[aggregate] Generating beeswarm plots by group…")
        beeswarm_plots(
            cherry_pick,
            transcript_index,
            fold_data,
            top_feats=top_feats,
            save_dir=out_dir / "beeswarm_plots",
        )
    else:
        # Write an empty placeholder so Snakemake output is always satisfied
        (out_dir / "cherry_pick_summary.csv").write_text(
            "group,transcript,full_id,fold\n"
        )

    # ── directory listing ─────────────────────────────────────────────────────
    print(f"\n[aggregate] Done. Output directory: {out_dir}")
    for p in sorted(out_dir.rglob("*"))[:40]:
        indent = "  " * (len(p.relative_to(out_dir).parts) - 1)
        print(f"  {indent}{p.name}")


if __name__ == "__main__":
    main()
