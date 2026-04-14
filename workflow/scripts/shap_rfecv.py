#!/usr/bin/env python3
"""RFECV-based feature selection using Random Forest with permutation importance ranking.

The key design: instead of using Gini-based `feature_importances_` for RFECV's
internal ranking, this script wraps RF in a thin class that recomputes
permutation importance after each fit and exposes it as `feature_importances_`.
This ensures the recursive elimination is guided by permutation importance
(model-agnostic, less biased toward high-cardinality features) rather than Gini.

Outputs
-------
  rfecv_feature_selection.csv  — per-feature rank + selected flag + cluster_rfecv
                                  column for direct use as --cluster-file in
                                  shap_train_fold.py / shap_config.yaml
  rfecv_cv_scores.csv          — per-step mean ± std cross-validation score
  permutation_importance.csv   — mean ± std permutation importance (final model)
  rfecv_cv_curve.png           — accuracy-vs-n-features curve with optimal marker
  rfecv_importance.png         — final permutation importance bar chart (top-N)
  rfecv.pkl                    — serialised RFECV object (for cache / inspection)

Compatibility with shap pipeline
---------------------------------
  shap_config.yaml:
    cluster_file: "results/.../rfecv/rfecv_feature_selection.csv"
    cluster_threshold: "rfecv"   # → selects column "cluster_rfecv"

Usage examples
--------------
  # Basic run on all transcripts, full feature set
  python shap_rfecv.py --dataset gencode.v47.common.cdhit.cv --fold 1 \\
      --te-features te_pipeline/.../all_transcripts_te_features.csv \\
      --nbd-features nonb-pipeline/.../features_nonb_features.csv \\
      --output-dir results/gencode.v47.common.cdhit.cv/rfecv

  # Start from correlation-filtered features (recommended, faster)
  python shap_rfecv.py ... --cluster-file results/.../feature_clusters_at_distances.csv

  # Subsample to N transcripts, use 1-SE rule for conservative feature count
  python shap_rfecv.py ... --max-transcripts 5000 --use-1se-rule
"""

import argparse
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.inspection import permutation_importance
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.preprocessing import PowerTransformer, StandardScaler

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parents[1]))  # paper/workflow/
from utils.entropy import load_dataset
from utils.features import filter_feature_columns, remove_constant_features
from utils.parsing import simple_load_ids

# ── RF wrapper that uses permutation importance for RFECV ranking ─────────────


class PermImportanceRF(RandomForestClassifier):
    """RandomForestClassifier where `feature_importances_` returns permutation
    importance (mean decrease in accuracy) computed on the training data.

    RFECV relies on `feature_importances_` to rank and prune features at each
    step.  Swapping Gini importance for permutation importance produces a less
    biased ranking, particularly useful when features have mixed cardinality.

    Parameters
    ----------
    perm_n_repeats : int
        Number of permutation repeats.  More repeats → more stable estimates
        but slower.  Default: 5.
    perm_random_state : int
        Seed for the permutation shuffles.
    All other kwargs forwarded to RandomForestClassifier.
    """

    def __init__(
        self, perm_n_repeats: int = 5, perm_random_state: int = 42, **rf_kwargs
    ):
        self.perm_n_repeats = perm_n_repeats
        self.perm_random_state = perm_random_state
        super().__init__(**rf_kwargs)
        self._perm_importances_cache: np.ndarray | None = None

    def fit(self, X, y, sample_weight=None):
        super().fit(X, y, sample_weight=sample_weight)
        # Recompute permutation importance on the same training data.
        # RFECV calls fit() on nested CV splits, so this is automatically
        # recomputed with the correct subset each time.
        result = permutation_importance(
            self,
            X,
            y,
            n_repeats=self.perm_n_repeats,
            random_state=self.perm_random_state,
            n_jobs=self.n_jobs,
            scoring="balanced_accuracy",
        )
        # Store as array; non-negative clip prevents RFECV from excluding
        # features that happen to have a tiny negative mean due to noise.
        self._perm_importances_cache = np.maximum(result.importances_mean, 0.0)
        return self

    @property
    def feature_importances_(self) -> np.ndarray:
        if self._perm_importances_cache is None:
            # Fallback to Gini if called before fit (e.g. parameter checks)
            return super().feature_importances_
        return self._perm_importances_cache


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(
        description="RFECV feature selection with RF + permutation importance"
    )

    # ── dataset / fold ────────────────────────────────────────────────────────
    p.add_argument("--dataset", required=True)
    p.add_argument(
        "--fold",
        type=int,
        default=1,
        help="Fold to use for feature selection (default: 1). "
        "For a stable selector, fold 1 is usually sufficient.",
    )
    p.add_argument(
        "--results-dir",
        default="results",
        help="Base results directory (default: results)",
    )

    # ── supplementary features ────────────────────────────────────────────────
    p.add_argument("--te-features", default="")
    p.add_argument("--nbd-features", default="")

    # ── feature mode & optional pre-filtering ────────────────────────────────
    p.add_argument(
        "--feature-mode",
        choices=["full", "filtered"],
        default="full",
        help="'full' = all features; 'filtered' = keep one feature per "
        "correlation cluster using --cluster-file (default: full)",
    )
    p.add_argument(
        "--cluster-file",
        default="",
        help="Correlation-cluster CSV used when --feature-mode filtered "
        "(output of 021_feature_clustering.ipynb)",
    )
    p.add_argument(
        "--cluster-threshold",
        default="0.25",
        help="Distance threshold column (float) or special key 'rfecv' "
        "→ column cluster_rfecv (default: 0.25)",
    )

    # ── transcript subsampling ────────────────────────────────────────────────
    p.add_argument(
        "--max-transcripts",
        default="none",
        help="Max transcripts to use for RFE ('none' = all). "
        "Subsampling stratified by class.",
    )

    # ── RFECV / RF parameters ─────────────────────────────────────────────────
    p.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of CV folds inside RFECV (default: 5)",
    )
    p.add_argument(
        "--cv-repeats",
        type=int,
        default=3,
        help="Number of repeated CV rounds (default: 3). "
        "Set 1 for StratifiedKFold without repeats.",
    )
    p.add_argument(
        "--scoring",
        default="balanced_accuracy",
        help="Sklearn scoring for RFECV (default: balanced_accuracy)",
    )
    p.add_argument(
        "--rfe-step",
        type=int,
        default=1,
        help="Features removed per iteration (default: 1)",
    )
    p.add_argument(
        "--perm-n-repeats",
        type=int,
        default=5,
        help="Permutation repeats for importance estimation (default: 5)",
    )
    p.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of parallel jobs for RF and permutation importance (-1 = all CPUs).",
    )
    p.add_argument(
        "--n-trees",
        type=int,
        default=200,
        help="Number of trees in each RF (default: 200)",
    )
    p.add_argument(
        "--use-1se-rule",
        action="store_true",
        help="Apply 1-SE rule to select a more parsimonious feature set",
    )
    p.add_argument(
        "--top-n-plot",
        type=int,
        default=30,
        help="Top-N features to show in importance plot (default: 30)",
    )

    # ── output / cache ────────────────────────────────────────────────────────
    p.add_argument("--output-dir", required=True)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument(
        "--force-rerun",
        action="store_true",
        help="Ignore cached rfecv.pkl and recompute",
    )

    return p.parse_args()


# ── helpers ───────────────────────────────────────────────────────────────────


def _parse_max_transcripts(val: str):
    if val in ("none", "null", "0", ""):
        return None
    return int(val)


def load_supplementary(te_path: str, nbd_path: str):
    def _load_df(path: str, label: str) -> pd.DataFrame:
        if not path:
            print(f"[rfecv] {label} features disabled")
            return pd.DataFrame()

        p = Path(path)
        if not p.exists():
            print(f"[rfecv] {label} features not found at {path} — skipping")
            return pd.DataFrame()

        raw = pd.read_csv(path)
        if "transcript_id" in raw.columns:
            df = raw.set_index("transcript_id")
        else:
            df = pd.read_csv(path, index_col=0)

        df = df.select_dtypes(include="number").fillna(0)
        if df.empty:
            return df
        return remove_constant_features(df)

    te = _load_df(te_path, "TE")
    nbd = _load_df(nbd_path, "NBD")
    return te, nbd


def build_feature_matrix(
    dataset_name,
    fold_i,
    results_dir,
    features_df,
    binary,
    te_feats,
    nbd_feats,
    feature_mode,
    cluster_file,
    cluster_threshold,
):
    """Assemble **training-only** feature matrix for one fold.

    Only training transcripts are used so that RFECV never sees test-set data.
    This ensures the feature selection step is part of the outer CV loop and
    cannot inflate downstream performance estimates.

    Parameters
    ----------
    feature_mode : str
        'full'     — all features, no correlation filtering.
        'filtered' — keep one representative per correlation cluster; requires
                     cluster_file to be set.
    cluster_threshold : str
        Float as string (e.g. '0.25') selects column 'cluster_0.25'.
        Special value 'rfecv' selects column 'cluster_rfecv'.
    """
    base = Path(results_dir) / dataset_name
    dataset_dir = base / "datasets" / f"fold{fold_i}"

    # ── training IDs only ─────────────────────────────────────────────────────
    # test_all.fa is intentionally excluded: RFECV must not see test transcripts
    all_ids: set = set()
    for fname in ("train_pc.fa", "train_lnc.fa"):
        all_ids |= set(simple_load_ids(str(dataset_dir / fname), simple=True))
    all_ids = all_ids.intersection(features_df.index)

    # ── feature table ─────────────────────────────────────────────────────────
    fold_features = features_df.loc[list(all_ids)]
    fold_binary = binary.loc[fold_features.index].copy()
    keep = filter_feature_columns(fold_features)
    fold_features = fold_features[keep]
    fold_features = remove_constant_features(fold_features)
    for col in fold_features.columns:
        fold_features[col] = pd.to_numeric(fold_features[col], errors="coerce").fillna(
            0
        )

    real = fold_binary["real"].astype(int)

    # ── join supplementary features ───────────────────────────────────────────
    all_feat = fold_features.copy()
    if not te_feats.empty:
        all_feat = all_feat.join(te_feats, how="left")
    if not nbd_feats.empty:
        all_feat = all_feat.join(nbd_feats, how="left", rsuffix="_nbd")
        all_feat.drop(
            columns=[c for c in all_feat.columns if c.endswith("_nbd")],
            inplace=True,
        )
    all_feat = all_feat.fillna(0)

    # ── optional correlation pre-filtering (only when feature_mode=filtered) ──
    if feature_mode == "filtered":
        if not cluster_file:
            raise ValueError(
                "--feature-mode filtered requires --cluster-file to be set."
            )
        cluster_df = pd.read_csv(cluster_file, index_col=0)
        col = (
            "cluster_rfecv"
            if str(cluster_threshold) == "rfecv"
            else f"cluster_{cluster_threshold}"
        )
        if col not in cluster_df.columns:
            raise ValueError(
                f"Column '{col}' not found in cluster file. "
                f"Available: {list(cluster_df.columns)}"
            )
        top_feats = cluster_df.groupby(col).head(1).index
        # Always keep binary/categorical columns regardless of clustering
        cat_substrings = ["_has_", "_present"]
        cat_col_names = ["ORF_frame_l_cpat"]
        cat_cols = [
            c
            for c in all_feat.columns
            if any(s in c for s in cat_substrings) or c in cat_col_names
        ]
        keep_cols = list(
            dict.fromkeys(cat_cols + list(top_feats))
        )  # preserve order, dedup
        keep_cols = [c for c in keep_cols if c in all_feat.columns]
        all_feat = all_feat[keep_cols]
        print(
            f"  Pre-filtered to {all_feat.shape[1]} features "
            f"(mode=filtered, threshold={cluster_threshold})"
        )
    else:
        print(f"  Using all {all_feat.shape[1]} features (mode=full)")

    all_feat = all_feat.loc[real.index]
    return all_feat, real


def stratified_subsample(X, y, n, random_state):
    """Return index arrays for a stratified subsample of size n."""
    idx = np.arange(len(y))
    classes, counts = np.unique(y, return_counts=True)
    fracs = counts / counts.sum()
    sampled = []
    rng = np.random.default_rng(random_state)
    for cls, frac in zip(classes, fracs):
        cls_idx = idx[y == cls]
        take = min(int(np.round(n * frac)), len(cls_idx))
        sampled.append(rng.choice(cls_idx, take, replace=False))
    return np.concatenate(sampled)


def apply_1se_rule(cv_results):
    """Return the smallest k where mean_score >= max_mean - 1 * se."""
    mean = cv_results["mean_test_score"]
    std = cv_results["std_test_score"]
    n_folds = cv_results.get("n_splits", 5)
    se = std / np.sqrt(n_folds)
    threshold = mean.max() - se[mean.argmax()]
    optimal_k = int(np.argmax(mean >= threshold) + 1)  # 1-indexed
    return optimal_k


# ── plots ─────────────────────────────────────────────────────────────────────


def plot_cv_curve(cv_results, optimal_k, k_1se, scoring, save_path):
    n_steps = len(cv_results["mean_test_score"])
    ks = np.arange(1, n_steps + 1)
    mean = cv_results["mean_test_score"]
    std = cv_results["std_test_score"]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ks, mean, linewidth=2, color="steelblue", label="CV mean")
    ax.fill_between(
        ks, mean - std, mean + std, alpha=0.2, color="steelblue", label="± 1 std"
    )
    ax.axvline(
        optimal_k,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"Optimal (max): {optimal_k} features",
    )
    if k_1se is not None and k_1se != optimal_k:
        ax.axvline(
            k_1se,
            color="orange",
            linestyle=":",
            linewidth=1.5,
            label=f"1-SE rule: {k_1se} features",
        )
    ax.set_xlabel("Number of features", fontsize=12)
    ax.set_ylabel(f"CV {scoring}", fontsize=12)
    ax.set_title(
        "RFECV: performance vs number of features\n"
        "(ranking by permutation importance)",
        fontsize=12,
    )
    ax.legend(fontsize=10)
    ax.grid(axis="both", linestyle="--", alpha=0.4)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_perm_importance(imp_df, top_n, save_path):
    top = imp_df.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, top_n * 0.35 + 1.5))
    colors = ["steelblue" if s else "lightgray" for s in top["selected"]]
    ax.barh(
        top["feature"],
        top["importance_mean"],
        xerr=top["importance_std"],
        capsize=3,
        color=colors,
        ecolor="#444",
        alpha=0.85,
        height=0.7,
    )
    ax.set_xlabel(
        "Permutation importance (mean decrease in balanced accuracy)", fontsize=10
    )
    ax.set_title(
        f"Top-{top_n} features by permutation importance\n"
        "(blue = RFECV-selected, gray = eliminated)",
        fontsize=11,
    )
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ── main ──────────────────────────────────────────────────────────────────────


def main():
    args = parse_args()
    np.random.seed(args.random_state)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rfecv_pkl = out_dir / "rfecv.pkl"
    sel_csv = out_dir / "rfecv_feature_selection.csv"

    # ── cache check ───────────────────────────────────────────────────────────
    if rfecv_pkl.exists() and sel_csv.exists() and not args.force_rerun:
        print(f"[rfecv] Cached results found — loading from {out_dir}")
        print(f"        Pass --force-rerun to recompute.")
        rfecv = joblib.load(rfecv_pkl)
        sel_df = pd.read_csv(sel_csv, index_col=0)
        selected = sel_df[sel_df["selected"]]["feature"].tolist()
        print(f"        {len(selected)} features selected previously.")
        return

    # ── load data ─────────────────────────────────────────────────────────────
    print(f"[rfecv] Loading dataset: {args.dataset}")
    dataset = load_dataset(args.dataset)
    features_df = dataset["features"]
    binary = dataset["binary"]

    print(f"[rfecv] Loading supplementary features…")
    te_feats, nbd_feats = load_supplementary(args.te_features, args.nbd_features)

    print(
        f"[rfecv] Building feature matrix (fold {args.fold}, "
        f"feature_mode={args.feature_mode})…"
    )
    X, y = build_feature_matrix(
        args.dataset,
        args.fold,
        args.results_dir,
        features_df,
        binary,
        te_feats,
        nbd_feats,
        args.feature_mode,
        args.cluster_file,
        args.cluster_threshold,
    )
    print(
        f"[rfecv] Matrix shape: {X.shape}  " f"(class balance: {y.mean():.2%} coding)"
    )

    # ── optional transcript subsampling ───────────────────────────────────────
    max_t = _parse_max_transcripts(args.max_transcripts)
    if max_t and len(y) > max_t:
        idx = stratified_subsample(X.values, y.values, max_t, args.random_state)
        X = X.iloc[idx]
        y = y.iloc[idx]
        print(
            f"[rfecv] Subsampled to {len(y)} transcripts "
            f"(stratified, seed={args.random_state})"
        )

    feature_cols = X.columns.tolist()
    print(
        f"[rfecv] Running RFECV on {len(feature_cols)} features, "
        f"{len(y)} transcripts…"
    )

    # ── CV strategy ───────────────────────────────────────────────────────────
    if args.cv_repeats > 1:
        cv = RepeatedStratifiedKFold(
            n_splits=args.cv_folds,
            n_repeats=args.cv_repeats,
            random_state=args.random_state,
        )
    else:
        cv = StratifiedKFold(
            n_splits=args.cv_folds,
            shuffle=True,
            random_state=args.random_state,
        )

    # ── estimator ─────────────────────────────────────────────────────────────
    estimator = PermImportanceRF(
        perm_n_repeats=args.perm_n_repeats,
        perm_random_state=args.random_state,
        n_estimators=args.n_trees,
        class_weight="balanced",
        random_state=args.random_state,
        n_jobs=args.n_jobs,
    )

    # ── RFECV ─────────────────────────────────────────────────────────────────
    rfecv = RFECV(
        estimator=estimator,
        step=args.rfe_step,
        cv=cv,
        scoring=args.scoring,
        n_jobs=1,  # outer parallelism; inner RF already uses n_jobs=-1
        verbose=1,
        min_features_to_select=1,
    )
    rfecv.fit(X.values, y.values)
    joblib.dump(rfecv, rfecv_pkl)
    print(f"[rfecv] RFECV complete. Optimal features: {rfecv.n_features_}")

    # ── 1-SE rule ─────────────────────────────────────────────────────────────
    k_1se = None
    if args.use_1se_rule:
        k_1se = apply_1se_rule(rfecv.cv_results_)
        print(
            f"[rfecv] 1-SE rule optimal k: {k_1se}  "
            f"(vs RFECV max: {rfecv.n_features_})"
        )
        final_k = k_1se
    else:
        final_k = rfecv.n_features_

    # ── build selection dataframe ─────────────────────────────────────────────
    # Re-rank by original RFECV ranking so we can apply the 1-SE cut correctly.
    # rfecv.ranking_: 1 = selected, >1 = rank of elimination (lower = better)
    ranking = rfecv.ranking_
    original_selected_mask = rfecv.support_  # True where ranking == 1

    # For 1-SE: we need to mark the top `final_k` features by ranking
    if args.use_1se_rule and final_k < rfecv.n_features_:
        # Re-select top final_k by ranking (1 is best)
        order = np.argsort(ranking)  # sorted indices by rank ascending
        final_selected_mask = np.zeros(len(feature_cols), dtype=bool)
        final_selected_mask[order[:final_k]] = True
    else:
        final_selected_mask = original_selected_mask

    sel_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "rfecv_rank": ranking,
            "selected": final_selected_mask,
            "selected_by_max": original_selected_mask,
        }
    ).sort_values("rfecv_rank")

    # ── cluster_rfecv column — each selected feature = its own cluster ────────
    # Non-selected features share a dummy cluster 0 (they will be outranked
    # by the selected features when the shap pipeline does groupby().head(1)).
    # The shap pipeline filters with:
    #   cluster_df.groupby("cluster_rfecv").head(1).index
    # so we need each SELECTED feature to be the HEAD of its own cluster.
    cluster_id = 0
    cluster_rfecv = np.zeros(len(sel_df), dtype=int)
    for i, row in sel_df.iterrows():
        if row["selected"]:
            cluster_id += 1
            cluster_rfecv[sel_df.index.get_loc(i)] = cluster_id
        # non-selected remain 0
    # Non-selected features in cluster 0 will never be head(1) because we only
    # select head(1) per non-zero cluster; re-sort so selected are first in each
    # cluster (guaranteed since each selected has its own cluster).
    sel_df["cluster_rfecv"] = cluster_rfecv
    sel_df = sel_df.set_index("feature")
    sel_df.to_csv(sel_csv)
    print(
        f"[rfecv] Selection table saved ({len(feature_cols)} features, "
        f"{final_selected_mask.sum()} selected) → {sel_csv}"
    )

    # ── CV scores ─────────────────────────────────────────────────────────────
    cv_df = pd.DataFrame(
        {
            "n_features": np.arange(1, len(rfecv.cv_results_["mean_test_score"]) + 1),
            "mean_score": rfecv.cv_results_["mean_test_score"],
            "std_score": rfecv.cv_results_["std_test_score"],
        }
    )
    cv_df.to_csv(out_dir / "rfecv_cv_scores.csv", index=False)

    # ── permutation importance on the final selected feature set ──────────────
    # Fit a fresh RF on the full X (not nested) for the final importance plot.
    print(f"[rfecv] Computing final permutation importance on selected features…")
    X_sel = X[sel_df[sel_df["selected"]].index.tolist()]
    final_rf = RandomForestClassifier(
        n_estimators=args.n_trees,
        class_weight="balanced",
        random_state=args.random_state,
        n_jobs=args.n_jobs,
    )
    final_rf.fit(X_sel.values, y.values)
    perm = permutation_importance(
        final_rf,
        X_sel.values,
        y.values,
        n_repeats=args.perm_n_repeats,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
        scoring=args.scoring,
    )
    imp_df = pd.DataFrame(
        {
            "feature": X_sel.columns,
            "importance_mean": perm.importances_mean,
            "importance_std": perm.importances_std,
            "selected": True,
        }
    ).sort_values("importance_mean", ascending=False)
    imp_df.to_csv(out_dir / "permutation_importance.csv", index=False)

    # ── plots ─────────────────────────────────────────────────────────────────
    print(f"[rfecv] Generating plots…")
    plot_cv_curve(
        rfecv.cv_results_,
        optimal_k=rfecv.n_features_,
        k_1se=k_1se,
        scoring=args.scoring,
        save_path=out_dir / "rfecv_cv_curve.png",
    )

    # Merge importance with full selection table for the bar chart
    plot_df = (
        sel_df.reset_index()
        .merge(
            imp_df[["feature", "importance_mean", "importance_std"]],
            on="feature",
            how="left",
        )
        .fillna({"importance_mean": 0, "importance_std": 0})
    )
    plot_df = plot_df.sort_values("importance_mean", ascending=False)

    plot_perm_importance(
        plot_df, args.top_n_plot, save_path=out_dir / "rfecv_importance.png"
    )

    # ── summary ───────────────────────────────────────────────────────────────
    best_score = rfecv.cv_results_["mean_test_score"].max()
    print(f"\n[rfecv] ── Summary ──────────────────────────────────────────────")
    print(f"  Input features        : {len(feature_cols)}")
    print(f"  Selected (RFECV max)  : {original_selected_mask.sum()}")
    if args.use_1se_rule:
        print(f"  Selected (1-SE rule)  : {final_selected_mask.sum()}  ← used")
    print(f"  Best CV {args.scoring:<20}: {best_score:.4f}")
    print(f"  Output dir            : {out_dir}")
    print(f"\n  To use with shap pipeline, set in shap_config.yaml:")
    print(f'    cluster_file: "{sel_csv}"')
    print(f'    cluster_threshold: "rfecv"')
    print(f"  And in shap.smk / shap_train_fold.py, --feature-mode filtered")


if __name__ == "__main__":
    main()
