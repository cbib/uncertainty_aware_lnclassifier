#!/usr/bin/env python3
"""Per-fold RF training + SHAP computation for the lncRNA classifier SHAP pipeline.

Outputs saved to OUTPUT_DIR/:
  rf_model.joblib             trained RandomForest
  X_test.csv                  test-set features used for SHAP
  y_test.csv                  true labels
  y_pred.csv                  predictions + class probabilities
  classification_report.csv   sklearn classification_report as dict
  shap_values.csv             SHAP values (n_test × n_features), class-1
  base_val.txt                SHAP expected value (float, class-1)

Cache behaviour
  If all outputs exist and --force-rerun is NOT set, the script exits early.
  Intermediate re-use: if rf_model.joblib exists and --force-rerun is not set,
  the RF is loaded from disk instead of retrained.
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

warnings.filterwarnings("ignore")

# ── import project utils ─────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parents[1]))  # paper/workflow/
from utils.entropy import load_dataset
from utils.features import filter_feature_columns, remove_constant_features
from utils.parsing import simple_load_ids


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Per-fold RF training + SHAP for lncRNA classifier"
    )
    p.add_argument(
        "--dataset",
        required=True,
        help="Experiment/dataset name (e.g. gencode.v47.common.cdhit.cv)",
    )
    p.add_argument("--fold", type=int, required=True, help="Fold number (1-based)")
    p.add_argument(
        "--results-dir",
        default="results",
        help="Base results directory (default: results)",
    )
    p.add_argument(
        "--output-dir", required=True, help="Output directory for this fold's artefacts"
    )

    # Feature-set mode
    p.add_argument(
        "--feature-mode",
        choices=["full", "filtered"],
        default="full",
        help="'full' = all features; 'filtered' = correlation-filtered subset",
    )
    p.add_argument(
        "--cluster-file",
        default="",
        help="Path to feature_clusters_at_distances.csv (required for filtered mode)",
    )
    p.add_argument(
        "--cluster-threshold",
        type=float,
        default=0.25,
        help="Correlation threshold column to use (default: 0.25)",
    )
    p.add_argument(
        "--selected-features",
        default="",
        help="Path to JSON file containing a list of feature names produced by "
        "shap_rfecv_consensus.py.  When set, takes priority over "
        "--cluster-file and implies --feature-mode filtered.",
    )

    # Supplementary feature files
    p.add_argument(
        "--te-features",
        default="",
        help="Path to TE features CSV (optional)",
    )
    p.add_argument(
        "--nbd-features",
        default="",
        help="Path to non-B-DNA features CSV (optional)",
    )

    # SHAP / RF parameters
    p.add_argument(
        "--max-transcripts",
        default="none",
        help="Max test transcripts for SHAP ('none' = all, or integer)",
    )
    p.add_argument(
        "--background-sample",
        type=int,
        default=500,
        help="SHAP background size (default: 500)",
    )
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument(
        "--skip-shap",
        action="store_true",
        help="Train RF and predict only; do not compute SHAP values",
    )

    # Cache control
    p.add_argument(
        "--force-rerun", action="store_true", help="Overwrite all existing outputs"
    )

    return p.parse_args()


# ── helpers ───────────────────────────────────────────────────────────────────
def _max_transcripts(val: str):
    """Parse --max-transcripts value; return None for 'none'/'0'/'null'."""
    if val in ("none", "null", "0", ""):
        return None
    return int(val)


def load_supplementary_features(te_path: str, nbd_path: str):
    def _load_df(path: str, label: str) -> pd.DataFrame:
        if not path:
            print(f"[supplementary] {label}: disabled")
            return pd.DataFrame()

        p = Path(path)
        if not p.exists():
            print(f"[supplementary] {label}: not found at {path} — skipping")
            return pd.DataFrame()

        raw = pd.read_csv(path)
        if "transcript_id" in raw.columns:
            df = raw.set_index("transcript_id")
        else:
            # Fallback for files already indexed by transcript ID.
            df = pd.read_csv(path, index_col=0)

        # Convert boolean/object True-False columns to int (0/1)
        # leaving non-boolean columns unchanged
        bool_like = df.select_dtypes(include=["object", "bool"])
        if not bool_like.empty:
            bool_map = {True: 1, False: 0, "True": 1, "False": 0}
            df[bool_like.columns] = bool_like.apply(
                lambda s: pd.to_numeric(s.map(bool_map), errors="ignore")
            )

        # Keep only numeric, fill NaNs with 0, and remove constant features
        df = df.select_dtypes(include="number")
        df = df.fillna(0)
        if df.empty:
            return df
        df = remove_constant_features(df)

        # We also remove transcript_length, as it is already encoded by RNA_size_feelnc
        df = df.drop(columns=["transcript_length"], errors="ignore")
        return df

    te = _load_df(te_path, "TE")
    nbd = _load_df(nbd_path, "NBD")
    return te, nbd


def build_fold_features(
    fold_i,
    dataset_name,
    results_dir,
    features_df,
    binary,
    te_feats,
    nbd_feats,
    feature_mode,
    top_feats,
):
    """Load and assemble X_train / X_test / y_train / y_test for one fold."""
    base = Path(results_dir) / dataset_name
    fold_name = f"fold{fold_i}"
    dataset_dir = base / "datasets" / fold_name

    # ── transcript IDs from FASTA files ──────────────────────────────────────
    print(f"[fold {fold_i}] Loading transcript IDs from FASTA files…")
    train_ids = set()
    for fname in ("train_pc.fa", "train_lnc.fa"):
        ids = set(simple_load_ids(str(dataset_dir / fname), simple=True))
        print(f"[fold {fold_i}]   {fname}: {len(ids)} IDs")
        train_ids |= ids
    test_ids = set(simple_load_ids(str(dataset_dir / "test_all.fa"), simple=True))
    print(f"[fold {fold_i}]   test_all.fa: {len(test_ids)} IDs")

    train_ids = train_ids.intersection(features_df.index)
    test_ids = test_ids.intersection(features_df.index)
    print(
        f"[fold {fold_i}] After intersecting with feature_df: "
        f"train={len(train_ids)}, test={len(test_ids)}"
    )

    # ── feature table from models ───────────────────────────────────────────────
    print(f"[fold {fold_i}] Building feature matrix…")
    fold_features = features_df.loc[list(train_ids.union(test_ids))]
    print(f"[fold {fold_i}]   Initial shape: {fold_features.shape}")

    fold_binary = binary.loc[fold_features.index].copy()

    keep_cols = filter_feature_columns(fold_features)
    print(f"[fold {fold_i}]   filter_feature_columns: {len(keep_cols)} kept")
    fold_features = fold_features[keep_cols]

    fold_features = remove_constant_features(fold_features)
    print(f"[fold {fold_i}]   After removing constant features: {fold_features.shape}")

    for col in fold_features.columns:
        fold_features[col] = pd.to_numeric(fold_features[col], errors="coerce").fillna(
            0
        )

    real = fold_binary["real"].astype(int)

    # ── join supplementary features ───────────────────────────────────────────
    print(f"[fold {fold_i}] Joining supplementary features…")
    print(f"[fold {fold_i}]   TE features: {te_feats.shape[1]} cols")
    print(f"[fold {fold_i}]   NBD features: {nbd_feats.shape[1]} cols")
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

    print(f"[fold {fold_i}]   All loaded columns: {all_feat.columns.tolist()}")

    # ── optional correlation filtering ────────────────────────────────────────
    if feature_mode == "filtered" and top_feats is not None:
        print(f"[fold {fold_i}] Applying feature filtering (mode={feature_mode})…")
        print(f"[fold {fold_i}]  Separating categorical and continuous features...")
        cat_substrings = ["_has_", "_present"]
        cat_col_names = [""]  # hardcode any specific categorical column names here
        cat_exceptions = [
            "motif_types_present"
        ]  # hardcode any exceptions to the cat_substrings rule here
        cat_cols = [
            col
            for col in all_feat.columns
            if any(sub in col for sub in cat_substrings) or col in cat_col_names
        ]
        cat_cols = [col for col in cat_cols if col not in cat_exceptions]

        print(f"[fold {fold_i}]   Categorical columns: {len(cat_cols)}")
        # Print all categorical columns as a list
        print(f"[fold {fold_i}]   Categorical columns: {cat_cols}")

        print(
            f"[fold {fold_i}]  Continuous features: {len(all_feat.columns) - len(cat_cols)}"
        )
        print(
            f"[fold {fold_i}]    Filtering continuous features based on consensus list..."
        )
        print(f"[fold {fold_i}]    Top features from consensus: {len(top_feats)}")

        all_top_cols = cat_cols + list(top_feats)
        all_top_cols = [c for c in all_top_cols if c in all_feat.columns]
        print(f"[fold {fold_i}]    Total cols after filtering: {len(all_top_cols)}")
        all_feat = all_feat[all_top_cols]
    else:
        print(f"[fold {fold_i}] Using all features (mode={feature_mode})")

    all_feat = all_feat.loc[real.index]
    print(f"[fold {fold_i}] Final feature matrix: {all_feat.shape}")
    print(f"[fold {fold_i}] Final feature columns: {all_feat.columns.tolist()}")

    X_train = all_feat.loc[list(train_ids)]
    y_train = real.loc[X_train.index]
    X_test = all_feat.loc[list(test_ids)]
    y_test = real.loc[X_test.index]

    print(
        f"[fold {fold_i}] Train/test split: "
        f"X_train={X_train.shape}, X_test={X_test.shape}"
    )

    return X_train, X_test, y_train, y_test


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    np.random.seed(args.random_state)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── output paths ─────────────────────────────────────────────────────────
    rf_path = out_dir / "rf_model.joblib"
    shap_path = out_dir / "shap_values.csv"
    bval_path = out_dir / "base_val.txt"
    xtest_path = out_dir / "X_test.csv"
    ytest_path = out_dir / "y_test.csv"
    ypred_path = out_dir / "y_pred.csv"
    rep_path = out_dir / "classification_report.csv"

    all_exist = all(
        p.exists()
        for p in [
            rf_path,
            shap_path,
            bval_path,
            xtest_path,
            ytest_path,
            ypred_path,
            rep_path,
        ]
    )
    if all_exist and not args.force_rerun:
        print(
            f"[fold {args.fold}] All outputs exist — skipping "
            f"(pass --force-rerun to overwrite)"
        )
        return

    # ── load data ─────────────────────────────────────────────────────────────
    print(f"[fold {args.fold}] Loading dataset: {args.dataset}")
    dataset = load_dataset(args.dataset)
    features_df = dataset["features"]
    binary = dataset["binary"]

    print(f"[fold {args.fold}] Loading supplementary features…")
    te_feats, nbd_feats = load_supplementary_features(
        args.te_features, args.nbd_features
    )
    print(
        f"[fold {args.fold}]   TE={te_feats.shape[1]} cols, NBD={nbd_feats.shape[1]} cols"
    )

    # ── feature list: consensus JSON takes priority over cluster-file ─────────
    top_feats = None
    if args.selected_features:
        # Pre-computed consensus feature list from shap_rfecv_consensus.py
        with open(args.selected_features) as fh:
            top_feats = json.load(fh)
        args.feature_mode = "filtered"  # consensus implies filtering
        print(
            f"[fold {args.fold}] Using {len(top_feats)} consensus features "
            f"from {args.selected_features}"
        )
    elif args.feature_mode == "filtered":
        if not args.cluster_file:
            raise ValueError("--cluster-file is required for --feature-mode filtered")
        cluster_df = pd.read_csv(args.cluster_file, index_col=0)
        col = f"cluster_{args.cluster_threshold}"
        # Threshold may be formatted differently in the file (e.g. "cluster_0.40" vs "cluster_0.4")
        if col not in cluster_df.columns:
            matches = [
                c
                for c in cluster_df.columns
                if c == f"cluster_{float(args.cluster_threshold):.2f}"
            ]
            if not matches:
                matches = [
                    c
                    for c in cluster_df.columns
                    if c.startswith("cluster_")
                    and float(c.split("_", 1)[1]) == float(args.cluster_threshold)
                ]
            if not matches:
                raise ValueError(
                    f"No column matching threshold {args.cluster_threshold} found in cluster file. "
                    f"Available columns: {list(cluster_df.columns)}"
                )
            col = matches[0]
        # Select top feature (best effect size)
        # NOTE: Always keep transcript length as the representative for its cluster
        if "RNA_size_feelnc" in cluster_df.index:
            length_cluster = cluster_df[cluster_df.index == "RNA_size_feelnc"][
                col
            ].values[0]
            features_to_remove = cluster_df[
                cluster_df[col] == length_cluster
            ].index.difference(["RNA_size_feelnc"])
            cluster_df = cluster_df.drop(index=features_to_remove)

        top_feats = cluster_df.groupby(col).head(1).index
        print(
            f"[fold {args.fold}] Loaded {len(top_feats)} clusters to keep representative features"
            f"(threshold={args.cluster_threshold})"
        )

    # ── build fold matrices ───────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = build_fold_features(
        args.fold,
        args.dataset,
        args.results_dir,
        features_df,
        binary,
        te_feats,
        nbd_feats,
        args.feature_mode,
        top_feats,
    )
    print(
        f"[fold {args.fold}] train={len(X_train)}, test={len(X_test)}, "
        f"features={X_train.shape[1]}"
    )

    # ── optional test-set subsampling ─────────────────────────────────────────
    max_t = _max_transcripts(args.max_transcripts)
    if max_t and len(X_test) > max_t:
        sample_idx = np.random.choice(len(X_test), max_t, replace=False)
        X_test = X_test.iloc[sample_idx]
        y_test = y_test.iloc[sample_idx]
        print(f"[fold {args.fold}] Subsampled test set to {len(X_test)} transcripts")

    # ── RF: train or load from cache ─────────────────────────────────────────
    if args.force_rerun or not rf_path.exists():
        print(f"[fold {args.fold}] Training RandomForest…")
        rf = RandomForestClassifier(
            n_jobs=-1, random_state=args.random_state, class_weight="balanced"
        )
        rf.fit(X_train, y_train)
        joblib.dump(rf, rf_path)
        print(f"[fold {args.fold}] RF saved to {rf_path}")
    else:
        print(f"[fold {args.fold}] Loading cached RF from {rf_path}")
        rf = joblib.load(rf_path)

    # ── predictions ──────────────────────────────────────────────────────────
    y_pred = rf.predict(X_test)
    y_pred_proba = rf.predict_proba(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    print(classification_report(y_test, y_pred))

    # ── save predictions + test matrices ─────────────────────────────────────
    y_pred_df = pd.DataFrame(
        {
            "y_true": y_test,
            "y_pred": y_pred,
            "y_pred_proba_class_0": y_pred_proba[:, 0],
            "y_pred_proba_class_1": y_pred_proba[:, 1],
        },
        index=y_test.index,
    )
    y_pred_df.to_csv(ypred_path)
    y_test.to_csv(ytest_path)
    X_test.to_csv(xtest_path)
    pd.DataFrame([report]).to_csv(rep_path)

    if args.skip_shap:
        print(f"[fold {args.fold}] SHAP skipped (--skip-shap)")
        return

    # ── SHAP: compute or load from cache ─────────────────────────────────────
    if args.force_rerun or not shap_path.exists():
        print(
            f"[fold {args.fold}] Computing SHAP values (background={args.background_sample})…"
        )
        background = X_train.sample(
            n=min(args.background_sample, len(X_train)),
            random_state=args.random_state,
        )
        explainer = shap.TreeExplainer(
            rf,
            data=background,
            feature_perturbation="interventional",
            model_output="probability",
        )
        shap_values = explainer.shap_values(X_test, check_additivity=False)

        # Handle both old (list) and new (ndarray) SHAP API
        if isinstance(shap_values, list):
            sv = shap_values[1]
            base_val = float(explainer.expected_value[1])
        elif shap_values.ndim == 3:
            sv = shap_values[:, :, 1]
            base_val = float(explainer.expected_value[1])
        else:
            sv = shap_values
            base_val = float(explainer.expected_value)

        shap_df = pd.DataFrame(sv, columns=X_train.columns, index=X_test.index)
        shap_df.to_csv(shap_path)
        bval_path.write_text(str(base_val))
        print(f"[fold {args.fold}] SHAP values saved ({shap_df.shape})")
    else:
        print(f"[fold {args.fold}] SHAP outputs exist — loading cached values")

    print(f"[fold {args.fold}] Done. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
