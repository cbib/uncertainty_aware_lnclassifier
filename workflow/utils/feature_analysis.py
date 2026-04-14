"""
Unified Feature Analysis Module

Centralizes statistical testing, effect size calculations, residualization,
and visualization functions for feature analysis across multiple notebooks.

This module enables reusable, configurable analysis workflows.
"""

import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from scipy.stats import mannwhitneyu, rankdata
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")


#
# CORRELATIONS
#


def correlation_and_distance(df: pd.DataFrame, method="spearman"):
    """
    Compute correlation matrix and plot heatmap with hierarchical clustering.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame of features (index = seq_ID)
    method : str
        Correlation method ('pearson', 'spearman', 'kendall')

    Returns:
    --------
    corr_matrix : pd.DataFrame
        Correlation matrix of numeric features
    dist_linkage : np.ndarray
        Linkage matrix for hierarchical clustering
    """
    df = df.copy()
    df = df.fillna(0)
    numeric_features = df.select_dtypes(include=[np.number])
    print(
        f"Selected {numeric_features.shape[1]} numeric features (out of {df.shape[1]} total) for correlation analysis"
    )

    corr_matrix = numeric_features.corr(method="spearman")
    corr_matrix = (corr_matrix + corr_matrix.T) / 2

    distance_matrix = 1 - np.abs(corr_matrix)
    distance_matrix = np.where(np.isfinite(distance_matrix), distance_matrix, 0)

    dist_linkage = hierarchy.ward(squareform(distance_matrix))
    return corr_matrix, dist_linkage


def cluster_features(df, dist_linkage, dist: float):
    """
    Cluster features based on hierarchical clustering of distance matrix.

    Parameters:
    -----------
    dist_linkage : np.ndarray
        Linkage matrix from hierarchical clustering
    dist : float
        Distance threshold for forming flat clusters

    Returns:
    --------
    cluster_df : pd.DataFrame
        DataFrame mapping feature indices to cluster IDs
    """
    cluster_ids = hierarchy.fcluster(dist_linkage, dist, criterion="distance")
    cluster_df = pd.DataFrame({"feature_name": df.columns, "cluster_id": cluster_ids})
    return cluster_df


def get_representative_features_from_distance(dist_linkage, dist: float, colnames):
    """
    Get representative features based on hierarchical clustering of distance matrix.
    """
    cluster_ids = hierarchy.fcluster(dist_linkage, dist, criterion="distance")
    cluster_id_to_feature_ids = defaultdict(list)
    for idx, cluster_id in enumerate(cluster_ids):
        cluster_id_to_feature_ids[cluster_id].append(idx)

    representative_features = [v[0] for v in cluster_id_to_feature_ids.values()]
    representative_feature_names = colnames[representative_features]
    return representative_feature_names


def plot_correlation_heatmap(
    corr_matrix, dist_linkage, figsize=(12, 10), output_file=None
):
    """
    Plot correlation heatmap with hierarchical clustering.

    Parameters:
    -----------
    corr_matrix : pd.DataFrame
        Correlation matrix of features
    dist_linkage : np.ndarray
        Linkage matrix for hierarchical clustering
    figsize : tuple
        Figure size
    output_file : str
        Path to save figure

    Returns:
    --------
    None (saves figure)
    """
    g = sns.heatmap(
        corr_matrix,
        center=0,
        vmax=1,
        vmin=-1,
        cmap="RdBu_r",
        cbar_kws={"shrink": 0.8},
        figsize=figsize,
    )

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"✓ Saved correlation heatmap to {output_file}")

    return g


def plot_dendrogram(
    dist_linkage, colnames, distance_threshold=0.5, figsize=(10, 5), output_file=None
):
    """
    Plot dendrogram from hierarchical clustering.

    Parameters:
    -----------
    dist_linkage : np.ndarray
        Linkage matrix from hierarchical clustering
    colnames : list
        List of feature names corresponding to indices
    distance_threshold : float
        Distance threshold for horizontal line on dendrogram
    figsize : tuple
        Figure size
    output_file : str
        Path to save figure

    Returns:
    --------
    None (saves figure)
    """
    plt.figure(figsize=figsize)
    hierarchy.dendrogram(
        dist_linkage,
        labels=colnames,
        leaf_rotation=90,
        color_threshold=distance_threshold,
    )
    plt.title("Feature Dendrogram", fontweight="bold", fontsize=12)
    plt.xlabel("Features", fontsize=11)
    plt.ylabel("Distance", fontsize=11)
    plt.axhline(distance_threshold, color="red", linestyle="--", linewidth=1)

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"✓ Saved dendrogram to {output_file}")

    return plt.gcf()


# ============================================================================
# EFFECT SIZE & STATISTICAL FUNCTIONS
# ============================================================================


def cliffs_delta(x, y):
    """
    Cliff's delta: non-parametric effect size measure [-1, 1].

    Parameters:
    -----------
    x, y : array-like
        Two samples to compare

    Returns:
    --------
    float: Effect size in range [-1, 1]
    """
    x = np.array(x)
    y = np.array(y)
    nx, ny = len(x), len(y)
    rx = rankdata(np.concatenate([x, y]))[:nx]
    return (2 * np.sum(rx) / (nx * (nx + ny)) - 1) * np.sign(
        np.median(x) - np.median(y)
    )


def vargha_delaney_A(d1, d2):
    """
    Vargha-Delaney A statistic: P(X1 > X2) for unequal sample sizes.

    Parameters:
    -----------
    d1, d2 : array-like
        Two samples to compare

    Returns:
    --------
    float: Probability in range [0, 1]
    """
    n1, n2 = len(d1), len(d2)
    U = mannwhitneyu(d1, d2, method="auto").statistic
    return U / (n1 * n2)


def perform_mann_whitney_tests(
    features_df, group_df, group_col="group", min_group_size=50, exclude_cols=None
):
    """
    Perform Mann-Whitney U tests comparing groups on numeric features.

    Parameters:
    -----------
    features_df : pd.DataFrame
        Numeric feature matrix (index = seq_ID)
    group_df : pd.DataFrame
        DataFrame with group assignments (index = seq_ID, contains group column)
    group_col : str
        Column name containing group assignments
    min_group_size : int
        Minimum samples required per group for a contrast
    exclude_cols : list
        Columns to exclude from analysis (e.g., entropy metrics)

    Returns:
    --------
    pd.DataFrame: Results with effect sizes, p-values, and rankings
    """
    if exclude_cols is None:
        exclude_cols = [
            "H_pred",
            "H_exp",
            "I_bald",
            "mean_prob",
            "variance",
            "std",
            "range",
            "ensemble_prob",
            group_col,
        ]

    # Merge features with groups
    df = features_df.join(group_df[[group_col]], how="inner")

    # Get numeric features only
    numeric_feats = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_feats = [f for f in numeric_feats if f not in exclude_cols]

    print(f"✓ Mann-Whitney tests on {len(numeric_feats)} numeric features")

    # Get unique groups
    groups = sorted(df[group_col].unique())
    print(f"✓ Groups: {groups}")
    contrasts = [(g1, g2) for i, g1 in enumerate(groups) for g2 in groups[i + 1 :]]
    print(f"✓ Contrasts: {len(contrasts)}")

    results = []
    for g1, g2 in contrasts:
        mask1, mask2 = df[group_col] == g1, df[group_col] == g2
        n1, n2 = mask1.sum(), mask2.sum()
        print(f"  {g1} vs {g2}: n={n1} vs {n2}", end=" → ")

        if n1 > min_group_size and n2 > min_group_size:
            n_tests = 0
            for feat in numeric_feats:
                d1, d2 = df.loc[mask1, feat].dropna(), df.loc[mask2, feat].dropna()
                if len(d1) > min_group_size and len(d2) > min_group_size:
                    stat, pval = mannwhitneyu(d1, d2, alternative="two-sided")
                    delta = cliffs_delta(d1.values, d2.values)
                    vda = vargha_delaney_A(d1.values, d2.values)
                    results.append(
                        {
                            "contrast": f"{g1} vs {g2}",
                            "feature": feat,
                            "n1": len(d1),
                            "n2": len(d2),
                            "U_stat": stat,
                            "p_value": pval,
                            "cliffs_delta": delta,
                            "VDA": vda,
                            "median_g1": d1.median(),
                            "median_g2": d2.median(),
                            "median_diff": d1.median() - d2.median(),
                        }
                    )
                    n_tests += 1
            print(f"{n_tests} features tested")
        else:
            print("skipped (too small)")

    if not results:
        print("⚠️ No viable contrasts found")
        return pd.DataFrame()

    results_df = pd.DataFrame(results)

    # Apply FDR correction per contrast
    fdr_table = []
    for contrast in results_df["contrast"].unique():
        sub = results_df[results_df["contrast"] == contrast].copy()
        if len(sub) > 1:
            reject, p_fdr, _, _ = multipletests(sub["p_value"], method="fdr_bh")
            sub["p_FDR"] = p_fdr
            sub["sig_FDR05"] = reject
        else:
            sub["p_FDR"] = sub["p_value"]
            sub["sig_FDR05"] = sub["p_value"] < 0.05
        fdr_table.append(sub)

    fdr_table = pd.concat(fdr_table, ignore_index=True)

    # Calculate effect size composite metrics
    fdr_table["cles_pct"] = fdr_table["VDA"] * 100
    fdr_table["effect_size"] = np.abs(fdr_table["cliffs_delta"].fillna(0))
    fdr_table["vda_deviation"] = np.abs(fdr_table["VDA"] - 0.5)

    # Normalize effect sizes into [0, 1]
    fdr_table["norm_diff"] = np.abs(fdr_table["median_diff"]) / fdr_table.groupby(
        "contrast"
    )["median_diff"].transform("std").fillna(1)

    # Meta-rank: average percentile ranks of three effect size metrics
    es_cols = ["effect_size", "vda_deviation", "norm_diff"]
    ranks = fdr_table[es_cols].rank(pct=True, method="min")
    fdr_table["meta_rank"] = ranks.mean(axis=1)

    # Log-scale p-value score
    p_floor = 1e-10
    fdr_table["p_FDR_cap"] = np.maximum(fdr_table["p_FDR"], p_floor)
    fdr_table["logp_score"] = -np.log10(fdr_table["p_FDR_cap"])

    print(
        f"✓ {len(fdr_table)} test results; {fdr_table['sig_FDR05'].sum()} significant (FDR < 0.05)"
    )

    return fdr_table


def perform_f_tests(
    features_df,
    group_df,
    group_col="group",
    exclude_cols=None,
    fdr_method="fdr_bh",
    fdr_alpha=0.05,
    include_mi=False,
):
    """
    Perform F-tests with optional Mutual Information scores.
    Note: For standalone MI, use perform_mutual_info_tests() instead.

    Parameters:
    -----------
    features_df : pd.DataFrame
        Numeric feature matrix (index = seq_ID)
    group_df : pd.DataFrame
        DataFrame with group assignments (index = seq_ID)
    group_col : str
        Column name containing group assignments
    exclude_cols : list
        Columns to exclude (entropy metrics, etc.)
    fdr_method : str
        FDR correction method ('fdr_bh', 'bonferroni', etc.)
    fdr_alpha : float
        Significance threshold
    include_mi : bool
        Whether to include Mutual Information scores (deprecated)

    Returns:
    --------
    pd.DataFrame: Results with F-scores, p-values, FDR-corrected p-values
    """
    if exclude_cols is None:
        exclude_cols = [
            "H_pred",
            "H_exp",
            "I_bald",
            "mean_prob",
            "variance",
            "std",
            "range",
            "ensemble_prob",
            group_col,
        ]

    # Merge features with groups
    df = features_df.join(group_df[[group_col]], how="inner")

    # Get numeric features only
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [f for f in numeric_cols if f not in exclude_cols]

    print(f"✓ F-test on {len(numeric_cols)} numeric features")

    # Prepare data
    X = df[numeric_cols].fillna(0)
    y = LabelEncoder().fit_transform(df[group_col])

    # F-test
    scores, pvalues = f_classif(X, y)

    # FDR correction
    reject, pvals_corrected, _, _ = multipletests(
        pvalues, alpha=fdr_alpha, method=fdr_method
    )

    results = pd.DataFrame(
        {
            "feature": numeric_cols,
            "f_score": scores,
            "p_value": pvalues,
            "p_FDR": pvals_corrected,
            "significant_fdr": reject,
        }
    ).sort_values("f_score", ascending=False)

    # Optional: Mutual Information (deprecated - use perform_mutual_info_tests instead)
    if include_mi:
        mi_scores = mutual_info_classif(X, y, random_state=42)
        results["mi_score"] = mi_scores
        # Normalize both scores for composite ranking
        results["f_norm"] = results["f_score"] / results["f_score"].max()
        results["mi_norm"] = results["mi_score"] / results["mi_score"].max()
        results["composite_score"] = (results["f_norm"] + results["mi_norm"]) / 2

    n_sig = (pvals_corrected < fdr_alpha).sum()
    print(f"✓ {n_sig} features significant (FDR < {fdr_alpha})")

    return results


def perform_mutual_info_tests(
    features_df,
    group_df,
    group_col="group",
    exclude_cols=None,
    fdr_method="fdr_bh",
    fdr_alpha=0.05,
):
    """
    Perform Mutual Information analysis across groups as standalone method.

    Parameters:
    -----------
    features_df : pd.DataFrame
        Numeric feature matrix (index = seq_ID)
    group_df : pd.DataFrame
        DataFrame with group assignments (index = seq_ID)
    group_col : str
        Column name containing group assignments
    exclude_cols : list
        Columns to exclude (entropy metrics, etc.)
    fdr_method : str
        FDR correction method ('fdr_bh', 'bonferroni', etc.)
    fdr_alpha : float
        Significance threshold

    Returns:
    --------
    pd.DataFrame: Results with mutual info scores and FDR-corrected p-values
    """
    if exclude_cols is None:
        exclude_cols = [
            "H_pred",
            "H_exp",
            "I_bald",
            "mean_prob",
            "variance",
            "std",
            "range",
            "ensemble_prob",
            group_col,
        ]

    # Merge features with groups
    df = features_df.join(group_df[[group_col]], how="inner")

    # Get numeric features only
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [f for f in numeric_cols if f not in exclude_cols]

    print(f"✓ Mutual Information analysis on {len(numeric_cols)} numeric features")

    # Prepare data
    X = df[numeric_cols].fillna(0)
    y = LabelEncoder().fit_transform(df[group_col])

    # Compute MI scores
    mi_scores = mutual_info_classif(X, y, random_state=42)

    results = pd.DataFrame(
        {
            "feature": numeric_cols,
            "mi_score": mi_scores,
        }
    ).sort_values("mi_score", ascending=False)

    print(f"✓ Computed MI scores for {len(results)} features")

    return results


# ============================================================================
# RESIDUALIZATION & PREPROCESSING
# ============================================================================


def residualize_features(features_df, length_col=None, length_source="features_df"):
    """
    Remove length confounding from features via linear regression.

    Parameters:
    -----------
    features_df : pd.DataFrame
        Feature matrix (index = seq_ID)
    length_col : str
        Name of length column ('RNA_size_feelnc', 'Transcript_length_lncDC', etc.)
        If None, will try to auto-detect
    length_source : str
        'features_df' (default) or alternative source

    Returns:
    --------
    pd.DataFrame: Residualized features (same index/columns as input)
    """
    df = features_df.copy()

    # Auto-detect length column if not specified
    if length_col is None:
        candidates = ["RNA_size_feelnc", "Transcript_length_lncDC", "transcript_length"]
        for cand in candidates:
            if cand in df.columns:
                length_col = cand
                break
        if length_col is None:
            print("⚠️ Length column not found; skipping residualization")
            return df

    if length_col not in df.columns:
        print(f"⚠️ {length_col} not found in features; skipping residualization")
        return df

    print(f"✓ Residualizing {len(df.columns)-1} features against {length_col}")

    # Get numeric columns (exclude length)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != length_col]

    # Prepare regressor
    X_length = df[[length_col]].fillna(0)

    # Residualize each feature
    residuals = pd.DataFrame(index=df.index, columns=numeric_cols)

    for col in numeric_cols:
        y = df[col].fillna(0)

        # Skip if constant or all NaN
        if y.std() == 0 or y.isna().all():
            residuals[col] = y
            continue

        # Fit regression
        lr = LinearRegression()
        lr.fit(X_length, y)

        # Residuals
        y_pred = lr.predict(X_length)
        residuals[col] = y - y_pred

    # Copy non-numeric columns
    for col in df.columns:
        if col not in numeric_cols and col != length_col:
            residuals[col] = df[col]

    # Add length column back for reference
    residuals[length_col] = df[length_col]

    print(f"✓ Residualized {len(numeric_cols)} features")

    return residuals


# ============================================================================
# FEATURE RANKING & SCORING
# ============================================================================
# TODO: Merge with rank_features_by_score. Some code will go to perform_mann_whitney_tests


def rank_features_by_composite(
    results_df, score_columns=["effect_size", "vda_deviation", "norm_diff"]
):
    """
    Rank features by composite normalized score across multiple metrics.

    Parameters:
    -----------
    results_df : pd.DataFrame
        Results with score columns (from Mann-Whitney tests)
    score_columns : list
        Columns to normalize and aggregate

    Returns:
    --------
    pd.DataFrame: Same as input with 'rank_score' and 'rank_percentile' columns
    """
    print(f"✓ Ranking features by composite of: {score_columns}")
    df = results_df.copy()

    # Ensure all score columns are present
    missing = [c for c in score_columns if c not in df.columns]
    if missing:
        print(f"⚠️ Missing columns: {missing}")
        return df

    # Normalize each score into [0, 1]
    for col in score_columns:
        min_val = df[col].min()
        max_val = df[col].max()
        if max_val > min_val:
            df[f"{col}_norm"] = (df[col] - min_val) / (max_val - min_val)
        else:
            df[f"{col}_norm"] = 0

    norm_cols = [f"{c}_norm" for c in score_columns]
    df["rank_score"] = df[norm_cols].mean(axis=1)
    df["rank_percentile"] = df["rank_score"].rank(pct=True)

    return df.sort_values("rank_score", ascending=False)


def rank_features_by_score(results_df, score_column="f_score", compute_all_ranks=True):
    """
    Rank features by configured score column(s).

    Optionally computes ranking for all available score columns:
    - f_score (from F-tests)
    - mi_score (from Mutual Information)
    - composite_score (for Mann-Whitney U tests)

    Parameters:
    -----------
    results_df : pd.DataFrame
        Results with score column(s)
    score_column : str
        Primary column to use for ranking (default='f_score')
    compute_all_ranks : bool
        If True, compute rank pairs for f_score, mi_score, and composite_score

    Returns:
    --------
    pd.DataFrame: Sorted by score_column descending, with rank columns
    """
    df = results_df.copy()

    # Compute ranking on primary column
    if score_column in df.columns:
        df[f"rank_{score_column}"] = df[score_column].rank(ascending=False)
        df[f"rank_pct_{score_column}"] = df[score_column].rank(pct=True)
        sort_col = score_column
    else:
        print(f"⚠️ {score_column} not found; using first numeric column")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            sort_col = numeric_cols[0]
            df[f"rank_{sort_col}"] = df[sort_col].rank(ascending=False)
            df[f"rank_pct_{sort_col}"] = df[sort_col].rank(pct=True)
        else:
            return df

    # Optionally compute ranks for other available score columns
    if compute_all_ranks:
        for col in ["f_score", "mi_score", "composite_score"]:
            if col in df.columns and col != score_column:
                df[f"rank_{col}"] = df[col].rank(ascending=False)
                df[f"rank_pct_{col}"] = df[col].rank(pct=True)

    return df.sort_values(sort_col, ascending=False)


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================


def plot_feature_distributions(
    features_df,
    group_df,
    group_col="group",
    cherry_picked_features=None,
    figsize=(12, 10),
    output_file=None,
):
    """
    Plot feature distributions across groups using KDE density plots.

    Parameters:
    -----------
    features_df : pd.DataFrame
        Feature matrix
    group_df : pd.DataFrame
        Group assignments
    group_col : str
        Column name with group labels
    cherry_picked_features : list
        Features to plot (if None, uses top 6 by variance)
    figsize : tuple
        Figure size
    output_file : str
        Path to save figure

    Returns:
    --------
    None (saves figure)
    """
    df = features_df.join(group_df[[group_col]], how="inner")

    # Select features to plot
    if cherry_picked_features is None:
        cherry_picked_features = (
            df.select_dtypes(include=[np.number]).var().nlargest(6).index.tolist()
        )

    plot_features = [f for f in cherry_picked_features if f in df.columns]

    if not plot_features:
        print("⚠️ No valid features to plot")
        return

    n_features = len(plot_features)
    n_rows = (n_features + 1) // 2
    n_cols = min(n_features, 2)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_features > 1 else [axes]

    for idx, feature in enumerate(plot_features):
        ax = axes[idx]
        sns.kdeplot(
            data=df,
            x=feature,
            hue=group_col,
            ax=ax,
            fill=True,
            common_norm=False,
            alpha=0.5,
        )
        ax.set_title(feature.replace("_", " "), fontweight="bold", fontsize=11)
        ax.grid(axis="y", alpha=0.3)

    # Hide unused axes
    for idx in range(len(plot_features), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"✓ Saved distribution plot to {output_file}")

    return fig


def plot_feature_heatmap(
    results_df, top_n=15, score_column="rank_score", figsize=(10, 8), output_file=None
):
    """
    Plot top features ranked by score as a heatmap.

    Parameters:
    -----------
    results_df : pd.DataFrame
        Results DataFrame with features and scores
    top_n : int
        Number of top features to display
    score_column : str
        Column to use for ranking
    figsize : tuple
        Figure size
    output_file : str
        Path to save figure

    Returns:
    --------
    None (saves figure)
    """
    top_features = results_df.nlargest(top_n, score_column)[
        ["feature", score_column]
    ].copy()
    top_features = top_features.sort_values(score_column, ascending=True)

    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top_features)))

    bars = ax.barh(
        range(len(top_features)), top_features[score_column].values, color=colors
    )
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features["feature"].values, fontsize=10)
    ax.set_xlabel(score_column.replace("_", " ").title(), fontsize=11)
    ax.set_title(
        f"Top {top_n} Features by {score_column}", fontweight="bold", fontsize=12
    )
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"✓ Saved heatmap to {output_file}")

    return fig


def plot_volcano(
    results_raw,
    results_resid=None,
    effect_size_col="cliffs_delta",
    pval_col="p_FDR",
    figsize=(12, 6),
    output_file=None,
):
    """
    Create a volcano plot comparing raw and residualized results.

    Parameters:
    -----------
    results_raw : pd.DataFrame
        Results from raw features
    results_resid : pd.DataFrame
        Results from residualized features (optional)
    effect_size_col : str
        Column name for effect size (x-axis)
    pval_col : str
        Column name for p-value (y-axis)
    figsize : tuple
        Figure size
    output_file : str
        Path to save figure

    Returns:
    --------
    None (saves figure)
    """
    fig, axes = plt.subplots(1, 2 if results_resid is not None else 1, figsize=figsize)
    axes = axes if isinstance(axes, np.ndarray) else [axes]

    for idx, (label, data) in enumerate(
        [("Raw", results_raw), ("Residualized", results_resid)]
    ):
        if data is None:
            continue

        ax = axes[idx]

        # Prepare data
        effect = data[effect_size_col].fillna(0)
        pval = -np.log10(data[pval_col].fillna(1))
        sig = (
            data["significant_fdr"]
            if "significant_fdr" in data.columns
            else pval > -np.log10(0.05)
        )

        # Plot
        colors = np.where(sig, "red", "gray")
        ax.scatter(effect, pval, alpha=0.6, c=colors, s=50)

        # Add significance threshold line
        ax.axhline(
            y=-np.log10(0.05), color="black", linestyle="--", linewidth=1, alpha=0.5
        )
        ax.axvline(x=0, color="black", linestyle="-", linewidth=0.5, alpha=0.5)

        ax.set_xlabel(f"{effect_size_col} (Effect Size)", fontsize=11)
        ax.set_ylabel(f"-log10(p-value)", fontsize=11)
        ax.set_title(f"Volcano Plot ({label})", fontweight="bold", fontsize=12)
        ax.grid(alpha=0.3)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"✓ Saved volcano plot to {output_file}")

    return fig


# ============================================================================
# DATA LOADING & DISCOVERY
# ============================================================================


def discover_feature_sets(base_dir, dataset_name):
    """
    Auto-discover available feature sets in the results directory.

    Parameters:
    -----------
    base_dir : str or Path
        Base directory for results
    dataset_name : str
        Name of the dataset

    Returns:
    --------
    dict: {feature_set_name: file_path}
    """
    base_dir = Path(base_dir)
    feature_sets = {}

    # Common feature set locations
    patterns = {
        "default": f"results/{dataset_name}/tables/{dataset_name}_full_table.tsv",
        "te_pipeline": "te_pipeline/results/te_analysis_flexible/features/all_transcripts_te_features.csv",
        "nbd_pipeline": "nonb-pipeline/results/gencode.v47/extended_analysis/features_nonb_features.csv",
    }

    for name, rel_path in patterns.items():
        full_path = base_dir / rel_path
        if full_path.exists():
            feature_sets[name] = str(full_path)
            print(f"✓ Found {name} at {rel_path}")
        else:
            print(f"⚠️ {name} not found at {rel_path}")

    return feature_sets


def load_feature_set(filepath, index_col=0):
    """
    Load a feature set from CSV/TSV file.

    Parameters:
    -----------
    filepath : str
        Path to feature file
    index_col : int
        Column to use as index

    Returns:
    --------
    pd.DataFrame: Loaded features
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Feature file not found: {filepath}")

    ext = filepath.suffix.lower()

    if ext == ".tsv":
        df = pd.read_csv(filepath, sep="\t", index_col=index_col)
    elif ext == ".csv":
        df = pd.read_csv(filepath, index_col=index_col)
    else:
        raise ValueError(f"Unknown file type: {ext}")

    print(
        f"✓ Loaded {df.shape[0]} samples × {df.shape[1]} features from {filepath.name}"
    )

    return df
