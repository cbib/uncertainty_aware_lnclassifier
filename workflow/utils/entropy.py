import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.metrics import balanced_accuracy_score

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.features import (
    custom_feature_scaling,
    filter_feature_columns,
    get_probabilities,
)
from utils.parsing import load_tables

VERBOSE = True


################################
# DATASET LOADING AND HANDLING #
################################
# TODO: Move loading functions to a different utils file
def load_dataset(dataset_name: str) -> dict[str, pd.DataFrame]:
    """
    Load dataset with predictions and labels.

    Parameters:
    -----------
    dataset_name : str
        Name of the analyzed dataset (e.g., 'gencode.v47.common.cdhit.cv')
    basedir : Path
        Base directory of the analysis pipeline

    Returns:
    dict[str, pd.DataFrame] with keys:
        - 'probs': DataFrame of tool probabilities (index=seq_ID, columns=tools)
        - 'labels': DataFrame with 'coding_class' and 'biotype' columns
        - 'features': DataFrame of features for the transcripts
        - 'binary': Original binary table with boolean prediction labels. Also contains 'real' column.
    """
    tables = load_tables(dataset_name)
    binary = tables["binary"].set_index("seq_ID")
    pc_ids = binary[binary["real"] == True].index.tolist()
    lnc_ids = binary[binary["real"] == False].index.tolist()
    ids_with_class = pc_ids + lnc_ids

    features_df = tables["full_table"].set_index("seq_ID")
    features_df = features_df.loc[ids_with_class]
    probs = get_probabilities(features_df)
    probs = probs.dropna()

    labels = pd.DataFrame(index=probs.index)
    labels["coding_class"] = labels.index.isin(pc_ids).astype(int)
    labels["biotype"] = labels["coding_class"].map({1: "coding", 0: "lncRNA"})

    return {
        "probs": probs,
        "labels": labels,
        "features": features_df.loc[probs.index],
        "binary": binary,
    }


def load_additional_features(
    dataset_name: str, basedir: Path, pipelines: dict[str, str] = None
) -> dict[str, pd.DataFrame]:
    """
    Load TE and NBD pipeline features if available.

    Parameters:
    -----------
    dataset_name : str
        Name of the analyzed dataset (e.g., 'gencode.v47.common.cdhit.cv')
    basedir : Path
        Base directory of the analysis pipeline

    Returns:
    --------
    dict[str, pd.DataFrame] with keys:
        - 'te_pipeline': DataFrame of TE pipeline features (if available)
        - 'nbd_pipeline': DataFrame of NBD pipeline features (if available)
        - 'entropy': DataFrame of uncertainty metrics (if available)
        Each DataFrame is indexed by 'seq_ID' and contains relevant features for the transcripts.
    """
    additional = {}
    if pipelines is None:
        pipelines = {
            "te_pipeline": "te_pipeline/results/te_analysis_flexible/features/all_transcripts_te_features.csv",
            "nbd_pipeline": "nonb-pipeline/results/gencode.v47/extended_analysis/features_nonb_features.csv",
            "entropy": f"results/{dataset_name}/features/entropy/{dataset_name}_uncertainty_analysis.tsv",
        }

    for key, rel_path in pipelines.items():
        path = basedir / rel_path
        if path.exists():
            ext = path.suffix
            df = pd.read_csv(path, sep="\t" if ext == ".tsv" else ",", index_col=0)
            additional[key] = df
            if VERBOSE:
                print(
                    f"✓ Loaded {key} ({df.shape[0]} samples × {df.shape[1]} features)"
                )
        else:
            if VERBOSE:
                print(f"⚠ {key} not found at {rel_path}")

    return additional


def assign_entropy_groups(
    entropy_df: pd.DataFrame,
    mode: str,
    low_th: int,
    high_th: int,
    entropy_column: str = "H_pred",
    entropy_column_high: str | None = "I_bald",
    high_threshold_sec: int | None = None,
    class_column: str = "coding_class",
) -> pd.Series:
    """Assign entropy groups from an uncertainty table.

    Parameters
    ----------
    entropy_df : pd.DataFrame
        Uncertainty table indexed by transcript ID.
    mode : str
        Either "overall" or "class_separated".
    low_th : int
        Lower percentile threshold for the low group.
    high_th : int
        Upper percentile threshold for the primary high-group column.
    entropy_column : str
        Primary entropy column used for grouping.
    entropy_column_high : str | None
        Optional secondary column required for membership in the high group.
    high_threshold_sec : int | None
        Percentile threshold for ``entropy_column_high``. If omitted and a
        secondary column is used, ``high_th`` is reused.
    class_column : str
        Class column used for class-separated grouping.

    Returns
    -------
    pd.Series
        Series named ``entropy_group`` indexed by transcript ID.
    """
    if mode not in {"overall", "class_separated"}:
        raise ValueError("mode must be one of {'overall', 'class_separated'}")

    if entropy_column not in entropy_df.columns:
        raise ValueError(f"Column '{entropy_column}' not found in entropy dataframe")

    secondary_threshold = high_th if high_threshold_sec is None else high_threshold_sec
    use_secondary = (
        entropy_column_high is not None and entropy_column_high in entropy_df.columns
    )

    groups = pd.Series("middle", index=entropy_df.index, name="entropy_group")

    if mode == "class_separated":
        if class_column not in entropy_df.columns:
            raise ValueError(
                f"Class column '{class_column}' not found in entropy dataframe"
            )

        class_labels = {
            0: "lncRNA",
            1: "coding",
            "lncRNA": "lncRNA",
            "coding": "coding",
        }
        for cls in pd.Series(entropy_df[class_column]).dropna().unique():
            mask = entropy_df[class_column] == cls
            label = class_labels.get(cls, str(cls))
            lo = np.percentile(entropy_df.loc[mask, entropy_column], low_th)
            hi_primary = np.percentile(entropy_df.loc[mask, entropy_column], high_th)

            groups[mask & (entropy_df[entropy_column] <= lo)] = f"low_{label}"

            high_mask = mask & (entropy_df[entropy_column] >= hi_primary)
            if use_secondary:
                hi_secondary = np.percentile(
                    entropy_df.loc[mask, entropy_column_high], secondary_threshold
                )
                high_mask = high_mask & (
                    entropy_df[entropy_column_high] >= hi_secondary
                )
                print(
                    f"  {label}: low <= {entropy_column} {lo:.4f} | "
                    f"high >= {entropy_column} {hi_primary:.4f} AND "
                    f"{entropy_column_high} {hi_secondary:.4f}"
                )
            else:
                print(
                    f"  {label}: low <= {entropy_column} {lo:.4f} | "
                    f"high >= {entropy_column} {hi_primary:.4f}"
                )
            groups[high_mask] = f"high_{label}"
    else:
        lo = np.percentile(entropy_df[entropy_column], low_th)
        hi_primary = np.percentile(entropy_df[entropy_column], high_th)
        groups[entropy_df[entropy_column] <= lo] = "low"

        high_mask = entropy_df[entropy_column] >= hi_primary
        if use_secondary:
            hi_secondary = np.percentile(
                entropy_df[entropy_column_high], secondary_threshold
            )
            high_mask = high_mask & (entropy_df[entropy_column_high] >= hi_secondary)
            print(
                f"  overall: low <= {entropy_column} {lo:.4f} | "
                f"high >= {entropy_column} {hi_primary:.4f} AND "
                f"{entropy_column_high} {hi_secondary:.4f}"
            )
        else:
            print(
                f"  overall: low <= {entropy_column} {lo:.4f} | "
                f"high >= {entropy_column} {hi_primary:.4f}"
            )
        groups[high_mask] = "high"

    return groups


def load_entropy_groups(
    groups_tsv: str | Path,
    transcript_index: pd.Index | None = None,
) -> pd.Series:
    """Load a persisted entropy-group TSV and optionally align it to an index."""
    groups_df = pd.read_csv(groups_tsv, sep="\t", index_col=0)
    if "entropy_group" not in groups_df.columns:
        raise ValueError(
            f"Expected column 'entropy_group' in {groups_tsv}; found {list(groups_df.columns)}"
        )
    groups = groups_df["entropy_group"]
    if transcript_index is not None:
        groups = groups.reindex(transcript_index)
    return groups


def split_entropy_group_indices(groups: pd.Series) -> tuple[pd.Index, pd.Index]:
    """Return low-group and high-group transcript indices from group labels."""
    group_labels = groups.dropna().astype(str)
    low_idx = group_labels[group_labels.str.startswith("low")].index
    high_idx = group_labels[group_labels.str.startswith("high")].index
    return low_idx, high_idx


def compute_uncertainty_metrics(probs):
    """
    Calculate predictive entropy, expected entropy, and BALD from tool probabilities.

    Parameters:
    -----------
    probs : pd.DataFrame
        DataFrame where each column is a tool and each row is a transcript,
        containing probability of coding class (0-1)

    Returns:
    --------
    pd.DataFrame with columns:
        - H_pred: Predictive entropy (uncertainty in ensemble prediction)
        - H_exp: Expected entropy (average uncertainty across models)
        - I_bald: BALD / Mutual information (model disagreement)
        - plus individual tool entropies
    """

    def compute_entropy_binary(p_coding: pd.Series) -> pd.Series:
        """Compute binary entropy for a probability value."""
        p_noncoding = 1.0 - p_coding
        p = np.stack([p_noncoding, p_coding], axis=1)
        return pd.Series(entropy(p, base=2, axis=1), index=p_coding.index)

    def compute_consensus_entropy(tool_probs: pd.Series) -> pd.Series:
        """Compute entropy of the ensemble (average) prediction."""
        consensus_prob_coding = np.mean(tool_probs)
        return compute_entropy_binary(consensus_prob_coding)

    # Initialize results DataFrame
    entropy_df = pd.DataFrame(index=probs.index)

    # Calculate entropy for each tool
    for col in probs.columns:
        p_noncoding = 1.0 - probs[col]
        entropy_df[f"{col}_entropy"] = compute_entropy_binary(probs[col])

    # Calculate expected entropy (mean across tools)
    entropy_df["H_exp"] = entropy_df[[f"{col}_entropy" for col in probs.columns]].mean(
        axis=1
    )

    # Calculate predictive entropy (entropy of ensemble prediction)
    entropy_df["H_pred"] = compute_entropy_binary(probs.mean(axis=1))

    # Calculate BALD (mutual information)
    entropy_df["I_bald"] = entropy_df["H_pred"] - entropy_df["H_exp"]

    print(f"✓ Computed uncertainty metrics for {len(entropy_df)} transcripts")
    print(
        f"  H_pred range: [{entropy_df['H_pred'].min():.3f}, {entropy_df['H_pred'].max():.3f}]"
    )
    print(
        f"  H_exp range: [{entropy_df['H_exp'].min():.3f}, {entropy_df['H_exp'].max():.3f}]"
    )
    print(
        f"  I_bald range: [{entropy_df['I_bald'].min():.3f}, {entropy_df['I_bald'].max():.3f}]"
    )

    return entropy_df


def analyze_tool_agreement(
    probs,
    labels,
    entropy_df,
    strong_lnc_thresh=0.2,
    strong_coding_thresh=0.8,
    extreme_thresh=2,
):
    """
    Analyze agreement and disagreement patterns across prediction tools.

    Parameters:
    -----------
    probs : pd.DataFrame
        Tool probabilities
    labels : pd.DataFrame
        Ground truth labels
    entropy_df : pd.DataFrame
        Uncertainty metrics from compute_uncertainty_metrics()
    strong_lnc_thresh : float, default 0.2
        Probability threshold for "strong lncRNA" prediction
    strong_coding_thresh : float, default 0.8
        Probability threshold for "strong coding" prediction
    extreme_thresh : int, default 2
        Minimum number of tools on each side for extreme disagreement

    Returns:
    --------
    pd.DataFrame with additional columns:
        - variance, range, std: Spread of predictions across tools
        - n_strong_lnc: Number of tools predicting strong lncRNA
        - n_strong_coding: Number of tools predicting strong coding
        - extreme_disagreement: Boolean flag for extreme disagreement
        - extreme_agreement: Boolean flag for extreme agreement
        - majority_agreement: Boolean flag for majority agreement
    """
    # Create comprehensive agreement DataFrame
    agreement_df = pd.DataFrame(index=probs.index)

    # Add uncertainty metrics
    agreement_df["H_pred"] = entropy_df["H_pred"]
    agreement_df["H_exp"] = entropy_df["H_exp"]
    agreement_df["I_bald"] = entropy_df["I_bald"]

    # Add prediction statistics
    agreement_df["mean_prob"] = probs.mean(axis=1)
    agreement_df["variance"] = probs.var(axis=1)
    agreement_df["std"] = probs.std(axis=1)
    agreement_df["range"] = probs.max(axis=1) - probs.min(axis=1)

    # Ensemble prediction (majority vote style)
    agreement_df["ensemble_prob"] = probs.mean(axis=1)
    agreement_df["ensemble_pred"] = (agreement_df["ensemble_prob"] > 0.5).astype(int)

    # Add ground truth
    agreement_df["true_label"] = labels["coding_class"]
    agreement_df["biotype"] = labels["biotype"]

    # Count strong predictions
    agreement_df["n_strong_lnc"] = (probs < strong_lnc_thresh).sum(axis=1)
    agreement_df["n_strong_coding"] = (probs > strong_coding_thresh).sum(axis=1)
    agreement_df["n_moderate"] = (
        (probs >= strong_lnc_thresh) & (probs <= strong_coding_thresh)
    ).sum(axis=1)

    # Define agreement categories
    agreement_df["extreme_disagreement"] = (
        agreement_df["n_strong_lnc"] >= extreme_thresh
    ) & (agreement_df["n_strong_coding"] >= extreme_thresh)

    agreement_df["extreme_agreement"] = (
        (agreement_df["n_strong_lnc"] >= extreme_thresh)
        & (agreement_df["n_strong_coding"] == 0)
    ) | (
        (agreement_df["n_strong_coding"] >= extreme_thresh)
        & (agreement_df["n_strong_lnc"] == 0)
    )

    n_tools = len(probs.columns)
    majority_thresh = n_tools // 2 + 1
    agreement_df["majority_agreement"] = (
        agreement_df["n_strong_lnc"] >= majority_thresh
    ) | (agreement_df["n_strong_coding"] >= majority_thresh)

    # Summary statistics
    n_extreme_disagree = agreement_df["extreme_disagreement"].sum()
    n_extreme_agree = agreement_df["extreme_agreement"].sum()
    n_majority = agreement_df["majority_agreement"].sum()

    print(f"✓ Agreement analysis complete:")
    print(
        f"  - Extreme disagreement: {n_extreme_disagree} ({100*n_extreme_disagree/len(agreement_df):.2f}%)"
    )
    print(
        f"  - Extreme agreement: {n_extreme_agree} ({100*n_extreme_agree/len(agreement_df):.2f}%)"
    )
    print(
        f"  - Majority agreement: {n_majority} ({100*n_majority/len(agreement_df):.2f}%)"
    )

    return agreement_df


def bootstrap_bin_accuracy(
    df: pd.DataFrame,
    mask: pd.Series = None,
    bin_col: str = None,
    group_label: str = None,
    n_boot=100,
    seed=42,
) -> np.ndarray:
    """
    Bootstrap accuracy for a specific group within a bin column.
    """
    if mask is None:
        if bin_col is None or group_label is None:
            raise ValueError(
                "Must provide either a mask or both bin_col and group_label"
            )
    else:
        df = df.loc[mask]

    rng = np.random.default_rng(seed)
    accs = []
    for _ in range(n_boot):
        sample_idx = rng.choice(df.index, size=len(df), replace=True)
        sample = df.loc[sample_idx]
        mask = sample[bin_col] == group_label
        if mask.sum() > 0:
            accs.append(
                balanced_accuracy_score(
                    sample.loc[mask, "true_label"], sample.loc[mask, "ensemble_pred"]
                )
            )
        else:
            accs.append(np.nan)
    return np.array(accs)


#
# Plotting
#


def plot_uncertainty_scatter(
    df,
    entropy_col="H_pred",
    bald_col="I_bald",
    color_by=None,
    title=None,
    figsize=(10, 8),
):
    """
    Create scatter plot of H_pred vs I_bald with optional coloring.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with uncertainty metrics
    entropy_col : str
        Column for x-axis (predictive entropy)
    bald_col : str
        Column for y-axis (BALD)
    color_by : str
        Column to use for coloring points
    title : str, optional
        Plot title
    figsize : tuple
        Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)

    if color_by and color_by in df.columns:
        for category in df[color_by].unique():
            mask = df[color_by] == category
            ax.scatter(
                df.loc[mask, entropy_col],
                df.loc[mask, bald_col],
                label=category,
                alpha=0.6,
                s=5,
            )
        ax.legend()
    else:
        ax.scatter(df[entropy_col], df[bald_col], alpha=0.6, s=5)

    ax.set_xlabel(f"{entropy_col} (Predictive Entropy)", fontweight="bold")
    ax.set_ylabel(f"{bald_col} (BALD)", fontweight="bold")
    ax.set_title(title or "Uncertainty Space", fontweight="bold", fontsize=14)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig
