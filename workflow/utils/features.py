import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy
from sklearn.decomposition import PCA
from sklearn.preprocessing import PowerTransformer, StandardScaler

####################################
# Constants for feature processing #
####################################
FEAT_TOOL_SUFFIXES = [
    "_rnasamba",
    "_feelnc",
    "_l_cpat",
    "_lncDC",
    "_mrnn",
    "_lncfinder",
    "_lncrnabert",
    "_plncpro",
]

FEAT_METADATA_COLS = ["metadata", "biotype", "temp_id"]

FEAT_PROB_COLS = [
    "coding_score_rnasamba",
    "coding_potential_feelnc",
    "Coding_prob_l_cpat",
    "Noncoding_prob_ss_lncDC",
    "coding_prob_mrnn",
    "P(pcRNA)_lncrnabert",
    "prob_coding_plncpro",
    "Coding.Potential_ss_lncfinder",
]

FEAT_TO_REMOVE = [
    "logit_coding_prob_mrnn",
    "prediction_plncpro",
    "num_label_feelnc",
    "Noncoding_prob_lncDC",
    "Noncoding_prob_ss_lncDC",
    "prob_noncoding_plncpro",
    "score_plncpro",
    "Coding.Potential_lncfinder",
    "Pred_lncfinder",
    "Pred_ss_lncfinder",
]

FEAT_LENGTH_COLS = [
    "transcript_length",
    "RNA_size_feelnc",
    "Transcript_length_lncDC",
    "length_plncpro",
]

# TODO: convert into a dictionary so that the feature gets renamed upon inversion
FEAT_INVERT_PROBS = ["Noncoding_prob_ss_lncDC"]

# Substrings that identify binary (0/1 presence) feature columns.
# Shared by clustering, statistical testing, and figure scripts.
# TODO: define as in a configuration yaml
CAT_FEATURE_SUBSTRINGS: tuple[str, ...] = ("_has_", "_present")
CAT_FEATURE_EXACT: frozenset[str] = frozenset({"ORF_frame_l_cpat"})
CAT_FEATURE_EXEPTIONS: frozenset[str] = frozenset({"motif_types_present"})


def custom_feature_scaling(df, use_power_transform=False):
    """
    Scale numeric features using optional power transformation + standard scaling.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing features to scale.
    use_power_transform : bool, default=False
        If True, apply Yeo-Johnson transformation before StandardScaler.

    Returns:
    --------
    np.ndarray
        Scaled feature array with NaN/inf values replaced by 0.
    """
    numeric_features = df.select_dtypes(include="number")
    print(
        f"Selected {numeric_features.shape[1]} numeric features out of {df.shape[1]} for scaling."
    )

    if use_power_transform:
        print("Applying Yeo-Johnson transformation...")
        transformer = PowerTransformer(method="yeo-johnson", standardize=False)
        numeric_features = transformer.fit_transform(numeric_features)

    print("Standardizing features...")
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(numeric_features)
    scaled_features = np.nan_to_num(scaled_features, nan=0.0, posinf=0.0, neginf=0.0)
    return scaled_features


def filter_feature_columns(
    df,
    tool_suffixes=FEAT_TOOL_SUFFIXES,
    metadata_cols=FEAT_METADATA_COLS,
    prob_colnames=FEAT_PROB_COLS,
    to_remove=FEAT_TO_REMOVE,
    length_cols=FEAT_LENGTH_COLS,
):
    """
    Filter DataFrame columns to keep only numeric features for analysis.
    Excludes metadata, class labels, probabilities, and other specified columns.

    Parameters:
    -----------
    df : pd.DataFrame
        Full feature DataFrame.
    tool_suffixes : list
        List of tool suffixes (e.g., ["_feelnc", "_cpat", "_lncDC"]).
    metadata_cols : list, optional
       Metadata columns to exclude. Defined at module level.
    prob_colnames : list, optional
        Probability column names to exclude.
    to_remove : list, optional
        Additional column names to manually exclude.

    Returns:
    --------
    list
        Filtered list of feature column names.
    """
    if metadata_cols is None:
        metadata_cols = []
    if prob_colnames is None:
        prob_colnames = []
    if to_remove is None:
        to_remove = []
    if length_cols is None:
        length_cols = []

    # Check length columns
    length_cols = [col for col in length_cols if col in df.columns]
    if len(length_cols) > 1:
        print(
            f"Identified length columns to exclude: {length_cols[1:]} (keeping {length_cols[0]} for reference)"
        )
        to_remove += length_cols[1:]  # Keep the first length column
    elif len(length_cols) == 1:
        print(f"No length columns to exclude (keeping {length_cols[0]} for reference)")

    label_cols = ["label" + c for c in df.columns]

    feature_cols = [col for col in df.columns if col.endswith(tuple(tool_suffixes))]
    feature_cols = [
        col
        for col in feature_cols
        if col not in metadata_cols + label_cols + prob_colnames + to_remove
    ]
    feature_cols = df[feature_cols].select_dtypes(include="number").columns.tolist()

    print(f"Total number of columns in features table: {df.shape[1]}")
    print(f"Number of kept feature columns: {len(feature_cols)}")
    print(f"Feature columns: {feature_cols}")

    return feature_cols


def remove_constant_features(df, name="Dataset"):
    nunique = df.nunique()
    constant_features = nunique[nunique <= 1].index.tolist()
    if constant_features:
        print(f"{name}: Removing {len(constant_features)} constant features")
        print(f"  Constant features: {constant_features}")
        df = df.drop(columns=constant_features)
    else:
        print(f"{name}: No constant features were removed")
    return df


def get_categorical_and_continuous_columns(df: pd.DataFrame) -> list[list]:
    """
    Separate categorical from continuous columns
    Columns matching CAT_FEATURE_SUBSTRINGS or CAT_FEATURE_EXACT are binary presence features.
    CAT_FEATURE_EXEPTIONS look categorical by name but is treated as continuous.

    Params
    ------
    df: pd.Dataframe

    Returns
    ------

    """
    cat_cols = [
        col
        for col in df.columns
        if (
            any(sub in col for sub in CAT_FEATURE_SUBSTRINGS)
            or col in CAT_FEATURE_EXACT
        )
        and col not in CAT_FEATURE_EXEPTIONS
    ]
    cont_cols = df.columns.difference(cat_cols).tolist()
    return cat_cols, cont_cols


def get_probabilities(df, prob_colnames=FEAT_PROB_COLS, invert_probs=FEAT_INVERT_PROBS):
    """
    Extract probability columns from DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing probability columns.
    prob_colnames : list
        List of probability column names to extract.
    invert_probs : list, optional
        Column names where values represent noncoding probability (to be inverted to coding).

    Returns:
    --------
    pd.DataFrame
        DataFrame with only the specified probability columns.
    """
    probs_df = df[prob_colnames].copy()
    print(f"Extracted {probs_df.shape[1]} probability columns.")
    if invert_probs is not None:
        print("Inverting noncoding probabilities...")
        for col in invert_probs:
            if col in probs_df.columns:
                print(f"  - Inverting column: {col}")
                probs_df[col] = 1 - probs_df[col]
    return probs_df


def calculate_ensemble_entropy(probs_df, noncoding_prob_cols_to_invert=None):
    """
    Calculate classification entropy from ensemble tool probabilities.

    Parameters:
    -----------
    probs_df : pd.DataFrame
        DataFrame with probability columns for each tool.
    noncoding_prob_cols_to_invert : list, optional
        Column names where values represent noncoding probability (to be inverted to coding).

    Returns:
    --------
    pd.DataFrame
        Original DataFrame with added entropy calculations:
        - mean_prob: average probability across tools
        - mean_inv_prob: inverse of mean_prob
        - entropy: Shannon entropy of [mean_inv_prob, mean_prob]
    """
    if noncoding_prob_cols_to_invert is None:
        noncoding_prob_cols_to_invert = []

    result = probs_df.copy()

    # Invert noncoding probabilities
    for col in noncoding_prob_cols_to_invert:
        if col in result.columns:
            result[col] = 1 - result[col]

    # Calculate ensemble statistics
    result["mean_prob"] = result.mean(axis=1)
    result["mean_inv_prob"] = 1 - result["mean_prob"]
    result["entropy"] = scipy_entropy(
        np.array(result[["mean_inv_prob", "mean_prob"]]).T, base=2
    )

    return result


def reduce_dimensions_pca(scaled_features, variance_explained=0.95, random_state=42):
    """
    Reduce feature dimensionality using PCA.

    Parameters:
    -----------
    scaled_features : np.ndarray
        Scaled feature array.
    variance_explained : float, default=0.95
        Target cumulative explained variance ratio.
    random_state : int, default=42
        Random seed for reproducibility.

    Returns:
    --------
    tuple
        (pca_features: np.ndarray, pca_model: PCA)
    """
    pca = PCA(n_components=variance_explained, random_state=random_state)
    features_pca = pca.fit_transform(scaled_features)

    print(f"PCA reduced features to {features_pca.shape[1]} dimensions")
    print(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")

    return features_pca, pca
