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


# ============================================================================
# SUPPLEMENTARY PIPELINE COVERAGE CHECK
# ============================================================================


TRACE_COLS = ["category", "pipeline", "detail"]


def _trace_rows(
    idx: pd.Index, category: str, pipeline: str, detail: str
) -> pd.DataFrame:
    """One long-format traceability block: rows in `idx`, all sharing category/pipeline/detail."""
    return pd.DataFrame(
        {"category": category, "pipeline": pipeline, "detail": detail},
        index=pd.Index(idx, name="seq_ID"),
    )


def build_supplementary_traceability(
    main_index: pd.Index,
    pipeline_dfs: dict[str, pd.DataFrame],
) -> tuple[pd.Index, pd.DataFrame]:
    """
    Build a full data-traceability report for the supplementary feature merge.

    Every transcript that is *not* carried into the clean merged matrix is
    recorded with an explicit reason, so nothing is silently dropped or
    zero-imputed.  Three categories are distinguished:

    ``missing_from_pipeline``
        A main-index transcript is absent from an active pipeline.  Zero-imputing
        its features (the old downstream ``fillna(0)`` behaviour) biases effect
        sizes toward zero, so it is excluded from the clean set instead.
    ``not_in_main_index``
        A pipeline supplies a transcript that is absent from the main analysis
        set.  Informational — it was never eligible and is dropped from the merge.
    ``invalid_data``
        A main-index transcript is present in a pipeline but carries missing or
        non-numeric feature values; it is excluded rather than coerced to zero.

    Parameters
    ----------
    main_index : pd.Index
        Transcript IDs in the main analysis set.
    pipeline_dfs : dict[str, pd.DataFrame]
        Mapping of pipeline label → loaded feature DataFrame.  Empty DataFrames
        (not configured / not found) are skipped.

    Returns
    -------
    clean_index : pd.Index
        Main transcripts present with valid data in every active pipeline.
    report : pd.DataFrame
        Long-format, one row per (transcript, issue); indexed by seq_ID with
        columns ``category``, ``pipeline``, ``detail``.  Empty (with those
        columns) when nothing is dropped.
    """
    active = {name: df for name, df in pipeline_dfs.items() if len(df) > 0}
    empty_report = pd.DataFrame(columns=TRACE_COLS)
    empty_report.index.name = "seq_ID"

    if not active:
        return main_index, empty_report

    parts: list[pd.DataFrame] = []
    excluded_main = main_index[:0]  # empty, same dtype — main transcripts to drop

    for name, df in active.items():
        # pd.Index.isin is hash-based (ms); np.isin on string arrays falls back to a
        # sort-based path costing >90s per call — keep isin on the hot path.

        # (1) main transcripts absent from this pipeline
        missing = main_index[~main_index.isin(df.index)]
        if len(missing):
            parts.append(
                _trace_rows(
                    missing,
                    "missing_from_pipeline",
                    name,
                    f"Absent from pipeline '{name}'",
                )
            )
            excluded_main = excluded_main.union(missing)

        # (2) pipeline transcripts absent from the main index (never eligible)
        extra = pd.Index(df.index[~df.index.isin(main_index)]).unique()
        if len(extra):
            parts.append(
                _trace_rows(
                    extra,
                    "not_in_main_index",
                    name,
                    f"Present in pipeline '{name}' but absent from main index",
                )
            )

        # (3) main transcripts present here but with missing/non-numeric values
        present = df.index.intersection(main_index)
        # ponytail: feature matrices are numeric; to_numeric(coerce)+isna catches both
        # NaN and stray non-numeric strings. If a column is ever legitimately categorical
        # this over-flags — narrow to that column's dtype then.
        numeric = df.loc[present].apply(pd.to_numeric, errors="coerce")
        bad = present[numeric.isna().any(axis=1).to_numpy()]
        if len(bad):
            parts.append(
                _trace_rows(
                    bad,
                    "invalid_data",
                    name,
                    f"Missing/non-numeric feature value(s) in pipeline '{name}'",
                )
            )
            excluded_main = excluded_main.union(pd.Index(bad))

    clean_index = main_index[~main_index.isin(excluded_main)]

    if not parts:
        return clean_index, empty_report

    report = pd.concat(parts)
    counts = report.groupby("category").size().to_dict()
    print(
        f"⚠ Traceability: {len(clean_index):,}/{len(main_index):,} main transcripts kept; "
        + ", ".join(f"{k}={v}" for k, v in counts.items())
    )
    return clean_index, report


# ============================================================================
# SUPPLEMENTARY PIPELINE LOADER
# ============================================================================


def _read_pipeline_file(path: str, sep: str = ",", label: str = "") -> pd.DataFrame:
    """Read a supplementary pipeline CSV/TSV; return empty DataFrame if path is falsy or missing."""
    from pathlib import (  # local import — utils/features.py has no top-level Path import
        Path,
    )

    if not path:
        return pd.DataFrame()
    p = Path(path)
    if not p.exists():
        print(f"⚠  {label}: not found at {p} — skipping")
        return pd.DataFrame()
    df = pd.read_csv(p, sep=sep, index_col=0)
    # Some pipeline outputs use transcript_id as a column rather than the index
    if "transcript_id" in df.columns:
        df = df.set_index("transcript_id")
    print(f"   {label}: {df.shape[0]:,} rows × {df.shape[1]} cols")
    return df


def _clean_supplementary(df: pd.DataFrame) -> pd.DataFrame:
    """Drop non-feature label/ID columns and coerce presence flags to 0/1.

    ``transcript_type``/``coding_class`` are string metadata and the per-motif
    ``*_transcript_id`` columns are IDs — none are numeric features. Left in, they
    coerce to all-NaN and flag every transcript as ``invalid_data`` (emptying the
    merge). ``*_has_*`` presence flags are stored True/False/NaN where NaN means the
    element is absent (= 0), not a data gap — coerce them to int so 0 is the true
    value rather than a dropped transcript.
    """
    drop = [
        c
        for c in df.columns
        if c in ("transcript_type", "coding_class") or c.endswith("_transcript_id")
    ]
    df = df.drop(columns=drop)
    flags = df.columns[df.columns.str.contains("_has_")]
    if len(flags):
        df[flags] = (
            df[flags].apply(pd.to_numeric, errors="coerce").fillna(0).astype("int8")
        )
    return df


def load_supplementary_features(
    te_rna_path: str = "",
    te_dna_path: str = "",
    nbd_path: str = "",
    scanfold_path: str = "",
    rg4_path: str = "",
) -> dict[str, pd.DataFrame]:
    """
    Load supplementary pipeline feature files, applying per-pipeline index and
    column transformations.

    Drops non-feature label/ID columns and converts ``*_has_*`` presence flags to
    0/1 (NaN = element absent = 0) via :func:`_clean_supplementary`. Does **not**
    apply general ``fillna`` over measured features, numeric-type filtering, or
    ``remove_constant_features`` — those differ per downstream step and remain the
    caller's responsibility.

    Parameters
    ----------
    te_rna_path   : path to RNA/spliced TE features CSV
    te_dna_path   : path to DNA/unspliced TE features CSV
    nbd_path      : path to Non-B DNA features CSV
    scanfold_path : path to ScanFold features TSV
    rg4_path      : path to rG4detector features CSV

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys: ``te_rna``, ``te_dna``, ``nbd``, ``scanfold``, ``rg4``.
        Empty DataFrame for any disabled or missing pipeline.
        Suitable for direct use with :func:`build_supplementary_traceability`.

    Per-pipeline transformations
    ----------------------------
    te_rna   : drops ``transcript_length``, metadata/flag-cleans, prefixes columns with ``rna_``
    te_dna   : drops ``transcript_length`` (unspliced genomic length), metadata/flag-cleans, prefixes with ``dna_``
    nbd      : renames ``transcript_length`` → ``unspliced_length``, metadata/flag-cleans
    scanfold : strips ``.win*`` suffix from index; deduplicates; drops ``length``, ``source_dir``
    rg4      : strips ``|…`` from index (keeps transcript ID only); deduplicates; drops ``transcript_length``
    """
    te_rna = _read_pipeline_file(te_rna_path, sep=",", label="TE RNA")
    if not te_rna.empty:
        te_rna.drop(columns=["transcript_length"], errors="ignore", inplace=True)
        te_rna = _clean_supplementary(te_rna)
        te_rna.columns = [f"rna_{c}" for c in te_rna.columns]

    te_dna = _read_pipeline_file(te_dna_path, sep=",", label="TE DNA")
    if not te_dna.empty:
        te_dna.drop(columns=["transcript_length"], errors="ignore", inplace=True)
        te_dna = _clean_supplementary(te_dna)
        te_dna.columns = [f"dna_{c}" for c in te_dna.columns]

    nbd = _read_pipeline_file(nbd_path, sep=",", label="NBD")
    if not nbd.empty:
        nbd.rename(columns={"transcript_length": "unspliced_length"}, inplace=True)
        nbd = _clean_supplementary(nbd)

    scanfold = _read_pipeline_file(scanfold_path, sep="\t", label="ScanFold")
    if not scanfold.empty:
        # Index looks like "ENST00000831533.1.win_120.stp_1.csv" — strip window suffix
        scanfold.index = scanfold.index.str.split(".win").str[0]
        # ponytail: keep-first dedup; TODO remove when ScanFold pipeline stops emitting duplicates
        scanfold = scanfold[~scanfold.index.duplicated(keep="first")]
        scanfold.drop(columns=["length", "source_dir"], errors="ignore", inplace=True)

    rg4 = _read_pipeline_file(rg4_path, sep=",", label="rG4")
    if not rg4.empty:
        # Index looks like "ENST00000832824.1|ENSG...|..." — keep only the transcript ID
        rg4.index = rg4.index.str.split("|").str[0]
        rg4 = rg4[~rg4.index.duplicated(keep="first")]
        rg4.drop(columns=["transcript_length"], errors="ignore", inplace=True)

    return {
        "te_rna": te_rna,
        "te_dna": te_dna,
        "nbd": nbd,
        "scanfold": scanfold,
        "rg4": rg4,
    }
