"""
Script modular para agregar resultados de clasificación de transcritos
de múltiples herramientas en tablas unificadas.
"""

import gzip
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from Bio import SeqIO
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

# Add scripts directory to path for utils import
scripts_dir = "/mnt/cbib/LNClassifier/paper/workflow/"
sys.path.insert(0, str(scripts_dir))

from utils.logging_utils import (
    log_job_completion,
    setup_basic_logging,
    setup_snakemake_logging,
)

# Setup module-wide logging
if "snakemake" in globals():
    logger = setup_snakemake_logging(snakemake, script_name=__file__)
else:
    logger = setup_basic_logging()


# ============================================================================
# CONFIGURATION
# ============================================================================

CATEGORY_MAP = {
    "lncrna": "noncoding",
    "mrna": "coding",
    "pcRNA": "coding",
    "ncRNA": "noncoding",
    0: "noncoding",
    1: "coding",
    "non-coding": "noncoding",
}

TOOL_THRESHOLDS = {
    "cpat": 0.364,  # Default CPAT cutoff for human transcripts
    "mrnn": 0.5,
    "lncrnanet": 0.5,
}

LABEL_COLUMN_RENAMES = {
    "predict_lncDC": "label_lncDC",
    "predict_ss_lncDC": "label_ss_lncDC",
    "classification_rnasamba": "label_rnasamba",
    "class_lncrnabert": "label_lncrnabert",
}


# ============================================================================
# TOOL PROCESSORS
# ============================================================================


def process_cpc2(file_path: str) -> pd.DataFrame:
    """Process CPC2 classification results."""
    logger.info(f"Processing CPC2 results from {file_path}")
    df = pd.read_csv(file_path, sep="\t")
    df["cpat_merge_ID"] = df["#ID"].str.upper()
    df.columns = [f"{col}_cpc2" for col in df.columns]
    logger.info(f"  Loaded {len(df)} CPC2 predictions")
    return df


def process_feelnc(file_path: str) -> pd.DataFrame:
    """Process FEELnc classification results."""
    logger.info(f"Processing FEELnc results from {file_path}")
    df = pd.read_csv(file_path, sep="\t")
    df.rename(columns={"label": "num_label"}, inplace=True)
    labels = {0: "noncoding", 1: "coding"}
    df["label"] = df["num_label"].map(labels)
    df.columns = [f"{col}_feelnc" for col in df.columns]
    logger.info(f"  Loaded {len(df)} FEELnc predictions")
    return df


def process_cpat(
    cpat_p_path: str, cpat_l_path: str, cpat_cutoff_path: str = None
) -> pd.DataFrame:
    """Process CPAT classification results (both best-orf modes)."""
    logger.info(f"Processing CPAT results")
    logger.info(f"  Best-orf (p): {cpat_p_path}")
    logger.info(f"  Best-orf (l): {cpat_l_path}")
    if cpat_cutoff_path is not None:
        logger.info(f"  Cutoff file: {cpat_cutoff_path}")
        cutoff_df = pd.read_csv(cpat_cutoff_path, header=0, sep="\t")
        cutoff = cutoff_df["Cutoff"].iloc[0]
        logger.info(f"  Using CPAT cutoff: {cutoff}")
    else:
        cutoff = TOOL_THRESHOLDS["cpat"]
        logger.info(f"  No cutoff file provided, using default CPAT cutoff: {cutoff}")

    def _process_single_cpat(path: str) -> pd.DataFrame:
        df = pd.read_csv(path, sep="\t")
        df["ID"] = df["ID"].str.split("|").str[-1]
        df["biotype"] = df["seq_ID"].str.split("|").str[-2]
        df["label"] = df["Coding_prob"].apply(
            lambda x: "noncoding" if x < cutoff else "coding"
        )

        # CPAT does not return ORF coverage, just ORF start and end
        orf_len = df["ORF_end"] - df["ORF_start"] + 1
        df["ORF_coverage"] = orf_len / df["mRNA"]
        df.drop(
            columns=["ORF_start", "ORF_end", "ORF_frame", "ORF_strand"], inplace=True
        )
        return df

    cpat_p = _process_single_cpat(cpat_p_path)
    cpat_l = _process_single_cpat(cpat_l_path)

    merged = (
        pd.merge(cpat_p, cpat_l, on="seq_ID", how="outer", suffixes=("_p", "_l"))
        .drop(columns=["biotype_l", "mRNA_l"])
        .rename(columns={"biotype_p": "biotype", "mRNA_p": "mRNA"})
    )

    merged.columns = [f"{col}_cpat" for col in merged.columns]
    logger.info(f"  Loaded {len(merged)} CPAT predictions")
    return merged


def process_lncdc(no_ss_path: str, ss_path: str) -> pd.DataFrame:
    """Process LncDC classification results (with and without secondary structure)."""
    logger.info(f"Processing LncDC results")
    logger.info(f"  No SS: {no_ss_path}")
    logger.info(f"  With SS: {ss_path}")

    df_no_ss = pd.read_csv(no_ss_path)
    df_no_ss["predict"] = df_no_ss["predict"].map(CATEGORY_MAP)
    # NOTE: Parse using "ENSG" to make compatible with lncfinder-derived IDs. \
    # These are present if lncDC was run using lncfinder secondary structure files.
    df_no_ss["simple_ID"] = df_no_ss["Description"].str.split("ENSG").str[0].str[:-1]

    df_ss = pd.read_csv(ss_path)
    df_ss["predict"] = df_ss["predict"].map(CATEGORY_MAP)
    df_ss["simple_ID"] = df_ss["Description"].str.split("ENSG").str[0].str[:-1]

    ss_columns = [
        "simple_ID",
        "Description",
        "Noncoding_prob",
        "predict",
        "SS_score_k1",
        "SS_score_k2",
        "SS_score_k3",
        "SS_score_k4",
        "SS_score_k5",
        "GC_content_paired_ss",
    ]

    merged = pd.merge(
        df_no_ss, df_ss[ss_columns], on="simple_ID", how="outer", suffixes=("", "_ss")
    ).drop(columns=["simple_ID"])

    merged.columns = [f"{col}_lncDC" for col in merged.columns]
    logger.info(f"  Loaded {len(merged)} LncDC predictions")
    return merged


def process_mrnn(file_path: str) -> pd.DataFrame:
    """Process mRNN classification results."""
    logger.info(f"Processing mRNN results from {file_path}")
    df = pd.read_csv(
        file_path, sep="\t", names=["seq_ID", "coding_prob", "logit_coding_prob"]
    )
    df["label"] = df["coding_prob"].apply(
        lambda x: "noncoding" if x < TOOL_THRESHOLDS["mrnn"] else "coding"
    )
    df.columns = [f"{col}_mrnn" for col in df.columns]
    logger.info(f"  Loaded {len(df)} mRNN predictions")
    return df


def process_rnasamba(full_path: str, partial_path: str = None) -> pd.DataFrame:
    """Process RNAsamba classification results (full and partial modes)."""
    logger.info(f"Processing RNAsamba results")
    logger.info(f"  Full: {full_path}")

    df_full = pd.read_csv(full_path, sep="\t")

    if partial_path is not None:
        logger.info(f"  Partial: {partial_path}")
        df_partial = pd.read_csv(partial_path, sep="\t")
        df_full = df_full.merge(
            df_partial, on="sequence_name", how="outer", suffixes=("_full", "_partial")
        )

    df_full.columns = [f"{col}_rnasamba" for col in df_full.columns]
    logger.info(f"  Loaded {len(df_full)} RNAsamba predictions")
    return df_full


def process_lncrnabert(file_path: str) -> pd.DataFrame:
    """Process LncRNABERT classification results."""
    logger.info(f"Processing LncRNABERT results from {file_path}")
    df = pd.read_csv(file_path)
    df["class"] = df["class"].map(CATEGORY_MAP)
    df.columns = [f"{col}_lncrnabert" for col in df.columns]
    logger.info(f"  Loaded {len(df)} LncRNABERT predictions")
    return df


def process_plncpro(file_path: str) -> pd.DataFrame:
    """Process PLncPRO classification results."""
    logger.info(f"Processing PLncPRO results from {file_path}")
    df = pd.read_csv(
        file_path,
        sep="\t",
        header=0,
    )
    df.drop(columns=["Label"], inplace=True)
    df["prediction"] = df["prediction"].astype(int)
    df["label"] = df["prediction"].map({0: "noncoding", 1: "coding"})
    df.columns = [f"{col}_plncpro" for col in df.columns]
    logger.info(f"  Loaded {len(df)} PLncPRO predictions")
    return df


def process_lncadeep(file_path: str) -> pd.DataFrame:
    """Process LncADeep classification results."""
    logger.info(f"Processing LncADeep results from {file_path}")
    df = pd.read_csv(file_path, sep="\t", skiprows=1)
    df.columns = ["ID", "MajorityVoteNum", "Index"] + [
        f"ModelScore{i}" for i in range(1, 22)
    ]
    df["label"] = df["Index"].str.lower()
    df.columns = [f"{col}_lncadeep" for col in df.columns]
    logger.info(f"  Loaded {len(df)} LncADeep predictions")
    return df


def process_lncfinder(ss_path: str, no_ss_path: str) -> pd.DataFrame:
    """Process LncFinder classification results (with and without secondary structure)."""
    logger.info(f"Processing LncFinder results")
    logger.info(f"  With SS: {ss_path}")
    logger.info(f"  No SS: {no_ss_path}")

    df_ss = pd.read_csv(ss_path, sep="\t")
    df_ss.rename(columns={"Unnamed: 0": "ID"}, inplace=True)
    df_ss["label"] = df_ss["Pred"].str.lower()

    df_no_ss = pd.read_csv(no_ss_path, sep="\t")
    df_no_ss.rename(columns={"Unnamed: 0": "ID"}, inplace=True)
    df_no_ss["label"] = df_no_ss["Pred"].str.lower()

    # NOTE: IDs of ss dataframe substitute "|" separator by "."
    df_ss["simple_ID"] = df_ss["ID"].str.split("ENSG").str[0].str[:-1]
    df_ss.drop(columns=["ID"], inplace=True)
    df_no_ss["simple_ID"] = df_no_ss["ID"].str.split("|").str[0]

    merged = df_ss.merge(
        df_no_ss[["ID", "simple_ID", "Pred", "Coding.Potential", "label"]],
        on="simple_ID",
        how="outer",
        suffixes=("_ss", ""),
    ).drop(columns=["simple_ID"])

    merged.columns = [f"{col}_lncfinder" for col in merged.columns]
    logger.info(f"  Loaded {len(merged)} LncFinder predictions")
    return merged


def process_lncrnanet(file_path: str) -> pd.DataFrame:
    """Process LncRNAnet classification results."""
    logger.info(f"Processing LncRNAnet results from {file_path}")
    df = pd.read_csv(
        file_path, sep="\t", names=["ID", "transcript_length", "coding_prob"]
    )
    # NOTE: Higher probability means noncoding for this tool
    df["label"] = df["coding_prob"].apply(
        lambda x: "noncoding" if x > TOOL_THRESHOLDS["lncrnanet"] else "coding"
    )
    df.columns = [f"{col}_lncrnanet" for col in df.columns]
    logger.info(f"  Loaded {len(df)} LncRNAnet predictions")
    return df


# ============================================================================
# MERGE PIPELINE
# ============================================================================
@dataclass
class ToolConfig:
    """Configuration for merging a tool's results."""

    df: pd.DataFrame
    tool_name: str
    id_column: str


def merge_all_tools(tools: list[ToolConfig]) -> pd.DataFrame:
    """
    Merge all tool results into a single dataframe.

    The first tool in the list is used as the base dataframe,
    and all subsequent tools are merged onto it.

    Args:
        tools: List of tool configurations to merge (first one is the base)

    Returns:
        Combined dataframe with all tool results

    Raises:
        ValueError: If tools list is empty
    """
    if not tools:
        raise ValueError("tools list cannot be empty")

    logger.info("Merging all tool results...")
    base_config = tools[0]
    result = base_config.df.copy()
    # Create temporary column "simple ID" (i.e., only ENST number + version)
    result["temp_id"] = result[base_config.id_column].str.split("ENSG").str[0].str[:-1]
    base_id = base_config.id_column
    logger.info(f"  Using {base_config.tool_name} as base ({len(result)} rows)")

    for tool_config in tools[1:]:
        before_len = len(result)
        result = result.merge(
            tool_config.df,
            left_on="temp_id",
            right_on=tool_config.df[tool_config.id_column]
            .str.split("ENSG")
            .str[0]
            .str[:-1],
            how="outer",
        )
        logger.info(
            f"  Merged {tool_config.tool_name} ({len(result)} rows, +{len(result)-before_len})"
        )

        if tool_config.id_column != base_id:
            result.drop(columns=[tool_config.id_column], inplace=True)

    result = result.rename(columns={base_id: "seq_ID"}).set_index("seq_ID")
    logger.info(f"Combined dataframe: {len(result)} total transcripts")
    return result


# ============================================================================
# OUTPUT GENERATION
# ============================================================================


def add_metadata_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add metadata and biotype columns extracted from sequence IDs."""
    logger.info("Adding metadata columns...")
    df = df.copy()
    df["metadata"] = df.index
    df.index = df["metadata"].str.split("|").str[0]
    df.index.name = "seq_ID"
    df["biotype"] = df["metadata"].str.split("|").str[-2]
    return df


def create_classification_table(df: pd.DataFrame) -> pd.DataFrame:
    """Extract only classification labels and biotype."""
    logger.info("Creating classification table...")
    labels = df.filter(regex="^label_").columns.tolist()
    logger.info(f"  Found {len(labels)} classification label columns")
    return df[labels + ["biotype"]].copy()


def load_reference_transcripts(
    pc_path: str, lnc_path: str
) -> Tuple[list[str], list[str]]:
    """Load reference transcript IDs from FASTA files."""
    logger.info("Loading reference transcripts...")
    logger.info(f"  PC transcripts: {pc_path}")
    logger.info(f"  LncRNA transcripts: {lnc_path}")

    with gzip.open(pc_path, "rt") as pc_file:
        pcs = SeqIO.to_dict(SeqIO.parse(pc_file, "fasta"))
        pc_ids = [t.split("|")[0] for t in pcs.keys()]

    with gzip.open(lnc_path, "rt") as nc_file:
        lncs = SeqIO.to_dict(SeqIO.parse(nc_file, "fasta"))
        lnc_ids = [t.split("|")[0] for t in lncs.keys()]

    logger.info(f"  Loaded {len(pc_ids)} PC transcript IDs")
    logger.info(f"  Loaded {len(lnc_ids)} lncRNA transcript IDs")

    return pc_ids, lnc_ids


def create_simple_classification_table(
    class_df: pd.DataFrame, pc_ids: list[str], lnc_ids: list[str]
) -> pd.DataFrame:
    """Create simplified classification table with only reference transcripts."""
    logger.info("Creating simple classification table...")
    simple_df = class_df.copy()
    simple_df.loc[simple_df.index.isin(pc_ids), "label_real"] = "coding"
    simple_df.loc[simple_df.index.isin(lnc_ids), "label_real"] = "noncoding"

    before_len = len(simple_df)
    simple_df.dropna(subset=["label_real"], inplace=True)
    logger.info(
        f"  Kept {len(simple_df)} reference transcripts (dropped {before_len - len(simple_df)})"
    )

    return simple_df


def create_unclassified_table(simple_class_df: pd.DataFrame) -> pd.DataFrame:
    """Extract transcripts that were not classified by at least one tool."""
    logger.info("Creating unclassified transcripts table...")
    no_class_df = simple_class_df[simple_class_df.isna().any(axis=1)].copy()
    logger.info(f"  Found {len(no_class_df)} transcripts with missing classifications")
    return no_class_df


def create_binary_classification_table(simple_class_df: pd.DataFrame) -> pd.DataFrame:
    """Create binary classification table (False=noncoding, True=coding)."""
    logger.info("Creating binary classification table...")
    before_len = len(simple_class_df)
    binary_df = (
        simple_class_df.dropna()
        .replace({"coding": True, "noncoding": False})
        .rename(lambda x: x.removeprefix("label_"), axis=1)
        .drop(columns=["biotype"])
    )
    logger.info(
        f"  Binary table: {len(binary_df)} complete cases (dropped {before_len - len(binary_df)} with NaN)"
    )
    return binary_df


def create_all_tables(
    df: pd.DataFrame, pc_ids: list[str], lnc_ids: list[str], output_prefix: str
) -> dict[str, pd.DataFrame]:
    """Create all classification tables."""
    logger.info("Creating all classification tables...")

    # Create classification table
    class_df = create_classification_table(df)
    class_table_path = f"{output_prefix}_class_table.tsv"
    logger.info(f"Writing classification table to {class_table_path}")
    class_df.to_csv(class_table_path, sep="\t")

    # Create simple classification table
    simple_class_df = create_simple_classification_table(class_df, pc_ids, lnc_ids)
    simple_class_path = f"{output_prefix}_simple_class_table.tsv"
    logger.info(f"Writing simple classification table to {simple_class_path}")
    simple_class_df.to_csv(simple_class_path, sep="\t")

    # Create unclassified transcripts table
    no_class_df = create_unclassified_table(simple_class_df)
    no_class_path = f"{output_prefix}_no_class_table.tsv"
    logger.info(f"Writing unclassified table to {no_class_path}")
    no_class_df.to_csv(no_class_path, sep="\t")

    # Create binary classification table
    binary_class_df = create_binary_classification_table(simple_class_df)
    binary_class_path = f"{output_prefix}_binary_class_table.tsv"
    logger.info(f"Writing binary classification table to {binary_class_path}")
    binary_class_df.to_csv(binary_class_path, sep="\t")
    return {
        "classification_table": class_df,
        "simple_classification_table": simple_class_df,
        "unclassified_table": no_class_df,
        "binary_classification_table": binary_class_df,
    }


def get_classification_scores(
    df: pd.DataFrame, reference_column: str = "real", average: str = "binary"
) -> pd.DataFrame:
    """
    Get classification scores for each tool in the input dataframe.
    The reference labels are expected to be in a column named 'real'.

    :param df: Dataframe containing the classification results as columns, with one column per tool and a colum for the real lables, named 'real'.
    :type df: pd.DataFrame
    :param reference_column: Name of the column containing the reference labels.
    :type reference_column: str
    :param average: Averaging method for classification metrics. Can be 'binary', 'micro', 'macro', 'weighted' or 'samples'.
    :type average: str
    :return: DataFrame containing classification scores for each tool.
    :rtype: DataFrame
    """
    reference = df[reference_column]
    columns = [
        col
        for col in df.columns
        if col
        not in [reference_column, "agreement", "agreement_unique", "agreement_diff"]
    ]
    scores = pd.DataFrame(
        index=columns,
        columns=["accuracy", "balanced_accuracy", "precision", "recall", "f1_score"],
    )
    for column in columns:
        scores.loc[column, "accuracy"] = accuracy_score(
            reference, df[column], normalize=True
        )
        scores.loc[column, "balanced_accuracy"] = balanced_accuracy_score(
            reference, df[column]
        )
        scores.loc[column, "precision"] = precision_score(
            reference, df[column], average=average, zero_division=0
        )
        scores.loc[column, "recall"] = recall_score(
            reference, df[column], average=average, zero_division=0
        )
        scores.loc[column, "f1_score"] = f1_score(
            reference, df[column], average=average, zero_division=0
        )

    scores = scores.sort_values(by="f1_score", ascending=False)
    return scores
