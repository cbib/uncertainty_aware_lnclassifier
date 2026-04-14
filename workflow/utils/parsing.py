import os
from typing import Dict, List, Set, Tuple

import pandas as pd
from Bio import SeqIO
from Bio.SeqIO.FastaIO import SimpleFastaParser
from Bio.SeqRecord import SeqRecord

_LIBRARY_DIR = os.path.dirname(os.path.abspath(__file__))
_RESULTS_DIR = os.path.join(_LIBRARY_DIR, "..", "..", "results")
_RESOURCES_DIR = os.path.join(_LIBRARY_DIR, "..", "..", "resources")

TABLE_PATHS = {
    "full_table": "tables/{dataset}_full_table.tsv",
    "binary": "tables/{dataset}_binary_class_table.tsv",
    # "dataset": "tables/{dataset}.dataset_info.tsv",
    # "inference": "training/{dataset}.inference_info.tsv",
}

# Default tables to load if none specified
DEFAULT_TABLES = ["full_table", "binary"]

PC_FASTA = os.path.join(_RESOURCES_DIR, "gencode.v47.pc_transcripts.fa")
LNC_FASTA = os.path.join(_RESOURCES_DIR, "gencode.v47.lncRNA_transcripts.fa")


def gencode_fasta_to_df(file):
    """
    Parses a GENCODE FASTA file and converts it into a pandas DataFrame.
    The function reads a FASTA file where each sequence header contains metadata
    separated by the '|' character. It extracts the following fields from the header:
    - ID: The first field in the header.
    - gene_id: The second field in the header.
    - description: The concatenated fields between the second and the second-to-last field.
    - biotype: The second-to-last field in the header.
    - sequence: The actual sequence from the FASTA file.
    Args:
        file (str): Path to the GENCODE FASTA file.
    Returns:
        pandas.DataFrame: A DataFrame with the following columns:
            - 'ID': The unique identifier for each sequence.
            - 'gene_id': The gene identifier associated with the sequence.
            - 'description': A description of the sequence.
            - 'biotype': The biotype of the sequence.
            - 'sequence': The nucleotide or protein sequence.
    """
    # Initialize lists for each column
    ids = []
    gene_ids = []
    descriptions = []
    biotypes = []
    sequences = []
    titles = []

    with open(file, "r") as handle:
        for title, seq in SimpleFastaParser(handle):
            titles.append(title)
            fields = title.split("|")

            # Append to separate lists
            ids.append(fields[0])
            gene_ids.append(fields[1])
            descriptions.append(" ".join(fields[2:-2]))
            biotypes.append(fields[-2])
            sequences.append(seq)

    # Create DataFrame once from dict of lists
    return pd.DataFrame(
        {
            "transcript_id": ids,
            "gene_id": gene_ids,
            "description": descriptions,
            "biotype": biotypes,
            "sequence": sequences,
            "full_id": titles,
        }
    )


def gencode_gtf_to_df(file):
    """
    Parses a GENCODE GTF file and converts it into a pandas DataFrame.
    The function reads a GTF file, which is a tab-separated format used for
    storing genomic features. It extracts the following columns:
    - seqname: The name of the sequence (chromosome or scaffold).
    - source: The source of the feature (e.g., Ensembl).
    - feature: The type of feature (e.g., gene, transcript, exon).
    - start: The starting position of the feature.
    - end: The ending position of the feature.
    - score: A score associated with the feature (can be '.' if not applicable).
    - strand: The strand of the feature ('+' or '-').
    - frame: The reading frame of the feature (can be '.' if not applicable).
    - attributes: A semicolon-separated list of key-value pairs providing additional information about the feature.
    Args:
        file (str): Path to the GENCODE GTF file.
    Returns:
        pandas.DataFrame: A DataFrame with the following columns:
            - 'seqname': The name of the sequence.
            - 'source': The source of the feature.
            - 'feature': The type of feature.
            - 'start': The starting position of the feature.
            - 'end': The ending position of the feature.
            - 'score': The score associated with the feature.
            - 'strand': The strand of the feature.
            - 'frame': The reading frame of the feature.
            - 'attributes': Additional attributes as a single string.
    """
    if file.endswith(".gz"):
        compression = "gzip"
    else:
        compression = None
    return pd.read_csv(
        file,
        sep="\t",
        comment="#",
        header=None,
        names=[
            "seqname",
            "source",
            "feature",
            "start",
            "end",
            "score",
            "strand",
            "frame",
            "attributes",
        ],
        compression=compression,
    )


def gencode_gtf_to_transcripts_df(file):
    """
    Parses a GENCODE GTF file and extracts transcript-level information into a pandas DataFrame.
    The function reads a GTF file, filters for transcript features, and extracts relevant
    information including transcript ID, gene ID, biotype, and description from the attributes column.
    Args:
        file (str): Path to the GENCODE GTF file.
    Returns:
        pandas.DataFrame: A DataFrame with the following columns:
            - 'seqname': The name of the sequence.
            - 'source': The source of the feature.
            - 'feature': The type of feature.
            - 'start': The starting position of the feature.
            - 'end': The ending position of the feature.
            - 'score': The score associated with the feature.
            - 'strand': The strand of the feature.
            - 'frame': The reading frame of the feature.
            - 'attributes': Additional attributes as a single string.
            - 'transcript_id': The unique identifier for each transcript.
            - 'gene_id': The gene identifier associated with the transcript.
            - 'biotype': The biotype of the transcript.
    """
    gtf_df = gencode_gtf_to_df(file)

    # Filter for transcript features
    transcripts_df = gtf_df[gtf_df["feature"] == "transcript"].copy()

    # Extract relevant attributes
    transcripts_df["gene_id"] = transcripts_df["attributes"].str.extract(
        r'gene_id "([^"]+)"'
    )
    transcripts_df["transcript_id"] = transcripts_df["attributes"].str.extract(
        r'transcript_id "([^"]+)"'
    )
    transcripts_df["biotype"] = transcripts_df["attributes"].str.extract(
        r'transcript_type "([^"]+)"'
    )
    return transcripts_df


def load_fasta(
    fasta_file: str, as_dict: bool = False
) -> List[SeqRecord] | Dict[str, SeqRecord]:
    """
    Load sequences from a FASTA file.

    Args:
        fasta_file: Path to FASTA file
        as_dict: If True, return a dictionary with sequence IDs as keys. Otherwise, return a list.

    Returns:
        List of SeqRecord objects
        or
        Dict of SeqRecord objects with string IDs as keys
    """
    if as_dict:
        return SeqIO.to_dict(SeqIO.parse(fasta_file, "fasta"))
    else:
        return list(SeqIO.parse(fasta_file, "fasta"))


def simple_load_ids(fasta_file: str, simple=False) -> List[str]:
    """
    Load sequences from a FASTA file and return a list of ids.

    Args:
        fasta_file: Path to FASTA file
        simple: If True, split ID by '|' and return only the simple ID

    Returns:
        List sequence ids
    """
    ids = []
    if fasta_file.endswith(".gz"):
        import gzip

        with gzip.open(fasta_file, "rt") as handle:
            for title, sequence in SeqIO.FastaIO.SimpleFastaParser(handle):
                ids.append(title)
    else:

        with open(fasta_file) as handle:
            for title, sequence in SeqIO.FastaIO.SimpleFastaParser(handle):
                ids.append(title)
    if simple:
        ids = [i.split("|")[0] for i in ids]

    return ids


def get_classification(seq_id, pc_fasta, lnc_fasta):
    """
    Determine if sequence is protein-coding or lncRNA based on Gencode reference files.

    Args:
        seq_id: Sequence ID to classify
        pc_fasta: Path to protein-coding reference FASTA file
        lnc_fasta: Path to lncRNA reference FASTA file

    Returns:
        str: "protein_coding", "lncRNA", or "other"
    NOTE: It is unlikely I will use this function, as it loads all sequences just for one ID check.
    It is preferred to use the one that annotates a full dataframe.
    """

    # Read reference files
    pc_ids = set(simple_load_ids(pc_fasta))
    lnc_ids = set(simple_load_ids(lnc_fasta))

    if seq_id in pc_ids:
        return "protein_coding"
    elif seq_id in lnc_ids:
        return "lncRNA"
    else:
        return "other"


def classify_transcripts(
    df: pd.DataFrame,
    pc_ids: Set[str],
    lnc_ids: Set[str],
    simple_id=False,
    version=True,
    id_column="transcript_id",
) -> pd.DataFrame:
    """
    Classify transcripts as coding, lncRNA, or other based on the provided ID sets

    Args:
        df: DataFrame with transcript_id column
        pc_ids: Set of protein-coding transcript IDs
        lnc_ids: Set of lncRNA transcript IDs
        version: Whether transcript IDs include version numbers (default: True)

    Returns:
        DataFrame with added 'coding_class' column
    """
    # Initialize all as 'other'
    df["coding_class"] = "other"

    # Extract base transcript ID
    if not simple_id:
        # Extract raw gencode IDs (with full metadata)
        base_ids = df[id_column]
    else:
        # Extract ENST IDs from full IDs
        base_ids = df[id_column].str.split("|").str[0]
        pc_ids = [i.split("|")[0] for i in pc_ids]
        lnc_ids = [i.split("|")[0] for i in lnc_ids]
        if not version:
            # Further strip version suffix
            base_ids = base_ids.str.split(".").str[0]
            pc_ids = [i.split(".")[0] for i in pc_ids]
            lnc_ids = [i.split(".")[0] for i in lnc_ids]

    # Classify coding transcripts
    coding_mask = base_ids.isin(pc_ids)
    df.loc[coding_mask, "coding_class"] = "coding"
    n_coding = coding_mask.sum()

    # Classify lncRNA transcripts
    lnc_mask = base_ids.isin(lnc_ids)
    df.loc[lnc_mask, "coding_class"] = "lncRNA"
    n_lnc = lnc_mask.sum()

    # Count other
    n_other = (df["coding_class"] == "other").sum()
    print(f"Classified transcripts: {n_coding} coding, {n_lnc} lncRNA, {n_other} other")
    return df


def classify_transcripts_from_files(
    df: pd.DataFrame,
    pc_fasta: str = PC_FASTA,
    lnc_fasta: str = LNC_FASTA,
    simple_id=True,
    version=True,
    id_column="transcript_id",
) -> pd.DataFrame:
    """
    Classify transcripts as coding, lncRNA, or other based on the provided ID sets
    NOTE: Full Gencode IDs change between files. Use of simple_id=True is recommended.

    Args:
        df: DataFrame with transcript_id column
        pc_fasta: Path to protein-coding reference FASTA file
        lnc_fasta: Path to lncRNA reference FASTA file
        version: Whether transcript IDs include version numbers (default: True)

    Returns:
        DataFrame with added 'coding_class' column
    """
    pc_ids = set(simple_load_ids(pc_fasta))
    lnc_ids = set(simple_load_ids(lnc_fasta))
    return classify_transcripts(df, pc_ids, lnc_ids, simple_id, version, id_column)


def parse_gencode_ids(record_ids: list, to_df=False):
    """
    Parse GENCODE transcript IDs to extract relevant information.

    Args:
        records: List of SeqRecord objects
        to_df: If True, return a pandas DataFrame. Otherwise, return lists.
    Returns:
        List of lists with parsed information
        0: full_ids,
        1: seq_ids, list of sequence IDs (first field in the FASTA header)
        2: clean_ids, list of IDs without version
        3: biotypes, list of biotypes (last field in the FASTA header)

    """
    full_ids = []
    seq_ids = []
    clean_ids = []
    id_versions = []
    biotypes = []

    for full_id in record_ids:
        parts = full_id.split("|")
        full_ids.append(full_id)
        seq_ids.append(parts[0])
        id_parts = parts[0].split(".")
        clean_ids.append(id_parts[0])
        id_versions.append(id_parts[1] if len(id_parts) > 1 else "")
        biotypes.append(parts[-2])

    if to_df:
        df = pd.DataFrame(
            {
                "full_id": full_ids,
                "seq_id": seq_ids,
                "clean_id": clean_ids,
                "id_version": id_versions,
                "biotype": biotypes,
            }
        )
        df.set_index("clean_id", inplace=True)
        return df

    return full_ids, seq_ids, clean_ids, biotypes


def write_fasta(sequences, output_file):
    """
    Write sequences to a FASTA file.

    Args:
        sequences: List of SeqRecord objects
        output_file: Path to output FASTA file
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    SeqIO.write(sequences, output_file, "fasta")


def get_dataset_subset(df: pd.DataFrame, subset: str, dset: str) -> pd.DataFrame:
    """
    Get a subset of the DataFrame based on the specified criteria.

    Args:
        df: Input DataFrame with 'coding_class' column
        subset: Subset criteria ('all', 'pc', 'lncRNA', 'other')
    """
    # TODO: Generalize path handling
    dataset_info = (
        f"/mnt/cbib/LNClassifier/paper/results/{dset}/training/{dset}.dataset_info.tsv"
    )
    if not os.path.exists(dataset_info):
        raise FileNotFoundError(f"Dataset info file not found: {dataset_info}")

    allowed_subsets = {
        "pc",
        "lncRNA",
        "test",
        "redun",
        "added",
        "no_class",
        "class_change",
        "",
    }

    # Validate subset
    if subset not in allowed_subsets and subset != "all":
        raise ValueError(
            f"Invalid subset: {subset}. Allowed values are: {allowed_subsets} or 'all'."
        )

    # Load table
    dataset_df = pd.read_csv(dataset_info, sep="\t")
    dataset_df["simple_ID"] = dataset_df["transcript_id"].str.split("|").str[0]

    # Define subset mappings using vectorized operations
    if subset == "all":
        selected_ids = set(dataset_df["simple_ID"])
    elif subset in ["pc", "lncRNA", "other"]:
        coding_class_map = {"pc": "coding", "lncRNA": "lncRNA", "other": "other"}
        coding_class = coding_class_map[subset]
        selected_ids = set(
            dataset_df.loc[dataset_df["coding_class"] == coding_class, "simple_ID"]
        )
    elif subset == "class_change":
        selected_ids = set(
            dataset_df.loc[
                dataset_df["source_file"].str.contains("class_change"), "simple_ID"
            ]
        )
    else:
        selected_ids = set(
            dataset_df.loc[
                dataset_df["source_file"].str.startswith(subset), "simple_ID"
            ]
        )

    return df[df.index.isin(selected_ids)].copy()


def load_tables(
    dataset: str, table_names: List[str] = None, all: bool = False
) -> Dict[str, pd.DataFrame]:
    """
    Load multiple analysis tables for a given dataset.

    Args:
        dataset: Dataset identifier (e.g., 'gencode.v47.cdhit')
        table_names: List of table names to load (keys from TABLE_PATHS).
                    If None, uses DEFAULT_TABLES: {DEFAULT_TABLES}
        all: If True, load all tables in TABLE_PATHS (overrides table_names)

    Returns:
        Dictionary mapping table names to pandas DataFrames

    Raises:
        FileNotFoundError: If any requested table file is not found
        ValueError: If dataset or table_names parameters are invalid
    """
    if not dataset or not isinstance(dataset, str):
        raise ValueError("dataset must be a non-empty string")

    # Load all tables if requested
    if all:
        table_names = list(TABLE_PATHS.keys())
    # Use default tables if none specified
    elif table_names is None:
        table_names = DEFAULT_TABLES

    if not table_names or not isinstance(table_names, list):
        raise ValueError("table_names must be a non-empty list")

    base_path = os.path.join(_RESULTS_DIR, dataset)

    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Dataset directory not found: {base_path}")

    tables = {}

    for table_name in table_names:
        if table_name not in TABLE_PATHS:
            raise ValueError(
                f"Unknown table name: '{table_name}'. Available tables: {list(TABLE_PATHS.keys())}"
            )

        # Build file path from template
        file_path = os.path.join(
            base_path, TABLE_PATHS[table_name].format(dataset=dataset)
        )

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Table file not found: {file_path}")

        # Load table with appropriate delimiter
        sep = "\t" if file_path.endswith(".tsv") else ","
        tables[table_name] = pd.read_csv(file_path, sep=sep)

    return tables
