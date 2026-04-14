import os
from typing import Dict, List, Set, Tuple

import pandas as pd
from Bio import SeqIO
from Bio.SeqIO.FastaIO import SimpleFastaParser
from Bio.SeqRecord import SeqRecord


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

    with open(file, "r") as handle:
        for title, seq in SimpleFastaParser(handle):
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
            "ID": ids,
            "gene_id": gene_ids,
            "description": descriptions,
            "biotype": biotypes,
            "sequence": sequences,
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
            - 'transcript_id': The unique identifier for each transcript.
            - 'gene_id': The gene identifier associated with the transcript.
            - 'biotype': The biotype of the transcript.
            - 'description': A description of the transcript.
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


def simple_load_ids(fasta_file: str) -> List[str]:
    """
    Load sequences from a FASTA file and return a list of ids.

    Args:
        fasta_file: Path to FASTA file

    Returns:
        List sequence ids
    """
    ids = []
    with open(fasta_file) as handle:
        for title, sequence in SeqIO.FastaIO.SimpleFastaParser(handle):
            ids.append(title)
    return ids


def parse_gencode_ids(record_ids: list, to_df=False):
    """
    Parse GENCODE transcript IDs to extract relevant information.

    Args:
        records: List of SeqRecord objects
    Returns:
        List of dictionaries with parsed information
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
