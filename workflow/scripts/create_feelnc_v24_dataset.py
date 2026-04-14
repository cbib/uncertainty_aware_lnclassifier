"""
Create FEELnc training and test datasets from GENCODE FASTA files.

This script:
1. Parses GENCODE FASTA files with biotype information
2. Filters for protein_coding and lncRNA (lincRNA/antisense) transcripts
3. Removes genes with ambiguous transcripts (both biotypes)
4. Splits selected transcripts into balanced train/test sets
5. Exports four FASTA files (lnc_train, lnc_test, pc_train, pc_test) and a TSV metadata file
"""

import logging
import random
import sys
from pathlib import Path
from typing import Dict, Set

import pandas as pd
from Bio import SeqIO

# Add workflow utilities to path
sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
from logging_utils import setup_basic_logging
from sklearn.model_selection import train_test_split

# Configure logging
logger = setup_basic_logging()
logger = logging.getLogger(__name__)

# Configuration
# TODO: Make these configurable via snakemake params and command-line
DB = "v24"
SEED = 42
NUMBER = 5000
INPUT_FASTA = Path("/mnt/cbib/LNClassifier/paper/resources/gencode.v24.transcripts.fa")
OUTPUT_DIR = Path("/mnt/cbib/LNClassifier/paper/resources/training/FEELnc_datasets")

random.seed(SEED)


def parse_fasta_header(header: str) -> Dict[str, str]:
    """
    Parse GENCODE FASTA header format.

    Args:
        header: FASTA header string (e.g., "ENST...|ENSG...|...|biotype|...")

    Returns:
        Dictionary with 'id', 'gene_id', and 'biotype' keys
    """
    parts = header.split("|")
    return {"id": parts[0], "gene_id": parts[1], "biotype": parts[-2]}


logger.info("Reading FASTA file...")
records = [r for r in SeqIO.parse(INPUT_FASTA, "fasta")]
logger.info(f"Total records read: {len(records)}")

logger.info("Processing records...")
data = [parse_fasta_header(r.id) for r in records]

df = pd.DataFrame(data)
logger.info(f"Initial dataframe shape: {df.shape}")

df = df[df["biotype"].isin(["protein_coding", "lincRNA", "antisense"])]
logger.info(f"Filtered dataframe shape: {df.shape}")

lncs = df[df["biotype"].isin(["lincRNA", "antisense"])]
pcs = df[df["biotype"].isin(["protein_coding"])]

logger.info(f"Number of lncRNA/antisense genes: {len(lncs)}")
logger.info(f"Number of protein_coding genes: {len(pcs)}")

unique_lnc_genes: Set[str] = set(lncs["gene_id"])
unique_pc_genes: Set[str] = set(pcs["gene_id"])

# These genes have both protein_coding and lincRNA/antisense transcripts (ambiguous)
common_genes = unique_lnc_genes.intersection(unique_pc_genes)
logger.info(f"Number of common genes (ambiguous): {len(common_genes)}")

# Remove ambiguous genes from the original dataset
unique_lnc_genes = unique_lnc_genes - common_genes
unique_pc_genes = unique_pc_genes - common_genes

logger.info(f"Unique lncRNA genes after filtering: {len(unique_lnc_genes)}")
logger.info(f"Unique protein_coding genes after filtering: {len(unique_pc_genes)}")

# Choose subset of NUMBER*2 genes for each class
if len(unique_lnc_genes) < NUMBER * 2 or len(unique_pc_genes) < NUMBER * 2:
    raise ValueError(
        f"Not enough unique genes to sample the required subsets. "
        f"Need {NUMBER * 2}, have lncRNA: {len(unique_lnc_genes)}, "
        f"protein_coding: {len(unique_pc_genes)}"
    )

lnc_gene_subset = random.sample(sorted(unique_lnc_genes), NUMBER * 2)
pc_gene_subset = random.sample(sorted(unique_pc_genes), NUMBER * 2)

logger.info(f"Selected lncRNA gene subset: {len(lnc_gene_subset)}")
logger.info(f"Selected protein_coding gene subset: {len(pc_gene_subset)}")

# Select one random transcript per gene
transcripts = df.groupby("gene_id").sample(1)[["id", "gene_id"]].reset_index(drop=True)
logger.info(f"Number of transcripts after grouping: {len(transcripts)}")

# Get the associated transcripts
lnc_transcripts = transcripts[transcripts["gene_id"].isin(lnc_gene_subset)][
    "id"
].tolist()
pc_transcripts = transcripts[transcripts["gene_id"].isin(pc_gene_subset)]["id"].tolist()

logger.info(f"Number of lncRNA transcripts: {len(lnc_transcripts)}")
logger.info(f"Number of protein_coding transcripts: {len(pc_transcripts)}")

if len(lnc_transcripts) < NUMBER * 2:
    raise ValueError(
        f"Not enough lncRNA transcripts to sample the required subsets. Found: {len(lnc_transcripts)}, Required: {NUMBER * 2}"
    )
elif len(pc_transcripts) < NUMBER * 2:
    raise ValueError(
        f"Not enough protein_coding transcripts to sample the required subsets. Found: {len(pc_transcripts)}, Required: {NUMBER * 2}"
    )

# Split the lists into two train/test chunks
lnc_train, lnc_test = train_test_split(
    lnc_transcripts, test_size=0.5, random_state=SEED
)
pc_train, pc_test = train_test_split(pc_transcripts, test_size=0.5, random_state=SEED)

lnc_train = set(lnc_train)
lnc_test = set(lnc_test)
pc_train = set(pc_train)
pc_test = set(pc_test)

logger.info(f"lncRNA train size: {len(lnc_train)}, test size: {len(lnc_test)}")
logger.info(f"protein_coding train size: {len(pc_train)}, test size: {len(pc_test)}")

# Create a table with IDs, train/test label, and biotype
# train_test_data = (
#     [(t, "train", "lincRNA") for t in lnc_train] +
#     [(t, "test", "lincRNA") for t in lnc_test] +
#     [(t, "train", "protein_coding") for t in pc_train] +
#     [(t, "test", "protein_coding") for t in pc_test]
# )

train_test_data = {
    "ID": list(lnc_train) + list(pc_train) + list(lnc_test) + list(pc_test),
    "Label": ["train"] * NUMBER * 2 + ["test"] * NUMBER * 2,
    "Biotype": (["lincRNA"] * NUMBER + ["protein_coding"] * NUMBER) * 2,
}

train_test_df = pd.DataFrame(train_test_data, columns=["ID", "Label", "Biotype"])
logger.info(f"Train/test dataframe shape: {train_test_df.shape}")

# Export the table to a TSV file
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
tsv_path = OUTPUT_DIR / "train_test_table.tsv"
train_test_df.to_csv(tsv_path, sep="\t", index=False)
logger.info(f"Train/test table saved to {tsv_path}")

# Evaluate the seqrecord list and export 4 fasta files
print("Writing FASTA files...")
with open(
    OUTPUT_DIR / f"FEELnc_{DB}_{NUMBER}_lnc_train.fasta", "w"
) as lnc_train_file, open(
    OUTPUT_DIR / f"FEELnc_{DB}_{NUMBER}_lnc_test.fasta", "w"
) as lnc_test_file, open(
    OUTPUT_DIR / f"FEELnc_{DB}_{NUMBER}_pc_train.fasta", "w"
) as pc_train_file, open(
    OUTPUT_DIR / f"FEELnc_{DB}_{NUMBER}_pc_test.fasta", "w"
) as pc_test_file, open(
    OUTPUT_DIR / f"FEELnc_{DB}_{NUMBER}_all_test.fasta", "w"
) as all_file:

    for record in records:
        transcript_id = record.id.split("|")[0]
        if transcript_id in lnc_train:
            SeqIO.write(record, lnc_train_file, "fasta")
        elif transcript_id in lnc_test:
            SeqIO.write(record, lnc_test_file, "fasta")
            SeqIO.write(record, all_file, "fasta")
        elif transcript_id in pc_train:
            SeqIO.write(record, pc_train_file, "fasta")
        elif transcript_id in pc_test:
            SeqIO.write(record, pc_test_file, "fasta")
            SeqIO.write(record, all_file, "fasta")

print("Datasets created successfully.")
