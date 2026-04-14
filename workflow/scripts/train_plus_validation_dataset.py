# This script gets a training dataset (FASTA) as input
# and produces a new dataset split between training and validation
# Enable imports from workflow directory
import sys
from pathlib import Path

from Bio import SeqIO
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent))

# Setup logging
from utils.logging_utils import log_job_completion, setup_snakemake_logging

logger = setup_snakemake_logging(snakemake, script_name=__file__)

# Parameters
SEED = snakemake.params["seed"]  # If it is None, dataset must be pre-shuffled
train_split = snakemake.params["train_split"] / 100
valid_split = snakemake.params["valid_split"] / 100
split = valid_split / train_split
# Train and valid splits are given out of the total dataset
# Input datasets contain only training transcripts for tools not needing validation
# We must calculate the subset of total training transcripts to be reserved for validation
# e.g.
# Tool without validation: train=80%, test=20%
# Tool with validation: train=60%, valid=20%, test=20%
# Calculate the share of the training transcripts to be used for validation:
# split = valid_split / train_split = 20% / 80% = 0.25
# Thus, to have a final validation set of 20%, 0.25 of the training transcripts will be used for validation

pc_fasta = snakemake.input["pc"]
lnc_fasta = snakemake.input["lnc"]
pc_train = snakemake.output["pc_train"]
lnc_train = snakemake.output["lnc_train"]
pc_valid = snakemake.output["pc_valid"]
lnc_valid = snakemake.output["lnc_valid"]

logger.info(
    f"Starting dataset split with seed: {SEED}, train_split: {train_split}, valid_split: {valid_split}, split ratio: {split}"
)

# Read input fasta
for file in [pc_fasta, lnc_fasta]:
    logger.info(f"Processing file: {file}")
    records = list(SeqIO.parse(file, "fasta"))
    total_records = len(records)
    logger.info(f"Total records in {file}: {total_records}")

    valid_count = int(
        total_records * split
    )  # Pass validation size as count instead of as a fraction
    train, valid = train_test_split(
        records, test_size=valid_count, random_state=SEED
    )  # test_size stands for validation size here

    train_count = len(train)

    assert (
        train_count + len(valid) == total_records
    ), "Record count mismatch after split"

    logger.info(
        f"Split completed for {file}:\n  - {train_count} training records\n  - {valid_count} validation records."
    )

    # Write output files
    if file == pc_fasta:
        SeqIO.write(train, pc_train, "fasta")
        SeqIO.write(valid, pc_valid, "fasta")
        logger.info(
            f"Training and validation datasets written for {file}: {pc_train}, {pc_valid}"
        )
    else:
        SeqIO.write(train, lnc_train, "fasta")
        SeqIO.write(valid, lnc_valid, "fasta")
        logger.info(
            f"Training and validation datasets written for {file}: {lnc_train}, {lnc_valid}"
        )

    logger.info(
        f"File {file} processed successfully: {total_records} total records, {train_count} training records, {valid_count} validation records."
    )

logger.info("Dataset split completed successfully.")

log_job_completion(logger)
