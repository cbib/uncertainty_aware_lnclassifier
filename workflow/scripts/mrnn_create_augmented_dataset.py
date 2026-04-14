import random

# Enable imports from workflow directory
import sys
from pathlib import Path

from Bio import SeqIO
from Bio.Seq import MutableSeq, Seq

sys.path.insert(0, str(Path(__file__).parent.parent))

# Setup logging
from utils.logging_utils import log_job_completion, setup_snakemake_logging

logger = setup_snakemake_logging(snakemake, script_name=__file__)

pc = snakemake.input["pc_train"]
lnc = snakemake.input["lnc_train"]
pc_augmented = snakemake.output["pc_augmented"]
lnc_augmented = snakemake.output["lnc_augmented"]
num_augmented = snakemake.params.get("num_augmented", 10)
SEED = snakemake.params.get("seed", 42)
random.seed(SEED)

logger.info(f"Random seed set to {SEED}")
logger.info(f"Number of augmented sequences per record: {num_augmented}")


def insert_random_nucleotide(sequence: Seq) -> MutableSeq:
    """
    Insert a random nucleotide (A, T, C, G) at a random position in the sequence.
    """
    nucleotides = ["A", "T", "C", "G"]
    insert_position = random.randint(0, len(sequence))
    random_nucleotide = random.choice(nucleotides)
    mutable_sequence = MutableSeq(sequence)
    mutable_sequence.insert(insert_position, random_nucleotide)
    return mutable_sequence


def create_augmented_record(record, id):
    augmented_record = record[:]  # Create a copy of the original record
    augmented_record.id = f"{record.id}_augmented_{id+1}_{SEED}"
    augmented_record.description = (
        f"{record.description} (augmented {id+1}, seed {SEED})"
    )
    augmented_record.seq = insert_random_nucleotide(record.seq)
    return augmented_record


def read_fasta_and_augment(input_fasta, output_fasta, num_augmented):
    augmented_records = []
    record_count = 0
    for record in SeqIO.parse(input_fasta, "fasta"):
        record_count += 1
        for i in range(num_augmented):
            augmented_records.append(create_augmented_record(record, i))

    with open(output_fasta, "w") as outfile:
        SeqIO.write(augmented_records, outfile, "fasta")


# Main execution
# Iterate over pc and lnc files to create augmented datasets
for input_fasta, output_fasta in [(pc, pc_augmented), (lnc, lnc_augmented)]:
    logger.info(f"Creating augmented dataset for {input_fasta}")
    read_fasta_and_augment(input_fasta, output_fasta, num_augmented)
    logger.info(f"Augmented dataset saved to {output_fasta}")

log_job_completion(logger)
