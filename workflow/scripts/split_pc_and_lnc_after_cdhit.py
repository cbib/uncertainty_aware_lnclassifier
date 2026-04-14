import argparse
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

from Bio import SeqIO
from sklearn.model_selection import train_test_split

scripts_dir = Path(__file__).parent.parent
sys.path.insert(0, str(scripts_dir))

from utils.logging_utils import log_job_completion, setup_snakemake_logging
from utils.parsing import load_fasta, simple_load_ids, write_fasta

#!/usr/bin/env python3
"""
Script to process CD-HIT cluster file and split sequences into PC/LNC categories.
"""


def representative_ids_from_cdhit_clusters(cluster_file: str) -> List[str]:
    """
    Parse CD-HIT .clstr file

    Returns:
        list of representative transcript IDs
    """
    representative_ids = []

    logger.info(f"Parsing cluster file: {cluster_file}")
    with open(cluster_file, "r") as f:
        for line in f:
            line = line.strip()

            if line.startswith(">Cluster"):
                # New cluster
                continue
            elif line and line.endswith("*"):
                # Parse sequence entry
                parts = line.split(">")
                if len(parts) < 2:
                    continue
                # Extract sequence ID (handle different formats)
                seq_id_full = parts[1].split()[0]  # Get first part before space
                # Get simple transcript ID from GENCODE FASTA header format
                seq_id = (
                    seq_id_full.split("|")[0] if "|" in seq_id_full else seq_id_full
                )
                representative_ids.append(seq_id)

    return representative_ids


def parse_cdhit_cluster(cluster_file: str) -> Set[str]:
    """Extract representative transcript IDs from CD-HIT cluster file."""
    representatives = set()

    with open(cluster_file, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                continue
            # Representative sequences are marked with '*'
            if "*" in line:
                # Extract transcript ID from line like: "0	2345aa, >transcript_id... *"
                parts = line.split(">")
                if len(parts) > 1:
                    transcript_id = parts[1].split(".")[0].split()[0]
                    representatives.add(transcript_id)

    return representatives


def main():
    parser = argparse.ArgumentParser(
        description="Process CD-HIT clusters and split into PC/LNC train/test sets"
    )
    parser.add_argument(
        "-c", "--cluster", required=True, help="CD-HIT cluster file (.clstr)"
    )
    parser.add_argument("-f", "--fasta", required=True, help="Input FASTA file")
    parser.add_argument(
        "-p", "--pc-ids", required=True, help="File with PC transcript IDs"
    )
    parser.add_argument(
        "-l", "--lnc-ids", required=True, help="File with LNC transcript IDs"
    )
    parser.add_argument(
        "-o", "--output-prefix", required=True, help="Prefix for output files"
    )
    parser.add_argument(
        "-t", "--train-frac", type=float, default=0.8, help="Training fraction"
    )
    parser.add_argument(
        "-b",
        "--balance-train",
        action="store_true",
        help="Balance PC/LNC in training set",
    )
    parser.add_argument("-s", "--seed", type=int, default=42, help="Random seed")

    if "snakemake" in globals():
        args = argparse.Namespace(
            cluster=snakemake.input.cluster,
            fasta=snakemake.input.fasta,
            pc_ids=snakemake.input.pc_ids,
            lnc_ids=snakemake.input.lnc_ids,
            output_prefix=snakemake.params.output_prefix,
            train_frac=snakemake.params.get("train_frac", 0.8),
            balance_train=snakemake.params.get("balance_train", False),
            seed=snakemake.params.get("seed", 42),
        )

        logger = setup_snakemake_logging(snakemake, script_name=__file__)
    else:
        logger = setup_basic_logging()
        args = parser.parse_args()

    # Load data
    logger.info("Loading all transcripts...")
    all_sequences = load_fasta(args.fasta, as_dict=True)

    logger.info("Loading PC and LNC IDs...")
    pc_ids = simple_load_ids(args.pc_ids)
    lnc_ids = simple_load_ids(args.lnc_ids)

    logger.info("Parsing CD-HIT cluster file...")
    representative_ids = representative_ids_from_cdhit_clusters(args.cluster)
    logger.info(f"Found {len(representative_ids)} representative sequences")

    logger.info("Extracting simple transcript IDs...")
    # Extract the simple transcript IDs (in case full headers were provided)
    representative_ids = set(
        [
            seq_id.split("|")[0] if "|" in seq_id else seq_id
            for seq_id in representative_ids
        ]
    )
    all_sequences = {
        seq_id.split("|")[0] if "|" in seq_id else seq_id: seq
        for seq_id, seq in all_sequences.items()
    }
    pc_ids = set(
        [seq_id.split("|")[0] if "|" in seq_id else seq_id for seq_id in pc_ids]
    )
    lnc_ids = set(
        [seq_id.split("|")[0] if "|" in seq_id else seq_id for seq_id in lnc_ids]
    )

    logger.info("Extracting representative and redundant sequences...")
    representatives = {seq_id: all_sequences[seq_id] for seq_id in representative_ids}
    redundant = {
        seq_id: all_sequences[seq_id]
        for seq_id in all_sequences
        if seq_id not in representative_ids
    }

    # Classify representatives
    logger.info("Separating representative sequences into PC and LNC...")
    pc_sequences = [seq for seq_id, seq in representatives.items() if seq_id in pc_ids]
    lnc_sequences = [
        seq for seq_id, seq in representatives.items() if seq_id in lnc_ids
    ]
    logger.info("Representative counts:")
    logger.info(f"  Total: {len(representatives)}")
    logger.info(f"  PC:    {len(pc_sequences)}")
    logger.info(f"  LNC:   {len(lnc_sequences)}")

    # Get redundant sequences (sequences left out of training/testing due to redundancy)
    logger.info("Separating redundant sequences into PC and LNC...")
    pc_redun = [seq for seq_id, seq in redundant.items() if seq_id in pc_ids]
    lnc_redun = [seq for seq_id, seq in redundant.items() if seq_id in lnc_ids]
    logger.info("Redundant counts:")
    logger.info(f"  Total: {len(redundant)}")
    logger.info(f"  PC:    {len(pc_redun)}")
    logger.info(f"  LNC:   {len(lnc_redun)}")

    # Calculate splits depending on whether the dataset has to be balanced
    train_size = args.train_frac
    if args.balance_train:
        # Check which class is smaller -> it will determine the size of training
        logger.info("Calculating adjusted fractions for balanced training set...")
        min_count = min(len(pc_sequences), len(lnc_sequences))
        train_size = int(min_count * args.train_frac)
        logger.info("Adjusted fractions for balancing:")
        logger.info(f"  Train fraction (pc): {train_size/len(pc_sequences):.4f}")
        logger.info(f"  Train fraction (lnc): {train_size/len(lnc_sequences):.4f}")
        # NOTE: Since here we only select the training dataset to be balanced, the test set will remain imbalanced.
        # This is acceptable, and no "leftover" sequences will be generated.
        # We will need to manage this when evaluating the performance statistics.

    # Split sequences
    logger.info("Splitting sequences into train/test/redun sets...")
    pc_train, pc_test = train_test_split(
        pc_sequences, train_size=train_size, random_state=args.seed
    )

    lnc_train, lnc_test = train_test_split(
        lnc_sequences, train_size=train_size, random_state=args.seed
    )

    # Write output files
    logger.info("Writing output files...")
    output_prefix = args.output_prefix
    pc_train_file = output_prefix + ".train_pc.fa"
    lnc_train_file = output_prefix + ".train_lnc.fa"
    pc_test_file = output_prefix + ".test_pc.fa"
    lnc_test_file = output_prefix + ".test_lnc.fa"
    pc_redun_file = output_prefix + ".redun_pc.fa"
    lnc_redun_file = output_prefix + ".redun_lnc.fa"

    logger.info(f"  Writing {pc_train_file}...")
    write_fasta(pc_train, pc_train_file)
    logger.info(f"  Writing {lnc_train_file}...")
    write_fasta(lnc_train, lnc_train_file)
    logger.info(f"  Writing {pc_test_file}...")
    write_fasta(pc_test, pc_test_file)
    logger.info(f"  Writing {lnc_test_file}...")
    write_fasta(lnc_test, lnc_test_file)
    logger.info(f"  Writing {pc_redun_file}...")
    write_fasta(pc_redun, pc_redun_file)
    logger.info(f"  Writing {lnc_redun_file}...")
    write_fasta(lnc_redun, lnc_redun_file)

    # Print summary statistics
    logger.info("=== SUMMARY ===")
    logger.info(f"Training set:   PC={len(pc_train)}, LNC={len(lnc_train)}")
    logger.info(f"Test set:       PC={len(pc_test)}, LNC={len(lnc_test)}")
    logger.info(f"Non-represetative set:   PC={len(pc_redun)}, LNC={len(lnc_redun)}")

    log_job_completion(logger)


if __name__ == "__main__":
    main()
