import argparse
import sys
from itertools import compress
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from Bio import SeqIO
from sklearn.model_selection import StratifiedKFold, train_test_split

scripts_dir = Path(__file__).parent.parent
sys.path.insert(0, str(scripts_dir))

from utils.logging_utils import (
    log_job_completion,
    setup_basic_logging,
    setup_snakemake_logging,
)
from utils.parsing import (
    classify_transcripts_from_files,
    gencode_fasta_to_df,
    load_fasta,
    simple_load_ids,
    write_fasta,
)


def run_and_save_cv_split(
    records: List[SeqIO.SeqRecord],
    y: np.ndarray,
    output_dir: str,
    n_splits: int = 5,
    SEED: int = 42,
    logger=None,
):
    """
    Perform stratified K-Fold CV split, balance training sets, and save FASTA files for each fold.
    Parameters:
    records: List[SeqIO.SeqRecord]
        List of all sequence records.
    y: np.ndarray
        Array of class labels corresponding to records.
    output_dir: str
        Directory to save fold FASTA files.
    n_splits: int
        Number of folds for Stratified K-Fold.
    SEED: int
        Random seed for reproducibility.
    logger: logging.Logger
        Logger for logging information.
    """

    X = np.arange(len(records))

    np.random.seed(SEED)
    logger.info(f"Starting CV split process using seed {SEED} and {n_splits} folds")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        ###########
        # Balance #
        ###########
        unique, counts = np.unique(y_train, return_counts=True)
        min_count = counts.min()
        minority_class = unique[np.argmin(counts)]
        logger.info(
            f"Fold {fold+1}: Undersampling from {np.max(counts)} to {min_count} samples each (minority class: {minority_class})"
        )

        balanced_idx = []
        for label in unique:
            label_idx = np.where(y_train == label)[0]
            sampled = np.random.choice(label_idx, size=min_count, replace=False)
            balanced_idx.extend(sampled)

        X_train_balanced = X_train[balanced_idx]
        y_train_balanced = y_train[balanced_idx]

        ###############
        # Save FASTAs #
        ###############
        fold_dir = Path(output_dir) / f"fold{fold+1}"
        fold_dir.mkdir(exist_ok=True)

        # Helper to get sequence IDs by label
        train_pc = X_train_balanced[y_train_balanced == "coding"]
        train_lnc = X_train_balanced[y_train_balanced == "lncRNA"]
        test_pc = X_test[y_test == "coding"]
        test_lnc = X_test[y_test == "lncRNA"]

        # Write sets
        logger.info(f"Fold {fold+1}: Writing FASTA files to {fold_dir}")
        write_fasta([records[i] for i in train_pc], fold_dir / "train_pc.fa")
        write_fasta([records[i] for i in train_lnc], fold_dir / "train_lnc.fa")
        write_fasta([records[i] for i in test_pc], fold_dir / "test_pc.fa")
        write_fasta([records[i] for i in test_lnc], fold_dir / "test_lnc.fa")
        write_fasta([records[i] for i in X_test], fold_dir / "test_all.fa")
        logger.info(
            f"Fold {fold+1}: Written. {len(train_pc)} train PC, {len(train_lnc)} train LNC, {len(test_pc)} test PC, {len(test_lnc)} test LNC sequences"
        )

    logger.info("CV split process completed.")
    return


def main():

    if "snakemake" in globals():
        args = argparse.Namespace(
            fasta=snakemake.input.fasta,
            pc_file=snakemake.input.pc_file,
            lnc_file=snakemake.input.lnc_file,
            output_dir=snakemake.params.output_dir,
            n_splits=snakemake.params.get("n_splits", 5),
            SEED=snakemake.params.get("seed", 42),
        )

        logger = setup_snakemake_logging(snakemake, script_name=__file__)

    else:
        logger = setup_basic_logging()

    # Load inputs
    logger.info("Loading all transcripts...")
    records = load_fasta(args.fasta)
    all_ids = [rec.id.split("|")[0] for rec in records]

    pc_ids = simple_load_ids(args.pc_file, simple=True)
    lnc_ids = simple_load_ids(args.lnc_file, simple=True)
    logger.info(
        f"Loaded {len(all_ids)} sequences, {len(pc_ids)} PC IDs and {len(lnc_ids)} LNC IDs"
    )

    all_df = pd.DataFrame({"transcript_id": all_ids})
    all_df = classify_transcripts_from_files(all_df, args.pc_file, args.lnc_file)
    labels = all_df["coding_class"].to_list()

    # Subset to only coding and lncRNA
    mask = all_df["coding_class"].isin(["coding", "lncRNA"]).to_numpy()
    records = list(compress(records, mask))
    labels = np.asarray(list(compress(labels, mask)))

    run_and_save_cv_split(
        records,
        labels,
        output_dir=args.output_dir,
        n_splits=args.n_splits,
        SEED=args.SEED,
        logger=logger,
    )

    # Create a completion flag
    (Path(args.output_dir) / "cv_split.done").touch()
    log_job_completion(logger)


if __name__ == "__main__":
    main()
