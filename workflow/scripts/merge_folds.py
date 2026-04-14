"""
Merge fold-level tables into aggregated dataset-level tables.

This script takes tables from multiple cross-validation folds and combines them,
adding a 'fold' column to track the source fold for each row.
"""

import logging
import sys
from pathlib import Path

import pandas as pd

script_dir = Path(__file__).parent
workflow_dir = script_dir.parent

# Ensure workflow directory is in path for imports
if str(workflow_dir) not in sys.path:
    sys.path.insert(0, str(workflow_dir))

# Import utilities - with better error handling
try:
    from utils.logging_utils import (
        log_job_completion,
        setup_basic_logging,
        setup_snakemake_logging,
    )
except ImportError as e:
    print(f"Error importing from workflow utils: {e}", file=sys.stderr)
    print(f"Script dir: {script_dir}", file=sys.stderr)
    print(f"Workflow dir: {workflow_dir}", file=sys.stderr)
    print(f"sys.path: {sys.path}", file=sys.stderr)
    raise

# Setup module-wide logging
if "snakemake" in globals():
    logger = setup_snakemake_logging(snakemake, script_name=__file__)
else:
    logger = setup_basic_logging()


def merge_fold_tables(input_files, output_file, table_type):
    """
    Merge tables from multiple folds into a single aggregated table.

    Parameters
    ----------
    input_files : list of str
        Paths to input TSV files, one per fold
    output_file : str
        Path to output merged TSV file
    table_type : str
        Description of table type for logging
    """
    logger.info(f"Merging {len(input_files)} {table_type} tables...")

    dfs = []
    for i, file_path in enumerate(input_files):
        fold_name = f"fold{i+1}"
        logger.info(f"  Reading {fold_name}: {file_path}")

        # Read table
        df = pd.read_csv(file_path, sep="\t", index_col=0)

        # Add fold identifier column
        df.insert(0, "fold", fold_name)

        dfs.append(df)
        logger.info(f"    {fold_name}: {len(df)} rows")

    # Concatenate all folds
    merged_df = pd.concat(dfs, ignore_index=False)

    logger.info(f"Merged table: {len(merged_df)} total rows")

    # Save merged table
    logger.info(f"Writing to {output_file}")
    merged_df.to_csv(output_file, sep="\t")

    return merged_df


def main():
    """Main execution function."""
    try:
        # Get Snakemake parameters
        input_dict = snakemake.input
        output_dict = snakemake.output

        # Determine if we're processing single or multiple table types
        if hasattr(input_dict, "keys"):
            # Multiple table types
            table_types = list(input_dict.keys())
            logger.info("=== Starting fold table merging (multiple table types) ===")
            logger.info(f"Table types to process: {', '.join(table_types)}")

            total_rows = 0
            for table_type in table_types:
                input_files = list(input_dict[table_type])
                output_file = output_dict[table_type]

                logger.info(f"\n--- Processing {table_type} ---")
                logger.info(f"Number of folds: {len(input_files)}")

                merged_df = merge_fold_tables(input_files, output_file, table_type)
                total_rows += len(merged_df)

            logger.info("\n=== SUMMARY ===")
            logger.info(f"Table types processed: {len(table_types)}")
            logger.info(f"Total rows across all tables: {total_rows}")

        else:
            # Single table type (legacy mode)
            input_files = list(input_dict)
            output_file = output_dict[0]
            table_type = snakemake.params.get("table_type", "unknown")

            logger.info("=== Starting fold table merging ===")
            logger.info(f"Table type: {table_type}")
            logger.info(f"Number of folds: {len(input_files)}")

            merged_df = merge_fold_tables(input_files, output_file, table_type)

            logger.info("=== SUMMARY ===")
            logger.info(f"Input files: {len(input_files)}")
            logger.info(f"Output rows: {len(merged_df)}")
            logger.info(f"Output file: {output_file}")

        log_job_completion(logger)

    except NameError:
        logger.error("This script must be run within Snakemake")
        raise NotImplementedError(
            "For now, this script can only be run within Snakemake."
        )


if __name__ == "__main__":
    main()
