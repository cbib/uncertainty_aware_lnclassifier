import logging
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

script_dir = Path(__file__).parent
workflow_dir = script_dir.parent
sys.path.insert(0, str(workflow_dir))

from utils.logging_utils import log_job_completion, setup_snakemake_logging
from utils.parsing import classify_transcripts_from_files, load_fasta, write_fasta

logger = logging.getLogger(__name__)


def main():
    logger = setup_snakemake_logging(snakemake, script_name=__file__)

    # Get input/output files from snakemake
    pc_fasta = snakemake.input.pc_fasta
    lnc_fasta = snakemake.input.lnc_fasta
    inference_files = snakemake.input.inference_files
    output_fasta = snakemake.output.fasta
    output_table = snakemake.output.info_table

    # Collect all records with provenance
    all_records = []
    all_record_sources = []

    logger.info(f"Processing {len(inference_files)} inference files...")

    for fasta_file in inference_files:
        source = fasta_file.split(".")[-2]  # Source/Dataset name
        logger.info(f"Reading {fasta_file} ({source})...")
        records = load_fasta(fasta_file)
        all_records.extend(records)
        all_record_sources.extend([source] * len(records))
        logger.info(f"  Added {len(records)} sequences from {source}")

    logger.info(f"Total sequences collected: {len(all_records)}")

    # Write combined fasta
    logger.info(f"Writing output files...")
    write_fasta(all_records, output_fasta)
    logger.info(f"Wrote {len(all_records)} sequences to {output_fasta}")

    # Create provenance table with classes
    df = pd.DataFrame(
        {
            "transcript_id": [record.id for record in all_records],
            "source_file": all_record_sources,
        }
    )

    df = classify_transcripts_from_files(df, pc_fasta, lnc_fasta, version=True)
    df.to_csv(output_table, sep="\t", index=False)
    logger.info(f"Provenance table written to {output_table}")
    log_job_completion(logger)


if __name__ == "__main__":
    if "snakemake" not in globals():
        raise NotImplementedError(
            "For now, this script is intended to be run within a Snakemake workflow."
        )
    main()
