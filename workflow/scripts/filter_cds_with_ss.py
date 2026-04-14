import sys
from pathlib import Path

from Bio.SeqIO.FastaIO import SimpleFastaParser

script_dir = Path(__file__).parent
workflow_dir = script_dir.parent
sys.path.insert(0, str(workflow_dir))

from utils.logging_utils import (
    log_job_completion,
    setup_basic_logging,
    setup_snakemake_logging,
)
from utils.parsing import simple_load_ids

# Setup module-wide logging
if "snakemake" in globals():
    logger = setup_snakemake_logging(snakemake, script_name=__file__)
else:
    logger = setup_basic_logging()


def filter_cds_with_ss(cds_file, ss_file, out_file):
    """
    Filters out CDS sequences that do not have corresponding secondary structure data.
    Parameters:
        cds_file (str): Path to the CDS FASTA file.
        ss_file (str): Path to the secondary structure FASTA file.
        out_file (str): Path to the output filtered CDS FASTA file.
    """
    logger.info(f"Reading SS data from {ss_file}.")
    # Index the mRNA SS FASTA into a dict
    ss_dict = {}
    with open(ss_file) as handle:
        ss_dict = dict(SimpleFastaParser(handle))
        # Map simple IDs to full IDs
        # NOTE: Files produced by lncfinder have IDs like 'ENST00000454124.1.ENSG00000115504.15...'
        ss_ids = {i.split(".ENSG")[0]: i for i in ss_dict.keys()}
    logger.info(list(ss_ids.keys())[:10])
    logger.info(f"Reading CDS sequences from {cds_file}")
    cds_ids = simple_load_ids(cds_file)
    # CDS ids have different format from Gencode transcripts.
    # Extract by splitting at spaces
    cds_ids = [i.split(" ")[0] for i in cds_ids]

    logger.info(f"Writing CDS sequences that have associated SS data to {out_file}")
    count = 0
    with open(out_file, "w") as outfh:
        for cds in cds_ids:
            full_id = ss_ids.get(cds)
            if full_id:
                # id found in SS dict
                seq = ss_dict.get(full_id)
                outfh.write(f">{full_id}\n{seq}\n")
                count += 1
            else:
                logger.info(f"CDS ID {cds} not found in SS data; skipping.")
    logger.info(f"Wrote {count} out of {len(cds_ids)} sequences.")
    return


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Filter CDS sequences to only those with secondary structure data."
    )
    parser.add_argument("-c", "--cds", required=True, help="Input CDS FASTA file.")
    parser.add_argument(
        "-s", "--ss", required=True, help="Input secondary structure FASTA file."
    )
    parser.add_argument(
        "-o", "--out", required=True, help="Output filtered CDS FASTA file."
    )

    if "snakemake" in globals():
        args = argparse.Namespace(
            cds=snakemake.input.cds,
            ss=snakemake.input.ss,
            out=snakemake.output[0],
        )
    else:
        args = parser.parse_args()

    filter_cds_with_ss(args.cds, args.ss, args.out)

    log_job_completion(logger)


if __name__ == "__main__":
    main()
