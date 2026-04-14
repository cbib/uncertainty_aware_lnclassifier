import sys
from pathlib import Path

from Bio.SeqIO.FastaIO import SimpleFastaParser

script_dir = Path(__file__).parent
workflow_dir = script_dir.parent
sys.path.insert(0, str(workflow_dir))

from utils.logging_utils import log_job_completion, setup_snakemake_logging
from utils.parsing import parse_gencode_ids, simple_load_ids

logger = setup_snakemake_logging(snakemake, script_name=__file__)

ss = snakemake.input["ss"]
fasta = snakemake.input["fasta"]
out_file = snakemake.output["out"]

# Load the IDs from the subset FASTA
# NOTE: LncFinder replaces '|' and '-' with '.' in their IDs
fasta_ids = simple_load_ids(fasta)
fasta_ids = [i.replace("|", ".").replace("-", ".") for i in fasta_ids]

# Index the large reference FASTA into a dict
ss_dict = {}
with open(ss) as handle:
    ss_dict = dict(SimpleFastaParser(handle))

# Write only the sequences present in fasta_ids (subset file)
count = 0
with open(out_file, "w") as outfh:
    for name in fasta_ids:
        seq = ss_dict.get(name)
        if seq:
            outfh.write(f">{name}\n{seq}\n")
            count += 1

logger.info(f"Found {count} out of {len(fasta_ids)} secondary structures in {ss}")
logger.info(f"Saved to {out_file}")
log_job_completion(logger)
