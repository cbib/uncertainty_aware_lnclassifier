import gzip
import sys
from pathlib import Path

from Bio import SeqIO

script_dir = Path(__file__).parent
workflow_dir = script_dir.parent
sys.path.insert(0, str(workflow_dir))

from utils.logging_utils import log_job_completion, setup_snakemake_logging
from utils.parsing import parse_gencode_ids, simple_load_ids

logger = setup_snakemake_logging(snakemake, script_name=__file__)

pc_file = snakemake.input["transcripts"]
cds_file = snakemake.input["cds"]
out_file = snakemake.output[0]


ids = simple_load_ids(pc_file)
if "|" in ids[0]:
    logger.info("Parsing sequence IDs from FASTA headers")
    _, ids, _, _ = parse_gencode_ids(ids)
ids_set = set(ids)
logger.info(f"Scanned {len(ids)} transcripts")

matching_cds = []
with gzip.open(cds_file, "rt") as handle:
    for cds in SeqIO.parse(handle, "fasta"):
        id = cds.id.split(" ")[0].strip(">")
        if id in ids_set:
            matching_cds.append(cds)

matching_cds.sort(key=lambda x: ids.index(x.id))

if ids == [record.id for record in matching_cds]:
    logger.info("Sequence order is the same")

count = SeqIO.write(matching_cds, out_file, "fasta")

logger.info(f"Saved {count} cds to '{out_file}'")
log_job_completion(logger)
