# Import statements
import gzip
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Set, Tuple

import pandas as pd
from Bio import SeqIO

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
    from utils.parsing import parse_gencode_ids, simple_load_ids
    from utils.process_tools import *
except ImportError as e:
    print(f"Error importing from workflow utils: {e}", file=sys.stderr)
    print(f"Script dir: {script_dir}", file=sys.stderr)
    print(f"Workflow dir: {workflow_dir}", file=sys.stderr)
    print(f"sys.path: {sys.path}", file=sys.stderr)
    raise

# Setup module-wide logging
if "snakemake" in globals():
    print("Snakemake object found; setting up Snakemake logging.", file=sys.stderr)
    logger = setup_snakemake_logging(snakemake, script_name=__file__)
else:
    print(
        "No snakemake object found; using basic logging configuration.", file=sys.stderr
    )
    logger = setup_basic_logging()

try:
    # Process Snakemake arguments
    input = dict(snakemake.input)
    output = snakemake.output[0]
    prefix = snakemake.params.get("prefix", "")
except NameError:
    # TODO: Make it possible to run outside Snakemake
    raise NotImplementedError("For now, this script can only be run within Snakemake.")

input_files = input
output_prefix = prefix

# ============================================================================
# MAIN PIPELINE
# ============================================================================

logger.info("=== Starting classification aggregation pipeline ===")

# Process each tool (only if present in input_files)
tool_processors = {
    "rnasamba": lambda: process_rnasamba(input_files["rnasamba_full"]),
    "cpc2": lambda: process_cpc2(input_files["cpc2"]),
    "feelnc": lambda: process_feelnc(input_files["feelnc"]),
    "cpat": lambda: process_cpat(
        input_files["cpat_p"],
        input_files["cpat_l"],
        input_files.get("cpat_cutoff", None),
    ),
    "lncdc": lambda: process_lncdc(input_files["lncDC_no_ss"], input_files["lncDC_ss"]),
    "mrnn": lambda: process_mrnn(input_files["mrnn"]),
    "lncrnabert": lambda: process_lncrnabert(input_files["lncrnabert"]),
    "plncpro": lambda: process_plncpro(input_files["plncpro"]),
    "lncadeep": lambda: process_lncadeep(input_files["lncadeep"]),
    "lncfinder": lambda: process_lncfinder(
        input_files["lncfinder_ss"], input_files["lncfinder_no_ss"]
    ),
    "lncrnanet": lambda: process_lncrnanet(input_files["lncrnanet"]),
}

# Check which tools have required input files
tool_file_requirements = {
    "rnasamba": ["rnasamba_full"],
    "cpc2": ["cpc2"],
    "feelnc": ["feelnc"],
    "cpat": ["cpat_p", "cpat_l"],  # 'cpat_cutoff' is optional, so we don't list it here
    "lncdc": ["lncDC_no_ss", "lncDC_ss"],
    "mrnn": ["mrnn"],
    "lncrnabert": ["lncrnabert"],
    "plncpro": ["plncpro"],
    "lncadeep": ["lncadeep"],
    "lncfinder": ["lncfinder_ss", "lncfinder_no_ss"],
    "lncrnanet": ["lncrnanet"],
}

tool_col_ids = {
    "cpc2": "#ID_cpc2",
    "feelnc": "name_feelnc",
    "cpat": "seq_ID_cpat",
    "lncdc": "Description_lncDC",
    "mrnn": "seq_ID_mrnn",
    "rnasamba": "sequence_name_rnasamba",
    "lncrnabert": "id_lncrnabert",
    "plncpro": "transcript_id_plncpro",
    "lncadeep": "ID_lncadeep",
    "lncfinder": "ID_lncfinder",
    "lncrnanet": "ID_lncrnanet",
}

# Process available tools
processed_tools = []
for tool_name, required_files in tool_file_requirements.items():
    if all(rf in input_files for rf in required_files):
        logger.info(f"Processing {tool_name}...")
        processed_tools.append(
            ToolConfig(tool_processors[tool_name](), tool_name, tool_col_ids[tool_name])
        )
    else:
        logger.warning(
            f"Skipping {tool_name}: missing required input files {required_files}"
        )

if not processed_tools:
    raise ValueError("No valid tool results found in input_files")

logger.info("Combining results...")
import os

logger.info(f"Current working directory: {os.getcwd()}")


# Merge all tools (first tool is used as base)
combined_df = merge_all_tools(processed_tools)

# Standardize column names
print(LABEL_COLUMN_RENAMES)
combined_df.rename(columns=LABEL_COLUMN_RENAMES, inplace=True)

# Add metadata
combined_df = add_metadata_columns(combined_df)

# Save full table
full_table_path = f"{output_prefix}_full_table.tsv"
logger.info(f"Writing full table to {full_table_path}")
combined_df.to_csv(full_table_path, sep="\t")

# Create classification table
class_df = create_classification_table(combined_df)
class_table_path = f"{output_prefix}_class_table.tsv"
logger.info(f"Writing classification table to {class_table_path}")
class_df.to_csv(class_table_path, sep="\t")

# Load reference transcripts
pc_ids, lnc_ids = load_reference_transcripts(
    input_files["pc_transcripts"], input_files["lncRNA_transcripts"]
)

# Create simple classification table
simple_class_df = create_simple_classification_table(class_df, pc_ids, lnc_ids)
simple_class_path = f"{output_prefix}_simple_class_table.tsv"
logger.info(f"Writing simple classification table to {simple_class_path}")
simple_class_df.to_csv(simple_class_path, sep="\t")

# Create unclassified transcripts table
no_class_df = create_unclassified_table(simple_class_df)
no_class_path = f"{output_prefix}_no_class_table.tsv"
logger.info(f"Writing unclassified table to {no_class_path}")
no_class_df.to_csv(no_class_path, sep="\t")

# Create binary classification table
binary_class_df = create_binary_classification_table(simple_class_df)
binary_class_path = f"{output_prefix}_binary_class_table.tsv"
logger.info(f"Writing binary classification table to {binary_class_path}")
binary_class_df.to_csv(binary_class_path, sep="\t")

logger.info("=== SUMMARY ===")
logger.info(f"Total transcripts in combined dataset: {len(combined_df)}")
logger.info(f"Reference transcripts (with ground truth): {len(simple_class_df)}")
logger.info(f"Complete classifications (no NaN): {len(binary_class_df)}")
logger.info(f"Transcripts with missing classifications: {len(no_class_df)}")

log_job_completion(logger)
