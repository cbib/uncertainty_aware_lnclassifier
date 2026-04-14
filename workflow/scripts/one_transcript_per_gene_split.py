import random

import numpy as np
import pandas as pd
from Bio import SeqIO

SEED = snakemake.params["seed"]
split = snakemake.params["split"] / 100
gtf_file = snakemake.input["gtf"]
pc_fasta = snakemake.input["pc"]
lnc_fasta = snakemake.input["lnc"]
balanced = snakemake.params["balanced"]
full_fasta_out = snakemake.output["one_per_gene"]
out_dir = snakemake.params["out_dir"]
prefix = snakemake.params["prefix"]


# Helper function to split sequences
def split_sequences(records, ids, split=20):
    # Subset IDs
    n_test = int(len(ids) * split)
    test_ids = set(ids[:n_test])
    train_ids = set(ids[n_test:])

    # Subset records
    test_records = []
    train_records = []
    leftover_records = []
    for rec in records:
        rec_id = rec.id.split("|")[0]
        if rec_id in test_ids:
            test_records.append(rec)
        elif rec_id in train_ids:
            train_records.append(rec)
        else:
            leftover_records.append(rec)
    return test_records, train_records, leftover_records


# Read GTF
gtf = pd.read_csv(gtf_file, sep="\t", comment="#", header=None)

# Read pc and lncRNA files, which hold the classification labels
# (i.e., transcripts in pc file= protein coding; transcripts in lncRNA file = lncRNA)
pc_records = list(SeqIO.parse(pc_fasta, "fasta"))
lnc_records = list(SeqIO.parse(lnc_fasta, "fasta"))
pc_ids = set([rec.id.split("|")[0] for rec in pc_records])
lnc_ids = set([rec.id.split("|")[0] for rec in lnc_records])

transcripts = gtf[gtf[2] == "transcript"].copy()
transcripts["gene_id"] = transcripts[8].str.split('"').str[1]
transcripts["transcript_id"] = transcripts[8].str.split('"').str[3]
transcripts = transcripts[transcripts["transcript_id"].isin(pc_ids | lnc_ids)].copy()
transcripts["label"] = np.where(transcripts["transcript_id"].isin(pc_ids), "pc", "lnc")

# Unique genes
gene_ids_in_db = transcripts["gene_id"].unique()
print(
    len(gene_ids_in_db)
)  # TODO: Check what are the missing genes, because pc_genes = 19433 and lncRNA genes = 35934

# Select random transcript per gene
random.shuffle(gene_ids_in_db)
selection = (
    transcripts.groupby("gene_id")
    .apply(lambda x: x.sample(1, random_state=SEED)[["gene_id", "transcript_id"]])
    .reset_index(drop=True)
)
selection = selection.merge(
    transcripts[["transcript_id", "label"]], on="transcript_id", how="left"
)
print(selection["label"].value_counts())

selection.to_csv(full_fasta_out, sep="\t", index=False, header="True")

# Split
pc_transcripts = selection.loc[selection["label"] == "pc", "transcript_id"].to_list()
lnc_transcripts = selection.loc[selection["label"] == "lnc", "transcript_id"].to_list()


# Save the test, train, and leftover records for pc transcripts
test_pc_fasta, train_pc_fasta, leftover_pc_fasta = split_sequences(
    pc_records, pc_transcripts, split
)
SeqIO.write(test_pc_fasta, f"{out_dir}/{prefix}.test_pc.fa", "fasta")
SeqIO.write(train_pc_fasta, f"{out_dir}/{prefix}.train_pc.fa", "fasta")
SeqIO.write(leftover_pc_fasta, f"{out_dir}/{prefix}.leftover_pc.fa", "fasta")

# Save the test, train, and leftover records for lnc transcripts
test_lnc_fasta, train_lnc_fasta, leftover_lnc_fasta = split_sequences(
    lnc_records, lnc_transcripts, split
)
SeqIO.write(test_lnc_fasta, f"{out_dir}/{prefix}.test_lnc.fa", "fasta")
SeqIO.write(train_lnc_fasta, f"{out_dir}/{prefix}.train_lnc.fa", "fasta")
SeqIO.write(leftover_lnc_fasta, f"{out_dir}/{prefix}.leftover_lnc.fa", "fasta")

# Save test set combined
test_combined = test_pc_fasta + test_lnc_fasta
SeqIO.write(test_combined, f"{out_dir}/{prefix}.test.fa", "fasta")

# Save dataset statistics
# Calculate and save dataset statistics
stats = {
    "seed": SEED,
    "total_transcripts": len(pc_ids) + len(lnc_ids),
    "pc_transcripts": len(pc_transcripts),
    "lnc_transcripts": len(lnc_transcripts),
    "total_genes": len(gene_ids_in_db),
    "train_total": len(train_pc_fasta) + len(train_lnc_fasta),
    "train_pc": len(train_pc_fasta),
    "train_lnc": len(train_lnc_fasta),
    "test_total": len(test_combined),
    "test_pc": len(test_pc_fasta),
    "test_lnc": len(test_lnc_fasta),
    "leftover_pc": len(leftover_pc_fasta),
    "leftover_lnc": len(leftover_lnc_fasta),
}

stats_df = pd.DataFrame([stats])
stats_df.to_csv(f"{out_dir}/{prefix}.{SEED}.stats.csv", index=False)
