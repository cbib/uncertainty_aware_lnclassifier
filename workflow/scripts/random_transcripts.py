import random
from collections import defaultdict

from Bio import SeqIO

seed = snakemake.params.seed


random.seed(snakemake.params.seed)

# Open the input files
mRNA_file = snakemake.input.mrna
lncRNA_file = snakemake.input.lncrna

# Read mRNA and lncRNA records
mRNA_records = list(SeqIO.parse(mRNA_file, "fasta"))
lncRNA_records = list(SeqIO.parse(lncRNA_file, "fasta"))


# Process number of transcripts
n = snakemake.wildcards.transcript_number
if not n.isdigit() or int(n) <= 0:
    raise ValueError(f"Invalid transcript number: {n}. It must be a positive integer.")

n = int(n)

if n > len(mRNA_records) or n > len(lncRNA_records):
    raise ValueError(
        f"Requested number of transcripts {n} exceeds available records in input files."
    )
    # Alternatively, set n to minimum of available records


def select_random_transcripts(records, n, transcript_type):
    """
    Select n random transcripts ensuring that no more than one transcript per gene is selected.
    """
    transcripts_per_gene = defaultdict(list)
    for rec in records:
        gene_id = rec.id.split("|")[1]
        transcripts_per_gene[gene_id].append(rec)

    selected_genes = random.sample(sorted(transcripts_per_gene.keys()), n)
    selected_transcripts = [
        random.choice(transcripts_per_gene[gene_id]) for gene_id in selected_genes
    ]
    if len(selected_transcripts) < n:
        raise ValueError(
            f"Not enough unique genes to select {n} {transcript_type}. Available: {len(selected_transcripts)}"
        )

    return selected_transcripts


chosen_mRNA = select_random_transcripts(mRNA_records, n, "mRNA transcripts")
chosen_lncRNA = select_random_transcripts(lncRNA_records, n, "lncRNA transcripts")


# Write the selected transcripts to the output file
out_mrna = snakemake.output.out_mrna
with open(out_mrna, "w") as out_handle:
    SeqIO.write(chosen_mRNA, out_handle, "fasta")

out_lncrna = snakemake.output.out_lncrna
with open(out_lncrna, "w") as out_handle:
    SeqIO.write(chosen_lncRNA, out_handle, "fasta")
