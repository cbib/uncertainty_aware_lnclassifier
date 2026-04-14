import pandas as pd
from Bio import SeqIO

# Snakemake input arguments
input = snakemake.input
orf_file_name = snakemake.output[0]

# Read the CPC2 output file
df = pd.read_csv(
    input.cpc2_txt, sep="\t", usecols=["#ID", "ORF_Start", "peptide_length"]
)
df["ORF_length_nt"] = (
    df["peptide_length"] * 3
)  # Convert peptide length to nucleotide length
# TODO: where is the ORF_length_nt column in Rahma's code?

# Rahma's code to extract ORFs
sequences = {}
for record in SeqIO.parse(input.fasta, "fasta"):
    # sequences[clean_id(record.id)] = record.seq # la clé = l'ID et la valeur = la séquence
    sequences[record.id] = record.seq  # la clé est l'ID et la valeur est la séquence
# ouverture du fichier des ORFs, vide
with open(orf_file_name, "w") as sample_orfs:
    for (
        index,
        row,
    ) in df.iterrows():  # pour chaque ligne du df sous forme de tuple (index, row)
        # transcript_id = clean_id(row["#ID"])
        transcript_id = row["#ID"]
        start = int(row["ORF_Start"])
        length = int(row["ORF_length_nt"])
        if (
            transcript_id in sequences
        ):  # correspondance entre l'ID dans le fichier fasta des ORFs extrait et l'ID dans le dictionnaire sequences{} des fichiers samplegene.fasta et sampletranscript.fasta
            orf_seq = sequences[transcript_id][
                start - 1 : start - 1 + length
            ]  # [a:b], a -> extraction à partir du caractère à la position start [jusqu'au] b -> caractère de la position end
            sample_orfs.write(f">{transcript_id}\n{orf_seq}\n")
            # TODO: Decice if we want to output ORFs as tsv as well.
