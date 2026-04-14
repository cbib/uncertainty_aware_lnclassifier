import os
import sys

from Bio import SeqIO

# Remove sequences containing 'N' characters
# Remove sequences longer than 100,000 nt

try:
    # Ensure output directory exists
    os.makedirs(os.path.dirname(snakemake.output[0]), exist_ok=True)

    filtered_records = []
    count_N = 0
    count_long = 0

    # Read and filter sequences
    with open(snakemake.input[0], "r") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            seq = str(record.seq)
            if "N" in seq:
                print(f"{record.id} contains 'N'")
                count_N += 1
            elif len(seq) > 100000:
                print(f"{record.id} is longer than 100,000 nt (length: {len(seq)})")
                count_long += 1
            else:
                filtered_records.append(record)

    print(f"Filtered out {count_N} sequences containing 'N'")
    print(f"Filtered out {count_long} sequences longer than 100,000 nt")

    # Write filtered records to output
    with open(snakemake.output[0], "w") as out_handle:
        SeqIO.write(filtered_records, out_handle, "fasta")

    print(
        f"File saved to {snakemake.output[0]} with {len(filtered_records)} sequences."
    )

except FileNotFoundError as e:
    print(f"Error: Input file not found: {e}")
    raise e
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    raise e
