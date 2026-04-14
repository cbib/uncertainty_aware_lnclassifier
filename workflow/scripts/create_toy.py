import argparse
import random

from Bio import SeqIO


def select_random_sequences(input_file, output_file, num_sequences, seed=None):
    # Set the random seed for reproducibility
    if seed is not None:
        print(f"Setting random seed to {seed}")
        random.seed(seed)

    # Read all sequences from the input FASTA file (handle compressed files if necessary)
    print(f"Reading sequences from {input_file}...")
    if input_file.endswith(".gz"):
        import gzip

        with gzip.open(input_file, "rt") as handle:
            sequences = list(SeqIO.parse(handle, "fasta"))
    else:
        sequences = list(SeqIO.parse(input_file, "fasta"))

    print(f"Total sequences read: {len(sequences)}")

    # Check if the number of sequences to select is greater than available sequences
    if num_sequences > len(sequences):
        raise ValueError(
            "Number of sequences to select exceeds the total number of sequences in the file."
        )

    # Randomly select the specified number of sequences
    print(f"Selecting {num_sequences} random sequences...")
    selected_sequences = random.sample(sequences, num_sequences)

    # Write the selected sequences to the output FASTA file
    print(f"Writing selected sequences to {output_file}...")
    SeqIO.write(selected_sequences, output_file, "fasta")

    print("Selection and writing completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Select random sequences from a FASTA file."
    )
    parser.add_argument("input_file", help="Path to the input FASTA file.")
    parser.add_argument("output_file", help="Path to the output FASTA file.")
    parser.add_argument(
        "num_sequences", type=int, help="Number of random sequences to select."
    )
    parser.add_argument(
        "-s", "--seed", type=int, default=None, help="Random seed for reproducibility."
    )
    args = parser.parse_args()

    try:
        print(
            f"Selecting {args.num_sequences} random sequences from {args.input_file}..."
        )
        select_random_sequences(
            args.input_file, args.output_file, args.num_sequences, args.seed
        )
        print(f"Random sequences have been written to {args.output_file}")
    except Exception as e:
        print(f"Error: {e}")
