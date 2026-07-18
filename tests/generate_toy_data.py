#!/usr/bin/env python3
"""
Generate toy FASTA datasets for LNClassifier pipeline testing.

Samples 50 protein-coding and 50 lncRNA sequences from GENCODE v47,
filtering for minimum length, deterministic via seed=42.

Outputs (in tests/data/toy/):
  toy.pc_transcripts.fa / .fa.gz
  toy.lncrna_transcripts.fa / .fa.gz

Usage:
  python tests/generate_toy_data.py
  python tests/generate_toy_data.py --n 20 --min-len 100  # faster smoke-test set
"""

import argparse
import gzip
import random
import sys
from pathlib import Path

from Bio import SeqIO
from Bio.SeqRecord import SeqRecord

REPO_ROOT = Path(__file__).resolve().parent.parent
PC_SOURCE = REPO_ROOT / "resources" / "gencode.v47.pc_transcripts.fa"
LNC_SOURCE = REPO_ROOT / "resources" / "gencode.v47.lncRNA_transcripts.fa"
OUT_DIR = REPO_ROOT / "tests" / "data" / "toy"

SEED = 42
N = 50
MIN_LEN = 200


def sample_sequences(
    fasta_path: Path, n: int, min_len: int, seed: int
) -> list[SeqRecord]:
    """Read all qualifying sequences then sample n of them."""
    candidates = [r for r in SeqIO.parse(fasta_path, "fasta") if len(r.seq) >= min_len]
    if len(candidates) < n:
        raise ValueError(
            f"{fasta_path.name}: only {len(candidates)} sequences >= {min_len} nt, need {n}"
        )
    rng = random.Random(seed)
    return rng.sample(candidates, n)


def write_fasta(records: list[SeqRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    SeqIO.write(records, path, "fasta")


def write_fasta_gz(records: list[SeqRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as fh:
        SeqIO.write(records, fh, "fasta")


def main(n: int = N, min_len: int = MIN_LEN) -> None:
    print(f"Sampling {n} sequences per class (min_len={min_len} nt, seed={SEED})")

    pc_records = sample_sequences(PC_SOURCE, n, min_len, SEED)
    lnc_records = sample_sequences(LNC_SOURCE, n, min_len, SEED)

    pc_fa = OUT_DIR / "toy.pc_transcripts.fa"
    lnc_fa = OUT_DIR / "toy.lncrna_transcripts.fa"

    write_fasta(pc_records, pc_fa)
    write_fasta(lnc_records, lnc_fa)
    write_fasta_gz(pc_records, pc_fa.with_suffix(".fa.gz"))
    write_fasta_gz(lnc_records, lnc_fa.with_suffix(".fa.gz"))

    print(f"Written {pc_fa}")
    print(f"Written {lnc_fa}")
    print(f"Written {pc_fa.with_suffix('.fa.gz')}")
    print(f"Written {lnc_fa.with_suffix('.fa.gz')}")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=N, help="Sequences per class")
    parser.add_argument(
        "--min-len", type=int, default=MIN_LEN, help="Minimum sequence length"
    )
    args = parser.parse_args()
    main(n=args.n, min_len=args.min_len)
