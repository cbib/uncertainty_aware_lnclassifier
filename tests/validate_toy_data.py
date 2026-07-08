#!/usr/bin/env python3
"""
Validate toy FASTA datasets for LNClassifier pipeline testing.

Checks:
  1. Both files exist (plain + gzipped)
  2. Correct sequence counts
  3. All sequences >= MIN_LEN nt
  4. All headers contain ENSG (required by ID normalisation in process_tools.py)
  5. All headers have >= 7 pipe-separated fields (ENST|ENSG|...|name|gene|length|...)
  6. No duplicate transcript IDs within a file
  7. No ID overlap between PC and lncRNA sets
  8. Gzipped files decode to the same IDs as the plain files

Usage:
  python tests/validate_toy_data.py            # uses defaults
  python tests/validate_toy_data.py --n 20     # if generated with --n 20
"""

import argparse
import gzip
import sys
from pathlib import Path

from Bio import SeqIO

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "tests" / "data" / "toy"

N = 50
MIN_LEN = 200


def enst_id(record) -> str:
    """First pipe-field of the FASTA header."""
    return record.id.split("|")[0]


def load_plain(path: Path) -> list:
    return list(SeqIO.parse(path, "fasta"))


def load_gz(path: Path) -> list:
    with gzip.open(path, "rt") as fh:
        return list(SeqIO.parse(fh, "fasta"))


def check(condition: bool, msg: str) -> bool:
    if condition:
        print(f"  PASS  {msg}")
    else:
        print(f"  FAIL  {msg}", file=sys.stderr)
    return condition


def validate(n: int = N, min_len: int = MIN_LEN) -> bool:
    pc_fa = OUT_DIR / "toy.pc_transcripts.fa"
    lnc_fa = OUT_DIR / "toy.lncrna_transcripts.fa"
    pc_gz = pc_fa.with_suffix(".fa.gz")
    lnc_gz = lnc_fa.with_suffix(".fa.gz")

    ok = True

    # 1. Files exist
    for p in (pc_fa, lnc_fa, pc_gz, lnc_gz):
        ok &= check(p.exists(), f"exists: {p.name}")

    if not ok:
        print("\nSome files missing — aborting further checks.", file=sys.stderr)
        return False

    pc = load_plain(pc_fa)
    lnc = load_plain(lnc_fa)
    pc_gz_records = load_gz(pc_gz)
    lnc_gz_records = load_gz(lnc_gz)

    # 2. Correct counts
    ok &= check(len(pc) == n, f"PC count == {n} (got {len(pc)})")
    ok &= check(len(lnc) == n, f"lncRNA count == {n} (got {len(lnc)})")

    # 3. Minimum length
    pc_short = [r for r in pc if len(r.seq) < min_len]
    lnc_short = [r for r in lnc if len(r.seq) < min_len]
    ok &= check(
        len(pc_short) == 0,
        f"all PC sequences >= {min_len} nt (violations: {len(pc_short)})",
    )
    ok &= check(
        len(lnc_short) == 0,
        f"all lncRNA sequences >= {min_len} nt (violations: {len(lnc_short)})",
    )

    # 4. ENSG in every header (required by ID normalisation in process_tools.py)
    pc_no_ensg = [r for r in pc if "ENSG" not in r.description]
    lnc_no_ensg = [r for r in lnc if "ENSG" not in r.description]
    ok &= check(
        len(pc_no_ensg) == 0,
        f"all PC headers contain ENSG (violations: {len(pc_no_ensg)})",
    )
    ok &= check(
        len(lnc_no_ensg) == 0,
        f"all lncRNA headers contain ENSG (violations: {len(lnc_no_ensg)})",
    )

    # 5. Header field count (>= 7 pipe fields needed for ID, gene, ..., length fields)
    def min_fields(records):
        return min(len(r.description.split("|")) for r in records)

    ok &= check(
        min_fields(pc) >= 7,
        f"all PC headers have >= 7 pipe fields (min={min_fields(pc)})",
    )
    ok &= check(
        min_fields(lnc) >= 7,
        f"all lncRNA headers have >= 7 pipe fields (min={min_fields(lnc)})",
    )

    # 6. No duplicate IDs within each file
    pc_ids = [enst_id(r) for r in pc]
    lnc_ids = [enst_id(r) for r in lnc]
    ok &= check(
        len(pc_ids) == len(set(pc_ids)),
        f"no duplicate IDs in PC (dupes: {len(pc_ids)-len(set(pc_ids))})",
    )
    ok &= check(
        len(lnc_ids) == len(set(lnc_ids)),
        f"no duplicate IDs in lncRNA (dupes: {len(lnc_ids)-len(set(lnc_ids))})",
    )

    # 7. No ID overlap between PC and lncRNA
    overlap = set(pc_ids) & set(lnc_ids)
    ok &= check(
        len(overlap) == 0,
        f"no ID overlap between PC and lncRNA (overlap: {len(overlap)})",
    )

    # 8. Gzipped files match plain files
    pc_gz_ids = [enst_id(r) for r in pc_gz_records]
    lnc_gz_ids = [enst_id(r) for r in lnc_gz_records]
    ok &= check(pc_ids == pc_gz_ids, "PC gzipped IDs match plain file")
    ok &= check(lnc_ids == lnc_gz_ids, "lncRNA gzipped IDs match plain file")

    # Summary stats
    pc_lens = [len(r.seq) for r in pc]
    lnc_lens = [len(r.seq) for r in lnc]
    print(
        f"\nPC   length: min={min(pc_lens)}, max={max(pc_lens)}, median={sorted(pc_lens)[n//2]}"
    )
    print(
        f"lncRNA length: min={min(lnc_lens)}, max={max(lnc_lens)}, median={sorted(lnc_lens)[n//2]}"
    )

    return ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=N)
    parser.add_argument("--min-len", type=int, default=MIN_LEN)
    args = parser.parse_args()

    passed = validate(n=args.n, min_len=args.min_len)
    print(f"\n{'All checks passed.' if passed else 'Some checks FAILED.'}")
    sys.exit(0 if passed else 1)
