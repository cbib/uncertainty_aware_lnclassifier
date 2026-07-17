#!/usr/bin/env python3
"""
Merge all supplementary pipeline feature files into a single cleaned TSV.

Applies per-pipeline index/column transformations via load_supplementary_features(),
checks coverage against the main analysis transcript index via check_supplementary_coverage(),
and writes the merged feature matrix + exclusion report.

Downstream rules consume the merged TSV directly instead of loading individual
pipeline files.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))
from utils.features import check_supplementary_coverage, load_supplementary_features


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--main-index",
        required=True,
        help="TSV with the main analysis transcript index (seq_ID as index column)",
    )
    p.add_argument("--te-features-rna", default="")
    p.add_argument("--te-features-dna", default="")
    p.add_argument("--nbd-features", default="")
    p.add_argument("--scanfold-features", default="")
    p.add_argument("--rg4-features", default="")
    p.add_argument(
        "--output", required=True, help="Output path for merged features TSV"
    )
    p.add_argument(
        "--excl-output", required=True, help="Output path for exclusion report TSV"
    )
    return p.parse_args()


def merge_features(
    clean_index: pd.Index, active: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """Concatenate active pipeline DataFrames, reindexed to clean_index, zeroing intra-pipeline NaN."""
    if not active:
        return pd.DataFrame(index=clean_index)

    merged = pd.concat(
        [df.reindex(clean_index, fill_value=0) for df in active.values()],
        axis=1,
    )
    merged.index.name = "seq_ID"
    return merged


def main():
    args = parse_args()

    main_index = pd.read_csv(args.main_index, sep="\t", index_col=0).index
    print(f"Main index: {len(main_index):,} transcripts", file=sys.stderr)

    supplementary = load_supplementary_features(
        te_rna_path=args.te_features_rna,
        te_dna_path=args.te_features_dna,
        nbd_path=args.nbd_features,
        scanfold_path=args.scanfold_features,
        rg4_path=args.rg4_features,
    )

    clean_index, excl_report = check_supplementary_coverage(main_index, supplementary)
    excl_report.to_csv(args.excl_output, sep="\t")
    if not excl_report.empty:
        print(
            f"⚠ Excluded {len(excl_report)} transcripts — see {args.excl_output}",
            file=sys.stderr,
        )

    active = {k: v for k, v in supplementary.items() if not v.empty}
    merged = merge_features(clean_index, active)
    if not active:
        print(
            "No supplementary features loaded — writing empty output.", file=sys.stderr
        )
    merged.to_csv(args.output, sep="\t")
    print(
        f"✓ Merged features: {merged.shape[0]:,} transcripts × {merged.shape[1]} features → {args.output}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
