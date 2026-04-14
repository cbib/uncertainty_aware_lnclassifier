"""compute_entropy_groups.py — Assign entropy groups from uncertainty analysis TSV.

Reads the uncertainty analysis TSV (output of compute_entropy_metrics) and
assigns each transcript to an entropy group based on the configured thresholding
strategy:

  overall        — percentile thresholds applied across all transcripts.
  class_separated — percentiles applied independently within coding / lncRNA.

Output TSV columns:
  seq_ID (index)
  entropy_group : low | high | middle                                    (overall)
                  low_lncRNA | high_lncRNA | low_coding | high_coding | middle
                                                                 (class_separated)

Usage
-----
python workflow/scripts/compute_entropy_groups.py \\
    --entropy-tsv results/.../features/entropy/..._uncertainty_analysis.tsv \\
    --output-tsv  results/.../features/entropy/..._entropy_groups.tsv \\
    --mode        class_separated \\
    --low-th      10 \\
    --high-th     90
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))

from utils.entropy import assign_entropy_groups


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Assign entropy groups per transcript")
    p.add_argument(
        "--entropy-tsv",
        required=True,
        metavar="FILE",
        help="Uncertainty analysis TSV (output of compute_entropy_metrics)",
    )
    p.add_argument(
        "--output-tsv",
        required=True,
        metavar="FILE",
        help="Output TSV with entropy_group column",
    )
    p.add_argument(
        "--mode",
        default="overall",
        choices=["overall", "class_separated"],
        help="Thresholding strategy (default: overall)",
    )
    p.add_argument(
        "--low-th",
        type=int,
        default=10,
        metavar="N",
        help="Lower percentile threshold (default: 10)",
    )
    p.add_argument(
        "--high-th",
        type=int,
        default=90,
        metavar="N",
        help="Upper percentile threshold (default: 90)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    entropy_df = pd.read_csv(args.entropy_tsv, sep="\t", index_col=0)

    print(f"Mode       : {args.mode}")
    print(f"Thresholds : {args.low_th}th / {args.high_th}th percentile")
    print(f"Transcripts: {len(entropy_df):,}")

    groups = assign_entropy_groups(
        entropy_df,
        mode=args.mode,
        low_th=args.low_th,
        high_th=args.high_th,
        entropy_column="H_pred",
        entropy_column_high="I_bald",
        high_threshold_sec=args.high_th,
        class_column="coding_class",
    )

    out = Path(args.output_tsv)
    out.parent.mkdir(parents=True, exist_ok=True)
    groups.to_frame().to_csv(out, sep="\t")

    counts = groups.value_counts()
    print("\nGroup counts:")
    for grp, n in counts.items():
        print(f"  {grp}: {n:,}")
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
