"""
compute_entropy.py — Compute uncertainty metrics (H_pred, H_exp, I_bald) for a dataset.

Outputs a TSV with per-transcript entropy values plus coding_class / biotype labels.

Usage:
    python compute_entropy.py \\
        --dataset gencode.v47.common.cdhit.cv \\
        --output-dir results/gencode.v47.common.cdhit.cv/uncertainty_analysis
"""

import argparse
import sys
from pathlib import Path

# ── Path setup (must happen before local imports) ─────────────────────────────
sys.path.insert(0, str(Path(__file__).parents[1]))

from utils.entropy import compute_uncertainty_metrics, load_dataset  # noqa: E402

# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(
        description="Compute per-transcript uncertainty metrics from tool probabilities."
    )
    p.add_argument(
        "--dataset",
        required=True,
        metavar="EXPT",
        help="Experiment / dataset name (e.g. gencode.v47.common.cdhit.cv)",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        metavar="DIR",
        help="Directory where the uncertainty_analysis TSV will be written",
    )
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_tsv = output_dir / f"{args.dataset}_uncertainty_analysis.tsv"

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load dataset ──────────────────────────────────────────────────────────
    print(f"── Loading dataset: {args.dataset} ──────────────────────────────────")
    data = load_dataset(args.dataset)
    probs = data["probs"]
    labels = data["labels"]
    print(f"  Probabilities : {probs.shape[0]:,} transcripts × {probs.shape[1]} tools")
    print(f"  Labels        : {labels['biotype'].value_counts().to_dict()}")

    # ── Compute uncertainty metrics ───────────────────────────────────────────
    print("\n── Computing uncertainty metrics ─────────────────────────────────────")
    entropy_df = compute_uncertainty_metrics(probs)

    # ── Merge with class labels ───────────────────────────────────────────────
    entropy_df["coding_class"] = labels["coding_class"]
    entropy_df["biotype"] = labels["biotype"]

    # ── Save ──────────────────────────────────────────────────────────────────
    entropy_df.index.name = "seq_ID"
    entropy_df.to_csv(output_tsv, sep="\t")
    print(f"\n✓ Saved uncertainty analysis → {output_tsv}")
    print(f"  Shape: {entropy_df.shape[0]:,} rows × {entropy_df.shape[1]} columns")


if __name__ == "__main__":
    main()
