"""
Minimal, framework-free self-check for the combined dropout report.

Feeds a tiny synthetic merged simple_class dataframe (one transcript missing a
tool label) through create_dropout_report and asserts the excluded transcript
appears in the report.

Run with:
    conda run -n lnc-datasets python workflow/scripts/test_merge_folds_dropout.py
"""

import sys
from pathlib import Path

import pandas as pd

# Make workflow utils importable regardless of CWD
workflow_dir = Path(__file__).resolve().parent.parent
if str(workflow_dir) not in sys.path:
    sys.path.insert(0, str(workflow_dir))

from utils.process_tools import create_dropout_report


def test_excluded_transcript_appears():
    # Synthetic merged simple_class table: index = seq_ID, label_<tool> + biotype.
    # "fold" column is present as it is in the real merged table, to prove it is
    # ignored by the report (only label_ columns matter).
    df = pd.DataFrame(
        {
            "fold": ["fold1", "fold2", "fold3"],
            "label_cpat": ["coding", "noncoding", "coding"],
            "label_feelnc": ["coding", None, "noncoding"],  # ENST2 missing feelnc
            "biotype": ["protein_coding", "lncRNA", "protein_coding"],
        },
        index=pd.Index(["ENST1", "ENST2", "ENST3"], name="seq_ID"),
    )

    report = create_dropout_report(df)

    # The transcript missing a tool label must appear; fully-labelled ones must not.
    assert "ENST2" in report.index, "excluded transcript ENST2 missing from report"
    assert "ENST1" not in report.index, "fully-labelled ENST1 wrongly reported"
    assert "ENST3" not in report.index, "fully-labelled ENST3 wrongly reported"

    # The report must name the tool that produced no output.
    assert "feelnc" in report.loc["ENST2", "missing_tools"]
    assert report.loc["ENST2", "n_missing"] == 1
    assert report.loc["ENST2", "biotype"] == "lncRNA"

    print("OK: create_dropout_report flags the excluded transcript (ENST2).")


if __name__ == "__main__":
    test_excluded_transcript_appears()
