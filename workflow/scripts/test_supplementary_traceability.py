#!/usr/bin/env python3
"""Self-check for build_supplementary_traceability: run directly, asserts on failure."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))
from utils.features import _clean_supplementary, build_supplementary_traceability


def _df(index, val=1.0):
    return pd.DataFrame({"f1": val, "f2": val}, index=pd.Index(index, name="seq_ID"))


def main():
    main_index = pd.Index(["A", "B", "C", "D"], name="seq_ID")

    # p1 covers A,B,C,D fully — but D has a NaN feature (invalid_data)
    p1 = _df(["A", "B", "C", "D"])
    p1.loc["D", "f2"] = np.nan
    # p2 is missing B (missing_from_pipeline) and carries an extra X (not_in_main_index)
    p2 = _df(["A", "C", "D", "X"])
    # p3 has a non-numeric value in C (invalid_data)
    p3 = _df(["A", "B", "C", "D"]).astype(object)
    p3.loc["C", "f1"] = "oops"

    clean, report = build_supplementary_traceability(
        main_index, {"p1": p1, "p2": p2, "p3": p3, "empty": pd.DataFrame()}
    )

    # A is the only transcript present with valid data in every active pipeline.
    assert list(clean) == ["A"], clean

    cats = report.groupby(["category", report.index]).size()
    assert ("missing_from_pipeline", "B") in cats.index  # B absent from p2
    assert ("not_in_main_index", "X") in cats.index  # X only in p2
    assert ("invalid_data", "D") in cats.index  # D NaN in p1
    assert ("invalid_data", "C") in cats.index  # C non-numeric in p3
    assert set(report.columns) == {"category", "pipeline", "detail"}

    # No active pipelines → whole main index kept, empty report with the columns.
    clean2, report2 = build_supplementary_traceability(
        main_index, {"e": pd.DataFrame()}
    )
    assert list(clean2) == list(main_index)
    assert report2.empty and list(report2.columns) == ["category", "pipeline", "detail"]

    # _clean_supplementary: drop metadata/ID cols; presence flags NaN->0, keeping
    # transcripts that used to be wrongly dropped as invalid_data.
    raw = pd.DataFrame(
        {
            "score": [1.0, 2.0],
            "te_has_line": [True, np.nan],  # NaN = element absent = 0
            "transcript_type": ["lncRNA", "coding"],  # non-feature label
            "coding_class": ["lncRNA", "coding"],  # non-feature label
            "z_transcript_id": ["ENST1", "ENST2"],  # non-feature id
        },
        index=pd.Index(["A", "B"], name="seq_ID"),
    )
    cleaned = _clean_supplementary(raw)
    assert list(cleaned.columns) == ["score", "te_has_line"], cleaned.columns
    assert cleaned["te_has_line"].tolist() == [1, 0], cleaned["te_has_line"].tolist()
    assert not cleaned.isna().any().any()

    print("ok")


if __name__ == "__main__":
    main()
