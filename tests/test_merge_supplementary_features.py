import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "workflow" / "scripts"))
from merge_supplementary_features import merge_features


def test_merge_features_concatenates_and_zero_fills_intra_pipeline_nan():
    clean_index = pd.Index(["t1", "t2"], name="seq_ID")
    active = {
        "te_rna": pd.DataFrame({"rna_orf_len": [10, 20]}, index=["t1", "t2"]),
        "nbd": pd.DataFrame({"apr_count": [1, None]}, index=["t1", "t2"]),
    }
    merged = merge_features(clean_index, active)
    assert list(merged.columns) == ["rna_orf_len", "apr_count"]
    assert merged.loc["t2", "apr_count"] == 0
    assert merged.index.name == "seq_ID"


def test_merge_features_no_active_pipelines_returns_empty_columns():
    clean_index = pd.Index(["t1", "t2"], name="seq_ID")
    merged = merge_features(clean_index, {})
    assert merged.shape == (2, 0)
    assert list(merged.index) == ["t1", "t2"]
