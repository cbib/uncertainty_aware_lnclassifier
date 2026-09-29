import pandas as pd


def _apply_nbd_rename(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={"transcript_length": "unspliced_length"})


def _apply_rg4_drop(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=["transcript_length"], errors="ignore")


def _apply_scanfold_drop(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=["length", "source_dir"], errors="ignore")


def test_nbd_rename_preserves_column():
    df = pd.DataFrame({"transcript_length": [9285, 35745], "apr_count": [3, 7]})
    result = _apply_nbd_rename(df)
    assert "unspliced_length" in result.columns
    assert "transcript_length" not in result.columns
    assert list(result["unspliced_length"]) == [9285, 35745]


def test_rg4_drop_removes_column():
    df = pd.DataFrame({"transcript_length": [2450, 7160], "peak_count": [1, 4]})
    result = _apply_rg4_drop(df)
    assert "transcript_length" not in result.columns
    assert "peak_count" in result.columns


def test_rg4_drop_tolerates_missing_column():
    df = pd.DataFrame({"peak_count": [1, 4]})
    result = _apply_rg4_drop(df)
    assert "peak_count" in result.columns


def test_scanfold_drop_removes_length_and_source_dir():
    df = pd.DataFrame(
        {
            "length": [1200, 3500],
            "source_dir": ["/path/a", "/path/b"],
            "n_paired": [400, 1100],
        }
    )
    result = _apply_scanfold_drop(df)
    assert "length" not in result.columns
    assert "source_dir" not in result.columns
    assert "n_paired" in result.columns


def test_scanfold_drop_tolerates_missing_columns():
    df = pd.DataFrame({"n_paired": [400, 1100]})
    result = _apply_scanfold_drop(df)
    assert "n_paired" in result.columns
