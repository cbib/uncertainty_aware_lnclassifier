import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from zipfile import ZipFile

import create_zenodo_zip


def test_manifest_restores_revision_files_to_pipeline_paths():
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        original = root / "results" / "same.csv"
        revised = root / "revision-wt" / "results" / "same.csv"
        for path in (original, revised):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(path.as_posix())

        archive = root / "data.zip"
        with patch.object(create_zenodo_zip, "PAPER_DIR", root):
            create_zenodo_zip.build_zip(
                {"Original": [original], "Revised": [revised]}, archive
            )

        with ZipFile(archive) as zf:
            manifest = json.loads(zf.read("manifest.json"))
            assert manifest == {
                "Original/same.csv": "results/same.csv",
                "Revised/same.csv": "results/same.csv",
            }
            assert set(zf.namelist()) == {*manifest, "manifest.json"}
