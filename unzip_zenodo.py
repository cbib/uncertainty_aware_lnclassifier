#!/usr/bin/env python3
"""
Unzip zenodo_deposit.zip, restoring each file to its original path
relative to the paper directory.

Usage: python unzip_zenodo.py [--zip zenodo_deposit.zip] [--paper-dir /path/to/paper]

--paper-dir defaults to the directory containing this script.
"""
import argparse
import json
import zipfile
from pathlib import Path


def main() -> None:
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--zip",
        default="zenodo_deposit.zip",
        help="Path to the zip file (default: zenodo_deposit.zip next to this script)",
    )
    parser.add_argument(
        "--paper-dir",
        default=str(script_dir),
        help="Root paper directory where 'results/' lives (default: script directory)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print destination paths without writing files",
    )
    args = parser.parse_args()

    zip_path = Path(args.zip) if Path(args.zip).is_absolute() else script_dir / args.zip
    paper_dir = Path(args.paper_dir).resolve()

    if not zip_path.exists():
        raise FileNotFoundError(f"Zip not found: {zip_path}")

    with zipfile.ZipFile(zip_path, "r") as zf:
        if "manifest.json" not in zf.namelist():
            raise ValueError(
                "manifest.json missing from zip — was this created by create_zenodo_zip.py?"
            )

        manifest: dict[str, str] = json.loads(zf.read("manifest.json"))

        for zip_entry, rel_path in manifest.items():
            dest = paper_dir / rel_path
            if args.dry_run:
                print(f"  {zip_entry}  →  {dest}")
                continue

            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(zf.read(zip_entry))
            print(f"  restored: {dest}")

    if args.dry_run:
        print("\n(dry run — no files written)")
    else:
        print(f"\nDone. Files restored under: {paper_dir}")


if __name__ == "__main__":
    main()
