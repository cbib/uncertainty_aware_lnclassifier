#!/usr/bin/env python3
"""
Parse zenodo_deposit.md, zip each file into a folder named after its ## section,
and embed a manifest.json for later extraction to original paths.

Usage: python create_zenodo_zip.py [--md zenodo_deposit.md] [--out zenodo_deposit.zip]
"""
import argparse
import json
import re
import zipfile
from pathlib import Path

PAPER_DIR = Path(__file__).resolve().parent


def parse_deposit_md(md_path: Path) -> dict[str, list[Path]]:
    """Return {section_title: [absolute_paths]} preserving order.

    Handles both regular spaces and non-breaking spaces (U+00A0) after ##.
    """
    sections: dict[str, list[Path]] = {}
    current = None
    for raw_line in md_path.read_text(encoding="utf-8").splitlines():
        # Normalise non-breaking spaces → regular spaces, then strip
        line = raw_line.replace("\u00a0", " ").strip()
        if re.match(r"^## ", line):
            current = line[3:].strip()
            sections[current] = []
        elif line and current is not None:
            sections[current].append(Path(line))
    return sections


def section_to_folder(title: str) -> str:
    """Convert '## Pipeline tables' → 'Pipeline_tables'."""
    return re.sub(r"\s+", "_", title)


def build_zip(sections: dict[str, list[Path]], out_path: Path) -> None:
    manifest: dict[str, str] = {}  # zip_path → relative-from-paper path

    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for title, paths in sections.items():
            folder = section_to_folder(title)
            for abs_path in paths:
                if not abs_path.exists():
                    print(f"  WARNING: not found, skipping: {abs_path}")
                    continue
                zip_entry = f"{folder}/{abs_path.name}"
                rel_path = abs_path.relative_to(PAPER_DIR)
                if rel_path.parts[0] == "revision-wt":
                    rel_path = Path(*rel_path.parts[1:])
                manifest[zip_entry] = str(rel_path)
                zf.write(abs_path, zip_entry)
                print(f"  + {zip_entry}")

        zf.writestr("manifest.json", json.dumps(manifest, indent=2))
        print(f"  + manifest.json")

    print(f"\nZip created: {out_path}  ({out_path.stat().st_size / 1024:.1f} KB)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--md", default="zenodo_deposit.md")
    parser.add_argument("--out", default="zenodo_deposit.zip")
    args = parser.parse_args()

    md_path = PAPER_DIR / args.md
    out_path = PAPER_DIR / args.out

    sections = parse_deposit_md(md_path)
    print(f"Sections found: {list(sections.keys())}")
    build_zip(sections, out_path)


if __name__ == "__main__":
    main()
