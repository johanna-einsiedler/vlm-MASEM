"""
tune_eval_split.py

Two-fold split of PDFs in 'data/raw_data/':
  (1) 10 tuning papers  → data/intermediate_data/tuning/
  (2) remaining papers  → data/intermediate_data/eval/

Also writes 'data/dataset_assignments.csv' with columns:
  identifier, filename, assignment, doi

The 'doi' column is left blank for manual completion.
Supplemental/translation files (e.g. *_translation.pdf, *supplemental*.pdf)
are copied alongside their parent study but not counted toward the tuning quota.

Usage:
    python utils/tune_eval_split.py [--seed SEED]
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw_data"
DATA_DIR = ROOT / "data" / "intermediate_data"
TUNING_DIR = DATA_DIR / "tuning"
EVAL_DIR = DATA_DIR / "eval"
CSV_PATH = ROOT / "data" / "dataset_assignments.csv"
TUNING_SIZE = 10

# Patterns that mark a file as supplemental/translation (not a primary study)
SUPPLEMENTAL_PATTERNS = re.compile(
    r"(_translation|_supplemental|supplemental_material)", re.IGNORECASE
)


def _identifier(filename: str) -> str:
    """Return the study identifier from a filename (stem, no suffix)."""
    return Path(filename).stem


def _is_supplemental(filename: str) -> bool:
    return bool(SUPPLEMENTAL_PATTERNS.search(filename))


def main(seed: int | None = None) -> None:
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"raw_data directory not found: {RAW_DIR}")

    if seed is not None:
        random.seed(seed)

    # Separate primary PDFs from supplemental/translation files
    all_pdfs = sorted(f.name for f in RAW_DIR.glob("*.pdf"))
    primary_pdfs = [f for f in all_pdfs if not _is_supplemental(f)]
    supplemental_pdfs = [f for f in all_pdfs if _is_supplemental(f)]

    non_pdf = [
        f.name for f in RAW_DIR.iterdir()
        if f.suffix.lower() in (".doc", ".docx")
    ]
    if non_pdf:
        print(f"Note: {len(non_pdf)} Word file(s) found — skipped:")
        for f in non_pdf:
            print(f"  {f}")

    if len(primary_pdfs) < TUNING_SIZE:
        raise ValueError(
            f"Not enough primary PDFs to sample {TUNING_SIZE} "
            f"(found {len(primary_pdfs)})"
        )

    # Random split
    tuning_set = set(random.sample(primary_pdfs, TUNING_SIZE))
    eval_set = set(primary_pdfs) - tuning_set

    # Create output dirs
    TUNING_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    # Copy primary PDFs
    assignments: list[dict] = []
    for pdf in sorted(tuning_set):
        shutil.copy2(RAW_DIR / pdf, TUNING_DIR / pdf)
        assignments.append({"identifier": _identifier(pdf), "filename": pdf,
                             "assignment": "tuning", "doi": ""})

    for pdf in sorted(eval_set):
        shutil.copy2(RAW_DIR / pdf, EVAL_DIR / pdf)
        assignments.append({"identifier": _identifier(pdf), "filename": pdf,
                             "assignment": "eval", "doi": ""})

    # Copy supplemental files alongside their parent study
    for supp in supplemental_pdfs:
        # Match to parent by checking which primary study is a prefix
        parent = next(
            (p for p in primary_pdfs if supp.startswith(Path(p).stem.split("_")[0])),
            None,
        )
        dest_dir = TUNING_DIR if (parent in tuning_set) else EVAL_DIR
        shutil.copy2(RAW_DIR / supp, dest_dir / supp)
        assignment = "tuning" if parent in tuning_set else "eval"
        assignments.append({"identifier": _identifier(supp), "filename": supp,
                             "assignment": assignment, "doi": ""})

    # Sort by identifier for readable CSV
    assignments.sort(key=lambda r: r["identifier"])

    # Write CSV
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["identifier", "filename", "assignment", "doi"])
        writer.writeheader()
        writer.writerows(assignments)

    print(f"\nTuning set  ({len(tuning_set):2d} papers): {TUNING_DIR}")
    print(f"Eval set    ({len(eval_set):2d} papers): {EVAL_DIR}")
    if supplemental_pdfs:
        print(f"Supplemental ({len(supplemental_pdfs):2d} files): copied alongside parent study")
    print(f"\nAssignment CSV written to: {CSV_PATH}")
    print("\nTuning papers:")
    for p in sorted(tuning_set):
        print(f"  {p}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split raw PDFs into tuning/eval sets.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility.")
    args = parser.parse_args()
    main(seed=args.seed)
