#!/usr/bin/env python3
"""Run detection for a PDF unless output already exists."""

from __future__ import annotations

import argparse
from pathlib import Path

import json

from pipeline_steps import (
    consistency_check,
    detection,
    evaluate,
    extraction,
    summarize,
)


def _process_pdf(pdf_name: str) -> None:
    study_name = Path(pdf_name).stem
    output_path = Path("data/detection") / f"{study_name}.json"
    if output_path.exists():
        print(f"Skipping detection; output exists at {output_path}")
    else:
        output_path = detection.run(pdf_name)

    if not output_path.exists():
        raise FileNotFoundError(f"Detection output not found at {output_path}")

    detection_data = json.loads(output_path.read_text(encoding="utf-8"))
    flagged_pages = []
    for page in detection_data.get("pages", []):
        result = str(page.get("result", "")).strip().upper()
        if result in ("A", "C"):
            flagged_pages.append(page.get("number"))

    flagged_pages = [p for p in flagged_pages if isinstance(p, int)]
    if not flagged_pages:
        print(f"No pages flagged for extraction in {study_name}.")
        return

    output_dir = Path("data/extraction")
    output_dir.mkdir(parents=True, exist_ok=True)

    extraction_outputs = []
    extraction_pages = []
    if len(flagged_pages) == 1:
        output_path = output_dir / f"{study_name}.json"
        if output_path.exists():
            print(f"Skipping extraction; output exists at {output_path}")
            extraction_outputs.append(output_path)
        else:
            extraction_outputs = extraction.run(
                pdf_name,
                page_numbers=flagged_pages,
            )
        extraction_pages = [flagged_pages[0]]
    else:
        suffixes = [chr(ord("a") + index) for index in range(len(flagged_pages))]
        output_paths = [
            output_dir / f"{study_name}{suffix}.json" for suffix in suffixes
        ]
        if all(path.exists() for path in output_paths):
            print("Skipping extraction; all output files already exist.")
            extraction_outputs = output_paths
        else:
            extraction_outputs = extraction.run(
                pdf_name,
                page_numbers=flagged_pages,
            )
        extraction_pages = flagged_pages

    image_dir = detection._resolve_image_dir(study_name)
    dataset = image_dir.parent.name
    suffixes = (
        [None]
        if len(extraction_outputs) == 1
        else [chr(ord("a") + index) for index in range(len(extraction_outputs))]
    )
    for extraction_path, page_number, suffix in zip(
        extraction_outputs, extraction_pages, suffixes
    ):
        attempts = 0
        while True:
            try:
                evaluate.run(
                    str(extraction_path),
                    dataset,
                    "data/ground_truth_codings.xlsx",
                )
                break
            except json.JSONDecodeError:
                attempts += 1
                print(
                    f"JSON decode error for {extraction_path}. "
                    f"Retrying extraction ({attempts}/3)..."
                )
                if attempts >= 3:
                    print(
                        f"Failed to parse extraction after {attempts} attempts: "
                        f"{extraction_path}"
                    )
                    break
                retry_outputs = extraction.run(
                    pdf_name,
                    page_numbers=[page_number],
                    output_suffix=suffix,
                )
                extraction_path = retry_outputs[0]

    consistency_check.run(study_name)
    summarize.run(study_name)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run detection/extraction/evaluation for PDFs."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--pdf",
        help="PDF file name (e.g., wise2000.pdf) used to locate images/output.",
    )
    group.add_argument(
        "--dataset",
        help="Dataset name: tune, val, eval (or tuning/validation).",
    )
    args = parser.parse_args()

    if args.pdf:
        _process_pdf(args.pdf)
        return

    dataset_map = {
        "tune": "tuning",
        "tuning": "tuning",
        "val": "validation",
        "validation": "validation",
        "eval": "eval",
    }
    dataset = dataset_map.get(args.dataset)
    if dataset is None:
        raise ValueError(
            "Unknown dataset. Expected one of: "
            + ", ".join(sorted(dataset_map.keys()))
        )
    dataset_dir = Path("data/intermediate_data") / dataset
    pdf_paths = sorted(dataset_dir.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDFs found in {dataset_dir}")

    for pdf_path in pdf_paths:
        _process_pdf(pdf_path.name)


if __name__ == "__main__":
    main()
