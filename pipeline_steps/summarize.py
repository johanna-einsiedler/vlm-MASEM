#!/usr/bin/env python3
"""Summarize detection confidence for each page."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional


def _confidence_label(score: Optional[float]) -> str:
    if score is None:
        return "unknown"
    if score < 0.3:
        return "very uncertain"
    if score < 1.0:
        return "weak evidence"
    if score < 2.0:
        return "moderate"
    if score <= 4.0:
        return "strong"
    return "extremely strong"


def _confidence_score(result: str, logprobs: dict) -> Optional[float]:
    if not isinstance(logprobs, dict):
        return None
    result = result.upper()
    if result not in logprobs:
        return None
    result_lp = logprobs.get(result)
    if result_lp is None:
        return None
    other_values = [
        value
        for key, value in logprobs.items()
        if key != result and value is not None
    ]
    if not other_values:
        return None
    return result_lp - max(other_values)


def run(study: str) -> Path:
    study = study
    detection_path = Path("data/detection") / f"{study}.json"
    if not detection_path.exists():
        raise FileNotFoundError(f"Detection file not found: {detection_path}")

    detection_data = json.loads(detection_path.read_text(encoding="utf-8"))
    pages = detection_data.get("pages") or []

    summaries = []
    relevant_pages = []
    for page in pages:
        result = str(page.get("result", "")).strip().upper()
        logprobs = page.get("logprobs") or {}
        score = _confidence_score(result, logprobs)
        page_number = page.get("number")
        image_matches = list(
            Path("data/image_data").glob(f"*/{study}/page{page_number}.png")
        )
        page_path = str(image_matches[0]) if image_matches else None
        summaries.append(
            {
                "number": page_number,
                "relevance": result in {"A", "B", "C"},
                "confidence": _confidence_label(score),
            }
        )
        if result in {"A", "B", "C"}:
            relevant_pages.append({"number": page_number, "path": page_path})

    extraction_dir = Path("data/extraction")
    extraction_paths = sorted(extraction_dir.glob(f"{study}*.json"))
    extractions = []
    for extraction_path in extraction_paths:
        try:
            extractions.append(
                json.loads(extraction_path.read_text(encoding="utf-8"))
            )
        except json.JSONDecodeError:
            extractions.append({"path": str(extraction_path), "error": "invalid_json"})

    output = {
        "study": study,
        "pages": summaries,
        "relevant_pages": relevant_pages,
        "extractions": extractions,
    }
    output_dir = Path("summaries")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{study}.json"
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize detection confidence for a study."
    )
    parser.add_argument(
        "study",
        help="Study name (e.g., erni1997) used to locate detection output.",
    )
    args = parser.parse_args()

    run(args.study)


if __name__ == "__main__":
    main()
