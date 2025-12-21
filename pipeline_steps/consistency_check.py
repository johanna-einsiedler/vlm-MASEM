#!/usr/bin/env python3
"""Check extraction outputs for structural consistency."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List


def _extract_json_blob(text: str) -> Dict:
    if not isinstance(text, str):
        raise ValueError("Extraction result is not a string.")

    fenced = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        return json.loads(fenced.group(1))

    fenced_generic = re.search(r"```\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced_generic:
        return json.loads(fenced_generic.group(1))

    inline = re.search(r"(\{.*\})", text, re.DOTALL)
    if inline:
        return json.loads(inline.group(1))

    return json.loads(text)


def _expected_keys() -> List[str]:
    keys = []
    for factor in ("F1", "F2", "F3"):
        for idx in range(1, 21):
            keys.append(f"{factor}.{idx}")
    return keys


def _values_in_range(values: Iterable[float]) -> bool:
    for value in values:
        if value is None:
            return False
        if not (-1 <= value <= 1):
            return False
    return True


def _notes_indicate_continued(notes: str) -> bool:
    if not notes:
        return False
    lowered = notes.lower()
    return any(
        phrase in lowered
        for phrase in ("continued", "cont.", "continues", "next page")
    )


def _all_values_zero(values: Iterable[float]) -> bool:
    has_any = False
    for value in values:
        if value is None:
            return False
        has_any = True
        if value != 0:
            return False
    return has_any


def _gather_samples(extraction_data: dict) -> List[dict]:
    pages = extraction_data.get("pages") or []
    if not pages:
        return []
    payload = _extract_json_blob(pages[0].get("result", ""))
    return payload.get("samples") or []


def run(study: str) -> Path:
    study = study
    extraction_paths = sorted(Path("data/extraction").glob(f"{study}*.json"))
    if not extraction_paths:
        raise FileNotFoundError(f"No extraction files found for {study}")

    expected_keys = _expected_keys()
    results = {}

    for extraction_path in extraction_paths:
        extraction_data = json.loads(extraction_path.read_text(encoding="utf-8"))
        samples = _gather_samples(extraction_data)
        if not samples:
            continue

        sample = samples[0]
        factor_loadings = sample.get("factor_loadings") or {}
        notes = sample.get("notes") or ""

        values = [factor_loadings.get(key) for key in expected_keys]

        all_fields = all(key in factor_loadings for key in expected_keys)
        possible_values = _values_in_range(values)
        table_likely_continued = _notes_indicate_continued(notes)
        all_zeros = _all_values_zero(values)

        results[extraction_path.stem] = {
            "all_fields": all_fields,
            "possible_values": possible_values,
            "table_likely_continued": table_likely_continued,
            "all_zeros": all_zeros,
        }

    output_dir = Path("consistency_checks")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{study}.json"
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check extraction results for field completeness and validity."
    )
    parser.add_argument(
        "study",
        help="Study name used to locate extraction output (e.g., erni1997).",
    )
    args = parser.parse_args()

    run(args.study)


if __name__ == "__main__":
    main()
