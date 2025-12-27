#!/usr/bin/env python3
"""Run page-level detection with Qwen on pre-rendered PNGs."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Iterable, Tuple

from PIL import Image
from dotenv import load_dotenv

from document_parsing.vlm_qwen import QwenParser

PROMPT_PATH = Path("prompts/01_detection_prompt.txt")
ALT_PROMPT_PATH = Path(
    "/Users/einsie0004/Documents/research/33_llm_replicability/"
    "vlm_pipeline/prompts/01_detection_prompt.txt"
)


def _load_prompt_text() -> str:
    prompt_path = ALT_PROMPT_PATH if ALT_PROMPT_PATH.exists() else PROMPT_PATH
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt not found at {prompt_path}")
    return prompt_path.read_text(encoding="utf-8").strip()


def _page_sort_key(path: Path) -> Tuple[int, str]:
    match = re.search(r"\d+", path.stem)
    return (int(match.group()) if match else 0, path.name)


def _iter_pngs(image_dir: Path) -> Iterable[Path]:
    pngs = sorted(image_dir.glob("*.png"), key=_page_sort_key)
    if not pngs:
        raise FileNotFoundError(f"No PNGs found in {image_dir}")
    return pngs


def _resolve_image_dir(study_name: str) -> Path:
    images_root = Path("data/image_data")
    matches = list(images_root.glob(f"*/{study_name}"))
    if matches:
        return matches[0]

    raise FileNotFoundError(
        f"Image folder not found in {images_root} for study '{study_name}'."
    )


def _progress_bar(iterable: Iterable[Path], total: int) -> Iterable[Path]:
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

    if tqdm is None:
        for index, item in enumerate(iterable, start=1):
            print(f"Processing {index}/{total}: {item.name}", end="\r")
            yield item
        print()
        return

    yield from tqdm(iterable, total=total, desc="Processing pages", unit="page")


def _serialize_logprobs(logprobs):
    if logprobs is None:
        return None
    if hasattr(logprobs, "model_dump"):
        return logprobs.model_dump()
    if hasattr(logprobs, "dict"):
        return logprobs.dict()
    if isinstance(logprobs, dict):
        return logprobs
    return logprobs


def _extract_class_logprobs(logprobs):
    data = _serialize_logprobs(logprobs)
    if not isinstance(data, dict):
        return {"A": None, "B": None, "C": None, "D": None}

    tokens = data.get("tokens") or []
    token_logprobs = data.get("token_logprobs") or []
    top_logprobs = data.get("top_logprobs") or []
    content = data.get("content") or []

    result = {"A": None, "B": None, "C": None, "D": None}

    for target in ("A", "B", "C", "D"):
        if tokens and token_logprobs:
            for token, logprob in zip(tokens, token_logprobs):
                if token == target:
                    result[target] = logprob
                    break

        if result[target] is None and top_logprobs:
            best = None
            for entry in top_logprobs:
                if isinstance(entry, dict) and target in entry:
                    value = entry[target]
                    if best is None or (value is not None and value > best):
                        best = value
            result[target] = best

        if result[target] is None and content:
            for entry in content:
                if not isinstance(entry, dict):
                    continue
                if entry.get("token") == target:
                    result[target] = entry.get("logprob")
                    break
            if result[target] is None:
                best = None
                for entry in content:
                    if not isinstance(entry, dict):
                        continue
                    for candidate in entry.get("top_logprobs") or []:
                        if (
                            isinstance(candidate, dict)
                            and candidate.get("token") == target
                        ):
                            value = candidate.get("logprob")
                            if best is None or (value is not None and value > best):
                                best = value
                result[target] = best

    return result


def run(pdf_name: str) -> Path:
    load_dotenv()
    study_name = Path(pdf_name).stem
    image_dir = _resolve_image_dir(study_name)
    prompt_text = _load_prompt_text()

    api_key = os.environ.get("TOGETHER_API_KEY")
    parser_client = QwenParser(api_key=api_key)

    png_paths = list(_iter_pngs(image_dir))
    pages = []
    for page_path in _progress_bar(png_paths, total=len(png_paths)):
        with Image.open(page_path) as img:
            result, logprobs = parser_client.parse_page(
                img.convert("RGB"),
                prompt=prompt_text,
                logprobs=4,
                return_logprobs=True,
            )
        page_number_match = re.search(r"\d+", page_path.stem)
        page_number = int(page_number_match.group()) if page_number_match else len(pages) + 1
        page_entry = {"number": page_number, "result": result}
        if logprobs is not None:
            page_entry["logprobs"] = _extract_class_logprobs(logprobs)
        pages.append(page_entry)

    output = {"study": study_name, "pages": pages}

    output_dir = Path("data/detection")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{study_name}.json"
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run Qwen detection on PNG pages that correspond to a PDF name."
        )
    )
    parser.add_argument(
        "pdf_name",
        help="PDF file name (e.g., wise2000.pdf) used to locate the image folder.",
    )
    args = parser.parse_args()
    run(args.pdf_name)


if __name__ == "__main__":
    main()
