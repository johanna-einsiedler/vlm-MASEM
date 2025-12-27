#!/usr/bin/env python3
"""Run page-level extraction with Qwen on pre-rendered PNGs."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Iterable, Tuple

from dotenv import load_dotenv
from PIL import Image

from document_parsing.vlm_qwen import QwenParser

PROMPT_PATH = Path("prompts/02_extraction_prompt.txt")
ALT_PROMPT_PATH = Path(
    "/Users/einsie0004/Documents/research/33_llm_replicability/"
    "vlm_pipeline/prompts/02_extraction_prompt.txt"
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


def run(
    pdf_name: str,
    page_numbers: Iterable[int] | None = None,
    output_suffix: str | None = None,
) -> list[Path]:
    load_dotenv()
    study_name = Path(pdf_name).stem
    image_dir = _resolve_image_dir(study_name)
    prompt_text = _load_prompt_text()

    api_key = os.environ.get("TOGETHER_API_KEY")
    parser_client = QwenParser(api_key=api_key, max_tokens=10000)

    if page_numbers is None:
        png_paths = list(_iter_pngs(image_dir))
    else:
        png_paths = []
        for page_number in page_numbers:
            page_path = image_dir / f"page{page_number}.png"
            if not page_path.exists():
                raise FileNotFoundError(f"Missing page image: {page_path}")
            png_paths.append(page_path)

    pages = []
    for page_path in _progress_bar(png_paths, total=len(png_paths)):
        with Image.open(page_path) as img:
            result = parser_client.parse_page(
                img.convert("RGB"),
                prompt=prompt_text,
                logprobs=0,
            )
        page_number_match = re.search(r"\d+", page_path.stem)
        page_number = (
            int(page_number_match.group()) if page_number_match else len(pages) + 1
        )
        pages.append({"number": page_number, "result": result})

    output_paths: list[Path] = []
    output_dir = Path("data/extraction")
    output_dir.mkdir(parents=True, exist_ok=True)

    if len(pages) <= 1:
        output = {"study": study_name, "pages": pages}
        if output_suffix:
            output_path = output_dir / f"{study_name}{output_suffix}.json"
        else:
            output_path = output_dir / f"{study_name}.json"
        output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
        output_paths.append(output_path)
    else:
        for index, page in enumerate(pages):
            suffix = chr(ord("a") + index)
            output = {"study": study_name, "pages": [page]}
            output_path = output_dir / f"{study_name}{suffix}.json"
            output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
            output_paths.append(output_path)

    print(json.dumps({"study": study_name, "pages": pages}, indent=2))
    return output_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description=("Run Qwen extraction on PNG pages that correspond to a PDF name.")
    )
    parser.add_argument(
        "pdf_name",
        help="PDF file name (e.g., wise2000.pdf) used to locate the image folder.",
    )
    parser.add_argument(
        "--pages",
        nargs="*",
        type=int,
        help="Optional page numbers to extract (e.g., --pages 1 2 5).",
    )
    args = parser.parse_args()
    run(args.pdf_name, page_numbers=args.pages)


if __name__ == "__main__":
    main()
