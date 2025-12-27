#!/usr/bin/env python3
"""Run page-level extraction with PaddleOCR PPStructureV3 on image files."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv
from paddleocr import PPStructureV3

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from document_parsing.llm_deepseek import DeepSeekParser

PROMPT_PATH = Path("prompts/02_md_extraction_prompt.txt")
ALT_PROMPT_PATH = Path(
    "/Users/einsie0004/Documents/research/33_llm_replicability/"
    "vlm_pipeline/prompts/02_md_extraction_prompt.txt"
)


def _load_prompt_text() -> str:
    prompt_path = ALT_PROMPT_PATH if ALT_PROMPT_PATH.exists() else PROMPT_PATH
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt not found at {prompt_path}")
    return prompt_path.read_text(encoding="utf-8").strip()


def _resolve_image_dir(study_name: str, dataset: str | None = None) -> Path:
    image_root = Path("data/image_data")

    # Try dataset-specific location first if dataset provided
    if dataset:
        image_dir = image_root / dataset / study_name
        if image_dir.exists():
            return image_dir

    # Fall back to non-dataset location
    image_dir = image_root / study_name
    if image_dir.exists():
        return image_dir

    raise FileNotFoundError(
        f"Image data folder not found: {image_dir}"
    )


def _resolve_detection_file(study_name: str) -> Path:
    detection_path = Path("data/detection") / f"{study_name}.json"
    if not detection_path.exists():
        raise FileNotFoundError(
            f"Detection file not found: {detection_path}. "
            "Run detection first to identify relevant pages."
        )
    return detection_path


def _get_flagged_pages(detection_path: Path) -> list[int]:
    """Get page numbers flagged as A or C from detection results."""
    detection_data = json.loads(detection_path.read_text(encoding="utf-8"))
    flagged_pages = []
    for page in detection_data.get("pages", []):
        result = str(page.get("result", "")).strip().upper()
        if result in ("A", "C"):
            flagged_pages.append(page.get("number"))

    flagged_pages = [p for p in flagged_pages if isinstance(p, int)]
    return flagged_pages


def _progress_bar(iterable: Iterable, total: int, desc: str = "Processing") -> Iterable:
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

    if tqdm is None:
        for index, item in enumerate(iterable, start=1):
            print(f"{desc} {index}/{total}", end="\r")
            yield item
        print()
        return

    yield from tqdm(iterable, total=total, desc=desc, unit="page")


def run(
    pdf_name: str,
    page_numbers: Iterable[int] | None = None,
    output_suffix: str | None = None,
    force: bool = False,
    model_name: str = "openai/gpt-oss-120b",
    dataset: str | None = None,
) -> list[Path]:
    load_dotenv()
    study_name = Path(pdf_name).stem
    image_dir = _resolve_image_dir(study_name, dataset)
    prompt_text = _load_prompt_text()

    # If page_numbers not specified, get them from detection results
    if page_numbers is None:
        detection_path = _resolve_detection_file(study_name)
        page_numbers = _get_flagged_pages(detection_path)
        if not page_numbers:
            print(f"No pages flagged for extraction in {study_name}.")
            return []

    # Create model-specific output directory with optional dataset subfolder
    model_suffix = "paddle"
    if dataset:
        output_dir = Path(f"data/md_extraction_{model_suffix}") / dataset
    else:
        output_dir = Path(f"data/md_extraction_{model_suffix}")

    if not force:
        page_list = list(page_numbers)
        if len(page_list) == 1:
            # Single page: check for {study}.json
            output_path = output_dir / f"{study_name}.json"
            if output_path.exists():
                print(f"Skipping {study_name}: already extracted")
                return [output_path]
        else:
            # Multiple pages: check for {study}a.json, {study}b.json, etc.
            expected_files = [
                output_dir / f"{study_name}{chr(ord('a') + i)}.json"
                for i in range(len(page_list))
            ]
            if all(f.exists() for f in expected_files):
                print(f"Skipping {study_name}: already extracted ({len(expected_files)} files)")
                return expected_files

    api_key = os.environ.get("TOGETHER_API_KEY")
    if not api_key:
        raise ValueError(
            "TOGETHER_API_KEY not found in environment. "
            "Please set it in your .env file."
        )

    # Initialize PaddleOCR PPStructureV3 pipeline
    paddle_pipeline = PPStructureV3(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False
    )

    parser_client = DeepSeekParser(api_key=api_key, max_tokens=10000, model_name=model_name)

    pages = []
    page_list = list(page_numbers)

    # Create temporary directory for paddle markdown outputs
    temp_dir = Path("output/paddle_temp")
    temp_dir.mkdir(parents=True, exist_ok=True)

    for page_number in _progress_bar(page_list, total=len(page_list), desc="Extracting"):
        # Look for image file (try common extensions)
        image_path = None
        for ext in [".png", ".jpg", ".jpeg", ".tiff", ".tif"]:
            candidate = image_dir / f"page{page_number}{ext}"
            if candidate.exists():
                image_path = candidate
                break

        if not image_path:
            print(f"Warning: Image file not found for page {page_number}")
            continue

        # Convert image to markdown using PaddleOCR
        output = paddle_pipeline.predict(input=str(image_path))

        # Save markdown to temporary location
        for res in output:
            res.save_to_markdown(save_path=str(temp_dir))

        # Read the generated markdown
        # PPStructureV3 saves with the image filename
        markdown_file = temp_dir / f"{image_path.stem}.md"
        if not markdown_file.exists():
            print(f"Warning: Markdown file not generated for page {page_number}")
            continue

        markdown_content = markdown_file.read_text(encoding="utf-8")

        # Extract structured data using LLM
        result = parser_client.parse_text(
            text=markdown_content,
            prompt=prompt_text,
        )

        pages.append({"number": page_number, "result": result})

    if not pages:
        print(f"No pages were successfully extracted for {study_name}.")
        return []

    output_paths: list[Path] = []
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
        description=(
            "Run PaddleOCR PPStructureV3 extraction on image pages."
        )
    )
    parser.add_argument(
        "pdf_name",
        help="PDF file name (e.g., wise2000.pdf) used to locate the image data folder.",
    )
    parser.add_argument(
        "--pages",
        nargs="*",
        type=int,
        help="Optional page numbers to extract (e.g., --pages 1 2 5). "
             "If not specified, uses detection results to identify relevant pages.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-extraction even if output already exists.",
    )
    parser.add_argument(
        "--model",
        default="openai/gpt-oss-120b",
        help="Model name to use for extraction (default: openai/gpt-oss-120b)",
    )
    parser.add_argument(
        "--dataset",
        help="Optional dataset name for organizing outputs (e.g., tuning, validation)",
    )
    args = parser.parse_args()
    run(args.pdf_name, page_numbers=args.pages, force=args.force, model_name=args.model, dataset=args.dataset)


if __name__ == "__main__":
    main()
