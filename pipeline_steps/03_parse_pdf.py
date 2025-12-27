#!/usr/bin/env python3
"""Parse all pages to markdown using VLM.

For pages marked A, B, or C in detection: uses refined table images (with cell detection overlays).
For pages marked D or other: uses original images.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from document_parsing.vlm_qwen import QwenParser

PROMPT_PATH = Path("prompts/03_parse_prompt.txt")
ALT_PROMPT_PATH = Path(
    "/Users/einsie0004/Documents/research/33_llm_replicability/"
    "vlm_pipeline/prompts/03_parse_prompt.txt"
)


def _load_prompt_text() -> str:
    """Load the parsing prompt."""
    prompt_path = ALT_PROMPT_PATH if ALT_PROMPT_PATH.exists() else PROMPT_PATH
    if not prompt_path.exists():
        # Use default prompt if file not found
        return "Convert this table image to markdown format. Preserve the structure and all values exactly as shown."
    return prompt_path.read_text(encoding="utf-8").strip()


def _resolve_refined_tables_dir(study_name: str, dataset: str | None = None) -> Path:
    """Resolve the refined tables directory for a given study."""
    refined_root = Path("data/refined_tables")

    # Try dataset-specific location first if dataset provided
    if dataset:
        refined_dir = refined_root / dataset / study_name
        if refined_dir.exists():
            return refined_dir

    # Fall back to non-dataset location
    refined_dir = refined_root / study_name
    if refined_dir.exists():
        return refined_dir

    raise FileNotFoundError(f"Refined tables folder not found: {refined_dir}")


def _resolve_image_dir(study_name: str, dataset: str | None = None) -> Path:
    """Resolve the image directory for a given study."""
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

    raise FileNotFoundError(f"Image data folder not found: {image_dir}")


def _resolve_detection_file(study_name: str) -> Path:
    """Resolve the detection file for a given study."""
    detection_path = Path("data/detection") / f"{study_name}.json"
    if not detection_path.exists():
        raise FileNotFoundError(
            f"Detection file not found: {detection_path}. "
            "Run detection first to identify relevant pages."
        )
    return detection_path


def _get_all_pages_with_labels(detection_path: Path) -> dict[int, str]:
    """Get all page numbers with their detection labels from detection results.

    Returns:
        Dictionary mapping page number to detection label (A, B, C, or D)
    """
    detection_data = json.loads(detection_path.read_text(encoding="utf-8"))
    pages_dict = {}
    for page in detection_data.get("pages", []):
        page_num = page.get("number")
        result = str(page.get("result", "")).strip().upper()
        if isinstance(page_num, int) and result:
            pages_dict[page_num] = result

    return pages_dict


def run(pdf_name: str, force: bool = False, dataset: str | None = None, use_original_images: bool = False) -> Path:
    """Parse all pages to markdown using VLM.

    For pages marked A, B, or C in detection, uses refined table images.
    For all other pages (D or other), uses original images.

    Args:
        pdf_name: PDF file name (e.g., wise2000.pdf)
        force: Force re-processing even if outputs already exist
        dataset: Optional dataset name for organizing outputs
        use_original_images: If True, force use of original images for ALL pages

    Returns:
        Path to output directory containing markdown files
    """
    load_dotenv()
    study_name = Path(pdf_name).stem

    # Get both image directories
    original_image_dir = _resolve_image_dir(study_name, dataset)

    # Try to get refined tables directory (may not exist if refinement was skipped)
    try:
        refined_tables_dir = _resolve_refined_tables_dir(study_name, dataset)
    except FileNotFoundError:
        refined_tables_dir = None
        print(f"Note: No refined tables found for {study_name}, will use original images for all pages")

    # Get all pages with their detection labels
    detection_path = _resolve_detection_file(study_name)
    pages_with_labels = _get_all_pages_with_labels(detection_path)

    if not pages_with_labels:
        print(f"No pages found in detection results for {study_name}.")
        return None

    # Create output directory with optional dataset subfolder
    if dataset:
        output_dir = Path("data/parsed_data") / dataset / study_name
    else:
        output_dir = Path("data/parsed_data") / study_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if already parsed (unless force=True)
    if not force:
        expected_md_files = [
            output_dir / f"page{page_num}.md" for page_num in pages_with_labels.keys()
        ]
        if all(f.exists() for f in expected_md_files):
            print(
                f"Skipping {study_name}: already parsed ({len(expected_md_files)} pages)"
            )
            return output_dir

    # Load VLM parser
    api_key = os.environ.get("TOGETHER_API_KEY")
    if not api_key:
        raise ValueError(
            "TOGETHER_API_KEY not found in environment. "
            "Please set it in your .env file."
        )

    parser_client = QwenParser(api_key=api_key, max_tokens=4000)
    prompt_text = _load_prompt_text()

    parsed_count = 0
    refined_count = 0
    original_count = 0

    for page_num, label in sorted(pages_with_labels.items()):
        # Determine which image to use based on label and availability
        if use_original_images:
            # Force original images for all pages
            image_source_dir = original_image_dir
            image_type = "original"
        elif label in ("A", "B", "C") and refined_tables_dir is not None:
            # Use refined table image for A, B, C pages if available
            refined_path = refined_tables_dir / f"page{page_num}.png"
            if refined_path.exists():
                image_source_dir = refined_tables_dir
                image_type = "refined"
                refined_count += 1
            else:
                # Fall back to original if refined doesn't exist
                image_source_dir = original_image_dir
                image_type = "original (refined not found)"
                original_count += 1
        else:
            # Use original image for D or other labels
            image_source_dir = original_image_dir
            image_type = "original"
            original_count += 1

        page_path = image_source_dir / f"page{page_num}.png"

        if not page_path.exists():
            print(f"Warning: No PNG file found for page {page_num} at {page_path}")
            continue

        # Check if this specific page already exists
        output_path = output_dir / f"page{page_num}.md"
        if not force and output_path.exists():
            print(f"Skipping page {page_num}: already parsed")
            parsed_count += 1
            continue

        # Parse the full page image to markdown using VLM
        with Image.open(page_path) as img:
            markdown_content = parser_client.parse_page(
                img.convert("RGB"),
                prompt=prompt_text,
                logprobs=0,
            )

        output_path.write_text(markdown_content, encoding="utf-8")
        print(f"Saved: {output_path} (label: {label}, source: {image_type})")
        parsed_count += 1

    print(f"\nCompleted parsing {parsed_count} pages to {output_dir}")
    if not use_original_images and refined_tables_dir is not None:
        print(f"  - {refined_count} pages from refined tables (A/B/C)")
        print(f"  - {original_count} pages from original images (D/other)")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description=("Parse refined table PNGs to markdown using VLM.")
    )
    parser.add_argument(
        "pdf_name",
        help="PDF file name (e.g., wise2000.pdf) used to locate the refined tables.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-processing even if output already exists.",
    )
    parser.add_argument(
        "--dataset",
        help="Optional dataset name for organizing outputs (e.g., tuning, validation)",
    )
    args = parser.parse_args()
    run(args.pdf_name, force=args.force, dataset=args.dataset)


if __name__ == "__main__":
    main()
