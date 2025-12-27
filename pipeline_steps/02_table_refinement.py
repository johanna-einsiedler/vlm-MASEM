#!/usr/bin/env python3
"""Run table detection and cell recognition with PaddleOCR on detected images (A, B, or C labels).

Outputs (saved to data/refined_tables/<dataset>/<study>/):
- One image per page with cell detection overlays:
  - pageN.png (full page with cell detection visualization)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


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


def _get_flagged_pages(detection_path: Path) -> list[int]:
    """Get page numbers flagged as A, B, or C from detection results."""
    detection_data = json.loads(detection_path.read_text(encoding="utf-8"))
    flagged_pages = []
    for page in detection_data.get("pages", []):
        result = str(page.get("result", "")).strip().upper()
        if result in ("A", "B", "C"):
            flagged_pages.append(page.get("number"))

    flagged_pages = [p for p in flagged_pages if isinstance(p, int)]
    return flagged_pages


def _progress_bar(iterable: Iterable, total: int, desc: str = "Processing") -> Iterable:
    """Display progress bar for iteration."""
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
    force: bool = False,
    dataset: str | None = None,
) -> list[Path]:
    """Run table detection and cell recognition on flagged pages with full-page overlay visualization.

    Args:
        pdf_name: PDF file name (e.g., wise2000.pdf)
        page_numbers: Optional page numbers to process. If None, uses detection results.
        force: Force re-processing even if outputs already exist
        dataset: Optional dataset name for organizing outputs

    Returns:
        List of paths to refined table images with cell detection overlays
    """
    study_name = Path(pdf_name).stem
    image_dir = _resolve_image_dir(study_name, dataset)

    # If page_numbers not specified, get them from detection results
    if page_numbers is None:
        detection_path = _resolve_detection_file(study_name)
        page_numbers = _get_flagged_pages(detection_path)
        if not page_numbers:
            print(f"No pages flagged for table refinement in {study_name}.")
            return []

    # Create output directory
    if dataset:
        output_dir = Path("data/refined_tables") / dataset / study_name
    else:
        output_dir = Path("data/refined_tables") / study_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if already processed (unless force=True)
    if not force:
        page_list = list(page_numbers)
        expected_files = [output_dir / f"page{page_num}.png" for page_num in page_list]
        if all(f.exists() for f in expected_files):
            print(
                f"Skipping {study_name}: already refined ({len(expected_files)} pages)"
            )
            return expected_files

    # Initialize PaddleOCR table recognition pipeline V2
    print("Initializing PaddleOCR table recognition pipeline V2...")
    from paddleocr import TableRecognitionPipelineV2

    # Use TableRecognitionPipelineV2 with layout detection enabled
    # paddle_pipeline = TableRecognitionPipelineV2(
    #     use_doc_orientation_classify=True,
    #     use_doc_unwarping=True,
    #     use_layout_detection=True,
    #     use_ocr_model=True,
    #     text_det_limit_side_len=1920,
    #     text_det_limit_type="max",
    #     text_det_box_thresh=0.4,  # more recall for small tokens
    #     text_det_unclip_ratio=1.5,
    # )

    paddle_pipeline = TableRecognitionPipelineV2(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_layout_detection=True,
        use_ocr_model=True,
    )

    output_paths: list[Path] = []
    page_list = list(page_numbers)

    for page_number in _progress_bar(
        page_list, total=len(page_list), desc="Processing pages with cell detection"
    ):
        # Check if already exists
        output_png = output_dir / f"page{page_number}.png"
        if not force and output_png.exists():
            print(f"Skipping page {page_number}: already exists")
            output_paths.append(output_png)
            continue

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

        # Run table recognition pipeline to get cell detection visualization
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                print(f"Warning: Failed to read image: {image_path}")
                continue

            # Run pipeline - it returns table recognition results with cell detection
            output = paddle_pipeline.predict(
                str(image_path),
                # use_table_orientation_classify=True,
            )

            # Save the visualization with cell detection overlay
            # PaddleOCR's save_to_img creates visualization files
            if output and len(output) > 0:
                # Get list of existing files before saving
                existing_files = set(output_dir.glob("*"))

                # Save the full visualization (creates multiple files)
                # Use the first result which contains the full page visualization
                output[0].save_to_img(str(output_dir))

                # Get list of new files created
                new_files = set(output_dir.glob("*")) - existing_files

                # Find the main visualization file
                # PaddleOCR saves multiple files, we want the one with cell detection overlay
                viz_file = None
                for f in new_files:
                    # Look for files with "table" in the name
                    if "table" in f.name.lower() and f.suffix in [".jpg", ".png"]:
                        viz_file = f
                        break

                if not viz_file:
                    # Fall back to any new image file
                    for f in new_files:
                        if f.suffix in [".jpg", ".png"]:
                            viz_file = f
                            break

                if viz_file:
                    # Read the visualization and save it with the standard filename
                    full_viz = cv2.imread(str(viz_file))
                    if full_viz is not None:
                        # Save the full page visualization
                        cv2.imwrite(str(output_png), full_viz)
                        output_paths.append(output_png)
                        print(f"Saved page with cell detection: {output_png}")
                    else:
                        print(f"Warning: Could not read visualization file: {viz_file}")

                # Clean up all temporary visualization files created by PaddleOCR
                for f in new_files:
                    if f.exists():
                        f.unlink()
            else:
                print(
                    f"Warning: No output from table recognition for page {page_number}"
                )

        except Exception as e:
            print(f"Error processing page {page_number}: {e}")
            import traceback

            traceback.print_exc()
            continue

    if not output_paths:
        print(f"No pages were successfully refined for {study_name}.")
        return []

    print(f"Completed table refinement for {study_name}: {len(output_paths)} pages")
    return output_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run PaddleOCR table detection and cell recognition with full-page visualization overlay."
        )
    )
    parser.add_argument(
        "pdf_name",
        help="PDF file name (e.g., wise2000.pdf) used to locate images.",
    )
    parser.add_argument(
        "--pages",
        nargs="*",
        type=int,
        help="Optional page numbers to process (e.g., --pages 1 2 5). "
        "If not specified, uses detection results to identify relevant pages.",
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
    run(args.pdf_name, page_numbers=args.pages, force=args.force, dataset=args.dataset)


if __name__ == "__main__":
    main()
