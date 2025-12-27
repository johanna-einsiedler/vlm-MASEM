#!/usr/bin/env python3
"""Convert PDF pages to PNG images."""

from __future__ import annotations

import argparse
from pathlib import Path

import fitz  # PyMuPDF


def run(pdf_path: str, output_dir: str | None = None, dpi: int = 200, force: bool = False) -> Path:
    """Convert PDF to PNG images.

    Args:
        pdf_path: Path to the PDF file
        output_dir: Optional output directory (default: data/image_data/<dataset>/<study_name>)
        dpi: DPI for image rendering (default: 200)
        force: Force re-conversion even if images already exist

    Returns:
        Path to the output directory containing the images
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    study_name = pdf_path.stem

    # Determine output directory
    if output_dir:
        output_path = Path(output_dir)
    else:
        # Try to determine dataset from PDF location
        dataset = None
        if "intermediate_data" in pdf_path.parts:
            # Extract dataset from path like data/intermediate_data/tuning/study.pdf
            try:
                idx = pdf_path.parts.index("intermediate_data")
                if idx + 1 < len(pdf_path.parts):
                    dataset = pdf_path.parts[idx + 1]
            except (ValueError, IndexError):
                pass

        if dataset:
            output_path = Path("data/image_data") / dataset / study_name
        else:
            output_path = Path("data/image_data") / study_name

    output_path.mkdir(parents=True, exist_ok=True)

    # Convert PDF to images
    print(f"Converting {pdf_path.name} to images...")
    doc = fitz.open(pdf_path)

    zoom = dpi / 72  # Convert DPI to zoom factor (72 is PDF default DPI)
    mat = fitz.Matrix(zoom, zoom)

    converted_count = 0
    skipped_count = 0

    for page_num in range(len(doc)):
        output_file = output_path / f"page{page_num + 1}.png"

        # Skip if file already exists and force is False
        if not force and output_file.exists():
            print(f"Skipping page {page_num + 1}: already exists")
            skipped_count += 1
            continue

        page = doc[page_num]
        pix = page.get_pixmap(matrix=mat)
        pix.save(output_file)
        print(f"Saved: {output_file}")
        converted_count += 1

    doc.close()

    if skipped_count > 0:
        print(f"\nCompleted: {converted_count} pages converted, {skipped_count} pages skipped (already exist)")
    else:
        print(f"\nCompleted converting {converted_count} pages to {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert PDF pages to PNG images."
    )
    parser.add_argument(
        "pdf_path",
        help="Path to the PDF file (e.g., data/intermediate_data/tuning/wise2000.pdf)",
    )
    parser.add_argument(
        "-o", "--output",
        help="Output directory for images (default: auto-determined from PDF path)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="DPI for image rendering (default: 200)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-conversion even if images already exist",
    )
    args = parser.parse_args()

    run(args.pdf_path, output_dir=args.output, dpi=args.dpi, force=args.force)


if __name__ == "__main__":
    main()
