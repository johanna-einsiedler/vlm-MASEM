#!/usr/bin/env python3
"""Convert PDFs to per-page PNG images for the Streamlit frontend.

This script is purely for display purposes — it is NOT part of the extraction
pipeline. The 1-step extraction (X1-5_masem_1step.py) renders images internally
from the PDF on the fly; this script pre-renders and saves them so the Streamlit
app can show the relevant context pages.

Output structure:
  data/image_data/<dataset>/<study>/page<N>.png   (1-indexed)

Usage
-----
  # Convert all PDFs in a dataset
  python pipeline_steps/pdf_to_images.py --dataset tuning

  # Convert a single PDF
  python pipeline_steps/pdf_to_images.py --pdf wise2000.pdf --dataset tuning

  # Re-generate existing images at a different DPI
  python pipeline_steps/pdf_to_images.py --dataset tuning --dpi 200 --force
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import fitz  # PyMuPDF

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Core conversion
# ---------------------------------------------------------------------------

def _convert_pdf(pdf_path: Path, output_dir: Path, dpi: int, force: bool) -> Path:
    """Convert a single PDF to per-page PNGs.

    Args:
        pdf_path:   Path to the PDF file.
        output_dir: Directory where page<N>.png files will be written.
        dpi:        Rendering resolution.
        force:      Re-render pages that already exist as PNG files.

    Returns:
        output_dir Path.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(str(pdf_path))
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    n_pages = len(doc)

    saved = skipped = 0
    for i, page in enumerate(doc, start=1):
        out_file = output_dir / f"page{i}.png"
        if out_file.exists() and not force:
            skipped += 1
            continue
        pix = page.get_pixmap(matrix=mat, alpha=False)
        pix.save(str(out_file))
        saved += 1

    doc.close()
    print(f"  {pdf_path.name}: {saved} saved, {skipped} skipped  → {output_dir}")
    return output_dir


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run(
    pdf_name: str,
    dataset: str | None = None,
    dpi: int = 150,
    force: bool = False,
) -> Path:
    """Convert a single PDF to per-page PNGs.

    Args:
        pdf_name: PDF file name (e.g., wise2000.pdf) or full path.
        dataset:  Dataset label (e.g., "tuning"). Used to locate the PDF and
                  name the output directory.
        dpi:      Rendering resolution (default: 150).
        force:    Re-render pages that already exist.

    Returns:
        Path to the output directory containing the PNG files.
    """
    pdf_path = Path(pdf_name)
    if not pdf_path.exists():
        candidates = []
        if dataset:
            candidates.append(ROOT / "data" / "intermediate_data" / dataset / pdf_path.name)
        candidates.append(ROOT / "data" / "intermediate_data" / pdf_path.name)
        for c in candidates:
            if c.exists():
                pdf_path = c
                break
        else:
            raise FileNotFoundError(f"PDF not found: {pdf_name}")

    study_name = pdf_path.stem
    if dataset:
        output_dir = ROOT / "data" / "image_data" / dataset / study_name
    else:
        output_dir = ROOT / "data" / "image_data" / study_name

    return _convert_pdf(pdf_path, output_dir, dpi=dpi, force=force)


def run_dataset(dataset: str, dpi: int = 150, force: bool = False) -> list[Path]:
    """Convert all PDFs in a dataset to per-page PNGs.

    Args:
        dataset: Dataset label (e.g., "tuning").
        dpi:     Rendering resolution (default: 150).
        force:   Re-render pages that already exist.

    Returns:
        List of output directory Paths, one per study.
    """
    pdf_dir = ROOT / "data" / "intermediate_data" / dataset
    if not pdf_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {pdf_dir}")

    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in {pdf_dir}")

    print(f"Converting {len(pdfs)} PDF(s) in '{dataset}' at {dpi} DPI...")
    output_dirs = []
    for pdf_path in pdfs:
        output_dir = ROOT / "data" / "image_data" / dataset / pdf_path.stem
        output_dirs.append(_convert_pdf(pdf_path, output_dir, dpi=dpi, force=force))

    print(f"Done. Images saved under data/image_data/{dataset}/")
    return output_dirs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert PDFs to per-page PNG images for the Streamlit frontend.\n"
            "Output: data/image_data/<dataset>/<study>/page<N>.png (1-indexed)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--pdf",
        default=None,
        help="PDF file name or full path (e.g., wise2000.pdf). Process a single PDF.",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Dataset label (e.g., tuning). If --pdf is omitted, converts all PDFs in the dataset.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Rendering resolution in DPI (default: 150).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-render pages even if PNG files already exist.",
    )
    args = parser.parse_args()

    if args.pdf:
        run(pdf_name=args.pdf, dataset=args.dataset, dpi=args.dpi, force=args.force)
    elif args.dataset:
        run_dataset(dataset=args.dataset, dpi=args.dpi, force=args.force)
    else:
        parser.error("Provide at least --pdf or --dataset.")


if __name__ == "__main__":
    main()
