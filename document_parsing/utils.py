"""Shared helpers reused by multiple document parsers."""

from __future__ import annotations

from typing import List, Optional

import fitz  # PyMuPDF
from PIL import Image


def pdf_to_images(pdf_path: str, dpi: Optional[int] = None) -> List[Image.Image]:
    """Convert each page of a PDF into a PIL image."""
    matrix = None
    if dpi is not None:
        zoom = dpi / 72
        matrix = fitz.Matrix(zoom, zoom)
    images: List[Image.Image] = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            pix = page.get_pixmap(matrix=matrix) if matrix else page.get_pixmap()
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            images.append(img)
    return images
