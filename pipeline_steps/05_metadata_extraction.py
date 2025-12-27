#!/usr/bin/env python3
"""Extract metadata using combined information from factor loadings, correlations, and parsed paper.

This step combines:
1. Extracted factor loadings (from extraction_factors/)
2. Extracted correlations (from extraction_correlations/)
3. All parsed markdown pages (from parsed_data/)

Then uses an LLM to extract metadata and saves to extraction_metadata/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from document_parsing.llm_deepseek import DeepSeekParser


def _load_prompt_text(prompt_filename: str) -> str:
    """Load a prompt text file."""
    prompt_path = Path("prompts") / prompt_filename
    alt_prompt_path = Path(
        "/Users/einsie0004/Documents/research/33_llm_replicability/"
        f"vlm_pipeline/prompts/{prompt_filename}"
    )

    if alt_prompt_path.exists():
        return alt_prompt_path.read_text(encoding="utf-8").strip()
    elif prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8").strip()
    else:
        raise FileNotFoundError(
            f"Prompt file not found: {prompt_filename}. "
            f"Looked in {prompt_path} and {alt_prompt_path}"
        )


def _resolve_parsed_dir(study_name: str, dataset: str | None = None) -> Path:
    """Resolve the parsed data directory for a given study."""
    parsed_root = Path("data/parsed_data")

    # Try dataset-specific location first if dataset provided
    if dataset:
        parsed_dir = parsed_root / dataset / study_name
        if parsed_dir.exists():
            return parsed_dir

    # Fall back to non-dataset location
    parsed_dir = parsed_root / study_name
    if parsed_dir.exists():
        return parsed_dir

    raise FileNotFoundError(f"Parsed data folder not found: {parsed_dir}")


def _resolve_extraction_dir(extraction_type: str, dataset: str | None = None) -> Path:
    """Resolve the extraction directory for a given extraction type."""
    if dataset:
        return Path("data") / extraction_type / dataset
    else:
        return Path("data") / extraction_type


def _load_extraction_json(extraction_dir: Path, study_name: str) -> dict | None:
    """Load extraction JSON if it exists, return None otherwise."""
    json_path = extraction_dir / f"{study_name}.json"
    if not json_path.exists():
        return None

    try:
        return json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"Warning: Could not load {json_path}: {e}")
        return None


def _concat_markdown_pages(parsed_dir: Path) -> str:
    """Concatenate all markdown pages in order."""
    md_files = sorted(parsed_dir.glob("page*.md"))

    if not md_files:
        return ""

    content_parts = []
    for md_file in md_files:
        page_num = md_file.stem.replace("page", "")
        content = md_file.read_text(encoding="utf-8").strip()
        content_parts.append(f"=== PAGE {page_num} ===\n{content}")

    return "\n\n".join(content_parts)


def run(pdf_name: str, force: bool = False, dataset: str | None = None) -> Path | None:
    """Extract metadata using factor loadings, correlations, and parsed paper.

    Args:
        pdf_name: PDF file name (e.g., wise2000.pdf)
        force: Force re-processing even if output already exists
        dataset: Optional dataset name for organizing outputs

    Returns:
        Path to metadata extraction JSON file, or None if extraction failed
    """
    load_dotenv()
    study_name = Path(pdf_name).stem

    # Create output directory
    if dataset:
        output_dir = Path("data/extraction_metadata") / dataset
    else:
        output_dir = Path("data/extraction_metadata")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{study_name}.json"

    # Check if already exists (unless force=True)
    if not force and output_path.exists():
        print(f"Skipping {study_name}: metadata already extracted")
        return output_path

    # Load factor loadings extraction
    factor_dir = _resolve_extraction_dir("extraction_factors", dataset)
    factor_data = _load_extraction_json(factor_dir, study_name)

    # Load correlations extraction
    correlation_dir = _resolve_extraction_dir("extraction_correlations", dataset)
    correlation_data = _load_extraction_json(correlation_dir, study_name)

    # Load parsed markdown
    try:
        parsed_dir = _resolve_parsed_dir(study_name, dataset)
        parsed_content = _concat_markdown_pages(parsed_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None

    if not parsed_content:
        print(f"Warning: No parsed content found for {study_name}")
        return None

    # Construct combined input for LLM
    combined_parts = []

    # Add factor loadings if available
    if factor_data:
        combined_parts.append("=== EXTRACTED FACTOR LOADINGS ===")
        combined_parts.append(json.dumps(factor_data, indent=2))
    else:
        combined_parts.append("=== EXTRACTED FACTOR LOADINGS ===")
        combined_parts.append("(No factor loadings extracted)")

    # Add correlations if available
    if correlation_data:
        combined_parts.append("\n=== EXTRACTED CORRELATIONS ===")
        combined_parts.append(json.dumps(correlation_data, indent=2))
    else:
        combined_parts.append("\n=== EXTRACTED CORRELATIONS ===")
        combined_parts.append("(No correlations extracted)")

    # Add full parsed paper
    combined_parts.append("\n=== FULL PAPER CONTENT ===")
    combined_parts.append(parsed_content)

    combined_input = "\n".join(combined_parts)

    # Load prompt
    try:
        prompt_text = _load_prompt_text("04_extraction_prompt_metadata.txt")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None

    # Initialize LLM parser
    api_key = os.environ.get("TOGETHER_API_KEY")
    if not api_key:
        raise ValueError(
            "TOGETHER_API_KEY not found in environment. "
            "Please set it in your .env file."
        )

    parser_client = DeepSeekParser(api_key=api_key, max_tokens=4000)

    # Extract metadata
    print(f"Extracting metadata for {study_name}...")
    try:
        extraction_result = parser_client.parse_text(
            text=combined_input,
            prompt=prompt_text,
        )

        # Save result
        output_path.write_text(extraction_result, encoding="utf-8")
        print(f"✓ Saved metadata extraction to: {output_path}")
        return output_path

    except Exception as e:
        print(f"Error during metadata extraction for {study_name}: {e}")
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract metadata using factor loadings, correlations, and parsed paper content."
        )
    )
    parser.add_argument(
        "pdf_name",
        help="PDF file name (e.g., wise2000.pdf) used to locate extractions.",
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
