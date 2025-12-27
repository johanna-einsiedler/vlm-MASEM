#!/usr/bin/env python3
"""Run extraction with GPT OSS on parsed markdown files.

Performs two separate extraction tasks:
1. Factor loadings: Pages A or C -> extraction_factors/
2. Correlations: Pages B or C -> extraction_correlations/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable

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

    # Try alternate path first, then local path
    if alt_prompt_path.exists():
        return alt_prompt_path.read_text(encoding="utf-8").strip()
    elif prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8").strip()
    else:
        raise FileNotFoundError(f"Prompt not found at {prompt_path} or {alt_prompt_path}")


def _resolve_parsed_dir(study_name: str, dataset: str | None = None) -> Path:
    """Resolve the parsed markdown directory for a given study."""
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

    raise FileNotFoundError(
        f"Parsed data folder not found: {parsed_dir}"
    )


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


def _filter_pages_by_labels(pages_dict: dict[int, str], labels: tuple[str, ...]) -> list[int]:
    """Filter pages by their labels.

    Args:
        pages_dict: Dictionary mapping page number to label
        labels: Tuple of labels to filter by (e.g., ("A", "C"))

    Returns:
        List of page numbers matching the labels
    """
    return sorted([page_num for page_num, label in pages_dict.items() if label in labels])


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


def _run_single_extraction(
    study_name: str,
    parsed_dir: Path,
    page_numbers: list[int],
    prompt_filename: str,
    output_dir: Path,
    parser_client: DeepSeekParser,
    force: bool,
) -> Path | None:
    """Run a single extraction task on specified pages.

    Args:
        study_name: Study name
        parsed_dir: Directory containing parsed markdown files
        page_numbers: List of page numbers to extract from
        prompt_filename: Name of the prompt file to use
        output_dir: Output directory for extraction results
        parser_client: DeepSeek parser client
        force: Force re-extraction even if output exists

    Returns:
        Path to extraction output file, or None if skipped/failed
    """
    if not page_numbers:
        return None

    output_path = output_dir / f"{study_name}.json"

    # Check if already exists
    if not force and output_path.exists():
        print(f"  Skipping: already exists at {output_path}")
        return output_path

    # Load prompt
    prompt_text = _load_prompt_text(prompt_filename)

    # Collect markdown content from specified pages
    combined_markdown = []
    page_metadata = []

    for page_number in page_numbers:
        markdown_path = parsed_dir / f"page{page_number}.md"
        if not markdown_path.exists():
            print(f"  Warning: Markdown file not found: {markdown_path}")
            continue

        markdown_content = markdown_path.read_text(encoding="utf-8")
        combined_markdown.append(markdown_content)
        page_metadata.append(page_number)

    if not combined_markdown:
        print(f"  No markdown files found for pages {page_numbers}")
        return None

    # Combine all markdown
    full_markdown = "\n\n".join(combined_markdown)

    # Run extraction
    print(f"  Extracting from {len(page_metadata)} page(s)...")
    result = parser_client.parse_text(
        text=full_markdown,
        prompt=prompt_text,
    )

    # Create output
    output = {
        "study": study_name,
        "pages": [{"number": page_metadata[0] if page_metadata else None, "result": result}]
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"  ✓ Saved to {output_path}")

    return output_path


def run(
    pdf_name: str,
    force: bool = False,
    dataset: str | None = None,
) -> list[Path]:
    """Run two extraction tasks on parsed markdown files.

    1. Factor loadings: Pages A or C -> extraction_factors/
    2. Correlations: Pages B or C -> extraction_correlations/

    Args:
        pdf_name: PDF file name (e.g., wise2000.pdf)
        force: Force re-extraction even if outputs already exist
        dataset: Optional dataset name for organizing outputs

    Returns:
        List of paths to extraction output JSON files
    """
    load_dotenv()
    study_name = Path(pdf_name).stem
    parsed_dir = _resolve_parsed_dir(study_name, dataset)

    # Get all pages with their labels
    detection_path = _resolve_detection_file(study_name)
    pages_with_labels = _get_all_pages_with_labels(detection_path)

    if not pages_with_labels:
        print(f"No pages found in detection results for {study_name}.")
        return []

    # Filter pages for each extraction task
    factor_pages = _filter_pages_by_labels(pages_with_labels, ("A", "C"))
    correlation_pages = _filter_pages_by_labels(pages_with_labels, ("B", "C"))

    print(f"\n{'=' * 60}")
    print(f"Running extractions for {study_name}")
    print(f"{'=' * 60}")
    print(f"Factor loadings pages (A or C): {factor_pages}")
    print(f"Correlation pages (B or C): {correlation_pages}")
    print(f"{'=' * 60}\n")

    # Initialize LLM client
    api_key = os.environ.get("TOGETHER_API_KEY")
    if not api_key:
        raise ValueError(
            "TOGETHER_API_KEY not found in environment. "
            "Please set it in your .env file."
        )

    parser_client = DeepSeekParser(api_key=api_key, max_tokens=10000)

    # Prepare output directories
    output_paths = []

    # Task 1: Factor loadings (A or C pages)
    print("Task 1: Factor loadings extraction (pages A or C)...")
    if dataset:
        factor_output_dir = Path("data/extraction_factors") / dataset
    else:
        factor_output_dir = Path("data/extraction_factors")
    factor_path = _run_single_extraction(
        study_name=study_name,
        parsed_dir=parsed_dir,
        page_numbers=factor_pages,
        prompt_filename="02_extraction_prompt_factor_loadings.txt",
        output_dir=factor_output_dir,
        parser_client=parser_client,
        force=force,
    )
    if factor_path:
        output_paths.append(factor_path)

    # Task 2: Correlations (B or C pages)
    print("\nTask 2: Correlations extraction (pages B or C)...")
    if dataset:
        correlation_output_dir = Path("data/extraction_correlations") / dataset
    else:
        correlation_output_dir = Path("data/extraction_correlations")

    # If no correlation pages found, create a default file with 0 values
    if not correlation_pages:
        correlation_output_dir.mkdir(parents=True, exist_ok=True)
        correlation_path = correlation_output_dir / f"{study_name}.json"

        # Check if file already exists
        if not force and correlation_path.exists():
            print(f"  Skipping: already exists at {correlation_path}")
            output_paths.append(correlation_path)
        else:
            # Create default correlations structure with all 0 values
            default_correlations = {
                "study": study_name,
                "pages": [
                    {
                        "number": None,
                        "result": json.dumps({
                            "samples": [
                                {
                                    "sample_id": "sample1",
                                    "factor_correlations": {
                                        "R1.2": 0,
                                        "R1.3": 0,
                                        "R1.4": 0,
                                        "R1.5": 0,
                                        "R2.3": 0,
                                        "R2.4": 0,
                                        "R2.5": 0,
                                        "R3.4": 0,
                                        "R3.5": 0,
                                        "R4.5": 0
                                    },
                                    "notes": "No correlation pages detected (no B or C pages). All correlations set to 0 (orthogonal rotation assumed)."
                                }
                            ]
                        })
                    }
                ]
            }
            correlation_path.write_text(json.dumps(default_correlations, indent=2), encoding="utf-8")
            print(f"  ✓ Created default (all 0) correlations at {correlation_path}")
            output_paths.append(correlation_path)
    else:
        correlation_path = _run_single_extraction(
            study_name=study_name,
            parsed_dir=parsed_dir,
            page_numbers=correlation_pages,
            prompt_filename="03_extraction_prompt_correlations.txt",
            output_dir=correlation_output_dir,
            parser_client=parser_client,
            force=force,
        )
        if correlation_path:
            output_paths.append(correlation_path)

    print(f"\n{'=' * 60}")
    print(f"Completed {len(output_paths)} extraction task(s) for {study_name}")
    print(f"{'=' * 60}\n")

    return output_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run two extraction tasks on parsed markdown pages:\n"
            "1. Factor loadings: Pages A or C -> extraction_factors/\n"
            "2. Correlations: Pages B or C -> extraction_correlations/"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "pdf_name",
        help="PDF file name (e.g., wise2000.pdf) used to locate the parsed data folder.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-extraction even if output already exists.",
    )
    parser.add_argument(
        "--dataset",
        help="Optional dataset name for organizing outputs (e.g., tuning, validation)",
    )
    args = parser.parse_args()
    run(args.pdf_name, force=args.force, dataset=args.dataset)


if __name__ == "__main__":
    main()
