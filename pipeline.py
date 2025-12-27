#!/usr/bin/env python3
"""Pipeline: Convert PDFs to PNG images and run detection."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import 00_pdf_to_images (has number prefix, needs special handling)
spec_00 = importlib.util.spec_from_file_location(
    "pdf_to_images_module",
    ROOT / "pipeline_steps" / "00_pdf_to_images.py"
)
pdf_to_images = importlib.util.module_from_spec(spec_00)
spec_00.loader.exec_module(pdf_to_images)

# Import 01_detection (has number prefix, needs special handling)
spec_01 = importlib.util.spec_from_file_location(
    "detection_module",
    ROOT / "pipeline_steps" / "01_detection.py"
)
detection = importlib.util.module_from_spec(spec_01)
spec_01.loader.exec_module(detection)

# Import 02_table_refinement (has number prefix, needs special handling)
spec_02 = importlib.util.spec_from_file_location(
    "table_refinement_module",
    ROOT / "pipeline_steps" / "02_table_refinement.py"
)
table_refinement = importlib.util.module_from_spec(spec_02)
spec_02.loader.exec_module(table_refinement)

# Import 03_parse_pdf (has number prefix, needs special handling)
spec_03 = importlib.util.spec_from_file_location(
    "parse_pdf_module",
    ROOT / "pipeline_steps" / "03_parse_pdf.py"
)
parse_pdf = importlib.util.module_from_spec(spec_03)
spec_03.loader.exec_module(parse_pdf)

# Import 04_factor_extraction (has number prefix, needs special handling)
spec_04 = importlib.util.spec_from_file_location(
    "factor_extraction_module",
    ROOT / "pipeline_steps" / "04_factor_extraction.py"
)
factor_extraction = importlib.util.module_from_spec(spec_04)
spec_04.loader.exec_module(factor_extraction)

# Import 05_metadata_extraction (has number prefix, needs special handling)
spec_05 = importlib.util.spec_from_file_location(
    "metadata_extraction_module",
    ROOT / "pipeline_steps" / "05_metadata_extraction.py"
)
metadata_extraction = importlib.util.module_from_spec(spec_05)
spec_05.loader.exec_module(metadata_extraction)

# Import 06_evaluate (has number prefix, needs special handling)
spec_06 = importlib.util.spec_from_file_location(
    "evaluate_module",
    ROOT / "pipeline_steps" / "06_evaluate.py"
)
evaluate = importlib.util.module_from_spec(spec_06)
spec_06.loader.exec_module(evaluate)

# Import 07_accuracy_plots (has number prefix, needs special handling)
spec_07 = importlib.util.spec_from_file_location(
    "accuracy_plots_module",
    ROOT / "pipeline_steps" / "07_accuracy_plots.py"
)
accuracy_plots = importlib.util.module_from_spec(spec_07)
spec_07.loader.exec_module(accuracy_plots)


def _process_single_pdf(pdf_path: Path, force: bool = False, stop_at: int | None = None, skip_refinement: bool = False) -> None:
    """Convert a single PDF to PNG images, run detection, and refine tables.

    Args:
        pdf_path: Path to the PDF file
        force: Force re-processing even if outputs already exist
        stop_at: Stop after the specified step (1-7), or None to run all steps
        skip_refinement: Skip step 3 (table refinement) and use original images for parsing
    """
    print(f"\n{'=' * 80}")
    print(f"Processing: {pdf_path.name}")
    print(f"{'=' * 80}\n")

    study_name = pdf_path.stem

    # Determine dataset from path
    dataset = None
    if "intermediate_data" in pdf_path.parts:
        try:
            idx = pdf_path.parts.index("intermediate_data")
            if idx + 1 < len(pdf_path.parts):
                dataset = pdf_path.parts[idx + 1]
        except (ValueError, IndexError):
            pass

    # Track whether we need to force subsequent steps
    force_remaining = force

    # Step 1: Convert PDF to images
    print("Step 1: Converting PDF to PNG images...")
    step1_output = Path("data/image_data") / (dataset if dataset else "") / study_name
    if dataset:
        step1_exists = step1_output.exists() and any(step1_output.glob("*.png"))
    else:
        # Try to find in any location
        step1_exists = any(Path("data/image_data").glob(f"*/{study_name}/*.png"))

    try:
        image_dir = pdf_to_images.run(str(pdf_path), force=force_remaining)
        print(f"✓ Images saved to: {image_dir}\n")
        # If we just ran this step, force all subsequent steps
        if not step1_exists or force_remaining:
            force_remaining = True
    except Exception as e:
        print(f"✗ Error converting PDF to images: {e}")
        return

    if stop_at == 1:
        print(f"{'=' * 80}")
        print(f"✓ Stopped after step 1 (PDF to images)")
        print(f"{'=' * 80}\n")
        return

    # Step 2: Run detection on images
    print("Step 2: Running detection on images...")
    detection_output = Path("data/detection") / f"{study_name}.json"
    step2_exists = detection_output.exists()

    if not force_remaining and step2_exists:
        print(f"Skipping detection: output already exists at {detection_output}\n")
    else:
        try:
            detection_path = detection.run(pdf_path.name)
            print(f"✓ Detection results saved to: {detection_path}\n")
            # If we just ran this step, force all subsequent steps
            if not step2_exists or force_remaining:
                force_remaining = True
        except Exception as e:
            print(f"✗ Error running detection: {e}")
            return

    if stop_at == 2:
        print(f"{'=' * 80}")
        print(f"✓ Stopped after step 2 (Detection)")
        print(f"{'=' * 80}\n")
        return

    # Step 3: Run table refinement on detected pages (A or C)
    if skip_refinement:
        print("Step 3: Skipping table refinement (--skip-refinement flag set)\n")
    else:
        print("Step 3: Running table cell recognition with full-page visualization on detected pages...")
        if dataset:
            step3_output = Path("data/refined_tables") / dataset / study_name
        else:
            step3_output = Path("data/refined_tables") / study_name
        step3_exists = step3_output.exists() and any(step3_output.glob("*.png"))

        try:
            refined_paths = table_refinement.run(
                pdf_path.name,
                force=force_remaining,
                dataset=dataset
            )
            if refined_paths:
                print(f"✓ Created {len(refined_paths)} full-page images with cell detection overlays\n")
                # If we just ran this step, force all subsequent steps
                if not step3_exists or force_remaining:
                    force_remaining = True
            else:
                print(f"No pages to process (no pages flagged as A or C)\n")
        except Exception as e:
            print(f"✗ Error running table refinement: {e}")
            # Don't return - this is not a critical error
            print()

    if stop_at == 3:
        print(f"{'=' * 80}")
        print(f"✓ Stopped after step 3 (Table refinement)")
        print(f"{'=' * 80}\n")
        return

    # Step 4: Parse refined table PNGs to markdown with VLM
    if skip_refinement:
        print("Step 4: Parsing original images to markdown (refinement skipped)...")
    else:
        print("Step 4: Parsing refined tables to markdown...")
    if dataset:
        step4_output = Path("data/parsed_data") / dataset / study_name
    else:
        step4_output = Path("data/parsed_data") / study_name
    step4_exists = step4_output.exists() and any(step4_output.glob("*.md"))

    try:
        output_dir = parse_pdf.run(
            pdf_path.name,
            force=force_remaining,
            dataset=dataset,
            use_original_images=skip_refinement
        )
        if output_dir:
            print(f"✓ Markdown saved to: {output_dir}\n")
            # If we just ran this step, force all subsequent steps
            if not step4_exists or force_remaining:
                force_remaining = True
        else:
            print(f"No pages to parse\n")
    except Exception as e:
        print(f"✗ Error parsing to markdown: {e}")
        # Don't return - this is not a critical error
        print()

    if stop_at == 4:
        print(f"{'=' * 80}")
        print(f"✓ Stopped after step 4 (Parse to markdown)")
        print(f"{'=' * 80}\n")
        return

    # Step 5: Extract structured data from markdown using GPT OSS
    print("Step 5: Extracting factors & correlations with GPT OSS...")
    # Check if both extraction outputs exist
    if dataset:
        step5_factor_output = Path("data/extraction_factors") / dataset / f"{study_name}.json"
        step5_corr_output = Path("data/extraction_correlations") / dataset / f"{study_name}.json"
    else:
        step5_factor_output = Path("data/extraction_factors") / f"{study_name}.json"
        step5_corr_output = Path("data/extraction_correlations") / f"{study_name}.json"
    step5_exists = step5_factor_output.exists() and step5_corr_output.exists()

    try:
        extraction_paths = factor_extraction.run(
            pdf_path.name,
            force=force_remaining,
            dataset=dataset
        )
        if extraction_paths:
            print(f"✓ Extracted data saved ({len(extraction_paths)} file(s))\n")
            # If we just ran this step, force all subsequent steps
            if not step5_exists or force_remaining:
                force_remaining = True
        else:
            print(f"No data extracted\n")
    except Exception as e:
        print(f"✗ Error extracting data: {e}")
        # Don't return - this is not a critical error
        print()

    if stop_at == 5:
        print(f"{'=' * 80}")
        print(f"✓ Stopped after step 5 (Extract structured data)")
        print(f"{'=' * 80}\n")
        return

    # Step 6: Extract metadata using factor loadings, correlations, and full paper
    print("Step 6: Extracting metadata with GPT OSS...")
    if dataset:
        step6_output = Path("data/extraction_metadata") / dataset / f"{study_name}.json"
    else:
        step6_output = Path("data/extraction_metadata") / f"{study_name}.json"
    step6_exists = step6_output.exists()

    try:
        metadata_path = metadata_extraction.run(
            pdf_path.name,
            force=force_remaining,
            dataset=dataset
        )
        if metadata_path:
            print(f"✓ Metadata saved to: {metadata_path}\n")
            # If we just ran this step, force all subsequent steps
            if not step6_exists or force_remaining:
                force_remaining = True
        else:
            print(f"No metadata extracted\n")
    except Exception as e:
        print(f"✗ Error extracting metadata: {e}")
        # Don't return - this is not a critical error
        print()

    if stop_at == 6:
        print(f"{'=' * 80}")
        print(f"✓ Stopped after step 6 (Extract metadata)")
        print(f"{'=' * 80}\n")
        return

    # Step 7: Evaluate extraction against ground truth
    print("Step 7: Evaluating extraction against ground truth...")
    try:
        eval_paths = evaluate.run(
            pdf_path.name,
            dataset=dataset,
            force=force_remaining
        )
        if eval_paths:
            print(f"✓ Evaluation saved ({len(eval_paths)} file(s))\n")
        else:
            print(f"No evaluation performed\n")
    except Exception as e:
        print(f"✗ Error evaluating: {e}")
        # Don't return - this is not a critical error
        print()

    print(f"{'=' * 80}")
    print(f"✓ Completed processing {pdf_path.name}")
    print(f"{'=' * 80}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Pipeline: Convert PDFs to images, detect tables, refine, parse, extract, and evaluate.\n"
            "By default, processes all PDFs in data/intermediate_data/tuning/\n"
            "\n"
            "Steps (per paper):\n"
            "  1. Convert PDF to PNGs -> data/image_data/<dataset>/<paper-name>/pageN.png\n"
            "  2. Run detection on PNGs -> data/detection/<paper-name>.json\n"
            "  3. Add cell detection overlays with PaddleOCR -> data/refined_tables/<dataset>/<paper-name>/pageN.png\n"
            "     (can be skipped with --skip-refinement to use original images)\n"
            "  4. Parse tables to markdown with VLM -> data/parsed_data/<dataset>/<paper-name>/pageN.md\n"
            "  5. Extract factors & correlations with GPT OSS -> data/extraction_factors/ and data/extraction_correlations/\n"
            "  6. Extract metadata with GPT OSS -> data/extraction_metadata/<dataset>/<paper-name>.json\n"
            "  7. Evaluate against ground truth -> data/evaluation/<dataset>/<paper-name>[a|b|...].json\n"
            "\n"
            "Final step (per dataset):\n"
            "  8. Generate accuracy plots -> figures/accuracy_plot_factors_<dataset>.png\n"
            "                             -> figures/accuracy_plot_correlations_<dataset>.png\n"
            "                             -> figures/accuracy_plot_metadata_<dataset>.png\n"
            "                             -> figures/accuracy_plot_metadata_fields_<dataset>.png"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--pdf",
        help="Process a single PDF file (e.g., --pdf data/intermediate_data/tuning/wise2000.pdf)",
    )
    group.add_argument(
        "--paper",
        help="Process a single paper by name (e.g., --paper wise2000)",
    )
    group.add_argument(
        "--dataset",
        help="Process all PDFs in a dataset (e.g., --dataset eval, --dataset validation)",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-processing even if outputs already exist",
    )

    parser.add_argument(
        "--stop-at",
        type=int,
        choices=[1, 2, 3, 4, 5, 6, 7],
        help="Stop pipeline after the specified step (1-7)",
    )

    parser.add_argument(
        "--skip-refinement",
        action="store_true",
        help="Skip step 3 (table refinement) and use original images for parsing",
    )

    args = parser.parse_args()

    # Determine which PDFs to process
    pdf_paths = []

    if args.pdf:
        # Single PDF by path
        pdf_path = Path(args.pdf)
        if not pdf_path.exists():
            print(f"Error: PDF not found: {pdf_path}")
            sys.exit(1)
        pdf_paths = [pdf_path]

    elif args.paper:
        # Single paper by name - search in all datasets
        paper_name = args.paper
        if not paper_name.endswith(".pdf"):
            paper_name = f"{paper_name}.pdf"

        # Look for the paper in intermediate_data subdirectories
        intermediate_data = Path("data/intermediate_data")
        if not intermediate_data.exists():
            print(f"Error: Directory not found: {intermediate_data}")
            sys.exit(1)

        # Search in all dataset subdirectories
        found = False
        for dataset_dir in intermediate_data.iterdir():
            if dataset_dir.is_dir():
                candidate = dataset_dir / paper_name
                if candidate.exists():
                    pdf_paths = [candidate]
                    found = True
                    print(f"Found {paper_name} in {dataset_dir.name} dataset")
                    break

        if not found:
            print(f"Error: Paper '{args.paper}' not found in any dataset")
            print(f"Searched in: {intermediate_data}")
            sys.exit(1)

    else:
        # Process dataset (default: tuning)
        dataset = args.dataset or "tuning"

        # Normalize dataset name
        dataset_map = {
            "tune": "tuning",
            "tuning": "tuning",
            "val": "validation",
            "validation": "validation",
            "eval": "eval",
        }
        dataset = dataset_map.get(dataset, dataset)

        dataset_dir = Path("data/intermediate_data") / dataset
        if not dataset_dir.exists():
            print(f"Error: Dataset directory not found: {dataset_dir}")
            print(f"Available options: {', '.join(dataset_map.keys())}")
            sys.exit(1)

        pdf_paths = sorted(dataset_dir.glob("*.pdf"))
        if not pdf_paths:
            print(f"Error: No PDFs found in {dataset_dir}")
            sys.exit(1)

        print(f"Found {len(pdf_paths)} PDFs in {dataset} dataset\n")

    # Determine dataset for accuracy plot
    dataset_for_plot = None
    if args.dataset:
        dataset_map = {
            "tune": "tuning",
            "tuning": "tuning",
            "val": "validation",
            "validation": "validation",
            "eval": "eval",
        }
        dataset_for_plot = dataset_map.get(args.dataset, args.dataset)
    elif pdf_paths:
        # Try to infer dataset from first PDF path
        first_pdf = pdf_paths[0]
        if "intermediate_data" in first_pdf.parts:
            try:
                idx = first_pdf.parts.index("intermediate_data")
                if idx + 1 < len(first_pdf.parts):
                    dataset_for_plot = first_pdf.parts[idx + 1]
            except (ValueError, IndexError):
                pass

    # Process each PDF
    for pdf_path in pdf_paths:
        _process_single_pdf(pdf_path, force=args.force, stop_at=args.stop_at, skip_refinement=args.skip_refinement)

    # Generate accuracy plots for the dataset (Step 8)
    # Skip if user requested to stop before step 7
    if args.stop_at is not None and args.stop_at < 7:
        print("\n" + "=" * 80)
        print(f"Skipping accuracy plot generation (stopped at step {args.stop_at})")
        print("=" * 80)
        return
    if dataset_for_plot and len(pdf_paths) > 0:
        print("\n" + "=" * 80)
        print("Generating accuracy plots for dataset...")
        print("=" * 80 + "\n")
        try:
            plot_paths = accuracy_plots.run(
                dataset=dataset_for_plot,
                force=args.force
            )
            if plot_paths:
                print(f"✓ Accuracy plots saved ({len(plot_paths)} plots)\n")
            else:
                print(f"No accuracy plots generated\n")
        except Exception as e:
            print(f"✗ Error generating accuracy plots: {e}\n")

    # Summary
    print("\n" + "=" * 80)
    print(f"Pipeline completed for {len(pdf_paths)} PDF(s)")
    print("=" * 80)


if __name__ == "__main__":
    main()
