#!/usr/bin/env python3
"""Runner: 1-step MASEM extraction + evaluation + accuracy plots.

For every PDF in a dataset this script:
  1. Runs X1-5_masem_1step  -> extraction_*/  <dataset>_1step/
  2. Runs 06_evaluate        -> data/evaluation/<dataset>_1step/
  3. Runs 07_accuracy_plots  -> figures/accuracy_plot_*_<dataset>_1step.png

Usage
-----
    # whole dataset
    python pipeline_steps/run_1step_eval.py --dataset tuning

    # single PDF (still uses _1step output dirs)
    python pipeline_steps/run_1step_eval.py --dataset tuning --pdf wise2000.pdf

    # force re-run everything
    python pipeline_steps/run_1step_eval.py --dataset tuning --force
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Dynamic import helper (handles numeric-prefix filenames)
# ---------------------------------------------------------------------------

def _load_step(filename: str) -> ModuleType:
    path = ROOT / "pipeline_steps" / filename
    spec = importlib.util.spec_from_file_location(Path(filename).stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run(
    dataset: str,
    pdf_name: str | None = None,
    provider: str = "openai",
    model: str | None = None,
    dpi: int = 200,
    max_output_tokens: int = 16000,
    chunk_size: int = 15,
    force: bool = False,
    skip_extraction: bool = False,
    skip_eval: bool = False,
    skip_plots: bool = False,
) -> None:
    tag = f"{dataset}_{provider}"

    # Collect PDFs to process
    if pdf_name:
        pdf_path = Path(pdf_name)
        if not pdf_path.exists():
            candidate = ROOT / "data" / "intermediate_data" / dataset / Path(pdf_name).name
            if candidate.exists():
                pdf_path = candidate
            else:
                raise FileNotFoundError(f"PDF not found: {pdf_name}")
        pdfs = [pdf_path]
    else:
        pdf_dir = ROOT / "data" / "intermediate_data" / dataset
        if not pdf_dir.exists():
            raise FileNotFoundError(f"Dataset folder not found: {pdf_dir}")
        pdfs = sorted(pdf_dir.glob("*.pdf"))
        if not pdfs:
            raise FileNotFoundError(f"No PDFs found in {pdf_dir}")

    print(f"\n{'=' * 60}")
    print(f"1-step pipeline: {len(pdfs)} PDF(s), dataset tag = '{tag}'")
    print(f"Model: {model}")
    print(f"{'=' * 60}\n")

    # ------------------------------------------------------------------ #
    # Step X1-5: extraction                                               #
    # ------------------------------------------------------------------ #
    if not skip_extraction:
        step_x15 = _load_step("extract.py")
        for pdf_path in pdfs:
            print(f"\n--- Extraction: {pdf_path.name} ---")
            step_x15.run(
                pdf_name=str(pdf_path),
                force=force,
                dataset=dataset,
                provider=provider,
                model=model,
                dpi=dpi,
                max_output_tokens=max_output_tokens,
                chunk_size=chunk_size,
            )
    else:
        print("Skipping extraction step (--skip-extraction).")

    # ------------------------------------------------------------------ #
    # Step 6: evaluate                                                    #
    # We patch _resolve_dataset_name so it accepts the _1step tag.       #
    # ------------------------------------------------------------------ #
    if not skip_eval:
        step_eval = _load_step("evaluate.py")

        for pdf_path in pdfs:
            print(f"\n--- Evaluate: {pdf_path.name} ---")
            step_eval.run(
                pdf_name=pdf_path.name,
                dataset=tag,
                force=force,
            )
    else:
        print("Skipping evaluation step (--skip-eval).")

    # ------------------------------------------------------------------ #
    # Step 7: accuracy plots                                              #
    # 07_accuracy_plots.run() uses dataset name as a raw path component, #
    # so no name resolution is needed.                                   #
    # ------------------------------------------------------------------ #
    if not skip_plots:
        step_plots = _load_step("accuracy_plots.py")
        print(f"\n--- Accuracy plots for '{tag}' ---")
        step_plots.run(dataset=tag, force=force)
    else:
        print("Skipping accuracy plots (--skip-plots).")

    print(f"\n{'=' * 60}")
    print(f"All done. Outputs are in:")
    print(f"  data/extraction_factors/{tag}/")
    print(f"  data/extraction_correlations/{tag}/")
    print(f"  data/extraction_metadata/{tag}/")
    print(f"  data/evaluation/{tag}/")
    print(f"  figures/accuracy_plot_*_{tag}.png")
    print(f"{'=' * 60}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "1-step MASEM pipeline: extraction (X1-5) + evaluation (step 6) + plots (step 7).\n"
            "Outputs land in <dataset>_1step/ subdirectories."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset name (e.g., tuning, validation). Output tag becomes <dataset>_1step.",
    )
    parser.add_argument(
        "--pdf",
        dest="pdf_name",
        default=None,
        help="Process a single PDF instead of the whole dataset.",
    )
    parser.add_argument(
        "--provider",
        default="openai",
        choices=["openai", "cohere", "gemini", "together"],
        help="API provider to use (default: openai).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name. Defaults to the provider's recommended model.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="DPI for PDF-to-image conversion (default: 200).",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=16000,
        dest="max_output_tokens",
        help="Max output tokens per GPT call (default: 8000).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=15,
        dest="chunk_size",
        help="Max pages per GPT call (default: 15). Longer papers are split into chunks.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run all steps even if outputs already exist.",
    )
    parser.add_argument(
        "--skip-extraction",
        action="store_true",
        dest="skip_extraction",
        help="Skip the X1-5 extraction step (use existing extraction outputs).",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        dest="skip_eval",
        help="Skip the evaluation step.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        dest="skip_plots",
        help="Skip the accuracy plots step.",
    )
    args = parser.parse_args()
    run(
        dataset=args.dataset,
        pdf_name=args.pdf_name,
        provider=args.provider,
        model=args.model,
        dpi=args.dpi,
        max_output_tokens=args.max_output_tokens,
        chunk_size=args.chunk_size,
        force=args.force,
        skip_extraction=args.skip_extraction,
        skip_eval=args.skip_eval,
        skip_plots=args.skip_plots,
    )


if __name__ == "__main__":
    main()
