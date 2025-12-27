#!/usr/bin/env python3
"""Create per-study accuracy bar plots from evaluation outputs.

Generates four separate plots:
1. Factor loadings accuracy (by study)
2. Correlations accuracy (by study)
3. Metadata accuracy (by study)
4. Metadata field accuracy (mean per field across all studies)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def _iter_eval_files(root: Path) -> List[Path]:
    return sorted(root.glob("**/*.json"))


def _collect_metrics_by_type(path: Path) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    """Collect metrics separately for factor loadings, correlations, and metadata.

    Returns:
        Tuple of (factors_metrics, correlations_metrics, metadata_metrics)
        Each metrics tuple contains (avg_all, avg_nonzero)
    """
    data = path.read_text(encoding="utf-8")
    payload = __import__("json").loads(data)

    # Collect factor loadings metrics
    factor_entries = (payload.get("factor_loadings") or {}).values()
    factor_accuracies = []
    factor_accuracies_nonzero = []
    for entry in factor_entries:
        if not isinstance(entry, dict):
            continue
        accuracy = entry.get("accuracy")
        truth = entry.get("true")
        if accuracy is None:
            continue
        factor_accuracies.append(accuracy)
        if truth not in (None, 0):
            factor_accuracies_nonzero.append(accuracy)

    factor_avg_all = sum(factor_accuracies) / len(factor_accuracies) if factor_accuracies else 0.0
    factor_avg_nonzero = (
        sum(factor_accuracies_nonzero) / len(factor_accuracies_nonzero)
        if factor_accuracies_nonzero
        else 0.0
    )

    # Collect correlations metrics
    corr_entries = (payload.get("factor_correlations") or {}).values()
    corr_accuracies = []
    corr_accuracies_nonzero = []
    for entry in corr_entries:
        if not isinstance(entry, dict):
            continue
        accuracy = entry.get("accuracy")
        truth = entry.get("true")
        if accuracy is None:
            continue
        corr_accuracies.append(accuracy)
        if truth not in (None, 0):
            corr_accuracies_nonzero.append(accuracy)

    corr_avg_all = sum(corr_accuracies) / len(corr_accuracies) if corr_accuracies else 0.0
    corr_avg_nonzero = (
        sum(corr_accuracies_nonzero) / len(corr_accuracies_nonzero)
        if corr_accuracies_nonzero
        else 0.0
    )

    # Collect metadata metrics
    meta_entries = (payload.get("metadata") or {}).values()
    meta_accuracies = []
    meta_accuracies_nonzero = []
    for entry in meta_entries:
        if not isinstance(entry, dict):
            continue
        accuracy = entry.get("accuracy")
        truth = entry.get("true")
        if accuracy is None:
            continue
        meta_accuracies.append(accuracy)
        if truth not in (None, 0):
            meta_accuracies_nonzero.append(accuracy)

    meta_avg_all = sum(meta_accuracies) / len(meta_accuracies) if meta_accuracies else 0.0
    meta_avg_nonzero = (
        sum(meta_accuracies_nonzero) / len(meta_accuracies_nonzero)
        if meta_accuracies_nonzero
        else 0.0
    )

    return (
        (factor_avg_all, factor_avg_nonzero),
        (corr_avg_all, corr_avg_nonzero),
        (meta_avg_all, meta_avg_nonzero),
    )


def _create_plot(
    studies: List[str],
    avg_all_vals: List[float],
    avg_nonzero_vals: List[float],
    title: str,
    output_file: Path,
) -> None:
    """Create and save a single accuracy plot."""
    x = range(len(studies))
    width = 0.4

    plt.figure(figsize=(max(10, len(studies) * 0.5), 6))
    plt.bar([i - width / 2 for i in x], avg_all_vals, width, label="Avg accuracy")
    plt.bar(
        [i + width / 2 for i in x],
        avg_nonzero_vals,
        width,
        label="Avg accuracy (non-zero true)",
    )
    plt.xticks(list(x), studies, rotation=90)
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file, dpi=200)
    plt.close()


def run(
    dataset: str | None = None,
    output_dir: str | None = None,
    force: bool = False,
) -> List[Path]:
    """Generate four accuracy plots from evaluation results.

    Args:
        dataset: Optional dataset name (tuning, validation, eval)
        output_dir: Optional output directory for the plots
        force: Force regeneration even if plots already exist

    Returns:
        List of paths to the generated plots
    """
    # Determine input directory
    if dataset:
        input_dir = Path("data/evaluation") / dataset
    else:
        input_dir = Path("data/evaluation")

    if not input_dir.exists():
        print(f"No evaluation directory found: {input_dir}")
        return []

    # Determine output directory and filenames
    if output_dir is None:
        out_dir = Path("figures")
    else:
        out_dir = Path(output_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    if dataset:
        output_files = {
            "factors": out_dir / f"accuracy_plot_factors_{dataset}.png",
            "correlations": out_dir / f"accuracy_plot_correlations_{dataset}.png",
            "metadata": out_dir / f"accuracy_plot_metadata_{dataset}.png",
            "metadata_fields": out_dir / f"accuracy_plot_metadata_fields_{dataset}.png",
        }
    else:
        output_files = {
            "factors": out_dir / "accuracy_plot_factors.png",
            "correlations": out_dir / "accuracy_plot_correlations.png",
            "metadata": out_dir / "accuracy_plot_metadata.png",
            "metadata_fields": out_dir / "accuracy_plot_metadata_fields.png",
        }

    # Check if outputs already exist
    if not force and all(f.exists() for f in output_files.values()):
        print(f"Skipping accuracy plots: all already exist")
        return list(output_files.values())

    files = _iter_eval_files(input_dir)
    if not files:
        print(f"No evaluation JSON files found in {input_dir}")
        return []

    # Collect metrics for each type
    study_factors_all: Dict[str, float] = {}
    study_factors_nonzero: Dict[str, float] = {}
    study_corr_all: Dict[str, float] = {}
    study_corr_nonzero: Dict[str, float] = {}
    study_meta_all: Dict[str, float] = {}
    study_meta_nonzero: Dict[str, float] = {}

    for file_path in files:
        study = file_path.stem
        (
            (factor_all, factor_nonzero),
            (corr_all, corr_nonzero),
            (meta_all, meta_nonzero),
        ) = _collect_metrics_by_type(file_path)

        study_factors_all[study] = factor_all
        study_factors_nonzero[study] = factor_nonzero
        study_corr_all[study] = corr_all
        study_corr_nonzero[study] = corr_nonzero
        study_meta_all[study] = meta_all
        study_meta_nonzero[study] = meta_nonzero

    studies = sorted(study_factors_all.keys())

    # Create factor loadings plot
    if not force and output_files["factors"].exists():
        print(f"Skipping factor loadings plot: already exists at {output_files['factors']}")
    else:
        factor_all_vals = [study_factors_all[s] for s in studies]
        factor_nonzero_vals = [study_factors_nonzero[s] for s in studies]
        _create_plot(
            studies,
            factor_all_vals,
            factor_nonzero_vals,
            "Factor Loadings Accuracy by Study",
            output_files["factors"],
        )
        print(f"✓ Saved factor loadings plot to: {output_files['factors']}")

    # Create correlations plot
    if not force and output_files["correlations"].exists():
        print(f"Skipping correlations plot: already exists at {output_files['correlations']}")
    else:
        corr_all_vals = [study_corr_all[s] for s in studies]
        corr_nonzero_vals = [study_corr_nonzero[s] for s in studies]
        _create_plot(
            studies,
            corr_all_vals,
            corr_nonzero_vals,
            "Correlations Accuracy by Study",
            output_files["correlations"],
        )
        print(f"✓ Saved correlations plot to: {output_files['correlations']}")

    # Create metadata plot
    if not force and output_files["metadata"].exists():
        print(f"Skipping metadata plot: already exists at {output_files['metadata']}")
    else:
        meta_all_vals = [study_meta_all[s] for s in studies]
        meta_nonzero_vals = [study_meta_nonzero[s] for s in studies]
        _create_plot(
            studies,
            meta_all_vals,
            meta_nonzero_vals,
            "Metadata Accuracy by Study",
            output_files["metadata"],
        )
        print(f"✓ Saved metadata plot to: {output_files['metadata']}")

    # Create metadata fields plot (mean accuracy per metadata field across all studies)
    if not force and output_files["metadata_fields"].exists():
        print(f"Skipping metadata fields plot: already exists at {output_files['metadata_fields']}")
    else:
        # Collect accuracy per metadata field across all studies
        field_accuracies: Dict[str, List[float]] = {}
        for file_path in files:
            data = file_path.read_text(encoding="utf-8")
            payload = __import__("json").loads(data)
            meta_entries = (payload.get("metadata") or {})

            for field_name, entry in meta_entries.items():
                if not isinstance(entry, dict):
                    continue
                accuracy = entry.get("accuracy")
                if accuracy is None:
                    continue
                if field_name not in field_accuracies:
                    field_accuracies[field_name] = []
                field_accuracies[field_name].append(accuracy)

        # Calculate mean accuracy per field
        field_means = {
            field: sum(accs) / len(accs) if accs else 0.0
            for field, accs in field_accuracies.items()
        }

        # Sort fields by name for consistent ordering
        sorted_fields = sorted(field_means.keys())
        field_mean_vals = [field_means[f] for f in sorted_fields]

        # Create bar plot
        plt.figure(figsize=(max(10, len(sorted_fields) * 0.6), 6))
        x = range(len(sorted_fields))
        plt.bar(x, field_mean_vals, color='steelblue')
        plt.xticks(list(x), sorted_fields, rotation=90)
        plt.ylim(0, 1)
        plt.ylabel("Mean Accuracy")
        plt.title("Mean Accuracy by Metadata Field (Across All Studies)")
        plt.tight_layout()
        plt.savefig(output_files["metadata_fields"], dpi=200)
        plt.close()
        print(f"✓ Saved metadata fields plot to: {output_files['metadata_fields']}")

    return list(output_files.values())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot per-study evaluation accuracy bars (4 separate plots)."
    )
    parser.add_argument(
        "--dataset",
        help="Optional dataset name (tuning, validation, eval).",
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory for the plots.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration even if plots already exist.",
    )
    args = parser.parse_args()

    run(dataset=args.dataset, output_dir=args.output_dir, force=args.force)


if __name__ == "__main__":
    main()
