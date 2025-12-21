#!/usr/bin/env python3
"""Create per-study accuracy bar plots from evaluation outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def _iter_eval_files(root: Path) -> List[Path]:
    return sorted(root.glob("**/*.json"))


def _collect_metrics(path: Path) -> Tuple[float, float]:
    data = path.read_text(encoding="utf-8")
    payload = __import__("json").loads(data)
    entries = []
    entries.extend((payload.get("factor_loadings") or {}).values())
    entries.extend((payload.get("factor_correlations") or {}).values())

    accuracies = []
    accuracies_nonzero = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        accuracy = entry.get("accuracy")
        truth = entry.get("true")
        if accuracy is None:
            continue
        accuracies.append(accuracy)
        if truth not in (None, 0):
            accuracies_nonzero.append(accuracy)

    avg_all = sum(accuracies) / len(accuracies) if accuracies else 0.0
    avg_nonzero = (
        sum(accuracies_nonzero) / len(accuracies_nonzero)
        if accuracies_nonzero
        else 0.0
    )
    return avg_all, avg_nonzero


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot per-study evaluation accuracy bars."
    )
    parser.add_argument(
        "--input",
        default="data/evaluation",
        help="Root folder containing evaluation JSON files.",
    )
    parser.add_argument(
        "--output",
        default="figures/accuracy_plot.png",
        help="Output PNG path for the plot.",
    )
    args = parser.parse_args()

    root = Path(args.input)
    files = _iter_eval_files(root)
    if not files:
        raise FileNotFoundError(f"No evaluation JSON files found under {root}")

    study_metrics: Dict[str, List[float]] = {}
    study_metrics_nonzero: Dict[str, List[float]] = {}
    for file_path in files:
        study = file_path.stem
        avg_all, avg_nonzero = _collect_metrics(file_path)
        study_metrics[study] = avg_all
        study_metrics_nonzero[study] = avg_nonzero

    studies = sorted(study_metrics.keys())
    avg_all_vals = [study_metrics[s] for s in studies]
    avg_nonzero_vals = [study_metrics_nonzero[s] for s in studies]

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
    plt.title("Evaluation Accuracy by Study")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output, dpi=200)


if __name__ == "__main__":
    main()
