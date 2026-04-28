from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.titlesize": 14,
    }
)

import matplotlib.pyplot as plt
import numpy as np


RESULTS_PATH = Path("results") / "random_tree_benchmark_checkpoints.csv"
PLOTS_DIR = Path("results") / "plots"

IMPURITY_LABELS = {
    "gini": "Gini",
    "entropy": "Entropy",
    "restricted_quadratic_global": "RQ-global",
    "restricted_quadratic_local": "RQ-local",
    "restricted_entropy_global": "RE-global",
    "restricted_entropy_local": "RE-local",
}

IMPURITY_ORDER = list(IMPURITY_LABELS.keys())


def load_rows(path: Path) -> List[dict[str, str | float]]:
    rows: List[dict[str, str | float]] = []
    numeric_columns = {
        "epsilon",
        "seed",
        "n_features",
        "omega",
        "max_depth",
        "min_leaf_depth",
        "n_samples_total",
        "checkpoint",
        "accuracy",
        "n_split_checks",
        "n_splits",
        "n_leaves",
        "mean_margin",
        "mean_margin_over_threshold",
    }

    with path.open(newline="", encoding="utf-8") as input_file:
        reader = csv.DictReader(input_file)
        for row in reader:
            parsed = dict(row)
            for column in numeric_columns:
                parsed[column] = float(row[column]) if row[column] else float("nan")
            rows.append(parsed)
    return rows


def grouped_mean_rows(
    rows: List[dict[str, str | float]],
    metric: str,
) -> Dict[tuple[int, float, str], List[tuple[float, float]]]:
    grouped: Dict[tuple[int, float, str], Dict[float, List[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for row in rows:
        key = (int(float(row["n_features"])), float(row["omega"]), str(row["impurity"]))
        grouped[key][float(row["checkpoint"])].append(float(row[metric]))

    result: Dict[tuple[int, float, str], List[tuple[float, float]]] = {}
    for key, values_by_checkpoint in grouped.items():
        result[key] = [
            (checkpoint, float(np.mean(values)))
            for checkpoint, values in sorted(values_by_checkpoint.items())
        ]
    return result


def plot_metric_grid(
    rows: List[dict[str, str | float]],
    metric: str,
    ylabel: str,
    title: str,
    output_name: str,
) -> Path:
    path = PLOTS_DIR / output_name
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    n_features_values = sorted({int(float(row["n_features"])) for row in rows})
    omega_values = sorted({float(row["omega"]) for row in rows})
    mean_rows = grouped_mean_rows(rows, metric)

    fig, axes = plt.subplots(
        len(n_features_values),
        len(omega_values),
        figsize=(11.8, 7.8),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    for row_index, n_features in enumerate(n_features_values):
        for col_index, omega in enumerate(omega_values):
            axis = axes[row_index, col_index]
            for impurity in IMPURITY_ORDER:
                series = mean_rows.get((n_features, omega, impurity), [])
                if not series:
                    continue
                x = [point[0] for point in series]
                y = [point[1] for point in series]
                axis.plot(
                    x,
                    y,
                    linewidth=1.6,
                    label=IMPURITY_LABELS[impurity],
                )
            axis.set_xscale("log")
            axis.grid(alpha=0.25)
            if row_index == 0:
                axis.set_title(rf"$\omega={omega:.1f}$")
            if col_index == 0:
                axis.set_ylabel(f"{ylabel}\nfeatures={n_features}")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.supxlabel("Processed samples")
    fig.legend(
        handles,
        labels,
        loc="outside right center",
        ncol=1,
        frameon=False,
    )
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def main() -> None:
    rows = load_rows(RESULTS_PATH)
    paths = [
        plot_metric_grid(
            rows,
            metric="accuracy",
            ylabel="Accuracy",
            title="Prequential accuracy on random-tree streams",
            output_name="random_tree_accuracy_over_time.png",
        ),
        plot_metric_grid(
            rows,
            metric="n_splits",
            ylabel="Splits",
            title="Tree growth on random-tree streams",
            output_name="random_tree_splits_over_time.png",
        ),
        plot_metric_grid(
            rows,
            metric="mean_margin_over_threshold",
            ylabel=r"Mean $\hat F/\epsilon$",
            title="Split-check margin relative to threshold",
            output_name="random_tree_margin_ratio_over_time.png",
        ),
    ]
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
