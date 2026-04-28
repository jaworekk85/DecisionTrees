from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update(
    {
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.titlesize": 14,
        "lines.linewidth": 1.8,
        "lines.markersize": 4.8,
    }
)

import matplotlib.pyplot as plt
import numpy as np


RESULTS_PATH = Path("results") / "wrong_split_probability.csv"
PLOTS_DIR = Path("results") / "plots"
PANEL_PARENT_PROBABILITIES = [0.2, 0.3, 0.4, 0.5]

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
        "p",
        "delta_a",
        "delta_b",
        "n_samples",
        "repetitions",
        "wrong_rate",
        "correct_rate",
        "tie_rate",
        "wrong_rate_se",
        "mean_empirical_margin",
        "std_empirical_margin",
    }

    with path.open(newline="", encoding="utf-8") as input_file:
        reader = csv.DictReader(input_file)
        for row in reader:
            parsed = dict(row)
            for column in numeric_columns:
                parsed[column] = float(row[column]) if row[column] else float("nan")
            rows.append(parsed)
    return rows


def plot_wrong_split_probability_by_p(rows: List[dict[str, str | float]]) -> Path:
    path = PLOTS_DIR / "wrong_split_probability_vs_sample_size_by_p.png"
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    available_parent_probabilities = sorted({float(row["p"]) for row in rows})
    parent_probabilities = [
        parent_probability
        for parent_probability in PANEL_PARENT_PROBABILITIES
        if any(
            abs(parent_probability - available_parent_probability) < 1e-12
            for available_parent_probability in available_parent_probabilities
        )
    ]
    n_cols = 2
    n_rows = int(np.ceil(len(parent_probabilities) / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(8.6, 6.8),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    flat_axes = list(np.ravel(axes))

    for axis, parent_probability in zip(flat_axes, parent_probabilities):
        p_rows = [
            row
            for row in rows
            if abs(float(row["p"]) - parent_probability) < 1e-12
        ]
        grouped: Dict[str, List[dict[str, str | float]]] = defaultdict(list)
        for row in p_rows:
            grouped[str(row["impurity"])].append(row)

        for impurity in IMPURITY_ORDER:
            impurity_rows = sorted(
                grouped[impurity],
                key=lambda row: float(row["n_samples"]),
            )
            x = [float(row["n_samples"]) for row in impurity_rows]
            y = [float(row["wrong_rate"]) for row in impurity_rows]
            axis.plot(
                x,
                y,
                marker="o",
                linewidth=1.8,
                markersize=4.6,
                label=IMPURITY_LABELS[impurity],
            )

        axis.set_xscale("log")
        axis.set_title(rf"$p={parent_probability:.2f}$")
        axis.grid(alpha=0.25)

    for axis in flat_axes[len(parent_probabilities):]:
        axis.set_visible(False)

    handles, labels = flat_axes[0].get_legend_handles_labels()
    fig.supxlabel(r"Sample size $n$", y=0.08)
    fig.supylabel("Wrong split probability", x=0.02)
    fig.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=3,
        frameon=False,
    )
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_wrong_split_probability_over_p(rows: List[dict[str, str | float]]) -> Path:
    path = PLOTS_DIR / "wrong_split_probability_over_p.png"
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    selected_sample_sizes = [100, 500, 2000]

    fig, axes = plt.subplots(
        1,
        len(selected_sample_sizes),
        figsize=(8.8, 4.6),
        sharey=True,
        constrained_layout=True,
    )

    for axis, n_samples in zip(axes, selected_sample_sizes):
        n_rows = [row for row in rows if int(float(row["n_samples"])) == n_samples]
        grouped: Dict[str, List[dict[str, str | float]]] = defaultdict(list)
        for row in n_rows:
            grouped[str(row["impurity"])].append(row)

        for impurity in IMPURITY_ORDER:
            impurity_rows = sorted(grouped[impurity], key=lambda row: float(row["p"]))
            axis.plot(
                [float(row["p"]) for row in impurity_rows],
                [float(row["wrong_rate"]) for row in impurity_rows],
                marker="o",
                linewidth=1.8,
                markersize=4.8,
                label=IMPURITY_LABELS[impurity],
            )

        axis.set_title(rf"$n={n_samples}$")
        axis.set_xlabel(r"$p$")
        axis.grid(alpha=0.25)

    axes[0].set_ylabel("Wrong split probability")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=3,
        frameon=False,
    )
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_wrong_split_probability_single_p(
    rows: List[dict[str, str | float]],
    parent_probability: float = 0.5,
) -> Path:
    path = PLOTS_DIR / "wrong_split_probability_vs_sample_size.png"
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    selected_rows = [
        row for row in rows if abs(float(row["p"]) - parent_probability) < 1e-12
    ]
    grouped: Dict[str, List[dict[str, str | float]]] = defaultdict(list)
    for row in selected_rows:
        grouped[str(row["impurity"])].append(row)

    first = selected_rows[0]
    plt.figure(figsize=(7.4, 5.2))
    for impurity in IMPURITY_ORDER:
        impurity_rows = sorted(grouped[impurity], key=lambda row: float(row["n_samples"]))
        x = [float(row["n_samples"]) for row in impurity_rows]
        y = [float(row["wrong_rate"]) for row in impurity_rows]
        yerr = [1.96 * float(row["wrong_rate_se"]) for row in impurity_rows]
        plt.errorbar(
            x,
            y,
            yerr=yerr,
            marker="o",
            linewidth=1.6,
            markersize=4.0,
            capsize=3.0,
            label=IMPURITY_LABELS[impurity],
        )

    plt.xscale("log")
    plt.xlabel(r"Sample size at the node  $n$")
    plt.ylabel("Wrong split probability")
    plt.title(
        rf"Empirical split error "
        rf"($p={float(first['p']):.2f}$, "
        rf"$\delta_A={float(first['delta_a']):.2f}$, "
        rf"$\delta_B={float(first['delta_b']):.2f}$)"
    )
    plt.legend(fontsize=8)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()
    return path


def plot_wrong_split_probability_legacy(rows: List[dict[str, str | float]]) -> Path:
    path = PLOTS_DIR / "wrong_split_probability_vs_sample_size_all_pooled.png"
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    grouped: Dict[str, List[dict[str, str | float]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["impurity"])].append(row)

    first = rows[0]
    plt.figure(figsize=(7.4, 5.2))
    for impurity in IMPURITY_ORDER:
        impurity_rows = sorted(grouped[impurity], key=lambda row: float(row["n_samples"]))
        x = [float(row["n_samples"]) for row in impurity_rows]
        y = [float(row["wrong_rate"]) for row in impurity_rows]
        yerr = [1.96 * float(row["wrong_rate_se"]) for row in impurity_rows]
        plt.errorbar(
            x,
            y,
            yerr=yerr,
            marker="o",
            linewidth=1.6,
            markersize=4.0,
            capsize=3.0,
            label=IMPURITY_LABELS[impurity],
        )

    plt.xscale("log")
    plt.xlabel(r"Sample size at the node  $n$")
    plt.ylabel("Wrong split probability")
    plt.title(
        rf"Empirical split error "
        rf"($p={float(first['p']):.2f}$, "
        rf"$\delta_A={float(first['delta_a']):.2f}$, "
        rf"$\delta_B={float(first['delta_b']):.2f}$)"
    )
    plt.legend(fontsize=8)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()
    return path


def main() -> None:
    rows = load_rows(RESULTS_PATH)
    paths = [
        plot_wrong_split_probability_by_p(rows),
        plot_wrong_split_probability_over_p(rows),
    ]
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
