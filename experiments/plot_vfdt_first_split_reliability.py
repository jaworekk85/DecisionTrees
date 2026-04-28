from __future__ import annotations

import csv
from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update(
    {
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
    }
)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


RESULTS_PATH = Path("results") / "vfdt_first_split_reliability.csv"
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


def load_rows(path: Path) -> List[dict[str, str | float | bool | None]]:
    rows: List[dict[str, str | float | bool | None]] = []
    numeric_columns = {
        "epsilon",
        "p",
        "delta_a",
        "delta_b",
        "repetition",
        "seed",
        "max_samples",
        "vfdt_delta",
        "grace_period",
        "min_samples_to_split",
        "samples_until_split",
        "first_split_feature",
        "margin_at_split",
        "threshold_at_split",
        "margin_over_threshold",
        "best_gain",
        "second_best_gain",
    }
    boolean_columns = {"split_occurred", "correct_first_split"}

    with path.open(newline="", encoding="utf-8") as input_file:
        reader = csv.DictReader(input_file)
        for row in reader:
            parsed: dict[str, str | float | bool | None] = dict(row)
            for column in numeric_columns:
                parsed[column] = float(row[column]) if row[column] else None
            for column in boolean_columns:
                parsed[column] = row[column] == "True"
            rows.append(parsed)
    return rows


def save_current_figure(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.name != "vfdt_margin_vs_threshold_by_p.png":
        plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()
    return path


def _rows_for(rows: List[dict[str, str | float | bool | None]], impurity: str, p: float):
    return [
        row
        for row in rows
        if row["impurity"] == impurity and float(row["p"]) == p
    ]


def plot_first_split_rates(rows: List[dict[str, str | float | bool | None]]) -> Path:
    path = PLOTS_DIR / "vfdt_first_split_rates_over_p.png"
    parent_probabilities = sorted({float(row["p"]) for row in rows})

    figure, axis = plt.subplots(figsize=(8.4, 5.4))
    for impurity in IMPURITY_ORDER:
        split_rates = []
        correct_rates = []
        for p in parent_probabilities:
            selected = _rows_for(rows, impurity, p)
            split_rates.append(np.mean([bool(row["split_occurred"]) for row in selected]))
            correct_rates.append(
                np.mean([bool(row["correct_first_split"]) for row in selected])
            )

        label = IMPURITY_LABELS[impurity]
        correct_line = axis.plot(
            parent_probabilities,
            correct_rates,
            marker="o",
            linewidth=2.0,
            label=label,
        )[0]
        axis.plot(
            parent_probabilities,
            split_rates,
            color=correct_line.get_color(),
            linestyle=(0, (4, 2)),
            marker="s",
            markerfacecolor="white",
            markeredgewidth=1.2,
            linewidth=2.4,
            alpha=0.95,
        )

    axis.set_xlabel(r"Parent-node class probability  $p$")
    axis.set_ylabel("Rate over stream replications")
    axis.set_title("VFDT-style first split reliability")
    axis.set_ylim(-0.02, 1.05)
    axis.grid(alpha=0.25)

    impurity_legend = axis.legend(ncol=2, frameon=False, loc="lower right")
    axis.add_artist(impurity_legend)
    style_handles = [
        Line2D([0], [0], color="black", marker="o", linewidth=2.0, label="Correct"),
        Line2D(
            [0],
            [0],
            color="black",
            linestyle=(0, (4, 2)),
            marker="s",
            markerfacecolor="white",
            linewidth=2.4,
            label="Any split",
        ),
    ]
    axis.legend(handles=style_handles, frameon=False, loc="lower left")
    return save_current_figure(path)


def plot_samples_until_first_split_over_p(
    rows: List[dict[str, str | float | bool | None]]
) -> Path:
    path = PLOTS_DIR / "vfdt_samples_until_first_split_over_p.png"
    parent_probabilities = sorted({float(row["p"]) for row in rows})

    plt.figure(figsize=(8.4, 5.4))
    for impurity in IMPURITY_ORDER:
        medians = []
        for p in parent_probabilities:
            selected = [
                row
                for row in _rows_for(rows, impurity, p)
                if bool(row["split_occurred"])
            ]
            medians.append(
                float(np.median([row["samples_until_split"] for row in selected]))
                if selected
                else np.nan
            )

        plt.plot(
            parent_probabilities,
            medians,
            marker="o",
            linewidth=1.8,
            label=IMPURITY_LABELS[impurity],
        )

    plt.xlabel(r"Parent-node class probability  $p$")
    plt.ylabel("Median samples until first split")
    plt.title("First split delay under the fixed VFDT-style trigger")
    plt.grid(alpha=0.25)
    plt.legend(ncol=2, frameon=False)
    return save_current_figure(path)


def plot_margin_vs_threshold_by_p(
    rows: List[dict[str, str | float | bool | None]]
) -> Path:
    path = PLOTS_DIR / "vfdt_margin_vs_threshold_by_p.png"
    split_rows = [row for row in rows if bool(row["split_occurred"])]
    parent_probabilities = sorted({float(row["p"]) for row in split_rows})

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(8.8, 7.2),
        sharex=True,
        sharey=True,
    )
    flat_axes = list(np.ravel(axes))

    all_values = [
        float(row["threshold_at_split"]) for row in split_rows
    ] + [float(row["margin_at_split"]) for row in split_rows]
    lower = min(all_values)
    upper = max(all_values)

    for axis, p in zip(flat_axes, parent_probabilities):
        p_rows = [row for row in split_rows if float(row["p"]) == p]
        for impurity in IMPURITY_ORDER:
            impurity_rows = [row for row in p_rows if row["impurity"] == impurity]
            axis.scatter(
                [float(row["threshold_at_split"]) for row in impurity_rows],
                [float(row["margin_at_split"]) for row in impurity_rows],
                s=18,
                alpha=0.65,
                label=IMPURITY_LABELS[impurity],
            )
        axis.plot([lower, upper], [lower, upper], color="black", linewidth=1.0)
        axis.set_title(rf"$p={p:.2f}$")
        axis.grid(alpha=0.25)

    handles, labels = flat_axes[0].get_legend_handles_labels()
    fig.subplots_adjust(
        left=0.16,
        right=0.98,
        top=0.92,
        bottom=0.26,
        wspace=0.12,
        hspace=0.28,
    )
    fig.text(
        0.57,
        0.14,
        r"Hoeffding threshold  $\epsilon(\delta,n)$",
        ha="center",
        va="center",
    )
    fig.text(
        0.035,
        0.59,
        r"Empirical gain margin  $\hat M_A-\hat M_B$",
        ha="center",
        va="center",
        rotation="vertical",
    )
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=3,
        frameon=False,
    )
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def main() -> None:
    rows = load_rows(RESULTS_PATH)
    paths = [
        plot_first_split_rates(rows),
        plot_samples_until_first_split_over_p(rows),
        plot_margin_vs_threshold_by_p(rows),
    ]
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
