from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


RESULTS_PATH = Path("results") / "population_margin_vs_curvature.csv"
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
        "p",
        "delta_a",
        "delta_b",
        "v_a",
        "v_b",
        "v_diff",
        "curvature",
        "exact_margin",
        "predicted_margin",
        "absolute_error",
        "relative_error",
    }

    with path.open(newline="", encoding="utf-8") as input_file:
        reader = csv.DictReader(input_file)
        for row in reader:
            parsed = dict(row)
            for column in numeric_columns:
                parsed[column] = float(row[column]) if row[column] else float("nan")
            rows.append(parsed)
    return rows


def save_current_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def plot_predicted_vs_exact(rows: List[dict[str, str | float]]) -> Path:
    path = PLOTS_DIR / "population_margin_predicted_vs_exact.png"
    plt.figure(figsize=(7.0, 5.2))

    for impurity in IMPURITY_ORDER:
        impurity_rows = [row for row in rows if row["impurity"] == impurity]
        plt.scatter(
            [float(row["predicted_margin"]) for row in impurity_rows],
            [float(row["exact_margin"]) for row in impurity_rows],
            label=IMPURITY_LABELS[impurity],
            s=28,
            alpha=0.75,
        )

    all_values = [
        float(row["predicted_margin"]) for row in rows
    ] + [float(row["exact_margin"]) for row in rows]
    lower = min(all_values)
    upper = max(all_values)
    plt.plot([lower, upper], [lower, upper], color="black", linewidth=1.0, label="y = x")
    plt.xlabel(r"Second-order approximation  $-\frac{1}{2} g''(p)(V_A - V_B)$")
    plt.ylabel(r"Exact population margin  $F = M_A - M_B$")
    plt.title("Exact margin versus second-order approximation")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.25)
    save_current_figure(path)
    return path


def plot_margin_by_impurity(rows: List[dict[str, str | float]]) -> Path:
    path = PLOTS_DIR / "population_margin_by_impurity.png"
    grouped = [
        [
            float(row["exact_margin"])
            for row in rows
            if row["impurity"] == impurity
        ]
        for impurity in IMPURITY_ORDER
    ]

    plt.figure(figsize=(8.2, 5.2))
    plt.boxplot(
        grouped,
        tick_labels=[IMPURITY_LABELS[name] for name in IMPURITY_ORDER],
    )
    plt.ylabel(r"Population margin  $F$")
    plt.title("Distribution of population margins over controlled split configurations")
    plt.xticks(rotation=25, ha="right")
    plt.grid(axis="y", alpha=0.25)
    save_current_figure(path)
    return path


def plot_margin_over_parent_probability(
    rows: List[dict[str, str | float]],
    delta_a: float = 0.05,
    delta_b: float = 0.025,
) -> Path:
    path = PLOTS_DIR / "population_margin_over_parent_probability.png"
    selected = [
        row
        for row in rows
        if float(row["delta_a"]) == delta_a and float(row["delta_b"]) == delta_b
    ]

    grouped: Dict[str, List[dict[str, str | float]]] = defaultdict(list)
    for row in selected:
        grouped[str(row["impurity"])].append(row)

    plt.figure(figsize=(7.4, 5.2))
    for impurity in IMPURITY_ORDER:
        impurity_rows = sorted(grouped[impurity], key=lambda row: float(row["p"]))
        plt.plot(
            [float(row["p"]) for row in impurity_rows],
            [float(row["exact_margin"]) for row in impurity_rows],
            marker="o",
            linewidth=1.6,
            markersize=4.0,
            label=IMPURITY_LABELS[impurity],
        )

    plt.xlabel(r"Parent-node class probability  $p$")
    plt.ylabel(r"Population margin  $F$")
    plt.title(
        rf"Population margin as a function of $p$ "
        rf"($\delta_A={delta_a}$, $\delta_B={delta_b}$)"
    )
    plt.legend(fontsize=8)
    plt.grid(alpha=0.25)
    save_current_figure(path)
    return path


def main() -> None:
    rows = load_rows(RESULTS_PATH)
    paths = [
        plot_predicted_vs_exact(rows),
        plot_margin_by_impurity(rows),
        plot_margin_over_parent_probability(rows),
    ]
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
