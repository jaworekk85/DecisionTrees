from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


CHECKPOINT_RESULTS_PATH = Path("results") / "random_tree_benchmark_checkpoints.csv"
STATS_RESULTS_PATH = Path("results") / "random_tree_statistical_comparison.csv"
LATEX_TABLE_PATH = Path("results") / "random_tree_statistical_comparison.tex"

COMPARISONS = [
    ("RQ-local - Gini", "restricted_quadratic_local", "gini"),
    ("RQ-global - Gini", "restricted_quadratic_global", "gini"),
    ("RE-local - Entropy", "restricted_entropy_local", "entropy"),
    ("RE-global - Entropy", "restricted_entropy_global", "entropy"),
]

METRICS = [
    ("accuracy_auc", "AUC"),
    ("final_accuracy", "Final acc."),
    ("final_splits", "Splits"),
    ("final_margin_ratio", r"Mean $\rho$"),
]


def load_checkpoint_rows() -> List[dict[str, str | float]]:
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

    with CHECKPOINT_RESULTS_PATH.open(newline="", encoding="utf-8") as input_file:
        reader = csv.DictReader(input_file)
        for row in reader:
            parsed = dict(row)
            for column in numeric_columns:
                parsed[column] = float(row[column]) if row[column] else float("nan")
            rows.append(parsed)
    return rows


def run_metrics(rows: List[dict[str, str | float]]) -> Dict[Tuple[int, float, int, str], dict[str, float]]:
    grouped: Dict[Tuple[int, float, int, str], List[dict[str, str | float]]] = defaultdict(list)
    for row in rows:
        key = (
            int(float(row["n_features"])),
            float(row["omega"]),
            int(float(row["seed"])),
            str(row["impurity"]),
        )
        grouped[key].append(row)

    metrics: Dict[Tuple[int, float, int, str], dict[str, float]] = {}
    for key, run_rows in grouped.items():
        run_rows = sorted(run_rows, key=lambda row: float(row["checkpoint"]))
        x = np.array([float(row["checkpoint"]) for row in run_rows])
        y = np.array([float(row["accuracy"]) for row in run_rows])
        final = run_rows[-1]

        metrics[key] = {
            "accuracy_auc": float(np.trapezoid(y, x) / (x[-1] - x[0])),
            "final_accuracy": float(final["accuracy"]),
            "final_splits": float(final["n_splits"]),
            "final_margin_ratio": float(final["mean_margin_over_threshold"]),
        }

    return metrics


def paired_differences(
    metrics_by_run: Dict[Tuple[int, float, int, str], dict[str, float]],
    treatment: str,
    baseline: str,
    metric: str,
) -> np.ndarray:
    diffs = []
    blocks = sorted(
        {
            (n_features, omega, seed)
            for n_features, omega, seed, impurity in metrics_by_run
            if impurity == baseline
        }
    )

    for n_features, omega, seed in blocks:
        treatment_key = (n_features, omega, seed, treatment)
        baseline_key = (n_features, omega, seed, baseline)
        if treatment_key not in metrics_by_run or baseline_key not in metrics_by_run:
            continue
        diffs.append(
            metrics_by_run[treatment_key][metric]
            - metrics_by_run[baseline_key][metric]
        )

    return np.array(diffs, dtype=float)


def bootstrap_ci(
    diffs: np.ndarray,
    rng: np.random.Generator,
    n_bootstrap: int = 20000,
) -> tuple[float, float]:
    indices = rng.integers(0, len(diffs), size=(n_bootstrap, len(diffs)))
    means = np.mean(diffs[indices], axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975])
    return float(lower), float(upper)


def paired_randomization_p_value(
    diffs: np.ndarray,
    rng: np.random.Generator,
    n_permutations: int = 50000,
) -> float:
    observed = abs(float(np.mean(diffs)))
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_permutations, len(diffs)))
    permuted = np.abs(np.mean(signs * diffs, axis=1))
    return float((np.sum(permuted >= observed) + 1.0) / (n_permutations + 1.0))


def holm_adjust(p_values: List[float]) -> List[float]:
    m = len(p_values)
    ordered = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [0.0] * m
    running_max = 0.0

    for rank, (index, p_value) in enumerate(ordered, start=1):
        value = min(1.0, (m - rank + 1) * p_value)
        running_max = max(running_max, value)
        adjusted[index] = running_max

    return adjusted


def analyze() -> List[dict[str, float | int | str]]:
    rng = np.random.default_rng(20260428)
    metrics_by_run = run_metrics(load_checkpoint_rows())
    rows: List[dict[str, float | int | str]] = []
    raw_p_values = []

    for comparison, treatment, baseline in COMPARISONS:
        for metric, metric_label in METRICS:
            diffs = paired_differences(
                metrics_by_run=metrics_by_run,
                treatment=treatment,
                baseline=baseline,
                metric=metric,
            )
            ci_lower, ci_upper = bootstrap_ci(diffs, rng=rng)
            p_value = paired_randomization_p_value(diffs, rng=rng)
            raw_p_values.append(p_value)

            rows.append(
                {
                    "comparison": comparison,
                    "metric": metric,
                    "metric_label": metric_label,
                    "n_blocks": len(diffs),
                    "baseline_mean": float(
                        np.mean(
                            [
                                metrics_by_run[(n_features, omega, seed, baseline)][metric]
                                for n_features, omega, seed, _ in metrics_by_run
                                if _ == baseline
                            ]
                        )
                    ),
                    "treatment_mean": float(
                        np.mean(
                            [
                                metrics_by_run[(n_features, omega, seed, treatment)][metric]
                                for n_features, omega, seed, _ in metrics_by_run
                                if _ == treatment
                            ]
                        )
                    ),
                    "mean_difference": float(np.mean(diffs)),
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "p_value": p_value,
                }
            )

    adjusted = holm_adjust(raw_p_values)
    for row, adjusted_p_value in zip(rows, adjusted):
        row["holm_p_value"] = adjusted_p_value

    return rows


def write_csv(rows: List[dict[str, float | int | str]]) -> None:
    STATS_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with STATS_RESULTS_PATH.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_p_value(value: float) -> str:
    if value < 0.001:
        return "$<0.001$"
    return f"{value:.3f}"


def write_latex(rows: List[dict[str, float | int | str]]) -> None:
    selected_metrics = {"accuracy_auc", "final_accuracy", "final_splits"}
    selected_rows = [row for row in rows if row["metric"] in selected_metrics]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Paired statistical comparison on the random-tree stream benchmark. Differences are computed over paired blocks $(d_{\mathrm{attr}},\omega,\mathrm{seed})$. Confidence intervals are bootstrap $95\%$ intervals; $p$-values are paired randomization tests with Holm correction.}",
        r"\label{tab:random_tree_statistical_comparison}",
        r"\begin{tabular}{llccc}",
        r"\hline",
        r"Comparison & Metric & Difference & 95\% CI & $p_{\mathrm{Holm}}$ \\",
        r"\hline",
    ]

    for row in selected_rows:
        lines.append(
            f"{row['comparison']} & {row['metric_label']} & "
            f"{float(row['mean_difference']):.4f} & "
            f"[{float(row['ci_lower']):.4f}, {float(row['ci_upper']):.4f}] & "
            f"{format_p_value(float(row['holm_p_value']))} \\\\"
        )

    lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    LATEX_TABLE_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    rows = analyze()
    write_csv(rows)
    write_latex(rows)
    print(f"wrote {len(rows)} rows to {STATS_RESULTS_PATH}")
    print(f"wrote LaTeX table to {LATEX_TABLE_PATH}")


if __name__ == "__main__":
    main()
