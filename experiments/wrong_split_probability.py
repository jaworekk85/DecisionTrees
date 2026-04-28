from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

from impurity import (
    EntropyImpurity,
    GiniImpurity,
    GloballyConcaveRestrictedEntropy,
    GloballyConcaveRestrictedQuadratic,
    LocallyConcaveRestrictedEntropy,
    LocallyConcaveRestrictedQuadratic,
)
from impurity.base import Impurity


RESULTS_PATH = Path("results") / "wrong_split_probability.csv"


@dataclass(frozen=True)
class ImpuritySpec:
    name: str
    impurity: Impurity
    epsilon: float | None = None


def impurity_specs(epsilon: float = 0.15) -> List[ImpuritySpec]:
    return [
        ImpuritySpec("gini", GiniImpurity()),
        ImpuritySpec("entropy", EntropyImpurity()),
        ImpuritySpec(
            "restricted_quadratic_global",
            GloballyConcaveRestrictedQuadratic(epsilon=epsilon),
            epsilon=epsilon,
        ),
        ImpuritySpec(
            "restricted_quadratic_local",
            LocallyConcaveRestrictedQuadratic(epsilon=epsilon),
            epsilon=epsilon,
        ),
        ImpuritySpec(
            "restricted_entropy_global",
            GloballyConcaveRestrictedEntropy(epsilon=epsilon),
            epsilon=epsilon,
        ),
        ImpuritySpec(
            "restricted_entropy_local",
            LocallyConcaveRestrictedEntropy(epsilon=epsilon),
            epsilon=epsilon,
        ),
    ]


def sample_two_split_node(
    n_samples: int,
    parent_probability: float,
    delta_a: float,
    delta_b: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    a = rng.integers(0, 2, size=n_samples)
    b = rng.integers(0, 2, size=n_samples)

    sign_a = 2 * a - 1
    sign_b = 2 * b - 1
    class_probabilities = parent_probability + delta_a * sign_a + delta_b * sign_b

    if np.any(class_probabilities < 0.0) or np.any(class_probabilities > 1.0):
        raise ValueError("Generated class probabilities leave [0, 1].")

    y = (rng.random(n_samples) < class_probabilities).astype(np.int64)
    x = np.column_stack([a, b]).astype(np.int64)
    return x, y


def empirical_gain(impurity: Impurity, x: np.ndarray, y: np.ndarray, feature: int) -> float:
    parent_probability = float(np.mean(y))
    weighted_child_impurity = 0.0

    for value in (0, 1):
        child_mask = x[:, feature] == value
        child_count = int(np.sum(child_mask))
        if child_count == 0:
            continue
        child_probability = float(np.mean(y[child_mask]))
        child_weight = child_count / len(y)
        weighted_child_impurity += child_weight * impurity.value(child_probability)

    return impurity.value(parent_probability) - weighted_child_impurity


def run_experiment(
    delta_a: float = 0.08,
    delta_b: float = 0.04,
    epsilon: float = 0.15,
    repetitions: int = 2000,
    seed: int = 20260428,
) -> List[dict[str, float | int | str | None]]:
    parent_probabilities = [0.2, 0.3, 0.4, 0.5]
    sample_sizes = [25, 50, 100, 200, 500, 1000, 2000]
    specs = impurity_specs(epsilon=epsilon)
    rng = np.random.default_rng(seed)
    rows: List[dict[str, float | int | str | None]] = []

    for parent_probability in parent_probabilities:
        probability_min = parent_probability - delta_a - delta_b
        probability_max = parent_probability + delta_a + delta_b
        if probability_min < 0.0 or probability_max > 1.0:
            raise ValueError(
                "parent_probability, delta_a, and delta_b create invalid "
                f"class probabilities: [{probability_min}, {probability_max}]"
            )

        for n_samples in sample_sizes:
            counters = {
                spec.name: {
                    "wrong": 0,
                    "correct": 0,
                    "tie": 0,
                    "margins": [],
                }
                for spec in specs
            }

            for _ in range(repetitions):
                x, y = sample_two_split_node(
                    n_samples=n_samples,
                    parent_probability=parent_probability,
                    delta_a=delta_a,
                    delta_b=delta_b,
                    rng=rng,
                )

                for spec in specs:
                    gain_a = empirical_gain(spec.impurity, x, y, feature=0)
                    gain_b = empirical_gain(spec.impurity, x, y, feature=1)
                    margin = gain_a - gain_b
                    counters[spec.name]["margins"].append(margin)

                    if margin > 0.0:
                        counters[spec.name]["correct"] += 1
                    elif margin < 0.0:
                        counters[spec.name]["wrong"] += 1
                    else:
                        counters[spec.name]["tie"] += 1

            for spec in specs:
                counts = counters[spec.name]
                wrong_rate = counts["wrong"] / repetitions
                correct_rate = counts["correct"] / repetitions
                tie_rate = counts["tie"] / repetitions
                standard_error = float(
                    np.sqrt(wrong_rate * (1.0 - wrong_rate) / repetitions)
                )

                rows.append(
                    {
                        "impurity": spec.name,
                        "epsilon": spec.epsilon,
                        "p": parent_probability,
                        "delta_a": delta_a,
                        "delta_b": delta_b,
                        "n_samples": n_samples,
                        "repetitions": repetitions,
                        "wrong_rate": wrong_rate,
                        "correct_rate": correct_rate,
                        "tie_rate": tie_rate,
                        "wrong_rate_se": standard_error,
                        "mean_empirical_margin": float(np.mean(counts["margins"])),
                        "std_empirical_margin": float(np.std(counts["margins"])),
                    }
                )

    return rows


def write_rows(rows: List[dict[str, float | int | str | None]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: List[dict[str, float | int | str | None]]) -> None:
    print(f"wrote {len(rows)} rows to {RESULTS_PATH}")
    final_n = max(int(row["n_samples"]) for row in rows)
    parent_probabilities = sorted({float(row["p"]) for row in rows})
    print(f"mean wrong split rate at n={final_n}:")
    for impurity in sorted({str(row["impurity"]) for row in rows}):
        values = [
            float(row["wrong_rate"])
            for row in rows
            if int(row["n_samples"]) == final_n and row["impurity"] == impurity
        ]
        print(f"  {impurity}: {float(np.mean(values)):.4f}")
    print(
        "parent probabilities: "
        + ", ".join(f"{parent_probability:.2f}" for parent_probability in parent_probabilities)
    )


def main() -> None:
    rows = run_experiment()
    write_rows(rows, RESULTS_PATH)
    summarize(rows)


if __name__ == "__main__":
    main()
