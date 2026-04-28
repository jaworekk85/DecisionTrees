from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

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


RESULTS_PATH = Path("results") / "population_margin_vs_curvature.csv"


@dataclass(frozen=True)
class ImpuritySpec:
    name: str
    impurity: Impurity
    epsilon: float | None = None


@dataclass(frozen=True)
class MarginCase:
    parent_probability: float
    delta_a: float
    delta_b: float

    @property
    def variance_a(self) -> float:
        return self.delta_a**2

    @property
    def variance_b(self) -> float:
        return self.delta_b**2


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


def margin_cases() -> Iterable[MarginCase]:
    parent_probabilities = np.linspace(0.25, 0.75, 11)
    delta_pairs = [
        (0.02, 0.01),
        (0.05, 0.025),
        (0.08, 0.04),
    ]

    for p in parent_probabilities:
        for delta_a, delta_b in delta_pairs:
            yield MarginCase(
                parent_probability=float(p),
                delta_a=delta_a,
                delta_b=delta_b,
            )


def split_gain(impurity: Impurity, p: float, delta: float) -> float:
    child_impurity = 0.5 * (
        impurity.value(p - delta) + impurity.value(p + delta)
    )
    return impurity.value(p) - child_impurity


def exact_margin(impurity: Impurity, case: MarginCase) -> float:
    return split_gain(impurity, case.parent_probability, case.delta_a) - split_gain(
        impurity,
        case.parent_probability,
        case.delta_b,
    )


def predicted_margin(impurity: Impurity, case: MarginCase) -> float:
    return -0.5 * impurity.second_derivative(case.parent_probability) * (
        case.variance_a - case.variance_b
    )


def build_rows() -> List[dict[str, float | str | None]]:
    rows: List[dict[str, float | str | None]] = []

    for spec in impurity_specs():
        for case in margin_cases():
            exact = exact_margin(spec.impurity, case)
            predicted = predicted_margin(spec.impurity, case)
            absolute_error = abs(exact - predicted)
            relative_error = absolute_error / abs(exact) if exact != 0.0 else 0.0

            rows.append(
                {
                    "impurity": spec.name,
                    "epsilon": spec.epsilon,
                    "p": case.parent_probability,
                    "delta_a": case.delta_a,
                    "delta_b": case.delta_b,
                    "v_a": case.variance_a,
                    "v_b": case.variance_b,
                    "v_diff": case.variance_a - case.variance_b,
                    "curvature": -spec.impurity.second_derivative(
                        case.parent_probability
                    ),
                    "exact_margin": exact,
                    "predicted_margin": predicted,
                    "absolute_error": absolute_error,
                    "relative_error": relative_error,
                }
            )

    return rows


def write_rows(rows: List[dict[str, float | str | None]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: List[dict[str, float | str | None]]) -> None:
    by_impurity = sorted({str(row["impurity"]) for row in rows})
    print(f"wrote {len(rows)} rows to {RESULTS_PATH}")
    print("mean relative approximation error:")
    for impurity in by_impurity:
        impurity_rows = [row for row in rows if row["impurity"] == impurity]
        mean_error = float(np.mean([row["relative_error"] for row in impurity_rows]))
        mean_margin = float(np.mean([row["exact_margin"] for row in impurity_rows]))
        print(f"  {impurity}: error={mean_error:.6f}, margin={mean_margin:.6f}")


def main() -> None:
    rows = build_rows()
    write_rows(rows, RESULTS_PATH)
    summarize(rows)


if __name__ == "__main__":
    main()
