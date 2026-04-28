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
from tree.vfdt_like import VFDTLikeTree


RESULTS_PATH = Path("results") / "vfdt_first_split_reliability.csv"


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


def sample_one(
    parent_probability: float,
    delta_a: float,
    delta_b: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
    a = int(rng.integers(0, 2))
    b = int(rng.integers(0, 2))
    class_probability = (
        parent_probability
        + delta_a * (2 * a - 1)
        + delta_b * (2 * b - 1)
    )
    y = int(rng.random() < class_probability)
    return np.array([a, b], dtype=np.int64), y


def run_once(
    impurity: Impurity,
    seed: int,
    parent_probability: float,
    delta_a: float,
    delta_b: float,
    max_samples: int,
    vfdt_delta: float,
    grace_period: int,
    min_samples_to_split: int,
) -> dict[str, float | int | bool | None]:
    rng = np.random.default_rng(seed)
    model = VFDTLikeTree(
        n_features=2,
        impurity=impurity,
        delta=vfdt_delta,
        grace_period=grace_period,
        min_samples_to_split=min_samples_to_split,
    )

    last_decision = None
    for sample_index in range(1, max_samples + 1):
        x, y = sample_one(
            parent_probability=parent_probability,
            delta_a=delta_a,
            delta_b=delta_b,
            rng=rng,
        )
        was_leaf = model.root.is_leaf
        decision = model.partial_fit_one(x, y)
        if decision is not None:
            last_decision = decision
        if was_leaf and not model.root.is_leaf:
            return {
                "split_occurred": True,
                "samples_until_split": sample_index,
                "first_split_feature": model.root.split_feature,
                "correct_first_split": model.root.split_feature == 0,
                "margin_at_split": decision.margin if decision is not None else None,
                "threshold_at_split": decision.epsilon if decision is not None else None,
                "margin_over_threshold": (
                    decision.margin / decision.epsilon
                    if decision is not None and decision.epsilon > 0.0
                    else None
                ),
                "best_gain": decision.best_gain if decision is not None else None,
                "second_best_gain": (
                    decision.second_best_gain if decision is not None else None
                ),
            }

    return {
        "split_occurred": False,
        "samples_until_split": None,
        "first_split_feature": None,
        "correct_first_split": False,
        "margin_at_split": last_decision.margin if last_decision is not None else None,
        "threshold_at_split": last_decision.epsilon if last_decision is not None else None,
        "margin_over_threshold": (
            last_decision.margin / last_decision.epsilon
            if last_decision is not None and last_decision.epsilon > 0.0
            else None
        ),
        "best_gain": last_decision.best_gain if last_decision is not None else None,
        "second_best_gain": (
            last_decision.second_best_gain if last_decision is not None else None
        ),
    }


def run_experiment(
    delta_a: float = 0.08,
    delta_b: float = 0.04,
    epsilon: float = 0.15,
    repetitions: int = 100,
    max_samples: int = 20000,
    vfdt_delta: float = 1e-3,
    grace_period: int = 50,
    min_samples_to_split: int = 100,
    seed: int = 20260428,
) -> List[dict[str, float | int | str | bool | None]]:
    rows: List[dict[str, float | int | str | bool | None]] = []
    parent_probabilities = [0.2, 0.3, 0.4, 0.5]

    for p_index, parent_probability in enumerate(parent_probabilities):
        probability_min = parent_probability - delta_a - delta_b
        probability_max = parent_probability + delta_a + delta_b
        if probability_min < 0.0 or probability_max > 1.0:
            raise ValueError(
                "parent_probability, delta_a, and delta_b create invalid "
                f"class probabilities: [{probability_min}, {probability_max}]"
            )

        for spec_index, spec in enumerate(impurity_specs(epsilon=epsilon)):
            for repetition in range(repetitions):
                run_seed = seed + p_index * 1000000 + spec_index * 100000 + repetition
                result = run_once(
                    impurity=spec.impurity,
                    seed=run_seed,
                    parent_probability=parent_probability,
                    delta_a=delta_a,
                    delta_b=delta_b,
                    max_samples=max_samples,
                    vfdt_delta=vfdt_delta,
                    grace_period=grace_period,
                    min_samples_to_split=min_samples_to_split,
                )
                rows.append(
                    {
                        "impurity": spec.name,
                        "epsilon": spec.epsilon,
                        "p": parent_probability,
                        "delta_a": delta_a,
                        "delta_b": delta_b,
                        "repetition": repetition,
                        "seed": run_seed,
                        "max_samples": max_samples,
                        "vfdt_delta": vfdt_delta,
                        "grace_period": grace_period,
                        "min_samples_to_split": min_samples_to_split,
                        **result,
                    }
                )

    return rows


def write_rows(rows: List[dict[str, float | int | str | bool | None]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: List[dict[str, float | int | str | bool | None]]) -> None:
    print(f"wrote {len(rows)} rows to {RESULTS_PATH}")
    impurities = sorted({str(row["impurity"]) for row in rows})
    parent_probabilities = sorted({float(row["p"]) for row in rows})
    for parent_probability in parent_probabilities:
        print(f"p={parent_probability:.2f}")
        p_rows = [row for row in rows if float(row["p"]) == parent_probability]
        for impurity in impurities:
            impurity_rows = [row for row in p_rows if row["impurity"] == impurity]
            split_rows = [row for row in impurity_rows if row["split_occurred"]]
            correct_rows = [row for row in impurity_rows if row["correct_first_split"]]
            split_rate = len(split_rows) / len(impurity_rows)
            correct_rate = len(correct_rows) / len(impurity_rows)
            median_samples = (
                float(np.median([row["samples_until_split"] for row in split_rows]))
                if split_rows
                else float("nan")
            )
            print(
                f"  {impurity}: split_rate={split_rate:.3f}, "
                f"correct_rate={correct_rate:.3f}, "
                f"median_samples={median_samples:.1f}"
            )


def main() -> None:
    rows = run_experiment()
    write_rows(rows, RESULTS_PATH)
    summarize(rows)


if __name__ == "__main__":
    main()
