from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np

from data.synthetic import generate_random_tree_stream
from impurity import (
    EntropyImpurity,
    GiniImpurity,
    GloballyConcaveRestrictedEntropy,
    GloballyConcaveRestrictedQuadratic,
    LocallyConcaveRestrictedEntropy,
    LocallyConcaveRestrictedQuadratic,
)
from impurity.base import Impurity
from tree.node import Node
from tree.vfdt_like import VFDTLikeTree


CHECKPOINT_RESULTS_PATH = Path("results") / "random_tree_benchmark_checkpoints.csv"
SUMMARY_RESULTS_PATH = Path("results") / "random_tree_benchmark_summary.csv"


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


def count_tree(node: Node) -> Tuple[int, int]:
    if node.is_leaf:
        return 0, 1
    assert node.left is not None
    assert node.right is not None
    left_splits, left_leaves = count_tree(node.left)
    right_splits, right_leaves = count_tree(node.right)
    return 1 + left_splits + right_splits, left_leaves + right_leaves


def split_check_summaries(model: VFDTLikeTree) -> tuple[float, float]:
    if not model.split_log:
        return 0.0, 0.0

    margins = [decision.margin for decision in model.split_log]
    ratios = [
        decision.margin / decision.epsilon
        for decision in model.split_log
        if decision.epsilon > 0.0
    ]
    return float(np.mean(margins)), float(np.mean(ratios)) if ratios else 0.0


def run_one(
    spec: ImpuritySpec,
    n_features: int,
    omega: float,
    seed: int,
    n_samples: int,
    max_depth: int,
    min_leaf_depth: int,
    checkpoints: set[int],
    vfdt_delta: float,
    grace_period: int,
    min_samples_to_split: int,
) -> List[dict[str, float | int | str | None]]:
    model = VFDTLikeTree(
        n_features=n_features,
        impurity=spec.impurity,
        delta=vfdt_delta,
        grace_period=grace_period,
        min_samples_to_split=min_samples_to_split,
    )
    stream = generate_random_tree_stream(
        n_samples=n_samples,
        n_features=n_features,
        max_depth=max_depth,
        omega=omega,
        min_leaf_depth=min_leaf_depth,
        leaf_prob_min=0.05,
        leaf_prob_max=0.95,
        feature_prob=0.5,
        label_noise=0.0,
        seed=seed,
    )

    correct = 0
    rows: List[dict[str, float | int | str | None]] = []

    for index, (x, y) in enumerate(stream, start=1):
        y_pred = model.predict_one(x)
        if y_pred == y:
            correct += 1
        model.partial_fit_one(x, y)

        if index in checkpoints:
            n_splits, n_leaves = count_tree(model.root)
            mean_margin, mean_margin_over_threshold = split_check_summaries(model)
            rows.append(
                {
                    "impurity": spec.name,
                    "epsilon": spec.epsilon,
                    "seed": seed,
                    "n_features": n_features,
                    "omega": omega,
                    "max_depth": max_depth,
                    "min_leaf_depth": min_leaf_depth,
                    "n_samples_total": n_samples,
                    "checkpoint": index,
                    "accuracy": correct / index,
                    "n_split_checks": len(model.split_log),
                    "n_splits": n_splits,
                    "n_leaves": n_leaves,
                    "mean_margin": mean_margin,
                    "mean_margin_over_threshold": mean_margin_over_threshold,
                }
            )

    return rows


def run_task(
    task: tuple[
        ImpuritySpec,
        int,
        float,
        int,
        int,
        int,
        int,
        set[int],
        float,
        int,
        int,
    ]
) -> List[dict[str, float | int | str | None]]:
    (
        spec,
        n_features,
        omega,
        seed,
        n_samples,
        max_depth,
        min_leaf_depth,
        checkpoints,
        vfdt_delta,
        grace_period,
        min_samples_to_split,
    ) = task
    return run_one(
        spec=spec,
        n_features=n_features,
        omega=omega,
        seed=seed,
        n_samples=n_samples,
        max_depth=max_depth,
        min_leaf_depth=min_leaf_depth,
        checkpoints=checkpoints,
        vfdt_delta=vfdt_delta,
        grace_period=grace_period,
        min_samples_to_split=min_samples_to_split,
    )


def run_experiment(
    n_samples: int = 1000000,
    max_depth: int = 10,
    min_leaf_depth: int = 3,
    epsilon: float = 0.15,
    n_seeds: int = 10,
    vfdt_delta: float = 1e-3,
    grace_period: int = 50,
    min_samples_to_split: int = 100,
    n_jobs: int = 1,
) -> List[dict[str, float | int | str | None]]:
    n_features_values = [6, 10, 20]
    omega_values = [0.1, 0.2, 0.3]
    checkpoints = {
        500,
        1000,
        1500,
        2000,
        3000,
        4000,
        5000,
        7500,
        10000,
        15000,
        20000,
        30000,
        40000,
        50000,
        75000,
        100000,
        150000,
        200000,
        300000,
        400000,
        500000,
        750000,
        1000000,
    }
    checkpoints = {checkpoint for checkpoint in checkpoints if checkpoint <= n_samples}
    specs = impurity_specs(epsilon=epsilon)
    tasks = []

    for n_features in n_features_values:
        for omega in omega_values:
            for seed_index in range(n_seeds):
                stream_seed = 20260428 + seed_index
                for spec in specs:
                    tasks.append(
                        (
                            spec,
                            n_features,
                            omega,
                            stream_seed,
                            n_samples,
                            max_depth,
                            min_leaf_depth,
                            checkpoints,
                            vfdt_delta,
                            grace_period,
                            min_samples_to_split,
                        )
                    )

    rows: List[dict[str, float | int | str | None]] = []
    if n_jobs == 1:
        for index, task in enumerate(tasks, start=1):
            rows.extend(run_task(task))
            _, n_features, omega, stream_seed, *_ = task
            print(
                f"finished task {index}/{len(tasks)}: "
                f"n_features={n_features}, omega={omega}, "
                f"seed={stream_seed}, n_samples={n_samples}"
            )
    else:
        worker_count = mp.cpu_count() if n_jobs <= 0 else n_jobs
        with mp.Pool(processes=worker_count) as pool:
            for index, task_rows in enumerate(
                pool.imap_unordered(run_task, tasks),
                start=1,
            ):
                rows.extend(task_rows)
                print(f"finished task {index}/{len(tasks)}")

    return rows


def write_rows(rows: List[dict[str, float | int | str | None]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_summary(
    rows: List[dict[str, float | int | str | None]]
) -> List[dict[str, float | int | str | None]]:
    final_checkpoint = max(int(row["checkpoint"]) for row in rows)
    summary_rows: List[dict[str, float | int | str | None]] = []

    keys = sorted(
        {
            (int(row["n_features"]), float(row["omega"]), str(row["impurity"]))
            for row in rows
            if int(row["checkpoint"]) == final_checkpoint
        }
    )

    for n_features, omega, impurity in keys:
        selected = [
            row
            for row in rows
            if int(row["checkpoint"]) == final_checkpoint
            and int(row["n_features"]) == n_features
            and float(row["omega"]) == omega
            and row["impurity"] == impurity
        ]
        auc_values = []
        for seed in sorted({int(row["seed"]) for row in selected}):
            seed_rows = sorted(
                [
                    row
                    for row in rows
                    if int(row["n_features"]) == n_features
                    and float(row["omega"]) == omega
                    and row["impurity"] == impurity
                    and int(row["seed"]) == seed
                ],
                key=lambda row: int(row["checkpoint"]),
            )
            x = np.array([row["checkpoint"] for row in seed_rows], dtype=float)
            y = np.array([row["accuracy"] for row in seed_rows], dtype=float)
            auc_values.append(float(np.trapz(y, x) / (x[-1] - x[0])))

        summary_rows.append(
            {
                "n_features": n_features,
                "omega": omega,
                "impurity": impurity,
                "n_runs": len(selected),
                "accuracy_auc_mean": float(np.mean(auc_values)),
                "accuracy_auc_std": float(np.std(auc_values)),
                "accuracy_mean": float(np.mean([row["accuracy"] for row in selected])),
                "accuracy_std": float(np.std([row["accuracy"] for row in selected])),
                "splits_mean": float(np.mean([row["n_splits"] for row in selected])),
                "splits_std": float(np.std([row["n_splits"] for row in selected])),
                "split_checks_mean": float(
                    np.mean([row["n_split_checks"] for row in selected])
                ),
                "split_checks_std": float(
                    np.std([row["n_split_checks"] for row in selected])
                ),
                "margin_ratio_mean": float(
                    np.mean([row["mean_margin_over_threshold"] for row in selected])
                ),
                "margin_ratio_std": float(
                    np.std([row["mean_margin_over_threshold"] for row in selected])
                ),
            }
        )

    return summary_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=1000000)
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--min-leaf-depth", type=int, default=3)
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--epsilon", type=float, default=0.15)
    parser.add_argument("--vfdt-delta", type=float, default=1e-3)
    parser.add_argument("--grace-period", type=int, default=50)
    parser.add_argument("--min-samples-to-split", type=int, default=100)
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of worker processes. Use 0 to use all CPU cores.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = run_experiment(
        n_samples=args.n_samples,
        max_depth=args.max_depth,
        min_leaf_depth=args.min_leaf_depth,
        epsilon=args.epsilon,
        n_seeds=args.n_seeds,
        vfdt_delta=args.vfdt_delta,
        grace_period=args.grace_period,
        min_samples_to_split=args.min_samples_to_split,
        n_jobs=args.n_jobs,
    )
    write_rows(rows, CHECKPOINT_RESULTS_PATH)
    summary_rows = build_summary(rows)
    write_rows(summary_rows, SUMMARY_RESULTS_PATH)
    print(f"wrote {len(rows)} rows to {CHECKPOINT_RESULTS_PATH}")
    print(f"wrote {len(summary_rows)} rows to {SUMMARY_RESULTS_PATH}")


if __name__ == "__main__":
    main()
