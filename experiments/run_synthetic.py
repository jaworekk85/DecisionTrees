from __future__ import annotations

from data.synthetic import generate_random_tree_stream
from evaluation.prequential import run_prequential
from impurity.entropy import EntropyImpurity
from impurity.gini import GiniImpurity
from tree.vfdt_like import VFDTLikeTree


def run_once(name: str, impurity) -> None:
    model = VFDTLikeTree(
        n_features=6,
        impurity=impurity,
        delta=1e-3,
        grace_period=50,
        min_samples_to_split=100,
    )

    stream = generate_random_tree_stream(
    n_samples=2000,
    n_features=6,
    max_depth=4,
    split_prob=0.9,
    leaf_prob_min=0.05,
    leaf_prob_max=0.95,
    feature_prob=0.5,
    label_noise=0.0,
    seed=42,
)

    results = run_prequential(model, stream)

    print(f"\n=== {name} ===")
    print(f"accuracy: {results['accuracy']:.4f}")
    print(f"n_samples: {results['n_samples']}")
    print(f"n_split_checks: {results['n_split_checks']}")

    if model.split_log:
        last = model.split_log[-1]
        print("last split check:")
        print(f"  n = {last.n}")
        print(f"  best_feature = {last.best_feature}")
        print(f"  second_best_feature = {last.second_best_feature}")
        print(f"  best_gain = {last.best_gain:.6f}")
        print(f"  second_best_gain = {last.second_best_gain:.6f}")
        print(f"  margin = {last.margin:.6f}")
        print(f"  epsilon = {last.epsilon:.6f}")


def main() -> None:
    run_once("Gini", GiniImpurity())
    run_once("Entropy", EntropyImpurity())


if __name__ == "__main__":
    main()