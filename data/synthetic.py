from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class _RandomTreeNode:
    feature: Optional[int] = None
    prob_class_one: Optional[float] = None
    left: Optional["_RandomTreeNode"] = None
    right: Optional["_RandomTreeNode"] = None

    @property
    def is_leaf(self) -> bool:
        return self.feature is None


def _validate_probability(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}")


def _make_random_tree(
    rng: np.random.Generator,
    available_features: Sequence[int],
    depth: int,
    max_depth: int,
    min_leaf_depth: int,
    split_prob: float,
    leaf_prob_min: float,
    leaf_prob_max: float,
) -> _RandomTreeNode:
    must_stop = depth >= max_depth or len(available_features) == 0
    must_split = depth < min_leaf_depth and len(available_features) > 0
    should_split = must_split or ((not must_stop) and rng.random() < split_prob)

    if not should_split:
        prob_class_one = float(rng.uniform(leaf_prob_min, leaf_prob_max))
        return _RandomTreeNode(prob_class_one=prob_class_one)

    feature_index = int(rng.integers(0, len(available_features)))
    feature = int(available_features[feature_index])
    child_features = (
        tuple(available_features[:feature_index])
        + tuple(available_features[feature_index + 1 :])
    )

    return _RandomTreeNode(
        feature=feature,
        left=_make_random_tree(
            rng=rng,
            available_features=child_features,
            depth=depth + 1,
            max_depth=max_depth,
            min_leaf_depth=min_leaf_depth,
            split_prob=split_prob,
            leaf_prob_min=leaf_prob_min,
            leaf_prob_max=leaf_prob_max,
        ),
        right=_make_random_tree(
            rng=rng,
            available_features=child_features,
            depth=depth + 1,
            max_depth=max_depth,
            min_leaf_depth=min_leaf_depth,
            split_prob=split_prob,
            leaf_prob_min=leaf_prob_min,
            leaf_prob_max=leaf_prob_max,
        ),
    )


def _leaf_probability(root: _RandomTreeNode, x: np.ndarray) -> float:
    node = root
    while not node.is_leaf:
        assert node.feature is not None
        node = node.left if int(x[node.feature]) == 0 else node.right
        assert node is not None

    assert node.prob_class_one is not None
    return node.prob_class_one


def generate_random_tree_stream(
    n_samples: int,
    n_features: int = 6,
    max_depth: int = 4,
    split_prob: float = 0.9,
    omega: Optional[float] = None,
    min_leaf_depth: int = 0,
    leaf_prob_min: float = 0.05,
    leaf_prob_max: float = 0.95,
    feature_prob: float = 0.5,
    label_noise: float = 0.0,
    seed: int = 42,
) -> Iterator[Tuple[np.ndarray, int]]:
    """
    Generate a binary stream from a hidden random decision tree.

    This follows the random-tree stream idea used in VFDT-style work:
    first sample a target tree, then draw examples from that fixed target.
    Internal nodes test binary features, leaves store class probabilities,
    and optional label noise flips the sampled label.
    """
    if n_samples < 0:
        raise ValueError(f"n_samples must be non-negative, got {n_samples}")
    if n_features <= 0:
        raise ValueError(f"n_features must be positive, got {n_features}")
    if max_depth < 0:
        raise ValueError(f"max_depth must be non-negative, got {max_depth}")
    if min_leaf_depth < 0:
        raise ValueError(f"min_leaf_depth must be non-negative, got {min_leaf_depth}")
    if min_leaf_depth > max_depth:
        raise ValueError(
            f"min_leaf_depth must not exceed max_depth, got {min_leaf_depth} > {max_depth}"
        )
    if omega is not None:
        _validate_probability("omega", omega)
        split_prob = 1.0 - omega
    _validate_probability("split_prob", split_prob)
    _validate_probability("leaf_prob_min", leaf_prob_min)
    _validate_probability("leaf_prob_max", leaf_prob_max)
    _validate_probability("feature_prob", feature_prob)
    _validate_probability("label_noise", label_noise)
    if leaf_prob_min > leaf_prob_max:
        raise ValueError(
            "leaf_prob_min must be less than or equal to leaf_prob_max, "
            f"got {leaf_prob_min} > {leaf_prob_max}"
        )

    tree_rng = np.random.default_rng(seed)
    sample_rng = np.random.default_rng(seed + 1)

    root = _make_random_tree(
        rng=tree_rng,
        available_features=tuple(range(n_features)),
        depth=0,
        max_depth=max_depth,
        min_leaf_depth=min_leaf_depth,
        split_prob=split_prob,
        leaf_prob_min=leaf_prob_min,
        leaf_prob_max=leaf_prob_max,
    )

    for _ in range(n_samples):
        x = sample_rng.binomial(1, feature_prob, size=n_features).astype(np.int64)
        prob_class_one = _leaf_probability(root, x)
        y = int(sample_rng.random() < prob_class_one)

        if sample_rng.random() < label_noise:
            y = 1 - y

        yield x, y


def generate_binary_stream(
    n_samples: int,
    n_features: int = 6,
    signal_strength: float = 0.9,
    seed: int = 42,
) -> Iterator[Tuple[np.ndarray, int]]:
    """
    Minimal synthetic stream.

    Feature 0 is informative:
    - y ~ Bernoulli(0.5)
    - x0 equals y with probability signal_strength
    Other features are noise ~ Bernoulli(0.5)
    """
    rng = np.random.default_rng(seed)

    for _ in range(n_samples):
        y = int(rng.integers(0, 2))
        x = rng.integers(0, 2, size=n_features, dtype=np.int64)

        if rng.random() < signal_strength:
            x[0] = y
        else:
            x[0] = 1 - y

        yield x.astype(np.int64), y
