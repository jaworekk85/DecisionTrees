from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from impurity.base import Impurity


@dataclass
class SplitScore:
    attribute: int
    gain: float
    weighted_child_impurity: float


class LeafStats:
    """
    Sufficient statistics for a leaf.

    Assumptions in version 1:
    - binary class labels: y in {0, 1}
    - binary attributes: x[a] in {0, 1}
    """

    def __init__(self, n_features: int) -> None:
        self.n_features = n_features
        self.class_counts = np.zeros(2, dtype=np.int64)
        # counts[feature, value, class]
        self.counts = np.zeros((n_features, 2, 2), dtype=np.int64)
        self.n_seen = 0

    def update(self, x: np.ndarray, y: int) -> None:
        y_int = int(y)
        self.class_counts[y_int] += 1
        for a in range(self.n_features):
            v = int(x[a])
            self.counts[a, v, y_int] += 1
        self.n_seen += 1

    def total_count(self) -> int:
        return int(self.n_seen)

    def parent_probability(self) -> float:
        n = self.total_count()
        if n == 0:
            return 0.5
        return float(self.class_counts[1]) / float(n)

    def majority_class(self) -> int:
        return int(self.class_counts[1] >= self.class_counts[0])

    def weighted_child_impurity(self, feature: int, impurity: Impurity) -> float:
        n = self.total_count()
        if n == 0:
            return 0.0

        total = 0.0
        for v in (0, 1):
            child_counts = self.counts[feature, v]
            child_n = int(child_counts.sum())
            if child_n == 0:
                continue
            p_child = float(child_counts[1]) / float(child_n)
            w_child = float(child_n) / float(n)
            total += w_child * impurity.value(p_child)
        return total

    def gain(self, feature: int, impurity: Impurity) -> float:
        p_parent = self.parent_probability()
        parent_impurity = impurity.value(p_parent)
        return parent_impurity - self.weighted_child_impurity(feature, impurity)

    def all_split_scores(self, impurity: Impurity) -> List[SplitScore]:
        scores: List[SplitScore] = []
        for feature in range(self.n_features):
            weighted = self.weighted_child_impurity(feature, impurity)
            gain = impurity.value(self.parent_probability()) - weighted
            scores.append(
                SplitScore(
                    attribute=feature,
                    gain=gain,
                    weighted_child_impurity=weighted,
                )
            )
        return scores