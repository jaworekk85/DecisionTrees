from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from impurity.base import Impurity
from tree.splitter import SplitDecision, choose_split
from tree.stats import LeafStats


class Node:
    def __init__(self, n_features: int) -> None:
        self.is_leaf = True
        self.stats = LeafStats(n_features=n_features)

        self.split_feature: Optional[int] = None
        self.left: Optional[Node] = None   # value 0
        self.right: Optional[Node] = None  # value 1

    def predict(self, x: np.ndarray) -> int:
        if self.is_leaf:
            return self.stats.majority_class()

        assert self.split_feature is not None
        child = self.left if int(x[self.split_feature]) == 0 else self.right
        assert child is not None
        return child.predict(x)

    def update(
        self,
        x: np.ndarray,
        y: int,
        impurity: Impurity,
        delta: float,
        grace_period: int,
        min_samples_to_split: int,
    ) -> Optional[SplitDecision]:
        if not self.is_leaf:
            assert self.split_feature is not None
            child = self.left if int(x[self.split_feature]) == 0 else self.right
            assert child is not None
            return child.update(
                x=x,
                y=y,
                impurity=impurity,
                delta=delta,
                grace_period=grace_period,
                min_samples_to_split=min_samples_to_split,
            )

        self.stats.update(x, y)

        n = self.stats.total_count()
        if n < min_samples_to_split:
            return None
        if n % grace_period != 0:
            return None

        decision = choose_split(self.stats, impurity, delta)

        if decision.should_split and decision.best_feature is not None:
            self._split(decision.best_feature)

        return decision

    def _split(self, feature: int) -> None:
        n_features = self.stats.n_features
        self.is_leaf = False
        self.split_feature = feature
        self.left = Node(n_features=n_features)
        self.right = Node(n_features=n_features)