from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from impurity.base import Impurity
from tree.node import Node
from tree.splitter import SplitDecision


class VFDTLikeTree:
    def __init__(
        self,
        n_features: int,
        impurity: Impurity,
        delta: float = 1e-3,
        grace_period: int = 50,
        min_samples_to_split: int = 100,
    ) -> None:
        self.n_features = n_features
        self.impurity = impurity
        self.delta = delta
        self.grace_period = grace_period
        self.min_samples_to_split = min_samples_to_split

        self.root = Node(n_features=n_features)
        self.split_log: List[SplitDecision] = []

    def predict_one(self, x: np.ndarray) -> int:
        return self.root.predict(x)

    def partial_fit_one(self, x: np.ndarray, y: int) -> Optional[SplitDecision]:
        decision = self.root.update(
            x=x,
            y=y,
            impurity=self.impurity,
            delta=self.delta,
            grace_period=self.grace_period,
            min_samples_to_split=self.min_samples_to_split,
        )
        if decision is not None:
            self.split_log.append(decision)
        return decision