from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from impurity.base import Impurity
from tree.stats import LeafStats, SplitScore


@dataclass
class SplitDecision:
    should_split: bool
    best_feature: Optional[int]
    second_best_feature: Optional[int]
    best_gain: float
    second_best_gain: float
    margin: float
    epsilon: float
    n: int


def hoeffding_epsilon(delta: float, n: int) -> float:
    if n <= 0:
        return float("inf")
    return math.sqrt(math.log(1.0 / delta) / (2.0 * n))


def choose_split(
    stats: LeafStats,
    impurity: Impurity,
    delta: float,
) -> SplitDecision:
    scores = stats.all_split_scores(impurity)
    scores = sorted(scores, key=lambda s: s.gain, reverse=True)

    n = stats.total_count()
    if len(scores) == 0:
        return SplitDecision(False, None, None, 0.0, 0.0, 0.0, float("inf"), n)

    best = scores[0]
    second = scores[1] if len(scores) > 1 else SplitScore(-1, 0.0, 0.0)

    margin = best.gain - second.gain
    epsilon = hoeffding_epsilon(delta=delta, n=n)
    should_split = margin > epsilon

    return SplitDecision(
        should_split=should_split,
        best_feature=best.attribute,
        second_best_feature=(None if second.attribute == -1 else second.attribute),
        best_gain=best.gain,
        second_best_gain=second.gain,
        margin=margin,
        epsilon=epsilon,
        n=n,
    )