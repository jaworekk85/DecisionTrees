from __future__ import annotations

import math

from impurity.base import Impurity


class EntropyImpurity(Impurity):
    """
    Normalized binary entropy:
        g(p) = [ -p log p - (1-p) log(1-p) ] / log(2)
    so that g(1/2) = 1.
    """

    def value(self, p: float) -> float:
        p = min(max(p, 0.0), 1.0)
        if p == 0.0 or p == 1.0:
            return 0.0
        return (-(p * math.log(p) + (1.0 - p) * math.log(1.0 - p))) / math.log(2.0)

    def second_derivative(self, p: float) -> float:
        if p <= 0.0 or p >= 1.0:
            raise ValueError("Entropy second derivative is singular at p=0 and p=1.")
        return -(1.0 / math.log(2.0)) * (1.0 / (p * (1.0 - p)))