from __future__ import annotations

from impurity.base import Impurity


class GiniImpurity(Impurity):
    """
    Normalized Gini:
        g(p) = 4 p (1-p)
    so that g(1/2) = 1.
    """

    def value(self, p: float) -> float:
        p = min(max(p, 0.0), 1.0)
        return 4.0 * p * (1.0 - p)

    def second_derivative(self, p: float) -> float:
        return -8.0