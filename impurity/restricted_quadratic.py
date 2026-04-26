from __future__ import annotations

from typing import Iterable

from impurity.base import Impurity


def _validate_epsilon(epsilon: float) -> None:
    if not 0.0 < epsilon < 0.5:
        raise ValueError(f"epsilon must be in (0, 0.5), got {epsilon}")


def _clip_probability(p: float) -> float:
    return min(max(float(p), 0.0), 1.0)


class _RestrictedQuadraticBase(Impurity):
    def __init__(self, epsilon: float = 0.15) -> None:
        _validate_epsilon(epsilon)
        self.epsilon = float(epsilon)

    def calculate(self, labels: Iterable[int]) -> float:
        labels_list = list(labels)
        if len(labels_list) == 0:
            return 0.0
        p = sum(int(label) for label in labels_list) / len(labels_list)
        return self.value(p)

    def curvature(self, p: float) -> float:
        return -self.second_derivative(p)

    def _half_probability(self, p: float) -> float:
        p = _clip_probability(p)
        return min(p, 1.0 - p)


class GloballyConcaveRestrictedQuadratic(_RestrictedQuadraticBase):
    """
    Globally concave restricted-domain quadratic impurity.

    On the central interval ``[epsilon, 1 - epsilon]`` this construction has
    constant curvature

        8 / (1 - 4 epsilon^2).

    Near the boundaries it is extended linearly to preserve global concavity
    and the standard normalization ``g(0)=g(1)=0``, ``g(1/2)=1``.
    """

    def __init__(self, epsilon: float = 0.15) -> None:
        super().__init__(epsilon=epsilon)
        self.central_curvature = 8.0 / (1.0 - 4.0 * self.epsilon**2)

    def value(self, p: float) -> float:
        q = self._half_probability(p)
        if q <= self.epsilon:
            return self.central_curvature * (0.5 - self.epsilon) * q
        return 1.0 - 0.5 * self.central_curvature * (0.5 - q) ** 2

    def second_derivative(self, p: float) -> float:
        p = _clip_probability(p)
        if self.epsilon < p < 1.0 - self.epsilon:
            return -self.central_curvature
        return 0.0


class LocallyConcaveRestrictedQuadratic(_RestrictedQuadraticBase):
    """
    Locally concave restricted-domain quadratic impurity.

    The function is flat outside ``[epsilon, 1 - epsilon]`` and quadratic
    inside the operational interval, where it has constant curvature

        8 / (1 - 2 epsilon)^2.
    """

    def __init__(self, epsilon: float = 0.15) -> None:
        super().__init__(epsilon=epsilon)
        self.central_curvature = 8.0 / (1.0 - 2.0 * self.epsilon) ** 2

    def value(self, p: float) -> float:
        q = self._half_probability(p)
        if q <= self.epsilon:
            return 0.0
        return 1.0 - 0.5 * self.central_curvature * (0.5 - q) ** 2

    def second_derivative(self, p: float) -> float:
        p = _clip_probability(p)
        if self.epsilon < p < 1.0 - self.epsilon:
            return -self.central_curvature
        return 0.0


RestrictedQuadraticGlobal = GloballyConcaveRestrictedQuadratic
RestrictedQuadraticLocal = LocallyConcaveRestrictedQuadratic
RestrictedQuadratic = GloballyConcaveRestrictedQuadratic
