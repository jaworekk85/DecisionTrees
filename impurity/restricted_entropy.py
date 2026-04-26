from __future__ import annotations

import math
from typing import Iterable

from impurity.base import Impurity


_LOG_2 = math.log(2.0)


def _validate_epsilon(epsilon: float) -> None:
    if not 0.0 < epsilon < 0.5:
        raise ValueError(f"epsilon must be in (0, 0.5), got {epsilon}")


def _clip_probability(p: float) -> float:
    return min(max(float(p), 0.0), 1.0)


def _entropy(p: float) -> float:
    p = _clip_probability(p)
    if p == 0.0 or p == 1.0:
        return 0.0
    return (-(p * math.log(p) + (1.0 - p) * math.log(1.0 - p))) / _LOG_2


def _entropy_first_derivative(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        raise ValueError("Entropy derivative is singular at p=0 and p=1.")
    return math.log((1.0 - p) / p) / _LOG_2


def _entropy_second_derivative(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        raise ValueError("Entropy second derivative is singular at p=0 and p=1.")
    return -(1.0 / _LOG_2) * (1.0 / (p * (1.0 - p)))


class _RestrictedEntropyBase(Impurity):
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


class GloballyConcaveRestrictedEntropy(_RestrictedEntropyBase):
    """
    Globally concave restricted-domain entropy impurity.

    The central interval keeps the entropy shape, shifted and rescaled so that
    ``g(1/2)=1``. Near the boundaries it is connected linearly to preserve
    global concavity and ``g(0)=g(1)=0``.
    """

    def __init__(self, epsilon: float = 0.15) -> None:
        super().__init__(epsilon=epsilon)
        h_epsilon = _entropy(self.epsilon)
        h_prime_epsilon = _entropy_first_derivative(self.epsilon)
        self.scale = 1.0 + self.epsilon * h_prime_epsilon - h_epsilon
        self.boundary_slope = h_prime_epsilon / self.scale
        self.central_curvature = (4.0 / _LOG_2) / self.scale

    def value(self, p: float) -> float:
        q = self._half_probability(p)
        if q <= self.epsilon:
            return self.boundary_slope * q

        h_epsilon = _entropy(self.epsilon)
        h_prime_epsilon = _entropy_first_derivative(self.epsilon)
        return (_entropy(q) + self.epsilon * h_prime_epsilon - h_epsilon) / self.scale

    def second_derivative(self, p: float) -> float:
        p = _clip_probability(p)
        if self.epsilon < p < 1.0 - self.epsilon:
            return _entropy_second_derivative(p) / self.scale
        return 0.0


class LocallyConcaveRestrictedEntropy(_RestrictedEntropyBase):
    """
    Locally concave restricted-domain entropy impurity.

    The function is flat outside ``[epsilon, 1 - epsilon]`` and uses a rescaled
    entropy profile inside the operational interval.
    """

    def __init__(self, epsilon: float = 0.15) -> None:
        super().__init__(epsilon=epsilon)
        self.scale = 1.0 - _entropy(self.epsilon)
        self.central_curvature = (4.0 / _LOG_2) / self.scale

    def value(self, p: float) -> float:
        q = self._half_probability(p)
        if q <= self.epsilon:
            return 0.0
        return (_entropy(q) - _entropy(self.epsilon)) / self.scale

    def second_derivative(self, p: float) -> float:
        p = _clip_probability(p)
        if self.epsilon < p < 1.0 - self.epsilon:
            return _entropy_second_derivative(p) / self.scale
        return 0.0


RestrictedEntropyGlobal = GloballyConcaveRestrictedEntropy
RestrictedEntropyLocal = LocallyConcaveRestrictedEntropy
RestrictedEntropy = GloballyConcaveRestrictedEntropy
