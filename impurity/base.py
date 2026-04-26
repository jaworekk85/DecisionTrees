from __future__ import annotations

from abc import ABC, abstractmethod


class Impurity(ABC):
    @abstractmethod
    def value(self, p: float) -> float:
        """Return impurity value for class probability p in [0, 1]."""
        raise NotImplementedError

    def second_derivative(self, p: float) -> float:
        """Optional analytical second derivative."""
        raise NotImplementedError("Second derivative not implemented for this impurity.")