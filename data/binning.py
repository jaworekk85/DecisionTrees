from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class BinnedFeature:
    source_feature: int
    threshold: float


class FixedThresholdBinarizer:
    """
    Convert numeric attributes into binary threshold indicators.

    Each numeric feature is expanded into indicators of the form
    ``x[feature] > threshold``. The current tree can consume the result because
    every output feature is binary.
    """

    def __init__(
        self,
        n_bins: int = 4,
        strategy: str = "quantile",
        thresholds: Optional[Sequence[Sequence[float]]] = None,
    ) -> None:
        if n_bins < 2:
            raise ValueError(f"n_bins must be at least 2, got {n_bins}")
        if strategy not in {"quantile", "uniform"}:
            raise ValueError(
                f"strategy must be 'quantile' or 'uniform', got {strategy!r}"
            )

        self.n_bins = n_bins
        self.strategy = strategy
        self._provided_thresholds = thresholds
        self.thresholds_: Optional[List[np.ndarray]] = None
        self.features_: Optional[List[BinnedFeature]] = None

    def fit(self, X: np.ndarray) -> "FixedThresholdBinarizer":
        X_array = self._as_2d_float_array(X)

        if self._provided_thresholds is not None:
            thresholds = [
                np.asarray(feature_thresholds, dtype=float)
                for feature_thresholds in self._provided_thresholds
            ]
            if len(thresholds) != X_array.shape[1]:
                raise ValueError(
                    "thresholds must contain one threshold list per input feature"
                )
        elif self.strategy == "quantile":
            thresholds = self._quantile_thresholds(X_array)
        else:
            thresholds = self._uniform_thresholds(X_array)

        self.thresholds_ = [np.unique(values) for values in thresholds]
        self.features_ = [
            BinnedFeature(source_feature=feature, threshold=float(threshold))
            for feature, feature_thresholds in enumerate(self.thresholds_)
            for threshold in feature_thresholds
        ]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X_array = self._as_2d_float_array(X)
        self._check_is_fitted()

        assert self.features_ is not None
        transformed = np.zeros((X_array.shape[0], len(self.features_)), dtype=np.int64)

        for output_feature, feature in enumerate(self.features_):
            transformed[:, output_feature] = (
                X_array[:, feature.source_feature] > feature.threshold
            ).astype(np.int64)

        return transformed

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def transform_one(self, x: np.ndarray) -> np.ndarray:
        x_array = np.asarray(x, dtype=float)
        if x_array.ndim != 1:
            raise ValueError(f"x must be one-dimensional, got shape {x_array.shape}")
        return self.transform(x_array.reshape(1, -1))[0]

    @property
    def n_output_features(self) -> int:
        self._check_is_fitted()
        assert self.features_ is not None
        return len(self.features_)

    def describe_features(self) -> List[BinnedFeature]:
        self._check_is_fitted()
        assert self.features_ is not None
        return list(self.features_)

    def _quantile_thresholds(self, X: np.ndarray) -> List[np.ndarray]:
        quantiles = np.linspace(0.0, 1.0, self.n_bins + 1)[1:-1]
        return [
            np.quantile(X[:, feature], quantiles)
            for feature in range(X.shape[1])
        ]

    def _uniform_thresholds(self, X: np.ndarray) -> List[np.ndarray]:
        thresholds = []
        for feature in range(X.shape[1]):
            feature_min = float(np.min(X[:, feature]))
            feature_max = float(np.max(X[:, feature]))
            if feature_min == feature_max:
                thresholds.append(np.array([], dtype=float))
                continue
            thresholds.append(
                np.linspace(feature_min, feature_max, self.n_bins + 1)[1:-1]
            )
        return thresholds

    def _check_is_fitted(self) -> None:
        if self.thresholds_ is None or self.features_ is None:
            raise RuntimeError("FixedThresholdBinarizer must be fitted first")

    @staticmethod
    def _as_2d_float_array(X: np.ndarray) -> np.ndarray:
        X_array = np.asarray(X, dtype=float)
        if X_array.ndim != 2:
            raise ValueError(f"X must be two-dimensional, got shape {X_array.shape}")
        return X_array


def binarize_stream(
    stream: Iterable[Tuple[np.ndarray, int]],
    binarizer: FixedThresholdBinarizer,
) -> Iterator[Tuple[np.ndarray, int]]:
    for x, y in stream:
        yield binarizer.transform_one(x), y
