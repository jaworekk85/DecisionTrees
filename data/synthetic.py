from __future__ import annotations

from typing import Iterator, Tuple

import numpy as np


def generate_binary_stream(
    n_samples: int,
    n_features: int = 6,
    signal_strength: float = 0.9,
    seed: int = 42,
) -> Iterator[Tuple[np.ndarray, int]]:
    """
    Minimal synthetic stream.

    Feature 0 is informative:
    - y ~ Bernoulli(0.5)
    - x0 equals y with probability signal_strength
    Other features are noise ~ Bernoulli(0.5)
    """
    rng = np.random.default_rng(seed)

    for _ in range(n_samples):
        y = int(rng.integers(0, 2))
        x = rng.integers(0, 2, size=n_features, dtype=np.int64)

        if rng.random() < signal_strength:
            x[0] = y
        else:
            x[0] = 1 - y

        yield x.astype(np.int64), y