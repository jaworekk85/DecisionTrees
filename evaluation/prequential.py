from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np

from evaluation.metrics import accuracy
from tree.vfdt_like import VFDTLikeTree


def run_prequential(
    model: VFDTLikeTree,
    stream: Iterable[Tuple[np.ndarray, int]],
) -> Dict[str, float]:
    correct = 0
    total = 0

    for x, y in stream:
        y_pred = model.predict_one(x)
        if y_pred == y:
            correct += 1
        total += 1
        model.partial_fit_one(x, y)

    return {
        "accuracy": accuracy(correct, total),
        "n_samples": total,
        "n_split_checks": len(model.split_log),
    }