from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from . import TRAITS


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def regression_metrics(
    labels: Sequence[Sequence[float]], predictions: Sequence[Sequence[float]]
) -> dict[str, Any]:
    truth = np.asarray(labels, dtype=np.float64)
    predicted = np.asarray(predictions, dtype=np.float64)
    if truth.shape != predicted.shape or truth.ndim != 2 or truth.shape[1] != 3:
        raise ValueError("labels/predictions shape은 [N,3]이어야 합니다")
    traits: dict[str, dict[str, float | None]] = {}
    for index, trait in enumerate(TRAITS):
        left = truth[:, index]
        right = predicted[:, index]
        rmse = float(np.sqrt(np.mean((left - right) ** 2)))
        rho: float | None = None
        if len(set(left.tolist())) > 1 and len(set(right.tolist())) > 1:
            value = float(np.corrcoef(_average_ranks(left), _average_ranks(right))[0, 1])
            rho = None if math.isnan(value) else value
        traits[trait] = {"rmse": rmse, "spearman": rho}
    available = [item["spearman"] for item in traits.values() if item["spearman"] is not None]
    return {
        "count": len(truth),
        "traits": traits,
        "overall": {
            "rmse": float(np.mean([traits[trait]["rmse"] for trait in TRAITS])),
            "spearman": float(np.mean(available)) if available else None,
        },
    }

