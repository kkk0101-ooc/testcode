"""Source-only post-hoc review diagnostics for Endpoint M4."""
from __future__ import annotations

import hashlib
import math
from typing import Iterable

import numpy as np
from scipy.stats import rankdata


def finite(values: Iterable[float | None]) -> list[float]:
    return [float(value) for value in values if value is not None and math.isfinite(float(value))]


def quantiles(values: Iterable[float | None]) -> dict:
    array = np.asarray(finite(values), dtype=np.float64)
    if not len(array):
        return {key: None for key in ("q10", "q25", "median", "q75", "q90")}
    return {name: float(np.quantile(array, quantile)) for name, quantile in
            (("q10", .10), ("q25", .25), ("median", .50), ("q75", .75), ("q90", .90))}


def bootstrap_median(values: Iterable[float | None], namespace: str, repetitions: int = 10000):
    array = np.asarray(finite(values), dtype=np.float64)
    if not len(array):
        return None
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(namespace.encode()).digest()[:8], "big"))
    estimates = np.empty(repetitions, dtype=np.float64)
    for start in range(0, repetitions, 256):
        count = min(256, repetitions - start)
        indexes = rng.integers(0, len(array), size=(count, len(array)))
        estimates[start:start + count] = np.median(array[indexes], axis=1)
    return [float(np.quantile(estimates, .025)), float(np.quantile(estimates, .975))]


def exact_sign(values: Iterable[float | None], tolerance: float = 1e-12) -> dict:
    array = finite(values); positive = sum(value > tolerance for value in array)
    negative = sum(value < -tolerance for value in array); ties = len(array) - positive - negative
    n = positive + negative
    if n == 0:
        p = 1.0
    else:
        tail = math.fsum(math.comb(n, index) for index in range(min(positive, negative) + 1)) / 2 ** n
        p = min(1.0, 2 * tail)
    return {"n": len(array), "positive": positive, "negative": negative, "ties": ties, "p": float(p)}


def holm(rows: list[dict]) -> list[dict]:
    ordered = sorted(enumerate(rows), key=lambda pair: (pair[1]["p"], pair[1]["test_id"]))
    adjusted = [1.0] * len(rows); running = 0.0
    for rank, (index, row) in enumerate(ordered):
        running = max(running, min(1.0, (len(rows) - rank) * row["p"]))
        adjusted[index] = running
    return [dict(row, holm_p=adjusted[index], reject=adjusted[index] < .05) for index, row in enumerate(rows)]


def spearman(left, right):
    pairs = [(float(a), float(b)) for a, b in zip(left, right)
             if a is not None and b is not None and math.isfinite(float(a)) and math.isfinite(float(b))]
    if len(pairs) < 4:
        return None
    a, b = (np.asarray(value, dtype=np.float64) for value in zip(*pairs))
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def partial_spearman(left, right, controls):
    rows = [(float(a), float(b), *(float(control[index]) for control in controls))
            for index, (a, b) in enumerate(zip(left, right))
            if a is not None and b is not None and all(control[index] is not None for control in controls)]
    if len(rows) < len(controls) + 4:
        return None
    array = np.asarray(rows, dtype=np.float64)
    ranked = np.column_stack([rankdata(array[:, index]) for index in range(array.shape[1])])
    design = np.column_stack([np.ones(len(ranked)), ranked[:, 2:]])
    residuals = []
    for index in (0, 1):
        residuals.append(ranked[:, index] - design @ np.linalg.lstsq(design, ranked[:, index], rcond=None)[0])
    denominator = float(np.linalg.norm(residuals[0]) * np.linalg.norm(residuals[1]))
    return None if denominator == 0 else float(np.dot(residuals[0], residuals[1]) / denominator)


def patch_distribution(array: np.ndarray):
    value = np.asarray(array, dtype=np.float64)
    if value.shape != (3, 224, 224):
        raise ValueError("MAP_SHAPE")
    patch = np.abs(value).reshape(3, 14, 16, 14, 16).sum((0, 2, 4)).reshape(-1)
    total = float(patch.sum())
    return None if total == 0 else patch / total


def entropy(distribution):
    if distribution is None:
        return None
    positive = distribution[distribution > 0]
    return float(-(positive * np.log(positive)).sum() / math.log(len(distribution)))


def total_variation(left, right):
    return None if left is None or right is None else float(.5 * np.abs(left - right).sum())
