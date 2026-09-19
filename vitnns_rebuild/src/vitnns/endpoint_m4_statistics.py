"""Image-first frozen statistical helpers for Endpoint M4."""
from __future__ import annotations

import hashlib
import math
from typing import Iterable

import numpy as np
from scipy.stats import rankdata

ATTENTION_METRICS = (
    "routing_share", "frozen_attention_endpoint_error",
    "frozen_attention_absolute_error", "contracted_gradient_error",
    "contracted_gradient_absolute_error", "routing_value_path_cancellation",
    "routing_value_cosine", "mean_query_saturation", "cls_query_saturation",
    "frozen_attention_jvp_error", "frozen_attention_jvp_absolute_error",
)
RESIDUAL_METRICS = (
    "identity_share", "branch_share", "path_cancellation",
    "identity_branch_cosine", "output_identity_patch_tv",
    "identity_map_cancellation", "branch_map_cancellation",
    "output_map_cancellation",
)
PHASES = {"early": (0, 1, 2, 3), "middle": (4, 5, 6, 7), "late": (8, 9, 10, 11)}


def finite(values: Iterable[float | None]) -> list[float]:
    return [float(value) for value in values if value is not None and math.isfinite(float(value))]


def median(values: Iterable[float | None]) -> float | None:
    values = finite(values)
    return None if not values else float(np.median(np.asarray(values, dtype=np.float64)))


def spearman(left: Iterable[float | None], right: Iterable[float | None]) -> float | None:
    pairs = [(float(a), float(b)) for a, b in zip(left, right)
             if a is not None and b is not None and math.isfinite(float(a)) and math.isfinite(float(b))]
    if len(pairs) < 4:
        return None
    a, b = (np.asarray(value, dtype=np.float64) for value in zip(*pairs))
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def bootstrap_median(values: Iterable[float | None], namespace: str, repetitions: int = 10000) -> list[float] | None:
    array = np.asarray(finite(values), dtype=np.float64)
    if array.size == 0:
        return None
    seed = int.from_bytes(hashlib.sha256(namespace.encode()).digest()[:8], "big")
    rng = np.random.default_rng(seed); estimates = np.empty(repetitions, dtype=np.float64)
    for start in range(0, repetitions, 256):
        count = min(256, repetitions - start)
        indexes = rng.integers(0, array.size, size=(count, array.size))
        estimates[start:start + count] = np.median(array[indexes], axis=1)
    return [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))]


def exact_two_sided_sign(values: Iterable[float | None], tolerance: float = 1e-12) -> dict:
    values = finite(values); positive = sum(value > tolerance for value in values)
    negative = sum(value < -tolerance for value in values); ties = len(values) - positive - negative
    n = positive + negative
    if n == 0:
        p = 1.0
    else:
        smaller = min(positive, negative)
        tail = math.fsum(math.comb(n, k) for k in range(smaller + 1)) / 2 ** n
        p = min(1.0, 2.0 * tail)
    return {"n": len(values), "positive": positive, "negative": negative,
            "ties": ties, "p": float(p)}


def holm(rows: list[dict], alpha: float = 0.05) -> list[dict]:
    ordered = sorted(enumerate(rows), key=lambda item: (float(item[1]["p"]), item[1]["test_id"]))
    adjusted = [1.0] * len(rows); running = 0.0; total = len(rows)
    for rank, (index, row) in enumerate(ordered):
        running = max(running, min(1.0, (total - rank) * float(row["p"])))
        adjusted[index] = running
    return [dict(row, holm_p=adjusted[index], reject=bool(adjusted[index] < alpha)) for index, row in enumerate(rows)]


def enrich_residual(row: dict) -> dict:
    value = dict(row)
    for edge in ("identity", "branch", "output"):
        value[f"{edge}_map_cancellation"] = row[edge]["map_cancellation"]
    return value


def image_values(result: dict) -> dict:
    attention = result["attention"]; residual = [enrich_residual(row) for row in result["residual"]]
    output = {"attention": {}, "attention_relations": {}, "residual": {}}
    for block in range(12):
        heads = [row for row in attention if row["block"] == block]
        for metric in ATTENTION_METRICS:
            output["attention"][(block, metric)] = median(row[metric] for row in heads)
        output["attention_relations"][(block, "saturation_routing")] = spearman(
            (row["mean_query_saturation"] for row in heads), (row["routing_share"] for row in heads))
        output["attention_relations"][(block, "saturation_error")] = spearman(
            (row["mean_query_saturation"] for row in heads),
            (row["frozen_attention_endpoint_error"] for row in heads))
        for residual_type in ("attention", "mlp"):
            row = next(item for item in residual if item["block"] == block and item["residual_type"] == residual_type)
            for metric in RESIDUAL_METRICS:
                output["residual"][(residual_type, block, metric)] = row[metric]
    output["contrasts"] = {}
    for metric in ATTENTION_METRICS:
        phase = {name: median(output["attention"][(block, metric)] for block in blocks) for name, blocks in PHASES.items()}
        output["contrasts"][f"attention_{metric}_late_minus_early"] = (
            None if phase["late"] is None or phase["early"] is None else phase["late"] - phase["early"]
        )
    for relation in ("saturation_routing", "saturation_error"):
        output["contrasts"][relation] = median(output["attention_relations"][(block, relation)] for block in range(12))
    for residual_type in ("attention", "mlp"):
        for metric in RESIDUAL_METRICS:
            phase = {name: median(output["residual"][(residual_type, block, metric)] for block in blocks)
                     for name, blocks in PHASES.items()}
            output["contrasts"][f"{residual_type}_{metric}_late_minus_early"] = (
                None if phase["late"] is None or phase["early"] is None else phase["late"] - phase["early"]
            )
    return output


def summarize(values: list[float | None], namespace: str) -> dict:
    valid = finite(values)
    return {"n": len(valid), "na": len(values) - len(valid), "median": median(valid),
            "ci95": bootstrap_median(valid, namespace)}
