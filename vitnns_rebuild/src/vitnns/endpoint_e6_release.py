"""Frozen CPU-only statistics for the ENDPOINT-E6 release.

This module operates on image-level values. It imports neither model nor image
libraries and contains no empirical inference path.
"""
from __future__ import annotations

import hashlib
import math
from typing import Any, Mapping, Sequence

import numpy as np


class EndpointE6Error(ValueError):
    """The frozen E6 statistical contract was violated."""


def require(condition: bool, code: str) -> None:
    if not condition:
        raise EndpointE6Error(code)


def average_ranks(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    require(array.ndim == 1 and array.size > 0 and np.isfinite(array).all(), "RANK_INPUT")
    order = np.argsort(array, kind="stable")
    ranks = np.empty(array.size, dtype=np.float64)
    start = 0
    while start < array.size:
        stop = start + 1
        while stop < array.size and array[order[stop]] == array[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.0
        start = stop
    return ranks


def spearman(left: Sequence[float], right: Sequence[float], *, min_components: int = 8) -> dict[str, Any]:
    x = np.asarray(left, dtype=np.float64); y = np.asarray(right, dtype=np.float64)
    require(x.ndim == y.ndim == 1 and x.size == y.size, "SPEARMAN_SHAPE")
    require(x.size >= min_components, "SPEARMAN_COMPONENT_COUNT")
    require(np.isfinite(x).all() and np.isfinite(y).all(), "SPEARMAN_NONFINITE_RAW")
    rx = average_ranks(x); ry = average_ranks(y)
    rx -= rx.mean(); ry -= ry.mean()
    denominator = float(np.sqrt(np.dot(rx, rx) * np.dot(ry, ry)))
    if denominator == 0.0:
        return {"value": None, "n": int(x.size), "reason": "NA_CONSTANT_VECTOR"}
    return {"value": float(np.dot(rx, ry) / denominator), "n": int(x.size), "reason": None}


def normalized_auc(effects: Sequence[float]) -> float:
    values = np.abs(np.asarray(effects, dtype=np.float64))
    require(values.shape == (4,) and np.isfinite(values).all(), "AUC_INPUT")
    areas = np.asarray((0.0, 0.0625, 0.125, 0.25), dtype=np.float64)
    return float(np.sum(np.diff(areas) * (values[:-1] + values[1:]) / 2.0) / 0.25)


def bootstrap_seed(namespace: str, family_id: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{namespace}|bootstrap|{family_id}".encode()).digest()[:8], "big")


def bootstrap_median_ci(
    columns: Mapping[str, Sequence[float]], *, namespace: str, family_id: str, replicates: int = 10000,
) -> dict[str, Any]:
    require(columns and replicates == 10000, "BOOTSTRAP_CONTRACT")
    arrays = {key: np.asarray(value, dtype=np.float64) for key, value in columns.items()}
    sizes = {value.size for value in arrays.values()}
    require(len(sizes) == 1 and next(iter(sizes)) > 0 and all(value.ndim == 1 and np.isfinite(value).all() for value in arrays.values()), "BOOTSTRAP_INPUT")
    n = next(iter(sizes)); seed = bootstrap_seed(namespace, family_id)
    generator = np.random.Generator(np.random.PCG64(seed))
    medians = {key: np.empty(replicates, dtype=np.float64) for key in arrays}
    chunk = 500
    for start in range(0, replicates, chunk):
        stop = min(replicates, start + chunk)
        indices = generator.integers(0, n, size=(stop - start, n), endpoint=False)
        for key, value in arrays.items():
            medians[key][start:stop] = np.median(value[indices], axis=1)
    return {
        "seed": seed, "replicates": replicates, "n": n,
        "results": {key: {"estimate": float(np.median(value)),
                            "ci_lower": float(np.quantile(medians[key], 0.025, method="linear")),
                            "ci_upper": float(np.quantile(medians[key], 0.975, method="linear"))}
                    for key, value in arrays.items()},
    }


def exact_two_sided_sign(values: Sequence[float], *, zero_threshold: float = 1e-12) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    require(array.ndim == 1 and np.isfinite(array).all(), "SIGN_INPUT")
    positive = int(np.sum(array > zero_threshold)); negative = int(np.sum(array < -zero_threshold))
    ties = int(array.size - positive - negative); nonzero = positive + negative
    if nonzero == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(nonzero, index) for index in range(min(positive, negative) + 1))
        p_value = min(1.0, float(2 * tail / (2 ** nonzero)))
    return {"n_total": int(array.size), "n_nonzero": nonzero, "n_positive": positive,
            "n_negative": negative, "n_tie": ties, "raw_p_value": p_value}


def holm_adjust(raw: Mapping[str, float], frozen_order: Sequence[str], *, alpha: float = 0.05) -> list[dict[str, Any]]:
    require(set(raw) == set(frozen_order) and len(raw) == len(frozen_order) == 28, "HOLM_FAMILY")
    position = {test_id: index for index, test_id in enumerate(frozen_order)}
    ordered = sorted(raw, key=lambda test_id: (float(raw[test_id]), test_id))
    running = 0.0; rows = []
    for index, test_id in enumerate(ordered):
        value = float(raw[test_id]); require(math.isfinite(value) and 0 <= value <= 1, "HOLM_P_VALUE")
        adjusted = max(running, min(1.0, (len(ordered) - index) * value)); running = adjusted
        rows.append({"test_id": test_id, "frozen_order": position[test_id], "holm_rank": index,
                     "raw_p_value": value, "adjusted_p_value": adjusted,
                     "reject_alpha_0_05": adjusted <= alpha})
    return rows


def decide_family(family: Mapping[str, Any], tests: Sequence[Mapping[str, Any]], *, invalid_reasons: Sequence[str] = ()) -> tuple[str, list[str]]:
    if invalid_reasons:
        return "INVALID", list(invalid_reasons)
    reasons = []
    if not family.get("exact_coverage", False): reasons.append("EXACT_COVERAGE_INCOMPLETE")
    if int(family["n_valid"]) < int(family["n_valid_min"]): reasons.append("N_VALID_BELOW_MIN")
    if any(int(test["n_nonzero"]) < int(test["min_nonzero_n"]) for test in tests): reasons.append("N_NONZERO_BELOW_MIN")
    if any(test.get("estimate") is None or test.get("ci_lower") is None or test.get("ci_upper") is None for test in tests): reasons.append("STATISTIC_MISSING")
    if any(test.get("ci_width") is not None and float(test["ci_width"]) > float(family["ci_width_max"]) for test in tests): reasons.append("CI_WIDTH_ABOVE_MAX")
    if reasons:
        return "INCONCLUSIVE", reasons
    support = all(float(test["estimate"]) >= float(family["delta_min"]) and float(test["estimate"]) > 0 and bool(test["holm_reject"]) for test in tests)
    return ("SUPPORTED", ["ALL_REGISTERED_SUPPORT_CONDITIONS_MET"]) if support else ("NOT_SUPPORTED", ["ONE_OR_MORE_REGISTERED_SUPPORT_CONDITIONS_FAILED"])


def aggregate_decision(statuses: Sequence[str]) -> str:
    require(statuses, "AGGREGATE_EMPTY")
    if any(value == "INVALID" for value in statuses): return "INVALID"
    if any(value == "INCONCLUSIVE" for value in statuses): return "INCONCLUSIVE"
    if all(value == "SUPPORTED" for value in statuses): return "SUPPORTED"
    return "NOT_SUPPORTED"
