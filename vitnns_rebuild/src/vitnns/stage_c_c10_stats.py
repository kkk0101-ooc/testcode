"""Pure statistical and decision helpers for Stage C/C10.

This module deliberately has no model, image, GPU, or filesystem entry points.
It implements only the frozen image-level statistical contract and the final
decision truth table.  Inputs are never filtered or imputed: an invalid value
is a contract error, not a reason to silently change the sample size.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from typing import Any

import numpy as np


BOOTSTRAP_BASE_SEED = 20260821
BOOTSTRAP_NAMESPACE = f"stage-c-bootstrap|{BOOTSTRAP_BASE_SEED}|"
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_CLUSTER_SIZE = 15
BOOTSTRAP_QUANTILE_METHOD = "linear"

EFFECT_MEDIAN_MINIMUM = 0.02
EFFECT_POSITIVE_COUNT_MINIMUM = 12
EFFECT_CI_LOWER_EXCLUSIVE = 0.0

SPATIAL_SAMPLE_COUNT = 15
SPATIAL_JOINT_COUNT_MINIMUM = 12
PATCH_SUPPORT_MINIMUM = 0.01
POSITIVE_PATCH_EVIDENCE_MINIMUM = 0.01
PATCH_RESIDUAL_L1_RATIO_MINIMUM = 0.05

STABILITY_POSITIVE_COUNT_MINIMUM = 12
STABILITY_MEDIAN_MINIMUM = 0.50
STABILITY_CI_LOWER_EXCLUSIVE = 0.0
STABILITY_PERTURBATIONS = (
    "translate_right_down_4",
    "center_crop_216_resize_224",
    "gaussian_rgb_std_0_01",
)
EXPECTED_EXPLORATORY_WINNERS = ((4, 2), (6, 10), (11, 2))

_SCOPE_SIZES = {"calibration": 5, "holdout": 15, "all_20": 20}
_VALIDITY_STATUSES = {
    "PASS": "PASS_VALIDITY",
    "PASS_VALIDITY": "PASS_VALIDITY",
    "FAIL": "FAIL_VALIDITY",
    "FAIL_VALIDITY": "FAIL_VALIDITY",
}
_EFFECT_STATUSES = frozenset(
    {"PASS_EFFECT", "FAIL_EFFECT", "UNCERTAIN_EFFECT"}
)
_SPATIAL_STATUSES = frozenset(
    {"PASS_SPATIAL", "FAIL_SPATIAL", "UNCERTAIN_SPATIAL"}
)
_COST_STATUSES = {
    "PASS": "PASS_COST",
    "PASS_COST": "PASS_COST",
    "FAIL": "FAIL_COST",
    "FAIL_COST": "FAIL_COST",
}


def _nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value.strip():
        raise ValueError(f"{name} must be non-empty")
    return value


def _identifier_string(value: Any, name: str) -> str:
    identifier = _nonempty_string(value, name)
    if identifier != identifier.strip():
        raise ValueError(f"{name} must not contain leading or trailing whitespace")
    return identifier


def _finite_real(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number (not bool)")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _integer(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer (not bool)")
    return int(value)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _materialize_values(
    values: Any, *, expected_size: int, name: str = "values"
) -> np.ndarray:
    if isinstance(values, (str, bytes, bytearray, Mapping)):
        raise TypeError(f"{name} must be a one-dimensional iterable of numbers")
    try:
        materialized = list(values)
    except TypeError as exc:
        raise TypeError(
            f"{name} must be a one-dimensional iterable of numbers"
        ) from exc
    if len(materialized) != expected_size:
        raise ValueError(
            f"{name} must contain exactly {expected_size} image-cluster values; "
            f"got {len(materialized)}"
        )
    checked = [
        _finite_real(value, f"{name}[{index}]")
        for index, value in enumerate(materialized)
    ]
    return np.asarray(checked, dtype=np.float64)


def _sample_id_list(sample_ids: Any, expected_size: int) -> list[str] | None:
    if sample_ids is None:
        return None
    if isinstance(sample_ids, (str, bytes, bytearray, Mapping)):
        raise TypeError("sample_ids must be an iterable of strings or None")
    try:
        identifiers = list(sample_ids)
    except TypeError as exc:
        raise TypeError(
            "sample_ids must be an iterable of strings or None"
        ) from exc
    if len(identifiers) != expected_size:
        raise ValueError(
            f"sample_ids must contain exactly {expected_size} items; "
            f"got {len(identifiers)}"
        )
    checked = [
        _identifier_string(value, f"sample_ids[{index}]")
        for index, value in enumerate(identifiers)
    ]
    if len(set(checked)) != len(checked):
        raise ValueError("sample_ids must be unique image-cluster identifiers")
    return checked


def metric_seed(metric_id: str) -> int:
    """Return the frozen deterministic bootstrap seed for ``metric_id``.

    The SHA-256 input is the UTF-8 encoding of
    ``stage-c-bootstrap|20260821|<metric_id>``.  The first eight digest bytes
    are interpreted as an unsigned big-endian integer.
    """

    identifier = _identifier_string(metric_id, "metric_id")
    digest = hashlib.sha256((BOOTSTRAP_NAMESPACE + identifier).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def bootstrap_seed(metric_id: str) -> int:
    """Compatibility name for :func:`metric_seed`."""

    return metric_seed(metric_id)


def ordered_sample_ids_sha256(sample_ids: Any) -> str:
    """Hash one ordered image-cluster identity vector canonically."""

    if isinstance(sample_ids, (str, bytes, bytearray, Mapping)):
        raise TypeError("sample_ids must be an iterable of strings")
    try:
        materialized = list(sample_ids)
    except TypeError as exc:
        raise TypeError("sample_ids must be an iterable of strings") from exc
    identifiers = _sample_id_list(materialized, len(materialized))
    assert identifiers is not None
    return hashlib.sha256(
        json.dumps(
            identifiers, sort_keys=False, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _distribution_canonical_bytes(distribution: np.ndarray) -> bytes:
    """Encode a bootstrap distribution as canonical little-endian float64."""

    canonical = np.ascontiguousarray(distribution, dtype=np.dtype("<f8"))
    if canonical.ndim != 1:
        raise ValueError("bootstrap distribution must be one-dimensional")
    return canonical.tobytes(order="C")


def paired_median_bootstrap(
    metric_id: str,
    values: Any,
    *,
    sample_ids: Any = None,
    resamples: int = BOOTSTRAP_RESAMPLES,
    cluster_size: int = BOOTSTRAP_CLUSTER_SIZE,
    return_distribution: bool = False,
) -> dict[str, Any]:
    """Compute the frozen 15-image paired-median percentile bootstrap.

    A single integer draw matrix is generated with ``Generator(PCG64(seed))``.
    Consequently all values that belong to the same paired comparison can use
    the same ``metric_id`` and receive exactly the same image-index draws.
    """

    identifier = _identifier_string(metric_id, "metric_id")
    checked_resamples = _integer(resamples, "resamples")
    checked_cluster_size = _integer(cluster_size, "cluster_size")
    if checked_resamples != BOOTSTRAP_RESAMPLES:
        raise ValueError(
            f"resamples is frozen at {BOOTSTRAP_RESAMPLES}, got {checked_resamples}"
        )
    if checked_cluster_size != BOOTSTRAP_CLUSTER_SIZE:
        raise ValueError(
            f"cluster_size is frozen at {BOOTSTRAP_CLUSTER_SIZE}, "
            f"got {checked_cluster_size}"
        )
    if not isinstance(return_distribution, (bool, np.bool_)):
        raise TypeError("return_distribution must be bool")

    observations = _materialize_values(
        values, expected_size=BOOTSTRAP_CLUSTER_SIZE
    )
    identifiers = _sample_id_list(sample_ids, BOOTSTRAP_CLUSTER_SIZE)
    seed = metric_seed(identifier)
    generator = np.random.Generator(np.random.PCG64(seed))
    draws = generator.integers(
        0,
        BOOTSTRAP_CLUSTER_SIZE,
        size=(BOOTSTRAP_RESAMPLES, BOOTSTRAP_CLUSTER_SIZE),
        dtype=np.int64,
    )
    distribution = np.median(observations[draws], axis=1).astype(
        np.float64, copy=False
    )
    lower, upper = np.quantile(
        distribution, [0.025, 0.975], method=BOOTSTRAP_QUANTILE_METHOD
    )
    canonical_bytes = _distribution_canonical_bytes(distribution)
    canonical_draw_bytes = np.ascontiguousarray(
        draws, dtype=np.dtype("<i8")
    ).tobytes(order="C")
    sample_order_digest = (
        None
        if identifiers is None
        else ordered_sample_ids_sha256(identifiers)
    )

    result: dict[str, Any] = {
        "metric_id": identifier,
        "seed": seed,
        "base_seed": BOOTSTRAP_BASE_SEED,
        "seed_namespace": BOOTSTRAP_NAMESPACE + identifier,
        "generator": "numpy.random.Generator",
        "bit_generator": "PCG64",
        "resamples": BOOTSTRAP_RESAMPLES,
        "cluster_size": BOOTSTRAP_CLUSTER_SIZE,
        "cluster_unit": "image",
        "sample_ids": identifiers,
        "ordered_sample_ids_sha256": sample_order_digest,
        "statistic": "paired_median",
        "confidence_level": 0.95,
        "quantiles": [0.025, 0.975],
        "quantile_method": BOOTSTRAP_QUANTILE_METHOD,
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "distribution_count": BOOTSTRAP_RESAMPLES,
        "distribution_dtype": "float64_le",
        "distribution_encoding": "ieee754_binary64_little_endian_c_order",
        "distribution_sha256": hashlib.sha256(canonical_bytes).hexdigest(),
        "draw_indices_dtype": "int64_le",
        "draw_indices_sha256": hashlib.sha256(canonical_draw_bytes).hexdigest(),
    }
    if bool(return_distribution):
        result["distribution"] = distribution.tolist()
    return result


def summarize_values(
    metric_id: str,
    values: Any,
    *,
    scope: str,
    sample_ids: Any = None,
) -> dict[str, Any]:
    """Summarize one image-level metric in a frozen C10 scope.

    ``scope`` is one of ``calibration`` (n=5), ``holdout`` (n=15), or
    ``all_20`` (n=20).  Only holdout is confirmatory and therefore only that
    scope receives the registered 15-cluster bootstrap interval.
    """

    identifier = _identifier_string(metric_id, "metric_id")
    checked_scope = _nonempty_string(scope, "scope")
    if checked_scope not in _SCOPE_SIZES:
        raise ValueError(
            "scope must be exactly one of 'calibration', 'holdout', or 'all_20'"
        )
    expected_size = _SCOPE_SIZES[checked_scope]
    observations = _materialize_values(values, expected_size=expected_size)
    identifiers = _sample_id_list(sample_ids, expected_size)

    per_sample = [
        {
            "sample_index": index,
            "sample_id": None if identifiers is None else identifiers[index],
            "value": float(value),
        }
        for index, value in enumerate(observations)
    ]
    with np.errstate(over="ignore", invalid="ignore"):
        mean = float(np.mean(observations))
        median = float(np.median(observations))
        sample_sd = float(np.std(observations, ddof=1))
    for field_name, aggregate in (
        ("mean", mean),
        ("median", median),
        ("standard_deviation_sample_ddof_1", sample_sd),
    ):
        if not math.isfinite(aggregate):
            raise ValueError(
                f"{field_name} is non-finite for the supplied finite values"
            )
    result: dict[str, Any] = {
        "metric_id": identifier,
        "scope": checked_scope,
        "unit": "image",
        "confirmatory": checked_scope == "holdout",
        "n": expected_size,
        "sample_ids": identifiers,
        "per_sample": per_sample,
        "values": observations.tolist(),
        "mean": mean,
        "median": median,
        "standard_deviation_sample_ddof_1": sample_sd,
        "positive_count": int(np.count_nonzero(observations > 0.0)),
        "bootstrap_95_percent_ci": None,
        "bootstrap": None,
    }
    if checked_scope == "holdout":
        bootstrap = paired_median_bootstrap(
            identifier, observations, sample_ids=identifiers
        )
        result["bootstrap_95_percent_ci"] = {
            "lower": bootstrap["ci_lower"],
            "upper": bootstrap["ci_upper"],
        }
        result["bootstrap"] = bootstrap
    return result


def _bounded_count(value: Any, name: str, *, maximum: int) -> int:
    count = _integer(value, name)
    if count < 0 or count > maximum:
        raise ValueError(f"{name} must be between 0 and {maximum}")
    return count


def _ci_from_summary(summary: Mapping[str, Any], name: str) -> tuple[float, float]:
    candidates: list[tuple[float, float]] = []
    if "bootstrap_95_percent_ci" in summary:
        interval = summary["bootstrap_95_percent_ci"]
        if isinstance(interval, Mapping):
            if "lower" not in interval or "upper" not in interval:
                raise ValueError(
                    f"{name}.bootstrap_95_percent_ci requires lower and upper"
                )
            candidates.append(
                (
                    _finite_real(interval["lower"], f"{name}.ci.lower"),
                    _finite_real(interval["upper"], f"{name}.ci.upper"),
                )
            )
        elif isinstance(interval, Sequence) and not isinstance(
            interval, (str, bytes, bytearray)
        ):
            if len(interval) != 2:
                raise ValueError(f"{name}.bootstrap_95_percent_ci must have length 2")
            candidates.append(
                (
                    _finite_real(interval[0], f"{name}.ci[0]"),
                    _finite_real(interval[1], f"{name}.ci[1]"),
                )
            )
        else:
            raise TypeError(
                f"{name}.bootstrap_95_percent_ci must be a mapping or length-2 sequence"
            )
    bootstrap = summary.get("bootstrap")
    if bootstrap is not None:
        bootstrap_mapping = _mapping(bootstrap, f"{name}.bootstrap")
        if "ci_lower" not in bootstrap_mapping or "ci_upper" not in bootstrap_mapping:
            raise ValueError(f"{name}.bootstrap requires ci_lower and ci_upper")
        candidates.append(
            (
                _finite_real(
                    bootstrap_mapping["ci_lower"], f"{name}.bootstrap.ci_lower"
                ),
                _finite_real(
                    bootstrap_mapping["ci_upper"], f"{name}.bootstrap.ci_upper"
                ),
            )
        )
    if "ci_lower" in summary or "ci_upper" in summary:
        if "ci_lower" not in summary or "ci_upper" not in summary:
            raise ValueError(f"{name} requires both ci_lower and ci_upper")
        candidates.append(
            (
                _finite_real(summary["ci_lower"], f"{name}.ci_lower"),
                _finite_real(summary["ci_upper"], f"{name}.ci_upper"),
            )
        )
    if not candidates:
        raise ValueError(f"{name} has no bootstrap confidence interval")
    lower, upper = candidates[0]
    if lower > upper:
        raise ValueError(f"{name} confidence interval lower exceeds upper")
    for other_lower, other_upper in candidates[1:]:
        if other_lower != lower or other_upper != upper:
            raise ValueError(f"{name} has conflicting confidence intervals")
    return lower, upper


def _summary_gate_inputs(
    summary: Any, name: str
) -> tuple[float, int, float, float]:
    checked = _mapping(summary, name)
    if checked.get("scope") != "holdout":
        raise ValueError(f"{name}.scope must be exactly 'holdout'")
    if "unit" in checked and checked["unit"] != "image":
        raise ValueError(f"{name}.unit must be exactly 'image' when present")
    if "confirmatory" in checked and checked["confirmatory"] is not True:
        raise ValueError(f"{name}.confirmatory must be true when present")
    if "n" not in checked:
        raise ValueError(f"{name}.n is required")
    if _integer(checked["n"], f"{name}.n") != 15:
        raise ValueError(f"{name}.n must be exactly 15")
    if "median" not in checked or "positive_count" not in checked:
        raise ValueError(f"{name} requires median and positive_count")
    median = _finite_real(checked["median"], f"{name}.median")
    positive_count = _bounded_count(
        checked["positive_count"], f"{name}.positive_count", maximum=15
    )
    lower, upper = _ci_from_summary(checked, name)
    if checked.get("sample_ids") is not None:
        _sample_id_list(checked["sample_ids"], 15)
    return median, positive_count, lower, upper


def evaluate_effect(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the frozen primary fine-4x4-cell effect gate."""

    median, positive_count, lower, upper = _summary_gate_inputs(
        summary, "summary"
    )
    fail_positive_count = positive_count < EFFECT_POSITIVE_COUNT_MINIMUM
    fail_ci_upper = upper < EFFECT_MEDIAN_MINIMUM
    pass_median = median >= EFFECT_MEDIAN_MINIMUM
    pass_positive_count = positive_count >= EFFECT_POSITIVE_COUNT_MINIMUM
    pass_ci_lower = lower > EFFECT_CI_LOWER_EXCLUSIVE

    # A registered failure condition is decisive even for a synthetic summary
    # whose other fields would otherwise look like a pass.
    if fail_positive_count or fail_ci_upper:
        status = "FAIL_EFFECT"
    elif pass_median and pass_positive_count and pass_ci_lower:
        status = "PASS_EFFECT"
    else:
        status = "UNCERTAIN_EFFECT"
    return {
        "status": status,
        "effect_status": status,
        "median": median,
        "positive_count": positive_count,
        "bootstrap_ci_lower": lower,
        "bootstrap_ci_upper": upper,
        "criteria": {
            "median_at_least_0_02": pass_median,
            "positive_count_at_least_12": pass_positive_count,
            "ci_lower_strictly_above_0": pass_ci_lower,
            "ci_upper_strictly_below_0_02": fail_ci_upper,
        },
        "thresholds": {
            "median_minimum": EFFECT_MEDIAN_MINIMUM,
            "positive_count_minimum": EFFECT_POSITIVE_COUNT_MINIMUM,
            "ci_lower_exclusive": EFFECT_CI_LOWER_EXCLUSIVE,
            "fail_ci_upper_exclusive_below": EFFECT_MEDIAN_MINIMUM,
        },
    }


def _one_of_fields(record: Mapping[str, Any], names: tuple[str, ...], context: str) -> float:
    found = [(name, record[name]) for name in names if name in record]
    if not found:
        raise ValueError(f"{context} requires one of {names!r}")
    parsed = [_finite_real(value, f"{context}.{name}") for name, value in found]
    if any(value != parsed[0] for value in parsed[1:]):
        raise ValueError(f"{context} contains conflicting aliases for {names!r}")
    return parsed[0]


def _one_of_nullable_fields(
    record: Mapping[str, Any], names: tuple[str, ...], context: str,
) -> float | None:
    found = [(name, record[name]) for name in names if name in record]
    if not found:
        raise ValueError(f"{context} requires one of {names!r}")
    if all(value is None for _name, value in found):
        return None
    if any(value is None for _name, value in found):
        raise ValueError(f"{context} contains conflicting nullable aliases")
    parsed = [_finite_real(value, f"{context}.{name}") for name, value in found]
    if any(value != parsed[0] for value in parsed[1:]):
        raise ValueError(f"{context} contains conflicting aliases for {names!r}")
    return parsed[0]


def _evaluate_stability_summary(
    perturbation_id: str, summary: Any
) -> dict[str, Any]:
    checked = _mapping(summary, f"stability_summaries[{perturbation_id!r}]")
    if checked.get("status") == "INCOMPLETE_VECTOR":
        if checked.get("scope") != "holdout" or checked.get("n") != 15:
            raise ValueError(
                f"stability_summaries[{perturbation_id!r}] incomplete vector "
                "must retain all 15 holdout images"
            )
        if checked.get("unit") != "image" or checked.get("confirmatory") is not True:
            raise ValueError(
                f"stability_summaries[{perturbation_id!r}] has invalid image scope"
            )
        _sample_id_list(checked.get("sample_ids"), 15)
        positive_count = _bounded_count(
            checked.get("positive_count"),
            f"stability_summaries[{perturbation_id!r}].positive_count",
            maximum=15,
        )
        na_count = _bounded_count(
            checked.get("na_count"),
            f"stability_summaries[{perturbation_id!r}].na_count",
            maximum=15,
        )
        if na_count == 0 or any(
            checked.get(field) is not None
            for field in ("median", "bootstrap_95_percent_ci", "bootstrap")
        ):
            raise ValueError(
                f"stability_summaries[{perturbation_id!r}] incomplete vector "
                "must retain NA without an imputed median or CI"
            )
        failed = positive_count < STABILITY_POSITIVE_COUNT_MINIMUM
        status = "FAIL_STABILITY" if failed else "UNCERTAIN_STABILITY"
        return {
            "perturbation_id": perturbation_id, "status": status,
            "stability_status": status, "median": None,
            "positive_count": positive_count, "na_count": na_count,
            "bootstrap_ci_lower": None, "bootstrap_ci_upper": None,
            "criteria": {
                "positive_count_at_least_12": not failed,
                "median_at_least_0_50": None,
                "ci_lower_strictly_above_0": None,
                "median_and_ci_upper_strictly_below_0_50": None,
            },
            "reason": "retained_residual_na_no_imputation_or_complete_case_ci",
        }
    median, positive_count, lower, upper = _summary_gate_inputs(
        checked, f"stability_summaries[{perturbation_id!r}]"
    )
    for value, field in (
        (median, "median"),
        (lower, "bootstrap_ci_lower"),
        (upper, "bootstrap_ci_upper"),
    ):
        if value < -1.0 or value > 1.0:
            raise ValueError(
                f"stability_summaries[{perturbation_id!r}].{field} "
                "must be in [-1, 1] for Spearman correlation"
            )
    fail_positive_count = positive_count < STABILITY_POSITIVE_COUNT_MINIMUM
    fail_below_threshold = (
        median < STABILITY_MEDIAN_MINIMUM and upper < STABILITY_MEDIAN_MINIMUM
    )
    pass_median = median >= STABILITY_MEDIAN_MINIMUM
    pass_positive_count = positive_count >= STABILITY_POSITIVE_COUNT_MINIMUM
    pass_ci_lower = lower > STABILITY_CI_LOWER_EXCLUSIVE
    if fail_positive_count or fail_below_threshold:
        status = "FAIL_STABILITY"
    elif pass_positive_count and pass_median and pass_ci_lower:
        status = "PASS_STABILITY"
    else:
        status = "UNCERTAIN_STABILITY"
    return {
        "perturbation_id": perturbation_id,
        "status": status,
        "stability_status": status,
        "median": median,
        "positive_count": positive_count,
        "bootstrap_ci_lower": lower,
        "bootstrap_ci_upper": upper,
        "criteria": {
            "positive_count_at_least_12": pass_positive_count,
            "median_at_least_0_50": pass_median,
            "ci_lower_strictly_above_0": pass_ci_lower,
            "median_and_ci_upper_strictly_below_0_50": fail_below_threshold,
        },
    }


def evaluate_spatial(
    per_sample: Any, stability_summaries: Mapping[str, Any]
) -> dict[str, Any]:
    """Apply joint clean-spatial evidence and all three stability gates.

    Each of the 15 ``per_sample`` mappings must expose ``s_patch``, a positive
    patch-evidence fraction, and clean residual ratio ``rho``.  The latter two
    also accept their artifact-facing aliases
    ``positive_patch_evidence_fraction`` and ``patch_residual_l1_ratio``.
    """

    if isinstance(per_sample, (str, bytes, bytearray, Mapping)):
        raise TypeError("per_sample must be an iterable of 15 mappings")
    try:
        samples = list(per_sample)
    except TypeError as exc:
        raise TypeError("per_sample must be an iterable of 15 mappings") from exc
    if len(samples) != SPATIAL_SAMPLE_COUNT:
        raise ValueError(
            f"per_sample must contain exactly {SPATIAL_SAMPLE_COUNT} images; "
            f"got {len(samples)}"
        )

    sample_records: list[dict[str, Any]] = []
    observed_sample_ids: list[str | None] = []
    joint_count = 0
    for index, raw_sample in enumerate(samples):
        sample = _mapping(raw_sample, f"per_sample[{index}]")
        support = _one_of_nullable_fields(
            sample, ("s_patch", "patch_support"), f"per_sample[{index}]"
        )
        positive_fraction = _one_of_nullable_fields(
            sample,
            (
                "positive_patch_evidence_fraction_of_token_l1",
                "positive_patch_evidence_fraction",
                "positive_evidence_fraction",
            ),
            f"per_sample[{index}]",
        )
        rho = _one_of_nullable_fields(
            sample,
            ("rho", "patch_residual_l1_ratio", "residual_l1_ratio"),
            f"per_sample[{index}]",
        )
        if support is not None and not 0.0 <= support <= 1.0:
            raise ValueError(f"per_sample[{index}].s_patch must be in [0, 1]")
        if positive_fraction is not None and not 0.0 <= positive_fraction <= 1.0:
            raise ValueError(
                f"per_sample[{index}].positive patch evidence fraction "
                "must be in [0, 1]"
            )
        if rho is not None and rho < 0.0:
            raise ValueError(f"per_sample[{index}].rho must be non-negative")
        sample_id = (
            None
            if "sample_id" not in sample
            else _identifier_string(sample["sample_id"], f"per_sample[{index}].sample_id")
        )
        observed_sample_ids.append(sample_id)
        passes = (
            support is not None
            and positive_fraction is not None
            and rho is not None
            and support >= PATCH_SUPPORT_MINIMUM
            and positive_fraction >= POSITIVE_PATCH_EVIDENCE_MINIMUM
            and rho >= PATCH_RESIDUAL_L1_RATIO_MINIMUM
        )
        joint_count += int(passes)
        sample_records.append(
            {
                "sample_index": index,
                "sample_id": sample_id,
                "s_patch": support,
                "positive_patch_evidence_fraction_of_token_l1": positive_fraction,
                "rho": rho,
                "contains_na": any(
                    value is None for value in (support, positive_fraction, rho)
                ),
                "passes_joint_spatial_evidence": passes,
            }
        )
    if any(item is not None for item in observed_sample_ids):
        if any(item is None for item in observed_sample_ids):
            raise ValueError("per_sample sample_id must be present on all rows or none")
        if len(set(observed_sample_ids)) != SPATIAL_SAMPLE_COUNT:
            raise ValueError("per_sample sample_id values must be unique")

    stability = _mapping(stability_summaries, "stability_summaries")
    expected = set(STABILITY_PERTURBATIONS)
    actual = set(stability)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected, key=str)
        raise ValueError(
            "stability_summaries must contain exactly the three registered "
            f"perturbations; missing={missing}, extra={extra}"
        )
    stability_results = {
        perturbation_id: _evaluate_stability_summary(
            perturbation_id, stability[perturbation_id]
        )
        for perturbation_id in STABILITY_PERTURBATIONS
    }

    joint_status = (
        "PASS_JOINT_SPATIAL"
        if joint_count >= SPATIAL_JOINT_COUNT_MINIMUM
        else "FAIL_JOINT_SPATIAL"
    )
    stability_statuses = {item["status"] for item in stability_results.values()}
    if joint_status == "FAIL_JOINT_SPATIAL" or "FAIL_STABILITY" in stability_statuses:
        overall = "FAIL_SPATIAL"
    elif "UNCERTAIN_STABILITY" in stability_statuses:
        overall = "UNCERTAIN_SPATIAL"
    else:
        overall = "PASS_SPATIAL"
    return {
        "status": overall,
        "spatial_status": overall,
        "joint_spatial_status": joint_status,
        "joint_pass_count": joint_count,
        "joint_positive_count": joint_count,
        "joint_required_count": SPATIAL_JOINT_COUNT_MINIMUM,
        "sample_count": SPATIAL_SAMPLE_COUNT,
        "per_sample": sample_records,
        "stability": stability_results,
        "perturbations": stability_results,
        "thresholds": {
            "s_patch_minimum": PATCH_SUPPORT_MINIMUM,
            "positive_patch_evidence_fraction_minimum": (
                POSITIVE_PATCH_EVIDENCE_MINIMUM
            ),
            "rho_minimum": PATCH_RESIDUAL_L1_RATIO_MINIMUM,
            "stability_positive_count_minimum": (
                STABILITY_POSITIVE_COUNT_MINIMUM
            ),
            "stability_median_minimum": STABILITY_MEDIAN_MINIMUM,
            "stability_ci_lower_exclusive": STABILITY_CI_LOWER_EXCLUSIVE,
        },
    }


def _status(value: Any, name: str, allowed: set[str] | frozenset[str]) -> str:
    checked = _nonempty_string(value, name)
    if checked not in allowed:
        raise ValueError(f"{name} must be one of {sorted(allowed)!r}")
    return checked


def decide(
    validity_status: str,
    effect_status: str,
    spatial_status: str,
    cost_status: str,
) -> str:
    """Return the one Stage C decision using the frozen priority order."""

    validity_raw = _status(
        validity_status, "validity_status", set(_VALIDITY_STATUSES)
    )
    effect = _status(effect_status, "effect_status", _EFFECT_STATUSES)
    spatial = _status(spatial_status, "spatial_status", _SPATIAL_STATUSES)
    cost_raw = _status(cost_status, "cost_status", set(_COST_STATUSES))
    validity = _VALIDITY_STATUSES[validity_raw]
    cost = _COST_STATUSES[cost_raw]

    if validity == "FAIL_VALIDITY":
        return "NO_GO_VALIDITY"
    if effect == "FAIL_EFFECT" or spatial == "FAIL_SPATIAL":
        return "NO_GO_NOVELTY"
    if effect == "UNCERTAIN_EFFECT" or spatial == "UNCERTAIN_SPATIAL":
        return "INCONCLUSIVE"
    if cost == "FAIL_COST":
        return "NO_GO_COST"
    return "GO"


def holm_step_down(
    raw_p_values: Mapping[str, Any], alpha: float = 0.05
) -> list[dict[str, Any]]:
    """Apply deterministic Holm step-down correction.

    Records are returned in ascending ``(raw_p_value, member)`` order.  Holm
    adjusted p-values are the running maximum of ``(m-rank+1)*p`` capped at
    one, and rejection stops permanently at the first non-rejection.
    """

    values = _mapping(raw_p_values, "raw_p_values")
    if len(values) != 3:
        raise ValueError("raw_p_values must contain exactly three family members")
    checked_alpha = _finite_real(alpha, "alpha")
    if not 0.0 < checked_alpha < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1")

    parsed: list[tuple[str, float]] = []
    for raw_member, raw_p_value in values.items():
        member = _identifier_string(raw_member, "raw_p_values member")
        p_value = _finite_real(raw_p_value, f"raw_p_values[{member!r}]")
        if p_value < 0.0 or p_value > 1.0:
            raise ValueError(
                f"raw_p_values[{member!r}] must be between 0 and 1 inclusive"
            )
        parsed.append((member, p_value))
    parsed.sort(key=lambda item: (item[1], item[0]))

    family_size = len(parsed)
    rejection_active = True
    running_adjusted = 0.0
    records: list[dict[str, Any]] = []
    for rank, (member, p_value) in enumerate(parsed, start=1):
        remaining = family_size - rank + 1
        threshold = checked_alpha / remaining
        reject = rejection_active and p_value <= threshold
        if not reject:
            rejection_active = False
        running_adjusted = max(running_adjusted, remaining * p_value)
        adjusted = min(1.0, running_adjusted)
        records.append(
            {
                "member": member,
                "rank": rank,
                "family_size": family_size,
                "raw_p_value": p_value,
                "threshold": threshold,
                "adjusted_p_value": adjusted,
                "reject": reject,
            }
        )
    return records


_CONTRACT_CONTAINER_KEYS = frozenset(
    {
        "holm_contract",
        "holm_raw_p_value_contract",
        "exploratory_holm_contract",
        "exploratory_holm_raw_p_value_contract",
        "raw_p_value_contract",
        "raw_p_value_definition",
    }
)
_CONTRACT_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "test_statistic": ("test_statistic", "statistic"),
    "null_hypothesis": ("null_hypothesis", "null"),
    "alternative_hypothesis": (
        "alternative_hypothesis",
        "alternative",
        "directional_alternative",
        "direction",
    ),
    "sidedness": ("sidedness", "tail", "tails"),
    "family_members": (
        "family_members",
        "exact_family_members",
        "members",
    ),
    "p_value_method": (
        "p_value_method",
        "raw_p_value_method",
        "bootstrap_to_p_value_rule",
        "p_value_definition",
    ),
}
_UNDERSPECIFIED_TEXT_MARKERS = (
    "holm_blocked_underspecified",
    "underspecified",
    "raw p-value is not defined",
    "raw p-value definition is missing",
    "no raw p-value definition",
    "没有明确注册原始p-value",
    "没有明确注册原始 p-value",
)
_DECLARATIVE_TEXT_FIELDS = {
    "test_statistic": "test statistic",
    "null_hypothesis": "null hypothesis",
    "alternative_hypothesis": "alternative hypothesis",
    "sidedness": "sidedness",
    "family_members": "family members",
    "p_value_method": "p-value method",
}


def _contract_candidates(config: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
    candidates: list[tuple[str, Mapping[str, Any]]] = []
    seen: set[int] = set()

    def visit(value: Any, path: str) -> None:
        if not isinstance(value, Mapping):
            return
        identifier = id(value)
        field_hits = sum(
            any(alias in value for alias in aliases)
            for aliases in _CONTRACT_FIELD_ALIASES.values()
        )
        if (
            path.rsplit(".", 1)[-1] in _CONTRACT_CONTAINER_KEYS
            or field_hits >= 4
        ) and identifier not in seen:
            candidates.append((path, value))
            seen.add(identifier)
        for key, item in value.items():
            if isinstance(key, str) and isinstance(item, Mapping):
                visit(item, f"{path}.{key}")

    visit(config, "config")
    return candidates


def _audit_field(
    contract: Mapping[str, Any], canonical_name: str
) -> tuple[Any | None, str | None]:
    aliases = _CONTRACT_FIELD_ALIASES[canonical_name]
    found = [(key, contract[key]) for key in aliases if key in contract]
    if not found:
        return None, canonical_name
    first = found[0][1]
    if any(value != first for _, value in found[1:]):
        return None, f"{canonical_name}:conflicting_aliases"
    return first, None


def _holm_family_member(value: Any, index: int) -> tuple[int, int]:
    """Normalize one registered exploratory winner to ``(block, head)``."""

    context = f"family_members[{index}]"
    if isinstance(value, Mapping):
        if set(value) >= {"block_index", "head_index"}:
            raw_block = value["block_index"]
            raw_head = value["head_index"]
        elif set(value) >= {"block", "head"}:
            raw_block = value["block"]
            raw_head = value["head"]
        else:
            raise ValueError(f"{context} must identify block and head")
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        if len(value) != 2:
            raise ValueError(f"{context} must be a block/head pair")
        raw_block, raw_head = value
    elif isinstance(value, str):
        normalized = value.strip().lower()
        pieces = normalized.replace("block_", "").replace("head_", "").split("_")
        if len(pieces) != 2 or not all(piece.isdigit() for piece in pieces):
            raise ValueError(f"{context} must be a canonical block/head reference")
        raw_block, raw_head = (int(piece) for piece in pieces)
    else:
        raise TypeError(f"{context} must be a block/head pair")
    block = _integer(raw_block, f"{context}.block")
    head = _integer(raw_head, f"{context}.head")
    if not 0 <= block < 12 or not 0 <= head < 12:
        raise ValueError(f"{context} block/head indices must be in [0, 11]")
    return block, head


def _defined_contract_string(value: Any, name: str) -> tuple[str | None, str | None]:
    if not isinstance(value, str) or not value.strip():
        return None, name
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    forbidden = {
        "none",
        "null",
        "unknown",
        "unspecified",
        "underspecified",
        "tbd",
        "not_registered",
        "not_defined",
    }
    if normalized in forbidden:
        return None, name
    return value.strip(), None


def _protocol_contract_candidates(protocol_text: str) -> list[Mapping[str, Any]]:
    """Find explicit structured raw-p-value contracts in protocol text."""

    decoder = json.JSONDecoder()
    decoded: list[Any] = []
    try:
        decoded.append(json.loads(protocol_text.strip()))
    except (TypeError, ValueError, json.JSONDecodeError):
        # A Markdown protocol may embed an object rather than consist solely of
        # JSON.  Decode at every object start and retain contract-shaped maps.
        for index, character in enumerate(protocol_text):
            if character != "{":
                continue
            try:
                value, _ = decoder.raw_decode(protocol_text[index:])
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            decoded.append(value)

    result: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for value in decoded:
        if not isinstance(value, Mapping):
            continue
        roots = [("protocol", value)]
        candidates = _contract_candidates(value)
        if candidates:
            roots = candidates
        for _, candidate in roots:
            field_hits = sum(
                any(alias in candidate for alias in aliases)
                for aliases in _CONTRACT_FIELD_ALIASES.values()
            )
            if field_hits < len(_CONTRACT_FIELD_ALIASES):
                continue
            identity = json.dumps(
                candidate,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            if identity not in seen:
                result.append(candidate)
                seen.add(identity)
    return result


def _contract_signature(contract: Mapping[str, Any]) -> tuple[Any, ...] | None:
    """Build a representation-independent signature for one contract."""

    values: list[Any] = []
    for field_name in _CONTRACT_FIELD_ALIASES:
        raw_value, error = _audit_field(contract, field_name)
        if error is not None:
            return None
        if field_name == "family_members":
            if isinstance(raw_value, (str, bytes, bytearray, Mapping)):
                return None
            try:
                members = tuple(
                    sorted(
                        _holm_family_member(member, index)
                        for index, member in enumerate(list(raw_value))
                    )
                )
            except (TypeError, ValueError):
                return None
            if len(members) != 3 or len(set(members)) != 3:
                return None
            values.append(members)
        else:
            parsed, error = _defined_contract_string(raw_value, field_name)
            if error is not None:
                return None
            values.append(parsed)
    try:
        alpha = _finite_real(contract.get("alpha"), "protocol Holm alpha")
    except (TypeError, ValueError):
        return None
    values.append(alpha)
    return tuple(values)


def audit_holm_contract(
    config_dict: Mapping[str, Any], protocol_text: str
) -> dict[str, Any]:
    """Audit whether frozen inputs uniquely define raw exploratory p-values.

    This helper intentionally does not infer a hypothesis test from descriptive
    bootstrap output.  A complete contract must be one unique structured
    mapping (for example ``statistics.holm_raw_p_value_contract``) defining the
    test statistic, null, alternative/direction, sidedness, exactly three family
    members, and the raw-p-value calculation method.  Holm and alpha=0.05 must
    also be registered.  Anything less fails closed.
    """

    config = _mapping(config_dict, "config_dict")
    text = _nonempty_string(protocol_text, "protocol_text")
    statistics = config.get("statistics", config)
    if not isinstance(statistics, Mapping):
        raise TypeError("config_dict.statistics must be a mapping when present")

    method_value = statistics.get(
        "exploratory_band_winner_multiple_testing",
        statistics.get("multiple_testing_method", statistics.get("method")),
    )
    method_registered = isinstance(method_value, str) and "holm" in method_value.lower()

    candidates = _contract_candidates(config)
    missing: list[str] = []
    ambiguous: list[str] = []
    definitions: dict[str, Any] = {}
    normalized_family_members: list[tuple[int, int]] = []
    contract_path: str | None = None

    if not method_registered:
        missing.append("holm_method")
    if len(candidates) == 0:
        missing.extend(_CONTRACT_FIELD_ALIASES)
        missing.extend(["alpha", "exactly_three_family_members"])
    elif len(candidates) > 1:
        ambiguous.append("multiple_raw_p_value_contracts")
    else:
        contract_path, contract = candidates[0]
        for field_name in _CONTRACT_FIELD_ALIASES:
            raw_value, error = _audit_field(contract, field_name)
            if error is not None:
                (ambiguous if "conflicting" in error else missing).append(error)
                continue
            if field_name == "family_members":
                if isinstance(raw_value, (str, bytes, bytearray, Mapping)):
                    missing.append(field_name)
                    continue
                try:
                    raw_members = list(raw_value)
                except TypeError:
                    missing.append(field_name)
                    continue
                try:
                    members = [
                        _holm_family_member(member, index)
                        for index, member in enumerate(raw_members)
                    ]
                except (TypeError, ValueError):
                    missing.append(field_name)
                    continue
                if len(members) != 3 or len(set(members)) != 3:
                    missing.append("exactly_three_family_members")
                    continue
                if set(members) != set(EXPECTED_EXPLORATORY_WINNERS):
                    ambiguous.append("family_members_do_not_match_frozen_exploratory_winners")
                    continue
                # Preserve the registered spelling for audit transparency while
                # exposing normalized block/head pairs at the report top level.
                definitions[field_name] = raw_members
                normalized_family_members = members
            else:
                parsed, string_error = _defined_contract_string(raw_value, field_name)
                if string_error is not None:
                    missing.append(string_error)
                else:
                    definitions[field_name] = parsed

        sidedness = definitions.get("sidedness")
        if isinstance(sidedness, str):
            normalized_sidedness = (
                sidedness.lower().replace("-", "_").replace(" ", "_")
            )
            allowed_sidedness = {
                "one_sided",
                "one_tailed",
                "two_sided",
                "two_tailed",
                "greater",
                "less",
            }
            if normalized_sidedness not in allowed_sidedness:
                missing.append("sidedness:unrecognized")

        alpha_value = contract.get("alpha", statistics.get("alpha"))
        if alpha_value is None and isinstance(method_value, str) and "alpha_0_05" in method_value.lower():
            alpha_value = 0.05
        if alpha_value is None:
            missing.append("alpha")
        else:
            try:
                parsed_alpha = _finite_real(alpha_value, "holm alpha")
            except (TypeError, ValueError):
                missing.append("alpha")
            else:
                if parsed_alpha != 0.05:
                    ambiguous.append("alpha_not_frozen_0_05")
                else:
                    definitions["alpha"] = parsed_alpha

        correction = contract.get("correction", contract.get("method"))
        if correction is not None and (
            not isinstance(correction, str) or "holm" not in correction.lower()
        ):
            ambiguous.append("correction_is_not_holm")

    lowered_text = text.lower()
    protocol_contracts = _protocol_contract_candidates(text)
    if candidates:
        if len(protocol_contracts) == 1 and len(candidates) == 1:
            config_signature = _contract_signature(candidates[0][1])
            protocol_signature = _contract_signature(protocol_contracts[0])
            if config_signature is None or protocol_signature != config_signature:
                ambiguous.append("config_protocol_raw_p_value_contract_mismatch")
        elif len(protocol_contracts) > 1:
            ambiguous.append("multiple_protocol_raw_p_value_contracts")
        elif len(candidates) == 1:
            # Plain-text protocols need not embed JSON, but they must explicitly
            # register every ingredient.  Merely mentioning Holm is insufficient.
            absent = [
                name
                for name, phrase in _DECLARATIVE_TEXT_FIELDS.items()
                if phrase not in lowered_text
            ]
            if absent:
                missing.append("matching_protocol_raw_p_value_contract")
    protocol_reports_underspecified = any(
        marker in lowered_text for marker in _UNDERSPECIFIED_TEXT_MARKERS
    )
    if protocol_reports_underspecified and candidates:
        ambiguous.append("protocol_explicitly_reports_underspecified_raw_p_values")

    missing = sorted(set(missing))
    ambiguous = sorted(set(ambiguous))
    complete = not missing and not ambiguous
    status = (
        "HOLM_CONTRACT_COMPLETE"
        if complete
        else "HOLM_BLOCKED_UNDERSPECIFIED"
    )
    return {
        "status": status,
        "completion_status": "READY" if complete else "BLOCKED",
        "can_compute_raw_p_values": complete,
        "can_perform_holm": complete,
        "holm_performed": False,
        "contract_path": contract_path,
        "alpha": definitions.get("alpha", 0.05),
        "expected_exploratory_winners": [
            list(item) for item in EXPECTED_EXPLORATORY_WINNERS
        ],
        "family_members": (
            [list(item) for item in normalized_family_members]
            if complete
            else []
        ),
        "candidate_contract_paths": [path for path, _ in candidates],
        "protocol_contract_count": len(protocol_contracts),
        "required_definitions": [
            "test_statistic",
            "null_hypothesis",
            "alternative_hypothesis",
            "sidedness",
            "family_members",
            "p_value_method",
            "holm_method",
            "alpha_0_05",
            "exactly_three_family_members",
        ],
        "missing_definitions": missing,
        "missing": missing,
        "ambiguous_definitions": ambiguous,
        "definitions": definitions if complete else {},
        "method_registration": method_value,
        "protocol_mentions_holm": "holm" in lowered_text,
        "protocol_mentions_p_value": (
            "p-value" in lowered_text or "p value" in lowered_text
        ),
        "protocol_reports_underspecified": protocol_reports_underspecified,
        "raw_p_values": [],
        "adjusted_p_values": [],
        "reject": [],
        "affects_primary_decision": False,
        "acceptance_blocking": not complete,
        "reason": (
            "the frozen inputs uniquely register the raw p-value contract"
            if complete
            else (
                "raw p-values are underspecified; do not synthesize p-values "
                "or run Holm correction"
            )
        ),
    }


__all__ = [
    "BOOTSTRAP_BASE_SEED",
    "BOOTSTRAP_CLUSTER_SIZE",
    "BOOTSTRAP_RESAMPLES",
    "STABILITY_PERTURBATIONS",
    "audit_holm_contract",
    "bootstrap_seed",
    "decide",
    "evaluate_effect",
    "evaluate_spatial",
    "holm_step_down",
    "metric_seed",
    "ordered_sample_ids_sha256",
    "paired_median_bootstrap",
    "summarize_values",
]
