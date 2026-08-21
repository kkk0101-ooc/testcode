"""Metric-ready Stage C spatial metrics with structured NA results.

The public functions deliberately accept only arrays tagged ``metric_ready``.
Raw tensors, bare ndarrays, and display products are contract violations.
Numerical invalidity discovered after that boundary (for example a tampered
non-finite artifact, a shape mismatch, or an empty mask) becomes an explicit
``StructuredNA`` rather than a missing row or an imputed zero.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .display import DisplayBoundaryError
from .retention import ArtifactArray, ArtifactKind, _deep_freeze
from .stage_c_maps import (
    NAReason,
    PATCH_SHAPE,
    SignSemantics,
    SignView,
    StructuredNA,
)


CONSTANT_RELATIVE_L2_THRESHOLD = 1e-12
ZERO_MASS_RELATIVE_L1_THRESHOLD = 1e-12
KL_EPSILON = 1e-12
HOTSPOT_FRACTIONS = frozenset({0.1, 0.2})


@dataclass(frozen=True)
class MetricValue:
    """A finite scalar metric and optional immutable audit details."""

    value: float
    details: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if isinstance(self.value, bool) or not isinstance(
            self.value, (int, float, np.integer, np.floating)
        ):
            raise TypeError("MetricValue.value must be numeric")
        value = float(self.value)
        if not math.isfinite(value):
            raise ValueError("MetricValue.value must be finite")
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "details", _deep_freeze(dict(self.details or {})))

    @property
    def status(self) -> str:
        return "OK"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": "OK",
            "value": self.value,
            "details": _thaw(self.details),
        }


@dataclass(frozen=True)
class BidirectionalKL:
    left_to_right: MetricValue
    right_to_left: MetricValue

    @property
    def status(self) -> str:
        return "OK"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": "OK",
            "left_to_right": self.left_to_right.to_dict(),
            "right_to_left": self.right_to_left.to_dict(),
        }


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    if isinstance(value, frozenset):
        return sorted(_thaw(item) for item in value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _parse_sign_semantics(value: SignSemantics | str, side: str) -> SignSemantics:
    try:
        return SignSemantics(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{side}_semantics is invalid: {value!r}") from exc


def _parse_view(value: SignView | str) -> SignView:
    try:
        return SignView(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"view is invalid: {value!r}") from exc


def _tagged_array(value: Any, side: str) -> tuple[np.ndarray | None, StructuredNA | None]:
    kind = getattr(value, "kind", None)
    if kind is None:
        raise DisplayBoundaryError(
            f"{side} must be explicitly tagged metric_ready; bare arrays are forbidden"
        )
    parsed = ArtifactKind.parse(kind)
    if parsed is ArtifactKind.DISPLAY:
        raise DisplayBoundaryError(f"{side} rejects display data")
    if parsed is not ArtifactKind.METRIC_READY:
        raise DisplayBoundaryError(
            f"{side} requires metric_ready input, got {parsed.value}"
        )
    data = getattr(value, "data", None)
    if not isinstance(data, np.ndarray):
        raise TypeError(f"{side} metric_ready payload must be a numpy array")
    if not np.issubdtype(data.dtype, np.number) or np.issubdtype(
        data.dtype, np.complexfloating
    ):
        raise TypeError(f"{side} must be a real numeric array")
    if data.dtype != np.dtype(np.float32):
        raise TypeError(f"{side} metric_ready payload must have dtype float32")
    if data.shape != PATCH_SHAPE:
        return None, StructuredNA(
            NAReason.SHAPE_MISMATCH, f"{side} shape {data.shape} != {PATCH_SHAPE}"
        )
    if not np.all(np.isfinite(data)):
        return None, StructuredNA(NAReason.NON_FINITE, f"{side} contains non-finite values")
    return np.asarray(data, dtype=np.float64), None


def _valid_mask(mask: Any) -> tuple[np.ndarray | None, StructuredNA | None]:
    if not isinstance(mask, np.ndarray):
        raise TypeError("mask must be an explicit numpy bool array")
    if mask.dtype != np.bool_:
        raise TypeError("mask must have boolean dtype")
    if mask.shape != PATCH_SHAPE:
        return None, StructuredNA(
            NAReason.SHAPE_MISMATCH, f"mask shape {mask.shape} != {PATCH_SHAPE}"
        )
    if not bool(np.any(mask)):
        return None, StructuredNA(NAReason.EMPTY_VALID_REGION)
    return mask, None


def _pair_inputs(
    left: Any, right: Any, mask: Any
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, StructuredNA | None]:
    left_array, error = _tagged_array(left, "left")
    if error is not None:
        return None, None, None, error
    right_array, error = _tagged_array(right, "right")
    if error is not None:
        return None, None, None, error
    valid, error = _valid_mask(mask)
    if error is not None:
        return None, None, None, error
    return left_array, right_array, valid, None


def _one_input(
    source: Any, mask: Any
) -> tuple[np.ndarray | None, np.ndarray | None, StructuredNA | None]:
    array, error = _tagged_array(source, "source")
    if error is not None:
        return None, None, error
    valid, error = _valid_mask(mask)
    if error is not None:
        return None, None, error
    return array, valid, None


def _view_pair(
    left: np.ndarray,
    right: np.ndarray,
    mask: np.ndarray,
    *,
    view: SignView | str,
    left_semantics: SignSemantics | str,
    right_semantics: SignSemantics | str,
    nonnegative_required: bool,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None, StructuredNA | None]:
    selected = _parse_view(view)
    left_sign = _parse_sign_semantics(left_semantics, "left")
    right_sign = _parse_sign_semantics(right_semantics, "right")
    if left_sign is SignSemantics.NONNEGATIVE and np.any(left < 0):
        raise ValueError("left violates its nonnegative sign semantics")
    if right_sign is SignSemantics.NONNEGATIVE and np.any(right < 0):
        raise ValueError("right violates its nonnegative sign semantics")
    if selected is SignView.SIGNED:
        if nonnegative_required or (
            left_sign is not SignSemantics.SIGNED
            or right_sign is not SignSemantics.SIGNED
        ):
            return None, None, None, None, StructuredNA(NAReason.UNSUPPORTED_SIGN_SEMANTICS)
        left_view, right_view = left, right
    elif selected is SignView.POSITIVE:
        left_view = np.maximum(left, 0.0)
        right_view = np.maximum(right, 0.0)
    else:
        if (
            left_sign is not SignSemantics.SIGNED
            or right_sign is not SignSemantics.SIGNED
        ):
            return None, None, None, None, StructuredNA(NAReason.UNSUPPORTED_SIGN_SEMANTICS)
        left_view = np.maximum(-left, 0.0)
        right_view = np.maximum(-right, 0.0)
    return left_view[mask], right_view[mask], left[mask], right[mask], None


def _view_one(
    source: np.ndarray,
    mask: np.ndarray,
    *,
    view: SignView | str,
    semantics: SignSemantics | str,
    nonnegative_required: bool,
) -> tuple[np.ndarray | None, np.ndarray | None, StructuredNA | None]:
    selected = _parse_view(view)
    sign = _parse_sign_semantics(semantics, "source")
    if sign is SignSemantics.NONNEGATIVE and np.any(source < 0):
        raise ValueError("source violates its nonnegative sign semantics")
    if selected is SignView.SIGNED:
        if nonnegative_required or sign is not SignSemantics.SIGNED:
            return None, None, StructuredNA(NAReason.UNSUPPORTED_SIGN_SEMANTICS)
        branch = source
    elif selected is SignView.POSITIVE:
        branch = np.maximum(source, 0.0)
    else:
        if sign is not SignSemantics.SIGNED:
            return None, None, StructuredNA(NAReason.UNSUPPORTED_SIGN_SEMANTICS)
        branch = np.maximum(-source, 0.0)
    return branch[mask], source[mask], None


def _constant(values: np.ndarray) -> bool:
    centered = values - np.mean(values, dtype=np.float64)
    centered_norm = float(np.linalg.norm(centered))
    scale = CONSTANT_RELATIVE_L2_THRESHOLD * max(
        1.0, float(np.linalg.norm(values))
    )
    return centered_norm <= scale


def _correlation_values(
    left: Any,
    right: Any,
    *,
    mask: Any,
    view: SignView | str,
    left_semantics: SignSemantics | str,
    right_semantics: SignSemantics | str,
) -> tuple[np.ndarray | None, np.ndarray | None, StructuredNA | None]:
    left_map, right_map, valid, error = _pair_inputs(left, right, mask)
    if error is not None:
        return None, None, error
    left_values, right_values, _, _, error = _view_pair(
        left_map, right_map, valid, view=view, left_semantics=left_semantics,
        right_semantics=right_semantics, nonnegative_required=False,
    )
    if error is not None:
        return None, None, error
    if _constant(left_values):
        return None, None, StructuredNA(NAReason.CONSTANT_LEFT)
    if _constant(right_values):
        return None, None, StructuredNA(NAReason.CONSTANT_RIGHT)
    return left_values, right_values, None


def _pearson_values(left: np.ndarray, right: np.ndarray) -> float:
    left_centered = left - np.mean(left, dtype=np.float64)
    right_centered = right - np.mean(right, dtype=np.float64)
    denominator = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
    return float(np.dot(left_centered, right_centered) / denominator)


def pearson(
    left: Any,
    right: Any,
    *,
    mask: Any,
    view: SignView | str,
    left_semantics: SignSemantics | str,
    right_semantics: SignSemantics | str,
) -> MetricValue | StructuredNA:
    values_left, values_right, error = _correlation_values(
        left, right, mask=mask, view=view, left_semantics=left_semantics,
        right_semantics=right_semantics,
    )
    if error is not None:
        return error
    return MetricValue(_pearson_values(values_left, values_right))


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        # Ranks are one-based.  The average of start+1 through end is
        # ((start+1)+end)/2.
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def spearman(
    left: Any,
    right: Any,
    *,
    mask: Any,
    view: SignView | str,
    left_semantics: SignSemantics | str,
    right_semantics: SignSemantics | str,
) -> MetricValue | StructuredNA:
    values_left, values_right, error = _correlation_values(
        left, right, mask=mask, view=view, left_semantics=left_semantics,
        right_semantics=right_semantics,
    )
    if error is not None:
        return error
    return MetricValue(
        _pearson_values(_average_ranks(values_left), _average_ranks(values_right)),
        {"tie_method": "average_rank"},
    )


def _normalized_distributions(
    left: Any,
    right: Any,
    *,
    mask: Any,
    view: SignView | str,
    left_semantics: SignSemantics | str,
    right_semantics: SignSemantics | str,
) -> tuple[np.ndarray | None, np.ndarray | None, StructuredNA | None]:
    left_map, right_map, valid, error = _pair_inputs(left, right, mask)
    if error is not None:
        return None, None, error
    left_branch, right_branch, left_signed, right_signed, error = _view_pair(
        left_map, right_map, valid, view=view, left_semantics=left_semantics,
        right_semantics=right_semantics, nonnegative_required=True,
    )
    if error is not None:
        return None, None, error
    left_mass = float(np.sum(left_branch, dtype=np.float64))
    right_mass = float(np.sum(right_branch, dtype=np.float64))
    left_scale = ZERO_MASS_RELATIVE_L1_THRESHOLD * float(
        np.sum(np.abs(left_signed), dtype=np.float64)
    )
    right_scale = ZERO_MASS_RELATIVE_L1_THRESHOLD * float(
        np.sum(np.abs(right_signed), dtype=np.float64)
    )
    if left_mass <= left_scale:
        return None, None, StructuredNA(NAReason.ZERO_MASS_LEFT)
    if right_mass <= right_scale:
        return None, None, StructuredNA(NAReason.ZERO_MASS_RIGHT)
    return left_branch / left_mass, right_branch / right_mass, None


def jensen_shannon_divergence(
    left: Any,
    right: Any,
    *,
    mask: Any,
    view: SignView | str,
    left_semantics: SignSemantics | str,
    right_semantics: SignSemantics | str,
) -> MetricValue | StructuredNA:
    p, q, error = _normalized_distributions(
        left, right, mask=mask, view=view, left_semantics=left_semantics,
        right_semantics=right_semantics,
    )
    if error is not None:
        return error
    mixture = 0.5 * (p + q)
    with np.errstate(divide="ignore", invalid="ignore"):
        left_terms = np.where(p > 0.0, p * np.log2(p / mixture), 0.0)
        right_terms = np.where(q > 0.0, q * np.log2(q / mixture), 0.0)
    value = 0.5 * (
        float(np.sum(left_terms, dtype=np.float64))
        + float(np.sum(right_terms, dtype=np.float64))
    )
    if value < -1e-14 or value > 1.0 + 1e-14:
        raise ArithmeticError(f"JSD escaped its base-2 [0,1] range: {value}")
    # Remove only sub-ulp roundoff at the mathematical endpoints.
    value = min(1.0, max(0.0, value))
    return MetricValue(value, {"log_base": 2, "epsilon": 0.0})


def bidirectional_kl(
    left: Any,
    right: Any,
    *,
    mask: Any,
    view: SignView | str,
    left_semantics: SignSemantics | str,
    right_semantics: SignSemantics | str,
) -> BidirectionalKL | StructuredNA:
    p, q, error = _normalized_distributions(
        left, right, mask=mask, view=view, left_semantics=left_semantics,
        right_semantics=right_semantics,
    )
    if error is not None:
        return error
    count = p.size
    denominator = 1.0 + count * KL_EPSILON
    p_epsilon = (p + KL_EPSILON) / denominator
    q_epsilon = (q + KL_EPSILON) / denominator
    left_to_right = float(
        np.sum(p_epsilon * np.log2(p_epsilon / q_epsilon), dtype=np.float64)
    )
    right_to_left = float(
        np.sum(q_epsilon * np.log2(q_epsilon / p_epsilon), dtype=np.float64)
    )
    if left_to_right < -1e-14 or right_to_left < -1e-14:
        raise ArithmeticError(
            "base-2 KL became materially negative: "
            f"left_to_right={left_to_right}, right_to_left={right_to_left}"
        )
    details = {"log_base": 2, "epsilon": KL_EPSILON}
    return BidirectionalKL(
        MetricValue(max(0.0, left_to_right), details),
        MetricValue(max(0.0, right_to_left), details),
    )


def hotspot_jaccard(
    left: Any,
    right: Any,
    *,
    mask: Any,
    view: SignView | str,
    left_semantics: SignSemantics | str,
    right_semantics: SignSemantics | str,
    fraction: float,
) -> MetricValue | StructuredNA:
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)):
        raise TypeError("fraction must be numeric")
    fraction = float(fraction)
    if fraction not in HOTSPOT_FRACTIONS:
        raise ValueError("fraction must be the frozen 0.1 or 0.2")
    left_map, right_map, valid, error = _pair_inputs(left, right, mask)
    if error is not None:
        return error
    left_values, right_values, _, _, error = _view_pair(
        left_map, right_map, valid, view=view, left_semantics=left_semantics,
        right_semantics=right_semantics, nonnegative_required=True,
    )
    if error is not None:
        return error
    if _constant(left_values):
        return StructuredNA(NAReason.CONSTANT_LEFT)
    if _constant(right_values):
        return StructuredNA(NAReason.CONSTANT_RIGHT)
    linear_indices = np.flatnonzero(valid)
    count = left_values.size
    top_count = int(math.ceil(fraction * count))

    def select(values: np.ndarray) -> tuple[set[int], float, int]:
        # lexsort uses the final key as primary: -value first, then row-major.
        order = np.lexsort((linear_indices, -values))
        chosen = order[:top_count]
        cutoff = float(values[order[top_count - 1]])
        cutoff_ties = int(np.count_nonzero(values == cutoff))
        return set(int(item) for item in linear_indices[chosen]), cutoff, cutoff_ties

    left_set, left_cutoff, left_ties = select(left_values)
    right_set, right_cutoff, right_ties = select(right_values)
    intersection = len(left_set & right_set)
    union = len(left_set | right_set)
    return MetricValue(
        intersection / union,
        {
            "fraction": fraction,
            "n_valid": count,
            "k": top_count,
            "left_cutoff": left_cutoff,
            "right_cutoff": right_cutoff,
            "left_cutoff_tie_count": left_ties,
            "right_cutoff_tie_count": right_ties,
            "tie_break": "value_descending_then_row_major_index_ascending",
        },
    )


def hotspot_jaccard_top_10_percent(*args: Any, **kwargs: Any) -> MetricValue | StructuredNA:
    return hotspot_jaccard(*args, **kwargs, fraction=0.1)


def hotspot_jaccard_top_20_percent(*args: Any, **kwargs: Any) -> MetricValue | StructuredNA:
    return hotspot_jaccard(*args, **kwargs, fraction=0.2)


def normalized_entropy(
    source: Any,
    *,
    mask: Any,
    view: SignView | str,
    semantics: SignSemantics | str,
) -> MetricValue | StructuredNA:
    source_map, valid, error = _one_input(source, mask)
    if error is not None:
        return error
    branch, signed, error = _view_one(
        source_map, valid, view=view, semantics=semantics, nonnegative_required=True
    )
    if error is not None:
        return error
    if branch.size < 2:
        # With one valid cell every map is necessarily constant; the frozen
        # reason vocabulary has no separate "too few cells" token.
        return StructuredNA(NAReason.CONSTANT_LEFT, "normalized entropy requires n_valid >= 2")
    mass = float(np.sum(branch, dtype=np.float64))
    scale = ZERO_MASS_RELATIVE_L1_THRESHOLD * float(
        np.sum(np.abs(signed), dtype=np.float64)
    )
    if mass <= scale:
        return StructuredNA(NAReason.ZERO_MASS_LEFT)
    probabilities = branch / mass
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(
            probabilities > 0.0, probabilities * np.log2(probabilities), 0.0
        )
    entropy = -float(np.sum(terms, dtype=np.float64)) / math.log2(branch.size)
    if entropy < -1e-14 or entropy > 1.0 + 1e-14:
        raise ArithmeticError(f"normalized entropy escaped [0,1]: {entropy}")
    return MetricValue(
        min(1.0, max(0.0, entropy)),
        {"log_base": 2, "n_valid": branch.size},
    )


__all__ = [
    "BidirectionalKL", "CONSTANT_RELATIVE_L2_THRESHOLD", "KL_EPSILON",
    "MetricValue", "ZERO_MASS_RELATIVE_L1_THRESHOLD", "bidirectional_kl",
    "hotspot_jaccard", "hotspot_jaccard_top_10_percent",
    "hotspot_jaccard_top_20_percent", "jensen_shannon_divergence",
    "normalized_entropy", "pearson", "spearman",
]
