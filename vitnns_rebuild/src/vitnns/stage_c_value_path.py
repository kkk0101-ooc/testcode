"""Stage C fixed-clean-attention head-direction Value-path methods.

This module implements the two same-target methods frozen by the Stage C
protocol.  Endpoint tensors are observed from the real timm forward at
``qkv`` and ``attn_drop``; no QK, softmax, or Transformer block is replayed.
The clean attention row and the endpoint-derived direction are detached and
frozen before attribution.  Consequently the differentiable callback used by
Integrated Gradients contains only the real upstream Value path.

The small numerical functions at the top of the module are intentionally pure
and accept an arbitrary token count/head width.  Production resolution below
separately enforces the canonical 12 x 12 x 197 x 64 ViT geometry.
"""

from __future__ import annotations

import math
import json
import re
import weakref
from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, field, replace
from secrets import token_bytes
from typing import Any

import numpy as np
import torch
from torch import nn

from . import attribution
from .attribution import IGResult
from .extractor import split_fused_qkv
from .stage_c_class_logit import (
    _model_binding_sha256,
    _model_state_signature,
    _preserve_model_attribution_state,
    _tensor_sha256,
    _validate_clean_input,
)
from .stage_c_integrity import CheapTensorSnapshot, model_live_signature
from .stage_c_maps import (
    ComparisonLine,
    MethodResult,
    NAReason,
    PATCH_SHAPE,
    SignSemantics,
    StructuredNA,
    pixel_to_patch_mass,
    raw_array,
    signed_rgb_channel_sum,
    stage_c_metric_ready,
)


METHOD_VERSION = "1.0.0"
MODEL_BLOCKS = 12
HEADS_PER_BLOCK = 12
TOKENS = 197
PATCH_TOKENS = 196
HEAD_DIM = 64
EMBED_DIM = 768
CLS_TOKEN_INDEX = 0

IG_STEPS = (512, 1024, 2048, 4096, 8192)
IG_STEP_CHUNK_SIZE = 8
IG_ATOL = 1e-4
IG_RTOL = 0.01

DIRECTION_RELATIVE_THRESHOLD = 1e-6
DIRECTION_SCALE_FLOOR = 1.0
ENDPOINT_PARITY_MAX_ABS = 1e-6
TOKEN_CONSERVATION_ATOL = 1e-6
TOKEN_CONSERVATION_RTOL = 1e-5
PATCH_SUPPORT_MINIMUM = 0.01
POSITIVE_PATCH_EVIDENCE_MINIMUM = 0.01

TARGET_REF = "fixed_clean_attention_cls_head_direction"
HEAD_DIRECTION_TARGET = TARGET_REF
NNS_METHOD_ID = (
    "conditional_fixed_clean_attention_head_direction_value_path_ig"
)
NNS_METHOD_NAME = (
    "NNS fixed-clean-attention head-direction Value-path IG"
)
NNS_EXACT_NAME = NNS_METHOD_NAME
MATCHED_METHOD_ID = "matched_fixed_clean_attention_patch_token_contribution"
MATCHED_METHOD_NAME = (
    "matched fixed-clean-attention patch-token contribution"
)
MATCHED_EXACT_NAME = MATCHED_METHOD_NAME

STATUS_OK = "OK"
DEGENERATE_DIRECTION = "DEGENERATE_DIRECTION"
DEGENERATE_PATCH_SUPPORT = "DEGENERATE_PATCH_SUPPORT"
DEGENERATE_POSITIVE_PATCH_EVIDENCE = (
    "DEGENERATE_POSITIVE_PATCH_EVIDENCE"
)

_TARGET_ISSUER_SECRET = token_bytes(32)
_ISSUED_TARGETS: "weakref.WeakKeyDictionary[FrozenHeadDirectionTarget, tuple[Any, ...]]" = (
    weakref.WeakKeyDictionary()
)


class StageCValuePathError(ValueError):
    """Raised when a C4 Value-path contract is violated."""


@dataclass(frozen=True, init=False)
class FrozenScalarEvaluation:
    """One fixed-A, fixed-direction Value-path scalar evaluation."""

    value: float
    target_kind: str
    block_index: int
    head_index: int
    query_token: int
    clean_input_sha256: str
    intervention_input_sha256: str
    runtime_model_binding_sha256: str

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise TypeError(
            "FrozenScalarEvaluation can only be issued by "
            "evaluate_frozen_head_direction_target"
        )

    @classmethod
    def _create(
        cls,
        *,
        value: float,
        block_index: int,
        head_index: int,
        clean_input_sha256: str,
        intervention_input_sha256: str,
        runtime_model_binding_sha256: str,
        issuer_secret: bytes,
    ) -> "FrozenScalarEvaluation":
        if issuer_secret is not _TARGET_ISSUER_SECRET:
            raise TypeError("scalar evaluations can only be issued by the evaluator")
        instance = object.__new__(cls)
        for name, field_value in {
            "value": value,
            "target_kind": TARGET_REF,
            "block_index": block_index,
            "head_index": head_index,
            "query_token": CLS_TOKEN_INDEX,
            "clean_input_sha256": clean_input_sha256,
            "intervention_input_sha256": intervention_input_sha256,
            "runtime_model_binding_sha256": runtime_model_binding_sha256,
        }.items():
            object.__setattr__(instance, name, field_value)
        instance.__post_init__()
        return instance

    def __post_init__(self) -> None:
        if isinstance(self.value, bool) or not isinstance(self.value, (int, float)):
            raise TypeError("value must be numeric")
        value = float(self.value)
        if not math.isfinite(value):
            raise StageCValuePathError("scalar evaluation value must be finite")
        object.__setattr__(self, "value", value)
        if self.target_kind != TARGET_REF:
            raise StageCValuePathError("scalar target_kind is not the frozen head target")
        _strict_index(self.block_index, name="block_index", upper_bound=MODEL_BLOCKS)
        _strict_index(self.head_index, name="head_index", upper_bound=HEADS_PER_BLOCK)
        if self.query_token != CLS_TOKEN_INDEX:
            raise StageCValuePathError("query_token must be the frozen CLS token")
        for name in (
            "clean_input_sha256",
            "intervention_input_sha256",
            "runtime_model_binding_sha256",
        ):
            if not re.fullmatch(r"[0-9a-f]{64}", getattr(self, name)):
                raise ValueError(f"{name} must be a lowercase SHA-256")

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "target_kind": self.target_kind,
            "block_index": self.block_index,
            "head_index": self.head_index,
            "query_token": self.query_token,
            "clean_input_sha256": self.clean_input_sha256,
            "intervention_input_sha256": self.intervention_input_sha256,
            "runtime_model_binding_sha256": self.runtime_model_binding_sha256,
            "fixed_clean_attention": True,
            "frozen_direction": True,
        }


class StageCValuePathConvergenceError(StageCValuePathError):
    """Raised after the complete frozen IG schedule fails conservation.

    ``ig_result`` is retained verbatim so callers can audit all attempts even
    though no :class:`MethodResult` is emitted.
    """

    def __init__(self, message: str, ig_result: IGResult) -> None:
        super().__init__(message)
        self.ig_result = ig_result
        self.result = ig_result
        self.attempts = ig_result.attempts


# A slightly more explicit spelling is kept as a public alias for callers that
# distinguish this failure from later Stage C convergence notions.
StageCValuePathIGConvergenceError = StageCValuePathConvergenceError


def _strict_index(value: Any, *, name: str, upper_bound: int) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer (not bool)")
    if value < 0 or value >= upper_bound:
        raise StageCValuePathError(
            f"{name} must be in [0, {upper_bound - 1}], got {value}"
        )
    return value


def _prebound_sha256(value: str | None, *, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{name} must be a lowercase SHA-256 or None")
    return value


@dataclass(frozen=True, order=True)
class HeadCandidate:
    """One explicit canonical block/head candidate."""

    block_index: int
    head_index: int

    def __post_init__(self) -> None:
        _strict_index(
            self.block_index, name="block_index", upper_bound=MODEL_BLOCKS
        )
        _strict_index(
            self.head_index, name="head_index", upper_bound=HEADS_PER_BLOCK
        )

    @property
    def head(self) -> int:
        return self.head_index

    @property
    def block(self) -> int:
        return self.block_index

    def to_dict(self) -> dict[str, int]:
        return {
            "block_index": self.block_index,
            "head_index": self.head_index,
        }


def _coerce_candidate(value: Any, *, context: str) -> HeadCandidate:
    if isinstance(value, HeadCandidate):
        return value
    if isinstance(value, Mapping) and set(value) == {"block_index", "head_index"}:
        return HeadCandidate(value["block_index"], value["head_index"])
    if (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes, bytearray))
        and len(value) == 2
    ):
        return HeadCandidate(value[0], value[1])
    raise TypeError(
        f"{context} must be HeadCandidate, a (block, head) pair, or an exact "
        "block_index/head_index mapping"
    )


def all_head_candidates() -> tuple[HeadCandidate, ...]:
    """Return all 144 candidates in block-major, head-minor order."""

    return tuple(
        HeadCandidate(block_index, head_index)
        for block_index in range(MODEL_BLOCKS)
        for head_index in range(HEADS_PER_BLOCK)
    )


def _candidate_tuple(
    candidates: Sequence[HeadCandidate] | Any, *, allow_empty: bool
) -> tuple[HeadCandidate, ...]:
    if isinstance(candidates, (str, bytes, bytearray, torch.Tensor)):
        raise TypeError("candidates must be an explicit iterable of candidates")
    try:
        raw = tuple(candidates)
    except TypeError as exc:
        raise TypeError(
            "candidates must be an explicit iterable of candidates"
        ) from exc
    if not raw and not allow_empty:
        raise StageCValuePathError("candidate selection must not be empty")
    normalized = tuple(
        _coerce_candidate(value, context=f"candidates[{index}]")
        for index, value in enumerate(raw)
    )
    if len(set(normalized)) != len(normalized):
        raise StageCValuePathError("candidate selection contains duplicates")
    return normalized


def chunk_head_candidates(
    candidates: Sequence[HeadCandidate] | Any, chunk_size: int
) -> tuple[tuple[HeadCandidate, ...], ...]:
    """Validate and divide an explicit selection without reordering it."""

    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer (not bool)")
    normalized = _candidate_tuple(candidates, allow_empty=True)
    return tuple(
        normalized[start : start + chunk_size]
        for start in range(0, len(normalized), chunk_size)
    )


def _owned_tensor(value: torch.Tensor) -> torch.Tensor:
    return value.detach().clone()


class _RuntimeIntegrityGuard(AbstractContextManager[tuple[tuple[Any, ...], ...]]):
    """Diagnose cheap live-state drift on success and exceptions."""

    def __init__(
        self,
        model: nn.Module,
        caller_inputs: torch.Tensor,
        owned_inputs: torch.Tensor,
        *,
        context: str,
        target: "FrozenHeadDirectionTarget | None" = None,
    ) -> None:
        self.model = model
        self.caller_inputs = caller_inputs
        self.owned_inputs = owned_inputs
        self.context = context
        self.target = target
        self.model_signature = _model_state_signature(model)
        self.model_lifecycle_signature = model_live_signature(
            model, include_lifecycle=True
        )
        self.caller_input_state = CheapTensorSnapshot.capture(caller_inputs)
        self.owned_input_state = CheapTensorSnapshot.capture(owned_inputs)
        self.target_identity = (
            None if target is None else _target_identity_snapshot(target)
        )

    def __enter__(self) -> tuple[tuple[Any, ...], ...]:
        return self.model_signature

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        violations: list[str] = []
        if _model_state_signature(self.model) != self.model_signature:
            violations.append("model parameter/buffer state")
        if (
            model_live_signature(self.model, include_lifecycle=True)
            != self.model_lifecycle_signature
        ):
            violations.append("model lifecycle state")
        if not self.caller_input_state.verify(self.caller_inputs).matched:
            violations.append("caller clean-input tensor")
        if not self.owned_input_state.verify(self.owned_inputs).matched:
            violations.append("private clean-input clone")
        if self.target is not None and (
            _ISSUED_TARGETS.get(self.target) != self.target_identity
            or _target_identity_snapshot(self.target) != self.target_identity
        ):
            violations.append("frozen head target")
        if violations:
            integrity_error = StageCValuePathError(
                f"{self.context} mutated " + ", ".join(violations)
            )
            if exc is not None:
                raise integrity_error from exc
            raise integrity_error
        return False


@dataclass(frozen=True)
class HeadDirectionResolution:
    """Pure endpoint resolution for one fixed-attention head direction."""

    candidate: HeadCandidate
    status: str
    clean_attention: torch.Tensor
    clean_values: torch.Tensor
    baseline_values: torch.Tensor
    clean_head_output: torch.Tensor
    baseline_head_output: torch.Tensor
    direction_delta: torch.Tensor
    direction_norm: float
    direction_unit: torch.Tensor | None
    direction_threshold: float
    na: StructuredNA | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, HeadCandidate):
            raise TypeError("candidate must be a HeadCandidate")
        if self.status not in {STATUS_OK, DEGENERATE_DIRECTION}:
            raise ValueError(
                f"status must be {STATUS_OK!r} or {DEGENERATE_DIRECTION!r}"
            )
        tensor_fields = (
            "clean_attention",
            "clean_values",
            "baseline_values",
            "clean_head_output",
            "baseline_head_output",
            "direction_delta",
        )
        for name in tensor_fields:
            value = getattr(self, name)
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
            if value.dtype is not torch.float32:
                raise TypeError(f"{name} must have float32 dtype")
            object.__setattr__(self, name, _owned_tensor(value))
        if self.direction_unit is not None:
            if not isinstance(self.direction_unit, torch.Tensor):
                raise TypeError("direction_unit must be a Tensor or None")
            if self.direction_unit.dtype is not torch.float32:
                raise TypeError("direction_unit must have float32 dtype")
            object.__setattr__(
                self, "direction_unit", _owned_tensor(self.direction_unit)
            )
        if isinstance(self.direction_norm, bool) or not isinstance(
            self.direction_norm, (int, float)
        ):
            raise TypeError("direction_norm must be numeric")
        if isinstance(self.direction_threshold, bool) or not isinstance(
            self.direction_threshold, (int, float)
        ):
            raise TypeError("direction_threshold must be numeric")
        object.__setattr__(self, "direction_norm", float(self.direction_norm))
        object.__setattr__(
            self, "direction_threshold", float(self.direction_threshold)
        )
        self._validate_math()

    def _validate_math(self) -> None:
        attention = self.clean_attention
        clean = self.clean_values
        baseline = self.baseline_values
        if attention.ndim != 1 or attention.numel() == 0:
            raise StageCValuePathError(
                "clean_attention must have non-empty shape [tokens]"
            )
        if clean.ndim != 2 or baseline.shape != clean.shape:
            raise StageCValuePathError(
                "Value endpoints must have identical [tokens,head_dim] shapes"
            )
        if clean.shape[0] != attention.shape[0] or clean.shape[1] <= 0:
            raise StageCValuePathError(
                "attention and Value endpoints have incompatible geometry"
            )
        head_dim = clean.shape[1]
        for name in (
            "clean_head_output",
            "baseline_head_output",
            "direction_delta",
        ):
            if getattr(self, name).shape != (head_dim,):
                raise StageCValuePathError(
                    f"{name} must have shape ({head_dim},)"
                )
        if bool((attention[torch.isfinite(attention)] < 0).any().item()):
            raise StageCValuePathError(
                "clean attention contains a negative finite value"
            )
        endpoint_finite = bool(
            torch.isfinite(attention).all().item()
            and torch.isfinite(clean).all().item()
            and torch.isfinite(baseline).all().item()
        )
        if endpoint_finite:
            attention_mass = float(attention.to(torch.float64).sum().item())
            if not math.isclose(attention_mass, 1.0, abs_tol=1e-6, rel_tol=1e-6):
                raise StageCValuePathError(
                    "clean attention row must have finite nonzero mass normalized "
                    "to one"
                )
            expected_clean = torch.sum(attention[:, None] * clean, dim=0)
            expected_baseline = torch.sum(attention[:, None] * baseline, dim=0)
            expected_delta = expected_clean - expected_baseline
            for actual, expected, name in (
                (self.clean_head_output, expected_clean, "clean_head_output"),
                (
                    self.baseline_head_output,
                    expected_baseline,
                    "baseline_head_output",
                ),
                (self.direction_delta, expected_delta, "direction_delta"),
            ):
                if not torch.allclose(actual, expected, atol=1e-7, rtol=1e-6):
                    raise StageCValuePathError(
                        f"{name} is inconsistent with the frozen endpoint formula"
                    )
            expected_norm = float(torch.linalg.vector_norm(expected_delta).item())
            expected_threshold = DIRECTION_RELATIVE_THRESHOLD * max(
                DIRECTION_SCALE_FLOOR,
                float(torch.linalg.vector_norm(expected_clean).item()),
                float(torch.linalg.vector_norm(expected_baseline).item()),
            )
            if not math.isclose(
                self.direction_norm, expected_norm, abs_tol=1e-12, rel_tol=1e-6
            ):
                raise StageCValuePathError(
                    "direction_norm is inconsistent with direction_delta"
                )
            if not math.isclose(
                self.direction_threshold,
                expected_threshold,
                abs_tol=1e-12,
                rel_tol=1e-6,
            ):
                raise StageCValuePathError(
                    "direction_threshold is inconsistent with endpoint norms"
                )
            if self.status == STATUS_OK:
                if expected_norm <= expected_threshold:
                    raise StageCValuePathError(
                        "an OK direction must be strictly above its threshold"
                    )
                unit = self.direction_unit
                if unit is None or unit.shape != (head_dim,) or not bool(
                    torch.isfinite(unit).all().item()
                ):
                    raise StageCValuePathError(
                        "an OK direction requires one finite head-width unit vector"
                    )
                expected_unit = expected_delta / expected_norm
                if not torch.allclose(unit, expected_unit, atol=1e-7, rtol=1e-6):
                    raise StageCValuePathError(
                        "direction_unit is inconsistent with direction_delta/r"
                    )
                if not math.isclose(
                    float(torch.linalg.vector_norm(unit).item()),
                    1.0,
                    abs_tol=1e-6,
                    rel_tol=1e-6,
                ):
                    raise StageCValuePathError(
                        "direction_unit must have L2 norm one"
                    )
            elif expected_norm > expected_threshold:
                raise StageCValuePathError(
                    "a finite degenerate direction must be at or below threshold"
                )
        elif self.status != DEGENERATE_DIRECTION:
            raise StageCValuePathError(
                "non-finite endpoint data must be DEGENERATE_DIRECTION"
            )
        if self.status == STATUS_OK:
            if self.direction_unit is None or self.na is not None:
                raise ValueError(
                    "an OK direction requires direction_unit and no NA state"
                )
        else:
            if self.direction_unit is not None:
                raise ValueError(
                    "a degenerate direction must not expose a unit direction"
                )
            if not isinstance(self.na, StructuredNA) or (
                self.na.reason is not NAReason.DEGENERATE_DIRECTION
            ):
                raise ValueError(
                    "a degenerate direction requires DEGENERATE_DIRECTION NA"
                )

    @property
    def block_index(self) -> int:
        return self.candidate.block_index

    @property
    def head_index(self) -> int:
        return self.candidate.head_index

    @property
    def A(self) -> torch.Tensor:
        return self.clean_attention

    @property
    def fixed_attention(self) -> torch.Tensor:
        """Alias for the fixed clean CLS attention row."""

        return self.clean_attention

    @property
    def g_clean(self) -> torch.Tensor:
        return self.clean_head_output

    @property
    def g_baseline(self) -> torch.Tensor:
        return self.baseline_head_output

    @property
    def d(self) -> torch.Tensor:
        return self.direction_delta

    @property
    def r(self) -> float:
        return self.direction_norm

    @property
    def u(self) -> torch.Tensor | None:
        return self.direction_unit

    @property
    def tau(self) -> float:
        return self.direction_threshold

    @property
    def reason(self) -> str | None:
        return None if self.na is None else DEGENERATE_DIRECTION


def resolve_head_direction_from_endpoints(
    attention_row: torch.Tensor | None = None,
    clean_values: torch.Tensor | None = None,
    baseline_values: torch.Tensor | None = None,
    *,
    block_index: int = 0,
    head_index: int = 0,
    clean_attention: torch.Tensor | None = None,
) -> HeadDirectionResolution:
    """Resolve ``A``, ``g(x)``, ``g(x0)``, ``d``, ``r``, and ``u``.

    Non-finite endpoint data is represented as structured direction
    degeneracy.  Structural errors, non-float32 data, device disagreement, or
    a finite negative attention entry fail closed with an exception.
    """

    if clean_attention is not None:
        if attention_row is not None:
            raise TypeError(
                "provide attention_row or clean_attention, not both"
            )
        attention_row = clean_attention
    candidate = HeadCandidate(block_index, head_index)
    for value, name in (
        (attention_row, "attention_row"),
        (clean_values, "clean_values"),
        (baseline_values, "baseline_values"),
    ):
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if value.dtype is not torch.float32:
            raise TypeError(f"{name} must have float32 dtype, got {value.dtype}")
    assert attention_row is not None
    assert clean_values is not None
    assert baseline_values is not None
    if attention_row.ndim != 1 or attention_row.numel() == 0:
        raise StageCValuePathError(
            "attention_row must have non-empty shape [tokens]"
        )
    if clean_values.ndim != 2 or baseline_values.ndim != 2:
        raise StageCValuePathError(
            "clean_values and baseline_values must have shape [tokens,head_dim]"
        )
    if clean_values.shape != baseline_values.shape:
        raise StageCValuePathError(
            "clean_values and baseline_values must have identical shapes"
        )
    if clean_values.shape[0] != attention_row.shape[0]:
        raise StageCValuePathError(
            "attention and Value endpoints must have the same token count"
        )
    if clean_values.shape[1] <= 0:
        raise StageCValuePathError("head_dim must be positive")
    if not (
        attention_row.device == clean_values.device == baseline_values.device
    ):
        raise StageCValuePathError(
            "attention and Value endpoints must share one device"
        )

    attention = _owned_tensor(attention_row)
    clean = _owned_tensor(clean_values)
    baseline = _owned_tensor(baseline_values)
    finite_attention = attention[torch.isfinite(attention)]
    if bool((finite_attention < 0).any().item()):
        raise StageCValuePathError(
            "clean attention contains a negative finite value"
        )
    if bool(torch.isfinite(attention).all().item()):
        attention_mass = float(attention.to(torch.float64).sum().item())
        if not math.isclose(attention_mass, 1.0, abs_tol=1e-6, rel_tol=1e-6):
            raise StageCValuePathError(
                "clean attention row must have finite nonzero mass normalized "
                "to one"
            )

    g_clean = torch.sum(attention[:, None] * clean, dim=0)
    # This is deliberately the *clean* A applied to natural baseline V.
    g_baseline = torch.sum(attention[:, None] * baseline, dim=0)
    direction_delta = g_clean - g_baseline
    clean_norm = float(torch.linalg.vector_norm(g_clean).item())
    baseline_norm = float(torch.linalg.vector_norm(g_baseline).item())
    direction_norm = float(torch.linalg.vector_norm(direction_delta).item())

    finite = bool(
        torch.isfinite(attention).all().item()
        and torch.isfinite(clean).all().item()
        and torch.isfinite(baseline).all().item()
        and torch.isfinite(g_clean).all().item()
        and torch.isfinite(g_baseline).all().item()
        and torch.isfinite(direction_delta).all().item()
        and math.isfinite(clean_norm)
        and math.isfinite(baseline_norm)
        and math.isfinite(direction_norm)
    )
    if finite:
        direction_threshold = DIRECTION_RELATIVE_THRESHOLD * max(
            DIRECTION_SCALE_FLOOR, clean_norm, baseline_norm
        )
    elif any(
        math.isinf(value)
        for value in (clean_norm, baseline_norm, direction_norm)
    ):
        direction_threshold = math.inf
    else:
        direction_threshold = math.nan

    if not finite:
        na = StructuredNA(
            NAReason.DEGENERATE_DIRECTION,
            "non-finite clean attention, Value endpoint, or derived direction",
        )
        status = DEGENERATE_DIRECTION
        unit = None
    elif direction_norm <= direction_threshold:
        na = StructuredNA(
            NAReason.DEGENERATE_DIRECTION,
            "direction norm is at or below the frozen relative threshold",
        )
        status = DEGENERATE_DIRECTION
        unit = None
    else:
        unit = _owned_tensor(direction_delta / direction_norm)
        if not bool(torch.isfinite(unit).all().item()):
            na = StructuredNA(
                NAReason.DEGENERATE_DIRECTION,
                "unit direction is non-finite",
            )
            status = DEGENERATE_DIRECTION
            unit = None
        else:
            na = None
            status = STATUS_OK

    return HeadDirectionResolution(
        candidate=candidate,
        status=status,
        clean_attention=attention,
        clean_values=clean,
        baseline_values=baseline,
        clean_head_output=g_clean,
        baseline_head_output=g_baseline,
        direction_delta=direction_delta,
        direction_norm=direction_norm,
        direction_unit=unit,
        direction_threshold=direction_threshold,
        na=na,
    )


@dataclass(frozen=True)
class MatchedTokenContribution:
    """All-token analytic decomposition of one resolved head target."""

    candidate: HeadCandidate
    status: str
    token_contributions: torch.Tensor | None
    cls_contribution: float | None
    patch_contributions: torch.Tensor | None
    patch_map: torch.Tensor | None
    direction_norm: float | None
    conservation_sum: float | None
    conservation_error: float | None
    conservation_tolerance: float | None
    patch_support: float | None
    positive_patch_evidence_fraction: float | None
    degeneracy_reasons: tuple[str, ...]
    na: StructuredNA | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, HeadCandidate):
            raise TypeError("candidate must be a HeadCandidate")
        object.__setattr__(
            self, "degeneracy_reasons", tuple(self.degeneracy_reasons)
        )
        for reason in self.degeneracy_reasons:
            if reason not in {
                DEGENERATE_DIRECTION,
                DEGENERATE_PATCH_SUPPORT,
                DEGENERATE_POSITIVE_PATCH_EVIDENCE,
            }:
                raise ValueError(f"unknown contribution degeneracy {reason!r}")
        if self.status == DEGENERATE_DIRECTION:
            if (
                not isinstance(self.na, StructuredNA)
                or self.na.reason is not NAReason.DEGENERATE_DIRECTION
            ):
                raise ValueError("degenerate contributions require structured NA")
            if self.degeneracy_reasons != (DEGENERATE_DIRECTION,):
                raise ValueError(
                    "degenerate contributions require exactly the direction flag"
                )
            for name in (
                "token_contributions",
                "cls_contribution",
                "patch_contributions",
                "patch_map",
                "direction_norm",
                "conservation_sum",
                "conservation_error",
                "conservation_tolerance",
                "patch_support",
                "positive_patch_evidence_fraction",
            ):
                if getattr(self, name) is not None:
                    raise ValueError(
                        "degenerate contributions cannot expose fabricated values"
                    )
            return
        if self.status != STATUS_OK or self.na is not None:
            raise ValueError("matched contribution status is inconsistent")
        expected_shapes = {
            "token_contributions": (TOKENS,),
            "patch_contributions": (PATCH_TOKENS,),
            "patch_map": PATCH_SHAPE,
        }
        for name, shape in expected_shapes.items():
            value = getattr(self, name)
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a Tensor for an OK result")
            if value.dtype is not torch.float32 or tuple(value.shape) != shape:
                raise StageCValuePathError(
                    f"{name} must be float32 with shape {shape}"
                )
            if not bool(torch.isfinite(value).all().item()):
                raise StageCValuePathError(f"{name} must be finite")
            object.__setattr__(self, name, _owned_tensor(value))
        for name in (
            "cls_contribution",
            "direction_norm",
            "conservation_sum",
            "conservation_error",
            "conservation_tolerance",
            "patch_support",
            "positive_patch_evidence_fraction",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be finite numeric")
            if not math.isfinite(float(value)):
                raise StageCValuePathError(f"{name} must be finite")
            object.__setattr__(self, name, float(value))
        token = self.token_contributions
        patch = self.patch_contributions
        patch_map = self.patch_map
        assert token is not None and patch is not None and patch_map is not None
        if not torch.equal(patch, token[1:]):
            raise StageCValuePathError(
                "patch_contributions must equal token_contributions[1:]"
            )
        if not torch.equal(patch_map.reshape(-1), patch):
            raise StageCValuePathError(
                "patch_map must preserve original patch-token order"
            )
        expected_cls = float(token[CLS_TOKEN_INDEX].item())
        expected_sum = float(token.to(torch.float64).sum().item())
        if self.cls_contribution != expected_cls:
            raise StageCValuePathError(
                "cls_contribution must equal token_contributions[0]"
            )
        if not math.isclose(
            self.conservation_sum, expected_sum, abs_tol=1e-12, rel_tol=1e-7
        ):
            raise StageCValuePathError(
                "conservation_sum must equal the all-token sum"
            )
        total_l1 = float(token.abs().to(torch.float64).sum().item())
        if total_l1 <= 0.0:
            raise StageCValuePathError(
                "matched token L1 denominator must be finite and positive"
            )
        expected_support = float(patch.abs().to(torch.float64).sum().item()) / total_l1
        expected_positive = float(
            patch.clamp_min(0).to(torch.float64).sum().item()
        ) / total_l1
        if not math.isclose(self.patch_support, expected_support, abs_tol=1e-12, rel_tol=1e-7):
            raise StageCValuePathError(
                "patch_support is inconsistent with token contributions"
            )
        if not math.isclose(
            self.positive_patch_evidence_fraction,
            expected_positive,
            abs_tol=1e-12,
            rel_tol=1e-7,
        ):
            raise StageCValuePathError(
                "positive patch evidence is inconsistent with token contributions"
            )
        expected_flags: list[str] = []
        if expected_support < PATCH_SUPPORT_MINIMUM:
            expected_flags.append(DEGENERATE_PATCH_SUPPORT)
        if expected_positive < POSITIVE_PATCH_EVIDENCE_MINIMUM:
            expected_flags.append(DEGENERATE_POSITIVE_PATCH_EVIDENCE)
        if self.degeneracy_reasons != tuple(expected_flags):
            raise StageCValuePathError(
                "matched contribution degeneracy flags are inconsistent"
            )
        if self.conservation_error < 0.0 or self.conservation_tolerance < 0.0:
            raise StageCValuePathError(
                "matched conservation error/tolerance must be non-negative"
            )
        expected_tolerance = TOKEN_CONSERVATION_ATOL + (
            TOKEN_CONSERVATION_RTOL * abs(self.direction_norm)
        )
        expected_error = abs(expected_sum - self.direction_norm)
        if not math.isclose(
            self.conservation_tolerance,
            expected_tolerance,
            abs_tol=1e-15,
            rel_tol=1e-12,
        ):
            raise StageCValuePathError(
                "conservation_tolerance does not use the frozen formula"
            )
        if not math.isclose(
            self.conservation_error, expected_error, abs_tol=1e-12, rel_tol=1e-7
        ):
            raise StageCValuePathError(
                "conservation_error is inconsistent with token sum and tolerance"
            )
        if self.conservation_error > self.conservation_tolerance:
            raise StageCValuePathError(
                "matched token conservation exceeds the frozen tolerance"
            )

    @property
    def contributions(self) -> torch.Tensor | None:
        return self.token_contributions

    @property
    def flags(self) -> tuple[str, ...]:
        return self.degeneracy_reasons

    @property
    def positive_patch_fraction(self) -> float | None:
        return self.positive_patch_evidence_fraction

    @property
    def token_sum(self) -> float | None:
        return self.conservation_sum

    @property
    def conservation_abs_error(self) -> float | None:
        return self.conservation_error


def _float_identity(value: float | None) -> str | None:
    return None if value is None else float(value).hex()


def _resolution_content_identity(
    resolution: HeadDirectionResolution,
) -> tuple[Any, ...]:
    return (
        resolution.candidate,
        resolution.status,
        _tensor_sha256(resolution.clean_attention),
        _tensor_sha256(resolution.clean_values),
        _tensor_sha256(resolution.baseline_values),
        _tensor_sha256(resolution.clean_head_output),
        _tensor_sha256(resolution.baseline_head_output),
        _tensor_sha256(resolution.direction_delta),
        _float_identity(resolution.direction_norm),
        None
        if resolution.direction_unit is None
        else _tensor_sha256(resolution.direction_unit),
        _float_identity(resolution.direction_threshold),
        None if resolution.na is None else resolution.na.to_dict(),
    )


def _resolution_live_identity(
    resolution: HeadDirectionResolution,
) -> tuple[Any, ...]:
    """Content-free identity used by method and scalar hot paths."""

    def tensor_state(value: torch.Tensor | None) -> tuple[Any, ...] | None:
        return (
            None
            if value is None
            else CheapTensorSnapshot.capture(value).signature
        )

    return (
        resolution.candidate,
        resolution.status,
        tensor_state(resolution.clean_attention),
        tensor_state(resolution.clean_values),
        tensor_state(resolution.baseline_values),
        tensor_state(resolution.clean_head_output),
        tensor_state(resolution.baseline_head_output),
        tensor_state(resolution.direction_delta),
        _float_identity(resolution.direction_norm),
        tensor_state(resolution.direction_unit),
        _float_identity(resolution.direction_threshold),
        None if resolution.na is None else resolution.na.to_dict(),
    )


@dataclass(frozen=True, init=False, eq=False)
class FrozenHeadDirectionTarget:
    """Resolver-issued, model/input/state-bound C4 target."""

    candidate: HeadCandidate
    resolution: HeadDirectionResolution
    clean_value_max_abs_error: float | None
    clean_replay_max_abs_error: float | None
    clean_model_output_max_abs_error: float | None
    baseline_replay_max_abs_error: float | None
    clean_input_sha256: str
    clean_input_shape: tuple[int, int, int, int]
    runtime_model_binding_sha256: str
    _fixed_attention: torch.Tensor = field(repr=False, compare=False)
    _model_ref: weakref.ReferenceType[nn.Module] = field(
        repr=False, compare=False
    )
    _model_state_signature: tuple[tuple[Any, ...], ...] = field(
        repr=False, compare=False
    )
    _clean_input_ref: weakref.ReferenceType[torch.Tensor] = field(
        repr=False, compare=False
    )
    _clean_input_live_signature: tuple[Any, ...] = field(
        repr=False, compare=False
    )
    _content_identity: tuple[Any, ...] = field(repr=False, compare=False)
    _serialized_target_json: str = field(repr=False, compare=False)
    _issuer_nonce: bytes = field(repr=False, compare=False)

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise TypeError(
            "FrozenHeadDirectionTarget cannot be constructed directly; use "
            "resolve_fixed_clean_attention_head_direction"
        )

    @classmethod
    def _create(
        cls,
        *,
        model: nn.Module,
        model_state_signature: tuple[tuple[Any, ...], ...],
        clean_inputs: torch.Tensor,
        clean_input_context: torch.Tensor | None = None,
        candidate: HeadCandidate,
        resolution: HeadDirectionResolution,
        fixed_attention: torch.Tensor,
        clean_value_max_abs_error: float | None,
        clean_replay_max_abs_error: float | None,
        clean_model_output_max_abs_error: float | None,
        baseline_replay_max_abs_error: float | None,
        issuer_secret: bytes,
    ) -> "FrozenHeadDirectionTarget":
        if issuer_secret is not _TARGET_ISSUER_SECRET:
            raise TypeError("head-direction targets can only be issued by the resolver")
        instance = object.__new__(cls)
        values = {
            "candidate": candidate,
            "resolution": resolution,
            "clean_value_max_abs_error": clean_value_max_abs_error,
            "clean_replay_max_abs_error": clean_replay_max_abs_error,
            "clean_model_output_max_abs_error": clean_model_output_max_abs_error,
            "baseline_replay_max_abs_error": baseline_replay_max_abs_error,
            "clean_input_sha256": _tensor_sha256(clean_inputs),
            "clean_input_shape": tuple(clean_inputs.shape),
            "runtime_model_binding_sha256": _model_binding_sha256(
                model_state_signature
            ),
            "_fixed_attention": _owned_tensor(fixed_attention),
            "_model_ref": weakref.ref(model),
            "_model_state_signature": model_state_signature,
            "_clean_input_ref": weakref.ref(
                clean_inputs if clean_input_context is None else clean_input_context
            ),
            "_clean_input_live_signature": CheapTensorSnapshot.capture(
                clean_inputs if clean_input_context is None else clean_input_context
            ).signature,
            "_issuer_nonce": token_bytes(32),
            "_serialized_target_json": "",
        }
        for name, value in values.items():
            object.__setattr__(instance, name, value)
        object.__setattr__(
            instance, "_content_identity",
            (
                _resolution_content_identity(instance.resolution),
                _tensor_sha256(instance._fixed_attention),
            ),
        )
        instance.__post_init__()
        object.__setattr__(
            instance,
            "_serialized_target_json",
            json.dumps(
                instance.to_dict(),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ),
        )
        _ISSUED_TARGETS[instance] = _target_identity_snapshot(instance)
        return instance

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, HeadCandidate):
            raise TypeError("candidate must be a HeadCandidate")
        if not isinstance(self.resolution, HeadDirectionResolution):
            raise TypeError("resolution must be HeadDirectionResolution")
        if self.resolution.candidate != self.candidate:
            raise StageCValuePathError(
                "target candidate does not match its direction resolution"
            )
        if self.resolution.clean_attention.shape != (TOKENS,):
            raise StageCValuePathError(
                f"production attention row must have shape ({TOKENS},)"
            )
        if self.resolution.clean_values.shape != (TOKENS, HEAD_DIM):
            raise StageCValuePathError(
                f"production Value endpoints must have shape ({TOKENS},{HEAD_DIM})"
            )
        fixed = self._fixed_attention
        if not isinstance(fixed, torch.Tensor) or fixed.dtype is not torch.float32:
            raise TypeError("fixed selected-head attention must be float32")
        if fixed.shape != (1, TOKENS, TOKENS):
            raise StageCValuePathError(
                "fixed selected-head attention must have shape [1,197,197]"
            )
        object.__setattr__(self, "_fixed_attention", _owned_tensor(fixed))
        endpoint_finite = bool(
            torch.isfinite(self.resolution.clean_attention).all().item()
            and torch.isfinite(self.resolution.clean_values).all().item()
            and torch.isfinite(self.resolution.baseline_values).all().item()
        )
        if endpoint_finite:
            for name in (
                "clean_value_max_abs_error",
                "clean_replay_max_abs_error",
                "clean_model_output_max_abs_error",
                "baseline_replay_max_abs_error",
            ):
                value = getattr(self, name)
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise TypeError(f"{name} must be numeric for an OK target")
                if not math.isfinite(float(value)) or not (
                    float(value) < ENDPOINT_PARITY_MAX_ABS
                ):
                    raise StageCValuePathError(
                        f"{name} must be strictly below {ENDPOINT_PARITY_MAX_ABS}"
                    )
                object.__setattr__(self, name, float(value))
        elif any(
            value is not None
            for value in (
                self.clean_value_max_abs_error,
                self.clean_replay_max_abs_error,
                self.clean_model_output_max_abs_error,
                self.baseline_replay_max_abs_error,
            )
        ):
            raise StageCValuePathError(
                "a non-finite degenerate target must not claim finite endpoint parity"
            )
        if not re.fullmatch(r"[0-9a-f]{64}", self.clean_input_sha256):
            raise ValueError("clean_input_sha256 must be a lowercase SHA-256")
        if self.clean_input_shape != (1, 3, 224, 224):
            raise StageCValuePathError(
                "clean_input_shape must be exactly (1,3,224,224)"
            )
        if not re.fullmatch(
            r"[0-9a-f]{64}", self.runtime_model_binding_sha256
        ):
            raise ValueError(
                "runtime_model_binding_sha256 must be a lowercase SHA-256"
            )
        if not isinstance(self._model_ref, weakref.ReferenceType):
            raise TypeError("target has an invalid live-model reference")
        if not isinstance(self._clean_input_ref, weakref.ReferenceType):
            raise TypeError("target has an invalid clean-input reference")
        if not isinstance(self._clean_input_live_signature, tuple):
            raise TypeError("target has an invalid clean-input live signature")
        if not isinstance(self._model_state_signature, tuple):
            raise TypeError("target has an invalid model-state signature")
        if not isinstance(self._content_identity, tuple):
            raise TypeError("target has an invalid content identity")
        if not isinstance(self._issuer_nonce, bytes) or len(self._issuer_nonce) != 32:
            raise TypeError("target has an invalid issuer nonce")

    @property
    def status(self) -> str:
        return self.resolution.status

    @property
    def na(self) -> StructuredNA | None:
        return self.resolution.na

    @property
    def block_index(self) -> int:
        return self.candidate.block_index

    @property
    def head_index(self) -> int:
        return self.candidate.head_index

    @property
    def clean_attention(self) -> torch.Tensor:
        return self.resolution.clean_attention

    @property
    def clean_values(self) -> torch.Tensor:
        return self.resolution.clean_values

    @property
    def baseline_values(self) -> torch.Tensor:
        return self.resolution.baseline_values

    @property
    def clean_head_output(self) -> torch.Tensor:
        return self.resolution.clean_head_output

    @property
    def baseline_head_output(self) -> torch.Tensor:
        return self.resolution.baseline_head_output

    @property
    def direction_delta(self) -> torch.Tensor:
        return self.resolution.direction_delta

    @property
    def direction_norm(self) -> float:
        return self.resolution.direction_norm

    @property
    def direction_unit(self) -> torch.Tensor | None:
        return self.resolution.direction_unit

    @property
    def direction_threshold(self) -> float:
        return self.resolution.direction_threshold

    @property
    def fixed_attention(self) -> torch.Tensor:
        return self._fixed_attention

    @property
    def clean_endpoint_parity_max_abs_error(self) -> float | None:
        return self.clean_model_output_max_abs_error

    @property
    def baseline_endpoint_parity_max_abs_error(self) -> float | None:
        return self.baseline_replay_max_abs_error

    # Formula-oriented aliases mirror HeadDirectionResolution.
    A = property(lambda self: self.clean_attention)
    g_clean = property(lambda self: self.clean_head_output)
    g_baseline = property(lambda self: self.baseline_head_output)
    d = property(lambda self: self.direction_delta)
    r = property(lambda self: self.direction_norm)
    u = property(lambda self: self.direction_unit)
    tau = property(lambda self: self.direction_threshold)

    def to_dict(self) -> dict[str, Any]:
        if self._serialized_target_json:
            result = json.loads(self._serialized_target_json)
            if not isinstance(result, dict):
                raise RuntimeError("cached target JSON is not an object")
            return result
        resolution = self.resolution
        result: dict[str, Any] = {
            "target_kind": TARGET_REF,
            "candidate": self.candidate.to_dict(),
            "query_token": CLS_TOKEN_INDEX,
            "head_dim": HEAD_DIM,
            "status": self.status,
            "attention_source": "clean_real_forward_attention_used",
            "baseline_uses_clean_fixed_attention": True,
            "direction_formula": "u=(g_clean-g_baseline)/l2_norm",
            "direction_norm": _finite_json_number(resolution.direction_norm),
            "direction_threshold": _finite_json_number(
                resolution.direction_threshold
            ),
            "clean_attention": _tensor_descriptor(resolution.clean_attention),
            "clean_values": _tensor_descriptor(resolution.clean_values),
            "baseline_values": _tensor_descriptor(resolution.baseline_values),
            "direction_delta": _tensor_descriptor(resolution.direction_delta),
            "direction_unit": None
            if resolution.direction_unit is None
            else _tensor_descriptor(resolution.direction_unit),
            "detached_and_frozen_before_ig": True,
            "clean_value_max_abs_error": self.clean_value_max_abs_error,
            "clean_replay_max_abs_error": self.clean_replay_max_abs_error,
            "clean_model_output_max_abs_error": (
                self.clean_model_output_max_abs_error
            ),
            "baseline_replay_max_abs_error": self.baseline_replay_max_abs_error,
            "endpoint_parity_requirement": "strictly_less_than_1e-6",
            "clean_input_sha256": self.clean_input_sha256,
            "clean_input_shape": list(self.clean_input_shape),
            "runtime_model_binding": {
                "kind": "same_live_model_instance_and_tensor_state_versions",
                "sha256": self.runtime_model_binding_sha256,
                "is_canonical_model_artifact_hash": False,
            },
        }
        if resolution.na is not None:
            result["na"] = resolution.na.to_dict()
        return result


def _finite_json_number(value: float) -> float | None:
    converted = float(value)
    return converted if math.isfinite(converted) else None


def _tensor_descriptor(value: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": list(value.shape),
        "dtype": "float32",
        "sha256": _tensor_sha256(value),
        "finite": bool(torch.isfinite(value).all().item()),
    }


def _target_identity_snapshot(
    target: FrozenHeadDirectionTarget,
) -> tuple[Any, ...]:
    return (
        target._issuer_nonce,
        target.candidate,
        _resolution_live_identity(target.resolution),
        _float_identity(target.clean_value_max_abs_error),
        _float_identity(target.clean_replay_max_abs_error),
        _float_identity(target.clean_model_output_max_abs_error),
        _float_identity(target.baseline_replay_max_abs_error),
        target.clean_input_sha256,
        target.clean_input_shape,
        target.runtime_model_binding_sha256,
        CheapTensorSnapshot.capture(target._fixed_attention).signature,
        id(target._model_ref),
        target._model_state_signature,
        id(target._clean_input_ref),
        target._clean_input_live_signature,
        target._content_identity,
        target._serialized_target_json,
    )


def _require_untampered_target_identity(
    target: Any,
) -> FrozenHeadDirectionTarget:
    if not isinstance(target, FrozenHeadDirectionTarget):
        raise TypeError(
            "target must come from "
            "resolve_fixed_clean_attention_head_direction"
        )
    if _ISSUED_TARGETS.get(target) != _target_identity_snapshot(target):
        raise StageCValuePathError(
            "head-direction target was not issued by the resolver or was "
            "mutated after issue"
        )
    return target


def matched_token_contributions(
    resolution: HeadDirectionResolution | FrozenHeadDirectionTarget,
) -> MatchedTokenContribution:
    """Compute the analytic contribution of all 197 tokens."""

    if isinstance(resolution, FrozenHeadDirectionTarget):
        resolved = _require_untampered_target_identity(resolution).resolution
    elif isinstance(resolution, HeadDirectionResolution):
        resolved = resolution
    else:
        raise TypeError(
            "resolution must be HeadDirectionResolution or "
            "FrozenHeadDirectionTarget"
        )
    if resolved.status == DEGENERATE_DIRECTION:
        return MatchedTokenContribution(
            candidate=resolved.candidate,
            status=DEGENERATE_DIRECTION,
            token_contributions=None,
            cls_contribution=None,
            patch_contributions=None,
            patch_map=None,
            direction_norm=None,
            conservation_sum=None,
            conservation_error=None,
            conservation_tolerance=None,
            patch_support=None,
            positive_patch_evidence_fraction=None,
            degeneracy_reasons=(DEGENERATE_DIRECTION,),
            na=resolved.na,
        )
    if resolved.clean_attention.shape != (TOKENS,) or (
        resolved.clean_values.shape[0] != TOKENS
    ):
        raise StageCValuePathError(
            "matched token contribution requires exactly 197 tokens"
        )
    unit = resolved.direction_unit
    if unit is None:
        raise StageCValuePathError("OK direction unexpectedly has no unit vector")
    return _matched_token_contributions_from_values(
        candidate=resolved.candidate,
        clean_attention=resolved.clean_attention,
        intervention_values=resolved.clean_values,
        baseline_values=resolved.baseline_values,
        direction_unit=unit,
        conservation_reference=resolved.direction_norm,
    )


def _matched_token_contributions_from_values(
    *,
    candidate: HeadCandidate,
    clean_attention: torch.Tensor,
    intervention_values: torch.Tensor,
    baseline_values: torch.Tensor,
    direction_unit: torch.Tensor,
    conservation_reference: float,
) -> MatchedTokenContribution:
    if clean_attention.shape != (TOKENS,):
        raise StageCValuePathError(
            "matched token contribution requires exactly 197 tokens"
        )
    if clean_attention.dtype is not torch.float32:
        raise TypeError("clean_attention must be float32")
    if not bool(torch.isfinite(clean_attention).all().item()):
        raise StageCValuePathError("clean_attention must be finite")
    if direction_unit.ndim != 1:
        raise StageCValuePathError("direction_unit must be rank-1")
    head_width = int(direction_unit.shape[0])
    for name, value in (
        ("intervention_values", intervention_values),
        ("baseline_values", baseline_values),
    ):
        if value.shape != (TOKENS, head_width):
            raise StageCValuePathError(
                f"{name} must have shape ({TOKENS}, {head_width})"
            )
        if value.dtype is not torch.float32:
            raise TypeError(f"{name} must be float32")
        if not bool(torch.isfinite(value).all().item()):
            raise StageCValuePathError(f"{name} must be finite")
    if direction_unit.dtype is not torch.float32:
        raise TypeError("direction_unit must be float32")
    if not bool(torch.isfinite(direction_unit).all().item()):
        raise StageCValuePathError("direction_unit must be finite")
    if isinstance(conservation_reference, bool) or not isinstance(
        conservation_reference, (int, float)
    ):
        raise TypeError("conservation_reference must be finite numeric")
    conservation_target = float(conservation_reference)
    if not math.isfinite(conservation_target):
        raise StageCValuePathError("conservation_reference must be finite")
    value_delta = intervention_values - baseline_values
    projected = torch.sum(value_delta * direction_unit[None, :], dim=1)
    contributions = _owned_tensor(clean_attention * projected)
    if contributions.dtype is not torch.float32 or not bool(
        torch.isfinite(contributions).all().item()
    ):
        raise StageCValuePathError(
            "matched token contributions must be finite float32"
        )
    contribution_sum = float(
        torch.sum(contributions.to(torch.float64)).item()
    )
    conservation_error = abs(contribution_sum - conservation_target)
    conservation_tolerance = TOKEN_CONSERVATION_ATOL + (
        TOKEN_CONSERVATION_RTOL * abs(conservation_target)
    )
    if conservation_error > conservation_tolerance:
        raise StageCValuePathError(
            "matched token conservation failed: "
            f"error={conservation_error:.17g}, "
            f"tolerance={conservation_tolerance:.17g}"
        )
    cls_contribution = float(contributions[CLS_TOKEN_INDEX].item())
    patch = _owned_tensor(contributions[1:])
    patch_map = _owned_tensor(patch.reshape(PATCH_SHAPE))
    cls_l1 = abs(cls_contribution)
    patch_l1 = float(torch.sum(patch.abs().to(torch.float64)).item())
    total_l1 = cls_l1 + patch_l1
    if not math.isfinite(total_l1) or total_l1 <= 0.0:
        raise StageCValuePathError(
            "matched token L1 denominator must be finite and positive"
        )
    patch_support = patch_l1 / total_l1
    positive_patch = float(
        torch.sum(torch.clamp_min(patch, 0.0).to(torch.float64)).item()
    )
    positive_fraction = positive_patch / total_l1
    reasons: list[str] = []
    if patch_support < PATCH_SUPPORT_MINIMUM:
        reasons.append(DEGENERATE_PATCH_SUPPORT)
    if positive_fraction < POSITIVE_PATCH_EVIDENCE_MINIMUM:
        reasons.append(DEGENERATE_POSITIVE_PATCH_EVIDENCE)
    return MatchedTokenContribution(
        candidate=candidate,
        status=STATUS_OK,
        token_contributions=contributions,
        cls_contribution=cls_contribution,
        patch_contributions=patch,
        patch_map=patch_map,
        direction_norm=conservation_target,
        conservation_sum=contribution_sum,
        conservation_error=conservation_error,
        conservation_tolerance=conservation_tolerance,
        patch_support=patch_support,
        positive_patch_evidence_fraction=positive_fraction,
        degeneracy_reasons=tuple(reasons),
        na=None,
    )


@dataclass(frozen=True)
class _EndpointBlockData:
    qkv: torch.Tensor
    attention_used: torch.Tensor
    pre_projection: torch.Tensor


class _DefensiveHooks(AbstractContextManager["_DefensiveHooks"]):
    """Own only hook IDs created here, including register-then-raise cases."""

    def __init__(self) -> None:
        self._owned: list[tuple[dict[Any, Any], tuple[Any, ...], Any]] = []

    def _register(
        self,
        module: nn.Module,
        callback: Any,
        *,
        pre: bool,
        prepend: bool = False,
    ) -> None:
        hooks = module._forward_pre_hooks if pre else module._forward_hooks
        before = set(hooks)
        register = (
            module.register_forward_pre_hook
            if pre
            else module.register_forward_hook
        )
        try:
            handle = register(callback, prepend=prepend)
        except Exception:
            # A hostile/custom registration implementation may call super()
            # and raise before returning the handle.  Remove only IDs that did
            # not exist before this individual registration call.
            for hook_id in set(hooks) - before:
                hooks.pop(hook_id, None)
            raise
        owned_ids = tuple(set(hooks) - before)
        self._owned.append((hooks, owned_ids, handle))

    def forward(
        self, module: nn.Module, callback: Any, *, prepend: bool = False
    ) -> None:
        self._register(module, callback, pre=False, prepend=prepend)

    def forward_pre(self, module: nn.Module, callback: Any) -> None:
        self._register(module, callback, pre=True)

    def __enter__(self) -> "_DefensiveHooks":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        cleanup_error: BaseException | None = None
        for hooks, owned_ids, handle in reversed(self._owned):
            try:
                handle.remove()
            except BaseException as error:  # pragma: no cover - hostile handle
                if cleanup_error is None:
                    cleanup_error = error
            finally:
                for hook_id in owned_ids:
                    hooks.pop(hook_id, None)
        self._owned.clear()
        if cleanup_error is not None:
            if exc is None:
                raise cleanup_error
            if hasattr(exc, "add_note"):
                exc.add_note(
                    "C4 hook cleanup also raised: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )


class _UnfusedSelectedBlocks(AbstractContextManager[None]):
    def __init__(self, attention_modules: Sequence[nn.Module]) -> None:
        self._attention_modules = tuple(attention_modules)
        self._states: list[tuple[nn.Module, bool, Any]] = []

    def __enter__(self) -> None:
        try:
            for attention_module in self._attention_modules:
                existed = hasattr(attention_module, "fused_attn")
                previous = (
                    getattr(attention_module, "fused_attn")
                    if existed
                    else None
                )
                self._states.append((attention_module, existed, previous))
                if existed:
                    setattr(attention_module, "fused_attn", False)
        except Exception:
            self.__exit__(None, None, None)
            raise
        return None

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        for attention_module, existed, previous in reversed(self._states):
            if existed:
                setattr(attention_module, "fused_attn", previous)
        self._states.clear()


def _resolve_production_attention_modules(
    model: nn.Module, selected_blocks: Sequence[int]
) -> dict[int, nn.Module]:
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    blocks = getattr(model, "blocks", None)
    if blocks is None:
        raise AttributeError("model must expose canonical model.blocks")
    try:
        block_count = len(blocks)
    except TypeError as exc:
        raise TypeError("model.blocks must be a sized module sequence") from exc
    if block_count != MODEL_BLOCKS:
        raise StageCValuePathError(
            f"C4 requires exactly {MODEL_BLOCKS} blocks, got {block_count}"
        )
    result: dict[int, nn.Module] = {}
    for block_index in selected_blocks:
        block = blocks[block_index]
        attention_module = getattr(block, "attn", None)
        if not isinstance(attention_module, nn.Module):
            raise AttributeError(
                f"model.blocks[{block_index}] must expose an attn module"
            )
        if getattr(attention_module, "num_heads", None) != HEADS_PER_BLOCK:
            raise StageCValuePathError(
                f"blocks[{block_index}].attn must expose num_heads={HEADS_PER_BLOCK}"
            )
        if getattr(attention_module, "head_dim", None) != HEAD_DIM:
            raise StageCValuePathError(
                f"blocks[{block_index}].attn must expose head_dim={HEAD_DIM}"
            )
        for name in ("qkv", "attn_drop", "proj"):
            if not isinstance(getattr(attention_module, name, None), nn.Module):
                raise AttributeError(
                    f"blocks[{block_index}].attn must expose {name} as a module"
                )
        result[block_index] = attention_module
    return result


class _EndpointCaptures(AbstractContextManager["_EndpointCaptures"]):
    def __init__(self, attentions: Mapping[int, nn.Module]) -> None:
        self._attentions = dict(attentions)
        self._hooks = _DefensiveHooks()
        self._calls: dict[int, dict[str, int]] = {
            block_index: {"qkv": 0, "attention": 0, "projection": 0}
            for block_index in attentions
        }
        self._values: dict[int, dict[str, torch.Tensor]] = {
            block_index: {} for block_index in attentions
        }

    def _output_hook(self, block_index: int, name: str) -> Any:
        def hook(
            _module: nn.Module, _arguments: tuple[Any, ...], output: Any
        ) -> None:
            self._calls[block_index][name] += 1
            if not isinstance(output, torch.Tensor):
                raise TypeError(
                    f"blocks[{block_index}] {name} output must be a Tensor"
                )
            self._values[block_index][name] = _owned_tensor(output)

        return hook

    def _projection_hook(self, block_index: int) -> Any:
        def hook(_module: nn.Module, arguments: tuple[Any, ...]) -> None:
            self._calls[block_index]["projection"] += 1
            if len(arguments) != 1 or not isinstance(arguments[0], torch.Tensor):
                raise TypeError(
                    f"blocks[{block_index}].attn.proj must receive one Tensor"
                )
            self._values[block_index]["projection"] = _owned_tensor(
                arguments[0]
            )

        return hook

    def __enter__(self) -> "_EndpointCaptures":
        self._hooks.__enter__()
        try:
            for block_index, attention_module in self._attentions.items():
                self._hooks.forward(
                    attention_module.qkv,
                    self._output_hook(block_index, "qkv"),
                )
                self._hooks.forward(
                    attention_module.attn_drop,
                    self._output_hook(block_index, "attention"),
                )
                self._hooks.forward_pre(
                    attention_module.proj,
                    self._projection_hook(block_index),
                )
        except Exception:
            self._hooks.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self._hooks.__exit__(exc_type, exc, traceback)

    def finish(self) -> dict[int, _EndpointBlockData]:
        result: dict[int, _EndpointBlockData] = {}
        for block_index in self._attentions:
            calls = self._calls[block_index]
            if any(value != 1 for value in calls.values()):
                raise StageCValuePathError(
                    f"selected block {block_index} observables must each execute "
                    f"exactly once, got {calls}"
                )
            values = self._values[block_index]
            result[block_index] = _EndpointBlockData(
                qkv=values["qkv"],
                attention_used=values["attention"],
                pre_projection=values["projection"],
            )
        return result


def _capture_real_endpoints(
    model: nn.Module,
    inputs: torch.Tensor,
    attentions: Mapping[int, nn.Module],
) -> tuple[dict[int, _EndpointBlockData], torch.Tensor]:
    with _UnfusedSelectedBlocks(tuple(attentions.values())):
        with _EndpointCaptures(attentions) as captures:
            with torch.no_grad():
                model_output = model(inputs)
        if not isinstance(model_output, torch.Tensor):
            raise TypeError("model must return a Tensor for clean replay parity")
        if not model_output.is_floating_point() or model_output.numel() == 0:
            raise StageCValuePathError(
                "model output for clean replay parity must be non-empty floating point"
            )
        return captures.finish(), _owned_tensor(model_output)


def _capture_selected_head_values(
    model: nn.Module,
    inputs: torch.Tensor,
    *,
    block_index: int,
    head_index: int,
) -> torch.Tensor:
    attention_module = _resolve_production_attention_modules(
        model, (block_index,)
    )[block_index]
    endpoints, _model_output = _capture_real_endpoints(
        model, inputs, {block_index: attention_module}
    )
    values, _attention_used, _head_outputs = _validate_endpoint_block_data(
        endpoints[block_index], block_index=block_index
    )
    return _owned_tensor(values[0, head_index])


def _validate_endpoint_block_data(
    value: _EndpointBlockData, *, block_index: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    qkv = value.qkv
    attention_used = value.attention_used
    pre_projection = value.pre_projection
    if qkv.dtype is not torch.float32:
        raise TypeError(
            f"blocks[{block_index}].attn.qkv output must be float32"
        )
    if qkv.shape != (1, TOKENS, 3 * EMBED_DIM):
        raise StageCValuePathError(
            f"blocks[{block_index}].attn.qkv output must have shape "
            f"[1,{TOKENS},{3 * EMBED_DIM}], got {tuple(qkv.shape)}"
        )
    if attention_used.dtype is not torch.float32:
        raise TypeError(
            f"blocks[{block_index}].attn_drop output must be float32"
        )
    if attention_used.shape != (
        1,
        HEADS_PER_BLOCK,
        TOKENS,
        TOKENS,
    ):
        raise StageCValuePathError(
            f"blocks[{block_index}].attn_drop output must have shape "
            f"[1,{HEADS_PER_BLOCK},{TOKENS},{TOKENS}], got "
            f"{tuple(attention_used.shape)}"
        )
    finite_attention = attention_used[torch.isfinite(attention_used)]
    if bool((finite_attention < 0).any().item()):
        raise StageCValuePathError(
            f"blocks[{block_index}].attention_used contains negative values"
        )
    if pre_projection.dtype is not torch.float32:
        raise TypeError(
            f"blocks[{block_index}].attn pre-projection output must be float32"
        )
    if pre_projection.shape != (1, TOKENS, EMBED_DIM):
        raise StageCValuePathError(
            f"blocks[{block_index}] pre-projection output must have shape "
            f"[1,{TOKENS},{EMBED_DIM}], got {tuple(pre_projection.shape)}"
        )
    _query, _key, values = split_fused_qkv(
        qkv, num_heads=HEADS_PER_BLOCK, head_dim=HEAD_DIM
    )
    head_outputs = pre_projection.reshape(
        1, TOKENS, HEADS_PER_BLOCK, HEAD_DIM
    ).permute(0, 2, 1, 3)
    return _owned_tensor(values), _owned_tensor(attention_used), _owned_tensor(
        head_outputs
    )


class _SelectedHeadIntervention(AbstractContextManager["_SelectedHeadIntervention"]):
    def __init__(
        self, attention_module: nn.Module, fixed_by_head: Mapping[int, torch.Tensor]
    ) -> None:
        self.attention_module = attention_module
        self.fixed_by_head = {
            _strict_index(
                head_index, name="head_index", upper_bound=HEADS_PER_BLOCK
            ): _owned_tensor(fixed)
            for head_index, fixed in fixed_by_head.items()
        }
        if not self.fixed_by_head:
            raise StageCValuePathError(
                "selected-head intervention requires at least one head"
            )
        for fixed in self.fixed_by_head.values():
            if fixed.dtype is not torch.float32:
                raise TypeError("fixed attention must have float32 dtype")
            if fixed.shape != (1, TOKENS, TOKENS):
                raise StageCValuePathError(
                    "fixed selected-head attention must have shape [1,197,197]"
                )
        self.call_count = 0
        self._hooks = _DefensiveHooks()

    def _hook(
        self, _module: nn.Module, _arguments: tuple[Any, ...], output: Any
    ) -> torch.Tensor:
        self.call_count += 1
        if not isinstance(output, torch.Tensor):
            raise TypeError("attn_drop must return a Tensor")
        if output.dtype is not torch.float32 or output.ndim != 4:
            raise StageCValuePathError(
                "attn_drop intervention output must be rank-4 float32"
            )
        if output.shape[1:] != (HEADS_PER_BLOCK, TOKENS, TOKENS):
            raise StageCValuePathError(
                "attn_drop intervention output has noncanonical geometry"
            )
        replaced = output.clone()
        for head_index, fixed in self.fixed_by_head.items():
            if fixed.device != output.device:
                raise StageCValuePathError(
                    "fixed attention device does not match model attention"
                )
            replaced[:, head_index] = fixed.expand(output.shape[0], -1, -1)
        return replaced

    def __enter__(self) -> "_SelectedHeadIntervention":
        self._hooks.__enter__()
        try:
            # Register after caller-owned hooks so their optional replacement
            # cannot override the frozen clean attention used downstream.  A
            # C4 capture is registered after this hook and therefore observes
            # the actual fixed value.
            self._hooks.forward(self.attention_module.attn_drop, self._hook)
        except Exception:
            self._hooks.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self._hooks.__exit__(exc_type, exc, traceback)


def _capture_fixed_head_replay(
    model: nn.Module,
    inputs: torch.Tensor,
    *,
    block_index: int,
    attention_module: nn.Module,
    fixed_by_head: Mapping[int, torch.Tensor],
) -> tuple[_EndpointBlockData, torch.Tensor]:
    selected = {block_index: attention_module}
    with _UnfusedSelectedBlocks((attention_module,)):
        with _SelectedHeadIntervention(
            attention_module, fixed_by_head
        ) as intervention:
            # Register capture second so ``attention_used`` is the actual
            # intervened tensor returned to the attention matmul.
            with _EndpointCaptures(selected) as captures:
                with torch.no_grad():
                    model_output = model(inputs)
            result = captures.finish()[block_index]
        if intervention.call_count != 1:
            raise StageCValuePathError(
                "selected attn_drop must execute exactly once per replay, "
                f"got {intervention.call_count}"
            )
    if not isinstance(model_output, torch.Tensor):
        raise TypeError("model must return a Tensor for clean replay parity")
    if not model_output.is_floating_point() or model_output.numel() == 0:
        raise StageCValuePathError(
            "model replay output must be non-empty floating point"
        )
    return result, _owned_tensor(model_output)


def _head_output(
    head_outputs: torch.Tensor, *, head_index: int
) -> torch.Tensor:
    return _owned_tensor(
        head_outputs[0, head_index, CLS_TOKEN_INDEX, :]
    )


def _max_abs_error(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.shape != right.shape:
        raise StageCValuePathError(
            f"endpoint parity shape mismatch: {tuple(left.shape)} vs "
            f"{tuple(right.shape)}"
        )
    difference = (left.to(torch.float32) - right.to(torch.float32)).abs()
    if difference.numel() == 0 or not bool(torch.isfinite(difference).all().item()):
        return math.inf
    return float(difference.max().item())


def resolve_fixed_clean_attention_head_directions(
    model: nn.Module,
    clean_inputs: torch.Tensor,
    candidates: Sequence[HeadCandidate] | Any,
    *,
    candidate_chunk_size: int = HEADS_PER_BLOCK,
) -> tuple[FrozenHeadDirectionTarget, ...]:
    """Resolve an explicit candidate selection from shared real forwards.

    Natural clean and baseline endpoints are captured once for every selected
    block.  Each explicit candidate chunk is then grouped by block so one
    clean/baseline replay serves every selected head in that group.  Chunking
    changes invocation count only; each head's fixed attention and target are
    independent of the other heads in its replay.
    """

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    caller_inputs = _validate_clean_input(clean_inputs)
    inputs = caller_inputs.detach().clone()
    selected = _candidate_tuple(candidates, allow_empty=False)
    chunks = chunk_head_candidates(selected, candidate_chunk_size)
    selected_blocks = tuple(
        dict.fromkeys(candidate.block_index for candidate in selected)
    )
    attention_modules = _resolve_production_attention_modules(
        model, selected_blocks
    )
    resolutions: dict[HeadCandidate, HeadDirectionResolution] = {}
    fixed_attention: dict[HeadCandidate, torch.Tensor] = {}
    parity: dict[HeadCandidate, tuple[float, float, float, float]] = {}

    with _RuntimeIntegrityGuard(
        model, caller_inputs, inputs, context="head-direction resolution"
    ) as model_state_before, _preserve_model_attribution_state(
        model, freeze_parameters=True
    ):
        clean_capture, clean_model_output = _capture_real_endpoints(
            model, inputs, attention_modules
        )
        baseline_inputs = torch.zeros_like(inputs)
        baseline_capture, _baseline_model_output = _capture_real_endpoints(
            model, baseline_inputs, attention_modules
        )
        clean_by_block: dict[
            int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ] = {}
        baseline_by_block: dict[
            int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ] = {}
        for block_index in selected_blocks:
            clean_by_block[block_index] = _validate_endpoint_block_data(
                clean_capture[block_index], block_index=block_index
            )
            baseline_by_block[block_index] = _validate_endpoint_block_data(
                baseline_capture[block_index], block_index=block_index
            )

        for candidate in selected:
            block_index = candidate.block_index
            head_index = candidate.head_index
            clean_values, attention_used, _natural_outputs = clean_by_block[
                block_index
            ]
            baseline_values, _baseline_attention, _baseline_outputs = (
                baseline_by_block[block_index]
            )
            selected_attention = _owned_tensor(
                attention_used[0, head_index, :, :]
            ).unsqueeze(0)
            fixed_attention[candidate] = selected_attention
            resolutions[candidate] = resolve_head_direction_from_endpoints(
                attention_used[
                    0, head_index, CLS_TOKEN_INDEX, :
                ],
                clean_values[0, head_index],
                baseline_values[0, head_index],
                block_index=block_index,
                head_index=head_index,
            )

        for chunk in chunks:
            by_block: dict[int, list[HeadCandidate]] = {}
            for candidate in chunk:
                # Finite threshold-degenerate heads still receive both parity
                # checks. Non-finite directions remain retained but cannot
                # produce a meaningful finite parity error.
                resolution = resolutions[candidate]
                if bool(torch.isfinite(resolution.clean_attention).all()) and bool(
                    torch.isfinite(resolution.clean_values).all()
                ) and bool(torch.isfinite(resolution.baseline_values).all()):
                    by_block.setdefault(candidate.block_index, []).append(candidate)
            for block_index, block_candidates in by_block.items():
                fixed_by_head = {
                    candidate.head_index: fixed_attention[candidate]
                    for candidate in block_candidates
                }
                clean_replay_capture, clean_replay_model_output = _capture_fixed_head_replay(
                    model,
                    inputs,
                    block_index=block_index,
                    attention_module=attention_modules[block_index],
                    fixed_by_head=fixed_by_head,
                )
                baseline_replay_capture, _baseline_replay_model_output = _capture_fixed_head_replay(
                    model,
                    baseline_inputs,
                    block_index=block_index,
                    attention_module=attention_modules[block_index],
                    fixed_by_head=fixed_by_head,
                )
                (
                    clean_replay_values,
                    clean_replay_attention,
                    clean_replay_outputs,
                ) = _validate_endpoint_block_data(
                    clean_replay_capture, block_index=block_index
                )
                (
                    baseline_replay_values,
                    baseline_replay_attention,
                    baseline_replay_outputs,
                ) = _validate_endpoint_block_data(
                    baseline_replay_capture, block_index=block_index
                )
                clean_values, natural_attention, natural_outputs = (
                    clean_by_block[block_index]
                )
                baseline_values = baseline_by_block[block_index][0]
                for candidate in block_candidates:
                    head_index = candidate.head_index
                    resolution = resolutions[candidate]
                    expected_fixed = fixed_attention[candidate]
                    clean_attention_error = _max_abs_error(
                        clean_replay_attention[:, head_index], expected_fixed
                    )
                    baseline_attention_error = _max_abs_error(
                        baseline_replay_attention[:, head_index], expected_fixed
                    )
                    if not (
                        clean_attention_error == 0.0
                        and baseline_attention_error == 0.0
                    ):
                        raise StageCValuePathError(
                            "selected-head attention intervention did not replay "
                            "the exact frozen clean attention"
                        )
                    clean_values_error = _max_abs_error(
                        clean_replay_values[0, head_index],
                        clean_values[0, head_index],
                    )
                    baseline_values_error = _max_abs_error(
                        baseline_replay_values[0, head_index],
                        baseline_values[0, head_index],
                    )
                    if not (
                        clean_values_error < ENDPOINT_PARITY_MAX_ABS
                        and baseline_values_error < ENDPOINT_PARITY_MAX_ABS
                    ):
                        raise StageCValuePathError(
                            "selected block Value endpoints changed under its "
                            "attention intervention"
                        )
                    natural_g = _head_output(
                        natural_outputs, head_index=head_index
                    )
                    clean_formula_error = _max_abs_error(
                        natural_g, resolution.clean_head_output
                    )
                    if not clean_formula_error < ENDPOINT_PARITY_MAX_ABS:
                        raise StageCValuePathError(
                            "natural clean pre-projection head output does not "
                            "match clean attention @ clean V"
                        )
                    clean_replay_error = _max_abs_error(
                        natural_g,
                        _head_output(
                            clean_replay_outputs, head_index=head_index
                        ),
                    )
                    baseline_replay_error = _max_abs_error(
                        _head_output(
                            baseline_replay_outputs, head_index=head_index
                        ),
                        resolution.baseline_head_output,
                    )
                    clean_model_output_error = _max_abs_error(
                        clean_model_output, clean_replay_model_output
                    )
                    if not clean_replay_error < ENDPOINT_PARITY_MAX_ABS:
                        raise StageCValuePathError(
                            "clean natural vs fixed-clean-A replay parity must "
                            "have max_abs_error < 1e-6"
                        )
                    if not baseline_replay_error < ENDPOINT_PARITY_MAX_ABS:
                        raise StageCValuePathError(
                            "baseline fixed-clean-A internal output vs independent "
                            "clean-A @ natural baseline V parity must have "
                            "max_abs_error < 1e-6"
                        )
                    if not clean_model_output_error < ENDPOINT_PARITY_MAX_ABS:
                        raise StageCValuePathError(
                            "clean natural vs fixed-clean-A complete model-output "
                            "parity must have max_abs_error < 1e-6"
                        )
                    parity[candidate] = (
                        clean_formula_error,
                        clean_replay_error,
                        clean_model_output_error,
                        baseline_replay_error,
                    )

    targets: list[FrozenHeadDirectionTarget] = []
    for candidate in selected:
        resolution = resolutions[candidate]
        if candidate in parity:
            clean_formula, clean_replay, clean_model_output, baseline_replay = parity[candidate]
        else:
            clean_formula = clean_replay = clean_model_output = baseline_replay = None
        targets.append(
            FrozenHeadDirectionTarget._create(
                model=model,
                model_state_signature=model_state_before,
                clean_inputs=inputs,
                clean_input_context=caller_inputs,
                candidate=candidate,
                resolution=resolution,
                fixed_attention=fixed_attention[candidate],
                clean_value_max_abs_error=clean_formula,
                clean_replay_max_abs_error=clean_replay,
                clean_model_output_max_abs_error=clean_model_output,
                baseline_replay_max_abs_error=baseline_replay,
                issuer_secret=_TARGET_ISSUER_SECRET,
            )
        )
    return tuple(targets)


def resolve_fixed_clean_attention_head_direction(
    model: nn.Module,
    clean_inputs: torch.Tensor,
    block_index: int,
    head_index: int,
) -> FrozenHeadDirectionTarget:
    """Resolve one canonical fixed-clean-attention head target."""

    candidate = HeadCandidate(block_index, head_index)
    return resolve_fixed_clean_attention_head_directions(
        model,
        clean_inputs,
        (candidate,),
        candidate_chunk_size=1,
    )[0]


def _verify_frozen_head_target(
    target: Any,
    model: nn.Module,
    clean_inputs: torch.Tensor,
    *,
    clean_input_sha256: str | None = None,
) -> FrozenHeadDirectionTarget:
    verified = _require_untampered_target_identity(target)
    if verified._model_ref() is not model:
        raise StageCValuePathError(
            "model instance does not match the frozen head target"
        )
    if tuple(clean_inputs.shape) != verified.clean_input_shape:
        raise StageCValuePathError(
            "clean input shape does not match the frozen head target"
        )
    if (
        verified._clean_input_ref() is not clean_inputs
        or CheapTensorSnapshot.capture(clean_inputs).signature
        != verified._clean_input_live_signature
    ):
        raise StageCValuePathError(
            "clean input bytes/live identity do not match the frozen head target"
        )
    clean_digest = _prebound_sha256(
        clean_input_sha256, name="clean_input_sha256"
    )
    if clean_digest is None:
        clean_digest = _tensor_sha256(clean_inputs)
    if clean_digest != verified.clean_input_sha256:
        raise StageCValuePathError(
            "clean input bytes do not match the frozen head target"
        )
    current_signature = _model_state_signature(model)
    if current_signature != verified._model_state_signature:
        raise StageCValuePathError(
            "model parameter/buffer state changed after target resolution"
        )
    if (
        _model_binding_sha256(current_signature)
        != verified.runtime_model_binding_sha256
    ):
        raise StageCValuePathError(
            "model runtime binding hash does not match the frozen head target"
        )
    return verified


def _verify_target_live_state(
    target: Any,
) -> FrozenHeadDirectionTarget:
    verified = _require_untampered_target_identity(target)
    model = verified._model_ref()
    if model is None:
        raise StageCValuePathError(
            "the model bound to the frozen head target no longer exists"
        )
    current_signature = _model_state_signature(model)
    if current_signature != verified._model_state_signature or (
        _model_binding_sha256(current_signature)
        != verified.runtime_model_binding_sha256
    ):
        raise StageCValuePathError(
            "model parameter/buffer state changed after target resolution"
        )
    return verified


def evaluate_frozen_head_direction_target(
    model: nn.Module,
    clean_inputs: torch.Tensor,
    intervention_inputs: torch.Tensor,
    target: FrozenHeadDirectionTarget,
    *,
    clean_input_sha256: str | None = None,
    intervention_input_sha256: str | None = None,
) -> FrozenScalarEvaluation:
    """Evaluate the issued fixed-clean-A, fixed-u head scalar at ``z``."""

    clean = _validate_clean_input(clean_inputs)
    intervention = _validate_clean_input(intervention_inputs)
    if clean.device != intervention.device:
        raise StageCValuePathError(
            "intervention input device must match the clean input context"
        )
    clean_digest = _prebound_sha256(
        clean_input_sha256, name="clean_input_sha256"
    )
    intervention_digest = _prebound_sha256(
        intervention_input_sha256, name="intervention_input_sha256"
    )
    frozen = _verify_frozen_head_target(
        target, model, clean, clean_input_sha256=clean_digest
    )
    if clean_digest is None:
        clean_digest = frozen.clean_input_sha256
    if intervention_digest is None:
        intervention_digest = _tensor_sha256(intervention)
    if frozen.status != STATUS_OK or frozen.direction_unit is None:
        raise StageCValuePathError(
            "frozen head-direction scalar is undefined for a degenerate target"
        )
    attention_module = _resolve_production_attention_modules(
        model, (frozen.block_index,)
    )[frozen.block_index]
    owned_inputs = intervention.detach().clone()
    fixed_attention = _owned_tensor(frozen.fixed_attention)
    frozen_direction = _owned_tensor(frozen.direction_unit)
    with _RuntimeIntegrityGuard(
        model, intervention, owned_inputs,
        context="frozen head-direction scalar evaluation", target=frozen,
    ), _preserve_model_attribution_state(
        model, freeze_parameters=True
    ), _UnfusedSelectedBlocks((attention_module,)):
        with _SelectedHeadIntervention(
            attention_module, {frozen.head_index: fixed_attention}
        ) as intervention_hook, _DifferentiableHeadOutputCapture(
            attention_module.proj
        ) as pre_projection, torch.no_grad():
            before_projection = pre_projection.call_count
            model(owned_inputs)
            if intervention_hook.call_count != 1:
                raise StageCValuePathError(
                    "selected attn_drop must execute exactly once per scalar evaluation"
                )
            scalar = pre_projection.scalar(
                before_calls=before_projection,
                head_index=frozen.head_index,
                direction_unit=frozen_direction,
            )
            if scalar.dtype is not torch.float32 or scalar.shape != (1,):
                raise StageCValuePathError(
                    "frozen head-direction scalar must have shape [1] and float32 dtype"
                )
            if not bool(torch.isfinite(scalar).all().item()):
                raise StageCValuePathError("frozen head-direction scalar is non-finite")
            value = float(scalar.item())
    _verify_frozen_head_target(
        frozen, model, clean, clean_input_sha256=clean_digest
    )
    return FrozenScalarEvaluation._create(
        value=value,
        block_index=frozen.block_index,
        head_index=frozen.head_index,
        clean_input_sha256=frozen.clean_input_sha256,
        intervention_input_sha256=intervention_digest,
        runtime_model_binding_sha256=frozen.runtime_model_binding_sha256,
        issuer_secret=_TARGET_ISSUER_SECRET,
    )


class _DifferentiableHeadOutputCapture(
    AbstractContextManager["_DifferentiableHeadOutputCapture"]
):
    def __init__(self, projection: nn.Module) -> None:
        self.projection = projection
        self.call_count = 0
        self.latest: torch.Tensor | None = None
        self._hooks = _DefensiveHooks()

    def _hook(self, _module: nn.Module, arguments: tuple[Any, ...]) -> None:
        self.call_count += 1
        if len(arguments) != 1 or not isinstance(arguments[0], torch.Tensor):
            raise TypeError("attention projection must receive one Tensor")
        output = arguments[0]
        if output.dtype is not torch.float32:
            raise TypeError("attention pre-projection output must be float32")
        if output.ndim != 3 or output.shape[1:] != (TOKENS, EMBED_DIM):
            raise StageCValuePathError(
                "attention pre-projection output must have shape "
                "[batch,197,768]"
            )
        if not bool(torch.isfinite(output.detach()).all().item()):
            raise StageCValuePathError(
                "attention pre-projection output contains non-finite values"
            )
        self.latest = output

    def __enter__(self) -> "_DifferentiableHeadOutputCapture":
        self._hooks.__enter__()
        try:
            self._hooks.forward_pre(self.projection, self._hook)
        except Exception:
            self._hooks.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self._hooks.__exit__(exc_type, exc, traceback)

    def scalar(
        self,
        *,
        before_calls: int,
        head_index: int,
        direction_unit: torch.Tensor,
    ) -> torch.Tensor:
        if self.call_count != before_calls + 1 or self.latest is None:
            raise StageCValuePathError(
                "selected attention projection must execute exactly once per "
                "Value-path callback"
            )
        head_output = self.latest.reshape(
            self.latest.shape[0], TOKENS, HEADS_PER_BLOCK, HEAD_DIM
        )[:, CLS_TOKEN_INDEX, head_index, :]
        if direction_unit.device != head_output.device:
            raise StageCValuePathError(
                "frozen direction device does not match the model Value path"
            )
        return torch.sum(head_output * direction_unit[None, :], dim=1)


def _attempt_numeric(
    attempt: Mapping[str, Any], key: str, attempt_index: int
) -> float:
    values = attempt.get(key)
    if (
        not isinstance(values, list)
        or len(values) != 1
        or isinstance(values[0], bool)
        or not isinstance(values[0], (int, float))
        or not math.isfinite(float(values[0]))
    ):
        raise StageCValuePathError(
            f"IG attempt {attempt_index} {key} must be one finite numeric value"
        )
    return float(values[0])


def _validate_head_direction_ig_result(
    result: Any,
    *,
    target: FrozenHeadDirectionTarget,
    expected_delta: float | None = None,
) -> IGResult:
    if not isinstance(result, IGResult):
        raise TypeError("integrated_gradients must return IGResult")
    if type(result.converged) is not bool:
        raise TypeError("IGResult.converged must be a boolean")
    if type(result.steps) is not int or result.steps not in IG_STEPS:
        raise StageCValuePathError(
            f"head-direction IG returned unregistered steps={result.steps!r}"
        )
    if not isinstance(result.attempts, tuple) or not result.attempts:
        raise StageCValuePathError(
            "head-direction IG must retain a non-empty tuple of attempts"
        )
    attempt_steps: list[int] = []
    attempt_passes: list[bool] = []
    required_attempt = {"steps", "delta", "sum", "abs", "rel", "pass"}
    for attempt_index, attempt in enumerate(result.attempts):
        if not isinstance(attempt, Mapping) or set(attempt) != required_attempt:
            raise StageCValuePathError(
                "head-direction IG attempt has the wrong fields"
            )
        steps = attempt.get("steps")
        if type(steps) is not int:
            raise StageCValuePathError(
                "head-direction IG attempt steps must be an integer"
            )
        attempt_steps.append(steps)
        delta = _attempt_numeric(attempt, "delta", attempt_index)
        attribution_sum = _attempt_numeric(attempt, "sum", attempt_index)
        absolute = _attempt_numeric(attempt, "abs", attempt_index)
        relative = _attempt_numeric(attempt, "rel", attempt_index)
        expected_absolute = abs(delta - attribution_sum)
        expected_relative = expected_absolute / max(
            abs(delta), torch.finfo(torch.float32).eps
        )
        if not math.isclose(
            absolute, expected_absolute, abs_tol=1e-12, rel_tol=1e-6
        ):
            raise StageCValuePathError(
                "head-direction IG attempt absolute error is inconsistent"
            )
        if not math.isclose(
            relative, expected_relative, abs_tol=1e-12, rel_tol=1e-6
        ):
            raise StageCValuePathError(
                "head-direction IG attempt relative error is inconsistent"
            )
        pass_values = attempt.get("pass")
        if (
            not isinstance(pass_values, list)
            or len(pass_values) != 1
            or type(pass_values[0]) is not bool
        ):
            raise StageCValuePathError(
                "head-direction IG attempt pass must be one boolean"
            )
        expected_pass = absolute <= (IG_ATOL + IG_RTOL * abs(delta))
        if pass_values[0] is not expected_pass:
            raise StageCValuePathError(
                "head-direction IG attempt pass flag is inconsistent"
            )
        attempt_passes.append(pass_values[0])
    expected_steps = list(IG_STEPS[: len(attempt_steps)])
    if attempt_steps != expected_steps or result.steps != attempt_steps[-1]:
        raise StageCValuePathError(
            "head-direction IG attempts must be an exact frozen schedule prefix "
            "ending at result.steps"
        )
    passing_indices = [
        index for index, passed in enumerate(attempt_passes) if passed
    ]
    if result.converged:
        if passing_indices != [len(attempt_passes) - 1]:
            raise StageCValuePathError(
                "head-direction IG must stop at its first conservation pass"
            )
    elif passing_indices or tuple(attempt_steps) != IG_STEPS:
        raise StageCValuePathError(
            "nonconverged head-direction IG must retain all five failed attempts"
        )

    native = result.attribution
    if not isinstance(native, torch.Tensor):
        raise TypeError("head-direction IG attribution must be a Tensor")
    if native.shape != (1, 3, 224, 224):
        raise StageCValuePathError(
            "head-direction IG attribution must have shape [1,3,224,224]"
        )
    if native.dtype is not torch.float32:
        raise TypeError("head-direction IG attribution must be float32")
    if not bool(torch.isfinite(native.detach()).all().item()):
        raise StageCValuePathError(
            "head-direction IG attribution contains non-finite values"
        )
    if int(torch.count_nonzero(native.detach()).item()) == 0:
        raise StageCValuePathError(
            "head-direction IG attribution must not be all zero"
        )
    required_report = {"delta", "sum", "abs", "rel", "pass"}
    if not isinstance(result.conservation, Mapping) or (
        set(result.conservation) != required_report
    ):
        raise StageCValuePathError(
            "head-direction IG conservation has the wrong fields"
        )
    for name, value in result.conservation.items():
        if not isinstance(value, torch.Tensor) or value.shape != (1,):
            raise StageCValuePathError(
                f"head-direction IG conservation {name} must have shape [1]"
            )
        if name == "pass":
            if value.dtype is not torch.bool:
                raise TypeError(
                    "head-direction IG conservation pass must have bool dtype"
                )
        elif not bool(torch.isfinite(value).all().item()):
            raise StageCValuePathError(
                f"head-direction IG conservation {name} is non-finite"
            )
    report_delta = result.conservation["delta"].to(torch.float32)
    report_sum = result.conservation["sum"].to(torch.float32)
    report_abs = result.conservation["abs"].to(torch.float32)
    report_rel = result.conservation["rel"].to(torch.float32)
    attribution_sum = native.flatten(1).sum(dim=1)
    if not torch.equal(attribution_sum, report_sum):
        raise StageCValuePathError(
            "head-direction IG conservation sum does not equal attribution sum"
        )
    expected_abs = (report_delta - report_sum).abs()
    expected_rel = expected_abs / report_delta.abs().clamp_min(
        torch.finfo(torch.float32).eps
    )
    expected_pass = expected_abs <= (IG_ATOL + IG_RTOL * report_delta.abs())
    if not torch.equal(expected_abs, report_abs):
        raise StageCValuePathError(
            "head-direction IG final absolute error is inconsistent"
        )
    if not torch.equal(expected_rel, report_rel):
        raise StageCValuePathError(
            "head-direction IG final relative error is inconsistent"
        )
    if not torch.equal(expected_pass, result.conservation["pass"]):
        raise StageCValuePathError(
            "head-direction IG final pass flag is inconsistent"
        )
    if bool(result.conservation["pass"].all().item()) is not result.converged:
        raise StageCValuePathError(
            "head-direction IG converged flag contradicts conservation"
        )
    final_attempt = result.attempts[-1]
    for name in required_report:
        if final_attempt[name][0] != result.conservation[name][0].item():
            raise StageCValuePathError(
                f"head-direction IG final attempt {name} does not match report"
            )

    # The generic IG core reports the two independently evaluated callback
    # endpoints in float32.  Bind that scalar to r under the registered IG
    # numerical tolerance; the separately measured replay parities above keep
    # their strict max-absolute-error < 1e-6 gates.
    bound_delta = (
        target.direction_norm
        if expected_delta is None
        else float(expected_delta)
    )
    delta_identity_error = abs(float(report_delta.item()) - bound_delta)
    delta_identity_tolerance = IG_ATOL + IG_RTOL * abs(float(report_delta.item()))
    if delta_identity_error > delta_identity_tolerance:
        raise StageCValuePathError(
            "head-direction IG endpoint delta does not match frozen r"
        )
    metadata = result.metadata
    if not isinstance(metadata, Mapping):
        raise TypeError("head-direction IG metadata must be a mapping")
    expected_metadata: dict[str, Any] = {
        "method": "integrated_gradients",
        "quadrature": "gauss_legendre",
        "integration_dtype": "float32",
        "input_shape": [1, 3, 224, 224],
        "step_schedule": list(IG_STEPS),
        "step_chunk_size": IG_STEP_CHUNK_SIZE,
        "atol": IG_ATOL,
        "rtol": IG_RTOL,
        "scope": "fixed_clean_attention_head_direction_value_path",
        "target": {
            "kind": TARGET_REF,
            "block_index": target.block_index,
            "head_index": target.head_index,
            "query_token": CLS_TOKEN_INDEX,
        },
        "includes_residual_paths": False,
        "is_complete_attention_attribution": False,
        "fixed_clean_attention": True,
        "frozen_direction": True,
    }
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            raise StageCValuePathError(
                f"head-direction IG metadata {key!r} does not match contract"
            )
    return result


def _caller_context(provenance: Mapping[str, Any] | None) -> dict[str, Any]:
    if provenance is None:
        return {}
    if not isinstance(provenance, Mapping):
        raise TypeError("provenance must be a mapping or None")
    return dict(provenance)


def _base_provenance(
    *,
    exact_name: str,
    target: FrozenHeadDirectionTarget,
    provenance: Mapping[str, Any] | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "stage": "C4",
        "exact_method_name": exact_name,
        "method_implementation_version": METHOD_VERSION,
        "comparison_line": "value_path",
        "class_agnostic": True,
        "target": target.to_dict(),
        "claim_boundaries": {
            "is_class_logit_explanation": False,
            "is_complete_qkv_attribution": False,
            "is_complete_attention_attribution": False,
            "explains_only_fixed_clean_attention_head_value_path": True,
        },
    }
    context = _caller_context(provenance)
    if context:
        result["caller_context"] = context
    return result


def _na_method_result(
    *,
    method_id: str,
    projection: str,
    provenance: Mapping[str, Any],
    na: StructuredNA,
) -> MethodResult:
    return MethodResult(
        method_id=method_id,
        method_version=METHOD_VERSION,
        comparison_line=ComparisonLine.VALUE_PATH,
        explanation_target=TARGET_REF,
        fidelity_target=TARGET_REF,
        class_agnostic=True,
        native_tensor=None,
        metric_map=None,
        projection=projection,
        sign_semantics=SignSemantics.SIGNED,
        valid_mask=np.zeros(PATCH_SHAPE, dtype=np.bool_),
        provenance=provenance,
        na=na,
    )


def _ig_provenance_summary(result: IGResult) -> dict[str, Any]:
    return {
        "selected_steps": result.steps,
        "converged": result.converged,
        "attempts": [dict(attempt) for attempt in result.attempts],
        "conservation": {
            key: value.detach().to(device="cpu").tolist()
            for key, value in result.conservation.items()
        },
        "metadata": dict(result.metadata),
    }


def conditional_fixed_clean_attention_head_direction_value_path_ig_at_intervention(
    model: nn.Module,
    clean_inputs: torch.Tensor,
    intervention_inputs: torch.Tensor,
    target: FrozenHeadDirectionTarget,
    provenance: Mapping[str, Any] | None = None,
    *,
    clean_input_sha256: str | None = None,
    intervention_input_sha256: str | None = None,
) -> MethodResult:
    """Attribute the issued fixed-clean-attention head scalar to RGB at ``z``."""

    if torch.is_inference_mode_enabled():
        raise RuntimeError(
            "head-direction Value-path IG cannot run inside "
            "torch.inference_mode()"
        )
    clean = _validate_clean_input(clean_inputs)
    inputs = _validate_clean_input(intervention_inputs)
    if clean.device != inputs.device:
        raise StageCValuePathError(
            "intervention input device must match the clean input context"
        )
    clean_digest = _prebound_sha256(
        clean_input_sha256, name="clean_input_sha256"
    )
    if clean_digest is None:
        clean_digest = _tensor_sha256(clean)
    intervention_digest = _prebound_sha256(
        intervention_input_sha256, name="intervention_input_sha256"
    )
    if intervention_digest is None:
        intervention_digest = _tensor_sha256(inputs)
    frozen = _verify_frozen_head_target(
        target, model, clean, clean_input_sha256=clean_digest
    )
    method_provenance = _base_provenance(
        exact_name=NNS_METHOD_NAME,
        target=frozen,
        provenance=provenance,
    )
    method_provenance["intervention_input_sha256"] = intervention_digest
    method_provenance["algorithm"] = {
        "model_scope": "real_upstream_value_path_to_selected_block",
        "attention_source": "clean_real_forward_attention_used",
        "fixed_attention_scope": "selected_head_only",
        "scalar": "dot(frozen_u,pre_projection_selected_head_cls_output)",
        "baseline": {
            "id": "zero_normalized",
            "normalized_value": 0.0,
            "canonical_rgb_value": 0.5,
            "is_black_image": False,
        },
        "quadrature": "gauss_legendre",
        "integration_dtype": "float32",
        "steps_schedule": list(IG_STEPS),
        "step_chunk_size": IG_STEP_CHUNK_SIZE,
        "atol": IG_ATOL,
        "rtol": IG_RTOL,
        "native_attribution": "signed_rgb_float32_3x224x224",
        "rgb_reduction": "signed_channel_sum_before_projection",
        "projection": "nonoverlapping_16x16_pixel_mass_to_patch_sum",
        "absolute_value_before_projection": False,
        "includes_residual_paths": False,
        "is_complete_attention_attribution": False,
        "claim_boundary": "fixed_clean_attention_head_value_path_only",
    }
    method_provenance["endpoint_parity"] = {
        "threshold": ENDPOINT_PARITY_MAX_ABS,
        "comparison": "strict_less_than",
        "clean_formula_max_abs_error": frozen.clean_value_max_abs_error,
        "clean_natural_vs_fixed_replay_max_abs_error": (
            frozen.clean_replay_max_abs_error
        ),
        "clean_natural_vs_fixed_replay_model_output_max_abs_error": (
            frozen.clean_model_output_max_abs_error
        ),
        "baseline_fixed_replay_vs_clean_A_matmul_natural_V0_max_abs_error": (
            frozen.baseline_replay_max_abs_error
        ),
    }
    if frozen.status == DEGENERATE_DIRECTION:
        if frozen.na is None:
            raise StageCValuePathError(
                "degenerate target does not contain structured NA"
            )
        method_provenance["status"] = DEGENERATE_DIRECTION
        return _na_method_result(
            method_id=NNS_METHOD_ID,
            projection="pixel_mass_to_patch_sum",
            provenance=method_provenance,
            na=frozen.na,
        )

    baseline_inputs = torch.zeros_like(inputs)
    intervention_endpoint = evaluate_frozen_head_direction_target(
        model, clean, inputs, frozen,
        clean_input_sha256=clean_digest,
        intervention_input_sha256=intervention_digest,
    )
    baseline_digest = _tensor_sha256(baseline_inputs)
    baseline_endpoint = evaluate_frozen_head_direction_target(
        model, clean, baseline_inputs, frozen,
        clean_input_sha256=clean_digest,
        intervention_input_sha256=baseline_digest,
    )
    expected_delta = intervention_endpoint.value - baseline_endpoint.value

    attention_module = _resolve_production_attention_modules(
        model, (frozen.block_index,)
    )[frozen.block_index]
    direction_unit = frozen.direction_unit
    if direction_unit is None:
        raise StageCValuePathError(
            "nondegenerate target unexpectedly has no unit direction"
        )
    # Private, owned clones are passed to hooks/callbacks.  The issued target
    # remains untouched and is re-verified after attribution.
    fixed_attention = _owned_tensor(frozen.fixed_attention)
    frozen_direction = _owned_tensor(direction_unit)
    caller_inputs = inputs
    owned_inputs = inputs.detach().clone()
    owned_baseline = baseline_inputs.detach().clone()
    with _RuntimeIntegrityGuard(
        model, caller_inputs, owned_inputs,
        context="head-direction Value-path IG", target=frozen,
    ), _preserve_model_attribution_state(
        model, freeze_parameters=True
    ), _UnfusedSelectedBlocks((attention_module,)):
        with _SelectedHeadIntervention(
            attention_module, {frozen.head_index: fixed_attention}
        ) as intervention, _DifferentiableHeadOutputCapture(
            attention_module.proj
        ) as pre_projection, torch.enable_grad():

            def callback(value: torch.Tensor) -> torch.Tensor:
                before_intervention = intervention.call_count
                before_projection = pre_projection.call_count
                model(value)
                if intervention.call_count != before_intervention + 1:
                    raise StageCValuePathError(
                        "selected attn_drop must execute exactly once per "
                        "Value-path callback"
                    )
                return pre_projection.scalar(
                    before_calls=before_projection,
                    head_index=frozen.head_index,
                    direction_unit=frozen_direction,
                )

            result = attribution.integrated_gradients(
                callback,
                owned_inputs,
                owned_baseline,
                steps=IG_STEPS,
                step_chunk_size=IG_STEP_CHUNK_SIZE,
                atol=IG_ATOL,
                rtol=IG_RTOL,
                parameters=model,
                metadata={
                    "scope": "fixed_clean_attention_head_direction_value_path",
                    "target": {
                        "kind": TARGET_REF,
                        "block_index": frozen.block_index,
                        "head_index": frozen.head_index,
                        "query_token": CLS_TOKEN_INDEX,
                    },
                    "includes_residual_paths": False,
                    "is_complete_attention_attribution": False,
                    "fixed_clean_attention": True,
                    "frozen_direction": True,
                },
            )
            intervention_calls = intervention.call_count
            projection_calls = pre_projection.call_count
        if not isinstance(result, IGResult):
            raise TypeError("integrated_gradients must return IGResult")
        result_with_counts = replace(
            result,
            metadata={
                **result.metadata,
                "intervention_calls": intervention_calls,
                "pre_projection_calls": projection_calls,
            },
        )
        validated = _validate_head_direction_ig_result(
            result_with_counts, target=frozen, expected_delta=expected_delta
        )
        _verify_frozen_head_target(
            frozen, model, clean,
            clean_input_sha256=clean_digest,
        )
    if not validated.converged:
        raise StageCValuePathConvergenceError(
            "head-direction Value-path IG did not converge on the complete "
            "frozen schedule",
            result,
        )

    native_values = (
        validated.attribution[0]
        .detach()
        .to(device="cpu", dtype=torch.float32)
        .numpy()
        .copy()
    )
    native = raw_array(native_values)
    metric = pixel_to_patch_mass(signed_rgb_channel_sum(native))
    if int(np.count_nonzero(metric.data)) == 0:
        raise StageCValuePathError(
            "head-direction IG signed channel-sum patch map is all zero"
        )
    method_provenance["status"] = STATUS_OK
    method_provenance["ig_result"] = _ig_provenance_summary(validated)
    return MethodResult(
        method_id=NNS_METHOD_ID,
        method_version=METHOD_VERSION,
        comparison_line=ComparisonLine.VALUE_PATH,
        explanation_target=TARGET_REF,
        fidelity_target=TARGET_REF,
        class_agnostic=True,
        native_tensor=native,
        metric_map=metric,
        projection="pixel_mass_to_patch_sum",
        sign_semantics=SignSemantics.SIGNED,
        valid_mask=np.ones(PATCH_SHAPE, dtype=np.bool_),
        provenance=method_provenance,
    )


def conditional_fixed_clean_attention_head_direction_value_path_ig(
    model: nn.Module,
    clean_inputs: torch.Tensor,
    target: FrozenHeadDirectionTarget,
    provenance: Mapping[str, Any] | None = None,
) -> MethodResult:
    """Attribute the issued fixed-clean-attention head scalar to RGB."""

    result = conditional_fixed_clean_attention_head_direction_value_path_ig_at_intervention(
        model,
        clean_inputs,
        clean_inputs,
        target,
        provenance=provenance,
        clean_input_sha256=target.clean_input_sha256,
        intervention_input_sha256=target.clean_input_sha256,
    )
    provenance_dict = dict(result.provenance)
    provenance_dict.pop("intervention_input_sha256", None)
    return replace(result, provenance=provenance_dict)


def matched_fixed_clean_attention_patch_token_contribution(
    target: FrozenHeadDirectionTarget,
    provenance: Mapping[str, Any] | None = None,
) -> MethodResult:
    """Return the issued target's exact endpoint patch-token baseline."""

    frozen = _verify_target_live_state(target)
    contribution = matched_token_contributions(frozen)
    method_provenance = _base_provenance(
        exact_name=MATCHED_METHOD_NAME,
        target=frozen,
        provenance=provenance,
    )
    method_provenance["algorithm"] = {
        "formula": "c_j=A_clean_cls_j*dot(u,V_clean_j-V_baseline_j)",
        "token_count": TOKENS,
        "patch_token_count": PATCH_TOKENS,
        "cls_contribution_retained_separately": True,
        "cls_given_spatial_location": False,
        "patch_order": "native_row_major_14x14",
        "projection": "patch_mass_to_pixel_uniform",
        "projection_semantics": (
            "mass_conserving_uniform_distribution_not_native_pixel_evidence"
        ),
        "patch_to_pixel_divisor": 256,
        "patch_to_fine_cell_divisor": 16,
        "is_complete_attention_attribution": False,
        "claim_boundary": "endpoint_patch_token_spatialization_only",
    }
    if contribution.status == DEGENERATE_DIRECTION:
        if contribution.na is None:
            raise StageCValuePathError(
                "degenerate contribution does not contain structured NA"
            )
        method_provenance["status"] = DEGENERATE_DIRECTION
        method_provenance["matched_contribution"] = {
            "status": DEGENERATE_DIRECTION,
            "degeneracy_reasons": list(contribution.degeneracy_reasons),
        }
        return _na_method_result(
            method_id=MATCHED_METHOD_ID,
            projection="patch_mass_to_pixel_uniform",
            provenance=method_provenance,
            na=contribution.na,
        )

    patch_map = contribution.patch_map
    if patch_map is None:
        raise StageCValuePathError(
            "OK matched contribution unexpectedly has no patch map"
        )
    patch_values = (
        patch_map.detach().to(device="cpu", dtype=torch.float32).numpy().copy()
    )
    # An all-zero patch map is a retained spatial degeneracy.  It is not
    # dropped: c0 remains separate and the map remains a valid signed 14x14
    # MethodResult with explicit uppercase support/evidence flags.
    method_provenance["status"] = STATUS_OK
    method_provenance["matched_contribution"] = {
        "status": STATUS_OK,
        "token_contribution_count": TOKENS,
        "token_contributions_sha256": _tensor_sha256(
            contribution.token_contributions
        ),
        "cls_contribution": contribution.cls_contribution,
        "patch_contributions_sha256": _tensor_sha256(
            contribution.patch_contributions
        ),
        "token_sum": contribution.conservation_sum,
        # At the clean endpoint dot(u, g_clean - g_baseline) equals
        # ||g_clean - g_baseline|| by construction.  Retain the scalar under
        # the same explicit name used by intervention records so C8 can audit
        # token conservation without substituting the vector norm by name.
        "scalar_delta_f_z_minus_f_x0": contribution.direction_norm,
        "direction_norm": contribution.direction_norm,
        "conservation_abs_error": contribution.conservation_error,
        "conservation_tolerance": contribution.conservation_tolerance,
        "patch_support": contribution.patch_support,
        "positive_patch_evidence_fraction": (
            contribution.positive_patch_evidence_fraction
        ),
        "degeneracy_reasons": list(contribution.degeneracy_reasons),
        "patch_support_minimum": PATCH_SUPPORT_MINIMUM,
        "positive_patch_evidence_minimum": (
            POSITIVE_PATCH_EVIDENCE_MINIMUM
        ),
        "all_zero_patch_map_retained": bool(
            int(torch.count_nonzero(patch_map).item()) == 0
        ),
    }
    native = raw_array(patch_values)
    return MethodResult(
        method_id=MATCHED_METHOD_ID,
        method_version=METHOD_VERSION,
        comparison_line=ComparisonLine.VALUE_PATH,
        explanation_target=TARGET_REF,
        fidelity_target=TARGET_REF,
        class_agnostic=True,
        native_tensor=native,
        metric_map=stage_c_metric_ready(patch_values),
        projection="patch_mass_to_pixel_uniform",
        sign_semantics=SignSemantics.SIGNED,
        valid_mask=np.ones(PATCH_SHAPE, dtype=np.bool_),
        provenance=method_provenance,
    )


def matched_fixed_clean_attention_patch_token_contribution_at_intervention(
    model: nn.Module,
    clean_inputs: torch.Tensor,
    intervention_inputs: torch.Tensor,
    target: FrozenHeadDirectionTarget,
    provenance: Mapping[str, Any] | None = None,
    *,
    clean_input_sha256: str | None = None,
    intervention_input_sha256: str | None = None,
) -> MethodResult:
    """Return the issued target's exact matched token map at arbitrary ``z``."""

    clean = _validate_clean_input(clean_inputs)
    inputs = _validate_clean_input(intervention_inputs)
    if clean.device != inputs.device:
        raise StageCValuePathError(
            "intervention input device must match the clean input context"
        )
    clean_digest = _prebound_sha256(
        clean_input_sha256, name="clean_input_sha256"
    )
    if clean_digest is None:
        clean_digest = _tensor_sha256(clean)
    intervention_digest = _prebound_sha256(
        intervention_input_sha256, name="intervention_input_sha256"
    )
    if intervention_digest is None:
        intervention_digest = _tensor_sha256(inputs)
    frozen = _verify_frozen_head_target(
        target, model, clean, clean_input_sha256=clean_digest
    )
    method_provenance = _base_provenance(
        exact_name=MATCHED_METHOD_NAME,
        target=frozen,
        provenance=provenance,
    )
    method_provenance["stage"] = "C5"
    method_provenance["intervention_input_sha256"] = intervention_digest
    method_provenance["algorithm"] = {
        "formula": "c_j(z)=A_clean_cls_j*dot(u,V(z)_j-V(x0)_j)",
        "token_count": TOKENS,
        "patch_token_count": PATCH_TOKENS,
        "cls_contribution_retained_separately": True,
        "cls_given_spatial_location": False,
        "patch_order": "native_row_major_14x14",
        "projection": "patch_mass_to_pixel_uniform",
        "projection_semantics": (
            "mass_conserving_uniform_distribution_not_native_pixel_evidence"
        ),
        "patch_to_pixel_divisor": 256,
        "patch_to_fine_cell_divisor": 16,
        "intervention_values_source": (
            "real_selected_block_head_qkv_value_endpoint"
        ),
        "fixed_attention_source": "issued_clean_attention_row",
        "frozen_direction_source": "issued_clean_head_direction_unit",
        "baseline_values_source": "issued_frozen_baseline_value_endpoint",
        "target_resolution_reused_without_reresolution": True,
        "replays_qk_or_transformer_block": False,
        "is_complete_attention_attribution": False,
        "claim_boundary": "matched_fixed_clean_attention_token_value_delta_only",
    }
    if frozen.status == DEGENERATE_DIRECTION:
        if frozen.na is None:
            raise StageCValuePathError(
                "degenerate contribution does not contain structured NA"
            )
        method_provenance["status"] = DEGENERATE_DIRECTION
        method_provenance["matched_contribution"] = {
            "status": DEGENERATE_DIRECTION,
            "degeneracy_reasons": [DEGENERATE_DIRECTION],
        }
        return _na_method_result(
            method_id=MATCHED_METHOD_ID,
            projection="patch_mass_to_pixel_uniform",
            provenance=method_provenance,
            na=frozen.na,
        )
    direction_unit = frozen.direction_unit
    if direction_unit is None:
        raise StageCValuePathError(
            "nondegenerate target unexpectedly has no unit direction"
        )
    caller_inputs = inputs
    owned_inputs = inputs.detach().clone()
    with _RuntimeIntegrityGuard(
        model, caller_inputs, owned_inputs,
        context="matched fixed-clean-attention patch-token contribution",
        target=frozen,
    ), _preserve_model_attribution_state(
        model, freeze_parameters=True
    ):
        intervention_values = _capture_selected_head_values(
            model,
            owned_inputs,
            block_index=frozen.block_index,
            head_index=frozen.head_index,
        )
        intervention_scalar = evaluate_frozen_head_direction_target(
            model, clean, owned_inputs, frozen,
            clean_input_sha256=clean_digest,
            intervention_input_sha256=intervention_digest,
        )
        baseline_scalar = float(torch.dot(frozen.u, frozen.g_baseline).item())
        contribution = _matched_token_contributions_from_values(
            candidate=frozen.candidate,
            clean_attention=frozen.clean_attention,
            intervention_values=intervention_values,
            baseline_values=frozen.baseline_values,
            direction_unit=direction_unit,
            conservation_reference=(
                intervention_scalar.value - baseline_scalar
            ),
        )
        _verify_frozen_head_target(
            frozen, model, clean,
            clean_input_sha256=clean_digest,
        )

    patch_map = contribution.patch_map
    if patch_map is None:
        raise StageCValuePathError(
            "OK matched contribution unexpectedly has no patch map"
        )
    patch_values = (
        patch_map.detach().to(device="cpu", dtype=torch.float32).numpy().copy()
    )
    method_provenance["status"] = STATUS_OK
    method_provenance["matched_contribution"] = {
        "status": STATUS_OK,
        "token_contribution_count": TOKENS,
        "token_contributions_sha256": _tensor_sha256(
            contribution.token_contributions
        ),
        "cls_contribution": contribution.cls_contribution,
        "patch_contributions_sha256": _tensor_sha256(
            contribution.patch_contributions
        ),
        "token_sum": contribution.conservation_sum,
        "scalar_delta_f_z_minus_f_x0": contribution.direction_norm,
        "conservation_abs_error": contribution.conservation_error,
        "conservation_tolerance": contribution.conservation_tolerance,
        "patch_support": contribution.patch_support,
        "positive_patch_evidence_fraction": (
            contribution.positive_patch_evidence_fraction
        ),
        "degeneracy_reasons": list(contribution.degeneracy_reasons),
        "patch_support_minimum": PATCH_SUPPORT_MINIMUM,
        "positive_patch_evidence_minimum": (
            POSITIVE_PATCH_EVIDENCE_MINIMUM
        ),
        "all_zero_patch_map_retained": bool(
            int(torch.count_nonzero(patch_map).item()) == 0
        ),
        "intervention_values_sha256": _tensor_sha256(intervention_values),
        "frozen_baseline_values_sha256": _tensor_sha256(
            frozen.baseline_values
        ),
        "fixed_clean_attention_row_sha256": _tensor_sha256(
            frozen.clean_attention
        ),
        "frozen_direction_sha256": _tensor_sha256(direction_unit),
    }
    native = raw_array(patch_values)
    return MethodResult(
        method_id=MATCHED_METHOD_ID,
        method_version=METHOD_VERSION,
        comparison_line=ComparisonLine.VALUE_PATH,
        explanation_target=TARGET_REF,
        fidelity_target=TARGET_REF,
        class_agnostic=True,
        native_tensor=native,
        metric_map=stage_c_metric_ready(patch_values),
        projection="patch_mass_to_pixel_uniform",
        sign_semantics=SignSemantics.SIGNED,
        valid_mask=np.ones(PATCH_SHAPE, dtype=np.bool_),
        provenance=method_provenance,
    )


__all__ = [
    "CLS_TOKEN_INDEX",
    "DEGENERATE_DIRECTION",
    "DEGENERATE_PATCH_SUPPORT",
    "DEGENERATE_POSITIVE_PATCH_EVIDENCE",
    "DIRECTION_RELATIVE_THRESHOLD",
    "DIRECTION_SCALE_FLOOR",
    "EMBED_DIM",
    "ENDPOINT_PARITY_MAX_ABS",
    "FrozenHeadDirectionTarget",
    "FrozenScalarEvaluation",
    "HEADS_PER_BLOCK",
    "HEAD_DIM",
    "HEAD_DIRECTION_TARGET",
    "HeadCandidate",
    "HeadDirectionResolution",
    "IG_ATOL",
    "IG_RTOL",
    "IG_STEPS",
    "IG_STEP_CHUNK_SIZE",
    "MATCHED_EXACT_NAME",
    "MATCHED_METHOD_ID",
    "MATCHED_METHOD_NAME",
    "METHOD_VERSION",
    "MODEL_BLOCKS",
    "MatchedTokenContribution",
    "NNS_EXACT_NAME",
    "NNS_METHOD_ID",
    "NNS_METHOD_NAME",
    "PATCH_SUPPORT_MINIMUM",
    "PATCH_TOKENS",
    "POSITIVE_PATCH_EVIDENCE_MINIMUM",
    "STATUS_OK",
    "StageCValuePathConvergenceError",
    "StageCValuePathError",
    "StageCValuePathIGConvergenceError",
    "TARGET_REF",
    "TOKENS",
    "TOKEN_CONSERVATION_ATOL",
    "TOKEN_CONSERVATION_RTOL",
    "all_head_candidates",
    "chunk_head_candidates",
    "conditional_fixed_clean_attention_head_direction_value_path_ig",
    "conditional_fixed_clean_attention_head_direction_value_path_ig_at_intervention",
    "evaluate_frozen_head_direction_target",
    "matched_fixed_clean_attention_patch_token_contribution",
    "matched_fixed_clean_attention_patch_token_contribution_at_intervention",
    "matched_token_contributions",
    "resolve_fixed_clean_attention_head_direction",
    "resolve_fixed_clean_attention_head_directions",
    "resolve_head_direction_from_endpoints",
]
