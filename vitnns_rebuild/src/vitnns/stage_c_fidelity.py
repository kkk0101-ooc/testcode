"""Stage C fidelity primitives for frozen scalar targets.

This module deliberately stops below execution orchestration.  It defines the
frozen target-response normalization, spatial ranking/corruption operators,
curve summaries, paired tie resolution, sanity rankings, and the registered
Value-path comparison.  It neither selects samples nor writes artifacts.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np
import torch
from torch import nn

from .fidelity_seed import FidelitySeedError, derive_fidelity_seed
from .retention import _deep_freeze, _immutable_array
from .stage_c_class_logit import (
    FrozenClassLogitTarget,
    FrozenScalarEvaluation as ClassScalarEvaluation,
    evaluate_frozen_class_logit_target,
    _tensor_sha256 as _frozen_target_tensor_sha256,
    _verify_frozen_target as _verify_frozen_class_target,
)
from .stage_c_maps import (
    ComparisonLine,
    MethodResult,
    NAReason,
    SignSemantics,
    StructuredNA,
    patch_to_fine_cell_mass,
    pixel_to_fine_cell_mass,
    raw_array,
    signed_rgb_channel_sum,
)
from .stage_c_integrity import CheapTensorSnapshot
from .stage_c_value_path import (
    DEGENERATE_DIRECTION,
    FrozenHeadDirectionTarget,
    FrozenScalarEvaluation as ValueScalarEvaluation,
    evaluate_frozen_head_direction_target,
    _verify_frozen_head_target,
)


TARGET_DELTA_RELATIVE_THRESHOLD = 1e-6
CURVE_FRACTIONS = tuple(index / 20.0 for index in range(21))
TIE_REPLICATES = 5
FIDELITY_BASE_SEED = 20260821
STAGE_C_FIDELITY_SEED_NAMESPACE = f"stage-c-fidelity|{FIDELITY_BASE_SEED}|"
SUFFICIENCY_FRACTION = 0.2
NORMALIZED_INPUT_SHAPE = (1, 3, 224, 224)
VALUE_TARGET_REF = "fixed_clean_attention_cls_head_direction"
CLASS_TARGET_REF = "clean_predicted_class_raw_logit"
NNS_METHOD_ID = (
    "conditional_fixed_clean_attention_head_direction_value_path_ig"
)
MATCHED_METHOD_ID = (
    "matched_fixed_clean_attention_patch_token_contribution"
)
VALUE_METHODS = frozenset({NNS_METHOD_ID, MATCHED_METHOD_ID})
CLASS_METHODS = frozenset(
    {
        "direct_raw_cls_attention_last_block_head_mean",
        "attention_rollout_mean_head_residual_all_blocks",
        "vit_grad_cam_block11_norm1",
        "full_model_class_logit_ig",
    }
)
PIXEL_NATIVE_METHODS = frozenset(
    {NNS_METHOD_ID, "full_model_class_logit_ig"}
)
_SCHEDULE_ISSUER_SECRET = object()
_RESPONSE_ISSUER_SECRET = object()


class FidelityContractError(ValueError):
    """Raised when a fidelity input crosses a frozen contract boundary."""


class FidelityScale(str, Enum):
    PATCH_16 = "patch_16x16"
    FINE_4 = "fine_4x4_cell"

    @property
    def grid_size(self) -> int:
        return 14 if self is FidelityScale.PATCH_16 else 56

    @property
    def unit_size(self) -> int:
        return 16 if self is FidelityScale.PATCH_16 else 4

    @property
    def unit_count(self) -> int:
        return self.grid_size * self.grid_size


class CurveDirection(str, Enum):
    DELETION = "deletion"
    INSERTION = "insertion"


class RankingKind(str, Enum):
    TARGET_ORIENTED_POSITIVE = "target_oriented_positive_tie_permuted"
    GLOBAL_RANDOM = "global_random"
    CENTER_DISTANCE_ROW_MAJOR = "center_distance_then_row_major"


def _parse_scale(value: FidelityScale | str) -> FidelityScale:
    try:
        return FidelityScale(value)
    except (TypeError, ValueError) as exc:
        raise FidelityContractError(f"unknown fidelity scale: {value!r}") from exc


def _parse_direction(value: CurveDirection | str) -> CurveDirection:
    try:
        return CurveDirection(value)
    except (TypeError, ValueError) as exc:
        raise FidelityContractError(f"unknown curve direction: {value!r}") from exc


def _parse_ranking_kind(value: RankingKind | str) -> RankingKind:
    try:
        return RankingKind(value)
    except (TypeError, ValueError) as exc:
        raise FidelityContractError(f"unknown ranking kind: {value!r}") from exc


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be numeric")
    converted = float(value)
    if not math.isfinite(converted):
        raise FidelityContractError(f"{name} must be finite")
    return converted


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


def _json_identity(value: Mapping[str, Any], name: str) -> tuple[Mapping[str, Any], str]:
    if not isinstance(value, Mapping) or not value:
        raise TypeError(f"{name} must be a non-empty mapping")
    plain = _thaw(value)
    if not isinstance(plain, dict):
        raise TypeError(f"{name} must thaw to a JSON object")
    try:
        encoded = json.dumps(
            plain, sort_keys=True, separators=(",", ":"),
            ensure_ascii=True, allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain only JSON-compatible values") from exc
    return _deep_freeze(plain), hashlib.sha256(encoded).hexdigest()


def _method_intervention_sha256(result: MethodResult) -> str | None:
    value = result.provenance.get("intervention_input_sha256")
    if value is None:
        caller = result.provenance.get("caller_context")
        if isinstance(caller, Mapping):
            value = caller.get("intervention_input_sha256")
    if value is not None and (
        not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
    ):
        raise FidelityContractError("MethodResult intervention hash is invalid")
    return value


def _array_sha256(array: np.ndarray) -> str:
    source = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(source.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(
        json.dumps(list(source.shape), separators=(",", ":")).encode("ascii")
    )
    digest.update(b"\0")
    digest.update(source.tobytes(order="C"))
    return digest.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    """Hash tensor exactly as the C3/C4 frozen-target clean binding does."""

    if not isinstance(value, torch.Tensor):
        raise TypeError("value must be a torch.Tensor")
    return _frozen_target_tensor_sha256(value)


def _validate_normalized_tensor(
    value: Any, *, name: str, require_zero: bool = False
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.dtype is not torch.float32:
        raise TypeError(f"{name} must have dtype torch.float32")
    if tuple(value.shape) != NORMALIZED_INPUT_SHAPE:
        raise FidelityContractError(
            f"{name} must have shape {NORMALIZED_INPUT_SHAPE}, got {tuple(value.shape)}"
        )
    if not bool(torch.isfinite(value.detach()).all().item()):
        raise FidelityContractError(f"{name} must contain only finite values")
    if require_zero and bool(torch.count_nonzero(value.detach()).item()):
        raise FidelityContractError(
            f"{name} must be the frozen normalized-zero replacement baseline"
        )
    return value


@runtime_checkable
class FrozenScalarEvaluator(Protocol):
    """Typed target adapter accepted by the fidelity curve evaluator."""

    @property
    def comparison_line(self) -> ComparisonLine: ...

    @property
    def target_ref(self) -> str: ...

    @property
    def target_identity(self) -> Mapping[str, Any]: ...

    def evaluate(self, inputs: torch.Tensor) -> float: ...


class ClassLogitScalarAdapter:
    """Bind one resolver-issued C3 class target for arbitrary ``z``."""

    def __init__(
        self, model: nn.Module, clean_inputs: torch.Tensor,
        target: FrozenClassLogitTarget,
    ) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError("model must be a torch.nn.Module")
        if not isinstance(target, FrozenClassLogitTarget):
            raise TypeError("target must be a resolver-issued FrozenClassLogitTarget")
        clean = _validate_normalized_tensor(clean_inputs, name="clean_inputs")
        self._model = model
        # Keep the resolver-bound clean tensor object.  It is never passed to a
        # model without a private clone, and target verification requires this
        # exact live identity to reject a substituted same-content context.
        self._clean_inputs = clean
        self._target = target
        _verify_frozen_class_target(target, clean, model)
        identity = {
            "comparison_line": ComparisonLine.CLASS_LOGIT.value,
            "target_ref": CLASS_TARGET_REF,
            "clean_input_sha256": target.clean_input_sha256,
            "frozen_target": target.to_dict(),
        }
        self._target_identity, self._target_identity_sha256 = _json_identity(
            identity, "class target identity"
        )

    @property
    def comparison_line(self) -> ComparisonLine:
        return ComparisonLine.CLASS_LOGIT

    @property
    def target_ref(self) -> str:
        return CLASS_TARGET_REF

    @property
    def target_identity(self) -> Mapping[str, Any]:
        return self._target_identity

    @property
    def endpoint_na(self) -> None:
        return None

    def evaluate(self, inputs: torch.Tensor) -> float:
        return self._evaluate(inputs, input_sha256=None)

    def evaluate_prehashed(
        self, inputs: torch.Tensor, *, input_sha256: str
    ) -> float:
        """Evaluate a schedule tensor whose digest was bound at construction."""

        return self._evaluate(inputs, input_sha256=input_sha256)

    def _evaluate(
        self, inputs: torch.Tensor, *, input_sha256: str | None
    ) -> float:
        value = _validate_normalized_tensor(inputs, name="intervention_inputs")
        if value.device != self._clean_inputs.device:
            raise FidelityContractError(
                "intervention and frozen clean context must share a device"
            )
        result = evaluate_frozen_class_logit_target(
            self._model, self._clean_inputs, value, self._target,
            clean_input_sha256=self._target.clean_input_sha256,
            intervention_input_sha256=input_sha256,
        )
        if not isinstance(result, ClassScalarEvaluation):
            raise FidelityContractError("class scalar evaluator returned the wrong type")
        expected_intervention = (
            _frozen_target_tensor_sha256(value)
            if input_sha256 is None else input_sha256
        )
        if (
            result.target_kind != CLASS_TARGET_REF
            or result.class_index != self._target.class_index
            or result.clean_input_sha256 != self._target.clean_input_sha256
            or result.runtime_model_binding_sha256
            != self._target.runtime_model_binding_sha256
            or getattr(result, "intervention_input_sha256", None)
            != expected_intervention
        ):
            raise FidelityContractError(
                "class scalar evaluation does not bind the frozen target and intervention"
            )
        return _finite_float(result.value, "class scalar evaluation")


class ValuePathScalarAdapter:
    """Bind one C4 fixed-clean-A/u/block/head scalar for arbitrary ``z``."""

    def __init__(
        self, model: nn.Module, clean_inputs: torch.Tensor,
        target: FrozenHeadDirectionTarget,
    ) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError("model must be a torch.nn.Module")
        if not isinstance(target, FrozenHeadDirectionTarget):
            raise TypeError("target must be a resolver-issued FrozenHeadDirectionTarget")
        clean = _validate_normalized_tensor(clean_inputs, name="clean_inputs")
        self._model = model
        self._clean_inputs = clean
        self._target = target
        _verify_frozen_head_target(target, model, clean)
        identity = {
            "comparison_line": ComparisonLine.VALUE_PATH.value,
            "target_ref": VALUE_TARGET_REF,
            "clean_input_sha256": target.clean_input_sha256,
            "frozen_target": target.to_dict(),
        }
        self._target_identity, self._target_identity_sha256 = _json_identity(
            identity, "Value-path target identity"
        )

    @property
    def comparison_line(self) -> ComparisonLine:
        return ComparisonLine.VALUE_PATH

    @property
    def target_ref(self) -> str:
        return VALUE_TARGET_REF

    @property
    def target_identity(self) -> Mapping[str, Any]:
        return self._target_identity

    @property
    def endpoint_na(self) -> StructuredNA | None:
        if self._target.status == DEGENERATE_DIRECTION:
            return StructuredNA(
                NAReason.DEGENERATE_TARGET_DELTA,
                "Value-path target-delta degeneracy aligns with direction validity failure",
            )
        return None

    def evaluate(self, inputs: torch.Tensor) -> float:
        return self._evaluate(inputs, input_sha256=None)

    def evaluate_prehashed(
        self, inputs: torch.Tensor, *, input_sha256: str
    ) -> float:
        """Evaluate a schedule tensor whose digest was bound at construction."""

        return self._evaluate(inputs, input_sha256=input_sha256)

    def _evaluate(
        self, inputs: torch.Tensor, *, input_sha256: str | None
    ) -> float:
        value = _validate_normalized_tensor(inputs, name="intervention_inputs")
        if value.device != self._clean_inputs.device:
            raise FidelityContractError(
                "intervention and frozen clean context must share a device"
            )
        result = evaluate_frozen_head_direction_target(
            self._model, self._clean_inputs, value, self._target,
            clean_input_sha256=self._target.clean_input_sha256,
            intervention_input_sha256=input_sha256,
        )
        if not isinstance(result, ValueScalarEvaluation):
            raise FidelityContractError("Value-path scalar evaluator returned the wrong type")
        expected_intervention = (
            _frozen_target_tensor_sha256(value)
            if input_sha256 is None else input_sha256
        )
        if (
            result.target_kind != VALUE_TARGET_REF
            or result.block_index != self._target.block_index
            or result.head_index != self._target.head_index
            or result.query_token != 0
            or result.clean_input_sha256 != self._target.clean_input_sha256
            or result.runtime_model_binding_sha256
            != self._target.runtime_model_binding_sha256
            or getattr(result, "intervention_input_sha256", None)
            != expected_intervention
        ):
            raise FidelityContractError(
                "Value-path scalar evaluation does not bind frozen A/u/head and intervention"
            )
        return _finite_float(result.value, "Value-path scalar evaluation")


def _evaluator_contract(
    evaluator: Any,
) -> tuple[ComparisonLine, str, Mapping[str, Any], str]:
    if not isinstance(evaluator, FrozenScalarEvaluator):
        raise TypeError(
            "evaluator must implement the typed FrozenScalarEvaluator contract; "
            "a bare callable is forbidden"
        )
    try:
        line = ComparisonLine(evaluator.comparison_line)
    except (TypeError, ValueError) as exc:
        raise FidelityContractError("evaluator has an invalid comparison line") from exc
    expected_ref = VALUE_TARGET_REF if line is ComparisonLine.VALUE_PATH else CLASS_TARGET_REF
    if evaluator.target_ref != expected_ref:
        raise FidelityContractError(
            f"evaluator target_ref {evaluator.target_ref!r} does not match {line.value}"
        )
    identity, identity_sha256 = _json_identity(
        evaluator.target_identity, "evaluator.target_identity"
    )
    if identity.get("comparison_line") != line.value:
        raise FidelityContractError(
            "evaluator target identity must bind its comparison line"
        )
    if identity.get("target_ref") != expected_ref:
        raise FidelityContractError("evaluator target identity must bind target_ref")
    clean_sha256 = identity.get("clean_input_sha256")
    if not isinstance(clean_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}", clean_sha256
    ):
        raise FidelityContractError(
            "evaluator target identity must bind clean_input_sha256"
        )
    return line, expected_ref, identity, identity_sha256


def _evaluate_scalar(
    evaluator: FrozenScalarEvaluator,
    inputs: torch.Tensor,
    *,
    expected_identity_sha256: str | None = None,
    expected_input_sha256: str | None = None,
) -> float:
    guard = CheapTensorSnapshot.capture(inputs)
    try:
        evaluate_prehashed = getattr(evaluator, "evaluate_prehashed", None)
        if expected_input_sha256 is not None and callable(evaluate_prehashed):
            raw_value = evaluate_prehashed(
                inputs, input_sha256=expected_input_sha256
            )
        else:
            raw_value = evaluator.evaluate(inputs)
        value = _finite_float(raw_value, "frozen scalar evaluation")
    finally:
        if not guard.verify(inputs).matched:
            raise FidelityContractError(
                "frozen scalar evaluator mutated an intervention tensor"
            )
        if expected_identity_sha256 is not None:
            _line, _target_ref, _identity, current_digest = _evaluator_contract(
                evaluator
            )
            if current_digest != expected_identity_sha256:
                raise FidelityContractError(
                    "frozen evaluator identity changed during scalar evaluation"
                )
    return value


@dataclass(frozen=True, init=False)
class TargetResponse:
    """Endpoint normalization for one exact frozen scalar identity."""

    comparison_line: ComparisonLine
    target_ref: str
    target_identity: Mapping[str, Any]
    target_identity_sha256: str
    clean_value: float | None
    baseline_value: float | None
    delta: float | None
    sigma: int | None
    threshold: float | None
    na: StructuredNA | None = None

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise TypeError("TargetResponse can only be issued by Stage C resolvers")

    @classmethod
    def _create(cls, *, issuer_secret: object, **values: Any) -> "TargetResponse":
        if issuer_secret is not _RESPONSE_ISSUER_SECRET:
            raise TypeError("TargetResponse issuer is invalid")
        instance = object.__new__(cls)
        for name, value in values.items():
            object.__setattr__(instance, name, value)
        instance.__post_init__()
        return instance

    def __post_init__(self) -> None:
        try:
            line = ComparisonLine(self.comparison_line)
        except (TypeError, ValueError) as exc:
            raise FidelityContractError("invalid target comparison line") from exc
        expected_ref = VALUE_TARGET_REF if line is ComparisonLine.VALUE_PATH else CLASS_TARGET_REF
        if self.target_ref != expected_ref:
            raise FidelityContractError("target_ref and comparison_line disagree")
        identity, digest = _json_identity(self.target_identity, "target_identity")
        if self.target_identity_sha256 != digest:
            raise FidelityContractError("target identity SHA-256 does not match identity")
        unresolved_direction = (
            line is ComparisonLine.VALUE_PATH
            and isinstance(self.na, StructuredNA)
            and self.na.reason is NAReason.DEGENERATE_TARGET_DELTA
            and self.clean_value is None
            and self.baseline_value is None
            and self.delta is None
            and self.threshold is None
        )
        if unresolved_direction:
            if self.sigma is not None:
                raise FidelityContractError(
                    "an unresolved degenerate Value-path target must not claim sigma"
                )
            object.__setattr__(self, "comparison_line", line)
            object.__setattr__(self, "target_identity", identity)
            return
        clean = _finite_float(self.clean_value, "clean_value")
        baseline = _finite_float(self.baseline_value, "baseline_value")
        delta = _finite_float(self.delta, "delta")
        threshold = _finite_float(self.threshold, "threshold")
        expected_delta = clean - baseline
        if delta != expected_delta:
            raise FidelityContractError("delta must equal clean_value-baseline_value")
        expected_threshold = TARGET_DELTA_RELATIVE_THRESHOLD * max(
            1.0, abs(clean), abs(baseline)
        )
        if threshold != expected_threshold:
            raise FidelityContractError("target-delta threshold violates the frozen formula")
        degenerate = abs(delta) <= threshold
        if degenerate:
            if self.sigma is not None:
                raise FidelityContractError("a degenerate target must not claim sigma")
            if not isinstance(self.na, StructuredNA) or (
                self.na.reason is not NAReason.DEGENERATE_TARGET_DELTA
            ):
                raise FidelityContractError(
                    "a degenerate target requires degenerate_target_delta NA"
                )
        else:
            expected_sigma = 1 if delta > 0.0 else -1
            if self.sigma != expected_sigma or self.na is not None:
                raise FidelityContractError(
                    "a nondegenerate target requires sign(delta) and no NA"
                )
            if line is ComparisonLine.VALUE_PATH and expected_sigma != 1:
                raise FidelityContractError(
                    "Value-path endpoint delta must be positive by construction"
                )
        object.__setattr__(self, "comparison_line", line)
        object.__setattr__(self, "target_identity", identity)
        object.__setattr__(self, "clean_value", clean)
        object.__setattr__(self, "baseline_value", baseline)
        object.__setattr__(self, "delta", delta)
        object.__setattr__(self, "threshold", threshold)

    @property
    def status(self) -> str:
        return "NA" if self.na is not None else "OK"

    def normalize(self, scalar_value: Any) -> float | StructuredNA:
        if self.na is not None:
            return self.na
        value = _finite_float(scalar_value, "scalar_value")
        assert self.sigma in {-1, 1}
        # Deliberately no clipping: overshoot and undershoot remain observable.
        assert self.baseline_value is not None and self.delta is not None
        normalized = self.sigma * (value - self.baseline_value) / abs(self.delta)
        if not math.isfinite(normalized):
            raise FidelityContractError("normalized target response became non-finite")
        return float(normalized)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "status": self.status,
            "comparison_line": self.comparison_line.value,
            "target_ref": self.target_ref,
            "target_identity": _thaw(self.target_identity),
            "target_identity_sha256": self.target_identity_sha256,
            "clean_value": self.clean_value,
            "baseline_value": self.baseline_value,
            "delta": self.delta,
            "sigma": self.sigma,
            "threshold": self.threshold,
            "formula": "sigma*(t(z)-t(x0))/abs(t(x)-t(x0))",
            "clipped": False,
        }
        if self.na is not None:
            result["na"] = self.na.to_dict()
        return result


def target_response_from_values(
    *,
    comparison_line: ComparisonLine | str,
    target_ref: str,
    target_identity: Mapping[str, Any],
    clean_value: Any,
    baseline_value: Any,
) -> TargetResponse:
    """Construct the exact signed endpoint normalization, retaining degeneracy."""

    line = ComparisonLine(comparison_line)
    clean = _finite_float(clean_value, "clean_value")
    baseline = _finite_float(baseline_value, "baseline_value")
    delta = clean - baseline
    threshold = TARGET_DELTA_RELATIVE_THRESHOLD * max(
        1.0, abs(clean), abs(baseline)
    )
    identity, digest = _json_identity(target_identity, "target_identity")
    degenerate = abs(delta) <= threshold
    detail = None
    if degenerate and line is ComparisonLine.VALUE_PATH:
        detail = "Value-path target-delta degeneracy aligns with direction validity failure"
    return TargetResponse._create(
        issuer_secret=_RESPONSE_ISSUER_SECRET,
        comparison_line=line,
        target_ref=target_ref,
        target_identity=identity,
        target_identity_sha256=digest,
        clean_value=clean,
        baseline_value=baseline,
        delta=delta,
        sigma=None if degenerate else (1 if delta > 0.0 else -1),
        threshold=threshold,
        na=StructuredNA(NAReason.DEGENERATE_TARGET_DELTA, detail)
        if degenerate
        else None,
    )


def resolve_target_response(
    evaluator: FrozenScalarEvaluator,
    clean_inputs: torch.Tensor,
    baseline_inputs: torch.Tensor,
) -> TargetResponse:
    """Evaluate the two endpoints without resolving a new target."""

    clean = _validate_normalized_tensor(clean_inputs, name="clean_inputs")
    baseline = _validate_normalized_tensor(
        baseline_inputs, name="baseline_inputs", require_zero=True
    )
    if clean.device != baseline.device:
        raise FidelityContractError("clean and baseline tensors must share a device")
    line, target_ref, identity, _digest = _evaluator_contract(evaluator)
    endpoint_na = getattr(evaluator, "endpoint_na", None)
    if endpoint_na is not None:
        if (
            line is not ComparisonLine.VALUE_PATH
            or not isinstance(endpoint_na, StructuredNA)
            or endpoint_na.reason is not NAReason.DEGENERATE_TARGET_DELTA
        ):
            raise FidelityContractError(
                "only a degenerate Value-path adapter may predeclare endpoint NA"
            )
        if identity["clean_input_sha256"] != tensor_sha256(clean):
            raise FidelityContractError(
                "degenerate evaluator clean context does not match clean_inputs"
            )
        return TargetResponse._create(
            issuer_secret=_RESPONSE_ISSUER_SECRET,
            comparison_line=line, target_ref=target_ref,
            target_identity=identity, target_identity_sha256=_digest,
            clean_value=None, baseline_value=None, delta=None, sigma=None,
            threshold=None, na=endpoint_na,
        )
    clean_value = _evaluate_scalar(
        evaluator, clean, expected_identity_sha256=_digest,
        expected_input_sha256=identity["clean_input_sha256"],
    )
    baseline_sha256 = tensor_sha256(baseline)
    baseline_value = _evaluate_scalar(
        evaluator, baseline, expected_identity_sha256=_digest,
        expected_input_sha256=baseline_sha256,
    )
    # Re-check identity after arbitrary model callbacks.
    _line_after, _ref_after, identity_after, _ = _evaluator_contract(evaluator)
    if identity_after != identity:
        raise FidelityContractError("frozen evaluator identity changed during endpoints")
    return target_response_from_values(
        comparison_line=line, target_ref=target_ref, target_identity=identity,
        clean_value=clean_value, baseline_value=baseline_value,
    )


@dataclass(frozen=True)
class UnitMassMap:
    method_id: str
    comparison_line: ComparisonLine
    target_ref: str
    target_identity_sha256: str
    source_clean_input_sha256: str
    scale: FidelityScale
    sign_semantics: SignSemantics
    masses: np.ndarray
    sha256: str

    def __post_init__(self) -> None:
        if self.method_id not in VALUE_METHODS | CLASS_METHODS:
            raise FidelityContractError("method is not fidelity eligible")
        line = ComparisonLine(self.comparison_line)
        scale = _parse_scale(self.scale)
        signs = SignSemantics(self.sign_semantics)
        for name in ("target_identity_sha256", "source_clean_input_sha256"):
            digest_value = getattr(self, name)
            if not isinstance(digest_value, str) or not re.fullmatch(
                r"[0-9a-f]{64}", digest_value
            ):
                raise FidelityContractError(f"{name} must be a lowercase SHA-256")
        array = np.asarray(self.masses)
        if array.dtype != np.float32 or array.shape != (scale.grid_size, scale.grid_size):
            raise FidelityContractError(
                f"unit masses must be float32 {(scale.grid_size, scale.grid_size)}"
            )
        if not np.all(np.isfinite(array)):
            raise FidelityContractError("unit masses must be finite")
        if signs is SignSemantics.NONNEGATIVE and np.any(array < 0):
            raise FidelityContractError("nonnegative unit masses contain negatives")
        digest = _array_sha256(array)
        if self.sha256 != digest:
            raise FidelityContractError("unit mass SHA-256 mismatch")
        object.__setattr__(self, "comparison_line", line)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "sign_semantics", signs)
        object.__setattr__(self, "masses", _immutable_array(array))

    def to_dict(self) -> dict[str, Any]:
        return {
            "method_id": self.method_id,
            "comparison_line": self.comparison_line.value,
            "target_ref": self.target_ref,
            "target_identity_sha256": self.target_identity_sha256,
            "source_clean_input_sha256": self.source_clean_input_sha256,
            "scale": self.scale.value,
            "shape": list(self.masses.shape),
            "dtype": "float32",
            "sign_semantics": self.sign_semantics.value,
            "sha256": self.sha256,
            "claim_boundary": (
                "patch ranking only, not pixel novelty"
                if self.scale is FidelityScale.PATCH_16
                else "4x4-cell/subpatch fidelity, not single-pixel fidelity"
            ),
        }


def method_unit_masses(
    result: MethodResult,
    response: TargetResponse,
    scale: FidelityScale | str,
) -> UnitMassMap:
    """Project a successful registered method into signed spatial units."""

    if not isinstance(result, MethodResult):
        raise TypeError("result must be a MethodResult")
    if not isinstance(response, TargetResponse):
        raise TypeError("response must be a TargetResponse")
    parsed_scale = _parse_scale(scale)
    if result.na is not None:
        raise FidelityContractError(
            f"method result is NA and cannot be ranked: {result.na.reason.value}"
        )
    if result.comparison_line is not response.comparison_line:
        raise FidelityContractError("method and frozen target cross comparison lines")
    if result.fidelity_target != response.target_ref:
        raise FidelityContractError("method fidelity target does not match frozen target")
    if not bool(np.all(result.valid_mask)):
        raise FidelityContractError(
            "clean fidelity rankings require an all-valid 14x14 MethodResult mask"
        )
    allowed = (
        VALUE_METHODS
        if response.comparison_line is ComparisonLine.VALUE_PATH
        else CLASS_METHODS
    )
    if result.method_id not in allowed:
        raise FidelityContractError("method is not registered for this fidelity line")
    if result.native_tensor is None or result.metric_map is None:
        raise FidelityContractError("successful fidelity method lacks its numeric map")
    result_target: Any = result.provenance.get("target")
    if isinstance(result_target, Mapping):
        clean_hash = result_target.get("clean_input_sha256")
        result_identity = {
            "comparison_line": result.comparison_line.value,
            "target_ref": result.fidelity_target,
            "clean_input_sha256": clean_hash,
            "frozen_target": _thaw(result_target),
        }
    else:
        caller_context = result.provenance.get("caller_context")
        binding = (
            caller_context.get("fidelity_target_identity")
            if isinstance(caller_context, Mapping)
            else None
        )
        if not isinstance(binding, Mapping):
            raise FidelityContractError(
                "class-agnostic fidelity map requires caller_context."
                "fidelity_target_identity binding"
            )
        result_identity = _thaw(binding)
        clean_hash = result_identity.get("clean_input_sha256")
    _frozen_result_identity, result_identity_sha256 = _json_identity(
        result_identity, "MethodResult fidelity target identity"
    )
    if result_identity_sha256 != response.target_identity_sha256:
        raise FidelityContractError(
            "MethodResult does not bind the exact frozen target identity"
        )
    if clean_hash != response.target_identity.get("clean_input_sha256"):
        raise FidelityContractError(
            "MethodResult clean input does not match the frozen target context"
        )
    observed_intervention = _method_intervention_sha256(result)
    if observed_intervention is not None and observed_intervention != clean_hash:
        raise FidelityContractError(
            "fidelity ranking source is not the frozen clean-input map"
        )
    if parsed_scale is FidelityScale.PATCH_16:
        masses = np.asarray(result.metric_map.data, dtype=np.float32)
    elif result.method_id in PIXEL_NATIVE_METHODS:
        pixels = signed_rgb_channel_sum(result.native_tensor)
        masses = np.asarray(pixel_to_fine_cell_mass(pixels).data, dtype=np.float32)
    else:
        # This is mass-conserving subpatch expansion, not native pixel evidence.
        masses = np.asarray(
            patch_to_fine_cell_mass(result.metric_map).data, dtype=np.float32
        )
    owned = np.array(masses, dtype=np.float32, copy=True)
    return UnitMassMap(
        method_id=result.method_id,
        comparison_line=result.comparison_line,
        target_ref=result.fidelity_target,
        target_identity_sha256=response.target_identity_sha256,
        source_clean_input_sha256=clean_hash,
        scale=parsed_scale,
        sign_semantics=result.sign_semantics,
        masses=owned,
        sha256=_array_sha256(owned),
    )


def target_oriented_scores(
    unit_map: UnitMassMap, response: TargetResponse
) -> np.ndarray:
    """Return ``positive_part(sigma * signed_mass)`` after aggregation."""

    if not isinstance(unit_map, UnitMassMap):
        raise TypeError("unit_map must be UnitMassMap")
    if not isinstance(response, TargetResponse):
        raise TypeError("response must be TargetResponse")
    if response.na is not None or response.sigma not in {-1, 1}:
        raise FidelityContractError("a degenerate target has no ranking orientation")
    if (
        unit_map.comparison_line is not response.comparison_line
        or unit_map.target_ref != response.target_ref
        or unit_map.target_identity_sha256 != response.target_identity_sha256
        or unit_map.source_clean_input_sha256
        != response.target_identity.get("clean_input_sha256")
    ):
        raise FidelityContractError("unit map and target response do not pair")
    oriented = np.maximum(
        np.float32(response.sigma) * unit_map.masses, np.float32(0.0)
    ).astype(np.float32, copy=False)
    return _immutable_array(oriented)


def fidelity_target_provenance_binding(response: TargetResponse) -> dict[str, Any]:
    """Return the exact caller-context fragment for class-agnostic C2 maps."""

    if not isinstance(response, TargetResponse):
        raise TypeError("response must be TargetResponse")
    return {"fidelity_target_identity": _thaw(response.target_identity)}


def fidelity_seed_for_namespace(
    namespace: str, sample_id: str, block: int, head: int, replicate: int
) -> int:
    """Derive a fidelity seed from an explicit, auditable namespace.

    The helper deliberately accepts the complete namespace prefix, including
    its trailing separator.  This keeps Stage C's historic seed schedule
    unchanged while allowing later stages to use their separately registered
    namespace without copying the hashing formula.
    """

    try:
        return derive_fidelity_seed(
            namespace, sample_id, block, head, replicate,
            repeat_count=TIE_REPLICATES,
        )
    except FidelitySeedError as error:
        message = str(error).replace("repeat_index", "replicate")
        raise FidelityContractError(message) from error


def fidelity_seed(
    sample_id: str, block: int, head: int, replicate: int
) -> int:
    """Derive the exact preregistered Stage C unsigned 64-bit seed."""

    return fidelity_seed_for_namespace(
        STAGE_C_FIDELITY_SEED_NAMESPACE, sample_id, block, head, replicate
    )


def _validate_scores(scores: Any, unit_count: int | None = None) -> np.ndarray:
    source = np.asarray(scores)
    if source.dtype != np.float32 or source.ndim != 1 or source.size == 0:
        raise TypeError("scores must be a non-empty one-dimensional float32 array")
    if unit_count is not None and source.size != unit_count:
        raise FidelityContractError("score count does not match fidelity scale")
    if not np.all(np.isfinite(source)) or np.any(source < 0):
        raise FidelityContractError("scores must be finite and nonnegative")
    return source


def _validate_ranking(ranking: Any, unit_count: int) -> np.ndarray:
    source = np.asarray(ranking)
    if source.dtype != np.int64 or source.shape != (unit_count,):
        raise TypeError(f"ranking must be an int64 permutation of length {unit_count}")
    if not np.array_equal(np.sort(source), np.arange(unit_count, dtype=np.int64)):
        raise FidelityContractError("ranking must contain every unit exactly once")
    return source


def tie_permuted_ranking(scores: Any, seed: int) -> np.ndarray:
    """Rank descending score, using one paired random priority within ties."""

    values = _validate_scores(scores)
    if type(seed) is not int or seed < 0 or seed >= 2**64:
        raise TypeError("seed must be an unsigned 64-bit integer")
    random_order = np.random.default_rng(seed).permutation(values.size)
    priority = np.empty(values.size, dtype=np.int64)
    priority[random_order] = np.arange(values.size, dtype=np.int64)
    order = np.lexsort((priority, -values.astype(np.float64))).astype(np.int64)
    return _immutable_array(order)


def tie_priority_sha256(unit_count: int, seed: int) -> str:
    """Hash the method-independent random priority used to resolve ties."""

    if type(unit_count) is not int or unit_count <= 0:
        raise TypeError("unit_count must be a positive integer")
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise TypeError("seed must be an unsigned 64-bit integer")
    random_order = np.random.default_rng(seed).permutation(unit_count)
    priority = np.empty(unit_count, dtype=np.int64)
    priority[random_order] = np.arange(unit_count, dtype=np.int64)
    return _array_sha256(priority)


def global_random_ranking(unit_count: int, seed: int) -> np.ndarray:
    if type(unit_count) is not int or unit_count <= 0:
        raise TypeError("unit_count must be a positive integer")
    if type(seed) is not int or seed < 0 or seed >= 2**64:
        raise TypeError("seed must be an unsigned 64-bit integer")
    return _immutable_array(
        np.random.default_rng(seed).permutation(unit_count).astype(np.int64)
    )


def center_distance_ranking(scale: FidelityScale | str) -> np.ndarray:
    parsed = _parse_scale(scale)
    grid = parsed.grid_size
    centers = (np.arange(grid, dtype=np.float64) + 0.5) * parsed.unit_size
    rows, cols = np.meshgrid(centers, centers, indexing="ij")
    distance = np.square(rows - 112.0) + np.square(cols - 112.0)
    row_major = np.arange(parsed.unit_count, dtype=np.int64)
    order = np.lexsort((row_major, distance.reshape(-1))).astype(np.int64)
    return _immutable_array(order)


def curve_unit_counts(
    unit_count: int, fractions: tuple[float, ...] = CURVE_FRACTIONS
) -> tuple[int, ...]:
    if type(unit_count) is not int or unit_count <= 0:
        raise TypeError("unit_count must be a positive integer")
    if fractions != CURVE_FRACTIONS:
        raise FidelityContractError("curve fractions must be the frozen 21-point grid")
    return (0,) + tuple(
        int(math.ceil(fraction * unit_count)) for fraction in fractions[1:]
    )


def actual_curve_fractions(unit_count: int) -> tuple[float, ...]:
    return tuple(count / unit_count for count in curve_unit_counts(unit_count))


def trapezoid_auc(actual_fractions: Any, responses: Any) -> float:
    x = np.asarray(actual_fractions, dtype=np.float64)
    y = np.asarray(responses, dtype=np.float64)
    if x.shape != (21,) or y.shape != (21,):
        raise FidelityContractError("AUC requires exactly 21 x/y values")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise FidelityContractError("AUC inputs must be finite")
    if x[0] != 0.0 or x[-1] != 1.0 or np.any(np.diff(x) <= 0.0):
        raise FidelityContractError("AUC x coordinates must increase from 0 to 1")
    return float(np.sum(np.diff(x) * (y[:-1] + y[1:]) * 0.5, dtype=np.float64))


class InterventionSchedule:
    """One integrity-checked, reusable 21-tensor corruption schedule."""

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("InterventionSchedule is immutable after construction")
        object.__setattr__(self, name, value)

    def __init__(
        self,
        *,
        clean_inputs: torch.Tensor,
        baseline_inputs: torch.Tensor,
        ranking: np.ndarray,
        scores: np.ndarray | None,
        source_method_id: str | None,
        source_target_identity_sha256: str | None,
        source_unit_map_sha256: str | None,
        scale: FidelityScale,
        direction: CurveDirection,
        ranking_kind: RankingKind,
        sample_id: str,
        block: int,
        head: int,
        seed: int | None,
        replicate: int | None,
        seed_namespace: str,
        issuer_secret: object,
    ) -> None:
        if issuer_secret is not _SCHEDULE_ISSUER_SECRET:
            raise TypeError(
                "InterventionSchedule can only be issued by Stage C builders"
            )
        clean = _validate_normalized_tensor(clean_inputs, name="clean_inputs")
        baseline = _validate_normalized_tensor(
            baseline_inputs, name="baseline_inputs", require_zero=True
        )
        if clean.device != baseline.device:
            raise FidelityContractError("clean and baseline tensors must share a device")
        order = _validate_ranking(ranking, scale.unit_count)
        if ranking_kind is RankingKind.TARGET_ORIENTED_POSITIVE:
            if source_method_id not in VALUE_METHODS | CLASS_METHODS:
                raise FidelityContractError(
                    "target-oriented schedule requires a registered source method"
                )
            for name, digest in (
                ("source_target_identity_sha256", source_target_identity_sha256),
                ("source_unit_map_sha256", source_unit_map_sha256),
            ):
                if not isinstance(digest, str) or not re.fullmatch(
                    r"[0-9a-f]{64}", digest
                ):
                    raise FidelityContractError(f"{name} must be a lowercase SHA-256")
        elif any(
            value is not None for value in (
                source_method_id, source_target_identity_sha256, source_unit_map_sha256
            )
        ):
            raise FidelityContractError(
                "sanity schedules must not claim a method-map ranking source"
            )
        # Validate context even for the deterministic center sanity so every
        # schedule remains attributable to one registered sample/config.
        fidelity_seed_for_namespace(seed_namespace, sample_id, block, head, 0)
        if ranking_kind is RankingKind.CENTER_DISTANCE_ROW_MAJOR:
            if seed is not None or replicate is not None:
                raise FidelityContractError("center ranking is deterministic and unseeded")
            expected_center = center_distance_ranking(scale)
            if not np.array_equal(order, expected_center):
                raise FidelityContractError(
                    "center ranking must use distance then row-major order"
                )
        else:
            if type(replicate) is not int or not 0 <= replicate < TIE_REPLICATES:
                raise TypeError("seeded ranking replicate must be an integer in [0,4]")
            expected_seed = fidelity_seed_for_namespace(
                seed_namespace, sample_id, block, head, replicate
            )
            if seed != expected_seed:
                raise FidelityContractError(
                    "seed does not match the preregistered sample/block/head/replicate hash"
                )
            if ranking_kind is RankingKind.GLOBAL_RANDOM:
                expected_order = global_random_ranking(scale.unit_count, expected_seed)
                if not np.array_equal(order, expected_order):
                    raise FidelityContractError(
                        "global-random ranking does not match its registered seed"
                    )
            else:
                if scores is None:
                    raise FidelityContractError(
                        "target-oriented ranking requires its exact score vector"
                    )
                score_values = _validate_scores(scores, scale.unit_count)
                expected_order = tie_permuted_ranking(score_values, expected_seed)
                if not np.array_equal(order, expected_order):
                    raise FidelityContractError(
                        "target-oriented ranking does not match scores and tie seed"
                    )
        self.scale = scale
        self.direction = direction
        self.ranking_kind = ranking_kind
        self.sample_id = sample_id
        self.block = block
        self.head = head
        self.seed = seed
        self.seed_namespace = seed_namespace
        self.replicate = replicate
        self.tie_priority_sha256 = (
            None if seed is None else tie_priority_sha256(scale.unit_count, seed)
        )
        self.source_method_id = source_method_id
        self.source_target_identity_sha256 = source_target_identity_sha256
        self.source_unit_map_sha256 = source_unit_map_sha256
        self.score_sha256 = (
            None if scores is None else _array_sha256(np.asarray(scores, np.float32))
        )
        self.unit_counts = curve_unit_counts(scale.unit_count)
        self.actual_fractions = actual_curve_fractions(scale.unit_count)
        self.ranking = _immutable_array(order)
        self.ranking_sha256 = _array_sha256(order)
        owned_clean = clean.detach().clone()
        owned_baseline = baseline.detach().clone()
        self.clean_sha256 = tensor_sha256(owned_clean)
        self.baseline_sha256 = tensor_sha256(owned_baseline)
        pair_digest = hashlib.sha256(
            f"{self.clean_sha256}|{self.baseline_sha256}".encode("ascii")
        ).hexdigest()
        self.input_pair_sha256 = pair_digest
        interventions: list[torch.Tensor] = []
        cell = scale.unit_size
        for count in self.unit_counts:
            selected = np.zeros(scale.unit_count, dtype=np.bool_)
            selected[order[:count]] = True
            unit_mask = selected.reshape(scale.grid_size, scale.grid_size)
            pixel_mask = np.repeat(np.repeat(unit_mask, cell, axis=0), cell, axis=1)
            mask = torch.from_numpy(pixel_mask.copy()).to(device=clean.device)
            mask = mask[None, None, :, :].expand_as(owned_clean)
            if direction is CurveDirection.DELETION:
                tensor = torch.where(mask, owned_baseline, owned_clean)
            else:
                tensor = torch.where(mask, owned_clean, owned_baseline)
            interventions.append(tensor.detach().clone())
        self._interventions = tuple(interventions)
        self.intervention_sha256 = tuple(
            tensor_sha256(value) for value in self._interventions
        )
        self._intervention_live_state = tuple(
            CheapTensorSnapshot.capture(value) for value in self._interventions
        )
        self._sealed = True

    def __len__(self) -> int:
        return len(self._interventions)

    def tensor_at(self, index: int) -> torch.Tensor:
        if type(index) is not int or not 0 <= index < len(self._interventions):
            raise IndexError(index)
        self._verify_point_integrity(index)
        return self._interventions[index].detach().clone()

    def evaluate_at(self, index: int, consumer: Any) -> Any:
        """Apply a scalar consumer to the one shared guarded tensor."""

        if type(index) is not int or not 0 <= index < len(self._interventions):
            raise IndexError(index)
        if not callable(consumer):
            raise TypeError("consumer must be callable")
        shared = self._interventions[index]
        live_state = self._intervention_live_state[index]
        if not live_state.verify(shared).matched:
            raise FidelityContractError(
                f"intervention tensor {index} changed live state"
            )
        try:
            result = consumer(shared)
        finally:
            if not live_state.verify(shared).matched:
                raise FidelityContractError(
                    f"intervention tensor {index} was mutated"
                )
        if isinstance(result, torch.Tensor):
            raise FidelityContractError(
                "intervention consumer must return a scalar, not a tensor alias"
            )
        return result

    def _verify_point_integrity(self, index: int) -> None:
        if not self._intervention_live_state[index].verify(
            self._interventions[index]
        ).matched:
            raise FidelityContractError(
                f"intervention tensor {index} changed live state"
            )

    def verify_integrity(self) -> None:
        if len(self._interventions) != 21:
            raise FidelityContractError("intervention schedule must contain 21 tensors")
        current = tuple(tensor_sha256(value) for value in self._interventions)
        if current != self.intervention_sha256:
            raise FidelityContractError("an intervention tensor was mutated")
        if self.direction is CurveDirection.DELETION:
            expected_first, expected_last = self.clean_sha256, self.baseline_sha256
        else:
            expected_first, expected_last = self.baseline_sha256, self.clean_sha256
        if current[0] != expected_first or current[-1] != expected_last:
            raise FidelityContractError("corruption endpoints violate deletion/insertion identity")

    def to_dict(self) -> dict[str, Any]:
        self.verify_integrity()
        result = {
            "scale": self.scale.value,
            "direction": self.direction.value,
            "ranking_kind": self.ranking_kind.value,
            "sample_id": self.sample_id,
            "block": self.block,
            "head": self.head,
            "seed": self.seed,
            "replicate": self.replicate,
            "tie_priority_sha256": self.tie_priority_sha256,
            "source_method_id": self.source_method_id,
            "source_target_identity_sha256": self.source_target_identity_sha256,
            "source_unit_map_sha256": self.source_unit_map_sha256,
            "score_sha256": self.score_sha256,
            "curve_fractions": list(CURVE_FRACTIONS),
            "unit_counts": list(self.unit_counts),
            "actual_fractions": list(self.actual_fractions),
            "ranking_sha256": self.ranking_sha256,
            "clean_sha256": self.clean_sha256,
            "baseline_sha256": self.baseline_sha256,
            "input_pair_sha256": self.input_pair_sha256,
            "intervention_sha256": list(self.intervention_sha256),
            "replacement": "normalized_zero_rgb_channels_together",
        }
        if self.seed_namespace != STAGE_C_FIDELITY_SEED_NAMESPACE:
            result["seed_namespace"] = self.seed_namespace
        return result


def build_intervention_schedule(
    clean_inputs: torch.Tensor,
    baseline_inputs: torch.Tensor,
    ranking: Any,
    *,
    scores: Any | None = None,
    source_method_id: str | None = None,
    source_target_identity_sha256: str | None = None,
    source_unit_map_sha256: str | None = None,
    scale: FidelityScale | str,
    direction: CurveDirection | str,
    ranking_kind: RankingKind | str = RankingKind.TARGET_ORIENTED_POSITIVE,
    sample_id: str,
    block: int,
    head: int,
    seed: int | None = None,
    replicate: int | None = None,
    seed_namespace: str = STAGE_C_FIDELITY_SEED_NAMESPACE,
) -> InterventionSchedule:
    parsed_kind = _parse_ranking_kind(ranking_kind)
    if parsed_kind is RankingKind.TARGET_ORIENTED_POSITIVE:
        raise FidelityContractError(
            "target-oriented schedules must be built from UnitMassMap via "
            "build_method_intervention_schedule"
        )
    return InterventionSchedule(
        clean_inputs=clean_inputs, baseline_inputs=baseline_inputs,
        ranking=_validate_ranking(ranking, _parse_scale(scale).unit_count),
        scores=None if scores is None else _validate_scores(
            scores, _parse_scale(scale).unit_count
        ),
        source_method_id=source_method_id,
        source_target_identity_sha256=source_target_identity_sha256,
        source_unit_map_sha256=source_unit_map_sha256,
        scale=_parse_scale(scale), direction=_parse_direction(direction),
        ranking_kind=parsed_kind,
        sample_id=sample_id, block=block, head=head,
        seed=seed, replicate=replicate,
        seed_namespace=seed_namespace,
        issuer_secret=_SCHEDULE_ISSUER_SECRET,
    )


def build_method_intervention_schedule(
    clean_inputs: torch.Tensor, baseline_inputs: torch.Tensor,
    unit_map: UnitMassMap, response: TargetResponse, *,
    sample_id: str, block: int, head: int, replicate: int,
    direction: CurveDirection | str,
    seed_namespace: str = STAGE_C_FIDELITY_SEED_NAMESPACE,
) -> InterventionSchedule:
    """Build a target-oriented schedule from its exact registered method map."""

    if not isinstance(unit_map, UnitMassMap):
        raise TypeError("unit_map must be UnitMassMap")
    scores_2d = target_oriented_scores(unit_map, response)
    if tensor_sha256(clean_inputs) != unit_map.source_clean_input_sha256:
        raise FidelityContractError(
            "method schedule clean tensor does not match UnitMassMap source"
        )
    scores = np.asarray(scores_2d, dtype=np.float32).reshape(-1)
    seed = fidelity_seed_for_namespace(
        seed_namespace, sample_id, block, head, replicate
    )
    ranking = tie_permuted_ranking(scores, seed)
    return InterventionSchedule(
        clean_inputs=clean_inputs, baseline_inputs=baseline_inputs, ranking=ranking,
        scores=scores,
        source_method_id=unit_map.method_id,
        source_target_identity_sha256=unit_map.target_identity_sha256,
        source_unit_map_sha256=unit_map.sha256,
        scale=unit_map.scale, direction=_parse_direction(direction),
        ranking_kind=RankingKind.TARGET_ORIENTED_POSITIVE,
        sample_id=sample_id, block=block, head=head, seed=seed,
        replicate=replicate, seed_namespace=seed_namespace,
        issuer_secret=_SCHEDULE_ISSUER_SECRET,
    )


@dataclass(frozen=True)
class FidelityCurveResult:
    method_id: str
    comparison_line: ComparisonLine
    target_ref: str
    target_identity_sha256: str
    scale: FidelityScale
    direction: CurveDirection
    ranking_kind: RankingKind
    sample_id: str
    block: int
    head: int
    replicate: int | None
    seed: int | None
    tie_priority_sha256: str | None
    source_unit_map_sha256: str | None
    score_sha256: str | None
    input_pair_sha256: str
    ranking_sha256: str
    intervention_sha256: tuple[str, ...]
    unit_counts: tuple[int, ...]
    actual_fractions: tuple[float, ...]
    responses: tuple[float, ...] | None
    auc: float | None
    na: StructuredNA | None = None
    seed_namespace: str = STAGE_C_FIDELITY_SEED_NAMESPACE

    def __post_init__(self) -> None:
        if self.method_id not in VALUE_METHODS | CLASS_METHODS:
            raise FidelityContractError("unknown fidelity method id")
        line = ComparisonLine(self.comparison_line)
        scale = _parse_scale(self.scale)
        direction = _parse_direction(self.direction)
        ranking_kind = _parse_ranking_kind(self.ranking_kind)
        if not isinstance(self.seed_namespace, str) or not self.seed_namespace.endswith("|"):
            raise FidelityContractError(
                "curve seed_namespace must be a string ending in '|'"
            )
        fidelity_seed_for_namespace(
            self.seed_namespace, self.sample_id, self.block, self.head, 0
        )
        if ranking_kind is RankingKind.CENTER_DISTANCE_ROW_MAJOR:
            if self.seed is not None or self.replicate is not None:
                raise FidelityContractError("center curve must be unseeded")
            if self.tie_priority_sha256 is not None:
                raise FidelityContractError("center curve has no tie priority")
            if self.source_unit_map_sha256 is not None or self.score_sha256 is not None:
                raise FidelityContractError("center curve has no method score source")
        else:
            if type(self.replicate) is not int or not 0 <= self.replicate < 5:
                raise FidelityContractError("seeded curve requires replicate 0..4")
            if self.seed != fidelity_seed_for_namespace(
                self.seed_namespace, self.sample_id, self.block, self.head,
                self.replicate
            ):
                raise FidelityContractError("curve seed violates the frozen namespace")
            expected_priority = tie_priority_sha256(
                _parse_scale(self.scale).unit_count, self.seed
            )
            if self.tie_priority_sha256 != expected_priority:
                raise FidelityContractError("curve tie-priority hash mismatch")
            if ranking_kind is RankingKind.TARGET_ORIENTED_POSITIVE:
                for name in ("source_unit_map_sha256", "score_sha256"):
                    digest_value = getattr(self, name)
                    if not isinstance(digest_value, str) or not re.fullmatch(
                        r"[0-9a-f]{64}", digest_value
                    ):
                        raise FidelityContractError(f"{name} must be a lowercase SHA-256")
            elif self.source_unit_map_sha256 is not None or self.score_sha256 is not None:
                raise FidelityContractError("global random curve has no method score source")
        expected_line = (
            ComparisonLine.VALUE_PATH
            if self.method_id in VALUE_METHODS
            else ComparisonLine.CLASS_LOGIT
            if self.method_id in CLASS_METHODS else line
        )
        expected_ref = (
            VALUE_TARGET_REF
            if expected_line is ComparisonLine.VALUE_PATH
            else CLASS_TARGET_REF
        )
        if line is not expected_line or self.target_ref != expected_ref:
            raise FidelityContractError(
                "fidelity curve method/line/target contract mismatch"
            )
        for digest_name, digest in (
            ("target identity", self.target_identity_sha256),
            ("input pair", self.input_pair_sha256),
            ("ranking", self.ranking_sha256),
        ):
            if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise FidelityContractError(f"{digest_name} SHA-256 is invalid")
        if len(self.intervention_sha256) != 21 or any(
            not isinstance(item, str) or len(item) != 64
            for item in self.intervention_sha256
        ):
            raise FidelityContractError("curve must bind 21 intervention hashes")
        if self.unit_counts != curve_unit_counts(scale.unit_count):
            raise FidelityContractError("curve unit counts violate ceil scheduling")
        if self.actual_fractions != actual_curve_fractions(scale.unit_count):
            raise FidelityContractError("curve x-axis must use actual k/N")
        if self.na is None:
            if self.responses is None or len(self.responses) != 21:
                raise FidelityContractError("successful curve requires 21 responses")
            values = tuple(_finite_float(value, "curve response") for value in self.responses)
            expected_auc = trapezoid_auc(self.actual_fractions, values)
            auc = _finite_float(self.auc, "auc")
            if auc != expected_auc:
                raise FidelityContractError("curve AUC is not actual-fraction trapezoid")
            object.__setattr__(self, "responses", values)
            object.__setattr__(self, "auc", auc)
        else:
            if not isinstance(self.na, StructuredNA):
                raise TypeError("na must be StructuredNA")
            if self.responses is not None or self.auc is not None:
                raise FidelityContractError("NA curve cannot contain fake responses/AUC")
        object.__setattr__(self, "comparison_line", line)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "ranking_kind", ranking_kind)

    @property
    def status(self) -> str:
        return "NA" if self.na is not None else "OK"

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "status": self.status, "method_id": self.method_id,
            "comparison_line": self.comparison_line.value,
            "target_ref": self.target_ref,
            "target_identity_sha256": self.target_identity_sha256,
            "scale": self.scale.value, "direction": self.direction.value,
            "ranking_kind": self.ranking_kind.value,
            "sample_id": self.sample_id, "block": self.block,
            "head": self.head,
            "replicate": self.replicate, "seed": self.seed,
            "tie_priority_sha256": self.tie_priority_sha256,
            "source_unit_map_sha256": self.source_unit_map_sha256,
            "score_sha256": self.score_sha256,
            "input_pair_sha256": self.input_pair_sha256,
            "ranking_sha256": self.ranking_sha256,
            "intervention_sha256": list(self.intervention_sha256),
            "curve_fractions": list(CURVE_FRACTIONS),
            "unit_counts": list(self.unit_counts),
            "actual_fractions": list(self.actual_fractions),
            "responses": None if self.responses is None else list(self.responses),
            "auc": self.auc, "response_clipped": False,
            "auc_integration": "trapezoid_over_actual_k_divided_by_units",
        }
        if self.seed_namespace != STAGE_C_FIDELITY_SEED_NAMESPACE:
            result["seed_namespace"] = self.seed_namespace
        if self.na is not None:
            result["na"] = self.na.to_dict()
        return result


def evaluate_fidelity_curve(
    evaluator: FrozenScalarEvaluator,
    response: TargetResponse,
    schedule: InterventionSchedule,
    *,
    method_id: str,
) -> FidelityCurveResult:
    """Evaluate one prebuilt shared schedule against one frozen scalar."""

    if not isinstance(response, TargetResponse):
        raise TypeError("response must be TargetResponse")
    if not isinstance(schedule, InterventionSchedule):
        raise TypeError("schedule must be InterventionSchedule")
    line, target_ref, _identity, digest = _evaluator_contract(evaluator)
    if (
        line is not response.comparison_line
        or target_ref != response.target_ref
        or digest != response.target_identity_sha256
    ):
        raise FidelityContractError("curve evaluator does not match target response")
    allowed = VALUE_METHODS if line is ComparisonLine.VALUE_PATH else CLASS_METHODS
    if method_id not in allowed:
        raise FidelityContractError("method is not eligible for this target line")
    if schedule.ranking_kind is RankingKind.TARGET_ORIENTED_POSITIVE and (
        schedule.source_method_id != method_id
        or schedule.source_target_identity_sha256 != digest
    ):
        raise FidelityContractError(
            "target-oriented schedule does not come from this method/target map"
        )
    clean_identity_sha256 = _identity["clean_input_sha256"]
    if schedule.clean_sha256 != clean_identity_sha256:
        raise FidelityContractError(
            "intervention schedule clean tensor does not match the frozen target context"
        )
    values: tuple[float, ...] | None
    auc: float | None
    schedule.verify_integrity()
    try:
        if response.na is not None:
            values = None
            auc = None
        else:
            computed: list[float] = []
            for index in range(len(schedule)):
                scalar = schedule.evaluate_at(
                    index,
                    lambda intervention, index=index: _evaluate_scalar(
                        evaluator, intervention,
                        expected_identity_sha256=digest,
                        expected_input_sha256=schedule.intervention_sha256[index],
                    ),
                )
                normalized = response.normalize(scalar)
                if isinstance(normalized, StructuredNA):
                    raise FidelityContractError(
                        "nondegenerate curve produced unexpected NA"
                    )
                computed.append(normalized)
            # Endpoints are mathematical identities and must not be silently repaired.
            if computed[0] != (1.0 if schedule.direction is CurveDirection.DELETION else 0.0):
                raise FidelityContractError("curve start does not equal its frozen endpoint")
            if computed[-1] != (0.0 if schedule.direction is CurveDirection.DELETION else 1.0):
                raise FidelityContractError("curve end does not equal its frozen endpoint")
            values = tuple(computed)
            auc = trapezoid_auc(schedule.actual_fractions, values)
    finally:
        # The group is strongly content-checked exactly at the curve exit,
        # including NA and consumer-exception paths and `.data` writes that do
        # not advance a version counter.
        schedule.verify_integrity()
    _line_after, _target_after, _identity_after, digest_after = _evaluator_contract(
        evaluator
    )
    if digest_after != digest:
        raise FidelityContractError(
            "frozen evaluator identity changed during curve evaluation"
        )
    return FidelityCurveResult(
        method_id=method_id, comparison_line=line, target_ref=target_ref,
        target_identity_sha256=digest, scale=schedule.scale,
        direction=schedule.direction, ranking_kind=schedule.ranking_kind,
        sample_id=schedule.sample_id, block=schedule.block, head=schedule.head,
        replicate=schedule.replicate, seed=schedule.seed,
        seed_namespace=schedule.seed_namespace,
        tie_priority_sha256=schedule.tie_priority_sha256,
        source_unit_map_sha256=schedule.source_unit_map_sha256,
        score_sha256=schedule.score_sha256,
        input_pair_sha256=schedule.input_pair_sha256,
        ranking_sha256=schedule.ranking_sha256,
        intervention_sha256=schedule.intervention_sha256,
        unit_counts=schedule.unit_counts,
        actual_fractions=schedule.actual_fractions, responses=values, auc=auc,
        na=response.na,
    )


@dataclass(frozen=True)
class CurveAggregate:
    curves: tuple[FidelityCurveResult, ...]
    response_mean: tuple[float, ...] | None
    response_std_population: tuple[float, ...] | None
    auc_mean: float | None
    auc_std_population: float | None
    na: StructuredNA | None = None

    def __post_init__(self) -> None:
        if len(self.curves) != TIE_REPLICATES:
            raise FidelityContractError("tie aggregate requires exactly five curves")
        first = self.curves[0]
        paired_fields = (
            "method_id", "comparison_line", "target_ref",
            "target_identity_sha256", "scale", "direction",
            "ranking_kind", "sample_id", "block", "head",
            "input_pair_sha256", "unit_counts",
            "actual_fractions", "source_unit_map_sha256", "score_sha256",
            "seed_namespace",
        )
        for curve in self.curves[1:]:
            if any(getattr(curve, field) != getattr(first, field) for field in paired_fields):
                raise FidelityContractError("five nuisance curves do not form a paired set")
        if tuple(curve.replicate for curve in self.curves) != tuple(range(5)):
            raise FidelityContractError("tie aggregate replicates must be ordered 0..4")
        if len({curve.seed for curve in self.curves}) != 5:
            raise FidelityContractError("tie aggregate requires five distinct seeds")
        for replicate, curve in enumerate(self.curves):
            if curve.seed != fidelity_seed_for_namespace(
                curve.seed_namespace, curve.sample_id, curve.block, curve.head,
                replicate
            ):
                raise FidelityContractError("aggregate curve seed violates namespace")
            if curve.tie_priority_sha256 != tie_priority_sha256(
                curve.scale.unit_count, curve.seed
            ):
                raise FidelityContractError(
                    "aggregate curve tie-priority hash violates its seed"
                )
        nas = tuple(curve.na for curve in self.curves)
        if any(item is not None for item in nas):
            if not all(item == nas[0] for item in nas):
                raise FidelityContractError("paired NA curves must retain one reason")
            if any(
                value is not None
                for value in (self.response_mean, self.response_std_population,
                              self.auc_mean, self.auc_std_population)
            ):
                raise FidelityContractError("NA aggregate cannot contain fake numeric values")
            if self.na != nas[0]:
                raise FidelityContractError("aggregate NA must match its curves")
            return
        matrix = np.asarray([curve.responses for curve in self.curves], dtype=np.float64)
        aucs = np.asarray([curve.auc for curve in self.curves], dtype=np.float64)
        expected_mean = tuple(float(value) for value in matrix.mean(axis=0))
        expected_std = tuple(float(value) for value in matrix.std(axis=0, ddof=0))
        if self.response_mean != expected_mean or self.response_std_population != expected_std:
            raise FidelityContractError("curve aggregate mean/std mismatch")
        if self.auc_mean != float(aucs.mean()) or self.auc_std_population != float(aucs.std(ddof=0)):
            raise FidelityContractError("AUC aggregate mean/std mismatch")
        if self.na is not None:
            raise FidelityContractError("successful curve aggregate cannot contain NA")

    @property
    def status(self) -> str:
        return "NA" if self.na is not None else "OK"

    def to_dict(self) -> dict[str, Any]:
        result = {
            "status": self.status,
            "replicate_role": "paired_nuisance_resolution_not_independent_samples",
            "replicate_count": 5,
            "standard_deviation": "population_ddof_0_over_complete_five_replicates",
            "curves": [curve.to_dict() for curve in self.curves],
            "response_mean": None if self.response_mean is None else list(self.response_mean),
            "response_std": None if self.response_std_population is None else list(self.response_std_population),
            "auc_mean": self.auc_mean, "auc_std": self.auc_std_population,
        }
        if self.na is not None:
            result["na"] = self.na.to_dict()
        return result


def aggregate_five_curves(curves: Any) -> CurveAggregate:
    if isinstance(curves, (str, bytes)):
        raise TypeError("curves must be an iterable of FidelityCurveResult")
    values = tuple(curves)
    if len(values) != 5 or any(not isinstance(item, FidelityCurveResult) for item in values):
        raise TypeError("curves must contain exactly five FidelityCurveResult values")
    if any(curve.na is not None for curve in values):
        na = values[0].na
        return CurveAggregate(values, None, None, None, None, na)
    matrix = np.asarray([curve.responses for curve in values], dtype=np.float64)
    aucs = np.asarray([curve.auc for curve in values], dtype=np.float64)
    return CurveAggregate(
        values, tuple(float(item) for item in matrix.mean(axis=0)),
        tuple(float(item) for item in matrix.std(axis=0, ddof=0)),
        float(aucs.mean()), float(aucs.std(ddof=0)), None,
    )


@dataclass(frozen=True)
class Top20Result:
    method_id: str
    comparison_line: ComparisonLine
    target_ref: str
    target_identity_sha256: str
    ranking_kind: RankingKind
    sample_id: str
    block: int
    head: int
    replicate: int | None
    seed: int | None
    tie_priority_sha256: str | None
    input_pair_sha256: str
    ranking_sha256: str
    scale: FidelityScale
    k: int
    actual_fraction: float
    sufficiency: float
    sufficiency_gap: float
    comprehensiveness: float

    def __post_init__(self) -> None:
        scale = _parse_scale(self.scale)
        line = ComparisonLine(self.comparison_line)
        kind = RankingKind(self.ranking_kind)
        if self.method_id not in VALUE_METHODS | CLASS_METHODS:
            raise FidelityContractError("top20 method is not fidelity eligible")
        if not re.fullmatch(r"[0-9a-f]{64}", self.target_identity_sha256):
            raise FidelityContractError("top20 target identity hash is invalid")
        if not re.fullmatch(r"[0-9a-f]{64}", self.input_pair_sha256) or not re.fullmatch(
            r"[0-9a-f]{64}", self.ranking_sha256
        ):
            raise FidelityContractError("top20 schedule hashes are invalid")
        expected_k = int(math.ceil(SUFFICIENCY_FRACTION * scale.unit_count))
        if self.k != expected_k or self.actual_fraction != expected_k / scale.unit_count:
            raise FidelityContractError("top-20% uses the wrong ceil count/fraction")
        sufficiency = _finite_float(self.sufficiency, "sufficiency")
        gap = _finite_float(self.sufficiency_gap, "sufficiency_gap")
        comprehensiveness = _finite_float(self.comprehensiveness, "comprehensiveness")
        if gap != 1.0 - sufficiency:
            raise FidelityContractError("sufficiency_gap must equal 1-sufficiency")
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "comparison_line", line)
        object.__setattr__(self, "ranking_kind", kind)
        object.__setattr__(self, "sufficiency", sufficiency)
        object.__setattr__(self, "sufficiency_gap", gap)
        object.__setattr__(self, "comprehensiveness", comprehensiveness)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": "OK", "method_id": self.method_id,
            "comparison_line": self.comparison_line.value,
            "target_ref": self.target_ref,
            "target_identity_sha256": self.target_identity_sha256,
            "ranking_kind": self.ranking_kind.value,
            "sample_id": self.sample_id, "block": self.block, "head": self.head,
            "replicate": self.replicate, "seed": self.seed,
            "tie_priority_sha256": self.tie_priority_sha256,
            "input_pair_sha256": self.input_pair_sha256,
            "ranking_sha256": self.ranking_sha256,
            "scale": self.scale.value,
            "fraction_requested": 0.2, "k": self.k,
            "actual_fraction": self.actual_fraction,
            "sufficiency": self.sufficiency,
            "sufficiency_gap": self.sufficiency_gap,
            "comprehensiveness": self.comprehensiveness,
            "decision_gate": False,
        }


def top20_from_curves(
    insertion: FidelityCurveResult, deletion: FidelityCurveResult
) -> Top20Result | StructuredNA:
    if not isinstance(insertion, FidelityCurveResult) or not isinstance(
        deletion, FidelityCurveResult
    ):
        raise TypeError("top20 requires FidelityCurveResult inputs")
    fields = (
        "method_id", "comparison_line", "target_ref",
        "target_identity_sha256", "scale", "ranking_kind",
        "sample_id", "block", "head", "replicate", "seed",
        "tie_priority_sha256", "input_pair_sha256", "ranking_sha256",
    )
    if any(getattr(insertion, name) != getattr(deletion, name) for name in fields):
        raise FidelityContractError("top20 curves are not a paired corruption")
    if insertion.direction is not CurveDirection.INSERTION or (
        deletion.direction is not CurveDirection.DELETION
    ):
        raise FidelityContractError("top20 requires insertion then deletion")
    if insertion.na is not None or deletion.na is not None:
        if insertion.na != deletion.na:
            raise FidelityContractError("paired top20 curves have different NA states")
        assert insertion.na is not None
        return insertion.na
    scale = insertion.scale
    k = int(math.ceil(0.2 * scale.unit_count))
    try:
        index = insertion.unit_counts.index(k)
    except ValueError as exc:
        raise FidelityContractError("top-20% point is missing from curve") from exc
    assert insertion.responses is not None and deletion.responses is not None
    sufficiency = insertion.responses[index]
    comprehensiveness = 1.0 - deletion.responses[index]
    return Top20Result(
        method_id=insertion.method_id, comparison_line=insertion.comparison_line,
        target_ref=insertion.target_ref,
        target_identity_sha256=insertion.target_identity_sha256,
        ranking_kind=insertion.ranking_kind, sample_id=insertion.sample_id,
        block=insertion.block, head=insertion.head, replicate=insertion.replicate,
        seed=insertion.seed, tie_priority_sha256=insertion.tie_priority_sha256,
        input_pair_sha256=insertion.input_pair_sha256,
        ranking_sha256=insertion.ranking_sha256,
        scale=scale, k=k, actual_fraction=k / scale.unit_count,
        sufficiency=sufficiency, sufficiency_gap=1.0 - sufficiency,
        comprehensiveness=comprehensiveness,
    )


@dataclass(frozen=True)
class ValuePathDeltaFidelity:
    target_identity_sha256: str
    nns_insertion_auc: float
    matched_insertion_auc: float
    nns_deletion_auc: float
    matched_deletion_auc: float
    value: float

    def __post_init__(self) -> None:
        if not isinstance(self.target_identity_sha256, str) or not re.fullmatch(
            r"[0-9a-f]{64}", self.target_identity_sha256
        ):
            raise FidelityContractError("delta_fid target identity hash is invalid")
        values = tuple(
            _finite_float(getattr(self, name), name)
            for name in (
                "nns_insertion_auc", "matched_insertion_auc",
                "nns_deletion_auc", "matched_deletion_auc",
            )
        )
        expected = 0.5 * ((values[0] - values[1]) + (values[3] - values[2]))
        if _finite_float(self.value, "delta_fid") != expected:
            raise FidelityContractError("delta_fid violates the frozen formula")

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": "OK", "comparison_line": "value_path",
            "target_ref": VALUE_TARGET_REF,
            "target_identity_sha256": self.target_identity_sha256,
            "scale": FidelityScale.FINE_4.value,
            "nns_insertion_auc": self.nns_insertion_auc,
            "matched_insertion_auc": self.matched_insertion_auc,
            "nns_deletion_auc": self.nns_deletion_auc,
            "matched_deletion_auc": self.matched_deletion_auc,
            "delta_fid": self.value,
            "formula": "0.5*((Ins_NNS-Ins_matched)+(Del_matched-Del_NNS))",
        }


def value_path_delta_fidelity(
    *,
    nns_insertion: CurveAggregate,
    nns_deletion: CurveAggregate,
    matched_insertion: CurveAggregate,
    matched_deletion: CurveAggregate,
) -> ValuePathDeltaFidelity | StructuredNA:
    aggregates = (nns_insertion, nns_deletion, matched_insertion, matched_deletion)
    if any(not isinstance(item, CurveAggregate) for item in aggregates):
        raise TypeError("delta_fid inputs must be CurveAggregate values")
    first_curves = tuple(item.curves[0] for item in aggregates)
    expected_methods = (NNS_METHOD_ID, NNS_METHOD_ID, MATCHED_METHOD_ID, MATCHED_METHOD_ID)
    expected_directions = (
        CurveDirection.INSERTION, CurveDirection.DELETION,
        CurveDirection.INSERTION, CurveDirection.DELETION,
    )
    identities = {curve.target_identity_sha256 for curve in first_curves}
    if len(identities) != 1:
        raise FidelityContractError("delta_fid cannot mix frozen target identities")
    for curve, method, direction in zip(first_curves, expected_methods, expected_directions):
        if (
            curve.method_id != method
            or curve.comparison_line is not ComparisonLine.VALUE_PATH
            or curve.target_ref != VALUE_TARGET_REF
            or curve.scale is not FidelityScale.FINE_4
            or curve.direction is not direction
            or curve.ranking_kind is not RankingKind.TARGET_ORIENTED_POSITIVE
        ):
            raise FidelityContractError(
                "delta_fid only accepts the registered same-target fine-4 Value-path pair"
            )
    # Five repeats are paired nuisance resolution.  Every corresponding
    # replicate must use the same registered seed and exact clean/baseline
    # tensor pair across both methods and both corruption directions.
    for replicate in range(TIE_REPLICATES):
        curves = tuple(item.curves[replicate] for item in aggregates)
        expected_replicate = replicate
        contexts = {
            (curve.sample_id, curve.block, curve.head) for curve in curves
        }
        seeds = {curve.seed for curve in curves}
        tie_priorities = {curve.tie_priority_sha256 for curve in curves}
        input_pairs = {curve.input_pair_sha256 for curve in curves}
        identities_here = {curve.target_identity_sha256 for curve in curves}
        if (
            any(curve.replicate != expected_replicate for curve in curves)
            or len(contexts) != 1
            or len(seeds) != 1
            or len(tie_priorities) != 1
            or len(input_pairs) != 1
            or len(identities_here) != 1
        ):
            raise FidelityContractError(
                "delta_fid inputs are not paired by replicate/seed/tie-priority/context/input/target"
            )
        sample_id, block, head = next(iter(contexts))
        namespaces = {curve.seed_namespace for curve in curves}
        if len(namespaces) != 1 or next(iter(seeds)) != fidelity_seed_for_namespace(
            next(iter(namespaces)), sample_id, block, head, replicate
        ):
            raise FidelityContractError("delta_fid paired seed violates namespace")
        if (
            nns_insertion.curves[replicate].ranking_sha256
            != nns_deletion.curves[replicate].ranking_sha256
            or matched_insertion.curves[replicate].ranking_sha256
            != matched_deletion.curves[replicate].ranking_sha256
        ):
            raise FidelityContractError(
                "delta_fid insertion/deletion rankings differ within a method"
            )
    nas = tuple(item.na for item in aggregates)
    if any(item is not None for item in nas):
        reasons = {None if item is None else item.reason for item in nas}
        if len(reasons) != 1:
            raise FidelityContractError("delta_fid inputs have inconsistent NA states")
        assert nas[0] is not None
        return nas[0]
    nns_ins = _finite_float(nns_insertion.auc_mean, "nns insertion AUC")
    nns_del = _finite_float(nns_deletion.auc_mean, "nns deletion AUC")
    matched_ins = _finite_float(matched_insertion.auc_mean, "matched insertion AUC")
    matched_del = _finite_float(matched_deletion.auc_mean, "matched deletion AUC")
    value = 0.5 * ((nns_ins - matched_ins) + (matched_del - nns_del))
    return ValuePathDeltaFidelity(
        target_identity_sha256=next(iter(identities)),
        nns_insertion_auc=nns_ins, matched_insertion_auc=matched_ins,
        nns_deletion_auc=nns_del, matched_deletion_auc=matched_del, value=value,
    )


def paired_rankings_for_method(
    scores: Any,
    *,
    sample_id: str,
    block: int,
    head: int,
) -> tuple[tuple[int, np.ndarray], ...]:
    """Return five seed/ranking pairs shared by method comparisons."""

    values = _validate_scores(scores)
    return tuple(
        (
            seed := fidelity_seed(sample_id, block, head, replicate),
            tie_permuted_ranking(values, seed),
        )
        for replicate in range(TIE_REPLICATES)
    )


def random_rankings(
    unit_count: int,
    *,
    sample_id: str,
    block: int,
    head: int,
) -> tuple[tuple[int, np.ndarray], ...]:
    """Return the five preregistered global-random sanity rankings."""

    return tuple(
        (
            seed := fidelity_seed(sample_id, block, head, replicate),
            global_random_ranking(unit_count, seed),
        )
        for replicate in range(TIE_REPLICATES)
    )


__all__ = [
    "CLASS_METHODS", "CLASS_TARGET_REF", "CURVE_FRACTIONS",
    "ClassLogitScalarAdapter", "CurveAggregate", "CurveDirection",
    "FIDELITY_BASE_SEED", "STAGE_C_FIDELITY_SEED_NAMESPACE",
    "FidelityContractError", "FidelityCurveResult", "FidelityScale",
    "FrozenScalarEvaluator", "InterventionSchedule", "MATCHED_METHOD_ID",
    "NNS_METHOD_ID", "RankingKind", "TIE_REPLICATES",
    "TARGET_DELTA_RELATIVE_THRESHOLD", "TargetResponse", "Top20Result",
    "UnitMassMap", "VALUE_METHODS", "VALUE_TARGET_REF",
    "ValuePathDeltaFidelity", "ValuePathScalarAdapter",
    "actual_curve_fractions",
    "aggregate_five_curves", "build_intervention_schedule",
    "build_method_intervention_schedule",
    "center_distance_ranking", "curve_unit_counts",
    "evaluate_fidelity_curve", "fidelity_seed",
    "fidelity_seed_for_namespace",
    "fidelity_target_provenance_binding",
    "global_random_ranking", "method_unit_masses",
    "paired_rankings_for_method", "random_rankings",
    "resolve_target_response", "target_oriented_scores",
    "target_response_from_values", "tensor_sha256",
    "tie_permuted_ranking", "tie_priority_sha256",
    "top20_from_curves", "trapezoid_auc",
    "value_path_delta_fidelity",
]
