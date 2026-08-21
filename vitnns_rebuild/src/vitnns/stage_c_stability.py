"""Frozen Stage C map stability and within-patch residual primitives."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .display import DisplayBoundaryError, validate_metric_input
from .retention import ArtifactArray, ArtifactKind, _deep_freeze, _immutable_array
from .stage_c_maps import (
    ComparisonLine, MethodResult, NAReason, SignSemantics, SignView,
    StructuredNA, patch_to_pixel_mass, raw_array, signed_rgb_channel_sum,
    stage_c_metric_ready,
)
from .stage_c_metrics import (
    CONSTANT_RELATIVE_L2_THRESHOLD, MetricValue, hotspot_jaccard_top_10_percent,
    hotspot_jaccard_top_20_percent, jensen_shannon_divergence, pearson, spearman,
)
from .stage_c_perturbations import PERTURBATION_IDS, perturbation_masks
from .stage_c_perturbations import PerturbationResult, inverse_align_pixel_map
from .stage_c_maps import pixel_to_patch_mass


PIXEL_SHAPE = (224, 224)
PATCH_SHAPE = (14, 14)
PATCH_SIZE = 16
_SHA256 = re.compile(r"[0-9a-f]{64}")
_TARGET_REF_BY_LINE = {
    ComparisonLine.VALUE_PATH: "fixed_clean_attention_cls_head_direction",
    ComparisonLine.CLASS_LOGIT: "clean_predicted_class_raw_logit",
}
_C2_CLASS_METHODS = {
    "direct_raw_cls_attention_last_block_head_mean",
    "attention_rollout_mean_head_residual_all_blocks",
}


class StageCStabilityError(ValueError):
    """Raised for malformed or cross-context stability inputs."""


def _json_safe(value: Any, context: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            raise StageCStabilityError(f"{context} contains a non-finite number")
        return
    if isinstance(value, np.integer):
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{context} keys must be strings")
            _json_safe(item, f"{context}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _json_safe(item, f"{context}[{index}]")
        return
    raise TypeError(f"{context} contains non-JSON-safe {type(value).__name__}")


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value.item() if isinstance(value, np.generic) else value


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _canonical_target_identity(
    value: Any, *, comparison_line: ComparisonLine | str
) -> Mapping[str, Any]:
    """Validate the exact identity envelope emitted by C5 adapters."""

    line = ComparisonLine(comparison_line)
    if not isinstance(value, Mapping):
        raise TypeError("target_identity must be a mapping")
    required = {
        "comparison_line", "target_ref", "clean_input_sha256",
        "frozen_target",
    }
    if set(value) != required:
        raise StageCStabilityError(
            "target_identity must have the exact canonical C5 fields"
        )
    _json_safe(value, "target_identity")
    expected_ref = _TARGET_REF_BY_LINE[line]
    if value["comparison_line"] != line.value or value["target_ref"] != expected_ref:
        raise StageCStabilityError(
            "target_identity comparison_line/target_ref mismatch"
        )
    clean_hash = _sha256(value["clean_input_sha256"], "clean_input_sha256")
    frozen = value["frozen_target"]
    if not isinstance(frozen, Mapping) or not frozen:
        raise StageCStabilityError("frozen_target must be a non-empty mapping")
    expected_kind = (
        "raw_logit"
        if line is ComparisonLine.CLASS_LOGIT
        else "fixed_clean_attention_cls_head_direction"
    )
    if frozen.get("target_kind") != expected_kind:
        raise StageCStabilityError(
            "frozen_target.target_kind does not match comparison_line"
        )
    if frozen.get("clean_input_sha256") != clean_hash:
        raise StageCStabilityError(
            "frozen_target clean_input_sha256 does not match identity envelope"
        )
    runtime = frozen.get("runtime_model_binding")
    runtime_sha = (
        runtime.get("sha256") if isinstance(runtime, Mapping)
        else frozen.get("runtime_model_binding_sha256")
    )
    _sha256(runtime_sha, "runtime_model_binding.sha256")
    if line is ComparisonLine.CLASS_LOGIT:
        class_index = frozen.get("class_index")
        class_count = frozen.get("class_count")
        if type(class_index) is not int or class_index < 0:
            raise StageCStabilityError("class target requires non-negative class_index")
        if class_count is not None and (
            type(class_count) is not int or class_count <= class_index
        ):
            raise StageCStabilityError("class target class_count is invalid")
    else:
        candidate = frozen.get("candidate")
        if not isinstance(candidate, Mapping) or set(candidate) != {
            "block_index", "head_index"
        }:
            raise StageCStabilityError("Value target requires exact candidate identity")
        for name in ("block_index", "head_index"):
            index = candidate[name]
            if type(index) is not int or not 0 <= index < 12:
                raise StageCStabilityError(f"Value target {name} is invalid")
        status = frozen.get("status")
        if status is not None and status not in {"OK", "DEGENERATE_DIRECTION"}:
            raise StageCStabilityError("Value target status is invalid")
    return _deep_freeze(_thaw(value))


def _method_target_identity(result: MethodResult) -> Mapping[str, Any]:
    if not isinstance(result, MethodResult):
        raise TypeError("result must be MethodResult")
    provenance = result.provenance
    target = provenance.get("target")
    if isinstance(target, Mapping):
        candidate = {
            "comparison_line": result.comparison_line.value,
            "target_ref": result.fidelity_target,
            "clean_input_sha256": target.get("clean_input_sha256"),
            "frozen_target": _thaw(target),
        }
    elif result.method_id in _C2_CLASS_METHODS:
        caller = provenance.get("caller_context")
        candidate = (
            caller.get("fidelity_target_identity")
            if isinstance(caller, Mapping) else None
        )
        if not isinstance(candidate, Mapping):
            raise StageCStabilityError(
                "C2 class map requires caller_context.fidelity_target_identity"
            )
    else:
        raise StageCStabilityError(
            "MethodResult provenance does not bind a frozen target identity"
        )
    identity = _canonical_target_identity(
        candidate, comparison_line=result.comparison_line
    )
    if identity["target_ref"] != result.fidelity_target:
        raise StageCStabilityError(
            "MethodResult fidelity target does not match target identity"
        )
    return identity


def _method_intervention_sha256(result: MethodResult) -> str | None:
    value = result.provenance.get("intervention_input_sha256")
    if value is None:
        caller = result.provenance.get("caller_context")
        if isinstance(caller, Mapping):
            value = caller.get("intervention_input_sha256")
    if value is not None:
        _sha256(value, "MethodResult intervention_input_sha256")
    return value


def _raw_pixels(value: Any, name: str) -> np.ndarray:
    kind = getattr(value, "kind", None)
    if kind is None:
        raise DisplayBoundaryError(f"{name} requires explicitly tagged raw data")
    parsed = ArtifactKind.parse(kind)
    if parsed is not ArtifactKind.RAW:
        raise DisplayBoundaryError(f"{name} requires raw input, got {parsed.value}")
    array = np.asarray(value.data)
    if array.shape != PIXEL_SHAPE:
        raise StageCStabilityError(f"{name} must have shape {PIXEL_SHAPE}")
    if array.dtype != np.float32:
        raise TypeError(f"{name} must have dtype float32")
    if not np.isfinite(array).all():
        raise StageCStabilityError(f"{name} must be finite")
    return array


def _registered_masks(
    perturbation_id: str, pixel: Any, patch: Any
) -> tuple[np.ndarray, np.ndarray]:
    if perturbation_id not in PERTURBATION_IDS:
        raise StageCStabilityError("perturbation_id is not registered")
    expected_pixel, expected_patch = perturbation_masks(perturbation_id)
    if not isinstance(pixel, np.ndarray) or pixel.dtype != np.bool_:
        raise TypeError("valid_pixel_mask must be a numpy bool array")
    if not isinstance(patch, np.ndarray) or patch.dtype != np.bool_:
        raise TypeError("fully_valid_patch_mask must be a numpy bool array")
    if not np.array_equal(pixel, expected_pixel):
        raise StageCStabilityError("valid_pixel_mask is not the registered perturbation mask")
    if not np.array_equal(patch, expected_patch):
        raise StageCStabilityError("fully_valid_patch_mask is not the registered perturbation mask")
    return expected_pixel, expected_patch


@dataclass(frozen=True)
class AlignedMapResult:
    """One inverse-aligned metric map bound to method, target and perturbation."""

    method_id: str
    method_version: str
    comparison_line: ComparisonLine
    target_identity: Mapping[str, Any]
    sign_semantics: SignSemantics
    perturbation_id: str
    perturbation_tensor_sha256: str
    metric_map: ArtifactArray
    valid_pixel_mask: np.ndarray
    fully_valid_patch_mask: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.method_id, str) or not self.method_id:
            raise ValueError("method_id must be non-empty")
        if not isinstance(self.method_version, str) or not self.method_version:
            raise ValueError("method_version must be non-empty")
        object.__setattr__(self, "comparison_line", ComparisonLine(self.comparison_line))
        object.__setattr__(self, "sign_semantics", SignSemantics(self.sign_semantics))
        identity = _canonical_target_identity(
            self.target_identity, comparison_line=self.comparison_line
        )
        object.__setattr__(self, "target_identity", identity)
        _sha256(self.perturbation_tensor_sha256, "perturbation_tensor_sha256")
        metric = validate_metric_input(self.metric_map, name="aligned metric_map")
        metric = np.asarray(metric)
        if metric.shape != PATCH_SHAPE or metric.dtype != np.float32 or not np.isfinite(metric).all():
            raise StageCStabilityError("aligned metric_map must be finite float32 14x14")
        pixels, patches = _registered_masks(
            self.perturbation_id, self.valid_pixel_mask, self.fully_valid_patch_mask
        )
        object.__setattr__(self, "valid_pixel_mask", pixels)
        object.__setattr__(self, "fully_valid_patch_mask", patches)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method_id": self.method_id, "method_version": self.method_version,
            "comparison_line": self.comparison_line.value,
            "target_identity": _thaw(self.target_identity),
            "sign_semantics": self.sign_semantics.value,
            "perturbation_id": self.perturbation_id,
            "perturbation_tensor_sha256": self.perturbation_tensor_sha256,
            "valid_pixel_count": int(self.valid_pixel_mask.sum()),
            "fully_valid_patch_count": int(self.fully_valid_patch_mask.sum()),
        }


def _issue_aligned_map_result(
    method_result: MethodResult, *, target_identity: Mapping[str, Any],
    perturbation_id: str, perturbation_tensor_sha256: str,
    metric_map: ArtifactArray, valid_pixel_mask: np.ndarray,
    fully_valid_patch_mask: np.ndarray,
) -> AlignedMapResult:
    if not isinstance(method_result, MethodResult):
        raise TypeError("method_result must be MethodResult")
    if method_result.na is not None:
        raise StageCStabilityError("cannot align a failed MethodResult")
    identity = _canonical_target_identity(
        target_identity, comparison_line=method_result.comparison_line
    )
    if _method_target_identity(method_result) != identity:
        raise StageCStabilityError(
            "aligned MethodResult provenance target identity mismatch"
        )
    return AlignedMapResult(
        method_result.method_id, method_result.method_version,
        method_result.comparison_line, identity, method_result.sign_semantics,
        perturbation_id, perturbation_tensor_sha256, metric_map,
        valid_pixel_mask, fully_valid_patch_mask,
    )


def inverse_aligned_map_result(
    method_result: MethodResult,
    *,
    target_identity: Mapping[str, Any],
    perturbation: PerturbationResult,
    perturbed_method_result: MethodResult,
) -> AlignedMapResult:
    """Inverse-align a raw pixel mass map and bind its perturbation bundle.

    Pixel-native methods use this helper so inverse geometry, the mass
    Jacobian, the registered masks, and the exact pre-generated perturbation
    tensor hash cannot be accidentally paired by hand.
    """

    if not isinstance(perturbation, PerturbationResult):
        raise TypeError("perturbation must be a PerturbationResult")
    if not isinstance(perturbed_method_result, MethodResult):
        raise TypeError("perturbed_method_result must be MethodResult")
    if perturbed_method_result.na is not None or perturbed_method_result.native_tensor is None:
        raise StageCStabilityError("perturbed MethodResult must contain a native map")
    intervention_hash = _method_intervention_sha256(perturbed_method_result)
    if intervention_hash != perturbation.normalized_tensor_sha256:
        raise StageCStabilityError(
            "perturbed MethodResult intervention hash does not match perturbation"
        )
    if (
        perturbed_method_result.method_id != method_result.method_id
        or perturbed_method_result.method_version != method_result.method_version
        or perturbed_method_result.comparison_line is not method_result.comparison_line
        or perturbed_method_result.sign_semantics is not method_result.sign_semantics
    ):
        raise StageCStabilityError(
            "clean and perturbed MethodResult method identity mismatch"
        )
    identity = _canonical_target_identity(
        target_identity, comparison_line=method_result.comparison_line
    )
    if (
        _method_target_identity(method_result) != identity
        or _method_target_identity(perturbed_method_result) != identity
    ):
        raise StageCStabilityError(
            "clean/perturbed MethodResult frozen target identity mismatch"
        )
    perturbation.verify_integrity()
    native = perturbed_method_result.native_tensor
    native_shape = tuple(native.data.shape)
    if native_shape == (3, 224, 224):
        perturbed_pixel_map = signed_rgb_channel_sum(native)
    elif native_shape == (14, 14):
        perturbed_pixel_map = patch_to_pixel_mass(
            perturbed_method_result.metric_map
        )
    else:
        raise StageCStabilityError(
            "perturbed native map must be RGB pixel-native or 14x14 patch-native"
        )
    aligned_pixels = inverse_align_pixel_map(
        perturbed_pixel_map, perturbation.perturbation_id
    )
    metric_map = pixel_to_patch_mass(aligned_pixels)
    result = _issue_aligned_map_result(
        perturbed_method_result, target_identity=target_identity,
        perturbation_id=perturbation.perturbation_id,
        perturbation_tensor_sha256=perturbation.normalized_tensor_sha256,
        metric_map=metric_map,
        valid_pixel_mask=perturbation.valid_pixel_mask,
        fully_valid_patch_mask=perturbation.fully_valid_patch_mask,
    )
    perturbation.verify_integrity()
    return result


def aligned_map_result(
    clean_method_result: MethodResult,
    *,
    target_identity: Mapping[str, Any],
    perturbation: PerturbationResult,
    perturbed_method_result: MethodResult,
) -> AlignedMapResult:
    """Public issued path: native perturbed map -> inverse -> patch metric."""

    return inverse_aligned_map_result(
        clean_method_result, target_identity=target_identity,
        perturbation=perturbation,
        perturbed_method_result=perturbed_method_result,
    )


@dataclass(frozen=True)
class PatchResidualResult:
    residual: ArtifactArray
    rho: MetricValue | StructuredNA
    status: str
    failure_reason: str | None
    method_id: str
    method_version: str
    comparison_line: ComparisonLine
    target_identity: Mapping[str, Any]
    perturbation_id: str
    perturbation_tensor_sha256: str
    valid_pixel_mask: np.ndarray
    fully_valid_patch_mask: np.ndarray
    details: Mapping[str, Any]

    def __post_init__(self) -> None:
        residual = _raw_pixels(self.residual, "residual")
        if not isinstance(self.rho, (MetricValue, StructuredNA)):
            raise TypeError("rho must be MetricValue or StructuredNA")
        if self.status not in {"OK", "SPATIAL_FAILURE"}:
            raise ValueError("residual status must be OK or SPATIAL_FAILURE")
        if (self.status == "OK") is (self.failure_reason is not None):
            raise ValueError("residual failure_reason/status are inconsistent")
        if not isinstance(self.method_id, str) or not self.method_id:
            raise ValueError("residual method_id must be non-empty")
        if not isinstance(self.method_version, str) or not self.method_version:
            raise ValueError("residual method_version must be non-empty")
        line = ComparisonLine(self.comparison_line)
        identity = _canonical_target_identity(
            self.target_identity, comparison_line=line
        )
        _sha256(self.perturbation_tensor_sha256, "perturbation_tensor_sha256")
        pixels, patches = _registered_masks(
            self.perturbation_id, self.valid_pixel_mask, self.fully_valid_patch_mask
        )
        if np.any(residual[~pixels] != 0):
            raise StageCStabilityError("residual must be zero outside valid pixels")
        _json_safe(self.details, "details")
        object.__setattr__(self, "valid_pixel_mask", pixels)
        object.__setattr__(self, "fully_valid_patch_mask", patches)
        object.__setattr__(self, "comparison_line", line)
        object.__setattr__(self, "target_identity", identity)
        object.__setattr__(self, "details", _deep_freeze(dict(self.details)))

    @property
    def spatial_failure(self) -> bool:
        return self.status == "SPATIAL_FAILURE"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status, "failure_reason": self.failure_reason,
            "method_id": self.method_id, "method_version": self.method_version,
            "comparison_line": self.comparison_line.value,
            "target_identity": _thaw(self.target_identity),
            "rho": self.rho.to_dict(), "perturbation_id": self.perturbation_id,
            "perturbation_tensor_sha256": self.perturbation_tensor_sha256,
            "valid_pixel_count": int(self.valid_pixel_mask.sum()),
            "fully_valid_patch_count": int(self.fully_valid_patch_mask.sum()),
            "details": _thaw(self.details),
        }


def patch_inner_residual(
    pixel_map: Any, *, method_id: str, method_version: str,
    comparison_line: ComparisonLine | str, target_identity: Mapping[str, Any],
    perturbation: PerturbationResult,
) -> PatchResidualResult:
    source = _raw_pixels(pixel_map, "pixel_map")
    if not isinstance(perturbation, PerturbationResult):
        raise TypeError("perturbation must be a PerturbationResult")
    perturbation.verify_integrity()
    mask, patch_mask = _registered_masks(
        perturbation.perturbation_id, perturbation.valid_pixel_mask,
        perturbation.fully_valid_patch_mask
    )
    uniform = np.zeros(PIXEL_SHAPE, dtype=np.float32)
    for row in range(14):
        rs = slice(row * PATCH_SIZE, (row + 1) * PATCH_SIZE)
        for col in range(14):
            cs = slice(col * PATCH_SIZE, (col + 1) * PATCH_SIZE)
            local_mask = mask[rs, cs]
            count = int(local_mask.sum())
            if count:
                local = source[rs, cs]
                mass = float(np.sum(local[local_mask], dtype=np.float64))
                uniform[rs, cs][local_mask] = np.float32(mass / count)
    residual = np.zeros(PIXEL_SHAPE, dtype=np.float32)
    residual[mask] = source[mask] - uniform[mask]
    denominator = float(np.sum(np.abs(source[mask]), dtype=np.float64))
    numerator = float(np.sum(np.abs(residual[mask]), dtype=np.float64))
    if denominator == 0.0:
        rho: MetricValue | StructuredNA = StructuredNA(
            NAReason.ZERO_MASS_LEFT, "within-patch residual ratio has zero map L1"
        )
        status, reason = "SPATIAL_FAILURE", "zero_map_l1"
    else:
        rho = MetricValue(numerator / denominator)
        values = residual[mask].astype(np.float64)
        centered = values - values.mean(dtype=np.float64)
        constant = float(np.linalg.norm(centered)) <= CONSTANT_RELATIVE_L2_THRESHOLD * max(
            1.0, float(np.linalg.norm(values))
        )
        status, reason = ("SPATIAL_FAILURE", "constant_residual") if constant else ("OK", None)
    return PatchResidualResult(
        raw_array(residual, metadata={"operator": "M_minus_U_of_P"}), rho,
        status, reason, method_id, method_version,
        ComparisonLine(comparison_line), target_identity,
        perturbation.perturbation_id, perturbation.normalized_tensor_sha256, mask, patch_mask,
        {"formula": "R(M)=M-U(P(M));rho=l1(R)/l1(M)",
         "partial_patch_mass_divisor": "valid_pixel_count",
         "numerator_l1": numerator, "denominator_l1": denominator},
    )


def residual_signed_spearman(
    clean: PatchResidualResult, perturbed: PatchResidualResult
) -> MetricValue | StructuredNA:
    if not isinstance(clean, PatchResidualResult) or not isinstance(perturbed, PatchResidualResult):
        raise TypeError("residual Spearman inputs must be PatchResidualResult")
    if clean.perturbation_id != perturbed.perturbation_id or clean.perturbation_tensor_sha256 != perturbed.perturbation_tensor_sha256:
        raise StageCStabilityError("residual pair perturbation identity mismatch")
    if (
        clean.method_id != perturbed.method_id
        or clean.method_version != perturbed.method_version
        or clean.comparison_line is not perturbed.comparison_line
        or clean.target_identity != perturbed.target_identity
    ):
        raise StageCStabilityError("residual pair method/target identity mismatch")
    if not np.array_equal(clean.valid_pixel_mask, perturbed.valid_pixel_mask) or not np.array_equal(clean.fully_valid_patch_mask, perturbed.fully_valid_patch_mask):
        raise StageCStabilityError("residual pair must use identical registered masks")
    if clean.spatial_failure:
        return StructuredNA(NAReason.CONSTANT_LEFT, clean.failure_reason)
    if perturbed.spatial_failure:
        return StructuredNA(NAReason.CONSTANT_RIGHT, perturbed.failure_reason)
    # Residual patch sums are mathematically zero, so C1's 14x14 metric cannot
    # represent the native residual test.  Compute the registered average-rank
    # Spearman on the identical registered pixel mask while retaining C1's NA rule.
    mask = clean.valid_pixel_mask
    left = clean.residual.data[mask].astype(np.float64)
    right = perturbed.residual.data[mask].astype(np.float64)
    def ranks(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="mergesort"); result = np.empty(values.size, np.float64); start = 0
        while start < values.size:
            end = start + 1
            while end < values.size and values[order[end]] == values[order[start]]: end += 1
            result[order[start:end]] = (start + 1 + end) / 2.0; start = end
        return result
    left_rank, right_rank = ranks(left), ranks(right)
    left_rank -= left_rank.mean(); right_rank -= right_rank.mean()
    return MetricValue(float(np.dot(left_rank, right_rank) / (np.linalg.norm(left_rank) * np.linalg.norm(right_rank))),
                       {"view": "signed", "tie_method": "average_rank", "common_valid_pixel_count": int(mask.sum())})


@dataclass(frozen=True)
class StabilityResult:
    method_id: str
    method_version: str
    comparison_line: ComparisonLine
    target_identity: Mapping[str, Any]
    perturbation_id: str
    perturbation_tensor_sha256: str
    clean_target_fixed: bool
    clean_prediction: PredictionObservation
    perturbed_natural_prediction: PredictionObservation
    prediction_unchanged: bool
    prediction_flipped: bool
    metrics: Mapping[str, Mapping[str, MetricValue | StructuredNA]]

    def __post_init__(self) -> None:
        object.__setattr__(self, "comparison_line", ComparisonLine(self.comparison_line))
        if not self.clean_target_fixed:
            raise StageCStabilityError("clean target must remain frozen")
        if not isinstance(self.clean_prediction, PredictionObservation) or not isinstance(
            self.perturbed_natural_prediction, PredictionObservation
        ):
            raise TypeError("predictions must be PredictionObservation values")
        expected = (
            self.clean_prediction.class_index
            == self.perturbed_natural_prediction.class_index
        )
        if type(self.prediction_unchanged) is not bool or self.prediction_unchanged is not expected:
            raise StageCStabilityError("prediction_unchanged is inconsistent")
        if type(self.prediction_flipped) is not bool or self.prediction_flipped is not (not expected):
            raise StageCStabilityError("prediction_flipped is inconsistent")
        _sha256(self.perturbation_tensor_sha256, "perturbation_tensor_sha256")
        identity = _canonical_target_identity(
            self.target_identity, comparison_line=self.comparison_line
        )
        if self.comparison_line is ComparisonLine.CLASS_LOGIT:
            frozen_class = identity["frozen_target"].get("class_index")
            if type(frozen_class) is not int or frozen_class < 0:
                raise StageCStabilityError(
                    "class target identity requires a non-negative class_index"
                )
            if self.clean_prediction.class_index != frozen_class:
                raise StageCStabilityError(
                    "clean_prediction must equal the frozen clean class index"
                )
            class_count = identity["frozen_target"].get("class_count")
            if class_count is not None and (
                self.clean_prediction.class_count != class_count
                or self.perturbed_natural_prediction.class_count != class_count
            ):
                raise StageCStabilityError("prediction class_count mismatch")
        runtime = identity["frozen_target"].get("runtime_model_binding")
        runtime_sha = (
            runtime.get("sha256") if isinstance(runtime, Mapping)
            else identity["frozen_target"].get("runtime_model_binding_sha256")
        )
        if (
            self.clean_prediction.runtime_model_binding_sha256 != runtime_sha
            or self.perturbed_natural_prediction.runtime_model_binding_sha256
            != runtime_sha
        ):
            raise StageCStabilityError("prediction model binding mismatch")
        if (
            self.clean_prediction.intervention_input_sha256
            != identity["clean_input_sha256"]
            or self.perturbed_natural_prediction.intervention_input_sha256
            != self.perturbation_tensor_sha256
        ):
            raise StageCStabilityError("prediction intervention hash mismatch")
        object.__setattr__(self, "target_identity", identity)
        if not isinstance(self.method_id, str) or not self.method_id or not isinstance(self.method_version, str) or not self.method_version:
            raise ValueError("method identity must be non-empty")
        if self.perturbation_id not in PERTURBATION_IDS:
            raise StageCStabilityError("perturbation_id is not registered")
        required = {"pearson", "spearman", "jensen_shannon", "top10_jaccard", "top20_jaccard"}
        correlation_views = {"signed", "positive", "negative"}
        distribution_views = {"positive", "negative"}
        if set(self.metrics) != required:
            raise StageCStabilityError("metrics must contain the five frozen stability metrics")
        for metric, views in self.metrics.items():
            expected = correlation_views if metric in {"pearson", "spearman"} else distribution_views
            if not isinstance(views, Mapping) or set(views) != expected or not all(
                isinstance(value, (MetricValue, StructuredNA)) for value in views.values()
            ):
                raise StageCStabilityError(
                    f"{metric} does not contain the frozen sign views"
                )
        object.__setattr__(
            self, "metrics",
            _deep_freeze({metric: dict(views) for metric, views in self.metrics.items()}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"method_id": self.method_id, "method_version": self.method_version,
                "comparison_line": self.comparison_line.value, "target_identity": _thaw(self.target_identity),
                "perturbation_id": self.perturbation_id, "perturbation_tensor_sha256": self.perturbation_tensor_sha256,
                "fixed_target_primary": {"clean_target_fixed": True, "metrics": {metric: {view: value.to_dict() for view, value in views.items()} for metric, views in self.metrics.items()}},
                "natural_prediction_descriptive": {"clean_prediction": self.clean_prediction.to_dict(), "perturbed_prediction": self.perturbed_natural_prediction.to_dict(),
                                                     "prediction_unchanged": self.prediction_unchanged, "prediction_flipped": self.prediction_flipped}}


@dataclass(frozen=True)
class PredictionObservation:
    """Natural perturbed prediction bound to model and intervention bytes."""

    class_index: int
    class_count: int
    intervention_input_sha256: str
    runtime_model_binding_sha256: str
    source: str = "natural_raw_logits_argmax"

    def __post_init__(self) -> None:
        if type(self.class_index) is not int or self.class_index < 0:
            raise TypeError("prediction class_index must be a non-negative integer")
        if type(self.class_count) is not int or self.class_count <= self.class_index:
            raise ValueError("prediction class_count must exceed class_index")
        _sha256(self.intervention_input_sha256, "intervention_input_sha256")
        _sha256(self.runtime_model_binding_sha256, "runtime_model_binding_sha256")
        if self.source != "natural_raw_logits_argmax":
            raise ValueError("prediction source must be natural_raw_logits_argmax")

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_index": self.class_index, "class_count": self.class_count,
            "intervention_input_sha256": self.intervention_input_sha256,
            "runtime_model_binding_sha256": self.runtime_model_binding_sha256,
            "source": self.source,
        }


def prediction_observation_from_dict(value: Mapping[str, Any]) -> PredictionObservation:
    if not isinstance(value, Mapping) or set(value) != {
        "class_index", "class_count", "intervention_input_sha256",
        "runtime_model_binding_sha256", "source",
    }:
        raise StageCStabilityError("prediction observation has wrong fields")
    return PredictionObservation(**dict(value))


def evaluate_map_stability(
    clean_method: MethodResult, aligned: AlignedMapResult, *,
    target_identity: Mapping[str, Any], clean_prediction: PredictionObservation,
    perturbed_natural_prediction: PredictionObservation,
) -> StabilityResult:
    if not isinstance(clean_method, MethodResult) or clean_method.na is not None or clean_method.metric_map is None:
        raise StageCStabilityError("clean_method must be a successful MethodResult")
    if not isinstance(aligned, AlignedMapResult):
        raise TypeError("aligned must be AlignedMapResult")
    identity = _canonical_target_identity(
        target_identity, comparison_line=clean_method.comparison_line
    )
    if not bool(np.all(clean_method.valid_mask)):
        raise StageCStabilityError(
            "clean MethodResult valid_mask must be all true"
        )
    if _method_target_identity(clean_method) != identity:
        raise StageCStabilityError(
            "clean MethodResult provenance target identity mismatch"
        )
    clean_intervention = _method_intervention_sha256(clean_method)
    if clean_intervention is not None and clean_intervention != identity[
        "clean_input_sha256"
    ]:
        raise StageCStabilityError(
            "clean MethodResult was not evaluated at the frozen clean input"
        )
    if (clean_method.method_id, clean_method.method_version, clean_method.comparison_line, clean_method.sign_semantics) != (aligned.method_id, aligned.method_version, aligned.comparison_line, aligned.sign_semantics):
        raise StageCStabilityError("clean/aligned method identity mismatch")
    if identity != aligned.target_identity:
        raise StageCStabilityError("clean/aligned target identity mismatch")
    kwargs = dict(mask=aligned.fully_valid_patch_mask, left_semantics=clean_method.sign_semantics, right_semantics=aligned.sign_semantics)
    def pair(function: Any, views: tuple[SignView, ...]) -> dict[str, MetricValue | StructuredNA]:
        return {
            view.value.replace("negative_magnitude", "negative"): function(
                clean_method.metric_map, aligned.metric_map, view=view, **kwargs
            )
            for view in views
        }

    correlation_views = (
        SignView.SIGNED, SignView.POSITIVE, SignView.NEGATIVE_MAGNITUDE
    )
    distribution_views = (SignView.POSITIVE, SignView.NEGATIVE_MAGNITUDE)
    metrics = {
        "pearson": pair(pearson, correlation_views),
        "spearman": pair(spearman, correlation_views),
        "jensen_shannon": pair(jensen_shannon_divergence, distribution_views),
        "top10_jaccard": pair(hotspot_jaccard_top_10_percent, distribution_views),
        "top20_jaccard": pair(hotspot_jaccard_top_20_percent, distribution_views),
    }
    unchanged = clean_prediction.class_index == perturbed_natural_prediction.class_index
    return StabilityResult(clean_method.method_id, clean_method.method_version, clean_method.comparison_line, identity,
                           aligned.perturbation_id, aligned.perturbation_tensor_sha256, True, clean_prediction,
                           perturbed_natural_prediction, unchanged, not unchanged, metrics)


__all__ = ["AlignedMapResult", "PatchResidualResult", "PredictionObservation",
           "StabilityResult", "StageCStabilityError",
           "aligned_map_result", "inverse_aligned_map_result",
           "prediction_observation_from_dict",
           "evaluate_map_stability", "patch_inner_residual",
           "residual_signed_spearman"]
