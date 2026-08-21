"""Stage C map, projection, and method-result contracts.

This module contains no attribution method.  It only defines the numerical
language that every later Stage C method must use.  In particular, display
arrays are never accepted as quantitative inputs, signed mass is projected
before sign views are split, and every retained metric map is float32.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from .display import DisplayBoundaryError, validate_metric_input
from .retention import ArtifactArray, ArtifactKind, _deep_freeze, _immutable_array


INPUT_SHAPE = (224, 224)
PATCH_SHAPE = (14, 14)
FINE_CELL_SHAPE = (56, 56)
PATCH_SIZE = 16
FINE_CELL_SIZE = 4
PROJECTION_ATOL = 1e-6
PROJECTION_RTOL = 1e-6


class ComparisonLine(str, Enum):
    VALUE_PATH = "value_path"
    CLASS_LOGIT = "class_logit"


class SignSemantics(str, Enum):
    SIGNED = "signed"
    NONNEGATIVE = "nonnegative"


class SignView(str, Enum):
    SIGNED = "signed"
    POSITIVE = "positive"
    NEGATIVE_MAGNITUDE = "negative_magnitude"


class NAReason(str, Enum):
    CONSTANT_LEFT = "constant_left"
    CONSTANT_RIGHT = "constant_right"
    ZERO_MASS_LEFT = "zero_mass_left"
    ZERO_MASS_RIGHT = "zero_mass_right"
    UNSUPPORTED_SIGN_SEMANTICS = "unsupported_sign_semantics"
    EMPTY_VALID_REGION = "empty_valid_region"
    SHAPE_MISMATCH = "shape_mismatch"
    NON_FINITE = "non_finite"
    DEGENERATE_DIRECTION = "degenerate_direction"
    DEGENERATE_TARGET_DELTA = "degenerate_target_delta"
    MISSING_ARTIFACT = "missing_artifact"


INVALIDATING_NA_REASONS = frozenset(
    {
        NAReason.EMPTY_VALID_REGION,
        NAReason.SHAPE_MISMATCH,
        NAReason.NON_FINITE,
        NAReason.DEGENERATE_DIRECTION,
        NAReason.MISSING_ARTIFACT,
    }
)


@dataclass(frozen=True)
class StructuredNA:
    """A serializable, never-imputed missing metric/result state."""

    reason: NAReason
    detail: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "reason", NAReason(self.reason))
        if self.detail is not None and (
            not isinstance(self.detail, str) or not self.detail.strip()
        ):
            raise ValueError("StructuredNA.detail must be a non-empty string or None")

    @property
    def status(self) -> str:
        return "NA"

    @property
    def value(self) -> None:
        return None

    @property
    def invalidating(self) -> bool:
        return self.reason in INVALIDATING_NA_REASONS

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "status": "NA",
            "value": None,
            "reason": self.reason.value,
        }
        if self.detail is not None:
            result["detail"] = self.detail
        return result


def _parse_enum(value: Any, enum: type[Enum], name: str) -> Any:
    try:
        return enum(value)
    except (TypeError, ValueError) as exc:
        choices = sorted(item.value for item in enum)
        raise ValueError(f"{name} must be one of {choices}, got {value!r}") from exc


def _reject_display(value: Any, name: str) -> Any:
    kind = getattr(value, "kind", None)
    if kind is not None and ArtifactKind.parse(kind) is ArtifactKind.DISPLAY:
        raise DisplayBoundaryError(f"{name} rejects display data")
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu().numpy()
    elif isinstance(value, ArtifactArray):
        value = value.data
    return value


def _float32_array(
    value: Any, *, name: str, shape: tuple[int, ...] | None = None
) -> np.ndarray:
    value = _reject_display(value, name)
    array = np.asarray(value)
    if array.dtype != np.dtype(np.float32):
        raise TypeError(f"{name} must have dtype float32, got {array.dtype}")
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _raw_float32_array(
    value: Any, *, name: str, shape: tuple[int, ...] | None = None
) -> np.ndarray:
    kind = getattr(value, "kind", None)
    if kind is None:
        raise DisplayBoundaryError(
            f"{name} requires an explicitly tagged raw value; bare arrays are forbidden"
        )
    parsed = ArtifactKind.parse(kind)
    if parsed is ArtifactKind.DISPLAY:
        raise DisplayBoundaryError(f"{name} rejects display data")
    if parsed is not ArtifactKind.RAW:
        raise DisplayBoundaryError(f"{name} requires raw input, got {parsed.value}")
    return _float32_array(value, name=name, shape=shape)


def _check_mass(source: np.ndarray, projected: np.ndarray, operation: str) -> None:
    before = float(np.sum(source, dtype=np.float64))
    after = float(np.sum(projected, dtype=np.float64))
    if not np.isclose(
        before, after, atol=PROJECTION_ATOL, rtol=PROJECTION_RTOL
    ):
        raise ArithmeticError(
            f"{operation} violated signed-mass conservation: "
            f"before={before:.17g}, after={after:.17g}"
        )


def raw_array(value: Any, *, metadata: Mapping[str, Any] | None = None) -> ArtifactArray:
    """Tag an owned finite float32 numeric tensor as native/raw data."""

    kind = getattr(value, "kind", None)
    if kind is not None and ArtifactKind.parse(kind) is not ArtifactKind.RAW:
        parsed = ArtifactKind.parse(kind)
        if parsed is ArtifactKind.DISPLAY:
            raise DisplayBoundaryError("raw array rejects display data")
        raise DisplayBoundaryError(
            f"raw array cannot relabel {parsed.value} data as raw"
        )
    value = _reject_display(value, "raw array")
    array = np.asarray(value)
    if array.dtype != np.dtype(np.float32):
        raise TypeError(f"raw array must have dtype float32, got {array.dtype}")
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("raw array must be non-empty and finite")
    return ArtifactArray(array, ArtifactKind.RAW, metadata=dict(metadata or {}))


def stage_c_metric_ready(value: Any) -> ArtifactArray:
    """Tag an exact float32 ``14 x 14`` map for Stage C metrics."""

    kind = getattr(value, "kind", None)
    if kind is not None and ArtifactKind.parse(kind) is not ArtifactKind.METRIC_READY:
        parsed = ArtifactKind.parse(kind)
        if parsed is ArtifactKind.DISPLAY:
            raise DisplayBoundaryError("Stage C metric map rejects display data")
        raise DisplayBoundaryError(
            f"Stage C metric map cannot relabel {parsed.value} data as metric_ready"
        )
    array = _float32_array(value, name="Stage C metric map", shape=PATCH_SHAPE)
    return ArtifactArray(
        array,
        ArtifactKind.METRIC_READY,
        metadata={"shape": list(PATCH_SHAPE), "dtype": "float32"},
    )


def signed_rgb_channel_sum(value: Any) -> ArtifactArray:
    """Reduce native CHW RGB attribution without taking absolute values."""

    rgb = _raw_float32_array(
        value, name="signed RGB attribution", shape=(3, *INPUT_SHAPE)
    )
    reduced = np.sum(rgb, axis=0, dtype=np.float64).astype(np.float32)
    _check_mass(rgb, reduced, "signed RGB channel sum")
    return raw_array(
        reduced, metadata={"operator": "signed_channel_sum", "source_shape": [3, 224, 224]}
    )


def pixel_to_patch_mass(value: Any) -> ArtifactArray:
    """Sum every non-overlapping ``16 x 16`` signed pixel region."""

    pixels = _raw_float32_array(value, name="pixel map", shape=INPUT_SHAPE)
    patches = pixels.reshape(14, 16, 14, 16).sum(
        axis=(1, 3), dtype=np.float64
    ).astype(np.float32)
    _check_mass(pixels, patches, "pixel_to_patch_mass")
    return stage_c_metric_ready(patches)


def patch_to_pixel_mass(value: Any) -> ArtifactArray:
    """Uniformly distribute each patch mass over its 256 pixels."""

    patches = validate_metric_input(value, name="patch map")
    patches = _float32_array(patches, name="patch map", shape=PATCH_SHAPE)
    pixels = np.repeat(np.repeat(patches / np.float32(256.0), 16, axis=0), 16, axis=1)
    pixels = np.asarray(pixels, dtype=np.float32)
    _check_mass(patches, pixels, "patch_to_pixel_mass")
    return raw_array(
        pixels, metadata={"operator": "uniform_mass_divide_by_256"}
    )


def pixel_to_fine_cell_mass(value: Any) -> ArtifactArray:
    """Sum the pixel map over every non-overlapping ``4 x 4`` cell."""

    pixels = _raw_float32_array(value, name="pixel map", shape=INPUT_SHAPE)
    cells = pixels.reshape(56, 4, 56, 4).sum(
        axis=(1, 3), dtype=np.float64
    ).astype(np.float32)
    _check_mass(pixels, cells, "pixel_to_fine_cell_mass")
    return raw_array(cells, metadata={"operator": "nonoverlapping_4x4_sum"})


def patch_to_fine_cell_mass(value: Any) -> ArtifactArray:
    """Uniformly distribute each patch mass over its sixteen 4x4 cells."""

    patches = validate_metric_input(value, name="patch map")
    patches = _float32_array(patches, name="patch map", shape=PATCH_SHAPE)
    cells = np.repeat(np.repeat(patches / np.float32(16.0), 4, axis=0), 4, axis=1)
    cells = np.asarray(cells, dtype=np.float32)
    _check_mass(patches, cells, "patch_to_fine_cell_mass")
    return raw_array(
        cells, metadata={"operator": "uniform_mass_divide_by_16"}
    )


def split_sign_views(value: Any, semantics: SignSemantics | str) -> Mapping[SignView, ArtifactArray | StructuredNA]:
    """Split an already-projected metric map into registered sign views."""

    source = validate_metric_input(value, name="metric map for sign split")
    source = _float32_array(source, name="metric map for sign split", shape=PATCH_SHAPE)
    parsed = _parse_enum(semantics, SignSemantics, "sign semantics")
    if parsed is SignSemantics.NONNEGATIVE and np.any(source < 0):
        raise ValueError("a nonnegative map contains negative values")
    signed = stage_c_metric_ready(source)
    positive = stage_c_metric_ready(np.maximum(source, np.float32(0.0)))
    negative: ArtifactArray | StructuredNA
    if parsed is SignSemantics.SIGNED:
        negative = stage_c_metric_ready(np.maximum(-source, np.float32(0.0)))
    else:
        negative = StructuredNA(NAReason.UNSUPPORTED_SIGN_SEMANTICS)
    return MappingProxyType(
        {
            SignView.SIGNED: signed,
            SignView.POSITIVE: positive,
            SignView.NEGATIVE_MAGNITUDE: negative,
        }
    )


_METHOD_CONTRACTS: Mapping[
    str, tuple[str, str | None, str | None, bool, str, SignSemantics]
] = {
    "direct_raw_cls_attention_per_head": (
        "value_path", None, None, True, "raw_cls_attention_to_patch",
        SignSemantics.NONNEGATIVE,
    ),
    "conditional_fixed_clean_attention_head_direction_value_path_ig": (
        "value_path",
        "fixed_clean_attention_cls_head_direction",
        "fixed_clean_attention_cls_head_direction",
        True,
        "pixel_mass_to_patch_sum", SignSemantics.SIGNED,
    ),
    "matched_fixed_clean_attention_patch_token_contribution": (
        "value_path",
        "fixed_clean_attention_cls_head_direction",
        "fixed_clean_attention_cls_head_direction",
        True,
        "patch_mass_to_pixel_uniform", SignSemantics.SIGNED,
    ),
    "direct_raw_cls_attention_last_block_head_mean": (
        "class_logit", None, "clean_predicted_class_raw_logit", True,
        "raw_cls_attention_to_patch", SignSemantics.NONNEGATIVE,
    ),
    "attention_rollout_mean_head_residual_all_blocks": (
        "class_logit", None, "clean_predicted_class_raw_logit", True,
        "rollout_cls_to_patch", SignSemantics.NONNEGATIVE,
    ),
    "vit_grad_cam_block11_norm1": (
        "class_logit",
        "clean_predicted_class_raw_logit",
        "clean_predicted_class_raw_logit",
        False,
        "identity_14x14", SignSemantics.NONNEGATIVE,
    ),
    "full_model_class_logit_ig": (
        "class_logit",
        "clean_predicted_class_raw_logit",
        "clean_predicted_class_raw_logit",
        False,
        "pixel_mass_to_patch_sum", SignSemantics.SIGNED,
    ),
}


@dataclass(frozen=True)
class MethodResult:
    """Immutable identity, target, native tensor, and common-map contract."""

    method_id: str
    method_version: str
    comparison_line: ComparisonLine
    explanation_target: str | None
    fidelity_target: str | None
    class_agnostic: bool
    native_tensor: ArtifactArray | None
    metric_map: ArtifactArray | None
    projection: str
    sign_semantics: SignSemantics
    valid_mask: np.ndarray
    provenance: Mapping[str, Any]
    na: StructuredNA | None = None

    def __post_init__(self) -> None:
        if self.method_id not in _METHOD_CONTRACTS:
            raise ValueError(f"unknown Stage C method_id: {self.method_id!r}")
        if not isinstance(self.method_version, str) or not re.fullmatch(
            r"[0-9]+\.[0-9]+\.[0-9]+(?:[-+][A-Za-z0-9.-]+)?", self.method_version
        ):
            raise ValueError("method_version must be an explicit semantic version")
        line = _parse_enum(self.comparison_line, ComparisonLine, "comparison_line")
        signs = _parse_enum(self.sign_semantics, SignSemantics, "sign_semantics")
        if type(self.class_agnostic) is not bool:
            raise TypeError("class_agnostic must be a boolean")
        expected = _METHOD_CONTRACTS[self.method_id]
        actual = (
            line.value, self.explanation_target, self.fidelity_target,
            self.class_agnostic, self.projection, signs,
        )
        if actual != expected:
            raise ValueError(
                f"method/target/projection contract mismatch for {self.method_id}: "
                f"expected {expected}, got {actual}"
            )
        if (
            self.class_agnostic
            and self.explanation_target == "clean_predicted_class_raw_logit"
        ):
            raise ValueError("a class-agnostic map cannot claim a class explanation")
        na = self.na
        if na is not None and not isinstance(na, StructuredNA):
            raise TypeError("na must be StructuredNA or None")
        if na is None and (self.native_tensor is None or self.metric_map is None):
            raise ValueError("a successful MethodResult requires native_tensor and metric_map")
        native: np.ndarray | None = None
        if self.native_tensor is not None:
            if not isinstance(self.native_tensor, ArtifactArray) or (
                self.native_tensor.kind is not ArtifactKind.RAW
            ):
                raise TypeError("native_tensor must be explicitly tagged raw")
            native = np.asarray(self.native_tensor.data)
            if native.dtype != np.float32 or native.size == 0 or not np.all(np.isfinite(native)):
                raise ValueError("native_tensor must be a non-empty finite float32 tensor")
        metric: np.ndarray | None = None
        if self.metric_map is not None:
            metric = validate_metric_input(self.metric_map, name="MethodResult.metric_map")
            metric = _float32_array(
                metric, name="MethodResult.metric_map", shape=PATCH_SHAPE
            )
            if signs is SignSemantics.NONNEGATIVE and np.any(metric < 0):
                raise ValueError("nonnegative MethodResult.metric_map contains negative values")
        if native is not None and metric is not None:
            if self.method_id in {
                "conditional_fixed_clean_attention_head_direction_value_path_ig",
                "full_model_class_logit_ig",
            }:
                if native.shape != (3, *INPUT_SHAPE):
                    raise ValueError(
                        "pixel-native IG methods require native RGB shape (3,224,224)"
                    )
                expected_metric = pixel_to_patch_mass(
                    signed_rgb_channel_sum(self.native_tensor)
                ).data
            else:
                if native.shape != PATCH_SHAPE:
                    raise ValueError(
                        f"patch-native method requires native shape {PATCH_SHAPE}"
                    )
                expected_metric = native
            if not np.allclose(
                metric, expected_metric, atol=PROJECTION_ATOL, rtol=PROJECTION_RTOL
            ):
                raise ValueError(
                    "metric_map does not equal the registered projection of native_tensor"
                )
        mask = np.asarray(self.valid_mask)
        if mask.dtype != np.bool_ or mask.shape != PATCH_SHAPE:
            raise ValueError("valid_mask must be a bool 14 x 14 array")
        if not np.any(mask) and not (na is not None and na.invalidating):
            raise ValueError(
                "an empty valid_mask requires an invalidating structured NA"
            )
        if not isinstance(self.provenance, Mapping) or not self.provenance:
            raise ValueError("provenance must be a non-empty mapping")
        if any(not isinstance(key, str) or not key for key in self.provenance):
            raise ValueError("provenance keys must be non-empty strings")
        _validate_json_fragment(self.provenance, "provenance")
        object.__setattr__(self, "comparison_line", line)
        object.__setattr__(self, "sign_semantics", signs)
        object.__setattr__(self, "valid_mask", _immutable_array(mask))
        object.__setattr__(self, "provenance", _deep_freeze(dict(self.provenance)))

    @property
    def sign_views(self) -> Mapping[SignView, ArtifactArray | StructuredNA]:
        if self.metric_map is None:
            if self.na is None:
                raise RuntimeError("MethodResult has neither a metric map nor an NA state")
            return MappingProxyType(
                {
                    SignView.SIGNED: self.na,
                    SignView.POSITIVE: self.na,
                    SignView.NEGATIVE_MAGNITUDE: self.na,
                }
            )
        return split_sign_views(self.metric_map, self.sign_semantics)


def _validate_json_fragment(value: Any, context: str) -> None:
    if value is None or type(value) in {bool, int, str}:
        return
    if type(value) is float:
        if not np.isfinite(value):
            raise ValueError(f"{context} contains a non-finite float")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise ValueError(f"{context} keys must be non-empty strings")
            _validate_json_fragment(item, f"{context}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_json_fragment(item, f"{context}[{index}]")
        return
    raise TypeError(
        f"{context} must contain only finite JSON-compatible values, got {type(value).__name__}"
    )


__all__ = [
    "ComparisonLine", "FINE_CELL_SHAPE", "INPUT_SHAPE",
    "INVALIDATING_NA_REASONS", "MethodResult", "NAReason",
    "PATCH_SHAPE", "PROJECTION_ATOL", "PROJECTION_RTOL",
    "SignSemantics", "SignView", "StructuredNA",
    "patch_to_fine_cell_mass", "patch_to_pixel_mass",
    "pixel_to_fine_cell_mass", "pixel_to_patch_mass", "raw_array",
    "signed_rgb_channel_sum", "split_sign_views", "stage_c_metric_ready",
]
