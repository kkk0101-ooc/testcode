"""Explicit, provenance-carrying transforms for display-only arrays.

Every operation in this module crosses the metric-ready -> display boundary.
The defaults are an identity transform with every legacy enhancement disabled.
Display output is tagged and metric validators reject it, preventing rendered
or interpolated values from silently entering quantitative evaluation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter, zoom

from .retention import (
    ArtifactArray,
    ArtifactKind,
    PolicyViolationError,
    _deep_freeze,
    _immutable_array,
    array_sha256,
)


class DisplayBoundaryError(PolicyViolationError):
    """Raised when display data crosses back into a metric computation."""


@dataclass(frozen=True)
class DisplayResult:
    """Immutable display array plus an explicit transformation record."""

    data: np.ndarray
    kind: ArtifactKind
    transforms: tuple[str, ...]
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        array = _immutable_array(self.data)
        kind = ArtifactKind.parse(self.kind)
        if kind is not ArtifactKind.DISPLAY:
            raise DisplayBoundaryError("DisplayResult must be tagged as display")
        object.__setattr__(self, "data", array)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "transforms", tuple(self.transforms))
        object.__setattr__(self, "metadata", _deep_freeze(dict(self.metadata)))

    @property
    def values(self) -> np.ndarray:
        return self.data

    @property
    def array(self) -> np.ndarray:
        return self.data

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def dtype(self) -> np.dtype[Any]:
        return self.data.dtype

    @property
    def sha256(self) -> str:
        return array_sha256(self.data)

    @property
    def transformed(self) -> bool:
        return self.transforms != ("identity",)

    def __array__(self, dtype: np.dtype[Any] | None = None) -> np.ndarray:
        return np.asarray(self.data, dtype=dtype)


@dataclass(frozen=True)
class DisplayTransform:
    """Configured display-only transformations, all disabled by default.

    Transform order is fixed and recorded: legacy fusion, percentile clipping,
    max pooling, Gaussian smoothing, then resize.  ``legacy_fusion_weight`` is
    the weight of ``class_feature``; the metric-ready attention input receives
    ``1 - weight``.  Setting it to 0.7 reproduces the historical 0.7/0.3 mix.
    """

    legacy_fusion_weight: float | None = None
    percentile_clip: tuple[float, float] | None = None
    max_pool_size: int | None = None
    gaussian_sigma: float | None = None
    resize: tuple[int, int] | None = None
    resize_order: int = 3

    def __post_init__(self) -> None:
        if self.legacy_fusion_weight is not None:
            weight = float(self.legacy_fusion_weight)
            if not 0.0 <= weight <= 1.0:
                raise ValueError("legacy_fusion_weight must be in [0, 1]")
            object.__setattr__(self, "legacy_fusion_weight", weight)
        if self.percentile_clip is not None:
            if len(self.percentile_clip) != 2:
                raise ValueError("percentile_clip must be (low, high)")
            low, high = (float(value) for value in self.percentile_clip)
            if not 0.0 <= low < high <= 100.0:
                raise ValueError("percentile_clip must satisfy 0 <= low < high <= 100")
            object.__setattr__(self, "percentile_clip", (low, high))
        if self.max_pool_size is not None:
            if (
                isinstance(self.max_pool_size, bool)
                or not isinstance(self.max_pool_size, int)
                or self.max_pool_size <= 0
            ):
                raise ValueError("max_pool_size must be a positive integer")
        if self.gaussian_sigma is not None:
            sigma = float(self.gaussian_sigma)
            if not np.isfinite(sigma) or sigma < 0.0:
                raise ValueError("gaussian_sigma must be finite and non-negative")
            object.__setattr__(self, "gaussian_sigma", sigma)
        if self.resize is not None:
            if len(self.resize) != 2 or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in self.resize
            ):
                raise ValueError("resize must be a positive (height, width) pair")
            object.__setattr__(self, "resize", tuple(self.resize))
        if isinstance(self.resize_order, bool) or self.resize_order not in range(6):
            raise ValueError("resize_order must be an integer from 0 through 5")

    @property
    def is_identity(self) -> bool:
        return (
            self.legacy_fusion_weight is None
            and self.percentile_clip is None
            and self.max_pool_size is None
            and (self.gaussian_sigma is None or self.gaussian_sigma == 0.0)
            and self.resize is None
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        if value["percentile_clip"] is not None:
            value["percentile_clip"] = list(value["percentile_clip"])
        if value["resize"] is not None:
            value["resize"] = list(value["resize"])
        return value

    def apply(
        self,
        metric_ready: Any,
        *,
        class_feature: Any | None = None,
    ) -> DisplayResult:
        """Apply configured operations without modifying either input."""

        source = validate_metric_input(metric_ready, name="metric_ready")
        native_shape = tuple(source.shape)
        source_hash = array_sha256(source)
        # A true identity preserves dtype, values, and therefore the canonical
        # array hash.  Enabled numerical display operations use float32 (which
        # also avoids scipy's float16 limitations) on a caller-independent copy.
        output = np.array(source, copy=True)
        if not self.is_identity:
            output = output.astype(np.float32, copy=False)
        transforms: list[str] = []
        transform_details: list[dict[str, Any]] = []

        if self.legacy_fusion_weight is not None:
            if class_feature is None:
                raise ValueError("class_feature is required when legacy fusion is enabled")
            feature = validate_metric_input(class_feature, name="class_feature")
            if feature.shape != source.shape:
                raise ValueError(
                    f"class_feature shape {feature.shape} does not match metric map {source.shape}"
                )
            weight = self.legacy_fusion_weight
            output = weight * feature.astype(np.float32, copy=False) + (1.0 - weight) * output
            transforms.append("legacy_fusion")
            transform_details.append({
                "name": "legacy_fusion",
                "class_feature_weight": weight,
                "attention_weight": 1.0 - weight,
                "class_feature_sha256": array_sha256(feature),
            })
        elif class_feature is not None:
            raise ValueError(
                "class_feature was supplied but legacy_fusion_weight is disabled"
            )

        if self.percentile_clip is not None:
            low, high = self.percentile_clip
            low_value, high_value = np.percentile(output, (low, high))
            output = np.clip(output, low_value, high_value)
            transforms.append("percentile_clip")
            transform_details.append({
                "name": "percentile_clip",
                "low_percentile": low,
                "high_percentile": high,
                "low_value": float(low_value),
                "high_value": float(high_value),
            })

        if self.max_pool_size is not None:
            output = maximum_filter(output, size=self.max_pool_size, mode="nearest")
            transforms.append("max_pool")
            transform_details.append({
                "name": "max_pool", "size": self.max_pool_size, "stride": 1
            })

        if self.gaussian_sigma is not None and self.gaussian_sigma > 0.0:
            output = gaussian_filter(output, sigma=self.gaussian_sigma, mode="nearest")
            transforms.append("gaussian_smoothing")
            transform_details.append({
                "name": "gaussian_smoothing", "sigma": self.gaussian_sigma
            })

        if self.resize is not None:
            factors = tuple(target / actual for target, actual in zip(self.resize, output.shape))
            output = zoom(output, zoom=factors, order=self.resize_order, mode="nearest", prefilter=True)
            # scipy's rounding should already match; slice/pad defensively so
            # display_shape is exactly the declared one for every odd size.
            output = _force_shape(output, self.resize)
            transforms.append("resize")
            transform_details.append({
                "name": "resize",
                "target_shape": list(self.resize),
                "order": self.resize_order,
            })

        if not transforms:
            transforms.append("identity")
            transform_details.append({"name": "identity"})

        output = np.ascontiguousarray(output)
        metadata = {
            "input_kind": ArtifactKind.METRIC_READY.value,
            "output_kind": ArtifactKind.DISPLAY.value,
            "native_shape": list(native_shape),
            "display_shape": list(output.shape),
            "source_dtype": str(source.dtype),
            "display_dtype": str(output.dtype),
            "source_sha256": source_hash,
            "transformed": transforms != ["identity"],
            "transforms": transform_details,
            "config": self.to_dict(),
        }
        return DisplayResult(output, ArtifactKind.DISPLAY, tuple(transforms), metadata)

    __call__ = apply


def metric_ready(value: Any) -> ArtifactArray:
    """Explicitly tag a finite 2D numeric array as metric-ready."""

    _reject_retagged_artifact(value, target=ArtifactKind.METRIC_READY)
    array = _validate_numeric_2d(value, name="metric_ready")
    return ArtifactArray(
        array,
        ArtifactKind.METRIC_READY,
        metadata={"native_shape": list(array.shape), "dtype": str(array.dtype)},
    )


def raw(value: Any) -> ArtifactArray:
    """Explicitly tag an unmodified finite RGB array as raw.

    Raw, metric-ready, and display identities are intentionally not
    interchangeable.  In particular, this constructor cannot be used to
    relabel a display result and thereby bypass the metric boundary.
    """

    _reject_retagged_artifact(value, target=ArtifactKind.RAW)
    array = _validate_rgb(value, name="raw")
    return ArtifactArray(
        array,
        ArtifactKind.RAW,
        metadata={"native_shape": list(array.shape), "dtype": str(array.dtype)},
    )


def validate_metric_input(
    value: Any, *, name: str = "metric input", allow_untagged: bool = False
) -> np.ndarray:
    """Return a metric-ready array, rejecting all display-tagged values.

    Inputs must be explicitly tagged ``metric_ready``.  ``allow_untagged`` is
    reserved for narrow compatibility adapters whose own boundary is already
    explicit; metric implementations should leave it false.
    """

    kind = getattr(value, "kind", None)
    if kind is not None:
        parsed = ArtifactKind.parse(kind)
        if parsed is ArtifactKind.DISPLAY:
            raise DisplayBoundaryError(
                f"{name} rejects display data; quantitative metrics require metric_ready input"
            )
        if parsed is not ArtifactKind.METRIC_READY:
            raise DisplayBoundaryError(
                f"{name} requires metric_ready input, got {parsed.value}"
            )
        value = getattr(value, "data", value)
    elif not allow_untagged:
        raise DisplayBoundaryError(
            f"{name} requires an explicitly tagged metric_ready value; "
            "use metric_ready(array) first"
        )
    return _validate_numeric_2d(value, name=name)


def require_metric_ready(value: Any, *, name: str = "metric input") -> np.ndarray:
    """Compatibility alias for metric entry points."""

    return validate_metric_input(value, name=name)


def validate_raw_input(value: Any, *, name: str = "raw input") -> np.ndarray:
    """Return a tagged raw RGB array and reject every other identity."""

    kind = getattr(value, "kind", None)
    if kind is None or ArtifactKind.parse(kind) is not ArtifactKind.RAW:
        observed = "untagged" if kind is None else ArtifactKind.parse(kind).value
        raise DisplayBoundaryError(f"{name} requires raw input, got {observed}")
    return _validate_rgb(getattr(value, "data", value), name=name)


def validate_display_input(value: Any, *, name: str = "display input") -> np.ndarray:
    """Return a tagged display array and reject raw/metric-ready inputs."""

    kind = getattr(value, "kind", None)
    if kind is None or ArtifactKind.parse(kind) is not ArtifactKind.DISPLAY:
        observed = "untagged" if kind is None else ArtifactKind.parse(kind).value
        raise DisplayBoundaryError(f"{name} requires display input, got {observed}")
    return _validate_numeric_2d(getattr(value, "data", value), name=name)


def _reject_retagged_artifact(value: Any, *, target: ArtifactKind) -> None:
    kind = getattr(value, "kind", None)
    if kind is not None:
        parsed = ArtifactKind.parse(kind)
        raise DisplayBoundaryError(
            f"cannot relabel {parsed.value} data as {target.value}; "
            "artifact identities are immutable"
        )


def _validate_numeric_2d(value: Any, *, name: str) -> np.ndarray:
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D map, got shape {array.shape}")
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.complexfloating):
        raise TypeError(f"{name} must be a real numeric map")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _validate_rgb(value: Any, *, name: str) -> np.ndarray:
    if hasattr(value, "detach") or hasattr(value, "cpu"):
        raise TypeError(f"{name} must be a NumPy-compatible RGB array, not a tensor")
    array = np.asarray(value)
    if array.ndim != 3 or array.shape[2] not in (3, 4):
        raise ValueError(f"{name} must be an HxWx3 or HxWx4 RGB(A) array")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must be numeric")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _force_shape(array: np.ndarray, target: Sequence[int]) -> np.ndarray:
    target_shape = tuple(target)
    result = array[: target_shape[0], : target_shape[1]]
    pad_h = target_shape[0] - result.shape[0]
    pad_w = target_shape[1] - result.shape[1]
    if pad_h or pad_w:
        result = np.pad(result, ((0, pad_h), (0, pad_w)), mode="edge")
    return result


__all__ = [
    "DisplayBoundaryError",
    "DisplayResult",
    "DisplayTransform",
    "metric_ready",
    "raw",
    "require_metric_ready",
    "validate_display_input",
    "validate_metric_input",
    "validate_raw_input",
]
