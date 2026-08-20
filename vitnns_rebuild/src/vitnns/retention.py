"""Retention boundaries and fail-closed artifact writing for Stage B.

The module deliberately keeps storage policy separate from attribution and
display code.  In particular, a ``paper`` run can only persist compact,
two-dimensional metric-ready maps.  An ``audit`` run retains the float32 RGB
source attribution as well as the float32 metric-ready map from which the
paper representation can be reproduced.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import threading
import uuid
from copy import deepcopy
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from numbers import Integral
from pathlib import Path, PurePath
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np


# Conservative, compression-independent allowance for manifest/status JSON and
# NPZ container metadata in static preflight estimates.  Actual artifact writes
# still use exact serialized byte counts.
CONTROL_OVERHEAD_BYTES = 64 * 1024
PER_ARTIFACT_OVERHEAD_BYTES = 4 * 1024


class RetentionError(RuntimeError):
    """Base error for retention-policy and artifact-write failures."""


class PolicyViolationError(RetentionError, ValueError):
    """Raised when an artifact violates its immutable retention policy."""


class BudgetExceededError(RetentionError):
    """Raised before a write whose exact serialized size exceeds budget."""


class OutputDirectoryNotEmptyError(RetentionError, FileExistsError):
    """Raised rather than overwriting or resuming an existing output."""


class IncompleteArtifactError(RetentionError):
    """Raised when an incomplete or failed artifact set cannot finalize."""


class ArtifactKind(str, Enum):
    """The three non-interchangeable stages of attribution data."""

    RAW = "raw"
    METRIC_READY = "metric_ready"
    DISPLAY = "display"

    # Lower-case aliases make configuration-oriented usage unsurprising while
    # keeping conventional upper-case Enum members as the documented API.
    raw = RAW
    metric_ready = METRIC_READY
    display = DISPLAY

    @classmethod
    def parse(cls, value: "ArtifactKind | str") -> "ArtifactKind":
        if isinstance(value, cls):
            return value
        normalized = str(value).strip().lower().replace("-", "_")
        try:
            return cls(normalized)
        except ValueError as exc:
            choices = ", ".join(item.value for item in cls)
            raise ValueError(f"Unknown artifact kind {value!r}; expected one of {choices}") from exc


class RetentionProfile(str, Enum):
    DEBUG = "debug"
    PAPER = "paper"
    AUDIT = "audit"

    debug = DEBUG
    paper = PAPER
    audit = AUDIT

    @classmethod
    def parse(cls, value: "RetentionProfile | str") -> "RetentionProfile":
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).strip().lower())
        except ValueError as exc:
            choices = ", ".join(item.value for item in cls)
            raise ValueError(f"Unknown retention profile {value!r}; expected one of {choices}") from exc


@dataclass(frozen=True)
class RetentionPolicy:
    """Immutable persistence rules for one Stage-B retention profile.

    ``None`` for a dtype means that the writer preserves the supplied dtype.
    JSON control/metric documents remain permitted for all profiles; the
    array-specific dtype and rank rules apply to NPZ members.
    """

    profile: RetentionProfile
    allowed_kinds: frozenset[ArtifactKind]
    raw_storage_dtype: str | None
    metric_storage_dtype: str | None
    display_storage_dtype: str | None
    metric_ndim: int | None
    raw_rgb_required: bool
    allow_png: bool
    allow_per_pixel_files: bool
    allow_per_dimension_files: bool
    required_npz_kinds: frozenset[ArtifactKind]

    @property
    def name(self) -> str:
        return self.profile.value

    @property
    def storage_dtype(self) -> str | None:
        """Paper-oriented alias for the metric-ready storage dtype."""

        return self.metric_storage_dtype

    @property
    def save_raw(self) -> bool:
        return ArtifactKind.RAW in self.allowed_kinds

    @property
    def save_metric_ready(self) -> bool:
        return ArtifactKind.METRIC_READY in self.allowed_kinds

    @property
    def save_display(self) -> bool:
        return ArtifactKind.DISPLAY in self.allowed_kinds

    @property
    def save_display_png(self) -> bool:
        return self.allow_png and self.save_display

    def allows(self, kind: ArtifactKind | str) -> bool:
        return ArtifactKind.parse(kind) in self.allowed_kinds

    def validate_artifact(
        self,
        *,
        kind: ArtifactKind | str,
        name: str,
        artifact_format: str,
        granularity: str = "aggregate",
    ) -> ArtifactKind:
        kind = ArtifactKind.parse(kind)
        if kind not in self.allowed_kinds:
            raise PolicyViolationError(
                f"{self.name} retention does not allow {kind.value!r} artifacts"
            )

        lowered_name = str(name).lower()
        lowered_format = str(artifact_format).lower().lstrip(".")
        if (lowered_format == "png" or lowered_name.endswith(".png")) and not self.allow_png:
            raise PolicyViolationError(f"{self.name} retention forbids PNG artifacts")

        normalized_granularity = str(granularity).strip().lower().replace("-", "_")
        if normalized_granularity in {"pixel", "per_pixel", "pixels"}:
            if not self.allow_per_pixel_files:
                raise PolicyViolationError(
                    f"{self.name} retention forbids one-file-per-pixel artifacts"
                )
        elif normalized_granularity in {
            "dimension", "dim", "per_dimension", "per_dim", "dimensions",
        }:
            if not self.allow_per_dimension_files:
                raise PolicyViolationError(
                    f"{self.name} retention forbids one-file-per-dimension artifacts"
                )
        elif normalized_granularity not in {"aggregate", "sample", "layer", "head", "target"}:
            raise ValueError(f"Unknown artifact granularity {granularity!r}")

        if self.profile is RetentionProfile.PAPER and _looks_like_granular_filename(name):
            raise PolicyViolationError(
                "paper retention forbids filenames that encode an individual pixel or dimension"
            )
        return kind


DEBUG_POLICY = RetentionPolicy(
    profile=RetentionProfile.DEBUG,
    allowed_kinds=frozenset(ArtifactKind),
    raw_storage_dtype=None,
    metric_storage_dtype=None,
    display_storage_dtype=None,
    metric_ndim=None,
    raw_rgb_required=False,
    allow_png=True,
    allow_per_pixel_files=True,
    allow_per_dimension_files=True,
    required_npz_kinds=frozenset(),
)

PAPER_POLICY = RetentionPolicy(
    profile=RetentionProfile.PAPER,
    allowed_kinds=frozenset({ArtifactKind.METRIC_READY}),
    raw_storage_dtype=None,
    metric_storage_dtype="float16",
    display_storage_dtype=None,
    metric_ndim=2,
    raw_rgb_required=False,
    allow_png=False,
    allow_per_pixel_files=False,
    allow_per_dimension_files=False,
    required_npz_kinds=frozenset({ArtifactKind.METRIC_READY}),
)

AUDIT_POLICY = RetentionPolicy(
    profile=RetentionProfile.AUDIT,
    allowed_kinds=frozenset({ArtifactKind.RAW, ArtifactKind.METRIC_READY}),
    raw_storage_dtype="float32",
    metric_storage_dtype="float32",
    display_storage_dtype=None,
    metric_ndim=2,
    raw_rgb_required=True,
    allow_png=False,
    allow_per_pixel_files=False,
    allow_per_dimension_files=False,
    required_npz_kinds=frozenset({ArtifactKind.RAW, ArtifactKind.METRIC_READY}),
)

_POLICIES = MappingProxyType({
    RetentionProfile.DEBUG: DEBUG_POLICY,
    RetentionProfile.PAPER: PAPER_POLICY,
    RetentionProfile.AUDIT: AUDIT_POLICY,
})


def get_retention_policy(
    profile: RetentionProfile | str | RetentionPolicy,
) -> RetentionPolicy:
    """Return the immutable singleton policy for ``profile``."""

    if isinstance(profile, RetentionPolicy):
        canonical = _POLICIES[RetentionProfile.parse(profile.profile)]
        if profile != canonical:
            raise PolicyViolationError(
                "custom policy overrides are forbidden; choose debug, paper, or audit"
            )
        return canonical
    return _POLICIES[RetentionProfile.parse(profile)]


def estimate_output_bytes(
    layer_count: int,
    shape: Sequence[int],
    profile: RetentionProfile | str | RetentionPolicy,
    *,
    head_count: int = 1,
    target_count: int = 1,
) -> int:
    """Estimate uncompressed numeric payload bytes before a run starts.

    ``shape`` may be the native ``(H, W)`` metric-map shape or its RGB
    ``(3, H, W)``/``(H, W, 3)`` raw-attribution shape.  The estimate is
    intentionally static and independent of NPZ compression ratios.  Paper
    retains one float16 map (2 bytes/pixel); audit retains a float32 metric map
    plus its float32 RGB raw attribution (16 bytes/pixel); debug budgets those
    plus one float32 display plane (20 bytes/pixel).
    """

    counts = {
        "layer_count": layer_count,
        "head_count": head_count,
        "target_count": target_count,
    }
    for label, value in counts.items():
        if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
            raise ValueError(f"{label} must be a positive integer")
    try:
        raw_dimensions = tuple(shape)
    except TypeError as exc:
        raise ValueError("shape must be a finite sequence of positive integers") from exc
    if any(
        isinstance(value, bool) or not isinstance(value, Integral)
        for value in raw_dimensions
    ):
        raise ValueError("shape must contain positive integers")
    dimensions = tuple(int(value) for value in raw_dimensions)
    if any(value <= 0 for value in dimensions):
        raise ValueError("shape must contain only positive dimensions")
    if len(dimensions) == 2:
        metric_dimensions = dimensions
    elif len(dimensions) == 3 and dimensions[0] == 3:
        metric_dimensions = dimensions[1:]
    elif len(dimensions) == 3 and dimensions[-1] == 3:
        metric_dimensions = dimensions[:2]
    else:
        raise ValueError(
            "shape must be (H, W), (3, H, W), or (H, W, 3)"
        )

    selected = get_retention_policy(profile).profile
    bytes_per_pixel = {
        RetentionProfile.PAPER: 2,
        RetentionProfile.AUDIT: 4 + (3 * 4),
        RetentionProfile.DEBUG: (3 * 4) + 4 + 4,
    }[selected]
    maps = layer_count * head_count * target_count
    payload_bytes = maps * math.prod(metric_dimensions) * bytes_per_pixel
    # A profile may pack maps into fewer NPZ files, but budgeting one container
    # allowance per logical map is intentionally conservative and static.
    return int(payload_bytes + CONTROL_OVERHEAD_BYTES + maps * PER_ARTIFACT_OVERHEAD_BYTES)


@dataclass(frozen=True)
class ArtifactArray:
    """An ndarray carrying its raw/metric-ready/display boundary tag."""

    data: np.ndarray
    kind: ArtifactKind
    transforms: tuple[str, ...] = ()
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        array = _immutable_array(self.data)
        object.__setattr__(self, "data", array)
        object.__setattr__(self, "kind", ArtifactKind.parse(self.kind))
        object.__setattr__(self, "transforms", tuple(self.transforms))
        object.__setattr__(self, "metadata", _deep_freeze(dict(self.metadata or {})))

    @property
    def values(self) -> np.ndarray:
        return self.data

    @property
    def array(self) -> np.ndarray:
        return self.data

    @property
    def dtype(self) -> np.dtype[Any]:
        return self.data.dtype

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def sha256(self) -> str:
        return array_sha256(self.data)

    def __array__(self, dtype: np.dtype[Any] | None = None) -> np.ndarray:
        return np.asarray(self.data, dtype=dtype)


def array_sha256(value: Any) -> str:
    """Hash array dtype, shape, and bytes using a stable canonical layout."""

    array = np.asarray(_unwrap_array(value))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(np.ascontiguousarray(array).tobytes(order="C"))
    return digest.hexdigest()


def quantize_paper_metric(metric_map: Any) -> np.ndarray:
    """Return the exact float16 representation persisted by paper policy."""

    source = _metric_array(metric_map)
    with np.errstate(over="ignore", invalid="ignore"):
        quantized = np.asarray(source, dtype=np.float16)
    if not np.all(np.isfinite(quantized)):
        raise PolicyViolationError("metric map is not representable as finite float16")
    return quantized


def paper_float16_roundtrip_error(metric_map: Any) -> dict[str, Any]:
    """Measure float32 -> paper float16 -> float32 storage error.

    The input must be an explicitly tagged, two-dimensional metric-ready
    array.  Untagged and display-tagged data are rejected.
    """

    source = _metric_array(metric_map).astype(np.float32, copy=False)
    with np.errstate(over="ignore", invalid="ignore"):
        quantized = source.astype(np.float16)
    if not np.all(np.isfinite(quantized)):
        raise PolicyViolationError("metric map is not representable as finite float16")
    roundtrip = quantized.astype(np.float32)
    difference = roundtrip.astype(np.float64) - source.astype(np.float64)
    absolute = np.abs(difference)
    denominator = np.maximum(np.abs(source.astype(np.float64)), np.finfo(np.float32).tiny)
    return {
        "source_dtype": "float32",
        "storage_dtype": "float16",
        "shape": list(source.shape),
        "max_abs_error": float(absolute.max(initial=0.0)),
        "mean_abs_error": float(absolute.mean()) if absolute.size else 0.0,
        "rmse": float(np.sqrt(np.mean(np.square(difference)))) if difference.size else 0.0,
        "max_relative_error": float((absolute / denominator).max(initial=0.0)),
    }


def audit_replay_report(
    raw_rgb: Any,
    metric_map: Any,
    *,
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> dict[str, Any]:
    """Replay an audit RGB attribution into its paper-ready 2D map.

    Both inputs must be finite float32 arrays.  ``raw_rgb`` has canonical CHW
    shape ``[3,H,W]`` and is reduced by a signed channel sum.  The report checks
    that reduction against ``metric_map`` and then verifies the paper float16
    decode against the same reference using the requested numerical tolerance
    plus the measured storage quantization error.
    """

    _validate_nonnegative_finite_tolerance(atol, "atol")
    _validate_nonnegative_finite_tolerance(rtol, "rtol")
    raw_kind = getattr(raw_rgb, "kind", None)
    if raw_kind is not None and ArtifactKind.parse(raw_kind) is not ArtifactKind.RAW:
        raise PolicyViolationError(
            f"audit replay raw input must have kind raw, got {ArtifactKind.parse(raw_kind).value}"
        )
    metric_kind = getattr(metric_map, "kind", None)
    if (
        metric_kind is not None
        and ArtifactKind.parse(metric_kind) is not ArtifactKind.METRIC_READY
    ):
        raise PolicyViolationError(
            "audit replay metric input must have kind metric_ready, got "
            f"{ArtifactKind.parse(metric_kind).value}"
        )

    raw = np.asarray(_unwrap_array(raw_rgb))
    metric = np.asarray(_unwrap_array(metric_map))
    if raw.dtype != np.dtype(np.float32):
        raise PolicyViolationError(f"audit raw RGB must be float32, got {raw.dtype}")
    if metric.dtype != np.dtype(np.float32):
        raise PolicyViolationError(f"audit metric map must be float32, got {metric.dtype}")
    if raw.ndim != 3 or raw.shape[0] != 3:
        raise PolicyViolationError(
            f"audit raw RGB must have shape [3,H,W], got {raw.shape}"
        )
    if metric.ndim != 2:
        raise PolicyViolationError(f"audit metric map must have shape [H,W], got {metric.shape}")
    if raw.shape[1:] != metric.shape:
        raise PolicyViolationError(
            f"audit raw/metric spatial shapes differ: {raw.shape[1:]} vs {metric.shape}"
        )
    if not np.all(np.isfinite(raw)) or not np.all(np.isfinite(metric)):
        raise PolicyViolationError("audit replay inputs must contain only finite values")

    # Match the Stage-B reducer: a float32, signed channel sum.
    replayed = raw.sum(axis=0, dtype=np.float32)
    residual = replayed.astype(np.float64) - metric.astype(np.float64)
    absolute = np.abs(residual)
    tolerance = atol + rtol * np.abs(metric.astype(np.float64))
    raw_pass = bool(np.all(absolute <= tolerance))

    metric_tagged = ArtifactArray(metric, ArtifactKind.METRIC_READY)
    quantization = paper_float16_roundtrip_error(metric_tagged)
    decoded = quantize_paper_metric(metric_tagged).astype(np.float32)
    decoded_residual = decoded.astype(np.float64) - replayed.astype(np.float64)
    decoded_absolute = np.abs(decoded_residual)
    # The exact measured per-pixel quantization error is the allowed additive
    # storage term; this is tighter and more meaningful than a global magic cap.
    quantization_absolute = np.abs(
        decoded.astype(np.float64) - metric.astype(np.float64)
    )
    decoded_tolerance = tolerance + quantization_absolute
    paper_pass = bool(np.all(decoded_absolute <= decoded_tolerance))

    return {
        "raw_dtype": str(raw.dtype),
        "metric_dtype": str(metric.dtype),
        "raw_shape": list(raw.shape),
        "metric_shape": list(metric.shape),
        "reducer": "signed_float32_sum_axis_0",
        "atol": float(atol),
        "rtol": float(rtol),
        "max_abs_error": float(absolute.max(initial=0.0)),
        "mean_abs_error": float(absolute.mean()) if absolute.size else 0.0,
        "rmse": float(np.sqrt(np.mean(np.square(residual)))) if residual.size else 0.0,
        "raw_replay_pass": raw_pass,
        "paper_float16_roundtrip_error": quantization,
        "paper_replay_max_abs_error": float(decoded_absolute.max(initial=0.0)),
        "paper_replay_rmse": (
            float(np.sqrt(np.mean(np.square(decoded_residual))))
            if decoded_residual.size
            else 0.0
        ),
        "paper_replay_pass": paper_pass,
        "pass": raw_pass and paper_pass,
    }


# Clear aliases for callers that use the shorter terminology.
float16_roundtrip_error = paper_float16_roundtrip_error
audit_paper_roundtrip_error = paper_float16_roundtrip_error


class BudgetTracker:
    """Thread-safe byte budget reserved before any artifact hits disk."""

    def __init__(self, max_bytes: int | None):
        if max_bytes is not None:
            if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 0:
                raise ValueError("max_bytes must be a non-negative integer or None")
        self.max_bytes = max_bytes
        self._used_bytes = 0
        self._failed = False
        self._allocations: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    @property
    def used_bytes(self) -> int:
        with self._lock:
            return self._used_bytes

    @property
    def actual_bytes(self) -> int:
        """Committed artifact payload bytes (control files excluded)."""

        return self.used_bytes

    @property
    def remaining_bytes(self) -> int | None:
        with self._lock:
            if self.max_bytes is None:
                return None
            return self.max_bytes - self._used_bytes

    @property
    def failed(self) -> bool:
        with self._lock:
            return self._failed

    @property
    def allocations(self) -> tuple[dict[str, Any], ...]:
        with self._lock:
            return tuple(deepcopy(self._allocations))

    def can_fit(self, byte_count: int) -> bool:
        byte_count = _validate_byte_count(byte_count)
        with self._lock:
            return self.max_bytes is None or self._used_bytes + byte_count <= self.max_bytes

    def reserve(self, byte_count: int, *, label: str | None = None) -> None:
        """Reserve exact serialized bytes, raising before any filesystem write."""

        byte_count = _validate_byte_count(byte_count)
        with self._lock:
            if self._failed:
                raise BudgetExceededError("byte budget is already in a failed state")
            proposed = self._used_bytes + byte_count
            if self.max_bytes is not None and proposed > self.max_bytes:
                self._failed = True
                raise BudgetExceededError(
                    f"artifact requires {byte_count} bytes but only "
                    f"{self.max_bytes - self._used_bytes} bytes remain"
                )
            self._used_bytes = proposed
            self._allocations.append({"label": label, "bytes": byte_count})

    consume = reserve

    def release(self, byte_count: int, *, label: str | None = None) -> None:
        """Release a reservation after an I/O failure (never after success)."""

        byte_count = _validate_byte_count(byte_count)
        with self._lock:
            if byte_count > self._used_bytes:
                raise ValueError("cannot release more bytes than have been reserved")
            self._used_bytes -= byte_count
            for index in range(len(self._allocations) - 1, -1, -1):
                candidate = self._allocations[index]
                if candidate["bytes"] == byte_count and (label is None or candidate["label"] == label):
                    del self._allocations[index]
                    break

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_bytes": self.max_bytes,
            "used_bytes": self.used_bytes,
            "actual_bytes": self.used_bytes,
            "accounting_scope": "requested artifacts; manifest/status control files excluded",
            "remaining_bytes": self.remaining_bytes,
            "failed": self.failed,
        }


class ArtifactWriter:
    """Write NPZ/JSON artifacts with policy, budget, and completion gates.

    The directory must not contain any entries when the writer is created.
    ``status.json`` starts as ``incomplete``.  Only a successful ``finalize``
    call may change it to ``complete``.  It is the authoritative commit record
    and binds the current manifest by SHA-256.  Manifest/status control bytes
    are covered by the static estimate's conservative allowance, not charged
    to ``BudgetTracker.actual_bytes`` (which tracks requested artifacts).
    """

    STATUS_FILENAME = "status.json"
    MANIFEST_FILENAME = "manifest.json"

    def __init__(
        self,
        output_dir: str | Path,
        policy: RetentionPolicy | RetentionProfile | str,
        budget: BudgetTracker | int | None = None,
        *,
        budget_bytes: int | None = None,
    ) -> None:
        if budget is not None and budget_bytes is not None:
            raise ValueError("pass budget or budget_bytes, not both")
        if isinstance(budget, BudgetTracker):
            tracker = budget
        elif budget is None:
            tracker = BudgetTracker(budget_bytes)
        elif isinstance(budget, int) and not isinstance(budget, bool):
            tracker = BudgetTracker(budget)
        else:
            raise TypeError("budget must be a BudgetTracker, integer, or None")

        self.output_dir = Path(output_dir).resolve()
        self.policy = get_retention_policy(policy)
        self.budget = tracker
        self.status_path = self.output_dir / self.STATUS_FILENAME
        self.manifest_path = self.output_dir / self.MANIFEST_FILENAME
        self._status = "incomplete"
        self._failure_reason: str | None = None
        self._terminal_failure = False
        self._artifacts: list[dict[str, Any]] = []
        self._paths: set[str] = set()
        self._lock = threading.RLock()
        created_output = False
        try:
            if self.output_dir.exists():
                if not self.output_dir.is_dir():
                    raise NotADirectoryError(self.output_dir)
                if next(self.output_dir.iterdir(), None) is not None:
                    raise OutputDirectoryNotEmptyError(
                        f"refusing to overwrite non-empty output directory: {self.output_dir}"
                    )
            else:
                self.output_dir.mkdir(parents=True, exist_ok=False)
                created_output = True
            self._write_controls()
        except Exception:
            # If initialization loses a race or cannot publish its control
            # files, remove only an empty directory created by this instance.
            if created_output:
                try:
                    self.output_dir.rmdir()
                except OSError:
                    pass
            raise

    @property
    def status(self) -> str:
        return self._status

    @property
    def complete(self) -> bool:
        return self._status == "complete"

    @property
    def artifact_bytes(self) -> int:
        """Exact serialized bytes of user artifacts (control files excluded)."""

        return self.budget.actual_bytes

    @property
    def actual_bytes(self) -> int:
        """Current regular-file bytes in the output tree, controls included."""

        if not self.output_dir.exists():
            return 0
        return sum(
            path.stat().st_size
            for path in self.output_dir.rglob("*")
            if path.is_file()
        )

    @property
    def total_bytes(self) -> int:
        return self.actual_bytes

    @property
    def manifest(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema_version": 1,
            "profile": self.policy.profile.value,
            "status": self._status,
            "complete": self.complete,
            "completion_authority": self.STATUS_FILENAME,
            "budget": self.budget.to_dict(),
            "artifacts": deepcopy(self._artifacts),
        }
        if self._failure_reason is not None:
            value["failure_reason"] = self._failure_reason
        return value

    def write_npz(
        self,
        kind: ArtifactKind | str,
        name: str,
        arrays: Mapping[str, Any],
        *,
        compressed: bool = True,
        granularity: str = "aggregate",
    ) -> dict[str, Any]:
        """Validate, serialize, budget, and atomically persist an NPZ artifact."""

        with self._lock:
            self._ensure_writable()
            try:
                parsed_kind = self.policy.validate_artifact(
                    kind=kind, name=name, artifact_format="npz", granularity=granularity
                )
                path, relative = self._artifact_path(parsed_kind, name, ".npz")
                prepared = self._prepare_arrays(parsed_kind, arrays)
                payload = _serialize_npz(prepared, compressed=compressed)
                array_metadata = {
                    key: {"dtype": str(value.dtype), "shape": list(value.shape)}
                    for key, value in prepared.items()
                }
                dtype: str | dict[str, str]
                shape: list[int] | dict[str, list[int]]
                if len(array_metadata) == 1:
                    only = next(iter(array_metadata.values()))
                    dtype = only["dtype"]
                    shape = only["shape"]
                else:
                    dtype = {key: value["dtype"] for key, value in array_metadata.items()}
                    shape = {key: value["shape"] for key, value in array_metadata.items()}
                extra: dict[str, Any] = {"arrays": array_metadata}
                if (
                    self.policy.profile is RetentionProfile.AUDIT
                    and parsed_kind is ArtifactKind.METRIC_READY
                ):
                    extra["paper_float16_roundtrip_error"] = {
                        key: paper_float16_roundtrip_error(
                            ArtifactArray(value, ArtifactKind.METRIC_READY)
                        )
                        for key, value in prepared.items()
                    }
                return self._commit_payload(
                    path=path,
                    relative=relative,
                    payload=payload,
                    kind=parsed_kind,
                    artifact_format="npz",
                    dtype=dtype,
                    shape=shape,
                    granularity=granularity,
                    extra=extra,
                )
            except Exception as exc:
                self._mark_failed(exc)
                raise

    def write_json(
        self,
        kind: ArtifactKind | str,
        name: str,
        value: Any,
        *,
        granularity: str = "aggregate",
    ) -> dict[str, Any]:
        """Validate, serialize, budget, and atomically persist a JSON artifact."""

        with self._lock:
            self._ensure_writable()
            try:
                parsed_kind = self.policy.validate_artifact(
                    kind=kind, name=name, artifact_format="json", granularity=granularity
                )
                path, relative = self._artifact_path(parsed_kind, name, ".json")
                tagged_kind = _find_tagged_artifact(value)
                if tagged_kind is not None:
                    raise PolicyViolationError(
                        "tagged numerical artifacts must be stored as NPZ; "
                        f"JSON value contains {tagged_kind.value} data"
                    )
                serializable = _json_compatible(value)
                payload = (
                    json.dumps(serializable, indent=2, sort_keys=True, allow_nan=False) + "\n"
                ).encode("utf-8")
                dtype, shape = _json_dtype_shape(value)
                return self._commit_payload(
                    path=path,
                    relative=relative,
                    payload=payload,
                    kind=parsed_kind,
                    artifact_format="json",
                    dtype=dtype,
                    shape=shape,
                    granularity=granularity,
                    extra={},
                )
            except Exception as exc:
                self._mark_failed(exc)
                raise

    def finalize(self) -> dict[str, Any]:
        """Validate required artifacts and publish the sole complete status."""

        with self._lock:
            if self.complete:
                return self.manifest
            if self._terminal_failure or self.budget.failed:
                raise IncompleteArtifactError(
                    f"cannot finalize failed artifact set: {self._failure_reason or 'budget failed'}"
                )
            try:
                self._validate_completion()
                self._status = "complete"
                self._write_controls()
            except Exception as exc:
                self._status = "incomplete"
                self._mark_failed(exc)
                raise
            return self.manifest

    def abort(self, reason: str) -> dict[str, Any]:
        """Explicitly leave the output incomplete and permanently non-finalizable."""

        with self._lock:
            self._mark_failed(IncompleteArtifactError(str(reason)))
            return self.manifest

    def _ensure_writable(self) -> None:
        if self.complete:
            raise IncompleteArtifactError("artifact set is already complete and immutable")
        if self._terminal_failure or self.budget.failed:
            raise IncompleteArtifactError("artifact writer is in a failed, incomplete state")

    def _artifact_path(
        self, kind: ArtifactKind, name: str, expected_suffix: str
    ) -> tuple[Path, str]:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("artifact name must be a non-empty relative path")
        supplied = PurePath(name)
        if supplied.is_absolute() or ".." in supplied.parts:
            raise ValueError("artifact name must stay within its kind directory")
        if supplied.suffix:
            if supplied.suffix.lower() != expected_suffix:
                raise PolicyViolationError(
                    f"{expected_suffix[1:].upper()} writer cannot create {supplied.suffix} artifacts"
                )
            relative_name = supplied
        else:
            relative_name = PurePath(f"{name}{expected_suffix}")
        relative = str(PurePath(kind.value) / relative_name)
        if relative in self._paths:
            raise FileExistsError(f"artifact path already written: {relative}")
        path = self.output_dir / relative
        if path.exists():
            raise FileExistsError(f"refusing to overwrite artifact: {path}")
        return path, relative

    def _prepare_arrays(
        self, kind: ArtifactKind, arrays: Mapping[str, Any]
    ) -> dict[str, np.ndarray]:
        if not isinstance(arrays, Mapping) or not arrays:
            raise ValueError("arrays must be a non-empty mapping")
        prepared: dict[str, np.ndarray] = {}
        for key, value in arrays.items():
            if not isinstance(key, str) or not key:
                raise ValueError("NPZ array keys must be non-empty strings")
            source_kind = getattr(value, "kind", None)
            if source_kind is not None and ArtifactKind.parse(source_kind) is not kind:
                raise PolicyViolationError(
                    f"cannot write {ArtifactKind.parse(source_kind).value} data as {kind.value}"
                )
            array = np.asarray(_unwrap_array(value))
            if array.dtype.hasobject:
                raise PolicyViolationError("object arrays are not auditable NPZ artifacts")
            if not np.issubdtype(array.dtype, np.number) and not np.issubdtype(array.dtype, np.bool_):
                raise PolicyViolationError(f"NPZ member {key!r} must be numeric")
            if array.size == 0:
                raise PolicyViolationError(f"NPZ member {key!r} must not be empty")

            if kind is ArtifactKind.METRIC_READY and self.policy.metric_ndim is not None:
                if array.ndim != self.policy.metric_ndim:
                    raise PolicyViolationError(
                        f"{self.policy.name} metric-ready maps must be "
                        f"{self.policy.metric_ndim}D, got shape {array.shape}"
                    )
            if self.policy.profile in {RetentionProfile.PAPER, RetentionProfile.AUDIT}:
                if kind is ArtifactKind.METRIC_READY and not np.issubdtype(array.dtype, np.floating):
                    raise PolicyViolationError("metric-ready maps must use a floating dtype")
                if kind in {ArtifactKind.RAW, ArtifactKind.METRIC_READY} and not np.all(np.isfinite(array)):
                    raise PolicyViolationError("retained attribution arrays must contain finite values")

            if kind is ArtifactKind.RAW and self.policy.raw_rgb_required:
                if array.ndim != 3 or array.shape[0] != 3:
                    raise PolicyViolationError(
                        f"audit raw attribution must be RGB [3,H,W], got {array.shape}"
                    )
            target_dtype = {
                ArtifactKind.RAW: self.policy.raw_storage_dtype,
                ArtifactKind.METRIC_READY: self.policy.metric_storage_dtype,
                ArtifactKind.DISPLAY: self.policy.display_storage_dtype,
            }[kind]
            if target_dtype is not None:
                with np.errstate(over="ignore", invalid="ignore"):
                    array = array.astype(target_dtype, copy=False)
            else:
                array = np.array(array, copy=True)
            if (
                kind in {ArtifactKind.RAW, ArtifactKind.METRIC_READY}
                and not np.all(np.isfinite(array))
            ):
                raise PolicyViolationError(
                    f"NPZ member {key!r} is not representable as finite {array.dtype}"
                )
            prepared[key] = np.ascontiguousarray(array)
        return prepared

    def _commit_payload(
        self,
        *,
        path: Path,
        relative: str,
        payload: bytes,
        kind: ArtifactKind,
        artifact_format: str,
        dtype: Any,
        shape: Any,
        granularity: str,
        extra: Mapping[str, Any],
    ) -> dict[str, Any]:
        actual_bytes = len(payload)
        digest = hashlib.sha256(payload).hexdigest()
        self.budget.reserve(actual_bytes, label=relative)
        artifact_published = False
        record_added = False
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            _atomic_create(path, payload)
            artifact_published = True
            on_disk_bytes = path.stat().st_size
            if on_disk_bytes != actual_bytes:
                raise OSError(
                    f"short artifact write for {relative}: {on_disk_bytes} != {actual_bytes}"
                )
            record = {
                "path": relative,
                "name": Path(relative).name,
                "format": artifact_format,
                "kind": kind.value,
                "bytes": on_disk_bytes,
                "actual_bytes": on_disk_bytes,
                "sha256": digest,
                "dtype": dtype,
                "shape": shape,
                "granularity": granularity,
                **deepcopy(dict(extra)),
            }
            self._artifacts.append(record)
            self._paths.add(relative)
            record_added = True
            self._write_controls()
            return deepcopy(record)
        except Exception:
            # A write is committed only after both the artifact and its updated
            # manifest/status are durable.  Roll back a partial publication so
            # callers never observe an unmanifested artifact.
            if record_added:
                self._artifacts.pop()
                self._paths.discard(relative)
            if artifact_published:
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
            self.budget.release(actual_bytes, label=relative)
            raise

    def _validate_completion(self) -> None:
        if not self._artifacts:
            raise IncompleteArtifactError("cannot finalize an empty artifact set")
        npz_kinds = {
            ArtifactKind.parse(record["kind"])
            for record in self._artifacts
            if record["format"] == "npz"
        }
        missing = self.policy.required_npz_kinds - npz_kinds
        if missing:
            names = ", ".join(sorted(kind.value for kind in missing))
            raise IncompleteArtifactError(
                f"{self.policy.name} retention is missing required NPZ kinds: {names}"
            )

    def _mark_failed(self, exc: Exception) -> None:
        if self.complete:
            return
        self._status = "incomplete"
        self._terminal_failure = True
        if self._failure_reason is None:
            self._failure_reason = f"{type(exc).__name__}: {exc}"
        try:
            self._write_controls()
        except Exception:
            # Preserve the original exception.  status.json was written as
            # incomplete at construction, so failure cannot invent complete.
            pass

    def _write_controls(self) -> None:
        manifest_payload = (
            json.dumps(self.manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")
        status_value = {
            "schema_version": 1,
            "profile": self.policy.profile.value,
            "status": self._status,
            "complete": self.complete,
            "manifest_sha256": hashlib.sha256(manifest_payload).hexdigest(),
        }
        if self._failure_reason is not None:
            status_value["failure_reason"] = self._failure_reason
        status_payload = (
            json.dumps(status_value, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")
        # Publish manifest first and status last.  A crash between the two leaves
        # an old/incomplete status or a manifest-hash mismatch, never a valid
        # complete commit marker.
        _atomic_replace(self.manifest_path, manifest_payload)
        _atomic_replace(self.status_path, status_payload)


def _unwrap_array(value: Any) -> Any:
    if isinstance(value, ArtifactArray):
        return value.data
    # Keep this duck-typed so display objects from compatible callers retain
    # their boundary tag without creating an import cycle.
    if hasattr(value, "data") and hasattr(value, "kind"):
        candidate = getattr(value, "data")
        if isinstance(candidate, np.ndarray):
            return candidate
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return value.detach().cpu().numpy()
    return value


def _immutable_array(value: Any) -> np.ndarray:
    """Copy an array onto an immutable bytes buffer.

    ``ndarray.setflags(write=True)`` cannot make a view over a Python ``bytes``
    object writable, unlike a merely write-disabled owning ndarray.
    """

    source = np.ascontiguousarray(np.asarray(_unwrap_array(value)))
    immutable = np.frombuffer(source.tobytes(order="C"), dtype=source.dtype)
    return immutable.reshape(source.shape)


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_deep_freeze(item) for item in value)
    if isinstance(value, np.ndarray):
        return _immutable_array(value)
    return value


def _find_tagged_artifact(value: Any, _seen: set[int] | None = None) -> ArtifactKind | None:
    """Find a tagged numerical value nested in a would-be JSON document."""

    seen = _seen if _seen is not None else set()
    identity = id(value)
    if identity in seen:
        return None
    seen.add(identity)
    kind = getattr(value, "kind", None)
    if kind is not None:
        try:
            return ArtifactKind.parse(kind)
        except ValueError:
            pass
    if is_dataclass(value) and not isinstance(value, type):
        for field in value.__dataclass_fields__:
            found = _find_tagged_artifact(getattr(value, field), seen)
            if found is not None:
                return found
    elif isinstance(value, Mapping):
        for item in value.values():
            found = _find_tagged_artifact(item, seen)
            if found is not None:
                return found
    elif isinstance(value, (tuple, list, set, frozenset)):
        for item in value:
            found = _find_tagged_artifact(item, seen)
            if found is not None:
                return found
    return None


def _metric_array(value: Any) -> np.ndarray:
    source_kind = getattr(value, "kind", None)
    if source_kind is None:
        raise PolicyViolationError(
            "paper quantization requires explicitly tagged metric_ready input"
        )
    if ArtifactKind.parse(source_kind) is not ArtifactKind.METRIC_READY:
        raise PolicyViolationError(
            f"paper quantization requires metric_ready input, got {ArtifactKind.parse(source_kind).value}"
        )
    array = np.asarray(_unwrap_array(value))
    if array.ndim != 2:
        raise PolicyViolationError(f"paper metric maps must be 2D, got shape {array.shape}")
    if not np.issubdtype(array.dtype, np.floating):
        raise PolicyViolationError("paper metric maps must use a floating dtype")
    if not np.all(np.isfinite(array)):
        raise PolicyViolationError("paper metric maps must contain finite values")
    return array


def _serialize_npz(arrays: Mapping[str, np.ndarray], *, compressed: bool) -> bytes:
    buffer = io.BytesIO()
    writer = np.savez_compressed if compressed else np.savez
    writer(buffer, **arrays)
    return buffer.getvalue()


def _atomic_create(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        # The destination was checked before serialization; link gives us an
        # atomic no-overwrite publication even if another process races us.
        os.link(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _atomic_replace(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _json_compatible(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _json_compatible(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_compatible(item) for item in value]
    return value


def _json_dtype_shape(value: Any) -> tuple[str, list[int] | None]:
    if isinstance(value, (np.ndarray, np.generic, list, tuple)):
        array = np.asarray(value)
        if not array.dtype.hasobject:
            return str(array.dtype), list(array.shape)
    return "json", None


def _validate_byte_count(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("byte_count must be a non-negative integer")
    return value


def _validate_nonnegative_finite_tolerance(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite non-negative number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return result


_GRANULAR_FILENAME = re.compile(
    r"(?:^|[/_.-])(?:per[/_.-]?)?(?:pixel|dimension|dim)[_-]?\d+(?:$|[/_.-])",
    re.IGNORECASE,
)


def _looks_like_granular_filename(name: str) -> bool:
    return _GRANULAR_FILENAME.search(str(name)) is not None


__all__ = [
    "ArtifactArray",
    "ArtifactKind",
    "ArtifactWriter",
    "AUDIT_POLICY",
    "BudgetExceededError",
    "BudgetTracker",
    "DEBUG_POLICY",
    "IncompleteArtifactError",
    "OutputDirectoryNotEmptyError",
    "PAPER_POLICY",
    "PolicyViolationError",
    "RetentionError",
    "RetentionPolicy",
    "RetentionProfile",
    "array_sha256",
    "audit_replay_report",
    "audit_paper_roundtrip_error",
    "estimate_output_bytes",
    "float16_roundtrip_error",
    "get_retention_policy",
    "paper_float16_roundtrip_error",
    "quantize_paper_metric",
]
