"""Closed-layout, immutable Stage D execution-attempt artifacts.

The status document is the sole commit marker.  A terminal transition publishes
payloads and provenance first, the manifest second, and status last.  A crash
at any earlier boundary leaves a retained, non-resumable attempt.

The module accepts in-memory numeric payloads only.  It contains no image, model,
state-dict, or scientific-outcome access.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import stat
import uuid
import zipfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any

import numpy as np

from .stage_d_plan import StageDImageGroup, StageDUnit
from .stage_d_provenance import (
    DOCUMENT_TYPE as PROVENANCE_DOCUMENT_TYPE,
    FAILURE_DOCUMENT_TYPE,
    FAILURE_PATH,
    SCIENTIFIC_FAILURE_STATUS,
    SUCCESS_STATUS,
    StageDProvenance,
    StageDProvenanceError,
    canonical_json_bytes,
    canonical_json_file_bytes,
    canonical_sha256,
    normalize_stage_d_scientific_failure,
    stage_d_scientific_failure_artifact,
    validate_stage_d_provenance,
)


SCHEMA_VERSION = 1
MANIFEST_FILENAME = "manifest.json"
STATUS_FILENAME = "status.json"
RESULT_PATH = "L0/result.json"
PROVENANCE_PATH = "L0/provenance.json"
MAP_PATH = "L1/metric_ready_map.npy"
CURVES_PATH = "L1/fidelity_curves.npz"
AUDIT_PATH = "L2/audit_evidence.npz"
MANIFEST_DOCUMENT_TYPE = "stage_d_execution_unit_manifest"
STATUS_DOCUMENT_TYPE = "stage_d_execution_unit_status"
RUNNING_STATUS = "running"
INFRASTRUCTURE_INTERRUPTION_STATUS = "infrastructure_interruption"

FAULT_AFTER_RUNNING = "after_running"
FAULT_AFTER_PAYLOAD = "after_payload"
FAULT_AFTER_MANIFEST = "after_manifest"
FAULT_BEFORE_TERMINAL_STATUS = "before_terminal_status"
# Compatibility spelling for callers familiar with the Stage C API.
FAULT_BEFORE_COMPLETE_STATUS = FAULT_BEFORE_TERMINAL_STATUS
FAULT_POINTS = frozenset({
    FAULT_AFTER_RUNNING, FAULT_AFTER_PAYLOAD, FAULT_AFTER_MANIFEST,
    FAULT_BEFORE_TERMINAL_STATUS,
})

_ATTEMPT_RE = re.compile(r"^attempt-(\d{6})$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ARRAY_KEY_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]*$")
AUDIT_REPEAT_MAP_KEY = "repeat_clean_metric_ready_map"
_EXPECTED_NON_AUDIT = {
    STATUS_FILENAME, MANIFEST_FILENAME, RESULT_PATH, PROVENANCE_PATH, MAP_PATH,
    CURVES_PATH,
}
_EXPECTED_AUDIT = _EXPECTED_NON_AUDIT | {AUDIT_PATH}
_PAYLOAD_PATHS = (RESULT_PATH, MAP_PATH, CURVES_PATH)
_SCIENTIFIC_PATHS = frozenset({RESULT_PATH, MAP_PATH, CURVES_PATH, AUDIT_PATH})
_ISSUER = object()
RETRYABLE_INFRASTRUCTURE_CODES = frozenset({
    "SCHEDULER_PREEMPTION", "EXTERNAL_PROCESS_TERMINATION", "HOST_LOSS",
    "GPU_DEVICE_LOSS_OR_RESET", "TRANSIENT_STORAGE_IO_BEFORE_RESULT_PUBLISH",
    "WORKER_PROCESS_CRASH_BEFORE_RESULT_PUBLISH",
    "STALE_CLAIM_OWNER_CONFIRMED_DEAD",
})
RETENTION_CONTRACT = {
    "closed_layout": True, "l1_metric_map_dtype": "float32",
    "l1_metric_map_shape": [56, 56], "l1_curves_dtype": "float64",
    "l1_curves_shape": [2, 5, 21], "l2_audit_only": True,
    "l3_allowed": False, "png_allowed": False,
    "forbidden_granularities": ["per_step", "per_pixel", "per_dimension"],
}


class StageDArtifactError(RuntimeError):
    """Base error for Stage D artifact persistence."""


class StageDArtifactPolicyError(StageDArtifactError, ValueError):
    """Raised for a path, type, layout, or retention violation."""


class StageDAttemptNotEmptyError(StageDArtifactError, FileExistsError):
    """Raised when a writer is pointed at an existing non-empty attempt."""


class StageDArtifactWriterStateError(StageDArtifactError):
    """Raised when an operation would mutate a sealed/crashed attempt."""


class InjectedStageDArtifactCrash(StageDArtifactError):
    """Named synthetic crash used to test every commit boundary."""

    def __init__(self, point: str):
        self.point = point
        super().__init__(f"injected Stage D artifact crash at {point}")


# Short aliases keep migration from the Stage C persistence API straightforward
# without sharing any Stage C identity or document type.
ArtifactPolicyError = StageDArtifactPolicyError
AttemptNotEmptyError = StageDAttemptNotEmptyError
ArtifactWriterStateError = StageDArtifactWriterStateError
InjectedArtifactCrash = InjectedStageDArtifactCrash


def _plain(value: Any, context: str = "value") -> Any:
    converter = getattr(value, "to_dict", None)
    if callable(converter) and not isinstance(value, Mapping):
        value = converter()
    if value is None or type(value) in {str, bool, int}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise StageDArtifactPolicyError(
                f"{context} contains a non-finite number"
            )
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, child in value.items():
            if not isinstance(key, str) or not key:
                raise StageDArtifactPolicyError(
                    f"{context} keys must be non-empty strings"
                )
            result[key] = _plain(child, f"{context}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [_plain(child, f"{context}[{index}]") for index, child in enumerate(value)]
    raise StageDArtifactPolicyError(
        f"{context} is not JSON-compatible: {type(value).__name__}"
    )


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(child) for key, child in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(child) for child in value)
    if isinstance(value, tuple):
        return tuple(_freeze(child) for child in value)
    return value


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _unit(value: Any) -> StageDUnit:
    if isinstance(value, StageDUnit):
        return value
    document = _plain(value, "unit")
    try:
        return StageDUnit.from_dict(document)
    except Exception as error:
        raise StageDArtifactPolicyError(f"invalid Stage D unit: {error}") from error


def _group(value: Any) -> StageDImageGroup:
    if isinstance(value, StageDImageGroup):
        return value
    document = _plain(value, "group")
    try:
        return StageDImageGroup.from_dict(document)
    except Exception as error:
        raise StageDArtifactPolicyError(f"invalid Stage D group: {error}") from error


def _attempt_number_from_path(path: Path) -> int:
    match = _ATTEMPT_RE.fullmatch(path.name)
    if match is None:
        raise StageDArtifactPolicyError(
            "attempt directory must be named attempt-NNNNNN"
        )
    number = int(match.group(1))
    if number <= 0:
        raise StageDArtifactPolicyError("attempt number must be positive")
    return number


def _absolute_path(value: str | Path) -> Path:
    raw = os.fspath(value)
    if not isinstance(raw, str) or not raw or "\x00" in raw:
        raise StageDArtifactPolicyError("path must be non-empty text")
    return Path(os.path.abspath(raw))


def _assert_directory_chain(path: Path, *, create: bool) -> None:
    if create:
        path.mkdir(parents=True, exist_ok=True)
    current = path
    while True:
        try:
            info = current.lstat()
        except FileNotFoundError:
            raise StageDArtifactPolicyError(f"directory does not exist: {current}")
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise StageDArtifactPolicyError(
                f"path contains a symlink or non-directory: {current}"
            )
        if current.parent == current:
            break
        current = current.parent


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_create(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        _fsync_directory(path.parent)
    finally:
        try:
            temporary.unlink()
            _fsync_directory(path.parent)
        except FileNotFoundError:
            pass


def _atomic_replace(path: Path, payload: bytes) -> None:
    if path.is_symlink():
        raise StageDArtifactPolicyError(f"refusing to replace symlink: {path}")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _ensure_payload_parent(root: Path, path: Path) -> None:
    if path.parent.parent != root or path.parent.name not in {"L0", "L1", "L2"}:
        raise StageDArtifactPolicyError("payload path is outside the closed layout")
    if path.parent.exists():
        info = path.parent.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise StageDArtifactPolicyError("payload parent is unsafe")
    else:
        path.parent.mkdir(mode=0o755, exist_ok=False)
        _fsync_directory(root)


def next_attempt_directory(unit_root: str | Path) -> Path:
    """Atomically create the next unique ``attempt-NNNNNN`` directory."""

    root = _absolute_path(unit_root)
    _assert_directory_chain(root, create=True)
    while True:
        highest = 0
        with os.scandir(root) as entries:
            for entry in entries:
                if entry.is_symlink():
                    raise StageDArtifactPolicyError(
                        f"attempt root contains a forbidden symlink: {entry.name}"
                    )
                match = _ATTEMPT_RE.fullmatch(entry.name)
                if match is None:
                    continue
                if not entry.is_dir(follow_symlinks=False):
                    raise StageDArtifactPolicyError(
                        f"attempt path is not a directory: {entry.name}"
                    )
                highest = max(highest, int(match.group(1)))
        if highest >= 999999:
            raise StageDArtifactError("attempt sequence exhausted")
        candidate = root / f"attempt-{highest + 1:06d}"
        try:
            candidate.mkdir(mode=0o755, exist_ok=False)
        except FileExistsError:
            continue
        _fsync_directory(root)
        return candidate


def attempt_directories(unit_root: str | Path) -> tuple[Path, ...]:
    """Return existing validly named attempt directories in numeric order."""

    root = _absolute_path(unit_root)
    if not root.exists():
        return ()
    _assert_directory_chain(root, create=False)
    result: list[tuple[int, Path]] = []
    with os.scandir(root) as entries:
        for entry in entries:
            if entry.is_symlink():
                raise StageDArtifactPolicyError(
                    f"attempt root contains a forbidden symlink: {entry.name}"
                )
            match = _ATTEMPT_RE.fullmatch(entry.name)
            if match is None:
                continue
            if not entry.is_dir(follow_symlinks=False):
                raise StageDArtifactPolicyError(
                    f"attempt path is not a directory: {entry.name}"
                )
            result.append((int(match.group(1)), Path(entry.path)))
    return tuple(path for _number, path in sorted(result))


def _finite_array(
    value: Any, *, dtype: np.dtype[Any], shape: tuple[int, ...], context: str,
) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != dtype:
        raise StageDArtifactPolicyError(
            f"{context} must already be {dtype.name}, got {array.dtype}"
        )
    if array.shape != shape:
        raise StageDArtifactPolicyError(
            f"{context} must have shape {list(shape)}, got {list(array.shape)}"
        )
    if not np.all(np.isfinite(array)):
        raise StageDArtifactPolicyError(f"{context} must contain only finite values")
    return np.ascontiguousarray(array)


def _audit_arrays(value: Any) -> dict[str, np.ndarray]:
    if not isinstance(value, Mapping) or not value:
        raise StageDArtifactPolicyError(
            "audit_evidence must be a non-empty mapping of arrays"
        )
    arrays: dict[str, np.ndarray] = {}
    for key, child in value.items():
        if not isinstance(key, str) or _SAFE_ARRAY_KEY_RE.fullmatch(key) is None:
            raise StageDArtifactPolicyError("audit evidence key is unsafe")
        array = np.asarray(child)
        if array.dtype.kind not in {"b", "i", "u", "f"} or array.dtype.hasobject:
            raise StageDArtifactPolicyError(
                "audit evidence arrays must be numeric or boolean"
            )
        if array.size == 0:
            raise StageDArtifactPolicyError("audit evidence arrays must be non-empty")
        if array.dtype.kind == "f" and not np.all(np.isfinite(array)):
            raise StageDArtifactPolicyError(
                "audit evidence arrays must contain only finite values"
            )
        arrays[key] = np.ascontiguousarray(array)
    repeat_map = arrays.get(AUDIT_REPEAT_MAP_KEY)
    if repeat_map is None:
        raise StageDArtifactPolicyError(
            f"audit evidence requires {AUDIT_REPEAT_MAP_KEY!r}"
        )
    if repeat_map.dtype != np.dtype(np.float32) or repeat_map.shape != (56, 56):
        raise StageDArtifactPolicyError(
            "audit repeat clean metric-ready map must be float32 shape [56, 56]"
        )
    return dict(sorted(arrays.items()))


def _serialize_npy(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return buffer.getvalue()


def _serialize_npz(arrays: Mapping[str, np.ndarray]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for key, array in sorted(arrays.items()):
            info = zipfile.ZipInfo(f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, _serialize_npy(array))
    return buffer.getvalue()


def _strict_json_loads(raw: bytes) -> Any:
    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, child in values:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key!r}")
            result[key] = child
        return result

    def constant(token: str) -> Any:
        raise ValueError(f"non-finite JSON number: {token}")

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("JSON is not UTF-8") from error
    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)


def _load_canonical_json(raw: bytes, context: str) -> dict[str, Any]:
    value = _strict_json_loads(raw)
    if not isinstance(value, dict):
        raise StageDArtifactPolicyError(f"{context} must be a JSON object")
    if raw != canonical_json_file_bytes(value):
        raise StageDArtifactPolicyError(f"{context} is not canonical JSON")
    return value


def _load_npy(raw: bytes) -> np.ndarray:
    value = np.load(io.BytesIO(raw), allow_pickle=False)
    if not isinstance(value, np.ndarray):
        raise StageDArtifactPolicyError("NPY did not decode to one array")
    return value


def _load_npz(raw: bytes) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        if len(archive.files) != len(set(archive.files)):
            raise StageDArtifactPolicyError("NPZ contains duplicate members")
        for key in sorted(archive.files):
            if _SAFE_ARRAY_KEY_RE.fullmatch(key) is None:
                raise StageDArtifactPolicyError("NPZ contains an unsafe member name")
            result[key] = np.asarray(archive[key])
    return result


def _record(
    *, path: str, raw: bytes, kind: str, artifact_format: str,
    dtype: Any, shape: Any,
) -> dict[str, Any]:
    return {
        "level": path.split("/", 1)[0], "path": path, "kind": kind,
        "format": artifact_format, "dtype": _plain(dtype),
        "shape": _plain(shape), "bytes": len(raw), "sha256": _sha(raw),
    }


@dataclass(frozen=True, slots=True)
class StageDUnitPayload:
    """The complete scientific payload for exactly one Stage D unit."""

    l0_scalars: Mapping[str, Any]
    metric_ready_map: np.ndarray = field(repr=False)
    fidelity_curves: np.ndarray = field(repr=False)
    audit_evidence: Mapping[str, np.ndarray] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        scalars = _plain(self.l0_scalars, "l0_scalars")
        if not isinstance(scalars, dict) or not scalars:
            raise StageDArtifactPolicyError("l0_scalars must be a non-empty object")
        metric_map = _finite_array(
            self.metric_ready_map, dtype=np.dtype(np.float32), shape=(56, 56),
            context="metric_ready_map",
        )
        curves = _finite_array(
            self.fidelity_curves, dtype=np.dtype(np.float64), shape=(2, 5, 21),
            context="fidelity_curves",
        )
        audit = None if self.audit_evidence is None else _audit_arrays(self.audit_evidence)
        object.__setattr__(self, "l0_scalars", _freeze(scalars))
        object.__setattr__(self, "metric_ready_map", metric_map.copy())
        object.__setattr__(self, "fidelity_curves", curves.copy())
        object.__setattr__(
            self, "audit_evidence",
            None if audit is None else MappingProxyType({
                key: array.copy() for key, array in audit.items()
            }),
        )

    @property
    def audit(self) -> bool:
        return self.audit_evidence is not None


@dataclass(frozen=True, slots=True)
class StageDAttemptValidation(Mapping[str, Any]):
    """Source-recomputed verdict for one Stage D attempt directory."""

    path: Path
    valid: bool
    terminal: bool
    complete: bool
    resumable: bool
    status: str | None
    errors: tuple[str, ...]
    digest: str | None
    manifest_sha256: str | None
    unit_digest: str | None
    artifact_count: int
    total_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path), "valid": self.valid,
            "terminal": self.terminal, "complete": self.complete,
            "resumable": self.resumable, "status": self.status,
            "errors": list(self.errors), "digest": self.digest,
            "manifest_sha256": self.manifest_sha256,
            "unit_digest": self.unit_digest,
            "artifact_count": self.artifact_count,
            "total_bytes": self.total_bytes,
        }

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())

    def require_resumable(self) -> None:
        if not self.resumable:
            reason = "; ".join(self.errors) or f"status is {self.status!r}"
            raise StageDArtifactError(f"attempt is not resumable: {reason}")


class StageDArtifactWriter:
    """Create exactly one new, closed-layout Stage D attempt."""

    def __init__(
        self, attempt_dir: str | Path, *, unit: StageDUnit | Mapping[str, Any],
        group: StageDImageGroup | Mapping[str, Any],
        fault_injector: Callable[[str], Any] | str | Sequence[str] | None = None,
    ) -> None:
        self.attempt_dir = _absolute_path(attempt_dir)
        self.output_dir = self.attempt_dir
        self.manifest_path = self.attempt_dir / MANIFEST_FILENAME
        self.status_path = self.attempt_dir / STATUS_FILENAME
        self.provenance_path = self.attempt_dir / PROVENANCE_PATH
        self.unit = _unit(unit)
        self.group = _group(group)
        if self.unit.canonical_digest not in {
            candidate.canonical_digest for candidate in self.group.units
        }:
            raise StageDArtifactPolicyError("unit is not a member of group")
        self.attempt_number = _attempt_number_from_path(self.attempt_dir)
        if self.attempt_number > 2:
            raise StageDArtifactPolicyError(
                "Stage D permits at most two attempts per unit"
            )
        self._fault_injector = fault_injector
        self._crashed = False
        self._terminal = False
        self._status = RUNNING_STATUS
        self._payload_records: list[dict[str, Any]] = []
        self._payload: StageDUnitPayload | None = None
        self._claim_empty_attempt()
        running_manifest = self._manifest(
            status=RUNNING_STATUS, terminal=False, complete=False, failure=None,
            provenance_sha256=None, artifacts=[],
        )
        manifest_raw = canonical_json_file_bytes(running_manifest)
        running_status = self._status_document(
            status=RUNNING_STATUS, terminal=False, complete=False,
            manifest_raw=manifest_raw, failure=None,
        )
        _atomic_create(self.manifest_path, manifest_raw)
        _atomic_create(self.status_path, canonical_json_file_bytes(running_status))
        self._maybe_fault(FAULT_AFTER_RUNNING)

    @property
    def status(self) -> str:
        return self._status

    @property
    def terminal(self) -> bool:
        return self._terminal

    @property
    def complete(self) -> bool:
        return self._terminal

    @property
    def artifacts(self) -> tuple[dict[str, Any], ...]:
        return tuple(_plain(item) for item in self._payload_records)

    def _claim_empty_attempt(self) -> None:
        path = self.attempt_dir
        if path.exists():
            _assert_directory_chain(path, create=False)
            if next(path.iterdir(), None) is not None:
                raise StageDAttemptNotEmptyError(
                    f"refusing to overwrite or append attempt: {path}"
                )
        else:
            _assert_directory_chain(path.parent, create=True)
            try:
                path.mkdir(mode=0o755, exist_ok=False)
            except FileExistsError as error:
                raise StageDAttemptNotEmptyError(
                    f"attempt was claimed concurrently: {path}"
                ) from error
            _fsync_directory(path.parent)
        _assert_directory_chain(path, create=False)

    def _ensure_running(self) -> None:
        if self._crashed:
            raise StageDArtifactWriterStateError(
                "crashed attempt is immutable; allocate a fresh attempt"
            )
        if self._terminal:
            raise StageDArtifactWriterStateError(
                "terminal attempt is immutable; allocate a fresh attempt"
            )

    def write_payload(self, payload: StageDUnitPayload) -> tuple[dict[str, Any], ...]:
        """Publish every required payload exactly once in closed order."""

        self._ensure_running()
        if self._payload is not None:
            raise StageDArtifactWriterStateError("payload is already immutable")
        if not isinstance(payload, StageDUnitPayload):
            raise TypeError("payload must be StageDUnitPayload")
        if payload.audit != self.unit.audit:
            raise StageDArtifactPolicyError(
                "audit payload presence must exactly match the typed unit"
            )
        values: list[tuple[str, bytes, str, str, Any, Any]] = []
        result_raw = canonical_json_file_bytes(_plain(payload.l0_scalars))
        values.append((RESULT_PATH, result_raw, "result", "json", "json", None))
        map_raw = _serialize_npy(payload.metric_ready_map)
        values.append((
            MAP_PATH, map_raw, "metric_ready_map", "npy", "float32",
            [56, 56],
        ))
        curves_raw = _serialize_npz({"curves": payload.fidelity_curves})
        values.append((
            CURVES_PATH, curves_raw, "fidelity_curves", "npz", "float64",
            [2, 5, 21],
        ))
        if payload.audit_evidence is not None:
            arrays = dict(payload.audit_evidence)
            audit_raw = _serialize_npz(arrays)
            values.append((
                AUDIT_PATH, audit_raw, "audit_evidence", "npz",
                {key: array.dtype.name for key, array in sorted(arrays.items())},
                {key: list(array.shape) for key, array in sorted(arrays.items())},
            ))
        for path, raw, kind, artifact_format, dtype, shape in values:
            destination = self.attempt_dir.joinpath(*PurePosixPath(path).parts)
            _ensure_payload_parent(self.attempt_dir, destination)
            if destination.exists() or destination.is_symlink():
                raise StageDArtifactPolicyError(
                    f"refusing to overwrite payload: {path}"
                )
            _atomic_create(destination, raw)
            record = _record(
                path=path, raw=raw, kind=kind, artifact_format=artifact_format,
                dtype=dtype, shape=shape,
            )
            self._payload_records.append(record)
            self._maybe_fault(FAULT_AFTER_PAYLOAD)
        self._payload = payload
        return self.artifacts

    def write_l0_scalars(self, l0_scalars: Mapping[str, Any]) -> dict[str, Any]:
        """Publish the success-result L0 independently for staged runners."""

        raw = canonical_json_file_bytes(_plain(l0_scalars, "l0_scalars"))
        return self._write_scientific_artifact(
            RESULT_PATH, raw, kind="result", artifact_format="json",
            dtype="json", shape=None,
        )

    def write_metric_ready_map(self, value: Any) -> dict[str, Any]:
        array = _finite_array(
            value, dtype=np.dtype(np.float32), shape=(56, 56),
            context="metric_ready_map",
        )
        return self._write_scientific_artifact(
            MAP_PATH, _serialize_npy(array), kind="metric_ready_map",
            artifact_format="npy", dtype="float32", shape=[56, 56],
        )

    def write_fidelity_curves(self, value: Any) -> dict[str, Any]:
        array = _finite_array(
            value, dtype=np.dtype(np.float64), shape=(2, 5, 21),
            context="fidelity_curves",
        )
        return self._write_scientific_artifact(
            CURVES_PATH, _serialize_npz({"curves": array}),
            kind="fidelity_curves", artifact_format="npz",
            dtype="float64", shape=[2, 5, 21],
        )

    def write_audit_evidence(self, value: Mapping[str, Any]) -> dict[str, Any]:
        if not self.unit.audit:
            raise StageDArtifactPolicyError("L2 evidence is audit-only")
        arrays = _audit_arrays(value)
        return self._write_scientific_artifact(
            AUDIT_PATH, _serialize_npz(arrays), kind="audit_evidence",
            artifact_format="npz",
            dtype={key: array.dtype.name for key, array in arrays.items()},
            shape={key: list(array.shape) for key, array in arrays.items()},
        )

    def _write_scientific_artifact(
        self, path: str, raw: bytes, *, kind: str, artifact_format: str,
        dtype: Any, shape: Any,
    ) -> dict[str, Any]:
        self._ensure_running()
        if path not in _SCIENTIFIC_PATHS:
            raise StageDArtifactPolicyError("scientific artifact path is forbidden")
        if any(item["path"] == path for item in self._payload_records):
            raise StageDArtifactWriterStateError(f"artifact is already immutable: {path}")
        destination = self.attempt_dir.joinpath(*PurePosixPath(path).parts)
        _ensure_payload_parent(self.attempt_dir, destination)
        if destination.exists() or destination.is_symlink():
            raise StageDArtifactPolicyError(f"refusing to overwrite payload: {path}")
        _atomic_create(destination, raw)
        record = _record(
            path=path, raw=raw, kind=kind, artifact_format=artifact_format,
            dtype=dtype, shape=shape,
        )
        self._payload_records.append(record)
        self._payload_records.sort(
            key=lambda item: [RESULT_PATH, MAP_PATH, CURVES_PATH, AUDIT_PATH].index(
                item["path"]
            )
        )
        self._maybe_fault(FAULT_AFTER_PAYLOAD)
        return _plain(record)

    def finalize(
        self, *, provenance: StageDProvenance | Mapping[str, Any],
        scientific_failure: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Publish success, or delegate a full-payload scientific failure."""

        self._ensure_running()
        if self._payload is None:
            raise StageDArtifactWriterStateError(
                "complete terminal attempt requires the full StageDUnitPayload"
            )
        if scientific_failure is not None:
            return self.fail_scientifically_partial(
                failure=scientific_failure, provenance=provenance,
                l0_scalars=_plain(self._payload.l0_scalars),
            )
        terminal_status = SUCCESS_STATUS
        failure = None
        terminal = {
            "status": terminal_status, "terminal": True, "complete": True,
        }
        try:
            validated = validate_stage_d_provenance(
                provenance, expected_unit=self.unit, expected_group=self.group,
                expected_attempt_number=self.attempt_number,
                expected_terminal=terminal, expected_failure=failure,
                expected_inventory=self._payload_records,
            )
        except StageDProvenanceError as error:
            raise StageDArtifactPolicyError(
                f"invalid Stage D provenance: {error}"
            ) from error
        provenance_raw = canonical_json_file_bytes(validated.to_dict())
        _ensure_payload_parent(self.attempt_dir, self.provenance_path)
        _atomic_create(self.provenance_path, provenance_raw)
        provenance_record = _record(
            path=PROVENANCE_PATH, raw=provenance_raw, kind="provenance",
            artifact_format="json", dtype="json", shape=None,
        )
        all_records = [*self._payload_records, provenance_record]
        manifest = self._manifest(
            status=terminal_status, terminal=True, complete=True, failure=failure,
            provenance_sha256=provenance_record["sha256"],
            artifacts=all_records,
        )
        manifest_raw = canonical_json_file_bytes(manifest)
        status = self._status_document(
            status=terminal_status, terminal=True, complete=True,
            manifest_raw=manifest_raw, failure=failure,
        )
        self._validate_pending_tree(allow_running_controls=True)
        _atomic_replace(self.manifest_path, manifest_raw)
        self._maybe_fault(FAULT_AFTER_MANIFEST)
        self._maybe_fault(FAULT_BEFORE_TERMINAL_STATUS)
        _atomic_replace(self.status_path, canonical_json_file_bytes(status))
        self._status = terminal_status
        self._terminal = True
        return _plain(manifest)

    def succeed(self, *, provenance: StageDProvenance | Mapping[str, Any]) -> dict[str, Any]:
        return self.finalize(provenance=provenance)

    def fail_scientifically(
        self, *, failure: Mapping[str, Any],
        provenance: StageDProvenance | Mapping[str, Any],
    ) -> dict[str, Any]:
        return self.finalize(provenance=provenance, scientific_failure=failure)

    def fail_scientifically_partial(
        self, *, failure: Mapping[str, Any],
        provenance: StageDProvenance | Mapping[str, Any],
        l0_scalars: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Seal a phase-specific scientific failure without inventing payload."""

        self._ensure_running()
        produced = [item["path"] for item in self._payload_records]
        normalized, expected_record, failure_raw = stage_d_scientific_failure_artifact(
            failure, l0_scalars=l0_scalars, audit=self.unit.audit, produced=produced,
        )
        prospective_records = [*self._payload_records, expected_record]
        terminal = {
            "status": SCIENTIFIC_FAILURE_STATUS, "terminal": True,
            "complete": True,
        }
        try:
            validated = validate_stage_d_provenance(
                provenance, expected_unit=self.unit, expected_group=self.group,
                expected_attempt_number=self.attempt_number,
                expected_terminal=terminal, expected_failure=normalized,
                expected_inventory=prospective_records,
            )
        except StageDProvenanceError as error:
            raise StageDArtifactPolicyError(
                f"invalid Stage D provenance: {error}"
            ) from error
        destination = self.attempt_dir / FAILURE_PATH
        _ensure_payload_parent(self.attempt_dir, destination)
        _atomic_create(destination, failure_raw)
        self._payload_records.append(expected_record)
        self._maybe_fault(FAULT_AFTER_PAYLOAD)
        provenance_raw = canonical_json_file_bytes(validated.to_dict())
        _ensure_payload_parent(self.attempt_dir, self.provenance_path)
        _atomic_create(self.provenance_path, provenance_raw)
        provenance_record = _record(
            path=PROVENANCE_PATH, raw=provenance_raw, kind="provenance",
            artifact_format="json", dtype="json", shape=None,
        )
        manifest = self._manifest(
            status=SCIENTIFIC_FAILURE_STATUS, terminal=True, complete=True,
            failure=normalized, provenance_sha256=provenance_record["sha256"],
            artifacts=[*self._payload_records, provenance_record],
        )
        manifest_raw = canonical_json_file_bytes(manifest)
        status = self._status_document(
            status=SCIENTIFIC_FAILURE_STATUS, terminal=True, complete=True,
            manifest_raw=manifest_raw, failure=normalized,
        )
        self._validate_pending_tree(allow_running_controls=True)
        _atomic_replace(self.manifest_path, manifest_raw)
        self._maybe_fault(FAULT_AFTER_MANIFEST)
        self._maybe_fault(FAULT_BEFORE_TERMINAL_STATUS)
        _atomic_replace(self.status_path, canonical_json_file_bytes(status))
        self._status = SCIENTIFIC_FAILURE_STATUS
        self._terminal = True
        return _plain(manifest)

    def mark_infrastructure_interruption(
        self, *, code: str, resources: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Leave a durable nonterminal marker; never publish result status."""

        self._ensure_running()
        if self._payload_records:
            raise StageDArtifactWriterStateError(
                "infrastructure retry is forbidden after scientific payload publication"
            )
        if code not in RETRYABLE_INFRASTRUCTURE_CODES:
            raise ValueError("infrastructure interruption code is not retryable")
        resource_value = _plain(resources)
        if not isinstance(resource_value, dict):
            raise TypeError("infrastructure resources must be an object")
        interruption = {"code": code, "resources": resource_value}
        manifest = self._manifest(
            status=INFRASTRUCTURE_INTERRUPTION_STATUS, terminal=False,
            complete=False, failure=interruption, provenance_sha256=None,
            artifacts=[],
        )
        manifest_raw = canonical_json_file_bytes(manifest)
        status = self._status_document(
            status=INFRASTRUCTURE_INTERRUPTION_STATUS, terminal=False,
            complete=False, manifest_raw=manifest_raw, failure=interruption,
        )
        _atomic_replace(self.manifest_path, manifest_raw)
        _atomic_replace(self.status_path, canonical_json_file_bytes(status))
        self._status = INFRASTRUCTURE_INTERRUPTION_STATUS
        self._crashed = True
        return _plain(status)

    # A concise alias used by orchestration code.
    interrupt = mark_infrastructure_interruption

    def _manifest(
        self, *, status: str, terminal: bool, complete: bool,
        failure: Mapping[str, Any] | None, provenance_sha256: str | None,
        artifacts: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "document_type": MANIFEST_DOCUMENT_TYPE,
            "status": status, "terminal": terminal, "complete": complete,
            "completion_authority": STATUS_FILENAME,
            "attempt_number": self.attempt_number,
            "unit_identity": self.unit.to_dict(),
            "unit_digest": self.unit.canonical_digest,
            "group_digest": self.group.canonical_digest,
            "pair_digest": self.group.pair_binding.canonical_digest,
            "audit": self.unit.audit,
            "provenance_path": PROVENANCE_PATH,
            "provenance_sha256": provenance_sha256,
            "retention": _plain(RETENTION_CONTRACT),
            "artifacts": [_plain(item) for item in artifacts],
        }
        if failure is not None:
            value["failure"] = _plain(failure)
        return value

    def _status_document(
        self, *, status: str, terminal: bool, complete: bool,
        manifest_raw: bytes, failure: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "document_type": STATUS_DOCUMENT_TYPE,
            "status": status, "terminal": terminal, "complete": complete,
            "attempt_number": self.attempt_number,
            "unit_digest": self.unit.canonical_digest,
            "manifest_sha256": _sha(manifest_raw),
        }
        if failure is not None:
            value["failure"] = _plain(failure)
        return value

    def _validate_pending_tree(self, *, allow_running_controls: bool) -> None:
        files, problems = _inventory_regular_files(self.attempt_dir)
        if problems:
            raise StageDArtifactPolicyError("; ".join(problems))
        expected = {STATUS_FILENAME, MANIFEST_FILENAME, PROVENANCE_PATH, *(
            item["path"] for item in self._payload_records
        )}
        if set(files) != expected:
            raise StageDArtifactPolicyError(
                f"attempt tree differs from closed pending layout: {sorted(files)}"
            )
        for item in self._payload_records:
            raw = files[item["path"]][0].read_bytes()
            if len(raw) != item["bytes"] or _sha(raw) != item["sha256"]:
                raise StageDArtifactPolicyError(
                    f"payload changed before terminal commit: {item['path']}"
                )

    def _maybe_fault(self, point: str) -> None:
        injector = self._fault_injector
        if injector is None:
            return
        if callable(injector):
            triggered = bool(injector(point))
        elif isinstance(injector, str):
            if injector not in FAULT_POINTS:
                raise ValueError(f"unknown fault point: {injector!r}")
            triggered = injector == point
        else:
            configured = set(injector)
            if configured - FAULT_POINTS:
                raise ValueError(
                    f"unknown fault points: {sorted(configured - FAULT_POINTS)}"
                )
            triggered = point in configured
        if triggered:
            self._crashed = True
            raise InjectedStageDArtifactCrash(point)


def _inventory_regular_files(
    root: Path,
) -> tuple[dict[str, tuple[Path, int]], list[str]]:
    files: dict[str, tuple[Path, int]] = {}
    problems: list[str] = []

    def visit(directory: Path, prefix: PurePosixPath) -> None:
        try:
            entries = sorted(os.scandir(directory), key=lambda item: item.name)
        except OSError as error:
            problems.append(f"cannot scan {directory}: {error}")
            return
        for entry in entries:
            relative = str(prefix / entry.name)
            try:
                if entry.is_symlink():
                    problems.append(f"forbidden symlink: {relative}")
                elif entry.is_dir(follow_symlinks=False):
                    child_prefix = prefix / entry.name
                    if len(child_prefix.parts) != 1 or entry.name not in {
                        "L0", "L1", "L2"
                    }:
                        problems.append(f"forbidden directory: {relative}")
                    else:
                        visit(Path(entry.path), child_prefix)
                elif entry.is_file(follow_symlinks=False):
                    info = entry.stat(follow_symlinks=False)
                    if info.st_nlink != 1:
                        problems.append(f"forbidden hardlink: {relative}")
                    else:
                        files[relative] = (Path(entry.path), info.st_size)
                else:
                    problems.append(f"non-regular entry: {relative}")
            except OSError as error:
                problems.append(f"cannot inspect {relative}: {error}")

    visit(root, PurePosixPath())
    return files, problems


def _validate_artifact_record(
    record: Mapping[str, Any], files: Mapping[str, tuple[Path, int]],
    errors: list[str],
) -> None:
    required = {
        "level", "path", "kind", "format", "dtype", "shape",
        "bytes", "sha256",
    }
    if set(record) != required:
        errors.append("artifact record has missing or unknown fields")
        return
    path = record.get("path")
    if not isinstance(path, str):
        errors.append("artifact path must be a string")
        return
    if path not in {RESULT_PATH, FAILURE_PATH, PROVENANCE_PATH, MAP_PATH, CURVES_PATH, AUDIT_PATH}:
        errors.append(f"artifact path is outside closed layout: {path!r}")
        return
    if record.get("level") != path.split("/", 1)[0]:
        errors.append(f"artifact level/path mismatch: {path}")
    entry = files.get(path)
    if entry is None:
        errors.append(f"missing artifact file: {path}")
        return
    try:
        raw = entry[0].read_bytes()
    except OSError as error:
        errors.append(f"cannot read artifact {path}: {error}")
        return
    if type(record.get("bytes")) is not int or record["bytes"] != len(raw):
        errors.append(f"artifact byte mismatch: {path}")
    digest = record.get("sha256")
    if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
        errors.append(f"artifact digest is invalid: {path}")
    elif digest != _sha(raw):
        errors.append(f"artifact hash mismatch: {path}")
    try:
        if path == RESULT_PATH:
            if (record.get("kind"), record.get("format"), record.get("dtype"), record.get("shape")) != (
                "result", "json", "json", None
            ):
                raise StageDArtifactPolicyError("result record metadata is invalid")
            _load_canonical_json(raw, "result")
        elif path == FAILURE_PATH:
            if (record.get("kind"), record.get("format"), record.get("dtype"), record.get("shape")) != (
                "scientific_failure", "json", "json", None
            ):
                raise StageDArtifactPolicyError("failure record metadata is invalid")
            document = _load_canonical_json(raw, "scientific failure evidence")
            if set(document) != {
                "schema_version", "document_type", "failure", "l0_scalars"
            } or document.get("schema_version") != 1 or document.get("document_type") != FAILURE_DOCUMENT_TYPE or not isinstance(document.get("l0_scalars"), Mapping):
                raise StageDArtifactPolicyError("failure evidence document is invalid")
        elif path == PROVENANCE_PATH:
            if (record.get("kind"), record.get("format"), record.get("dtype"), record.get("shape")) != (
                "provenance", "json", "json", None
            ):
                raise StageDArtifactPolicyError("provenance record metadata is invalid")
        elif path == MAP_PATH:
            array = _load_npy(raw)
            if (
                record.get("kind") != "metric_ready_map"
                or record.get("format") != "npy"
                or record.get("dtype") != "float32"
                or record.get("shape") != [56, 56]
                or array.dtype != np.dtype(np.float32)
                or array.shape != (56, 56)
                or not np.all(np.isfinite(array))
            ):
                raise StageDArtifactPolicyError("metric-ready map contract failed")
        elif path == CURVES_PATH:
            arrays = _load_npz(raw)
            array = arrays.get("curves")
            if (
                set(arrays) != {"curves"}
                or record.get("kind") != "fidelity_curves"
                or record.get("format") != "npz"
                or record.get("dtype") != "float64"
                or record.get("shape") != [2, 5, 21]
                or array is None or array.dtype != np.dtype(np.float64)
                or array.shape != (2, 5, 21)
                or not np.all(np.isfinite(array))
            ):
                raise StageDArtifactPolicyError("fidelity-curves contract failed")
        elif path == AUDIT_PATH:
            arrays = _load_npz(raw)
            repeat_map = arrays.get(AUDIT_REPEAT_MAP_KEY)
            if (
                not arrays or record.get("kind") != "audit_evidence"
                or record.get("format") != "npz"
                or record.get("dtype") != {
                    key: array.dtype.name for key, array in arrays.items()
                }
                or record.get("shape") != {
                    key: list(array.shape) for key, array in arrays.items()
                }
                or any(
                    array.size == 0 or array.dtype.kind not in {"b", "i", "u", "f"}
                    or (array.dtype.kind == "f" and not np.all(np.isfinite(array)))
                    for array in arrays.values()
                )
                or repeat_map is None
                or repeat_map.dtype != np.dtype(np.float32)
                or repeat_map.shape != (56, 56)
            ):
                raise StageDArtifactPolicyError("audit evidence contract failed")
    except Exception as error:
        errors.append(f"artifact payload contract failed for {path}: {error}")


def _validate_headers(
    status: Mapping[str, Any], manifest: Mapping[str, Any], errors: list[str],
) -> None:
    status_fields = {
        "schema_version", "document_type", "status", "terminal",
        "complete", "attempt_number", "unit_digest", "manifest_sha256",
    }
    manifest_fields = {
        "schema_version", "document_type", "status", "terminal",
        "complete", "completion_authority", "attempt_number",
        "unit_identity", "unit_digest", "group_digest", "pair_digest",
        "audit", "provenance_path", "provenance_sha256", "retention",
        "artifacts",
    }
    state = status.get("status")
    if "failure" in status:
        status_fields.add("failure")
    if "failure" in manifest:
        manifest_fields.add("failure")
    if set(status) != status_fields:
        errors.append("status has missing or unknown fields")
    if set(manifest) != manifest_fields:
        errors.append("manifest has missing or unknown fields")
    if status.get("schema_version") != 1 or status.get("document_type") != STATUS_DOCUMENT_TYPE:
        errors.append("status identity is invalid")
    if manifest.get("schema_version") != 1 or manifest.get("document_type") != MANIFEST_DOCUMENT_TYPE:
        errors.append("manifest identity is invalid")
    expected_flags = {
        RUNNING_STATUS: (False, False),
        INFRASTRUCTURE_INTERRUPTION_STATUS: (False, False),
        SUCCESS_STATUS: (True, True),
        SCIENTIFIC_FAILURE_STATUS: (True, True),
    }.get(state)
    if expected_flags is None:
        errors.append("status state is invalid")
    elif (status.get("terminal"), status.get("complete")) != expected_flags:
        errors.append("status terminal/complete flags are inconsistent")
    if manifest.get("status") != state or (
        manifest.get("terminal"), manifest.get("complete")
    ) != (status.get("terminal"), status.get("complete")):
        errors.append("status and manifest lifecycle disagree")
    for key in ("attempt_number", "unit_digest", "failure"):
        if status.get(key) != manifest.get(key):
            errors.append(f"status and manifest {key} disagree")
    if state == SCIENTIFIC_FAILURE_STATUS:
        if not isinstance(status.get("failure"), Mapping) or not status["failure"]:
            errors.append("scientific failure lacks structured failure")
    elif state == INFRASTRUCTURE_INTERRUPTION_STATUS:
        if not isinstance(status.get("failure"), Mapping) or not status["failure"]:
            errors.append("infrastructure interruption lacks control failure")
    elif "failure" in status or "failure" in manifest:
        errors.append("non-failure lifecycle contains failure")
    if manifest.get("completion_authority") != STATUS_FILENAME:
        errors.append("manifest completion authority is invalid")
    if manifest.get("provenance_path") != PROVENANCE_PATH:
        errors.append("manifest provenance path is invalid")
    if manifest.get("retention") != RETENTION_CONTRACT:
        errors.append("manifest retention contract is invalid")
    for key in ("unit_digest", "group_digest", "pair_digest"):
        digest = manifest.get(key)
        if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
            errors.append(f"manifest {key} is invalid")
    if type(manifest.get("audit")) is not bool:
        errors.append("manifest audit must be boolean")


def validate_stage_d_attempt(
    attempt_dir: str | Path, *,
    expected_unit: StageDUnit | Mapping[str, Any] | None = None,
    expected_group: StageDImageGroup | Mapping[str, Any] | None = None,
    expected_plan_digest: str | None = None,
    expected_mode: str | None = None,
    expected_provenance_bindings: Mapping[str, Any] | None = None,
    observed_git_state: Mapping[str, Any] | None = None,
) -> StageDAttemptValidation:
    """Recompute the complete status/manifest/inventory/provenance chain."""

    root = _absolute_path(attempt_dir)
    errors: list[str] = []
    files: dict[str, tuple[Path, int]] = {}
    manifest_digest: str | None = None
    attempt_digest: str | None = None
    status_value: dict[str, Any] | None = None
    manifest: dict[str, Any] | None = None
    unit_digest: str | None = None
    artifact_count = 0
    total_bytes = 0
    try:
        _assert_directory_chain(root, create=False)
        attempt_number = _attempt_number_from_path(root)
        files, problems = _inventory_regular_files(root)
        errors.extend(problems)
        total_bytes = sum(size for _path, size in files.values())
    except Exception as error:
        errors.append(f"cannot inspect attempt: {error}")
        attempt_number = None

    status_raw: bytes | None = None
    manifest_raw: bytes | None = None
    if STATUS_FILENAME not in files:
        errors.append("missing status.json")
    else:
        try:
            status_raw = files[STATUS_FILENAME][0].read_bytes()
            status_value = _load_canonical_json(status_raw, "status.json")
        except Exception as error:
            errors.append(f"invalid status.json: {error}")
    if MANIFEST_FILENAME not in files:
        errors.append("missing manifest.json")
    else:
        try:
            manifest_raw = files[MANIFEST_FILENAME][0].read_bytes()
            manifest_digest = _sha(manifest_raw)
            manifest = _load_canonical_json(manifest_raw, "manifest.json")
        except Exception as error:
            errors.append(f"invalid manifest.json: {error}")
    if status_value is not None and manifest is not None and manifest_raw is not None:
        _validate_headers(status_value, manifest, errors)
        if status_value.get("manifest_sha256") != manifest_digest:
            errors.append("status does not bind the manifest bytes")
        if attempt_number is not None and status_value.get("attempt_number") != attempt_number:
            errors.append("status attempt number disagrees with directory")
        unit_digest = manifest.get("unit_digest") if isinstance(
            manifest.get("unit_digest"), str
        ) else None
        manifest_unit: StageDUnit | None = None
        try:
            manifest_unit = _unit(manifest.get("unit_identity"))
            if manifest_unit.canonical_digest != manifest.get("unit_digest"):
                errors.append("manifest unit identity/digest mismatch")
            if manifest_unit.pair_binding_digest != manifest.get("pair_digest"):
                errors.append("manifest unit/pair digest mismatch")
            if manifest_unit.audit is not manifest.get("audit"):
                errors.append("manifest unit/audit flag mismatch")
        except Exception as error:
            errors.append(f"manifest unit identity is invalid: {error}")

        records = manifest.get("artifacts")
        if not isinstance(records, list):
            errors.append("manifest artifacts must be an ordered array")
            records = []
        artifact_count = len(records)
        paths: list[Any] = []
        for raw_record in records:
            if not isinstance(raw_record, Mapping):
                errors.append("manifest artifact record is not an object")
                continue
            paths.append(raw_record.get("path"))
            _validate_artifact_record(raw_record, files, errors)
        comparable_paths = [path for path in paths if isinstance(path, str)]
        if len(comparable_paths) != len(paths):
            errors.append("manifest contains a non-string artifact path")
        if len(comparable_paths) != len(set(comparable_paths)):
            errors.append("manifest contains duplicate artifact paths")
        audit = manifest.get("audit")
        if type(audit) is not bool:
            errors.append("manifest audit flag is invalid")
            audit = False
        state = status_value.get("status")
        if state in {SUCCESS_STATUS, SCIENTIFIC_FAILURE_STATUS}:
            if state == SUCCESS_STATUS:
                expected_payload_order = list(_PAYLOAD_PATHS) + (
                    [AUDIT_PATH] if audit else []
                )
            else:
                failure_value = status_value.get("failure")
                try:
                    normalized_failure = normalize_stage_d_scientific_failure(
                        failure_value, audit=audit,
                        produced=(
                            failure_value.get("produced", [])
                            if isinstance(failure_value, Mapping) else []
                        ),
                    )
                    expected_payload_order = [
                        *normalized_failure["produced"], FAILURE_PATH,
                    ]
                except Exception as error:
                    errors.append(f"invalid scientific failure declaration: {error}")
                    expected_payload_order = [FAILURE_PATH]
            expected_order = [*expected_payload_order, PROVENANCE_PATH]
            if paths != expected_order:
                errors.append("terminal manifest does not use exact artifact order/layout")
            expected_files = {
                STATUS_FILENAME, MANIFEST_FILENAME, PROVENANCE_PATH,
                *expected_payload_order,
            }
            if set(files) != expected_files:
                errors.append(
                    "terminal attempt has missing or extra files: "
                    f"missing={sorted(expected_files - set(files))}, "
                    f"extra={sorted(set(files) - expected_files)}"
                )
            if state == SCIENTIFIC_FAILURE_STATUS and FAILURE_PATH in files:
                try:
                    failure_document = _load_canonical_json(
                        files[FAILURE_PATH][0].read_bytes(),
                        "scientific failure evidence",
                    )
                    if failure_document.get("failure") != status_value.get("failure"):
                        errors.append(
                            "failure evidence disagrees with status/manifest failure"
                        )
                except Exception as error:
                    errors.append(f"invalid scientific failure evidence: {error}")
            provenance_records = [
                item for item in records if isinstance(item, Mapping)
                and item.get("path") == PROVENANCE_PATH
            ]
            if len(provenance_records) != 1:
                errors.append("terminal manifest lacks one provenance record")
            elif manifest.get("provenance_sha256") != provenance_records[0].get("sha256"):
                errors.append("manifest provenance digest mismatch")
            provenance_entry = files.get(PROVENANCE_PATH)
            if provenance_entry is not None:
                try:
                    provenance_raw = provenance_entry[0].read_bytes()
                    provenance = _load_canonical_json(provenance_raw, "provenance")
                    if provenance.get("document_type") != PROVENANCE_DOCUMENT_TYPE:
                        raise StageDProvenanceError("wrong provenance document_type")
                    provenance_coordinates = provenance.get("bindings", {}).get(
                        "coordinates", {}
                    )
                    for provenance_key, manifest_key in (
                        ("unit_digest", "unit_digest"),
                        ("group_digest", "group_digest"),
                        ("pair_digest", "pair_digest"),
                    ):
                        if provenance_coordinates.get(provenance_key) != manifest.get(
                            manifest_key
                        ):
                            raise StageDProvenanceError(
                                f"provenance {provenance_key} disagrees with manifest"
                            )
                    validate_stage_d_provenance(
                        provenance, expected_unit=(
                            manifest_unit if manifest_unit is not None else expected_unit
                        ),
                        expected_group=expected_group,
                        expected_plan_digest=expected_plan_digest,
                        expected_mode=expected_mode,
                        expected_attempt_number=attempt_number,
                        expected_terminal={
                            "status": state, "terminal": True, "complete": True,
                        },
                        expected_failure=status_value.get("failure"),
                        expected_inventory=[
                            item for item in records
                            if isinstance(item, Mapping)
                            and item.get("path") != PROVENANCE_PATH
                        ],
                        expected_bindings=expected_provenance_bindings,
                        observed_git_state=observed_git_state,
                    )
                except Exception as error:
                    errors.append(f"invalid provenance: {error}")
        else:
            if records:
                errors.append("nonterminal manifest cannot claim scientific artifacts")
            if manifest.get("provenance_sha256") is not None:
                errors.append("nonterminal manifest cannot bind provenance")
            if set(files) != {STATUS_FILENAME, MANIFEST_FILENAME}:
                errors.append("nonterminal attempt contains published payload or extra files")

        if expected_unit is not None:
            try:
                typed_unit = _unit(expected_unit)
                if manifest.get("unit_identity") != typed_unit.to_dict():
                    errors.append("manifest unit does not match expected unit")
                if manifest.get("unit_digest") != typed_unit.canonical_digest:
                    errors.append("manifest unit digest does not match expected unit")
            except Exception as error:
                errors.append(f"invalid expected unit: {error}")
        if expected_group is not None:
            try:
                typed_group = _group(expected_group)
                if manifest.get("group_digest") != typed_group.canonical_digest:
                    errors.append("manifest group digest does not match expected group")
                if manifest.get("pair_digest") != typed_group.pair_binding.canonical_digest:
                    errors.append("manifest pair digest does not match expected group")
            except Exception as error:
                errors.append(f"invalid expected group: {error}")

    if files:
        inventory = []
        try:
            for relative in sorted(files):
                path, size = files[relative]
                raw = path.read_bytes()
                inventory.append({"path": relative, "bytes": size, "sha256": _sha(raw)})
            attempt_digest = canonical_sha256({"files": inventory})
        except OSError as error:
            errors.append(f"cannot digest complete attempt inventory: {error}")
    state = status_value.get("status") if status_value is not None else None
    terminal = bool(status_value.get("terminal")) if status_value is not None else False
    complete = bool(status_value.get("complete")) if status_value is not None else False
    valid = not errors
    # Both terminal scientific outcomes are safe to resume-skip.
    resumable = valid and terminal and complete and state in {
        SUCCESS_STATUS, SCIENTIFIC_FAILURE_STATUS,
    }
    return StageDAttemptValidation(
        path=root, valid=valid, terminal=terminal, complete=complete,
        resumable=resumable, status=state if isinstance(state, str) else None,
        errors=tuple(errors), digest=attempt_digest,
        manifest_sha256=manifest_digest, unit_digest=unit_digest,
        artifact_count=artifact_count, total_bytes=total_bytes,
    )


# Descriptive alias for callers that use the Stage C validation naming style.
validate_execution_unit = validate_stage_d_attempt


__all__ = [
    "AUDIT_PATH", "AUDIT_REPEAT_MAP_KEY", "ArtifactPolicyError",
    "ArtifactWriterStateError",
    "AttemptNotEmptyError", "CURVES_PATH", "FAULT_AFTER_MANIFEST",
    "FAULT_AFTER_PAYLOAD", "FAULT_AFTER_RUNNING", "FAULT_BEFORE_COMPLETE_STATUS",
    "FAULT_BEFORE_TERMINAL_STATUS", "FAULT_POINTS",
    "INFRASTRUCTURE_INTERRUPTION_STATUS", "InjectedArtifactCrash",
    "InjectedStageDArtifactCrash", "MANIFEST_DOCUMENT_TYPE",
    "FAILURE_PATH", "MANIFEST_FILENAME", "MAP_PATH", "PROVENANCE_PATH",
    "RESULT_PATH",
    "STATUS_DOCUMENT_TYPE", "STATUS_FILENAME", "StageDArtifactError",
    "RETENTION_CONTRACT", "RETRYABLE_INFRASTRUCTURE_CODES",
    "StageDArtifactPolicyError", "StageDArtifactWriter",
    "StageDArtifactWriterStateError", "StageDAttemptNotEmptyError",
    "StageDAttemptValidation", "StageDUnitPayload",
    "attempt_directories", "next_attempt_directory",
    "validate_execution_unit", "validate_stage_d_attempt",
]
