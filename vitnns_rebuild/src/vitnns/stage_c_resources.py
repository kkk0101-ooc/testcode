"""Small, composable resource instrumentation primitives for Stage C.

This module deliberately does not own execution directories, artifact writing,
manifests, resume behavior, or budget decisions.  A caller supplies the exact
files whose sizes belong to one measurement and explicitly increments the call
counter at the same batched-model-invocation boundary used by the protocol.
"""

from __future__ import annotations

import math
import threading
import time
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch


CUDA_MEASURED = "measured"
CUDA_NOT_APPLICABLE = "not_applicable_cpu"
CUDA_MEASUREMENT_FAILED = "measurement_failed"
PAYLOAD_FILE_LIST_SCOPE = "explicit_payload_file_list"
ROOT_TREE_SCOPE = "exhaustive_regular_file_tree"
_CUDA_OWNERS_LOCK = threading.Lock()
_CUDA_OWNERS: set[int] = set()
REQUIRED_PHASES = (
    "attribution_or_map_compute",
    "fidelity_compute",
    "integrity_checking",
    "artifact_finalize",
)
PHASE_CLOCK = "perf_counter"
PHASE_SYNCHRONIZATION = "device_before_and_after_each_phase"
TOTAL_BOUNDARY = (
    "unit_start_through_first_complete_validation_"
    "excluding_telemetry_publication"
)


def _positive_count(value: Any, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer (not bool)")
    return value


@dataclass(frozen=True)
class PhaseTiming:
    """Accumulated synchronized wall time for one named unit phase."""

    wall_time_seconds: float
    invocations: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.wall_time_seconds, bool)
            or not isinstance(self.wall_time_seconds, (int, float))
            or not math.isfinite(float(self.wall_time_seconds))
            or self.wall_time_seconds < 0.0
        ):
            raise ValueError(
                "phase wall_time_seconds must be finite and non-negative"
            )
        if type(self.invocations) is not int or self.invocations < 0:
            raise ValueError("phase invocations must be a non-negative integer")
        object.__setattr__(self, "wall_time_seconds", float(self.wall_time_seconds))

    def to_dict(self) -> dict[str, Any]:
        return {
            "wall_time_seconds": self.wall_time_seconds,
            "invocations": self.invocations,
        }


class PhaseTimer:
    """Synchronized, additive phase timing inside one execution unit.

    Phase intervals may be nested, so their durations are deliberately not
    represented as a partition of total wall time.  Model invocation counts
    remain owned exclusively by :class:`CallCounter`.
    """

    def __init__(
        self, *, device: torch.device | str, clock: Any = time.perf_counter
    ) -> None:
        self.device = torch.device(device)
        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError("phase timing supports only CPU or CUDA devices")
        if not callable(clock):
            raise TypeError("clock must be callable")
        self._clock = clock
        self._durations = {name: 0.0 for name in REQUIRED_PHASES}
        self._invocations = {name: 0 for name in REQUIRED_PHASES}
        self._total_start: float | None = None
        self._total_wall: float | None = None

    def _synchronize(self) -> None:
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA phase timing requested but CUDA is unavailable"
                )
            if self.device.index is None:
                self.device = torch.device("cuda", torch.cuda.current_device())
            torch.cuda.synchronize(self.device)

    def _now(self, context: str) -> float:
        value = self._clock()
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise RuntimeError(f"{context} clock reading is not finite")
        return float(value)

    def start(self) -> None:
        if self._total_start is not None or self._total_wall is not None:
            raise RuntimeError("PhaseTimer instances are single-use")
        self._synchronize()
        self._total_start = self._now("total start")

    @contextmanager
    def measure(self, phase: str):
        if phase not in self._durations:
            raise ValueError(f"unknown execution-unit phase: {phase!r}")
        if self._total_start is None or self._total_wall is not None:
            raise RuntimeError("phase timing requires an active total boundary")
        self._synchronize()
        started = self._now(f"{phase} start")
        body_error: BaseException | None = None
        try:
            yield
        except BaseException as error:
            body_error = error
            raise
        finally:
            try:
                self._synchronize()
                elapsed = self._now(f"{phase} end") - started
                if not math.isfinite(elapsed) or elapsed < 0.0:
                    raise RuntimeError(
                        f"{phase} duration was non-finite or negative"
                    )
                self._durations[phase] += elapsed
                self._invocations[phase] += 1
            except BaseException as timing_error:
                if body_error is None:
                    raise
                if hasattr(body_error, "add_note"):
                    try:
                        body_error.add_note(
                            f"Stage C {phase} timing also failed: "
                            f"{type(timing_error).__name__}: {timing_error}"
                        )
                    except BaseException:
                        pass

    def stop(self) -> None:
        if self._total_start is None or self._total_wall is not None:
            raise RuntimeError("total phase boundary is not active")
        self._synchronize()
        elapsed = self._now("total end") - self._total_start
        if not math.isfinite(elapsed) or elapsed < 0.0:
            raise RuntimeError("total wall duration was non-finite or negative")
        self._total_wall = elapsed

    @property
    def stopped(self) -> bool:
        return self._total_wall is not None

    def to_dict(self) -> dict[str, Any]:
        if self._total_wall is None:
            raise RuntimeError("phase timings are unavailable before total stop")
        phases = {
            name: PhaseTiming(
                self._durations[name], self._invocations[name]
            ).to_dict()
            for name in REQUIRED_PHASES
        }
        return {
            "clock": PHASE_CLOCK,
            "synchronization": PHASE_SYNCHRONIZATION,
            "phases": phases,
            "synchronized_total_wall_time_seconds": self._total_wall,
            "total_boundary": TOTAL_BOUNDARY,
            "phases_are_mutually_exclusive": False,
        }


def measurement_environment(device: torch.device | str) -> dict[str, Any]:
    """Return JSON-safe runtime identity for persisted timing evidence."""

    resolved = torch.device(device)
    if resolved.type not in {"cpu", "cuda"}:
        raise ValueError("measurement environment supports only CPU or CUDA")
    if resolved.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA measurement environment requested but CUDA is unavailable"
            )
        if resolved.index is None:
            resolved = torch.device("cuda", torch.cuda.current_device())
        device_name: str | None = str(torch.cuda.get_device_name(resolved))
        cuda_runtime: str | None = torch.version.cuda
        cudnn_value = torch.backends.cudnn.version()
        cudnn_version: int | None = (
            None if cudnn_value is None else int(cudnn_value)
        )
    else:
        device_name = None
        cuda_runtime = None
        cudnn_version = None
    return {
        "device_type": resolved.type,
        "device": str(resolved),
        "device_name": device_name,
        "cuda_runtime_version": cuda_runtime,
        "cudnn_version": cudnn_version,
    }


@dataclass(frozen=True)
class CallCounts:
    """Immutable counts in units of batched model invocations."""

    forward_calls: int
    backward_calls: int
    accounting_unit: str = "batched_model_invocation"

    def __post_init__(self) -> None:
        for name in ("forward_calls", "backward_calls"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer (not bool)")
        if self.accounting_unit != "batched_model_invocation":
            raise ValueError("accounting_unit must be 'batched_model_invocation'")

    def to_dict(self) -> dict[str, Any]:
        return {
            "accounting_unit": self.accounting_unit,
            "forward_calls": self.forward_calls,
            "backward_calls": self.backward_calls,
        }


class CallCounter:
    """Thread-safe explicit counter; it never infers calls from tensor work."""

    def __init__(self) -> None:
        self._forward_calls = 0
        self._backward_calls = 0
        self._lock = threading.Lock()

    def record_forward(self, count: int = 1) -> None:
        count = _positive_count(count, "forward count")
        with self._lock:
            self._forward_calls += count

    def record_backward(self, count: int = 1) -> None:
        count = _positive_count(count, "backward count")
        with self._lock:
            self._backward_calls += count

    def snapshot(self) -> CallCounts:
        with self._lock:
            return CallCounts(self._forward_calls, self._backward_calls)

    def counted_forward(self, function: Any):
        """Return a wrapper that counts each actual callable invocation."""

        if not callable(function):
            raise TypeError("forward function must be callable")

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            self.record_forward()
            return function(*args, **kwargs)

        return wrapped

    def counted_backward(self, function: Any):
        """Return a wrapper that counts each actual backward invocation."""

        if not callable(function):
            raise TypeError("backward function must be callable")

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            self.record_backward()
            return function(*args, **kwargs)

        return wrapped


@dataclass(frozen=True)
class ArtifactFileBytes:
    path: str
    bytes: int

    def __post_init__(self) -> None:
        if not isinstance(self.path, str) or not self.path:
            raise ValueError("artifact path must be a non-empty string")
        if type(self.bytes) is not int or self.bytes < 0:
            raise ValueError("artifact bytes must be a non-negative integer")

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "bytes": self.bytes}


@dataclass(frozen=True)
class ArtifactByteMeasurement:
    """Exact sizes of an explicit, duplicate-free set of regular files."""

    files: tuple[ArtifactFileBytes, ...]
    total_bytes: int
    scope: str = PAYLOAD_FILE_LIST_SCOPE
    root: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.files, tuple) or not all(
            isinstance(item, ArtifactFileBytes) for item in self.files
        ):
            raise TypeError("files must be a tuple of ArtifactFileBytes")
        if type(self.total_bytes) is not int or self.total_bytes < 0:
            raise ValueError("total_bytes must be a non-negative integer")
        if self.total_bytes != sum(item.bytes for item in self.files):
            raise ValueError("total_bytes does not equal the explicit file-size sum")
        if self.scope not in {PAYLOAD_FILE_LIST_SCOPE, ROOT_TREE_SCOPE}:
            raise ValueError("invalid artifact byte measurement scope")
        if self.scope == PAYLOAD_FILE_LIST_SCOPE and self.root is not None:
            raise ValueError("payload file-list measurements cannot name a root")
        if self.scope == ROOT_TREE_SCOPE and (
            not isinstance(self.root, str) or not self.root
        ):
            raise ValueError("root-tree measurements require a root")

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_bytes": self.total_bytes,
            "artifact_bytes_scope": self.scope,
            "artifact_root": self.root,
            "files": [item.to_dict() for item in self.files],
        }


def measure_artifact_bytes(paths: Iterable[Path | str]) -> ArtifactByteMeasurement:
    """Measure only the explicitly supplied regular files.

    Directories and duplicate files are rejected so callers cannot accidentally
    change the accounting scope through recursive traversal or double counting.
    """

    if isinstance(paths, (str, bytes, Path)):
        raise TypeError("artifact paths must be an explicit iterable of file paths")
    try:
        candidates = tuple(paths)
    except TypeError as exc:
        raise TypeError(
            "artifact paths must be an explicit iterable of file paths"
        ) from exc

    seen: set[Path] = set()
    measured: list[ArtifactFileBytes] = []
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, (str, Path)):
            raise TypeError(f"artifact paths[{index}] must be a path or string")
        path = Path(candidate).resolve()
        if path in seen:
            raise ValueError(f"duplicate artifact file: {path}")
        seen.add(path)
        if not path.is_file():
            raise FileNotFoundError(f"artifact is not a regular file: {path}")
        size = path.stat().st_size
        measured.append(ArtifactFileBytes(str(path), size))
    files = tuple(measured)
    return ArtifactByteMeasurement(files, sum(item.bytes for item in files))


def measure_artifact_tree_bytes(root: Path | str) -> ArtifactByteMeasurement:
    """Measure every regular file currently present below one explicit root.

    This is read-only accounting, not directory ownership or retention
    orchestration. Symlinks are rejected so the declared tree cannot escape its
    root or alias another file.
    """

    if not isinstance(root, (str, Path)):
        raise TypeError("artifact root must be a path or string")
    resolved = Path(root).resolve()
    if not resolved.is_dir():
        raise NotADirectoryError(f"artifact root is not a directory: {resolved}")
    measured: list[ArtifactFileBytes] = []
    for path in sorted(resolved.rglob("*"), key=lambda item: str(item)):
        if path.is_symlink():
            raise ValueError(f"artifact tree contains a symlink: {path}")
        if path.is_file():
            measured.append(ArtifactFileBytes(str(path.resolve()), path.stat().st_size))
    files = tuple(measured)
    return ArtifactByteMeasurement(
        files, sum(item.bytes for item in files), scope=ROOT_TREE_SCOPE, root=str(resolved)
    )


@dataclass(frozen=True)
class InstrumentationError:
    phase: str
    exception_type: str
    message: str

    @classmethod
    def from_exception(cls, phase: str, error: BaseException) -> "InstrumentationError":
        try:
            message = str(error)
        except BaseException as formatting_error:
            message = (
                "<exception message unavailable: "
                f"{type(formatting_error).__module__}.{type(formatting_error).__qualname__}>"
            )
        return cls(
            phase, f"{type(error).__module__}.{type(error).__qualname__}", message
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "phase": self.phase,
            "type": self.exception_type,
            "message": self.message,
        }


@dataclass(frozen=True)
class ResourceRecord:
    """Immutable, JSON-safe result for one measured execution boundary."""

    status: str
    wall_time_seconds: float | None
    cuda_status: str
    cuda_device: str | None
    cuda_peak_allocated_bytes: int | None
    cuda_peak_reserved_bytes: int | None
    calls: CallCounts
    artifacts: ArtifactByteMeasurement
    exception_type: str | None = None
    exception_message: str | None = None
    instrumentation_errors: tuple[InstrumentationError, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in {"completed", "failed"}:
            raise ValueError("status must be 'completed' or 'failed'")
        if self.wall_time_seconds is not None:
            if isinstance(self.wall_time_seconds, bool) or not isinstance(
                self.wall_time_seconds, (int, float)
            ):
                raise TypeError("wall_time_seconds must be numeric or None")
            wall = float(self.wall_time_seconds)
            if not math.isfinite(wall) or wall < 0.0:
                raise ValueError("wall_time_seconds must be finite and non-negative")
            object.__setattr__(self, "wall_time_seconds", wall)
        if self.cuda_status == CUDA_NOT_APPLICABLE:
            if any(
                value is not None
                for value in (
                    self.cuda_device,
                    self.cuda_peak_allocated_bytes,
                    self.cuda_peak_reserved_bytes,
                )
            ):
                raise ValueError("CPU records cannot contain CUDA measurements")
        elif self.cuda_status == CUDA_MEASURED:
            if not isinstance(self.cuda_device, str) or not self.cuda_device:
                raise ValueError("measured CUDA records require cuda_device")
            for name in ("cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes"):
                value = getattr(self, name)
                if type(value) is not int or value < 0:
                    raise ValueError(f"{name} must be a non-negative integer")
        elif self.cuda_status == CUDA_MEASUREMENT_FAILED:
            if not isinstance(self.cuda_device, str) or not self.cuda_device:
                raise ValueError("failed CUDA measurements require cuda_device")
            for name in ("cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes"):
                value = getattr(self, name)
                if value is not None and (type(value) is not int or value < 0):
                    raise ValueError(f"{name} must be None or a non-negative integer")
        else:
            raise ValueError("invalid cuda_status")
        if not isinstance(self.calls, CallCounts):
            raise TypeError("calls must be CallCounts")
        if not isinstance(self.artifacts, ArtifactByteMeasurement):
            raise TypeError("artifacts must be ArtifactByteMeasurement")
        if not isinstance(self.instrumentation_errors, tuple) or not all(
            isinstance(item, InstrumentationError) for item in self.instrumentation_errors
        ):
            raise TypeError("instrumentation_errors must be InstrumentationError values")
        if self.status == "completed":
            if self.exception_type is not None or self.exception_message is not None:
                raise ValueError("completed records cannot contain exception details")
            if self.instrumentation_errors:
                raise ValueError("completed records cannot contain instrumentation errors")
        elif self.exception_type is None and not self.instrumentation_errors:
            raise ValueError("failed records require a body or instrumentation error")
        elif self.exception_type is not None and (
            not isinstance(self.exception_type, str)
            or not isinstance(self.exception_message, str)
        ):
            raise TypeError("failed record exception details must be strings")

    @property
    def forward_calls(self) -> int:
        return self.calls.forward_calls

    @property
    def backward_calls(self) -> int:
        return self.calls.backward_calls

    @property
    def artifact_bytes(self) -> int:
        return self.artifacts.total_bytes

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "wall_time_seconds": self.wall_time_seconds,
            "cuda": {
                "status": self.cuda_status,
                "device": self.cuda_device,
                "peak_allocated_bytes": self.cuda_peak_allocated_bytes,
                "peak_reserved_bytes": self.cuda_peak_reserved_bytes,
            },
            "calls": self.calls.to_dict(),
            **self.artifacts.to_dict(),
            "exception": None
            if self.exception_type is None
            else {
                "type": self.exception_type,
                "message": self.exception_message,
            },
            "instrumentation_errors": [
                item.to_dict() for item in self.instrumentation_errors
            ],
        }


class ResourceMeter(AbstractContextManager["ResourceMeter"]):
    """Measure wall time and optional CUDA peaks for one caller-owned unit."""

    def __init__(
        self,
        *,
        device: torch.device | str,
        artifact_files: Iterable[Path | str] = (),
        call_counter: CallCounter | None = None,
    ) -> None:
        self.device = torch.device(device)
        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError("resource measurement supports only CPU or CUDA devices")
        if isinstance(artifact_files, (str, bytes, Path)):
            raise TypeError("artifact_files must be an explicit iterable")
        self._artifact_files = tuple(artifact_files)
        if call_counter is not None and not isinstance(call_counter, CallCounter):
            raise TypeError("call_counter must be a CallCounter or None")
        self.call_counter = call_counter or CallCounter()
        self._started = False
        self._finished = False
        self._start_time: float | None = None
        self._start_calls: CallCounts | None = None
        self._record: ResourceRecord | None = None
        self._cuda_index: int | None = None
        self._owns_cuda = False

    @property
    def record(self) -> ResourceRecord:
        if self._record is None:
            raise RuntimeError("resource record is unavailable before the boundary exits")
        return self._record

    def record_forward(self, count: int = 1) -> None:
        if not self._started or self._finished:
            raise RuntimeError("forward calls can only be recorded inside the boundary")
        self.call_counter.record_forward(count)

    def record_backward(self, count: int = 1) -> None:
        if not self._started or self._finished:
            raise RuntimeError("backward calls can only be recorded inside the boundary")
        self.call_counter.record_backward(count)

    def __enter__(self) -> "ResourceMeter":
        if self._started or self._finished:
            raise RuntimeError("ResourceMeter instances are single-use")
        try:
            if self.device.type == "cuda":
                if not torch.cuda.is_available():
                    raise RuntimeError(
                        "CUDA resource measurement requested but CUDA is unavailable"
                    )
                index = (
                    torch.cuda.current_device()
                    if self.device.index is None
                    else self.device.index
                )
                concrete = torch.device("cuda", index)
                with _CUDA_OWNERS_LOCK:
                    if index in _CUDA_OWNERS:
                        raise RuntimeError(
                            f"CUDA resource meter already active for {concrete}"
                        )
                    _CUDA_OWNERS.add(index)
                self._cuda_index = index
                self._owns_cuda = True
                self.device = concrete
                torch.cuda.synchronize(concrete)
                torch.cuda.reset_peak_memory_stats(concrete)
            self._start_calls = self.call_counter.snapshot()
            self._start_time = time.perf_counter()
            self._started = True
            return self
        except BaseException:
            self._release_cuda_owner()
            self._cuda_index = None
            self._start_calls = None
            self._start_time = None
            self._started = False
            raise

    def _release_cuda_owner(self) -> None:
        if self._owns_cuda and self._cuda_index is not None:
            with _CUDA_OWNERS_LOCK:
                _CUDA_OWNERS.discard(self._cuda_index)
        self._owns_cuda = False

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        if self._start_time is None or self._finished:
            raise RuntimeError("ResourceMeter boundary is not active")
        errors: list[InstrumentationError] = []
        raised: list[tuple[BaseException, Any]] = []

        def attempt(phase: str, function: Any, default: Any) -> Any:
            try:
                return function()
            except BaseException as error:
                errors.append(InstrumentationError.from_exception(phase, error))
                raised.append((error, error.__traceback__))
                return default

        try:
            if self.device.type == "cuda":
                attempt("cuda_synchronize_after", lambda: torch.cuda.synchronize(self.device), None)
            end_time = attempt("wall_clock_after", time.perf_counter, None)
            elapsed = None if end_time is None else end_time - self._start_time
            if elapsed is not None and (not math.isfinite(elapsed) or elapsed < 0.0):
                error = RuntimeError(
                    "wall-clock duration was non-finite or negative"
                )
                errors.append(InstrumentationError.from_exception("wall_clock_after", error))
                raised.append((error, error.__traceback__))
                elapsed = None
            artifacts = attempt(
                "artifact_payload_measurement",
                lambda: measure_artifact_bytes(self._artifact_files),
                ArtifactByteMeasurement((), 0),
            )
            if self.device.type == "cuda":
                cuda_device = str(self.device)
                peak_allocated = attempt(
                    "cuda_peak_allocated",
                    lambda: int(torch.cuda.max_memory_allocated(self.device)), None,
                )
                peak_reserved = attempt(
                    "cuda_peak_reserved",
                    lambda: int(torch.cuda.max_memory_reserved(self.device)), None,
                )
                for phase, value in (
                    ("cuda_peak_allocated", peak_allocated),
                    ("cuda_peak_reserved", peak_reserved),
                ):
                    if value is not None and value < 0:
                        error = RuntimeError(f"{phase} returned a negative byte count")
                        errors.append(InstrumentationError.from_exception(phase, error))
                        raised.append((error, error.__traceback__))
                        if phase == "cuda_peak_allocated":
                            peak_allocated = None
                        else:
                            peak_reserved = None
                cuda_status = CUDA_MEASURED if not any(
                    item.phase.startswith("cuda_") for item in errors
                ) else CUDA_MEASUREMENT_FAILED
            else:
                cuda_status = CUDA_NOT_APPLICABLE
                cuda_device = None
                peak_allocated = None
                peak_reserved = None
        finally:
            self._release_cuda_owner()
        exception_type = None
        exception_message = None
        if exc_type is not None:
            exception_type = f"{exc_type.__module__}.{exc_type.__qualname__}"
            try:
                exception_message = str(exc)
            except BaseException as formatting_error:
                exception_message = (
                    "<exception message unavailable: "
                    f"{type(formatting_error).__module__}."
                    f"{type(formatting_error).__qualname__}>"
                )
        assert self._start_calls is not None
        end_calls = self.call_counter.snapshot()
        boundary_calls = CallCounts(
            end_calls.forward_calls - self._start_calls.forward_calls,
            end_calls.backward_calls - self._start_calls.backward_calls,
        )
        if exc is not None and errors:
            if hasattr(exc, "add_note"):
                try:
                    exc.add_note(
                        "Stage C resource instrumentation also failed: "
                        + "; ".join(
                            f"{item.phase}: {item.exception_type}: {item.message}"
                            for item in errors
                        )
                    )
                except BaseException as annotation_error:
                    errors.append(
                        InstrumentationError.from_exception(
                            "exception_annotation", annotation_error
                        )
                    )
            retained_errors = tuple(item.to_dict() for item in errors)
            try:
                setattr(
                    exc, "stage_c_resource_instrumentation_errors", retained_errors
                )
            except BaseException:
                # The immutable ResourceRecord constructed below is authoritative.
                pass
        self._record = ResourceRecord(
            status="completed" if exc_type is None and not errors else "failed",
            wall_time_seconds=elapsed,
            cuda_status=cuda_status,
            cuda_device=cuda_device,
            cuda_peak_allocated_bytes=peak_allocated,
            cuda_peak_reserved_bytes=peak_reserved,
            calls=boundary_calls,
            artifacts=artifacts,
            exception_type=exception_type,
            exception_message=exception_message,
            instrumentation_errors=tuple(errors),
        )
        self._finished = True
        self._started = False
        if exc_type is None and raised:
            error, error_traceback = raised[0]
            raise error.with_traceback(error_traceback)
        return False


__all__ = [
    "ArtifactByteMeasurement",
    "ArtifactFileBytes",
    "CUDA_MEASURED",
    "CUDA_MEASUREMENT_FAILED",
    "CUDA_NOT_APPLICABLE",
    "CallCounter",
    "CallCounts",
    "InstrumentationError",
    "PAYLOAD_FILE_LIST_SCOPE",
    "PHASE_CLOCK",
    "PHASE_SYNCHRONIZATION",
    "PhaseTimer",
    "PhaseTiming",
    "REQUIRED_PHASES",
    "ROOT_TREE_SCOPE",
    "ResourceMeter",
    "ResourceRecord",
    "TOTAL_BOUNDARY",
    "measurement_environment",
    "measure_artifact_bytes",
    "measure_artifact_tree_bytes",
]
