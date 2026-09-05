"""Fail-closed Stage D runtime orchestration.

This module is the D4 control plane.  It deliberately cannot issue a D7
execution authorization.  Callers must supply an execution gate implemented
by D7; the production default always refuses before an output root is changed
or a backend factory is called.  D4 tests use the private synthetic context
issuer at the bottom of this file.

The runtime treats scientific artifacts as opaque.  Its durable state is one
canonical checkpoint containing the complete shard/group/unit state machine
and one failure-table row for every planned unit.  In particular, retry
decisions use attempt controls and inventory declarations, never scientific
payload values.
"""

from __future__ import annotations

import copy
import fcntl
import hashlib
import inspect
import json
import math
import os
import re
import secrets
import stat
import tempfile
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

from .stage_d_plan import (
    StageDEstimate,
    StageDExecutionPlan,
    StageDImageGroup,
    StageDPlanError,
    StageDPreflight,
    StageDUnit,
    canonical_json_bytes,
    canonical_sha256,
    validate_stage_d_estimate,
    validate_stage_d_plan,
    validate_stage_d_preflight,
    validate_stage_d_shard_count,
    validate_stage_d_shard_mapping,
)


RUNTIME_CHECKPOINT_FILENAME = "runtime_checkpoint.json"
LIFECYCLE_LOCK_FILENAME = ".stage-d-lifecycle"
CHECKPOINT_LOCK_FILENAME = ".stage-d-checkpoint"
ARTIFACT_ACTIVITY_LOCK_FILENAME = ".stage-d-artifact-activity"

RETRYABLE_INFRASTRUCTURE_FAILURE_CODES = (
    "SCHEDULER_PREEMPTION",
    "EXTERNAL_PROCESS_TERMINATION",
    "HOST_LOSS",
    "GPU_DEVICE_LOSS_OR_RESET",
    "TRANSIENT_STORAGE_IO_BEFORE_RESULT_PUBLISH",
    "WORKER_PROCESS_CRASH_BEFORE_RESULT_PUBLISH",
    "STALE_CLAIM_OWNER_CONFIRMED_DEAD",
)
NONRETRYABLE_FAILURE_CLASSES = (
    "SCIENTIFIC_INVALIDITY",
    "OUT_OF_MEMORY_FOR_FROZEN_WORKLOAD",
    "LAUNCH_SAFETY_CAP_BREACH",
    "AUTHORIZATION_OR_CONTRACT_DRIFT",
    "TAMPER_OR_UNSAFE_ROOT",
    "AMBIGUOUS_ATTEMPT_OR_OWNERSHIP",
    "MANUAL_CANCELLATION_AFTER_SCIENTIFIC_RESULT_ACCESS",
)

_SYNTHETIC_SCOPE = "SYNTHETIC_D4_TEST_ONLY"
_NOT_ISSUED = "NOT_ISSUED"
_FORMAL_EXECUTION_AUTHORIZATION = "AUTHORIZED_STAGE_D_" + "FIDELITY_SCOPE"
_CHECKPOINT_DOCUMENT_TYPE = "stage_d_runtime_checkpoint"
_RETRY_DOCUMENT_TYPE = "stage_d_runtime_retry_authorization"
_RECOVERY_DOCUMENT_TYPE = "stage_d_runtime_claim_recovery_authorization"
_SUMMARY_DOCUMENT_TYPE = "stage_d_shard_summary"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ATTEMPT_ID_RE = re.compile(r"^attempt-[0-9]{6}$")
_MAX_CHECKPOINT_BYTES = 64 * 1024 * 1024
_LOCK_NAMES = (
    LIFECYCLE_LOCK_FILENAME,
    CHECKPOINT_LOCK_FILENAME,
    ARTIFACT_ACTIVITY_LOCK_FILENAME,
)
_ACCESS_DIRECTORY_NAME = "access"
ACCESS_LIFECYCLE_FILENAME = "lifecycle.json"
ACCESS_JOURNAL_DIRECTORY = "journal"
_ROOT_ENTRIES = frozenset({RUNTIME_CHECKPOINT_FILENAME, *_LOCK_NAMES, "units"})
_ACCESS_STATES = frozenset({"ACCESS_STARTED", "RUNNABLE", "BLOCKED"})
_ACCESS_EVENTS = frozenset({
    "ACCESS_STARTED", "ASSET_ACCESS_INTENT", "ASSET_ACCESS_RESUMED",
    "ASSET_VALIDATED", "ASSET_REJECTED", "ACCESS_COMPLETE",
})
_ACCESS_OBSERVATION_KEYS = frozenset({
    "source_kind", "source_payload_sha256", "image_sha256",
    "size_bytes", "width", "height", "mode", "image_format",
    "official_ilsvrc2012_class_id", "official_wnid",
    "imagenet1k_index", "label_source",
})
_ACCESS_ASSET_KEYS = frozenset({
    "selection_order", "sample_id", "relative_path",
    "expected_image_sha256", "registry_row_digest", "observation",
})
_ACCESS_EVENT_KEYS = frozenset({
    "schema_version", "document_type", "sequence", "event",
    "previous_event_digest", "plan_digest", "runtime_binding_digest",
    "asset", "failure_code", "canonical_digest",
})
_ACCESS_LIFECYCLE_KEYS = frozenset({
    "schema_version", "document_type", "state", "plan_digest",
    "registry_digest", "ordered_sample_ids_digest",
    "runtime_binding_digest", "runtime_checkpoint_at_access_start",
    "expected_asset_count", "validated_asset_count",
    "next_selection_order", "journal_entry_count",
    "journal_head_digest", "live_assets_digest", "blocked_reasons",
    "unit_execution_started_during_initialization", "canonical_digest",
})
_UNIT_STATES = frozenset({
    "pending", "claimed", "succeeded", "scientific_failed",
    "retry_pending", "blocked",
})
_GROUP_STATES = frozenset({
    "pending", "claimed", "terminal", "retry_pending", "blocked",
})
_SHARD_STATES = frozenset({
    "pending", "leased", "completed", "retry_pending", "blocked",
})
_LIFECYCLE_STATES = frozenset({"initialized", "running", "complete", "blocked"})
_ATTEMPT_STATES = frozenset({
    "claimed", "succeeded", "scientific_failed",
    "infrastructure_failed", "blocked",
})
_RESOURCE_KEYS = (
    "elapsed_seconds",
    "forward_calls",
    "backward_calls",
    "l0_l1_payload_bytes",
    "l2_payload_bytes",
    "control_and_provenance_bytes",
    "cuda_peak_reserved_bytes",
)
_ZERO_RESOURCES: dict[str, int | float] = {
    "elapsed_seconds": 0.0,
    "forward_calls": 0,
    "backward_calls": 0,
    "l0_l1_payload_bytes": 0,
    "l2_payload_bytes": 0,
    "control_and_provenance_bytes": 0,
    "cuda_peak_reserved_bytes": 0,
}


class StageDRuntimeError(ValueError):
    """Base class for Stage D runtime control-plane failures."""


class StageDExecutionAuthorizationRequired(StageDRuntimeError):
    """Raised when D7 has not supplied a validated execution context."""


class StageDRecoveryAuthorizationRequired(StageDRuntimeError):
    """Raised when a coordinator has not authorized stale-claim recovery."""


class StageDRuntimeIntegrityError(StageDRuntimeError):
    """Raised for checkpoint, binding, or artifact-layout drift."""


class StageDRuntimeBlockedError(StageDRuntimeError):
    """Raised when the root has reached a fail-closed blocked state."""

    def __init__(self, message: str, *, reasons: Sequence[str] = ()) -> None:
        super().__init__(message)
        self.reasons = tuple(reasons)


class StageDConcurrentClaimError(StageDRuntimeBlockedError):
    """Raised for a live or stale lease/claim; D4 never steals it."""


@runtime_checkable
class ValidatedExecutionContext(Protocol):
    """Opaque context returned by a later D7 implementation.

    D4 consumes these bindings but intentionally defines no constructor for a
    formal context and knows no formal authorization token value.
    """

    formal: bool
    authorization_scope: str
    execution_authorization: str
    authorization_slot: Mapping[str, Any]
    shard_count: int
    shard_mapping: Mapping[int, int]
    plan_digest: str
    budget_digest: str
    preflight_digest: str
    code_binding: Mapping[str, Any]
    dependency_binding: Mapping[str, Any]
    model_binding: Mapping[str, Any]
    source_bindings: Mapping[str, Any]
    access_attestation: Mapping[str, bool]
    canonical_digest: str


@dataclass(frozen=True, slots=True)
class _RuntimeExecutionCapability:
    """Formal context sealed only after the runtime validates D7 output."""

    formal: bool
    authorization_scope: str
    execution_authorization: str
    authorization_slot: Mapping[str, Any]
    shard_count: int
    shard_mapping: Mapping[int, int]
    plan_digest: str
    budget_digest: str
    preflight_digest: str
    code_binding: Mapping[str, Any]
    dependency_binding: Mapping[str, Any]
    model_binding: Mapping[str, Any]
    source_bindings: Mapping[str, Any]
    access_attestation: Mapping[str, bool]
    canonical_digest: str
    _issuer: object = field(repr=False, compare=False)


_RUNTIME_CAPABILITY_ISSUER = object()


@runtime_checkable
class StageDExecutionGate(Protocol):
    """D7-owned validator consumed by D4 init and shard operations."""

    def validate_execution_context(
        self,
        context: object | None,
        *,
        operation: str,
        output_root: Path | str,
        plan: StageDExecutionPlan,
        estimate: StageDEstimate,
        preflight: StageDPreflight,
    ) -> ValidatedExecutionContext:
        ...


@runtime_checkable
class StageDRecoveryGate(Protocol):
    """Coordinator-owned validator for an abandoned claim transition."""

    def validate_recovery_authorization(
        self, authorization: object | None, *, unit: StageDUnit,
        attempt: Mapping[str, Any], owner_token_digest: str, failure_code: str,
        execution_context_digest: str, checkpoint_digest: str,
    ) -> object:
        ...


class UnavailableD7ExecutionGate:
    """Production default until D7 installs its independent gate."""

    def validate_execution_context(
        self,
        context: object | None,
        *,
        operation: str,
        output_root: Path | str,
        plan: StageDExecutionPlan,
        estimate: StageDEstimate,
        preflight: StageDPreflight,
    ) -> ValidatedExecutionContext:
        del context, operation, output_root, plan, estimate, preflight
        raise StageDExecutionAuthorizationRequired(
            "Stage D execution requires a D7 validated execution context"
        )

    __call__ = validate_execution_context


class UnavailableStageDRecoveryGate:
    """Production default: D4 cannot declare a worker dead by itself."""

    def validate_recovery_authorization(self, *_args: Any, **_kwargs: Any) -> object:
        raise StageDRecoveryAuthorizationRequired(
            "Stage D stale/crashed claim recovery requires coordinator authorization"
        )

    __call__ = validate_recovery_authorization


def _plain(value: Any, context: str = "value") -> Any:
    """Return a detached JSON value, preserving strict scalar types."""

    converter = getattr(value, "to_dict", None)
    if callable(converter) and not isinstance(value, Mapping):
        value = converter()
    if value is None or type(value) in {str, bool, int}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise StageDRuntimeIntegrityError(f"{context} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, child in value.items():
            if not isinstance(key, str) or not key:
                raise StageDRuntimeIntegrityError(
                    f"{context} keys must be non-empty strings"
                )
            result[key] = _plain(child, f"{context}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [
            _plain(child, f"{context}[{index}]")
            for index, child in enumerate(value)
        ]
    raise StageDRuntimeIntegrityError(
        f"{context} is not JSON-compatible: {type(value).__name__}"
    )


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(child) for key, child in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(child) for child in value)
    return copy.deepcopy(value)


def _require_mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise StageDRuntimeIntegrityError(f"{context} must be an object")
    return value


def _require_exact_keys(
    value: Mapping[str, Any], expected: Sequence[str] | set[str], context: str,
) -> None:
    wanted = set(expected)
    actual = set(value)
    if actual != wanted:
        raise StageDRuntimeIntegrityError(
            f"{context} fields differ: missing={sorted(wanted - actual)}, "
            f"unknown={sorted(actual - wanted)}"
        )


def _require_sha256(value: Any, context: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise StageDRuntimeIntegrityError(f"{context} must be a lowercase SHA-256")
    return value


def _require_int(
    value: Any, context: str, *, minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if type(value) is not int:
        raise StageDRuntimeIntegrityError(f"{context} must be an integer")
    if minimum is not None and value < minimum:
        raise StageDRuntimeIntegrityError(f"{context} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise StageDRuntimeIntegrityError(f"{context} must be <= {maximum}")
    return value


def _self_digest(value: Mapping[str, Any]) -> str:
    return canonical_sha256({
        key: child for key, child in value.items() if key != "canonical_digest"
    })


def _signed(value: Mapping[str, Any]) -> dict[str, Any]:
    result = _plain(value)
    result["canonical_digest"] = canonical_sha256(result)
    return result


def _context_member(context: object, names: Sequence[str], label: str) -> Any:
    for name in names:
        if isinstance(context, Mapping) and name in context:
            return context[name]
        try:
            return getattr(context, name)
        except AttributeError:
            continue
    raise StageDRuntimeIntegrityError(
        f"validated execution context lacks {label}"
    )


def _validate_control_inputs(
    plan: Any, estimate: Any, preflight: Any,
) -> tuple[StageDExecutionPlan, StageDEstimate, StageDPreflight]:
    try:
        typed_plan = validate_stage_d_plan(plan)
        typed_estimate = validate_stage_d_estimate(estimate)
        typed_preflight = validate_stage_d_preflight(preflight)
    except (StageDPlanError, TypeError, ValueError) as error:
        raise StageDRuntimeIntegrityError(
            "Stage D plan/estimate/preflight validation failed"
        ) from error
    if typed_estimate.plan_digest != typed_plan.canonical_digest:
        raise StageDRuntimeIntegrityError(
            "Stage D estimate is not bound to the supplied plan"
        )
    if (
        typed_preflight.plan_digest != typed_plan.canonical_digest
        or typed_preflight.estimate_digest != typed_estimate.canonical_digest
    ):
        raise StageDRuntimeIntegrityError(
            "Stage D preflight is not bound to the supplied plan/estimate"
        )
    if (
        typed_estimate.budget_status == "BLOCKED_NO_LAUNCH"
        or typed_preflight.status == "BLOCKED_NO_LAUNCH"
    ):
        raise StageDRuntimeBlockedError(
            "Stage D preflight blocks launch",
            reasons=tuple(typed_preflight.violations),
        )
    return typed_plan, typed_estimate, typed_preflight


def _consume_execution_gate(
    gate: StageDExecutionGate | Callable[..., Any] | None,
    context: object | None,
    *,
    operation: str,
    output_root: Path | str,
    plan: StageDExecutionPlan,
    estimate: StageDEstimate,
    preflight: StageDPreflight,
) -> object:
    selected: Any = UnavailableD7ExecutionGate() if gate is None else gate
    validator = getattr(selected, "validate_execution_context", None)
    if not callable(validator):
        validator = selected if callable(selected) else None
    if validator is None:
        raise StageDRuntimeIntegrityError(
            "execution_gate must implement validate_execution_context"
        )
    checked = validator(
        context, operation=operation, output_root=output_root, plan=plan,
        estimate=estimate, preflight=preflight,
    )
    if checked is None:
        raise StageDRuntimeIntegrityError(
            "execution gate did not return a validated execution context"
        )
    return checked


def _normalize_execution_context(
    context: object, *, plan: StageDExecutionPlan,
    estimate: StageDEstimate, preflight: StageDPreflight,
) -> dict[str, Any]:
    formal = _context_member(context, ("formal",), "formal mode")
    if type(formal) is not bool:
        raise StageDRuntimeIntegrityError(
            "validated execution context formal flag must be boolean"
        )
    scope = _context_member(
        context, ("authorization_scope", "scope"), "authorization scope"
    )
    execution_authorization = _context_member(
        context, ("execution_authorization",), "execution authorization slot"
    )
    if not isinstance(scope, str) or not scope:
        raise StageDRuntimeIntegrityError(
            "validated execution context scope must be non-empty"
        )
    if not isinstance(execution_authorization, str) or not execution_authorization:
        raise StageDRuntimeIntegrityError(
            "validated execution context authorization slot is invalid"
        )
    slot = _plain(_require_mapping(
        _context_member(context, ("authorization_slot",), "authorization slot"),
        "validated execution context authorization_slot",
    ))
    shard_count = validate_stage_d_shard_count(
        _context_member(context, ("shard_count",), "shard count")
    )
    raw_mapping = _context_member(
        context, ("shard_mapping", "mapping"), "shard mapping"
    )
    try:
        shard_mapping = validate_stage_d_shard_mapping(
            plan, shard_count, raw_mapping
        )
    except (StageDPlanError, TypeError, ValueError) as error:
        raise StageDRuntimeIntegrityError(
            "validated execution context shard mapping drifted"
        ) from error

    plan_digest = _context_member(
        context, ("plan_digest",), "plan binding"
    )
    budget_digest = _context_member(
        context, ("budget_digest", "estimate_digest"), "budget binding"
    )
    preflight_digest = _context_member(
        context, ("preflight_digest",), "preflight binding"
    )
    for value, expected, label in (
        (plan_digest, plan.canonical_digest, "plan"),
        (budget_digest, estimate.canonical_digest, "budget"),
        (preflight_digest, preflight.canonical_digest, "preflight"),
    ):
        _require_sha256(value, f"execution context {label}_digest")
        if value != expected:
            raise StageDRuntimeIntegrityError(
                f"validated execution context {label} binding drifted"
            )

    bindings: dict[str, dict[str, Any]] = {}
    for output_name, aliases in (
        ("code_binding", ("code_binding", "code")),
        ("dependency_binding", ("dependency_binding", "dependencies")),
        ("model_binding", ("model_binding", "model")),
        ("source_bindings", ("source_bindings",)),
    ):
        bindings[output_name] = _plain(_require_mapping(
            _context_member(context, aliases, output_name),
            f"validated execution context {output_name}",
        ))

    access = _plain(_require_mapping(
        _context_member(
            context, ("access_attestation",), "access attestation"
        ),
        "validated execution context access_attestation",
    ))
    synthetic_access = {
        "reserve_image_bytes_accessed": False,
        "model_accessed": False,
        "cuda_accessed": False,
        "outcome_accessed": False,
    }
    formal_access = {
        "reserve_image_bytes_accessed": False,
        "model_loaded_for_identity_verification": True,
        "model_used_for_inference": False,
        "cuda_accessed": False,
        "outcome_accessed": False,
    }
    if not formal:
        if (
            scope != _SYNTHETIC_SCOPE
            or execution_authorization != _NOT_ISSUED
            or access != synthetic_access
        ):
            raise StageDRuntimeIntegrityError(
                "synthetic D4 context crossed its non-formal access boundary"
            )
    elif (
        execution_authorization != _FORMAL_EXECUTION_AUTHORIZATION
        or isinstance(context, Mapping) or access != formal_access
    ):
        raise StageDExecutionAuthorizationRequired(
            "formal Stage D context is not an opaque D7-gate result"
        )

    supplied_digest = _context_member(
        context, ("canonical_digest", "context_digest"), "context digest"
    )
    _require_sha256(supplied_digest, "validated execution context digest")
    context_payload = {
        "formal": formal,
        "authorization_scope": scope,
        "execution_authorization": execution_authorization,
        "authorization_slot": slot,
        "shard_count": shard_count,
        "shard_mapping": [shard_mapping[index] for index in range(500)],
        "plan_digest": plan_digest,
        "budget_digest": budget_digest,
        "preflight_digest": preflight_digest,
        **bindings,
        "access_attestation": access,
    }
    if supplied_digest != canonical_sha256(context_payload):
        raise StageDRuntimeIntegrityError(
            "validated execution context canonical digest mismatch"
        )
    projection = {
        "formal": formal,
        "authorization_scope": scope,
        "execution_authorization": execution_authorization,
        "authorization_slot_digest": canonical_sha256(slot),
        "shard_count": shard_count,
        "shard_mapping": [shard_mapping[index] for index in range(500)],
        "plan_digest": plan_digest,
        "budget_digest": budget_digest,
        "preflight_digest": preflight_digest,
        "code_binding_digest": canonical_sha256(bindings["code_binding"]),
        "dependency_binding_digest": canonical_sha256(
            bindings["dependency_binding"]
        ),
        "model_binding_digest": canonical_sha256(bindings["model_binding"]),
        "source_bindings_digest": canonical_sha256(bindings["source_bindings"]),
        "access_attestation": access,
        "gate_context_digest": supplied_digest,
        "provenance_bindings": {
            "d7_authorization": (
                {"status": _NOT_ISSUED, "digest": None}
                if not formal else {
                    "status": _FORMAL_EXECUTION_AUTHORIZATION,
                    "digest": canonical_sha256(slot),
                }
            ),
            "producer": bindings["code_binding"],
            "dependencies": bindings["dependency_binding"],
            "model": bindings["model_binding"],
            "source_bindings": bindings["source_bindings"],
        },
    }
    projection["runtime_binding_digest"] = canonical_sha256(projection)
    return projection


def _is_runtime_execution_capability(context: object) -> bool:
    """Recognize only issuer-backed contexts accepted by the D4 producer."""

    if isinstance(context, _RuntimeExecutionCapability):
        return context._issuer is _RUNTIME_CAPABILITY_ISSUER
    synthetic_type = globals().get("_SyntheticD4ExecutionContext")
    synthetic_issuer = globals().get("_SYNTHETIC_ISSUER")
    return (
        synthetic_type is not None
        and isinstance(context, synthetic_type)
        and getattr(context, "_issuer", None) is synthetic_issuer
    )


def _seal_execution_context(
    context: object, context_binding: Mapping[str, Any],
) -> object:
    """Seal a fully normalized gate result before backend construction."""

    if not context_binding["formal"]:
        if not _is_runtime_execution_capability(context):
            raise StageDExecutionAuthorizationRequired(
                "synthetic D4 context was not issued by the private test gate"
            )
        return context
    values = {
        "formal": True,
        "authorization_scope": _context_member(
            context, ("authorization_scope", "scope"), "authorization scope"
        ),
        "execution_authorization": _context_member(
            context, ("execution_authorization",), "execution authorization slot"
        ),
        "authorization_slot": _freeze(_plain(_context_member(
            context, ("authorization_slot",), "authorization slot"
        ))),
        "shard_count": context_binding["shard_count"],
        "shard_mapping": MappingProxyType({
            index: context_binding["shard_mapping"][index]
            for index in range(500)
        }),
        "plan_digest": context_binding["plan_digest"],
        "budget_digest": context_binding["budget_digest"],
        "preflight_digest": context_binding["preflight_digest"],
        "code_binding": _freeze(_plain(_context_member(
            context, ("code_binding", "code"), "code_binding"
        ))),
        "dependency_binding": _freeze(_plain(_context_member(
            context, ("dependency_binding", "dependencies"),
            "dependency_binding",
        ))),
        "model_binding": _freeze(_plain(_context_member(
            context, ("model_binding", "model"), "model_binding"
        ))),
        "source_bindings": _freeze(_plain(_context_member(
            context, ("source_bindings",), "source_bindings"
        ))),
        "access_attestation": _freeze(_plain(_context_member(
            context, ("access_attestation",), "access attestation"
        ))),
        "canonical_digest": context_binding["gate_context_digest"],
    }
    return _RuntimeExecutionCapability(
        **values, _issuer=_RUNTIME_CAPABILITY_ISSUER
    )


def _absolute_lexical_root(value: Path | str, context: str) -> Path:
    raw = os.fspath(value)
    if not isinstance(raw, str) or not raw or "\x00" in raw or "\\" in raw:
        raise StageDRuntimeIntegrityError(
            f"{context} must be a non-empty absolute POSIX path"
        )
    pure = PurePosixPath(raw)
    if not pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts[1:]):
        raise StageDRuntimeIntegrityError(
            f"{context} must be a normalized absolute path"
        )
    normalized = os.path.normpath(raw)
    if normalized != raw or normalized == os.path.sep:
        raise StageDRuntimeIntegrityError(
            f"{context} must be a normalized, non-root absolute path"
        )
    return Path(raw)


def _reject_symlink_components(path: Path, *, include_leaf: bool) -> None:
    current = Path(path.anchor)
    parts = path.parts[1:] if include_leaf else path.parts[1:-1]
    for component in parts:
        current /= component
        try:
            metadata = os.lstat(current)
        except FileNotFoundError:
            raise StageDRuntimeIntegrityError(
                f"unsafe root path has a missing parent: {current}"
            ) from None
        except OSError as error:
            raise StageDRuntimeIntegrityError(
                f"cannot inspect root path component: {current}"
            ) from error
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise StageDRuntimeIntegrityError(
                f"root path contains a symlink or non-directory: {current}"
            )


def _fresh_root(value: Path | str) -> Path:
    root = _absolute_lexical_root(value, "Stage D output root")
    _reject_symlink_components(root, include_leaf=False)
    if os.path.lexists(root):
        raise StageDRuntimeBlockedError(
            "Stage D output root must be a previously nonexistent path"
        )
    return root


def _existing_root(value: Path | str) -> Path:
    root = _absolute_lexical_root(value, "Stage D output root")
    _reject_symlink_components(root, include_leaf=True)
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        descriptor = os.open(root, flags)
    except OSError as error:
        raise StageDRuntimeIntegrityError(
            "Stage D output root is missing or unsafe"
        ) from error
    try:
        opened = os.fstat(descriptor)
        named = os.lstat(root)
        if (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino):
            raise StageDRuntimeIntegrityError(
                "Stage D output root changed while being opened"
            )
    finally:
        os.close(descriptor)
    return root


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _create_lock_file(path: Path) -> None:
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise StageDRuntimeIntegrityError(
                f"Stage D lock is not a single-link regular file: {path.name}"
            )
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def _safe_lock(
    root: Path, name: str, *, exclusive: bool, blocking: bool = True,
) -> Iterator[None]:
    if name not in _LOCK_NAMES:
        raise StageDRuntimeIntegrityError("unknown Stage D lock name")
    path = root / name
    flags = os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise StageDRuntimeIntegrityError(
            f"cannot safely open Stage D lock {name}"
        ) from error
    locked = False
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise StageDRuntimeIntegrityError(
                f"Stage D lock {name} must be a single-link regular file"
            )
        operation = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        if not blocking:
            operation |= fcntl.LOCK_NB
        try:
            fcntl.flock(descriptor, operation)
        except BlockingIOError as error:
            raise StageDConcurrentClaimError(
                f"Stage D lock {name} is already held"
            ) from error
        locked = True
        after = os.fstat(descriptor)
        named = os.lstat(path)
        if (
            (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino)
            or (after.st_dev, after.st_ino) != (named.st_dev, named.st_ino)
            or not stat.S_ISREG(named.st_mode)
            or named.st_nlink != 1
        ):
            raise StageDRuntimeIntegrityError(
                f"Stage D lock {name} changed while being acquired"
            )
        yield
    finally:
        if locked:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _lifecycle_lock(root: Path, *, exclusive: bool) -> Iterator[None]:
    return _safe_lock(root, LIFECYCLE_LOCK_FILENAME, exclusive=exclusive)


def _checkpoint_lock(root: Path) -> Iterator[None]:
    return _safe_lock(root, CHECKPOINT_LOCK_FILENAME, exclusive=True)


def _artifact_activity_lock(root: Path, *, exclusive: bool) -> Iterator[None]:
    return _safe_lock(root, ARTIFACT_ACTIVITY_LOCK_FILENAME, exclusive=exclusive)


def _checkpoint_bytes(value: Mapping[str, Any]) -> bytes:
    return canonical_json_bytes(value) + b"\n"


def _atomic_create_checkpoint(path: Path, value: Mapping[str, Any]) -> None:
    payload = _checkpoint_bytes(value)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(path.parent)
    except BaseException:
        # A partially initialized execution root is retained as failure
        # evidence.  It is never silently cleaned up or reused.
        raise


def _atomic_replace_checkpoint(path: Path, value: Mapping[str, Any]) -> None:
    try:
        current = os.lstat(path)
    except OSError as error:
        raise StageDRuntimeIntegrityError(
            "runtime checkpoint disappeared before replacement"
        ) from error
    if (
        not stat.S_ISREG(current.st_mode)
        or current.st_nlink != 1
        or stat.S_ISLNK(current.st_mode)
    ):
        raise StageDRuntimeIntegrityError(
            "runtime checkpoint is not a single-link regular file"
        )
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=".stage-d-checkpoint-write-", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_checkpoint_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _atomic_replace_control_json(path: Path, value: Mapping[str, Any]) -> None:
    """Atomically replace an existing canonical JSON control file."""

    current = os.lstat(path)
    if not stat.S_ISREG(current.st_mode) or current.st_nlink != 1:
        raise StageDRuntimeIntegrityError(
            "attempt control is not a single-link regular file"
        )
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.recovery-", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_json_bytes(value) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _strict_checkpoint_json(path: Path) -> dict[str, Any]:
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise StageDRuntimeIntegrityError(
            "cannot safely open runtime checkpoint"
        ) from error
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size > _MAX_CHECKPOINT_BYTES
        ):
            raise StageDRuntimeIntegrityError(
                "runtime checkpoint is not a bounded single-link regular file"
            )
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            block = os.read(descriptor, min(1024 * 1024, remaining))
            if not block:
                raise StageDRuntimeIntegrityError(
                    "runtime checkpoint was truncated while being read"
                )
            chunks.append(block)
            remaining -= len(block)
        after = os.fstat(descriptor)
        named = os.lstat(path)
        identity = lambda item: (
            item.st_dev, item.st_ino, item.st_mode, item.st_nlink, item.st_size,
            item.st_mtime_ns, item.st_ctime_ns,
        )
        if identity(before) != identity(after) or (
            after.st_dev, after.st_ino
        ) != (named.st_dev, named.st_ino):
            raise StageDRuntimeIntegrityError(
                "runtime checkpoint changed while being read"
            )
    finally:
        os.close(descriptor)
    payload = b"".join(chunks)

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, child in items:
            if key in result:
                raise StageDRuntimeIntegrityError(
                    f"duplicate runtime checkpoint key: {key}"
                )
            result[key] = child
        return result

    try:
        value = json.loads(
            payload.decode("utf-8"), object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite number {token}")
            ),
        )
    except StageDRuntimeIntegrityError:
        raise
    except Exception as error:
        raise StageDRuntimeIntegrityError(
            "runtime checkpoint is not strict UTF-8 JSON"
        ) from error
    if not isinstance(value, dict):
        raise StageDRuntimeIntegrityError(
            "runtime checkpoint root must be an object"
        )
    if payload != _checkpoint_bytes(value):
        raise StageDRuntimeIntegrityError(
            "runtime checkpoint bytes are not canonical"
        )
    return value


def _unit_identity(unit: StageDUnit) -> dict[str, Any]:
    return {
        "unit_digest": unit.canonical_digest,
        "group_selection_order": unit.selection_order,
        "sample_id": unit.sample_id,
        "method_index": unit.method_index,
        "method_id": unit.method_id,
        "pair_binding_digest": unit.pair_binding_digest,
    }


def _pending_unit_row(group: StageDImageGroup, unit: StageDUnit) -> dict[str, Any]:
    return {
        **_unit_identity(unit),
        "group_digest": group.canonical_digest,
        "state": "pending",
        "terminal": False,
        "owner_token_digest": None,
        "attempts": [],
        "retry_authorization_digest": None,
        "failure_class": None,
        "failure_code": None,
        "resources": dict(_ZERO_RESOURCES),
    }


def _failure_rows(units: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "unit_digest": row["unit_digest"],
            "group_selection_order": row["group_selection_order"],
            "method_index": row["method_index"],
            "state": row["state"],
            "terminal": row["terminal"],
            "attempt_count": len(row["attempts"]),
            "failure_class": row["failure_class"],
            "failure_code": row["failure_code"],
        }
        for row in units
    ]


def _sum_resources(units: Sequence[Mapping[str, Any]]) -> dict[str, int | float]:
    result: dict[str, int | float] = dict(_ZERO_RESOURCES)
    for row in units:
        resources = row["resources"]
        for key in _RESOURCE_KEYS:
            if key == "cuda_peak_reserved_bytes":
                result[key] = max(result[key], resources[key])
            else:
                result[key] += resources[key]
    return result


def _lifecycle_projection(
    state: str, shards: Sequence[Mapping[str, Any]],
    groups: Sequence[Mapping[str, Any]], units: Sequence[Mapping[str, Any]],
    blocked_reasons: Sequence[str],
) -> dict[str, Any]:
    return {
        "state": state,
        "active_shard_leases": sum(row["state"] == "leased" for row in shards),
        "active_group_claims": sum(row["state"] == "claimed" for row in groups),
        "terminal_groups": sum(row["terminal"] is True for row in groups),
        "terminal_units": sum(row["terminal"] is True for row in units),
        "scientific_failure_units": sum(
            row["state"] == "scientific_failed" for row in units
        ),
        "retry_pending_units": sum(
            row["state"] == "retry_pending" for row in units
        ),
        "blocked_units": sum(row["state"] == "blocked" for row in units),
        "blocked_reasons": list(blocked_reasons),
    }


def _expected_lifecycle_state(
    state: str, generation: int, shards: Sequence[Mapping[str, Any]],
) -> str:
    if state == "blocked":
        return "blocked"
    if all(row["terminal"] for row in shards):
        return "complete"
    if generation == 0 and all(row["state"] == "pending" for row in shards):
        return "initialized"
    return "running"


def _initial_checkpoint(
    plan: StageDExecutionPlan, estimate: StageDEstimate,
    preflight: StageDPreflight, context_binding: Mapping[str, Any],
) -> dict[str, Any]:
    count = context_binding["shard_count"]
    mapping = context_binding["shard_mapping"]
    groups = [
        {
            "group_digest": group.canonical_digest,
            "selection_order": group.selection_order,
            "sample_id": group.sample_id,
            "shard_id": mapping[group.selection_order],
            "state": "pending",
            "terminal": False,
            "owner_token_digest": None,
            "unit_digests": [unit.canonical_digest for unit in group.units],
        }
        for group in plan.groups
    ]
    units = [
        _pending_unit_row(group, unit)
        for group in plan.groups for unit in group.units
    ]
    by_shard = {index: [] for index in range(count)}
    for group in groups:
        by_shard[group["shard_id"]].append(group["group_digest"])
    shards = [
        {
            "shard_id": shard_id,
            "state": "pending",
            "terminal": False,
            "owner_token_digest": None,
            "group_count": len(by_shard[shard_id]),
            "group_digests": by_shard[shard_id],
        }
        for shard_id in range(count)
    ]
    lifecycle = _lifecycle_projection(
        "initialized", shards, groups, units, ()
    )
    return _signed({
        "schema_version": 1,
        "document_type": _CHECKPOINT_DOCUMENT_TYPE,
        "generation": 0,
        "previous_checkpoint_digest": None,
        "plan_digest": plan.canonical_digest,
        "estimate_digest": estimate.canonical_digest,
        "preflight_digest": preflight.canonical_digest,
        "execution_context": _plain(context_binding),
        "lifecycle": lifecycle,
        "shards": shards,
        "groups": groups,
        "units": units,
        "failure_rows": _failure_rows(units),
        "resources": _sum_resources(units),
    })


def _access_start_checkpoint_digest(
    plan: StageDExecutionPlan, estimate: StageDEstimate,
    preflight: StageDPreflight, context_binding: Mapping[str, Any],
) -> str:
    """Recompute the immutable checkpoint identity bound by D8.0.

    The live runtime checkpoint advances after every lease, claim and terminal
    publication.  The access lifecycle intentionally remains bound to the
    generation-zero checkpoint that existed before the first reserve access,
    so later shards must compare it with this deterministic reconstruction
    rather than with the current mutable checkpoint.
    """

    return _initial_checkpoint(
        plan, estimate, preflight, context_binding,
    )["canonical_digest"]


_CHECKPOINT_KEYS = {
    "schema_version", "document_type", "generation",
    "previous_checkpoint_digest", "plan_digest", "estimate_digest",
    "preflight_digest", "execution_context", "lifecycle", "shards",
    "groups", "units", "failure_rows", "resources",
    "canonical_digest",
}
_CONTEXT_KEYS = {
    "formal", "authorization_scope", "execution_authorization",
    "authorization_slot_digest", "shard_count", "shard_mapping",
    "plan_digest", "budget_digest", "preflight_digest",
    "code_binding_digest", "dependency_binding_digest",
    "model_binding_digest", "source_bindings_digest",
    "access_attestation", "gate_context_digest", "runtime_binding_digest",
    "provenance_bindings",
}
_SHARD_KEYS = {
    "shard_id", "state", "terminal", "owner_token_digest",
    "group_count", "group_digests",
}
_GROUP_KEYS = {
    "group_digest", "selection_order", "sample_id", "shard_id",
    "state", "terminal", "owner_token_digest", "unit_digests",
}
_UNIT_KEYS = {
    "unit_digest", "group_selection_order", "sample_id",
    "method_index", "method_id", "pair_binding_digest",
    "group_digest", "state", "terminal", "owner_token_digest",
    "attempts", "retry_authorization_digest", "failure_class",
    "failure_code", "resources",
}
_ATTEMPT_KEYS = {
    "attempt_id", "attempt_number", "attempt_path", "state",
    "owner_token_digest", "status_sha256", "attempt_inventory_digest",
    "failure_class",
    "failure_code", "result_manifest_published",
    "published_l0_l1_payload_count", "retry_authorization_digest",
    "resources", "canonical_digest",
}
_FAILURE_ROW_KEYS = {
    "unit_digest", "group_selection_order", "method_index", "state",
    "terminal", "attempt_count", "failure_class", "failure_code",
}
_LIFECYCLE_KEYS = {
    "state", "active_shard_leases", "active_group_claims",
    "terminal_groups", "terminal_units", "scientific_failure_units",
    "retry_pending_units", "blocked_units", "blocked_reasons",
}


def _validate_resources(value: Any, context: str) -> dict[str, int | float]:
    document = _plain(_require_mapping(value, context), context)
    _require_exact_keys(document, set(_RESOURCE_KEYS), context)
    for key in _RESOURCE_KEYS:
        item = document[key]
        if key == "elapsed_seconds":
            if type(item) not in {int, float} or not math.isfinite(float(item)) or item < 0:
                raise StageDRuntimeIntegrityError(
                    f"{context}.{key} must be a finite non-negative number"
                )
            document[key] = float(item)
        elif type(item) is not int or item < 0:
            raise StageDRuntimeIntegrityError(
                f"{context}.{key} must be a non-negative integer"
            )
    return document


def _validate_attempt(
    value: Any, *, root: Path, plan: StageDExecutionPlan,
    group: StageDImageGroup, unit: StageDUnit,
    context_binding: Mapping[str, Any], attempt_index: int,
    allow_published_claim: bool = False,
) -> dict[str, Any]:
    attempt = _plain(_require_mapping(value, "checkpoint attempt"))
    _require_exact_keys(attempt, _ATTEMPT_KEYS, "checkpoint attempt")
    expected_number = attempt_index + 1
    expected_id = f"attempt-{expected_number:06d}"
    if (
        attempt["attempt_id"] != expected_id
        or attempt["attempt_number"] != expected_number
        or attempt["state"] not in _ATTEMPT_STATES
    ):
        raise StageDRuntimeIntegrityError(
            "checkpoint attempt identity/order/state drifted"
        )
    expected_path = root / "units" / unit.canonical_digest / "attempts" / expected_id
    if attempt["attempt_path"] != str(expected_path):
        raise StageDRuntimeIntegrityError(
            "checkpoint attempt path escaped its unit"
        )
    _require_sha256(attempt["owner_token_digest"], "attempt owner token digest")
    status_sha = attempt["status_sha256"]
    if status_sha is not None:
        _require_sha256(status_sha, "attempt status SHA-256")
    inventory_digest = attempt["attempt_inventory_digest"]
    if inventory_digest is not None:
        _require_sha256(inventory_digest, "attempt inventory digest")
    retry_digest = attempt["retry_authorization_digest"]
    if retry_digest is not None:
        _require_sha256(retry_digest, "attempt retry authorization digest")
    if type(attempt["result_manifest_published"]) is not bool:
        raise StageDRuntimeIntegrityError(
            "attempt result_manifest_published must be boolean"
        )
    _require_int(
        attempt["published_l0_l1_payload_count"],
        "attempt published payload count", minimum=0,
    )
    attempt["resources"] = _validate_resources(
        attempt["resources"], "attempt resources"
    )
    if attempt["canonical_digest"] != _self_digest(attempt):
        raise StageDRuntimeIntegrityError(
            "checkpoint attempt canonical digest mismatch"
        )
    if attempt["state"] == "claimed":
        if any(attempt[key] is not None for key in (
            "status_sha256", "attempt_inventory_digest",
            "failure_class", "failure_code"
        )) or attempt["result_manifest_published"] or attempt["published_l0_l1_payload_count"]:
            raise StageDRuntimeIntegrityError(
                "claimed attempt contains published result evidence"
            )
        if expected_path.exists():
            if allow_published_claim:
                try:
                    _source_validate_attempt(
                        expected_path, plan=plan, group=group, unit=unit,
                        context_binding=context_binding,
                    )
                except StageDRuntimeIntegrityError:
                    phase = _inspect_recovery_attempt(
                        expected_path, group=group, unit=unit
                    )["phase"]
                    if phase not in {
                        "empty", "running", "manifest_published",
                        "infrastructure_published",
                    }:
                        raise StageDRuntimeIntegrityError(
                            "claimed checkpoint attempt has an invalid transition state"
                        )
            else:
                observed = _source_validate_open_attempt(
                    expected_path, plan=plan, group=group, unit=unit,
                    context_binding=context_binding,
                )
                if observed["state"] != "claimed":
                    raise StageDRuntimeIntegrityError(
                        "claimed checkpoint attempt has published completion evidence"
                    )
    elif attempt["state"] == "abandoned":
        raise StageDRuntimeIntegrityError(
            "legacy abandoned attempt state is not accepted"
        )
    elif attempt["status_sha256"] is None or inventory_digest is None:
        raise StageDRuntimeIntegrityError(
            "finished attempt lacks status/inventory digests"
        )
    else:
        observed = _source_validate_attempt(
            expected_path, plan=plan, group=group, unit=unit,
            context_binding=context_binding,
        )
        for key in (
            "state", "attempt_id", "attempt_path", "status_sha256",
            "attempt_inventory_digest", "failure_class", "failure_code",
            "result_manifest_published", "published_l0_l1_payload_count",
            "resources",
        ):
            if attempt[key] != observed[key]:
                raise StageDRuntimeIntegrityError(
                    f"checkpoint attempt {key} disagrees with source evidence"
                )
    return attempt


def _attempt_inventory_digest(path: Path) -> str:
    """Hash an attempt's closed file inventory without parsing payload values."""

    try:
        metadata = os.lstat(path)
    except OSError as error:
        raise StageDRuntimeIntegrityError(
            "claimed attempt directory is missing"
        ) from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise StageDRuntimeIntegrityError("attempt path is not a safe directory")
    records: list[dict[str, Any]] = []
    for directory, directory_names, file_names in os.walk(path, followlinks=False):
        directory_path = Path(directory)
        for name in sorted(directory_names):
            child = directory_path / name
            info = os.lstat(child)
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                raise StageDRuntimeIntegrityError(
                    "attempt inventory contains an unsafe directory"
                )
        for name in sorted(file_names):
            child = directory_path / name
            info = os.lstat(child)
            if (
                stat.S_ISLNK(info.st_mode)
                or not stat.S_ISREG(info.st_mode)
                or info.st_nlink != 1
            ):
                raise StageDRuntimeIntegrityError(
                    "attempt inventory contains an unsafe file"
                )
            flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
            descriptor = os.open(child, flags)
            try:
                opened = os.fstat(descriptor)
                digest = hashlib.sha256()
                observed = 0
                while True:
                    block = os.read(descriptor, 1024 * 1024)
                    if not block:
                        break
                    observed += len(block)
                    digest.update(block)
                after = os.fstat(descriptor)
                if (
                    (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns)
                    != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
                    or observed != opened.st_size
                ):
                    raise StageDRuntimeIntegrityError(
                        "attempt file changed while being inventoried"
                    )
            finally:
                os.close(descriptor)
            records.append({
                "path": child.relative_to(path).as_posix(),
                "bytes": info.st_size, "sha256": digest.hexdigest(),
            })
    return canonical_sha256({"files": records})


def _single_regular_file_sha256(path: Path, context: str) -> str:
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise StageDRuntimeIntegrityError(f"cannot safely open {context}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise StageDRuntimeIntegrityError(
                f"{context} must be a single-link regular file"
            )
        digest = hashlib.sha256()
        observed = 0
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            observed += len(block)
            digest.update(block)
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
        ) or observed != before.st_size:
            raise StageDRuntimeIntegrityError(f"{context} changed while hashing")
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def _strict_control_json(path: Path, context: str) -> dict[str, Any]:
    """Read one canonical, descriptor-bound control document."""

    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise StageDRuntimeIntegrityError(f"cannot safely open {context}") from error
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size > 16 * 1024 * 1024
        ):
            raise StageDRuntimeIntegrityError(
                f"{context} is not a bounded single-link regular file"
            )
        payload = b""
        while len(payload) < before.st_size:
            block = os.read(descriptor, min(1024 * 1024, before.st_size - len(payload)))
            if not block:
                break
            payload += block
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
        ) or len(payload) != before.st_size:
            raise StageDRuntimeIntegrityError(f"{context} changed while being read")
    finally:
        os.close(descriptor)

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, child in items:
            if key in result:
                raise StageDRuntimeIntegrityError(f"duplicate key in {context}: {key}")
            result[key] = child
        return result

    try:
        value = json.loads(
            payload.decode("utf-8"), object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite number {token}")
            ),
        )
    except StageDRuntimeIntegrityError:
        raise
    except Exception as error:
        raise StageDRuntimeIntegrityError(f"{context} is invalid JSON") from error
    if not isinstance(value, dict) or payload != canonical_json_bytes(value) + b"\n":
        raise StageDRuntimeIntegrityError(f"{context} is not canonical JSON")
    return value


def _atomic_create_control_json(path: Path, value: Mapping[str, Any]) -> None:
    """Create one immutable canonical control and fsync its directory."""

    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    _atomic_create_checkpoint(path, value)


def _access_event(
    *, sequence: int, event: str, previous: str | None, plan_digest: str,
    runtime_binding_digest: str, asset: Mapping[str, Any] | None = None,
    failure_code: str | None = None,
) -> dict[str, Any]:
    if event not in _ACCESS_EVENTS:
        raise StageDRuntimeIntegrityError("unknown Stage D access event")
    return _signed({
        "schema_version": 1,
        "document_type": "stage_d_access_journal_event",
        "sequence": sequence,
        "event": event,
        "previous_event_digest": previous,
        "plan_digest": plan_digest,
        "runtime_binding_digest": runtime_binding_digest,
        "asset": None if asset is None else _plain(asset),
        "failure_code": failure_code,
    })


def _access_lifecycle(
    *, state: str, plan: StageDExecutionPlan, runtime_binding_digest: str,
    checkpoint_digest: str, validated_asset_count: int,
    journal_entry_count: int, journal_head_digest: str | None,
    live_assets_digest: str | None, blocked_reasons: Sequence[str] = (),
) -> dict[str, Any]:
    if state not in _ACCESS_STATES:
        raise StageDRuntimeIntegrityError("unknown Stage D access lifecycle state")
    return _signed({
        "schema_version": 1,
        "document_type": "stage_d_access_lifecycle",
        "state": state,
        "plan_digest": plan.canonical_digest,
        "registry_digest": plan.registry_digest,
        "ordered_sample_ids_digest": plan.ordered_sample_ids_digest,
        "runtime_binding_digest": runtime_binding_digest,
        "runtime_checkpoint_at_access_start": checkpoint_digest,
        "expected_asset_count": len(plan.groups),
        "validated_asset_count": validated_asset_count,
        "next_selection_order": (
            validated_asset_count if validated_asset_count < len(plan.groups) else None
        ),
        "journal_entry_count": journal_entry_count,
        "journal_head_digest": journal_head_digest,
        "live_assets_digest": live_assets_digest,
        "blocked_reasons": list(blocked_reasons),
        "unit_execution_started_during_initialization": False,
    })


def _validate_access_event(
    value: Any, *, sequence: int, previous: str | None,
    plan_digest: str, runtime_binding_digest: str,
) -> dict[str, Any]:
    event = _plain(_require_mapping(value, "access journal event"))
    _require_exact_keys(event, _ACCESS_EVENT_KEYS, "access journal event")
    if (
        event["schema_version"] != 1
        or event["document_type"] != "stage_d_access_journal_event"
        or event["sequence"] != sequence
        or event["event"] not in _ACCESS_EVENTS
        or event["previous_event_digest"] != previous
        or event["plan_digest"] != plan_digest
        or event["runtime_binding_digest"] != runtime_binding_digest
        or event["canonical_digest"] != _self_digest(event)
    ):
        raise StageDRuntimeIntegrityError("access journal event binding drifted")
    if event["event"] in {"ACCESS_STARTED", "ACCESS_COMPLETE"}:
        if event["asset"] is not None or event["failure_code"] is not None:
            raise StageDRuntimeIntegrityError("boundary access event has asset data")
    else:
        asset = _plain(_require_mapping(event["asset"], "access event asset"))
        _require_exact_keys(asset, _ACCESS_ASSET_KEYS, "access event asset")
        if event["event"] == "ASSET_REJECTED":
            if not isinstance(event["failure_code"], str) or not event["failure_code"]:
                raise StageDRuntimeIntegrityError("rejected asset lacks a failure code")
        elif event["failure_code"] is not None:
            raise StageDRuntimeIntegrityError("non-rejected access event has a failure")
    return event


def _read_access_chain(
    root: Path, *, plan: StageDExecutionPlan, runtime_binding_digest: str,
) -> list[dict[str, Any]]:
    journal = root / _ACCESS_DIRECTORY_NAME / ACCESS_JOURNAL_DIRECTORY
    try:
        children = sorted(journal.iterdir(), key=lambda item: item.name)
    except OSError as error:
        raise StageDRuntimeIntegrityError("access journal is missing or unsafe") from error
    expected_names = [f"{index:06d}.json" for index in range(len(children))]
    if [child.name for child in children] != expected_names:
        raise StageDRuntimeIntegrityError("access journal is not a closed sequence")
    events: list[dict[str, Any]] = []
    previous: str | None = None
    for index, child in enumerate(children):
        event = _validate_access_event(
            _strict_control_json(child, f"access journal entry {index}"),
            sequence=index, previous=previous, plan_digest=plan.canonical_digest,
            runtime_binding_digest=runtime_binding_digest,
        )
        events.append(event)
        previous = event["canonical_digest"]
    return events


def _validate_access_layout(root: Path) -> None:
    access_root = root / _ACCESS_DIRECTORY_NAME
    journal = access_root / ACCESS_JOURNAL_DIRECTORY
    try:
        access_info = os.lstat(access_root)
        journal_info = os.lstat(journal)
        lifecycle_info = os.lstat(access_root / ACCESS_LIFECYCLE_FILENAME)
    except OSError as error:
        raise StageDRuntimeIntegrityError("access controls are missing or unsafe") from error
    if (
        stat.S_ISLNK(access_info.st_mode) or not stat.S_ISDIR(access_info.st_mode)
        or stat.S_ISLNK(journal_info.st_mode) or not stat.S_ISDIR(journal_info.st_mode)
        or stat.S_ISLNK(lifecycle_info.st_mode)
        or not stat.S_ISREG(lifecycle_info.st_mode)
        or lifecycle_info.st_nlink != 1
        or {child.name for child in access_root.iterdir()}
        != {ACCESS_LIFECYCLE_FILENAME, ACCESS_JOURNAL_DIRECTORY}
    ):
        raise StageDRuntimeIntegrityError("access control layout is not closed")
    for child in journal.iterdir():
        info = os.lstat(child)
        if (
            stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
        ):
            raise StageDRuntimeIntegrityError("access journal entry is unsafe")


def _validated_access_rows(
    events: Sequence[Mapping[str, Any]], plan: StageDExecutionPlan,
    contracts: Sequence[Mapping[str, Any]] | None = None,
    *, require_live_identity: bool = True,
) -> tuple[list[dict[str, Any]], int | None, str | None]:
    if not events or events[0]["event"] != "ACCESS_STARTED":
        raise StageDRuntimeIntegrityError("access journal lacks ACCESS_STARTED")
    validated: list[dict[str, Any]] = []
    pending: int | None = None
    failure: str | None = None
    complete = False
    for event in events[1:]:
        kind = event["event"]
        if complete or failure is not None:
            raise StageDRuntimeIntegrityError("access journal continued after terminal event")
        if kind in {"ASSET_ACCESS_INTENT", "ASSET_ACCESS_RESUMED"}:
            if len(validated) >= len(plan.groups):
                raise StageDRuntimeIntegrityError("access intent sequence is ambiguous")
            asset = event["asset"]
            group = plan.groups[len(validated)]
            if (
                asset["selection_order"] != group.selection_order
                or asset["sample_id"] != group.sample_id
                or asset["relative_path"] != group.relative_path
                or asset["expected_image_sha256"] != group.image_sha256
                or asset["registry_row_digest"] != group.registry_row_digest
                or asset["observation"] is not None
            ):
                raise StageDRuntimeIntegrityError("access intent differs from frozen plan")
            if (kind == "ASSET_ACCESS_INTENT" and pending is not None) or (
                kind == "ASSET_ACCESS_RESUMED" and pending != group.selection_order
            ):
                raise StageDRuntimeIntegrityError("access resume sequence is ambiguous")
            pending = group.selection_order
        elif kind in {"ASSET_VALIDATED", "ASSET_REJECTED"}:
            if pending != len(validated):
                raise StageDRuntimeIntegrityError("access result lacks matching durable intent")
            asset = event["asset"]
            group = plan.groups[pending]
            observation = asset["observation"]
            if (
                asset["selection_order"] != group.selection_order
                or asset["sample_id"] != group.sample_id
                or asset["relative_path"] != group.relative_path
                or asset["expected_image_sha256"] != group.image_sha256
                or asset["registry_row_digest"] != group.registry_row_digest
            ):
                raise StageDRuntimeIntegrityError("access result differs from frozen plan")
            if kind == "ASSET_VALIDATED":
                observed = _plain(_require_mapping(observation, "asset observation"))
                _require_exact_keys(observed, _ACCESS_OBSERVATION_KEYS, "asset observation")
                expected_metadata = (
                    {} if contracts is None else {key: contracts[pending][key] for key in (
                        "size_bytes", "width", "height", "mode",
                        "image_format", "official_ilsvrc2012_class_id",
                        "official_wnid", "imagenet1k_index", "label_source",
                    )}
                )
                if (
                    not isinstance(observed["image_sha256"], str)
                    or _SHA256_RE.fullmatch(observed["image_sha256"]) is None
                    or (require_live_identity and observed["image_sha256"] != group.image_sha256)
                    or not isinstance(observed["source_payload_sha256"], str)
                    or _SHA256_RE.fullmatch(observed["source_payload_sha256"]) is None
                    or (
                        require_live_identity
                        and observed["source_payload_sha256"] != group.image_sha256
                    )
                    or any(observed[key] != value for key, value in expected_metadata.items())
                ):
                    raise StageDRuntimeIntegrityError("validated asset digest drifted")
                validated.append({
                    "selection_order": group.selection_order,
                    "sample_id": group.sample_id,
                    "relative_path": group.relative_path,
                    "image_sha256": observed["image_sha256"],
                    "size_bytes": observed["size_bytes"],
                    "width": observed["width"], "height": observed["height"],
                    "mode": observed["mode"],
                    "image_format": observed["image_format"],
                    "official_ilsvrc2012_class_id": observed["official_ilsvrc2012_class_id"],
                    "official_wnid": observed["official_wnid"],
                    "imagenet1k_index": observed["imagenet1k_index"],
                    "label_source": observed["label_source"],
                })
            else:
                if observation is not None:
                    raise StageDRuntimeIntegrityError(
                        "rejected asset leaked an observation"
                    )
                failure = event["failure_code"]
            pending = None
        elif kind == "ACCESS_COMPLETE":
            if pending is not None or len(validated) != len(plan.groups):
                raise StageDRuntimeIntegrityError("access completed before full validation")
            complete = True
        else:
            raise StageDRuntimeIntegrityError("unexpected access journal transition")
    return validated, pending, failure


def _validate_access_lifecycle(
    root: Path, *, plan: StageDExecutionPlan, runtime_binding_digest: str,
    checkpoint_digest: str, contracts: Sequence[Mapping[str, Any]] | None = None,
    repair_lag: bool = False, require_live_identity: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], int | None, str | None]:
    _validate_access_layout(root)
    lifecycle = _strict_control_json(
        root / _ACCESS_DIRECTORY_NAME / ACCESS_LIFECYCLE_FILENAME,
        "access lifecycle",
    )
    _require_exact_keys(lifecycle, _ACCESS_LIFECYCLE_KEYS, "access lifecycle")
    if (
        lifecycle["schema_version"] != 1
        or lifecycle["document_type"] != "stage_d_access_lifecycle"
        or lifecycle["state"] not in _ACCESS_STATES
        or lifecycle["plan_digest"] != plan.canonical_digest
        or lifecycle["registry_digest"] != plan.registry_digest
        or lifecycle["ordered_sample_ids_digest"] != plan.ordered_sample_ids_digest
        or lifecycle["runtime_binding_digest"] != runtime_binding_digest
        or lifecycle["runtime_checkpoint_at_access_start"] != checkpoint_digest
        or lifecycle["unit_execution_started_during_initialization"] is not False
        or lifecycle["canonical_digest"] != _self_digest(lifecycle)
    ):
        raise StageDRuntimeIntegrityError("access lifecycle binding drifted")
    events = _read_access_chain(
        root, plan=plan, runtime_binding_digest=runtime_binding_digest,
    )
    validated, pending, failure = _validated_access_rows(
        events, plan, contracts, require_live_identity=require_live_identity,
    )
    live_digest = canonical_sha256({"assets": validated}) if len(validated) == len(plan.groups) else None
    expected_state = "BLOCKED" if failure is not None else (
        "RUNNABLE" if events[-1]["event"] == "ACCESS_COMPLETE" else "ACCESS_STARTED"
    )
    expected = _access_lifecycle(
        state=expected_state, plan=plan, runtime_binding_digest=runtime_binding_digest,
        checkpoint_digest=checkpoint_digest, validated_asset_count=len(validated),
        journal_entry_count=len(events), journal_head_digest=events[-1]["canonical_digest"],
        live_assets_digest=live_digest if expected_state == "RUNNABLE" else None,
        blocked_reasons=(() if failure is None else (failure,)),
    )
    if lifecycle != expected:
        if not repair_lag:
            raise StageDRuntimeIntegrityError("access lifecycle differs from journal source")
        count = lifecycle["journal_entry_count"]
        if type(count) is not int or count < 1 or count > len(events):
            raise StageDRuntimeIntegrityError("access lifecycle journal count is invalid")
        prefix = events[:count]
        prefix_validated, _prefix_pending, prefix_failure = _validated_access_rows(
            prefix, plan, contracts, require_live_identity=require_live_identity,
        )
        prefix_state = "BLOCKED" if prefix_failure is not None else (
            "RUNNABLE" if prefix[-1]["event"] == "ACCESS_COMPLETE"
            else "ACCESS_STARTED"
        )
        prefix_live_digest = (
            canonical_sha256({"assets": prefix_validated})
            if prefix_state == "RUNNABLE" else None
        )
        prefix_expected = _access_lifecycle(
            state=prefix_state, plan=plan,
            runtime_binding_digest=runtime_binding_digest,
            checkpoint_digest=checkpoint_digest,
            validated_asset_count=len(prefix_validated),
            journal_entry_count=len(prefix),
            journal_head_digest=prefix[-1]["canonical_digest"],
            live_assets_digest=prefix_live_digest,
            blocked_reasons=(() if prefix_failure is None else (prefix_failure,)),
        )
        if lifecycle != prefix_expected:
            raise StageDRuntimeIntegrityError("access lifecycle is not a valid journal prefix")
        _replace_access_lifecycle(root, expected)
        lifecycle = expected
    return lifecycle, events, validated, pending, failure


def _access_asset_projection(
    group: StageDImageGroup, observation: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
        "selection_order": group.selection_order, "sample_id": group.sample_id,
        "relative_path": group.relative_path,
        "expected_image_sha256": group.image_sha256,
        "registry_row_digest": group.registry_row_digest,
        "observation": None if observation is None else _plain(observation),
    }


def _validate_access_contracts(
    values: Sequence[Mapping[str, Any]], plan: StageDExecutionPlan,
) -> tuple[dict[str, Any], ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise StageDRuntimeIntegrityError("asset contracts must be a sequence")
    if len(values) != len(plan.groups):
        raise StageDRuntimeIntegrityError("asset contracts must cover all 500 groups")
    required = {
        "selection_order", "sample_id", "relative_path", "image_sha256",
        "canonical_digest", "size_bytes", "width", "height", "mode",
        "image_format", "official_ilsvrc2012_class_id", "official_wnid",
        "imagenet1k_index", "label_source",
    }
    result: list[dict[str, Any]] = []
    for index, (value, group) in enumerate(zip(values, plan.groups)):
        source = _plain(_require_mapping(value, f"asset contract {index}"))
        if not required <= set(source):
            raise StageDRuntimeIntegrityError("asset contract fields are incomplete")
        if (
            source["selection_order"] != group.selection_order
            or source["sample_id"] != group.sample_id
            or source["relative_path"] != group.relative_path
            or source["image_sha256"] != group.image_sha256
            or source["canonical_digest"] != group.registry_row_digest
        ):
            raise StageDRuntimeIntegrityError("asset contract differs from frozen plan")
        for name in ("size_bytes", "width", "height", "official_ilsvrc2012_class_id", "imagenet1k_index"):
            if type(source[name]) is not int or source[name] < 0:
                raise StageDRuntimeIntegrityError(f"asset contract {name} is invalid")
        if source["width"] < 1 or source["height"] < 1:
            raise StageDRuntimeIntegrityError("asset contract dimensions are invalid")
        for name in ("mode", "image_format", "official_wnid", "label_source"):
            if not isinstance(source[name], str) or not source[name]:
                raise StageDRuntimeIntegrityError(f"asset contract {name} is invalid")
        result.append(source)
    return tuple(result)


def _append_access_event(
    root: Path, events: Sequence[Mapping[str, Any]], *, event: str,
    plan: StageDExecutionPlan, runtime_binding_digest: str,
    asset: Mapping[str, Any] | None = None, failure_code: str | None = None,
) -> dict[str, Any]:
    result = _access_event(
        sequence=len(events), event=event,
        previous=None if not events else events[-1]["canonical_digest"],
        plan_digest=plan.canonical_digest,
        runtime_binding_digest=runtime_binding_digest, asset=asset,
        failure_code=failure_code,
    )
    _atomic_create_control_json(
        root / _ACCESS_DIRECTORY_NAME / ACCESS_JOURNAL_DIRECTORY
        / f"{len(events):06d}.json",
        result,
    )
    return result


def _replace_access_lifecycle(root: Path, document: Mapping[str, Any]) -> None:
    _atomic_replace_control_json(
        root / _ACCESS_DIRECTORY_NAME / ACCESS_LIFECYCLE_FILENAME, document,
    )


def _ensure_access_started(
    root: Path, *, plan: StageDExecutionPlan, runtime_binding_digest: str,
    checkpoint_digest: str,
) -> None:
    """Finish only the pre-access prefix after an initialization crash."""

    access_root = root / _ACCESS_DIRECTORY_NAME
    journal = access_root / ACCESS_JOURNAL_DIRECTORY
    lifecycle_path = access_root / ACCESS_LIFECYCLE_FILENAME
    if not os.path.lexists(access_root):
        os.mkdir(access_root, 0o700)
        _fsync_directory(root)
    else:
        info = os.lstat(access_root)
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise StageDRuntimeIntegrityError("access root is unsafe")
    children = {child.name for child in access_root.iterdir()}
    if children - {ACCESS_JOURNAL_DIRECTORY, ACCESS_LIFECYCLE_FILENAME}:
        raise StageDRuntimeIntegrityError("partial access root has unknown entries")
    if not os.path.lexists(journal):
        if ACCESS_LIFECYCLE_FILENAME in children:
            raise StageDRuntimeIntegrityError("access lifecycle exists without journal")
        os.mkdir(journal, 0o700)
        _fsync_directory(access_root)
    else:
        info = os.lstat(journal)
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise StageDRuntimeIntegrityError("access journal root is unsafe")
    entries = list(journal.iterdir())
    if not entries:
        first = _append_access_event(
            root, (), event="ACCESS_STARTED", plan=plan,
            runtime_binding_digest=runtime_binding_digest,
        )
    else:
        events = _read_access_chain(
            root, plan=plan, runtime_binding_digest=runtime_binding_digest,
        )
        if len(events) != 1 or events[0]["event"] != "ACCESS_STARTED":
            if not os.path.lexists(lifecycle_path):
                raise StageDRuntimeIntegrityError(
                    "access journal advanced without a durable lifecycle"
                )
            return
        first = events[0]
    if not os.path.lexists(lifecycle_path):
        _atomic_create_control_json(
            lifecycle_path, _access_lifecycle(
                state="ACCESS_STARTED", plan=plan,
                runtime_binding_digest=runtime_binding_digest,
                checkpoint_digest=checkpoint_digest, validated_asset_count=0,
                journal_entry_count=1,
                journal_head_digest=first["canonical_digest"],
                live_assets_digest=None,
            ),
        )
        _fsync_directory(root)


def _require_pristine_runtime_checkpoint(checkpoint: Mapping[str, Any]) -> None:
    if (
        checkpoint["generation"] != 0
        or checkpoint["lifecycle"]["state"] != "initialized"
        or any(row["state"] != "pending" for row in checkpoint["shards"])
        or any(row["state"] != "pending" for row in checkpoint["groups"])
        or any(row["state"] != "pending" or row["attempts"] for row in checkpoint["units"])
    ):
        raise StageDRuntimeBlockedError(
            "D8.0 access initialization requires a pristine no-unit runtime root"
        )


def initialize_stage_d_access_root(
    output_root: Path | str, plan: Any, estimate: Any, preflight: Any,
    *, execution_context: object | None = None,
    execution_gate: StageDExecutionGate | Callable[..., Any] | None = None,
    asset_contracts: Sequence[Mapping[str, Any]] = (),
    asset_validator: Callable[[StageDImageGroup, Mapping[str, Any]], Mapping[str, Any]] | None = None,
    fault_injector: Callable[[str, int | None], Any] | None = None,
) -> Mapping[str, Any]:
    """Run the D8.0 durable access boundary without executing a unit.

    D7 verifies this exact implementation using an injected synthetic asset
    validator.  A future authorized D8.0 supplies the production validator.
    Process-control interruptions leave a durable intent and are resumed from
    that exact selection order; ordinary validation failures durably block the
    root.
    """

    typed_plan, typed_estimate, typed_preflight = _validate_control_inputs(
        plan, estimate, preflight
    )
    contracts = _validate_access_contracts(asset_contracts, typed_plan)
    if not callable(asset_validator):
        raise StageDRuntimeIntegrityError(
            "D8.0 requires an explicit post-authorization asset validator"
        )
    checked_context = _consume_execution_gate(
        execution_gate, execution_context, operation="initialize-access",
        output_root=output_root, plan=typed_plan, estimate=typed_estimate,
        preflight=typed_preflight,
    )
    context_binding = _normalize_execution_context(
        checked_context, plan=typed_plan, estimate=typed_estimate,
        preflight=typed_preflight,
    )
    _seal_execution_context(checked_context, context_binding)
    lexical_root = _absolute_lexical_root(output_root, "Stage D output root")
    if not os.path.lexists(lexical_root):
        _initialize_validated_output_root(
            lexical_root, typed_plan, typed_estimate, typed_preflight,
            context_binding,
        )
    root = _existing_root(lexical_root)
    _validate_closed_layout(root, typed_plan)

    def fault(point: str, selection_order: int | None = None) -> None:
        if fault_injector is not None:
            fault_injector(point, selection_order)

    with _lifecycle_lock(root, exclusive=True):
        with _checkpoint_lock(root):
            checkpoint = _read_valid_checkpoint_locked(
                root, typed_plan, typed_estimate, typed_preflight, context_binding
            )
        _require_pristine_runtime_checkpoint(checkpoint)
        access_root = root / _ACCESS_DIRECTORY_NAME
        journal = access_root / ACCESS_JOURNAL_DIRECTORY
        if not os.path.lexists(access_root):
            _ensure_access_started(
                root, plan=typed_plan,
                runtime_binding_digest=context_binding["runtime_binding_digest"],
                checkpoint_digest=checkpoint["canonical_digest"],
            )
            fault("after_access_started", None)
        else:
            _ensure_access_started(
                root, plan=typed_plan,
                runtime_binding_digest=context_binding["runtime_binding_digest"],
                checkpoint_digest=checkpoint["canonical_digest"],
            )
        _validate_closed_layout(root, typed_plan)
        lifecycle, events, validated, pending, failure = _validate_access_lifecycle(
            root, plan=typed_plan,
            runtime_binding_digest=context_binding["runtime_binding_digest"],
            checkpoint_digest=checkpoint["canonical_digest"],
            contracts=contracts,
            repair_lag=True,
            require_live_identity=context_binding["formal"],
        )
        if failure is not None or lifecycle["state"] == "BLOCKED":
            raise StageDRuntimeBlockedError(
                "D8.0 access root is durably blocked", reasons=(failure or "BLOCKED",)
            )
        if lifecycle["state"] == "RUNNABLE":
            return lifecycle

        while len(validated) < len(typed_plan.groups):
            group = typed_plan.groups[len(validated)]
            intent_kind = "ASSET_ACCESS_RESUMED" if pending is not None else "ASSET_ACCESS_INTENT"
            intent = _append_access_event(
                root, events, event=intent_kind, plan=typed_plan,
                runtime_binding_digest=context_binding["runtime_binding_digest"],
                asset=_access_asset_projection(group, None),
            )
            events.append(intent)
            lifecycle = _access_lifecycle(
                state="ACCESS_STARTED", plan=typed_plan,
                runtime_binding_digest=context_binding["runtime_binding_digest"],
                checkpoint_digest=checkpoint["canonical_digest"],
                validated_asset_count=len(validated),
                journal_entry_count=len(events),
                journal_head_digest=events[-1]["canonical_digest"],
                live_assets_digest=None,
            )
            _replace_access_lifecycle(root, lifecycle)
            fault("after_asset_intent", group.selection_order)
            try:
                observation = _plain(
                    _require_mapping(
                        asset_validator(group, contracts[group.selection_order]),
                        "asset validation observation",
                    )
                )
                _require_exact_keys(
                    observation, _ACCESS_OBSERVATION_KEYS, "asset validation observation"
                )
                contract = contracts[group.selection_order]
                expected_registry = {key: contract[key] for key in (
                    "size_bytes", "width", "height", "mode", "image_format",
                    "official_ilsvrc2012_class_id", "official_wnid",
                    "imagenet1k_index", "label_source",
                )}
                if (
                    (
                        context_binding["formal"]
                        and observation["source_payload_sha256"] != group.image_sha256
                    )
                    or (
                        context_binding["formal"]
                        and observation["image_sha256"] != group.image_sha256
                    )
                    or not isinstance(observation["source_kind"], str)
                    or not observation["source_kind"]
                    or any(value is None for value in expected_registry.values())
                    or any(observation[key] != value for key, value in expected_registry.items())
                ):
                    raise StageDRuntimeIntegrityError("live asset observation drifted")
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as error:
                rejected = _append_access_event(
                    root, events, event="ASSET_REJECTED", plan=typed_plan,
                    runtime_binding_digest=context_binding["runtime_binding_digest"],
                    asset=_access_asset_projection(group, None),
                    failure_code="LIVE_ASSET_VALIDATION_FAILED",
                )
                events.append(rejected)
                blocked = _access_lifecycle(
                    state="BLOCKED", plan=typed_plan,
                    runtime_binding_digest=context_binding["runtime_binding_digest"],
                    checkpoint_digest=checkpoint["canonical_digest"],
                    validated_asset_count=len(validated),
                    journal_entry_count=len(events),
                    journal_head_digest=events[-1]["canonical_digest"],
                    live_assets_digest=None,
                    blocked_reasons=("LIVE_ASSET_VALIDATION_FAILED",),
                )
                _replace_access_lifecycle(root, blocked)
                raise StageDRuntimeBlockedError(
                    "D8.0 live asset validation failed",
                    reasons=("LIVE_ASSET_VALIDATION_FAILED",),
                ) from error
            result = _append_access_event(
                root, events, event="ASSET_VALIDATED", plan=typed_plan,
                runtime_binding_digest=context_binding["runtime_binding_digest"],
                asset=_access_asset_projection(group, observation),
            )
            events.append(result)
            fault("after_asset_validated_event", group.selection_order)
            validated.append({
                "selection_order": group.selection_order,
                "sample_id": group.sample_id, "relative_path": group.relative_path,
                "image_sha256": observation["image_sha256"],
                **{key: observation[key] for key in (
                    "size_bytes", "width", "height", "mode", "image_format",
                    "official_ilsvrc2012_class_id", "official_wnid",
                    "imagenet1k_index", "label_source",
                )},
            })
            pending = None
            lifecycle = _access_lifecycle(
                state="ACCESS_STARTED", plan=typed_plan,
                runtime_binding_digest=context_binding["runtime_binding_digest"],
                checkpoint_digest=checkpoint["canonical_digest"],
                validated_asset_count=len(validated), journal_entry_count=len(events),
                journal_head_digest=events[-1]["canonical_digest"],
                live_assets_digest=None,
            )
            _replace_access_lifecycle(root, lifecycle)
            fault("after_asset_validated", group.selection_order)

        final = _append_access_event(
            root, events, event="ACCESS_COMPLETE", plan=typed_plan,
            runtime_binding_digest=context_binding["runtime_binding_digest"],
        )
        events.append(final)
        fault("after_access_complete_event", None)
        runnable = _access_lifecycle(
            state="RUNNABLE", plan=typed_plan,
            runtime_binding_digest=context_binding["runtime_binding_digest"],
            checkpoint_digest=checkpoint["canonical_digest"],
            validated_asset_count=len(validated), journal_entry_count=len(events),
            journal_head_digest=events[-1]["canonical_digest"],
            live_assets_digest=canonical_sha256({"assets": validated}),
        )
        _replace_access_lifecycle(root, runnable)
        checked, *_ = _validate_access_lifecycle(
            root, plan=typed_plan,
            runtime_binding_digest=context_binding["runtime_binding_digest"],
            checkpoint_digest=checkpoint["canonical_digest"],
            contracts=contracts,
            require_live_identity=context_binding["formal"],
        )
        return checked


def _source_resources(value: Any) -> dict[str, int | float]:
    source = _plain(_require_mapping(value, "attempt resource evidence"))
    result: dict[str, int | float] = dict(_ZERO_RESOURCES)
    aliases = {"wall_time_seconds": "elapsed_seconds"}
    for raw_key, raw_value in source.items():
        key = aliases.get(raw_key, raw_key)
        if key not in result:
            continue
        if key == "elapsed_seconds":
            if type(raw_value) not in {int, float} or not math.isfinite(float(raw_value)) or raw_value < 0:
                raise StageDRuntimeIntegrityError(
                    "attempt elapsed resource evidence is invalid"
                )
            result[key] = float(raw_value)
        elif type(raw_value) is not int or raw_value < 0:
            raise StageDRuntimeIntegrityError(
                f"attempt resource evidence {raw_key} is invalid"
            )
        elif key == "cuda_peak_reserved_bytes":
            result[key] = max(result[key], raw_value)
        else:
            result[key] += raw_value
    return result


def _source_validate_attempt(
    path: Path, *, plan: StageDExecutionPlan, group: StageDImageGroup,
    unit: StageDUnit, context_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute status, manifest, provenance and inventory from disk."""

    from .stage_d_artifacts import (
        INFRASTRUCTURE_INTERRUPTION_STATUS, MANIFEST_FILENAME,
        STATUS_FILENAME, validate_stage_d_attempt,
    )
    from .stage_d_provenance import (
        FORMAL_MODE, SCIENTIFIC_FAILURE_STATUS, SUCCESS_STATUS, SYNTHETIC_MODE,
    )

    mode = FORMAL_MODE if context_binding["formal"] else SYNTHETIC_MODE
    verdict = validate_stage_d_attempt(
        path, expected_unit=unit, expected_group=group,
        expected_plan_digest=plan.canonical_digest, expected_mode=mode,
        expected_provenance_bindings=context_binding["provenance_bindings"],
    )
    if not verdict.valid or verdict.digest is None:
        raise StageDRuntimeIntegrityError(
            "Stage D attempt failed source-recomputing validation: "
            + "; ".join(verdict.errors)
        )
    status = _strict_control_json(path / STATUS_FILENAME, "attempt status")
    manifest = _strict_control_json(path / MANIFEST_FILENAME, "attempt manifest")
    status_sha256 = _single_regular_file_sha256(
        path / STATUS_FILENAME, "attempt status"
    )
    artifact_rows = manifest.get("artifacts")
    if not isinstance(artifact_rows, list):
        raise StageDRuntimeIntegrityError("attempt manifest artifacts are invalid")
    scientific_payload_count = sum(
        isinstance(row, Mapping)
        and isinstance(row.get("path"), str)
        and row["path"].startswith(("L0/", "L1/"))
        and row.get("kind") not in {"provenance", "scientific_failure"}
        for row in artifact_rows
    )
    l0_l1_bytes = 0
    l2_bytes = 0
    for record in artifact_rows:
        if not isinstance(record, Mapping):
            raise StageDRuntimeIntegrityError(
                "attempt manifest contains a non-object artifact record"
            )
        level = record.get("level")
        byte_count = record.get("bytes")
        if type(byte_count) is not int or byte_count < 0:
            raise StageDRuntimeIntegrityError(
                "attempt manifest contains invalid artifact bytes"
            )
        if level in {"L0", "L1"} and record.get("kind") != "provenance":
            l0_l1_bytes += byte_count
        elif level == "L2":
            l2_bytes += byte_count
    source_status = verdict.status
    if source_status == SUCCESS_STATUS:
        state = "succeeded"
        failure_class = failure_code = None
    elif source_status == SCIENTIFIC_FAILURE_STATUS:
        state = "scientific_failed"
        failure_class = "SCIENTIFIC_INVALIDITY"
        failure = status.get("failure")
        failure_code = failure.get("code") if isinstance(failure, Mapping) else None
        if not isinstance(failure_code, str) or not failure_code:
            raise StageDRuntimeIntegrityError(
                "scientific failure lacks a source-validated failure code"
            )
    elif source_status == INFRASTRUCTURE_INTERRUPTION_STATUS:
        state = "infrastructure_failed"
        failure_class = "INFRASTRUCTURE_FAILURE"
        failure = status.get("failure")
        failure_code = failure.get("code") if isinstance(failure, Mapping) else None
        if failure_code not in RETRYABLE_INFRASTRUCTURE_FAILURE_CODES:
            raise StageDRuntimeIntegrityError(
                "infrastructure attempt code is outside the retry allowlist"
            )
    else:
        raise StageDRuntimeIntegrityError(
            "attempt has no supported source-validated control status"
        )
    if state in {"succeeded", "scientific_failed"}:
        provenance = _strict_control_json(
            path / "L0/provenance.json", "attempt provenance"
        )
        resource_source = provenance.get("bindings", {}).get("resources")
    else:
        resource_source = status.get("failure", {}).get("resources")
    resources = _source_resources(resource_source)
    resources["l0_l1_payload_bytes"] = l0_l1_bytes
    resources["l2_payload_bytes"] = l2_bytes
    # Control/provenance bytes are an observed inventory fact, not a producer
    # assertion embedded in provenance.
    control_names = {"status.json", "manifest.json", "L0/provenance.json"}
    resources["control_and_provenance_bytes"] = sum(
        child.stat().st_size for child in path.rglob("*")
        if child.is_file() and child.relative_to(path).as_posix() in control_names
    )
    return {
        "state": state,
        "attempt_id": path.name,
        "attempt_path": str(path),
        "status_sha256": status_sha256,
        "attempt_inventory_digest": verdict.digest,
        "failure_class": failure_class,
        "failure_code": failure_code,
        "result_manifest_published": verdict.terminal and verdict.complete,
        "published_l0_l1_payload_count": scientific_payload_count,
        "resources": resources,
    }


def _source_validate_open_attempt(
    path: Path, *, plan: StageDExecutionPlan, group: StageDImageGroup,
    unit: StageDUnit, context_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a writer's running control without treating it as terminal."""

    from .stage_d_artifacts import validate_stage_d_attempt
    from .stage_d_provenance import FORMAL_MODE, SYNTHETIC_MODE

    mode = FORMAL_MODE if context_binding["formal"] else SYNTHETIC_MODE
    verdict = validate_stage_d_attempt(
        path, expected_unit=unit, expected_group=group,
        expected_plan_digest=plan.canonical_digest, expected_mode=mode,
        expected_provenance_bindings=context_binding["provenance_bindings"],
    )
    if (
        not verdict.valid or verdict.status != "running" or verdict.terminal
        or verdict.complete or verdict.resumable or verdict.artifact_count != 0
        or verdict.digest is None
    ):
        raise StageDRuntimeIntegrityError(
            "claimed attempt is not a valid no-result running control: "
            + "; ".join(verdict.errors)
        )
    return {
        "state": "claimed", "status_sha256": _single_regular_file_sha256(
            path / "status.json", "running attempt status"
        ),
        "attempt_inventory_digest": verdict.digest,
    }


def _inspect_recovery_attempt(
    path: Path, *, group: StageDImageGroup, unit: StageDUnit,
) -> dict[str, Any]:
    """Classify only control publication phases; never inspect payload values."""

    from .stage_d_artifacts import (
        INFRASTRUCTURE_INTERRUPTION_STATUS, MANIFEST_FILENAME, RUNNING_STATUS,
        STATUS_FILENAME,
    )
    if not path.exists():
        return {"phase": "missing"}
    info = os.lstat(path)
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        return {"phase": "unsafe"}
    entries = {child.relative_to(path).as_posix(): child for child in path.rglob("*")}
    files = {name: child for name, child in entries.items() if child.is_file()}
    if not files:
        return {"phase": "empty"}
    scientific = [name for name in files if name.startswith(("L0/", "L1/", "L2/"))]
    if scientific:
        return {"phase": "scientific_or_partial"}
    allowed = {STATUS_FILENAME, MANIFEST_FILENAME}
    if set(files) - allowed:
        return {"phase": "unsafe"}
    manifest = (
        _strict_control_json(files[MANIFEST_FILENAME], "recovery attempt manifest")
        if MANIFEST_FILENAME in files else None
    )
    status = (
        _strict_control_json(files[STATUS_FILENAME], "recovery attempt status")
        if STATUS_FILENAME in files else None
    )
    expected = {
        "unit_digest": unit.canonical_digest,
        "group_digest": group.canonical_digest,
        "pair_digest": group.pair_binding.canonical_digest,
    }
    if manifest is not None and any(manifest.get(key) != value for key, value in expected.items()):
        return {"phase": "unsafe"}
    if status is not None and status.get("unit_digest") != unit.canonical_digest:
        return {"phase": "unsafe"}
    manifest_state = None if manifest is None else manifest.get("status")
    status_state = None if status is None else status.get("status")
    if manifest_state == INFRASTRUCTURE_INTERRUPTION_STATUS:
        if status_state == INFRASTRUCTURE_INTERRUPTION_STATUS:
            return {"phase": "infrastructure_published"}
        if status_state == RUNNING_STATUS:
            return {"phase": "manifest_published"}
    if manifest_state == RUNNING_STATUS and status_state == RUNNING_STATUS:
        return {"phase": "running"}
    if status_state in {"succeeded", "scientific_failure"}:
        return {"phase": "terminal_published"}
    return {"phase": "unsafe"}


def _validate_checkpoint(
    checkpoint: Any, *, root: Path, plan: StageDExecutionPlan,
    estimate: StageDEstimate, preflight: StageDPreflight,
    context_binding: Mapping[str, Any],
    transitioning_unit_digest: str | None = None,
) -> dict[str, Any]:
    document = _plain(_require_mapping(checkpoint, "runtime checkpoint"))
    _require_exact_keys(document, _CHECKPOINT_KEYS, "runtime checkpoint")
    if (
        document["schema_version"] != 1
        or document["document_type"] != _CHECKPOINT_DOCUMENT_TYPE
    ):
        raise StageDRuntimeIntegrityError(
            "runtime checkpoint identity drifted"
        )
    _require_int(document["generation"], "checkpoint generation", minimum=0)
    previous = document["previous_checkpoint_digest"]
    if document["generation"] == 0:
        if previous is not None:
            raise StageDRuntimeIntegrityError(
                "initial checkpoint cannot claim a predecessor"
            )
    else:
        _require_sha256(previous, "previous checkpoint digest")
    if document["canonical_digest"] != _self_digest(document):
        raise StageDRuntimeIntegrityError(
            "runtime checkpoint canonical digest mismatch"
        )
    if (
        document["plan_digest"] != plan.canonical_digest
        or document["estimate_digest"] != estimate.canonical_digest
        or document["preflight_digest"] != preflight.canonical_digest
    ):
        raise StageDRuntimeIntegrityError(
            "runtime checkpoint plan/budget/preflight binding drifted"
        )
    persisted_context = _require_mapping(
        document["execution_context"], "checkpoint execution context"
    )
    _require_exact_keys(persisted_context, _CONTEXT_KEYS, "checkpoint execution context")
    if persisted_context != context_binding:
        raise StageDRuntimeIntegrityError(
            "execution context drift or formal/synthetic root upgrade attempted"
        )

    shards = document["shards"]
    groups = document["groups"]
    units = document["units"]
    if not isinstance(shards, list) or len(shards) != context_binding["shard_count"]:
        raise StageDRuntimeIntegrityError("checkpoint shard coverage drifted")
    if not isinstance(groups, list) or len(groups) != len(plan.groups):
        raise StageDRuntimeIntegrityError("checkpoint group coverage drifted")
    if not isinstance(units, list) or len(units) != len(plan.units):
        raise StageDRuntimeIntegrityError(
            "checkpoint failure table must cover all 1000 planned units"
        )
    normalized_shards: list[dict[str, Any]] = []
    expected_by_shard = {index: [] for index in range(len(shards))}
    for group in plan.groups:
        expected_by_shard[context_binding["shard_mapping"][group.selection_order]].append(
            group.canonical_digest
        )
    for index, raw in enumerate(shards):
        row = _plain(_require_mapping(raw, f"checkpoint shard {index}"))
        _require_exact_keys(row, _SHARD_KEYS, f"checkpoint shard {index}")
        if (
            row["shard_id"] != index
            or row["state"] not in _SHARD_STATES
            or type(row["terminal"]) is not bool
            or row["terminal"] is not (row["state"] == "completed")
            or row["group_count"] != len(expected_by_shard[index])
            or row["group_digests"] != expected_by_shard[index]
        ):
            raise StageDRuntimeIntegrityError(
                f"checkpoint shard {index} identity/state drifted"
            )
        token = row["owner_token_digest"]
        if row["state"] == "leased":
            _require_sha256(token, f"checkpoint shard {index} owner digest")
        elif token is not None:
            raise StageDRuntimeIntegrityError(
                f"unleased checkpoint shard {index} retains an owner"
            )
        normalized_shards.append(row)

    normalized_groups: list[dict[str, Any]] = []
    for index, (raw, expected_group) in enumerate(zip(groups, plan.groups)):
        row = _plain(_require_mapping(raw, f"checkpoint group {index}"))
        _require_exact_keys(row, _GROUP_KEYS, f"checkpoint group {index}")
        expected_units = [unit.canonical_digest for unit in expected_group.units]
        if (
            row["group_digest"] != expected_group.canonical_digest
            or row["selection_order"] != expected_group.selection_order
            or row["sample_id"] != expected_group.sample_id
            or row["shard_id"] != context_binding["shard_mapping"][index]
            or row["unit_digests"] != expected_units
            or row["state"] not in _GROUP_STATES
            or type(row["terminal"]) is not bool
            or row["terminal"] is not (row["state"] == "terminal")
        ):
            raise StageDRuntimeIntegrityError(
                f"checkpoint group {index} identity/state drifted"
            )
        token = row["owner_token_digest"]
        if row["state"] == "claimed":
            _require_sha256(token, f"checkpoint group {index} owner digest")
        elif token is not None:
            raise StageDRuntimeIntegrityError(
                f"unclaimed checkpoint group {index} retains an owner"
            )
        normalized_groups.append(row)

    normalized_units: list[dict[str, Any]] = []
    for index, (raw, expected_unit) in enumerate(zip(units, plan.units)):
        row = _plain(_require_mapping(raw, f"checkpoint unit {index}"))
        _require_exact_keys(row, _UNIT_KEYS, f"checkpoint unit {index}")
        expected_group = plan.groups[expected_unit.selection_order]
        identity = _unit_identity(expected_unit)
        if (
            any(row[key] != value for key, value in identity.items())
            or row["group_digest"] != expected_group.canonical_digest
            or row["state"] not in _UNIT_STATES
            or type(row["terminal"]) is not bool
            or row["terminal"] is not (
                row["state"] in {"succeeded", "scientific_failed"}
            )
            or not isinstance(row["attempts"], list)
            or len(row["attempts"]) > 2
        ):
            raise StageDRuntimeIntegrityError(
                f"checkpoint unit {index} identity/state drifted"
            )
        row["attempts"] = [
            _validate_attempt(
                attempt, root=root, plan=plan, group=expected_group,
                unit=expected_unit, context_binding=context_binding,
                attempt_index=attempt_index,
                allow_published_claim=(
                    expected_unit.canonical_digest == transitioning_unit_digest
                ),
            )
            for attempt_index, attempt in enumerate(row["attempts"])
        ]
        attempts_root = (
            root / "units" / expected_unit.canonical_digest / "attempts"
        )
        try:
            disk_entries = list(attempts_root.iterdir())
        except OSError as error:
            raise StageDRuntimeIntegrityError(
                f"cannot enumerate checkpoint unit {index} attempts"
            ) from error
        disk_names: set[str] = set()
        for entry in disk_entries:
            info = os.lstat(entry)
            if (
                entry.name in disk_names
                or _ATTEMPT_ID_RE.fullmatch(entry.name) is None
                or stat.S_ISLNK(info.st_mode)
                or not stat.S_ISDIR(info.st_mode)
            ):
                raise StageDRuntimeIntegrityError(
                    f"checkpoint unit {index} has an extra or unsafe attempt"
                )
            disk_names.add(entry.name)
        checkpoint_names = {item["attempt_id"] for item in row["attempts"]}
        if not disk_names.issubset(checkpoint_names):
            raise StageDRuntimeIntegrityError(
                f"checkpoint unit {index} has an unrecorded attempt directory"
            )
        missing = checkpoint_names - disk_names
        allowed_missing = {
            row["attempts"][-1]["attempt_id"]
        } if row["attempts"] and row["attempts"][-1]["state"] == "claimed" else set()
        if missing - allowed_missing:
            raise StageDRuntimeIntegrityError(
                f"checkpoint unit {index} is missing a recorded attempt directory"
            )
        owner = row["owner_token_digest"]
        if row["state"] == "claimed":
            _require_sha256(owner, f"checkpoint unit {index} owner digest")
            if not row["attempts"] or row["attempts"][-1]["state"] != "claimed":
                raise StageDRuntimeIntegrityError(
                    f"claimed checkpoint unit {index} lacks a claimed attempt"
                )
        elif owner is not None:
            raise StageDRuntimeIntegrityError(
                f"unclaimed checkpoint unit {index} retains an owner"
            )
        if row["state"] == "pending" and row["attempts"]:
            raise StageDRuntimeIntegrityError(
                f"pending checkpoint unit {index} has attempts"
            )
        if row["state"] == "retry_pending":
            if (
                len(row["attempts"]) != 1
                or row["attempts"][0]["state"] != "infrastructure_failed"
                or row["failure_code"] not in RETRYABLE_INFRASTRUCTURE_FAILURE_CODES
                or row["attempts"][0]["result_manifest_published"]
                or row["attempts"][0]["published_l0_l1_payload_count"] != 0
            ):
                raise StageDRuntimeIntegrityError(
                    f"retry-pending checkpoint unit {index} is not no-result eligible"
                )
        if row["terminal"] and (
            not row["attempts"]
            or row["attempts"][-1]["state"] != row["state"]
        ):
            raise StageDRuntimeIntegrityError(
                f"terminal checkpoint unit {index} attempt state drifted"
            )
        if row["retry_authorization_digest"] is not None:
            _require_sha256(
                row["retry_authorization_digest"],
                f"checkpoint unit {index} retry authorization digest",
            )
        row["resources"] = _validate_resources(
            row["resources"], f"checkpoint unit {index} resources"
        )
        expected_resource = dict(_ZERO_RESOURCES)
        for attempt in row["attempts"]:
            for key in _RESOURCE_KEYS:
                if key == "cuda_peak_reserved_bytes":
                    expected_resource[key] = max(
                        expected_resource[key], attempt["resources"][key]
                    )
                else:
                    expected_resource[key] += attempt["resources"][key]
        if row["resources"] != expected_resource:
            raise StageDRuntimeIntegrityError(
                f"checkpoint unit {index} resource totals drifted"
            )
        normalized_units.append(row)

    # Group state is a strict projection of its two ordered unit rows.
    by_unit = {row["unit_digest"]: row for row in normalized_units}
    for index, group in enumerate(normalized_groups):
        children = [by_unit[digest] for digest in group["unit_digests"]]
        if group["state"] == "claimed":
            expected_group_state = "claimed"
        elif any(row["state"] == "blocked" for row in children):
            expected_group_state = "blocked"
        elif all(row["terminal"] for row in children):
            expected_group_state = "terminal"
        elif any(row["state"] == "retry_pending" for row in children) and all(
            row["terminal"] or row["state"] in {"pending", "retry_pending"}
            for row in children
        ):
            expected_group_state = "retry_pending"
        elif all(row["state"] == "pending" for row in children):
            expected_group_state = "pending"
        else:
            raise StageDRuntimeIntegrityError(
                f"checkpoint group {index} has an impossible child-state projection"
            )
        if group["state"] != expected_group_state:
            raise StageDRuntimeIntegrityError(
                f"checkpoint group {index} state disagrees with its units"
            )

    by_group = {row["group_digest"]: row for row in normalized_groups}
    for index, shard in enumerate(normalized_shards):
        children = [by_group[digest] for digest in shard["group_digests"]]
        if shard["state"] == "leased":
            expected_shard_state = "leased"
        elif all(row["state"] == "terminal" for row in children):
            expected_shard_state = "completed"
        elif any(row["state"] == "blocked" for row in children):
            expected_shard_state = "blocked"
        elif any(row["state"] == "retry_pending" for row in children) and all(
            row["state"] in {"pending", "terminal", "retry_pending"}
            for row in children
        ):
            expected_shard_state = "retry_pending"
        elif all(row["state"] == "pending" for row in children):
            expected_shard_state = "pending"
        else:
            raise StageDRuntimeIntegrityError(
                f"checkpoint shard {index} has an impossible group-state projection"
            )
        if shard["state"] != expected_shard_state:
            raise StageDRuntimeIntegrityError(
                f"checkpoint shard {index} state disagrees with its groups"
            )

    failure_rows = document["failure_rows"]
    if not isinstance(failure_rows, list) or len(failure_rows) != len(plan.units):
        raise StageDRuntimeIntegrityError(
            "checkpoint failure_rows must contain all 1000 units"
        )
    for index, row in enumerate(failure_rows):
        _require_exact_keys(
            _require_mapping(row, f"failure row {index}"),
            _FAILURE_ROW_KEYS, f"failure row {index}",
        )
    if failure_rows != _failure_rows(normalized_units):
        raise StageDRuntimeIntegrityError(
            "checkpoint failure rows disagree with unit controls"
        )
    if document["resources"] != _sum_resources(normalized_units):
        raise StageDRuntimeIntegrityError(
            "checkpoint resource totals disagree with unit controls"
        )
    lifecycle = _plain(_require_mapping(document["lifecycle"], "checkpoint lifecycle"))
    _require_exact_keys(lifecycle, _LIFECYCLE_KEYS, "checkpoint lifecycle")
    if lifecycle["state"] not in _LIFECYCLE_STATES:
        raise StageDRuntimeIntegrityError("checkpoint lifecycle state drifted")
    reasons = lifecycle["blocked_reasons"]
    if not isinstance(reasons, list) or any(
        not isinstance(item, str) or not item for item in reasons
    ):
        raise StageDRuntimeIntegrityError(
            "checkpoint blocked reasons are invalid"
        )
    expected_lifecycle = _lifecycle_projection(
        lifecycle["state"], normalized_shards, normalized_groups,
        normalized_units, reasons,
    )
    if lifecycle != expected_lifecycle:
        raise StageDRuntimeIntegrityError(
            "checkpoint lifecycle counters drifted"
        )
    if lifecycle["state"] == "complete" and lifecycle["terminal_units"] != len(plan.units):
        raise StageDRuntimeIntegrityError(
            "complete checkpoint does not have all units terminal"
        )
    if lifecycle["state"] == "blocked" and not reasons:
        raise StageDRuntimeIntegrityError(
            "blocked checkpoint lacks a control-plane reason"
        )
    active_shards = [row for row in normalized_shards if row["state"] == "leased"]
    active_groups = [row for row in normalized_groups if row["state"] == "claimed"]
    active_units = [row for row in normalized_units if row["state"] == "claimed"]
    if len(active_shards) > 1 or len(active_groups) > 1 or len(active_units) > 1:
        raise StageDRuntimeIntegrityError(
            "checkpoint contains concurrent shard/group/unit ownership"
        )
    if active_groups and (
        not active_shards
        or active_groups[0]["shard_id"] != active_shards[0]["shard_id"]
    ):
        raise StageDRuntimeIntegrityError(
            "checkpoint group ownership is outside the leased shard"
        )
    if active_units and (
        not active_groups
        or active_units[0]["group_digest"] != active_groups[0]["group_digest"]
    ):
        raise StageDRuntimeIntegrityError(
            "checkpoint unit ownership is outside the claimed group"
        )
    expected_lifecycle_state = _expected_lifecycle_state(
        lifecycle["state"], document["generation"], normalized_shards,
    )
    if lifecycle["state"] != expected_lifecycle_state:
        raise StageDRuntimeIntegrityError(
            "checkpoint lifecycle state disagrees with child controls"
        )
    document["shards"] = normalized_shards
    document["groups"] = normalized_groups
    document["units"] = normalized_units
    document["lifecycle"] = lifecycle
    return document


def _read_valid_checkpoint_locked(
    root: Path, plan: StageDExecutionPlan, estimate: StageDEstimate,
    preflight: StageDPreflight, context_binding: Mapping[str, Any],
    *, transitioning_unit_digest: str | None = None,
) -> dict[str, Any]:
    # All directory enumeration shares the checkpoint lock with atomic replace,
    # so its private temporary file cannot be mistaken for foreign residue.
    _validate_closed_layout(root, plan)
    return _validate_checkpoint(
        _strict_checkpoint_json(root / RUNTIME_CHECKPOINT_FILENAME),
        root=root, plan=plan, estimate=estimate, preflight=preflight,
        context_binding=context_binding,
        transitioning_unit_digest=transitioning_unit_digest,
    )


def _replace_checkpoint(
    root: Path, previous: Mapping[str, Any], changed: Mapping[str, Any], *,
    plan: StageDExecutionPlan, estimate: StageDEstimate,
    preflight: StageDPreflight, context_binding: Mapping[str, Any],
    transitioning_unit_digest: str | None = None,
) -> dict[str, Any]:
    next_document = _plain(changed, "next runtime checkpoint")
    next_document["generation"] = previous["generation"] + 1
    next_document["previous_checkpoint_digest"] = previous["canonical_digest"]
    next_document["failure_rows"] = _failure_rows(next_document["units"])
    next_document["resources"] = _sum_resources(next_document["units"])
    lifecycle = next_document["lifecycle"]
    next_document["lifecycle"] = _lifecycle_projection(
        lifecycle["state"], next_document["shards"],
        next_document["groups"], next_document["units"],
        lifecycle["blocked_reasons"],
    )
    next_document.pop("canonical_digest", None)
    next_document = _signed(next_document)
    # Never persist a state that this runtime cannot subsequently consume.
    _validate_checkpoint(
        next_document, root=root, plan=plan, estimate=estimate,
        preflight=preflight, context_binding=context_binding,
        transitioning_unit_digest=transitioning_unit_digest,
    )
    _validate_closed_layout(root, plan)
    _atomic_replace_checkpoint(
        root / RUNTIME_CHECKPOINT_FILENAME, next_document
    )
    return next_document


def _validate_closed_layout(root: Path, plan: StageDExecutionPlan) -> None:
    expected_units = {unit.canonical_digest for unit in plan.units}
    try:
        children = list(root.iterdir())
    except OSError as error:
        raise StageDRuntimeIntegrityError(
            "cannot enumerate Stage D output root"
        ) from error
    observed = {child.name for child in children}
    allowed = (_ROOT_ENTRIES, _ROOT_ENTRIES | {_ACCESS_DIRECTORY_NAME})
    if observed not in allowed:
        raise StageDRuntimeIntegrityError(
            "Stage D root layout is not closed; "
            f"missing={sorted(_ROOT_ENTRIES - observed)}, "
            f"unknown={sorted(observed - (_ROOT_ENTRIES | {_ACCESS_DIRECTORY_NAME}))}"
        )
    for child in children:
        metadata = os.lstat(child)
        if stat.S_ISLNK(metadata.st_mode):
            raise StageDRuntimeIntegrityError(
                f"Stage D root contains a symlink: {child.name}"
            )
        if child.name in {*_LOCK_NAMES, RUNTIME_CHECKPOINT_FILENAME}:
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise StageDRuntimeIntegrityError(
                    f"Stage D control is not a single-link file: {child.name}"
                )
        elif child.name in {"units", _ACCESS_DIRECTORY_NAME} and not stat.S_ISDIR(metadata.st_mode):
            raise StageDRuntimeIntegrityError(
                f"Stage D {child.name} root is not a directory"
            )
    units_root = root / "units"
    observed_units: set[str] = set()
    for unit_root in units_root.iterdir():
        metadata = os.lstat(unit_root)
        if (
            unit_root.name not in expected_units
            or stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISDIR(metadata.st_mode)
        ):
            raise StageDRuntimeIntegrityError(
                f"unexpected or unsafe Stage D unit entry: {unit_root.name}"
            )
        observed_units.add(unit_root.name)
        contents = list(unit_root.iterdir())
        if len(contents) != 1 or contents[0].name != "attempts":
            raise StageDRuntimeIntegrityError(
                f"Stage D unit layout drifted: {unit_root.name}"
            )
        attempts_root = contents[0]
        attempt_metadata = os.lstat(attempts_root)
        if stat.S_ISLNK(attempt_metadata.st_mode) or not stat.S_ISDIR(
            attempt_metadata.st_mode
        ):
            raise StageDRuntimeIntegrityError(
                f"Stage D attempts root is unsafe: {unit_root.name}"
            )
        attempts = list(attempts_root.iterdir())
        if len(attempts) > 2:
            raise StageDRuntimeIntegrityError(
                f"Stage D unit exceeds two attempts: {unit_root.name}"
            )
        expected_names = [f"attempt-{index:06d}" for index in range(1, len(attempts) + 1)]
        if sorted(item.name for item in attempts) != expected_names:
            raise StageDRuntimeIntegrityError(
                f"Stage D attempt sequence is ambiguous: {unit_root.name}"
            )
        for attempt in attempts:
            metadata = os.lstat(attempt)
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                raise StageDRuntimeIntegrityError(
                    f"Stage D attempt entry is unsafe: {attempt}"
                )
    if observed_units != expected_units:
        raise StageDRuntimeIntegrityError(
            "Stage D unit directory coverage differs from the exact plan"
        )


def _initialize_validated_output_root(
    output_root: Path | str, plan: StageDExecutionPlan, estimate: StageDEstimate,
    preflight: StageDPreflight, context_binding: Mapping[str, Any],
) -> Path:
    root = _fresh_root(output_root)
    try:
        os.mkdir(root, 0o700)
    except FileExistsError as error:
        raise StageDRuntimeBlockedError(
            "Stage D output root was claimed concurrently"
        ) from error
    # Initialization is deliberately one-way.  Any exception below leaves a
    # non-fresh root which must be diagnosed and superseded, never reused.
    for name in _LOCK_NAMES:
        _create_lock_file(root / name)
    os.mkdir(root / "units", 0o700)
    for unit in plan.units:
        unit_root = root / "units" / unit.canonical_digest
        os.mkdir(unit_root, 0o700)
        os.mkdir(unit_root / "attempts", 0o700)
    _fsync_directory(root / "units")
    checkpoint = _initial_checkpoint(
        plan, estimate, preflight, context_binding
    )
    _atomic_create_checkpoint(root / RUNTIME_CHECKPOINT_FILENAME, checkpoint)
    _fsync_directory(root)
    return root


def initialize_stage_d_output_root(
    output_root: Path | str,
    plan: Any,
    estimate: Any,
    preflight: Any,
    *,
    execution_context: object | None = None,
    execution_gate: StageDExecutionGate | Callable[..., Any] | None = None,
) -> Path:
    """Initialize a synthetic D4 root without constructing a backend.

    Formal roots must use :func:`initialize_stage_d_access_root`; this generic
    initializer cannot bypass the durable D8.0 access boundary.
    """

    typed_plan, typed_estimate, typed_preflight = _validate_control_inputs(
        plan, estimate, preflight
    )
    checked_context = _consume_execution_gate(
        execution_gate, execution_context, operation="init",
        output_root=output_root, plan=typed_plan, estimate=typed_estimate,
        preflight=typed_preflight,
    )
    context_binding = _normalize_execution_context(
        checked_context, plan=typed_plan, estimate=typed_estimate,
        preflight=typed_preflight,
    )
    _seal_execution_context(checked_context, context_binding)
    if context_binding["formal"]:
        raise StageDExecutionAuthorizationRequired(
            "formal Stage D roots require the D8.0 access initializer"
        )
    return _initialize_validated_output_root(
        output_root, typed_plan, typed_estimate, typed_preflight, context_binding
    )


def _find_unit_row(
    checkpoint: Mapping[str, Any], unit_digest: str,
) -> tuple[int, dict[str, Any]]:
    matches = [
        (index, row) for index, row in enumerate(checkpoint["units"])
        if row["unit_digest"] == unit_digest
    ]
    if len(matches) != 1:
        raise StageDRuntimeIntegrityError(
            "runtime checkpoint unit lookup is ambiguous"
        )
    return matches[0]


def _find_group_row(
    checkpoint: Mapping[str, Any], group: StageDImageGroup,
) -> tuple[int, dict[str, Any]]:
    matches = [
        (index, row) for index, row in enumerate(checkpoint["groups"])
        if row["group_digest"] == group.canonical_digest
    ]
    if len(matches) != 1:
        raise StageDRuntimeIntegrityError(
            "runtime checkpoint group lookup is ambiguous"
        )
    return matches[0]


def _owner() -> tuple[str, str]:
    token = secrets.token_hex(32)
    return token, hashlib.sha256(token.encode("ascii")).hexdigest()


def _lease_shard(
    root: Path, plan: StageDExecutionPlan, estimate: StageDEstimate,
    preflight: StageDPreflight, context_binding: Mapping[str, Any],
    shard_id: int,
) -> tuple[str | None, str | None, dict[str, Any]]:
    with _checkpoint_lock(root):
        checkpoint = _read_valid_checkpoint_locked(
            root, plan, estimate, preflight, context_binding,
        )
        lifecycle = checkpoint["lifecycle"]["state"]
        if lifecycle == "blocked":
            raise StageDRuntimeBlockedError(
                "Stage D root is blocked",
                reasons=checkpoint["lifecycle"]["blocked_reasons"],
            )
        row = checkpoint["shards"][shard_id]
        if row["state"] == "completed":
            return None, None, checkpoint
        active = [
            item["shard_id"] for item in checkpoint["shards"]
            if item["state"] == "leased"
        ]
        if active:
            raise StageDConcurrentClaimError(
                "Stage D permits only one active shard lease: "
                + ", ".join(str(item) for item in active)
            )
        if any(
            item["state"] != "completed"
            for item in checkpoint["shards"][:shard_id]
        ):
            raise StageDRuntimeBlockedError(
                "Stage D shards require serial lower-index handoff"
            )
        if any(
            item["state"] != "pending"
            for item in checkpoint["shards"][shard_id + 1:]
        ):
            raise StageDRuntimeIntegrityError(
                "Stage D later-shard state violates serial handoff"
            )
        if row["state"] == "leased":
            raise StageDConcurrentClaimError(
                f"Stage D shard {shard_id} already has an owner; stale leases "
                "require explicit coordinator recovery and are never stolen"
            )
        if row["state"] == "blocked":
            raise StageDRuntimeBlockedError(
                f"Stage D shard {shard_id} is blocked"
            )
        if row["state"] not in {"pending", "retry_pending"}:
            raise StageDRuntimeIntegrityError(
                f"Stage D shard {shard_id} cannot be leased from {row['state']}"
            )
        token, digest = _owner()
        row["state"] = "leased"
        row["owner_token_digest"] = digest
        if checkpoint["lifecycle"]["state"] == "initialized":
            checkpoint["lifecycle"]["state"] = "running"
        checkpoint = _replace_checkpoint(
            root, checkpoint, checkpoint, plan=plan, estimate=estimate,
            preflight=preflight, context_binding=context_binding,
        )
        return token, digest, checkpoint


def _claim_group(
    root: Path, plan: StageDExecutionPlan, estimate: StageDEstimate,
    preflight: StageDPreflight, context_binding: Mapping[str, Any],
    group: StageDImageGroup, *, shard_id: int, shard_owner_digest: str,
) -> tuple[str | None, str | None]:
    with _checkpoint_lock(root):
        checkpoint = _read_valid_checkpoint_locked(
            root, plan, estimate, preflight, context_binding,
        )
        shard = checkpoint["shards"][shard_id]
        if (
            shard["state"] != "leased"
            or shard["owner_token_digest"] != shard_owner_digest
        ):
            raise StageDConcurrentClaimError(
                "Stage D shard ownership changed before group claim"
            )
        index, row = _find_group_row(checkpoint, group)
        if row["state"] == "terminal":
            return None, None
        if row["state"] == "claimed":
            raise StageDConcurrentClaimError(
                f"Stage D group {group.selection_order} already has an owner; "
                "claims are never stolen"
            )
        if row["state"] == "blocked":
            raise StageDRuntimeBlockedError(
                f"Stage D group {group.selection_order} is blocked"
            )
        if row["state"] not in {"pending", "retry_pending"}:
            raise StageDRuntimeIntegrityError(
                "Stage D group state forbids a claim"
            )
        token, digest = _owner()
        checkpoint["groups"][index] = {
            **row, "state": "claimed", "owner_token_digest": digest,
        }
        _replace_checkpoint(
            root, checkpoint, checkpoint, plan=plan, estimate=estimate,
            preflight=preflight, context_binding=context_binding,
        )
        return token, digest


def validate_stage_d_retry_authorization(
    authorization: Any, *, unit_row: Mapping[str, Any],
    execution_context_digest: str,
) -> dict[str, Any]:
    """Validate a coordinator retry record without inspecting payload values."""

    document = _plain(_require_mapping(authorization, "retry authorization"))
    expected = {
        "schema_version", "document_type", "status", "unit_digest",
        "prior_attempt_digest", "failure_code", "next_attempt_number",
        "execution_context_digest", "canonical_digest",
    }
    _require_exact_keys(document, expected, "retry authorization")
    if (
        document["schema_version"] != 1
        or document["document_type"] != _RETRY_DOCUMENT_TYPE
        or document["status"] != "COORDINATOR_AUTHORIZED_RETRY"
        or document["unit_digest"] != unit_row["unit_digest"]
        or document["failure_code"] != unit_row["failure_code"]
        or document["failure_code"] not in RETRYABLE_INFRASTRUCTURE_FAILURE_CODES
        or document["next_attempt_number"] != len(unit_row["attempts"]) + 1
        or document["next_attempt_number"] != 2
        or document["execution_context_digest"] != execution_context_digest
        or len(unit_row["attempts"]) != 1
        or document["prior_attempt_digest"]
        != unit_row["attempts"][0]["attempt_inventory_digest"]
        or document["canonical_digest"] != _self_digest(document)
    ):
        raise StageDRuntimeBlockedError(
            "coordinator retry authorization does not bind the exact failed attempt"
        )
    return document


def _retry_authorization_map(
    values: Mapping[str, Any] | Sequence[Any],
) -> dict[str, Any]:
    if isinstance(values, Mapping):
        return dict(values)
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise StageDRuntimeIntegrityError(
            "retry_authorizations must be a mapping or sequence"
        )
    result: dict[str, Any] = {}
    for item in values:
        document = _require_mapping(item, "retry authorization")
        digest = document.get("unit_digest")
        if not isinstance(digest, str) or digest in result:
            raise StageDRuntimeIntegrityError(
                "retry authorizations contain an invalid or duplicate unit"
            )
        result[digest] = item
    return result


def _consume_recovery_gate(
    gate: StageDRecoveryGate | Callable[..., Any] | None, authorization: object,
    *, unit: StageDUnit, attempt: Mapping[str, Any], owner_token_digest: str,
    failure_code: str, execution_context_digest: str, checkpoint_digest: str,
) -> object:
    selected: Any = UnavailableStageDRecoveryGate() if gate is None else gate
    validator = getattr(selected, "validate_recovery_authorization", None)
    if not callable(validator):
        validator = selected if callable(selected) else None
    if validator is None:
        raise StageDRuntimeIntegrityError(
            "recovery_gate must implement validate_recovery_authorization"
        )
    checked = validator(
        authorization, unit=unit, attempt=_plain(attempt),
        owner_token_digest=owner_token_digest, failure_code=failure_code,
        execution_context_digest=execution_context_digest,
        checkpoint_digest=checkpoint_digest,
    )
    if checked is None:
        raise StageDRecoveryAuthorizationRequired(
            "recovery gate did not return validated authorization"
        )
    return checked


def validate_stage_d_access_root(
    output_root: Path | str, plan: Any, estimate: Any, preflight: Any,
    *, execution_context: object | None = None,
    execution_gate: StageDExecutionGate | Callable[..., Any] | None = None,
    asset_contracts: Sequence[Mapping[str, Any]] = (),
    require_runnable: bool = True,
    repair_lag: bool = False,
) -> Mapping[str, Any]:
    """Independently source-recompute the D8.0 journal and lifecycle."""

    typed_plan, typed_estimate, typed_preflight = _validate_control_inputs(
        plan, estimate, preflight
    )
    contracts = _validate_access_contracts(asset_contracts, typed_plan)
    checked_context = _consume_execution_gate(
        execution_gate, execution_context, operation="validate-access",
        output_root=output_root, plan=typed_plan, estimate=typed_estimate,
        preflight=typed_preflight,
    )
    context_binding = _normalize_execution_context(
        checked_context, plan=typed_plan, estimate=typed_estimate,
        preflight=typed_preflight,
    )
    _seal_execution_context(checked_context, context_binding)
    root = _existing_root(output_root)
    with _lifecycle_lock(root, exclusive=False):
        with _checkpoint_lock(root):
            checkpoint = _read_valid_checkpoint_locked(
                root, typed_plan, typed_estimate, typed_preflight, context_binding
            )
        lifecycle, events, validated, pending, failure = _validate_access_lifecycle(
            root, plan=typed_plan,
            runtime_binding_digest=context_binding["runtime_binding_digest"],
            checkpoint_digest=_access_start_checkpoint_digest(
                typed_plan, typed_estimate, typed_preflight, context_binding,
            ),
            contracts=contracts,
            repair_lag=repair_lag,
            require_live_identity=context_binding["formal"],
        )
    if require_runnable and lifecycle["state"] != "RUNNABLE":
        raise StageDRuntimeBlockedError(
            "D8.0 root is not runnable", reasons=tuple(lifecycle["blocked_reasons"]),
        )
    return MappingProxyType({
        "state": lifecycle["state"],
        "validated_asset_count": len(validated),
        "expected_asset_count": len(typed_plan.groups),
        "journal_entry_count": len(events),
        "journal_head_digest": events[-1]["canonical_digest"],
        "live_assets_digest": lifecycle["live_assets_digest"],
        "pending_selection_order": pending,
        "failure_code": failure,
        "unit_execution_started_during_initialization": False,
        "canonical_digest": lifecycle["canonical_digest"],
    })


def _recovery_digest(value: object) -> str:
    supplied = (
        value.get("canonical_digest") if isinstance(value, Mapping)
        else getattr(value, "canonical_digest", None)
    )
    return _require_sha256(supplied, "recovery authorization digest")


def recover_stage_d_claim(
    plan: Any, *, output_root: Path | str, estimate: Any, preflight: Any,
    unit_digest: str, owner_token_digest: str, failure_code: str,
    recovery_authorization: object | None = None,
    recovery_gate: StageDRecoveryGate | Callable[..., Any] | None = None,
    execution_context: object | None = None,
    execution_gate: StageDExecutionGate | Callable[..., Any] | None = None,
    resources: Mapping[str, Any] | None = None,
    fault_injector: Callable[[str], Any] | str | None = None,
) -> str:
    """Coordinator-authorized, outcome-blind transition of one dead claim.

    The attempt must have no published scientific payload.  If its writer left
    only a valid running control, the coordinator replaces it with a canonical
    infrastructure-interruption control using ``StageDArtifactWriter``'s exact
    schema.  A missing attempt directory is initialized through the same writer
    first.  Stale claims are never recovered implicitly by ``run_stage_d_shard``.
    """

    if failure_code not in {
        "WORKER_PROCESS_CRASH_BEFORE_RESULT_PUBLISH",
        "STALE_CLAIM_OWNER_CONFIRMED_DEAD",
    }:
        raise StageDRuntimeBlockedError(
            "claim recovery code is not a dead-owner retry code"
        )
    typed_plan, typed_estimate, typed_preflight = _validate_control_inputs(
        plan, estimate, preflight
    )
    checked_context = _consume_execution_gate(
        execution_gate, execution_context, operation="recover-claim",
        output_root=output_root, plan=typed_plan, estimate=typed_estimate,
        preflight=typed_preflight,
    )
    context_binding = _normalize_execution_context(
        checked_context, plan=typed_plan, estimate=typed_estimate,
        preflight=typed_preflight,
    )
    _seal_execution_context(checked_context, context_binding)
    root = _existing_root(output_root)
    normalized_resources = _validate_resources(
        dict(_ZERO_RESOURCES) if resources is None else resources,
        "claim recovery resources",
    )

    def fault(point: str) -> None:
        selected = fault_injector
        triggered = selected == point if isinstance(selected, str) else (
            bool(selected(point)) if callable(selected) else False
        )
        if triggered:
            raise RuntimeError(f"injected Stage D recovery crash at {point}")

    with _lifecycle_lock(root, exclusive=False):
        with _checkpoint_lock(root):
            checkpoint = _read_valid_checkpoint_locked(
                root, typed_plan, typed_estimate, typed_preflight, context_binding,
                transitioning_unit_digest=unit_digest,
            )
            unit_index, row = _find_unit_row(checkpoint, unit_digest)
            if (
                row["state"] != "claimed"
                or row["owner_token_digest"] != owner_token_digest
                or len(row["attempts"]) != 1
                or row["attempts"][0]["state"] != "claimed"
            ):
                raise StageDConcurrentClaimError(
                    "claim recovery does not bind the unique active first attempt"
                )
            unit = typed_plan.units[unit_index]
            attempt = row["attempts"][0]
            checked_recovery = _consume_recovery_gate(
                recovery_gate, recovery_authorization, unit=unit, attempt=attempt,
                owner_token_digest=owner_token_digest, failure_code=failure_code,
                execution_context_digest=context_binding["runtime_binding_digest"],
                checkpoint_digest=checkpoint["canonical_digest"],
            )
            recovery_digest = _recovery_digest(checked_recovery)
            attempt_path = Path(attempt["attempt_path"])
            with _artifact_activity_lock(root, exclusive=True):
                phase = _inspect_recovery_attempt(
                    attempt_path, group=typed_plan.groups[unit.selection_order],
                    unit=unit,
                )["phase"]
                if phase in {"terminal_published", "scientific_or_partial", "unsafe"}:
                    raise StageDRuntimeBlockedError(
                        "claim recovery found terminal, partial scientific, or unsafe evidence"
                    )
                if phase in {"running", "manifest_published"}:
                    from .stage_d_artifacts import (
                        MANIFEST_FILENAME, STATUS_FILENAME,
                        INFRASTRUCTURE_INTERRUPTION_STATUS,
                    )
                    interruption = {
                        "code": failure_code, "resources": normalized_resources,
                    }
                    manifest = _strict_control_json(
                        attempt_path / MANIFEST_FILENAME, "recovery attempt manifest"
                    )
                    if phase == "running":
                        manifest = {
                            **manifest, "status": INFRASTRUCTURE_INTERRUPTION_STATUS,
                            "terminal": False, "complete": False,
                            "failure": interruption,
                        }
                        _atomic_replace_control_json(
                            attempt_path / MANIFEST_FILENAME, manifest
                        )
                    fault("after_manifest")
                    manifest_bytes = canonical_json_bytes(manifest) + b"\n"
                    status = {
                        "schema_version": 1,
                        "document_type": "stage_d_execution_unit_status",
                        "status": INFRASTRUCTURE_INTERRUPTION_STATUS,
                        "terminal": False, "complete": False,
                        "attempt_number": attempt["attempt_number"],
                        "unit_digest": unit.canonical_digest,
                        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                        "failure": interruption,
                    }
                    _atomic_replace_control_json(
                        attempt_path / STATUS_FILENAME, status
                    )
                    fault("after_status")
                elif phase in {"missing", "empty"}:
                    if phase == "empty":
                        attempt_path.rmdir()
                    from .stage_d_artifacts import StageDArtifactWriter
                    group = typed_plan.groups[unit.selection_order]
                    writer = StageDArtifactWriter(attempt_path, unit=unit, group=group)
                    writer.mark_infrastructure_interruption(
                        code=failure_code, resources=normalized_resources
                    )
                    fault("after_status")
                elif phase != "infrastructure_published":
                    raise StageDRuntimeIntegrityError(
                        "claim recovery encountered an unknown publication phase"
                    )
                group = typed_plan.groups[unit.selection_order]
                observed = _source_validate_attempt(
                    attempt_path, plan=typed_plan, group=group, unit=unit,
                    context_binding=context_binding,
                )
            final_attempt = _signed({
                **{key: attempt[key] for key in _ATTEMPT_KEYS if key != "canonical_digest"},
                **observed,
            })
            fault("before_checkpoint")
            checkpoint["units"][unit_index] = {
                **row, "state": "retry_pending", "terminal": False,
                "owner_token_digest": None, "attempts": [final_attempt],
                "retry_authorization_digest": recovery_digest,
                "failure_class": "INFRASTRUCTURE_FAILURE",
                "failure_code": failure_code,
                "resources": observed["resources"],
            }
            group_index = unit.selection_order
            group_row = checkpoint["groups"][group_index]
            if group_row["state"] != "claimed":
                raise StageDRuntimeIntegrityError(
                    "claim recovery unit has no active group owner"
                )
            checkpoint["groups"][group_index] = {
                **group_row, "state": "retry_pending",
                "owner_token_digest": None,
            }
            shard_id = group_row["shard_id"]
            shard_row = checkpoint["shards"][shard_id]
            if shard_row["state"] != "leased":
                raise StageDRuntimeIntegrityError(
                    "claim recovery group has no active shard lease"
                )
            checkpoint["shards"][shard_id] = {
                **shard_row, "state": "retry_pending",
                "owner_token_digest": None,
            }
            updated = _replace_checkpoint(
                root, checkpoint, checkpoint, plan=typed_plan,
                estimate=typed_estimate, preflight=typed_preflight,
                context_binding=context_binding,
                transitioning_unit_digest=unit.canonical_digest,
            )
            return updated["canonical_digest"]


def _claim_unit(
    root: Path, plan: StageDExecutionPlan, estimate: StageDEstimate,
    preflight: StageDPreflight, context_binding: Mapping[str, Any],
    group: StageDImageGroup, unit: StageDUnit, *, group_owner_digest: str,
    retry_authorizations: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    with _checkpoint_lock(root):
        checkpoint = _read_valid_checkpoint_locked(
            root, plan, estimate, preflight, context_binding,
        )
        _, group_row = _find_group_row(checkpoint, group)
        if (
            group_row["state"] != "claimed"
            or group_row["owner_token_digest"] != group_owner_digest
        ):
            raise StageDConcurrentClaimError(
                "Stage D group ownership changed before unit claim"
            )
        index, row = _find_unit_row(checkpoint, unit.canonical_digest)
        if row["terminal"]:
            return None, row
        if row["state"] == "claimed":
            raise StageDConcurrentClaimError(
                f"Stage D unit {unit.canonical_digest} already has an owner"
            )
        retry_document = None
        if row["state"] == "retry_pending":
            candidate = retry_authorizations.get(unit.canonical_digest)
            if candidate is None:
                return None, row
            retry_document = validate_stage_d_retry_authorization(
                candidate, unit_row=row,
                execution_context_digest=context_binding["runtime_binding_digest"],
            )
        elif row["state"] != "pending":
            if row["state"] == "blocked":
                raise StageDRuntimeBlockedError(
                    f"Stage D unit {unit.canonical_digest} is blocked"
                )
            raise StageDRuntimeIntegrityError(
                "Stage D unit state forbids a claim"
            )
        token, owner_digest = _owner()
        number = len(row["attempts"]) + 1
        attempt_id = f"attempt-{number:06d}"
        attempt_path = root / "units" / unit.canonical_digest / "attempts" / attempt_id
        attempt = _signed({
            "attempt_id": attempt_id,
            "attempt_number": number,
            "attempt_path": str(attempt_path),
            "state": "claimed",
            "owner_token_digest": owner_digest,
            "status_sha256": None,
            "attempt_inventory_digest": None,
            "failure_class": None,
            "failure_code": None,
            "result_manifest_published": False,
            "published_l0_l1_payload_count": 0,
            "retry_authorization_digest": (
                None if retry_document is None else retry_document["canonical_digest"]
            ),
            "resources": dict(_ZERO_RESOURCES),
        })
        row = {
            **row,
            "state": "claimed",
            "terminal": False,
            "owner_token_digest": owner_digest,
            "attempts": [*row["attempts"], attempt],
            "retry_authorization_digest": (
                None if retry_document is None else retry_document["canonical_digest"]
            ),
        }
        checkpoint["units"][index] = row
        _replace_checkpoint(
            root, checkpoint, checkpoint, plan=plan, estimate=estimate,
            preflight=preflight, context_binding=context_binding,
            transitioning_unit_digest=unit.canonical_digest,
        )
        return {
            "owner_token": token,
            "owner_token_digest": owner_digest,
            "group_owner_token_digest": group_owner_digest,
            "unit_digest": unit.canonical_digest,
            "group_digest": group.canonical_digest,
            "attempt_id": attempt_id,
            "attempt_number": number,
            "attempt_path": str(attempt_path),
        }, row


def _control_mapping(value: Any, context: str) -> dict[str, Any]:
    converted = value.to_dict() if callable(getattr(value, "to_dict", None)) else value
    return _plain(_require_mapping(converted, context), context)


def _normalize_runner_result(
    value: Any, *, plan: StageDExecutionPlan, group: StageDImageGroup,
    unit: StageDUnit, context_binding: Mapping[str, Any],
    claim: Mapping[str, Any],
) -> dict[str, Any]:
    document = _control_mapping(value, "Stage D unit-runner result")
    state = document.get("state", document.get("status"))
    aliases = {
        "success": "succeeded",
        "scientific_failure": "scientific_failed",
        "infrastructure_interruption": "infrastructure_failed",
        "TERMINAL_SUCCESS": "succeeded",
        "TERMINAL_SCIENTIFIC_FAILURE": "scientific_failed",
        "RETRY_PENDING": "infrastructure_failed",
        "BLOCKED": "blocked",
    }
    state = aliases.get(state, state)
    if state not in {
        "succeeded", "scientific_failed", "infrastructure_failed",
        "blocked",
    }:
        raise StageDRuntimeIntegrityError(
            "unit runner returned an invalid control state"
        )
    attempt_id = document.get("attempt_id")
    attempt_path = document.get("attempt_path", document.get("path"))
    if attempt_id is None and isinstance(attempt_path, (str, os.PathLike)):
        attempt_id = Path(attempt_path).name
    if attempt_id != claim["attempt_id"] or os.fspath(attempt_path) != claim["attempt_path"]:
        raise StageDRuntimeIntegrityError(
            "unit runner returned an attempt outside the claimed coordinate"
        )
    if document.get("unit_digest", unit.canonical_digest) != unit.canonical_digest:
        raise StageDRuntimeIntegrityError(
            "unit runner result is bound to a different unit"
        )
    observed = _source_validate_attempt(
        Path(claim["attempt_path"]), plan=plan, group=group, unit=unit,
        context_binding=context_binding,
    )
    # The runner is an untrusted producer.  Only compare its control state and
    # identity; publication flags, counts, resources, status hash, and whole
    # inventory come exclusively from source-recomputed disk evidence.
    if state != observed["state"]:
        raise StageDRuntimeIntegrityError(
            "unit runner state disagrees with source-validated attempt"
        )
    return observed


def _resource_cap_errors(
    result: Mapping[str, Any], estimate: StageDEstimate,
) -> list[str]:
    resources = result["resources"]
    caps = estimate.launch_safety_caps
    comparisons = (
        ("l0_l1_payload_bytes", "l0_l1_bytes_per_execution_unit_max"),
        ("forward_calls", "forward_calls_per_execution_unit_max"),
        ("backward_calls", "backward_calls_per_execution_unit_max"),
        ("cuda_peak_reserved_bytes", "cuda_peak_reserved_bytes_per_execution_unit_max"),
    )
    return [
        cap for resource, cap in comparisons
        if resources[resource] > caps[cap]
    ]


def _commit_unit(
    root: Path, plan: StageDExecutionPlan, estimate: StageDEstimate,
    preflight: StageDPreflight, context_binding: Mapping[str, Any],
    unit: StageDUnit, *, claim: Mapping[str, Any], result: Mapping[str, Any],
) -> dict[str, Any]:
    normalized = _normalize_runner_result(
        result, plan=plan, group=plan.groups[unit.selection_order], unit=unit,
        context_binding=context_binding, claim=claim,
    )
    cap_errors = _resource_cap_errors(normalized, estimate)
    with _checkpoint_lock(root):
        checkpoint = _read_valid_checkpoint_locked(
            root, plan, estimate, preflight, context_binding,
            transitioning_unit_digest=unit.canonical_digest,
        )
        index, row = _find_unit_row(checkpoint, unit.canonical_digest)
        if (
            row["state"] != "claimed"
            or row["owner_token_digest"] != claim["owner_token_digest"]
            or not row["attempts"]
        ):
            raise StageDConcurrentClaimError(
                "Stage D unit owner changed before checkpoint publication"
            )
        attempt = row["attempts"][-1]
        if (
            attempt["state"] != "claimed"
            or attempt["attempt_id"] != claim["attempt_id"]
            or attempt["owner_token_digest"] != claim["owner_token_digest"]
        ):
            raise StageDRuntimeIntegrityError(
                "Stage D attempt claim binding drifted"
            )
        final_attempt = _signed({
            **{key: attempt[key] for key in _ATTEMPT_KEYS if key != "canonical_digest"},
            "state": normalized["state"],
            "status_sha256": normalized["status_sha256"],
            "attempt_inventory_digest": normalized["attempt_inventory_digest"],
            "failure_class": normalized["failure_class"],
            "failure_code": normalized["failure_code"],
            "result_manifest_published": normalized["result_manifest_published"],
            "published_l0_l1_payload_count": normalized["published_l0_l1_payload_count"],
            "resources": normalized["resources"],
        })
        state = normalized["state"]
        disposition_failure_class = normalized["failure_class"]
        disposition_failure_code = normalized["failure_code"]
        terminal = state in {"succeeded", "scientific_failed"}
        if cap_errors:
            state = "blocked"
            terminal = False
            disposition_failure_class = "LAUNCH_SAFETY_CAP_BREACH"
            disposition_failure_code = "RUNTIME_LAUNCH_SAFETY_CAP_BREACH"
        elif state == "infrastructure_failed":
            no_result = (
                normalized["result_manifest_published"] is False
                and normalized["published_l0_l1_payload_count"] == 0
            )
            if no_result and len(row["attempts"]) == 1:
                state = "retry_pending"
            else:
                state = "blocked"
                disposition_failure_class = (
                    "AMBIGUOUS_ATTEMPT_OR_OWNERSHIP" if not no_result
                    else "AUTHORIZATION_OR_CONTRACT_DRIFT"
                )
                disposition_failure_code = (
                    "INFRASTRUCTURE_FAILURE_AFTER_RESULT_PUBLICATION"
                    if not no_result else "RETRY_LIMIT_EXHAUSTED"
                )
        elif state == "blocked":
            terminal = False
        row = {
            **row,
            "state": state,
            "terminal": terminal,
            "owner_token_digest": None,
            "attempts": [*row["attempts"][:-1], final_attempt],
            "failure_class": disposition_failure_class,
            "failure_code": disposition_failure_code,
        }
        row["resources"] = _sum_resources([
            {"resources": item["resources"]} for item in row["attempts"]
        ])
        checkpoint["units"][index] = row
        if state == "blocked":
            checkpoint["lifecycle"]["state"] = "blocked"
            reason = f"{unit.canonical_digest}:{row['failure_class']}"
            checkpoint["lifecycle"]["blocked_reasons"] = list(dict.fromkeys([
                *checkpoint["lifecycle"]["blocked_reasons"], reason,
            ]))
        _replace_checkpoint(
            root, checkpoint, checkpoint, plan=plan, estimate=estimate,
            preflight=preflight, context_binding=context_binding,
            transitioning_unit_digest=unit.canonical_digest,
        )
        return row


def _finish_group(
    root: Path, plan: StageDExecutionPlan, estimate: StageDEstimate,
    preflight: StageDPreflight, context_binding: Mapping[str, Any],
    group: StageDImageGroup, *, group_owner_digest: str,
) -> str:
    with _checkpoint_lock(root):
        checkpoint = _read_valid_checkpoint_locked(
            root, plan, estimate, preflight, context_binding
        )
        index, row = _find_group_row(checkpoint, group)
        if (
            row["state"] != "claimed"
            or row["owner_token_digest"] != group_owner_digest
        ):
            raise StageDConcurrentClaimError(
                "Stage D group owner changed before release"
            )
        by_unit = {item["unit_digest"]: item for item in checkpoint["units"]}
        children = [by_unit[digest] for digest in row["unit_digests"]]
        if any(item["state"] == "claimed" for item in children):
            raise StageDRuntimeIntegrityError(
                "Stage D group cannot release with an active unit claim"
            )
        if any(item["state"] == "blocked" for item in children):
            state = "blocked"
        elif all(item["terminal"] for item in children):
            state = "terminal"
        elif any(item["state"] == "retry_pending" for item in children) and all(
            item["terminal"] or item["state"] == "retry_pending"
            for item in children
        ):
            state = "retry_pending"
        else:
            state = "blocked"
            checkpoint["lifecycle"]["state"] = "blocked"
            checkpoint["lifecycle"]["blocked_reasons"] = list(dict.fromkeys([
                *checkpoint["lifecycle"]["blocked_reasons"],
                f"{group.canonical_digest}:INCOMPLETE_GROUP_RUNNER_RETURN",
            ]))
        checkpoint["groups"][index] = {
            **row, "state": state, "terminal": state == "terminal",
            "owner_token_digest": None,
        }
        _replace_checkpoint(
            root, checkpoint, checkpoint, plan=plan, estimate=estimate,
            preflight=preflight, context_binding=context_binding,
        )
        return state


def _finish_shard(
    root: Path, plan: StageDExecutionPlan, estimate: StageDEstimate,
    preflight: StageDPreflight, context_binding: Mapping[str, Any],
    shard_id: int, *, shard_owner_digest: str,
) -> tuple[str, dict[str, Any]]:
    with _checkpoint_lock(root):
        checkpoint = _read_valid_checkpoint_locked(
            root, plan, estimate, preflight, context_binding
        )
        shard = checkpoint["shards"][shard_id]
        if (
            shard["state"] != "leased"
            or shard["owner_token_digest"] != shard_owner_digest
        ):
            raise StageDConcurrentClaimError(
                "Stage D shard owner changed before release"
            )
        by_group = {row["group_digest"]: row for row in checkpoint["groups"]}
        children = [by_group[digest] for digest in shard["group_digests"]]
        if any(row["state"] == "claimed" for row in children):
            raise StageDRuntimeIntegrityError(
                "Stage D shard cannot release with an active group claim"
            )
        if any(row["state"] == "blocked" for row in children):
            state = "blocked"
        elif all(row["terminal"] for row in children):
            state = "completed"
        elif any(row["state"] == "retry_pending" for row in children) and all(
            row["terminal"] or row["state"] == "retry_pending"
            for row in children
        ):
            state = "retry_pending"
        else:
            state = "blocked"
            checkpoint["lifecycle"]["state"] = "blocked"
            checkpoint["lifecycle"]["blocked_reasons"] = list(dict.fromkeys([
                *checkpoint["lifecycle"]["blocked_reasons"],
                f"shard-{shard_id}:INCOMPLETE_SHARD_RETURN",
            ]))
        checkpoint["shards"][shard_id] = {
            **shard, "state": state, "terminal": state == "completed",
            "owner_token_digest": None,
        }
        if state == "blocked":
            checkpoint["lifecycle"]["state"] = "blocked"
        elif all(row["terminal"] for row in checkpoint["groups"]):
            checkpoint["lifecycle"]["state"] = "complete"
        result = _replace_checkpoint(
            root, checkpoint, checkpoint, plan=plan, estimate=estimate,
            preflight=preflight, context_binding=context_binding,
        )
        return state, result


def _call_backend_factory(
    factory: Callable[..., Any], *, context: object, device: str,
) -> Any:
    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError):
        return factory(context=context, device=device)
    parameters = signature.parameters
    has_var_keywords = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    keywords: dict[str, Any] = {}
    if has_var_keywords or "context" in parameters:
        keywords["context"] = context
    if has_var_keywords or "device" in parameters:
        keywords["device"] = device
    return factory(**keywords)


def _trace_mode_for_unit(unit: Any) -> str:
    """Return the closed trace policy for one execution unit.

    New unit implementations expose ``trace_mode`` directly.  The D3 plan
    objects predate that field, so their frozen audit split is the compatibility
    source: ordinary units are fast and the bounded audit subset is full.
    """

    mode = (
        unit.get("trace_mode")
        if isinstance(unit, Mapping) and "trace_mode" in unit
        else getattr(unit, "trace_mode", None)
    )
    audit = (
        unit.get("audit")
        if isinstance(unit, Mapping)
        else getattr(unit, "audit", None)
    )
    if type(audit) is not bool:
        raise StageDRuntimeIntegrityError(
            "trace policy requires a boolean unit.audit"
        )
    expected = "full" if audit else "fast"
    if mode is None:
        mode = expected
    if mode not in {"fast", "full"}:
        raise StageDRuntimeIntegrityError(
            "unit trace_mode must be exactly 'fast' or 'full'"
        )
    if mode != expected:
        raise StageDRuntimeIntegrityError(
            "unit trace_mode disagrees with unit.audit"
        )
    return mode


def _accepts_keyword(
    callable_value: Callable[..., Any], name: str, *, variadic: bool = True,
) -> bool:
    try:
        parameters = inspect.signature(callable_value).parameters
    except (TypeError, ValueError):
        return False
    return name in parameters or (variadic and any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    ))


class _TraceModeBackendProxy:
    """Forward a per-unit trace policy through a legacy unit runner."""

    def __init__(self, backend: Any, trace_mode: str) -> None:
        self._backend = backend
        self._trace_mode = trace_mode

    def __getattr__(self, name: str) -> Any:
        value = getattr(self._backend, name)
        if name not in {"run_stage_d_unit", "run"} or not callable(value):
            return value
        if not _accepts_keyword(value, "trace_mode", variadic=False):
            return value

        def invoke(*args: Any, **kwargs: Any) -> Any:
            kwargs["trace_mode"] = self._trace_mode
            return value(*args, **kwargs)

        return invoke


def _call_unit_runner(
    runner: Callable[..., Any], unit: StageDUnit, **keywords: Any,
) -> Any:
    """Invoke a unit runner, forwarding trace policy when it supports it.

    Signature-aware forwarding keeps existing injected D4 runners usable while
    ensuring the canonical runner receives the per-unit policy explicitly.
    """

    trace_mode = _trace_mode_for_unit(unit)
    # A generic **kwargs wrapper may forward unknown keywords to a legacy
    # runner, so only an explicit parameter proves this boundary owns the new
    # contract.  Backend proxying below handles that compatibility case.
    if _accepts_keyword(runner, "trace_mode", variadic=False):
        keywords["trace_mode"] = trace_mode
    elif "backend" in keywords:
        backend = keywords["backend"]
        backend_runner = getattr(backend, "run_stage_d_unit", None)
        if not callable(backend_runner):
            backend_runner = getattr(backend, "run", None)
        if callable(backend_runner) and _accepts_keyword(
            backend_runner, "trace_mode", variadic=False
        ):
            keywords["backend"] = _TraceModeBackendProxy(
                backend, trace_mode
            )
    return runner(unit, **keywords)


def _default_group_runner() -> Callable[..., Any]:
    try:
        from .stage_d_producer import run_stage_d_image_group
    except (ImportError, AttributeError) as error:
        raise StageDRuntimeIntegrityError(
            "Stage D producer is unavailable; inject group_runner for tests"
        ) from error
    return run_stage_d_image_group


def _default_unit_runner() -> Callable[..., Any]:
    try:
        from . import stage_d_producer
    except ImportError as error:
        raise StageDRuntimeIntegrityError(
            "Stage D producer is unavailable; inject unit_runner for tests"
        ) from error
    for name in ("run_stage_d_unit", "run_stage_d_execution_unit"):
        candidate = getattr(stage_d_producer, name, None)
        if callable(candidate):
            return candidate
    raise StageDRuntimeIntegrityError(
        "Stage D producer exposes no default unit runner; inject unit_runner"
    )


_PROCESS_BACKEND_GUARD = threading.Lock()
_PROCESS_BACKEND_OWNER: tuple[int, str] | None = None


@contextmanager
def _one_process_backend_device(backend: Any, device: str) -> Iterator[None]:
    global _PROCESS_BACKEND_OWNER
    if not isinstance(device, str) or not device:
        raise StageDRuntimeIntegrityError("device must be a non-empty string")
    if not _PROCESS_BACKEND_GUARD.acquire(blocking=False):
        raise StageDConcurrentClaimError(
            "this process already owns a Stage D backend/device execution"
        )
    try:
        if _PROCESS_BACKEND_OWNER is not None:
            raise StageDConcurrentClaimError(
                "this process already owns a Stage D backend/device execution"
            )
        _PROCESS_BACKEND_OWNER = (id(backend), device)
        yield
    finally:
        _PROCESS_BACKEND_OWNER = None
        _PROCESS_BACKEND_GUARD.release()


@dataclass(frozen=True, slots=True)
class StageDShardSummary(Mapping[str, Any]):
    """Control/resource-only shard result; scientific keys cannot enter it."""

    schema_version: int
    document_type: str
    shard_id: int
    shard_count: int
    state: str
    counts: Mapping[str, int]
    resources: Mapping[str, int | float]
    checkpoint_generation: int
    checkpoint_digest: str
    execution_context_digest: str

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.document_type != _SUMMARY_DOCUMENT_TYPE:
            raise StageDRuntimeIntegrityError("Stage D shard summary identity drifted")
        if self.state not in {"completed", "retry_pending", "blocked"}:
            raise StageDRuntimeIntegrityError("Stage D shard summary state is invalid")
        object.__setattr__(self, "counts", _freeze(self.counts))
        object.__setattr__(self, "resources", _freeze(self.resources))
        _require_sha256(self.checkpoint_digest, "summary checkpoint digest")
        _require_sha256(
            self.execution_context_digest, "summary execution context digest"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "document_type": self.document_type,
            "shard_id": self.shard_id,
            "shard_count": self.shard_count,
            "state": self.state,
            "counts": _plain(self.counts),
            "resources": _plain(self.resources),
            "checkpoint_generation": self.checkpoint_generation,
            "checkpoint_digest": self.checkpoint_digest,
            "execution_context_digest": self.execution_context_digest,
        }

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())


def _summary(
    checkpoint: Mapping[str, Any], *, shard_id: int,
    visited_groups: int, claimed_groups: int, attempted_units: int,
    skipped_terminal_units: int,
) -> StageDShardSummary:
    shard = checkpoint["shards"][shard_id]
    group_set = set(shard["group_digests"])
    groups = [
        row for row in checkpoint["groups"] if row["group_digest"] in group_set
    ]
    group_orders = {row["selection_order"] for row in groups}
    units = [
        row for row in checkpoint["units"]
        if row["group_selection_order"] in group_orders
    ]
    counts = {
        "expected_groups": len(groups),
        "visited_groups": visited_groups,
        "claimed_groups": claimed_groups,
        "terminal_groups": sum(row["terminal"] for row in groups),
        "expected_units": len(units),
        "attempted_units": attempted_units,
        "terminal_units": sum(row["terminal"] for row in units),
        "succeeded_units": sum(row["state"] == "succeeded" for row in units),
        "scientific_failed_units": sum(
            row["state"] == "scientific_failed" for row in units
        ),
        "retry_pending_units": sum(
            row["state"] == "retry_pending" for row in units
        ),
        "blocked_units": sum(row["state"] == "blocked" for row in units),
        "skipped_terminal_units": skipped_terminal_units,
    }
    return StageDShardSummary(
        schema_version=1, document_type=_SUMMARY_DOCUMENT_TYPE,
        shard_id=shard_id, shard_count=len(checkpoint["shards"]),
        state=shard["state"], counts=counts, resources=_sum_resources(units),
        checkpoint_generation=checkpoint["generation"],
        checkpoint_digest=checkpoint["canonical_digest"],
        execution_context_digest=checkpoint["execution_context"][
            "runtime_binding_digest"
        ],
    )


def run_stage_d_shard(
    plan: Any,
    *,
    output_root: Path | str,
    estimate: Any,
    preflight: Any,
    shard_id: int,
    execution_context: object | None = None,
    execution_gate: StageDExecutionGate | Callable[..., Any] | None = None,
    backend_factory: Callable[..., Any] | None = None,
    group_runner: Callable[..., Any] | None = None,
    unit_runner: Callable[..., Any] | None = None,
    device: str = "cpu",
    retry_authorizations: Mapping[str, Any] | Sequence[Any] = (),
    asset_contracts: Sequence[Mapping[str, Any]] = (),
) -> StageDShardSummary:
    """Run one modulo shard with atomic group ownership and recovery.

    The injected unit runner has the narrow form::

        runner(unit, *, output_root, backend, context, claim,
               retry_authorization=None) -> control mapping

    The default group runner is ``run_stage_d_image_group``.  It receives a
    wrapped unit callback, so producer ordering remains NNS then matched and a
    scientific failure or retry-pending first unit does not suppress its pair.
    """

    typed_plan, typed_estimate, typed_preflight = _validate_control_inputs(
        plan, estimate, preflight
    )
    if type(shard_id) is not int:
        raise StageDRuntimeIntegrityError("shard_id must be an integer")
    if (
        isinstance(execution_context, Mapping)
        and "authorized_next_shard" in execution_context
        and execution_context["authorized_next_shard"] != shard_id
    ):
        raise StageDExecutionAuthorizationRequired(
            "D8 continuation authorization is bound to a different shard"
        )
    checked_context = _consume_execution_gate(
        execution_gate, execution_context, operation="run-shard",
        output_root=output_root, plan=typed_plan, estimate=typed_estimate,
        preflight=typed_preflight,
    )
    context_binding = _normalize_execution_context(
        checked_context, plan=typed_plan, estimate=typed_estimate,
        preflight=typed_preflight,
    )
    runtime_context = _seal_execution_context(checked_context, context_binding)
    shard_count = context_binding["shard_count"]
    if shard_id < 0 or shard_id >= shard_count:
        raise StageDRuntimeIntegrityError(
            "shard_id must select the validated zero-based shard range"
        )
    root = _existing_root(output_root)
    retry_map = _retry_authorization_map(retry_authorizations)
    visited_groups = claimed_groups = attempted_units = skipped_terminal_units = 0
    with _lifecycle_lock(root, exclusive=False):
        with _checkpoint_lock(root):
            checkpoint = _read_valid_checkpoint_locked(
                root, typed_plan, typed_estimate, typed_preflight, context_binding
            )
        if context_binding["formal"]:
            contracts = _validate_access_contracts(asset_contracts, typed_plan)
            lifecycle, _events, _validated, _pending, _failure = (
                _validate_access_lifecycle(
                    root, plan=typed_plan,
                    runtime_binding_digest=context_binding["runtime_binding_digest"],
                    checkpoint_digest=_access_start_checkpoint_digest(
                        typed_plan, typed_estimate, typed_preflight,
                        context_binding,
                    ),
                    contracts=contracts,
                    require_live_identity=True,
                )
            )
            if lifecycle["state"] != "RUNNABLE":
                raise StageDRuntimeBlockedError(
                    "formal Stage D shards require a RUNNABLE D8.0 access gate"
                )
        shard_token, shard_owner_digest, checkpoint = _lease_shard(
            root, typed_plan, typed_estimate, typed_preflight, context_binding,
            shard_id,
        )
        if shard_token is None or shard_owner_digest is None:
            return _summary(
                checkpoint, shard_id=shard_id, visited_groups=0,
                claimed_groups=0, attempted_units=0,
                skipped_terminal_units=checkpoint["shards"][shard_id][
                    "group_count"
                ] * 2,
            )

        selected_backend = None
        current_group: StageDImageGroup | None = None
        current_group_owner: str | None = None
        try:
            # Factory construction is intentionally after gate, safe-root,
            # closed-layout, checkpoint validation, and atomic shard lease.
            if backend_factory is None:
                raise StageDRuntimeIntegrityError(
                    "run_stage_d_shard requires a post-gate backend_factory"
                )
            selected_backend = _call_backend_factory(
                backend_factory, context=runtime_context, device=device
            )
            selected_group_runner = (
                _default_group_runner() if group_runner is None else group_runner
            )
            selected_unit_runner = (
                _default_unit_runner() if unit_runner is None else unit_runner
            )
            if not callable(selected_group_runner) or not callable(selected_unit_runner):
                raise StageDRuntimeIntegrityError(
                    "group_runner and unit_runner must be callable"
                )
            groups = tuple(
                group for group in typed_plan.groups
                if context_binding["shard_mapping"][group.selection_order] == shard_id
            )
            with _one_process_backend_device(selected_backend, device):
                for group in groups:
                    visited_groups += 1
                    group_token, group_owner_digest = _claim_group(
                        root, typed_plan, typed_estimate, typed_preflight,
                        context_binding, group, shard_id=shard_id,
                        shard_owner_digest=shard_owner_digest,
                    )
                    if group_token is None or group_owner_digest is None:
                        skipped_terminal_units += 2
                        continue
                    current_group = group
                    current_group_owner = group_owner_digest
                    claimed_groups += 1

                    def run_one(unit: StageDUnit, *, backend: Any, device: str) -> dict[str, Any]:
                        nonlocal attempted_units, skipped_terminal_units
                        claim, row = _claim_unit(
                            root, typed_plan, typed_estimate, typed_preflight,
                            context_binding, group, unit,
                            group_owner_digest=group_owner_digest,
                            retry_authorizations=retry_map,
                        )
                        if claim is None:
                            skipped_terminal_units += int(row is not None and row["terminal"])
                            state = (
                                "SKIPPED_VALID_TERMINAL"
                                if row is not None and row["terminal"]
                                else "RETRY_PENDING"
                            )
                            return {
                                "state": state,
                                "unit_digest": unit.canonical_digest,
                                "pair_binding_digest": unit.pair_binding_digest,
                            }
                        attempted_units += 1
                        retry_document = retry_map.get(unit.canonical_digest)
                        try:
                            with _artifact_activity_lock(root, exclusive=False):
                                raw = _call_unit_runner(
                                    selected_unit_runner, unit,
                                    output_root=root, backend=backend,
                                    context=runtime_context, claim=claim,
                                    group=group, plan=typed_plan,
                                    device=device,
                                    retry_authorization=retry_document,
                                )
                        except (KeyboardInterrupt, SystemExit):
                            # Process-control interruption intentionally leaves
                            # the durable claim; workers may never steal it.
                            raise
                        except Exception:
                            # An exception without a returned, durable control
                            # result is a crash, not a retry classification.
                            # Preserve shard/group/unit ownership so another
                            # worker cannot steal an ambiguous or stale claim.
                            raise
                        row = _commit_unit(
                            root, typed_plan, typed_estimate, typed_preflight,
                            context_binding, unit, claim=claim, result=raw,
                        )
                        state = {
                            "succeeded": "TERMINAL_SUCCESS",
                            "scientific_failed": "TERMINAL_SCIENTIFIC_FAILURE",
                            "retry_pending": "RETRY_PENDING",
                            "blocked": "BLOCKED",
                        }[row["state"]]
                        return {
                            "state": state,
                            "unit_digest": unit.canonical_digest,
                            "pair_binding_digest": unit.pair_binding_digest,
                            "attempt_digest": row["attempts"][-1]["canonical_digest"],
                            "failure_class": row["failure_class"],
                            "failure_code": row["failure_code"],
                        }

                    selected_group_runner(
                        group, backend=selected_backend, run_unit=run_one,
                        device=device, on_unit_terminal=None,
                    )
                    state = _finish_group(
                        root, typed_plan, typed_estimate, typed_preflight,
                        context_binding, group,
                        group_owner_digest=group_owner_digest,
                    )
                    current_group = None
                    current_group_owner = None
                    if state == "blocked":
                        break
            state, checkpoint = _finish_shard(
                root, typed_plan, typed_estimate, typed_preflight,
                context_binding, shard_id,
                shard_owner_digest=shard_owner_digest,
            )
        except BaseException:
            # A crash before the normal release path intentionally leaves the
            # durable ownership chain in place.  Only a coordinator-confirmed
            # recovery event may classify it; workers never auto-steal it.
            raise
    return _summary(
        checkpoint, shard_id=shard_id, visited_groups=visited_groups,
        claimed_groups=claimed_groups, attempted_units=attempted_units,
        skipped_terminal_units=skipped_terminal_units,
    )


@dataclass(frozen=True, slots=True)
class _SyntheticD4ExecutionContext:
    formal: bool
    authorization_scope: str
    execution_authorization: str
    authorization_slot: Mapping[str, Any]
    shard_count: int
    shard_mapping: Mapping[int, int]
    plan_digest: str
    budget_digest: str
    preflight_digest: str
    code_binding: Mapping[str, Any]
    dependency_binding: Mapping[str, Any]
    model_binding: Mapping[str, Any]
    source_bindings: Mapping[str, Any]
    access_attestation: Mapping[str, bool]
    canonical_digest: str
    _issuer: object = field(repr=False, compare=False)


_SYNTHETIC_ISSUER = object()


class _SyntheticD4ExecutionGate:
    def validate_execution_context(
        self, context: object | None, **_kwargs: Any,
    ) -> _SyntheticD4ExecutionContext:
        if (
            not isinstance(context, _SyntheticD4ExecutionContext)
            or context._issuer is not _SYNTHETIC_ISSUER
        ):
            raise StageDExecutionAuthorizationRequired(
                "D4 synthetic gate requires its private test-only permit"
            )
        return context


def _issue_synthetic_d4_test_context(
    plan: StageDExecutionPlan, estimate: StageDEstimate,
    preflight: StageDPreflight, *, shard_count: int,
) -> tuple[_SyntheticD4ExecutionContext, _SyntheticD4ExecutionGate]:
    """Private test helper; never creates a formal execution capability."""

    typed_plan, typed_estimate, typed_preflight = _validate_control_inputs(
        plan, estimate, preflight
    )
    count = validate_stage_d_shard_count(shard_count)
    mapping = dict(validate_stage_d_shard_mapping(typed_plan, count))
    synthetic_sha = {
        name: hashlib.sha256(("stage-d-d4-synthetic-" + name).encode()).hexdigest()
        for name in ("dependency", "model", "d0", "d1", "d2", "c10")
    }
    values = {
        "formal": False,
        "authorization_scope": _SYNTHETIC_SCOPE,
        "execution_authorization": _NOT_ISSUED,
        "authorization_slot": {
            "status": "NOT_ISSUED", "purpose": _SYNTHETIC_SCOPE,
        },
        "shard_count": count,
        "shard_mapping": [mapping[index] for index in range(500)],
        "plan_digest": typed_plan.canonical_digest,
        "budget_digest": typed_estimate.canonical_digest,
        "preflight_digest": typed_preflight.canonical_digest,
        "code_binding": {
            "commit": "d" * 40, "tree": "e" * 40, "clean": True,
        },
        "dependency_binding": {
            "status": "SYNTHETIC_NO_DEPENDENCY_AUTHORITY",
            "lock_sha256": synthetic_sha["dependency"],
        },
        "model_binding": {
            "status": "SYNTHETIC_NO_MODEL_ACCESS",
            "config_sha256": synthetic_sha["model"],
        },
        "source_bindings": {
            name: {
                "status": "SYNTHETIC_NO_RESERVE_ACCESS",
                "canonical_digest": synthetic_sha[name],
            }
            for name in ("d0", "d1", "d2", "c10")
        },
        "access_attestation": {
            "reserve_image_bytes_accessed": False,
            "model_accessed": False,
            "cuda_accessed": False,
            "outcome_accessed": False,
        },
    }
    digest = canonical_sha256(values)
    context = _SyntheticD4ExecutionContext(
        **{**values, "shard_mapping": MappingProxyType(mapping)},
        canonical_digest=digest, _issuer=_SYNTHETIC_ISSUER,
    )
    return context, _SyntheticD4ExecutionGate()


__all__ = [
    "ACCESS_JOURNAL_DIRECTORY", "ACCESS_LIFECYCLE_FILENAME",
    "ARTIFACT_ACTIVITY_LOCK_FILENAME", "CHECKPOINT_LOCK_FILENAME",
    "LIFECYCLE_LOCK_FILENAME", "NONRETRYABLE_FAILURE_CLASSES",
    "RETRYABLE_INFRASTRUCTURE_FAILURE_CODES",
    "RUNTIME_CHECKPOINT_FILENAME", "StageDConcurrentClaimError",
    "StageDExecutionAuthorizationRequired", "StageDExecutionGate",
    "StageDRecoveryAuthorizationRequired", "StageDRecoveryGate",
    "StageDRuntimeBlockedError", "StageDRuntimeError",
    "StageDRuntimeIntegrityError", "StageDShardSummary",
    "UnavailableD7ExecutionGate", "UnavailableStageDRecoveryGate",
    "ValidatedExecutionContext", "initialize_stage_d_access_root",
    "initialize_stage_d_output_root", "validate_stage_d_access_root",
    "recover_stage_d_claim", "run_stage_d_shard",
    "validate_stage_d_retry_authorization",
]
