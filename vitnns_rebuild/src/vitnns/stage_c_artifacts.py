"""Crash-safe, fail-closed artifact persistence for Stage C execution units.

Stage C has retention rules that intentionally differ from the older Stage B
profiles: L1 is float32 14 by 14, L2 is heterogeneous float32 audit data, and
L3 is display-only PNG data.  Every execution attempt is a new immutable
directory.  A status file is the commit marker and is always published after
the manifest it hashes.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import stat
import struct
import uuid
import zipfile
import zlib
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np


DEFAULT_L0_L1_CAP_BYTES = 64 * 1024 * 1024
DEFAULT_L2_CAP_BYTES = 1024 * 1024 * 1024
SCHEMA_VERSION = 1
MANIFEST_FILENAME = "manifest.json"
STATUS_FILENAME = "status.json"
PROVENANCE_PATH = "L0/provenance.json"
PROVENANCE_DOCUMENT_TYPE = "stage_c_execution_unit_provenance"
MANIFEST_DOCUMENT_TYPE = "stage_c_execution_unit_manifest"
STATUS_DOCUMENT_TYPE = "stage_c_execution_unit_status"
RUNNER_TELEMETRY_SCHEMA_VERSION = 1
_REQUIRED_PHASES = {
    "attribution_or_map_compute",
    "fidelity_compute",
    "integrity_checking",
    "artifact_finalize",
}

FAULT_AFTER_RUNNING = "after_running"
FAULT_AFTER_PAYLOAD = "after_payload"
FAULT_AFTER_MANIFEST = "after_manifest"
FAULT_BEFORE_COMPLETE_STATUS = "before_complete_status"
FAULT_POINTS = frozenset(
    {
        FAULT_AFTER_RUNNING,
        FAULT_AFTER_PAYLOAD,
        FAULT_AFTER_MANIFEST,
        FAULT_BEFORE_COMPLETE_STATUS,
    }
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_COMPONENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_SAFE_KIND_RE = re.compile(r"^[a-z][a-z0-9_.-]*$")
_ATTEMPT_RE = re.compile(r"^attempt-(\d{6})$")
_GRANULAR_RE = re.compile(
    r"(?:^|[_.-])(?:per[_.-]?)?(?:quadrature[_.-]?)?"
    r"(?:step|pixel|dimension|dim)[_.-]?\d+(?:[_.-]\d+)*(?:$|[_.-])",
    re.IGNORECASE,
)
_STRICT_PROVENANCE_BINDING_FIELDS = {
    "code", "model", "input", "dependencies", "registry", "protocol",
    "acceptance", "configuration", "selection", "method", "target", "map",
    "baseline", "ig_schedule", "seeds", "perturbations", "status", "failure",
    "resources", "retention", "artifact_inventory",
}
_VALUE_PATH_IG_METHOD_ID = (
    "conditional_fixed_clean_attention_head_direction_value_path_ig"
)
_VALUE_PATH_CONVERGENCE_ERROR = (
    "vitnns.stage_c_value_path.StageCValuePathConvergenceError"
)
_FULL_MODEL_IG_METHOD_ID = "full_model_class_logit_ig"
_FULL_MODEL_IG_CONVERGENCE_ERROR = (
    "vitnns.stage_c_class_logit.StageCClassLogitIGConvergenceError"
)
_FAILURE_OBSERVABILITY_PATH = "L0/execution_lifecycle.json"
_VALUE_PATH_IG_STEPS = (512, 1024, 2048, 4096, 8192)
_VALUE_PATH_IG_CHUNK_SIZE = 8
_VALUE_PATH_IG_ATOL = 1e-4
_VALUE_PATH_IG_RTOL = 1e-2
_VALUE_PATH_IG_METADATA_FIELDS = {
    "method", "quadrature", "integration_dtype", "input_shape",
    "step_schedule", "step_chunk_size", "atol", "rtol", "scope",
    "target", "includes_residual_paths",
    "is_complete_attention_attribution", "fixed_clean_attention",
    "frozen_direction",
}
_MAP_WORK_ITEM_FIELDS = {
    "kind", "input_variant", "repeat_index", "fidelity_scale",
    "curve_direction", "ranking_kind", "replicate_index",
    "fraction_index", "fraction", "pairing_block_index",
    "pairing_head_index",
}


class StageCArtifactError(RuntimeError):
    """Base error for Stage C artifact persistence."""


class ArtifactPolicyError(StageCArtifactError, ValueError):
    """An artifact violates a frozen L0/L1/L2/L3 boundary."""


class ArtifactBudgetError(StageCArtifactError):
    """The exact attempt would exceed a byte cap."""


class AttemptNotEmptyError(StageCArtifactError, FileExistsError):
    """An attempt directory is not fresh and empty."""


class ArtifactWriterStateError(StageCArtifactError):
    """An operation is illegal for the current writer state."""


class InjectedArtifactCrash(StageCArtifactError):
    """Named fault used to exercise crash recovery."""

    def __init__(self, point: str):
        self.point = point
        super().__init__(f"injected Stage C artifact crash at {point}")


def _value_path_ig_attempt_values(
    attempt: Any, *, context: str, errors: list[str],
) -> tuple[int | None, bool | None]:
    required = {"steps", "delta", "sum", "abs", "rel", "pass"}
    if not isinstance(attempt, dict) or set(attempt) != required:
        errors.append(f"{context} has missing or unknown fields")
        return None, None
    steps = attempt.get("steps")
    if type(steps) is not int or steps <= 0:
        errors.append(f"{context} steps are invalid")
        steps = None
    values: dict[str, float] = {}
    for field in ("delta", "sum", "abs", "rel"):
        raw = attempt.get(field)
        if (
            not isinstance(raw, list) or len(raw) != 1
            or isinstance(raw[0], bool)
            or not isinstance(raw[0], (int, float))
            or not math.isfinite(float(raw[0]))
        ):
            errors.append(f"{context} {field} is invalid")
        else:
            values[field] = float(raw[0])
    raw_pass = attempt.get("pass")
    passed = None
    if (
        not isinstance(raw_pass, list) or len(raw_pass) != 1
        or type(raw_pass[0]) is not bool
    ):
        errors.append(f"{context} pass is invalid")
    else:
        passed = raw_pass[0]
    if len(values) == 4:
        expected_abs = abs(values["delta"] - values["sum"])
        expected_rel = expected_abs / max(
            abs(values["delta"]), np.finfo(np.float32).eps
        )
        if not math.isclose(
            values["abs"], expected_abs, rel_tol=1e-6, abs_tol=1e-12
        ):
            errors.append(f"{context} conservation abs is inconsistent")
        if not math.isclose(
            values["rel"], expected_rel, rel_tol=1e-6, abs_tol=1e-12
        ):
            errors.append(f"{context} conservation rel is inconsistent")
        expected_pass = expected_abs <= (
            _VALUE_PATH_IG_ATOL
            + _VALUE_PATH_IG_RTOL * abs(values["delta"])
        )
        if passed is not None and passed is not expected_pass:
            errors.append(f"{context} conservation pass is inconsistent")
    return steps, passed


def _bound_class_index(provenance: Any, *, context: str, errors: list[str]) -> int | None:
    """Return the frozen class index from the signed provenance document."""

    bindings = provenance.get("bindings") if isinstance(provenance, Mapping) else None
    target = bindings.get("target") if isinstance(bindings, Mapping) else None
    identity = target.get("identity") if isinstance(target, Mapping) else None
    frozen = identity.get("frozen_target") if isinstance(identity, Mapping) else None
    class_index = frozen.get("class_index") if isinstance(frozen, Mapping) else None
    if type(class_index) is not int or class_index < 0:
        errors.append(f"{context} has no valid provenance-bound class_index")
        return None
    return class_index


def _validate_ig_map_record(
    value: Any, *, identity: Mapping[str, Any], expected_variant: tuple[str, Any],
    failed: bool, failure_details: Mapping[str, Any], method_id: str,
    class_index: int | None, context: str, errors: list[str],
) -> None:
    required = {
        "method_id", "input_variant", "repeat_index", "status",
        "ig_result", "attempted_steps_total", "backward_invocations",
        "actual_calls", "exception",
    }
    if not isinstance(value, dict) or set(value) != required:
        errors.append(f"{context} has missing or unknown fields")
        return
    if value.get("method_id") != method_id:
        errors.append(f"{context} method_id differs from unit identity")
    if (value.get("input_variant"), value.get("repeat_index")) != expected_variant:
        errors.append(f"{context} map coverage/order differs from frozen prefix")
    expected_status = "failed" if failed else "completed"
    if value.get("status") != expected_status:
        errors.append(f"{context} status is invalid")

    exception = value.get("exception")
    if failed:
        expected_exception = {
            "type": failure_details.get("type"),
            "message": failure_details.get("message"),
        }
        if exception != expected_exception:
            errors.append(f"{context} exception differs from terminal failure")
    elif exception is not None:
        errors.append(f"{context} completed map contains an exception")

    summary = value.get("ig_result")
    summary_fields = {"selected_steps", "converged", "attempts", "metadata"}
    if not isinstance(summary, dict) or set(summary) != summary_fields:
        errors.append(f"{context} IG result has missing or unknown fields")
        return
    if type(summary.get("converged")) is not bool:
        errors.append(f"{context} IG converged flag is invalid")
    elif summary["converged"] is failed:
        errors.append(f"{context} IG converged state contradicts map status")

    attempts = summary.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        errors.append(f"{context} IG attempts are missing")
        steps: list[int] = []
        passes: list[bool] = []
    else:
        steps = []
        passes = []
        for index, attempt in enumerate(attempts):
            step, passed = _value_path_ig_attempt_values(
                attempt, context=f"{context} IG attempt {index}", errors=errors
            )
            if step is not None:
                steps.append(step)
            if passed is not None:
                passes.append(passed)
    expected_steps = list(_VALUE_PATH_IG_STEPS if failed else _VALUE_PATH_IG_STEPS[:len(steps)])
    if steps != expected_steps:
        errors.append(f"{context} IG schedule is not the frozen {'complete schedule' if failed else 'prefix'}")
    if not steps or summary.get("selected_steps") != steps[-1]:
        errors.append(f"{context} IG selected_steps is invalid")
    expected_passes = (
        [False] * len(passes)
        if failed else [False] * (len(passes) - 1) + [True]
    )
    if passes != expected_passes:
        errors.append(f"{context} IG stop/pass rule is invalid")

    expected_attempted = sum(steps)
    expected_backward = sum(
        math.ceil(step / _VALUE_PATH_IG_CHUNK_SIZE) for step in steps
    )
    metadata = summary.get("metadata")
    expected_metadata: dict[str, Any] = {
        "method": "integrated_gradients",
        "quadrature": "gauss_legendre",
        "integration_dtype": "float32",
        "input_shape": [1, 3, 224, 224],
        "step_schedule": list(_VALUE_PATH_IG_STEPS),
        "step_chunk_size": _VALUE_PATH_IG_CHUNK_SIZE,
        "atol": _VALUE_PATH_IG_ATOL,
        "rtol": _VALUE_PATH_IG_RTOL,
    }
    if method_id == _VALUE_PATH_IG_METHOD_ID:
        expected_metadata.update({
            "scope": "fixed_clean_attention_head_direction_value_path",
            "target": {
                "kind": "fixed_clean_attention_cls_head_direction",
                "block_index": identity.get("block_index"),
                "head_index": identity.get("head_index"),
                "query_token": 0,
            },
            "includes_residual_paths": False,
            "is_complete_attention_attribution": False,
            "fixed_clean_attention": True,
            "frozen_direction": True,
        })
        expected_metadata_fields = set(_VALUE_PATH_IG_METADATA_FIELDS)
    else:
        expected_metadata.update({
            "scope": "full_model",
            "target": {"kind": "class_logit", "class_index": class_index},
            "includes_residual_paths": True,
        })
        expected_metadata_fields = {
            "method", "quadrature", "integration_dtype", "input_shape",
            "step_schedule", "step_chunk_size", "atol", "rtol",
            "scope", "target", "includes_residual_paths",
        }
    if method_id == _VALUE_PATH_IG_METHOD_ID and not failed:
        expected_metadata.update({
            "intervention_calls": expected_backward + 2,
            "pre_projection_calls": expected_backward + 2,
        })
        expected_metadata_fields.update({
            "intervention_calls", "pre_projection_calls",
        })
    if (
        not isinstance(metadata, dict)
        or set(metadata) != expected_metadata_fields
        or any(metadata.get(key) != expected for key, expected in expected_metadata.items())
    ):
        errors.append(f"{context} IG metadata differs from frozen contract")
    if value.get("attempted_steps_total") != expected_attempted:
        errors.append(f"{context} attempted_steps_total is inconsistent")
    if value.get("backward_invocations") != expected_backward:
        errors.append(f"{context} backward_invocations is inconsistent")
    calls = value.get("actual_calls")
    if not isinstance(calls, dict) or set(calls) != {
        "accounting_unit", "forward_calls", "backward_calls"
    }:
        errors.append(f"{context} actual_calls is invalid")
    else:
        if calls.get("accounting_unit") != "batched_model_invocation":
            errors.append(f"{context} call accounting unit is invalid")
        if calls.get("backward_calls") != expected_backward:
            errors.append(f"{context} actual backward_calls is inconsistent")
        forward_overhead = (
            4 if method_id == _VALUE_PATH_IG_METHOD_ID
            or expected_variant[0] not in {"clean", "repeat_clean"}
            else 2
        )
        if calls.get("forward_calls") != expected_backward + forward_overhead:
            errors.append(f"{context} actual forward_calls is inconsistent")


def _load_bound_plan_map_sequence(
    attempt_root: Path, manifest: Mapping[str, Any], provenance: Any,
    errors: list[str],
) -> list[tuple[str, Any]] | None:
    """Load the frozen map order from the plan bound by provenance."""

    context = "value-path convergence failure observability"
    if not isinstance(provenance, Mapping):
        errors.append(f"{context} cannot bind map order without provenance")
        return None
    bindings = provenance.get("bindings")
    configuration = (
        bindings.get("configuration")
        if isinstance(bindings, Mapping) else None
    )
    expected_digest = (
        configuration.get("execution_plan_sha256")
        if isinstance(configuration, Mapping) else None
    )
    if not isinstance(expected_digest, str) or not _SHA256_RE.fullmatch(
        expected_digest
    ):
        errors.append(f"{context} has no valid execution-plan binding")
        return None

    # attempts live at <root>/units/<unit>/attempts/attempt-NNNNNN.
    try:
        output_root = attempt_root.parents[3]
    except IndexError:
        errors.append(f"{context} cannot locate execution_plan.json")
        return None
    plan_path = output_root / "execution_plan.json"
    try:
        _assert_no_symlink_components(plan_path)
        raw = plan_path.read_bytes()
        plan = _strict_json_loads(raw)
    except (OSError, ValueError, TypeError, ArtifactPolicyError) as exc:
        errors.append(f"{context} cannot load bound execution plan: {exc}")
        return None
    if not isinstance(plan, dict) or _json_file_bytes(plan) != raw:
        errors.append(f"{context} execution plan is not canonical")
        return None
    payload = dict(plan)
    claimed_digest = payload.pop("canonical_digest", None)
    observed_digest = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    if claimed_digest != expected_digest or observed_digest != expected_digest:
        errors.append(f"{context} execution plan does not match provenance")
        return None
    units = plan.get("units")
    identity = manifest.get("unit_identity")
    matching = [
        unit for unit in units if isinstance(unit, Mapping)
        and unit.get("key") == identity
    ] if isinstance(units, list) else []
    if len(matching) != 1:
        errors.append(f"{context} unit is absent or duplicated in execution plan")
        return None
    work_items = matching[0].get("work_items")
    if not isinstance(work_items, list):
        errors.append(f"{context} execution-plan work_items are missing")
        return None
    result: list[tuple[str, Any]] = []
    for index, item in enumerate(work_items):
        if not isinstance(item, Mapping) or set(item) != _MAP_WORK_ITEM_FIELDS:
            errors.append(
                f"{context} execution-plan work item {index} is malformed"
            )
            return None
        if item.get("kind") != "map":
            continue
        if any(
            item.get(field) is not None
            for field in _MAP_WORK_ITEM_FIELDS
            - {"kind", "input_variant", "repeat_index"}
        ):
            errors.append(
                f"{context} execution-plan map item {index} has curve fields"
            )
            return None
        result.append((item.get("input_variant"), item.get("repeat_index")))
    if not result or result[0] != ("clean", 0):
        errors.append(f"{context} execution-plan map order is invalid")
        return None
    return result


def _failed_ig_maps(observability: Any) -> list[Mapping[str, Any]]:
    if not isinstance(observability, Mapping):
        return []
    maps = observability.get("map_execution_records")
    if not isinstance(maps, list):
        return []
    return [
        item for item in maps
        if isinstance(item, Mapping)
        and item.get("status") == "failed"
        and isinstance(item.get("ig_result"), Mapping)
    ]


def _validate_ig_convergence_failure_observability(
    attempt_root: Path, manifest: Mapping[str, Any], status: Any,
    provenance: Any, observability: Any,
    errors: list[str],
) -> None:
    identity = manifest.get("unit_identity")
    failure = manifest.get("failure")
    details = failure.get("details") if isinstance(failure, Mapping) else None
    method_id = identity.get("method_id") if isinstance(identity, Mapping) else None
    profiles = {
        _VALUE_PATH_IG_METHOD_ID: _VALUE_PATH_CONVERGENCE_ERROR,
        _FULL_MODEL_IG_METHOD_ID: _FULL_MODEL_IG_CONVERGENCE_ERROR,
    }
    is_production_ig_failure = (
        manifest.get("status") == "failed"
        and isinstance(identity, Mapping)
        and "canonical_digest" in identity
        and method_id in profiles
    )
    if not is_production_ig_failure:
        return
    context = f"{method_id} convergence failure observability"
    expected_failure_type = profiles[method_id]
    claimed_convergence_failure = (
        isinstance(details, Mapping)
        and details.get("type") == expected_failure_type
    )
    # Do not trust a record's self-declared method_id when deciding whether IG
    # evidence exists.  Otherwise jointly re-signing that field and the
    # terminal exception type could disable semantic validation.
    failed_ig_maps = _failed_ig_maps(observability)
    has_convergence_evidence = bool(failed_ig_maps)
    if claimed_convergence_failure != has_convergence_evidence:
        errors.append(
            f"{context} terminal failure type and failed IG evidence mismatch"
        )
    if not claimed_convergence_failure and not has_convergence_evidence:
        return
    records = manifest.get("artifacts")
    evidence_records = [
        item for item in records if isinstance(item, Mapping)
        and item.get("path") == _FAILURE_OBSERVABILITY_PATH
    ] if isinstance(records, list) else []
    if len(evidence_records) != 1 or evidence_records[0].get("kind") != "failure_evidence":
        errors.append(f"{context} requires exactly one failure_evidence artifact")
    if not isinstance(observability, dict) or set(observability) != {
        "lifecycle", "map_execution_records"
    }:
        errors.append(f"{context} has missing or unknown fields")
        return
    telemetry = status.get("runner_telemetry") if isinstance(status, Mapping) else None
    # The runner performs one terminal validation before publishing telemetry,
    # then a second validation with ``require_runner_telemetry=True``.  Let the
    # first pass validate the immutable failure payload; when telemetry is
    # present, bind the duplicated lifecycle evidence exactly.
    if telemetry is not None and (
        not isinstance(telemetry, Mapping)
        or observability.get("lifecycle") != telemetry.get("lifecycle")
    ):
        errors.append(f"{context} lifecycle differs from status telemetry")
    lifecycle = observability.get("lifecycle")
    integrity = (
        lifecycle.get("integrity")
        if isinstance(lifecycle, Mapping) else None
    )
    if (
        not isinstance(integrity, Mapping)
        or integrity.get("body_exception_type")
        != details.get("type")
    ):
        errors.append(f"{context} lifecycle exception differs from terminal failure")
    maps = observability.get("map_execution_records")
    if not isinstance(maps, list) or not maps:
        errors.append(f"{context} map_execution_records are missing")
        return
    expected_maps = _load_bound_plan_map_sequence(
        attempt_root, manifest, provenance, errors
    )
    if expected_maps is None:
        return
    if len(maps) > len(expected_maps):
        errors.append(f"{context} map coverage exceeds frozen work items")
    if len(failed_ig_maps) != 1 or maps[-1] is not failed_ig_maps[0]:
        errors.append(f"{context} requires exactly one final failed IG map")
    class_index = (
        _bound_class_index(provenance, context=context, errors=errors)
        if method_id == _FULL_MODEL_IG_METHOD_ID else None
    )
    for index, item in enumerate(maps):
        expected_variant = (
            expected_maps[index]
            if index < len(expected_maps) else ("<unplanned_map>", None)
        )
        _validate_ig_map_record(
            item, identity=identity, expected_variant=expected_variant,
            failed=index == len(maps) - 1, failure_details=details,
            method_id=method_id, class_index=class_index,
            context=f"{context} map {index}", errors=errors,
        )


@dataclass(frozen=True)
class ArtifactCaps:
    """Exact ordinary-file byte caps gating a complete execution attempt.

    L0/L1 accounting also includes L3, manifest, and status.  L2 files are
    charged separately to the audit L2 cap.  Provenance is an L0 artifact.
    """

    l0_l1_bytes_max: int = DEFAULT_L0_L1_CAP_BYTES
    l2_bytes_max: int = DEFAULT_L2_CAP_BYTES

    def __post_init__(self) -> None:
        _nonnegative_integer(self.l0_l1_bytes_max, "l0_l1_bytes_max")
        _nonnegative_integer(self.l2_bytes_max, "l2_bytes_max")

    def to_dict(self) -> dict[str, int]:
        return {
            "l0_l1_bytes_max": self.l0_l1_bytes_max,
            "l2_bytes_max": self.l2_bytes_max,
        }

    @classmethod
    def parse(
        cls,
        value: ArtifactCaps | Mapping[str, Any] | None = None,
        *,
        l0_l1_cap_bytes: int | None = None,
        l2_cap_bytes: int | None = None,
    ) -> ArtifactCaps:
        if isinstance(value, cls):
            first = value.l0_l1_bytes_max
            second = value.l2_bytes_max
        elif value is None:
            first = DEFAULT_L0_L1_CAP_BYTES
            second = DEFAULT_L2_CAP_BYTES
        elif isinstance(value, Mapping):
            first = value.get(
                "l0_l1_bytes_max",
                value.get(
                    "bytes_per_l0_l1_execution_unit_max",
                    DEFAULT_L0_L1_CAP_BYTES,
                ),
            )
            second = value.get(
                "l2_bytes_max",
                value.get(
                    "bytes_per_audit_sample_l2_max", DEFAULT_L2_CAP_BYTES
                ),
            )
        else:
            raise TypeError("caps must be ArtifactCaps, a mapping, or None")
        if l0_l1_cap_bytes is not None:
            first = l0_l1_cap_bytes
        if l2_cap_bytes is not None:
            second = l2_cap_bytes
        return cls(
            _nonnegative_integer(first, "l0_l1_cap_bytes"),
            _nonnegative_integer(second, "l2_cap_bytes"),
        )


@dataclass(frozen=True)
class ExecutionUnitValidation(Mapping[str, Any]):
    """Read-only result.  Only valid complete attempts are resumable."""

    path: Path
    valid: bool
    status: str | None
    terminal: bool
    complete: bool
    resumable: bool
    unit_identity_sha256: str | None
    manifest_sha256: str | None
    artifact_count: int
    l0_l1_bytes: int
    l2_bytes: int
    total_bytes: int
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "valid": self.valid,
            "status": self.status,
            "terminal": self.terminal,
            "complete": self.complete,
            "resumable": self.resumable,
            "unit_identity_sha256": self.unit_identity_sha256,
            "manifest_sha256": self.manifest_sha256,
            "artifact_count": self.artifact_count,
            "l0_l1_bytes": self.l0_l1_bytes,
            "l2_bytes": self.l2_bytes,
            "total_bytes": self.total_bytes,
            "errors": list(self.errors),
        }

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())

    def require_resumable(self) -> None:
        if not self.resumable:
            detail = "; ".join(self.errors) or f"status is {self.status!r}"
            raise StageCArtifactError(
                f"execution unit is not a valid complete commit: {detail}"
            )


def canonical_json_bytes(value: Any) -> bytes:
    """Canonical compact JSON used for semantic identity digests."""

    return json.dumps(
        _json_compatible(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def unit_identity_sha256(unit_identity: Mapping[str, Any]) -> str:
    identity = _identity_copy(unit_identity)
    # Stage C UnitKey identities carry their own canonical digest.  That
    # digest hashes the seven semantic fields and deliberately excludes the
    # digest field itself.  Recompute it here instead of hashing the envelope,
    # so artifact/status/provenance all bind exactly the same identity.
    if "canonical_digest" in identity:
        expected_fields = {
            "schema_version", "sample_id", "method_id", "config_id",
            "block_index", "head_index", "target_ref", "canonical_digest",
        }
        if set(identity) != expected_fields:
            raise ValueError(
                "UnitKey identity has missing or unknown fields"
            )
        claimed = identity["canonical_digest"]
        if not isinstance(claimed, str) or not _SHA256_RE.fullmatch(claimed):
            raise ValueError(
                "UnitKey canonical_digest must be lowercase SHA-256"
            )
        semantic = {
            key: identity[key]
            for key in (
                "schema_version", "sample_id", "method_id", "config_id",
                "block_index", "head_index", "target_ref",
            )
        }
        observed = hashlib.sha256(canonical_json_bytes(semantic)).hexdigest()
        if claimed != observed:
            raise ValueError(
                "UnitKey canonical_digest does not match semantic fields"
            )
        return claimed
    return hashlib.sha256(canonical_json_bytes(identity)).hexdigest()


def next_attempt_directory(unit_root: str | Path) -> Path:
    """Atomically create the next fresh attempt-NNNNNN directory."""

    root = _absolute_path(unit_root)
    _ensure_directory_no_symlink(root, create=True)
    while True:
        highest = 0
        with os.scandir(root) as entries:
            for entry in entries:
                if entry.is_symlink():
                    raise ArtifactPolicyError(
                        f"attempt root contains forbidden symlink: {entry.path}"
                    )
                match = _ATTEMPT_RE.fullmatch(entry.name)
                if match is None:
                    continue
                if not entry.is_dir(follow_symlinks=False):
                    raise ArtifactPolicyError(
                        f"attempt path is not a directory: {entry.path}"
                    )
                highest = max(highest, int(match.group(1)))
        if highest >= 999999:
            raise StageCArtifactError(f"attempt sequence exhausted under {root}")
        candidate = root / f"attempt-{highest + 1:06d}"
        try:
            candidate.mkdir(mode=0o755, exist_ok=False)
        except FileExistsError:
            continue
        _fsync_directory(root)
        return candidate


def attempt_directories(unit_root: str | Path) -> tuple[Path, ...]:
    """Enumerate numbered attempts without modifying them."""

    root = _absolute_path(unit_root)
    if not root.exists():
        return ()
    _ensure_directory_no_symlink(root, create=False)
    result: list[tuple[int, Path]] = []
    with os.scandir(root) as entries:
        for entry in entries:
            if entry.is_symlink():
                raise ArtifactPolicyError(
                    f"attempt root contains forbidden symlink: {entry.path}"
                )
            match = _ATTEMPT_RE.fullmatch(entry.name)
            if match and entry.is_dir(follow_symlinks=False):
                result.append((int(match.group(1)), Path(entry.path)))
    return tuple(path for _index, path in sorted(result))


class StageCArtifactWriter:
    """Writer for one new immutable Stage C execution attempt.

    Construction immediately publishes a durable running manifest/status.
    Provenance is held in memory until a terminal transition, allowing actual
    resource and scientific status bindings to be supplied after computation.
    """

    def __init__(
        self,
        attempt_dir: str | Path,
        *,
        unit_identity: Mapping[str, Any],
        audit: bool,
        provenance: Mapping[str, Any] | None = None,
        caps: ArtifactCaps | Mapping[str, Any] | None = None,
        l0_l1_cap_bytes: int | None = None,
        l2_cap_bytes: int | None = None,
        fault_injector: Callable[[str], Any] | str | Sequence[str] | None = None,
    ) -> None:
        if type(audit) is not bool:
            raise TypeError("audit must be bool")
        self.attempt_dir = _absolute_path(attempt_dir)
        self.output_dir = self.attempt_dir
        self.manifest_path = self.attempt_dir / MANIFEST_FILENAME
        self.status_path = self.attempt_dir / STATUS_FILENAME
        self.provenance_path = self.attempt_dir / PROVENANCE_PATH
        self.unit_identity = _identity_copy(unit_identity)
        self.unit_identity_sha256 = unit_identity_sha256(self.unit_identity)
        self.audit = audit
        self.caps = ArtifactCaps.parse(
            caps,
            l0_l1_cap_bytes=l0_l1_cap_bytes,
            l2_cap_bytes=l2_cap_bytes,
        )
        self._provenance: dict[str, Any] | None = None
        self._artifacts: list[dict[str, Any]] = []
        self._artifact_paths: set[str] = set()
        self._status = "running"
        self._terminal = False
        self._complete = False
        self._crashed = False
        self._failure: dict[str, Any] | None = None
        self._fault_injector = fault_injector
        self._published_manifest: dict[str, Any] | None = None

        self._claim_empty_attempt()
        if provenance is not None:
            self.set_provenance(provenance)
        manifest = self._make_manifest("running", False, False)
        manifest_payload = _json_file_bytes(manifest)
        status = self._make_status(
            "running", False, False, manifest_payload
        )
        _atomic_create(self.manifest_path, manifest_payload)
        _atomic_create(self.status_path, _json_file_bytes(status))
        self._published_manifest = manifest
        self._maybe_fault(FAULT_AFTER_RUNNING)

    @property
    def status(self) -> str:
        return self._status

    @property
    def terminal(self) -> bool:
        return self._terminal

    @property
    def complete(self) -> bool:
        return self._complete

    @property
    def manifest(self) -> dict[str, Any]:
        return _json_copy(self._published_manifest or {})

    @property
    def artifacts(self) -> tuple[dict[str, Any], ...]:
        return tuple(_json_copy(item) for item in self._artifacts)

    @property
    def actual_bytes(self) -> int:
        try:
            files, _problems = _inventory_regular_files(self.attempt_dir)
        except OSError:
            return 0
        return sum(size for _path, size in files.values())

    def set_provenance(self, provenance: Mapping[str, Any]) -> None:
        self._ensure_running()
        if PROVENANCE_PATH in self._artifact_paths:
            raise ArtifactWriterStateError("provenance is already immutable")
        self._provenance = _normalize_provenance(
            provenance, self.unit_identity, self.unit_identity_sha256
        )

    def write_l0(
        self, name: str, value: Any, *, kind: str = "record"
    ) -> dict[str, Any]:
        try:
            if PurePosixPath(name).name in {"provenance", "provenance.json"}:
                raise ArtifactPolicyError(
                    "L0/provenance.json is reserved for the terminal binding"
                )
            if PurePosixPath(name).name in {"failure", "failure.json"}:
                raise ArtifactPolicyError(
                    "L0/failure.json is reserved for terminal failures"
                )
            parsed_kind = _artifact_kind(kind)
            if parsed_kind in {"provenance", "failure"}:
                raise ArtifactPolicyError(
                    f"artifact kind {parsed_kind!r} is reserved"
                )
            return self._write_payload(
                "L0", name, ".json", _json_file_bytes(value),
                parsed_kind, "json", None, "json"
            )
        except InjectedArtifactCrash:
            raise
        except Exception as exc:
            self._fail_after_error(exc)
            raise

    def write_l1(self, name: str, value: Any) -> dict[str, Any]:
        try:
            array = _float32_array(value, "L1", (14, 14))
            return self._write_payload(
                "L1", name, ".npy", _serialize_npy(array),
                "metric_ready", "float32", [14, 14], "npy"
            )
        except InjectedArtifactCrash:
            raise
        except Exception as exc:
            self._fail_after_error(exc)
            raise

    def write_l2(
        self, name: str, value: Any, *, kind: str = "audit"
    ) -> dict[str, Any]:
        """Write heterogeneous float32 audit data as NPY or NPZ."""

        try:
            self._ensure_running()
            if not self.audit:
                raise ArtifactPolicyError("L2 artifacts are restricted to audit units")
            parsed_kind = _artifact_kind(kind)
            if isinstance(value, Mapping):
                if not value:
                    raise ArtifactPolicyError("L2 array mapping must not be empty")
                arrays: dict[str, np.ndarray] = {}
                for key, item in value.items():
                    if (
                        not isinstance(key, str)
                        or not _SAFE_COMPONENT_RE.fullmatch(key)
                    ):
                        raise ArtifactPolicyError(
                            "L2 NPZ keys must be safe non-empty identifiers"
                        )
                    arrays[key] = _float32_array(item, "L2")
                arrays = dict(sorted(arrays.items()))
                payload = _serialize_npz(arrays)
                dtype: str | dict[str, str] = {
                    key: "float32" for key in arrays
                }
                shape: list[int] | dict[str, list[int]] = {
                    key: list(array.shape) for key, array in arrays.items()
                }
                suffix = ".npz"
                artifact_format = "npz"
            else:
                array = _float32_array(value, "L2")
                payload = _serialize_npy(array)
                dtype = "float32"
                shape = list(array.shape)
                suffix = ".npy"
                artifact_format = "npy"
            return self._write_payload(
                "L2", name, suffix, payload, parsed_kind, dtype, shape,
                artifact_format
            )
        except InjectedArtifactCrash:
            raise
        except Exception as exc:
            self._fail_after_error(exc)
            raise

    def write_l3(
        self,
        name: str,
        png_bytes: bytes | bytearray | memoryview,
        *,
        failure: bool = False,
    ) -> dict[str, Any]:
        """Write a PNG marked permanently as display-only and non-metric."""

        try:
            self._ensure_running()
            if type(failure) is not bool:
                raise TypeError("failure must be bool")
            if not self.audit and not failure:
                raise ArtifactPolicyError(
                    "L3 display PNG requires an audit unit or failure=True"
                )
            if not isinstance(png_bytes, (bytes, bytearray, memoryview)):
                raise ArtifactPolicyError("L3 artifact must be PNG bytes")
            payload = bytes(png_bytes)
            _verify_png(payload)
            return self._write_payload(
                "L3",
                name,
                ".png",
                payload,
                "display",
                "bytes",
                None,
                "png",
                extra={
                    "display_only": True,
                    "metric_input": False,
                    "failure": failure,
                },
            )
        except InjectedArtifactCrash:
            raise
        except Exception as exc:
            self._fail_after_error(exc)
            raise

    def finalize(
        self, *, provenance: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        """Validate and atomically publish the sole resumable state."""

        self._ensure_running()
        try:
            if provenance is not None:
                self.set_provenance(provenance)
            if self._provenance is None:
                raise ArtifactWriterStateError(
                    "final provenance must be supplied before completion"
                )
            if not any(
                item["level"] == "L0" and item["kind"] != "provenance"
                for item in self._artifacts
            ):
                raise ArtifactPolicyError(
                    "complete unit requires at least one L0 record"
                )
            if not any(item["level"] == "L1" for item in self._artifacts):
                raise ArtifactPolicyError(
                    "complete unit requires at least one L1 map"
                )
            if not self.audit and any(
                item["level"] == "L3" for item in self._artifacts
            ):
                raise ArtifactPolicyError(
                    "a non-audit success cannot retain failure-display PNGs"
                )
            self._publish_provenance_if_needed()
            manifest = self._make_manifest("complete", True, True)
            manifest_payload = _json_file_bytes(manifest)
            status = self._make_status(
                "complete", True, True, manifest_payload
            )
            status_payload = _json_file_bytes(status)
            self._validate_pending_tree()
            self._enforce_complete_caps(manifest_payload, status_payload)
            _atomic_replace(self.manifest_path, manifest_payload)
            self._published_manifest = manifest
            self._maybe_fault(FAULT_AFTER_MANIFEST)
            self._maybe_fault(FAULT_BEFORE_COMPLETE_STATUS)
            _atomic_replace(self.status_path, status_payload)
            self._status = "complete"
            self._terminal = True
            self._complete = True
            return _json_copy(manifest)
        except InjectedArtifactCrash:
            raise
        except Exception:
            # A completion failure cannot safely synthesize provenance for a
            # different terminal state: the caller's completed provenance is
            # already frozen and may already be on disk.  Leave the attempt
            # non-terminal/non-resumable; the runner records the failure in a
            # fresh attempt or its batch failure table.
            raise

    def publish_runner_telemetry(
        self, telemetry: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Atomically bind post-finalize timing and lifecycle evidence.

        The synchronized total deliberately stops after the first terminal
        validation.  This final status replacement is excluded because a
        status document cannot recursively include the time required to write
        itself.  Batch controls bind the resulting status bytes by SHA-256.
        """

        if not self._terminal:
            raise ArtifactWriterStateError(
                "runner telemetry requires a terminal attempt"
            )
        value = _json_copy(telemetry)
        errors: list[str] = []
        _validate_runner_telemetry(value, errors)
        if errors:
            raise ArtifactPolicyError(
                "invalid runner telemetry: " + "; ".join(errors)
            )
        status = _strict_json_loads(self.status_path.read_bytes())
        if "runner_telemetry" in status:
            raise ArtifactWriterStateError(
                "runner telemetry is already immutable"
            )
        telemetry_payload = canonical_json_bytes(value)
        manifest = _strict_json_loads(self.manifest_path.read_bytes())
        if "runner_telemetry_sha256" in manifest:
            raise ArtifactWriterStateError(
                "runner telemetry digest is already immutable"
            )
        manifest["runner_telemetry_sha256"] = hashlib.sha256(
            telemetry_payload
        ).hexdigest()
        manifest_payload = _json_file_bytes(manifest)
        status["manifest_sha256"] = hashlib.sha256(manifest_payload).hexdigest()
        status["runner_telemetry"] = value
        status_payload = _json_file_bytes(status)
        # The terminal telemetry is an L0 control and remains inside the same
        # per-unit cap even though its own publication is outside timed total.
        artifact_size = sum(int(item["bytes"]) for item in self._artifacts)
        l2_size = sum(
            int(item["bytes"]) for item in self._artifacts
            if item["level"] == "L2"
        )
        if len(manifest_payload) + len(status_payload) + artifact_size - l2_size > (
            self.caps.l0_l1_bytes_max
        ):
            raise ArtifactBudgetError(
                "terminal runner telemetry would exceed the L0/L1 cap"
            )
        _atomic_replace(self.manifest_path, manifest_payload)
        self._published_manifest = manifest
        _atomic_replace(self.status_path, status_payload)
        return _json_copy(status)

    def fail(
        self,
        reason: str,
        details: Any = None,
        *,
        provenance: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Publish an explicit terminal failure with complete=false."""

        if self._status == "failed" and self._terminal:
            return self.manifest
        self._ensure_running()
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("failure reason must be a non-empty string")
        self.prepare_failure(reason.strip(), details)
        if provenance is not None:
            self.set_provenance(provenance)
        return self._publish_failure(reason.strip(), details)

    def prepare_failure(
        self, reason: str, details: Any = None
    ) -> dict[str, Any]:
        """Persist the failure L0 record before final provenance is built.

        A runner that binds an exact inventory into provenance calls this
        first, builds provenance from writer.artifacts, and then calls fail.
        Repeating it is allowed only for an exactly identical failure.
        """

        self._ensure_running()
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("failure reason must be a non-empty string")
        failure_value: dict[str, Any] = {"reason": reason.strip()}
        if details is not None:
            failure_value["details"] = _json_compatible(details)
        failure_value = _json_copy(failure_value)
        if self._failure is not None:
            if self._failure != failure_value:
                raise ArtifactWriterStateError(
                    "attempt already has a different prepared failure"
                )
            record = next(
                item
                for item in self._artifacts
                if item["path"] == "L0/failure.json"
            )
            return _json_copy(record)
        record = self._write_payload(
            "L0",
            "failure",
            ".json",
            _json_file_bytes(failure_value),
            "failure",
            "json",
            None,
            "json",
            inject_fault=False,
            enforce_budget=False,
        )
        self._failure = failure_value
        return record

    # Explicit aliases keep caller code self-documenting without widening the
    # accepted formats.
    write_l0_json = write_l0
    write_l1_map = write_l1
    write_l2_array = write_l2
    write_l3_png = write_l3

    def _claim_empty_attempt(self) -> None:
        path = self.attempt_dir
        if path.exists():
            _ensure_directory_no_symlink(path, create=False)
            if next(path.iterdir(), None) is not None:
                raise AttemptNotEmptyError(
                    f"refusing to overwrite or append attempt: {path}"
                )
        else:
            _ensure_directory_no_symlink(path.parent, create=True)
            try:
                path.mkdir(exist_ok=False)
            except FileExistsError as exc:
                raise AttemptNotEmptyError(
                    f"attempt directory was claimed concurrently: {path}"
                ) from exc
            _fsync_directory(path.parent)
        _ensure_directory_no_symlink(path, create=False)

    def _ensure_running(self) -> None:
        if self._crashed:
            raise ArtifactWriterStateError(
                "fault-injected attempt is immutable; allocate a fresh attempt"
            )
        if self._status != "running" or self._terminal:
            raise ArtifactWriterStateError(
                f"attempt is already terminal with status {self._status!r}"
            )

    def _write_payload(
        self,
        level: str,
        name: str,
        suffix: str,
        payload: bytes,
        kind: str,
        dtype: Any,
        shape: Any,
        artifact_format: str,
        *,
        extra: Mapping[str, Any] | None = None,
        inject_fault: bool = True,
        enforce_budget: bool = True,
    ) -> dict[str, Any]:
        self._ensure_running()
        relative = _artifact_relative_path(level, name, suffix)
        if relative in self._artifact_paths:
            raise FileExistsError(f"artifact was already written: {relative}")
        destination = self.attempt_dir.joinpath(*PurePosixPath(relative).parts)
        _ensure_artifact_parent(self.attempt_dir, destination.parent)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"refusing to overwrite artifact: {destination}")
        if enforce_budget:
            self._preflight_payload(level, len(payload))
        _atomic_create(destination, payload)
        info = destination.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_size != len(payload):
            raise OSError(f"artifact publication failed for {relative}")
        record = {
            "level": level,
            "path": relative,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
            "kind": kind,
            "dtype": _json_copy(dtype),
            "shape": _json_copy(shape),
            "format": artifact_format,
        }
        if extra:
            record.update(_json_copy(extra))
        self._artifacts.append(record)
        self._artifact_paths.add(relative)
        if inject_fault:
            self._maybe_fault(FAULT_AFTER_PAYLOAD)
        return _json_copy(record)

    def _preflight_payload(self, level: str, byte_count: int) -> None:
        if level == "L2":
            used = sum(
                int(item["bytes"])
                for item in self._artifacts
                if item["level"] == "L2"
            )
            if used + byte_count > self.caps.l2_bytes_max:
                raise ArtifactBudgetError(
                    f"L2 bytes {used + byte_count} exceed cap "
                    f"{self.caps.l2_bytes_max}"
                )
            return
        used = self.status_path.stat().st_size + self.manifest_path.stat().st_size
        used += sum(
            int(item["bytes"])
            for item in self._artifacts
            if item["level"] != "L2"
        )
        if used + byte_count > self.caps.l0_l1_bytes_max:
            raise ArtifactBudgetError(
                f"L0/L1/L3/control bytes {used + byte_count} exceed cap "
                f"{self.caps.l0_l1_bytes_max}"
            )

    def _publish_provenance_if_needed(
        self, *, enforce_budget: bool = True
    ) -> None:
        if PROVENANCE_PATH in self._artifact_paths:
            return
        if self._provenance is None:
            self._provenance = _normalize_provenance(
                {}, self.unit_identity, self.unit_identity_sha256
            )
        self._write_payload(
            "L0",
            "provenance",
            ".json",
            _json_file_bytes(self._provenance),
            "provenance",
            "json",
            None,
            "json",
            enforce_budget=enforce_budget,
        )

    def _publish_failure(self, reason: str, details: Any) -> dict[str, Any]:
        self._ensure_running()
        failure_value: dict[str, Any] = {"reason": reason}
        if details is not None:
            failure_value["details"] = _json_compatible(details)
        failure_value = _json_copy(failure_value)
        if self._failure is None:
            self.prepare_failure(reason, details)
        elif self._failure != failure_value:
            raise ArtifactWriterStateError(
                "terminal failure does not match prepared failure"
            )
        self._publish_provenance_if_needed(enforce_budget=False)
        manifest = self._make_manifest("failed", True, False)
        manifest_payload = _json_file_bytes(manifest)
        status = self._make_status("failed", True, False, manifest_payload)
        status_payload = _json_file_bytes(status)
        self._validate_pending_tree()
        # A cap violation is itself failure evidence.  Completion is gated on
        # the cap, but the small terminal failure controls must still be
        # publishable so an over-budget attempt never remains ambiguously
        # running.
        _atomic_replace(self.manifest_path, manifest_payload)
        self._published_manifest = manifest
        self._maybe_fault(FAULT_AFTER_MANIFEST)
        _atomic_replace(self.status_path, status_payload)
        self._status = "failed"
        self._terminal = True
        self._complete = False
        return _json_copy(manifest)

    def _fail_after_error(self, exc: Exception) -> None:
        if self._status != "running" or self._crashed:
            return
        try:
            self._publish_failure(
                f"{type(exc).__name__}: {exc}",
                {"exception_type": type(exc).__name__},
            )
        except Exception:
            # Preserve the original exception.  Initial controls still prove
            # that this attempt was never complete.
            return

    def _make_manifest(
        self, status: str, terminal: bool, complete: bool
    ) -> dict[str, Any]:
        provenance_record = next(
            (
                item
                for item in self._artifacts
                if item["path"] == PROVENANCE_PATH
            ),
            None,
        )
        value: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "document_type": MANIFEST_DOCUMENT_TYPE,
            "status": status,
            "terminal": terminal,
            "complete": complete,
            "completion_authority": STATUS_FILENAME,
            "unit_identity": _json_copy(self.unit_identity),
            "unit_identity_sha256": self.unit_identity_sha256,
            "provenance_path": PROVENANCE_PATH,
            "provenance_sha256": (
                provenance_record["sha256"] if provenance_record else None
            ),
            "audit": self.audit,
            "retention": {
                "l1_storage_dtype": "float32",
                "l1_shape": [14, 14],
                "l2_audit_only": True,
                "l3_display_only": True,
                "display_is_never_metric_input": True,
                "forbidden_granularities": [
                    "per_step",
                    "per_pixel",
                    "per_dimension",
                ],
            },
            "caps": self.caps.to_dict(),
            "artifacts": _json_copy(self._artifacts),
        }
        if self._failure is not None:
            value["failure"] = _json_copy(self._failure)
        return value

    def _make_status(
        self,
        status: str,
        terminal: bool,
        complete: bool,
        manifest_payload: bytes,
    ) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "document_type": STATUS_DOCUMENT_TYPE,
            "status": status,
            "terminal": terminal,
            "complete": complete,
            "unit_identity_sha256": self.unit_identity_sha256,
            "manifest_sha256": hashlib.sha256(manifest_payload).hexdigest(),
        }
        if self._failure is not None:
            value["failure"] = _json_copy(self._failure)
        return value

    def _validate_pending_tree(self) -> None:
        files, problems = _inventory_regular_files(self.attempt_dir)
        if problems:
            raise ArtifactPolicyError("; ".join(problems))
        expected = {MANIFEST_FILENAME, STATUS_FILENAME, *self._artifact_paths}
        actual = set(files)
        if actual != expected:
            raise ArtifactPolicyError(
                "attempt file set mismatch; "
                f"missing={sorted(expected - actual)}, "
                f"extra={sorted(actual - expected)}"
            )
        records = {item["path"]: item for item in self._artifacts}
        for relative, record in records.items():
            path, size = files[relative]
            raw = path.read_bytes()
            if size != record["bytes"] or len(raw) != record["bytes"]:
                raise ArtifactPolicyError(f"artifact byte mismatch: {relative}")
            if hashlib.sha256(raw).hexdigest() != record["sha256"]:
                raise ArtifactPolicyError(f"artifact hash mismatch: {relative}")

    def _enforce_complete_caps(
        self, manifest_payload: bytes, status_payload: bytes
    ) -> None:
        l2_bytes = sum(
            int(item["bytes"])
            for item in self._artifacts
            if item["level"] == "L2"
        )
        l0_l1_bytes = len(manifest_payload) + len(status_payload) + sum(
            int(item["bytes"])
            for item in self._artifacts
            if item["level"] != "L2"
        )
        if l0_l1_bytes > self.caps.l0_l1_bytes_max:
            raise ArtifactBudgetError(
                f"complete L0/L1/L3/control bytes {l0_l1_bytes} exceed cap "
                f"{self.caps.l0_l1_bytes_max}"
            )
        if l2_bytes > self.caps.l2_bytes_max:
            raise ArtifactBudgetError(
                f"complete L2 bytes {l2_bytes} exceed cap "
                f"{self.caps.l2_bytes_max}"
            )

    def _maybe_fault(self, point: str) -> None:
        injector = self._fault_injector
        if injector is None:
            return
        triggered = False
        try:
            if callable(injector):
                triggered = bool(injector(point))
            elif isinstance(injector, str):
                if injector not in FAULT_POINTS:
                    raise ValueError(f"unknown fault point: {injector!r}")
                triggered = injector == point
            else:
                configured = set(injector)
                unknown = configured - FAULT_POINTS
                if unknown:
                    raise ValueError(f"unknown fault points: {sorted(unknown)}")
                triggered = point in configured
        except BaseException:
            self._crashed = True
            raise
        if triggered:
            self._crashed = True
            raise InjectedArtifactCrash(point)


def validate_execution_unit(
    attempt_dir: str | Path,
    *,
    expected_unit_identity: Mapping[str, Any] | None = None,
    expected_unit_digest: str | None = None,
    expected_unit_identity_sha256: str | None = None,
    expected_provenance_bindings: Mapping[str, Any] | None = None,
    observed_git_state: Mapping[str, Any] | None = None,
    caps: ArtifactCaps | Mapping[str, Any] | None = None,
    l0_l1_cap_bytes: int | None = None,
    l2_cap_bytes: int | None = None,
    require_runner_telemetry: bool = False,
) -> ExecutionUnitValidation:
    """Validate an attempt without changing any filesystem state."""

    if type(require_runner_telemetry) is not bool:
        raise TypeError("require_runner_telemetry must be bool")

    root = _absolute_path(attempt_dir)
    errors: list[str] = []
    status_value: dict[str, Any] | None = None
    manifest: dict[str, Any] | None = None
    manifest_digest: str | None = None
    identity_digest: str | None = None
    artifact_count = 0
    files: dict[str, tuple[Path, int]] = {}
    if expected_unit_identity_sha256 is not None:
        if (
            expected_unit_digest is not None
            and expected_unit_digest != expected_unit_identity_sha256
        ):
            errors.append("conflicting expected unit digests")
        else:
            expected_unit_digest = expected_unit_identity_sha256

    try:
        if not root.exists():
            errors.append(f"attempt directory is missing: {root}")
        else:
            try:
                _assert_no_symlink_components(root)
            except (OSError, ArtifactPolicyError) as exc:
                errors.append(str(exc))
            info = root.lstat()
            if stat.S_ISLNK(info.st_mode):
                errors.append(f"attempt directory is a forbidden symlink: {root}")
            elif not stat.S_ISDIR(info.st_mode):
                errors.append(f"attempt path is not a directory: {root}")
            else:
                files, inventory_errors = _inventory_regular_files(root)
                errors.extend(inventory_errors)
    except OSError as exc:
        errors.append(f"cannot inspect attempt directory: {exc}")

    manifest_raw = _read_control(
        files, MANIFEST_FILENAME, "manifest", errors
    )
    status_raw = _read_control(files, STATUS_FILENAME, "status", errors)
    if manifest_raw is not None:
        manifest_digest = hashlib.sha256(manifest_raw).hexdigest()
        manifest = _load_canonical_json(manifest_raw, "manifest", errors)
    if status_raw is not None:
        status_value = _load_canonical_json(status_raw, "status", errors)

    if manifest is not None:
        _validate_manifest_header(manifest, errors)
        identity = manifest.get("unit_identity")
        try:
            identity_digest = unit_identity_sha256(identity)
        except Exception as exc:
            errors.append(f"manifest unit identity is invalid: {exc}")
        else:
            if manifest.get("unit_identity_sha256") != identity_digest:
                errors.append("manifest unit identity digest mismatch")
        if expected_unit_identity is not None:
            try:
                expected_identity = _identity_copy(expected_unit_identity)
            except Exception as exc:
                errors.append(f"expected unit identity is invalid: {exc}")
            else:
                if identity != expected_identity:
                    errors.append(
                        "execution unit identity does not match expected identity"
                    )
        if expected_unit_digest is not None:
            if (
                not isinstance(expected_unit_digest, str)
                or not _SHA256_RE.fullmatch(expected_unit_digest)
            ):
                errors.append("expected unit digest is not lowercase SHA-256")
            elif identity_digest != expected_unit_digest:
                errors.append(
                    "execution unit digest does not match expected digest"
                )

    if status_value is not None:
        _validate_status_header(
            status_value, errors,
            require_runner_telemetry=require_runner_telemetry,
        )
        if (
            manifest_digest is not None
            and status_value.get("manifest_sha256") != manifest_digest
        ):
            errors.append("status manifest_sha256 does not match manifest bytes")
        if manifest is not None:
            for field in (
                "status", "terminal", "complete", "unit_identity_sha256"
            ):
                if status_value.get(field) != manifest.get(field):
                    errors.append(f"status/manifest {field} mismatch")
            if status_value.get("failure") != manifest.get("failure"):
                errors.append("status/manifest failure mismatch")
            telemetry = status_value.get("runner_telemetry")
            telemetry_digest = manifest.get("runner_telemetry_sha256")
            if (telemetry is None) != (telemetry_digest is None):
                errors.append("status/manifest runner telemetry binding mismatch")
            elif telemetry is not None:
                try:
                    observed_telemetry_digest = hashlib.sha256(
                        canonical_json_bytes(telemetry)
                    ).hexdigest()
                except Exception as exc:
                    errors.append(f"runner telemetry cannot be hashed: {exc}")
                else:
                    if observed_telemetry_digest != telemetry_digest:
                        errors.append(
                            "manifest runner_telemetry_sha256 does not match status"
                        )

    expected_files = {MANIFEST_FILENAME, STATUS_FILENAME}
    records: list[Any] = []
    if manifest is not None:
        candidate_records = manifest.get("artifacts")
        if isinstance(candidate_records, list):
            records = candidate_records
            artifact_count = len(records)
        else:
            errors.append("manifest artifacts must be a list")
    seen_paths: set[str] = set()
    provenance: dict[str, Any] | None = None
    failure_observability: dict[str, Any] | None = None
    for index, candidate in enumerate(records):
        context = f"artifact[{index}]"
        if not isinstance(candidate, dict):
            errors.append(f"{context} must be an object")
            continue
        relative = candidate.get("path")
        if not _valid_inventory_path(relative, candidate.get("level")):
            errors.append(f"{context} has unsafe or level-mismatched path")
            continue
        if relative in seen_paths:
            errors.append(f"duplicate artifact inventory path: {relative}")
            continue
        seen_paths.add(relative)
        expected_files.add(relative)
        file_item = files.get(relative)
        if file_item is None:
            errors.append(f"missing artifact file: {relative}")
            continue
        path, observed_size = file_item
        if candidate.get("level") == "L3":
            try:
                observed_digest, raw, streamed_size = _validate_png_path(path)
            except (OSError, ArtifactPolicyError) as exc:
                errors.append(
                    f"{context} payload contract violation: {exc}"
                )
                continue
        else:
            try:
                raw = path.read_bytes()
            except OSError as exc:
                errors.append(f"cannot read artifact {relative}: {exc}")
                continue
            observed_digest = hashlib.sha256(raw).hexdigest()
        byte_count = candidate.get("bytes")
        if type(byte_count) is not int or byte_count < 0:
            errors.append(f"{context} bytes must be a non-negative integer")
        elif byte_count != observed_size:
            errors.append(f"artifact byte mismatch: {relative}")
        elif candidate.get("level") == "L3" and streamed_size != observed_size:
            errors.append(f"artifact changed while validating: {relative}")
        elif candidate.get("level") != "L3" and observed_size != len(raw):
            errors.append(f"short artifact read: {relative}")
        digest = candidate.get("sha256")
        if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
            errors.append(f"{context} sha256 is invalid")
        elif observed_digest != digest:
            errors.append(f"artifact hash mismatch: {relative}")
        _validate_artifact_payload(
            candidate, raw, context, errors,
            png_structure_already_validated=(candidate.get("level") == "L3"),
        )
        if relative == PROVENANCE_PATH:
            provenance = _load_canonical_json(raw, "provenance", errors)
        elif relative == _FAILURE_OBSERVABILITY_PATH:
            failure_observability = _load_canonical_json(
                raw, "failure observability", errors
            )

    actual_files = set(files)
    missing_files = sorted(expected_files - actual_files)
    extra_files = sorted(actual_files - expected_files)
    if missing_files:
        errors.append(f"missing files from exact inventory: {missing_files}")
    if extra_files:
        errors.append(f"extra files outside exact inventory: {extra_files}")

    if manifest is not None:
        provenance_record = next(
            (
                item
                for item in records
                if isinstance(item, dict)
                and item.get("path") == PROVENANCE_PATH
            ),
            None,
        )
        state = manifest.get("status")
        if state in {"complete", "failed"}:
            if provenance_record is None:
                errors.append(
                    "terminal manifest is missing L0 provenance inventory"
                )
            elif (
                manifest.get("provenance_sha256")
                != provenance_record.get("sha256")
            ):
                errors.append("manifest provenance_sha256 mismatch")
        elif manifest.get("provenance_sha256") is not None:
            errors.append("running manifest cannot claim a provenance hash")
        if provenance is not None:
            _validate_provenance(
                provenance,
                manifest,
                expected_provenance_bindings,
                observed_git_state,
                errors,
            )
        elif expected_provenance_bindings is not None:
            errors.append(
                "expected provenance bindings but provenance is missing"
            )

        state = manifest.get("status")
        mapping_records = [item for item in records if isinstance(item, dict)]
        l0_records = [
            item for item in mapping_records if item.get("level") == "L0"
        ]
        l1_records = [
            item for item in mapping_records if item.get("level") == "L1"
        ]
        l2_records = [
            item for item in mapping_records if item.get("level") == "L2"
        ]
        l3_records = [
            item for item in mapping_records if item.get("level") == "L3"
        ]
        if not manifest.get("audit") and l2_records:
            errors.append("non-audit manifest contains L2 artifacts")
        if not manifest.get("audit") and any(
            item.get("failure") is not True for item in l3_records
        ):
            errors.append(
                "non-audit L3 artifact is not explicitly labelled failure"
            )
        provenance_records = [
            item
            for item in l0_records
            if item.get("path") == PROVENANCE_PATH
        ]
        if provenance_records and any(
            item.get("kind") != "provenance"
            for item in provenance_records
        ):
            errors.append("L0 provenance path has the wrong kind")
        if any(
            item.get("kind") == "provenance"
            and item.get("path") != PROVENANCE_PATH
            for item in mapping_records
        ):
            errors.append("provenance kind is used outside its reserved path")
        failure_records = [
            item
            for item in l0_records
            if item.get("path") == "L0/failure.json"
        ]
        if failure_records and any(
            item.get("kind") != "failure" for item in failure_records
        ):
            errors.append("L0 failure path has the wrong kind")
        if any(
            item.get("kind") == "failure"
            and item.get("path") != "L0/failure.json"
            for item in mapping_records
        ):
            errors.append("failure kind is used outside its reserved path")
        observability_records = [
            item
            for item in l0_records
            if item.get("path") == _FAILURE_OBSERVABILITY_PATH
        ]
        if observability_records and any(
            item.get("kind") != "failure_evidence"
            for item in observability_records
        ):
            errors.append(
                "L0 execution lifecycle path has the wrong kind"
            )
        if any(
            item.get("kind") == "failure_evidence"
            and item.get("path") != _FAILURE_OBSERVABILITY_PATH
            for item in mapping_records
        ):
            errors.append(
                "failure_evidence kind is used outside its reserved path"
            )
        if state == "complete":
            if not any(
                item.get("kind") not in {"provenance", "failure"}
                for item in l0_records
            ):
                errors.append("complete unit has no L0 result record")
            if not l1_records:
                errors.append("complete unit has no L1 metric map")
            if failure_records:
                errors.append("complete unit contains a terminal failure record")
            if not manifest.get("audit") and l3_records:
                errors.append("non-audit complete unit contains L3 artifacts")
        elif state == "failed":
            if len(failure_records) != 1:
                errors.append(
                    "failed unit must contain exactly one L0 failure record"
                )
            elif files.get("L0/failure.json") is not None:
                try:
                    failure_document = _strict_json_loads(
                        files["L0/failure.json"][0].read_bytes()
                    )
                except Exception as exc:
                    errors.append(f"failure record is invalid JSON: {exc}")
                else:
                    if failure_document != manifest.get("failure"):
                        errors.append(
                            "L0 failure record does not match terminal failure"
                        )
        elif state == "running" and mapping_records:
            errors.append("running manifest cannot inventory payload artifacts")

        _validate_ig_convergence_failure_observability(
            root, manifest, status_value, provenance,
            failure_observability, errors,
        )

    manifest_caps: ArtifactCaps | None = None
    if manifest is not None:
        try:
            manifest_caps = ArtifactCaps.parse(manifest.get("caps"))
            if manifest.get("caps") != manifest_caps.to_dict():
                errors.append("manifest caps use a non-canonical shape")
        except Exception as exc:
            errors.append(f"manifest caps are invalid: {exc}")
    explicitly_requested = (
        caps is not None
        or l0_l1_cap_bytes is not None
        or l2_cap_bytes is not None
    )
    try:
        requested_caps = (
            ArtifactCaps.parse(
                caps,
                l0_l1_cap_bytes=l0_l1_cap_bytes,
                l2_cap_bytes=l2_cap_bytes,
            )
            if explicitly_requested
            else manifest_caps
        )
    except Exception as exc:
        errors.append(f"expected caps are invalid: {exc}")
        requested_caps = manifest_caps
    if (
        explicitly_requested
        and requested_caps is not None
        and manifest_caps is not None
        and requested_caps != manifest_caps
    ):
        errors.append("manifest caps do not match expected caps")

    l2_bytes = sum(
        size
        for relative, (_path, size) in files.items()
        if relative.startswith("L2/")
    )
    l0_l1_bytes = sum(
        size
        for relative, (_path, size) in files.items()
        if not relative.startswith("L2/")
    )
    total_bytes = l0_l1_bytes + l2_bytes
    # The hard byte caps gate complete results.  A bounded failure record must
    # remain publishable even when the reason for failure is budget exhaustion;
    # it is terminal evidence but can never become resumable.
    manifest_state = manifest.get("status") if manifest is not None else None
    if requested_caps is not None and manifest_state == "complete":
        if l0_l1_bytes > requested_caps.l0_l1_bytes_max:
            errors.append(
                f"L0/L1/L3/control bytes {l0_l1_bytes} exceed cap "
                f"{requested_caps.l0_l1_bytes_max}"
            )
        if l2_bytes > requested_caps.l2_bytes_max:
            errors.append(
                f"L2 bytes {l2_bytes} exceed cap "
                f"{requested_caps.l2_bytes_max}"
            )

    state = status_value.get("status") if status_value else None
    terminal = (
        bool(status_value.get("terminal")) if status_value else False
    )
    complete = (
        bool(status_value.get("complete")) if status_value else False
    )
    valid = not errors
    resumable = valid and state == "complete" and terminal and complete
    return ExecutionUnitValidation(
        path=root,
        valid=valid,
        status=state if isinstance(state, str) else None,
        terminal=terminal,
        complete=complete,
        resumable=resumable,
        unit_identity_sha256=identity_digest,
        manifest_sha256=manifest_digest,
        artifact_count=artifact_count,
        l0_l1_bytes=l0_l1_bytes,
        l2_bytes=l2_bytes,
        total_bytes=total_bytes,
        errors=tuple(errors),
    )


def _validate_manifest_header(value: dict[str, Any], errors: list[str]) -> None:
    required = {
        "schema_version", "document_type", "status", "terminal", "complete",
        "completion_authority", "unit_identity", "unit_identity_sha256",
        "provenance_path", "provenance_sha256", "audit", "retention",
        "caps", "artifacts",
    }
    allowed = required | {"failure", "runner_telemetry_sha256"}
    if not required <= set(value) or set(value) - allowed:
        errors.append("manifest has missing or unknown fields")
    if value.get("schema_version") != SCHEMA_VERSION:
        errors.append("manifest schema_version is invalid")
    if value.get("document_type") != MANIFEST_DOCUMENT_TYPE:
        errors.append("manifest document_type is invalid")
    state = value.get("status")
    expected = {
        "running": (False, False),
        "failed": (True, False),
        "complete": (True, True),
    }.get(state)
    if expected is None:
        errors.append("manifest status is invalid")
    elif (value.get("terminal"), value.get("complete")) != expected:
        errors.append("manifest terminal/complete flags are inconsistent")
    if (
        type(value.get("terminal")) is not bool
        or type(value.get("complete")) is not bool
    ):
        errors.append("manifest terminal/complete flags must be bool")
    if type(value.get("audit")) is not bool:
        errors.append("manifest audit must be bool")
    if value.get("completion_authority") != STATUS_FILENAME:
        errors.append("manifest completion authority is invalid")
    if value.get("provenance_path") != PROVENANCE_PATH:
        errors.append("manifest provenance path is invalid")
    expected_retention = {
        "l1_storage_dtype": "float32",
        "l1_shape": [14, 14],
        "l2_audit_only": True,
        "l3_display_only": True,
        "display_is_never_metric_input": True,
        "forbidden_granularities": [
            "per_step", "per_pixel", "per_dimension"
        ],
    }
    if value.get("retention") != expected_retention:
        errors.append("manifest retention contract is invalid")
    if state == "failed" and not isinstance(value.get("failure"), dict):
        errors.append("failed manifest must contain a failure object")
    if state != "failed" and "failure" in value:
        errors.append("non-failed manifest cannot contain failure")
    telemetry_digest = value.get("runner_telemetry_sha256")
    if telemetry_digest is not None and (
        not isinstance(telemetry_digest, str)
        or not _SHA256_RE.fullmatch(telemetry_digest)
    ):
        errors.append("manifest runner_telemetry_sha256 is invalid")


def _validate_status_header(
    value: dict[str, Any], errors: list[str], *,
    require_runner_telemetry: bool = False,
) -> None:
    required = {
        "schema_version", "document_type", "status", "terminal", "complete",
        "unit_identity_sha256", "manifest_sha256",
    }
    allowed = required | {"failure", "runner_telemetry"}
    if not required <= set(value) or set(value) - allowed:
        errors.append("status has missing or unknown fields")
    if value.get("schema_version") != SCHEMA_VERSION:
        errors.append("status schema_version is invalid")
    if value.get("document_type") != STATUS_DOCUMENT_TYPE:
        errors.append("status document_type is invalid")
    state = value.get("status")
    expected = {
        "running": (False, False),
        "failed": (True, False),
        "complete": (True, True),
    }.get(state)
    if expected is None:
        errors.append("status state is invalid")
    elif (value.get("terminal"), value.get("complete")) != expected:
        errors.append("status terminal/complete flags are inconsistent")
    if (
        type(value.get("terminal")) is not bool
        or type(value.get("complete")) is not bool
    ):
        errors.append("status terminal/complete flags must be bool")
    for field in ("unit_identity_sha256", "manifest_sha256"):
        if (
            not isinstance(value.get(field), str)
            or not _SHA256_RE.fullmatch(value[field])
        ):
            errors.append(f"status {field} is invalid")
    if state == "failed" and not isinstance(value.get("failure"), dict):
        errors.append("failed status must contain failure")
    if state != "failed" and "failure" in value:
        errors.append("non-failed status cannot contain failure")
    telemetry = value.get("runner_telemetry")
    if state in {"complete", "failed"}:
        if telemetry is None and require_runner_telemetry:
            errors.append("terminal status is missing runner_telemetry")
        elif telemetry is not None:
            _validate_runner_telemetry(telemetry, errors)
    elif telemetry is not None:
        errors.append("running status cannot contain runner_telemetry")


def _validate_runner_telemetry(value: Any, errors: list[str]) -> None:
    context = "status runner_telemetry"
    if not isinstance(value, dict):
        errors.append(f"{context} must be an object")
        return
    required = {
        "schema_version", "phase_timings", "lifecycle",
        "measurement_environment",
    }
    if set(value) != required:
        errors.append(f"{context} has missing or unknown fields")
        return
    if value.get("schema_version") != RUNNER_TELEMETRY_SCHEMA_VERSION:
        errors.append(f"{context} schema_version is invalid")

    timings = value.get("phase_timings")
    timing_fields = {
        "clock", "synchronization", "phases",
        "synchronized_total_wall_time_seconds", "total_boundary",
        "phases_are_mutually_exclusive",
    }
    if not isinstance(timings, dict) or set(timings) != timing_fields:
        errors.append(f"{context}.phase_timings has missing or unknown fields")
    else:
        if timings.get("clock") != "perf_counter":
            errors.append(f"{context}.phase_timings clock is invalid")
        if timings.get("synchronization") != (
            "device_before_and_after_each_phase"
        ):
            errors.append(f"{context}.phase_timings synchronization is invalid")
        if timings.get("total_boundary") != (
            "unit_start_through_first_complete_validation_"
            "excluding_telemetry_publication"
        ):
            errors.append(f"{context}.phase_timings total boundary is invalid")
        if timings.get("phases_are_mutually_exclusive") is not False:
            errors.append(f"{context}.phase_timings must declare overlapping phases")
        total = timings.get("synchronized_total_wall_time_seconds")
        if not _finite_nonnegative(total):
            errors.append(f"{context}.phase_timings total is invalid")
            total = None
        phases = timings.get("phases")
        if not isinstance(phases, dict) or set(phases) != _REQUIRED_PHASES:
            errors.append(f"{context}.phase_timings phases are invalid")
        else:
            for name in sorted(_REQUIRED_PHASES):
                item = phases[name]
                if not isinstance(item, dict) or set(item) != {
                    "wall_time_seconds", "invocations"
                }:
                    errors.append(f"{context}.phase_timings.{name} is invalid")
                    continue
                duration = item.get("wall_time_seconds")
                invocations = item.get("invocations")
                if not _finite_nonnegative(duration):
                    errors.append(
                        f"{context}.phase_timings.{name} duration is invalid"
                    )
                elif total is not None and duration > total:
                    errors.append(
                        f"{context}.phase_timings.{name} exceeds total wall"
                    )
                if type(invocations) is not int or invocations < 0:
                    errors.append(
                        f"{context}.phase_timings.{name} invocations is invalid"
                    )

    lifecycle_envelope = value.get("lifecycle")
    if (
        not isinstance(lifecycle_envelope, dict)
        or set(lifecycle_envelope) != {"integrity", "restoration"}
    ):
        errors.append(f"{context}.lifecycle has missing or unknown fields")
        lifecycle = None
        restoration = None
    else:
        lifecycle = lifecycle_envelope.get("integrity")
        restoration = lifecycle_envelope.get("restoration")
    lifecycle_fields = {
        "schema_version", "algorithm", "checked", "entry_sha256",
        "exit_sha256", "matched", "restored", "differences",
        "signature_calls", "verification_error_type",
        "verification_error_message", "body_exception_type",
    }
    if not isinstance(lifecycle, dict) or set(lifecycle) != lifecycle_fields:
        errors.append(
            f"{context}.lifecycle.integrity has missing or unknown fields"
        )
    else:
        if lifecycle.get("schema_version") != 1:
            errors.append(f"{context}.lifecycle schema_version is invalid")
        if not isinstance(lifecycle.get("algorithm"), str) or not lifecycle["algorithm"]:
            errors.append(f"{context}.lifecycle algorithm is invalid")
        checked = lifecycle.get("checked")
        if not isinstance(checked, list) or not checked or not all(
            isinstance(item, str) and item for item in checked
        ):
            errors.append(f"{context}.lifecycle checked is invalid")
        for field in ("entry_sha256", "exit_sha256"):
            digest = lifecycle.get(field)
            if digest is not None and (
                not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest)
            ):
                errors.append(f"{context}.lifecycle {field} is invalid")
        if lifecycle.get("matched") is not lifecycle.get("restored"):
            errors.append(f"{context}.lifecycle matched/restored disagree")
        if lifecycle.get("matched") is True and (
            lifecycle.get("entry_sha256") != lifecycle.get("exit_sha256")
        ):
            errors.append(f"{context}.lifecycle matched digests disagree")
        differences = lifecycle.get("differences")
        if not isinstance(differences, list) or not all(
            isinstance(item, str) and item for item in differences
        ):
            errors.append(f"{context}.lifecycle differences are invalid")
        elif bool(differences) is lifecycle.get("matched"):
            errors.append(f"{context}.lifecycle differences disagree with matched")
        calls = lifecycle.get("signature_calls")
        if type(calls) is not int or calls not in {1, 2}:
            errors.append(f"{context}.lifecycle signature_calls is invalid")
        for field in (
            "verification_error_type", "verification_error_message",
            "body_exception_type",
        ):
            if lifecycle.get(field) is not None and (
                not isinstance(lifecycle[field], str) or not lifecycle[field]
            ):
                errors.append(f"{context}.lifecycle {field} is invalid")
    restoration_fields = {
        "schema_version", "checked", "changes_detected", "restored",
        "differences", "errors",
    }
    if restoration is not None:
        if not isinstance(restoration, dict) or set(restoration) != restoration_fields:
            errors.append(
                f"{context}.lifecycle.restoration has missing or unknown fields"
            )
        else:
            if restoration.get("schema_version") != 1:
                errors.append(
                    f"{context}.lifecycle.restoration schema_version is invalid"
                )
            for field in ("changes_detected", "restored"):
                if type(restoration.get(field)) is not bool:
                    errors.append(
                        f"{context}.lifecycle.restoration {field} is invalid"
                    )
            for field in ("checked", "differences", "errors"):
                items = restoration.get(field)
                if not isinstance(items, list) or not all(
                    isinstance(item, str) and item for item in items
                ):
                    errors.append(
                        f"{context}.lifecycle.restoration {field} is invalid"
                    )

    environment = value.get("measurement_environment")
    environment_fields = {
        "device_type", "device", "device_name",
        "cuda_runtime_version", "cudnn_version",
    }
    if not isinstance(environment, dict) or set(environment) != environment_fields:
        errors.append(
            f"{context}.measurement_environment has missing or unknown fields"
        )
    elif environment.get("device_type") == "cpu":
        if environment.get("device") != "cpu" or any(
            environment.get(name) is not None
            for name in ("device_name", "cuda_runtime_version", "cudnn_version")
        ):
            errors.append(f"{context}.measurement_environment CPU fields are invalid")
    elif environment.get("device_type") == "cuda":
        if not all(
            isinstance(environment.get(name), str) and environment[name]
            for name in ("device", "device_name", "cuda_runtime_version")
        ) or type(environment.get("cudnn_version")) is not int:
            errors.append(f"{context}.measurement_environment CUDA fields are invalid")
    else:
        errors.append(f"{context}.measurement_environment device_type is invalid")


def validate_runner_telemetry(value: Mapping[str, Any]) -> None:
    """Validate the public terminal runner-telemetry contract."""

    errors: list[str] = []
    _validate_runner_telemetry(value, errors)
    if errors:
        raise ArtifactPolicyError(
            "invalid runner telemetry: " + "; ".join(errors)
        )


def _finite_nonnegative(value: Any) -> bool:
    return (
        type(value) in {int, float}
        and math.isfinite(float(value))
        and value >= 0
    )


def _validate_artifact_payload(
    record: dict[str, Any], raw: bytes, context: str, errors: list[str],
    *, png_structure_already_validated: bool = False,
) -> None:
    base_fields = {
        "level", "path", "sha256", "bytes", "kind", "dtype", "shape",
        "format",
    }
    level = record.get("level")
    allowed = base_fields | (
        {"display_only", "metric_input", "failure"}
        if level == "L3"
        else set()
    )
    if set(record) != allowed:
        errors.append(f"{context} has missing or unknown inventory fields")
    kind = record.get("kind")
    if not isinstance(kind, str) or not _SAFE_KIND_RE.fullmatch(kind):
        errors.append(f"{context} kind is invalid")
    artifact_format = record.get("format")
    try:
        if level == "L0":
            if (
                artifact_format != "json"
                or record.get("dtype") != "json"
                or record.get("shape") is not None
            ):
                raise ArtifactPolicyError(
                    "L0 must be JSON with dtype=json and shape=null"
                )
            decoded = _strict_json_loads(raw)
            if _json_file_bytes(decoded) != raw:
                raise ArtifactPolicyError("L0 JSON is not canonical")
        elif level == "L1":
            if artifact_format != "npy" or kind != "metric_ready":
                raise ArtifactPolicyError("L1 must be metric_ready NPY")
            array = _load_npy(raw)
            if array.dtype != np.dtype(np.float32) or array.shape != (14, 14):
                raise ArtifactPolicyError(
                    "L1 must be float32 shape [14,14]"
                )
            if (
                record.get("dtype") != "float32"
                or record.get("shape") != [14, 14]
            ):
                raise ArtifactPolicyError("L1 inventory dtype/shape mismatch")
            if not np.all(np.isfinite(array)):
                raise ArtifactPolicyError("L1 contains non-finite values")
        elif level == "L2":
            if artifact_format == "npy":
                array = _load_npy(raw)
                if (
                    array.dtype != np.dtype(np.float32)
                    or array.size == 0
                    or not np.all(np.isfinite(array))
                ):
                    raise ArtifactPolicyError(
                        "L2 NPY must be non-empty finite float32"
                    )
                if (
                    record.get("dtype") != "float32"
                    or record.get("shape") != list(array.shape)
                ):
                    raise ArtifactPolicyError(
                        "L2 NPY inventory dtype/shape mismatch"
                    )
            elif artifact_format == "npz":
                arrays = _load_npz(raw)
                dtype = {key: "float32" for key in arrays}
                shape = {
                    key: list(value.shape) for key, value in arrays.items()
                }
                if not arrays or any(
                    value.dtype != np.dtype(np.float32)
                    or value.size == 0
                    or not np.all(np.isfinite(value))
                    for value in arrays.values()
                ):
                    raise ArtifactPolicyError(
                        "L2 NPZ members must be non-empty finite float32"
                    )
                if (
                    record.get("dtype") != dtype
                    or record.get("shape") != shape
                ):
                    raise ArtifactPolicyError(
                        "L2 NPZ inventory dtype/shape mismatch"
                    )
            else:
                raise ArtifactPolicyError("L2 format must be npy or npz")
        elif level == "L3":
            if artifact_format != "png" or kind != "display":
                raise ArtifactPolicyError("L3 must be display PNG")
            if (
                record.get("dtype") != "bytes"
                or record.get("shape") is not None
            ):
                raise ArtifactPolicyError("L3 inventory dtype/shape is invalid")
            if (
                record.get("display_only") is not True
                or record.get("metric_input") is not False
            ):
                raise ArtifactPolicyError(
                    "L3 must be marked display-only/non-metric"
                )
            if type(record.get("failure")) is not bool:
                raise ArtifactPolicyError("L3 failure marker must be bool")
            if not png_structure_already_validated:
                _verify_png_header(raw)
        else:
            raise ArtifactPolicyError(f"unknown retention level {level!r}")
    except Exception as exc:
        errors.append(f"{context} payload contract violation: {exc}")


def _validate_provenance(
    provenance: dict[str, Any],
    manifest: dict[str, Any],
    expected_bindings: Mapping[str, Any] | None,
    observed_git_state: Mapping[str, Any] | None,
    errors: list[str],
) -> None:
    required = {
        "schema_version", "document_type", "unit_identity",
        "unit_identity_sha256", "bindings",
    }
    if set(provenance) != required:
        errors.append("provenance envelope has missing or unknown fields")
    if provenance.get("schema_version") != SCHEMA_VERSION:
        errors.append("provenance schema_version is invalid")
    if provenance.get("document_type") != PROVENANCE_DOCUMENT_TYPE:
        errors.append("provenance document_type is invalid")
    if provenance.get("unit_identity") != manifest.get("unit_identity"):
        errors.append("provenance unit identity does not match manifest")
    if (
        provenance.get("unit_identity_sha256")
        != manifest.get("unit_identity_sha256")
    ):
        errors.append("provenance unit identity digest does not match manifest")
    bindings = provenance.get("bindings")
    if not isinstance(bindings, dict):
        errors.append("provenance bindings must be an object")
    if expected_bindings is not None and isinstance(bindings, dict):
        try:
            expected = _json_copy(expected_bindings)
        except Exception as exc:
            errors.append(f"expected provenance bindings are invalid: {exc}")
        else:
            if (
                isinstance(expected, dict)
                and expected.get("document_type") == PROVENANCE_DOCUMENT_TYPE
            ):
                target: Any = provenance
            else:
                target = bindings
            mismatch = _first_subset_mismatch(expected, target, "provenance")
            if mismatch is not None:
                errors.append(
                    f"expected provenance binding mismatch: {mismatch}"
                )
    if isinstance(bindings, dict):
        declared_inventory = bindings.get("artifact_inventory")
        if declared_inventory is not None:
            actual_inventory = [
                item
                for item in manifest.get("artifacts", [])
                if isinstance(item, dict) and item.get("path") != PROVENANCE_PATH
            ]
            if declared_inventory != actual_inventory:
                errors.append(
                    "provenance artifact_inventory does not exactly match "
                    "manifest inventory excluding provenance"
                )
            resources = bindings.get("resources")
            actual = resources.get("actual") if isinstance(resources, dict) else None
            if isinstance(actual, dict):
                expected_files = [
                    {"path": item["path"], "bytes": item["bytes"]}
                    for item in actual_inventory
                ]
                if actual.get("files") != expected_files:
                    errors.append(
                        "provenance resource file measurement does not exactly "
                        "match artifact inventory"
                    )
                if actual.get("artifact_bytes") != sum(
                    item["bytes"] for item in actual_inventory
                ):
                    errors.append(
                        "provenance resource artifact_bytes does not match inventory"
                    )
        if manifest.get("status") == "failed":
            failure = bindings.get("failure")
            manifest_failure = manifest.get("failure")
            if isinstance(failure, dict) and isinstance(manifest_failure, dict):
                details = manifest_failure.get("details")
                # Production failure bindings have reason/phase/type/message;
                # runner.fail embeds that exact object as details.  Generic
                # callers may use any JSON details and are compared through
                # expected_provenance_bindings when requested.
                if set(failure) == {"reason", "phase", "type", "message"} and (
                    details != failure
                    or manifest_failure.get("reason") != failure.get("reason")
                ):
                    errors.append(
                        "provenance failure does not match manifest failure"
                    )
        production_identity = "canonical_digest" in manifest.get(
            "unit_identity", {}
        )
        if production_identity and set(bindings) != _STRICT_PROVENANCE_BINDING_FIELDS:
            errors.append(
                "production UnitKey requires complete Stage C provenance bindings"
            )
        elif set(bindings) == _STRICT_PROVENANCE_BINDING_FIELDS:
            try:
                from .stage_c_provenance import validate_stage_c_provenance

                validate_stage_c_provenance(
                    provenance,
                    expected_unit=manifest.get("unit_identity"),
                    observed_git_state=observed_git_state,
                )
            except Exception as exc:
                errors.append(f"strict Stage C provenance is invalid: {exc}")


def _first_subset_mismatch(
    expected: Any, actual: Any, path: str
) -> str | None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return f"{path} is not an object"
        for key, value in expected.items():
            if key not in actual:
                return f"{path}.{key} is missing"
            mismatch = _first_subset_mismatch(
                value, actual[key], f"{path}.{key}"
            )
            if mismatch is not None:
                return mismatch
        return None
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            return f"{path} list differs"
        for index, value in enumerate(expected):
            mismatch = _first_subset_mismatch(
                value, actual[index], f"{path}[{index}]"
            )
            if mismatch is not None:
                return mismatch
        return None
    if type(expected) is not type(actual) or expected != actual:
        return f"{path} differs"
    return None


def _normalize_provenance(
    value: Mapping[str, Any], identity: dict[str, Any], digest: str
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("provenance must be a mapping")
    candidate = _json_copy(value)
    envelope_keys = {
        "schema_version", "document_type", "unit_identity",
        "unit_identity_sha256", "bindings",
    }
    if (
        candidate.get("document_type") == PROVENANCE_DOCUMENT_TYPE
        or set(candidate) == envelope_keys
    ):
        if set(candidate) != envelope_keys:
            raise ArtifactPolicyError(
                "full provenance envelope has missing or unknown fields"
            )
        if candidate.get("schema_version") != SCHEMA_VERSION:
            raise ArtifactPolicyError("provenance schema_version is invalid")
        if candidate.get("document_type") != PROVENANCE_DOCUMENT_TYPE:
            raise ArtifactPolicyError("provenance document_type is invalid")
        if candidate.get("unit_identity") != identity:
            raise ArtifactPolicyError(
                "provenance unit identity does not match writer"
            )
        if candidate.get("unit_identity_sha256") != digest:
            raise ArtifactPolicyError(
                "provenance unit identity digest does not match writer"
            )
        if not isinstance(candidate.get("bindings"), dict):
            raise ArtifactPolicyError("provenance bindings must be an object")
        if "canonical_digest" in identity:
            if set(candidate["bindings"]) != _STRICT_PROVENANCE_BINDING_FIELDS:
                raise ArtifactPolicyError(
                    "production UnitKey requires complete Stage C provenance bindings"
                )
            try:
                from .stage_c_provenance import validate_stage_c_provenance

                validate_stage_c_provenance(candidate, expected_unit=identity)
            except Exception as exc:
                raise ArtifactPolicyError(
                    f"strict Stage C provenance is invalid: {exc}"
                ) from exc
        return candidate
    if "canonical_digest" in identity:
        raise ArtifactPolicyError(
            "production UnitKey requires a full Stage C provenance envelope"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "document_type": PROVENANCE_DOCUMENT_TYPE,
        "unit_identity": _json_copy(identity),
        "unit_identity_sha256": digest,
        "bindings": candidate,
    }


def _identity_copy(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError("unit_identity must be a non-empty mapping")
    result = _json_copy(value)
    if not isinstance(result, dict) or not result:
        raise ValueError("unit_identity must be a non-empty JSON object")
    return result


def _artifact_kind(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("artifact kind must be a string")
    normalized = value.strip().lower().replace("-", "_")
    if not _SAFE_KIND_RE.fullmatch(normalized):
        raise ArtifactPolicyError(f"invalid artifact kind: {value!r}")
    return normalized


def _artifact_relative_path(level: str, name: str, suffix: str) -> str:
    if level not in {"L0", "L1", "L2", "L3"}:
        raise ArtifactPolicyError(f"unknown artifact level: {level!r}")
    if not isinstance(name, str) or not name.strip():
        raise ArtifactPolicyError(
            "artifact name must be a non-empty relative path"
        )
    if "\\" in name:
        raise ArtifactPolicyError("artifact name may not contain backslashes")
    supplied = PurePosixPath(name)
    if supplied.is_absolute() or any(
        part in {"", ".", ".."} for part in supplied.parts
    ):
        raise ArtifactPolicyError(
            "artifact name must not escape its level directory"
        )
    for component in supplied.parts:
        if not _SAFE_COMPONENT_RE.fullmatch(component):
            raise ArtifactPolicyError(
                f"unsafe artifact path component: {component!r}"
            )
        if _GRANULAR_RE.search(component):
            raise ArtifactPolicyError(
                "per-step, per-pixel, and per-dimension artifact files "
                "are forbidden"
            )
    if supplied.suffix:
        if supplied.suffix.lower() != suffix:
            raise ArtifactPolicyError(
                f"{level} writer requires {suffix} artifacts, "
                f"got {supplied.suffix}"
            )
        with_suffix = supplied
    else:
        with_suffix = supplied.with_name(supplied.name + suffix)
    return str(PurePosixPath(level) / with_suffix)


def _valid_inventory_path(value: Any, level: Any) -> bool:
    if (
        not isinstance(value, str)
        or level not in {"L0", "L1", "L2", "L3"}
        or "\\" in value
    ):
        return False
    path = PurePosixPath(value)
    return (
        not path.is_absolute()
        and len(path.parts) >= 2
        and path.parts[0] == level
        and all(part not in {"", ".", ".."} for part in path.parts)
        and all(
            _SAFE_COMPONENT_RE.fullmatch(part) for part in path.parts[1:]
        )
        and not any(_GRANULAR_RE.search(part) for part in path.parts[1:])
    )


def _float32_array(
    value: Any, level: str, required_shape: tuple[int, ...] | None = None
) -> np.ndarray:
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu().numpy()
    elif (
        hasattr(value, "data")
        and isinstance(getattr(value, "data"), np.ndarray)
    ):
        value = value.data
    array = np.asarray(value)
    if array.dtype != np.dtype(np.float32):
        raise ArtifactPolicyError(
            f"{level} arrays must already be float32, got {array.dtype}"
        )
    if required_shape is not None and array.shape != required_shape:
        raise ArtifactPolicyError(
            f"{level} array must have shape {list(required_shape)}, "
            f"got {list(array.shape)}"
        )
    if array.size == 0:
        raise ArtifactPolicyError(f"{level} arrays must not be empty")
    if not np.all(np.isfinite(array)):
        raise ArtifactPolicyError(
            f"{level} arrays must contain only finite values"
        )
    return np.ascontiguousarray(array)


def _serialize_npy(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return buffer.getvalue()


def _serialize_npz(arrays: Mapping[str, np.ndarray]) -> bytes:
    """Serialize a deterministic NPZ with fixed member timestamps/order."""

    buffer = io.BytesIO()
    with zipfile.ZipFile(
        buffer, mode="w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for key, array in sorted(arrays.items()):
            info = zipfile.ZipInfo(
                f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0)
            )
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, _serialize_npy(array))
    return buffer.getvalue()


def _load_npy(raw: bytes) -> np.ndarray:
    value = np.load(io.BytesIO(raw), allow_pickle=False)
    if not isinstance(value, np.ndarray):
        raise ArtifactPolicyError("NPY did not decode to one ndarray")
    return value


def _load_npz(raw: bytes) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        if len(archive.files) != len(set(archive.files)):
            raise ArtifactPolicyError("NPZ contains duplicate members")
        for key in sorted(archive.files):
            if not _SAFE_COMPONENT_RE.fullmatch(key):
                raise ArtifactPolicyError("NPZ contains an unsafe member name")
            result[key] = np.asarray(archive[key])
    return result


def _verify_png(payload: bytes) -> None:
    _verify_png_header(payload)
    try:
        from PIL import Image

        with Image.open(io.BytesIO(payload)) as image:
            if image.format != "PNG":
                raise ArtifactPolicyError("L3 image format is not PNG")
            image.verify()
    except ArtifactPolicyError:
        raise
    except Exception as exc:
        raise ArtifactPolicyError(f"invalid PNG payload: {exc}") from exc


def _verify_png_header(payload: bytes) -> None:
    """Check the display boundary without decoding pixels."""

    if len(payload) < 24 or not payload.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ArtifactPolicyError("L3 bytes are not a PNG")
    if payload[12:16] != b"IHDR":
        raise ArtifactPolicyError("L3 PNG does not begin with IHDR")


def _validate_png_path(path: Path) -> tuple[str, bytes, int]:
    """Stream-validate PNG chunks/CRCs and hash the same opened file.

    This checks storage integrity only.  It deliberately never decodes pixels
    and therefore cannot turn a display artifact into quantitative input.
    """

    digest = hashlib.sha256()
    prefix = bytearray()
    streamed_size = 0

    def consume(block: bytes) -> None:
        nonlocal streamed_size
        digest.update(block)
        streamed_size += len(block)
        if len(prefix) < 32:
            prefix.extend(block[: 32 - len(prefix)])

    seen_ihdr = False
    seen_iend = False
    with path.open("rb") as handle:
        signature = handle.read(8)
        consume(signature)
        if signature != b"\x89PNG\r\n\x1a\n":
            raise ArtifactPolicyError("L3 bytes are not a PNG")
        while not seen_iend:
            header = handle.read(8)
            consume(header)
            if len(header) != 8:
                raise ArtifactPolicyError("truncated PNG chunk header")
            length = struct.unpack(">I", header[:4])[0]
            chunk_type = header[4:]
            if not seen_ihdr and chunk_type != b"IHDR":
                raise ArtifactPolicyError("L3 PNG does not begin with IHDR")
            if seen_ihdr and chunk_type == b"IHDR":
                raise ArtifactPolicyError("PNG contains duplicate IHDR")
            if length > 2**31 - 1:
                raise ArtifactPolicyError("PNG chunk length is invalid")
            remaining = length
            crc = zlib.crc32(chunk_type)
            while remaining:
                block = handle.read(min(1024 * 1024, remaining))
                if not block:
                    raise ArtifactPolicyError("truncated PNG chunk data")
                consume(block)
                crc = zlib.crc32(block, crc)
                remaining -= len(block)
            stored_crc = handle.read(4)
            consume(stored_crc)
            if len(stored_crc) != 4:
                raise ArtifactPolicyError("truncated PNG chunk CRC")
            if struct.unpack(">I", stored_crc)[0] != (crc & 0xFFFFFFFF):
                raise ArtifactPolicyError("PNG chunk CRC mismatch")
            seen_ihdr = True
            if chunk_type == b"IEND":
                if length != 0:
                    raise ArtifactPolicyError("PNG IEND must be empty")
                seen_iend = True
        trailing = handle.read(1)
        if trailing:
            raise ArtifactPolicyError("PNG contains bytes after IEND")
    return digest.hexdigest(), bytes(prefix), streamed_size


def _read_control(
    files: Mapping[str, tuple[Path, int]],
    relative: str,
    label: str,
    errors: list[str],
) -> bytes | None:
    item = files.get(relative)
    if item is None:
        errors.append(f"missing {label} control file: {relative}")
        return None
    try:
        return item[0].read_bytes()
    except OSError as exc:
        errors.append(f"cannot read {label}: {exc}")
        return None


def _load_canonical_json(
    raw: bytes, label: str, errors: list[str]
) -> dict[str, Any] | None:
    try:
        value = _strict_json_loads(raw)
    except Exception as exc:
        errors.append(f"{label} is not strict JSON: {exc}")
        return None
    if not isinstance(value, dict):
        errors.append(f"{label} must be a JSON object")
        return None
    if _json_file_bytes(value) != raw:
        errors.append(f"{label} bytes are not canonical")
    return value


def _strict_json_loads(raw: bytes) -> Any:
    def reject_constant(value: str) -> Any:
        raise ValueError(f"non-finite JSON number {value}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    return json.loads(
        raw.decode("utf-8"),
        parse_constant=reject_constant,
        object_pairs_hook=reject_duplicates,
    )


def _json_file_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            _json_compatible(value),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _json_copy(value: Any) -> Any:
    return json.loads(canonical_json_bytes(value).decode("ascii"))


def _json_compatible(value: Any, _seen: set[int] | None = None) -> Any:
    seen = set() if _seen is None else _seen
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON values must be finite")
        return value
    if isinstance(value, np.generic):
        return _json_compatible(value.item(), seen)
    if isinstance(value, np.ndarray):
        return _json_compatible(value.tolist(), seen)
    if isinstance(value, Enum):
        return _json_compatible(value.value, seen)
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return _json_compatible(asdict(value), seen)
    identity = id(value)
    if isinstance(value, Mapping):
        if identity in seen:
            raise ValueError("cyclic JSON mapping")
        seen.add(identity)
        try:
            result: dict[str, Any] = {}
            for key, item in value.items():
                if not isinstance(key, str):
                    raise TypeError("JSON object keys must be strings")
                result[key] = _json_compatible(item, seen)
            return result
        finally:
            seen.remove(identity)
    if isinstance(value, (list, tuple)):
        if identity in seen:
            raise ValueError("cyclic JSON sequence")
        seen.add(identity)
        try:
            return [_json_compatible(item, seen) for item in value]
        finally:
            seen.remove(identity)
    raise TypeError(
        f"value of type {type(value).__name__} is not JSON-compatible"
    )


def _absolute_path(value: str | Path) -> Path:
    return Path(os.path.abspath(os.fspath(value)))


def _ensure_directory_no_symlink(path: Path, *, create: bool) -> None:
    _assert_no_symlink_components(path, allow_missing=create)
    if create:
        path.mkdir(parents=True, exist_ok=True)
        _assert_no_symlink_components(path)
    if not path.exists():
        raise FileNotFoundError(path)
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode):
        raise ArtifactPolicyError(f"symlink directory is forbidden: {path}")
    if not stat.S_ISDIR(info.st_mode):
        raise NotADirectoryError(path)


def _assert_no_symlink_components(
    path: Path, *, allow_missing: bool = False
) -> None:
    """Reject symlinks in every existing component of an absolute path."""

    absolute = _absolute_path(path)
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current = current / component
        if not os.path.lexists(current):
            if allow_missing:
                continue
            raise FileNotFoundError(current)
        if stat.S_ISLNK(current.lstat().st_mode):
            raise ArtifactPolicyError(
                f"symlink path component is forbidden: {current}"
            )


def _ensure_artifact_parent(root: Path, parent: Path) -> None:
    relative = parent.relative_to(root)
    current = root
    for component in relative.parts:
        current = current / component
        try:
            current.mkdir(exist_ok=False)
        except FileExistsError:
            info = current.lstat()
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                raise ArtifactPolicyError(
                    f"artifact parent is not a real directory: {current}"
                )
        else:
            _fsync_directory(current.parent)


def _inventory_regular_files(
    root: Path,
) -> tuple[dict[str, tuple[Path, int]], list[str]]:
    files: dict[str, tuple[Path, int]] = {}
    problems: list[str] = []

    def visit(directory: Path, relative_parent: PurePosixPath) -> None:
        try:
            entries = sorted(os.scandir(directory), key=lambda item: item.name)
        except OSError as exc:
            problems.append(f"cannot scan {directory}: {exc}")
            return
        for entry in entries:
            relative = relative_parent / entry.name
            relative_text = str(relative)
            try:
                if entry.is_symlink():
                    problems.append(f"forbidden symlink: {relative_text}")
                elif entry.is_dir(follow_symlinks=False):
                    visit(Path(entry.path), relative)
                elif entry.is_file(follow_symlinks=False):
                    info = entry.stat(follow_symlinks=False)
                    files[relative_text] = (Path(entry.path), info.st_size)
                else:
                    problems.append(
                        f"non-regular filesystem entry: {relative_text}"
                    )
            except OSError as exc:
                problems.append(f"cannot inspect {relative_text}: {exc}")

    visit(root, PurePosixPath())
    return files, problems


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
        raise ArtifactPolicyError(f"refusing to replace symlink: {path}")
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


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _nonnegative_integer(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


__all__ = [
    "ArtifactBudgetError",
    "ArtifactCaps",
    "ArtifactPolicyError",
    "ArtifactWriterStateError",
    "AttemptNotEmptyError",
    "DEFAULT_L0_L1_CAP_BYTES",
    "DEFAULT_L2_CAP_BYTES",
    "ExecutionUnitValidation",
    "FAULT_AFTER_MANIFEST",
    "FAULT_AFTER_PAYLOAD",
    "FAULT_AFTER_RUNNING",
    "FAULT_BEFORE_COMPLETE_STATUS",
    "FAULT_POINTS",
    "InjectedArtifactCrash",
    "MANIFEST_FILENAME",
    "PROVENANCE_DOCUMENT_TYPE",
    "PROVENANCE_PATH",
    "STATUS_FILENAME",
    "StageCArtifactError",
    "StageCArtifactWriter",
    "attempt_directories",
    "canonical_json_bytes",
    "next_attempt_directory",
    "unit_identity_sha256",
    "validate_execution_unit",
    "validate_runner_telemetry",
]
