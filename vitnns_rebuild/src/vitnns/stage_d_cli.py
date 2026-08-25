"""Non-interactive JSON CLI for the Stage D control and runtime planes.

Importing the module performs no Git query, filesystem validation, runtime
import, model construction, image access, or scientific computation.  The D4
runtime commands consume a separately issued D7 authorization and fail closed
when its validator is unavailable; this module never issues that authority.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import re
import stat
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from .constants import REPO_ROOT
from .stage_d_authorization import (
    StageDAuthorizationError,
    build_stage_d_preparation_authorization,
    capture_stage_d_preparation_inputs,
    validate_stage_d_preparation_authorization,
)
from .stage_d_config import (
    StageDConfigError,
    load_stage_d_contracts,
    strict_json_bytes,
)
from .stage_d_manifest import (
    StageDManifestError,
    verify_stage_d_manifest,
)
from .stage_d_plan import (
    StageDPlanError,
    build_stage_d_plan,
    estimate_stage_d_run,
    preflight_stage_d,
    validate_stage_d_estimate,
    validate_stage_d_plan,
    validate_stage_d_preflight,
)


EXIT_SUCCESS = 0
EXIT_INPUT_ERROR = 1
EXIT_PREFLIGHT_BLOCKED = 2
EXIT_INTEGRITY_FAILURE = 4

_PROFILE_FILE_LIMIT = 16 * 1024 * 1024
_AUTHORIZATION_FILE_LIMIT = 16 * 1024 * 1024
_RETRY_AUTHORIZATION_FILE_LIMIT = 16 * 1024 * 1024
_MAX_RETRY_AUTHORIZATIONS = 1000
_RUNTIME_COMMANDS = frozenset({"init", "run-shard"})

_SHARD_SUMMARY_KEYS = frozenset({
    "schema_version", "document_type", "shard_id", "shard_count",
    "state", "counts", "resources", "checkpoint_generation",
    "checkpoint_digest", "execution_context_digest",
})
_SHARD_COUNT_KEYS = frozenset({
    "expected_groups", "visited_groups", "claimed_groups",
    "terminal_groups", "expected_units", "attempted_units",
    "terminal_units", "succeeded_units", "scientific_failed_units",
    "retry_pending_units", "blocked_units", "skipped_terminal_units",
})
_SHARD_RESOURCE_KEYS = frozenset({
    "elapsed_seconds", "forward_calls", "backward_calls",
    "l0_l1_payload_bytes", "l2_payload_bytes",
    "control_and_provenance_bytes", "cuda_peak_reserved_bytes",
})
_INITIALIZATION_OUTPUT_KEYS = frozenset({
    "schema_version", "document_type", "status", "control",
    "digests", "counts",
})
_INITIALIZATION_CONTROL_KEYS = frozenset({
    "operation", "output_root", "external_execution_gate_validated",
    "analysis_authorization",
})
_INITIALIZATION_DIGEST_KEYS = frozenset({"plan_digest"})
_SHARD_OUTPUT_KEYS = frozenset({
    "schema_version", "document_type", "status", "control",
    "terminal", "resources", "digests", "counts",
})
_SHARD_CONTROL_KEYS = frozenset({
    "shard_index", "shard_count", "checkpoint_generation",
})
_SHARD_TERMINAL_KEYS = frozenset({"state"})
_SHARD_DIGEST_KEYS = frozenset({
    "checkpoint_digest", "execution_context_digest",
})
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ArgumentError(ValueError):
    """Raised instead of allowing argparse to emit a non-JSON error."""


class StageDCLIInputError(ValueError):
    """Raised for malformed optional CLI inputs such as profile JSON."""


class StageDCLIIntegrityError(RuntimeError):
    """Raised when a live control-plane observation cannot be trusted."""


class StageDCLIRuntimeBlockedError(RuntimeError):
    """Raised when the external D7/runtime gate refuses execution."""


class StageDCLIRuntimeIntegrityError(RuntimeError):
    """Raised when D4 runtime validation fails closed."""


class _JSONArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise ArgumentError(message)


@dataclass(frozen=True, slots=True)
class StageDCLIPipeline:
    """Verified source objects shared by one CLI command invocation."""

    contracts: Any
    manifest: Any
    plan: Any | None = None


def _add_source_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--repo-root", type=Path, default=REPO_ROOT,
        help="repository root containing the frozen Stage D sources",
    )
    parser.add_argument(
        "--config", type=Path,
        help="frozen Stage D config JSON",
    )
    parser.add_argument(
        "--run-spec", type=Path,
        help="frozen Stage D run-spec JSON",
    )
    parser.add_argument(
        "--config-schema", type=Path,
        help="Stage D config schema",
    )
    parser.add_argument(
        "--run-spec-schema", type=Path,
        help="Stage D run-spec schema",
    )


def _add_profile_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--profiles", type=Path,
        help=(
            "strict JSON array or method/unit-kind mapping of non-authoritative "
            "pre-D7 resource profiles"
        ),
    )


def _nonnegative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "must be a non-negative integer"
        ) from error
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def _add_runtime_arguments(
    parser: argparse.ArgumentParser, *, run_shard: bool,
) -> None:
    _add_source_arguments(parser)
    parser.add_argument(
        "--output-root", type=Path, required=True,
        help="fresh or initialized formal Stage D output root",
    )
    parser.add_argument(
        "--authorization", type=Path, required=True,
        help="strict D7 execution-authorization JSON",
    )
    if run_shard:
        parser.add_argument(
            "--shard-index", type=_nonnegative_int, required=True,
            help="zero-based shard coordinate frozen by D7",
        )
        parser.add_argument(
            "--device", required=True,
            help="runtime device selected within the D7-authorized scope",
        )
        parser.add_argument(
            "--retry-authorization", type=Path, action="append", default=[],
            help=(
                "strict coordinator retry-authorization JSON; repeat once "
                "for each authorized retry"
            ),
        )


def build_parser() -> argparse.ArgumentParser:
    parser = _JSONArgumentParser(
        prog="vitnns-stage-d",
        description="Strict metadata-only Stage D planning and preparation",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser(
        "validate-contracts",
        help="load frozen contracts and fully verify their source manifest",
    )
    _add_source_arguments(validate)

    plan = commands.add_parser(
        "plan", help="build and validate the exact typed execution plan",
    )
    _add_source_arguments(plan)

    estimate = commands.add_parser(
        "estimate", help="produce a metadata-only estimate and preflight",
    )
    _add_source_arguments(estimate)
    _add_profile_argument(estimate)

    prepare = commands.add_parser(
        "prepare-authorization",
        help="build and revalidate a preparation-only authorization",
    )
    _add_source_arguments(prepare)
    _add_profile_argument(prepare)

    initialize = commands.add_parser(
        "init",
        help="initialize a formal root under an externally issued D7 gate",
    )
    _add_runtime_arguments(initialize, run_shard=False)

    run_shard = commands.add_parser(
        "run-shard",
        help="run one D7-authorized shard without exposing scientific values",
    )
    _add_runtime_arguments(run_shard, run_shard=True)
    return parser


def _json_value(value: Any, context: str = "value") -> Any:
    converter = getattr(value, "to_dict", None)
    if callable(converter):
        value = converter()
    if value is None or type(value) in {str, bool, int}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise StageDCLIIntegrityError(
                f"{context} contains a non-finite number"
            )
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, child in value.items():
            if not isinstance(key, str):
                raise StageDCLIIntegrityError(
                    f"{context} contains a non-string JSON key"
                )
            result[key] = _json_value(child, f"{context}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [
            _json_value(child, f"{context}[{index}]")
            for index, child in enumerate(value)
        ]
    raise StageDCLIIntegrityError(
        f"{context} contains non-JSON value {type(value).__name__}"
    )


def _write_json(value: Any, stream: TextIO) -> None:
    document = _json_value(value)
    stream.write(json.dumps(
        document, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False,
    ))
    stream.write("\n")
    stream.flush()


def _runtime_mapping(value: Any, context: str) -> dict[str, Any]:
    try:
        document = _json_value(value, context)
    except StageDCLIIntegrityError as error:
        raise StageDCLIRuntimeIntegrityError(
            "runtime result is not finite, strict control JSON"
        ) from error
    except Exception as error:
        raise StageDCLIRuntimeIntegrityError(
            "runtime result could not be converted to safe control JSON"
        ) from error
    if not isinstance(document, dict):
        raise StageDCLIRuntimeIntegrityError(
            "runtime result must be a JSON object"
        )
    return document


def _require_exact_runtime_keys(
    value: Any, expected: frozenset[str], context: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise StageDCLIRuntimeIntegrityError(
            f"{context} does not match the closed control schema"
        )
    return value


def _validate_initialization_output(value: Any) -> dict[str, Any]:
    document = _runtime_mapping(value, "initialization output")
    _require_exact_runtime_keys(
        document, _INITIALIZATION_OUTPUT_KEYS, "initialization output",
    )
    control = _require_exact_runtime_keys(
        document["control"], _INITIALIZATION_CONTROL_KEYS,
        "initialization output.control",
    )
    digests = _require_exact_runtime_keys(
        document["digests"], _INITIALIZATION_DIGEST_KEYS,
        "initialization output.digests",
    )
    counts = _require_exact_runtime_keys(
        document["counts"], frozenset(), "initialization output.counts",
    )
    if (
        document["schema_version"] != 1
        or document["document_type"] != "stage_d_initialization_control"
        or document["status"] != "PASS"
        or control["operation"] != "init"
        or not isinstance(control["output_root"], str)
        or not control["output_root"]
        or control["external_execution_gate_validated"] is not True
        or control["analysis_authorization"] != "NOT_EVALUATED"
        or not isinstance(digests["plan_digest"], str)
        or _SHA256_RE.fullmatch(digests["plan_digest"]) is None
        or counts
    ):
        raise StageDCLIRuntimeIntegrityError(
            "initialization output contains invalid control values"
        )
    return document


def _validate_shard_output(value: Any) -> dict[str, Any]:
    document = _runtime_mapping(value, "shard output")
    _require_exact_runtime_keys(document, _SHARD_OUTPUT_KEYS, "shard output")
    control = _require_exact_runtime_keys(
        document["control"], _SHARD_CONTROL_KEYS, "shard output.control",
    )
    terminal = _require_exact_runtime_keys(
        document["terminal"], _SHARD_TERMINAL_KEYS, "shard output.terminal",
    )
    resources = _require_exact_runtime_keys(
        document["resources"], _SHARD_RESOURCE_KEYS, "shard output.resources",
    )
    digests = _require_exact_runtime_keys(
        document["digests"], _SHARD_DIGEST_KEYS, "shard output.digests",
    )
    counts = _require_exact_runtime_keys(
        document["counts"], _SHARD_COUNT_KEYS, "shard output.counts",
    )
    state = terminal["state"]
    if (
        document["schema_version"] != 1
        or document["document_type"] != "stage_d_shard_control"
        or state not in {"completed", "retry_pending", "blocked"}
        or document["status"]
        != ("PASS" if state == "completed" else "BLOCKED")
        or type(control["shard_index"]) is not int
        or type(control["shard_count"]) is not int
        or not 0 <= control["shard_index"] < control["shard_count"]
        or type(control["checkpoint_generation"]) is not int
        or control["checkpoint_generation"] < 0
        or any(
            not isinstance(digests[key], str)
            or _SHA256_RE.fullmatch(digests[key]) is None
            for key in _SHARD_DIGEST_KEYS
        )
        or any(type(item) is not int or item < 0 for item in counts.values())
    ):
        raise StageDCLIRuntimeIntegrityError(
            "shard output contains invalid control values"
        )
    for key, item in resources.items():
        valid = (
            type(item) in {int, float}
            and math.isfinite(item)
            and item >= 0
        ) if key == "elapsed_seconds" else type(item) is int and item >= 0
        if not valid:
            raise StageDCLIRuntimeIntegrityError(
                "shard output contains invalid resource values"
            )
    return document


def _closed_shard_summary(value: Any) -> Mapping[str, Any]:
    """Convert the runtime's closed summary to CLI output categories."""

    document = _runtime_mapping(value, "runtime shard summary")
    if set(document) != _SHARD_SUMMARY_KEYS:
        raise StageDCLIRuntimeIntegrityError(
            "Stage D shard summary does not match its closed schema"
        )
    if (
        document["schema_version"] != 1
        or document["document_type"] != "stage_d_shard_summary"
        or type(document["shard_id"]) is not int
        or type(document["shard_count"]) is not int
        or not 0 <= document["shard_id"] < document["shard_count"]
        or document["state"] not in {"completed", "retry_pending", "blocked"}
        or type(document["checkpoint_generation"]) is not int
        or document["checkpoint_generation"] < 0
        or not isinstance(document["checkpoint_digest"], str)
        or _SHA256_RE.fullmatch(document["checkpoint_digest"]) is None
        or not isinstance(document["execution_context_digest"], str)
        or _SHA256_RE.fullmatch(document["execution_context_digest"]) is None
    ):
        raise StageDCLIRuntimeIntegrityError(
            "Stage D shard summary control fields are invalid"
        )
    counts = document["counts"]
    resources = document["resources"]
    if not isinstance(counts, Mapping) or set(counts) != _SHARD_COUNT_KEYS:
        raise StageDCLIRuntimeIntegrityError(
            "Stage D shard summary count fields are invalid"
        )
    if any(type(item) is not int or item < 0 for item in counts.values()):
        raise StageDCLIRuntimeIntegrityError(
            "Stage D shard summary counts must be non-negative integers"
        )
    if not isinstance(resources, Mapping) or set(resources) != _SHARD_RESOURCE_KEYS:
        raise StageDCLIRuntimeIntegrityError(
            "Stage D shard summary resource fields are invalid"
        )
    for key, item in resources.items():
        if key == "elapsed_seconds":
            valid = (
                type(item) in {int, float}
                and math.isfinite(item)
                and item >= 0
            )
        else:
            valid = type(item) is int and item >= 0
        if not valid:
            raise StageDCLIRuntimeIntegrityError(
                "Stage D shard summary resources are invalid"
            )
    output = {
        "schema_version": 1,
        "document_type": "stage_d_shard_control",
        "status": "PASS" if document["state"] == "completed" else "BLOCKED",
        "control": {
            "shard_index": document["shard_id"],
            "shard_count": document["shard_count"],
            "checkpoint_generation": document["checkpoint_generation"],
        },
        "terminal": {"state": document["state"]},
        "resources": dict(resources),
        "digests": {
            "checkpoint_digest": document["checkpoint_digest"],
            "execution_context_digest": document["execution_context_digest"],
        },
        "counts": dict(counts),
    }
    return _validate_shard_output(output)


def _write_runtime_json(value: Any, stream: TextIO) -> dict[str, Any]:
    raw = _runtime_mapping(value, "runtime output")
    document_type = raw.get("document_type")
    if document_type == "stage_d_initialization_control":
        document = _validate_initialization_output(raw)
    elif document_type == "stage_d_shard_control":
        document = _validate_shard_output(raw)
    else:
        raise StageDCLIRuntimeIntegrityError(
            "runtime output document type is outside the closed CLI surface"
        )
    _write_json(document, stream)
    return document


def _load_pipeline(args: argparse.Namespace) -> StageDCLIPipeline:
    """Load each source once and perform full C10 verification by default."""

    repo_root = Path(args.repo_root)
    contracts = load_stage_d_contracts(
        config_path=args.config or repo_root / "config/stage_d.json",
        run_spec_path=args.run_spec or repo_root / "config/stage_d_run_spec.json",
        config_schema_path=args.config_schema or repo_root / "config/stage_d.schema.json",
        run_spec_schema_path=(
            args.run_spec_schema or repo_root / "config/stage_d_run_spec.schema.json"
        ),
        repo_root=args.repo_root,
    )
    manifest = verify_stage_d_manifest(
        contracts.config, repo_root=args.repo_root, verify_c10=True,
    )
    if args.command == "validate-contracts":
        return StageDCLIPipeline(contracts, manifest)
    plan = build_stage_d_plan(
        contracts.config, contracts.run_spec, manifest,
    )
    plan = validate_stage_d_plan(plan)
    return StageDCLIPipeline(contracts, manifest, plan)


def _require_pipeline(
    value: Any, *, require_plan: bool,
) -> StageDCLIPipeline:
    if isinstance(value, StageDCLIPipeline):
        pipeline = value
    elif all(hasattr(value, name) for name in ("contracts", "manifest")):
        pipeline = StageDCLIPipeline(
            value.contracts, value.manifest, getattr(value, "plan", None),
        )
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ) and len(value) in (2, 3):
        pipeline = StageDCLIPipeline(
            value[0], value[1], value[2] if len(value) == 3 else None,
        )
    else:
        raise StageDCLIIntegrityError(
            "pipeline provider returned an invalid result"
        )
    if require_plan and pipeline.plan is None:
        raise StageDCLIIntegrityError(
            "pipeline provider omitted the validated Stage D plan"
        )
    return pipeline


def _field(value: Any, name: str, context: str) -> Any:
    if hasattr(value, name):
        return getattr(value, name)
    if isinstance(value, Mapping) and name in value:
        return value[name]
    raise StageDCLIIntegrityError(f"{context} does not expose {name}")


def _contract_summary(pipeline: StageDCLIPipeline) -> dict[str, Any]:
    contracts = pipeline.contracts
    config = _field(contracts, "config", "contracts")
    run_spec = _field(contracts, "run_spec", "contracts")
    manifest = pipeline.manifest
    bindings = _field(manifest, "bindings", "verified source manifest")
    if not isinstance(bindings, Mapping):
        raise StageDCLIIntegrityError(
            "verified source manifest bindings are not an object"
        )
    source_status = {
        stage: _field(binding, "verification_status", f"{stage} binding")
        for stage, binding in bindings.items()
    }
    try:
        registry_rows = _field(
            manifest, "registry_rows", "verified source manifest",
        )
    except StageDCLIIntegrityError:
        registry = _field(
            manifest, "registry_document", "verified source manifest",
        )
        registry_rows = _field(registry, "rows", "verified registry")
    return {
        "schema_version": 1,
        "document_type": "stage_d_contract_validation",
        "status": "PASS",
        "config": {
            "canonical_digest": _field(
                config, "canonical_digest", "Stage D config",
            ),
            "file_sha256": _field(config, "file_sha256", "Stage D config"),
            "schema_file_sha256": _field(
                config, "schema_file_sha256", "Stage D config",
            ),
        },
        "run_spec": {
            "canonical_digest": _field(
                run_spec, "canonical_digest", "Stage D run spec",
            ),
            "file_sha256": _field(
                run_spec, "file_sha256", "Stage D run spec",
            ),
            "schema_file_sha256": _field(
                run_spec, "schema_file_sha256", "Stage D run spec",
            ),
        },
        "verified_source_manifest_canonical_digest": _field(
            manifest, "canonical_digest", "verified source manifest",
        ),
        "source_verification": source_status,
        "registry_row_count": len(registry_rows),
        "execution_authorization": "NOT_ISSUED",
        "analysis_authorization": "NOT_EVALUATED",
        "stage_d_execution_started": False,
        "scientific_result": "NOT_EVALUATED",
    }


def _strict_bounded_json(path: Path, *, context: str, limit: int) -> Any:
    if not hasattr(os, "O_NOFOLLOW"):
        raise StageDCLIInputError(
            f"safe no-follow {context} loading is unavailable on this platform"
        )
    absolute = Path(os.path.abspath(os.fspath(path)))
    flags = os.O_RDONLY | os.O_NOFOLLOW
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NONBLOCK", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(absolute, flags)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise StageDCLIInputError(
                f"{context} must be a single-link regular file"
            )
        if before.st_size > limit:
            raise StageDCLIInputError(
                f"{context} JSON exceeds the safe size limit"
            )
        chunks: list[bytes] = []
        remaining = before.st_size + 1
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        after = os.fstat(descriptor)
        identity_before = (
            before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns,
            before.st_ctime_ns,
        )
        identity_after = (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if len(payload) != before.st_size or identity_after != identity_before:
            raise StageDCLIInputError(
                f"{context} JSON changed while it was being read"
            )
    except StageDCLIInputError:
        raise
    except OSError as error:
        raise StageDCLIInputError(
            f"cannot safely read {context} JSON: {absolute}"
        ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
    try:
        return strict_json_bytes(payload, f"Stage D {context}")
    except StageDConfigError as error:
        raise StageDCLIInputError(str(error)) from error


def _strict_profile_json(path: Path) -> Any:
    """Read one bounded regular file and parse strict profile JSON."""

    profiles = _strict_bounded_json(
        path, context="resource profiles", limit=_PROFILE_FILE_LIMIT,
    )
    if not isinstance(profiles, (dict, list)):
        raise StageDCLIInputError(
            "Stage D resource profiles must be a JSON object or array"
        )
    return profiles


def _strict_authorization_json(path: Path) -> Mapping[str, Any]:
    authorization = _strict_bounded_json(
        path, context="D7 execution authorization",
        limit=_AUTHORIZATION_FILE_LIMIT,
    )
    if not isinstance(authorization, Mapping):
        raise StageDCLIInputError(
            "Stage D D7 execution authorization must be a JSON object"
        )
    return authorization


def _strict_retry_authorizations(
    paths: Sequence[Path],
) -> tuple[Mapping[str, Any], ...]:
    if len(paths) > _MAX_RETRY_AUTHORIZATIONS:
        raise StageDCLIInputError(
            "too many Stage D coordinator retry authorizations"
        )
    documents: list[Mapping[str, Any]] = []
    for path in paths:
        document = _strict_bounded_json(
            path, context="coordinator retry authorization",
            limit=_RETRY_AUTHORIZATION_FILE_LIMIT,
        )
        if not isinstance(document, Mapping):
            raise StageDCLIInputError(
                "Stage D coordinator retry authorization must be a JSON object"
            )
        documents.append(document)
    return tuple(documents)


def _runtime_api() -> Any:
    """Import runtime code only when a D4 command actually needs it."""

    return importlib.import_module(".stage_d_runtime", __package__)


def _runtime_callable(injected: Callable[..., Any] | None, name: str) -> Callable[..., Any]:
    if injected is not None:
        return injected
    candidate = getattr(_runtime_api(), name, None)
    if not callable(candidate):
        raise StageDCLIRuntimeIntegrityError(
            f"Stage D runtime does not expose {name}"
        )
    return candidate


def _execution_gate(factory: Callable[..., Any] | None) -> Any:
    gate_factory = factory
    if gate_factory is None:
        gate_factory = getattr(_runtime_api(), "UnavailableD7ExecutionGate", None)
    if not callable(gate_factory):
        raise StageDCLIRuntimeIntegrityError(
            "Stage D runtime has no D7 execution gate implementation"
        )
    gate = gate_factory()
    validator = getattr(gate, "validate_execution_context", None)
    if not callable(validator) and not callable(gate):
        raise StageDCLIRuntimeIntegrityError(
            "D7 execution gate factory returned an invalid gate"
        )
    return gate


def _runtime_failure(error: BaseException) -> StageDCLIRuntimeBlockedError | StageDCLIRuntimeIntegrityError:
    """Map runtime failures to non-scientific, stable CLI messages."""

    kind = type(error).__name__
    if kind in {
        "StageDExecutionAuthorizationRequired",
        "StageDRuntimeBlockedError",
        "StageDConcurrentClaimError",
    }:
        return StageDCLIRuntimeBlockedError(
            "Stage D execution is blocked because its D7/runtime gate "
            "did not authorize a safe operation"
        )
    return StageDCLIRuntimeIntegrityError(
        "Stage D runtime validation failed closed before a safe control "
        "result was available"
    )


def _call_runtime(operation: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
    try:
        return operation(*args, **kwargs)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as error:
        raise _runtime_failure(error) from error


def _initialization_document(
    result: Any, *, args: argparse.Namespace, pipeline: StageDCLIPipeline,
) -> Mapping[str, Any]:
    if not isinstance(result, (str, os.PathLike)):
        raise StageDCLIRuntimeIntegrityError(
            "Stage D initializer returned an invalid control result"
        )
    initialized = Path(result)
    requested = Path(args.output_root)
    if initialized.absolute() != requested.absolute():
        raise StageDCLIRuntimeIntegrityError(
            "Stage D initializer returned a different output root"
        )
    document = {
        "schema_version": 1,
        "document_type": "stage_d_initialization_control",
        "status": "PASS",
        "control": {
            "operation": "init",
            "output_root": str(initialized),
            "external_execution_gate_validated": True,
            "analysis_authorization": "NOT_EVALUATED",
        },
        "digests": {
            "plan_digest": _field(
                pipeline.plan, "canonical_digest", "Stage D plan",
            ),
        },
        "counts": {},
    }
    return _validate_initialization_output(document)


def _run_runtime_command(
    args: argparse.Namespace, pipeline: StageDCLIPipeline,
    *, runtime_initializer: Callable[..., Any] | None,
    shard_runner: Callable[..., Any] | None,
    execution_gate_factory: Callable[..., Any] | None,
    backend_factory: Callable[..., Any] | None,
) -> Mapping[str, Any]:
    # D3 metadata is independently rebuilt before the D7 consumer is invoked.
    estimate, preflight = _estimate_and_preflight(pipeline, None)
    authorization = _strict_authorization_json(args.authorization)
    retry_authorizations = (
        _strict_retry_authorizations(args.retry_authorization)
        if args.command == "run-shard" else ()
    )
    gate = _execution_gate(execution_gate_factory)

    if args.command == "init":
        initializer = _runtime_callable(
            runtime_initializer, "initialize_stage_d_output_root",
        )
        result = _call_runtime(
            initializer, args.output_root, pipeline.plan, estimate, preflight,
            execution_context=authorization, execution_gate=gate,
        )
        return _initialization_document(result, args=args, pipeline=pipeline)

    runner = _runtime_callable(shard_runner, "run_stage_d_shard")
    result = _call_runtime(
        runner, pipeline.plan, output_root=args.output_root,
        estimate=estimate, preflight=preflight, shard_id=args.shard_index,
        execution_context=authorization, execution_gate=gate,
        backend_factory=backend_factory, device=args.device,
        retry_authorizations=retry_authorizations,
    )
    if not isinstance(result, Mapping) and not callable(
        getattr(result, "to_dict", None)
    ):
        raise StageDCLIRuntimeIntegrityError(
            "Stage D shard runner returned an invalid control result"
        )
    return _closed_shard_summary(result)


def _estimate_and_preflight(
    pipeline: StageDCLIPipeline, profile_path: Path | None,
) -> tuple[Any, Any]:
    profiles = None if profile_path is None else _strict_profile_json(profile_path)
    try:
        estimate = estimate_stage_d_run(
            _field(pipeline.contracts, "config", "contracts"),
            pipeline.plan, profiles,
        )
    except (StageDPlanError, TypeError) as error:
        if profile_path is None:
            raise
        raise StageDCLIInputError(
            f"invalid Stage D resource profiles: {error}"
        ) from error
    estimate = validate_stage_d_estimate(estimate)
    preflight = preflight_stage_d(estimate)
    preflight = validate_stage_d_preflight(preflight)
    return estimate, preflight


def _error_document(error: BaseException, code: str) -> dict[str, Any]:
    return {
        "error": {
            "code": code,
            "message": str(error),
            "type": type(error).__name__,
        }
    }


def main(
    argv: list[str] | None = None,
    *,
    pipeline_provider: Callable[[argparse.Namespace], Any] | None = None,
    preparation_input_provider: Callable[..., Any] | None = None,
    runtime_initializer: Callable[..., Any] | None = None,
    shard_runner: Callable[..., Any] | None = None,
    execution_gate_factory: Callable[..., Any] | None = None,
    backend_factory: Callable[..., Any] | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Run one Stage D command and return its stable exit code.

    The four D3 commands remain metadata-only.  Runtime dependencies are
    resolved only for D4 commands, and ``backend_factory`` is passed through
    uncalled so the runtime can validate authority and root controls first.
    """

    output = sys.stdout if stdout is None else stdout
    errors = sys.stderr if stderr is None else stderr
    try:
        try:
            args = build_parser().parse_args(argv)
        except SystemExit as error:
            return EXIT_SUCCESS if error.code == 0 else EXIT_INPUT_ERROR

        provider = _load_pipeline if pipeline_provider is None else pipeline_provider
        pipeline = _require_pipeline(
            provider(args), require_plan=args.command != "validate-contracts",
        )

        if args.command == "validate-contracts":
            _write_json(_contract_summary(pipeline), output)
            return EXIT_SUCCESS

        if args.command == "plan":
            _write_json(pipeline.plan, output)
            return EXIT_SUCCESS

        if args.command in _RUNTIME_COMMANDS:
            result = _run_runtime_command(
                args, pipeline, runtime_initializer=runtime_initializer,
                shard_runner=shard_runner,
                execution_gate_factory=execution_gate_factory,
                backend_factory=backend_factory,
            )
            document = _write_runtime_json(result, output)
            return (
                EXIT_PREFLIGHT_BLOCKED
                if document.get("status") == "BLOCKED"
                else EXIT_SUCCESS
            )

        estimate, preflight = _estimate_and_preflight(
            pipeline, getattr(args, "profiles", None),
        )
        blocked = _field(
            preflight, "status", "Stage D preflight",
        ) == "BLOCKED_NO_LAUNCH"

        if args.command == "estimate":
            _write_json({
                "schema_version": 1,
                "document_type": "stage_d_estimate_report",
                "status": "BLOCKED" if blocked else "PASS",
                "estimate": estimate,
                "preflight": preflight,
            }, output)
            return EXIT_PREFLIGHT_BLOCKED if blocked else EXIT_SUCCESS

        if args.command != "prepare-authorization":  # defensive boundary
            raise StageDCLIIntegrityError(
                f"unsupported Stage D command: {args.command}"
            )

        preparation_capture = (
            capture_stage_d_preparation_inputs
            if preparation_input_provider is None
            else preparation_input_provider
        )
        preparation_inputs = preparation_capture(repo_root=args.repo_root)
        authorization = build_stage_d_preparation_authorization(
            contracts=pipeline.contracts, manifest=pipeline.manifest,
            plan=pipeline.plan, budget=estimate, preflight=preflight,
            preparation_inputs=preparation_inputs,
        )
        authorization = validate_stage_d_preparation_authorization(
            authorization, contracts=pipeline.contracts,
            manifest=pipeline.manifest, plan=pipeline.plan, budget=estimate,
            preflight=preflight, preparation_inputs=preparation_inputs,
        )
        _write_json(authorization, output)
        return EXIT_PREFLIGHT_BLOCKED if blocked else EXIT_SUCCESS
    except (ArgumentError, StageDCLIInputError) as error:
        _write_json(_error_document(error, "STAGE_D_INPUT_ERROR"), errors)
        return EXIT_INPUT_ERROR
    except StageDCLIRuntimeBlockedError as error:
        _write_json(_error_document(error, "STAGE_D_EXECUTION_BLOCKED"), errors)
        return EXIT_PREFLIGHT_BLOCKED
    except StageDCLIRuntimeIntegrityError as error:
        _write_json(
            _error_document(error, "STAGE_D_RUNTIME_INTEGRITY_ERROR"), errors,
        )
        return EXIT_INTEGRITY_FAILURE
    except (
        StageDCLIIntegrityError, StageDConfigError, StageDManifestError,
        StageDPlanError, StageDAuthorizationError,
    ) as error:
        _write_json(_error_document(error, "STAGE_D_INTEGRITY_ERROR"), errors)
        return EXIT_INTEGRITY_FAILURE
    except Exception as error:  # fail closed at the CLI boundary
        _write_json(_error_document(error, "STAGE_D_INTERNAL_ERROR"), errors)
        return EXIT_INTEGRITY_FAILURE


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EXIT_INPUT_ERROR", "EXIT_INTEGRITY_FAILURE",
    "EXIT_PREFLIGHT_BLOCKED", "EXIT_SUCCESS", "ArgumentError",
    "StageDCLIInputError", "StageDCLIIntegrityError",
    "StageDCLIRuntimeBlockedError", "StageDCLIRuntimeIntegrityError",
    "StageDCLIPipeline", "build_parser", "main",
]
