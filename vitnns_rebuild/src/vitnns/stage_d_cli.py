"""Non-interactive JSON CLI for the Stage D control and runtime planes.

Importing the module performs no Git query, filesystem validation, runtime
import, model construction, image access, or scientific computation.  The D4
runtime commands consume a separately issued D7 authorization and fail closed
when its validator is unavailable; this module never issues that authority.
The D5 finalization commands independently rebuild the metadata controls, load
their finalization implementation only on demand, and expose only a closed
control summary rather than scientific result values.
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
_D4_RUNTIME_COMMANDS = frozenset({"init", "run-shard"})
_D5_FINALIZATION_COMMANDS = frozenset({"finalize", "validate"})
_D6_ANALYSIS_COMMANDS = frozenset({
    "aggregate", "build-paper-package", "validate-paper-package",
})
_D7_COMMANDS = frozenset({"smoke", "budget", "authorize-execution"})

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
_FINALIZATION_OUTPUT_KEYS = frozenset({
    "schema_version", "document_type", "operation", "status",
    "package_complete", "scientific_validity", "plan_digest",
    "final_status_sha256", "root_inventory",
})
_ROOT_INVENTORY_KEYS = frozenset({
    "file_count", "total_bytes", "canonical_digest",
})
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ArgumentError(ValueError):
    """Raised instead of allowing argparse to emit a non-JSON error."""


class StageDCLIInputError(ValueError):
    """Raised for malformed optional CLI inputs such as profile JSON."""


class StageDCLIIntegrityError(RuntimeError):
    """Raised when a live control-plane observation cannot be trusted."""


class StageDCLIFinalizationBlockedError(RuntimeError):
    """Raised when D5 cannot safely seal or validate the requested root."""


class StageDCLIRuntimeBlockedError(RuntimeError):
    """Raised when the external D7/runtime gate refuses execution."""


class StageDCLIRuntimeIntegrityError(RuntimeError):
    """Raised when D4 runtime validation fails closed."""


class StageDCLIAnalysisIntegrityError(RuntimeError):
    """Raised when D6 cannot produce or validate a safe package."""


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


def _add_finalization_arguments(parser: argparse.ArgumentParser) -> None:
    """Add D5 inputs without adding a D4 execution gate or backend."""

    _add_source_arguments(parser)
    parser.add_argument(
        "--output-root", type=Path, required=True,
        help="initialized Stage D output root to finalize or validate",
    )


def _add_analysis_arguments(
    parser: argparse.ArgumentParser, *, operation: str,
) -> None:
    """Add only synthetic D6 package paths; never execution controls."""

    _add_source_arguments(parser)
    if operation == "aggregate":
        parser.add_argument(
            "--output-root", type=Path, required=True,
            help="sealed synthetic D5 root; formal roots are rejected",
        )
        parser.add_argument(
            "--analysis-root", type=Path, required=True,
            help="fresh D6 synthetic analysis-package destination",
        )
    elif operation == "build-paper-package":
        parser.add_argument(
            "--analysis-root", type=Path, required=True,
            help="validated synthetic D6 analysis package",
        )
        parser.add_argument(
            "--paper-root", type=Path, required=True,
            help="fresh D6 synthetic paper-package destination",
        )
    elif operation == "validate-paper-package":
        parser.add_argument(
            "--paper-root", type=Path, required=True,
            help="synthetic D6 package to validate",
        )
    else:  # defensive boundary
        raise ValueError(f"unknown D6 operation: {operation}")


def build_parser() -> argparse.ArgumentParser:
    parser = _JSONArgumentParser(
        prog="vitnns-stage-d",
        description=(
            "Strict Stage D planning, authorized execution, and "
            "control-only finalization"
        ),
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

    finalize = commands.add_parser(
        "finalize",
        help="seal a complete Stage D root without exposing result values",
    )
    _add_finalization_arguments(finalize)

    validate_root = commands.add_parser(
        "validate",
        help="independently validate a sealed Stage D root",
    )
    _add_finalization_arguments(validate_root)

    aggregate = commands.add_parser(
        "aggregate",
        help="aggregate one sealed synthetic D5 root into a fresh D6 package",
    )
    _add_analysis_arguments(aggregate, operation="aggregate")

    build_paper = commands.add_parser(
        "build-paper-package",
        help="copy a validated synthetic D6 analysis package to fresh paper data",
    )
    _add_analysis_arguments(build_paper, operation="build-paper-package")

    validate_paper = commands.add_parser(
        "validate-paper-package",
        help="independently validate a closed synthetic D6 paper package",
    )
    _add_analysis_arguments(validate_paper, operation="validate-paper-package")

    smoke = commands.add_parser(
        "smoke", help="run the isolated non-reserve D7 synthetic smoke",
    )
    smoke.add_argument("--repo-root", type=Path, default=REPO_ROOT)

    budget = commands.add_parser(
        "budget", help="recompute the D7 final budget from sealed telemetry",
    )
    budget.add_argument("--repo-root", type=Path, default=REPO_ROOT)

    authorize = commands.add_parser(
        "authorize-execution",
        help="write D7 execution authority at a clean frozen producer",
    )
    authorize.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    authorize.add_argument("--shard-count", type=_nonnegative_int, default=8)
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


def _finalization_mapping(value: Any) -> Mapping[str, Any]:
    """Expose a finalizer result as a mapping without serializing its rows."""

    if isinstance(value, Mapping):
        return value
    converter = getattr(value, "to_dict", None)
    if not callable(converter):
        raise StageDCLIIntegrityError(
            "Stage D finalization returned an invalid control result"
        )
    try:
        document = converter()
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as error:
        raise StageDCLIIntegrityError(
            "Stage D finalization result could not be converted to control JSON"
        ) from error
    if not isinstance(document, Mapping):
        raise StageDCLIIntegrityError(
            "Stage D finalization result must convert to an object"
        )
    return document


def _finalization_field(
    document: Mapping[str, Any], *names: str, required: bool = True,
) -> Any:
    """Read one safe control field without reflecting its value in errors."""

    try:
        present = [name for name in names if name in document]
        if not present:
            if required:
                raise StageDCLIIntegrityError(
                    "Stage D finalization result omits a required control field"
                )
            return None
        values = [document[name] for name in present]
    except StageDCLIIntegrityError:
        raise
    except Exception as error:
        raise StageDCLIIntegrityError(
            "Stage D finalization control fields could not be read safely"
        ) from error
    first = values[0]
    try:
        conflicting = any(value != first for value in values[1:])
    except Exception as error:
        raise StageDCLIIntegrityError(
            "Stage D finalization control aliases could not be compared safely"
        ) from error
    if conflicting:
        raise StageDCLIIntegrityError(
            "Stage D finalization result has conflicting control fields"
        )
    return first


def _finalization_document(
    value: Any, *, operation: str, pipeline: StageDCLIPipeline,
) -> dict[str, Any]:
    """Project a rich D5 validation result onto the closed CLI surface."""

    source = _finalization_mapping(value)
    schema_version = _finalization_field(
        source, "schema_version", required=False,
    )
    document_type = _finalization_field(
        source, "document_type", required=False,
    )
    reported_operation = _finalization_field(
        source, "operation", required=False,
    )
    status = _finalization_field(source, "status", "workflow_status")
    package_complete = _finalization_field(source, "package_complete")
    scientific_validity = _finalization_field(source, "scientific_validity")
    plan_digest = _finalization_field(source, "plan_digest")
    final_status_sha256 = _finalization_field(
        source, "final_status_sha256",
    )
    root_inventory = _finalization_field(source, "root_inventory")

    if (
        (
            schema_version is not None
            and (type(schema_version) is not int or schema_version != 1)
        )
        or (
            document_type is not None
            and (
                not isinstance(document_type, str)
                or document_type != "stage_d_finalization_control"
            )
        )
        or (
            reported_operation is not None
            and (
                not isinstance(reported_operation, str)
                or reported_operation != operation
            )
        )
        or operation not in _D5_FINALIZATION_COMMANDS
        or not isinstance(status, str)
        or status != "PASS"
        or package_complete is not True
        or not isinstance(scientific_validity, str)
        or scientific_validity not in {"PASS", "FAIL"}
        or not isinstance(plan_digest, str)
        or _SHA256_RE.fullmatch(plan_digest) is None
        or plan_digest
        != _field(pipeline.plan, "canonical_digest", "Stage D plan")
        or not isinstance(final_status_sha256, str)
        or _SHA256_RE.fullmatch(final_status_sha256) is None
        or not isinstance(root_inventory, Mapping)
    ):
        raise StageDCLIIntegrityError(
            "Stage D finalization output contains invalid control values"
        )
    try:
        inventory_keys = set(root_inventory)
        file_count = root_inventory["file_count"]
        total_bytes = root_inventory["total_bytes"]
        inventory_digest = root_inventory["canonical_digest"]
    except Exception as error:
        raise StageDCLIIntegrityError(
            "Stage D finalization root inventory is not a safe control object"
        ) from error
    if (
        inventory_keys != _ROOT_INVENTORY_KEYS
        or type(file_count) is not int
        or file_count < 0
        or type(total_bytes) is not int
        or total_bytes < 0
        or not isinstance(inventory_digest, str)
        or _SHA256_RE.fullmatch(inventory_digest) is None
    ):
        raise StageDCLIIntegrityError(
            "Stage D finalization root inventory contains invalid controls"
        )

    document = {
        "schema_version": 1,
        "document_type": "stage_d_finalization_control",
        "operation": operation,
        "status": status,
        "package_complete": package_complete,
        "scientific_validity": scientific_validity,
        "plan_digest": plan_digest,
        "final_status_sha256": final_status_sha256,
        "root_inventory": {
            "file_count": file_count,
            "total_bytes": total_bytes,
            "canonical_digest": inventory_digest,
        },
    }
    if set(document) != _FINALIZATION_OUTPUT_KEYS:
        raise StageDCLIIntegrityError(
            "Stage D finalization output does not match its closed schema"
        )
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


def _production_backend_factory(*, context: Any, device: str) -> Any:
    """Build the recovery-v2 Stage D backend after the D7 gate passes."""

    d7 = importlib.import_module(".stage_d_d7", __package__)
    return d7.production_backend_factory(context=context, device=device)


def _finalization_api() -> Any:
    """Import finalization code only when a D5 command actually needs it."""

    try:
        return importlib.import_module(".stage_d_finalize", __package__)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as error:
        raise StageDCLIIntegrityError(
            "Stage D finalization support could not be loaded safely"
        ) from error


def _analysis_api() -> tuple[Any, Any, Any]:
    """Load D6-only modules lazily, after D3 controls have been rebuilt."""

    try:
        return (
            importlib.import_module(".stage_d_analysis_input", __package__),
            importlib.import_module(".stage_d_analysis", __package__),
            importlib.import_module(".stage_d_paper", __package__),
        )
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as error:
        raise StageDCLIAnalysisIntegrityError(
            "Stage D D6 analysis support could not be loaded safely"
        ) from error


def _runtime_callable(injected: Callable[..., Any] | None, name: str) -> Callable[..., Any]:
    if injected is not None:
        return injected
    candidate = getattr(_runtime_api(), name, None)
    if not callable(candidate):
        raise StageDCLIRuntimeIntegrityError(
            f"Stage D runtime does not expose {name}"
        )
    return candidate


def _finalization_callable(
    injected: Callable[..., Any] | None, name: str,
) -> Callable[..., Any]:
    if injected is not None:
        if not callable(injected):
            raise StageDCLIIntegrityError(
                "injected Stage D finalization operation is not callable"
            )
        return injected
    candidate = getattr(_finalization_api(), name, None)
    if not callable(candidate):
        raise StageDCLIIntegrityError(
            f"Stage D finalization support does not expose {name}"
        )
    return candidate


def _finalization_exception_types(
) -> tuple[type[Exception], type[Exception], type[Exception]]:
    """Resolve and validate the canonical D5 exception hierarchy lazily."""

    api = _finalization_api()
    names = (
        "StageDFinalizationError",
        "StageDFinalizationBlockedError",
        "StageDFinalizationIntegrityError",
    )
    candidates = tuple(getattr(api, name, None) for name in names)
    if any(
        not isinstance(candidate, type)
        or not issubclass(candidate, Exception)
        for candidate in candidates
    ):
        raise StageDCLIIntegrityError(
            "Stage D finalization support has an invalid exception contract"
        )
    base, blocked, integrity = candidates
    if (
        base in {Exception, BaseException}
        or blocked is base
        or integrity is base
        or blocked is integrity
        or not issubclass(blocked, base)
        or not issubclass(integrity, base)
    ):
        raise StageDCLIIntegrityError(
            "Stage D finalization support has an invalid exception hierarchy"
        )
    return base, blocked, integrity


def _execution_gate(
    factory: Callable[..., Any] | None, *, allow_d7_default: bool,
) -> Any:
    gate_factory = factory
    if gate_factory is None and allow_d7_default:
        try:
            gate_factory = getattr(
                importlib.import_module(".stage_d_d7", __package__),
                "D7ExecutionGate", None,
            )
        except Exception:
            gate_factory = None
    if gate_factory is None:
        gate_factory = getattr(
            _runtime_api(), "UnavailableD7ExecutionGate", None
        )
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


def _finalization_failure(
    error: BaseException,
) -> StageDCLIFinalizationBlockedError | StageDCLIIntegrityError:
    """Map D5 failures without copying potentially scientific messages."""

    base_type, blocked_type, integrity_type = _finalization_exception_types()
    if isinstance(error, blocked_type):
        return StageDCLIFinalizationBlockedError(
            "Stage D finalization is blocked because the output root is "
            "not safely sealable"
        )
    # Canonical integrity/base failures and unexpected injected failures all
    # share one fail-closed public surface.  Keep the explicit checks here so
    # a same-named foreign exception can never acquire D5 blocked semantics.
    if isinstance(error, (integrity_type, base_type)):
        return StageDCLIIntegrityError(
            "Stage D finalization validation failed closed before a safe "
            "control result was available"
        )
    return StageDCLIIntegrityError(
        "Stage D finalization validation failed closed before a safe "
        "control result was available"
    )


def _call_finalization(
    operation: Callable[..., Any], /, *args: Any, **kwargs: Any,
) -> Any:
    try:
        return operation(*args, **kwargs)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as error:
        raise _finalization_failure(error) from error


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
    pipeline_provider_is_default: bool,
) -> Mapping[str, Any]:
    # D3 metadata is independently rebuilt before the D7 consumer is invoked.
    authorization = _strict_authorization_json(args.authorization)
    if execution_gate_factory is None and pipeline_provider_is_default:
        d7 = importlib.import_module(".stage_d_d7", __package__)
        pipeline = d7.load_recovery_pipeline(repo_root=args.repo_root)
        estimate, preflight = d7.controls_from_authorization(
            authorization, repo_root=args.repo_root,
        )
    else:
        estimate, preflight = _estimate_and_preflight(pipeline, None)
    retry_authorizations = (
        _strict_retry_authorizations(args.retry_authorization)
        if args.command == "run-shard" else ()
    )
    gate = _execution_gate(
        execution_gate_factory, allow_d7_default=pipeline_provider_is_default,
    )

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
    selected_backend_factory = (
        _production_backend_factory if backend_factory is None else backend_factory
    )
    result = _call_runtime(
        runner, pipeline.plan, output_root=args.output_root,
        estimate=estimate, preflight=preflight, shard_id=args.shard_index,
        execution_context=authorization, execution_gate=gate,
        backend_factory=selected_backend_factory, device=args.device,
        retry_authorizations=retry_authorizations,
    )
    if not isinstance(result, Mapping) and not callable(
        getattr(result, "to_dict", None)
    ):
        raise StageDCLIRuntimeIntegrityError(
            "Stage D shard runner returned an invalid control result"
        )
    return _closed_shard_summary(result)


def _run_finalization_command(
    args: argparse.Namespace, pipeline: StageDCLIPipeline,
    *, finalizer: Callable[..., Any] | None,
    root_validator: Callable[..., Any] | None,
) -> Mapping[str, Any]:
    """Run D5 from rebuilt metadata controls, without a gate or backend."""

    estimate, preflight = _estimate_and_preflight(pipeline, None)
    if args.command == "finalize":
        operation = _finalization_callable(
            finalizer, "finalize_stage_d_root",
        )
    elif args.command == "validate":
        operation = _finalization_callable(
            root_validator, "validate_stage_d_root",
        )
    else:  # defensive boundary
        raise StageDCLIIntegrityError(
            "unsupported Stage D finalization operation"
        )
    result = _call_finalization(
        operation, args.output_root, pipeline.plan, estimate, preflight,
    )
    return _finalization_document(
        result, operation=args.command, pipeline=pipeline,
    )


def _d6_control_document(value: Any, *, operation: str) -> dict[str, Any]:
    """Validate the deliberately non-numeric, non-identifying D6 CLI surface."""

    source = _json_value(value, "D6 control summary")
    expected = {
        "schema_version", "document_type", "status", "synthetic_only",
        "formal_result", "analysis_canonical_digest", "coordinate_count",
        "execution_authorization", "analysis_authorization", "stage_d_execution_started",
    }
    if (
        set(source) != expected
        or source["schema_version"] != 1
        or source["document_type"] != "stage_d_d6_control_summary"
        or source["status"] != "PASS"
        or source["synthetic_only"] is not True
        or source["formal_result"] is not False
        or not isinstance(source["analysis_canonical_digest"], str)
        or _SHA256_RE.fullmatch(source["analysis_canonical_digest"]) is None
        or source["coordinate_count"] != 500
        or source["execution_authorization"] != "NOT_ISSUED"
        or source["analysis_authorization"] != "NOT_EVALUATED"
        or source["stage_d_execution_started"] is not False
    ):
        raise StageDCLIAnalysisIntegrityError(
            "Stage D D6 operation did not return a safe closed control summary"
        )
    return source


def _run_analysis_command(
    args: argparse.Namespace, pipeline: StageDCLIPipeline,
) -> Mapping[str, Any]:
    """Run D6 on synthetic-only inputs and project no numerical values."""

    try:
        input_api, analysis_api, paper_api = _analysis_api()
        if args.command == "aggregate":
            estimate, preflight = _estimate_and_preflight(pipeline, None)
            receipt = input_api.validate_stage_d_analysis_input(
                args.output_root, pipeline.plan, estimate, preflight,
            )
            result = analysis_api.aggregate_stage_d_analysis(receipt, pipeline.plan)
            paper_api.build_stage_d_analysis_package(
                result, args.analysis_root, repo_root=args.repo_root,
            )
            return analysis_api.closed_analysis_summary(result)
        if args.command == "build-paper-package":
            paper_api.validate_stage_d_analysis_package(
                args.analysis_root, repo_root=args.repo_root,
            )
            paper_api.build_stage_d_paper_package(
                args.analysis_root, args.paper_root, repo_root=args.repo_root,
            )
            analysis = paper_api.analysis_control_from_package(
                args.paper_root, repo_root=args.repo_root,
            )
            return analysis_api.closed_analysis_summary(analysis)
        if args.command == "validate-paper-package":
            paper_api.validate_stage_d_paper_package(
                args.paper_root, repo_root=args.repo_root,
            )
            analysis = paper_api.analysis_control_from_package(
                args.paper_root, repo_root=args.repo_root,
            )
            return analysis_api.closed_analysis_summary(analysis)
        raise StageDCLIAnalysisIntegrityError("unsupported Stage D D6 operation")
    except (KeyboardInterrupt, SystemExit, StageDCLIAnalysisIntegrityError):
        raise
    except Exception as error:
        raise StageDCLIAnalysisIntegrityError(
            "Stage D D6 synthetic analysis or package validation failed closed"
        ) from error


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
    finalizer: Callable[..., Any] | None = None,
    root_validator: Callable[..., Any] | None = None,
    execution_gate_factory: Callable[..., Any] | None = None,
    backend_factory: Callable[..., Any] | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Run one Stage D command and return its stable exit code.

    The four D3 commands remain metadata-only.  Runtime dependencies are
    resolved only for D4 commands, and ``backend_factory`` is passed through
    uncalled so the runtime can validate authority and root controls first.
    D5 finalization is also loaded on demand; it receives freshly rebuilt
    metadata controls but no authorization document, execution gate, backend,
    profile, device, or scientific result value.
    """

    output = sys.stdout if stdout is None else stdout
    errors = sys.stderr if stderr is None else stderr
    try:
        try:
            args = build_parser().parse_args(argv)
        except SystemExit as error:
            return EXIT_SUCCESS if error.code == 0 else EXIT_INPUT_ERROR

        if args.command in _D7_COMMANDS:
            d7 = importlib.import_module(".stage_d_d7", __package__)
            if args.command == "smoke":
                document = d7.run_full_chain_synthetic_smoke(
                    repo_root=args.repo_root
                )
            elif args.command == "budget":
                document = d7.write_d7_budget(repo_root=args.repo_root)
            else:
                document = d7.build_execution_authorization(
                    repo_root=args.repo_root, shard_count=args.shard_count,
                )
                d7._write_atomic(
                    Path(args.repo_root).resolve() / d7.AUTHORIZATION_PATH, document,
                )
            _write_json(document, output)
            return EXIT_SUCCESS

        provider = _load_pipeline if pipeline_provider is None else pipeline_provider
        if args.command in _D4_RUNTIME_COMMANDS and pipeline_provider is None:
            d7 = importlib.import_module(".stage_d_d7", __package__)
            pipeline = d7.load_recovery_pipeline(repo_root=args.repo_root)
        else:
            pipeline = _require_pipeline(
                provider(args), require_plan=args.command != "validate-contracts",
            )

        if args.command == "validate-contracts":
            _write_json(_contract_summary(pipeline), output)
            return EXIT_SUCCESS

        if args.command == "plan":
            _write_json(pipeline.plan, output)
            return EXIT_SUCCESS

        if args.command in _D4_RUNTIME_COMMANDS:
            result = _run_runtime_command(
                args, pipeline, runtime_initializer=runtime_initializer,
                shard_runner=shard_runner,
                execution_gate_factory=execution_gate_factory,
                backend_factory=backend_factory,
                pipeline_provider_is_default=pipeline_provider is None,
            )
            document = _write_runtime_json(result, output)
            return (
                EXIT_PREFLIGHT_BLOCKED
                if document.get("status") == "BLOCKED"
                else EXIT_SUCCESS
            )

        if args.command in _D5_FINALIZATION_COMMANDS:
            document = _run_finalization_command(
                args, pipeline, finalizer=finalizer,
                root_validator=root_validator,
            )
            _write_json(document, output)
            return EXIT_SUCCESS

        if args.command in _D6_ANALYSIS_COMMANDS:
            document = _d6_control_document(
                _run_analysis_command(args, pipeline), operation=args.command,
            )
            _write_json(document, output)
            return EXIT_SUCCESS

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
    except StageDCLIFinalizationBlockedError as error:
        _write_json(
            _error_document(error, "STAGE_D_FINALIZATION_BLOCKED"), errors,
        )
        return EXIT_PREFLIGHT_BLOCKED
    except StageDCLIRuntimeBlockedError as error:
        _write_json(_error_document(error, "STAGE_D_EXECUTION_BLOCKED"), errors)
        return EXIT_PREFLIGHT_BLOCKED
    except StageDCLIRuntimeIntegrityError as error:
        _write_json(
            _error_document(error, "STAGE_D_RUNTIME_INTEGRITY_ERROR"), errors,
        )
        return EXIT_INTEGRITY_FAILURE
    except StageDCLIAnalysisIntegrityError as error:
        _write_json(
            _error_document(error, "STAGE_D_INTEGRITY_ERROR"), errors,
        )
        return EXIT_INTEGRITY_FAILURE
    except (
        StageDCLIIntegrityError, StageDConfigError, StageDManifestError,
        StageDPlanError, StageDAuthorizationError,
    ) as error:
        _write_json(_error_document(error, "STAGE_D_INTEGRITY_ERROR"), errors)
        return EXIT_INTEGRITY_FAILURE
    except ValueError as error:
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
    "StageDCLIFinalizationBlockedError",
    "StageDCLIRuntimeBlockedError", "StageDCLIRuntimeIntegrityError",
    "StageDCLIAnalysisIntegrityError",
    "StageDCLIPipeline", "build_parser", "main",
]
