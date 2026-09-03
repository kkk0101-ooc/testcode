"""Non-interactive JSON CLI for Stage C planning and orchestration."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable, TextIO

from .constants import DEFAULT_STATE_DICT, MODEL_SHA256, REPO_ROOT
from .io_utils import sha256_file
from .runner import seed_everything
from .stage_c_config import DEFAULT_STAGE_C_SCHEMA, load_stage_c_config, strict_json_load
from .stage_c_provenance import capture_git_state
from . import stage_c_c8 as _stage_c_c8_runtime
from .stage_c_benchmark import (
    DIAGNOSTIC_GATE_FILENAME,
    REPORT_FILENAME,
    StageCBenchmarkBlockedError,
    StageCBenchmarkIntegrityError,
    capture_benchmark_environment,
    require_fresh_c7_output_root,
    require_c7_launch_gate,
    validate_c7_diagnostic_plan,
    validate_c7_execution_plan,
    write_benchmark_report,
    write_c7_diagnostic_gate,
    write_c7_launch_authorization,
)
from .recovery_c7_launch import (
    RecoveryC7LaunchError,
    capture_authorized_c7_source_state,
    load_recovery_c7_launch_record,
)
from .recovery_c8_launch import (
    RecoveryC8LaunchError,
    capture_authorized_c8_source_state,
    load_recovery_c8_launch_record,
    recovery_provenance_context as recovery_c8_provenance_context,
)
from .recovery_c8_salvage import (
    RecoveryC8SalvageError,
    capture_authorized_c8_salvage_source_state,
    load_recovery_c8_salvage_launch_record,
)
from .recovery_c8_salvage_repair import (
    RecoveryC8SalvageRepairError,
    capture_authorized_c8_salvage_repair_source_state,
    load_recovery_c8_salvage_repair_launch_record,
)
from .recovery_c9_launch import (
    RecoveryC9LaunchError, load_recovery_c9_launch_record,
    load_salvage_selection_state,
)
from .recovery_c10_launch import (
    RecoveryC10LaunchError, load_recovery_c10_launch_record,
)
from .stage_c_c8_salvage import (
    finalize_c8_salvage, initialize_c8_salvage_gap_root,
    run_c8_salvage_gap_shard, validate_c8_salvage_from_source,
    validate_c8_salvage_completion_report,
    write_c8_salvage_completion_report,
)
from .stage_c_c8_salvage_selection import (
    build_c8_salvage_selection, write_c8_salvage_selection_outputs,
)
from .stage_c_c8_salvage_repair import (
    finalize_c8_salvage_repair, initialize_c8_salvage_repair_root,
    run_c8_salvage_repair_shard, validate_c8_salvage_repair_from_source,
    validate_c8_salvage_repair_completion_report,
    write_c8_salvage_repair_completion_report,
)
from .stage_c_c8_salvage_repair_selection import (
    build_c8_salvage_repair_selection,
    validate_c8_salvage_repair_selection_report,
    write_c8_salvage_repair_selection_outputs,
)
from .stage_c_plan import (
    DEFAULT_ACCEPTANCE,
    DEFAULT_DEPENDENCY_LOCK,
    StageCEstimate,
    build_execution_plan,
    estimate_stage_c_run,
    load_run_spec,
    load_stage_c_registry,
    load_stage_c_selection,
)
from .stage_c_c8 import (
    AUTHORIZATION_FILENAME as C8_AUTHORIZATION_FILENAME,
    build_c8_launch_authorization,
    build_c8_measured_preflight,
    finalize_c8_output_root,
    initialize_c8_output_root,
    run_c8_shard,
    validate_c8_execution_plan,
    validate_c8_launch_authorization,
    validate_fixed_c7_report,
)
from .stage_c_c9 import (
    build_c9_launch_authorization, build_recovery_c9_launch_authorization,
    finalize_c9_output_root,
    initialize_c9_output_root,
    load_c9_initialized_state,
    run_c9_shard,
    validate_c9_execution_plan,
    validate_c9_output_root,
)
from .stage_c_c10 import (
    StageC10BlockedError, StageC10IntegrityError,
    build_c10_output_root, validate_c10_output_root,
)
from .stage_c_selection import (
    StageCSelectionBlockedError,
    StageCSelectionIntegrityError,
    build_c8_calibration_evidence,
    build_stage_c_selection,
    validate_stage_c_selection_from_source,
    write_stage_c_selection_outputs,
)
from .stage_c_runner import (
    ProductionStageCBackend,
    StageCBatchValidationError,
    aggregate_stage_c_batch,
    run_stage_c_batch,
    run_stage_c_sample,
    validate_stage_c_batch,
)


EXIT_SUCCESS = 0
EXIT_INPUT_ERROR = 1
EXIT_PREFLIGHT_BLOCKED = 2
EXIT_TERMINAL_FAILURE = 3
EXIT_INTEGRITY_FAILURE = 4
STAGE_C_SEED = 20260821
_TRACKED_STAGE_C_SELECTION = REPO_ROOT / "selection/stage_c_selection.json"
_C8_COMMANDS = frozenset({
    "c8-init", "c8-run-shard", "c8-finalize", "c8-select",
})
_C8_SALVAGE_COMMANDS = frozenset({
    "c8-salvage-init", "c8-salvage-run-shard",
    "c8-salvage-finalize", "c8-salvage-validate",
    "c8-salvage-select", "c8-salvage-complete",
})
_C8_SALVAGE_REPAIR_COMMANDS = frozenset({
    "c8-salvage-repair-init", "c8-salvage-repair-run-shard",
    "c8-salvage-repair-finalize", "c8-salvage-repair-validate",
    "c8-salvage-repair-select", "c8-salvage-repair-complete",
})
_C9_COMMANDS = frozenset({
    "c9-init", "c9-run-shard", "c9-finalize", "c9-validate",
})
_C10_COMMANDS = frozenset({"c10-build", "c10-validate"})
class ArgumentError(ValueError):
    """Raised instead of allowing argparse to emit non-JSON errors."""


class _JSONArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise ArgumentError(message)


def _add_plan_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-spec", type=Path, required=True)
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--data-root", type=Path)


def _add_c8_plan_arguments(parser: argparse.ArgumentParser) -> None:
    """Add calibration-only planning inputs; a prior selection is forbidden."""

    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-spec", type=Path, required=True)
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--data-root", type=Path)


def _add_c9_plan_arguments(parser: argparse.ArgumentParser) -> None:
    """Add formal C9 inputs without a caller-overridable selection."""

    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-spec", type=Path, required=True)
    parser.add_argument(
        "--recovery-c9-launch-record", type=Path,
        help="validated recovery-v2 C9 authority bound to salvage C8",
    )


def _add_c10_plan_arguments(parser: argparse.ArgumentParser) -> None:
    """Add recovery C10 inputs without execution or selection overrides."""

    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-spec", type=Path, required=True)
    parser.add_argument(
        "--recovery-c10-launch-record", type=Path, required=True,
        help="validated recovery-v2 C10-only launch authority",
    )


def _add_recovery_c7_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--recovery-c7-launch-record", type=Path,
        help=(
            "validated recovery-v2 launch authority; required for the exact "
            "C7 diagnostic and formal scopes"
        ),
    )


def _add_recovery_c8_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--recovery-c8-launch-record", type=Path,
        help="validated recovery-v2 C8-only launch authority",
    )


def _add_c8_salvage_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--salvage-authority", type=Path, required=True,
        help="validated amendment-authorized recovery C8 salvage authority",
    )


def _add_c8_salvage_repair_argument(
    parser: argparse.ArgumentParser,
) -> None:
    parser.add_argument(
        "--repair-authority", type=Path, required=True,
        help="validated amendment-authorized recovery C8 repair authority",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = _JSONArgumentParser(
        prog="vitnns-stage-c",
        description="Auditable Stage C planning, execution, validation, and L0 aggregation",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    estimate = commands.add_parser("estimate")
    _add_plan_arguments(estimate)

    for name in ("run-sample", "run-batch"):
        command = commands.add_parser(name)
        _add_plan_arguments(command)
        _add_recovery_c7_argument(command)
        command.add_argument("--output-root", type=Path, required=True)
        command.add_argument("--device", default="cpu")
        command.add_argument("--state-dict", type=Path)
        command.add_argument(
            "--no-resume", action="store_true",
            help="always allocate a new attempt, preserving every old attempt",
        )
        command.add_argument(
            "--c7-diagnostic-gate", type=Path,
            help=(
                "required for the exact formal C7 plan; must name a passing "
                "report from the separate one-unit diagnostic workload"
            ),
        )
        if name == "run-sample":
            command.add_argument("--sample-id", required=True)

    benchmark = commands.add_parser("benchmark")
    _add_plan_arguments(benchmark)
    _add_recovery_c7_argument(benchmark)
    benchmark.add_argument("--output-root", type=Path, required=True)
    benchmark.add_argument("--device", default="cpu")
    benchmark.add_argument("--state-dict", type=Path)
    benchmark.add_argument("--c7-diagnostic-gate", type=Path, required=True)

    for name in ("validate", "aggregate", "c7-diagnostic-gate", "benchmark-report"):
        command = commands.add_parser(name)
        _add_plan_arguments(command)
        _add_recovery_c7_argument(command)
        command.add_argument("--output-root", type=Path, required=True)
        if name in {"aggregate", "c7-diagnostic-gate", "benchmark-report"}:
            command.add_argument("--output", type=Path)
        if name in {"c7-diagnostic-gate", "benchmark-report"}:
            command.add_argument("--device", default="cuda:0")
        if name == "benchmark-report":
            command.add_argument("--c7-diagnostic-gate", type=Path, required=True)

    c8_init = commands.add_parser("c8-init")
    _add_c8_plan_arguments(c8_init)
    _add_recovery_c8_argument(c8_init)
    c8_init.add_argument("--output-root", type=Path, required=True)
    c8_init.add_argument("--c7-report", type=Path, required=True)

    c8_shard = commands.add_parser("c8-run-shard")
    _add_c8_plan_arguments(c8_shard)
    _add_recovery_c8_argument(c8_shard)
    c8_shard.add_argument("--output-root", type=Path, required=True)
    c8_shard.add_argument("--device", default="cpu")
    c8_shard.add_argument("--state-dict", type=Path)
    c8_shard.add_argument(
        "--shard-index", type=int, required=True,
        help="zero-based index of this deterministic plan-order shard",
    )
    c8_shard.add_argument(
        "--shard-count", type=int, required=True,
        help="total number of deterministic plan-order shards",
    )

    c8_finalize = commands.add_parser("c8-finalize")
    _add_c8_plan_arguments(c8_finalize)
    _add_recovery_c8_argument(c8_finalize)
    c8_finalize.add_argument("--output-root", type=Path, required=True)

    c8_select = commands.add_parser("c8-select")
    _add_c8_plan_arguments(c8_select)
    _add_recovery_c8_argument(c8_select)
    c8_select.add_argument("--output-root", type=Path, required=True)
    c8_select.add_argument("--mode", choices=("build", "validate"), default="build")
    c8_select.add_argument("--report-output", type=Path, required=True)
    c8_select.add_argument("--selection-output", type=Path)

    for name in (
        "c8-salvage-init", "c8-salvage-finalize",
        "c8-salvage-validate", "c8-salvage-select",
        "c8-salvage-complete",
    ):
        command = commands.add_parser(name)
        _add_c8_plan_arguments(command)
        _add_c8_salvage_argument(command)
        if name in {"c8-salvage-select", "c8-salvage-complete"}:
            command.add_argument(
                "--mode", choices=("build", "validate"), default="build"
            )
    salvage_shard = commands.add_parser("c8-salvage-run-shard")
    _add_c8_plan_arguments(salvage_shard)
    _add_c8_salvage_argument(salvage_shard)
    salvage_shard.add_argument("--device", default="cuda:0")
    salvage_shard.add_argument("--shard-index", type=int, required=True)
    salvage_shard.add_argument("--shard-count", type=int, required=True)

    for name in (
        "c8-salvage-repair-init", "c8-salvage-repair-finalize",
        "c8-salvage-repair-validate", "c8-salvage-repair-select",
        "c8-salvage-repair-complete",
    ):
        command = commands.add_parser(name)
        _add_c8_plan_arguments(command)
        _add_c8_salvage_repair_argument(command)
        if name in {
            "c8-salvage-repair-select",
            "c8-salvage-repair-complete",
        }:
            command.add_argument(
                "--mode", choices=("build", "validate"), default="build"
            )
    repair_shard = commands.add_parser("c8-salvage-repair-run-shard")
    _add_c8_plan_arguments(repair_shard)
    _add_c8_salvage_repair_argument(repair_shard)
    repair_shard.add_argument("--device", default="cuda:0")
    repair_shard.add_argument("--shard-index", type=int, required=True)
    repair_shard.add_argument("--shard-count", type=int, required=True)

    c9_init = commands.add_parser("c9-init")
    _add_c9_plan_arguments(c9_init)
    c9_init.add_argument("--output-root", type=Path, required=True)
    c9_init.add_argument("--c8-root", type=Path)
    c9_init.add_argument("--c8-report", type=Path)
    c9_init.add_argument("--c8-recovery-selection", type=Path)

    c9_shard = commands.add_parser("c9-run-shard")
    _add_c9_plan_arguments(c9_shard)
    c9_shard.add_argument("--output-root", type=Path, required=True)
    c9_shard.add_argument("--device", default="cpu")
    c9_shard.add_argument(
        "--shard-index", type=int, required=True,
        help="zero-based index of the core-owned round-robin shard",
    )
    c9_shard.add_argument(
        "--shard-count", type=int, required=True,
        help="total number of core-owned round-robin shards",
    )

    for name in ("c9-finalize", "c9-validate"):
        command = commands.add_parser(name)
        _add_c9_plan_arguments(command)
        command.add_argument("--output-root", type=Path, required=True)

    for name in ("c10-build", "c10-validate"):
        command = commands.add_parser(name)
        _add_c10_plan_arguments(command)
        command.add_argument("--c9-root", type=Path, required=True)
        command.add_argument("--output-root", type=Path, required=True)
    return parser


def _write_json(value: Any, stream: TextIO) -> None:
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        value = to_dict()
    print(
        json.dumps(
            value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
        ),
        file=stream,
    )


def _load_pipeline(
    args: argparse.Namespace, *, c8: bool = False, c9: bool = False,
    recovery_record: Any | None = None, recovery_c8_record: Any | None = None,
    recovery_c9_record: Any | None = None,
) -> tuple[Any, Any, Any, Any, Any, Any]:
    if c8 and c9:
        raise ValueError("a pipeline cannot be both C8 and C9")
    config = load_stage_c_config(args.config)
    registry_kwargs: dict[str, Any] = {}
    if getattr(args, "registry", None) is not None:
        registry_kwargs["registry_path"] = args.registry
    if getattr(args, "data_root", None) is not None:
        registry_kwargs["data_root"] = args.data_root
    recovery_authority = (
        recovery_c9_record or recovery_c8_record or recovery_record
    )
    if recovery_authority is not None:
        registry_kwargs["recovery_binding"] = recovery_authority.binding
    if c8 or c9:
        # Planning must never decode or hash the 15 holdout images.  The
        # production backend later reads only a UnitSample already present in
        # the exact calibration plan.
        registry_kwargs["verify_live_assets"] = False
    registry = load_stage_c_registry(config, **registry_kwargs)
    if c8 or recovery_record is not None:
        # Recovery C7 is a pre-C8 diagnostic/formal replay.  It must not
        # consume the repository's historical C8 selection, even when that
        # file exists at the configured default path.
        selection = None
    elif c9 and recovery_c9_record is not None:
        selection = load_salvage_selection_state(recovery_c9_record)
    elif c9:
        # Formal C9 always consumes the repository-tracked canonical
        # selection.  Its parser deliberately has no --selection escape hatch.
        selection = load_stage_c_selection(
            config, selection_path=_TRACKED_STAGE_C_SELECTION, required=True,
            config_path=args.config,
        )
    else:
        selection = load_stage_c_selection(
            config, selection_path=args.selection, required=False
        )
    run_spec = load_run_spec(args.run_spec)
    plan = build_execution_plan(
        config, registry, run_spec, selection=selection
    )
    estimate = estimate_stage_c_run(config, plan)
    return config, registry, selection, run_spec, plan, estimate


def _require_exact_c8_plan(plan: Any, config: Any) -> dict[str, Any]:
    try:
        result = validate_c8_execution_plan(plan, config)
    except (TypeError, ValueError) as error:
        raise ArgumentError(f"C8 requires the exact checked-in execution scope: {error}") from error
    if not isinstance(result, Mapping) or result.get("valid") is not True:
        violations = result.get("violations", ()) if isinstance(result, Mapping) else ()
        detail = "; ".join(str(item) for item in violations)
        raise ArgumentError(
            "C8 requires the exact checked-in execution scope"
            + (f": {detail}" if detail else "")
        )
    return dict(result)


def _require_exact_c9_plan(plan: Any, config: Any) -> dict[str, Any]:
    try:
        result = validate_c9_execution_plan(plan, config)
    except (TypeError, ValueError) as error:
        raise ArgumentError(
            f"C9 requires the exact checked-in execution scope: {error}"
        ) from error
    if not isinstance(result, Mapping) or result.get("valid") is not True:
        violations = result.get("violations", ()) if isinstance(
            result, Mapping
        ) else ()
        detail = "; ".join(str(item) for item in violations)
        raise ArgumentError(
            "C9 requires the exact checked-in execution scope"
            + (f": {detail}" if detail else "")
        )
    return dict(result)


def _require_exact_c10_input_bytes(args: argparse.Namespace) -> None:
    from .stage_c_c10_source import (
        C9_CONFIG_FILE_SHA256, C9_RUN_SPEC_FILE_SHA256,
    )

    if sha256_file(args.config) != C9_CONFIG_FILE_SHA256:
        raise ArgumentError("C10 requires the exact frozen Stage C config bytes")
    if sha256_file(args.run_spec) != C9_RUN_SPEC_FILE_SHA256:
        raise ArgumentError("C10 requires the exact frozen C9 RunSpec bytes")


def _require_clean_c8_git(git_state: Mapping[str, Any]) -> None:
    if (
        set(git_state) != {"commit", "dirty"}
        or not isinstance(git_state.get("commit"), str)
        or len(git_state["commit"]) != 40
        or git_state.get("dirty") is not False
    ):
        raise StageCBenchmarkBlockedError(
            "formal C8 requires a clean Git worktree",
            reasons=("formal C8 requires a clean Git worktree",),
        )


def _require_clean_c9_git(git_state: Mapping[str, Any]) -> None:
    if (
        set(git_state) != {"commit", "dirty"}
        or not isinstance(git_state.get("commit"), str)
        or len(git_state["commit"]) != 40
        or git_state.get("dirty") is not False
    ):
        raise StageCBenchmarkBlockedError(
            "formal C9 requires a clean Git worktree",
            reasons=("formal C9 requires a clean Git worktree",),
        )


def _validate_c9_shard_arguments(
    plan: Any, *, shard_index: int, shard_count: int
) -> None:
    """Validate only shard coordinates; core owns exact partitioning."""

    unit_count = len(tuple(plan.units))
    if type(shard_count) is not int or not 1 <= shard_count <= unit_count:
        raise ArgumentError(
            "--shard-count must be positive and cannot exceed the exact "
            "C9 execution-unit count"
        )
    if type(shard_index) is not int or not 0 <= shard_index < shard_count:
        raise ArgumentError(
            "--shard-index must be zero-based and less than --shard-count"
        )


def _load_c8_initialized_estimate(
    output_root: Path | str, plan: Any, *,
    trusted_recovery_context: Any | None = None,
) -> tuple[StageCEstimate, dict[str, Any]]:
    root = Path(output_root).absolute()
    try:
        preflight = StageCEstimate.from_dict(
            strict_json_load(root / "preflight.json")
        )
        authorization = strict_json_load(root / C8_AUTHORIZATION_FILENAME)
        validate_c8_launch_authorization(
            authorization, plan=plan, estimate=preflight, output_root=root,
            trusted_recovery_context=trusted_recovery_context,
        )
        if strict_json_load(root / "execution_plan.json") != plan.to_dict():
            raise StageCBenchmarkIntegrityError(
                "C8 initialized execution plan differs from the requested plan"
            )
    except StageCBenchmarkIntegrityError:
        raise
    except Exception as error:
        raise StageCBenchmarkIntegrityError(
            f"C8 initialized controls are missing or invalid: {error}"
        ) from error
    return preflight, authorization


def _c8_shard_unit_digests(
    plan: Any, *, shard_index: int, shard_count: int
) -> tuple[str, ...]:
    """Partition the full plan into stable interleaved balanced shards.

    Plan-order modulo assignment preserves a deterministic order inside each
    shard while spreading the three very different Value-method costs across
    devices.  It changes only orchestration, never the scientific workload.
    """

    if type(shard_count) is not int or shard_count < 1:
        raise ArgumentError("--shard-count must be a positive integer")
    if type(shard_index) is not int or not 0 <= shard_index < shard_count:
        raise ArgumentError(
            "--shard-index must be zero-based and less than --shard-count"
        )
    units = tuple(plan.units)
    if shard_count > len(units):
        raise ArgumentError(
            "--shard-count cannot exceed the exact C8 execution-unit count"
        )
    return tuple(
        unit.key.canonical_digest
        for index, unit in enumerate(units)
        if index % shard_count == shard_index
    )


def _validate_c8_shard_checkpoint_locked(
    output_root: Path | str, plan: Any, unit_digests: tuple[str, ...]
) -> None:
    """Use the runtime's current locked checkpoint/status contract."""

    root = Path(output_root).absolute()
    with _stage_c_c8_runtime._checkpoint_lock(root):
        rows, _status = _stage_c_c8_runtime._validate_checkpoint_controls(
            root, plan, allowed_lifecycle=("running",)
        )
        by_digest = {
            row["unit_key"]["canonical_digest"]: row for row in rows
        }
        unavailable = [
            digest for digest in unit_digests
            if by_digest[digest].get("state") != "pending"
        ]
        if unavailable:
            raise ArgumentError(
                f"C8 shard overlaps units that are not pending: {unavailable}"
            )


def _c8_input_hashes(
    *, config: Any, config_path: Path, evidence: Mapping[str, Any],
    source_git_state: Mapping[str, Any],
    recovery_record: Any | None = None,
) -> dict[str, Any]:
    protocol_path = REPO_ROOT / config.protocol.path
    return {
        "code_commit": source_git_state["commit"],
        "code_tree_clean": not source_git_state["dirty"],
        "model_sha256": (
            MODEL_SHA256 if recovery_record is None
            else recovery_record.model_identity["state_dict_sha256"]
        ),
        "dependency_lock_sha256": sha256_file(DEFAULT_DEPENDENCY_LOCK),
        "registry_sha256": config.registry.sha256,
        "registry_metadata_sha256": config.registry.metadata_sha256,
        "protocol_sha256": sha256_file(protocol_path),
        "acceptance_sha256": sha256_file(DEFAULT_ACCEPTANCE),
        "config_sha256": sha256_file(config_path),
        "config_schema_sha256": sha256_file(DEFAULT_STAGE_C_SCHEMA),
        "calibration_artifact_manifest_sha256": evidence[
            "calibration_artifact_manifest_sha256"
        ],
    }


def _c8_report_binding(
    authorization: Mapping[str, Any], *, c7_report: Mapping[str, Any]
) -> dict[str, Any]:
    """Build selection-budget evidence only from the freshly validated C7 report."""

    if authorization.get("authority_kind") == "recovery_c8_launch_record":
        report_binding = authorization.get("c7_report")
        launch_binding = authorization.get("c7_launch")
        projections = authorization.get("projections")
        if not all(isinstance(item, Mapping) for item in (
            report_binding, launch_binding, projections,
        )):
            raise StageCBenchmarkIntegrityError(
                "recovery C8 authorization lacks C7 report projections"
            )
        binding = {
            "c7_report_path": str(REPO_ROOT / report_binding["path"]),
            "c7_report_file_sha256": report_binding["file_sha256"],
            "c7_report_canonical_digest": report_binding["canonical_digest"],
            "c7_code_commit": report_binding["source_commit"],
            "c7_launch_path": launch_binding["path"],
            "c7_launch_file_sha256": launch_binding["file_sha256"],
            "c7_launch_canonical_digest": launch_binding["canonical_digest"],
            "c7_launch_commit": launch_binding["launch_commit"],
            **{key: projections[key] for key in (
                "c7_slowest_unit_seconds", "c7_peak_reserved_bytes",
                "c7_projected_c8_h20_gpu_hours",
                "c7_projected_c8_artifact_bytes",
                "c7_projected_c8_l2_bytes_per_audit",
            )},
        }
    else:
        binding = {
            key: authorization[key]
            for key in (
                "c7_report_path", "c7_report_file_sha256",
                "c7_report_canonical_digest", "c7_code_commit",
                "c7_archive_ref", "c7_archive_tree",
                "c7_slowest_unit_seconds", "c7_peak_reserved_bytes",
                "c7_projected_c8_h20_gpu_hours",
                "c7_projected_c8_artifact_bytes",
                "c7_projected_c8_l2_bytes_per_audit",
            )
        }
    report_bindings = c7_report.get("bindings")
    measurement = c7_report.get("measurement_boundary")
    if not isinstance(report_bindings, Mapping) or not isinstance(
        measurement, Mapping
    ):
        raise StageCBenchmarkIntegrityError(
            "fresh C7 report lacks selection budget bindings"
        )
    recomputed = report_bindings.get("recomputed_estimate")
    control_bytes = measurement.get("source_batch_control_bytes")
    if not isinstance(recomputed, Mapping) or type(control_bytes) is not int:
        raise StageCBenchmarkIntegrityError(
            "fresh C7 report lacks its measured storage operands"
        )
    binding["recomputed_estimate"] = json.loads(
        json.dumps(recomputed, allow_nan=False)
    )
    binding["source_batch_control_bytes"] = control_bytes
    return binding


def _c8_authorization_source_state(
    authorization: Mapping[str, Any],
) -> Mapping[str, Any]:
    if authorization.get("authority_kind") == "recovery_c8_launch_record":
        source = authorization.get("source")
        code = (
            source.get("execution_git_state")
            if isinstance(source, Mapping) else None
        )
    else:
        code = authorization.get("code")
    if not isinstance(code, Mapping) or set(code) != {"commit", "dirty"}:
        raise StageCBenchmarkIntegrityError(
            "C8 authorization lacks its execution Git state"
        )
    return code


def _c8_authorization_report_path(
    authorization: Mapping[str, Any], recovery_record: Any | None,
) -> Path | str:
    if authorization.get("authority_kind") == "recovery_c8_launch_record":
        if recovery_record is None:
            raise StageCBenchmarkIntegrityError(
                "recovery C8 authorization lacks its launch authority"
            )
        return REPO_ROOT / recovery_record.document["c7_report"]["path"]
    return authorization["c7_report_path"]


def _default_backend_factory(
    *, config: Any, registry: Any, plan: Any, args: argparse.Namespace
) -> ProductionStageCBackend:
    rows = registry.by_sample_id()
    planned_sample_ids = {
        unit.sample.sample_id for unit in getattr(plan, "units", ())
    }
    data_root = Path(registry.data_root).resolve()

    def load_sample(sample: Any, model: Any) -> dict[str, Any]:
        import torch
        from PIL import Image
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms import functional as transform_functional
        from .modeling import canonical_preprocess_spec

        sample_id = getattr(sample, "sample_id", None)
        if planned_sample_ids and sample_id not in planned_sample_ids:
            raise ValueError(
                f"sample is outside the current execution plan: {sample_id}"
            )
        if sample_id not in rows:
            raise ValueError(f"sample is absent from the registry: {sample_id}")
        row = rows[sample_id]
        image_path = data_root / row.relative_path
        if sha256_file(image_path) != row.image_sha256:
            raise ValueError(f"image SHA-256 mismatch: {row.sample_id}")
        specification = canonical_preprocess_spec(model)
        if (
            specification.input_size != (3, 224, 224)
            or specification.resize_shorter_side != 248
            or specification.center_crop != (224, 224)
            or specification.interpolation != "bicubic"
            or specification.mean != (0.5, 0.5, 0.5)
            or specification.std != (0.5, 0.5, 0.5)
        ):
            raise RuntimeError("canonical preprocessing contract drifted")
        with Image.open(image_path) as source:
            rgb = source.convert("RGB")
            resized = transform_functional.resize(
                rgb, specification.resize_shorter_side,
                interpolation=InterpolationMode.BICUBIC, antialias=True,
            )
            cropped = transform_functional.center_crop(
                resized, list(specification.center_crop)
            )
            canonical = transform_functional.pil_to_tensor(cropped).to(
                dtype=torch.float32
            ).div(255.0).unsqueeze(0)
        mean = torch.tensor(specification.mean, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(specification.std, dtype=torch.float32).view(1, 3, 1, 1)
        normalized = (canonical - mean) / std
        return {
            "canonical_rgb": canonical,
            "normalized_input": normalized,
            "registry_row": row,
            "image_path": image_path,
        }

    recovery_binding = getattr(args, "_recovery_binding", None)
    recovery_c8_record = getattr(args, "_recovery_c8_launch_record", None)
    recovery_c8_salvage_record = getattr(
        args, "_recovery_c8_salvage_launch_record", None
    )
    recovery_c8_salvage_repair_record = getattr(
        args, "_recovery_c8_salvage_repair_launch_record", None
    )
    state_dict_path = (
        recovery_binding.model_path if recovery_binding is not None
        else (getattr(args, "state_dict", None) or DEFAULT_STATE_DICT)
    )
    return ProductionStageCBackend(
        sample_loader=load_sample, state_dict_path=state_dict_path,
        recovery_binding=recovery_binding,
        recovery_launch_record=getattr(args, "_recovery_launch_record", None),
        recovery_c8_launch_record=recovery_c8_record,
        recovery_c8_salvage_launch_record=recovery_c8_salvage_record,
        recovery_c8_salvage_repair_launch_record=(
            recovery_c8_salvage_repair_record
        ),
        config=config, registry=registry, plan=plan, config_path=args.config,
        code_state=(
            getattr(args, "_recovery_code_state", None)
            or capture_git_state(REPO_ROOT)
        ),
    )


def _make_backend(
    factory: Callable[..., Any],
    *, config: Any, registry: Any, plan: Any, args: argparse.Namespace,
) -> Any:
    return factory(config=config, registry=registry, plan=plan, args=args)


def _is_exact_c7_plan(plan: Any, config: Any) -> bool:
    """Recognize the frozen C7 plan without changing other run semantics."""

    try:
        result = validate_c7_execution_plan(plan, config)
    except (TypeError, ValueError):
        return False
    return bool(result.get("valid")) if isinstance(result, dict) else False


def _is_exact_c7_diagnostic_plan(plan: Any, config: Any) -> bool:
    try:
        result = validate_c7_diagnostic_plan(plan, config)
    except (TypeError, ValueError):
        return False
    return bool(result.get("valid")) if isinstance(result, dict) else False


def _default_report_path(output_root: Path, suffix: str) -> Path:
    root = Path(output_root)
    return root.with_name(root.name + suffix)


def _same_path(left: Path | str, right: Path | str) -> bool:
    return Path(left).resolve() == Path(right).resolve()


def _recovery_c7_report_paths(record: Any) -> tuple[Path, Path]:
    return (
        Path(record.reports_root) / DIAGNOSTIC_GATE_FILENAME,
        Path(record.reports_root) / REPORT_FILENAME,
    )


def _validate_recovery_c7_paths(
    args: argparse.Namespace, record: Any, *, diagnostic: bool, formal: bool,
) -> None:
    """Reject every caller path that disagrees with launch authority."""

    if not _same_path(
        args.config,
        REPO_ROOT / record.document["c7_contracts"]["stage_c_config"]["path"],
    ):
        raise ArgumentError("--config disagrees with recovery C7 launch record")
    contract = "diagnostic_run_spec" if diagnostic else "formal_run_spec"
    if not _same_path(
        args.run_spec, REPO_ROOT / record.document["c7_contracts"][contract]["path"]
    ):
        raise ArgumentError("--run-spec disagrees with recovery C7 launch record")
    if getattr(args, "registry", None) is not None and not _same_path(
        args.registry, REPO_ROOT / record.binding.stage_c_input["registry_path"]
    ):
        raise ArgumentError("--registry disagrees with recovery C7 launch record")
    if getattr(args, "data_root", None) is not None and not _same_path(
        args.data_root, record.data_root
    ):
        raise ArgumentError("--data-root disagrees with recovery C7 launch record")
    if getattr(args, "state_dict", None) is not None and not _same_path(
        args.state_dict, record.model_path
    ):
        raise ArgumentError("--state-dict disagrees with recovery C7 launch record")
    expected_root = record.diagnostic_root if diagnostic else record.formal_root
    if not _same_path(args.output_root, expected_root):
        raise ArgumentError("--output-root disagrees with recovery C7 launch record")

    gate_path, report_path = _recovery_c7_report_paths(record)
    if args.command == "c7-diagnostic-gate":
        requested = args.output or gate_path
        if not _same_path(requested, gate_path):
            raise ArgumentError(
                "--output disagrees with recovery C7 diagnostic gate path"
            )
        args.output = gate_path
    elif args.command == "benchmark-report":
        requested = args.output or report_path
        if not _same_path(requested, report_path):
            raise ArgumentError(
                "--output disagrees with recovery C7 benchmark report path"
            )
        args.output = report_path
        if not _same_path(args.c7_diagnostic_gate, gate_path):
            raise ArgumentError(
                "--c7-diagnostic-gate disagrees with recovery C7 launch record"
            )
    elif (
        formal
        and getattr(args, "c7_diagnostic_gate", None) is not None
        and not _same_path(args.c7_diagnostic_gate, gate_path)
    ):
        raise ArgumentError(
            "--c7-diagnostic-gate disagrees with recovery C7 launch record"
        )


def _trusted_recovery_context(record: Any) -> dict[str, Any]:
    return {
        "recovery_binding": record.binding,
        "recovery_launch_record": record,
    }


def _validate_recovery_c8_arguments(
    args: argparse.Namespace, record: Any,
) -> None:
    """Bind every caller-selected C8 path to the validated authority."""

    contracts = record.document["c8_contracts"]
    config_binding = contracts.get("stage_c_config")
    run_spec_binding = contracts.get("run_spec")
    if not isinstance(config_binding, Mapping) or not _same_path(
        args.config, REPO_ROOT / config_binding["path"]
    ):
        raise ArgumentError("--config disagrees with recovery C8 launch record")
    if not isinstance(run_spec_binding, Mapping) or not _same_path(
        args.run_spec, REPO_ROOT / run_spec_binding["path"]
    ):
        raise ArgumentError("--run-spec disagrees with recovery C8 launch record")
    if getattr(args, "registry", None) is not None and not _same_path(
        args.registry, REPO_ROOT / record.binding.stage_c_input["registry_path"]
    ):
        raise ArgumentError("--registry disagrees with recovery C8 launch record")
    if getattr(args, "data_root", None) is not None and not _same_path(
        args.data_root, record.data_root
    ):
        raise ArgumentError("--data-root disagrees with recovery C8 launch record")
    if getattr(args, "state_dict", None) is not None and not _same_path(
        args.state_dict, record.model_path
    ):
        raise ArgumentError("--state-dict disagrees with recovery C8 launch record")
    if not _same_path(args.output_root, record.formal_root):
        raise ArgumentError("--output-root disagrees with recovery C8 launch record")
    if args.command == "c8-init":
        c7_report = REPO_ROOT / record.document["c7_report"]["path"]
        if not _same_path(args.c7_report, c7_report):
            raise ArgumentError("--c7-report disagrees with recovery C8 launch record")
    if args.command == "c8-select":
        expected_report = Path(record.report_path)
        expected_selection = Path(record.selection_path)
        if not _same_path(args.report_output, expected_report):
            raise ArgumentError(
                "--report-output disagrees with recovery C8 launch record"
            )
        if args.selection_output is None or not _same_path(
            args.selection_output, expected_selection
        ):
            raise ArgumentError(
                "--selection-output disagrees with recovery C8 launch record"
            )
        if _same_path(args.selection_output, _TRACKED_STAGE_C_SELECTION):
            raise ArgumentError(
                "recovery C8 must not overwrite the historical selection"
            )


def _c8_recovery_kwargs(record: Any | None) -> dict[str, Any]:
    if record is None:
        return {}
    return {
        "trusted_recovery_context": record,
    }


def _prevalidate_recovery_c7_arguments(
    args: argparse.Namespace, record: Any,
) -> None:
    """Validate caller-controlled bindings before loading the pipeline."""

    contracts = record.document["c7_contracts"]
    if not _same_path(
        args.config, REPO_ROOT / contracts["stage_c_config"]["path"]
    ):
        raise ArgumentError("--config disagrees with recovery C7 launch record")
    diagnostic_spec = _same_path(
        args.run_spec, REPO_ROOT / contracts["diagnostic_run_spec"]["path"]
    )
    formal_spec = _same_path(
        args.run_spec, REPO_ROOT / contracts["formal_run_spec"]["path"]
    )
    if not (diagnostic_spec or formal_spec):
        raise ArgumentError("--run-spec disagrees with recovery C7 launch record")
    if args.command == "c7-diagnostic-gate" and not diagnostic_spec:
        raise ArgumentError(
            "c7-diagnostic-gate requires the recovery diagnostic run spec"
        )
    if (
        args.command in {"benchmark", "benchmark-report"}
        and not formal_spec
    ):
        raise ArgumentError(
            f"{args.command} requires the recovery formal run spec"
        )
    if args.command in {"validate", "aggregate"} and not (
        diagnostic_spec or formal_spec
    ):
        raise ArgumentError(
            f"{args.command} requires a recovery C7 run spec"
        )
    if getattr(args, "registry", None) is not None and not _same_path(
        args.registry, REPO_ROOT / record.binding.stage_c_input["registry_path"]
    ):
        raise ArgumentError("--registry disagrees with recovery C7 launch record")
    if getattr(args, "data_root", None) is not None and not _same_path(
        args.data_root, record.data_root
    ):
        raise ArgumentError("--data-root disagrees with recovery C7 launch record")
    if getattr(args, "state_dict", None) is not None and not _same_path(
        args.state_dict, record.model_path
    ):
        raise ArgumentError("--state-dict disagrees with recovery C7 launch record")
    expected_root = (
        record.diagnostic_root if diagnostic_spec else record.formal_root
    )
    if not _same_path(args.output_root, expected_root):
        raise ArgumentError("--output-root disagrees with recovery C7 launch record")
    gate_path, report_path = _recovery_c7_report_paths(record)
    if args.command == "c7-diagnostic-gate":
        if args.output is not None and not _same_path(args.output, gate_path):
            raise ArgumentError(
                "--output disagrees with recovery C7 diagnostic gate path"
            )
        args.output = gate_path
    if args.command == "benchmark-report":
        if args.output is not None and not _same_path(args.output, report_path):
            raise ArgumentError(
                "--output disagrees with recovery C7 benchmark report path"
            )
        args.output = report_path
    supplied_gate = getattr(args, "c7_diagnostic_gate", None)
    if supplied_gate is not None and formal_spec and not _same_path(
        supplied_gate, gate_path
    ):
        raise ArgumentError(
            "--c7-diagnostic-gate disagrees with recovery C7 launch record"
        )


def _configure_c7_execution(device: str) -> None:
    """Establish the frozen deterministic boundary for C7 only."""

    import torch

    parsed = torch.device(device)
    if parsed.type == "cuda" and os.environ.get(
        "CUBLAS_WORKSPACE_CONFIG"
    ) not in {":4096:8", ":16:8"}:
        raise RuntimeError(
            "C7 CUDA execution requires CUBLAS_WORKSPACE_CONFIG=:4096:8 "
            "(or :16:8) before process start"
        )
    seed_everything(STAGE_C_SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    if parsed.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def _configure_c8_execution(device: str) -> None:
    """Establish the frozen deterministic boundary for formal C8."""

    import torch

    parsed = torch.device(device)
    if parsed.type == "cuda" and os.environ.get(
        "CUBLAS_WORKSPACE_CONFIG"
    ) not in {":4096:8", ":16:8"}:
        raise RuntimeError(
            "C8 CUDA execution requires CUBLAS_WORKSPACE_CONFIG=:4096:8 "
            "(or :16:8) before process start"
        )
    seed_everything(STAGE_C_SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    if parsed.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def _configure_c9_execution(device: str) -> None:
    """Establish the frozen deterministic boundary for formal C9."""

    import torch

    parsed = torch.device(device)
    if parsed.type == "cuda" and os.environ.get(
        "CUBLAS_WORKSPACE_CONFIG"
    ) not in {":4096:8", ":16:8"}:
        raise RuntimeError(
            "C9 CUDA execution requires CUBLAS_WORKSPACE_CONFIG=:4096:8 "
            "(or :16:8) before process start"
        )
    seed_everything(STAGE_C_SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    if parsed.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def _c9_report_exit_code(report: Mapping[str, Any]) -> int:
    if report.get("status") == "PASS":
        return EXIT_SUCCESS
    if report.get("integrity_valid") is False:
        return EXIT_INTEGRITY_FAILURE
    return (
        EXIT_TERMINAL_FAILURE
        if report.get("orchestration_complete")
        else EXIT_INTEGRITY_FAILURE
    )


def main(
    argv: list[str] | None = None,
    *,
    backend_factory: Callable[..., Any] | None = None,
) -> int:
    try:
        try:
            args = build_parser().parse_args(argv)
        except SystemExit as error:
            return EXIT_SUCCESS if error.code == 0 else EXIT_INPUT_ERROR

        is_c8_command = args.command in _C8_COMMANDS
        is_c8_salvage_command = args.command in _C8_SALVAGE_COMMANDS
        is_c8_salvage_repair_command = (
            args.command in _C8_SALVAGE_REPAIR_COMMANDS
        )
        is_c9_command = args.command in _C9_COMMANDS
        is_c10_command = args.command in _C10_COMMANDS
        recovery_record = None
        recovery_c8_record = None
        salvage_record = None
        repair_record = None
        recovery_c9_record = None
        recovery_c10_record = None
        if is_c8_salvage_command:
            salvage_record = load_recovery_c8_salvage_launch_record(
                args.salvage_authority, repo_root=REPO_ROOT,
                require_tracked=True, require_fresh_outputs=False,
            )
        if is_c8_salvage_repair_command:
            repair_record = load_recovery_c8_salvage_repair_launch_record(
                args.repair_authority, repo_root=REPO_ROOT,
                require_tracked=True, require_fresh_outputs=False,
            )
        recovery_c8_path = getattr(args, "recovery_c8_launch_record", None)
        if is_c8_command and recovery_c8_path is None:
            raise ArgumentError(
                "exact C8 commands require --recovery-c8-launch-record"
            )
        if recovery_c8_path is not None:
            recovery_c8_record = load_recovery_c8_launch_record(
                recovery_c8_path, repo_root=REPO_ROOT,
                require_fresh_roots=False, require_tracked=True,
            )
            _validate_recovery_c8_arguments(args, recovery_c8_record)
            args._recovery_binding = recovery_c8_record.binding
            args._recovery_launch_record = recovery_c8_record.c7_launch_record
            args._recovery_c8_launch_record = recovery_c8_record
        recovery_path = getattr(args, "recovery_c7_launch_record", None)
        if recovery_path is not None:
            recovery_record = load_recovery_c7_launch_record(
                recovery_path, repo_root=REPO_ROOT,
                require_fresh_roots=False,
                require_tracked=True,
            )
            _prevalidate_recovery_c7_arguments(args, recovery_record)
            args._recovery_binding = recovery_record.binding
            args._recovery_launch_record = recovery_record
        recovery_c9_path = getattr(args, "recovery_c9_launch_record", None)
        if is_c9_command and recovery_c9_path is not None:
            recovery_c9_record = load_recovery_c9_launch_record(
                recovery_c9_path, repo_root=REPO_ROOT, require_tracked=True,
                full_c8_validation=False,
            )
            args._recovery_binding = recovery_c9_record.binding
            args._recovery_launch_record = recovery_c9_record.c7_launch_record
            args._recovery_c8_salvage_repair_launch_record = (
                recovery_c9_record.repair_record
            )
            args._recovery_code_state = capture_git_state(REPO_ROOT)
        if is_c10_command:
            recovery_c10_record = load_recovery_c10_launch_record(
                args.recovery_c10_launch_record, repo_root=REPO_ROOT,
                require_tracked=True, validate_runtime=True,
            )
            recovery_c9_record = recovery_c10_record.c9_record
        if is_c9_command or is_c10_command:
            pipeline = _load_pipeline(
                args, c9=True, recovery_c9_record=recovery_c9_record,
            )
        elif (
            is_c8_command or is_c8_salvage_command
            or is_c8_salvage_repair_command
        ):
            pipeline = _load_pipeline(
                args, c8=True, recovery_c8_record=(
                    recovery_c8_record if recovery_c8_record is not None
                    else (
                        salvage_record if salvage_record is not None
                        else repair_record
                    )
                ),
            )
        else:
            pipeline = _load_pipeline(args, recovery_record=recovery_record)
        config, registry, _selection, _spec, plan, estimate = pipeline
        if (
            is_c8_command or is_c8_salvage_command
            or is_c8_salvage_repair_command
        ):
            _require_exact_c8_plan(plan, config)
        elif is_c9_command or is_c10_command:
            _require_exact_c9_plan(plan, config)
        if is_c10_command:
            _require_exact_c10_input_bytes(args)

        if args.command in _C10_COMMANDS:
            operation = (
                build_c10_output_root
                if args.command == "c10-build"
                else validate_c10_output_root
            )
            kwargs = {}
            if args.command == "c10-build":
                git_state = capture_git_state(REPO_ROOT)
                kwargs["observed_git_state"] = git_state
            result = operation(
                args.c9_root, args.output_root, plan, config,
                recovery_authority=recovery_c10_record, **kwargs
            )
            _write_json(result, sys.stdout)
            return (
                EXIT_SUCCESS if result.get("status") == "PASS"
                else EXIT_PREFLIGHT_BLOCKED
            )

        if is_c8_salvage_repair_command:
            assert repair_record is not None
            if args.command == "c8-salvage-repair-init":
                result = initialize_c8_salvage_repair_root(
                    repair_record, plan, config,
                )
            elif args.command == "c8-salvage-repair-run-shard":
                runtime_git_state = (
                    capture_authorized_c8_salvage_repair_source_state(
                        repair_record, "repair_execution"
                    )
                )
                _configure_c8_execution(args.device)
                args._recovery_binding = repair_record.binding
                args._recovery_launch_record = repair_record.c7_launch_record
                args._recovery_c8_salvage_repair_launch_record = repair_record
                args._recovery_code_state = runtime_git_state
                backend = _make_backend(
                    backend_factory or _default_backend_factory,
                    config=config, registry=registry, plan=plan, args=args,
                )
                units = run_c8_salvage_repair_shard(
                    repair_record, plan, backend,
                    shard_index=args.shard_index,
                    shard_count=args.shard_count, device=args.device,
                    runtime_git_state=runtime_git_state,
                )
                unit_results = [
                    item.to_dict()
                    if callable(getattr(item, "to_dict", None)) else item
                    for item in units
                ]
                states = [
                    item.get("state") if isinstance(item, Mapping) else None
                    for item in unit_results
                ]
                orchestration_complete = bool(states) and all(
                    state in {"succeeded", "failed"} for state in states
                )
                scientific_successful = bool(states) and all(
                    state == "succeeded" for state in states
                )
                result = {
                    "status": (
                        "PASS" if orchestration_complete else "BLOCKED"
                    ),
                    "orchestration_complete": orchestration_complete,
                    "scientific_successful": scientific_successful,
                    "unit_count": len(unit_results),
                    "unit_results": unit_results,
                    "holdout_accessed": False,
                }
            elif args.command == "c8-salvage-repair-finalize":
                result = finalize_c8_salvage_repair(
                    repair_record, plan, config
                )
            elif args.command == "c8-salvage-repair-validate":
                result = validate_c8_salvage_repair_from_source(
                    repair_record, plan, config
                )
            elif args.command == "c8-salvage-repair-complete":
                result = (
                    validate_c8_salvage_repair_completion_report(
                        repair_record, plan, config
                    )
                    if args.mode == "validate"
                    else write_c8_salvage_repair_completion_report(
                        repair_record, plan, config
                    )
                )
            else:
                generator = (
                    capture_authorized_c8_salvage_repair_source_state(
                        repair_record, "selection_validation"
                        if args.mode == "validate"
                        else "selection_generation"
                    )
                )
                if args.mode == "validate":
                    report = strict_json_load(
                        repair_record.selection_report_path
                    )
                    selection = (
                        strict_json_load(repair_record.selection_path)
                        if Path(repair_record.selection_path).exists() else None
                    )
                    validate_c8_salvage_repair_selection_report(
                        repair_record, plan, config, report, selection,
                        generator_git_state=generator,
                    )
                    result = report
                else:
                    selection, report = build_c8_salvage_repair_selection(
                        repair_record, plan, config,
                        generator_git_state=generator,
                    )
                    result = write_c8_salvage_repair_selection_outputs(
                        repair_record, plan, config, selection, report,
                        generator_git_state=generator,
                    )
            _write_json(result, sys.stdout)
            return (
                EXIT_SUCCESS if result.get("status") == "PASS"
                else EXIT_TERMINAL_FAILURE
            )

        if is_c8_salvage_command:
            assert salvage_record is not None
            if args.command == "c8-salvage-init":
                result = initialize_c8_salvage_gap_root(
                    salvage_record, plan, config,
                )
            elif args.command == "c8-salvage-run-shard":
                runtime_git_state = capture_authorized_c8_salvage_source_state(
                    salvage_record, "gap_execution"
                )
                _configure_c8_execution(args.device)
                args._recovery_binding = salvage_record.binding
                args._recovery_launch_record = salvage_record.c7_launch_record
                args._recovery_c8_salvage_launch_record = salvage_record
                args._recovery_code_state = runtime_git_state
                backend = _make_backend(
                    backend_factory or _default_backend_factory,
                    config=config, registry=registry, plan=plan, args=args,
                )
                units = run_c8_salvage_gap_shard(
                    salvage_record, plan, backend,
                    shard_index=args.shard_index, shard_count=args.shard_count,
                    device=args.device, runtime_git_state=runtime_git_state,
                )
                result = {
                    "status": (
                        "PASS" if all(
                            item.get("state") == "succeeded" for item in units
                        ) else "BLOCKED"
                    ),
                    "unit_count": len(units), "unit_results": units,
                    "holdout_accessed": False,
                }
            elif args.command == "c8-salvage-finalize":
                result = finalize_c8_salvage(salvage_record, plan, config)
            elif args.command == "c8-salvage-validate":
                result = validate_c8_salvage_from_source(
                    salvage_record, plan, config,
                )
            elif args.command == "c8-salvage-complete":
                result = (
                    validate_c8_salvage_completion_report(
                        salvage_record, plan, config,
                    )
                    if args.mode == "validate"
                    else write_c8_salvage_completion_report(
                        salvage_record, plan, config,
                    )
                )
            else:
                if args.mode == "validate":
                    selection = strict_json_load(salvage_record.selection_path)
                    report = strict_json_load(
                        salvage_record.selection_report_path
                    )
                    verifier = report.get("verifier_identity")
                    if not isinstance(verifier, Mapping):
                        raise RecoveryC8SalvageError(
                            "salvage selection report lacks verifier identity"
                        )
                    generator = {
                        "commit": verifier.get("commit"),
                        "dirty": verifier.get("dirty"),
                    }
                    from .stage_c_c8_salvage_selection import (
                        validate_c8_salvage_selection_report,
                    )
                    validate_c8_salvage_selection_report(
                        salvage_record, plan, config, report, selection,
                        generator_git_state=generator,
                    )
                    result = report
                else:
                    generator = capture_authorized_c8_salvage_source_state(
                        salvage_record, "selection_generation"
                    )
                    selection, report = build_c8_salvage_selection(
                        salvage_record, plan, config,
                        generator_git_state=generator,
                    )
                    result = write_c8_salvage_selection_outputs(
                        salvage_record, selection, report,
                    )
            _write_json(result, sys.stdout)
            return (
                EXIT_SUCCESS if result.get("status") == "PASS"
                else EXIT_TERMINAL_FAILURE
            )

        if args.command == "c9-init":
            git_state = capture_git_state(REPO_ROOT)
            _require_clean_c9_git(git_state)
            if recovery_c9_record is not None:
                authorization, measured = build_recovery_c9_launch_authorization(
                    plan, config, recovery_record=recovery_c9_record,
                    output_root=args.output_root, observed_git_state=git_state,
                    repo_root=REPO_ROOT,
                )
            else:
                if any(value is None for value in (
                    args.c8_root, args.c8_report, args.c8_recovery_selection,
                )):
                    raise ArgumentError(
                        "historical C9 init requires --c8-root, --c8-report, "
                        "and --c8-recovery-selection"
                    )
                authorization, measured = build_c9_launch_authorization(
                    plan, config, c8_root=args.c8_root,
                    c8_report_path=args.c8_report,
                    recovery_selection_path=args.c8_recovery_selection,
                    output_root=args.output_root,
                    observed_git_state=git_state, repo_root=REPO_ROOT,
                )
            root = initialize_c9_output_root(
                args.output_root, plan, measured, authorization, config=config,
                observed_git_state=git_state, repo_root=REPO_ROOT,
            )
            _write_json(
                {
                    "document_type": "stage_c_c9_initialization",
                    "status": "PASS",
                    "output_root": str(root),
                    "execution_plan_sha256": plan.canonical_digest,
                    "preflight_sha256": measured.canonical_digest,
                    "authorization_sha256": authorization["canonical_digest"],
                    "expected_unit_count": len(plan.units),
                    "holdout_accessed": False,
                    "selection_performed": False,
                    "bootstrap_performed": False,
                    "holm_performed": False,
                    "decision_performed": False,
                },
                sys.stdout,
            )
            return EXIT_SUCCESS

        if args.command in {"c9-finalize", "c9-validate"}:
            git_state = capture_git_state(REPO_ROOT)
            _require_clean_c9_git(git_state)
            operation = (
                finalize_c9_output_root
                if args.command == "c9-finalize"
                else validate_c9_output_root
            )
            result = operation(
                args.output_root, plan, observed_git_state=git_state
            )
            _write_json(result, sys.stdout)
            return _c9_report_exit_code(result)

        if args.command == "c9-run-shard":
            _validate_c9_shard_arguments(
                plan, shard_index=args.shard_index,
                shard_count=args.shard_count,
            )
            git_state = capture_git_state(REPO_ROOT)
            _require_clean_c9_git(git_state)
            measured, _authorization = load_c9_initialized_state(
                args.output_root, plan, observed_git_state=git_state,
                repo_root=REPO_ROOT, require_checkpoint=True,
            )
            _configure_c9_execution(args.device)
            factory = backend_factory or _default_backend_factory
            backend = _make_backend(
                factory, config=config, registry=registry, plan=plan, args=args
            )
            results = run_c9_shard(
                plan, output_root=args.output_root, backend=backend,
                estimate=measured, shard_index=args.shard_index,
                shard_count=args.shard_count, device=args.device,
                observed_git_state=git_state,
            )
            result_documents = [
                item.to_dict() if callable(getattr(item, "to_dict", None)) else item
                for item in results
            ]
            states = [
                item.get("state") if isinstance(item, Mapping) else None
                for item in result_documents
            ]
            successful = bool(states) and all(
                state == "succeeded" for state in states
            )
            terminal = bool(states) and all(
                state in {"succeeded", "failed"} for state in states
            )
            payload = {
                "document_type": "stage_c_c9_shard_result",
                "status": "PASS" if successful else "BLOCKED",
                "successful": successful,
                "orchestration_complete": terminal,
                "shard_index": args.shard_index,
                "shard_count": args.shard_count,
                "unit_count": len(results),
                "unit_results": result_documents,
                "holdout_accessed": True,
                "selection_performed": False,
                "bootstrap_performed": False,
                "holm_performed": False,
                "decision_performed": False,
            }
            _write_json(payload, sys.stdout)
            if successful:
                return EXIT_SUCCESS
            return EXIT_TERMINAL_FAILURE if terminal else EXIT_INTEGRITY_FAILURE

        if args.command == "c8-init":
            if recovery_c8_record is None:
                git_state = capture_git_state(REPO_ROOT)
                _require_clean_c8_git(git_state)
                recovery_kwargs: dict[str, Any] = {}
            else:
                git_state = capture_authorized_c8_source_state(
                    recovery_c8_record, "pre_formal"
                )
                recovery_kwargs = _c8_recovery_kwargs(recovery_c8_record)
            authorization, measured = build_c8_launch_authorization(
                plan, config, c7_report_path=args.c7_report,
                output_root=args.output_root, observed_git_state=git_state,
                repo_root=REPO_ROOT, **recovery_kwargs,
            )
            root = initialize_c8_output_root(
                args.output_root, plan, measured, authorization, config=config,
                observed_git_state=git_state, repo_root=REPO_ROOT,
                **recovery_kwargs,
            )
            _write_json(
                {
                    "document_type": "stage_c_c8_initialization",
                    "status": "PASS",
                    "output_root": str(root),
                    "execution_plan_sha256": plan.canonical_digest,
                    "preflight_sha256": measured.canonical_digest,
                    "authorization_sha256": authorization["canonical_digest"],
                    "expected_unit_count": len(plan.units),
                    "holdout_accessed": False,
                },
                sys.stdout,
            )
            return EXIT_SUCCESS

        if args.command == "c8-finalize":
            if recovery_c8_record is None:
                git_state = capture_git_state(REPO_ROOT)
                recovery_kwargs = {}
            else:
                git_state = capture_authorized_c8_source_state(
                    recovery_c8_record, "finalize"
                )
                recovery_kwargs = _c8_recovery_kwargs(recovery_c8_record)
            result = finalize_c8_output_root(
                args.output_root, plan, observed_git_state=git_state,
                repo_root=REPO_ROOT,
                **recovery_kwargs,
            )
            _write_json(result, sys.stdout)
            if result.get("status") == "PASS":
                return EXIT_SUCCESS
            return (
                EXIT_TERMINAL_FAILURE
                if result.get("orchestration_complete")
                else EXIT_INTEGRITY_FAILURE
            )

        if args.command == "c8-select":
            recovery_context = recovery_c8_provenance_context(
                recovery_c8_record
            )
            initialized_preflight, authorization = _load_c8_initialized_estimate(
                args.output_root, plan,
                trusted_recovery_context=recovery_c8_record,
            )
            report = validate_fixed_c7_report(
                _c8_authorization_report_path(
                    authorization, recovery_c8_record
                ), repo_root=REPO_ROOT,
                **_c8_recovery_kwargs(recovery_c8_record),
            )
            measured = build_c8_measured_preflight(
                plan, config, report,
                trusted_recovery_context=recovery_c8_record,
            )
            if measured.to_dict() != initialized_preflight.to_dict():
                raise StageCBenchmarkIntegrityError(
                    "freshly measured C8 preflight differs from initialized preflight"
                )
            if recovery_c8_record is None:
                generator_git_state = capture_git_state(REPO_ROOT)
                _require_clean_c8_git(generator_git_state)
            else:
                generator_git_state = capture_authorized_c8_source_state(
                    recovery_c8_record, "selection"
                )
            source_git_state = _c8_authorization_source_state(authorization)
            c7_binding = _c8_report_binding(authorization, c7_report=report)
            evidence = build_c8_calibration_evidence(
                args.output_root, plan, config, source_git_state,
                c7_binding, trusted_recovery_authority=recovery_c8_record,
                trusted_recovery_context=recovery_context,
            )
            input_hashes = _c8_input_hashes(
                config=config, config_path=args.config, evidence=evidence,
                source_git_state=source_git_state,
                recovery_record=recovery_c8_record,
            )
            if args.mode == "validate":
                report = validate_stage_c_selection_from_source(
                    args.selection_output, args.report_output,
                    output_root=args.output_root, plan=plan, config=config,
                    observed_git_state=source_git_state, input_hashes=input_hashes,
                    c7_report_binding=c7_binding,
                    generator_git_state=generator_git_state,
                    trusted_recovery_authority=recovery_c8_record,
                    trusted_recovery_context=recovery_context,
                )
                _write_json(report, sys.stdout)
                return (
                    EXIT_SUCCESS if report.get("status") == "PASS"
                    else EXIT_TERMINAL_FAILURE
                )
            if args.selection_output is None:
                raise ArgumentError(
                    "c8-select --mode build requires --selection-output"
                )
            report = build_stage_c_selection(
                evidence, input_hashes,
                generator_git_state=generator_git_state,
                trusted_recovery_authority=recovery_c8_record,
                trusted_recovery_context=recovery_context,
            )
            outputs = write_stage_c_selection_outputs(
                args.selection_output, args.report_output, report,
                evidence=evidence, input_hashes=input_hashes,
                generator_git_state=generator_git_state,
                trusted_recovery_authority=recovery_c8_record,
                trusted_recovery_context=recovery_context,
            )
            result = {
                **outputs,
                "document_type": "stage_c_c8_selection_publish",
                "report_output": str(args.report_output.absolute()),
                "selection_output": (
                    str(args.selection_output.absolute())
                    if outputs.get("selection_sha256") is not None else None
                ),
                "holdout_accessed": False,
            }
            _write_json(result, sys.stdout)
            return (
                EXIT_SUCCESS if result.get("status") == "PASS"
                else EXIT_TERMINAL_FAILURE
            )

        if args.command == "c8-run-shard":
            digests = _c8_shard_unit_digests(
                plan, shard_index=args.shard_index, shard_count=args.shard_count
            )
            initialized_preflight, authorization = _load_c8_initialized_estimate(
                args.output_root, plan,
                trusted_recovery_context=recovery_c8_record,
            )
            recovery_context = recovery_c8_record
            report = validate_fixed_c7_report(
                _c8_authorization_report_path(
                    authorization, recovery_c8_record
                ), repo_root=REPO_ROOT,
                **_c8_recovery_kwargs(recovery_c8_record),
            )
            measured = build_c8_measured_preflight(
                plan, config, report,
                trusted_recovery_context=recovery_c8_record,
            )
            if measured.to_dict() != initialized_preflight.to_dict():
                raise StageCBenchmarkIntegrityError(
                    "freshly measured C8 preflight differs from initialized preflight"
                )
            git_state = (
                capture_authorized_c8_source_state(
                    recovery_c8_record, "formal"
                )
                if recovery_c8_record is not None
                else capture_git_state(REPO_ROOT)
            )
            if git_state != _c8_authorization_source_state(authorization):
                raise StageCBenchmarkIntegrityError(
                    "C8 shard requires the clean Git state bound at initialization"
                )
            _validate_c8_shard_checkpoint_locked(
                args.output_root, plan, digests
            )
            _configure_c8_execution(args.device)
            if recovery_c8_record is not None:
                args._recovery_code_state = git_state
            factory = backend_factory or _default_backend_factory
            backend = _make_backend(
                factory, config=config, registry=registry, plan=plan, args=args
            )
            results = run_c8_shard(
                plan, output_root=args.output_root, backend=backend,
                estimate=measured, unit_digests=digests, device=args.device,
                observed_git_state=git_state,
                repo_root=REPO_ROOT,
                trusted_recovery_context=recovery_context,
            )
            result_documents = [
                item.to_dict() if callable(getattr(item, "to_dict", None)) else item
                for item in results
            ]
            states = [
                item.get("state") if isinstance(item, Mapping) else None
                for item in result_documents
            ]
            successful = bool(states) and all(state == "succeeded" for state in states)
            terminal = bool(states) and all(
                state in {"succeeded", "failed"} for state in states
            )
            payload = {
                "document_type": "stage_c_c8_shard_result",
                "status": "PASS" if successful else "BLOCKED",
                "successful": successful,
                "orchestration_complete": terminal,
                "shard_index": args.shard_index,
                "shard_count": args.shard_count,
                "unit_count": len(results),
                "unit_results": result_documents,
                "holdout_accessed": False,
            }
            _write_json(payload, sys.stdout)
            if successful:
                return EXIT_SUCCESS
            return EXIT_TERMINAL_FAILURE if terminal else EXIT_INTEGRITY_FAILURE

        if args.command == "estimate":
            _write_json(estimate, sys.stdout)
            return EXIT_SUCCESS if estimate.within_budget else EXIT_PREFLIGHT_BLOCKED

        if args.command == "validate":
            validation_kwargs: dict[str, Any] = {}
            if recovery_record is not None:
                is_c7 = _is_exact_c7_plan(plan, config)
                is_c7_diagnostic = _is_exact_c7_diagnostic_plan(plan, config)
                if not (is_c7 or is_c7_diagnostic):
                    raise ArgumentError(
                        "recovery C7 validate requires an exact C7 scope"
                    )
                _validate_recovery_c7_paths(
                    args, recovery_record, diagnostic=is_c7_diagnostic,
                    formal=is_c7,
                )
                validation_kwargs = {
                    "observed_git_state": capture_authorized_c7_source_state(
                        recovery_record, "report",
                    ),
                    "trusted_recovery_context": _trusted_recovery_context(
                        recovery_record
                    ),
                }
            result = validate_stage_c_batch(
                args.output_root, plan, **validation_kwargs
            )
            _write_json(result, sys.stdout)
            if not result.valid:
                return EXIT_INTEGRITY_FAILURE
            return EXIT_SUCCESS if result.successful else EXIT_TERMINAL_FAILURE

        if args.command == "aggregate":
            aggregate_kwargs: dict[str, Any] = {"output_path": args.output}
            if recovery_record is not None:
                is_c7 = _is_exact_c7_plan(plan, config)
                is_c7_diagnostic = _is_exact_c7_diagnostic_plan(plan, config)
                if not (is_c7 or is_c7_diagnostic):
                    raise ArgumentError(
                        "recovery C7 aggregate requires an exact C7 scope"
                    )
                _validate_recovery_c7_paths(
                    args, recovery_record, diagnostic=is_c7_diagnostic,
                    formal=is_c7,
                )
                aggregate_kwargs.update({
                    "observed_git_state": capture_authorized_c7_source_state(
                        recovery_record, "report",
                    ),
                    "trusted_recovery_context": _trusted_recovery_context(
                        recovery_record
                    ),
                })
            result = aggregate_stage_c_batch(
                args.output_root, plan, **aggregate_kwargs
            )
            _write_json(result, sys.stdout)
            return (
                EXIT_SUCCESS if result.get("successful", True)
                else EXIT_TERMINAL_FAILURE
            )

        if args.command == "c7-diagnostic-gate":
            is_c7_diagnostic = _is_exact_c7_diagnostic_plan(plan, config)
            if is_c7_diagnostic and recovery_record is None:
                raise ArgumentError(
                    "the exact C7 diagnostic requires "
                    "--recovery-c7-launch-record"
                )
            if recovery_record is not None:
                if not is_c7_diagnostic:
                    raise ArgumentError(
                        "recovery C7 diagnostic gate requires the exact "
                        "checked-in diagnostic scope"
                    )
                _validate_recovery_c7_paths(
                    args, recovery_record, diagnostic=True, formal=False,
                )
            destination = args.output or _default_report_path(
                args.output_root, ".c7-diagnostic-gate.json"
            )
            git_state = (
                capture_authorized_c7_source_state(
                    recovery_record, "diagnostic_gate"
                )
                if recovery_record is not None
                else capture_git_state(REPO_ROOT)
            )
            _configure_c7_execution(args.device)
            result = write_c7_diagnostic_gate(
                destination, args.output_root, plan, config=config,
                environment=capture_benchmark_environment(args.device),
                run_spec_path=args.run_spec,
                run_spec_file_sha256=sha256_file(args.run_spec),
                observed_git_state=git_state,
                **({
                    "trusted_recovery_context": _trusted_recovery_context(
                        recovery_record
                    )
                } if recovery_record is not None else {}),
            )
            _write_json(result, sys.stdout)
            return (
                EXIT_SUCCESS
                if result.get("status") == "PASS"
                else EXIT_PREFLIGHT_BLOCKED
            )

        if args.command == "benchmark-report":
            is_c7 = _is_exact_c7_plan(plan, config)
            if is_c7 and recovery_record is None:
                raise ArgumentError(
                    "the exact C7 benchmark report requires "
                    "--recovery-c7-launch-record"
                )
            if recovery_record is not None:
                if not is_c7:
                    raise ArgumentError(
                        "recovery C7 benchmark report requires the exact "
                        "checked-in C7 scope"
                    )
                _validate_recovery_c7_paths(
                    args, recovery_record, diagnostic=False, formal=True,
                )
            destination = args.output or _default_report_path(
                args.output_root, ".benchmark-report.json"
            )
            git_state = (
                capture_authorized_c7_source_state(recovery_record, "report")
                if recovery_record is not None
                else capture_git_state(REPO_ROOT)
            )
            _configure_c7_execution(args.device)
            result = write_benchmark_report(
                destination, args.output_root, plan, config=config,
                environment=capture_benchmark_environment(args.device),
                run_spec_path=args.run_spec,
                run_spec_file_sha256=sha256_file(args.run_spec),
                recomputed_estimate=estimate,
                observed_git_state=git_state,
                c7_diagnostic_gate=args.c7_diagnostic_gate,
                **({
                    "trusted_recovery_context": _trusted_recovery_context(
                        recovery_record
                    )
                } if recovery_record is not None else {}),
            )
            _write_json(result, sys.stdout)
            if result.get("status") == "PASS":
                return EXIT_SUCCESS
            exit_class = result.get("exit_class")
            if exit_class == "integrity":
                return EXIT_INTEGRITY_FAILURE
            if exit_class == "preflight":
                return EXIT_PREFLIGHT_BLOCKED
            return EXIT_TERMINAL_FAILURE

        if args.command == "benchmark" and not _is_exact_c7_plan(plan, config):
            raise ArgumentError(
                "benchmark requires the exact checked-in C7 execution scope"
            )
        is_c7 = _is_exact_c7_plan(plan, config)
        is_c7_diagnostic = _is_exact_c7_diagnostic_plan(plan, config)
        if (is_c7 or is_c7_diagnostic) and recovery_record is None:
            raise ArgumentError(
                "the exact C7 diagnostic/formal run requires "
                "--recovery-c7-launch-record"
            )
        if recovery_record is not None:
            if not (is_c7 or is_c7_diagnostic):
                raise ArgumentError(
                    "--recovery-c7-launch-record is restricted to exact C7 scopes"
                )
            _validate_recovery_c7_paths(
                args, recovery_record, diagnostic=is_c7_diagnostic, formal=is_c7,
            )
        authorization = None
        if is_c7:
            if args.c7_diagnostic_gate is None:
                _write_json(
                    {
                        "document_type": "stage_c_c7_launch_gate",
                        "status": "BLOCKED",
                        "blocked_reasons": [
                            "the exact C7 run requires --c7-diagnostic-gate"
                        ],
                    },
                    sys.stdout,
                )
                return EXIT_PREFLIGHT_BLOCKED
            git_state = capture_authorized_c7_source_state(
                recovery_record, "formal"
            )
            authorization = require_c7_launch_gate(
                args.c7_diagnostic_gate,
                plan,
                config=config,
                output_root=args.output_root,
                observed_git_state=git_state,
                run_spec_path=args.run_spec,
                trusted_recovery_context=_trusted_recovery_context(
                    recovery_record
                ),
            )
        elif is_c7_diagnostic:
            require_fresh_c7_output_root(
                args.output_root, kind="diagnostic"
            )
            args._recovery_code_state = capture_authorized_c7_source_state(
                recovery_record, "pre_diagnostic"
            )
        if not estimate.within_budget:
            _write_json(estimate, sys.stdout)
            return EXIT_PREFLIGHT_BLOCKED
        if is_c7 or is_c7_diagnostic:
            _configure_c7_execution(args.device)
        if authorization is not None:
            write_c7_launch_authorization(
                args.output_root, authorization,
                trusted_recovery_context=_trusted_recovery_context(
                    recovery_record
                ),
            )
            args._recovery_code_state = git_state
        factory = backend_factory or _default_backend_factory
        backend = _make_backend(
            factory, config=config, registry=registry, plan=plan, args=args
        )
        budget = getattr(config, "budget", None)
        caps = budget.to_dict() if callable(getattr(budget, "to_dict", None)) else None
        common = {
            "output_root": args.output_root,
            "backend": backend,
            "device": args.device,
            "resume": False if args.command == "benchmark" else not args.no_resume,
            "estimate": estimate,
            "caps": caps,
        }
        if recovery_record is not None:
            common["observed_git_state"] = args._recovery_code_state
        if args.command == "run-sample":
            result = run_stage_c_sample(plan, args.sample_id, **common)
        else:
            result = run_stage_c_batch(plan, **common)
        _write_json(result, sys.stdout)
        if result.successful:
            return EXIT_SUCCESS
        if result.orchestration_complete:
            return EXIT_TERMINAL_FAILURE
        return EXIT_INTEGRITY_FAILURE
    except StageCBatchValidationError as error:
        _write_json(
            {
                "error": {
                    "message": str(error),
                    "type": type(error).__name__,
                }
            },
            sys.stderr,
        )
        return EXIT_INTEGRITY_FAILURE
    except StageC10IntegrityError as error:
        _write_json(
            {"error": {"message": str(error), "type": type(error).__name__}},
            sys.stderr,
        )
        return EXIT_INTEGRITY_FAILURE
    except StageC10BlockedError as error:
        _write_json(
            {
                "document_type": "stage_c_c10_status",
                "status": "BLOCKED", "blocking_reasons": [str(error)],
                "downstream_authorized": False, "stage_d_started": False,
            },
            sys.stdout,
        )
        return EXIT_PREFLIGHT_BLOCKED
    except StageCBenchmarkIntegrityError as error:
        _write_json(
            {
                "error": {
                    "message": str(error),
                    "type": type(error).__name__,
                }
            },
            sys.stderr,
        )
        return EXIT_INTEGRITY_FAILURE
    except RecoveryC7LaunchError as error:
        _write_json(
            {"error": {"message": str(error), "type": type(error).__name__}},
            sys.stderr,
        )
        return EXIT_INTEGRITY_FAILURE
    except RecoveryC8LaunchError as error:
        _write_json(
            {"error": {"message": str(error), "type": type(error).__name__}},
            sys.stderr,
        )
        return EXIT_INTEGRITY_FAILURE
    except RecoveryC8SalvageError as error:
        _write_json(
            {"error": {"message": str(error), "type": type(error).__name__}},
            sys.stderr,
        )
        return EXIT_INTEGRITY_FAILURE
    except RecoveryC8SalvageRepairError as error:
        _write_json(
            {"error": {"message": str(error), "type": type(error).__name__}},
            sys.stderr,
        )
        return EXIT_INTEGRITY_FAILURE
    except RecoveryC9LaunchError as error:
        _write_json(
            {"error": {"message": str(error), "type": type(error).__name__}},
            sys.stderr,
        )
        return EXIT_INTEGRITY_FAILURE
    except RecoveryC10LaunchError as error:
        _write_json(
            {"error": {"message": str(error), "type": type(error).__name__}},
            sys.stderr,
        )
        return EXIT_INTEGRITY_FAILURE
    except StageCBenchmarkBlockedError as error:
        command = getattr(locals().get("args"), "command", "")
        if command in {"c9-run-shard", "c9-finalize", "c9-validate"}:
            # These commands operate on an already initialized formal root.
            # An incomplete/tampered attempt, online cap breach, or active
            # claim is post-launch integrity evidence, never a launch gate.
            _write_json(
                {
                    "error": {
                        "message": str(error),
                        "type": type(error).__name__,
                        "reasons": list(
                            getattr(error, "reasons", ()) or (str(error),)
                        ),
                    }
                },
                sys.stderr,
            )
            return EXIT_INTEGRITY_FAILURE
        report = getattr(error, "report", None)
        if isinstance(report, dict):
            _write_json(report, sys.stdout)
        else:
            reasons = list(getattr(error, "reasons", ()) or (str(error),))
            _write_json(
                {
                    "document_type": (
                        "stage_c_c9_launch_authorization"
                        if command == "c9-init" else (
                            "stage_c_c8_launch_authorization"
                            if command in _C8_COMMANDS
                            else "stage_c_c7_launch_gate"
                        )
                    ),
                    "status": "BLOCKED",
                    "blocked_reasons": reasons,
                },
                sys.stdout,
            )
        return EXIT_PREFLIGHT_BLOCKED
    except StageCSelectionBlockedError as error:
        report = getattr(error, "report", None)
        _write_json(
            report if isinstance(report, Mapping) else {
                "document_type": "stage_c_c8_selection_report",
                "status": "BLOCKED",
                "blocked_reasons": [str(error)],
                "holdout_accessed": False,
            },
            sys.stdout,
        )
        return EXIT_TERMINAL_FAILURE
    except StageCSelectionIntegrityError as error:
        _write_json(
            {"error": {"message": str(error), "type": type(error).__name__}},
            sys.stderr,
        )
        return EXIT_INTEGRITY_FAILURE
    except ArgumentError as error:
        _write_json(
            {"error": {"message": str(error), "type": "ArgumentError"}},
            sys.stderr,
        )
        return EXIT_INPUT_ERROR
    except Exception as error:
        _write_json(
            {
                "error": {
                    "message": str(error),
                    "type": type(error).__name__,
                }
            },
            sys.stderr,
        )
        return EXIT_INPUT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EXIT_INPUT_ERROR", "EXIT_INTEGRITY_FAILURE",
    "EXIT_PREFLIGHT_BLOCKED", "EXIT_SUCCESS",
    "EXIT_TERMINAL_FAILURE", "ArgumentError", "build_parser", "main",
]
