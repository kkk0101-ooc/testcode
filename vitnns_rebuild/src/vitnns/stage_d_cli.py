"""Non-interactive, metadata-only JSON CLI for the Stage D control plane.

The commands in this module stop at preparation authorization.  Importing the
module performs no Git query, filesystem validation, model construction, image
access, or scientific computation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
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


class ArgumentError(ValueError):
    """Raised instead of allowing argparse to emit a non-JSON error."""


class StageDCLIInputError(ValueError):
    """Raised for malformed optional CLI inputs such as profile JSON."""


class StageDCLIIntegrityError(RuntimeError):
    """Raised when a live control-plane observation cannot be trusted."""


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


def _strict_profile_json(path: Path) -> Any:
    """Read one bounded regular file and parse strict profile JSON."""

    if not hasattr(os, "O_NOFOLLOW"):
        raise StageDCLIInputError(
            "safe no-follow profile loading is unavailable on this platform"
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
                "resource profiles must be a single-link regular file"
            )
        if before.st_size > _PROFILE_FILE_LIMIT:
            raise StageDCLIInputError(
                "resource profile JSON exceeds the safe size limit"
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
                "resource profile JSON changed while it was being read"
            )
    except StageDCLIInputError:
        raise
    except OSError as error:
        raise StageDCLIInputError(
            f"cannot safely read resource profile JSON: {absolute}"
        ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
    try:
        profiles = strict_json_bytes(payload, "Stage D resource profiles")
    except StageDConfigError as error:
        raise StageDCLIInputError(str(error)) from error
    if not isinstance(profiles, (dict, list)):
        raise StageDCLIInputError(
            "Stage D resource profiles must be a JSON object or array"
        )
    return profiles


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
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Run one Stage D metadata command and return its stable exit code."""

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
    "StageDCLIPipeline", "build_parser", "main",
]
