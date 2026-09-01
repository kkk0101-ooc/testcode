"""Independent recovery-v2 C8 launch and C7-to-C8 handoff authority.

This module is metadata-only.  It reloads the committed recovery C7 launch
record and terminal PASS report, binds the exact C8 contracts and a clean
producer commit, and authorizes only calibration/selection work under fresh
repository-local roots.  It never loads tensors, images, or CUDA.
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from jsonschema import Draft202012Validator

from . import recovery_c7_launch as _c7
from .recovery_v2 import RecoveryV2Binding, RecoveryV2BindingError
from .stage_c_benchmark import FROZEN_CAPS, HARD_CAP_KEYS
from .stage_c_config import load_stage_c_config
from .stage_c_plan import (
    CALIBRATION_SAMPLE_IDS,
    CLASS_METHOD_IDS,
    RunSpec,
    VALUE_METHOD_IDS,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCHEMA_PATH = REPO_ROOT / "config/recovery_c8_launch_record.schema.json"
DEFAULT_C7_LAUNCH_RECORD_PATH = (
    REPO_ROOT
    / "evidence/stage_c/c7_recovery_v2_026b622/recovery_c7_launch_record.json"
)
DEFAULT_C7_REPORT_PATH = (
    REPO_ROOT
    / "evidence/stage_c/c7_recovery_v2_026b622/reports/"
    "stage_c_benchmark_report.json"
)
DEFAULT_C8_CONFIG_PATH = REPO_ROOT / "config/stage_c.json"
DEFAULT_C8_RUN_SPEC_PATH = REPO_ROOT / "config/stage_c_c8_run_spec.json"
LAUNCH_RECORD_FILENAME = "recovery_c8_launch_record.json"
C8_REPORT_FILENAME = "stage_c_c8_selection_report.json"
C8_SELECTION_FILENAME = "stage_c_selection.json"
DOCUMENT_TYPE = "vitnns_recovery_v2_c8_launch_record"
RECORD_ID = "RECOVERY_V2_C8_LAUNCH_20260902"
STATUS = "AUTHORIZED_FOR_RECOVERY_C8_ONLY"
SOURCE_SUBTREE_PATH = "src/vitnns"
SCHEMA_RELATIVE_PATH = "config/recovery_c8_launch_record.schema.json"
C7_LAUNCH_RELATIVE_PATH = (
    "evidence/stage_c/c7_recovery_v2_026b622/recovery_c7_launch_record.json"
)
C7_REPORT_RELATIVE_PATH = (
    "evidence/stage_c/c7_recovery_v2_026b622/reports/"
    "stage_c_benchmark_report.json"
)
C7_PRODUCER_COMMIT = "026b622577370c50babbec6d1f26342875665c8e"
C7_LAUNCH_COMMIT = "b05307241337eff42757d7ef0638223034da50be"
C7_REPORT_COMMIT = "064a0b8d96e68d78ac652c3677c307cb8b745630"
C7_LAUNCH_FILE_SHA256 = (
    "9bc7c15971ccc2422175dd8548bf5851b635e2456f44cb07b24e26ee0f4733e5"
)
C7_LAUNCH_CANONICAL_DIGEST = (
    "fa063b7c1ec4d5c44bdc3de8587eabfcd3247ca5cdbe457c41a88e5e558d6fea"
)
C7_REPORT_FILE_SHA256 = (
    "02565d66b9ef2d51fcd47f5956271cd3d35ba8ce51cec245fd2572180962970f"
)
C7_REPORT_CANONICAL_DIGEST = (
    "4f256c3f677c3da342f00716a1e172f76a2cdb761ad6d2c9fb904decf11d91b7"
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_GIT_OBJECT_RE = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
_MAX_REPORT_BYTES = 32 * 1024 * 1024
_SCOPE = {
    "execution_authorization": "RECOVERY_C8_CALIBRATION_AND_SELECTION_ONLY",
    "rerun_interpretation": "recovery_replication_not_new_confirmatory",
    "c7_authorized": False,
    "c8_authorized": True,
    "c8_selection_authorized": True,
    "stage_c_holdout_access_authorized": False,
    "c9_authorized": False,
    "c10_authorized": False,
    "stage_d_authorized": False,
}
_PHASE_ROOTS = {
    "pre_formal": (),
    "formal": ("formal_root",),
    # CLI spelling for the same terminal validation boundary.
    "finalize": ("formal_root",),
    "report": ("formal_root", "reports_root"),
    "selection": ("formal_root", "reports_root", "selection_root"),
}


class RecoveryC8LaunchError(ValueError):
    """Raised when recovery C8 launch authority cannot be established."""


def _from_c7(function: Any, *args: Any, **kwargs: Any) -> Any:
    try:
        return function(*args, **kwargs)
    except (_c7.RecoveryC7LaunchError, RecoveryV2BindingError) as error:
        raise RecoveryC8LaunchError(str(error)) from error


def _repository_identity(repo: Path) -> tuple[Path, str]:
    return _from_c7(_c7._repository_identity, repo)


def _project_path(
    repo: Path, raw: Path | str, context: str, *, require_exists: bool = False,
) -> tuple[Path, str]:
    return _from_c7(
        _c7._project_path, repo, raw, context, require_exists=require_exists,
    )


def _top_relative(top: Path, path: Path) -> str:
    return _from_c7(_c7._top_relative, top, path)


def _run_git(repo: Path, *arguments: str) -> str:
    return _from_c7(_c7._run_git, repo, *arguments)


def _git_status_entries(repo: Path) -> list[dict[str, str]]:
    return _from_c7(_c7._git_status_entries, repo)


def _is_ancestor(repo: Path, older: str, newer: str, context: str) -> None:
    _from_c7(_c7._is_ancestor, repo, older, newer, context)


def _require_tracked(repo: Path, top_path: str, revision: str = "HEAD") -> None:
    _from_c7(_c7._require_tracked, repo, top_path, revision)


def _verify_lfs_pointer_at_revision(
    repo: Path, top: Path, path: Path, *, revision: str = "HEAD",
    expected_sha256: str | None = None, require_clean: bool = True,
) -> tuple[str, int]:
    return _from_c7(
        _c7._verify_lfs_pointer_at_revision, repo, top, path,
        revision=revision, expected_sha256=expected_sha256,
        require_clean=require_clean,
    )


def _verify_lfs_pointer_at_head(
    repo: Path, top: Path, path: Path, *, expected_sha256: str | None = None,
) -> tuple[str, int]:
    return _verify_lfs_pointer_at_revision(
        repo, top, path, revision="HEAD", expected_sha256=expected_sha256,
        require_clean=True,
    )


def _capture_clean_producer(repo: Path, top: Path) -> dict[str, Any]:
    if _git_status_entries(repo):
        raise RecoveryC8LaunchError(
            "Recovery C8 launch generation requires a clean Git worktree"
        )
    commit = _run_git(repo, "rev-parse", "--verify", "HEAD^{commit}")
    tree = _run_git(repo, "rev-parse", "--verify", "HEAD^{tree}")
    source, _relative = _project_path(
        repo, SOURCE_SUBTREE_PATH, "producer source subtree",
        require_exists=True,
    )
    source_tree = _run_git(
        repo, "rev-parse", f"HEAD:{_top_relative(top, source)}"
    )
    if (
        _COMMIT_RE.fullmatch(commit) is None
        or _GIT_OBJECT_RE.fullmatch(tree) is None
        or _GIT_OBJECT_RE.fullmatch(source_tree) is None
    ):
        raise RecoveryC8LaunchError("Producer Git identity is invalid")
    return {
        "commit": commit, "tree": tree,
        "source_subtree_path": SOURCE_SUBTREE_PATH,
        "source_subtree_tree": source_tree, "dirty": False,
    }


def _publish_exclusive(repo: Path, launch: Path, payload: bytes) -> None:
    _from_c7(_c7._publish_exclusive, repo, launch, payload)


sha256_file = _c7.sha256_file
canonical_json_bytes = _c7.canonical_json_bytes
canonical_sha256 = _c7.canonical_sha256
_json_file_bytes = _c7._json_file_bytes


def launch_record_digest(document: Mapping[str, Any]) -> str:
    return canonical_sha256({
        key: value for key, value in document.items()
        if key != "canonical_digest"
    })


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(child) for key, child in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(child) for child in value)
    return copy.deepcopy(value)


def _regular_single_link_file(path: Path, context: str) -> None:
    _from_c7(_c7._regular_single_link_file, path, context)


def _read_json(path: Path, context: str, *, maximum: int = 4 * 1024 * 1024) -> Any:
    _regular_single_link_file(path, context)
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise RecoveryC8LaunchError(f"Cannot read {context}: {path}") from error
    if len(payload) > maximum:
        raise RecoveryC8LaunchError(f"{context} exceeds the metadata size limit")

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise RecoveryC8LaunchError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def nonfinite(token: str) -> None:
        raise RecoveryC8LaunchError(f"Non-finite JSON value is forbidden: {token}")

    try:
        return json.loads(
            payload.decode("utf-8"), object_pairs_hook=pairs,
            parse_constant=nonfinite,
        )
    except UnicodeDecodeError as error:
        raise RecoveryC8LaunchError(f"{context} is not UTF-8") from error
    except json.JSONDecodeError as error:
        raise RecoveryC8LaunchError(f"Invalid {context}: {error}") from error


def _require_self_digest(document: Mapping[str, Any], context: str) -> None:
    digest = document.get("canonical_digest")
    if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
        raise RecoveryC8LaunchError(f"{context} canonical digest is invalid")
    if digest != canonical_sha256({
        key: value for key, value in document.items()
        if key != "canonical_digest"
    }):
        raise RecoveryC8LaunchError(f"{context} canonical digest drifted")


@dataclass(frozen=True)
class RecoveryC8LaunchRecord:
    """Validated C8-only authority and its already validated C7 handoff."""

    path: Path
    schema_path: Path
    document: Mapping[str, Any]
    binding: RecoveryV2Binding
    c7_launch_record: _c7.RecoveryC7LaunchRecord
    model_path: Path
    data_root: Path
    devkit_path: Path
    formal_root: Path
    reports_root: Path
    selection_root: Path
    report_path: Path
    selection_path: Path
    model_identity: Mapping[str, Any]
    file_sha256: str
    launch_commit: str
    repo_root: Path = REPO_ROOT

    @property
    def canonical_digest(self) -> str:
        return str(self.document["canonical_digest"])


def _validate_output_paths(
    repo: Path, launch_raw: Path | str, formal_raw: Path | str,
    reports_raw: Path | str, selection_raw: Path | str, *,
    require_fresh_roots: bool, launch_must_exist: bool,
) -> tuple[Path, Path, Path, Path, Path]:
    launch, launch_rel = _project_path(
        repo, launch_raw, "launch record path", require_exists=launch_must_exist,
    )
    if launch.name != LAUNCH_RECORD_FILENAME:
        raise RecoveryC8LaunchError(
            f"Launch record filename must be {LAUNCH_RECORD_FILENAME}"
        )
    common = launch.parent
    common_rel = PurePosixPath(launch_rel).parent
    if len(common_rel.parts) < 3 or common_rel.parts[:2] != ("evidence", "stage_c"):
        raise RecoveryC8LaunchError(
            "Recovery C8 common root must be named under evidence/stage_c"
        )
    roots: list[Path] = []
    for raw, expected_name, context in (
        (formal_raw, "formal", "formal root"),
        (reports_raw, "reports", "reports root"),
        (selection_raw, "selection", "selection root"),
    ):
        path, _relative = _project_path(repo, raw, context)
        if path.parent != common:
            raise RecoveryC8LaunchError(
                f"{context} must be a direct child of the launch common root"
            )
        if path.name != expected_name:
            raise RecoveryC8LaunchError(f"{context} must be named {expected_name}")
        if path.exists() or path.is_symlink():
            if require_fresh_roots:
                raise RecoveryC8LaunchError(f"{context} is not fresh")
            if path.is_symlink() or not path.is_dir():
                raise RecoveryC8LaunchError(f"{context} must be a directory")
        roots.append(path)
    if common.exists() and (common.is_symlink() or not common.is_dir()):
        raise RecoveryC8LaunchError(
            "Launch common root must be an ordinary directory"
        )
    if require_fresh_roots and common.exists():
        allowed = {launch.name} if launch_must_exist else set()
        children = list(common.iterdir())
        if {child.name for child in children} != allowed:
            raise RecoveryC8LaunchError(
                "Fresh common root may contain only the launch record"
            )
        if launch_must_exist and (
            len(children) != 1 or children[0].is_symlink()
            or not children[0].is_file()
        ):
            raise RecoveryC8LaunchError(
                "Fresh common root must contain one ordinary launch record"
            )
    return launch, common, roots[0], roots[1], roots[2]


def _validate_lfs_coverage(
    repo: Path, launch: Path, formal: Path, reports: Path, selection: Path,
) -> None:
    for probe in (
        launch, formal / ".probe", reports / ".probe", selection / ".probe",
    ):
        relative = probe.relative_to(repo).as_posix()
        if not _run_git(repo, "check-attr", "filter", "--", relative).endswith(
            ": filter: lfs"
        ):
            raise RecoveryC8LaunchError(
                "Recovery C8 evidence lacks Git LFS coverage"
            )


def _validate_schema(document: Mapping[str, Any], schema: Any) -> None:
    try:
        Draft202012Validator.check_schema(schema)
    except Exception as error:
        raise RecoveryC8LaunchError("Recovery C8 launch schema is invalid") from error
    errors = sorted(
        Draft202012Validator(schema).iter_errors(document),
        key=lambda error: tuple(str(item) for item in error.absolute_path),
    )
    if errors:
        error = errors[0]
        location = ".".join(str(item) for item in error.absolute_path) or "root"
        raise RecoveryC8LaunchError(
            f"Recovery C8 launch schema failure at {location}: {error.message}"
        )


def _validate_intrinsic_authority(document: Mapping[str, Any]) -> None:
    if (
        document.get("$schema") != "recovery_c8_launch_record.schema.json"
        or document.get("schema_version") != 1
        or document.get("document_type") != DOCUMENT_TYPE
        or document.get("record_id") != RECORD_ID
        or document.get("status") != STATUS
        or document.get("scope") != _SCOPE
    ):
        raise RecoveryC8LaunchError("Recovery C8 intrinsic authority drifted")


def _load_c7_authority(
    repo: Path, path: Path | str, *, require_tracked: bool = True,
) -> _c7.RecoveryC7LaunchRecord:
    """Reload C7 semantically while preserving its historical source boundary.

    The C7 loader intentionally requires current ``HEAD:src/vitnns`` to equal
    the C7 execution source.  A later C8 producer necessarily changes that
    subtree, so C8 repeats every C7 semantic check and validates the old
    producer/launch relationship at the committed C7 revisions instead of
    weakening or monkey-patching the C7 loader.
    """

    try:
        top, _prefix = _c7._repository_identity(repo)
        launch, launch_rel = _c7._project_path(
            repo, path, "recovery C7 launch record", require_exists=True,
        )
        if launch_rel != C7_LAUNCH_RELATIVE_PATH:
            raise _c7.RecoveryC7LaunchError(
                "Unexpected recovery C7 launch record path"
            )
        schema = repo / "config/recovery_c7_launch_record.schema.json"
        document = _c7._read_json(launch, "recovery C7 launch record")
        schema_document = _c7._read_json(schema, "recovery C7 launch schema")
        if not isinstance(document, Mapping):
            raise _c7.RecoveryC7LaunchError(
                "Recovery C7 launch record must be an object"
            )
        _c7._validate_schema(document, schema_document)
        _c7._validate_intrinsic_authority(document)
        if document["canonical_digest"] != _c7.launch_record_digest(document):
            raise _c7.RecoveryC7LaunchError(
                "Recovery C7 launch canonical digest drifted"
            )
        if (
            sha256_file(launch) != C7_LAUNCH_FILE_SHA256
            or document["canonical_digest"] != C7_LAUNCH_CANONICAL_DIGEST
        ):
            raise _c7.RecoveryC7LaunchError(
                "Recovery C7 launch frozen identity drifted"
            )
        if document["launch_schema"] != {
            "path": "config/recovery_c7_launch_record.schema.json",
            "file_sha256": sha256_file(schema),
        }:
            raise _c7.RecoveryC7LaunchError(
                "Recovery C7 launch schema binding drifted"
            )
        outputs = document["outputs"]
        checked, _common, diagnostic, formal, reports = _c7._validate_output_paths(
            repo, launch, outputs["diagnostic_root"], outputs["formal_root"],
            outputs["reports_root"], require_fresh_roots=False,
            launch_must_exist=True,
        )
        binding_record = document["recovery_binding"]
        binding = _c7.load_recovery_v2_binding(
            repo / binding_record["path"],
            schema_path=repo / binding_record["schema_path"],
            repo_root=repo, verify_assets=True,
        )
        if canonical_json_bytes(binding_record) != canonical_json_bytes(
            _c7._binding_record(repo, binding)
        ):
            raise _c7.RecoveryC7LaunchError(
                "Recovery-v2 binding identity drifted"
            )
        lfs = _c7.verify_recovery_v2_lfs_identity(binding, repo_root=repo)
        if (
            lfs["oid_sha256"] != binding.model["lfs_oid_sha256"]
            or not lfs["hydrated"]
        ):
            raise _c7.RecoveryC7LaunchError(
                "Recovery-v2 model LFS identity drifted"
            )
        if canonical_json_bytes(document["model"]) != canonical_json_bytes(
            _c7._model_record(binding)
        ):
            raise _c7.RecoveryC7LaunchError(
                "Recovery-v2 model launch identity drifted"
            )
        if document["dependencies"] != _c7._dependency_record(repo, binding):
            raise _c7.RecoveryC7LaunchError(
                "Recovery-v2 dependency binding drifted"
            )
        contracts = _c7._contract_records(repo, binding)
        if canonical_json_bytes(document["c7_contracts"]) != canonical_json_bytes(
            contracts
        ):
            raise _c7.RecoveryC7LaunchError(
                "Recovery-v2 C7 contracts drifted"
            )
        amendment = binding.document["amendment"]
        if document["amendment"] != {
            "id": amendment["id"], "path": amendment["path"],
            "commit": amendment["commit"],
            "file_sha256": amendment["sha256"],
        }:
            raise _c7.RecoveryC7LaunchError(
                "Recovery amendment launch binding drifted"
            )
        receipt = repo / document["stage_a"]["receipt_path"]
        stage_a = _c7._validate_stage_a_receipt(
            repo, top, receipt, binding, document["stage_a"],
        )
        launch_commit = _validate_historical_c7_git_authority(
            repo, top, checked, document,
            _c7._critical_paths(repo, schema, binding, stage_a, contracts),
            require_tracked=require_tracked,
        )
        record = _c7._make_record(
            repo, checked, schema, document, binding, diagnostic, formal,
            reports, launch_commit,
        )
    except (
        _c7.RecoveryC7LaunchError, RecoveryV2BindingError,
        OSError, KeyError, TypeError,
    ) as error:
        raise RecoveryC8LaunchError(
            f"Recovery C7 launch validation failed: {error}"
        ) from error
    if not isinstance(record, _c7.RecoveryC7LaunchRecord):
        raise RecoveryC8LaunchError("Recovery C7 loader returned an invalid authority")
    return record


def _validate_historical_c7_git_authority(
    repo: Path, top: Path, launch: Path, document: Mapping[str, Any],
    critical: Sequence[Path], *, require_tracked: bool,
) -> str:
    if not require_tracked:
        raise _c7.RecoveryC7LaunchError(
            "C8 handoff requires a committed recovery C7 launch record"
        )
    producer = document["producer_git"]
    if (
        producer.get("commit") != C7_PRODUCER_COMMIT
        or document.get("launch_record_commit") not in {
            "SELF", C7_LAUNCH_COMMIT,
        }
        or producer.get("dirty") is not False
    ):
        raise _c7.RecoveryC7LaunchError(
            "Recovery C7 producer identity drifted"
        )
    producer_tree = _run_git(
        repo, "rev-parse", "--verify", f"{C7_PRODUCER_COMMIT}^{{tree}}"
    )
    if producer_tree != producer.get("tree"):
        raise _c7.RecoveryC7LaunchError("Recovery C7 producer tree drifted")
    source = repo / producer["source_subtree_path"]
    source_top = _top_relative(top, source)
    if (
        _run_git(repo, "rev-parse", f"{C7_PRODUCER_COMMIT}:{source_top}")
        != producer.get("source_subtree_tree")
    ):
        raise _c7.RecoveryC7LaunchError(
            "Recovery C7 producer source subtree drifted"
        )
    head = _run_git(repo, "rev-parse", "--verify", "HEAD^{commit}")
    for older, newer, context in (
        (document["stage_a"]["producer_commit"], C7_PRODUCER_COMMIT,
         "Stage A producer commit"),
        (document["amendment"]["commit"], C7_PRODUCER_COMMIT,
         "Recovery amendment commit"),
        (C7_LAUNCH_COMMIT, head, "Recovery C7 launch commit"),
    ):
        _is_ancestor(repo, older, newer, context)
    launch_top = _top_relative(top, launch)
    resolved = _run_git(
        repo, "log", "-n", "1", "--format=%H", C7_LAUNCH_COMMIT,
        "--", f":(top){launch_top}",
    )
    if resolved != C7_LAUNCH_COMMIT:
        raise _c7.RecoveryC7LaunchError(
            "Recovery C7 launch containing commit drifted"
        )
    parents = _run_git(
        repo, "rev-list", "--parents", "-n", "1", C7_LAUNCH_COMMIT,
    ).split()
    if parents != [C7_LAUNCH_COMMIT, C7_PRODUCER_COMMIT]:
        raise _c7.RecoveryC7LaunchError(
            "Recovery C7 launch is not directly after its producer"
        )
    changed = [line for line in _run_git(
        repo, "diff", "--name-only", C7_PRODUCER_COMMIT, C7_LAUNCH_COMMIT,
        "--",
    ).splitlines() if line]
    if changed != [launch_top]:
        raise _c7.RecoveryC7LaunchError(
            "Recovery C7 producer-to-launch diff drifted"
        )
    if (
        _run_git(repo, "rev-parse", f"{C7_LAUNCH_COMMIT}:{launch_top}")
        != _run_git(repo, "rev-parse", f"HEAD:{launch_top}")
    ):
        raise _c7.RecoveryC7LaunchError(
            "Current C7 launch blob differs from its launch commit"
        )
    _verify_lfs_pointer_at_revision(
        repo, top, launch, revision=C7_LAUNCH_COMMIT,
        expected_sha256=C7_LAUNCH_FILE_SHA256, require_clean=True,
    )
    for path in critical:
        top_path = _top_relative(top, path)
        _require_tracked(repo, top_path, C7_PRODUCER_COMMIT)
        before = _run_git(repo, "rev-parse", f"{C7_PRODUCER_COMMIT}:{top_path}")
        launch_value = _run_git(
            repo, "rev-parse", f"{C7_LAUNCH_COMMIT}:{top_path}"
        )
        current = _run_git(repo, "rev-parse", f"HEAD:{top_path}")
        if before != launch_value or before != current:
            raise _c7.RecoveryC7LaunchError(
                f"Recovery C7 critical input drifted: {top_path}"
            )
    return C7_LAUNCH_COMMIT


def _c7_launch_record(
    repo: Path, record: _c7.RecoveryC7LaunchRecord,
) -> dict[str, Any]:
    return {
        "path": record.path.relative_to(repo).as_posix(),
        "file_sha256": record.file_sha256,
        "canonical_digest": record.canonical_digest,
        "launch_commit": record.launch_commit,
        "producer_commit": record.document["producer_git"]["commit"],
        "producer_tree": record.document["producer_git"]["tree"],
    }


def _validate_c7_report(
    repo: Path, top: Path, path: Path,
    c7_record: _c7.RecoveryC7LaunchRecord,
    expected: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    required = c7_record.reports_root / "stage_c_benchmark_report.json"
    if path != required:
        raise RecoveryC8LaunchError(
            "C7 PASS report must be the report bound to the C7 launch record"
        )
    report = _read_json(path, "C7 PASS report", maximum=_MAX_REPORT_BYTES)
    if not isinstance(report, Mapping):
        raise RecoveryC8LaunchError("C7 PASS report must be an object")
    _require_self_digest(report, "C7 PASS report")
    # The complete report bytes are pinned below.  Do not pass the historical
    # C7 record through the standalone C7 consumer here: that consumer
    # correctly requires current HEAD's source tree to remain at the C7
    # execution tree, whereas this handoff intentionally authorizes a later
    # C8 implementation commit.  The checks below re-establish the terminal
    # PASS facts consumed by C8 and cross-bind them to the historical record.
    batch = report.get("batch_validation")
    records = report.get("unit_records")
    if (
        report.get("document_type") != "stage_c_c7_benchmark_report"
        or set(report) != {
            "schema_version", "document_type", "status",
            "integrity_valid", "blocking_kind", "exit_class",
            "blocked_reasons", "scope", "bindings", "environment",
            "batch_validation", "unit_records", "projections",
            "measurement_boundary", "selection_performed",
            "bootstrap_performed", "decision_performed",
            "holdout_accessed", "canonical_digest",
        }
        or report.get("schema_version") != 1
        or report.get("status") != "PASS"
        or report.get("integrity_valid") is not True
        or report.get("exit_class") != "success"
        or report.get("blocking_kind") is not None
        or report.get("blocked_reasons") != []
        or any(report.get(flag) is not False for flag in (
            "holdout_accessed", "selection_performed",
            "bootstrap_performed", "decision_performed",
        ))
        or not isinstance(batch, Mapping)
        or batch.get("valid") is not True
        or batch.get("successful") is not True
        or batch.get("orchestration_complete") is not True
        or batch.get("errors") != []
        or not isinstance(records, list)
        or len(records) != 40
        or any(
            not isinstance(item, Mapping)
            or item.get("state") != "succeeded"
            or item.get("trusted_by_batch_validator") is not True
            for item in records
        )
    ):
        raise RecoveryC8LaunchError(
            "C7 report is not an integrity-valid 40/40 PASS handoff"
        )
    projections = report.get("projections")
    if not isinstance(projections, Mapping):
        raise RecoveryC8LaunchError("C7 report projections are invalid")
    completeness = projections.get("measurement_completeness", {})
    if completeness != {
        "bytes_measured": True, "calls_measured": True,
        "cuda_measured": True, "expected_units": 40,
        "timing_measured": True, "usable_units": 40,
    }:
        raise RecoveryC8LaunchError(
            "C7 report does not contain exact 40/40 measured completeness"
        )
    checks = projections.get("hard_cap_checks", [])
    if (
        not isinstance(checks, list)
        or len(checks) != len(HARD_CAP_KEYS)
        or [item.get("cap_key") for item in checks if isinstance(item, Mapping)]
        != list(HARD_CAP_KEYS)
        or any(
            not isinstance(item, Mapping)
            or item.get("status") != "PASS"
            or item.get("measured") is not True
            or item.get("cap") != FROZEN_CAPS.get(item.get("cap_key"))
            for item in checks
        )
    ):
        raise RecoveryC8LaunchError(
            "C7 report does not contain ten measured frozen-cap PASS checks"
        )
    bindings = report.get("bindings")
    source = bindings.get("observed_git_state") if isinstance(bindings, Mapping) else None
    expected_source = {"commit": c7_record.launch_commit, "dirty": False}
    if source != expected_source:
        raise RecoveryC8LaunchError(
            "C7 report source commit differs from the C7 launch authority"
        )
    launch_authorization = (
        bindings.get("launch_authorization")
        if isinstance(bindings, Mapping) else None
    )
    shared = (
        bindings.get("shared_provenance")
        if isinstance(bindings, Mapping) else None
    )
    shared_code = (
        shared.get("code") if isinstance(shared, Mapping) else None
    )
    launch_code = (
        launch_authorization.get("observed_git_state")
        if isinstance(launch_authorization, Mapping) else None
    )
    if shared_code != expected_source or launch_code != expected_source:
        raise RecoveryC8LaunchError(
            "C7 report clean source bindings drifted"
        )
    shared_model = shared.get("model") if isinstance(shared, Mapping) else None
    model = c7_record.document["model"]
    required_model = {
        "name": model["name"],
        "identity": model["identity"],
        "state_dict_path": model["state_dict_path"],
        "state_dict_sha256": model["state_dict_sha256"],
        "canonical_tensor_digest": model["canonical_tensor_digest"],
        "recovery_binding_path": c7_record.document["recovery_binding"]["path"],
        "recovery_binding_file_sha256": c7_record.document["recovery_binding"]["file_sha256"],
        "recovery_binding_canonical_digest": c7_record.document["recovery_binding"]["canonical_digest"],
        "recovery_launch_record_path": c7_record.path.relative_to(repo).as_posix(),
        "recovery_launch_record_file_sha256": c7_record.file_sha256,
        "recovery_launch_record_canonical_digest": c7_record.canonical_digest,
    }
    if not isinstance(shared_model, Mapping) or any(
        shared_model.get(key) != value for key, value in required_model.items()
    ):
        raise RecoveryC8LaunchError(
            "C7 report recovery model or launch binding drifted"
        )
    report_top = _top_relative(top, path)
    _require_tracked(repo, report_top, "HEAD")
    report_commit = _run_git(
        repo, "log", "-n", "1", "--format=%H", "HEAD", "--",
        f":(top){report_top}",
    )
    head = _run_git(repo, "rev-parse", "--verify", "HEAD^{commit}")
    if not report_commit or _COMMIT_RE.fullmatch(report_commit) is None:
        raise RecoveryC8LaunchError("C7 PASS report has no containing commit")
    if report_commit != C7_REPORT_COMMIT:
        raise RecoveryC8LaunchError("C7 PASS report containing commit drifted")
    report_parents = _run_git(
        repo, "rev-list", "--parents", "-n", "1", report_commit,
    ).split()
    if report_parents != [C7_REPORT_COMMIT, C7_LAUNCH_COMMIT]:
        raise RecoveryC8LaunchError(
            "C7 PASS report is not directly after the C7 launch commit"
        )
    _is_ancestor(repo, c7_record.launch_commit, report_commit, "C7 launch commit")
    _is_ancestor(repo, report_commit, head, "C7 report commit")
    report_blob = _run_git(repo, "rev-parse", f"{report_commit}:{report_top}")
    head_blob = _run_git(repo, "rev-parse", f"HEAD:{report_top}")
    if report_blob != head_blob:
        raise RecoveryC8LaunchError(
            "Current HEAD C7 report blob differs from its report commit"
        )
    digest = sha256_file(path)
    if (
        digest != C7_REPORT_FILE_SHA256
        or report["canonical_digest"] != C7_REPORT_CANONICAL_DIGEST
    ):
        raise RecoveryC8LaunchError("C7 PASS report frozen identity drifted")
    _verify_lfs_pointer_at_revision(
        repo, top, path, revision=report_commit, expected_sha256=digest,
        require_clean=True,
    )
    result = {
        "path": path.relative_to(repo).as_posix(),
        "file_sha256": digest,
        "canonical_digest": report["canonical_digest"],
        "report_commit": report_commit,
        "status": "PASS",
        "source_commit": source["commit"],
    }
    if expected is not None and canonical_json_bytes(expected) != canonical_json_bytes(result):
        raise RecoveryC8LaunchError("C7 PASS report launch binding drifted")
    return result


def _expected_c8_run_spec() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "document_type": "stage_c_run_spec",
        "sample_ids": list(CALIBRATION_SAMPLE_IDS),
        "method_ids": [*VALUE_METHOD_IDS, *CLASS_METHOD_IDS],
        "value_config_source": "calibration_all_144",
        "value_candidates": [],
        "fidelity_pairing_context": {"block_index": 0, "head_index": 0},
        "audit_repeatability_runs": 2,
        "audit_repeatability_scope": "clean_map_only",
        "run_registered_fidelity": True,
        "run_sanity": True,
        "run_stability": True,
    }


def _contract_records(
    repo: Path, c7_record: _c7.RecoveryC7LaunchRecord,
    config_path: Path | str, run_spec_path: Path | str,
) -> dict[str, Any]:
    config, config_rel = _project_path(
        repo, config_path, "C8 Stage C config", require_exists=True,
    )
    run_spec, run_spec_rel = _project_path(
        repo, run_spec_path, "C8 run spec", require_exists=True,
    )
    expected_config = c7_record.document["c7_contracts"]["stage_c_config"]
    config_digest = sha256_file(config)
    if (
        config_rel != expected_config["path"]
        or config_digest != expected_config["file_sha256"]
    ):
        raise RecoveryC8LaunchError(
            "C8 Stage C config differs from the C7-bound frozen config"
        )
    try:
        loaded_config = load_stage_c_config(
            config, schema_path=repo / "config/stage_c.schema.json",
        ).to_dict()
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise RecoveryC8LaunchError("C8 Stage C config is invalid") from error
    if (
        loaded_config.get("selection", {}).get("stage") != "C8"
        or loaded_config.get("selection", {}).get("eligible_role")
        != "calibration"
        or loaded_config.get("selection", {}).get("forbidden_role")
        != "holdout"
    ):
        raise RecoveryC8LaunchError("C8 config selection scope drifted")
    run_spec_document = _read_json(run_spec, "C8 run spec")
    try:
        parsed_spec = RunSpec.from_dict(run_spec_document).to_dict()
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise RecoveryC8LaunchError("C8 run spec is invalid") from error
    if parsed_spec != _expected_c8_run_spec():
        raise RecoveryC8LaunchError(
            "C8 run spec differs from the exact calibration-only contract"
        )
    return {
        "stage_c_config": {
            "path": config_rel, "file_sha256": config_digest,
            "canonical_sha256": canonical_sha256(loaded_config),
        },
        "run_spec": {
            "path": run_spec_rel, "file_sha256": sha256_file(run_spec),
            "canonical_sha256": canonical_sha256(parsed_spec),
        },
    }


def _critical_paths(
    repo: Path, top: Path, schema: Path, c7_record: _c7.RecoveryC7LaunchRecord,
    c7_report: Mapping[str, Any], contracts: Mapping[str, Any],
) -> list[Path]:
    document = c7_record.document
    paths = [
        schema, c7_record.path, c7_record.schema_path,
        repo / c7_report["path"], c7_record.binding.path,
        c7_record.binding.schema_path, c7_record.model_path,
        repo / document["amendment"]["path"],
        repo / document["stage_a"]["receipt_path"],
        repo / "config/stage_c.schema.json",
        repo / "config/stage_c_registry.schema.json",
        repo / "config/stage_c_selection.schema.json",
        top / ".gitattributes",
    ]
    paths.extend(
        repo / item["path"] for item in document["stage_a"]["artifacts"].values()
    )
    paths.extend(
        repo / item["path"] for item in document["c7_contracts"].values()
    )
    paths.extend(repo / item["path"] for item in contracts.values())
    dependency = document.get("dependencies", {})
    if isinstance(dependency, Mapping) and isinstance(
        dependency.get("requirements_lock_path"), str,
    ):
        paths.append(repo / dependency["requirements_lock_path"])
    # Preserve order while avoiding repeated config/run-spec paths.
    return list(dict.fromkeys(paths))


def _validate_git_authority(
    repo: Path, top: Path, launch: Path, document: Mapping[str, Any],
    critical: Sequence[Path], *, require_tracked: bool,
) -> str:
    producer = document["producer_git"]
    commit = producer["commit"]
    if not isinstance(commit, str) or _COMMIT_RE.fullmatch(commit) is None:
        raise RecoveryC8LaunchError("Producer commit is invalid")
    if producer.get("dirty") is not False:
        raise RecoveryC8LaunchError("Producer Git record is dirty")
    tree = _run_git(repo, "rev-parse", "--verify", f"{commit}^{{tree}}")
    if tree != producer["tree"]:
        raise RecoveryC8LaunchError("Producer tree drifted")
    source, _relative = _project_path(
        repo, producer["source_subtree_path"], "producer source subtree",
        require_exists=True,
    )
    source_top = _top_relative(top, source)
    producer_source = _run_git(repo, "rev-parse", f"{commit}:{source_top}")
    if producer_source != producer["source_subtree_tree"]:
        raise RecoveryC8LaunchError("Producer source subtree drifted")
    head = _run_git(repo, "rev-parse", "--verify", "HEAD^{commit}")
    for older, context in (
        (commit, "Producer commit"),
        (document["stage_a"]["producer_commit"], "Stage A producer commit"),
        (document["amendment"]["commit"], "Recovery amendment commit"),
        (document["c7_launch"]["launch_commit"], "C7 launch commit"),
        (document["c7_report"]["report_commit"], "C7 report commit"),
    ):
        _is_ancestor(repo, older, commit if older != commit else head, context)
    if _run_git(repo, "rev-parse", f"HEAD:{source_top}") != producer_source:
        raise RecoveryC8LaunchError(
            "Critical producer source drifted after C8 launch"
        )
    launch_top = _top_relative(top, launch)
    declared = document["launch_record_commit"]
    if not require_tracked:
        if declared != "SELF" or head != commit:
            raise RecoveryC8LaunchError(
                "Uncommitted launch validation requires SELF at producer HEAD"
            )
        if _git_status_entries(repo) != [{"status": "??", "path": launch_top}]:
            raise RecoveryC8LaunchError(
                "Uncommitted validation permits only the fresh launch record"
            )
        return "SELF"
    if declared == "SELF":
        launch_commit = _run_git(
            repo, "log", "-n", "1", "--format=%H", "HEAD", "--",
            f":(top){launch_top}",
        )
        if not launch_commit:
            raise RecoveryC8LaunchError(
                "SELF launch record has no containing Git commit"
            )
    else:
        launch_commit = declared
    if _COMMIT_RE.fullmatch(launch_commit) is None:
        raise RecoveryC8LaunchError("Launch-record commit is invalid")
    _is_ancestor(repo, launch_commit, head, "C8 launch-record commit")
    parents = _run_git(
        repo, "rev-list", "--parents", "-n", "1", launch_commit,
    ).split()
    if parents != [launch_commit, commit]:
        raise RecoveryC8LaunchError(
            "C8 launch record must be one commit directly after the producer"
        )
    changed = [line for line in _run_git(
        repo, "diff", "--name-only", commit, launch_commit, "--",
    ).splitlines() if line]
    if changed != [launch_top]:
        raise RecoveryC8LaunchError(
            "Producer-to-launch commit drift exceeds the C8 launch record"
        )
    if (
        _run_git(repo, "rev-parse", f"{launch_commit}:{launch_top}")
        != _run_git(repo, "rev-parse", f"HEAD:{launch_top}")
    ):
        raise RecoveryC8LaunchError(
            "Current HEAD C8 launch-record blob differs from its launch commit"
        )
    _verify_lfs_pointer_at_revision(
        repo, top, launch, revision=launch_commit,
        expected_sha256=sha256_file(launch), require_clean=True,
    )
    for path in critical:
        top_path = _top_relative(top, path)
        _require_tracked(repo, top_path, commit)
        before = _run_git(repo, "rev-parse", f"{commit}:{top_path}")
        after = _run_git(repo, "rev-parse", f"{launch_commit}:{top_path}")
        current = _run_git(repo, "rev-parse", f"HEAD:{top_path}")
        if before != after or before != current:
            raise RecoveryC8LaunchError(
                f"Critical C8 production input drifted after producer: {top_path}"
            )
    return launch_commit


def _make_record(
    repo: Path, launch: Path, schema: Path, document: Mapping[str, Any],
    c7_record: _c7.RecoveryC7LaunchRecord, formal: Path, reports: Path,
    selection: Path, launch_commit: str,
) -> RecoveryC8LaunchRecord:
    return RecoveryC8LaunchRecord(
        path=launch, schema_path=schema, document=_freeze(document),
        binding=c7_record.binding, c7_launch_record=c7_record,
        model_path=c7_record.model_path, data_root=c7_record.data_root,
        devkit_path=c7_record.devkit_path, formal_root=formal,
        reports_root=reports, selection_root=selection,
        report_path=reports / C8_REPORT_FILENAME,
        selection_path=selection / C8_SELECTION_FILENAME,
        model_identity=_freeze(document["model"]),
        file_sha256=sha256_file(launch), launch_commit=launch_commit,
        repo_root=repo,
    )


def generate_recovery_c8_launch_record(
    *, repo_root: Path | str, launch_record_path: Path | str,
    formal_root: Path | str, reports_root: Path | str,
    selection_root: Path | str,
    c7_launch_record_path: Path | str = DEFAULT_C7_LAUNCH_RECORD_PATH,
    c7_report_path: Path | str = DEFAULT_C7_REPORT_PATH,
    config_path: Path | str = DEFAULT_C8_CONFIG_PATH,
    run_spec_path: Path | str = DEFAULT_C8_RUN_SPEC_PATH,
    schema_path: Path | str = DEFAULT_SCHEMA_PATH,
) -> dict[str, Any]:
    """Generate a C8-only launch document from a pristine producer HEAD."""

    repo = Path(repo_root).resolve(strict=True)
    top, _prefix = _repository_identity(repo)
    launch, common, formal, reports, selection = _validate_output_paths(
        repo, launch_record_path, formal_root, reports_root, selection_root,
        require_fresh_roots=True, launch_must_exist=False,
    )
    if launch.exists() or launch.is_symlink():
        raise RecoveryC8LaunchError("Launch record path is not fresh")
    schema, schema_rel = _project_path(
        repo, schema_path, "C8 launch schema", require_exists=True,
    )
    if schema_rel != SCHEMA_RELATIVE_PATH:
        raise RecoveryC8LaunchError("Unexpected recovery C8 launch schema path")
    schema_document = _read_json(schema, "recovery C8 launch schema")
    try:
        Draft202012Validator.check_schema(schema_document)
    except Exception as error:
        raise RecoveryC8LaunchError("Recovery C8 launch schema is invalid") from error
    schema_sha256 = sha256_file(schema)
    c7_path, _c7_rel = _project_path(
        repo, c7_launch_record_path, "recovery C7 launch record",
        require_exists=True,
    )
    c7_record = _load_c7_authority(repo, c7_path, require_tracked=True)
    report_path, _report_rel = _project_path(
        repo, c7_report_path, "C7 PASS report", require_exists=True,
    )
    c7_report = _validate_c7_report(repo, top, report_path, c7_record)
    contracts = _contract_records(repo, c7_record, config_path, run_spec_path)
    producer = _capture_clean_producer(repo, top)
    for older, context in (
        (c7_record.document["stage_a"]["producer_commit"], "Stage A producer commit"),
        (c7_record.launch_commit, "C7 launch commit"),
        (c7_report["report_commit"], "C7 report commit"),
        (c7_record.document["amendment"]["commit"], "Recovery amendment commit"),
    ):
        _is_ancestor(repo, older, producer["commit"], context)
    _validate_lfs_coverage(repo, launch, formal, reports, selection)
    if _capture_clean_producer(repo, top) != producer:
        raise RecoveryC8LaunchError(
            "Producer Git state changed during C8 launch generation"
        )
    source = c7_record.document
    document: dict[str, Any] = {
        "$schema": "recovery_c8_launch_record.schema.json",
        "schema_version": 1,
        "document_type": DOCUMENT_TYPE,
        "record_id": RECORD_ID,
        "status": STATUS,
        "launch_record_commit": "SELF",
        "launch_schema": {"path": schema_rel, "file_sha256": schema_sha256},
        "amendment": copy.deepcopy(source["amendment"]),
        "recovery_binding": copy.deepcopy(source["recovery_binding"]),
        "stage_a": copy.deepcopy(source["stage_a"]),
        "c7_launch": _c7_launch_record(repo, c7_record),
        "c7_report": c7_report,
        "model": copy.deepcopy(source["model"]),
        "c8_contracts": contracts,
        "producer_git": producer,
        "outputs": {
            "common_root": common.relative_to(repo).as_posix(),
            "launch_record_path": launch.relative_to(repo).as_posix(),
            "formal_root": formal.relative_to(repo).as_posix(),
            "reports_root": reports.relative_to(repo).as_posix(),
            "selection_root": selection.relative_to(repo).as_posix(),
        },
        "scope": dict(_SCOPE),
    }
    document["canonical_digest"] = launch_record_digest(document)
    _validate_schema(document, schema_document)
    _validate_intrinsic_authority(document)
    if sha256_file(schema) != schema_sha256:
        raise RecoveryC8LaunchError(
            "Recovery C8 launch schema changed during generation"
        )
    return document


def write_recovery_c8_launch_record(
    *, repo_root: Path | str, launch_record_path: Path | str,
    formal_root: Path | str, reports_root: Path | str,
    selection_root: Path | str,
    c7_launch_record_path: Path | str = DEFAULT_C7_LAUNCH_RECORD_PATH,
    c7_report_path: Path | str = DEFAULT_C7_REPORT_PATH,
    config_path: Path | str = DEFAULT_C8_CONFIG_PATH,
    run_spec_path: Path | str = DEFAULT_C8_RUN_SPEC_PATH,
    schema_path: Path | str = DEFAULT_SCHEMA_PATH,
) -> RecoveryC8LaunchRecord:
    """Generate and exclusively publish an uncommitted C8 launch record."""

    repo = Path(repo_root).resolve(strict=True)
    document = generate_recovery_c8_launch_record(
        repo_root=repo, launch_record_path=launch_record_path,
        formal_root=formal_root, reports_root=reports_root,
        selection_root=selection_root,
        c7_launch_record_path=c7_launch_record_path,
        c7_report_path=c7_report_path, config_path=config_path,
        run_spec_path=run_spec_path, schema_path=schema_path,
    )
    launch, _common, formal, reports, selection = _validate_output_paths(
        repo, launch_record_path, formal_root, reports_root, selection_root,
        require_fresh_roots=True, launch_must_exist=False,
    )
    top, _prefix = _repository_identity(repo)
    if _capture_clean_producer(repo, top) != document["producer_git"]:
        raise RecoveryC8LaunchError(
            "Producer Git state changed before C8 launch publication"
        )
    _publish_exclusive(repo, launch, _json_file_bytes(document))
    c7_record = _load_c7_authority(
        repo, repo / document["c7_launch"]["path"], require_tracked=True,
    )
    return _make_record(
        repo, launch, repo / document["launch_schema"]["path"], document,
        c7_record, formal, reports, selection, "SELF",
    )


def load_recovery_c8_launch_record(
    path: Path | str, *, repo_root: Path | str = REPO_ROOT,
    schema_path: Path | str = DEFAULT_SCHEMA_PATH,
    require_fresh_roots: bool = True, require_tracked: bool = True,
) -> RecoveryC8LaunchRecord:
    """Reload every upstream source and validate a C8 launch authority."""

    repo = Path(repo_root).resolve(strict=True)
    top, _prefix = _repository_identity(repo)
    launch, launch_rel = _project_path(
        repo, path, "C8 launch record path", require_exists=True,
    )
    schema, schema_rel = _project_path(
        repo, schema_path, "C8 launch schema", require_exists=True,
    )
    if schema_rel != SCHEMA_RELATIVE_PATH:
        raise RecoveryC8LaunchError("Unexpected recovery C8 launch schema path")
    document = _read_json(launch, "recovery C8 launch record")
    schema_document = _read_json(schema, "recovery C8 launch schema")
    if not isinstance(document, Mapping):
        raise RecoveryC8LaunchError("Recovery C8 launch record must be an object")
    _validate_schema(document, schema_document)
    _validate_intrinsic_authority(document)
    if document["canonical_digest"] != launch_record_digest(document):
        raise RecoveryC8LaunchError(
            "Recovery C8 launch canonical digest drifted"
        )
    if document["launch_schema"] != {
        "path": schema_rel, "file_sha256": sha256_file(schema),
    }:
        raise RecoveryC8LaunchError("Recovery C8 launch schema binding drifted")
    if document["outputs"]["launch_record_path"] != launch_rel:
        raise RecoveryC8LaunchError("C8 launch record path binding drifted")
    outputs = document["outputs"]
    checked, common, formal, reports, selection = _validate_output_paths(
        repo, launch, outputs["formal_root"], outputs["reports_root"],
        outputs["selection_root"], require_fresh_roots=require_fresh_roots,
        launch_must_exist=True,
    )
    if outputs["common_root"] != common.relative_to(repo).as_posix():
        raise RecoveryC8LaunchError("C8 launch common-root binding drifted")
    _validate_lfs_coverage(repo, checked, formal, reports, selection)
    c7_path, _c7_rel = _project_path(
        repo, document["c7_launch"]["path"],
        "recovery C7 launch record", require_exists=True,
    )
    c7_record = _load_c7_authority(repo, c7_path, require_tracked=True)
    if canonical_json_bytes(document["c7_launch"]) != canonical_json_bytes(
        _c7_launch_record(repo, c7_record)
    ):
        raise RecoveryC8LaunchError("Recovery C7 launch handoff drifted")
    source = c7_record.document
    for field, message in (
        ("amendment", "Recovery amendment handoff drifted"),
        ("recovery_binding", "Recovery-v2 binding handoff drifted"),
        ("stage_a", "Stage A handoff drifted"),
        ("model", "Recovery-v2 model handoff drifted"),
    ):
        if canonical_json_bytes(document[field]) != canonical_json_bytes(source[field]):
            raise RecoveryC8LaunchError(message)
    report_path, _report_rel = _project_path(
        repo, document["c7_report"]["path"], "C7 PASS report",
        require_exists=True,
    )
    c7_report = _validate_c7_report(
        repo, top, report_path, c7_record, document["c7_report"],
    )
    contracts = _contract_records(
        repo, c7_record, document["c8_contracts"]["stage_c_config"]["path"],
        document["c8_contracts"]["run_spec"]["path"],
    )
    if canonical_json_bytes(document["c8_contracts"]) != canonical_json_bytes(contracts):
        raise RecoveryC8LaunchError("Recovery C8 contracts drifted")
    launch_commit = _validate_git_authority(
        repo, top, checked, document,
        _critical_paths(repo, top, schema, c7_record, c7_report, contracts),
        require_tracked=require_tracked,
    )
    return _make_record(
        repo, checked, schema, document, c7_record, formal, reports, selection,
        launch_commit,
    )


def validate_recovery_c8_launch_record(
    path: Path | str, *, repo_root: Path | str = REPO_ROOT,
    schema_path: Path | str = DEFAULT_SCHEMA_PATH,
    require_fresh_roots: bool = True, require_tracked: bool = True,
) -> RecoveryC8LaunchRecord:
    """CLI-friendly alias for source-recomputed C8 validation."""

    return load_recovery_c8_launch_record(
        path, repo_root=repo_root, schema_path=schema_path,
        require_fresh_roots=require_fresh_roots, require_tracked=require_tracked,
    )


def capture_authorized_c8_source_state(
    record: RecoveryC8LaunchRecord, allowed_phase: str,
) -> dict[str, Any]:
    """Validate phase-scoped evidence dirt and return logical clean code state."""

    if not isinstance(record, RecoveryC8LaunchRecord):
        raise RecoveryC8LaunchError("A validated RecoveryC8LaunchRecord is required")
    if allowed_phase not in _PHASE_ROOTS:
        raise RecoveryC8LaunchError(f"Unknown recovery C8 phase: {allowed_phase}")
    repo = record.repo_root.resolve(strict=True)
    top, _prefix = _repository_identity(repo)
    if allowed_phase == "pre_formal":
        _validate_output_paths(
            repo, record.path, record.formal_root, record.reports_root,
            record.selection_root, require_fresh_roots=True,
            launch_must_exist=True,
        )
    head_before = _run_git(repo, "rev-parse", "--verify", "HEAD^{commit}")
    if record.launch_commit == "SELF" or head_before != record.launch_commit:
        raise RecoveryC8LaunchError(
            "A committed C8 launch record at the current HEAD is required"
        )
    producer = record.document["producer_git"]
    source = repo / producer["source_subtree_path"]
    source_top = _top_relative(top, source)
    if _run_git(repo, "rev-parse", f"HEAD:{source_top}") != producer["source_subtree_tree"]:
        raise RecoveryC8LaunchError("C8 producer source subtree drifted")
    status_before = _git_status_entries(repo)
    _verify_lfs_pointer_at_head(
        repo, top, record.path, expected_sha256=record.file_sha256,
    )
    allowed_roots = tuple(
        getattr(record, name) for name in _PHASE_ROOTS[allowed_phase]
    )
    allowed_prefixes = tuple(
        _top_relative(top, root).rstrip("/") + "/" for root in allowed_roots
    )
    allowed_exact = {_top_relative(top, root) for root in allowed_roots}
    launch_top = _top_relative(top, record.path)
    for entry in status_before:
        paths = [entry["path"]]
        if "source_path" in entry:
            paths.append(entry["source_path"])
        if launch_top in paths:
            raise RecoveryC8LaunchError("C8 launch record itself is dirty")
        if entry.get("status") != "??" or "source_path" in entry:
            raise RecoveryC8LaunchError(
                "Recovery C8 evidence dirt must be newly created and untracked"
            )
        if any(
            path not in allowed_exact and not path.startswith(allowed_prefixes)
            for path in paths
        ):
            raise RecoveryC8LaunchError(
                f"Git dirt is outside roots authorized for {allowed_phase}"
            )
    head_after = _run_git(repo, "rev-parse", "--verify", "HEAD^{commit}")
    status_after = _git_status_entries(repo)
    if head_after != head_before or status_after != status_before:
        raise RecoveryC8LaunchError(
            "Git HEAD or status changed during recovery C8 source capture"
        )
    return {"commit": head_before, "dirty": False}


def recovery_provenance_context(
    record: RecoveryC8LaunchRecord,
) -> dict[str, Any]:
    """Return the existing validated C7 pair consumed by provenance code."""

    if not isinstance(record, RecoveryC8LaunchRecord):
        raise RecoveryC8LaunchError("A validated RecoveryC8LaunchRecord is required")
    return {
        "recovery_binding": record.binding,
        "recovery_launch_record": record.c7_launch_record,
    }


__all__ = [
    "C8_REPORT_FILENAME", "C8_SELECTION_FILENAME",
    "DEFAULT_C7_LAUNCH_RECORD_PATH", "DEFAULT_C7_REPORT_PATH",
    "DEFAULT_C8_CONFIG_PATH", "DEFAULT_C8_RUN_SPEC_PATH",
    "DEFAULT_SCHEMA_PATH", "DOCUMENT_TYPE", "LAUNCH_RECORD_FILENAME",
    "RECORD_ID", "STATUS", "RecoveryC8LaunchError",
    "RecoveryC8LaunchRecord", "canonical_json_bytes", "canonical_sha256",
    "capture_authorized_c8_source_state",
    "generate_recovery_c8_launch_record", "launch_record_digest",
    "load_recovery_c8_launch_record", "recovery_provenance_context",
    "validate_recovery_c8_launch_record", "write_recovery_c8_launch_record",
]
