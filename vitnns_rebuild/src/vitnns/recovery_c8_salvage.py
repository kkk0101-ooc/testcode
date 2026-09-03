"""Fail-closed authority for recovery-v2 C8 salvage.

The authority binds one immutable interrupted C8 source and one fresh gap
root.  It authorizes only calibration-unit verification, gap execution, and
machine selection.  It never reads images or initializes CUDA.
"""

from __future__ import annotations

import json
import os
import re
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from jsonschema import Draft202012Validator

from .constants import REPO_ROOT
from .io_utils import sha256_file
from .recovery_c8_launch import (
    RecoveryC8LaunchRecord, load_historical_recovery_c8_launch_record,
    _contract_records,
)
from .stage_c_provenance import canonical_sha256

DEFAULT_SCHEMA_PATH = REPO_ROOT / "config/recovery_c8_salvage_launch_record.schema.json"
DEFAULT_AMENDMENT_PATH = REPO_ROOT / "docs/recovery_c8_salvage_amendment_20260902.md"
DEFAULT_BASE_C8_LAUNCH_PATH = (
    REPO_ROOT / "evidence/stage_c/c8_recovery_v2_a342003/recovery_c8_launch_record.json"
)
DEFAULT_SOURCE_ROOT = REPO_ROOT / "evidence/stage_c/c8_recovery_v2_a342003/formal"
SOURCE_EVIDENCE_COMMIT = "e0edf3e51906b45feff094174921ca8bfd7eacf8"
DOCUMENT_TYPE = "vitnns_recovery_v2_c8_salvage_launch_record"
RECORD_ID = "RECOVERY_V2_C8_SALVAGE_20260902"
STATUS = "AUTHORIZED_FOR_RECOVERY_C8_SALVAGE_ONLY"
LAUNCH_FILENAME = "recovery_c8_salvage_launch_record.json"
SCHEMA_VERSION = 2
SOURCE_SUBTREE_PATH = "src/vitnns"
DECLARED_EVIDENCE_ROOT_KEYS = (
    "gap_root", "reports_root", "selection_root", "logs_root",
)
POST_LAUNCH_HANDOFF_PATHS = (
    "docs/recovery_and_rerun.md", "docs/阶段C-step-by-step.md",
)
CONTROL_NAMES = (
    "execution_plan.json", "preflight.json", "c8_launch_authorization.json",
    "failure_table.json", "batch_status.json",
)


class RecoveryC8SalvageError(ValueError):
    """Raised when salvage authority or custody validation fails."""


def _plain(value: Any) -> Any:
    if value is None or type(value) in {str, bool, int, float}:
        return value
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    method = getattr(value, "to_dict", None)
    if callable(method):
        return _plain(method())
    raise TypeError(f"value is not JSON-compatible: {type(value).__name__}")


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(k): _freeze(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _read_json(
    path: Path, context: str, *, maximum: int = 16 * 1024 * 1024,
) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise RecoveryC8SalvageError(f"duplicate JSON key in {context}: {key}")
            result[key] = value
        return result

    try:
        raw = path.read_bytes()
        if len(raw) > maximum:
            raise RecoveryC8SalvageError(f"{context} is too large")
        value = json.loads(
            raw.decode("utf-8"), object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON value {token}")
            ),
        )
    except RecoveryC8SalvageError:
        raise
    except Exception as error:
        raise RecoveryC8SalvageError(f"invalid {context}: {error}") from error
    if not isinstance(value, dict):
        raise RecoveryC8SalvageError(f"{context} must be an object")
    return value


def _execution_plan_digest(path: Path) -> str:
    """Read the leading self digest without materializing the 590 MiB plan."""

    try:
        with path.open("rb") as handle:
            prefix = handle.read(4096)
    except OSError as error:
        raise RecoveryC8SalvageError("cannot read source execution plan") from error
    match = re.search(rb'"canonical_digest"\s*:\s*"([0-9a-f]{64})"', prefix)
    if match is None:
        raise RecoveryC8SalvageError("source execution plan lacks a leading digest")
    return match.group(1).decode("ascii")


def _git(repo: Path, *args: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(repo), *args], check=True, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as error:
        raise RecoveryC8SalvageError(
            f"git {' '.join(args)} failed: {error}"
        ) from error


def _git_bytes(repo: Path, *args: str) -> bytes:
    """Run Git without stripping porcelain delimiters or leading status bytes."""

    try:
        return subprocess.run(
            ["git", "-C", str(repo), *args], check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120,
        ).stdout
    except (OSError, subprocess.SubprocessError) as error:
        raise RecoveryC8SalvageError(
            f"git {' '.join(args)} failed: {error}"
        ) from error


def _project_path(repo: Path, raw: Path | str, context: str, *, exists: bool = False) -> Path:
    value = Path(raw)
    path = value if value.is_absolute() else repo / value
    path = Path(os.path.abspath(os.fspath(path)))
    try:
        path.relative_to(repo)
    except ValueError as error:
        raise RecoveryC8SalvageError(f"{context} escapes repository") from error
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if os.path.lexists(current) and stat.S_ISLNK(current.lstat().st_mode):
            raise RecoveryC8SalvageError(f"{context} contains a symlink")
    if exists and not path.exists():
        raise RecoveryC8SalvageError(f"{context} does not exist")
    return path


def _rel(repo: Path, path: Path) -> str:
    return path.relative_to(repo).as_posix()


def _tree_inventory(repo: Path, root: Path, *, revision: str | None = None) -> dict[str, Any]:
    """Hash a closed ordinary-file tree in bytewise relative-path order."""

    if revision is not None:
        top_relative = _rel(repo.parent, root)
        # Git LFS clean filters compare hydrated payloads with the committed
        # pointer identity.  One closed pathspec check avoids trusting the
        # index and avoids one subprocess per evidence file.
        _git(repo, "diff", "--quiet", revision, "--", f":(top){top_relative}")
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix().encode()):
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            if stat.S_ISDIR(info.st_mode):
                continue
            raise RecoveryC8SalvageError(f"unsafe source evidence entry: {path}")
        if info.st_nlink != 1:
            raise RecoveryC8SalvageError(f"source evidence has a hard-linked file: {path}")
        relative = path.relative_to(root).as_posix()
        records.append({"path": relative, "bytes": info.st_size, "sha256": sha256_file(path)})
    return {
        "algorithm": "sha256_canonical_json_path_bytes_sha256_v1",
        "ordering": "relative_path_utf8_bytewise_ascending",
        "file_count": len(records),
        "total_bytes": sum(item["bytes"] for item in records),
        "digest": canonical_sha256({"domain": "vitnns-c8-salvage-root-v1", "files": records}),
    }


@dataclass(frozen=True)
class RecoveryC8SalvageLaunchRecord:
    path: Path
    schema_path: Path
    document: Mapping[str, Any]
    binding: Any
    c7_launch_record: Any
    source_c8_record: RecoveryC8LaunchRecord
    source_root: Path
    gap_root: Path
    reports_root: Path
    selection_root: Path
    logs_root: Path
    combined_manifest_path: Path
    combined_report_path: Path
    selection_path: Path
    selection_report_path: Path
    model_path: Path
    data_root: Path
    launch_commit: str
    file_sha256: str
    repo_root: Path = REPO_ROOT

    @property
    def canonical_digest(self) -> str:
        return str(self.document["canonical_digest"])

    @property
    def authorization_launch_commit(self) -> str:
        """Commit that authorized execution, distinct from a later runtime HEAD."""

        return self.launch_commit


def _capture_producer(repo: Path) -> dict[str, Any]:
    if _git(repo, "status", "--porcelain=v1"):
        raise RecoveryC8SalvageError("salvage producer worktree must be clean")
    commit = _git(repo, "rev-parse", "HEAD^{commit}")
    top = Path(_git(repo, "rev-parse", "--show-toplevel"))
    source_top = (repo / SOURCE_SUBTREE_PATH).relative_to(top).as_posix()
    return {
        "commit": commit, "tree": _git(repo, "rev-parse", "HEAD^{tree}"),
        "source_subtree_path": SOURCE_SUBTREE_PATH,
        "source_subtree_tree": _git(repo, "rev-parse", f"HEAD:{source_top}"),
        "dirty": False,
    }


def _nul_paths(payload: bytes, context: str) -> list[str]:
    if payload and not payload.endswith(b"\0"):
        raise RecoveryC8SalvageError(f"invalid NUL-delimited Git output for {context}")
    try:
        return [
            item.decode("utf-8") for item in payload.split(b"\0") if item
        ]
    except UnicodeDecodeError as error:
        raise RecoveryC8SalvageError(
            f"non-UTF-8 repository path in {context}"
        ) from error


def _changed_paths(repo: Path, older: str, newer: str) -> list[str]:
    return _nul_paths(
        _git_bytes(
            repo, "diff", "--name-only", "--no-renames", "-z",
            older, newer, "--",
        ),
        "committed change inventory",
    )


def _worktree_dirty_paths(repo: Path) -> list[str]:
    unstaged = _nul_paths(
        _git_bytes(
            repo, "diff", "--name-only", "--no-renames", "-z",
            "HEAD", "--",
        ),
        "unstaged worktree change inventory",
    )
    staged = _nul_paths(
        _git_bytes(
            repo, "diff", "--cached", "--name-only",
            "--no-renames", "-z", "HEAD", "--",
        ),
        "staged worktree change inventory",
    )
    untracked = _nul_paths(
        _git_bytes(
            repo, "ls-files", "--full-name", "--others",
            "--exclude-standard", "-z", "--",
        ),
        "untracked worktree change inventory",
    )
    return sorted(
        set(unstaged + staged + untracked),
        key=lambda item: item.encode("utf-8"),
    )


def _is_under_declared_root(path: str, roots: tuple[str, ...]) -> bool:
    return any(path == root or path.startswith(root + "/") for root in roots)


def _validated_output_paths(
    repo: Path, launch: Path, outputs: Mapping[str, Any],
) -> tuple[Path, Path, Path, Path, tuple[str, ...]]:
    """Resolve the closed set of roots in which runtime evidence may change."""

    common = _project_path(repo, outputs["common_root"], "salvage common root")
    if common != launch.parent:
        raise RecoveryC8SalvageError(
            "salvage common root does not contain the launch record"
        )
    common_relative = common.relative_to(repo)
    if (
        len(common_relative.parts) < 3
        or common_relative.parts[:2] != ("evidence", "stage_c")
    ):
        raise RecoveryC8SalvageError(
            "salvage common root must be under evidence/stage_c"
        )
    if launch.name != LAUNCH_FILENAME:
        raise RecoveryC8SalvageError(
            f"salvage launch record must be named {LAUNCH_FILENAME}"
        )
    declared_launch = _project_path(
        repo, outputs["launch_record_path"], "declared salvage launch record",
    )
    if declared_launch != launch:
        raise RecoveryC8SalvageError("salvage launch output binding drifted")

    expected_names = {
        "gap_root": "gap",
        "reports_root": "reports",
        "selection_root": "selection",
        "logs_root": "logs",
    }
    resolved: dict[str, Path] = {}
    for key in DECLARED_EVIDENCE_ROOT_KEYS:
        root = _project_path(repo, outputs[key], f"declared {key}")
        if root.parent != common or root.name != expected_names[key]:
            raise RecoveryC8SalvageError(
                f"{key} must be the {expected_names[key]} child of the common root"
            )
        resolved[key] = root
    if len(set(resolved.values())) != len(resolved):
        raise RecoveryC8SalvageError("salvage evidence roots must be distinct")

    expected_files = {
        "combined_manifest_path": resolved["reports_root"]
        / "c8_salvage_combined_manifest.json",
        "combined_report_path": resolved["reports_root"]
        / "c8_salvage_combined_report.json",
        "selection_path": resolved["selection_root"]
        / "stage_c_salvage_selection.json",
        "selection_report_path": resolved["reports_root"]
        / "stage_c_c8_salvage_selection_report.json",
    }
    for key, expected in expected_files.items():
        if _project_path(repo, outputs[key], f"declared {key}") != expected:
            raise RecoveryC8SalvageError(f"salvage {key} binding drifted")

    top = Path(_git(repo, "rev-parse", "--show-toplevel"))
    allowed = tuple(_rel(top, resolved[key]) for key in DECLARED_EVIDENCE_ROOT_KEYS)
    return (
        resolved["gap_root"], resolved["reports_root"],
        resolved["selection_root"], resolved["logs_root"], allowed,
    )


def _head_identity(repo: Path) -> dict[str, Any]:
    top = Path(_git(repo, "rev-parse", "--show-toplevel"))
    source_top = (repo / SOURCE_SUBTREE_PATH).relative_to(top).as_posix()
    return {
        "commit": _git(repo, "rev-parse", "--verify", "HEAD^{commit}"),
        "tree": _git(repo, "rev-parse", "--verify", "HEAD^{tree}"),
        "source_subtree_path": SOURCE_SUBTREE_PATH,
        "source_subtree_tree": _git(
            repo, "rev-parse", "--verify", f"HEAD:{source_top}",
        ),
    }


def _resolve_tracked_launch_commit(
    repo: Path, launch: Path, declared: str, producer_commit: str,
) -> str:
    """Resolve SELF and enforce the one-file, direct-parent launch commit."""

    top = Path(_git(repo, "rev-parse", "--show-toplevel"))
    launch_top = _rel(top, launch)
    launch_commit = declared
    if launch_commit == "SELF":
        launch_commit = _git(
            repo, "log", "-n", "1", "--format=%H", "HEAD", "--",
            f":(top){launch_top}",
        )
    parents = _git(
        repo, "rev-list", "--parents", "-n", "1", launch_commit,
    ).split()
    if parents != [launch_commit, producer_commit]:
        raise RecoveryC8SalvageError(
            "salvage launch commit must have the producer as its direct parent"
        )
    if _changed_paths(repo, producer_commit, launch_commit) != [launch_top]:
        raise RecoveryC8SalvageError(
            "salvage launch commit must change only its launch record"
        )
    return launch_commit


def _validate_runtime_git_identity(
    repo: Path, *, launch_commit: str, producer: Mapping[str, Any],
    declared_evidence_roots: tuple[str, ...], check_worktree: bool,
    declared_committed_paths: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Validate and return the real execution identity at the current HEAD."""

    identity = _head_identity(repo)
    try:
        _git(
            repo, "merge-base", "--is-ancestor", launch_commit,
            identity["commit"],
        )
    except RecoveryC8SalvageError as error:
        raise RecoveryC8SalvageError(
            "salvage runtime HEAD does not descend from launch commit"
        ) from error
    if identity["source_subtree_tree"] != producer["source_subtree_tree"]:
        raise RecoveryC8SalvageError(
            "salvage runtime source subtree differs from the authorized producer"
        )

    committed = _changed_paths(repo, launch_commit, identity["commit"])
    escaped = [
        path for path in committed
        if (
            not _is_under_declared_root(path, declared_evidence_roots)
            and path not in declared_committed_paths
        )
    ]
    if escaped:
        raise RecoveryC8SalvageError(
            "committed changes after salvage launch escape declared evidence roots: "
            + escaped[0]
        )

    dirty_paths = _worktree_dirty_paths(repo) if check_worktree else []
    escaped = [
        path for path in dirty_paths
        if not _is_under_declared_root(path, declared_evidence_roots)
    ]
    if escaped:
        raise RecoveryC8SalvageError(
            "salvage runtime has worktree dirt outside declared evidence roots: "
            + escaped[0]
        )
    # Declared evidence may be in flight, but the authorized code snapshot is
    # still clean because source-subtree worktree dirt was rejected above.
    identity["dirty"] = False
    identity["evidence_worktree_dirty"] = bool(dirty_paths)
    identity["authorization_launch_commit"] = launch_commit
    return identity


def _base_record(repo: Path, path: Path) -> RecoveryC8LaunchRecord:
    """Validate the historical authority against its immutable Git commits."""

    try:
        return load_historical_recovery_c8_launch_record(
            path, repo_root=repo, custody_commit=SOURCE_EVIDENCE_COMMIT,
        )
    except Exception as error:
        raise RecoveryC8SalvageError(
            f"historical base C8 authority is invalid: {error}"
        ) from error


def generate_recovery_c8_salvage_launch_record(
    *, repo_root: Path | str, launch_record_path: Path | str,
    gap_root: Path | str, reports_root: Path | str, selection_root: Path | str,
    source_root: Path | str = DEFAULT_SOURCE_ROOT,
    base_c8_launch_path: Path | str = DEFAULT_BASE_C8_LAUNCH_PATH,
    amendment_path: Path | str = DEFAULT_AMENDMENT_PATH,
    schema_path: Path | str = DEFAULT_SCHEMA_PATH,
) -> dict[str, Any]:
    repo = Path(repo_root).resolve(strict=True)
    launch = _project_path(repo, launch_record_path, "salvage launch path")
    schema = _project_path(repo, schema_path, "salvage schema", exists=True)
    amendment = _project_path(repo, amendment_path, "salvage amendment", exists=True)
    source = _project_path(repo, source_root, "interrupted C8 root", exists=True)
    base_path = _project_path(repo, base_c8_launch_path, "base C8 authority", exists=True)
    if launch.exists():
        raise RecoveryC8SalvageError("salvage launch record path is not fresh")
    common = launch.parent
    gap = _project_path(repo, gap_root, "fresh gap root")
    reports = _project_path(repo, reports_root, "reports root")
    selection = _project_path(repo, selection_root, "selection root")
    if any(path.parent != common for path in (gap, reports, selection)):
        raise RecoveryC8SalvageError("salvage output roots must share the launch parent")
    if any(path.exists() for path in (gap, reports, selection)):
        raise RecoveryC8SalvageError("salvage output roots must be fresh")
    if common.exists() and any(common.iterdir()):
        raise RecoveryC8SalvageError("salvage common root must be empty")
    base = _base_record(repo, base_path)
    controls = {name: sha256_file(source / name) for name in CONTROL_NAMES}
    authorization = _read_json(source / "c8_launch_authorization.json", "source authorization")
    plan_digest = _execution_plan_digest(source / "execution_plan.json")
    if plan_digest != authorization.get("c8_execution_plan_sha256"):
        raise RecoveryC8SalvageError("source plan/authorization binding drifted")
    custody_rel = _rel(repo.parent, source)
    tree = _git(repo, "rev-parse", f"{SOURCE_EVIDENCE_COMMIT}:{custody_rel}")
    producer = _capture_producer(repo)
    amendment_commit = _git(repo, "log", "-n", "1", "--format=%H", "--", f":(top){_rel(repo.parent, amendment)}")
    base_doc = base.document
    contracts = _contract_records(
        repo, base.c7_launch_record, base_doc["c8_contracts"]["stage_c_config"]["path"],
        base_doc["c8_contracts"]["run_spec"]["path"],
    )
    document: dict[str, Any] = {
        "$schema": schema.name, "schema_version": SCHEMA_VERSION,
        "document_type": DOCUMENT_TYPE, "record_id": RECORD_ID, "status": STATUS,
        "launch_record_commit": "SELF",
        "launch_schema": {"path": _rel(repo, schema), "file_sha256": sha256_file(schema)},
        "amendment": {"id": "RECOVERY-C8-SALVAGE-20260902", "path": _rel(repo, amendment), "file_sha256": sha256_file(amendment), "containing_commit": amendment_commit},
        "base_c8_authority": {"path": _rel(repo, base.path), "file_sha256": base.file_sha256, "canonical_digest": base.canonical_digest, "launch_commit": base.launch_commit, "producer_commit": base_doc["producer_git"]["commit"]},
        "source_interrupted_root": {"formal_root": _rel(repo, source), "evidence_commit": SOURCE_EVIDENCE_COMMIT, "tree_object": tree, "execution_commit": authorization["source"]["execution_git_state"]["commit"], "plan_digest": plan_digest, "control_sha256": controls},
        "contracts": {"config": contracts["stage_c_config"], "run_spec": contracts["run_spec"]},
        "model": {key: base_doc["model"][key] for key in ("identity", "state_dict_path", "state_dict_sha256", "canonical_tensor_digest", "historical_artifact_recovered")},
        "producer_git": producer,
        "outputs": {
            "common_root": _rel(repo, common), "launch_record_path": _rel(repo, launch),
            "gap_root": _rel(repo, gap), "reports_root": _rel(repo, reports),
            "selection_root": _rel(repo, selection),
            "logs_root": _rel(repo, common / "logs"),
            "post_launch_handoff_paths": list(POST_LAUNCH_HANDOFF_PATHS),
            "combined_manifest_path": _rel(repo, reports / "c8_salvage_combined_manifest.json"),
            "combined_report_path": _rel(repo, reports / "c8_salvage_combined_report.json"),
            "selection_path": _rel(repo, selection / "stage_c_salvage_selection.json"),
            "selection_report_path": _rel(repo, reports / "stage_c_c8_salvage_selection_report.json"),
        },
        "policy": {"expected_unit_count": 2180, "source_root_read_only": True, "source_resume_allowed": False, "artifact_copy_allowed": False, "virtual_union": True, "source_recomputation_required": True},
        "scope": {"c8_salvage_authorized": True, "holdout_access_authorized": False, "c9_authorized": False, "c10_authorized": False, "stage_d_authorized": False},
    }
    document["canonical_digest"] = canonical_sha256(document)
    Draft202012Validator(_read_json(schema, "salvage schema")).validate(document)
    return document


def write_recovery_c8_salvage_launch_record(**kwargs: Any) -> RecoveryC8SalvageLaunchRecord:
    repo = Path(kwargs["repo_root"]).resolve(strict=True)
    path = _project_path(repo, kwargs["launch_record_path"], "salvage launch path")
    document = generate_recovery_c8_salvage_launch_record(**kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False).encode() + b"\n")
        handle.flush(); os.fsync(handle.fileno())
    return load_recovery_c8_salvage_launch_record(path, repo_root=repo, require_tracked=False)


def load_recovery_c8_salvage_launch_record(
    path: Path | str, *, repo_root: Path | str = REPO_ROOT,
    schema_path: Path | str = DEFAULT_SCHEMA_PATH, require_tracked: bool = True,
    require_fresh_outputs: bool = False,
) -> RecoveryC8SalvageLaunchRecord:
    repo = Path(repo_root).resolve(strict=True)
    launch = _project_path(repo, path, "salvage launch record", exists=True)
    schema = _project_path(repo, schema_path, "salvage schema", exists=True)
    document = _read_json(launch, "salvage launch record")
    Draft202012Validator(_read_json(schema, "salvage schema")).validate(document)
    if document["canonical_digest"] != canonical_sha256({k: v for k, v in document.items() if k != "canonical_digest"}):
        raise RecoveryC8SalvageError("salvage launch canonical digest drifted")
    if document["launch_schema"] != {"path": _rel(repo, schema), "file_sha256": sha256_file(schema)}:
        raise RecoveryC8SalvageError("salvage schema binding drifted")
    amendment = repo / document["amendment"]["path"]
    if sha256_file(amendment) != document["amendment"]["file_sha256"]:
        raise RecoveryC8SalvageError("salvage amendment bytes drifted")
    amendment_commit = document["amendment"]["containing_commit"]
    try:
        if _git(repo, "rev-parse", f"{amendment_commit}^{{commit}}") != amendment_commit:
            raise RecoveryC8SalvageError(
                "salvage amendment containing commit identity drifted"
            )
        committed_amendment = _git_bytes(
            repo, "show",
            f"{amendment_commit}:{_rel(repo.parent, amendment)}",
        )
    except RecoveryC8SalvageError:
        raise
    if committed_amendment != amendment.read_bytes():
        raise RecoveryC8SalvageError(
            "salvage amendment differs from its containing commit"
        )
    base = _base_record(repo, repo / document["base_c8_authority"]["path"])
    expected_base = {"path": _rel(repo, base.path), "file_sha256": base.file_sha256, "canonical_digest": base.canonical_digest, "launch_commit": base.launch_commit, "producer_commit": base.document["producer_git"]["commit"]}
    if document["base_c8_authority"] != expected_base:
        raise RecoveryC8SalvageError("base C8 authority binding drifted")
    source = repo / document["source_interrupted_root"]["formal_root"]
    source_binding = document["source_interrupted_root"]
    if _git(repo, "rev-parse", f"{source_binding['evidence_commit']}:{_rel(repo.parent, source)}") != source_binding["tree_object"]:
        raise RecoveryC8SalvageError("source custody tree drifted")
    for name, digest in source_binding["control_sha256"].items():
        if sha256_file(source / name) != digest:
            raise RecoveryC8SalvageError(f"source control drifted: {name}")
    outputs = document["outputs"]
    if tuple(outputs["post_launch_handoff_paths"]) != POST_LAUNCH_HANDOFF_PATHS:
        raise RecoveryC8SalvageError(
            "salvage post-launch handoff path contract drifted"
        )
    gap, reports, selection, logs, evidence_roots = _validated_output_paths(
        repo, launch, outputs,
    )
    if require_fresh_outputs and any(
        item.exists() for item in (gap, reports, selection, logs)
    ):
        raise RecoveryC8SalvageError("salvage output roots are not fresh")
    producer = document["producer_git"]
    top = Path(_git(repo, "rev-parse", "--show-toplevel"))
    source_top = (repo / SOURCE_SUBTREE_PATH).relative_to(top).as_posix()
    if _git(repo, "rev-parse", f"{producer['commit']}^{{tree}}") != producer["tree"] or _git(repo, "rev-parse", f"{producer['commit']}:{source_top}") != producer["source_subtree_tree"]:
        raise RecoveryC8SalvageError("salvage producer Git identity drifted")
    try:
        _git(
            repo, "merge-base", "--is-ancestor",
            amendment_commit, producer["commit"],
        )
    except RecoveryC8SalvageError as error:
        raise RecoveryC8SalvageError(
            "salvage amendment commit is not an ancestor of the producer"
        ) from error
    launch_commit = document["launch_record_commit"]
    if require_tracked:
        launch_commit = _resolve_tracked_launch_commit(
            repo, launch, launch_commit, producer["commit"],
        )
        _validate_runtime_git_identity(
            repo, launch_commit=launch_commit, producer=producer,
            declared_evidence_roots=evidence_roots,
            declared_committed_paths=tuple(
                _rel(top, repo / path)
                for path in outputs["post_launch_handoff_paths"]
            ),
            check_worktree=True,
        )
    else:
        if launch_commit != "SELF" or _git(repo, "rev-parse", "HEAD") != producer["commit"]:
            raise RecoveryC8SalvageError("uncommitted salvage authority requires producer HEAD")
    return RecoveryC8SalvageLaunchRecord(
        path=launch, schema_path=schema, document=_freeze(document), binding=base.binding,
        c7_launch_record=base.c7_launch_record, source_c8_record=base, source_root=source,
        gap_root=gap, reports_root=reports, selection_root=selection, logs_root=logs,
        combined_manifest_path=repo / outputs["combined_manifest_path"],
        combined_report_path=repo / outputs["combined_report_path"],
        selection_path=repo / outputs["selection_path"],
        selection_report_path=repo / outputs["selection_report_path"],
        model_path=base.model_path, data_root=base.data_root, launch_commit=launch_commit,
        file_sha256=sha256_file(launch), repo_root=repo,
    )


def validate_recovery_c8_salvage_launch_record(*args: Any, **kwargs: Any) -> RecoveryC8SalvageLaunchRecord:
    return load_recovery_c8_salvage_launch_record(*args, **kwargs)


def capture_authorized_c8_salvage_source_state(
    record: RecoveryC8SalvageLaunchRecord, phase: str = "runtime",
) -> dict[str, Any]:
    identity = capture_c8_salvage_verifier_identity(record, phase)
    return {"commit": identity["commit"], "dirty": False}


def capture_c8_salvage_verifier_identity(
    record: RecoveryC8SalvageLaunchRecord, phase: str = "runtime",
) -> dict[str, Any]:
    """Return full, verified runtime Git identity for audit records."""

    authoritative = load_recovery_c8_salvage_launch_record(record.path, repo_root=record.repo_root)
    top = Path(_git(record.repo_root, "rev-parse", "--show-toplevel"))
    evidence_roots = tuple(
        _rel(top, path) for path in (
            authoritative.gap_root, authoritative.reports_root,
            authoritative.selection_root, authoritative.logs_root,
        )
    )
    try:
        return _validate_runtime_git_identity(
            record.repo_root, launch_commit=authoritative.launch_commit,
            producer=authoritative.document["producer_git"],
            declared_evidence_roots=evidence_roots,
            declared_committed_paths=tuple(
                _rel(top, authoritative.repo_root / path)
                for path in authoritative.document["outputs"][
                    "post_launch_handoff_paths"
                ]
            ),
            check_worktree=True,
        )
    except RecoveryC8SalvageError as error:
        raise RecoveryC8SalvageError(f"salvage {phase}: {error}") from error


def recovery_provenance_context(record: RecoveryC8SalvageLaunchRecord) -> RecoveryC8SalvageLaunchRecord:
    if type(record) is not RecoveryC8SalvageLaunchRecord:
        raise RecoveryC8SalvageError("validated salvage authority required")
    return record


__all__ = [
    "DEFAULT_SCHEMA_PATH", "RecoveryC8SalvageError",
    "RecoveryC8SalvageLaunchRecord", "capture_authorized_c8_salvage_source_state",
    "capture_c8_salvage_verifier_identity",
    "generate_recovery_c8_salvage_launch_record", "load_recovery_c8_salvage_launch_record",
    "recovery_provenance_context", "validate_recovery_c8_salvage_launch_record",
    "write_recovery_c8_salvage_launch_record", "_tree_inventory",
]
