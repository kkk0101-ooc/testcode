"""Fail-closed launch authority for repair of an interrupted C8 salvage.

This module is metadata-only.  It binds the original interrupted C8 source,
the prior partial salvage root at an immutable custody commit, and fresh
repair outputs.  It never loads images, tensors, or CUDA.
"""

from __future__ import annotations

import json
import os
import re
import stat
import subprocess
from hashlib import sha256
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from jsonschema import Draft202012Validator

from .constants import REPO_ROOT
from .io_utils import sha256_file
from . import recovery_c8_salvage as _salvage
from .recovery_c8_salvage import RecoveryC8SalvageLaunchRecord
from .stage_c_provenance import canonical_sha256

DEFAULT_SCHEMA_PATH = (
    REPO_ROOT / "config/recovery_c8_salvage_repair_launch_record.schema.json"
)
DEFAULT_AMENDMENT_PATH = (
    REPO_ROOT / "docs/recovery_c8_salvage_repair_amendment_20260903.md"
)
DEFAULT_ORIGINAL_SALVAGE_AUTHORITY_PATH = (
    REPO_ROOT / "evidence/stage_c/c8_salvage_recovery_v2_ac77b47/"
    "recovery_c8_salvage_launch_record.json"
)
DEFAULT_PRIOR_PARTIAL_ROOT = (
    REPO_ROOT / "evidence/stage_c/c8_salvage_recovery_v2_ac77b47"
)
PRIOR_PARTIAL_CUSTODY_COMMIT = "5e5ea9d01574c4ecf9f9e72f436c7063eedfe665"
PRIOR_PARTIAL_COMMON_TREE = "d139579c7c51ec57a93fb7ca4557740a5d3659db"
PRIOR_PARTIAL_GAP_TREE = "e24d8ea4c1287bf52300c2deed96981128b87510"
ORIGINAL_SALVAGE_FILE_SHA256 = (
    "cfe680f1c7cb6d66b0f4918dc735ec2b98417346e16393de2264d5dce1eb29ef"
)
ORIGINAL_SALVAGE_CANONICAL_DIGEST = (
    "8ba38ed4a10b06b601bb25ff0bcd1e9af2908ea59fcbf91cf28f52486fffdf62"
)
ORIGINAL_SALVAGE_LAUNCH_COMMIT = "6a6655b604f7310e298490837cac7e66ebbf2744"
ORIGINAL_SALVAGE_PRODUCER_COMMIT = "ac77b4715ed2524b803f9e681f15b5afc96ee78f"
PRIOR_PARTIAL_CONTROL_SHA256 = {
    "execution_plan.json": (
        "da24efd1a700c59ae378a87b7b5b575f5d09edcbf40111c515f1321b2db899f9"
    ),
    "preflight.json": (
        "f50dfa0b9dba0ecb22fefa638cea4bc35d0cd888738c2a4f343e8e6f77add7ee"
    ),
    "salvage_scope.json": (
        "056650484144f8d8f2c86a04c9d9cee115e76b59ca5219560f40a3f8a38d0e61"
    ),
    "salvage_status.json": (
        "cbd80af508da7b8fd488dd4e4752d8e1889c6c07b88ea4e2fcf5a69704b3675c"
    ),
    "c8_salvage_source_audit.json": (
        "3e2e38e0a68e9e5272773c4c3468c8fc38a40af4cd79ae4eb518ad24bda9f39d"
    ),
    "c8_partial_scientific_failure_diagnosis.json": (
        "4425f0f2b29bd8dc53fef812e055c8f3bc17f49c4a17fdb3d259f3535fc4655b"
    ),
}
PRIOR_PARTIAL_GAP_INVENTORY = {
    "algorithm": "sha256_canonical_json_path_bytes_sha256_v1",
    "ordering": "relative_path_utf8_bytewise_ascending",
    "file_count": 6421,
    "total_bytes": 1121468359,
    "digest": "56b3eda4a23ea15506252aae38bc7e93be53f0942f5dde306140e92d44cdae2e",
}
DOCUMENT_TYPE = "vitnns_recovery_c8_salvage_repair_launch_record"
RECORD_ID = "RECOVERY_C8_SALVAGE_REPAIR_20260903"
STATUS = "AUTHORIZED_FOR_RECOVERY_C8_SALVAGE_REPAIR_ONLY"
LAUNCH_FILENAME = "recovery_c8_salvage_repair_launch_record.json"
SOURCE_SUBTREE_PATH = "src/vitnns"
PARTIAL_CONTROL_PATHS = {
    "execution_plan.json": "gap/execution_plan.json",
    "preflight.json": "gap/preflight.json",
    "salvage_scope.json": "gap/salvage_scope.json",
    "salvage_status.json": "gap/salvage_status.json",
    "c8_salvage_source_audit.json": "reports/c8_salvage_source_audit.json",
    "c8_partial_scientific_failure_diagnosis.json": (
        "reports/c8_partial_scientific_failure_diagnosis.json"
    ),
}
REPORTED_COUNTS = {
    "succeeded": 568, "terminal_failed_candidate": 1,
    "unfinished": 151, "pending": 143, "incomplete": 8,
}
ORIGINS = {
    "original_interrupted": "inherited_from_original_interrupted_root",
    "prior_partial_salvage": "inherited_from_prior_partial_salvage_root",
    "fresh_repair": "newly_run_in_salvage_repair_root",
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


class RecoveryC8SalvageRepairError(ValueError):
    """Raised when repair authority or custody validation fails."""


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
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
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
                raise RecoveryC8SalvageRepairError(
                    f"duplicate JSON key in {context}: {key}"
                )
            result[key] = value
        return result

    try:
        payload = path.read_bytes()
        if len(payload) > maximum:
            raise RecoveryC8SalvageRepairError(f"{context} is too large")
        value = json.loads(
            payload.decode("utf-8"), object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON value {token}")
            ),
        )
    except RecoveryC8SalvageRepairError:
        raise
    except Exception as error:
        raise RecoveryC8SalvageRepairError(
            f"invalid {context}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise RecoveryC8SalvageRepairError(f"{context} must be an object")
    return value


def _git(repo: Path, *arguments: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(repo), *arguments], check=True, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as error:
        raise RecoveryC8SalvageRepairError(
            f"git {' '.join(arguments)} failed: {error}"
        ) from error


def _git_bytes(repo: Path, *arguments: str) -> bytes:
    try:
        return subprocess.run(
            ["git", "-C", str(repo), *arguments], check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120,
        ).stdout
    except (OSError, subprocess.SubprocessError) as error:
        raise RecoveryC8SalvageRepairError(
            f"git {' '.join(arguments)} failed: {error}"
        ) from error


def _project_path(
    repo: Path, raw: Path | str, context: str, *, exists: bool = False,
) -> Path:
    candidate = Path(raw)
    path = candidate if candidate.is_absolute() else repo / candidate
    path = Path(os.path.abspath(os.fspath(path)))
    try:
        path.relative_to(repo)
    except ValueError as error:
        raise RecoveryC8SalvageRepairError(f"{context} escapes repository") from error
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if os.path.lexists(current) and stat.S_ISLNK(current.lstat().st_mode):
            raise RecoveryC8SalvageRepairError(f"{context} contains a symlink")
    if exists and not path.exists():
        raise RecoveryC8SalvageRepairError(f"{context} does not exist")
    return path


def _top(repo: Path) -> Path:
    return Path(_git(repo, "rev-parse", "--show-toplevel"))


def _top_relative(repo: Path, path: Path) -> str:
    return path.relative_to(_top(repo)).as_posix()


def _repo_relative(repo: Path, path: Path) -> str:
    return path.relative_to(repo).as_posix()


def _nul_paths(payload: bytes, context: str) -> list[str]:
    if payload and not payload.endswith(b"\0"):
        raise RecoveryC8SalvageRepairError(
            f"invalid NUL-delimited Git output for {context}"
        )
    try:
        return [item.decode("utf-8") for item in payload.split(b"\0") if item]
    except UnicodeDecodeError as error:
        raise RecoveryC8SalvageRepairError(
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
    unstaged = _nul_paths(_git_bytes(
        repo, "diff", "--name-only", "--no-renames", "-z",
        "HEAD", "--",
    ), "unstaged worktree inventory")
    staged = _nul_paths(_git_bytes(
        repo, "diff", "--cached", "--name-only", "--no-renames",
        "-z", "HEAD", "--",
    ), "staged worktree inventory")
    untracked = _nul_paths(_git_bytes(
        repo, "ls-files", "--full-name", "--others",
        "--exclude-standard", "-z", "--",
    ), "untracked worktree inventory")
    return sorted(set(unstaged + staged + untracked), key=lambda item: item.encode())


def _under(path: str, roots: tuple[str, ...]) -> bool:
    return any(path == root or path.startswith(root + "/") for root in roots)


def _lfs_pointer_digest(blob: bytes) -> tuple[int, str] | None:
    try:
        lines = blob.decode("utf-8").splitlines()
    except UnicodeDecodeError:
        return None
    if not lines or lines[0] != "version https://git-lfs.github.com/spec/v1":
        return None
    values: dict[str, str] = {}
    for line in lines[1:]:
        if " " not in line:
            return None
        key, value = line.split(" ", 1)
        values[key] = value
    oid = values.get("oid", "")
    size = values.get("size", "")
    if not oid.startswith("sha256:") or _SHA256_RE.fullmatch(oid[7:]) is None:
        return None
    if not size.isdecimal():
        return None
    return int(size), oid[7:]


def _revision_tree_records(repo: Path, root: Path, revision: str) -> list[dict[str, Any]]:
    root_path = _repo_relative(repo, root)
    payload = _git_bytes(
        repo, "ls-tree", "-r", "-z", "--long", revision, "--", root_path,
    )
    if not payload:
        raise RecoveryC8SalvageRepairError(
            "historical source missing from its custody commit"
        )
    records: list[dict[str, Any]] = []
    for item in payload.split(b"\0"):
        if not item:
            continue
        try:
            meta, path_bytes = item.split(b"\t", 1)
            mode, object_type, object_id, _size = meta.decode("ascii").split()
            path = path_bytes.decode("utf-8")
        except (ValueError, UnicodeDecodeError) as error:
            raise RecoveryC8SalvageRepairError(
                "invalid custody tree inventory"
            ) from error
        if object_type != "blob" or mode not in {"100644", "100755"}:
            raise RecoveryC8SalvageRepairError(
                "historical source contains non-ordinary committed entries"
            )
        if not (path == root_path or path.startswith(root_path + "/")):
            raise RecoveryC8SalvageRepairError(
                "historical source custody path escaped root"
            )
        blob = _git_bytes(repo, "cat-file", "-p", object_id)
        pointer = _lfs_pointer_digest(blob)
        if pointer is None:
            byte_count = len(blob)
            digest = sha256(blob).hexdigest()
        else:
            byte_count, digest = pointer
        relative = path.removeprefix(root_path + "/")
        records.append({"path": relative, "bytes": byte_count, "sha256": digest})
    return sorted(records, key=lambda record: record["path"].encode("utf-8"))


def _materialized_tree_records(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(
        root.rglob("*"),
        key=lambda item: item.relative_to(root).as_posix().encode("utf-8"),
    ):
        info = path.lstat()
        if stat.S_ISDIR(info.st_mode):
            continue
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise RecoveryC8SalvageRepairError(
                f"unsafe prior partial evidence entry: {path}"
            )
        if info.st_nlink != 1:
            raise RecoveryC8SalvageRepairError(
                f"prior partial evidence has a hard-linked file: {path}"
            )
        records.append({
            "path": path.relative_to(root).as_posix(), "bytes": info.st_size,
            "sha256": sha256_file(path),
        })
    return records


def _require_materialized_custody(
    repo: Path, root: Path, revision: str,
) -> list[dict[str, Any]]:
    expected = {record["path"]: record for record in _revision_tree_records(
        repo, root, revision,
    )}
    actual_records = _materialized_tree_records(root)
    actual = {record["path"]: record for record in actual_records}
    if set(actual) - set(expected):
        raise RecoveryC8SalvageRepairError(
            "historical source contains files outside its custody commit"
        )
    if set(expected) - set(actual):
        raise RecoveryC8SalvageRepairError(
            "historical source is missing files from its custody commit"
        )
    if any(actual[path] != expected[path] for path in expected):
        raise RecoveryC8SalvageRepairError(
            "historical source differs from its custody commit"
        )
    return actual_records


def _tree_inventory(
    repo: Path, root: Path, *, revision: str | None = None,
) -> dict[str, Any]:
    """Hash a closed materialized tree after optional custody comparison."""

    if revision is not None:
        records = _require_materialized_custody(repo, root, revision)
    else:
        records = _materialized_tree_records(root)
    return {
        "algorithm": "sha256_canonical_json_path_bytes_sha256_v1",
        "ordering": "relative_path_utf8_bytewise_ascending",
        "file_count": len(records),
        "total_bytes": sum(record["bytes"] for record in records),
        "digest": canonical_sha256({
            "domain": "vitnns-c8-salvage-root-v1", "files": records,
        }),
    }


def _require_custody_clean(repo: Path, root: Path, revision: str) -> None:
    """Require the materialized historical root to equal its custody tree."""

    _require_materialized_custody(repo, root, revision)


@dataclass(frozen=True)
class RecoveryC8SalvageRepairLaunchRecord:
    path: Path
    schema_path: Path
    document: Mapping[str, Any]
    binding: Any
    c7_launch_record: Any
    source_c8_record: Any
    original_salvage_record: RecoveryC8SalvageLaunchRecord
    source_root: Path
    prior_gap_root: Path
    repair_root: Path
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
    archived_custody_commit: str | None = None
    expected_launch_commit: str | None = None
    verify_archived_custody: bool = True

    @property
    def canonical_digest(self) -> str:
        return str(self.document["canonical_digest"])

    @property
    def authorization_launch_commit(self) -> str:
        return self.launch_commit

    @property
    def prior_authority_digest(self) -> str:
        return str(self.document["original_salvage_authority"]["canonical_digest"])


def _capture_producer(repo: Path) -> dict[str, Any]:
    if _git(repo, "status", "--porcelain=v1"):
        raise RecoveryC8SalvageRepairError(
            "salvage repair producer worktree must be clean"
        )
    commit = _git(repo, "rev-parse", "HEAD^{commit}")
    top = _top(repo)
    source_top = (repo / SOURCE_SUBTREE_PATH).relative_to(top).as_posix()
    return {
        "commit": commit, "tree": _git(repo, "rev-parse", "HEAD^{tree}"),
        "source_subtree_path": SOURCE_SUBTREE_PATH,
        "source_subtree_tree": _git(repo, "rev-parse", f"HEAD:{source_top}"),
        "dirty": False,
    }


def _historical_original_salvage(
    repo: Path, path: Path,
) -> RecoveryC8SalvageLaunchRecord:
    """Load the prior authority against custody while granting no live authority."""

    try:
        document = _salvage._read_json(path, "historical salvage authority")
        schema = _project_path(
            repo, _salvage.DEFAULT_SCHEMA_PATH, "historical salvage schema",
            exists=True,
        )
        errors = list(Draft202012Validator(
            _salvage._read_json(schema, "historical salvage schema")
        ).iter_errors(document))
        if errors:
            raise RecoveryC8SalvageRepairError(
                f"historical salvage schema failure: {errors[0].message}"
            )
        if document["canonical_digest"] != canonical_sha256({
            key: value for key, value in document.items()
            if key != "canonical_digest"
        }):
            raise RecoveryC8SalvageRepairError(
                "historical salvage canonical digest drifted"
            )
        if document["launch_schema"] != {
            "path": _repo_relative(repo, schema),
            "file_sha256": sha256_file(schema),
        }:
            raise RecoveryC8SalvageRepairError(
                "historical salvage schema binding drifted"
            )
        if (
            sha256_file(path) != ORIGINAL_SALVAGE_FILE_SHA256
            or document["canonical_digest"]
            != ORIGINAL_SALVAGE_CANONICAL_DIGEST
            or document["producer_git"]["commit"]
            != ORIGINAL_SALVAGE_PRODUCER_COMMIT
        ):
            raise RecoveryC8SalvageRepairError(
                "unexpected original salvage authority identity"
            )
        base = _salvage._base_record(
            repo, _project_path(
                repo, document["base_c8_authority"]["path"],
                "base C8 authority", exists=True,
            ),
        )
        source = _project_path(
            repo, document["source_interrupted_root"]["formal_root"],
            "original interrupted root", exists=True,
        )
        source_binding = document["source_interrupted_root"]
        if _git(
            repo, "rev-parse",
            f"{source_binding['evidence_commit']}:{_top_relative(repo, source)}",
        ) != source_binding["tree_object"]:
            raise RecoveryC8SalvageRepairError(
                "original interrupted root tree drifted"
            )
        for name, digest in source_binding["control_sha256"].items():
            if sha256_file(source / name) != digest:
                raise RecoveryC8SalvageRepairError(
                    f"original interrupted root control drifted: {name}"
                )
        outputs = document["outputs"]
        gap = _project_path(repo, outputs["gap_root"], "historical salvage gap")
        reports = _project_path(
            repo, outputs["reports_root"], "historical salvage reports",
        )
        selection = _project_path(
            repo, outputs["selection_root"], "historical salvage selection",
        )
        logs = _project_path(repo, outputs["logs_root"], "historical salvage logs")
        launch_commit = ORIGINAL_SALVAGE_LAUNCH_COMMIT
        record = RecoveryC8SalvageLaunchRecord(
            path=path, schema_path=schema, document=_freeze(document),
            binding=base.binding, c7_launch_record=base.c7_launch_record,
            source_c8_record=base, source_root=source, gap_root=gap,
            reports_root=reports, selection_root=selection, logs_root=logs,
            combined_manifest_path=repo / outputs["combined_manifest_path"],
            combined_report_path=repo / outputs["combined_report_path"],
            selection_path=repo / outputs["selection_path"],
            selection_report_path=repo / outputs["selection_report_path"],
            model_path=base.model_path, data_root=base.data_root,
            launch_commit=launch_commit, file_sha256=sha256_file(path),
            repo_root=repo,
        )
    except Exception as error:
        if isinstance(error, RecoveryC8SalvageRepairError):
            raise
        raise RecoveryC8SalvageRepairError(
            f"original salvage authority is invalid: {error}"
        ) from error
    if _git(
        repo, "rev-parse", f"{ORIGINAL_SALVAGE_PRODUCER_COMMIT}^{{tree}}",
    ) != document["producer_git"]["tree"]:
        raise RecoveryC8SalvageRepairError(
            "historical salvage producer tree drifted"
        )
    source_top = _top_relative(repo, repo / SOURCE_SUBTREE_PATH)
    if _git(
        repo, "rev-parse",
        f"{ORIGINAL_SALVAGE_PRODUCER_COMMIT}:{source_top}",
    ) != document["producer_git"]["source_subtree_tree"]:
        raise RecoveryC8SalvageRepairError(
            "historical salvage producer source subtree drifted"
        )
    if _git(
        repo, "rev-list", "--parents", "-n", "1",
        ORIGINAL_SALVAGE_LAUNCH_COMMIT,
    ).split() != [ORIGINAL_SALVAGE_LAUNCH_COMMIT, ORIGINAL_SALVAGE_PRODUCER_COMMIT]:
        raise RecoveryC8SalvageRepairError(
            "historical salvage launch ancestry drifted"
        )
    if _changed_paths(
        repo, ORIGINAL_SALVAGE_PRODUCER_COMMIT, ORIGINAL_SALVAGE_LAUNCH_COMMIT,
    ) != [_top_relative(repo, path)]:
        raise RecoveryC8SalvageRepairError(
            "historical salvage launch is not launch-only"
        )
    try:
        _git(
            repo, "merge-base", "--is-ancestor", record.launch_commit,
            PRIOR_PARTIAL_CUSTODY_COMMIT,
        )
    except RecoveryC8SalvageRepairError as error:
        raise RecoveryC8SalvageRepairError(
            "original salvage launch is not an ancestor of partial custody"
        ) from error
    top_path = _top_relative(repo, path)
    if _git(repo, "rev-parse", f"{record.launch_commit}:{top_path}") != _git(
        repo, "rev-parse", f"{PRIOR_PARTIAL_CUSTODY_COMMIT}:{top_path}",
    ):
        raise RecoveryC8SalvageRepairError(
            "original salvage authority blob drifted at partial custody"
        )
    return record


def _prior_controls(root: Path) -> dict[str, str]:
    return {
        name: sha256_file(root / relative)
        for name, relative in PARTIAL_CONTROL_PATHS.items()
    }


def _prior_binding(repo: Path, root: Path) -> dict[str, Any]:
    _require_custody_clean(repo, root, PRIOR_PARTIAL_CUSTODY_COMMIT)
    top_root = _top_relative(repo, root)
    gap = root / "gap"
    plan_digest = _salvage._execution_plan_digest(gap / "execution_plan.json")
    preflight = _read_json(gap / "preflight.json", "prior partial preflight")
    scope = _read_json(gap / "salvage_scope.json", "prior partial scope")
    status = _read_json(gap / "salvage_status.json", "prior partial status")
    source_audit = _read_json(
        root / PARTIAL_CONTROL_PATHS["c8_salvage_source_audit.json"],
        "prior source audit",
    )
    diagnosis = _read_json(
        root / PARTIAL_CONTROL_PATHS[
            "c8_partial_scientific_failure_diagnosis.json"
        ],
        "prior partial diagnosis",
    )
    scope_digest = scope.get("canonical_digest")
    if (
        not isinstance(scope_digest, str)
        or _SHA256_RE.fullmatch(scope_digest) is None
        or scope_digest != canonical_sha256({
            key: value for key, value in scope.items()
            if key != "canonical_digest"
        })
    ):
        raise RecoveryC8SalvageRepairError("prior partial scope digest drifted")
    states = scope.get("states")
    gap_digests = scope.get("gap_unit_digests")
    inherited_digests = scope.get("inherited_unit_digests")
    if (
        not isinstance(states, Mapping)
        or not isinstance(gap_digests, list) or len(gap_digests) != 720
        or len(set(gap_digests)) != 720 or set(states) != set(gap_digests)
        or not isinstance(inherited_digests, list)
        or len(inherited_digests) != 1460
        or len(set(inherited_digests)) != 1460
        or set(gap_digests) & set(inherited_digests)
        or scope.get("claim_tokens") != {}
        or any(
            value not in {"succeeded", "failed", "pending", "incomplete"}
            for value in states.values()
        )
    ):
        raise RecoveryC8SalvageRepairError(
            "prior partial scope is not a closed zero-claim partition"
        )
    preflight_digest = preflight.get("canonical_digest")
    audit_digest = source_audit.get("canonical_digest")
    if (
        preflight.get("plan_digest") != plan_digest
        or preflight_digest != canonical_sha256({
            key: value for key, value in preflight.items()
            if key != "canonical_digest"
        })
        or scope.get("plan_digest") != plan_digest
        or source_audit.get("plan_digest") != plan_digest
        or audit_digest != canonical_sha256({
            key: value for key, value in source_audit.items()
            if key != "canonical_digest"
        })
        or scope.get("source_audit_digest") != audit_digest
    ):
        raise RecoveryC8SalvageRepairError(
            "prior partial plan/preflight/audit chain drifted"
        )
    observed = {
        "succeeded": sum(value == "succeeded" for value in states.values()),
        "terminal_failed_candidate": sum(
            value == "failed" for value in states.values()
        ),
        "pending": sum(value == "pending" for value in states.values()),
        "incomplete": sum(value == "incomplete" for value in states.values()),
    }
    observed["unfinished"] = observed["pending"] + observed["incomplete"]
    expected_status = {
        "schema_version": 1,
        "document_type": "stage_c_c8_salvage_status",
        "state": "running", "active_claim_count": 0,
        "gap_count": len(states),
        "succeeded_count": observed["succeeded"],
        "failed_count": observed["terminal_failed_candidate"],
        "incomplete_count": observed["incomplete"],
        "pending_count": observed["pending"],
    }
    if status != expected_status or observed != REPORTED_COUNTS:
        raise RecoveryC8SalvageRepairError(
            "prior partial reported state counts drifted"
        )
    if (
        diagnosis.get("terminal_snapshot", {}).get("succeeded") != 568
        or diagnosis.get("terminal_snapshot", {}).get("scientific_failed")
        != 1
        or diagnosis.get("disposition", {}).get(
            "required_fresh_repair_unit_count_before_reaudit"
        ) != 151
        or diagnosis.get("disposition", {}).get(
            "scientific_failed_unit_may_be_rerun"
        ) is not False
        or diagnosis.get("authority", {}).get("canonical_digest")
        != ORIGINAL_SALVAGE_CANONICAL_DIGEST
        or diagnosis.get("authority", {}).get("launch_commit")
        != ORIGINAL_SALVAGE_LAUNCH_COMMIT
        or diagnosis.get("gap_root_inventory_at_stop")
        != PRIOR_PARTIAL_GAP_INVENTORY
    ):
        raise RecoveryC8SalvageRepairError(
            "prior partial diagnosis policy drifted"
        )
    return {
        "common_root": _repo_relative(repo, root),
        "gap_root": _repo_relative(repo, gap),
        "evidence_commit": PRIOR_PARTIAL_CUSTODY_COMMIT,
        "common_tree_object": _git(
            repo, "rev-parse", f"{PRIOR_PARTIAL_CUSTODY_COMMIT}:{top_root}",
        ),
        "gap_tree_object": _git(
            repo, "rev-parse",
            f"{PRIOR_PARTIAL_CUSTODY_COMMIT}:{_top_relative(repo, gap)}",
        ),
        "control_sha256": _prior_controls(root),
        "inventory": _tree_inventory(repo, gap, revision=PRIOR_PARTIAL_CUSTODY_COMMIT),
        "reported_counts": observed,
        "reported_counts_are_advisory": True,
    }


def _validate_outputs(
    repo: Path, launch: Path, outputs: Mapping[str, Any],
) -> tuple[Path, Path, Path, Path, tuple[str, ...]]:
    common = _project_path(repo, outputs["common_root"], "repair common root")
    if common != launch.parent or launch.name != LAUNCH_FILENAME:
        raise RecoveryC8SalvageRepairError(
            "repair launch must be the fixed filename in its common root"
        )
    relative = common.relative_to(repo)
    if len(relative.parts) < 3 or relative.parts[:2] != ("evidence", "stage_c"):
        raise RecoveryC8SalvageRepairError(
            "repair common root must be under evidence/stage_c"
        )
    if _project_path(repo, outputs["launch_record_path"], "declared launch") != launch:
        raise RecoveryC8SalvageRepairError("repair launch path binding drifted")
    expected = {
        "repair_root": "repair", "reports_root": "reports",
        "selection_root": "selection", "logs_root": "logs",
    }
    roots: dict[str, Path] = {}
    for key, name in expected.items():
        root = _project_path(repo, outputs[key], f"declared {key}")
        if root.parent != common or root.name != name:
            raise RecoveryC8SalvageRepairError(
                f"{key} must be the {name} child of repair common root"
            )
        roots[key] = root
    exact_files = {
        "combined_manifest_path": roots["reports_root"]
        / "c8_salvage_repair_combined_manifest.json",
        "combined_report_path": roots["reports_root"]
        / "c8_salvage_repair_combined_report.json",
        "selection_path": roots["selection_root"]
        / "stage_c_salvage_repair_selection.json",
        "selection_report_path": roots["reports_root"]
        / "stage_c_c8_salvage_repair_selection_report.json",
    }
    for key, expected_path in exact_files.items():
        if _project_path(repo, outputs[key], f"declared {key}") != expected_path:
            raise RecoveryC8SalvageRepairError(f"repair {key} binding drifted")
    allowed = tuple(_top_relative(repo, roots[key]) for key in expected)
    return (
        roots["repair_root"], roots["reports_root"],
        roots["selection_root"], roots["logs_root"], allowed,
    )


def _head_identity(repo: Path) -> dict[str, Any]:
    top = _top(repo)
    source_top = (repo / SOURCE_SUBTREE_PATH).relative_to(top).as_posix()
    return {
        "commit": _git(repo, "rev-parse", "HEAD^{commit}"),
        "tree": _git(repo, "rev-parse", "HEAD^{tree}"),
        "source_subtree_path": SOURCE_SUBTREE_PATH,
        "source_subtree_tree": _git(repo, "rev-parse", f"HEAD:{source_top}"),
    }


def _resolve_launch(
    repo: Path, launch: Path, declared: str, producer_commit: str, *,
    expected_launch_commit: str | None = None,
) -> str:
    launch_top = _top_relative(repo, launch)
    launch_commit = declared
    if expected_launch_commit is not None:
        if _COMMIT_RE.fullmatch(expected_launch_commit) is None:
            raise RecoveryC8SalvageRepairError(
                "expected repair launch commit is invalid"
            )
        if launch_commit not in {"SELF", expected_launch_commit}:
            raise RecoveryC8SalvageRepairError(
                "repair launch record disagrees with expected launch commit"
            )
        launch_commit = expected_launch_commit
    elif launch_commit == "SELF":
        launch_commit = _git(
            repo, "log", "-n", "1", "--format=%H", "HEAD", "--",
            f":(top){launch_top}",
        )
    if _COMMIT_RE.fullmatch(launch_commit) is None:
        raise RecoveryC8SalvageRepairError("repair launch commit is invalid")
    parents = _git(
        repo, "rev-list", "--parents", "-n", "1", launch_commit,
    ).split()
    if parents != [launch_commit, producer_commit]:
        raise RecoveryC8SalvageRepairError(
            "repair launch must directly follow its producer"
        )
    if _changed_paths(repo, producer_commit, launch_commit) != [launch_top]:
        raise RecoveryC8SalvageRepairError(
            "repair launch commit must change only its authority"
        )
    return launch_commit


def _validate_runtime(
    repo: Path, *, launch_commit: str, producer: Mapping[str, Any],
    evidence_roots: tuple[str, ...], check_worktree: bool,
    archived_custody_commit: str | None = None,
) -> dict[str, Any]:
    identity = _head_identity(repo)
    if archived_custody_commit is not None:
        if _COMMIT_RE.fullmatch(archived_custody_commit) is None:
            raise RecoveryC8SalvageRepairError(
                "repair archive custody commit is invalid"
            )
        try:
            _git(repo, "merge-base", "--is-ancestor", launch_commit,
                 archived_custody_commit)
        except RecoveryC8SalvageRepairError as error:
            raise RecoveryC8SalvageRepairError(
                "repair archive custody does not descend from launch"
            ) from error
        dirt = _worktree_dirty_paths(repo) if check_worktree else []
        escaped = [path for path in dirt if not _under(path, evidence_roots)]
        if escaped:
            raise RecoveryC8SalvageRepairError(
                "worktree dirt escapes archived repair evidence roots: "
                + escaped[0]
            )
        identity = {
            "commit": launch_commit,
            "tree": _git(repo, "rev-parse", f"{launch_commit}^{{tree}}"),
            "source_subtree_path": SOURCE_SUBTREE_PATH,
            "source_subtree_tree": producer["source_subtree_tree"],
            "dirty": False, "evidence_worktree_dirty": bool(dirt),
            "authorization_launch_commit": launch_commit,
        }
        return identity
    try:
        _git(
            repo, "merge-base", "--is-ancestor", launch_commit,
            identity["commit"],
        )
    except RecoveryC8SalvageRepairError as error:
        raise RecoveryC8SalvageRepairError(
            "repair runtime HEAD does not descend from launch"
        ) from error
    if identity["source_subtree_tree"] != producer["source_subtree_tree"]:
        raise RecoveryC8SalvageRepairError(
            "repair runtime source subtree differs from producer"
        )
    escaped = [
        path for path in _changed_paths(repo, launch_commit, identity["commit"])
        if not _under(path, evidence_roots)
    ]
    if escaped:
        raise RecoveryC8SalvageRepairError(
            "post-launch commit escapes repair evidence roots: " + escaped[0]
        )
    dirt = _worktree_dirty_paths(repo) if check_worktree else []
    escaped = [path for path in dirt if not _under(path, evidence_roots)]
    if escaped:
        raise RecoveryC8SalvageRepairError(
            "worktree dirt escapes repair evidence roots: " + escaped[0]
        )
    identity.update({
        "dirty": False, "evidence_worktree_dirty": bool(dirt),
        "authorization_launch_commit": launch_commit,
    })
    return identity


def _validate_schema(document: Mapping[str, Any], schema: Mapping[str, Any]) -> None:
    try:
        Draft202012Validator.check_schema(schema)
        errors = sorted(
            Draft202012Validator(schema).iter_errors(document),
            key=lambda error: tuple(str(item) for item in error.absolute_path),
        )
    except Exception as error:
        raise RecoveryC8SalvageRepairError("repair launch schema is invalid") from error
    if errors:
        location = ".".join(str(item) for item in errors[0].absolute_path) or "root"
        raise RecoveryC8SalvageRepairError(
            f"repair launch schema failure at {location}: {errors[0].message}"
        )


def generate_recovery_c8_salvage_repair_launch_record(
    *, repo_root: Path | str, launch_record_path: Path | str,
    repair_root: Path | str, reports_root: Path | str,
    selection_root: Path | str,
    prior_partial_root: Path | str = DEFAULT_PRIOR_PARTIAL_ROOT,
    original_salvage_authority_path: Path | str = (
        DEFAULT_ORIGINAL_SALVAGE_AUTHORITY_PATH
    ),
    amendment_path: Path | str = DEFAULT_AMENDMENT_PATH,
    schema_path: Path | str = DEFAULT_SCHEMA_PATH,
) -> dict[str, Any]:
    repo = Path(repo_root).resolve(strict=True)
    launch = _project_path(repo, launch_record_path, "repair launch path")
    schema = _project_path(repo, schema_path, "repair schema", exists=True)
    amendment = _project_path(repo, amendment_path, "repair amendment", exists=True)
    prior = _project_path(repo, prior_partial_root, "prior partial root", exists=True)
    original_path = _project_path(
        repo, original_salvage_authority_path, "original salvage authority",
        exists=True,
    )
    common = launch.parent
    repair = _project_path(repo, repair_root, "fresh repair root")
    reports = _project_path(repo, reports_root, "fresh repair reports root")
    selection = _project_path(repo, selection_root, "fresh repair selection root")
    logs = common / "logs"
    if launch.exists() or launch.is_symlink():
        raise RecoveryC8SalvageRepairError("repair launch path is not fresh")
    if any(path.parent != common for path in (repair, reports, selection, logs)):
        raise RecoveryC8SalvageRepairError(
            "repair output roots must share the launch parent"
        )
    if any(path.exists() or path.is_symlink() for path in (repair, reports, selection, logs)):
        raise RecoveryC8SalvageRepairError("repair output roots must be fresh")
    if common.exists() and any(common.iterdir()):
        raise RecoveryC8SalvageRepairError("repair common root must be empty")

    original = _historical_original_salvage(repo, original_path)
    prior_binding = _prior_binding(repo, prior)
    if prior_binding["common_tree_object"] != PRIOR_PARTIAL_COMMON_TREE:
        raise RecoveryC8SalvageRepairError("prior partial common tree drifted")
    if prior_binding["gap_tree_object"] != PRIOR_PARTIAL_GAP_TREE:
        raise RecoveryC8SalvageRepairError("prior partial gap tree drifted")
    if prior_binding["control_sha256"] != PRIOR_PARTIAL_CONTROL_SHA256:
        raise RecoveryC8SalvageRepairError("prior partial controls drifted")
    if prior_binding["inventory"] != PRIOR_PARTIAL_GAP_INVENTORY:
        raise RecoveryC8SalvageRepairError("prior partial inventory drifted")
    producer = _capture_producer(repo)
    amendment_commit = _git(
        repo, "log", "-n", "1", "--format=%H", "HEAD", "--",
        f":(top){_top_relative(repo, amendment)}",
    )
    for ancestor, context in (
        (amendment_commit, "repair amendment"),
        (PRIOR_PARTIAL_CUSTODY_COMMIT, "prior partial custody"),
    ):
        try:
            _git(
                repo, "merge-base", "--is-ancestor", ancestor,
                producer["commit"],
            )
        except RecoveryC8SalvageRepairError as error:
            raise RecoveryC8SalvageRepairError(
                f"{context} commit is not an ancestor of the producer"
            ) from error
    document: dict[str, Any] = {
        "$schema": schema.name, "schema_version": 1,
        "document_type": DOCUMENT_TYPE, "record_id": RECORD_ID,
        "status": STATUS, "launch_record_commit": "SELF",
        "launch_schema": {
            "path": _repo_relative(repo, schema),
            "file_sha256": sha256_file(schema),
        },
        "amendment": {
            "id": "RECOVERY-C8-SALVAGE-REPAIR-20260903",
            "path": _repo_relative(repo, amendment),
            "file_sha256": sha256_file(amendment),
            "containing_commit": amendment_commit,
        },
        "original_salvage_authority": {
            "path": _repo_relative(repo, original.path),
            "file_sha256": original.file_sha256,
            "canonical_digest": original.canonical_digest,
            "launch_commit": original.launch_commit,
            "producer_commit": original.document["producer_git"]["commit"],
        },
        "original_interrupted_root": _plain(
            original.document["source_interrupted_root"]
        ),
        "prior_partial_root": prior_binding,
        "origins": dict(ORIGINS),
        "contracts": _plain(original.document["contracts"]),
        "model": _plain(original.document["model"]),
        "producer_git": producer,
        "outputs": {
            "common_root": _repo_relative(repo, common),
            "launch_record_path": _repo_relative(repo, launch),
            "repair_root": _repo_relative(repo, repair),
            "reports_root": _repo_relative(repo, reports),
            "selection_root": _repo_relative(repo, selection),
            "logs_root": _repo_relative(repo, logs),
            "combined_manifest_path": _repo_relative(
                repo, reports / "c8_salvage_repair_combined_manifest.json",
            ),
            "combined_report_path": _repo_relative(
                repo, reports / "c8_salvage_repair_combined_report.json",
            ),
            "selection_path": _repo_relative(
                repo, selection / "stage_c_salvage_repair_selection.json",
            ),
            "selection_report_path": _repo_relative(
                repo, reports / "stage_c_c8_salvage_repair_selection_report.json",
            ),
        },
        "policy": {
            "expected_unit_count": 2180, "three_source_union": True,
            "source_roots_read_only": True, "source_resume_allowed": False,
            "artifact_copy_allowed": False,
            "source_recomputation_required": True,
            "valid_terminal_failure_inherited": True,
            "reported_counts_trusted": False,
        },
        "scope": {
            "c8_salvage_repair_authorized": True,
            "holdout_access_authorized": False, "c9_authorized": False,
            "c10_authorized": False, "stage_d_authorized": False,
        },
    }
    document["canonical_digest"] = canonical_sha256(document)
    _validate_schema(document, _read_json(schema, "repair launch schema"))
    return document


def write_recovery_c8_salvage_repair_launch_record(
    **kwargs: Any,
) -> RecoveryC8SalvageRepairLaunchRecord:
    repo = Path(kwargs["repo_root"]).resolve(strict=True)
    path = _project_path(repo, kwargs["launch_record_path"], "repair launch path")
    document = generate_recovery_c8_salvage_repair_launch_record(**kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(
            json.dumps(
                document, indent=2, sort_keys=True, ensure_ascii=False,
            ).encode("utf-8") + b"\n"
        )
        handle.flush()
        os.fsync(handle.fileno())
    return load_recovery_c8_salvage_repair_launch_record(
        path, repo_root=repo, require_tracked=False,
    )


def load_recovery_c8_salvage_repair_launch_record(
    path: Path | str, *, repo_root: Path | str = REPO_ROOT,
    schema_path: Path | str = DEFAULT_SCHEMA_PATH, require_tracked: bool = True,
    require_fresh_outputs: bool = False,
    archived_custody_commit: str | None = None,
    expected_launch_commit: str | None = None,
    verify_archived_custody: bool = True,
) -> RecoveryC8SalvageRepairLaunchRecord:
    repo = Path(repo_root).resolve(strict=True)
    launch = _project_path(repo, path, "repair launch record", exists=True)
    schema = _project_path(repo, schema_path, "repair launch schema", exists=True)
    document = _read_json(launch, "repair launch record")
    _validate_schema(document, _read_json(schema, "repair launch schema"))
    expected_digest = canonical_sha256({
        key: value for key, value in document.items() if key != "canonical_digest"
    })
    if document["canonical_digest"] != expected_digest:
        raise RecoveryC8SalvageRepairError("repair launch canonical digest drifted")
    if document["launch_schema"] != {
        "path": _repo_relative(repo, schema), "file_sha256": sha256_file(schema),
    }:
        raise RecoveryC8SalvageRepairError("repair schema binding drifted")
    amendment = _project_path(
        repo, document["amendment"]["path"], "repair amendment", exists=True,
    )
    if sha256_file(amendment) != document["amendment"]["file_sha256"]:
        raise RecoveryC8SalvageRepairError("repair amendment bytes drifted")
    amendment_commit = document["amendment"]["containing_commit"]
    if (
        _git(repo, "rev-parse", f"{amendment_commit}^{{commit}}")
        != amendment_commit
        or _git_bytes(
            repo, "show",
            f"{amendment_commit}:{_top_relative(repo, amendment)}",
        ) != amendment.read_bytes()
    ):
        raise RecoveryC8SalvageRepairError(
            "repair amendment differs from its containing commit"
        )

    original = _historical_original_salvage(
        repo, _project_path(
            repo, document["original_salvage_authority"]["path"],
            "original salvage authority", exists=True,
        ),
    )
    expected_original = {
        "path": _repo_relative(repo, original.path),
        "file_sha256": original.file_sha256,
        "canonical_digest": original.canonical_digest,
        "launch_commit": original.launch_commit,
        "producer_commit": original.document["producer_git"]["commit"],
    }
    if document["original_salvage_authority"] != expected_original:
        raise RecoveryC8SalvageRepairError(
            "original salvage authority binding drifted"
        )
    if document["original_interrupted_root"] != _plain(
        original.document["source_interrupted_root"]
    ):
        raise RecoveryC8SalvageRepairError(
            "original interrupted root binding drifted"
        )
    if document["contracts"] != _plain(original.document["contracts"]):
        raise RecoveryC8SalvageRepairError("repair contracts drifted")
    if document["model"] != _plain(original.document["model"]):
        raise RecoveryC8SalvageRepairError("repair model binding drifted")

    prior = _project_path(
        repo, document["prior_partial_root"]["common_root"],
        "prior partial root", exists=True,
    )
    actual_prior = _prior_binding(repo, prior)
    if actual_prior != document["prior_partial_root"]:
        raise RecoveryC8SalvageRepairError("prior partial custody drifted")
    if (
        actual_prior["common_tree_object"] != PRIOR_PARTIAL_COMMON_TREE
        or actual_prior["gap_tree_object"] != PRIOR_PARTIAL_GAP_TREE
    ):
        raise RecoveryC8SalvageRepairError("fixed prior partial tree drifted")
    if actual_prior["control_sha256"] != PRIOR_PARTIAL_CONTROL_SHA256:
        raise RecoveryC8SalvageRepairError("fixed prior partial controls drifted")
    if actual_prior["inventory"] != PRIOR_PARTIAL_GAP_INVENTORY:
        raise RecoveryC8SalvageRepairError("fixed prior partial inventory drifted")

    repair, reports, selection, logs, roots = _validate_outputs(
        repo, launch, document["outputs"],
    )
    if require_fresh_outputs and any(
        item.exists() or item.is_symlink()
        for item in (repair, reports, selection, logs)
    ):
        raise RecoveryC8SalvageRepairError("repair output roots are not fresh")
    producer = document["producer_git"]
    if (
        _git(repo, "rev-parse", f"{producer['commit']}^{{tree}}")
        != producer["tree"]
        or _git(
            repo, "rev-parse",
            f"{producer['commit']}:{_top_relative(repo, repo / SOURCE_SUBTREE_PATH)}",
        ) != producer["source_subtree_tree"]
    ):
        raise RecoveryC8SalvageRepairError("repair producer identity drifted")
    try:
        _git(
            repo, "merge-base", "--is-ancestor", amendment_commit,
            producer["commit"],
        )
    except RecoveryC8SalvageRepairError as error:
        raise RecoveryC8SalvageRepairError(
            "repair amendment commit is not an ancestor of the producer"
        ) from error
    try:
        _git(
            repo, "merge-base", "--is-ancestor",
            PRIOR_PARTIAL_CUSTODY_COMMIT, producer["commit"],
        )
    except RecoveryC8SalvageRepairError as error:
        raise RecoveryC8SalvageRepairError(
            "prior partial custody commit is not an ancestor of the producer"
        ) from error

    launch_commit = document["launch_record_commit"]
    if require_tracked:
        launch_commit = _resolve_launch(
            repo, launch, launch_commit, producer["commit"],
            expected_launch_commit=expected_launch_commit,
        )
        if archived_custody_commit is not None and verify_archived_custody:
            common_root = repo / document["outputs"]["common_root"]
            _require_custody_clean(repo, common_root, archived_custody_commit)
        _validate_runtime(
            repo, launch_commit=launch_commit, producer=producer,
            evidence_roots=roots, check_worktree=True,
            archived_custody_commit=archived_custody_commit,
        )
    elif (
        launch_commit != "SELF"
        or _git(repo, "rev-parse", "HEAD^{commit}") != producer["commit"]
    ):
        raise RecoveryC8SalvageRepairError(
            "uncommitted repair authority requires producer HEAD"
        )
    elif _worktree_dirty_paths(repo) != [_top_relative(repo, launch)]:
        raise RecoveryC8SalvageRepairError(
            "uncommitted validation permits only the fresh repair authority"
        )
    return RecoveryC8SalvageRepairLaunchRecord(
        path=launch, schema_path=schema, document=_freeze(document),
        binding=original.binding, c7_launch_record=original.c7_launch_record,
        source_c8_record=original.source_c8_record,
        original_salvage_record=original, source_root=original.source_root,
        prior_gap_root=prior / "gap", repair_root=repair,
        reports_root=reports, selection_root=selection, logs_root=logs,
        combined_manifest_path=repo / document["outputs"]["combined_manifest_path"],
        combined_report_path=repo / document["outputs"]["combined_report_path"],
        selection_path=repo / document["outputs"]["selection_path"],
        selection_report_path=repo / document["outputs"]["selection_report_path"],
        model_path=original.model_path, data_root=original.data_root,
        launch_commit=launch_commit, file_sha256=sha256_file(launch),
        repo_root=repo, archived_custody_commit=archived_custody_commit,
        expected_launch_commit=expected_launch_commit,
        verify_archived_custody=verify_archived_custody,
    )


def validate_recovery_c8_salvage_repair_launch_record(
    *args: Any, **kwargs: Any,
) -> RecoveryC8SalvageRepairLaunchRecord:
    return load_recovery_c8_salvage_repair_launch_record(*args, **kwargs)


def revalidate_recovery_c8_salvage_repair_runtime_record(
    record: RecoveryC8SalvageRepairLaunchRecord,
) -> RecoveryC8SalvageRepairLaunchRecord:
    """Revalidate a fully loaded repair authority at a hot boundary.

    Public repair entry points first use the full loader, which revalidates
    both historical roots and their complete inventories.  Unit provenance
    validation then invokes this bounded check repeatedly.  It verifies the
    tracked authority bytes, direct-child launch, producer source tree, and
    current runtime/worktree boundary without rehashing the immutable 1.1 GB
    historical source for every individual unit.  Source audits still hash
    both inherited trees before and after their per-unit validation.
    """

    if type(record) is not RecoveryC8SalvageRepairLaunchRecord:
        raise RecoveryC8SalvageRepairError(
            "validated salvage repair authority required"
        )
    repo = Path(record.repo_root).resolve(strict=True)
    launch = _project_path(
        repo, record.path, "repair launch record", exists=True,
    )
    schema = _project_path(
        repo, record.schema_path, "repair launch schema", exists=True,
    )
    document = _read_json(launch, "repair launch record")
    _validate_schema(document, _read_json(schema, "repair launch schema"))
    if document["canonical_digest"] != canonical_sha256({
        key: value for key, value in document.items()
        if key != "canonical_digest"
    }):
        raise RecoveryC8SalvageRepairError(
            "repair launch canonical digest drifted"
        )
    if (
        document != _plain(record.document)
        or sha256_file(launch) != record.file_sha256
        or document["launch_schema"] != {
            "path": _repo_relative(repo, schema),
            "file_sha256": sha256_file(schema),
        }
    ):
        raise RecoveryC8SalvageRepairError(
            "repair runtime authority bytes drifted"
        )
    repair, reports, selection, logs, roots = _validate_outputs(
        repo, launch, document["outputs"],
    )
    expected_paths = (
        repair, reports, selection, logs,
        repo / document["outputs"]["combined_manifest_path"],
        repo / document["outputs"]["combined_report_path"],
        repo / document["outputs"]["selection_path"],
        repo / document["outputs"]["selection_report_path"],
    )
    observed_paths = (
        record.repair_root, record.reports_root, record.selection_root,
        record.logs_root, record.combined_manifest_path,
        record.combined_report_path, record.selection_path,
        record.selection_report_path,
    )
    if tuple(Path(path) for path in observed_paths) != expected_paths:
        raise RecoveryC8SalvageRepairError(
            "repair runtime output projection drifted"
        )
    producer = document["producer_git"]
    if (
        _git(repo, "rev-parse", f"{producer['commit']}^{{tree}}")
        != producer["tree"]
        or _git(
            repo, "rev-parse",
            f"{producer['commit']}:{_top_relative(repo, repo / SOURCE_SUBTREE_PATH)}",
        ) != producer["source_subtree_tree"]
    ):
        raise RecoveryC8SalvageRepairError(
            "repair runtime producer identity drifted"
        )
    launch_commit = _resolve_launch(
        repo, launch, document["launch_record_commit"], producer["commit"],
        expected_launch_commit=record.expected_launch_commit,
    )
    if launch_commit != record.launch_commit:
        raise RecoveryC8SalvageRepairError(
            "repair runtime launch identity drifted"
        )
    _validate_runtime(
        repo, launch_commit=launch_commit, producer=producer,
        evidence_roots=roots, check_worktree=True,
        archived_custody_commit=record.archived_custody_commit,
    )
    return record


def capture_c8_salvage_repair_verifier_identity(
    record: RecoveryC8SalvageRepairLaunchRecord, phase: str = "runtime",
) -> dict[str, Any]:
    if type(record) is not RecoveryC8SalvageRepairLaunchRecord:
        raise RecoveryC8SalvageRepairError(
            "validated salvage repair authority required"
        )
    authoritative = revalidate_recovery_c8_salvage_repair_runtime_record(record)
    roots = tuple(
        _top_relative(record.repo_root, root)
        for root in (
            authoritative.repair_root, authoritative.reports_root,
            authoritative.selection_root, authoritative.logs_root,
        )
    )
    try:
        return _validate_runtime(
            record.repo_root, launch_commit=authoritative.launch_commit,
            producer=authoritative.document["producer_git"],
            evidence_roots=roots, check_worktree=True,
            archived_custody_commit=record.archived_custody_commit,
        )
    except RecoveryC8SalvageRepairError as error:
        raise RecoveryC8SalvageRepairError(f"repair {phase}: {error}") from error


def capture_authorized_c8_salvage_repair_source_state(
    record: RecoveryC8SalvageRepairLaunchRecord, phase: str = "runtime",
) -> dict[str, Any]:
    identity = capture_c8_salvage_repair_verifier_identity(record, phase)
    return {"commit": identity["commit"], "dirty": False}


def recovery_provenance_context(
    record: RecoveryC8SalvageRepairLaunchRecord,
) -> RecoveryC8SalvageRepairLaunchRecord:
    if type(record) is not RecoveryC8SalvageRepairLaunchRecord:
        raise RecoveryC8SalvageRepairError(
            "validated salvage repair authority required"
        )
    return record


__all__ = [
    "DEFAULT_SCHEMA_PATH", "PRIOR_PARTIAL_CUSTODY_COMMIT",
    "RecoveryC8SalvageRepairError", "RecoveryC8SalvageRepairLaunchRecord",
    "capture_authorized_c8_salvage_repair_source_state",
    "capture_c8_salvage_repair_verifier_identity",
    "generate_recovery_c8_salvage_repair_launch_record",
    "load_recovery_c8_salvage_repair_launch_record",
    "revalidate_recovery_c8_salvage_repair_runtime_record",
    "recovery_provenance_context",
    "validate_recovery_c8_salvage_repair_launch_record",
    "write_recovery_c8_salvage_repair_launch_record", "_tree_inventory",
]
