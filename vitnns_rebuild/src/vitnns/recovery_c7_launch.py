"""Independent recovery-v2 C7 launch-record generation and validation.

The module is metadata-only: it never loads model tensors, images, or CUDA.
It turns the recovery amendment, machine binding, a sealed Stage A receipt,
the C7 contracts, and a clean implementation commit into one narrowly scoped
launch authority.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from jsonschema import Draft202012Validator

from .io_utils import sha256_file
from .recovery_v2 import (
    DEFAULT_BINDING_PATH,
    RecoveryV2Binding,
    load_recovery_v2_binding,
    verify_recovery_v2_lfs_identity,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCHEMA_PATH = REPO_ROOT / "config/recovery_c7_launch_record.schema.json"
DEFAULT_STAGE_A_RECEIPT_PATH = (
    REPO_ROOT / "evidence/stage_a/recovery_v2_d8c2f84/stage_a_verification.json"
)
LAUNCH_RECORD_FILENAME = "recovery_c7_launch_record.json"
DOCUMENT_TYPE = "vitnns_recovery_v2_c7_launch_record"
RECORD_ID = "RECOVERY_V2_C7_LAUNCH_20260901"
STATUS = "AUTHORIZED_FOR_RECOVERY_C7_ONLY"
SOURCE_SUBTREE_PATH = "src/vitnns"
SCHEMA_RELATIVE_PATH = "config/recovery_c7_launch_record.schema.json"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_GIT_OBJECT_RE = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
_LFS_POINTER_RE = re.compile(
    rb"version https://git-lfs.github.com/spec/v1\n"
    rb"oid sha256:([0-9a-f]{64})\n"
    rb"size (0|[1-9][0-9]*)\n?\Z"
)
_GIT_TIMEOUT_SECONDS = 30
_MAX_JSON_BYTES = 4 * 1024 * 1024

_RECEIPT_KEYS = {
    "schema_version", "document_type", "status", "binding", "amendment",
    "scope", "model", "input", "environment", "source_git_state",
    "numeric_gates", "artifacts", "output", "git_end_state_policy",
    "legacy_pickle", "observed_git_state_before_terminal_seal",
    "canonical_digest",
}
_GIT_STATE_KEYS = {
    "commit", "tree", "branch", "remote_tracking_ref",
    "remote_tracking_commit", "head_matches_remote_tracking",
    "remote_tracking_is_ancestor", "ahead_count", "behind_count", "dirty",
    "status_entries",
}
_RECEIPT_BINDING_KEYS = {
    "binding_file_sha256", "binding_id", "binding_path",
    "binding_schema_path", "binding_schema_sha256", "canonical_digest",
    "status",
}
_RECEIPT_MODEL_KEYS = {
    "name", "identity", "state_dict_path", "state_dict_sha256",
    "state_dict_bytes", "canonical_tensor_digest",
    "canonical_tensor_digest_algorithm", "key_count", "parameter_count",
    "historical_artifact_recovered", "historical_state_dict_sha256", "lfs",
    "loader", "architecture",
}
_RECEIPT_INPUT_KEYS = {
    "sample_id", "path", "sha256", "bytes", "width", "height",
    "ground_truth_wnid", "ground_truth_index", "label_source",
    "source_manifest_path", "source_manifest_sha256", "official_devkit_path",
    "official_devkit_sha256",
}
_RECEIPT_ENVIRONMENT_KEYS = {"binding", "conda", "requirements_lock", "runtime"}
_VALIDATION_KEYS = {
    "schema_version", "document_type", "status", "all_pass",
    "attention_actual_vs_used_max_abs_error", "attention_pass",
    "attention_recomputation_max_abs_error", "attention_threshold",
    "canonical_digest", "fused_vs_unfused_logits_max_abs_error",
    "hook_noninvasive_pass", "legacy_pickle_enabled", "legacy_pickle_used",
    "logits_pass", "logits_threshold",
    "per_block_attention_recomputation_max_abs_error",
    "reference_vs_extractor_block_input_max_abs_error",
    "reference_vs_extractor_block_output_max_abs_error",
    "reference_vs_extractor_logits_max_abs_error", "top5_match_pass",
    "unfused_plain_vs_hooked_logits_max_abs_error",
}
_PROVENANCE_KEYS = {
    "schema_version", "document_type", "status", "binding", "amendment",
    "scope", "model", "input", "preprocessing", "prediction", "extractor",
    "numeric_gates", "environment", "source_git_state", "artifacts",
    "legacy_pickle", "canonical_digest",
}
_ARTIFACT_NAMES = ("extraction.pt", "provenance.json", "validation.json")
_PHASE_ROOTS = {
    "pre_diagnostic": (),
    "diagnostic_gate": ("diagnostic_root",),
    "formal": ("diagnostic_root", "reports_root"),
    "report": ("diagnostic_root", "formal_root", "reports_root"),
}
_SCOPE = {
    "execution_authorization": "RECOVERY_C7_DIAGNOSTIC_AND_FORMAL_ONLY",
    "rerun_interpretation": "recovery_replication_not_new_confirmatory",
    "c7_diagnostic_authorized": True,
    "c7_formal_authorized": True,
    "stage_c_holdout_access_authorized": False,
    "c8_authorized": False,
    "c9_authorized": False,
    "c10_authorized": False,
    "stage_d_d7_authorized": False,
    "stage_d_d8_authorized": False,
    "stage_d_reserve_access_authorized": False,
}


class RecoveryC7LaunchError(ValueError):
    """Raised when recovery C7 launch authority cannot be established."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RecoveryC7LaunchError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite(token: str) -> None:
    raise RecoveryC7LaunchError(f"Non-finite JSON value is forbidden: {token}")


def _plain(value: Any, context: str = "value") -> Any:
    if value is None or type(value) in {str, bool, int}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise RecoveryC7LaunchError(f"{context} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, child in value.items():
            if not isinstance(key, str) or not key or key in result:
                raise RecoveryC7LaunchError(
                    f"{context} keys must be unique non-empty strings"
                )
            result[key] = _plain(child, f"{context}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [_plain(child, f"{context}[{index}]")
                for index, child in enumerate(value)]
    raise RecoveryC7LaunchError(
        f"{context} is not JSON-compatible: {type(value).__name__}"
    )


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(child) for key, child in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(child) for child in value)
    return copy.deepcopy(value)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        _plain(value, "canonical JSON input"), sort_keys=True,
        separators=(",", ":"), ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def launch_record_digest(document: Mapping[str, Any]) -> str:
    return canonical_sha256({
        key: value for key, value in document.items()
        if key != "canonical_digest"
    })


def _json_file_bytes(value: Any) -> bytes:
    text = json.dumps(
        _plain(value), sort_keys=True, indent=2, ensure_ascii=False,
        allow_nan=False,
    )
    return (text + "\n").encode("utf-8")


def _regular_single_link_file(path: Path, context: str) -> None:
    try:
        info = path.lstat()
    except OSError as error:
        raise RecoveryC7LaunchError(f"Cannot inspect {context}: {path}") from error
    if path.is_symlink() or not path.is_file() or info.st_nlink != 1:
        raise RecoveryC7LaunchError(
            f"{context} must be an ordinary single-link file"
        )


def _sha256_regular_file(path: Path, context: str) -> str:
    _regular_single_link_file(path, context)
    return sha256_file(path)


def _read_json(path: Path, context: str) -> Any:
    _regular_single_link_file(path, context)
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise RecoveryC7LaunchError(f"Cannot read {context}: {path}") from error
    if len(payload) > _MAX_JSON_BYTES:
        raise RecoveryC7LaunchError(f"{context} exceeds the metadata size limit")
    try:
        return json.loads(
            payload.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except UnicodeDecodeError as error:
        raise RecoveryC7LaunchError(f"{context} is not UTF-8") from error
    except json.JSONDecodeError as error:
        raise RecoveryC7LaunchError(f"Invalid {context}: {error}") from error


def _exact_keys(value: Any, expected: set[str], context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RecoveryC7LaunchError(f"{context} must be an object")
    actual = set(value)
    if actual != expected:
        raise RecoveryC7LaunchError(
            f"{context} fields drifted: missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}"
        )
    return value


def _require_self_digest(document: Mapping[str, Any], context: str) -> None:
    digest = document.get("canonical_digest")
    if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
        raise RecoveryC7LaunchError(f"{context} canonical digest is invalid")
    if digest != canonical_sha256({
        key: value for key, value in document.items()
        if key != "canonical_digest"
    }):
        raise RecoveryC7LaunchError(f"{context} canonical digest drifted")


def _git_process(
    repo_root: Path, arguments: Sequence[str], *, check: bool = True,
    text: bool = True,
) -> subprocess.CompletedProcess[Any]:
    environment = os.environ.copy()
    environment.update({
        "GIT_OPTIONAL_LOCKS": "0", "GIT_TERMINAL_PROMPT": "0",
        "GIT_LFS_SKIP_SMUDGE": "1",
    })
    try:
        return subprocess.run(
            ["git", *arguments], cwd=repo_root, check=check,
            capture_output=True, text=text, timeout=_GIT_TIMEOUT_SECONDS,
            env=environment,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise RecoveryC7LaunchError(
            f"Cannot inspect Git state: git {' '.join(arguments)}"
        ) from error


def _run_git(repo_root: Path, *arguments: str) -> str:
    return _git_process(repo_root, arguments).stdout.strip()


def _repository_identity(repo_root: Path) -> tuple[Path, str]:
    try:
        root = Path(repo_root).resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise RecoveryC7LaunchError("repo_root must be an existing directory") from error
    if not root.is_dir() or root.is_symlink():
        raise RecoveryC7LaunchError("repo_root must be an ordinary directory")
    top = Path(_run_git(root, "rev-parse", "--show-toplevel")).resolve(strict=True)
    prefix = _run_git(root, "rev-parse", "--show-prefix")
    if prefix:
        parsed = PurePosixPath(prefix)
        if (not prefix.endswith("/") or parsed.is_absolute() or "\\" in prefix
                or any(part in {"", ".", ".."} for part in parsed.parts)):
            raise RecoveryC7LaunchError("Git repository prefix is unsafe")
        project = top.joinpath(*parsed.parts).resolve(strict=True)
    else:
        project = top
    if project != root:
        raise RecoveryC7LaunchError(
            "repo_root does not match the active Git project identity"
        )
    return top, prefix


def _project_path(
    repo_root: Path, raw: Path | str, context: str, *, require_exists: bool = False,
) -> tuple[Path, str]:
    text = os.fspath(raw)
    if not isinstance(text, str) or not text or "\\" in text:
        raise RecoveryC7LaunchError(f"{context} must be a safe project path")
    supplied = Path(text)
    if supplied.is_absolute():
        candidate = Path(os.path.abspath(text))
        try:
            relative = candidate.relative_to(repo_root).as_posix()
        except ValueError as error:
            raise RecoveryC7LaunchError(f"{context} escapes the active project") from error
    else:
        pieces = text.split("/")
        if any(piece in {"", ".", ".."} for piece in pieces):
            raise RecoveryC7LaunchError(f"{context} is not normalized")
        relative = text
        candidate = repo_root.joinpath(*pieces)
    current = repo_root
    for piece in relative.split("/"):
        current = current / piece
        if current.is_symlink():
            raise RecoveryC7LaunchError(f"{context} must not traverse symlinks")
    candidate = Path(os.path.abspath(os.fspath(candidate)))
    if not candidate.is_relative_to(repo_root):
        raise RecoveryC7LaunchError(f"{context} escapes the active project")
    if require_exists and not candidate.exists():
        raise RecoveryC7LaunchError(f"{context} does not exist")
    return candidate, relative


def _top_relative(top: Path, path: Path) -> str:
    try:
        return path.relative_to(top).as_posix()
    except ValueError as error:
        raise RecoveryC7LaunchError("Path escapes the Git repository") from error


def _git_status_entries(repo_root: Path) -> list[dict[str, str]]:
    result = _git_process(
        repo_root,
        ("-c", "status.relativePaths=false", "status", "--porcelain=v1", "-z",
         "--untracked-files=all"), text=False,
    )
    try:
        records = result.stdout.decode("utf-8", errors="strict").split("\0")
    except UnicodeDecodeError as error:
        raise RecoveryC7LaunchError("Git status is not valid UTF-8") from error
    entries: list[dict[str, str]] = []
    index = 0
    while index < len(records):
        record = records[index]
        index += 1
        if not record:
            continue
        if len(record) < 4 or record[2] != " ":
            raise RecoveryC7LaunchError("Unexpected Git status record")
        entry = {"status": record[:2], "path": record[3:]}
        if record[0] in "RC" or record[1] in "RC":
            if index >= len(records) or not records[index]:
                raise RecoveryC7LaunchError("Incomplete Git rename/copy record")
            entry["source_path"] = records[index]
            index += 1
        entries.append(entry)
    return entries


def _is_ancestor(repo_root: Path, older: str, newer: str, context: str) -> None:
    result = _git_process(
        repo_root, ("merge-base", "--is-ancestor", older, newer), check=False,
    )
    if result.returncode == 1:
        raise RecoveryC7LaunchError(f"{context} is not an ancestor of {newer}")
    if result.returncode != 0:
        raise RecoveryC7LaunchError(f"Cannot verify {context} ancestry")


def _require_tracked(repo_root: Path, top_path: str, revision: str = "HEAD") -> None:
    result = _git_process(
        repo_root, ("cat-file", "-e", f"{revision}:{top_path}"), check=False,
    )
    if result.returncode != 0:
        raise RecoveryC7LaunchError(
            f"Required path is not tracked at {revision}: {top_path}"
        )


def _verify_lfs_pointer_at_revision(
    repo_root: Path, top: Path, path: Path, *, revision: str = "HEAD",
    expected_sha256: str | None = None, require_clean: bool = True,
) -> tuple[str, int]:
    top_path = _top_relative(top, path)
    _require_tracked(repo_root, top_path, revision)
    blob = _git_process(
        repo_root, ("cat-file", "blob", f"{revision}:{top_path}"), text=False,
    ).stdout
    match = _LFS_POINTER_RE.fullmatch(blob)
    if match is None:
        raise RecoveryC7LaunchError(
            f"Tracked evidence is not a strict Git LFS pointer at {revision}: {top_path}"
        )
    oid = match.group(1).decode("ascii")
    size = int(match.group(2))
    _regular_single_link_file(path, f"hydrated Git LFS file {top_path}")
    live_sha = sha256_file(path)
    if oid != live_sha or size != path.stat().st_size:
        raise RecoveryC7LaunchError(
            f"Hydrated Git LFS content drifted from {revision}: {top_path}"
        )
    if expected_sha256 is not None and live_sha != expected_sha256:
        raise RecoveryC7LaunchError(f"Git LFS SHA-256 drifted: {top_path}")
    if require_clean:
        status = _run_git(
            top, "status", "--porcelain=v1", "--untracked-files=all", "--", top_path,
        )
        if status:
            raise RecoveryC7LaunchError(f"Tracked Git LFS file is dirty: {top_path}")
    return oid, size


def _verify_lfs_pointer_at_head(
    repo_root: Path, top: Path, path: Path, *, expected_sha256: str | None = None,
) -> tuple[str, int]:
    return _verify_lfs_pointer_at_revision(
        repo_root, top, path, revision="HEAD", expected_sha256=expected_sha256,
        require_clean=True,
    )


@dataclass(frozen=True)
class RecoveryC7LaunchRecord:
    """Validated launch authority with paths ready for CLI consumption."""

    path: Path
    schema_path: Path
    document: Mapping[str, Any]
    binding: RecoveryV2Binding
    model_path: Path
    data_root: Path
    devkit_path: Path
    diagnostic_root: Path
    formal_root: Path
    reports_root: Path
    model_identity: Mapping[str, Any]
    file_sha256: str
    launch_commit: str
    repo_root: Path = REPO_ROOT

    @property
    def canonical_digest(self) -> str:
        return str(self.document["canonical_digest"])


def _validate_output_paths(
    repo: Path, launch_raw: Path | str, diagnostic_raw: Path | str,
    formal_raw: Path | str, reports_raw: Path | str, *,
    require_fresh_roots: bool, launch_must_exist: bool,
) -> tuple[Path, Path, Path, Path, Path]:
    launch, launch_rel = _project_path(
        repo, launch_raw, "launch record path", require_exists=launch_must_exist,
    )
    if launch.name != LAUNCH_RECORD_FILENAME:
        raise RecoveryC7LaunchError(
            f"Launch record filename must be {LAUNCH_RECORD_FILENAME}"
        )
    common = launch.parent
    common_rel = PurePosixPath(launch_rel).parent
    if len(common_rel.parts) < 3 or common_rel.parts[:2] != ("evidence", "stage_c"):
        raise RecoveryC7LaunchError(
            "Recovery C7 common root must be named under evidence/stage_c"
        )
    roots: list[Path] = []
    for raw, context in (
        (diagnostic_raw, "diagnostic root"), (formal_raw, "formal root"),
        (reports_raw, "reports root"),
    ):
        path, _relative = _project_path(repo, raw, context)
        if path.parent != common:
            raise RecoveryC7LaunchError(
                f"{context} must be a direct child of the launch common root"
            )
        if path == launch:
            raise RecoveryC7LaunchError(f"{context} overlaps the launch record")
        if path.exists() or path.is_symlink():
            if require_fresh_roots:
                raise RecoveryC7LaunchError(f"{context} is not fresh")
            if path.is_symlink() or not path.is_dir():
                raise RecoveryC7LaunchError(f"{context} must be a directory")
        roots.append(path)
    if len(set(roots)) != 3:
        raise RecoveryC7LaunchError(
            "Diagnostic, formal, and reports roots must be mutually distinct"
        )
    if common.exists() and (common.is_symlink() or not common.is_dir()):
        raise RecoveryC7LaunchError("Launch common root must be an ordinary directory")
    if require_fresh_roots and common.exists():
        allowed = {launch.name} if launch_must_exist else set()
        children = list(common.iterdir())
        if {child.name for child in children} != allowed:
            raise RecoveryC7LaunchError(
                "Fresh common root may contain only the launch record"
            )
        if launch_must_exist and (
            len(children) != 1 or children[0].is_symlink()
            or not children[0].is_file()
        ):
            raise RecoveryC7LaunchError(
                "Fresh common root must contain one ordinary launch record"
            )
    return launch, common, roots[0], roots[1], roots[2]


def _validate_schema(document: Mapping[str, Any], schema: Any) -> None:
    try:
        Draft202012Validator.check_schema(schema)
    except Exception as error:
        raise RecoveryC7LaunchError("Recovery C7 launch schema is invalid") from error
    errors = sorted(
        Draft202012Validator(schema).iter_errors(document),
        key=lambda error: tuple(str(item) for item in error.absolute_path),
    )
    if errors:
        error = errors[0]
        location = ".".join(str(item) for item in error.absolute_path) or "root"
        raise RecoveryC7LaunchError(
            f"Recovery C7 launch schema failure at {location}: {error.message}"
        )


def _validate_intrinsic_authority(document: Mapping[str, Any]) -> None:
    """Enforce authority semantics independently of the supplied schema."""

    if (
        document.get("$schema") != "recovery_c7_launch_record.schema.json"
        or document.get("schema_version") != 1
        or document.get("document_type") != DOCUMENT_TYPE
        or document.get("record_id") != RECORD_ID
        or document.get("status") != STATUS
        or document.get("scope") != _SCOPE
    ):
        raise RecoveryC7LaunchError("Recovery C7 intrinsic authority drifted")


def _binding_record(repo: Path, binding: RecoveryV2Binding) -> dict[str, Any]:
    return {
        "binding_id": binding.document["binding_id"],
        "status": binding.document["status"],
        "path": binding.path.relative_to(repo).as_posix(),
        "file_sha256": _sha256_regular_file(binding.path, "recovery-v2 binding"),
        "schema_path": binding.schema_path.relative_to(repo).as_posix(),
        "schema_file_sha256": _sha256_regular_file(
            binding.schema_path, "recovery-v2 binding schema"
        ),
        "canonical_digest": binding.canonical_digest,
    }


def _model_record(binding: RecoveryV2Binding) -> dict[str, Any]:
    return {key: binding.model[key] for key in (
        "name", "identity", "state_dict_path", "state_dict_sha256",
        "state_dict_bytes", "lfs_oid_sha256", "canonical_tensor_digest",
        "canonical_tensor_digest_algorithm", "expected_key_count",
        "expected_parameter_count", "historical_artifact_recovered",
        "historical_state_dict_sha256",
    )}


def _dependency_record(repo: Path, binding: RecoveryV2Binding) -> dict[str, Any]:
    environment = binding.document["environment"]
    path, relative = _project_path(
        repo, environment["requirements_lock_path"], "requirements lock",
        require_exists=True,
    )
    _regular_single_link_file(path, "requirements lock")
    digest = sha256_file(path)
    if digest != environment["requirements_lock_sha256"]:
        raise RecoveryC7LaunchError("Requirements lock SHA-256 drifted")
    return {
        "requirements_lock_path": relative,
        "requirements_lock_sha256": digest,
        "requirements_lock_bytes": path.stat().st_size,
    }


def _contract_records(repo: Path, binding: RecoveryV2Binding) -> dict[str, Any]:
    source = binding.document["c7_contracts"]
    definitions = {
        "stage_c_config": ("stage_c_config_path", "stage_c_config_sha256", None),
        "protocol": ("protocol_path", "protocol_sha256", None),
        "acceptance": ("acceptance_path", "acceptance_sha256", None),
        "diagnostic_run_spec": (
            "diagnostic_run_spec_path", "diagnostic_run_spec_sha256",
            "diagnostic_run_spec_canonical_sha256",
        ),
        "formal_run_spec": (
            "formal_run_spec_path", "formal_run_spec_sha256",
            "formal_run_spec_canonical_sha256",
        ),
    }
    result: dict[str, Any] = {}
    for name, (path_key, hash_key, canonical_key) in definitions.items():
        path, relative = _project_path(
            repo, source[path_key], f"C7 {name}", require_exists=True,
        )
        _regular_single_link_file(path, f"C7 {name}")
        digest = sha256_file(path)
        if digest != source[hash_key]:
            raise RecoveryC7LaunchError(f"C7 contract file drifted: {name}")
        item = {"path": relative, "file_sha256": digest}
        if canonical_key:
            canonical = canonical_sha256(_read_json(path, f"C7 {name}"))
            if canonical != source[canonical_key]:
                raise RecoveryC7LaunchError(
                    f"C7 run-spec canonical digest drifted: {name}"
                )
            item["canonical_sha256"] = canonical
        result[name] = item
    return result


def _artifact_record(root: str, name: str, item: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path": f"{root}/{name}", "bytes": item["bytes"],
        "file_sha256": item["sha256"],
    }


def _validate_stage_a_receipt(
    repo: Path, top: Path, receipt_path: Path, binding: RecoveryV2Binding,
    expected: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    receipt = _read_json(receipt_path, "Stage A terminal receipt")
    _exact_keys(receipt, _RECEIPT_KEYS, "Stage A terminal receipt")
    _require_self_digest(receipt, "Stage A terminal receipt")
    if (receipt["schema_version"] != 1
            or receipt["document_type"] != "recovery_v2_stage_a_verification"
            or receipt["status"] != "PASS"):
        raise RecoveryC7LaunchError("Stage A terminal receipt is not PASS")
    root = receipt_path.parent.relative_to(repo).as_posix()
    receipt_relative = receipt_path.relative_to(repo).as_posix()
    if root.split("/")[:2] != ["evidence", "stage_a"] or len(root.split("/")) != 3:
        raise RecoveryC7LaunchError("Stage A evidence root is unsafe")
    if receipt["output"] != {
        "root": root, "fresh_at_start": True, "repository_relative": True,
        "git_lfs_filter": "lfs",
    }:
        raise RecoveryC7LaunchError("Stage A output custody drifted")
    if receipt["legacy_pickle"] != {"enabled": False, "used": False}:
        raise RecoveryC7LaunchError("Stage A legacy-pickle boundary drifted")
    source = _exact_keys(
        receipt["source_git_state"], _GIT_STATE_KEYS, "Stage A source Git state",
    )
    if (not isinstance(source["commit"], str)
            or _COMMIT_RE.fullmatch(source["commit"]) is None
            or not isinstance(source["tree"], str)
            or _GIT_OBJECT_RE.fullmatch(source["tree"]) is None
            or source["dirty"] is not False or source["status_entries"] != []
            or source["branch"] != "stage-d-d3-pass"
            or source["remote_tracking_ref"]
               != "refs/remotes/origin/stage-d-d3-pass"
            or source["behind_count"] != 0
            or type(source["ahead_count"]) is not int or source["ahead_count"] < 0
            or source["remote_tracking_is_ancestor"] is not True
            or type(source["head_matches_remote_tracking"]) is not bool
            or source["head_matches_remote_tracking"]
               != (source["ahead_count"] == 0)):
        raise RecoveryC7LaunchError("Stage A producer was not clean")
    for field, context in (
        (source["remote_tracking_commit"], "Stage A remote-tracking commit"),
        (source["tree"], "Stage A producer tree"),
    ):
        if not isinstance(field, str) or _GIT_OBJECT_RE.fullmatch(field) is None:
            raise RecoveryC7LaunchError(f"{context} is invalid")
    actual_tree = _run_git(
        repo, "rev-parse", "--verify", f"{source['commit']}^{{tree}}",
    )
    if actual_tree != source["tree"]:
        raise RecoveryC7LaunchError("Stage A producer tree drifted")
    tracked_commit = _run_git(
        repo, "rev-parse", "--verify",
        f"{source['remote_tracking_ref']}^{{commit}}",
    )
    if tracked_commit != source["remote_tracking_commit"]:
        raise RecoveryC7LaunchError("Stage A remote-tracking identity drifted")
    _is_ancestor(
        repo, tracked_commit, source["commit"], "Stage A remote-tracking commit",
    )
    observed = _exact_keys(
        receipt["observed_git_state_before_terminal_seal"], _GIT_STATE_KEYS,
        "Stage A pre-seal Git state",
    )
    identity_keys = _GIT_STATE_KEYS - {"dirty", "status_entries"}
    if any(observed[key] != source[key] for key in identity_keys):
        raise RecoveryC7LaunchError("Stage A identity changed before sealing")
    expected_untracked = {
        _top_relative(top, receipt_path.parent / name) for name in _ARTIFACT_NAMES
    }
    entries = observed["status_entries"]
    if (observed["dirty"] is not True or not isinstance(entries, list)
            or {entry.get("path") for entry in entries} != expected_untracked
            or any(set(entry) != {"path", "status"} or entry["status"] != "??"
                   for entry in entries)):
        raise RecoveryC7LaunchError("Stage A pre-seal artifact set drifted")
    expected_policy = {
        "identity_must_equal_source": True,
        "only_fresh_output_files_may_be_untracked": True,
        "expected_output_files": [*_ARTIFACT_NAMES, "stage_a_verification.json"],
        "checked_before_terminal_seal": True,
        "terminal_receipt_published_last": True,
    }
    if receipt["git_end_state_policy"] != expected_policy:
        raise RecoveryC7LaunchError("Stage A terminal-seal policy drifted")

    binding_record = _exact_keys(
        receipt["binding"], _RECEIPT_BINDING_KEYS, "Stage A binding",
    )
    expected_binding = {
        "binding_file_sha256": _sha256_regular_file(
            binding.path, "recovery-v2 binding"
        ),
        "binding_id": binding.document["binding_id"],
        "binding_path": binding.path.relative_to(repo).as_posix(),
        "binding_schema_path": binding.schema_path.relative_to(repo).as_posix(),
        "binding_schema_sha256": _sha256_regular_file(
            binding.schema_path, "recovery-v2 binding schema"
        ),
        "canonical_digest": binding.canonical_digest,
        "status": binding.document["status"],
    }
    if dict(binding_record) != expected_binding:
        raise RecoveryC7LaunchError("Stage A recovery binding drifted")
    if canonical_json_bytes(receipt["amendment"]) != canonical_json_bytes(
        binding.document["amendment"]
    ) or canonical_json_bytes(receipt["scope"]) != canonical_json_bytes(
        binding.document["scope"]
    ):
        raise RecoveryC7LaunchError("Stage A amendment or scope drifted")

    model = _exact_keys(receipt["model"], _RECEIPT_MODEL_KEYS, "Stage A model")
    model_pairs = {
        "name": "name", "identity": "identity",
        "state_dict_path": "state_dict_path",
        "state_dict_sha256": "state_dict_sha256",
        "state_dict_bytes": "state_dict_bytes",
        "canonical_tensor_digest": "canonical_tensor_digest",
        "canonical_tensor_digest_algorithm": "canonical_tensor_digest_algorithm",
        "key_count": "expected_key_count", "parameter_count": "expected_parameter_count",
        "historical_artifact_recovered": "historical_artifact_recovered",
        "historical_state_dict_sha256": "historical_state_dict_sha256",
    }
    if not isinstance(model, Mapping) or any(
        model.get(left) != binding.model[right] for left, right in model_pairs.items()
    ):
        raise RecoveryC7LaunchError("Stage A model identity drifted")
    if model.get("lfs") != {
        "path": _top_relative(top, binding.model_path),
        "oid_sha256": binding.model["lfs_oid_sha256"], "hydrated": True,
    }:
        raise RecoveryC7LaunchError("Stage A model LFS identity drifted")
    environment_record = _exact_keys(
        receipt["environment"], _RECEIPT_ENVIRONMENT_KEYS, "Stage A environment",
    )
    if canonical_json_bytes(environment_record["binding"]) != canonical_json_bytes(
        binding.document["environment"]
    ):
        raise RecoveryC7LaunchError("Stage A environment binding drifted")
    input_record = _exact_keys(
        receipt["input"], _RECEIPT_INPUT_KEYS, "Stage A input",
    )
    inputs = binding.stage_c_input
    expected_input = {
        "sample_id": f"ILSVRC2012_val/{inputs['sample_relative_path']}",
        "path": f"{inputs['data_root']}/{inputs['sample_relative_path']}",
        "sha256": inputs["sample_sha256"],
        "ground_truth_wnid": inputs["sample_wnid"],
        "ground_truth_index": inputs["sample_imagenet1k_index"],
        "label_source": "official_devkit_val_gt",
        "source_manifest_path": inputs["source_manifest_path"],
        "source_manifest_sha256": inputs["source_manifest_sha256"],
        "official_devkit_path": inputs["official_devkit_path"],
        "official_devkit_sha256": inputs["official_devkit_sha256"],
    }
    if any(input_record[key] != value for key, value in expected_input.items()):
        raise RecoveryC7LaunchError("Stage A input identity drifted")
    sample_path = repo / input_record["path"]
    if (type(input_record["bytes"]) is not int or input_record["bytes"] <= 0
            or sample_path.stat().st_size != input_record["bytes"]
            or sha256_file(sample_path) != input_record["sha256"]
            or type(input_record["width"]) is not int or input_record["width"] <= 0
            or type(input_record["height"]) is not int or input_record["height"] <= 0):
        raise RecoveryC7LaunchError("Stage A input file metadata drifted")
    lock_path = repo / binding.document["environment"]["requirements_lock_path"]
    if environment_record["requirements_lock"] != {
        "path": binding.document["environment"]["requirements_lock_path"],
        "sha256": binding.document["environment"]["requirements_lock_sha256"],
        "bytes": lock_path.stat().st_size,
    }:
        raise RecoveryC7LaunchError("Stage A requirements lock drifted")

    artifacts = _exact_keys(
        receipt["artifacts"], set(_ARTIFACT_NAMES), "Stage A artifacts",
    )
    result_artifacts: dict[str, Any] = {}
    loaded: dict[str, Any] = {}
    for name in _ARTIFACT_NAMES:
        item = _exact_keys(
            artifacts[name], {"bytes", "sha256"}, f"Stage A artifact {name}",
        )
        path = receipt_path.parent / name
        _regular_single_link_file(path, f"Stage A artifact {name}")
        if (type(item["bytes"]) is not int or item["bytes"] <= 0
                or not isinstance(item["sha256"], str)
                or _SHA256_RE.fullmatch(item["sha256"]) is None
                or path.stat().st_size != item["bytes"]
                or sha256_file(path) != item["sha256"]):
            raise RecoveryC7LaunchError(f"Stage A artifact drifted: {name}")
        _verify_lfs_pointer_at_head(
            repo, top, path, expected_sha256=item["sha256"],
        )
        result_artifacts[name] = _artifact_record(root, name, item)
        if name.endswith(".json"):
            loaded[name] = _read_json(path, f"Stage A artifact {name}")
    validation = _exact_keys(
        loaded["validation.json"], _VALIDATION_KEYS, "Stage A numeric validation",
    )
    _require_self_digest(validation, "Stage A numeric validation")
    pass_keys = (
        "all_pass", "attention_pass", "hook_noninvasive_pass", "logits_pass",
        "top5_match_pass",
    )
    if (validation.get("document_type") != "recovery_v2_stage_a_numeric_validation"
            or validation.get("status") != "PASS"
            or any(validation.get(key) is not True for key in pass_keys)
            or validation.get("legacy_pickle_enabled") is not False
            or validation.get("legacy_pickle_used") is not False
            or canonical_json_bytes(receipt["numeric_gates"])
               != canonical_json_bytes(validation)):
        raise RecoveryC7LaunchError("Stage A numeric PASS evidence drifted")
    provenance = _exact_keys(
        loaded["provenance.json"], _PROVENANCE_KEYS, "Stage A provenance",
    )
    _require_self_digest(provenance, "Stage A provenance")
    if (provenance.get("document_type") != "recovery_v2_stage_a_provenance"
            or provenance.get("status") != "PASS"
            or provenance.get("artifacts") != {
                name: dict(artifacts[name])
                for name in ("extraction.pt", "validation.json")
            }
            or any(canonical_json_bytes(provenance.get(key))
                   != canonical_json_bytes(receipt[key])
                   for key in ("binding", "amendment", "scope", "model", "input",
                               "environment", "source_git_state", "numeric_gates",
                               "legacy_pickle"))):
        raise RecoveryC7LaunchError("Stage A provenance binding drifted")
    receipt_sha = sha256_file(receipt_path)
    _verify_lfs_pointer_at_head(repo, top, receipt_path, expected_sha256=receipt_sha)
    result = {
        "root": root, "receipt_path": receipt_relative,
        "receipt_file_sha256": receipt_sha,
        "receipt_canonical_digest": receipt["canonical_digest"], "status": "PASS",
        "producer_commit": source["commit"], "producer_tree": source["tree"],
        "artifacts": result_artifacts,
    }
    if expected is not None and canonical_json_bytes(expected) != canonical_json_bytes(result):
        raise RecoveryC7LaunchError("Stage A launch binding drifted")
    return result


def _capture_clean_producer(repo: Path, top: Path) -> dict[str, Any]:
    if _git_status_entries(repo):
        raise RecoveryC7LaunchError(
            "Recovery C7 launch generation requires a clean Git worktree"
        )
    commit = _run_git(repo, "rev-parse", "--verify", "HEAD^{commit}")
    tree = _run_git(repo, "rev-parse", "--verify", "HEAD^{tree}")
    source, _relative = _project_path(
        repo, SOURCE_SUBTREE_PATH, "producer source subtree", require_exists=True,
    )
    source_tree = _run_git(repo, "rev-parse", f"HEAD:{_top_relative(top, source)}")
    if (_COMMIT_RE.fullmatch(commit) is None
            or _GIT_OBJECT_RE.fullmatch(tree) is None
            or _GIT_OBJECT_RE.fullmatch(source_tree) is None):
        raise RecoveryC7LaunchError("Producer Git identity is invalid")
    return {
        "commit": commit, "tree": tree, "source_subtree_path": SOURCE_SUBTREE_PATH,
        "source_subtree_tree": source_tree, "dirty": False,
    }


def _critical_paths(
    repo: Path, schema: Path, binding: RecoveryV2Binding,
    stage_a: Mapping[str, Any], contracts: Mapping[str, Any],
) -> list[Path]:
    paths = [schema, binding.path, binding.schema_path, binding.model_path]
    paths.append(repo / binding.document["environment"]["requirements_lock_path"])
    paths.append(repo / binding.document["amendment"]["path"])
    paths.extend(repo / item["path"] for item in contracts.values())
    paths.append(repo / stage_a["receipt_path"])
    paths.extend(repo / item["path"] for item in stage_a["artifacts"].values())
    return paths


def _validate_git_authority(
    repo: Path, top: Path, launch: Path, document: Mapping[str, Any],
    critical: Sequence[Path], *, require_tracked: bool,
) -> str:
    producer = document["producer_git"]
    commit = producer["commit"]
    if not isinstance(commit, str) or _COMMIT_RE.fullmatch(commit) is None:
        raise RecoveryC7LaunchError("Producer commit is invalid")
    if producer.get("dirty") is not False:
        raise RecoveryC7LaunchError("Producer Git record is dirty")
    tree = _run_git(repo, "rev-parse", "--verify", f"{commit}^{{tree}}")
    if tree != producer["tree"]:
        raise RecoveryC7LaunchError("Producer tree drifted")
    source, _relative = _project_path(
        repo, producer["source_subtree_path"], "producer source subtree",
        require_exists=True,
    )
    source_top = _top_relative(top, source)
    producer_source = _run_git(repo, "rev-parse", f"{commit}:{source_top}")
    if producer_source != producer["source_subtree_tree"]:
        raise RecoveryC7LaunchError("Producer source subtree drifted")
    head = _run_git(repo, "rev-parse", "--verify", "HEAD^{commit}")
    _is_ancestor(repo, commit, head, "Producer commit")
    _is_ancestor(repo, document["stage_a"]["producer_commit"], commit,
                 "Stage A producer commit")
    _is_ancestor(repo, document["amendment"]["commit"], commit,
                 "Recovery amendment commit")
    if _run_git(repo, "rev-parse", f"HEAD:{source_top}") != producer_source:
        raise RecoveryC7LaunchError("Critical producer source drifted after launch")
    launch_top = _top_relative(top, launch)
    declared = document["launch_record_commit"]
    if not require_tracked:
        if declared != "SELF" or head != commit:
            raise RecoveryC7LaunchError(
                "Uncommitted launch validation requires SELF at producer HEAD"
            )
        entries = _git_status_entries(repo)
        if entries != [{"status": "??", "path": launch_top}]:
            raise RecoveryC7LaunchError(
                "Uncommitted validation permits only the fresh launch record"
            )
        return "SELF"
    if declared == "SELF":
        launch_commit = _run_git(
            repo, "log", "-n", "1", "--format=%H", "HEAD", "--",
            f":(top){launch_top}",
        )
        if not launch_commit:
            raise RecoveryC7LaunchError(
                "SELF launch record has no containing Git commit"
            )
    else:
        launch_commit = declared
    if not isinstance(launch_commit, str) or _COMMIT_RE.fullmatch(launch_commit) is None:
        raise RecoveryC7LaunchError("Launch-record commit is invalid")
    _is_ancestor(repo, launch_commit, head, "Launch-record commit")
    parents = _run_git(
        repo, "rev-list", "--parents", "-n", "1", launch_commit,
    ).split()
    if parents != [launch_commit, commit]:
        raise RecoveryC7LaunchError(
            "Launch record must be one commit directly after the producer"
        )
    changed = [line for line in _run_git(
        repo, "diff", "--name-only", commit, launch_commit, "--",
    ).splitlines() if line]
    if changed != [launch_top]:
        raise RecoveryC7LaunchError(
            "Producer-to-launch commit drift exceeds the launch record"
        )
    launch_blob = _run_git(repo, "rev-parse", f"{launch_commit}:{launch_top}")
    head_blob = _run_git(repo, "rev-parse", f"HEAD:{launch_top}")
    if launch_blob != head_blob:
        raise RecoveryC7LaunchError(
            "Current HEAD launch-record blob differs from its launch commit"
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
            raise RecoveryC7LaunchError(
                f"Critical production input drifted after producer: {top_path}"
            )
    return launch_commit


def _make_record(
    repo: Path, launch: Path, schema: Path, document: Mapping[str, Any],
    binding: RecoveryV2Binding, diagnostic: Path, formal: Path, reports: Path,
    launch_commit: str,
) -> RecoveryC7LaunchRecord:
    return RecoveryC7LaunchRecord(
        path=launch, schema_path=schema, document=_freeze(document), binding=binding,
        model_path=binding.model_path, data_root=binding.data_root,
        devkit_path=binding.devkit_path, diagnostic_root=diagnostic,
        formal_root=formal, reports_root=reports,
        model_identity=_freeze(document["model"]), file_sha256=sha256_file(launch),
        launch_commit=launch_commit, repo_root=repo,
    )


def _publish_exclusive(repo: Path, launch: Path, payload: bytes) -> None:
    """Create the launch file through no-follow directory descriptors."""

    try:
        relative_parent = launch.parent.relative_to(repo)
    except ValueError as error:
        raise RecoveryC7LaunchError("Launch parent escapes the active project") from error
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    descriptor = -1
    created = False
    complete = False
    try:
        descriptor = os.open(repo, directory_flags | nofollow)
        for part in relative_parent.parts:
            try:
                os.mkdir(part, mode=0o755, dir_fd=descriptor)
            except FileExistsError:
                pass
            child = os.open(part, directory_flags | nofollow, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | nofollow
        file_descriptor = os.open(launch.name, flags, 0o644, dir_fd=descriptor)
        created = True
        with os.fdopen(file_descriptor, "w+b") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            handle.seek(0)
            observed = handle.read()
            info = os.fstat(handle.fileno())
            if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
                    or observed != payload):
                raise RecoveryC7LaunchError("Published launch record drifted")
            complete = True
    except RecoveryC7LaunchError:
        raise
    except OSError as error:
        raise RecoveryC7LaunchError("Cannot exclusively publish launch record") from error
    finally:
        if descriptor >= 0:
            if created and not complete:
                try:
                    os.unlink(launch.name, dir_fd=descriptor)
                except FileNotFoundError:
                    pass
                except OSError:
                    pass
            os.close(descriptor)


def generate_recovery_c7_launch_record(
    *, repo_root: Path | str, launch_record_path: Path | str,
    diagnostic_root: Path | str, formal_root: Path | str,
    reports_root: Path | str, binding_path: Path | str = DEFAULT_BINDING_PATH,
    stage_a_receipt_path: Path | str = DEFAULT_STAGE_A_RECEIPT_PATH,
    schema_path: Path | str = DEFAULT_SCHEMA_PATH,
) -> dict[str, Any]:
    """Generate a launch document from a pristine implementation HEAD."""

    repo = Path(repo_root).resolve(strict=True)
    top, _prefix = _repository_identity(repo)
    launch, common, diagnostic, formal, reports = _validate_output_paths(
        repo, launch_record_path, diagnostic_root, formal_root, reports_root,
        require_fresh_roots=True, launch_must_exist=False,
    )
    if launch.exists() or launch.is_symlink():
        raise RecoveryC7LaunchError("Launch record path is not fresh")
    schema, schema_rel = _project_path(
        repo, schema_path, "launch schema", require_exists=True,
    )
    if schema_rel != SCHEMA_RELATIVE_PATH:
        raise RecoveryC7LaunchError("Unexpected recovery C7 launch schema path")
    schema_document = _read_json(schema, "recovery C7 launch schema")
    try:
        Draft202012Validator.check_schema(schema_document)
    except Exception as error:
        raise RecoveryC7LaunchError("Recovery C7 launch schema is invalid") from error
    schema_sha256 = sha256_file(schema)
    binding_file, _binding_rel = _project_path(
        repo, binding_path, "recovery-v2 binding", require_exists=True,
    )
    binding = load_recovery_v2_binding(
        binding_file, schema_path=repo / "config/recovery_v2_binding.schema.json",
        repo_root=repo, verify_assets=True,
    )
    lfs = verify_recovery_v2_lfs_identity(binding, repo_root=repo)
    if lfs["oid_sha256"] != binding.model["lfs_oid_sha256"] or not lfs["hydrated"]:
        raise RecoveryC7LaunchError("Recovery-v2 model LFS identity drifted")
    receipt, _receipt_rel = _project_path(
        repo, stage_a_receipt_path, "Stage A terminal receipt", require_exists=True,
    )
    stage_a = _validate_stage_a_receipt(repo, top, receipt, binding)
    contracts = _contract_records(repo, binding)
    dependencies = _dependency_record(repo, binding)
    producer = _capture_clean_producer(repo, top)
    _is_ancestor(repo, stage_a["producer_commit"], producer["commit"],
                 "Stage A producer commit")
    for probe in (launch, diagnostic / ".probe", formal / ".probe", reports / ".probe"):
        relative = probe.relative_to(repo).as_posix()
        if not _run_git(repo, "check-attr", "filter", "--", relative).endswith(
            ": filter: lfs"
        ):
            raise RecoveryC7LaunchError("Recovery C7 evidence lacks Git LFS coverage")
    # Detect races after all metadata and LFS checks and before returning an
    # authorization payload.  The writer performs one more identical check
    # immediately before its first filesystem mutation.
    if _capture_clean_producer(repo, top) != producer:
        raise RecoveryC7LaunchError("Producer Git state changed during generation")
    amendment = binding.document["amendment"]
    document: dict[str, Any] = {
        "$schema": "recovery_c7_launch_record.schema.json", "schema_version": 1,
        "document_type": DOCUMENT_TYPE, "record_id": RECORD_ID, "status": STATUS,
        "launch_record_commit": "SELF",
        "launch_schema": {"path": schema_rel, "file_sha256": schema_sha256},
        "amendment": {
            "id": amendment["id"], "path": amendment["path"],
            "commit": amendment["commit"], "file_sha256": amendment["sha256"],
        },
        "recovery_binding": _binding_record(repo, binding), "stage_a": stage_a,
        "model": _model_record(binding), "dependencies": dependencies,
        "c7_contracts": contracts, "producer_git": producer,
        "outputs": {
            "common_root": common.relative_to(repo).as_posix(),
            "launch_record_path": launch.relative_to(repo).as_posix(),
            "diagnostic_root": diagnostic.relative_to(repo).as_posix(),
            "formal_root": formal.relative_to(repo).as_posix(),
            "reports_root": reports.relative_to(repo).as_posix(),
        },
        "scope": dict(_SCOPE),
    }
    document["canonical_digest"] = launch_record_digest(document)
    _validate_schema(document, schema_document)
    _validate_intrinsic_authority(document)
    if sha256_file(schema) != schema_sha256:
        raise RecoveryC7LaunchError("Recovery C7 launch schema changed during generation")
    return document


def write_recovery_c7_launch_record(
    *, repo_root: Path | str, launch_record_path: Path | str,
    diagnostic_root: Path | str, formal_root: Path | str,
    reports_root: Path | str, binding_path: Path | str = DEFAULT_BINDING_PATH,
    stage_a_receipt_path: Path | str = DEFAULT_STAGE_A_RECEIPT_PATH,
    schema_path: Path | str = DEFAULT_SCHEMA_PATH,
) -> RecoveryC7LaunchRecord:
    """Generate and exclusively publish an uncommitted launch record."""

    repo = Path(repo_root).resolve(strict=True)
    document = generate_recovery_c7_launch_record(
        repo_root=repo, launch_record_path=launch_record_path,
        diagnostic_root=diagnostic_root, formal_root=formal_root,
        reports_root=reports_root, binding_path=binding_path,
        stage_a_receipt_path=stage_a_receipt_path, schema_path=schema_path,
    )
    launch, _common, diagnostic, formal, reports = _validate_output_paths(
        repo, launch_record_path, diagnostic_root, formal_root, reports_root,
        require_fresh_roots=True, launch_must_exist=False,
    )
    expected_producer = document["producer_git"]
    top, _prefix = _repository_identity(repo)
    if _capture_clean_producer(repo, top) != expected_producer:
        raise RecoveryC7LaunchError("Producer Git state changed before publication")
    payload = _json_file_bytes(document)
    _publish_exclusive(repo, launch, payload)
    binding = load_recovery_v2_binding(
        repo / document["recovery_binding"]["path"],
        schema_path=repo / document["recovery_binding"]["schema_path"],
        repo_root=repo, verify_assets=False,
    )
    return _make_record(
        repo, launch, repo / document["launch_schema"]["path"], document, binding,
        diagnostic, formal, reports, "SELF",
    )


def load_recovery_c7_launch_record(
    path: Path | str, *, repo_root: Path | str = REPO_ROOT,
    schema_path: Path | str = DEFAULT_SCHEMA_PATH,
    require_fresh_roots: bool = True, require_tracked: bool = True,
) -> RecoveryC7LaunchRecord:
    """Source-recompute and validate a recovery C7 launch record.

    Freshness is required before the first diagnostic. Gate and report
    consumers disable it once bound roots contain immutable attempt evidence.
    """

    repo = Path(repo_root).resolve(strict=True)
    top, _prefix = _repository_identity(repo)
    launch, launch_rel = _project_path(
        repo, path, "launch record path", require_exists=True,
    )
    schema, schema_rel = _project_path(
        repo, schema_path, "launch schema", require_exists=True,
    )
    if schema_rel != SCHEMA_RELATIVE_PATH:
        raise RecoveryC7LaunchError("Unexpected recovery C7 launch schema path")
    document = _read_json(launch, "recovery C7 launch record")
    schema_document = _read_json(schema, "recovery C7 launch schema")
    if not isinstance(document, Mapping):
        raise RecoveryC7LaunchError("Recovery C7 launch record must be an object")
    _validate_schema(document, schema_document)
    _validate_intrinsic_authority(document)
    if document["canonical_digest"] != launch_record_digest(document):
        raise RecoveryC7LaunchError("Recovery C7 launch canonical digest drifted")
    if document["launch_schema"] != {
        "path": schema_rel, "file_sha256": sha256_file(schema),
    }:
        raise RecoveryC7LaunchError("Recovery C7 launch schema binding drifted")
    if document["outputs"]["launch_record_path"] != launch_rel:
        raise RecoveryC7LaunchError("Launch record path binding drifted")
    outputs = document["outputs"]
    checked, common, diagnostic, formal, reports = _validate_output_paths(
        repo, launch, outputs["diagnostic_root"], outputs["formal_root"],
        outputs["reports_root"], require_fresh_roots=require_fresh_roots,
        launch_must_exist=True,
    )
    if outputs["common_root"] != common.relative_to(repo).as_posix():
        raise RecoveryC7LaunchError("Launch common-root binding drifted")

    binding_record = document["recovery_binding"]
    binding = load_recovery_v2_binding(
        repo / binding_record["path"],
        schema_path=repo / binding_record["schema_path"], repo_root=repo,
        verify_assets=True,
    )
    if canonical_json_bytes(binding_record) != canonical_json_bytes(
        _binding_record(repo, binding)
    ):
        raise RecoveryC7LaunchError("Recovery-v2 binding identity drifted")
    lfs = verify_recovery_v2_lfs_identity(binding, repo_root=repo)
    if lfs["oid_sha256"] != binding.model["lfs_oid_sha256"] or not lfs["hydrated"]:
        raise RecoveryC7LaunchError("Recovery-v2 model LFS identity drifted")
    if canonical_json_bytes(document["model"]) != canonical_json_bytes(
        _model_record(binding)
    ):
        raise RecoveryC7LaunchError("Recovery-v2 model launch identity drifted")
    if document["dependencies"] != _dependency_record(repo, binding):
        raise RecoveryC7LaunchError("Recovery-v2 dependency binding drifted")
    contracts = _contract_records(repo, binding)
    if canonical_json_bytes(document["c7_contracts"]) != canonical_json_bytes(contracts):
        raise RecoveryC7LaunchError("Recovery-v2 C7 contracts drifted")
    amendment = binding.document["amendment"]
    if document["amendment"] != {
        "id": amendment["id"], "path": amendment["path"],
        "commit": amendment["commit"], "file_sha256": amendment["sha256"],
    }:
        raise RecoveryC7LaunchError("Recovery amendment launch binding drifted")
    receipt, _receipt_rel = _project_path(
        repo, document["stage_a"]["receipt_path"], "Stage A terminal receipt",
        require_exists=True,
    )
    stage_a = _validate_stage_a_receipt(
        repo, top, receipt, binding, document["stage_a"],
    )
    launch_commit = _validate_git_authority(
        repo, top, checked, document,
        _critical_paths(repo, schema, binding, stage_a, contracts),
        require_tracked=require_tracked,
    )
    record = _make_record(
        repo, checked, schema, document, binding, diagnostic, formal, reports,
        launch_commit,
    )
    return record


def validate_recovery_c7_launch_record(
    path: Path | str, *, repo_root: Path | str = REPO_ROOT,
    schema_path: Path | str = DEFAULT_SCHEMA_PATH,
    require_fresh_roots: bool = True, require_tracked: bool = True,
) -> RecoveryC7LaunchRecord:
    """CLI-friendly alias for read-only launch-record loading/validation."""

    return load_recovery_c7_launch_record(
        path, repo_root=repo_root, schema_path=schema_path,
        require_fresh_roots=require_fresh_roots, require_tracked=require_tracked,
    )


def capture_authorized_c7_source_state(
    record: RecoveryC7LaunchRecord, allowed_phase: str,
) -> dict[str, Any]:
    """Validate phase-scoped evidence dirt and return logical clean code state."""

    if not isinstance(record, RecoveryC7LaunchRecord):
        raise RecoveryC7LaunchError("A validated RecoveryC7LaunchRecord is required")
    if allowed_phase not in _PHASE_ROOTS:
        raise RecoveryC7LaunchError(f"Unknown recovery C7 phase: {allowed_phase}")
    repo = record.repo_root.resolve(strict=True)
    top, _prefix = _repository_identity(repo)
    if allowed_phase == "pre_diagnostic":
        _validate_output_paths(
            repo, record.path, record.diagnostic_root, record.formal_root,
            record.reports_root, require_fresh_roots=True,
            launch_must_exist=True,
        )
    head_before = _run_git(repo, "rev-parse", "--verify", "HEAD^{commit}")
    if record.launch_commit == "SELF" or head_before != record.launch_commit:
        raise RecoveryC7LaunchError(
            "A committed launch record at the current HEAD is required"
        )
    status_before = _git_status_entries(repo)
    _verify_lfs_pointer_at_head(
        repo, top, record.path, expected_sha256=record.file_sha256,
    )
    allowed_roots = tuple(getattr(record, name) for name in _PHASE_ROOTS[allowed_phase])
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
            raise RecoveryC7LaunchError("Launch record itself is dirty")
        if entry.get("status") != "??" or "source_path" in entry:
            raise RecoveryC7LaunchError(
                "Recovery C7 evidence dirt must be newly created and untracked"
            )
        if any(path not in allowed_exact and not path.startswith(allowed_prefixes)
               for path in paths):
            raise RecoveryC7LaunchError(
                f"Git dirt is outside roots authorized for {allowed_phase}"
            )
    head_after = _run_git(repo, "rev-parse", "--verify", "HEAD^{commit}")
    status_after = _git_status_entries(repo)
    if head_after != head_before or status_after != status_before:
        raise RecoveryC7LaunchError(
            "Git HEAD or status changed during recovery C7 source capture"
        )
    return {"commit": head_before, "dirty": False}


def recovery_provenance_context(record: RecoveryC7LaunchRecord) -> dict[str, Any]:
    """Return the validated object pair consumed by Stage C provenance."""

    if not isinstance(record, RecoveryC7LaunchRecord):
        raise RecoveryC7LaunchError("A validated RecoveryC7LaunchRecord is required")
    return {
        "recovery_binding": record.binding,
        "recovery_launch_record": record,
    }


__all__ = [
    "DEFAULT_SCHEMA_PATH", "DEFAULT_STAGE_A_RECEIPT_PATH", "DOCUMENT_TYPE",
    "LAUNCH_RECORD_FILENAME", "RECORD_ID", "STATUS", "RecoveryC7LaunchError",
    "RecoveryC7LaunchRecord", "canonical_json_bytes", "canonical_sha256",
    "capture_authorized_c7_source_state", "generate_recovery_c7_launch_record",
    "launch_record_digest", "load_recovery_c7_launch_record",
    "recovery_provenance_context", "validate_recovery_c7_launch_record",
    "write_recovery_c7_launch_record",
]
