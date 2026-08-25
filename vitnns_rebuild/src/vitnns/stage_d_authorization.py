"""Immutable, preparation-only authorization for the Stage D control plane.

Artifact construction consumes issued, already validated metadata objects and
performs no image, model, outcome, or GPU access.  The separate capture helper
performs bounded, descriptor-safe reads of exactly two small metadata files and
captures a clean Git commit/tree.  A Stage D preparation artifact is not an
execution capability: execution remains unissued until the separate D7 gate,
and analysis and scientific results remain unevaluated.
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
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any

from .stage_d_config import (
    STAGE_D_CONFIG_CANONICAL_DIGEST,
    STAGE_D_CONFIG_FILE_SHA256,
    STAGE_D_CONFIG_SCHEMA_FILE_SHA256,
    STAGE_D_RUN_SPEC_CANONICAL_DIGEST,
    STAGE_D_RUN_SPEC_FILE_SHA256,
    STAGE_D_RUN_SPEC_SCHEMA_FILE_SHA256,
    StageDContracts,
)
from .stage_d_manifest import (
    D1_COMPLETION_COMMIT,
    D1_COMPLETION_REF,
    D2_COMPLETION_COMMIT,
    D2_COMPLETION_REF,
    StageDManifest,
)
from .stage_d_plan import (
    StageDPlanError,
    preflight_stage_d,
    validate_stage_d_estimate,
    validate_stage_d_plan,
    validate_stage_d_preflight,
)
from .stage_d_registry import (
    D0_COMPLETION_COMMIT,
    D0_COMPLETION_REF,
    D0_SCOPE_CANONICAL_DIGEST,
    D0_SCOPE_FILE_SHA256,
    D0_SCOPE_MARKDOWN_SHA256,
    D0_SCOPE_SCHEMA_SHA256,
    PACKAGE_FILES,
    PACKAGE_ID,
)
from .constants import REPO_ROOT


SCHEMA_VERSION = 1
DOCUMENT_TYPE = "stage_d_preparation_authorization"
STATUS = "PASS"
PREPARATION_AUTHORIZATION = "APPROVED_TO_PREPARE_STAGE_D"
EXECUTION_AUTHORIZATION = "NOT_ISSUED"
ANALYSIS_AUTHORIZATION = "NOT_EVALUATED"
SCIENTIFIC_RESULT = "NOT_EVALUATED"
REQUIREMENTS_LOCK_PATH = "requirements.lock.txt"
REQUIREMENTS_LOCK_SHA256 = (
    "81beabd6aba9ede777e973e2ee5f0c60ce2aa88ce55b29d025e95d67ab175984"
)
REQUIREMENTS_LOCK_BYTES = 1788
MODEL_NAME = "vit_base_patch16_224.augreg2_in21k_ft_in1k"
MODEL_CONFIG_PATH = "config/model.json"
MODEL_CONFIG_SHA256 = (
    "cc8d386736b3f0ae299e75184e86772b21b8139cab0a7c0eab17a961d1ad55a3"
)
MODEL_CONFIG_BYTES = 636
MODEL_STATE_DICT_SHA256 = (
    "2e718037429f18ec77aa629471beb2e99993d7383375188bab777aba04d5e4d0"
)

# Keep the prohibited D7 capability out of emitted D3 artifacts.  Splitting the
# source literal also helps static reviewers distinguish a guard from issuance.
_D7_EXECUTION_CAPABILITY = "AUTHORIZED_STAGE_D_" + "FIDELITY_SCOPE"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")

_ROOT_KEYS = (
    "schema_version",
    "document_type",
    "status",
    "preparation_authorization",
    "execution_authorization",
    "analysis_authorization",
    "stage_d_execution_started",
    "scientific_result",
    "c10_historical_status",
    "source_bindings",
    "plan_binding",
    "budget_binding",
    "preparation_inputs_digest",
    "code",
    "dependencies",
    "model",
    "canonical_digest",
)

_C10_HISTORICAL_STATUS = {
    "primary_decision": "NO_GO_NOVELTY",
    "spatial_status": "FAIL_SPATIAL",
    "stage_c_acceptance": "BLOCKED",
    "workflow_blocker": "HOLM_BLOCKED_UNDERSPECIFIED",
    "downstream_authorized": False,
    "stage_d_started": False,
}

_D0_CONFIG_KEYS = (
    "completion_ref", "completion_commit", "machine_record_path",
    "machine_record_file_sha256", "machine_record_canonical_digest",
    "schema_file_sha256", "markdown_file_sha256",
)
_D0_MANIFEST_KEYS = (*_D0_CONFIG_KEYS, "verification_status", "schema_path", "document_path")
_D1_CONFIG_KEYS = (
    "completion_ref", "completion_commit", "package_id",
    "package_manifest_path", "package_manifest_file_sha256",
    "package_manifest_canonical_digest", "registry_path",
    "registry_file_sha256", "registry_canonical_digest",
    "registry_schema_file_sha256", "registry_schema_canonical_digest",
    "source_binding_file_sha256", "source_binding_canonical_digest",
    "registry_metadata_file_sha256",
    "registry_metadata_canonical_digest", "attestation_path",
    "attestation_file_sha256", "attestation_canonical_digest",
    "ordered_sample_ids_sha256", "candidate_recorded_image_hashes_only",
    "live_image_equality_status",
)
_D1_MANIFEST_KEYS = (
    *_D1_CONFIG_KEYS, "verification_status", "package_dir",
    "package_file_count", "package_file_sha256",
    "package_canonical_digests", "registry_row_count",
    "audit_overlay_count", "audit_sample_ids",
    "execution_authorization",
)
_D2_MANIFEST_KEYS = (
    "verification_status", "completion_ref", "completion_commit",
    "config_path", "config_file_sha256",
    "config_canonical_digest", "config_schema_path",
    "config_schema_file_sha256",
)
_D2_ARTIFACT_KEYS = (
    *_D2_MANIFEST_KEYS, "run_spec_path", "run_spec_file_sha256",
    "run_spec_canonical_digest", "run_spec_schema_path",
    "run_spec_schema_file_sha256",
)
_C10_CONFIG_KEYS = (
    "formal_root", "completion_commit", "producer_ref",
    "producer_commit", "producer_tree", "producer_source_tree",
    "status_file_sha256", "status_canonical_digest",
    "report_file_sha256", "report_canonical_digest",
    "decision_file_sha256", "decision_canonical_digest",
    "holm_file_sha256", "holm_canonical_digest",
    "provenance_file_sha256", "provenance_canonical_digest",
    "manifest_file_sha256", "manifest_canonical_digest",
    "whole_root_file_count", "whole_root_total_bytes",
    "whole_root_inventory_digest", "primary_decision",
    "stage_c_acceptance", "workflow_blocker", "spatial_status",
    "downstream_authorized", "stage_d_started", "cost_evidence_role",
)
_C10_MANIFEST_KEYS = (*_C10_CONFIG_KEYS, "verification_status", "scientific_results_recomputed")

_REGISTRY_KEYS = (
    "schema_ref", "schema_version", "document_type", "package_id",
    "status", "source_binding_canonical_digest",
    "candidate_reserve_binding", "cohort", "image_byte_scope",
    "rows", "ordered_sample_ids_sha256",
    "ordered_relative_paths_sha256", "ordered_image_sha256s_sha256",
    "rows_canonical_digest", "audit_selection", "canonical_digest",
)
_PLAN_BINDING_KEYS = (
    "document_type", "canonical_digest", "config_digest",
    "run_spec_digest", "registry_digest",
    "ordered_sample_ids_digest", "context_digest",
    "coordinate_digest", "counts",
)
_BUDGET_BINDING_KEYS = (
    "estimate_document_type", "estimate_canonical_digest",
    "plan_digest", "profile_status", "budget_status",
    "launch_safety_caps", "completed_campaign_gates",
    "launch_violations", "completed_gate_violations",
    "preflight_document_type", "preflight_canonical_digest",
    "preflight_status", "preflight_execution_authorization",
    "preflight_analysis_authorization", "d7_final",
)
_CODE_KEYS = ("commit", "tree", "dirty")
_DEPENDENCY_KEYS = (
    "requirements_lock_path", "requirements_lock_sha256",
    "requirements_lock_bytes",
)
_MODEL_KEYS = (
    "model_name", "model_config_path", "model_config_sha256",
    "model_config_bytes", "state_dict_sha256",
    "state_dict_file_accessed",
)
_PREPARATION_INPUT_KEYS = (
    "schema_version", "document_type", "status", "code",
    "dependencies", "model", "canonical_digest",
)
_PREPARATION_INPUTS_ISSUER = object()
_MAX_STATIC_METADATA_BYTES = 1024 * 1024
_MAX_GIT_OUTPUT_BYTES = 1024 * 1024
_GIT_TIMEOUT_SECONDS = 15


class StageDAuthorizationError(ValueError):
    """Raised when a Stage D preparation artifact fails closed."""


def _plain(value: Any, context: str = "value") -> Any:
    convert = getattr(value, "to_dict", None)
    if callable(convert):
        try:
            value = convert()
        except Exception as error:  # pragma: no cover - defensive adapter
            raise StageDAuthorizationError(
                f"{context}.to_dict() failed"
            ) from error
    if value is None or type(value) in {str, bool, int}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise StageDAuthorizationError(
                f"{context} contains a non-finite number"
            )
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise StageDAuthorizationError(
                    f"{context} keys must be non-empty strings"
                )
            result[key] = _plain(item, f"{context}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [
            _plain(item, f"{context}[{index}]")
            for index, item in enumerate(value)
        ]
    raise StageDAuthorizationError(
        f"{context} is not JSON-compatible: {type(value).__name__}"
    )


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return copy.deepcopy(value)


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return copy.deepcopy(value)


def canonical_json_bytes(value: Any) -> bytes:
    """Return the repository's compact, sorted, finite UTF-8 JSON bytes."""

    return json.dumps(
        _plain(value, "canonical JSON input"), sort_keys=True,
        separators=(",", ":"), ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _same_json(left: Any, right: Any) -> bool:
    """Compare JSON values without Python's bool/int equality aliasing."""

    return canonical_json_bytes(left) == canonical_json_bytes(right)


def preparation_authorization_digest(document: Any) -> str:
    value = _mapping(document, "preparation authorization")
    return canonical_sha256({
        key: item for key, item in value.items() if key != "canonical_digest"
    })


def _mapping(value: Any, context: str) -> dict[str, Any]:
    plain = _plain(value, context)
    if not isinstance(plain, dict):
        raise StageDAuthorizationError(
            f"{context} must be a mapping or expose to_dict()"
        )
    return plain


def _exact_keys(value: Any, keys: Sequence[str], context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise StageDAuthorizationError(f"{context} must be an object")
    expected = set(keys)
    actual = set(value)
    if actual != expected:
        raise StageDAuthorizationError(
            f"{context} fields drifted: missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}"
        )
    return value


def _sha256(value: Any, context: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise StageDAuthorizationError(
            f"{context} must be a lowercase SHA-256"
        )
    return value


def _self_digest(document: Mapping[str, Any]) -> str:
    return canonical_sha256({
        key: item for key, item in document.items() if key != "canonical_digest"
    })


def _assert_self_digest(
    document: Mapping[str, Any], expected: str, context: str,
) -> None:
    observed = document.get("canonical_digest")
    _sha256(observed, f"{context}.canonical_digest")
    if observed != _self_digest(document) or observed != expected:
        raise StageDAuthorizationError(
            f"{context} canonical digest or trusted identity drifted"
        )


def _contains_prohibited_capability(value: Any) -> bool:
    if isinstance(value, str):
        return _D7_EXECUTION_CAPABILITY in value
    if isinstance(value, Mapping):
        return any(
            _contains_prohibited_capability(key)
            or _contains_prohibited_capability(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_prohibited_capability(item) for item in value)
    return False


def _validate_code(value: Any) -> dict[str, Any]:
    code = _mapping(value, "code metadata")
    _exact_keys(code, _CODE_KEYS, "code metadata")
    if (
        not isinstance(code["commit"], str)
        or _COMMIT_RE.fullmatch(code["commit"]) is None
    ):
        raise StageDAuthorizationError(
            "code metadata commit must be a lowercase 40-hex Git commit"
        )
    if (
        not isinstance(code["tree"], str)
        or _COMMIT_RE.fullmatch(code["tree"]) is None
    ):
        raise StageDAuthorizationError(
            "code metadata tree must be a lowercase 40-hex Git tree"
        )
    if code["dirty"] is not False:
        raise StageDAuthorizationError(
            "Stage D preparation refuses a dirty producer code state"
        )
    return {
        "commit": code["commit"], "tree": code["tree"],
        "dirty": False,
    }


def _safe_relative_path(value: str, context: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise StageDAuthorizationError(
            f"{context} must be a normalized relative POSIX path"
        )
    candidate = PurePosixPath(value)
    if (
        candidate.is_absolute()
        or candidate.as_posix() != value
        or any(part in {"", ".", ".."} for part in candidate.parts)
    ):
        raise StageDAuthorizationError(
            f"{context} must be a normalized relative POSIX path"
        )
    return value


def _safe_read_static_metadata(
    repo_root: Path | str, relative_path: str, *, expected_sha256: str,
) -> tuple[bytes, str]:
    """Read one small ordinary single-link file from stable descriptors."""

    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        raise StageDAuthorizationError(
            "safe no-follow metadata access is unavailable on this platform"
        )
    normalized = _safe_relative_path(relative_path, "static metadata path")
    absolute = Path(os.path.abspath(os.fspath(repo_root)))
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    directory_flags |= getattr(os, "O_CLOEXEC", 0)
    file_flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    file_flags |= getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(absolute.anchor, directory_flags)
    except OSError as error:
        raise StageDAuthorizationError(
            "cannot safely open repository filesystem anchor"
        ) from error
    try:
        for component in absolute.parts[1:]:
            try:
                next_descriptor = os.open(
                    component, directory_flags, dir_fd=descriptor,
                )
            except OSError as error:
                raise StageDAuthorizationError(
                    "repository path contains a missing, symlink, or unsafe "
                    f"component: {component}"
                ) from error
            os.close(descriptor)
            descriptor = next_descriptor
        parts = PurePosixPath(normalized).parts
        for component in parts[:-1]:
            try:
                next_descriptor = os.open(
                    component, directory_flags, dir_fd=descriptor,
                )
            except OSError as error:
                raise StageDAuthorizationError(
                    "static metadata path contains a missing, symlink, or "
                    f"unsafe directory: {normalized}"
                ) from error
            os.close(descriptor)
            descriptor = next_descriptor
        try:
            path_before = os.stat(
                parts[-1], dir_fd=descriptor, follow_symlinks=False,
            )
            file_descriptor = os.open(
                parts[-1], file_flags, dir_fd=descriptor,
            )
        except OSError as error:
            raise StageDAuthorizationError(
                f"cannot safely open static metadata: {normalized}"
            ) from error
        try:
            before = os.fstat(file_descriptor)
            identity = lambda info: (
                info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns,
                info.st_ctime_ns, info.st_nlink,
            )
            if (
                not stat.S_ISREG(path_before.st_mode)
                or not stat.S_ISREG(before.st_mode)
                or before.st_nlink != 1
                or path_before.st_nlink != 1
                or identity(path_before) != identity(before)
            ):
                raise StageDAuthorizationError(
                    f"static metadata must be a single-link regular file: {normalized}"
                )
            if before.st_size > _MAX_STATIC_METADATA_BYTES:
                raise StageDAuthorizationError(
                    f"static metadata file is unexpectedly large: {normalized}"
                )
            chunks: list[bytes] = []
            remaining = _MAX_STATIC_METADATA_BYTES + 1
            while remaining:
                block = os.read(file_descriptor, min(128 * 1024, remaining))
                if not block:
                    break
                chunks.append(block)
                remaining -= len(block)
            payload = b"".join(chunks)
            after = os.fstat(file_descriptor)
            path_after = os.stat(
                parts[-1], dir_fd=descriptor, follow_symlinks=False,
            )
        finally:
            os.close(file_descriptor)
    except OSError as error:
        raise StageDAuthorizationError(
            f"cannot safely read static metadata: {normalized}"
        ) from error
    finally:
        os.close(descriptor)
    if (
        identity(before) != identity(after)
        or identity(after) != identity(path_after)
        or len(payload) != before.st_size
    ):
        raise StageDAuthorizationError(
            f"static metadata changed during read: {normalized}"
        )
    observed = hashlib.sha256(payload).hexdigest()
    if observed != expected_sha256:
        raise StageDAuthorizationError(
            f"static metadata SHA-256 drifted: {normalized}"
        )
    return payload, observed


def _run_git(
    repo_root: Path | str, arguments: Sequence[str],
) -> str:
    environment = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "GIT_CONFIG_COUNT": "2",
        "GIT_CONFIG_KEY_0": "core.fsmonitor",
        "GIT_CONFIG_VALUE_0": "false",
        "GIT_CONFIG_KEY_1": "core.hooksPath",
        "GIT_CONFIG_VALUE_1": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_LITERAL_PATHSPECS": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_PAGER": "cat",
        "LANG": "C",
        "LC_ALL": "C",
    }
    command = ["git", *arguments]
    try:
        with tempfile.TemporaryFile() as output:
            subprocess.run(
                command, cwd=Path(repo_root), check=True, stdin=subprocess.DEVNULL,
                stdout=output, stderr=output, env=environment,
                timeout=_GIT_TIMEOUT_SECONDS,
            )
            size = output.tell()
            if size > _MAX_GIT_OUTPUT_BYTES:
                raise StageDAuthorizationError(
                    "Git metadata output exceeds the safe size limit"
                )
            output.seek(0)
            payload = output.read()
        return payload.decode("utf-8", errors="strict").strip()
    except StageDAuthorizationError:
        raise
    except (OSError, subprocess.SubprocessError, UnicodeDecodeError) as error:
        raise StageDAuthorizationError(
            f"cannot capture Git metadata: {' '.join(arguments)}"
        ) from error


def _capture_git_state(repo_root: Path | str) -> dict[str, Any]:
    root = Path(os.path.abspath(os.fspath(repo_root)))
    top_level = _run_git(root, ("rev-parse", "--show-toplevel"))
    prefix = _run_git(root, ("rev-parse", "--show-prefix"))
    try:
        observed_root = Path(top_level).resolve(strict=True)
        expected_root = root.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise StageDAuthorizationError(
            "cannot resolve the Stage D repository root"
        ) from error
    if prefix and (
        not prefix.endswith("/")
        or "\\" in prefix
        or PurePosixPath(prefix).is_absolute()
        or any(part in {"", ".", ".."} for part in PurePosixPath(prefix).parts)
    ):
        raise StageDAuthorizationError("Git repository prefix is unsafe")
    observed_project_root = (observed_root / prefix).resolve(strict=True)
    if observed_project_root != expected_root:
        raise StageDAuthorizationError(
            "repo_root does not match the Git repository top-level/prefix identity"
        )
    code = {
        "commit": _run_git(root, ("rev-parse", "HEAD")),
        "tree": _run_git(root, ("rev-parse", "HEAD^{tree}")),
        "dirty": bool(_run_git(
            root, ("status", "--porcelain=v1", "--untracked-files=all"),
        )),
    }
    return _validate_code(code)


@dataclass(frozen=True, slots=True, init=False)
class StageDPreparationInputs(Mapping[str, Any]):
    """Privately issued live code and static model/lock metadata proof."""

    schema_version: int
    document_type: str
    status: str
    code: Mapping[str, Any]
    dependencies: Mapping[str, Any]
    model: Mapping[str, Any]
    canonical_digest: str
    _issuer: object = field(default=None, repr=False, compare=False)

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise TypeError(
            "StageDPreparationInputs can only be issued by "
            "capture_stage_d_preparation_inputs"
        )

    @classmethod
    def _issue(
        cls, *, issuer: object, code: Mapping[str, Any],
        dependencies: Mapping[str, Any], model: Mapping[str, Any],
    ) -> "StageDPreparationInputs":
        if issuer is not _PREPARATION_INPUTS_ISSUER:
            raise StageDAuthorizationError(
                "Stage D preparation-input issuer is invalid"
            )
        payload = {
            "schema_version": 1,
            "document_type": "stage_d_preparation_inputs",
            "status": "PASS",
            "code": _plain(code, "captured code metadata"),
            "dependencies": _plain(
                dependencies, "captured dependency metadata"
            ),
            "model": _plain(model, "captured model metadata"),
        }
        instance = object.__new__(cls)
        for name, value in (
            ("schema_version", 1),
            ("document_type", "stage_d_preparation_inputs"),
            ("status", "PASS"),
            ("code", _freeze(payload["code"])),
            ("dependencies", _freeze(payload["dependencies"])),
            ("model", _freeze(payload["model"])),
            ("canonical_digest", canonical_sha256(payload)),
            ("_issuer", _PREPARATION_INPUTS_ISSUER),
        ):
            object.__setattr__(instance, name, value)
        instance.__post_init__()
        return instance

    def __post_init__(self) -> None:
        if self._issuer is not _PREPARATION_INPUTS_ISSUER:
            raise StageDAuthorizationError(
                "Stage D preparation inputs were not captured by this module"
            )
        document = self.to_dict()
        _exact_keys(document, _PREPARATION_INPUT_KEYS, "Stage D preparation inputs")
        if (
            type(self.schema_version) is not int
            or self.schema_version != 1
            or self.document_type != "stage_d_preparation_inputs"
            or self.status != "PASS"
        ):
            raise StageDAuthorizationError(
                "Stage D preparation-input identity/status drifted"
            )
        _validate_code(self.code)
        dependencies = _exact_keys(
            self.dependencies, _DEPENDENCY_KEYS,
            "captured dependency metadata",
        )
        model = _exact_keys(self.model, _MODEL_KEYS, "captured model metadata")
        if not _same_json(dependencies, {
            "requirements_lock_path": REQUIREMENTS_LOCK_PATH,
            "requirements_lock_sha256": REQUIREMENTS_LOCK_SHA256,
            "requirements_lock_bytes": REQUIREMENTS_LOCK_BYTES,
        }):
            raise StageDAuthorizationError(
                "captured dependency metadata drifted"
            )
        if (
            type(dependencies["requirements_lock_bytes"]) is not int
            or dependencies["requirements_lock_bytes"] < 1
        ):
            raise StageDAuthorizationError(
                "captured dependency byte count is invalid"
            )
        if not _same_json(model, {
            "model_name": MODEL_NAME,
            "model_config_path": MODEL_CONFIG_PATH,
            "model_config_sha256": MODEL_CONFIG_SHA256,
            "model_config_bytes": MODEL_CONFIG_BYTES,
            "state_dict_sha256": MODEL_STATE_DICT_SHA256,
            "state_dict_file_accessed": False,
        }):
            raise StageDAuthorizationError("captured model metadata drifted")
        if (
            type(model["model_config_bytes"]) is not int
            or model["model_config_bytes"] < 1
        ):
            raise StageDAuthorizationError(
                "captured model-config byte count is invalid"
            )
        if self.canonical_digest != _self_digest(document):
            raise StageDAuthorizationError(
                "Stage D preparation-input canonical digest mismatch"
            )

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "document_type": self.document_type,
            "status": self.status,
            "code": _thaw(self.code),
            "dependencies": _thaw(self.dependencies),
            "model": _thaw(self.model),
        }

    def to_dict(self) -> dict[str, Any]:
        result = self._payload()
        result["canonical_digest"] = self.canonical_digest
        return result

    def __getitem__(self, key: str) -> Any:
        if key not in _PREPARATION_INPUT_KEYS:
            raise KeyError(key)
        return getattr(self, key)

    def __iter__(self) -> Iterator[str]:
        return iter(_PREPARATION_INPUT_KEYS)

    def __len__(self) -> int:
        return len(_PREPARATION_INPUT_KEYS)


def capture_stage_d_preparation_inputs(
    repo_root: Path | str = REPO_ROOT,
) -> StageDPreparationInputs:
    """Capture clean Git and exact small metadata files, never the model.

    Tests that need a controlled Git observation may monkeypatch the private
    :func:`_run_git` helper.  Production callers cannot inject a claimed clean
    state through this public API.
    """

    code_before = _capture_git_state(repo_root)
    model_bytes, model_sha256 = _safe_read_static_metadata(
        repo_root, MODEL_CONFIG_PATH, expected_sha256=MODEL_CONFIG_SHA256,
    )
    lock_bytes, lock_sha256 = _safe_read_static_metadata(
        repo_root, REQUIREMENTS_LOCK_PATH, expected_sha256=REQUIREMENTS_LOCK_SHA256,
    )
    code_after = _capture_git_state(repo_root)
    if code_after != code_before:
        raise StageDAuthorizationError(
            "Git state changed during preparation-input capture"
        )
    try:
        model_document = json.loads(model_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise StageDAuthorizationError(
            "config/model.json is not valid UTF-8 JSON"
        ) from error
    if (
        len(model_bytes) != MODEL_CONFIG_BYTES
        or len(lock_bytes) != REQUIREMENTS_LOCK_BYTES
    ):
        raise StageDAuthorizationError(
            "captured static metadata byte count drifted"
        )
    if (
        not isinstance(model_document, Mapping)
        or model_document.get("model_name") != MODEL_NAME
        or model_document.get("state_dict_sha256") != MODEL_STATE_DICT_SHA256
    ):
        raise StageDAuthorizationError(
            "config/model.json semantic identity drifted"
        )
    return StageDPreparationInputs._issue(
        issuer=_PREPARATION_INPUTS_ISSUER, code=code_after,
        dependencies={
            "requirements_lock_path": REQUIREMENTS_LOCK_PATH,
            "requirements_lock_sha256": lock_sha256,
            "requirements_lock_bytes": len(lock_bytes),
        },
        model={
            "model_name": MODEL_NAME,
            "model_config_path": MODEL_CONFIG_PATH,
            "model_config_sha256": model_sha256,
            "model_config_bytes": len(model_bytes),
            "state_dict_sha256": MODEL_STATE_DICT_SHA256,
            "state_dict_file_accessed": False,
        },
    )


def _optional_file_identity(
    document: Any, *, expected_file: str, expected_schema: str, context: str,
) -> None:
    """Reject metadata drift when a validated wrapper exposes byte hashes."""

    file_sha = getattr(document, "file_sha256", None)
    schema_sha = getattr(document, "schema_file_sha256", None)
    if file_sha is not None and file_sha != expected_file:
        raise StageDAuthorizationError(f"{context} file source binding drifted")
    if schema_sha is not None and schema_sha != expected_schema:
        raise StageDAuthorizationError(f"{context} schema source binding drifted")


def _validate_contracts(contracts: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(contracts, StageDContracts):
        raise StageDAuthorizationError(
            "contracts must be issued by load_stage_d_contracts"
        )
    try:
        contracts.config.__post_init__()
        contracts.run_spec.__post_init__()
        contracts.__post_init__()
    except (TypeError, ValueError) as error:
        raise StageDAuthorizationError(
            "contracts were not validly issued by load_stage_d_contracts"
        ) from error
    original_config = contracts.config
    original_run_spec = contracts.run_spec
    bundle = _mapping(contracts, "Stage D contracts")
    _exact_keys(bundle, ("config", "run_spec"), "Stage D contracts")
    config = _mapping(bundle["config"], "Stage D config")
    run_spec = _mapping(bundle["run_spec"], "Stage D run spec")
    _assert_self_digest(
        config, STAGE_D_CONFIG_CANONICAL_DIGEST, "Stage D config"
    )
    _assert_self_digest(
        run_spec, STAGE_D_RUN_SPEC_CANONICAL_DIGEST, "Stage D run spec"
    )
    if original_config is not None:
        _optional_file_identity(
            original_config, expected_file=STAGE_D_CONFIG_FILE_SHA256,
            expected_schema=STAGE_D_CONFIG_SCHEMA_FILE_SHA256,
            context="Stage D config",
        )
    if original_run_spec is not None:
        _optional_file_identity(
            original_run_spec, expected_file=STAGE_D_RUN_SPEC_FILE_SHA256,
            expected_schema=STAGE_D_RUN_SPEC_SCHEMA_FILE_SHA256,
            context="Stage D run spec",
        )

    authority = config.get("authority")
    if not isinstance(authority, Mapping) or (
        authority.get("preparation_authorization") != PREPARATION_AUTHORIZATION
        or authority.get("execution_authorization") != EXECUTION_AUTHORIZATION
        or authority.get("analysis_authorization") != ANALYSIS_AUTHORIZATION
        or authority.get("stage_d_execution_started") is not False
        or authority.get("does_not_override_c10") is not True
    ):
        raise StageDAuthorizationError(
            "D2 preparation/authorization boundary drifted"
        )
    protocol = config.get("protocol")
    if not isinstance(protocol, Mapping) or protocol.get("result_status") != SCIENTIFIC_RESULT:
        raise StageDAuthorizationError("D2 scientific result boundary drifted")

    config_binding = run_spec.get("config_binding")
    expected_config_binding = {
        "config_path": "config/stage_d.json",
        "config_file_sha256": STAGE_D_CONFIG_FILE_SHA256,
        "config_canonical_digest": STAGE_D_CONFIG_CANONICAL_DIGEST,
    }
    if not _same_json(config_binding, expected_config_binding):
        raise StageDAuthorizationError(
            "D2 run-spec/config source binding drifted"
        )
    return config, run_spec


def _validate_manifest(
    manifest: Any, config: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not isinstance(manifest, StageDManifest):
        raise StageDAuthorizationError(
            "manifest must be issued by verify_stage_d_manifest"
        )
    try:
        manifest.__post_init__()
    except (TypeError, ValueError) as error:
        raise StageDAuthorizationError(
            "manifest was not validly issued by verify_stage_d_manifest"
        ) from error
    document = _mapping(manifest, "verified source manifest")
    _exact_keys(
        document,
        ("schema_version", "document_type", "status", "bindings",
         "registry_document", "canonical_digest"),
        "verified source manifest",
    )
    if (
        type(document["schema_version"]) is not int
        or document["schema_version"] != 1
        or document["document_type"] != "stage_d_verified_source_manifest"
        or document["status"] != "PASS"
    ):
        raise StageDAuthorizationError(
            "verified source manifest identity/status drifted"
        )
    _sha256(document["canonical_digest"], "source manifest canonical_digest")
    if document["canonical_digest"] != _self_digest(document):
        raise StageDAuthorizationError(
            "verified source manifest canonical digest mismatch"
        )

    bindings = _exact_keys(
        document["bindings"], ("d0", "d1", "d2", "c10"),
        "verified source manifest bindings",
    )
    upstream = _exact_keys(
        config.get("upstream"), ("d0", "d1", "c10"),
        "Stage D config upstream",
    )
    d0 = _exact_keys(bindings["d0"], _D0_MANIFEST_KEYS, "manifest D0 binding")
    d1 = _exact_keys(bindings["d1"], _D1_MANIFEST_KEYS, "manifest D1 binding")
    d2 = _exact_keys(bindings["d2"], _D2_MANIFEST_KEYS, "manifest D2 binding")
    c10 = _exact_keys(bindings["c10"], _C10_MANIFEST_KEYS, "manifest C10 binding")

    if not _same_json(
        {key: d0[key] for key in _D0_CONFIG_KEYS}, upstream["d0"]
    ):
        raise StageDAuthorizationError("D0 source binding drifted")
    if not _same_json(
        {key: d1[key] for key in _D1_CONFIG_KEYS}, upstream["d1"]
    ):
        raise StageDAuthorizationError("D1 source binding drifted")
    if not _same_json(
        {key: c10[key] for key in _C10_CONFIG_KEYS}, upstream["c10"]
    ):
        raise StageDAuthorizationError("C10 source binding drifted")

    d0_extras = {
        "verification_status": "PASS",
        "schema_path": "config/stage_d_scope_decision.schema.json",
        "document_path": "docs/stage_d_scope_decision.md",
    }
    if any(not _same_json(d0[key], value) for key, value in d0_extras.items()):
        raise StageDAuthorizationError("D0 independent verification drifted")
    expected_d2 = {
        "verification_status": "PASS",
        "completion_ref": D2_COMPLETION_REF,
        "completion_commit": D2_COMPLETION_COMMIT,
        "config_path": "config/stage_d.json",
        "config_file_sha256": STAGE_D_CONFIG_FILE_SHA256,
        "config_canonical_digest": STAGE_D_CONFIG_CANONICAL_DIGEST,
        "config_schema_path": "config/stage_d.schema.json",
        "config_schema_file_sha256": STAGE_D_CONFIG_SCHEMA_FILE_SHA256,
    }
    if not _same_json(d2, expected_d2):
        raise StageDAuthorizationError("D2 independent source binding drifted")

    expected_package_files = {
        "stage_d_registry_package.schema.json": d1["registry_schema_file_sha256"],
        "source_binding.json": d1["source_binding_file_sha256"],
        "registry.json": d1["registry_file_sha256"],
        "pre_authorization_access_attestation.json": d1["attestation_file_sha256"],
        "registry_metadata.json": d1["registry_metadata_file_sha256"],
        "manifest.json": d1["package_manifest_file_sha256"],
    }
    expected_package_canonicals = {
        "stage_d_registry_package.schema.json": d1["registry_schema_canonical_digest"],
        "source_binding.json": d1["source_binding_canonical_digest"],
        "registry.json": d1["registry_canonical_digest"],
        "pre_authorization_access_attestation.json": d1["attestation_canonical_digest"],
        "registry_metadata.json": d1["registry_metadata_canonical_digest"],
        "manifest.json": d1["package_manifest_canonical_digest"],
    }
    d1_extras = {
        "verification_status": "PASS",
        "package_dir": "manifests/stage_d_fidelity_v1",
        "package_file_count": len(PACKAGE_FILES),
        "package_file_sha256": expected_package_files,
        "package_canonical_digests": expected_package_canonicals,
        "registry_row_count": 500,
        "audit_overlay_count": 5,
        "audit_sample_ids": config["audit"]["audit_sample_ids"],
        "execution_authorization": EXECUTION_AUTHORIZATION,
    }
    if any(not _same_json(d1[key], value) for key, value in d1_extras.items()):
        raise StageDAuthorizationError("D1 independent verification drifted")
    if (
        c10["verification_status"] != "PASS"
        or c10["scientific_results_recomputed"] is not False
    ):
        raise StageDAuthorizationError(
            "C10 must be independently verified without recomputing results"
        )
    if any(
        not _same_json(c10[key], value)
        for key, value in _C10_HISTORICAL_STATUS.items()
    ):
        raise StageDAuthorizationError("C10 historical status drifted")

    registry = _exact_keys(
        document["registry_document"], _REGISTRY_KEYS,
        "verified D1 registry document",
    )
    if (
        type(registry["schema_version"]) is not int
        or registry["schema_version"] != 1
        or registry["document_type"] != "stage_d_fidelity_v1_registry"
        or registry["package_id"] != PACKAGE_ID
        or registry["status"] != "FROZEN_METADATA_ONLY"
        or registry["canonical_digest"] != d1["registry_canonical_digest"]
        or _self_digest(registry) != d1["registry_canonical_digest"]
    ):
        raise StageDAuthorizationError("verified D1 registry identity drifted")
    rows = registry["rows"]
    if not isinstance(rows, list) or len(rows) != 500:
        raise StageDAuthorizationError("verified D1 registry must contain 500 rows")
    if any(
        not isinstance(row, Mapping) or row.get("selection_order") != index
        for index, row in enumerate(rows)
    ):
        raise StageDAuthorizationError(
            "verified D1 registry rows are missing, duplicate, or reordered"
        )
    ordered_digest = canonical_sha256([row.get("sample_id") for row in rows])
    if (
        ordered_digest != d1["ordered_sample_ids_sha256"]
        or registry["ordered_sample_ids_sha256"] != ordered_digest
    ):
        raise StageDAuthorizationError(
            "verified D1 registry ordered sample binding drifted"
        )
    return document, [dict(row) for row in rows]


def _expected_plan_context(
    config: Mapping[str, Any], run_spec: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> str:
    payload = {
        "schema_version": 1,
        "document_type": "stage_d_plan_context",
        "config_digest": plan["config_digest"],
        "run_spec_digest": plan["run_spec_digest"],
        "registry_digest": plan["registry_digest"],
        "ordered_sample_ids_digest": plan["ordered_sample_ids_digest"],
        "model_and_preprocessing": config["model_and_preprocessing"],
        "primary_scope": config["primary_scope"],
        "target_and_map": config["target_and_map"],
        "fidelity": config["fidelity"],
        "audit": config["audit"],
        "run_execution_coordinates": run_spec["execution_coordinates"],
        "run_fidelity": run_spec["fidelity"],
        "run_audit": run_spec["audit"],
        "forbidden_workload": {
            "config": config["forbidden_workload"],
            "run_spec": run_spec["forbidden_workload"],
        },
        "sharding": run_spec["sharding"],
    }
    return canonical_sha256(payload)


def _validate_plan(
    value: Any, *, config: Mapping[str, Any], run_spec: Mapping[str, Any],
    manifest: Mapping[str, Any], registry_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    try:
        typed = validate_stage_d_plan(value)
    except (StageDPlanError, TypeError, ValueError) as error:
        raise StageDAuthorizationError(
            "Stage D plan failed independent validation"
        ) from error
    plan = _mapping(typed, "validated Stage D plan")
    d1 = manifest["bindings"]["d1"]
    expected_identities = {
        "config_digest": STAGE_D_CONFIG_CANONICAL_DIGEST,
        "run_spec_digest": STAGE_D_RUN_SPEC_CANONICAL_DIGEST,
        "registry_digest": d1["registry_canonical_digest"],
        "ordered_sample_ids_digest": d1["ordered_sample_ids_sha256"],
    }
    if any(plan.get(key) != expected for key, expected in expected_identities.items()):
        raise StageDAuthorizationError(
            "Stage D plan source binding drifted"
        )
    expected_context = _expected_plan_context(config, run_spec, plan)
    if plan.get("context_digest") != expected_context:
        raise StageDAuthorizationError(
            "Stage D plan context source binding drifted"
        )
    groups = plan.get("groups")
    if not isinstance(groups, list) or len(groups) != len(registry_rows):
        raise StageDAuthorizationError(
            "Stage D plan/registry group coverage drifted"
        )
    for index, (group, row) in enumerate(zip(groups, registry_rows)):
        if not isinstance(group, Mapping):
            raise StageDAuthorizationError(
                f"Stage D plan group {index} is not an object"
            )
        expected = {
            "selection_order": row.get("selection_order"),
            "sample_id": row.get("sample_id"),
            "relative_path": row.get("relative_path"),
            "image_sha256": row.get("image_sha256"),
            "registry_row_digest": row.get("canonical_digest"),
            "audit": row.get("audit_overlay"),
            "audit_overlay_id": row.get("audit_overlay_id"),
        }
        if any(
            not _same_json(group.get(key), item)
            for key, item in expected.items()
        ):
            raise StageDAuthorizationError(
                f"Stage D plan group {index} source binding drifted"
            )

    workload = config["workload"]
    expected_counts = {
        "image_groups": workload["image_groups"],
        "execution_units": workload["execution_units"],
        "primary_clean_maps": workload["primary_clean_maps"],
        "audit_repeat_clean_maps": workload["audit_repeat_clean_maps"],
        "map_work_items": workload["map_work_items"],
        "fidelity_curves": workload["fidelity_curves"],
        "fidelity_curve_points": workload["fidelity_curve_points"],
        "logical_work_items": workload["logical_work_items"],
        "statistical_n": config["cohort"]["statistical_n"],
        "statistical_n_increment": config["audit"]["statistical_n_increment"],
    }
    if not _same_json(plan.get("counts"), expected_counts):
        raise StageDAuthorizationError("Stage D plan count binding drifted")
    return plan


def _validate_budget_and_preflight(
    budget: Any, preflight: Any, *, plan: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        typed_budget = validate_stage_d_estimate(budget)
    except (StageDPlanError, TypeError, ValueError) as error:
        raise StageDAuthorizationError(
            "Stage D budget estimate failed independent validation"
        ) from error
    estimate = _mapping(typed_budget, "validated Stage D budget estimate")
    if (
        estimate.get("plan_digest") != plan["canonical_digest"]
        or not _same_json(estimate.get("counts"), plan["counts"])
    ):
        raise StageDAuthorizationError(
            "Stage D budget/plan source binding drifted"
        )
    try:
        typed_preflight = (
            preflight_stage_d(typed_budget)
            if preflight is None
            else validate_stage_d_preflight(preflight)
        )
    except (StageDPlanError, TypeError, ValueError) as error:
        raise StageDAuthorizationError(
            "Stage D preflight failed independent validation"
        ) from error
    preflight_document = _mapping(typed_preflight, "validated Stage D preflight")
    expected_status = {
        "NOT_EVALUATED": "PROFILE_REQUIRED",
        "BLOCKED_NO_LAUNCH": "BLOCKED_NO_LAUNCH",
    }.get(estimate.get("budget_status"))
    expected_violations = list(estimate.get("launch_violations", [])) + list(
        estimate.get("completed_gate_violations", [])
    )
    if (
        expected_status is None
        or estimate.get("profile_status") not in {
            "D7_PROFILE_REQUIRED", "UNVALIDATED_PROJECTED_FOR_D7",
        }
        or preflight_document.get("plan_digest") != plan["canonical_digest"]
        or preflight_document.get("estimate_digest")
        != estimate.get("canonical_digest")
        or preflight_document.get("status") != expected_status
        or preflight_document.get("violations") != expected_violations
        or preflight_document.get("execution_authorization")
        != EXECUTION_AUTHORIZATION
        or preflight_document.get("analysis_authorization")
        != ANALYSIS_AUTHORIZATION
        or preflight_document.get("d7_final") is not False
    ):
        raise StageDAuthorizationError(
            "Stage D preflight/estimate source binding drifted"
        )
    return estimate, preflight_document


def _validate_preparation_inputs(
    value: Any, config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str]:
    if not isinstance(value, StageDPreparationInputs):
        raise StageDAuthorizationError(
            "preparation_inputs must be issued by "
            "capture_stage_d_preparation_inputs"
        )
    if value._issuer is not _PREPARATION_INPUTS_ISSUER:
        raise StageDAuthorizationError(
            "Stage D preparation-input issuer is invalid"
        )
    document = _mapping(value, "Stage D preparation inputs")
    _exact_keys(document, _PREPARATION_INPUT_KEYS, "Stage D preparation inputs")
    if (
        type(document["schema_version"]) is not int
        or document["schema_version"] != 1
        or document["document_type"] != "stage_d_preparation_inputs"
        or document["status"] != "PASS"
        or document["canonical_digest"] != _self_digest(document)
    ):
        raise StageDAuthorizationError(
            "Stage D preparation-input identity/digest drifted"
        )
    code = _validate_code(document["code"])
    dependencies = _mapping(document["dependencies"], "dependency metadata")
    model = _mapping(document["model"], "model metadata")
    _exact_keys(dependencies, _DEPENDENCY_KEYS, "dependency metadata")
    _exact_keys(model, _MODEL_KEYS, "model metadata")
    configured = config["model_and_preprocessing"]
    expected_dependencies = {
        "requirements_lock_path": configured["requirements_lock_path"],
        "requirements_lock_sha256": configured["requirements_lock_sha256"],
        "requirements_lock_bytes": REQUIREMENTS_LOCK_BYTES,
    }
    expected_model = {
        "model_name": configured["model_name"],
        "model_config_path": configured["model_config_path"],
        "model_config_sha256": configured["model_config_sha256"],
        "model_config_bytes": MODEL_CONFIG_BYTES,
        "state_dict_sha256": configured["state_dict_sha256"],
        "state_dict_file_accessed": False,
    }
    if not _same_json(dependencies, expected_dependencies):
        raise StageDAuthorizationError("dependency source binding drifted")
    if not _same_json(model, expected_model):
        raise StageDAuthorizationError("model source binding drifted")
    if (
        type(dependencies["requirements_lock_bytes"]) is not int
        or dependencies["requirements_lock_bytes"] < 1
        or type(model["model_config_bytes"]) is not int
        or model["model_config_bytes"] < 1
    ):
        raise StageDAuthorizationError(
            "captured static metadata byte count is invalid"
        )
    return code, dependencies, model, document["canonical_digest"]


def _source_bindings(
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    bindings = manifest["bindings"]
    d2 = dict(bindings["d2"])
    d2.update({
        "run_spec_path": "config/stage_d_run_spec.json",
        "run_spec_file_sha256": STAGE_D_RUN_SPEC_FILE_SHA256,
        "run_spec_canonical_digest": STAGE_D_RUN_SPEC_CANONICAL_DIGEST,
        "run_spec_schema_path": "config/stage_d_run_spec.schema.json",
        "run_spec_schema_file_sha256": STAGE_D_RUN_SPEC_SCHEMA_FILE_SHA256,
    })
    return {
        "verified_source_manifest_canonical_digest": manifest["canonical_digest"],
        "d0": copy.deepcopy(bindings["d0"]),
        "d1": copy.deepcopy(bindings["d1"]),
        "d2": d2,
        "c10": copy.deepcopy(bindings["c10"]),
    }


def _plan_binding(plan: Mapping[str, Any]) -> dict[str, Any]:
    return {key: copy.deepcopy(plan[key]) for key in _PLAN_BINDING_KEYS}


def _budget_binding(
    estimate: Mapping[str, Any], preflight: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "estimate_document_type": estimate["document_type"],
        "estimate_canonical_digest": estimate["canonical_digest"],
        "plan_digest": estimate["plan_digest"],
        "profile_status": estimate["profile_status"],
        "budget_status": estimate["budget_status"],
        "launch_safety_caps": copy.deepcopy(estimate["launch_safety_caps"]),
        "completed_campaign_gates": copy.deepcopy(
            estimate["completed_campaign_gates"]
        ),
        "launch_violations": list(estimate["launch_violations"]),
        "completed_gate_violations": list(
            estimate["completed_gate_violations"]
        ),
        "preflight_document_type": preflight["document_type"],
        "preflight_canonical_digest": preflight["canonical_digest"],
        "preflight_status": preflight["status"],
        "preflight_execution_authorization": preflight["execution_authorization"],
        "preflight_analysis_authorization": preflight["analysis_authorization"],
        "d7_final": preflight["d7_final"],
    }


def _expected_payload(
    *, contracts: Any, manifest: Any, plan: Any, budget: Any,
    preparation_inputs: Any, preflight: Any = None,
) -> dict[str, Any]:
    config, run_spec = _validate_contracts(contracts)
    (
        code_document, dependency_document, model_document,
        preparation_inputs_digest,
    ) = _validate_preparation_inputs(preparation_inputs, config)
    source_manifest, registry_rows = _validate_manifest(manifest, config)
    plan_document = _validate_plan(
        plan, config=config, run_spec=run_spec, manifest=source_manifest,
        registry_rows=registry_rows,
    )
    estimate, preflight_document = _validate_budget_and_preflight(
        budget, preflight, plan=plan_document,
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "document_type": DOCUMENT_TYPE,
        "status": STATUS,
        "preparation_authorization": PREPARATION_AUTHORIZATION,
        "execution_authorization": EXECUTION_AUTHORIZATION,
        "analysis_authorization": ANALYSIS_AUTHORIZATION,
        "stage_d_execution_started": False,
        "scientific_result": SCIENTIFIC_RESULT,
        "c10_historical_status": copy.deepcopy(_C10_HISTORICAL_STATUS),
        "source_bindings": _source_bindings(source_manifest),
        "plan_binding": _plan_binding(plan_document),
        "budget_binding": _budget_binding(estimate, preflight_document),
        "preparation_inputs_digest": preparation_inputs_digest,
        "code": code_document,
        "dependencies": dependency_document,
        "model": model_document,
    }
    if _contains_prohibited_capability(payload):  # pragma: no cover - invariant
        raise StageDAuthorizationError(
            "D3 preparation metadata contains a prohibited D7 capability"
        )
    return payload


def _validate_intrinsic(document: Mapping[str, Any]) -> None:
    _exact_keys(document, _ROOT_KEYS, "Stage D preparation authorization")
    fixed = {
        "schema_version": SCHEMA_VERSION,
        "document_type": DOCUMENT_TYPE,
        "status": STATUS,
        "preparation_authorization": PREPARATION_AUTHORIZATION,
        "execution_authorization": EXECUTION_AUTHORIZATION,
        "analysis_authorization": ANALYSIS_AUTHORIZATION,
        "stage_d_execution_started": False,
        "scientific_result": SCIENTIFIC_RESULT,
    }
    if any(
        not _same_json(document.get(key), expected)
        for key, expected in fixed.items()
    ):
        raise StageDAuthorizationError(
            "Stage D preparation authority/result boundary drifted"
        )
    if not _same_json(
        document.get("c10_historical_status"), _C10_HISTORICAL_STATUS
    ):
        raise StageDAuthorizationError("C10 historical status drifted")
    _sha256(
        document.get("preparation_inputs_digest"),
        "authorization preparation_inputs_digest",
    )
    if _contains_prohibited_capability(document):
        raise StageDAuthorizationError(
            "D3 artifact contains a prohibited execution capability"
        )
    _validate_code(document.get("code"))

    dependencies = _exact_keys(
        document.get("dependencies"), _DEPENDENCY_KEYS,
        "authorization dependency binding",
    )
    model = _exact_keys(
        document.get("model"), _MODEL_KEYS, "authorization model binding",
    )
    for key in ("requirements_lock_sha256",):
        _sha256(dependencies[key], f"authorization dependencies.{key}")
    for key in ("model_config_sha256", "state_dict_sha256"):
        _sha256(model[key], f"authorization model.{key}")
    if not _same_json(dependencies, {
        "requirements_lock_path": REQUIREMENTS_LOCK_PATH,
        "requirements_lock_sha256": REQUIREMENTS_LOCK_SHA256,
        "requirements_lock_bytes": REQUIREMENTS_LOCK_BYTES,
    }):
        raise StageDAuthorizationError(
            "authorization dependency binding drifted"
        )
    if not _same_json(model, {
        "model_name": MODEL_NAME,
        "model_config_path": MODEL_CONFIG_PATH,
        "model_config_sha256": MODEL_CONFIG_SHA256,
        "model_config_bytes": MODEL_CONFIG_BYTES,
        "state_dict_sha256": MODEL_STATE_DICT_SHA256,
        "state_dict_file_accessed": False,
    }):
        raise StageDAuthorizationError("authorization model binding drifted")
    expected_preparation_inputs_digest = canonical_sha256({
        "schema_version": 1,
        "document_type": "stage_d_preparation_inputs",
        "status": "PASS",
        "code": document["code"],
        "dependencies": dependencies,
        "model": model,
    })
    if document["preparation_inputs_digest"] != expected_preparation_inputs_digest:
        raise StageDAuthorizationError(
            "authorization preparation-input binding drifted"
        )

    source = _exact_keys(
        document.get("source_bindings"),
        ("verified_source_manifest_canonical_digest", "d0", "d1",
         "d2", "c10"),
        "authorization source bindings",
    )
    _sha256(
        source["verified_source_manifest_canonical_digest"],
        "authorization source manifest digest",
    )
    d0 = _exact_keys(source["d0"], _D0_MANIFEST_KEYS, "authorization D0 binding")
    d1 = _exact_keys(source["d1"], _D1_MANIFEST_KEYS, "authorization D1 binding")
    d2 = _exact_keys(source["d2"], _D2_ARTIFACT_KEYS, "authorization D2 binding")
    c10 = _exact_keys(source["c10"], _C10_MANIFEST_KEYS, "authorization C10 binding")
    if (
        d0["completion_ref"] != D0_COMPLETION_REF
        or d0["completion_commit"] != D0_COMPLETION_COMMIT
        or d0["machine_record_path"] != "config/stage_d_scope_decision.json"
        or d0["machine_record_file_sha256"] != D0_SCOPE_FILE_SHA256
        or d0["machine_record_canonical_digest"] != D0_SCOPE_CANONICAL_DIGEST
        or d0["schema_file_sha256"] != D0_SCOPE_SCHEMA_SHA256
        or d0["markdown_file_sha256"] != D0_SCOPE_MARKDOWN_SHA256
        or d0["verification_status"] != "PASS"
        or d0["schema_path"]
        != "config/stage_d_scope_decision.schema.json"
        or d0["document_path"] != "docs/stage_d_scope_decision.md"
    ):
        raise StageDAuthorizationError("authorization D0 binding drifted")
    if (
        d1["completion_ref"] != D1_COMPLETION_REF
        or d1["completion_commit"] != D1_COMPLETION_COMMIT
        or d1["package_id"] != PACKAGE_ID
        or d1["package_manifest_path"]
        != "manifests/stage_d_fidelity_v1/manifest.json"
        or d1["registry_path"]
        != "manifests/stage_d_fidelity_v1/registry.json"
        or d1["attestation_path"]
        != (
            "manifests/stage_d_fidelity_v1/"
            "pre_authorization_access_attestation.json"
        )
        or d1["candidate_recorded_image_hashes_only"] is not True
        or d1["live_image_equality_status"]
        != "NOT_EVALUATED_D1_DEFERRED_TO_D8_0"
        or d1["verification_status"] != "PASS"
        or d1["package_dir"] != "manifests/stage_d_fidelity_v1"
        or d1["package_file_count"] != len(PACKAGE_FILES)
        or d1["registry_row_count"] != 500
        or d1["audit_overlay_count"] != 5
        or d1["execution_authorization"] != EXECUTION_AUTHORIZATION
    ):
        raise StageDAuthorizationError("authorization D1 binding drifted")
    expected_package_files = {
        "stage_d_registry_package.schema.json": d1["registry_schema_file_sha256"],
        "source_binding.json": d1["source_binding_file_sha256"],
        "registry.json": d1["registry_file_sha256"],
        "pre_authorization_access_attestation.json": d1["attestation_file_sha256"],
        "registry_metadata.json": d1["registry_metadata_file_sha256"],
        "manifest.json": d1["package_manifest_file_sha256"],
    }
    expected_package_canonicals = {
        "stage_d_registry_package.schema.json": d1["registry_schema_canonical_digest"],
        "source_binding.json": d1["source_binding_canonical_digest"],
        "registry.json": d1["registry_canonical_digest"],
        "pre_authorization_access_attestation.json": d1["attestation_canonical_digest"],
        "registry_metadata.json": d1["registry_metadata_canonical_digest"],
        "manifest.json": d1["package_manifest_canonical_digest"],
    }
    if (
        not _same_json(d1["package_file_sha256"], expected_package_files)
        or not _same_json(
            d1["package_canonical_digests"], expected_package_canonicals
        )
    ):
        raise StageDAuthorizationError(
            "authorization D1 package inventory binding drifted"
        )
    expected_d2 = {
        "completion_ref": D2_COMPLETION_REF,
        "completion_commit": D2_COMPLETION_COMMIT,
        "config_file_sha256": STAGE_D_CONFIG_FILE_SHA256,
        "config_canonical_digest": STAGE_D_CONFIG_CANONICAL_DIGEST,
        "config_schema_file_sha256": STAGE_D_CONFIG_SCHEMA_FILE_SHA256,
        "run_spec_file_sha256": STAGE_D_RUN_SPEC_FILE_SHA256,
        "run_spec_canonical_digest": STAGE_D_RUN_SPEC_CANONICAL_DIGEST,
        "run_spec_schema_file_sha256": STAGE_D_RUN_SPEC_SCHEMA_FILE_SHA256,
    }
    if d2["verification_status"] != "PASS" or any(
        d2.get(key) != expected for key, expected in expected_d2.items()
    ):
        raise StageDAuthorizationError("authorization D2 binding drifted")
    if (
        d2["config_path"] != "config/stage_d.json"
        or d2["config_schema_path"] != "config/stage_d.schema.json"
        or d2["run_spec_path"] != "config/stage_d_run_spec.json"
        or d2["run_spec_schema_path"]
        != "config/stage_d_run_spec.schema.json"
    ):
        raise StageDAuthorizationError("authorization D2 path binding drifted")
    if (
        c10["verification_status"] != "PASS"
        or c10["scientific_results_recomputed"] is not False
        or any(
            not _same_json(c10.get(key), value)
            for key, value in _C10_HISTORICAL_STATUS.items()
        )
    ):
        raise StageDAuthorizationError("authorization C10 binding drifted")

    plan = _exact_keys(
        document.get("plan_binding"), _PLAN_BINDING_KEYS,
        "authorization plan binding",
    )
    if plan["document_type"] != "stage_d_execution_plan":
        raise StageDAuthorizationError("authorization plan type drifted")
    for key in (
        "canonical_digest", "config_digest", "run_spec_digest",
        "registry_digest", "ordered_sample_ids_digest",
        "context_digest", "coordinate_digest",
    ):
        _sha256(plan[key], f"authorization plan.{key}")
    if (
        plan["config_digest"] != d2["config_canonical_digest"]
        or plan["run_spec_digest"] != d2["run_spec_canonical_digest"]
        or plan["registry_digest"] != d1["registry_canonical_digest"]
        or plan["ordered_sample_ids_digest"]
        != d1["ordered_sample_ids_sha256"]
    ):
        raise StageDAuthorizationError(
            "authorization plan/source cross-binding drifted"
        )
    expected_counts = {
        "image_groups": 500,
        "execution_units": 1000,
        "primary_clean_maps": 1000,
        "audit_repeat_clean_maps": 10,
        "map_work_items": 1010,
        "fidelity_curves": 10000,
        "fidelity_curve_points": 210000,
        "logical_work_items": 211010,
        "statistical_n": 500,
        "statistical_n_increment": 0,
    }
    if not _same_json(plan["counts"], expected_counts):
        raise StageDAuthorizationError(
            "authorization plan counts drifted"
        )

    budget = _exact_keys(
        document.get("budget_binding"), _BUDGET_BINDING_KEYS,
        "authorization budget binding",
    )
    if (
        budget["estimate_document_type"] != "stage_d_preflight_estimate"
        or budget["preflight_document_type"] != "stage_d_metadata_preflight"
        or budget["plan_digest"] != plan["canonical_digest"]
        or budget["preflight_execution_authorization"]
        != EXECUTION_AUTHORIZATION
        or budget["preflight_analysis_authorization"]
        != ANALYSIS_AUTHORIZATION
        or budget["d7_final"] is not False
    ):
        raise StageDAuthorizationError(
            "authorization budget/preflight boundary drifted"
        )
    expected_preflight_status = {
        "NOT_EVALUATED": "PROFILE_REQUIRED",
        "BLOCKED_NO_LAUNCH": "BLOCKED_NO_LAUNCH",
    }.get(budget["budget_status"])
    if (
        expected_preflight_status is None
        or budget["profile_status"] not in {
            "D7_PROFILE_REQUIRED", "UNVALIDATED_PROJECTED_FOR_D7",
        }
        or budget["preflight_status"] != expected_preflight_status
    ):
        raise StageDAuthorizationError(
            "authorization cannot treat D3 budget metadata as D7 authority"
        )
    _sha256(
        budget["estimate_canonical_digest"],
        "authorization estimate digest",
    )
    _sha256(
        budget["preflight_canonical_digest"],
        "authorization preflight digest",
    )

    observed = document.get("canonical_digest")
    _sha256(observed, "authorization canonical_digest")
    if observed != _self_digest(document):
        raise StageDAuthorizationError(
            "Stage D preparation authorization canonical digest mismatch"
        )


@dataclass(frozen=True, slots=True)
class StageDPreparationAuthorization(Mapping[str, Any]):
    """Deeply immutable, canonical D3 preparation attestation."""

    schema_version: int
    document_type: str
    status: str
    preparation_authorization: str
    execution_authorization: str
    analysis_authorization: str
    stage_d_execution_started: bool
    scientific_result: str
    c10_historical_status: Mapping[str, Any]
    source_bindings: Mapping[str, Any]
    plan_binding: Mapping[str, Any]
    budget_binding: Mapping[str, Any]
    preparation_inputs_digest: str
    code: Mapping[str, Any]
    dependencies: Mapping[str, Any]
    model: Mapping[str, Any]
    canonical_digest: str

    def __post_init__(self) -> None:
        for name in (
            "c10_historical_status", "source_bindings", "plan_binding",
            "budget_binding", "code", "dependencies", "model",
        ):
            object.__setattr__(self, name, _freeze(getattr(self, name)))
        _validate_intrinsic(self.to_dict())

    @property
    def digest(self) -> str:
        return self.canonical_digest

    @property
    def is_execution_authorized(self) -> bool:
        return False

    def _payload(self) -> dict[str, Any]:
        return {
            key: _thaw(getattr(self, key))
            for key in _ROOT_KEYS if key != "canonical_digest"
        }

    def to_dict(self) -> dict[str, Any]:
        result = self._payload()
        result["canonical_digest"] = self.canonical_digest
        return result

    @classmethod
    def create(cls, payload: Mapping[str, Any]) -> "StageDPreparationAuthorization":
        expected = set(_ROOT_KEYS) - {"canonical_digest"}
        _exact_keys(payload, tuple(expected), "authorization payload")
        document = _plain(payload, "authorization payload")
        document["canonical_digest"] = canonical_sha256(document)
        return cls(**document)

    @classmethod
    def from_dict(cls, value: Any) -> "StageDPreparationAuthorization":
        document = _mapping(value, "Stage D preparation authorization")
        _exact_keys(document, _ROOT_KEYS, "Stage D preparation authorization")
        return cls(**document)

    def __getitem__(self, key: str) -> Any:
        if key not in _ROOT_KEYS:
            raise KeyError(key)
        return getattr(self, key)

    def __iter__(self) -> Iterator[str]:
        return iter(_ROOT_KEYS)

    def __len__(self) -> int:
        return len(_ROOT_KEYS)


def build_stage_d_preparation_authorization(
    *,
    contracts: Any,
    manifest: Any,
    plan: Any,
    budget: Any,
    preparation_inputs: StageDPreparationInputs,
    preflight: Any = None,
) -> StageDPreparationAuthorization:
    """Build a canonical preparation artifact from independently supplied data.

    ``contracts``, ``manifest``, and ``preparation_inputs`` must be issued by
    their strict loader/verifier/capture APIs.  Omitted preflight is derived
    from ``budget``.
    """

    payload = _expected_payload(
        contracts=contracts, manifest=manifest, plan=plan, budget=budget,
        preparation_inputs=preparation_inputs, preflight=preflight,
    )
    return StageDPreparationAuthorization.create(payload)


def validate_stage_d_preparation_authorization(
    artifact: Any,
    *,
    contracts: Any,
    manifest: Any,
    plan: Any,
    budget: Any,
    preparation_inputs: StageDPreparationInputs,
    preflight: Any = None,
) -> StageDPreparationAuthorization:
    """Validate intrinsic integrity and every independent source binding.

    Recomputing an artifact's self-digest cannot bless a mutation: the complete
    expected payload is reconstructed from the independently supplied D0--D2
    manifest, frozen contracts, typed plan, estimate/preflight, and Git state.
    """

    candidate = (
        artifact
        if isinstance(artifact, StageDPreparationAuthorization)
        else StageDPreparationAuthorization.from_dict(artifact)
    )
    document = candidate.to_dict()
    _validate_intrinsic(document)
    expected = _expected_payload(
        contracts=contracts, manifest=manifest, plan=plan, budget=budget,
        preparation_inputs=preparation_inputs, preflight=preflight,
    )
    if not _same_json(candidate._payload(), expected):
        raise StageDAuthorizationError(
            "Stage D preparation authorization source binding drifted"
        )
    return candidate


validate_stage_d_authorization = validate_stage_d_preparation_authorization


__all__ = [
    "ANALYSIS_AUTHORIZATION", "DOCUMENT_TYPE",
    "EXECUTION_AUTHORIZATION", "MODEL_CONFIG_BYTES",
    "MODEL_CONFIG_PATH",
    "MODEL_CONFIG_SHA256", "MODEL_NAME",
    "MODEL_STATE_DICT_SHA256", "PREPARATION_AUTHORIZATION",
    "REQUIREMENTS_LOCK_BYTES", "REQUIREMENTS_LOCK_PATH",
    "REQUIREMENTS_LOCK_SHA256",
    "SCHEMA_VERSION", "SCIENTIFIC_RESULT", "STATUS",
    "StageDAuthorizationError", "StageDPreparationAuthorization",
    "StageDPreparationInputs", "build_stage_d_preparation_authorization",
    "canonical_json_bytes", "capture_stage_d_preparation_inputs",
    "canonical_sha256", "preparation_authorization_digest",
    "validate_stage_d_authorization",
    "validate_stage_d_preparation_authorization",
]
