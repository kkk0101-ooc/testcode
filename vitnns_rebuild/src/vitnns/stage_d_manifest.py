"""Fail-closed Stage D metadata and source-binding verification.

This module is deliberately limited to JSON/Markdown metadata and Git object
identity.  It never opens a registry-referenced image, imports a model stack,
or recomputes a scientific result.
"""

from __future__ import annotations

import copy
import hashlib
import os
import stat
import subprocess
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any

from .constants import REPO_ROOT
from .stage_d_registry import (
    D0_COMPLETION_COMMIT,
    D0_COMPLETION_REF,
    D0_SCOPE_CANONICAL_DIGEST,
    D0_SCOPE_FILE_SHA256,
    D0_SCOPE_MARKDOWN_SHA256,
    D0_SCOPE_SCHEMA_SHA256,
    PACKAGE_FILES,
    PACKAGE_ID,
    PACKAGE_SCHEMA_NAME,
    StageDRegistryError,
    canonical_file_bytes,
    canonical_sha256,
    semantic_digest,
    strict_json_bytes,
    validate_stage_d_registry_package_bytes,
)
from .stage_d_scope import (
    C10_ARCHIVE_REF,
    C10_COMPLETION_COMMIT,
    C10_FORMAL_ROOT,
    C10_KEY_CANONICAL_DIGESTS,
    C10_KEY_FILE_SHA256,
    C10_PRODUCER_COMMIT,
    C10_PRODUCER_TREE,
    C10_ROOT_FILE_COUNT,
    C10_ROOT_INVENTORY_DIGEST,
    C10_ROOT_TOTAL_BYTES,
    C10_SOURCE_TREE,
    StageDScopeError,
    verify_c10_sealed_evidence,
)


class StageDManifestError(ValueError):
    """Raised when a frozen Stage D metadata/source binding drifts."""


D1_COMPLETION_REF = "refs/archive/stage-d-d1-completion"
D1_COMPLETION_COMMIT = "d594eadae5b5aa715d12346f6dc93507f1e093a9"
D2_COMPLETION_REF = "refs/archive/stage-d-d2-completion"
D2_COMPLETION_COMMIT = "c57438640e882604031775c944d8ff5ce7932713"

D2_CONFIG_PATH = "config/stage_d.json"
D2_CONFIG_SCHEMA_PATH = "config/stage_d.schema.json"
D2_CONFIG_FILE_SHA256 = (
    "88fdc9cb2b4a5bd74ff7d8e68c79463cafac6437e29b5564c2115a13f1f1136d"
)
D2_CONFIG_CANONICAL_DIGEST = (
    "21f9dcf1d615f2531f7d7d9a36319b958e06d0606c325890dd31ad89b7c2caae"
)
D2_CONFIG_SCHEMA_SHA256 = (
    "64fed10aed29a1c97807ebee2d91a29c3dc218993909b4f0c8fab345d8f78c32"
)

D0_MACHINE_PATH = "config/stage_d_scope_decision.json"
D0_SCHEMA_PATH = "config/stage_d_scope_decision.schema.json"
D0_DOCUMENT_PATH = "docs/stage_d_scope_decision.md"
D1_PACKAGE_DIR = "manifests/stage_d_fidelity_v1"
D1_MANIFEST_PATH = f"{D1_PACKAGE_DIR}/manifest.json"
D1_REGISTRY_PATH = f"{D1_PACKAGE_DIR}/registry.json"
_MAX_METADATA_FILE_BYTES = 16 * 1024 * 1024
_MAX_GIT_OUTPUT_BYTES = 1024 * 1024
_GIT_TIMEOUT_SECONDS = 15
_MANIFEST_ISSUER = object()

_MANIFEST_KEYS = (
    "schema_version",
    "document_type",
    "status",
    "bindings",
    "registry_document",
    "canonical_digest",
)
_BINDING_KEYS = ("d0", "d1", "d2", "c10")
_D0_CONFIG_KEYS = (
    "completion_ref", "completion_commit", "machine_record_path",
    "machine_record_file_sha256", "machine_record_canonical_digest",
    "schema_file_sha256", "markdown_file_sha256",
)
_D0_SUMMARY_KEYS = _D0_CONFIG_KEYS + (
    "verification_status", "schema_path", "document_path",
)
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
_D1_SUMMARY_KEYS = _D1_CONFIG_KEYS + (
    "verification_status", "package_dir", "package_file_count",
    "package_file_sha256", "package_canonical_digests",
    "registry_row_count", "audit_overlay_count", "audit_sample_ids",
    "execution_authorization",
)
_D2_SUMMARY_KEYS = (
    "verification_status", "completion_ref", "completion_commit",
    "config_path", "config_file_sha256", "config_canonical_digest",
    "config_schema_path", "config_schema_file_sha256",
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
_C10_SUMMARY_KEYS = _C10_CONFIG_KEYS + (
    "verification_status", "scientific_results_recomputed",
)
_C10_STATUS = {
    "primary_decision": "NO_GO_NOVELTY",
    "stage_c_acceptance": "BLOCKED",
    "workflow_blocker": "HOLM_BLOCKED_UNDERSPECIFIED",
    "spatial_status": "FAIL_SPATIAL",
    "downstream_authorized": False,
    "stage_d_started": False,
    "cost_evidence_role": "HISTORICAL_8000_UNIT_FEASIBILITY_ONLY",
}
_C10_FILE_FIELDS = {
    "status_file_sha256": "status.json",
    "report_file_sha256": "stage_c_report.json",
    "decision_file_sha256": "decision.json",
    "holm_file_sha256": "holm.json",
    "provenance_file_sha256": "provenance.json",
    "manifest_file_sha256": "manifest.json",
}
_C10_CANONICAL_FIELDS = {
    "status_canonical_digest": "status.json",
    "report_canonical_digest": "stage_c_report.json",
    "decision_canonical_digest": "decision.json",
    "holm_canonical_digest": "holm.json",
    "provenance_canonical_digest": "provenance.json",
    "manifest_canonical_digest": "manifest.json",
}
_D1_EXPECTED_FILE_SHA256 = {
    "manifest.json": (
        "f38bf9773ad172100c22f50ec3b4fe5251664e5c9f5c7ee78ae46532aec372fe"
    ),
    "registry.json": (
        "9aace3abf02a2c07d992d88b765c53cfeffa7f83a6ed72ecefb8cecf9eb79b6b"
    ),
    PACKAGE_SCHEMA_NAME: (
        "2cbeb0536f680103ea5fb3d43e71532832c10fa5b432bed58cda2feaadf9f4fd"
    ),
    "source_binding.json": (
        "f1109304a23674bdadc62306bc927e7ac10b397f68754fae13967971b4f94cac"
    ),
    "registry_metadata.json": (
        "0633f0e78501686be0242332db1bc08d4cf069b360a3b2e51d5d85f03ab98c7f"
    ),
    "pre_authorization_access_attestation.json": (
        "610794455a1d75ef0243c5252c1e2bc14f6a4ea36f6c2ff2acf380dfd2c27644"
    ),
}
_D1_EXPECTED_CANONICAL_DIGESTS = {
    "manifest.json": (
        "cf3e21b88488dcb928f9a98557c2ee9896a7c5210ddb8dda64427c1792fa299d"
    ),
    "registry.json": (
        "514ce4dd2e1fedc1744893e5543651ea42c053458b233ff11a45f700bb4bc028"
    ),
    PACKAGE_SCHEMA_NAME: (
        "b5316e2571dac3d2ef82ccc39fbfc9044819d56d0d5bc9c5982972bcb79fba29"
    ),
    "source_binding.json": (
        "2eafef71035b95035a92c5ccfba7d536c3b924df4081def86fb0a1fa06dba2b8"
    ),
    "registry_metadata.json": (
        "e4c35f5935c119c46935cabd69ffbc6f9f168665d6643d4c253fc5ff0e3c6f52"
    ),
    "pre_authorization_access_attestation.json": (
        "9d89a6596de5027f72aeb76aca4aa600f03b6bb4cf45c8259840fd01e85638fe"
    ),
}
_D1_ORDERED_SAMPLE_IDS_SHA256 = (
    "b187ff5cbacc1b595f06d12e309602b09965e01e7ee1461fd8a694354d4b8f50"
)


@dataclass(frozen=True, slots=True)
class _BoundFile:
    relative_path: str
    size_bytes: int
    sha256: str
    payload: bytes


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


def _json_exact_equal(left: Any, right: Any) -> bool:
    """Compare JSON values without Python's ``True == 1`` coercion."""

    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(
            _json_exact_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            _json_exact_equal(a, b) for a, b in zip(left, right)
        )
    return type(left) is type(right) and left == right


def _plain_config(config: Any) -> dict[str, Any]:
    convert = getattr(config, "to_dict", None)
    value = convert() if callable(convert) else config
    if not isinstance(value, Mapping):
        raise StageDManifestError(
            "Stage D config must be a mapping or expose to_dict()"
        )
    plain = _thaw(value)
    if not isinstance(plain, dict):  # pragma: no cover - guarded above
        raise StageDManifestError("Stage D config must be a JSON object")
    # Canonicalization is also the recursive JSON-safety/type check.
    try:
        canonical_sha256(plain)
    except (StageDRegistryError, TypeError, ValueError) as error:
        raise StageDManifestError("Stage D config is not JSON-safe") from error
    return plain


def _require_mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise StageDManifestError(f"{context} must be an object")
    return value


def _require_exact_keys(
    value: Mapping[str, Any], keys: Sequence[str], context: str,
) -> None:
    expected = set(keys)
    actual = set(value)
    if actual != expected:
        raise StageDManifestError(
            f"{context} fields drifted: missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}"
        )


def _safe_relative_path(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise StageDManifestError(f"{context} must be a normalized relative POSIX path")
    candidate = PurePosixPath(value)
    if (
        candidate.is_absolute()
        or candidate.as_posix() != value
        or any(part in {"", ".", ".."} for part in candidate.parts)
    ):
        raise StageDManifestError(f"{context} must be a normalized relative POSIX path")
    return value


def _open_directory(path: Path | str) -> int:
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        raise StageDManifestError(
            "platform lacks required no-follow directory-open support"
        )
    absolute = Path(os.path.abspath(os.fspath(path)))
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        current = os.open(absolute.anchor, flags)
    except OSError as error:
        raise StageDManifestError("cannot open repository filesystem anchor") from error
    try:
        for component in absolute.parts[1:]:
            try:
                next_fd = os.open(component, flags, dir_fd=current)
            except OSError as error:
                raise StageDManifestError(
                    "repository path contains a missing, symlink, or unsafe component: "
                    f"{component}"
                ) from error
            os.close(current)
            current = next_fd
        if not stat.S_ISDIR(os.fstat(current).st_mode):
            raise StageDManifestError("repository root is not a directory")
        return current
    except BaseException:
        os.close(current)
        raise


def _open_relative_directory(root_fd: int, relative_path: str) -> int:
    normalized = _safe_relative_path(relative_path, "metadata directory path")
    current = os.dup(root_fd)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        for component in PurePosixPath(normalized).parts:
            try:
                next_fd = os.open(component, flags, dir_fd=current)
            except OSError as error:
                raise StageDManifestError(
                    "metadata directory has a missing, symlink, or unsafe component: "
                    f"{relative_path}"
                ) from error
            os.close(current)
            current = next_fd
        return current
    except BaseException:
        os.close(current)
        raise


def _file_identity(
    info: os.stat_result,
) -> tuple[int, int, int, int, int, int, int]:
    return (
        info.st_dev, info.st_ino, info.st_mode, info.st_size,
        info.st_mtime_ns, info.st_ctime_ns, info.st_nlink,
    )


def _read_relative(root_fd: int, relative_path: str) -> _BoundFile:
    normalized = _safe_relative_path(relative_path, "metadata file path")
    parts = PurePosixPath(normalized).parts
    directory_fd = os.dup(root_fd)
    try:
        directory_flags = (
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        for component in parts[:-1]:
            try:
                next_fd = os.open(component, directory_flags, dir_fd=directory_fd)
            except OSError as error:
                raise StageDManifestError(
                    f"metadata path contains a missing, symlink, or unsafe directory: {normalized}"
                ) from error
            os.close(directory_fd)
            directory_fd = next_fd

        file_flags = (
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        try:
            path_before = os.stat(
                parts[-1], dir_fd=directory_fd, follow_symlinks=False,
            )
        except OSError as error:
            raise StageDManifestError(
                f"cannot safely inspect metadata file: {normalized}"
            ) from error
        if (
            not stat.S_ISREG(path_before.st_mode)
            or path_before.st_nlink != 1
        ):
            raise StageDManifestError(
                f"metadata source is not a single-link regular file: {normalized}"
            )
        try:
            fd = os.open(parts[-1], file_flags, dir_fd=directory_fd)
        except OSError as error:
            raise StageDManifestError(
                f"cannot safely open metadata file: {normalized}"
            ) from error
        try:
            before = os.fstat(fd)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_nlink != 1
                or _file_identity(path_before) != _file_identity(before)
            ):
                raise StageDManifestError(
                    "metadata source changed before safe open or is not a "
                    f"single-link regular file: {normalized}"
                )
            if before.st_size > _MAX_METADATA_FILE_BYTES:
                raise StageDManifestError(
                    f"metadata source exceeds safe size limit: {normalized}"
                )
            chunks: list[bytes] = []
            total_bytes = 0
            while True:
                chunk = os.read(fd, 1024 * 1024)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > _MAX_METADATA_FILE_BYTES:
                    raise StageDManifestError(
                        f"metadata source exceeds safe size limit: {normalized}"
                    )
                chunks.append(chunk)
            payload = b"".join(chunks)
            after = os.fstat(fd)
        finally:
            os.close(fd)
        try:
            path_after = os.stat(
                parts[-1], dir_fd=directory_fd, follow_symlinks=False,
            )
        except OSError as error:
            raise StageDManifestError(
                f"metadata source changed while being read: {normalized}"
            ) from error
    finally:
        os.close(directory_fd)
    if (
        _file_identity(before) != _file_identity(after)
        or _file_identity(after) != _file_identity(path_after)
        or len(payload) != before.st_size
    ):
        raise StageDManifestError(f"metadata source changed while being read: {normalized}")
    return _BoundFile(
        relative_path=normalized,
        size_bytes=before.st_size,
        sha256=hashlib.sha256(payload).hexdigest(),
        payload=payload,
    )


def _read_repository_files(
    repo_root: Path | str, relative_paths: Sequence[str],
) -> dict[str, _BoundFile]:
    root_fd = _open_directory(repo_root)
    try:
        return {path: _read_relative(root_fd, path) for path in relative_paths}
    finally:
        os.close(root_fd)


def _read_package_snapshot(repo_root: Path | str) -> dict[str, _BoundFile]:
    root_fd = _open_directory(repo_root)
    try:
        package_fd = _open_relative_directory(root_fd, D1_PACKAGE_DIR)
        try:
            expected = set(PACKAGE_FILES)
            before_names = set(os.listdir(package_fd))
            if before_names != expected:
                raise StageDManifestError(
                    "D1 package inventory drifted while taking snapshot: "
                    f"missing={sorted(expected - before_names)}, "
                    f"extra={sorted(before_names - expected)}"
                )
            snapshot = {name: _read_relative(package_fd, name) for name in PACKAGE_FILES}
            after_names = set(os.listdir(package_fd))
            if after_names != before_names:
                raise StageDManifestError(
                    "D1 package inventory changed while taking snapshot"
                )
            return snapshot
        finally:
            os.close(package_fd)
    finally:
        os.close(root_fd)


def _git_output(repo_root: Path | str, arguments: Sequence[str]) -> str:
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
    try:
        with tempfile.TemporaryFile() as output:
            subprocess.run(
                ["git", *arguments], cwd=repo_root, check=True,
                stdin=subprocess.DEVNULL, stdout=output, stderr=output,
                env=environment, timeout=_GIT_TIMEOUT_SECONDS,
            )
            if output.tell() > _MAX_GIT_OUTPUT_BYTES:
                raise StageDManifestError(
                    "Git metadata output exceeds the safe size limit"
                )
            output.seek(0)
            payload = output.read()
        return payload.decode("utf-8", errors="strict").strip()
    except StageDManifestError:
        raise
    except (OSError, subprocess.SubprocessError, UnicodeDecodeError) as error:
        raise StageDManifestError(
            f"cannot query required Git metadata: {' '.join(arguments)}"
        ) from error


def _verify_repository_identity(repo_root: Path | str) -> None:
    root = Path(os.path.abspath(os.fspath(repo_root))).resolve(strict=True)
    top_level = Path(_git_output(root, ("rev-parse", "--show-toplevel"))).resolve(
        strict=True
    )
    prefix = _git_output(root, ("rev-parse", "--show-prefix"))
    if prefix and (
        not prefix.endswith("/")
        or "\\" in prefix
        or PurePosixPath(prefix).is_absolute()
        or any(part in {"", ".", ".."} for part in PurePosixPath(prefix).parts)
    ):
        raise StageDManifestError("Git repository prefix is unsafe")
    if (top_level / prefix).resolve(strict=True) != root:
        raise StageDManifestError(
            "repo_root does not match the Git repository top-level/prefix identity"
        )


def _git_rev_parse(repo_root: Path | str, revision: str) -> str:
    return _git_output(repo_root, ("rev-parse", "--verify", revision))


def _verify_completion_ref(
    repo_root: Path | str, *, stage: str, reference: str, expected: str,
) -> str:
    observed = _git_rev_parse(repo_root, f"{reference}^{{commit}}")
    if observed != expected:
        raise StageDManifestError(
            f"{stage} completion ref drifted: expected {expected}, observed {observed}"
        )
    return observed


def _parse_mapping(bound: _BoundFile, context: str) -> dict[str, Any]:
    try:
        value = strict_json_bytes(bound.payload, context)
    except StageDRegistryError as error:
        raise StageDManifestError(f"invalid {context}") from error
    if not isinstance(value, dict):
        raise StageDManifestError(f"{context} must contain a JSON object")
    return value


def _verify_hash(bound: _BoundFile, expected: Any, context: str) -> None:
    if not isinstance(expected, str) or bound.sha256 != expected:
        raise StageDManifestError(
            f"{context} file SHA-256 drifted: expected {expected!r}, "
            f"observed {bound.sha256}"
        )


def _verify_d2_config(
    config: Mapping[str, Any], repo_root: Path | str,
) -> dict[str, Any]:
    bound = _read_repository_files(
        repo_root, (D2_CONFIG_PATH, D2_CONFIG_SCHEMA_PATH),
    )
    _verify_hash(bound[D2_CONFIG_PATH], D2_CONFIG_FILE_SHA256, "D2 config")
    _verify_hash(
        bound[D2_CONFIG_SCHEMA_PATH], D2_CONFIG_SCHEMA_SHA256, "D2 config schema",
    )
    source = _parse_mapping(bound[D2_CONFIG_PATH], "D2 config")
    schema = _parse_mapping(bound[D2_CONFIG_SCHEMA_PATH], "D2 config schema")
    if not _json_exact_equal(source, config):
        raise StageDManifestError(
            "loaded Stage D config differs from the safely bound D2 config bytes"
        )
    if source.get("$schema") != Path(D2_CONFIG_SCHEMA_PATH).name:
        raise StageDManifestError("D2 config schema path drifted")
    observed_canonical = semantic_digest(source)
    if (
        source.get("canonical_digest") != observed_canonical
        or observed_canonical != D2_CONFIG_CANONICAL_DIGEST
    ):
        raise StageDManifestError("D2 config canonical digest drifted")
    try:
        from jsonschema import Draft202012Validator

        Draft202012Validator.check_schema(schema)
        errors = sorted(
            Draft202012Validator(schema).iter_errors(source),
            key=lambda error: list(error.path),
        )
    except ImportError as error:
        raise StageDManifestError(
            "jsonschema is required for Stage D source verification"
        ) from error
    except Exception as error:
        raise StageDManifestError(
            "D2 config or schema validation failed"
        ) from error
    if errors:
        raise StageDManifestError(
            f"D2 config schema validation failed: {errors[0].message}"
        )
    completion = _verify_completion_ref(
        repo_root, stage="D2", reference=D2_COMPLETION_REF,
        expected=D2_COMPLETION_COMMIT,
    )
    return {
        "verification_status": "PASS",
        "completion_ref": D2_COMPLETION_REF,
        "completion_commit": completion,
        "config_path": D2_CONFIG_PATH,
        "config_file_sha256": bound[D2_CONFIG_PATH].sha256,
        "config_canonical_digest": observed_canonical,
        "config_schema_path": D2_CONFIG_SCHEMA_PATH,
        "config_schema_file_sha256": bound[D2_CONFIG_SCHEMA_PATH].sha256,
    }


def _verify_d0(
    config: Mapping[str, Any], repo_root: Path | str,
) -> dict[str, Any]:
    upstream = _require_mapping(config.get("upstream"), "config.upstream")
    binding = _require_mapping(upstream.get("d0"), "config.upstream.d0")
    expected_binding = {
        "completion_ref": D0_COMPLETION_REF,
        "completion_commit": D0_COMPLETION_COMMIT,
        "machine_record_path": D0_MACHINE_PATH,
        "machine_record_file_sha256": D0_SCOPE_FILE_SHA256,
        "machine_record_canonical_digest": D0_SCOPE_CANONICAL_DIGEST,
        "schema_file_sha256": D0_SCOPE_SCHEMA_SHA256,
        "markdown_file_sha256": D0_SCOPE_MARKDOWN_SHA256,
    }
    if dict(binding) != expected_binding:
        raise StageDManifestError("config D0 source binding drifted")
    _safe_relative_path(binding["machine_record_path"], "D0 machine record path")
    bound = _read_repository_files(
        repo_root, (D0_MACHINE_PATH, D0_SCHEMA_PATH, D0_DOCUMENT_PATH),
    )
    _verify_hash(
        bound[D0_MACHINE_PATH], binding["machine_record_file_sha256"],
        "D0 machine record",
    )
    _verify_hash(
        bound[D0_SCHEMA_PATH], binding["schema_file_sha256"], "D0 schema",
    )
    _verify_hash(
        bound[D0_DOCUMENT_PATH], binding["markdown_file_sha256"],
        "D0 document",
    )
    document = _parse_mapping(bound[D0_MACHINE_PATH], "D0 scope decision")
    schema = _parse_mapping(bound[D0_SCHEMA_PATH], "D0 scope schema")
    observed_canonical = semantic_digest(document)
    if (
        document.get("canonical_digest") != observed_canonical
        or observed_canonical != binding["machine_record_canonical_digest"]
    ):
        raise StageDManifestError("D0 machine record canonical digest drifted")
    authority = _require_mapping(document.get("authority"), "D0 authority")
    historical = _require_mapping(
        _require_mapping(document.get("source_c10"), "D0 source_c10").get(
            "historical_status"
        ),
        "D0 source_c10.historical_status",
    )
    stage_d_scope = _require_mapping(
        document.get("stage_d_scope"), "D0 stage_d_scope",
    )
    if (
        authority.get("execution_authorization") != "NOT_ISSUED"
        or authority.get("analysis_authorization") != "NOT_EVALUATED"
        or authority.get("does_not_override_c10") is not True
        or stage_d_scope.get("result_status") != "NOT_EVALUATED"
        or any(
            not _json_exact_equal(historical.get(key), expected)
            for key, expected in _C10_STATUS.items()
            if key != "cost_evidence_role"
        )
    ):
        raise StageDManifestError("D0 authority or C10 preservation drifted")
    try:
        from jsonschema import Draft202012Validator

        Draft202012Validator.check_schema(schema)
        errors = sorted(
            Draft202012Validator(schema).iter_errors(document),
            key=lambda error: list(error.path),
        )
    except ImportError as error:
        raise StageDManifestError(
            "jsonschema is required for Stage D source verification"
        ) from error
    except Exception as error:
        raise StageDManifestError(
            "D0 scope or schema validation failed"
        ) from error
    if errors:
        raise StageDManifestError(
            f"D0 scope schema validation failed: {errors[0].message}"
        )
    completion = _verify_completion_ref(
        repo_root, stage="D0", reference=binding["completion_ref"],
        expected=binding["completion_commit"],
    )
    result = dict(binding)
    result.update({
        "verification_status": "PASS",
        "completion_commit": completion,
        "schema_path": D0_SCHEMA_PATH,
        "document_path": D0_DOCUMENT_PATH,
    })
    return result


def _verify_d1(
    config: Mapping[str, Any], repo_root: Path | str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    upstream = _require_mapping(config.get("upstream"), "config.upstream")
    binding = _require_mapping(upstream.get("d1"), "config.upstream.d1")
    _require_exact_keys(binding, _D1_CONFIG_KEYS, "config.upstream.d1")
    if (
        binding["completion_ref"] != D1_COMPLETION_REF
        or binding["completion_commit"] != D1_COMPLETION_COMMIT
        or binding["package_id"] != PACKAGE_ID
        or binding["package_manifest_path"] != D1_MANIFEST_PATH
        or binding["registry_path"] != D1_REGISTRY_PATH
        or binding["attestation_path"]
        != f"{D1_PACKAGE_DIR}/pre_authorization_access_attestation.json"
        or binding["candidate_recorded_image_hashes_only"] is not True
        or binding["live_image_equality_status"]
        != "NOT_EVALUATED_D1_DEFERRED_TO_D8_0"
    ):
        raise StageDManifestError("config D1 identity or image-byte boundary drifted")
    for key in ("package_manifest_path", "registry_path", "attestation_path"):
        _safe_relative_path(binding[key], f"config.upstream.d1.{key}")

    completion = _verify_completion_ref(
        repo_root, stage="D1", reference=binding["completion_ref"],
        expected=binding["completion_commit"],
    )
    # Take the bounded descriptor snapshot first so an oversized replacement
    # cannot reach the existing package validator's unbounded byte reader.
    snapshot = _read_package_snapshot(repo_root)
    observed_files = {name: item.sha256 for name, item in snapshot.items()}
    try:
        validated = validate_stage_d_registry_package_bytes(
            {name: item.payload for name, item in snapshot.items()},
            repo_root=repo_root,
            verify_sources=False, verify_rebuild=False,
        )
    except (StageDRegistryError, OSError, ValueError) as error:
        raise StageDManifestError("D1 registry package validation failed") from error
    if not isinstance(validated, Mapping):
        raise StageDManifestError(
            "D1 registry package verifier returned a non-object summary"
        )
    if (
        validated.get("status") != "PASS"
        or validated.get("package_id") != PACKAGE_ID
        or validated.get("file_count") != len(PACKAGE_FILES)
        or validated.get("registry_rows") != 500
        or validated.get("audit_overlay_count") != 5
        or validated.get("execution_authorization") != "NOT_ISSUED"
        or validated.get("reserve_image_bytes_accessed") is not False
        or validated.get("reserve_result_accessed") is not False
        or validated.get("model_or_gpu_used") is not False
    ):
        raise StageDManifestError("D1 registry package summary semantics drifted")

    observed_canonicals = validated.get("canonical_digests")
    if validated.get("file_sha256") != observed_files:
        raise StageDManifestError(
            "D1 package changed between validation and immutable snapshot"
        )
    if (
        not isinstance(observed_canonicals, Mapping)
        or set(observed_canonicals) != set(PACKAGE_FILES)
    ):
        raise StageDManifestError("D1 package canonical digest inventory drifted")

    expected_files = {
        "manifest.json": binding["package_manifest_file_sha256"],
        "registry.json": binding["registry_file_sha256"],
        PACKAGE_SCHEMA_NAME: binding["registry_schema_file_sha256"],
        "source_binding.json": binding["source_binding_file_sha256"],
        "registry_metadata.json": binding["registry_metadata_file_sha256"],
        "pre_authorization_access_attestation.json": binding["attestation_file_sha256"],
    }
    expected_canonicals = {
        "manifest.json": binding["package_manifest_canonical_digest"],
        "registry.json": binding["registry_canonical_digest"],
        PACKAGE_SCHEMA_NAME: binding["registry_schema_canonical_digest"],
        "source_binding.json": binding["source_binding_canonical_digest"],
        "registry_metadata.json": binding["registry_metadata_canonical_digest"],
        "pre_authorization_access_attestation.json": binding["attestation_canonical_digest"],
    }
    if observed_files != expected_files:
        raise StageDManifestError("D1 package file hashes differ from config pins")
    if dict(observed_canonicals) != expected_canonicals:
        raise StageDManifestError("D1 package canonical digests differ from config pins")
    if (
        expected_files != _D1_EXPECTED_FILE_SHA256
        or expected_canonicals != _D1_EXPECTED_CANONICAL_DIGESTS
        or binding["ordered_sample_ids_sha256"]
        != _D1_ORDERED_SAMPLE_IDS_SHA256
    ):
        raise StageDManifestError("D1 frozen package identity drifted")

    registry = _parse_mapping(snapshot["registry.json"], "D1 registry")
    if snapshot["registry.json"].payload != canonical_file_bytes(registry):
        raise StageDManifestError("D1 registry snapshot bytes are not canonical")
    try:
        registry_canonical = semantic_digest(registry)
    except (StageDRegistryError, TypeError, ValueError) as error:
        raise StageDManifestError("D1 registry snapshot is not JSON-safe") from error
    if (
        registry_canonical != binding["registry_canonical_digest"]
        or registry.get("canonical_digest") != binding["registry_canonical_digest"]
        or registry.get("package_id") != PACKAGE_ID
    ):
        raise StageDManifestError("D1 registry snapshot identity drifted")
    rows = registry.get("rows")
    if not isinstance(rows, list) or len(rows) != 500:
        raise StageDManifestError("D1 registry snapshot must contain 500 rows")
    if any(
        not isinstance(row, Mapping)
        or type(row.get("selection_order")) is not int
        or row.get("selection_order") != index
        for index, row in enumerate(rows)
    ):
        raise StageDManifestError("D1 registry snapshot row order drifted")
    ordered_digest = canonical_sha256([row.get("sample_id") for row in rows])
    if (
        ordered_digest != binding["ordered_sample_ids_sha256"]
        or registry.get("ordered_sample_ids_sha256") != ordered_digest
    ):
        raise StageDManifestError("D1 registry ordered sample identity drifted")

    result = dict(binding)
    result.update({
        "verification_status": "PASS",
        "completion_commit": completion,
        "package_dir": D1_PACKAGE_DIR,
        "package_file_count": len(PACKAGE_FILES),
        "package_file_sha256": observed_files,
        "package_canonical_digests": dict(observed_canonicals),
        "registry_row_count": len(rows),
        "audit_overlay_count": validated["audit_overlay_count"],
        "audit_sample_ids": list(validated["audit_sample_ids"]),
        "execution_authorization": "NOT_ISSUED",
    })
    return result, registry


def _verify_c10_binding(
    config: Mapping[str, Any], repo_root: Path | str, *, verify_full: bool,
) -> dict[str, Any]:
    upstream = _require_mapping(config.get("upstream"), "config.upstream")
    binding = _require_mapping(upstream.get("c10"), "config.upstream.c10")
    _require_exact_keys(binding, _C10_CONFIG_KEYS, "config.upstream.c10")
    expected_identity = {
        "formal_root": str(C10_FORMAL_ROOT),
        "completion_commit": C10_COMPLETION_COMMIT,
        "producer_ref": C10_ARCHIVE_REF,
        "producer_commit": C10_PRODUCER_COMMIT,
        "producer_tree": C10_PRODUCER_TREE,
        "producer_source_tree": C10_SOURCE_TREE,
        "whole_root_file_count": C10_ROOT_FILE_COUNT,
        "whole_root_total_bytes": C10_ROOT_TOTAL_BYTES,
        "whole_root_inventory_digest": C10_ROOT_INVENTORY_DIGEST,
    }
    for key, expected in {**expected_identity, **_C10_STATUS}.items():
        if binding.get(key) != expected:
            raise StageDManifestError(f"config C10 binding/status drifted: {key}")
    for field, filename in _C10_FILE_FIELDS.items():
        if binding.get(field) != C10_KEY_FILE_SHA256[filename]:
            raise StageDManifestError(f"config C10 file binding drifted: {field}")
    for field, filename in _C10_CANONICAL_FIELDS.items():
        if binding.get(field) != C10_KEY_CANONICAL_DIGESTS[filename]:
            raise StageDManifestError(f"config C10 canonical binding drifted: {field}")

    result = dict(binding)
    if not verify_full:
        result.update({
            "verification_status": "CONFIG_BINDINGS_ONLY",
            "scientific_results_recomputed": False,
        })
        return result

    try:
        verified = verify_c10_sealed_evidence(
            binding["formal_root"], repo_root=repo_root,
        )
    except (StageDScopeError, OSError, ValueError) as error:
        raise StageDManifestError("sealed C10 source verification failed") from error
    if not isinstance(verified, Mapping) or verified.get("status") != "PASS":
        raise StageDManifestError("sealed C10 verifier did not return PASS")
    checks = {
        "file_count": binding["whole_root_file_count"],
        "total_bytes": binding["whole_root_total_bytes"],
        "whole_root_inventory_digest": binding["whole_root_inventory_digest"],
        "archive_ref": binding["producer_ref"],
        "producer_commit": binding["producer_commit"],
        "producer_tree": binding["producer_tree"],
        "source_tree": binding["producer_source_tree"],
        "scientific_results_recomputed": False,
    }
    for key, expected in checks.items():
        if verified.get(key) != expected:
            raise StageDManifestError(f"sealed C10 verifier binding drifted: {key}")
    root_value = verified.get("root")
    if not isinstance(root_value, (str, os.PathLike)):
        raise StageDManifestError("sealed C10 verifier root is invalid")
    verified_root = os.path.abspath(os.fspath(root_value))
    expected_root = os.path.abspath(os.fspath(binding["formal_root"]))
    if verified_root != expected_root:
        raise StageDManifestError("sealed C10 verifier root drifted")
    file_hashes = _require_mapping(verified.get("file_sha256"), "C10 verifier file hashes")
    canonical_digests = _require_mapping(
        verified.get("canonical_digests"), "C10 verifier canonical digests",
    )
    for field, filename in _C10_FILE_FIELDS.items():
        if file_hashes.get(filename) != binding[field]:
            raise StageDManifestError(f"sealed C10 file hash drifted: {filename}")
    for field, filename in _C10_CANONICAL_FIELDS.items():
        if canonical_digests.get(filename) != binding[field]:
            raise StageDManifestError(
                f"sealed C10 canonical digest drifted: {filename}"
            )
    result.update({
        "verification_status": "PASS",
        "scientific_results_recomputed": False,
    })
    return result


@dataclass(frozen=True, slots=True, init=False)
class StageDManifest(Mapping[str, Any]):
    """Immutable verified D0/D1/D2/C10 source and registry snapshot.

    ``to_dict`` returns a detached JSON-shaped copy.  The object itself and all
    nested mappings/sequences are recursively immutable.
    """

    schema_version: int
    document_type: str
    status: str
    bindings: Mapping[str, Any]
    registry_document: Mapping[str, Any]
    canonical_digest: str
    _issuer: object = field(default=None, repr=False, compare=False)

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise TypeError(
            "StageDManifest can only be issued by verify_stage_d_manifest"
        )

    @classmethod
    def _issue(
        cls, *, issuer: object, bindings: Mapping[str, Any],
        registry_document: Mapping[str, Any], canonical_digest: str,
    ) -> "StageDManifest":
        if issuer is not _MANIFEST_ISSUER:
            raise StageDManifestError("Stage D source manifest issuer is invalid")
        instance = object.__new__(cls)
        for name, value in (
            ("schema_version", 1),
            ("document_type", "stage_d_verified_source_manifest"),
            ("status", "PASS"),
            ("bindings", bindings),
            ("registry_document", registry_document),
            ("canonical_digest", canonical_digest),
            ("_issuer", issuer),
        ):
            object.__setattr__(instance, name, value)
        instance.__post_init__()
        return instance

    def __post_init__(self) -> None:
        if self._issuer is not _MANIFEST_ISSUER:
            raise StageDManifestError(
                "Stage D source manifests must come from verify_stage_d_manifest"
            )
        frozen_bindings = _freeze(self.bindings)
        frozen_registry = _freeze(self.registry_document)
        object.__setattr__(self, "bindings", frozen_bindings)
        object.__setattr__(self, "registry_document", frozen_registry)
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise StageDManifestError("Stage D source manifest schema version drifted")
        if self.document_type != "stage_d_verified_source_manifest":
            raise StageDManifestError("Stage D source manifest document type drifted")
        if self.status != "PASS":
            raise StageDManifestError("Stage D source manifest status must be PASS")
        if not isinstance(frozen_bindings, Mapping):
            raise StageDManifestError("Stage D source bindings must be an object")
        _require_exact_keys(frozen_bindings, _BINDING_KEYS, "source manifest bindings")
        for name, keys in (
            ("d0", _D0_SUMMARY_KEYS),
            ("d1", _D1_SUMMARY_KEYS),
            ("d2", _D2_SUMMARY_KEYS),
            ("c10", _C10_SUMMARY_KEYS),
        ):
            summary = _require_mapping(
                frozen_bindings[name], f"source manifest bindings.{name}",
            )
            _require_exact_keys(
                summary, keys, f"source manifest bindings.{name}",
            )
        if (
            frozen_bindings["d0"]["verification_status"] != "PASS"
            or frozen_bindings["d0"]["completion_ref"] != D0_COMPLETION_REF
            or frozen_bindings["d0"]["completion_commit"]
            != D0_COMPLETION_COMMIT
            or frozen_bindings["d0"]["machine_record_canonical_digest"]
            != D0_SCOPE_CANONICAL_DIGEST
            or frozen_bindings["d0"]["machine_record_file_sha256"]
            != D0_SCOPE_FILE_SHA256
            or frozen_bindings["d0"]["schema_file_sha256"]
            != D0_SCOPE_SCHEMA_SHA256
            or frozen_bindings["d0"]["markdown_file_sha256"]
            != D0_SCOPE_MARKDOWN_SHA256
            or frozen_bindings["d0"]["machine_record_path"]
            != D0_MACHINE_PATH
            or frozen_bindings["d0"]["schema_path"] != D0_SCHEMA_PATH
            or frozen_bindings["d0"]["document_path"]
            != D0_DOCUMENT_PATH
        ):
            raise StageDManifestError("Stage D source manifest D0 binding drifted")
        if (
            frozen_bindings["d1"]["verification_status"] != "PASS"
            or frozen_bindings["d1"]["completion_ref"] != D1_COMPLETION_REF
            or frozen_bindings["d1"]["completion_commit"]
            != D1_COMPLETION_COMMIT
            or frozen_bindings["d1"]["package_id"] != PACKAGE_ID
            or frozen_bindings["d1"]["registry_row_count"] != 500
            or frozen_bindings["d1"]["audit_overlay_count"] != 5
            or frozen_bindings["d1"]["execution_authorization"]
            != "NOT_ISSUED"
            or frozen_bindings["d1"]["candidate_recorded_image_hashes_only"]
            is not True
            or frozen_bindings["d1"]["live_image_equality_status"]
            != "NOT_EVALUATED_D1_DEFERRED_TO_D8_0"
            or frozen_bindings["d1"]["package_manifest_path"]
            != D1_MANIFEST_PATH
            or frozen_bindings["d1"]["registry_path"]
            != D1_REGISTRY_PATH
            or frozen_bindings["d1"]["attestation_path"]
            != (
                f"{D1_PACKAGE_DIR}/"
                "pre_authorization_access_attestation.json"
            )
            or frozen_bindings["d1"]["package_dir"] != D1_PACKAGE_DIR
            or frozen_bindings["d1"]["package_file_count"]
            != len(PACKAGE_FILES)
        ):
            raise StageDManifestError("Stage D source manifest D1 binding drifted")
        d1_files = _require_mapping(
            frozen_bindings["d1"]["package_file_sha256"],
            "source manifest bindings.d1.package_file_sha256",
        )
        d1_canonicals = _require_mapping(
            frozen_bindings["d1"]["package_canonical_digests"],
            "source manifest bindings.d1.package_canonical_digests",
        )
        if set(d1_files) != set(PACKAGE_FILES) or set(d1_canonicals) != set(
            PACKAGE_FILES
        ):
            raise StageDManifestError(
                "Stage D source manifest D1 package inventory drifted"
            )
        for filename, expected_value in {
            "manifest.json": frozen_bindings["d1"][
                "package_manifest_file_sha256"
            ],
            "registry.json": frozen_bindings["d1"][
                "registry_file_sha256"
            ],
            PACKAGE_SCHEMA_NAME: frozen_bindings["d1"][
                "registry_schema_file_sha256"
            ],
            "source_binding.json": frozen_bindings["d1"][
                "source_binding_file_sha256"
            ],
            "registry_metadata.json": frozen_bindings["d1"][
                "registry_metadata_file_sha256"
            ],
            "pre_authorization_access_attestation.json": frozen_bindings[
                "d1"
            ]["attestation_file_sha256"],
        }.items():
            if d1_files.get(filename) != expected_value:
                raise StageDManifestError(
                    f"Stage D source manifest D1 file binding drifted: {filename}"
                )
        for filename, expected_value in {
            "manifest.json": frozen_bindings["d1"][
                "package_manifest_canonical_digest"
            ],
            "registry.json": frozen_bindings["d1"][
                "registry_canonical_digest"
            ],
            PACKAGE_SCHEMA_NAME: frozen_bindings["d1"][
                "registry_schema_canonical_digest"
            ],
            "source_binding.json": frozen_bindings["d1"][
                "source_binding_canonical_digest"
            ],
            "registry_metadata.json": frozen_bindings["d1"][
                "registry_metadata_canonical_digest"
            ],
            "pre_authorization_access_attestation.json": frozen_bindings[
                "d1"
            ]["attestation_canonical_digest"],
        }.items():
            if d1_canonicals.get(filename) != expected_value:
                raise StageDManifestError(
                    "Stage D source manifest D1 canonical binding drifted: "
                    f"{filename}"
                )
        if (
            dict(d1_files) != _D1_EXPECTED_FILE_SHA256
            or dict(d1_canonicals) != _D1_EXPECTED_CANONICAL_DIGESTS
            or frozen_bindings["d1"]["ordered_sample_ids_sha256"]
            != _D1_ORDERED_SAMPLE_IDS_SHA256
        ):
            raise StageDManifestError(
                "Stage D source manifest D1 frozen identity drifted"
            )
        if (
            frozen_bindings["d2"]["verification_status"] != "PASS"
            or frozen_bindings["d2"]["completion_ref"] != D2_COMPLETION_REF
            or frozen_bindings["d2"]["completion_commit"]
            != D2_COMPLETION_COMMIT
            or frozen_bindings["d2"]["config_file_sha256"]
            != D2_CONFIG_FILE_SHA256
            or frozen_bindings["d2"]["config_canonical_digest"]
            != D2_CONFIG_CANONICAL_DIGEST
            or frozen_bindings["d2"]["config_schema_file_sha256"]
            != D2_CONFIG_SCHEMA_SHA256
            or frozen_bindings["d2"]["config_path"] != D2_CONFIG_PATH
            or frozen_bindings["d2"]["config_schema_path"]
            != D2_CONFIG_SCHEMA_PATH
        ):
            raise StageDManifestError("Stage D source manifest D2 binding drifted")
        c10 = frozen_bindings["c10"]
        if c10["verification_status"] not in {
            "PASS", "CONFIG_BINDINGS_ONLY",
        } or c10["scientific_results_recomputed"] is not False:
            raise StageDManifestError("Stage D source manifest C10 verification drifted")
        for key, expected_value in _C10_STATUS.items():
            if not _json_exact_equal(c10[key], expected_value):
                raise StageDManifestError(
                    f"Stage D source manifest C10 status drifted: {key}"
                )
        for key, expected_value in {
            "formal_root": str(C10_FORMAL_ROOT),
            "completion_commit": C10_COMPLETION_COMMIT,
            "producer_ref": C10_ARCHIVE_REF,
            "producer_commit": C10_PRODUCER_COMMIT,
            "producer_tree": C10_PRODUCER_TREE,
            "producer_source_tree": C10_SOURCE_TREE,
            "whole_root_file_count": C10_ROOT_FILE_COUNT,
            "whole_root_total_bytes": C10_ROOT_TOTAL_BYTES,
            "whole_root_inventory_digest": C10_ROOT_INVENTORY_DIGEST,
        }.items():
            if not _json_exact_equal(c10[key], expected_value):
                raise StageDManifestError(
                    f"Stage D source manifest C10 binding drifted: {key}"
                )
        for field_name, filename in _C10_FILE_FIELDS.items():
            if c10[field_name] != C10_KEY_FILE_SHA256[filename]:
                raise StageDManifestError(
                    "Stage D source manifest C10 file binding drifted: "
                    f"{filename}"
                )
        for field_name, filename in _C10_CANONICAL_FIELDS.items():
            if c10[field_name] != C10_KEY_CANONICAL_DIGESTS[filename]:
                raise StageDManifestError(
                    "Stage D source manifest C10 canonical binding drifted: "
                    f"{filename}"
                )
        if not isinstance(frozen_registry, Mapping):
            raise StageDManifestError("Stage D registry document must be an object")
        rows = frozen_registry.get("rows")
        if not isinstance(rows, tuple) or len(rows) != 500:
            raise StageDManifestError("Stage D source manifest must expose 500 registry rows")
        if any(not isinstance(row, Mapping) for row in rows):
            raise StageDManifestError(
                "Stage D source manifest registry rows must be objects"
            )
        try:
            registry_canonical = semantic_digest(_thaw(frozen_registry))
            ordered_digest = canonical_sha256([row.get("sample_id") for row in rows])
            expected = canonical_sha256(self._payload())
        except (StageDRegistryError, TypeError, ValueError) as error:
            raise StageDManifestError(
                "Stage D source manifest contains invalid JSON state"
            ) from error
        if (
            frozen_registry.get("package_id") != PACKAGE_ID
            or frozen_registry.get("status") != "FROZEN_METADATA_ONLY"
            or frozen_registry.get("canonical_digest")
            != _D1_EXPECTED_CANONICAL_DIGESTS["registry.json"]
            or frozen_bindings["d1"]["registry_canonical_digest"]
            != _D1_EXPECTED_CANONICAL_DIGESTS["registry.json"]
            or registry_canonical
            != _D1_EXPECTED_CANONICAL_DIGESTS["registry.json"]
            or frozen_registry.get("ordered_sample_ids_sha256")
            != _D1_ORDERED_SAMPLE_IDS_SHA256
            or frozen_bindings["d1"]["ordered_sample_ids_sha256"]
            != _D1_ORDERED_SAMPLE_IDS_SHA256
            or ordered_digest != _D1_ORDERED_SAMPLE_IDS_SHA256
            or any(
                type(row.get("selection_order")) is not int
                or row.get("selection_order") != index
                for index, row in enumerate(rows)
            )
        ):
            raise StageDManifestError(
                "Stage D source manifest registry binding drifted"
            )
        audit = frozen_registry.get("audit_selection")
        if (
            not isinstance(audit, Mapping)
            or audit.get("audit_image_count") != 5
            or tuple(audit.get("selected_sample_ids", ()))
            != tuple(frozen_bindings["d1"]["audit_sample_ids"])
        ):
            raise StageDManifestError(
                "Stage D source manifest audit binding drifted"
            )
        if self.canonical_digest != expected:
            raise StageDManifestError(
                "Stage D source manifest canonical digest mismatch"
            )

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "document_type": self.document_type,
            "status": self.status,
            "bindings": _thaw(self.bindings),
            "registry_document": _thaw(self.registry_document),
        }

    @property
    def d0(self) -> Mapping[str, Any]:
        return self.bindings["d0"]

    @property
    def d1(self) -> Mapping[str, Any]:
        return self.bindings["d1"]

    @property
    def d2(self) -> Mapping[str, Any]:
        return self.bindings["d2"]

    @property
    def c10(self) -> Mapping[str, Any]:
        return self.bindings["c10"]

    @property
    def source_bindings(self) -> Mapping[str, Any]:
        return self.bindings

    @property
    def registry_rows(self) -> tuple[Mapping[str, Any], ...]:
        return self.registry_document["rows"]

    @property
    def registry(self) -> Mapping[str, Any]:
        return self.registry_document

    @property
    def registry_snapshot(self) -> Mapping[str, Any]:
        return self.registry_document

    def to_dict(self) -> dict[str, Any]:
        result = self._payload()
        result["canonical_digest"] = self.canonical_digest
        return result

    def __getitem__(self, key: str) -> Any:
        if key not in _MANIFEST_KEYS:
            raise KeyError(key)
        return getattr(self, key)

    def __iter__(self) -> Iterator[str]:
        return iter(_MANIFEST_KEYS)

    def __len__(self) -> int:
        return len(_MANIFEST_KEYS)


def verify_stage_d_manifest(
    config: Any, *, repo_root: Path | str = REPO_ROOT, verify_c10: bool = True,
) -> StageDManifest:
    """Verify and snapshot frozen Stage D metadata without live-data access.

    Args:
        config: A loaded config mapping or an object whose ``to_dict`` returns
            the exact loaded Stage D config document.
        repo_root: Repository root containing the frozen D0--D2 sources.
        verify_c10: When true (the default), run the existing full sealed-C10
            byte/hash/status/ref verifier.  False verifies only its exact frozen
            config bindings and reports ``CONFIG_BINDINGS_ONLY`` in ``c10``.
    """

    if type(verify_c10) is not bool:
        raise StageDManifestError("verify_c10 must be a boolean")
    _verify_repository_identity(repo_root)
    plain_config = _plain_config(config)
    # Validate the D2 source first: this makes every subsequently interpreted
    # upstream value an exact member of the frozen D2 document.
    d2 = _verify_d2_config(plain_config, repo_root)
    authority = _require_mapping(plain_config.get("authority"), "config.authority")
    protocol = _require_mapping(plain_config.get("protocol"), "config.protocol")
    if (
        authority.get("execution_authorization") != "NOT_ISSUED"
        or authority.get("analysis_authorization") != "NOT_EVALUATED"
        or authority.get("stage_d_execution_started") is not False
        or authority.get("does_not_override_c10") is not True
        or protocol.get("result_status") != "NOT_EVALUATED"
    ):
        raise StageDManifestError("D2 authority or result status drifted")
    d0 = _verify_d0(plain_config, repo_root)
    d1, registry = _verify_d1(plain_config, repo_root)
    c10 = _verify_c10_binding(
        plain_config, repo_root, verify_full=verify_c10,
    )
    payload = {
        "schema_version": 1,
        "document_type": "stage_d_verified_source_manifest",
        "status": "PASS",
        "bindings": {"d0": d0, "d1": d1, "d2": d2, "c10": c10},
        "registry_document": registry,
    }
    return StageDManifest._issue(
        issuer=_MANIFEST_ISSUER,
        bindings={"d0": d0, "d1": d1, "d2": d2, "c10": c10},
        registry_document=registry,
        canonical_digest=canonical_sha256(payload),
    )


__all__ = [
    "D1_COMPLETION_COMMIT",
    "D1_COMPLETION_REF",
    "D2_COMPLETION_COMMIT",
    "D2_COMPLETION_REF",
    "D2_CONFIG_CANONICAL_DIGEST",
    "D2_CONFIG_FILE_SHA256",
    "D2_CONFIG_PATH",
    "D2_CONFIG_SCHEMA_PATH",
    "D2_CONFIG_SCHEMA_SHA256",
    "StageDManifest",
    "StageDManifestError",
    "verify_stage_d_manifest",
]
