"""Stage D execution-root sealing and source-recomputing validation.

This module is deliberately a post-execution control plane.  It never issues or
consumes a new execution capability and never imports the model stack.  A root
is sealed from the D4 checkpoint and immutable attempt bytes under the D4
lifecycle/checkpoint/artifact locks.  Validation independently reconstructs the
four published payload documents from those source bytes.

The retained L0 contract has two closed forms: D4's deliberately small
synthetic fixture and the production envelope checked by
``stage_d_scientific``.  An otherwise intact terminal package with an
explicitly recorded scientific failure remains package-complete but has
scientific validity ``FAIL``; malformed successful evidence fails closed as
an integrity error.
"""

from __future__ import annotations

import copy
import errno
import hashlib
import io
import json
import math
import os
import secrets
import stat
import tempfile
from contextlib import ExitStack, contextmanager
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any

import numpy as np

from .stage_d_artifacts import (
    AUDIT_PATH, CURVES_PATH, MAP_PATH, PROVENANCE_PATH, RESULT_PATH,
    SCIENTIFIC_FAILURE_STATUS, SUCCESS_STATUS, attempt_directories,
)
from .stage_d_plan import (
    MATCHED_METHOD_ID, NNS_METHOD_ID, StageDEstimate, StageDExecutionPlan,
    StageDPreflight, canonical_json_bytes, validate_stage_d_estimate,
    validate_stage_d_plan, validate_stage_d_preflight,
)
from .stage_d_scientific import (
    StageDScientificValidation, validate_stage_d_pair_bindings,
    validate_stage_d_scientific_evidence,
)
from . import stage_d_runtime as _runtime


FINAL_DIRECTORY_NAME = "final"
FINAL_PAYLOAD_FILES = (
    "evidence_report.json", "reconciliation.json",
    "source_inventory.json", "payload_manifest.json",
)
FINAL_STATUS_FILENAME = "status.json"
FINAL_FILES = FINAL_PAYLOAD_FILES + (FINAL_STATUS_FILENAME,)
_RUNTIME_ROOT_ENTRIES = frozenset({
    _runtime.RUNTIME_CHECKPOINT_FILENAME,
    _runtime.LIFECYCLE_LOCK_FILENAME,
    _runtime.CHECKPOINT_LOCK_FILENAME,
    _runtime.ARTIFACT_ACTIVITY_LOCK_FILENAME,
    "units",
})
_FINAL_FILE_SET = frozenset(FINAL_FILES)
_SCIENTIFIC_FAILURE_CODES = frozenset({
    "empty_valid_region", "shape_mismatch", "non_finite",
    "degenerate_direction", "degenerate_target_delta",
    "missing_artifact", "audit_repeatability_failed",
    "endpoint_parity_failed", "ig_conservation_failed",
    "matched_token_conservation_failed",
})
_RESULT_KEYS = frozenset({
    "package_complete", "scientific_validity", "workflow_status",
    "plan_digest", "final_status_sha256", "root_inventory",
    "evidence_report", "reconciliation", "source_inventory",
    "payload_manifest", "whole_root_ordered_inventory",
})
_SOURCE_INVENTORY_KEYS = frozenset({
    "schema_version", "document_type", "root_scope", "files",
    "file_count", "total_bytes", "canonical_digest",
})
_RECONCILIATION_KEYS = frozenset({
    "schema_version", "document_type", "plan_digest",
    "expected_image_count", "expected_unit_count", "image_count",
    "unit_count", "expected_counts", "observed_counts",
    "state_counts", "group_rows", "rows", "failure_rows",
    "na_rows", "resources", "retention",
    "checkpoint_failure_table_reconciled",
    "checkpoint_resources_reconciled", "no_rows_dropped",
    "zero_imputation_performed", "canonical_digest",
})
_EVIDENCE_KEYS = frozenset({
    "schema_version", "document_type", "package_complete",
    "scientific_validity", "workflow_status", "plan_digest",
    "estimate_digest", "preflight_digest",
    "execution_context_digest", "counts", "scientific_failures",
    "authorization", "analysis_boundary", "analysis_performed",
    "bootstrap_performed", "decision_performed",
    "reconciliation_digest", "source_inventory_digest",
    "canonical_digest",
})
_PAYLOAD_MANIFEST_KEYS = frozenset({
    "schema_version", "document_type", "immutable",
    "artifact_count", "artifacts", "canonical_digest",
})
_FINAL_STATUS_KEYS = frozenset({
    "schema_version", "document_type", "terminal",
    "package_complete", "scientific_validity", "workflow_status",
    "plan_digest", "payload_manifest_sha256",
    "sealed_root_inventory", "canonical_digest",
})
_DOCUMENT_KEYS = {
    "evidence_report.json": _EVIDENCE_KEYS,
    "reconciliation.json": _RECONCILIATION_KEYS,
    "source_inventory.json": _SOURCE_INVENTORY_KEYS,
    "payload_manifest.json": _PAYLOAD_MANIFEST_KEYS,
    "status.json": _FINAL_STATUS_KEYS,
}
_EXPECTED_COUNTS = {
    "registry_rows": 500, "image_groups": 500,
    "execution_units": 1000, "primary_clean_maps": 1000,
    "audit_repeat_clean_maps": 10, "map_work_items": 1010,
    "fidelity_curves": 10000, "fidelity_curve_points": 210000,
    "logical_work_items": 211010,
}
_FROZEN_PLAN_DIGEST = (
    "5a40051ac17a7bfd6663640b1d2478c69d45eea07d3182653b90d2c7ca760443"
)
_FROZEN_COORDINATE_DIGEST = (
    "cc560e2a2d4e1220ccd812c073119a2b679de213f9770358f7decf51824661fb"
)
_FROZEN_PLAN_COUNTS = {
    "image_groups": 500, "execution_units": 1000,
    "primary_clean_maps": 1000, "audit_repeat_clean_maps": 10,
    "map_work_items": 1010, "fidelity_curves": 10000,
    "fidelity_curve_points": 210000, "logical_work_items": 211010,
    "statistical_n": 500, "statistical_n_increment": 0,
}
_RETRY_AUTHORIZATION_KEYS = frozenset({
    "schema_version", "document_type", "status", "unit_digest",
    "prior_attempt_digest", "failure_code", "next_attempt_number",
    "execution_context_digest", "canonical_digest",
})
_SEALED_INVENTORY_KEYS = frozenset({
    "scope", "ordering", "excluded_paths", "files",
    "file_count", "total_bytes", "canonical_digest",
})
_RUNTIME_CONTEXT_KEYS = frozenset({
    "formal", "authorization_scope", "execution_authorization",
    "authorization_slot_digest", "shard_count", "shard_mapping",
    "plan_digest", "budget_digest", "preflight_digest",
    "code_binding_digest", "dependency_binding_digest",
    "model_binding_digest", "source_bindings_digest",
    "access_attestation", "gate_context_digest",
    "runtime_binding_digest", "provenance_bindings",
})


class StageDFinalizationError(ValueError):
    """Base class for D5 finalization and validation failures."""


class StageDFinalizationBlockedError(StageDFinalizationError):
    """Raised when an incomplete, active, or unauthorized root cannot seal."""


class StageDFinalizationIntegrityError(StageDFinalizationError):
    """Raised when persisted bytes or their bindings are inconsistent."""


# Compatibility spellings retained for the first D5 draft.
StageDFinalizeError = StageDFinalizationError
StageDFinalizeBlockedError = StageDFinalizationBlockedError
StageDFinalizeIntegrityError = StageDFinalizationIntegrityError


def _plain(value: Any, context: str = "value") -> Any:
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, child in value.items():
            if not isinstance(key, str):
                raise StageDFinalizeIntegrityError(f"{context} has a non-string key")
            result[key] = _plain(child, f"{context}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [_plain(child, context) for child in value]
    if value is None or isinstance(value, (str, bool, int)):
        return copy.deepcopy(value)
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise StageDFinalizeIntegrityError(f"{context} is not finite JSON data")


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(child) for key, child in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(child) for child in value)
    return value


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return canonical_json_bytes(_plain(value)) + b"\n"


def _self_document(document_type: str, **values: Any) -> dict[str, Any]:
    result = {"schema_version": 1, "document_type": document_type, **values}
    result["canonical_digest"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    return result


def _strict_json(raw: bytes, context: str) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise StageDFinalizeIntegrityError(f"{context} has duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8"), object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON token {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise StageDFinalizeIntegrityError(f"invalid {context}: {error}") from error
    if not isinstance(value, dict):
        raise StageDFinalizeIntegrityError(f"{context} must be a JSON object")
    _plain(value, context)
    if raw != _json_bytes(value):
        raise StageDFinalizeIntegrityError(f"{context} is not canonical JSON")
    return value


def _exact(value: Any, keys: set[str] | frozenset[str], context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(keys):
        raise StageDFinalizeIntegrityError(f"{context} fields are missing or unknown")
    return _plain(value, context)


def _safe_root(value: Path | str) -> Path:
    raw = os.fspath(value)
    if not isinstance(raw, str) or not raw or "\x00" in raw or "\\" in raw:
        raise StageDFinalizationIntegrityError(
            "Stage D root must be a non-empty absolute POSIX path"
        )
    pure = PurePosixPath(raw)
    if (
        not pure.is_absolute()
        or pure.anchor != os.path.sep
        or any(part in {"", ".", ".."} for part in pure.parts[1:])
        or os.path.normpath(raw) != raw
        or raw == os.path.sep
    ):
        raise StageDFinalizationIntegrityError(
            "Stage D root must be normalized, absolute, and non-root"
        )
    root = Path(raw)
    current = Path(root.anchor)
    for part in root.parts[1:]:
        current /= part
        try:
            metadata = os.lstat(current)
        except OSError as error:
            raise StageDFinalizationIntegrityError(
                f"root component is missing: {current}"
            ) from error
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise StageDFinalizationIntegrityError(
                f"root contains a symlink or non-directory: {current}"
            )
    flags = (
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor: int | None = None
    try:
        descriptor = os.open(root, flags)
        opened = os.fstat(descriptor)
        named = os.lstat(root)
        if (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino):
            raise StageDFinalizationIntegrityError(
                "Stage D root changed while being opened"
            )
    except OSError as error:
        raise StageDFinalizationIntegrityError(
            "Stage D root is missing or unsafe"
        ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
    return root


def _root_identity(root: Path) -> tuple[int, int]:
    metadata = os.lstat(root)
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise StageDFinalizationIntegrityError(
            "Stage D root identity is no longer a safe directory"
        )
    return metadata.st_dev, metadata.st_ino


def _open_root_descriptor(root: Path) -> tuple[int, tuple[int, int]]:
    """Retain the exact execution-root inode for one complete operation."""

    flags = (
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor: int | None = None
    try:
        descriptor = os.open(os.path.sep, flags)
        for component in root.parts[1:]:
            child = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        opened = os.fstat(descriptor)
        named = os.lstat(root)
        if (
            not stat.S_ISDIR(opened.st_mode)
            or (opened.st_dev, opened.st_ino)
            != (named.st_dev, named.st_ino)
        ):
            raise ValueError("root pathname and descriptor differ")
        return descriptor, (opened.st_dev, opened.st_ino)
    except (OSError, ValueError) as error:
        try:
            if descriptor is not None:
                os.close(descriptor)
        except OSError:
            pass
        raise StageDFinalizationIntegrityError(
            "Stage D root is missing, unsafe, or unstable"
        ) from error


def _descriptor_root(root_descriptor: int) -> Path:
    """Return a pathname whose lookup starts at a retained root fd."""

    # Child lookup follows the procfs descriptor link to the retained inode;
    # the root itself is traversed by duplicating ``root_descriptor``.
    return Path(f"/proc/self/fd/{root_descriptor}")


@contextmanager
def _retained_root(
    named_root: Path,
) -> Iterator[tuple[Path, int, tuple[int, int]]]:
    descriptor, identity = _open_root_descriptor(named_root)
    try:
        yield _descriptor_root(descriptor), descriptor, identity
    finally:
        os.close(descriptor)


def _verify_root_binding(
    named_root: Path, root_descriptor: int, expected: tuple[int, int],
) -> None:
    opened = os.fstat(root_descriptor)
    if (
        not stat.S_ISDIR(opened.st_mode)
        or (opened.st_dev, opened.st_ino) != expected
    ):
        raise StageDFinalizationIntegrityError(
            "retained Stage D root descriptor changed identity"
        )
    _recheck_root_identity(named_root, expected)


@contextmanager
def _root_locks(root: Path) -> Iterator[None]:
    """Acquire all D4 locks through the retained-root namespace."""

    with ExitStack() as stack:
        stack.enter_context(_runtime._lifecycle_lock(root, exclusive=True))
        stack.enter_context(_runtime._checkpoint_lock(root))
        stack.enter_context(_runtime._artifact_activity_lock(root, exclusive=True))
        yield


def _recheck_root_identity(root: Path, expected: tuple[int, int]) -> None:
    checked = _safe_root(root)
    if checked != root or _root_identity(checked) != expected:
        raise StageDFinalizationIntegrityError(
            "Stage D root identity changed during finalization"
        )


def _read_regular(path: Path, context: str) -> bytes:
    flags = (
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    )
    descriptor: int | None = None
    try:
        before_path = os.lstat(path)
        if (
            stat.S_ISLNK(before_path.st_mode)
            or not stat.S_ISREG(before_path.st_mode)
            or before_path.st_nlink != 1
        ):
            raise ValueError("not a single-link regular file")
        descriptor = os.open(path, flags)
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or (before_path.st_dev, before_path.st_ino) != (before.st_dev, before.st_ino)
        ):
            raise ValueError("not a single-link stable regular file")
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        after = os.fstat(descriptor)
        after_path = os.lstat(path)
        identity = lambda item: (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns, item.st_ctime_ns)
        raw = b"".join(chunks)
        if identity(before) != identity(after) or identity(before) != identity(after_path) or len(raw) != before.st_size:
            raise ValueError("file changed while being read")
        return raw
    except (OSError, ValueError) as error:
        raise StageDFinalizeIntegrityError(f"{context} is missing, unsafe, or unstable") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _inventory(
    root: Path, *, exclude_final: bool,
    exclude_top_level: frozenset[str] = frozenset(),
    _captured_bytes: dict[str, bytes] | None = None,
    _captured_directories: list[str] | None = None,
    _root_descriptor: int | None = None,
) -> list[dict[str, Any]]:
    """Hash one root through directory-relative, no-follow descriptors.

    The returned ordered digest inventory is the D5 source snapshot.  Later
    semantic reads must match these hashes, and callers take a second snapshot
    before returning/publishing.  This avoids mixing generations without
    retaining potentially very large formal payloads in memory.
    """

    records: list[dict[str, Any]] = []
    directory_flags = (
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    file_flags = (
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    )

    def identity(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
        return (
            value.st_dev, value.st_ino, value.st_mode, value.st_size,
            value.st_mtime_ns, value.st_ctime_ns,
        )

    def visit(descriptor: int, prefix: PurePosixPath) -> None:
        before_directory = os.fstat(descriptor)
        if not stat.S_ISDIR(before_directory.st_mode):
            raise StageDFinalizationIntegrityError(
                f"unsafe directory in root: {prefix}"
            )
        try:
            entries = sorted(os.scandir(descriptor), key=lambda item: item.name)
        except OSError as error:
            raise StageDFinalizationIntegrityError(
                f"cannot enumerate root directory: {prefix}"
            ) from error
        if len({entry.name for entry in entries}) != len(entries):
            raise StageDFinalizationIntegrityError(
                f"duplicate directory entry in root: {prefix}"
            )
        for entry in entries:
            name = entry.name
            if not isinstance(name, str) or name in {"", ".", ".."} or "/" in name:
                raise StageDFinalizationIntegrityError(
                    f"invalid directory entry in root: {prefix}"
                )
            if not prefix.parts and name.startswith(".d5-final-staging-"):
                if name not in exclude_top_level:
                    raise StageDFinalizationIntegrityError(
                        "unexpected D5 staging directory is present in the root"
                    )
                continue
            if not prefix.parts and (
                (exclude_final and name == FINAL_DIRECTORY_NAME)
                or name in exclude_top_level
            ):
                continue
            relative = prefix / name
            try:
                named = os.stat(
                    name, dir_fd=descriptor, follow_symlinks=False,
                )
            except OSError as error:
                raise StageDFinalizationIntegrityError(
                    f"cannot inspect root entry: {relative}"
                ) from error
            if stat.S_ISLNK(named.st_mode):
                raise StageDFinalizationIntegrityError(
                    f"unsafe symlink in root: {relative}"
                )
            if stat.S_ISDIR(named.st_mode):
                allowed_directory = (
                    (not prefix.parts and name in {"units", FINAL_DIRECTORY_NAME})
                    or (len(prefix.parts) == 1 and prefix.parts[0] == "units")
                    or (len(prefix.parts) == 2 and prefix.parts[0] == "units" and name == "attempts")
                    or (len(prefix.parts) == 3 and prefix.parts[0] == "units" and name.startswith("attempt-"))
                    or (len(prefix.parts) == 4 and prefix.parts[0] == "units" and name in {"L0", "L1", "L2"})
                )
                if not allowed_directory:
                    raise StageDFinalizationIntegrityError(
                        f"unexpected directory in closed Stage D root: {relative}"
                    )
                child_descriptor: int | None = None
                try:
                    child_descriptor = os.open(
                        name, directory_flags, dir_fd=descriptor,
                    )
                    opened = os.fstat(child_descriptor)
                    if identity(opened) != identity(named):
                        raise StageDFinalizationIntegrityError(
                            f"root directory changed while opening: {relative}"
                        )
                    if _captured_directories is not None:
                        _captured_directories.append(relative.as_posix())
                    visit(child_descriptor, relative)
                    if identity(os.fstat(child_descriptor)) != identity(opened):
                        raise StageDFinalizationIntegrityError(
                            f"root directory changed while hashing: {relative}"
                        )
                except OSError as error:
                    raise StageDFinalizationIntegrityError(
                        f"cannot safely open root directory: {relative}"
                    ) from error
                finally:
                    if child_descriptor is not None:
                        os.close(child_descriptor)
                continue
            if not stat.S_ISREG(named.st_mode) or named.st_nlink != 1:
                raise StageDFinalizationIntegrityError(
                    f"root entry is not a single-link regular file: {relative}"
                )
            file_descriptor: int | None = None
            try:
                file_descriptor = os.open(name, file_flags, dir_fd=descriptor)
                opened = os.fstat(file_descriptor)
                if (
                    not stat.S_ISREG(opened.st_mode)
                    or opened.st_nlink != 1
                    or identity(opened) != identity(named)
                ):
                    raise StageDFinalizationIntegrityError(
                        f"root file changed while opening: {relative}"
                    )
                digest = hashlib.sha256()
                observed = 0
                chunks: list[bytes] | None = (
                    [] if _captured_bytes is not None else None
                )
                while True:
                    block = os.read(file_descriptor, 1024 * 1024)
                    if not block:
                        break
                    observed += len(block)
                    digest.update(block)
                    if chunks is not None:
                        chunks.append(block)
                after = os.fstat(file_descriptor)
                after_named = os.stat(
                    name, dir_fd=descriptor, follow_symlinks=False,
                )
                if (
                    identity(opened) != identity(after)
                    or identity(opened) != identity(after_named)
                    or observed != opened.st_size
                ):
                    raise StageDFinalizationIntegrityError(
                        f"root file changed while hashing: {relative}"
                    )
                records.append({
                    "path": relative.as_posix(), "bytes": observed,
                    "sha256": digest.hexdigest(),
                })
                if _captured_bytes is not None:
                    assert chunks is not None
                    _captured_bytes[relative.as_posix()] = b"".join(chunks)
            except OSError as error:
                raise StageDFinalizationIntegrityError(
                    f"cannot safely hash root file: {relative}"
                ) from error
            finally:
                if file_descriptor is not None:
                    os.close(file_descriptor)
        after_directory = os.fstat(descriptor)
        if identity(before_directory) != identity(after_directory):
            raise StageDFinalizationIntegrityError(
                f"root directory changed while hashing: {prefix}"
            )

    root_descriptor: int | None = None
    try:
        root_descriptor = (
            os.dup(_root_descriptor)
            if _root_descriptor is not None else os.open(root, directory_flags)
        )
        visit(root_descriptor, PurePosixPath())
    except StageDFinalizationError:
        raise
    except OSError as error:
        raise StageDFinalizationIntegrityError(
            "cannot safely capture Stage D root inventory"
        ) from error
    finally:
        if root_descriptor is not None:
            os.close(root_descriptor)
    records.sort(key=lambda item: item["path"])
    if len({item["path"] for item in records}) != len(records):
        raise StageDFinalizationIntegrityError(
            "root inventory contains duplicate paths"
        )
    return records


def _source_snapshot(
    root: Path, *, sealed: bool, root_descriptor: int | None = None,
) -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], Mapping[str, bytes],
    tuple[str, ...],
]:
    """Capture one exact byte generation for all D5 recomputation.

    Formal roots are refused before this helper is entered.  The temporary
    The capture contains only bytes below the supplied execution root (never
    an image/model path).  For validation it includes the existing final
    controls; source derivation always excludes that subtree.
    """

    captured: dict[str, bytes] = {}
    directories: list[str] = []
    whole = _inventory(
        root, exclude_final=not sealed,
        _captured_bytes=captured,
        _captured_directories=directories,
        _root_descriptor=root_descriptor,
    )
    if set(captured) != {row["path"] for row in whole}:
        raise StageDFinalizationIntegrityError(
            "source snapshot bytes and ordered inventory disagree"
        )
    source = [
        row for row in whole
        if not row["path"].startswith(FINAL_DIRECTORY_NAME + "/")
    ]
    if not sealed and source != whole:
        raise StageDFinalizationIntegrityError(
            "unsealed source snapshot unexpectedly contains final controls"
        )
    return (
        source, whole, MappingProxyType(captured),
        tuple(sorted(directories)),
    )


def _materialize_source_snapshot(
    source_records: Sequence[Mapping[str, Any]],
    captured_bytes: Mapping[str, bytes],
    captured_directories: Sequence[str],
) -> tuple[tempfile.TemporaryDirectory[str], Path]:
    """Create a private immutable-by-convention mirror of captured source bytes."""

    temporary = tempfile.TemporaryDirectory(
        prefix="stage-d-d5-source-snapshot-", dir=os.environ.get(
            "VITNNS_D5_SNAPSHOT_TMPDIR"
        ),
    )
    mirror = Path(temporary.name) / "execution-root"
    mirror.mkdir(mode=0o700)
    try:
        for relative in sorted(
            captured_directories,
            key=lambda value: (len(PurePosixPath(value).parts), value),
        ):
            if relative == FINAL_DIRECTORY_NAME or relative.startswith(
                FINAL_DIRECTORY_NAME + "/"
            ):
                continue
            target = mirror / PurePosixPath(relative)
            target.mkdir(mode=0o700, parents=True, exist_ok=True)
        for row in source_records:
            relative = row["path"]
            payload = captured_bytes[relative]
            target = mirror / PurePosixPath(relative)
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            descriptor = os.open(
                target, os.O_WRONLY | os.O_CREAT | os.O_EXCL
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
            try:
                view = memoryview(payload)
                while view:
                    written = os.write(descriptor, view)
                    if written <= 0:
                        raise StageDFinalizationIntegrityError(
                            "source snapshot mirror write made no progress"
                        )
                    view = view[written:]
            finally:
                os.close(descriptor)
        if _inventory(mirror, exclude_final=True) != list(source_records):
            raise StageDFinalizationIntegrityError(
                "materialized source snapshot differs from captured bytes"
            )
        return temporary, mirror
    except Exception:
        temporary.cleanup()
        raise


def _validate_source_directories(
    source_records: Sequence[Mapping[str, Any]],
    captured_directories: Sequence[str],
    plan: StageDExecutionPlan,
    checkpoint: Mapping[str, Any],
) -> None:
    """Reject empty/extra directories omitted by a file-only inventory."""

    expected = {"units"}
    for unit in plan.units:
        base = PurePosixPath("units") / unit.canonical_digest
        expected.update({base.as_posix(), (base / "attempts").as_posix()})
    raw_units = checkpoint.get("units")
    if not isinstance(raw_units, list):
        raise StageDFinalizationIntegrityError(
            "checkpoint units are invalid while reconciling directories"
        )
    by_digest: dict[str, Mapping[str, Any]] = {}
    for row in raw_units:
        if not isinstance(row, Mapping):
            raise StageDFinalizationIntegrityError(
                "checkpoint unit row is invalid while reconciling directories"
            )
        digest = row.get("unit_digest")
        if not isinstance(digest, str) or digest in by_digest:
            raise StageDFinalizationIntegrityError(
                "checkpoint unit directory join is invalid or duplicated"
            )
        by_digest[digest] = row
    for unit in plan.units:
        row = by_digest.get(unit.canonical_digest)
        if row is None or not isinstance(row.get("attempts"), list):
            raise StageDFinalizationIntegrityError(
                "checkpoint unit directory join is incomplete"
            )
        for index, attempt in enumerate(row["attempts"], start=1):
            if not isinstance(attempt, Mapping) or attempt.get("attempt_id") != (
                f"attempt-{index:06d}"
            ):
                raise StageDFinalizationIntegrityError(
                    "checkpoint attempt directory sequence is invalid"
                )
            expected.add((
                PurePosixPath("units") / unit.canonical_digest / "attempts"
                / f"attempt-{index:06d}"
            ).as_posix())
    for row in source_records:
        path = PurePosixPath(row["path"])
        parent = path.parent
        while parent != PurePosixPath("."):
            expected.add(parent.as_posix())
            parent = parent.parent
    observed = {
        path for path in captured_directories
        if path != FINAL_DIRECTORY_NAME
        and not path.startswith(FINAL_DIRECTORY_NAME + "/")
    }
    if len(observed) != len([
        path for path in captured_directories
        if path != FINAL_DIRECTORY_NAME
        and not path.startswith(FINAL_DIRECTORY_NAME + "/")
    ]) or observed != expected:
        raise StageDFinalizationIntegrityError(
            "Stage D source directory layout is not closed; "
            f"missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )


def _source_inventory_now(root: Path) -> list[dict[str, Any]]:
    return _inventory(root, exclude_final=True)


def _whole_inventory_now(
    root: Path, *, root_descriptor: int | None = None,
) -> list[dict[str, Any]]:
    return _inventory(
        root, exclude_final=False, _root_descriptor=root_descriptor,
    )


def _inventory_and_directories(
    root: Path, *, exclude_final: bool,
    exclude_top_level: frozenset[str] = frozenset(),
    root_descriptor: int | None = None,
) -> tuple[list[dict[str, Any]], tuple[str, ...]]:
    directories: list[str] = []
    records = _inventory(
        root, exclude_final=exclude_final,
        exclude_top_level=exclude_top_level,
        _captured_directories=directories,
        _root_descriptor=root_descriptor,
    )
    return records, tuple(sorted(directories))


def _inventory_index(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for raw in records:
        row = _plain(raw, "source inventory row")
        if (
            not isinstance(row, dict)
            or set(row) != {"path", "bytes", "sha256"}
            or not isinstance(row["path"], str)
            or type(row["bytes"]) is not int
            or row["bytes"] < 0
            or not isinstance(row["sha256"], str)
            or len(row["sha256"]) != 64
            or row["path"] in result
        ):
            raise StageDFinalizationIntegrityError(
                "source inventory contains an invalid or duplicate record"
            )
        result[row["path"]] = row
    return result


def _read_snapshot_regular(
    root: Path, path: Path, source_index: Mapping[str, Mapping[str, Any]],
    context: str, *, captured_bytes: Mapping[str, bytes] | None = None,
) -> bytes:
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError as error:
        raise StageDFinalizationIntegrityError(
            f"{context} escaped the Stage D root"
        ) from error
    expected = source_index.get(relative)
    if expected is None:
        raise StageDFinalizationIntegrityError(
            f"{context} is absent from the captured source snapshot"
        )
    if captured_bytes is None:
        raw = _read_regular(path, context)
    else:
        try:
            raw = captured_bytes[relative]
        except KeyError as error:
            raise StageDFinalizationIntegrityError(
                f"{context} bytes are absent from the captured snapshot"
            ) from error
    if (
        len(raw) != expected["bytes"]
        or hashlib.sha256(raw).hexdigest() != expected["sha256"]
    ):
        raise StageDFinalizationIntegrityError(
            f"{context} differs from the captured source snapshot"
        )
    return raw


def _attempt_snapshot_digest(
    root: Path, attempt: Path,
    source_index: Mapping[str, Mapping[str, Any]],
) -> str:
    prefix = attempt.relative_to(root).as_posix() + "/"
    records = [
        {
            "path": path[len(prefix):],
            "bytes": row["bytes"],
            "sha256": row["sha256"],
        }
        for path, row in source_index.items() if path.startswith(prefix)
    ]
    records.sort(key=lambda item: item["path"])
    if not records:
        raise StageDFinalizationIntegrityError(
            "attempt is absent from the captured source snapshot"
        )
    return hashlib.sha256(
        canonical_json_bytes({"files": records})
    ).hexdigest()


def _inventory_with_overlay(
    source_records: Sequence[Mapping[str, Any]],
    final_payloads: Mapping[str, bytes],
) -> list[dict[str, Any]]:
    """Return the sealed-root inventory before the self-binding status file."""

    records = _plain(list(source_records), "source inventory overlay")
    records.extend(
        {
            "path": f"{FINAL_DIRECTORY_NAME}/{name}",
            "bytes": len(final_payloads[name]),
            "sha256": hashlib.sha256(final_payloads[name]).hexdigest(),
        }
        for name in FINAL_PAYLOAD_FILES
    )
    records.sort(key=lambda item: item["path"])
    return records


def _inventory_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    plain = _plain(list(records), "inventory")
    return {
        "file_count": len(plain),
        "total_bytes": sum(item["bytes"] for item in plain),
        "canonical_digest": hashlib.sha256(canonical_json_bytes({"files": plain})).hexdigest(),
    }


def _sealed_inventory(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    files = _plain(list(records), "sealed-root inventory")
    unsigned = {
        "scope": "whole_root_excluding_status_self",
        "ordering": "path_lexicographic",
        "excluded_paths": [f"{FINAL_DIRECTORY_NAME}/{FINAL_STATUS_FILENAME}"],
        "files": files,
        "file_count": len(files),
        "total_bytes": sum(item["bytes"] for item in files),
    }
    return {
        **unsigned,
        "canonical_digest": hashlib.sha256(
            canonical_json_bytes(unsigned)
        ).hexdigest(),
    }


def _validate_sealed_inventory(
    root: Path, status: Mapping[str, Any],
) -> list[dict[str, Any]]:
    sealed = _exact(
        status.get("sealed_root_inventory"), _SEALED_INVENTORY_KEYS,
        "status sealed-root inventory",
    )
    whole = _inventory(root, exclude_final=False)
    status_path = f"{FINAL_DIRECTORY_NAME}/{FINAL_STATUS_FILENAME}"
    without_status = [item for item in whole if item["path"] != status_path]
    if len(without_status) + 1 != len(whole):
        raise StageDFinalizationIntegrityError(
            "whole-root inventory lacks exactly one terminal status"
        )
    expected = _sealed_inventory(without_status)
    if sealed != expected:
        raise StageDFinalizationIntegrityError(
            "terminal status sealed-root inventory differs from disk"
        )
    return whole


def _fsync_directory(path: Path | None = None, *, descriptor: int | None = None) -> None:
    if (path is None) == (descriptor is None):
        raise ValueError("supply exactly one directory path or descriptor")
    owned = descriptor is None
    if descriptor is None:
        descriptor = os.open(
            path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
    try:
        os.fsync(descriptor)
    finally:
        if owned:
            os.close(descriptor)


def _create_staging_directory(
    root_descriptor: int,
) -> tuple[str, int, tuple[int, int]]:
    """Create and retain an unpredictable staging directory below root."""

    flags = (
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    for _unused in range(128):
        name = f".d5-final-staging-{secrets.token_hex(16)}"
        created_identity: tuple[int, int] | None = None
        try:
            os.mkdir(name, mode=0o700, dir_fd=root_descriptor)
            created = os.stat(
                name, dir_fd=root_descriptor, follow_symlinks=False,
            )
            if not stat.S_ISDIR(created.st_mode):
                raise StageDFinalizationIntegrityError(
                    "new D5 staging object is not a directory"
                )
            created_identity = (created.st_dev, created.st_ino)
        except FileExistsError:
            continue
        descriptor: int | None = None
        try:
            descriptor = os.open(name, flags, dir_fd=root_descriptor)
            opened = os.fstat(descriptor)
            named = os.stat(
                name, dir_fd=root_descriptor, follow_symlinks=False,
            )
            identity = (opened.st_dev, opened.st_ino)
            if (
                not stat.S_ISDIR(opened.st_mode)
                or not stat.S_ISDIR(named.st_mode)
                or identity != (named.st_dev, named.st_ino)
            ):
                raise StageDFinalizationIntegrityError(
                    "D5 staging directory changed while being retained"
                )
            return name, descriptor, identity
        except Exception:
            if descriptor is not None:
                os.close(descriptor)
            if created_identity is not None:
                # The directory is still empty here.  Remove only the exact
                # inode we just created; never clean up a substituted name.
                try:
                    current = os.stat(
                        name, dir_fd=root_descriptor, follow_symlinks=False,
                    )
                    if (
                        stat.S_ISDIR(current.st_mode)
                        and (current.st_dev, current.st_ino)
                        == created_identity
                    ):
                        os.rmdir(name, dir_fd=root_descriptor)
                        _fsync_directory(descriptor=root_descriptor)
                except OSError:
                    pass
            raise
    raise StageDFinalizationIntegrityError(
        "unable to allocate a unique D5 staging directory"
    )


def _atomic_create_at(directory_descriptor: int, name: str, raw: bytes) -> None:
    """Create one immutable control through a retained directory fd."""

    if name not in _FINAL_FILE_SET:
        raise StageDFinalizationIntegrityError(
            "unknown D5 final control filename"
        )
    descriptor: int | None = None
    try:
        descriptor = os.open(
            name, os.O_WRONLY | os.O_CREAT | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
            0o600, dir_fd=directory_descriptor,
        )
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise StageDFinalizationIntegrityError(
                f"D5 final control is unsafe: {name}"
            )
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise StageDFinalizationIntegrityError(
                    f"D5 final control write made no progress: {name}"
                )
            view = view[written:]
        os.fsync(descriptor)
        after = os.fstat(descriptor)
        named = os.stat(
            name, dir_fd=directory_descriptor, follow_symlinks=False,
        )
        identity = lambda item: (
            item.st_dev, item.st_ino, item.st_mode, item.st_nlink,
            item.st_size, item.st_mtime_ns, item.st_ctime_ns,
        )
        if (
            identity(before)[:4] != identity(after)[:4]
            or identity(after) != identity(named)
            or after.st_size != len(raw)
        ):
            raise StageDFinalizationIntegrityError(
                f"D5 final control changed while being written: {name}"
            )
    except FileExistsError as error:
        raise StageDFinalizationBlockedError(
            f"refusing to overwrite final control: {name}"
        ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _publish_directory_no_replace(
    root_descriptor: int, staging_name: str, final_name: str,
    staging_descriptor: int,
) -> None:
    """Atomically publish a descriptor-bound sibling without replacement.

    Linux renameat2(RENAME_NOREPLACE) is used when available.  The hard-link
    fallback is intentionally not used for directories; on platforms without
    no-replace rename support D5 fails closed instead of exposing a partial or
    overwriting package.
    """

    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise StageDFinalizationIntegrityError(
            "atomic no-replace directory publication is unavailable"
        )
    renameat2.argtypes = (
        ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    retained = os.fstat(staging_descriptor)
    expected_staging_identity = (retained.st_dev, retained.st_ino)
    if not stat.S_ISDIR(retained.st_mode):
        raise StageDFinalizationIntegrityError(
            "retained D5 staging object is not a directory"
        )
    try:
        before = os.stat(
            staging_name, dir_fd=root_descriptor, follow_symlinks=False,
        )
    except OSError as error:
        raise StageDFinalizationIntegrityError(
            "D5 staging directory disappeared before publication"
        ) from error
    if (
        not stat.S_ISDIR(before.st_mode)
        or (before.st_dev, before.st_ino) != expected_staging_identity
    ):
        raise StageDFinalizationIntegrityError(
            "D5 staging directory identity changed before publication"
        )
    result = renameat2(
        root_descriptor, os.fsencode(staging_name),
        root_descriptor, os.fsencode(final_name), 1,
    )
    if result == 0:
        retained_after = os.fstat(staging_descriptor)
        try:
            published = os.stat(
                final_name, dir_fd=root_descriptor, follow_symlinks=False,
            )
        except OSError as error:
            raise StageDFinalizationIntegrityError(
                "published final directory is missing"
            ) from error
        if (
            not stat.S_ISDIR(published.st_mode)
            or (retained_after.st_dev, retained_after.st_ino)
            != expected_staging_identity
            or (published.st_dev, published.st_ino)
            != expected_staging_identity
        ):
            raise StageDFinalizationIntegrityError(
                "published final directory is not the staged inode"
            )
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise StageDFinalizationBlockedError(
            "final directory has already been claimed"
        )
    raise StageDFinalizationIntegrityError(
        f"cannot atomically publish final directory: errno {error_number}"
    )


def _cleanup_staging_directory(
    root: Path, staging_name: str, expected_identity: tuple[int, int],
    *, root_descriptor: int | None = None,
) -> None:
    """Best-effort cleanup limited to the private directory and five files."""

    staging = root / staging_name
    staging_descriptor: int | None = None
    try:
        metadata = (
            os.stat(
                staging_name, dir_fd=root_descriptor, follow_symlinks=False,
            )
            if root_descriptor is not None else os.lstat(staging)
        )
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISDIR(metadata.st_mode)
            or (metadata.st_dev, metadata.st_ino) != expected_identity
        ):
            return
        directory_flags = (
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        if root_descriptor is not None:
            staging_descriptor = os.open(
                staging_name, directory_flags, dir_fd=root_descriptor,
            )
        else:
            staging_descriptor = os.open(staging, directory_flags)
        opened = os.fstat(staging_descriptor)
        if (opened.st_dev, opened.st_ino) != expected_identity:
            return
        for name in FINAL_FILES:
            try:
                child_metadata = os.stat(
                    name, dir_fd=staging_descriptor, follow_symlinks=False,
                )
            except FileNotFoundError:
                continue
            if (
                stat.S_ISREG(child_metadata.st_mode)
                and child_metadata.st_nlink == 1
            ):
                os.unlink(name, dir_fd=staging_descriptor)
        os.rmdir(
            staging_name, dir_fd=root_descriptor,
        ) if root_descriptor is not None else staging.rmdir()
        if root_descriptor is not None:
            _fsync_directory(descriptor=root_descriptor)
        else:
            _fsync_directory(staging.parent)
    except OSError:
        # Never broaden cleanup to an unexpected namespace.  A private staging
        # residue is preferable to deleting a path that was externally altered.
        return
    finally:
        if staging_descriptor is not None:
            os.close(staging_descriptor)


def _validate_controls(plan: Any, estimate: Any, preflight: Any) -> tuple[StageDExecutionPlan, StageDEstimate, StageDPreflight]:
    try:
        typed_plan = validate_stage_d_plan(plan)
        typed_estimate = validate_stage_d_estimate(estimate)
        typed_preflight = validate_stage_d_preflight(preflight)
        _runtime._validate_control_inputs(typed_plan, typed_estimate, typed_preflight)
    except Exception as error:
        raise StageDFinalizeIntegrityError(f"invalid Stage D control inputs: {error}") from error
    if (
        typed_plan.canonical_digest != _FROZEN_PLAN_DIGEST
        or typed_plan.coordinate_digest != _FROZEN_COORDINATE_DIGEST
        or dict(typed_plan.counts) != _FROZEN_PLAN_COUNTS
        or len(typed_plan.groups) != 500
        or len(typed_plan.units) != 1000
    ):
        raise StageDFinalizeIntegrityError(
            "D5 requires the exact frozen plan, coordinate stream, and counts"
        )
    return typed_plan, typed_estimate, typed_preflight


def _checkpoint_control(
    root: Path, plan: StageDExecutionPlan, estimate: StageDEstimate,
    preflight: StageDPreflight, *, raw_bytes: bytes | None = None,
) -> dict[str, Any]:
    try:
        raw = (
            _runtime._strict_checkpoint_json(
                root / _runtime.RUNTIME_CHECKPOINT_FILENAME
            )
            if raw_bytes is None else _strict_json(
                raw_bytes, "runtime checkpoint"
            )
        )
        # This narrow preflight intentionally precedes runtime checkpoint
        # validation because that validator walks every attempt.  D5 must
        # reject formal roots from checkpoint control bytes alone and must not
        # inspect their outcome tree.
        if set(raw) != {
            "schema_version", "document_type", "generation",
            "previous_checkpoint_digest", "plan_digest", "estimate_digest",
            "preflight_digest", "execution_context", "lifecycle", "shards",
            "groups", "units", "failure_rows", "resources",
            "canonical_digest",
        }:
            raise StageDFinalizationIntegrityError(
                "runtime checkpoint fields are missing or unknown"
            )
        supplied_digest = raw.get("canonical_digest")
        unsigned = {
            key: value for key, value in raw.items() if key != "canonical_digest"
        }
        if (
            raw.get("schema_version") != 1
            or raw.get("document_type") != "stage_d_runtime_checkpoint"
            or supplied_digest
            != hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
            or raw.get("plan_digest") != plan.canonical_digest
            or raw.get("estimate_digest") != estimate.canonical_digest
            or raw.get("preflight_digest") != preflight.canonical_digest
        ):
            raise StageDFinalizationIntegrityError(
                "runtime checkpoint identity or control binding drifted"
            )
        context = raw.get("execution_context")
        if not isinstance(context, Mapping):
            raise StageDFinalizeIntegrityError("checkpoint execution context is invalid")
        if context.get("formal") is True:
            raise StageDFinalizationBlockedError(
                "D5 unconditionally refuses formal roots; formal sealing is outside D5 authority"
            )
        if context.get("formal") is not False:
            raise StageDFinalizationIntegrityError(
                "checkpoint execution context formal flag is invalid"
            )
        context_document = _exact(
            context, _RUNTIME_CONTEXT_KEYS, "checkpoint execution context",
        )
        provenance_bindings = context_document["provenance_bindings"]
        if (
            not isinstance(provenance_bindings, Mapping)
            or set(provenance_bindings) != {
                "d7_authorization", "producer", "dependencies",
                "model", "source_bindings",
            }
        ):
            raise StageDFinalizationIntegrityError(
                "synthetic checkpoint provenance bindings are not closed"
            )
        synthetic_slot = {
            "status": "NOT_ISSUED", "purpose": "SYNTHETIC_D4_TEST_ONLY",
        }
        if (
            context_document["plan_digest"] != plan.canonical_digest
            or context_document["budget_digest"] != estimate.canonical_digest
            or context_document["preflight_digest"] != preflight.canonical_digest
            or context_document["authorization_scope"]
            != "SYNTHETIC_D4_TEST_ONLY"
            or context_document["execution_authorization"] != "NOT_ISSUED"
            or context_document["access_attestation"] != {
                "reserve_image_bytes_accessed": False,
                "model_accessed": False,
                "cuda_accessed": False,
                "outcome_accessed": False,
            }
            or provenance_bindings["d7_authorization"] != {
                "status": "NOT_ISSUED", "digest": None,
            }
            or context_document["authorization_slot_digest"]
            != hashlib.sha256(canonical_json_bytes(synthetic_slot)).hexdigest()
            or context_document["code_binding_digest"]
            != hashlib.sha256(
                canonical_json_bytes(provenance_bindings["producer"])
            ).hexdigest()
            or context_document["dependency_binding_digest"]
            != hashlib.sha256(
                canonical_json_bytes(provenance_bindings["dependencies"])
            ).hexdigest()
            or context_document["model_binding_digest"]
            != hashlib.sha256(
                canonical_json_bytes(provenance_bindings["model"])
            ).hexdigest()
            or context_document["source_bindings_digest"]
            != hashlib.sha256(
                canonical_json_bytes(provenance_bindings["source_bindings"])
            ).hexdigest()
        ):
            raise StageDFinalizationIntegrityError(
                "synthetic checkpoint context control bindings drifted"
            )
        gate_payload = {
            "formal": False,
            "authorization_scope": context_document["authorization_scope"],
            "execution_authorization": context_document["execution_authorization"],
            "authorization_slot": synthetic_slot,
            "shard_count": context_document["shard_count"],
            "shard_mapping": context_document["shard_mapping"],
            "plan_digest": context_document["plan_digest"],
            "budget_digest": context_document["budget_digest"],
            "preflight_digest": context_document["preflight_digest"],
            "code_binding": provenance_bindings["producer"],
            "dependency_binding": provenance_bindings["dependencies"],
            "model_binding": provenance_bindings["model"],
            "source_bindings": provenance_bindings["source_bindings"],
            "access_attestation": context_document["access_attestation"],
        }
        if context_document["gate_context_digest"] != hashlib.sha256(
            canonical_json_bytes(gate_payload)
        ).hexdigest():
            raise StageDFinalizationIntegrityError(
                "synthetic checkpoint gate-context digest mismatch"
            )
        projection_without_runtime_digest = {
            key: value for key, value in context_document.items()
            if key != "runtime_binding_digest"
        }
        if context_document["runtime_binding_digest"] != hashlib.sha256(
            canonical_json_bytes(projection_without_runtime_digest)
        ).hexdigest():
            raise StageDFinalizationIntegrityError(
                "synthetic checkpoint context runtime digest mismatch"
            )
        synthetic_sha = {
            name: hashlib.sha256(
                ("stage-d-d4-synthetic-" + name).encode("utf-8")
            ).hexdigest()
            for name in ("dependency", "model", "d0", "d1", "d2", "c10")
        }
        shard_count = context_document["shard_count"]
        if (
            type(shard_count) is not int
            or shard_count < 1
            or context_document["shard_mapping"]
            != [index % shard_count for index in range(500)]
            or provenance_bindings != {
                "d7_authorization": {"status": "NOT_ISSUED", "digest": None},
                "producer": {
                    "commit": "d" * 40, "tree": "e" * 40, "clean": True,
                },
                "dependencies": {
                    "status": "SYNTHETIC_NO_DEPENDENCY_AUTHORITY",
                    "lock_sha256": synthetic_sha["dependency"],
                },
                "model": {
                    "status": "SYNTHETIC_NO_MODEL_ACCESS",
                    "config_sha256": synthetic_sha["model"],
                },
                "source_bindings": {
                    name: {
                        "status": "SYNTHETIC_NO_RESERVE_ACCESS",
                        "canonical_digest": synthetic_sha[name],
                    }
                    for name in ("d0", "d1", "d2", "c10")
                },
            }
        ):
            raise StageDFinalizationIntegrityError(
                "synthetic checkpoint context is not the D4-issued test binding"
            )
        return raw
    except StageDFinalizeError:
        raise
    except Exception as error:
        raise StageDFinalizeIntegrityError(
            f"runtime checkpoint control validation failed: {error}"
        ) from error


def _reject_active_checkpoint_control(
    checkpoint: Mapping[str, Any],
) -> None:
    """Fail closed on declared ownership before traversing attempt bytes."""

    lifecycle = checkpoint.get("lifecycle")
    units = checkpoint.get("units")
    groups = checkpoint.get("groups")
    shards = checkpoint.get("shards")
    if (
        not isinstance(lifecycle, Mapping)
        or not isinstance(units, list)
        or not isinstance(groups, list)
        or not isinstance(shards, list)
    ):
        raise StageDFinalizationIntegrityError(
            "runtime checkpoint ownership controls are invalid"
        )
    counters = (
        lifecycle.get("active_shard_leases"),
        lifecycle.get("active_group_claims"),
    )
    if any(type(value) is not int or value < 0 for value in counters):
        raise StageDFinalizationIntegrityError(
            "runtime checkpoint active-claim counters are invalid"
        )
    def has_state(rows: Sequence[Any], state: str) -> bool:
        return any(
            isinstance(row, Mapping) and row.get("state") == state
            for row in rows
        )
    if (
        any(counters)
        or has_state(shards, "leased")
        or has_state(groups, "claimed")
        or has_state(units, "claimed")
    ):
        raise StageDFinalizationBlockedError(
            "cannot finalize or validate a root with active claims"
        )


def _checkpoint(
    root: Path, plan: StageDExecutionPlan, estimate: StageDEstimate,
    preflight: StageDPreflight, *, sealed: bool,
    source_index: Mapping[str, Mapping[str, Any]],
    prechecked: Mapping[str, Any],
    captured_bytes: Mapping[str, bytes] | None = None,
    logical_root: Path | None = None,
) -> dict[str, Any]:
    """Validate the full checkpoint against one captured source inventory."""

    try:
        raw_bytes = _read_snapshot_regular(
            root, root / _runtime.RUNTIME_CHECKPOINT_FILENAME, source_index,
            "runtime checkpoint", captured_bytes=captured_bytes,
        )
        raw = _strict_json(raw_bytes, "runtime checkpoint")
        if raw != _plain(prechecked, "prechecked runtime checkpoint"):
            raise StageDFinalizationIntegrityError(
                "runtime checkpoint changed after the formal-mode precheck"
            )
        _root_layout(root, plan, sealed=sealed)
        logical = root if logical_root is None else logical_root
        mirrored = copy.deepcopy(raw)
        for unit, source_row, mirrored_row in zip(
            plan.units, raw["units"], mirrored["units"],
        ):
            for index, (source_attempt, mirrored_attempt) in enumerate(zip(
                source_row["attempts"], mirrored_row["attempts"],
            ), start=1):
                expected_source_path = (
                    logical / "units" / unit.canonical_digest / "attempts"
                    / f"attempt-{index:06d}"
                )
                if (
                    source_attempt.get("attempt_path") != str(expected_source_path)
                    or source_attempt.get("canonical_digest")
                    != hashlib.sha256(canonical_json_bytes({
                        key: value for key, value in source_attempt.items()
                        if key != "canonical_digest"
                    })).hexdigest()
                ):
                    raise StageDFinalizationIntegrityError(
                        "checkpoint attempt path or self digest drifted"
                    )
                mirrored_attempt["attempt_path"] = str(
                    root / "units" / unit.canonical_digest / "attempts"
                    / f"attempt-{index:06d}"
                )
                mirrored_attempt["canonical_digest"] = hashlib.sha256(
                    canonical_json_bytes({
                        key: value for key, value in mirrored_attempt.items()
                        if key != "canonical_digest"
                    })
                ).hexdigest()
        mirrored["canonical_digest"] = hashlib.sha256(
            canonical_json_bytes({
                key: value for key, value in mirrored.items()
                if key != "canonical_digest"
            })
        ).hexdigest()
        _runtime._validate_checkpoint(
            mirrored, root=root, plan=plan, estimate=estimate,
            preflight=preflight, context_binding=raw["execution_context"],
        )
        return raw
    except StageDFinalizeError:
        raise
    except Exception as error:
        raise StageDFinalizeIntegrityError(
            f"runtime checkpoint validation failed: {error}"
        ) from error


def _root_layout(
    root: Path, plan: StageDExecutionPlan, *, sealed: bool,
) -> None:
    observed = set(os.listdir(root))
    expected = set(_RUNTIME_ROOT_ENTRIES) | ({FINAL_DIRECTORY_NAME} if sealed else set())
    if observed != expected:
        raise StageDFinalizeIntegrityError(
            f"Stage D root layout is not closed; missing={sorted(expected-observed)}, extra={sorted(observed-expected)}"
        )
    for name in expected:
        child = root / name
        metadata = os.lstat(child)
        if stat.S_ISLNK(metadata.st_mode):
            raise StageDFinalizeIntegrityError(f"Stage D root contains symlink {name}")
        if name in {"units", FINAL_DIRECTORY_NAME}:
            if not stat.S_ISDIR(metadata.st_mode):
                raise StageDFinalizationIntegrityError(f"{name} is not a directory")
        elif not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise StageDFinalizeIntegrityError(f"root control is not a single-link file: {name}")
    if sealed:
        _final_layout(root / FINAL_DIRECTORY_NAME)

    # Runtime owns the exact recursive D4 layout validator.  Unsealed roots
    # use it directly; sealed roots apply the same unit/attempt namespace
    # rules after the D5-specific top-level and final-directory checks above.
    if not sealed:
        try:
            _runtime._validate_closed_layout(root, plan)
        except Exception as error:
            raise StageDFinalizationIntegrityError(
                f"Stage D recursive layout validation failed: {error}"
            ) from error
        return
    expected_units = {unit.canonical_digest for unit in plan.units}
    units_root = root / "units"
    try:
        unit_entries = list(units_root.iterdir())
    except OSError as error:
        raise StageDFinalizationIntegrityError(
            "cannot enumerate sealed Stage D units root"
        ) from error
    observed_units: set[str] = set()
    for unit_root in unit_entries:
        metadata = os.lstat(unit_root)
        if (
            unit_root.name not in expected_units
            or stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISDIR(metadata.st_mode)
        ):
            raise StageDFinalizationIntegrityError(
                f"unexpected or unsafe Stage D unit entry: {unit_root.name}"
            )
        observed_units.add(unit_root.name)
        contents = list(unit_root.iterdir())
        if len(contents) != 1 or contents[0].name != "attempts":
            raise StageDFinalizationIntegrityError(
                f"Stage D unit layout drifted: {unit_root.name}"
            )
        attempts_root = contents[0]
        attempt_metadata = os.lstat(attempts_root)
        if (
            stat.S_ISLNK(attempt_metadata.st_mode)
            or not stat.S_ISDIR(attempt_metadata.st_mode)
        ):
            raise StageDFinalizationIntegrityError(
                f"Stage D attempts root is unsafe: {unit_root.name}"
            )
        attempts = list(attempts_root.iterdir())
        expected_names = [
            f"attempt-{index:06d}" for index in range(1, len(attempts) + 1)
        ]
        if (
            len(attempts) > 2
            or sorted(item.name for item in attempts) != expected_names
        ):
            raise StageDFinalizationIntegrityError(
                f"Stage D attempt sequence is ambiguous: {unit_root.name}"
            )
        for attempt in attempts:
            metadata = os.lstat(attempt)
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                raise StageDFinalizationIntegrityError(
                    f"Stage D attempt entry is unsafe: {attempt}"
                )
    if observed_units != expected_units:
        raise StageDFinalizationIntegrityError(
            "Stage D unit directory coverage differs from the exact plan"
        )


def _final_layout(final: Path) -> None:
    metadata = os.lstat(final)
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise StageDFinalizationIntegrityError("final is not a safe directory")
    observed = set(os.listdir(final))
    if observed != _FINAL_FILE_SET:
        raise StageDFinalizationIntegrityError(
            "final layout is not closed; "
            f"missing={sorted(_FINAL_FILE_SET-observed)}, "
            f"extra={sorted(observed-_FINAL_FILE_SET)}"
        )
    for name in FINAL_FILES:
        metadata = os.lstat(final / name)
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
        ):
            raise StageDFinalizationIntegrityError(
                f"final control is not a single-link regular file: {name}"
            )


def _load_npy(raw: bytes, context: str) -> np.ndarray:
    try:
        value = np.load(io.BytesIO(raw), allow_pickle=False)
    except Exception as error:
        raise StageDFinalizeIntegrityError(f"invalid {context}: {error}") from error
    if isinstance(value, np.lib.npyio.NpzFile):
        value.close()
        raise StageDFinalizeIntegrityError(f"{context} is not NPY")
    return np.asarray(value)


def _load_npz(raw: bytes, context: str) -> dict[str, np.ndarray]:
    try:
        with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
            if len(archive.files) != len(set(archive.files)):
                raise ValueError("duplicate archive members")
            return {name: np.asarray(archive[name]) for name in archive.files}
    except Exception as error:
        raise StageDFinalizeIntegrityError(f"invalid {context}: {error}") from error


def _scientific_validation(
    attempt: Path, unit: Any, *, root: Path | None = None,
    source_index: Mapping[str, Mapping[str, Any]] | None = None,
    captured_bytes: Mapping[str, bytes] | None = None,
) -> StageDScientificValidation:
    def read(path: Path, context: str) -> bytes:
        if root is not None and source_index is not None:
            return _read_snapshot_regular(
                root, path, source_index, context,
                captured_bytes=captured_bytes,
            )
        return _read_regular(path, context)

    l0 = _strict_json(
        read(attempt / RESULT_PATH, "L0 result"), "L0 result"
    )
    metric = _load_npy(
        read(attempt / MAP_PATH, "metric-ready map"),
        "metric-ready map",
    )
    archive = _load_npz(
        read(attempt / CURVES_PATH, "fidelity curves"),
        "fidelity curves",
    )
    curves = archive.get("curves") if set(archive) == {"curves"} else None
    audit = (
        _load_npz(
            read(attempt / AUDIT_PATH, "audit evidence"),
            "audit evidence",
        )
        if unit.audit else None
    )
    provenance: Mapping[str, Any] | None = None
    if set(l0) != {"endpoint_parity"}:
        provenance = _strict_json(
            read(attempt / PROVENANCE_PATH, "L0 provenance"),
            "L0 provenance",
        )
    return validate_stage_d_scientific_evidence(
        l0, metric, curves, audit, unit, provenance=provenance,
    )


def _resource_sum(rows: Sequence[Mapping[str, Any]]) -> dict[str, int | float]:
    """Apply D4's resource aggregation to attempt or unit resource rows.

    Additive counters are summed and CUDA peak is the campaign/unit maximum.
    D5 first applies this to a unit's attempts, then to the checkpoint's unit
    aggregates; it never double-counts both levels in the same call.
    """

    keys = (
        "elapsed_seconds", "forward_calls", "backward_calls",
        "l0_l1_payload_bytes", "l2_payload_bytes",
        "control_and_provenance_bytes", "cuda_peak_reserved_bytes",
    )
    result: dict[str, int | float] = {key: 0.0 if key == "elapsed_seconds" else 0 for key in keys}
    for row in rows:
        resources = row["resources"]
        for key in keys:
            if key == "cuda_peak_reserved_bytes":
                result[key] = max(result[key], resources[key])
            else:
                result[key] += resources[key]
    return result


def _validate_retry_history(
    checkpoint_row: Mapping[str, Any], source_attempts: Sequence[Mapping[str, Any]],
    *, unit: Any, context: Mapping[str, Any],
) -> None:
    if not source_attempts or len(source_attempts) not in {1, 2}:
        raise StageDFinalizationIntegrityError(
            "terminal unit must retain exactly one or two attempts"
        )
    if source_attempts[0]["retry_authorization_digest"] is not None:
        raise StageDFinalizationIntegrityError(
            "first attempt cannot carry retry authorization"
        )
    final_provenance = source_attempts[-1].get("provenance")
    if not isinstance(final_provenance, Mapping):
        raise StageDFinalizationIntegrityError(
            "terminal attempt lacks provenance execution binding"
        )
    execution = final_provenance.get("execution")
    if not isinstance(execution, Mapping) or set(execution) != {
        "attempt", "retry", "claim", "shard",
    }:
        raise StageDFinalizationIntegrityError(
            "terminal provenance execution binding is not closed"
        )
    expected_attempt = {
        "number": len(source_attempts),
        "id": source_attempts[-1]["attempt_id"],
    }
    if execution["attempt"] != expected_attempt:
        raise StageDFinalizationIntegrityError(
            "terminal provenance attempt identity drifted"
        )

    unit_retry_digest = checkpoint_row["retry_authorization_digest"]
    if len(source_attempts) == 1:
        if unit_retry_digest is not None or execution["retry"] != {
            "status": "NOT_A_RETRY",
        }:
            raise StageDFinalizationIntegrityError(
                "single-attempt unit contains a retry authorization"
            )
        return

    first, second = source_attempts
    if (
        first["state"] != "infrastructure_failed"
        or first["failure_code"]
        not in _runtime.RETRYABLE_INFRASTRUCTURE_FAILURE_CODES
        or first["result_manifest_published"] is not False
        or first["published_l0_l1_payload_count"] != 0
        or not isinstance(unit_retry_digest, str)
        or second["retry_authorization_digest"] != unit_retry_digest
    ):
        raise StageDFinalizationIntegrityError(
            "two-attempt unit lacks the exact outcome-blind retry chain"
        )
    retry = _exact(
        execution["retry"], _RETRY_AUTHORIZATION_KEYS,
        "provenance retry authorization",
    )
    unsigned = {
        key: value for key, value in retry.items() if key != "canonical_digest"
    }
    if (
        retry["schema_version"] != 1
        or retry["document_type"] != "stage_d_runtime_retry_authorization"
        or retry["status"] != "COORDINATOR_AUTHORIZED_RETRY"
        or retry["canonical_digest"]
        != hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
        or retry["canonical_digest"] != unit_retry_digest
        or retry["unit_digest"] != unit.canonical_digest
        or retry["prior_attempt_digest"] != first["attempt_inventory_digest"]
        or retry["failure_code"] != first["failure_code"]
        or retry["next_attempt_number"] != 2
        or retry["execution_context_digest"]
        != context["runtime_binding_digest"]
    ):
        raise StageDFinalizationIntegrityError(
            "retry authorization does not bind the exact unit/failure/attempt"
        )


def _validate_self_document(
    value: Mapping[str, Any], keys: frozenset[str], document_type: str,
    context: str,
) -> dict[str, Any]:
    document = _exact(value, keys, context)
    if document["schema_version"] != 1 or document["document_type"] != document_type:
        raise StageDFinalizationIntegrityError(f"{context} identity drifted")
    supplied = document["canonical_digest"]
    unsigned = {key: child for key, child in document.items() if key != "canonical_digest"}
    expected = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    if supplied != expected:
        raise StageDFinalizationIntegrityError(f"{context} self digest mismatch")
    return document


_SHA256_HEX = frozenset("0123456789abcdef")
_INVENTORY_ROW_KEYS = frozenset({"path", "bytes", "sha256"})
_RESOURCE_KEYS = frozenset({
    "elapsed_seconds", "forward_calls", "backward_calls",
    "l0_l1_payload_bytes", "l2_payload_bytes",
    "control_and_provenance_bytes", "cuda_peak_reserved_bytes",
})
_COUNT_KEYS = frozenset({
    *_EXPECTED_COUNTS, "attempts", "attempt_statuses",
    "attempt_manifests", "provenance_records", "artifact_records",
    "source_files",
})
_GROUP_ROW_KEYS = frozenset({
    "selection_order", "sample_id", "relative_path", "image_sha256",
    "registry_row_digest", "audit", "audit_overlay_id",
    "group_digest", "pair_binding", "unit_digests",
})
_UNIT_ROW_KEYS = frozenset({
    "selection_order", "sample_id", "group_digest", "pair_digest",
    "unit_digest", "method_index", "method_id", "audit",
    "terminal_state", "terminal", "failure_class", "failure_code",
    "retry_authorization_digest", "resources", "attempts",
    "scientific_errors",
})
_ATTEMPT_ROW_KEYS = frozenset({
    "attempt_number", "attempt_id", "attempt_path",
    "owner_token_digest", "checkpoint_attempt_digest", "state",
    "status", "terminal", "complete", "status_sha256",
    "manifest_sha256", "status_manifest_sha256", "manifest_status",
    "manifest_terminal", "manifest_complete",
    "attempt_inventory_digest", "failure_class", "failure_code",
    "failure", "retry_authorization_digest",
    "result_manifest_published", "published_l0_l1_payload_count",
    "resources", "artifact_count", "artifacts", "provenance",
})
_ARTIFACT_ROW_KEYS = frozenset({
    "level", "path", "kind", "format", "dtype", "shape",
    "bytes", "sha256",
})
_PROVENANCE_PROJECTION_KEYS = frozenset({
    "sha256", "canonical_digest", "mode", "unit_identity_sha256",
    "bindings_digest", "execution",
})
_EXECUTION_PROJECTION_KEYS = frozenset({"attempt", "retry", "claim", "shard"})
_FAILURE_ROW_KEYS = frozenset({
    "unit_digest", "group_selection_order", "method_index", "state",
    "terminal", "attempt_count", "failure_class", "failure_code",
})
_RETENTION_LEVELS = frozenset({"L0", "L1", "L2", "L3"})


def _schema_error(context: str, message: str) -> None:
    raise StageDFinalizationIntegrityError(f"{context} {message}")


def _schema_sha(value: Any, context: str, *, nullable: bool = False) -> None:
    if nullable and value is None:
        return
    if (
        not isinstance(value, str) or len(value) != 64
        or any(character not in _SHA256_HEX for character in value)
    ):
        _schema_error(context, "must be a lowercase SHA-256")


def _schema_int(value: Any, context: str, *, minimum: int = 0) -> None:
    if type(value) is not int or value < minimum:
        _schema_error(context, f"must be an integer >= {minimum}")


def _schema_bool(value: Any, context: str) -> None:
    if type(value) is not bool:
        _schema_error(context, "must be boolean")


def _schema_string(
    value: Any, context: str, *, allowed: set[str] | frozenset[str] | None = None,
) -> None:
    if not isinstance(value, str) or not value or (allowed is not None and value not in allowed):
        _schema_error(context, "has an invalid string value")


def _schema_array(value: Any, context: str) -> list[Any]:
    if not isinstance(value, list):
        _schema_error(context, "must be an array")
    return value


def _schema_path(value: Any, context: str, *, prefixes: tuple[str, ...] = ()) -> None:
    if not isinstance(value, str) or not value or "\\" in value:
        _schema_error(context, "must be a safe relative POSIX path")
    pure = PurePosixPath(value)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        _schema_error(context, "must be a safe relative POSIX path")
    if prefixes and not value.startswith(prefixes):
        _schema_error(context, "is outside the closed path namespace")


def _validate_inventory_row(value: Any, context: str) -> dict[str, Any]:
    row = _exact(value, _INVENTORY_ROW_KEYS, context)
    _schema_path(row["path"], context + ".path")
    _schema_int(row["bytes"], context + ".bytes")
    _schema_sha(row["sha256"], context + ".sha256")
    return row


def _validate_inventory_rows(value: Any, context: str) -> list[dict[str, Any]]:
    rows = [
        _validate_inventory_row(item, f"{context}[{index}]")
        for index, item in enumerate(_schema_array(value, context))
    ]
    paths = [row["path"] for row in rows]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        _schema_error(context, "must be uniquely path-sorted")
    return rows


def _validate_resources(value: Any, context: str) -> dict[str, Any]:
    resources = _exact(value, _RESOURCE_KEYS, context)
    elapsed = resources["elapsed_seconds"]
    if type(elapsed) not in {int, float} or not math.isfinite(float(elapsed)) or elapsed < 0:
        _schema_error(context + ".elapsed_seconds", "must be finite and non-negative")
    for name in _RESOURCE_KEYS - {"elapsed_seconds"}:
        _schema_int(resources[name], f"{context}.{name}")
    return resources


def _validate_pair_binding(value: Any, context: str) -> dict[str, Any]:
    keys = frozenset({
        "schema_version", "document_type", "sample_id",
        "selection_order", "image_sha256", "model_context_digest",
        "target_context_digest", "fidelity_context_digest",
        "plan_context_digest", "canonical_digest",
    })
    row = _exact(value, keys, context)
    if row["schema_version"] != 1 or row["document_type"] != "stage_d_pair_binding":
        _schema_error(context, "identity drifted")
    _schema_string(row["sample_id"], context + ".sample_id")
    _schema_int(row["selection_order"], context + ".selection_order")
    for name in (
        "image_sha256", "model_context_digest", "target_context_digest",
        "fidelity_context_digest", "plan_context_digest",
    ):
        _schema_sha(row[name], f"{context}.{name}")
    _validate_self_document(row, keys, "stage_d_pair_binding", context)
    return row


def _validate_retry(value: Any, context: str) -> dict[str, Any]:
    if isinstance(value, Mapping) and set(value) == {"status"}:
        row = _exact(value, {"status"}, context)
        if row["status"] != "NOT_A_RETRY":
            _schema_error(context, "has an invalid non-retry status")
        return row
    row = _exact(value, _RETRY_AUTHORIZATION_KEYS, context)
    if (
        row["schema_version"] != 1
        or row["document_type"] != "stage_d_runtime_retry_authorization"
        or row["status"] != "COORDINATOR_AUTHORIZED_RETRY"
        or row["failure_code"] not in _runtime.RETRYABLE_INFRASTRUCTURE_FAILURE_CODES
        or row["next_attempt_number"] != 2
    ):
        _schema_error(context, "retry identity or value drifted")
    for name in ("unit_digest", "prior_attempt_digest", "execution_context_digest"):
        _schema_sha(row[name], f"{context}.{name}")
    _validate_self_document(
        row, _RETRY_AUTHORIZATION_KEYS,
        "stage_d_runtime_retry_authorization", context,
    )
    return row


def _validate_failure(value: Any, context: str, state: str) -> Any:
    if state == "succeeded":
        if value is not None:
            _schema_error(context, "must be null for success")
        return None
    if state == "infrastructure_failed":
        row = _exact(value, {"code", "resources"}, context)
        _schema_string(
            row["code"], context + ".code",
            allowed=frozenset(_runtime.RETRYABLE_INFRASTRUCTURE_FAILURE_CODES),
        )
        _validate_resources(row["resources"], context + ".resources")
        return row
    row = _exact(
        value, {
            "code", "phase", "source", "produced", "absent",
            "details_present", "details_sha256",
        }, context,
    )
    _schema_string(row["code"], context + ".code", allowed=_SCIENTIFIC_FAILURE_CODES)
    _schema_string(row["phase"], context + ".phase")
    if row["source"] != "stage_d_scientific_execution":
        _schema_error(context + ".source", "is invalid")
    allowed_paths = {RESULT_PATH, MAP_PATH, CURVES_PATH, AUDIT_PATH}
    produced = _schema_array(row["produced"], context + ".produced")
    absent = _schema_array(row["absent"], context + ".absent")
    if any(path not in allowed_paths for path in [*produced, *absent]):
        _schema_error(context, "contains an invalid scientific path")
    if len(produced) != len(set(produced)) or len(absent) != len(set(absent)) or set(produced) & set(absent):
        _schema_error(context, "scientific paths are duplicated or overlap")
    _schema_bool(row["details_present"], context + ".details_present")
    _schema_sha(
        row["details_sha256"], context + ".details_sha256", nullable=True,
    )
    if (row["details_sha256"] is None) is not (
        row["details_present"] is False
    ):
        _schema_error(context, "details presence/digest binding is invalid")
    return row


def _failure_projection(value: Any, state: str) -> Any:
    """Project source failure evidence into a recursively closed D5 row."""

    if state != "scientific_failed":
        return _plain(value, "attempt failure")
    source = _plain(value, "scientific failure")
    if not isinstance(source, dict):
        raise StageDFinalizationIntegrityError(
            "scientific failure source is not an object"
        )
    details_present = "details" in source
    details_sha256 = (
        hashlib.sha256(canonical_json_bytes(source["details"])).hexdigest()
        if details_present else None
    )
    return {
        "code": source.get("code"),
        "phase": source.get("phase"),
        "source": source.get("source"),
        "produced": source.get("produced"),
        "absent": source.get("absent"),
        "details_present": details_present,
        "details_sha256": details_sha256,
    }


def _validate_artifact(value: Any, context: str) -> dict[str, Any]:
    row = _exact(value, _ARTIFACT_ROW_KEYS, context)
    _schema_string(row["level"], context + ".level", allowed=_RETENTION_LEVELS - {"L3"})
    _schema_path(row["path"], context + ".path", prefixes=("L0/", "L1/", "L2/"))
    if row["path"].split("/", 1)[0] != row["level"]:
        _schema_error(context, "level/path mismatch")
    allowed = {
        RESULT_PATH: ("result", "json", "json", None),
        "L0/failure.json": ("scientific_failure", "json", "json", None),
        PROVENANCE_PATH: ("provenance", "json", "json", None),
        MAP_PATH: ("metric_ready_map", "npy", "float32", [56, 56]),
        CURVES_PATH: ("fidelity_curves", "npz", "float64", [2, 5, 21]),
    }
    if row["path"] == AUDIT_PATH:
        if row["kind"] != "audit_evidence" or row["format"] != "npz":
            _schema_error(context, "audit artifact identity is invalid")
        dtype = row["dtype"]
        shape = row["shape"]
        if (
            not isinstance(dtype, Mapping) or not isinstance(shape, Mapping)
            or not dtype or set(dtype) != set(shape)
            or "repeat_clean_metric_ready_map" not in dtype
            or any(
                not isinstance(key, str) or not key
                or not isinstance(item, str) or not item
                for key, item in dtype.items()
            )
        ):
            _schema_error(context, "audit dtype/shape maps are invalid")
        if dtype["repeat_clean_metric_ready_map"] != "float32":
            _schema_error(context + ".dtype", "is invalid")
        if shape["repeat_clean_metric_ready_map"] != [56, 56]:
            _schema_error(context + ".shape", "is invalid")
        for key, item in shape.items():
            if not isinstance(item, list) or not item or any(
                type(dimension) is not int or dimension < 1 for dimension in item
            ):
                _schema_error(f"{context}.shape.{key}", "is invalid")
    elif row["path"] not in allowed or (
        row["kind"], row["format"], row["dtype"], row["shape"]
    ) != allowed[row["path"]]:
        _schema_error(context, "artifact identity or representation is invalid")
    _schema_int(row["bytes"], context + ".bytes")
    _schema_sha(row["sha256"], context + ".sha256")
    return row


def _validate_provenance_projection(value: Any, context: str) -> dict[str, Any]:
    row = _exact(value, _PROVENANCE_PROJECTION_KEYS, context)
    for name in ("sha256", "canonical_digest", "unit_identity_sha256", "bindings_digest"):
        _schema_sha(row[name], f"{context}.{name}")
    if row["mode"] != "SYNTHETIC_D4_TEST_ONLY":
        _schema_error(context + ".mode", "is outside D5 synthetic scope")
    execution = _exact(row["execution"], _EXECUTION_PROJECTION_KEYS, context + ".execution")
    attempt = _exact(execution["attempt"], {"number", "id"}, context + ".execution.attempt")
    _schema_int(attempt["number"], context + ".execution.attempt.number", minimum=1)
    if attempt["number"] not in {1, 2} or attempt["id"] != f"attempt-{attempt['number']:06d}":
        _schema_error(context + ".execution.attempt", "identity is invalid")
    _validate_retry(execution["retry"], context + ".execution.retry")
    claim = _exact(
        execution["claim"], {"owner_token_digest", "group_owner_token_digest"},
        context + ".execution.claim",
    )
    _schema_sha(claim["owner_token_digest"], context + ".execution.claim.owner_token_digest")
    _schema_sha(claim["group_owner_token_digest"], context + ".execution.claim.group_owner_token_digest")
    shard = _exact(execution["shard"], {"id", "count"}, context + ".execution.shard")
    _schema_int(shard["id"], context + ".execution.shard.id")
    _schema_int(shard["count"], context + ".execution.shard.count", minimum=1)
    if shard["id"] >= shard["count"]:
        _schema_error(context + ".execution.shard", "id is outside count")
    return row


def _validate_attempt_projection(value: Any, context: str) -> dict[str, Any]:
    row = _exact(value, _ATTEMPT_ROW_KEYS, context)
    state = row["state"]
    _schema_string(
        state, context + ".state",
        allowed={"succeeded", "scientific_failed", "infrastructure_failed"},
    )
    number = row["attempt_number"]
    _schema_int(number, context + ".attempt_number", minimum=1)
    if number not in {1, 2} or row["attempt_id"] != f"attempt-{number:06d}":
        _schema_error(context, "attempt identity is invalid")
    _schema_path(row["attempt_path"], context + ".attempt_path", prefixes=("units/",))
    for name in (
        "owner_token_digest", "checkpoint_attempt_digest", "status_sha256",
        "manifest_sha256", "status_manifest_sha256",
        "attempt_inventory_digest",
    ):
        _schema_sha(row[name], f"{context}.{name}")
    _schema_sha(row["retry_authorization_digest"], context + ".retry_authorization_digest", nullable=True)
    expected_status = {
        "succeeded": "succeeded",
        "scientific_failed": "scientific_failure",
        "infrastructure_failed": "infrastructure_interruption",
    }[state]
    terminal = state != "infrastructure_failed"
    if any(row[name] != expected_status for name in ("status", "manifest_status")) or any(
        row[name] is not terminal
        for name in ("terminal", "complete", "manifest_terminal", "manifest_complete", "result_manifest_published")
    ):
        _schema_error(context, "attempt lifecycle projection is invalid")
    expected_class = {
        "succeeded": None, "scientific_failed": "SCIENTIFIC_INVALIDITY",
        "infrastructure_failed": "INFRASTRUCTURE_FAILURE",
    }[state]
    if row["failure_class"] != expected_class:
        _schema_error(context + ".failure_class", "is invalid")
    if state == "succeeded":
        if row["failure_code"] is not None:
            _schema_error(context + ".failure_code", "must be null")
    elif state == "scientific_failed":
        _schema_string(row["failure_code"], context + ".failure_code", allowed=_SCIENTIFIC_FAILURE_CODES)
    else:
        _schema_string(row["failure_code"], context + ".failure_code", allowed=frozenset(_runtime.RETRYABLE_INFRASTRUCTURE_FAILURE_CODES))
    _validate_failure(row["failure"], context + ".failure", state)
    _schema_int(row["published_l0_l1_payload_count"], context + ".published_l0_l1_payload_count")
    _validate_resources(row["resources"], context + ".resources")
    artifacts = [
        _validate_artifact(item, f"{context}.artifacts[{index}]")
        for index, item in enumerate(_schema_array(row["artifacts"], context + ".artifacts"))
    ]
    if row["artifact_count"] != len(artifacts):
        _schema_error(context + ".artifact_count", "does not match artifacts")
    if state == "infrastructure_failed":
        if artifacts or row["provenance"] is not None:
            _schema_error(context, "infrastructure attempt retains terminal payload")
    else:
        _validate_provenance_projection(row["provenance"], context + ".provenance")
    return row


def _validate_reconciliation(document: Mapping[str, Any]) -> None:
    for scalar in ("expected_image_count", "expected_unit_count", "image_count", "unit_count"):
        _schema_int(document[scalar], f"reconciliation.{scalar}")
    if document["expected_image_count"] != 500 or document["expected_unit_count"] != 1000 or document["image_count"] != 500 or document["unit_count"] != 1000:
        _schema_error("reconciliation", "frozen cardinalities drifted")
    _schema_sha(document["plan_digest"], "reconciliation.plan_digest")
    for family in ("expected_counts", "observed_counts"):
        counts = _exact(document[family], _COUNT_KEYS, f"reconciliation.{family}")
        for name, value in counts.items():
            _schema_int(value, f"reconciliation.{family}.{name}")
    if any(document["expected_counts"][name] != value for name, value in _EXPECTED_COUNTS.items()):
        _schema_error("reconciliation.expected_counts", "frozen workload counts drifted")
    states = document["state_counts"]
    if not isinstance(states, Mapping) or not states or any(
        key not in {"succeeded", "scientific_failed"} for key in states
    ):
        _schema_error("reconciliation.state_counts", "has unknown states")
    for name, value in states.items():
        _schema_int(value, f"reconciliation.state_counts.{name}")
    group_rows = _schema_array(document["group_rows"], "reconciliation.group_rows")
    unit_rows = _schema_array(document["rows"], "reconciliation.rows")
    failure_rows = _schema_array(document["failure_rows"], "reconciliation.failure_rows")
    na_rows = _schema_array(document["na_rows"], "reconciliation.na_rows")
    if not group_rows or len(unit_rows) != 2 * len(group_rows) or len(failure_rows) != len(unit_rows):
        _schema_error("reconciliation", "row cardinalities do not form complete method pairs")
    projected_group_units: list[str] = []
    for index, value in enumerate(group_rows):
        row = _exact(value, _GROUP_ROW_KEYS, f"reconciliation.group_rows[{index}]")
        _schema_int(row["selection_order"], f"reconciliation.group_rows[{index}].selection_order")
        for name in ("sample_id", "relative_path", "audit_overlay_id"):
            if name == "audit_overlay_id" and row[name] is None:
                continue
            _schema_string(row[name], f"reconciliation.group_rows[{index}].{name}")
        for name in ("image_sha256", "registry_row_digest", "group_digest"):
            _schema_sha(row[name], f"reconciliation.group_rows[{index}].{name}")
        _schema_bool(row["audit"], f"reconciliation.group_rows[{index}].audit")
        _validate_pair_binding(row["pair_binding"], f"reconciliation.group_rows[{index}].pair_binding")
        digests = _schema_array(row["unit_digests"], f"reconciliation.group_rows[{index}].unit_digests")
        if len(digests) != 2:
            _schema_error(f"reconciliation.group_rows[{index}].unit_digests", "must contain the exact method pair")
        for position, digest in enumerate(digests):
            _schema_sha(digest, f"reconciliation.group_rows[{index}].unit_digests[{position}]")
        projected_group_units.extend(digests)
    normalized_unit_rows: list[dict[str, Any]] = []
    for index, value in enumerate(unit_rows):
        row = _exact(value, _UNIT_ROW_KEYS, f"reconciliation.rows[{index}]")
        for name in ("selection_order", "method_index"):
            _schema_int(row[name], f"reconciliation.rows[{index}].{name}")
        if row["method_index"] not in {0, 1}:
            _schema_error(f"reconciliation.rows[{index}].method_index", "is invalid")
        _schema_string(row["sample_id"], f"reconciliation.rows[{index}].sample_id")
        for name in ("group_digest", "pair_digest", "unit_digest"):
            _schema_sha(row[name], f"reconciliation.rows[{index}].{name}")
        _schema_string(row["method_id"], f"reconciliation.rows[{index}].method_id", allowed={NNS_METHOD_ID, MATCHED_METHOD_ID})
        _schema_bool(row["audit"], f"reconciliation.rows[{index}].audit")
        _schema_bool(row["terminal"], f"reconciliation.rows[{index}].terminal")
        if row["terminal"] is not True:
            _schema_error(f"reconciliation.rows[{index}].terminal", "must be true")
        _schema_string(row["terminal_state"], f"reconciliation.rows[{index}].terminal_state", allowed={"succeeded", "scientific_failed"})
        expected_class = None if row["terminal_state"] == "succeeded" else "SCIENTIFIC_INVALIDITY"
        if row["failure_class"] != expected_class:
            _schema_error(f"reconciliation.rows[{index}].failure_class", "is invalid")
        _schema_sha(row["retry_authorization_digest"], f"reconciliation.rows[{index}].retry_authorization_digest", nullable=True)
        _validate_resources(row["resources"], f"reconciliation.rows[{index}].resources")
        attempts = _schema_array(row["attempts"], f"reconciliation.rows[{index}].attempts")
        if len(attempts) not in {1, 2}:
            _schema_error(f"reconciliation.rows[{index}].attempts", "has invalid cardinality")
        for attempt_index, attempt in enumerate(attempts):
            _validate_attempt_projection(attempt, f"reconciliation.rows[{index}].attempts[{attempt_index}]")
        if [attempt["attempt_number"] for attempt in attempts] != list(range(1, len(attempts) + 1)) or attempts[-1]["state"] != row["terminal_state"]:
            _schema_error(f"reconciliation.rows[{index}].attempts", "sequence or terminal state is invalid")
        if len(attempts) == 2 and (
            attempts[0]["state"] != "infrastructure_failed"
            or attempts[1]["retry_authorization_digest"] != row["retry_authorization_digest"]
        ):
            _schema_error(f"reconciliation.rows[{index}].attempts", "retry chain is invalid")
        if _resource_sum([{"resources": attempt["resources"]} for attempt in attempts]) != row["resources"]:
            _schema_error(f"reconciliation.rows[{index}].resources", "does not equal attempt resources")
        errors = _schema_array(row["scientific_errors"], f"reconciliation.rows[{index}].scientific_errors")
        if any(not isinstance(item, str) or not item for item in errors):
            _schema_error(f"reconciliation.rows[{index}].scientific_errors", "is invalid")
        normalized_unit_rows.append(row)
    if [row["unit_digest"] for row in normalized_unit_rows] != projected_group_units or len(set(projected_group_units)) != len(projected_group_units):
        _schema_error("reconciliation", "group/unit left join is reordered or duplicated")
    normalized_failure_rows: list[dict[str, Any]] = []
    for index, value in enumerate(failure_rows):
        row = _exact(value, _FAILURE_ROW_KEYS, f"reconciliation.failure_rows[{index}]")
        _schema_sha(row["unit_digest"], f"reconciliation.failure_rows[{index}].unit_digest")
        for name in ("group_selection_order", "method_index", "attempt_count"):
            _schema_int(row[name], f"reconciliation.failure_rows[{index}].{name}")
        if row["method_index"] not in {0, 1} or row["attempt_count"] not in {1, 2}:
            _schema_error(f"reconciliation.failure_rows[{index}]", "method/attempt count is invalid")
        _schema_string(row["state"], f"reconciliation.failure_rows[{index}].state", allowed={"succeeded", "scientific_failed"})
        _schema_bool(row["terminal"], f"reconciliation.failure_rows[{index}].terminal")
        if row["terminal"] is not True:
            _schema_error(f"reconciliation.failure_rows[{index}].terminal", "must be true")
        expected_class = None if row["state"] == "succeeded" else "SCIENTIFIC_INVALIDITY"
        if row["failure_class"] != expected_class:
            _schema_error(f"reconciliation.failure_rows[{index}].failure_class", "is invalid")
        if row["state"] == "succeeded":
            if row["failure_code"] is not None:
                _schema_error(f"reconciliation.failure_rows[{index}].failure_code", "must be null")
        else:
            _schema_string(row["failure_code"], f"reconciliation.failure_rows[{index}].failure_code", allowed=_SCIENTIFIC_FAILURE_CODES)
        normalized_failure_rows.append(row)
    expected_failure_rows = [{
        "unit_digest": row["unit_digest"],
        "group_selection_order": row["selection_order"],
        "method_index": row["method_index"],
        "state": row["terminal_state"], "terminal": row["terminal"],
        "attempt_count": len(row["attempts"]),
        "failure_class": row["failure_class"],
        "failure_code": row["failure_code"],
    } for row in normalized_unit_rows]
    if normalized_failure_rows != expected_failure_rows:
        _schema_error("reconciliation.failure_rows", "does not project unit rows exactly")
    normalized_na_rows: list[dict[str, Any]] = []
    for index, value in enumerate(na_rows):
        row = _exact(value, {"unit_digest", "reason"}, f"reconciliation.na_rows[{index}]")
        _schema_sha(row["unit_digest"], f"reconciliation.na_rows[{index}].unit_digest")
        _schema_string(row["reason"], f"reconciliation.na_rows[{index}].reason", allowed=_SCIENTIFIC_FAILURE_CODES)
        normalized_na_rows.append(row)
    expected_na = [{"unit_digest": row["unit_digest"], "reason": row["failure_code"]} for row in normalized_unit_rows if row["terminal_state"] == "scientific_failed"]
    if normalized_na_rows != expected_na:
        _schema_error("reconciliation.na_rows", "does not reconcile scientific failures")
    _validate_resources(document["resources"], "reconciliation.resources")
    retention = _exact(document["retention"], {"expected", "observed", "audit_image_count", "audit_unit_count"}, "reconciliation.retention")
    expected = _exact(retention["expected"], {"L0_unit_coverage", "L1_unit_coverage", "L2_unit_coverage", "L3_file_count"}, "reconciliation.retention.expected")
    observed = _exact(retention["observed"], {"L0_unit_coverage", "L1_unit_coverage", "L2_unit_coverage", "L3_file_count", "files_by_level", "file_unit_coverage"}, "reconciliation.retention.observed")
    for family, values in (("expected", expected), ("observed", observed)):
        for name in ("L0_unit_coverage", "L1_unit_coverage", "L2_unit_coverage", "L3_file_count"):
            _schema_int(values[name], f"reconciliation.retention.{family}.{name}")
    if expected != {"L0_unit_coverage": 1000, "L1_unit_coverage": 1000, "L2_unit_coverage": 10, "L3_file_count": 0}:
        _schema_error("reconciliation.retention.expected", "frozen policy drifted")
    for family in ("files_by_level", "file_unit_coverage"):
        values = _exact(observed[family], _RETENTION_LEVELS, f"reconciliation.retention.observed.{family}")
        for name, value in values.items():
            _schema_int(value, f"reconciliation.retention.observed.{family}.{name}")
    if retention["audit_image_count"] != 5 or retention["audit_unit_count"] != 10:
        _schema_error("reconciliation.retention", "audit cardinality drifted")
    for name in ("checkpoint_failure_table_reconciled", "checkpoint_resources_reconciled", "no_rows_dropped"):
        if document[name] is not True:
            _schema_error(f"reconciliation.{name}", "must be true")
    if document["zero_imputation_performed"] is not False:
        _schema_error("reconciliation.zero_imputation_performed", "must be false")


def _validate_final_document(name: str, document: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively validate one of the five immutable D5 controls."""

    document_types = {
        "evidence_report.json": "stage_d_final_evidence_report",
        "reconciliation.json": "stage_d_reconciliation",
        "source_inventory.json": "stage_d_source_inventory",
        "payload_manifest.json": "stage_d_final_payload_manifest",
        "status.json": "stage_d_final_status",
    }
    if name not in document_types:
        _schema_error("final document", "has an unknown filename")
    value = _validate_self_document(
        document, _DOCUMENT_KEYS[name], document_types[name], name,
    )
    if name == "evidence_report.json":
        for field in ("package_complete", "analysis_performed", "bootstrap_performed", "decision_performed"):
            _schema_bool(value[field], f"{name}.{field}")
        if value["package_complete"] is not True or any(value[field] for field in ("analysis_performed", "bootstrap_performed", "decision_performed")):
            _schema_error(name, "package/analysis boundary is invalid")
        _schema_string(value["scientific_validity"], name + ".scientific_validity", allowed={"PASS", "FAIL"})
        if value["workflow_status"] != "PASS":
            _schema_error(name + ".workflow_status", "must be PASS")
        for field in ("plan_digest", "estimate_digest", "preflight_digest", "execution_context_digest", "reconciliation_digest", "source_inventory_digest"):
            _schema_sha(value[field], f"{name}.{field}")
        counts = _exact(value["counts"], {"images", "groups", "units", "attempts", "source_files", "scientific_failures"}, name + ".counts")
        for field, item in counts.items():
            _schema_int(item, f"{name}.counts.{field}")
        if counts["images"] != 500 or counts["groups"] != 500 or counts["units"] != 1000:
            _schema_error(name + ".counts", "frozen cardinality drifted")
        for index, item in enumerate(_schema_array(value["scientific_failures"], name + ".scientific_failures")):
            row = _exact(item, {"unit_digest", "reasons"}, f"{name}.scientific_failures[{index}]")
            _schema_sha(row["unit_digest"], f"{name}.scientific_failures[{index}].unit_digest")
            reasons = _schema_array(row["reasons"], f"{name}.scientific_failures[{index}].reasons")
            if not reasons or any(not isinstance(reason, str) or not reason for reason in reasons):
                _schema_error(f"{name}.scientific_failures[{index}].reasons", "is invalid")
        authorization = _exact(value["authorization"], {"formal", "execution_authorization", "authorization_scope", "analysis_authorization", "stage_d_execution_started"}, name + ".authorization")
        if authorization != {"formal": False, "execution_authorization": "NOT_ISSUED", "authorization_scope": "SYNTHETIC_D4_TEST_ONLY", "analysis_authorization": "NOT_EVALUATED", "stage_d_execution_started": False}:
            _schema_error(name + ".authorization", "crosses the D5 authority boundary")
        boundary = _exact(value["analysis_boundary"], {"status", "analysis_performed", "bootstrap_performed", "decision_performed", "delta_fid_computed", "aggregate_effects_computed", "d6_eligibility"}, name + ".analysis_boundary")
        if boundary != {"status": "NOT_PERFORMED", "analysis_performed": False, "bootstrap_performed": False, "decision_performed": False, "delta_fid_computed": False, "aggregate_effects_computed": False, "d6_eligibility": "SYNTHETIC_ONLY"}:
            _schema_error(name + ".analysis_boundary", "is invalid")
    elif name == "reconciliation.json":
        _validate_reconciliation(value)
    elif name == "source_inventory.json":
        if value["root_scope"] != "entire_stage_d_root_excluding_final_subtree":
            _schema_error(name + ".root_scope", "is invalid")
        rows = _validate_inventory_rows(value["files"], name + ".files")
        if value["file_count"] != len(rows) or value["total_bytes"] != sum(row["bytes"] for row in rows):
            _schema_error(name, "inventory totals drifted")
    elif name == "payload_manifest.json":
        if value["immutable"] is not True or value["artifact_count"] != 3:
            _schema_error(name, "immutability/cardinality drifted")
        rows = _validate_inventory_rows(value["artifacts"], name + ".artifacts")
        if [row["path"] for row in rows] != list(FINAL_PAYLOAD_FILES[:-1]):
            _schema_error(name + ".artifacts", "does not bind the exact payload set")
    else:
        if value["terminal"] is not True or value["package_complete"] is not True or value["workflow_status"] != "PASS":
            _schema_error(name, "terminal/package/workflow state is invalid")
        _schema_string(value["scientific_validity"], name + ".scientific_validity", allowed={"PASS", "FAIL"})
        _schema_sha(value["plan_digest"], name + ".plan_digest")
        _schema_sha(value["payload_manifest_sha256"], name + ".payload_manifest_sha256")
        sealed = _exact(value["sealed_root_inventory"], _SEALED_INVENTORY_KEYS, name + ".sealed_root_inventory")
        if sealed["scope"] != "whole_root_excluding_status_self" or sealed["ordering"] != "path_lexicographic" or sealed["excluded_paths"] != ["final/status.json"]:
            _schema_error(name + ".sealed_root_inventory", "identity is invalid")
        rows = _validate_inventory_rows(sealed["files"], name + ".sealed_root_inventory.files")
        if sealed["file_count"] != len(rows) or sealed["total_bytes"] != sum(row["bytes"] for row in rows):
            _schema_error(name + ".sealed_root_inventory", "totals drifted")
        expected = hashlib.sha256(canonical_json_bytes({key: child for key, child in sealed.items() if key != "canonical_digest"})).hexdigest()
        if sealed["canonical_digest"] != expected:
            _schema_error(name + ".sealed_root_inventory", "self digest mismatch")
    return value


def _validate_root_validation_document(value: Mapping[str, Any]) -> dict[str, Any]:
    document = _exact(value, _RESULT_KEYS, "root validation")
    _schema_bool(document["package_complete"], "root validation.package_complete")
    if document["package_complete"] is not True or document["workflow_status"] != "PASS":
        _schema_error("root validation", "conclusion is invalid")
    _schema_string(document["scientific_validity"], "root validation.scientific_validity", allowed={"PASS", "FAIL"})
    _schema_sha(document["plan_digest"], "root validation.plan_digest")
    _schema_sha(document["final_status_sha256"], "root validation.final_status_sha256")
    summary = _exact(document["root_inventory"], {"file_count", "total_bytes", "canonical_digest"}, "root validation.root_inventory")
    _schema_int(summary["file_count"], "root validation.root_inventory.file_count")
    _schema_int(summary["total_bytes"], "root validation.root_inventory.total_bytes")
    _schema_sha(summary["canonical_digest"], "root validation.root_inventory.canonical_digest")
    for field, filename in (
        ("evidence_report", "evidence_report.json"),
        ("reconciliation", "reconciliation.json"),
        ("source_inventory", "source_inventory.json"),
        ("payload_manifest", "payload_manifest.json"),
    ):
        _validate_final_document(filename, document[field])
    rows = _validate_inventory_rows(
        document["whole_root_ordered_inventory"],
        "root validation.whole_root_ordered_inventory",
    )
    if summary != _inventory_summary(rows):
        _schema_error("root validation.root_inventory", "summary differs from rows")
    return document


def _build_source_documents(
    root: Path, plan: StageDExecutionPlan, estimate: StageDEstimate,
    preflight: StageDPreflight, *, allow_formal: bool, sealed: bool,
    source_records: Sequence[Mapping[str, Any]],
    prechecked_checkpoint: Mapping[str, Any],
    captured_bytes: Mapping[str, bytes] | None = None,
    logical_root: Path | None = None,
) -> tuple[dict[str, bytes], dict[str, Any]]:
    del allow_formal  # Kept in the public signature for compatibility only.
    source_records = _plain(list(source_records), "captured source inventory")
    source_index = _inventory_index(source_records)
    checkpoint = _checkpoint(
        root, plan, estimate, preflight, sealed=sealed,
        source_index=source_index, prechecked=prechecked_checkpoint,
        captured_bytes=captured_bytes,
        logical_root=logical_root,
    )
    context = checkpoint["execution_context"]
    if context["formal"] is True:
        raise StageDFinalizationBlockedError(
            "D5 unconditionally refuses formal roots; formal sealing is outside D5 authority"
        )
    if context["formal"] is False and (
        context["execution_authorization"] != "NOT_ISSUED"
        or context["authorization_scope"] != "SYNTHETIC_D4_TEST_ONLY"
        or preflight.analysis_authorization != "NOT_EVALUATED"
    ):
        raise StageDFinalizationIntegrityError(
            "synthetic root authority boundary drifted"
        )
    lifecycle = checkpoint["lifecycle"]
    if lifecycle["active_shard_leases"] or lifecycle["active_group_claims"] or any(row["state"] == "claimed" for row in checkpoint["units"]):
        raise StageDFinalizationBlockedError("cannot finalize a root with active claims")
    if lifecycle["state"] != "complete" or any(not row["terminal"] for row in checkpoint["shards"]):
        raise StageDFinalizationBlockedError(
            "cannot finalize before every shard and unit is terminal"
        )
    for name, expected in _EXPECTED_COUNTS.items():
        if name == "registry_rows":
            continue
        if plan.counts.get(name) != expected:
            raise StageDFinalizationIntegrityError(
                f"plan count drifted for {name}"
            )

    rows: list[dict[str, Any]] = []
    groups: list[dict[str, Any]] = []
    validations: dict[str, StageDScientificValidation] = {}
    observed_attempts: list[dict[str, Any]] = []
    observed_artifact_kinds: Counter[str] = Counter()
    observed_statuses = 0
    observed_manifests = 0
    observed_provenance = 0
    contract_artifact_records = 0
    contract_primary_maps = 0
    contract_audit_maps = 0
    contract_curve_bundles = 0
    retention_level_files: Counter[str] = Counter()
    retention_units: dict[str, set[str]] = {level: set() for level in ("L0", "L1", "L2", "L3")}
    retained_file_units: dict[str, set[str]] = {
        level: set() for level in ("L0", "L1", "L2", "L3")
    }
    expected_retained_units: dict[str, set[str]] = {
        level: set() for level in ("L0", "L1", "L2", "L3")
    }
    covered_primary_maps = 0
    covered_audit_maps = 0
    covered_curve_bundles = 0
    by_checkpoint = {row["unit_digest"]: row for row in checkpoint["units"]}
    if len(by_checkpoint) != 1000:
        raise StageDFinalizationIntegrityError(
            "checkpoint unit join is duplicated or incomplete"
        )
    for group in plan.groups:
        groups.append({
            "selection_order": group.selection_order,
            "sample_id": group.sample_id,
            "relative_path": group.relative_path,
            "image_sha256": group.image_sha256,
            "registry_row_digest": group.registry_row_digest,
            "audit": group.audit,
            "audit_overlay_id": group.audit_overlay_id,
            "group_digest": group.canonical_digest,
            "pair_binding": group.pair_binding.to_dict(),
            "unit_digests": [unit.canonical_digest for unit in group.units],
        })
        group_rows: list[dict[str, Any]] = []
        for unit in group.units:
            row = by_checkpoint.get(unit.canonical_digest)
            if row is None:
                raise StageDFinalizationIntegrityError(
                    "plan unit is absent from checkpoint"
                )
            attempts = attempt_directories(root / "units" / unit.canonical_digest / "attempts")
            if not attempts or len(attempts) != len(row["attempts"]):
                raise StageDFinalizationIntegrityError(
                    f"attempt coverage drifted for {unit.canonical_digest}"
                )
            source_attempts: list[dict[str, Any]] = []
            for index, attempt in enumerate(attempts):
                try:
                    source = _runtime._source_validate_attempt(
                        attempt, plan=plan, group=group, unit=unit, context_binding=context,
                    )
                except Exception as error:
                    raise StageDFinalizationIntegrityError(
                        f"attempt source validation failed for {unit.canonical_digest}: {error}"
                    ) from error
                snapshot_attempt_digest = _attempt_snapshot_digest(
                    root, attempt, source_index,
                )
                if source["attempt_inventory_digest"] != snapshot_attempt_digest:
                    raise StageDFinalizationIntegrityError(
                        "attempt validation differs from the captured source snapshot"
                    )
                recorded = row["attempts"][index]
                for key in ("state", "attempt_id", "status_sha256", "attempt_inventory_digest", "failure_class", "failure_code", "result_manifest_published", "published_l0_l1_payload_count", "resources"):
                    if source[key] != recorded[key]:
                        raise StageDFinalizationIntegrityError(
                            f"checkpoint/attempt {key} mismatch for {unit.canonical_digest}"
                        )
                status_raw = _read_snapshot_regular(
                    root, attempt / "status.json", source_index, "attempt status",
                    captured_bytes=captured_bytes,
                )
                manifest_raw = _read_snapshot_regular(
                    root, attempt / "manifest.json", source_index, "attempt manifest",
                    captured_bytes=captured_bytes,
                )
                status_document = _strict_json(status_raw, "attempt status")
                manifest_document = _strict_json(manifest_raw, "attempt manifest")
                observed_statuses += 1
                observed_manifests += 1
                if source["state"] == "infrastructure_failed":
                    expected_attempt_artifacts = 0
                elif source["state"] == "succeeded":
                    expected_attempt_artifacts = 4 + int(unit.audit)
                elif source["state"] == "scientific_failed":
                    declared_failure = manifest_document.get("failure")
                    declared_produced = (
                        declared_failure.get("produced")
                        if isinstance(declared_failure, Mapping) else None
                    )
                    if (
                        not isinstance(declared_produced, list)
                        or any(not isinstance(path, str) for path in declared_produced)
                    ):
                        raise StageDFinalizationIntegrityError(
                            "scientific failure lacks closed produced-artifact evidence"
                        )
                    expected_attempt_artifacts = len(declared_produced) + 2
                else:
                    raise StageDFinalizationIntegrityError(
                        "source attempt has an unsupported state"
                    )
                contract_artifact_records += expected_attempt_artifacts
                artifacts = _plain(manifest_document.get("artifacts", []), "manifest artifacts")
                if not isinstance(artifacts, list):
                    raise StageDFinalizationIntegrityError(
                        "attempt manifest artifacts are not an array"
                    )
                for artifact in artifacts:
                    if not isinstance(artifact, Mapping):
                        raise StageDFinalizationIntegrityError(
                            "attempt manifest contains a non-object artifact"
                        )
                    level = artifact.get("level")
                    if level not in retention_units:
                        raise StageDFinalizationIntegrityError(
                            "attempt manifest contains an unknown retention level"
                        )
                    retention_level_files[level] += 1
                    retention_units[level].add(unit.canonical_digest)
                    retained_file_units[level].add(unit.canonical_digest)
                    kind = artifact.get("kind")
                    path = artifact.get("path")
                    if not isinstance(kind, str) or not isinstance(path, str):
                        raise StageDFinalizationIntegrityError(
                            "attempt manifest artifact identity is invalid"
                        )
                    observed_artifact_kinds[kind] += 1
                    if level == "L2" and not unit.audit:
                        raise StageDFinalizationIntegrityError(
                            "L2 evidence is retained for a non-audit unit"
                        )
                    if level == "L3":
                        raise StageDFinalizationIntegrityError(
                            "L3 evidence is forbidden"
                        )
                provenance_sha256 = manifest_document.get("provenance_sha256")
                provenance_facts: dict[str, Any] | None = None
                if provenance_sha256 is not None:
                    provenance_raw = _read_snapshot_regular(
                        root, attempt / PROVENANCE_PATH, source_index,
                        "attempt provenance",
                        captured_bytes=captured_bytes,
                    )
                    provenance_document = _strict_json(
                        provenance_raw, "attempt provenance"
                    )
                    bindings = provenance_document.get("bindings")
                    execution = (
                        bindings.get("execution")
                        if isinstance(bindings, Mapping) else None
                    )
                    provenance_facts = {
                        "sha256": hashlib.sha256(provenance_raw).hexdigest(),
                        "canonical_digest": provenance_document.get("canonical_digest"),
                        "mode": provenance_document.get("mode"),
                        "unit_identity_sha256": provenance_document.get("unit_identity_sha256"),
                        "bindings_digest": hashlib.sha256(canonical_json_bytes(
                            provenance_document.get("bindings", {})
                        )).hexdigest(),
                        "execution": _plain(execution, "provenance execution"),
                    }
                    observed_provenance += 1
                    if provenance_facts["sha256"] != provenance_sha256:
                        raise StageDFinalizationIntegrityError(
                            "attempt provenance hash disagrees with manifest"
                        )
                attempt_projection = {
                    "attempt_number": recorded["attempt_number"],
                    "attempt_id": source["attempt_id"],
                    "attempt_path": attempt.relative_to(root).as_posix(),
                    "owner_token_digest": recorded["owner_token_digest"],
                    "checkpoint_attempt_digest": recorded["canonical_digest"],
                    "state": source["state"],
                    "status": status_document.get("status"),
                    "terminal": status_document.get("terminal"),
                    "complete": status_document.get("complete"),
                    "status_sha256": source["status_sha256"],
                    "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
                    "status_manifest_sha256": status_document.get("manifest_sha256"),
                    "manifest_status": manifest_document.get("status"),
                    "manifest_terminal": manifest_document.get("terminal"),
                    "manifest_complete": manifest_document.get("complete"),
                    "attempt_inventory_digest": source["attempt_inventory_digest"],
                    "failure_class": source["failure_class"],
                    "failure_code": source["failure_code"],
                    "failure": _failure_projection(
                        manifest_document.get("failure"), source["state"],
                    ),
                    "retry_authorization_digest": recorded["retry_authorization_digest"],
                    "result_manifest_published": source["result_manifest_published"],
                    "published_l0_l1_payload_count": source["published_l0_l1_payload_count"],
                    "resources": source["resources"],
                    "artifact_count": len(artifacts),
                    "artifacts": artifacts,
                    "provenance": provenance_facts,
                }
                if attempt_projection["manifest_sha256"] != attempt_projection["status_manifest_sha256"]:
                    raise StageDFinalizationIntegrityError(
                        "attempt status-to-manifest binding drifted"
                    )
                source_attempts.append(attempt_projection)
                observed_attempts.append({"resources": source["resources"]})
            _validate_retry_history(
                row, source_attempts, unit=unit, context=context,
            )
            final_attempt = source_attempts[-1]
            source_paths = {
                artifact["path"] for artifact in final_attempt["artifacts"]
                if artifact["kind"] != "provenance"
            }
            accounted_paths = set(source_paths)
            failure = final_attempt["failure"]
            if isinstance(failure, Mapping):
                absent = failure.get("absent", [])
                if not isinstance(absent, list) or any(
                    not isinstance(path, str) for path in absent
                ):
                    raise StageDFinalizationIntegrityError(
                        "scientific failure absent-artifact coverage is invalid"
                    )
                accounted_paths.update(absent)
                for path in absent:
                    level = path.split("/", 1)[0]
                    if level not in retention_units:
                        raise StageDFinalizationIntegrityError(
                            "scientific failure absence has an unknown retention level"
                        )
            required_paths = {RESULT_PATH, MAP_PATH, CURVES_PATH}
            if unit.audit:
                required_paths.add(AUDIT_PATH)
            if not required_paths <= accounted_paths:
                raise StageDFinalizationIntegrityError(
                    "terminal attempt does not account for every scientific path"
                )
            expected_retained_units["L0"].add(unit.canonical_digest)
            if row["state"] == "succeeded":
                expected_retained_units["L1"].add(unit.canonical_digest)
                if unit.audit:
                    expected_retained_units["L2"].add(unit.canonical_digest)
                contract_primary_maps += 1
                contract_curve_bundles += 1
                contract_audit_maps += int(unit.audit)
            elif isinstance(failure, Mapping):
                for path in failure.get("produced", []):
                    level = path.split("/", 1)[0]
                    if level in expected_retained_units:
                        expected_retained_units[level].add(unit.canonical_digest)
                produced = set(failure.get("produced", []))
                contract_primary_maps += int(MAP_PATH in produced)
                contract_curve_bundles += int(CURVES_PATH in produced)
                contract_audit_maps += int(unit.audit and AUDIT_PATH in produced)
            covered_primary_maps += int(MAP_PATH in source_paths)
            covered_audit_maps += int(unit.audit and AUDIT_PATH in source_paths)
            covered_curve_bundles += int(CURVES_PATH in source_paths)
            if row["state"] not in {"succeeded", "scientific_failed"} or source_attempts[-1]["state"] != row["state"]:
                raise StageDFinalizationIntegrityError(
                    f"unit lacks one authoritative terminal attempt: {unit.canonical_digest}"
                )
            if _resource_sum([{"resources": item["resources"]} for item in source_attempts]) != row["resources"]:
                raise StageDFinalizationIntegrityError(
                    f"unit resource reconciliation failed for {unit.canonical_digest}"
                )
            unit_scientific: list[str] = []
            if row["state"] == "scientific_failed":
                if row["failure_class"] != "SCIENTIFIC_INVALIDITY" or row["failure_code"] not in _SCIENTIFIC_FAILURE_CODES:
                    raise StageDFinalizationIntegrityError(
                        "terminal scientific failure taxonomy drifted"
                    )
                unit_scientific.append(str(row["failure_code"]))
            else:
                validation = _scientific_validation(
                    attempts[-1], unit, root=root, source_index=source_index,
                    captured_bytes=captured_bytes,
                )
                validations[unit.canonical_digest] = validation
                if validation.errors:
                    raise StageDFinalizationIntegrityError(
                        "successful attempt fails source scientific recomputation: "
                        + "; ".join(validation.errors)
                    )
            projected = {
                "selection_order": group.selection_order, "sample_id": group.sample_id,
                "group_digest": group.canonical_digest, "pair_digest": group.pair_binding.canonical_digest,
                "unit_digest": unit.canonical_digest, "method_index": unit.method_index,
                "method_id": unit.method_id, "audit": unit.audit,
                "terminal_state": row["state"], "terminal": row["terminal"],
                "failure_class": row["failure_class"],
                "failure_code": row["failure_code"],
                "retry_authorization_digest": row["retry_authorization_digest"],
                "resources": row["resources"], "attempts": source_attempts,
                "scientific_errors": unit_scientific,
            }
            rows.append(projected)
            group_rows.append(projected)
        if [item["method_id"] for item in group_rows] != [NNS_METHOD_ID, MATCHED_METHOD_ID]:
            raise StageDFinalizationIntegrityError(
                "image-group method order drifted"
            )
        pair_validations = [validations.get(item["unit_digest"]) for item in group_rows]
        if all(item is not None for item in pair_validations):
            bindings = [item.pair_binding for item in pair_validations if item is not None]
            if any(binding is not None for binding in bindings):
                pair_errors = validate_stage_d_pair_bindings(bindings[0], bindings[1])
                if pair_errors:
                    raise StageDFinalizationIntegrityError(
                        "successful method pair fails shared-binding recomputation: "
                        + "; ".join(pair_errors)
                    )
    if len(rows) != 1000 or [row["selection_order"] for row in rows[::2]] != list(range(500)):
        raise StageDFinalizationIntegrityError("exact plan left join failed")
    for index in range(0, len(rows), 2):
        pair = rows[index:index + 2]
        if [item["method_id"] for item in pair] != [NNS_METHOD_ID, MATCHED_METHOD_ID] or len({item["pair_digest"] for item in pair}) != 1:
            raise StageDFinalizationIntegrityError(
                "image-group method pair binding drifted"
            )

    recomputed_resources = _resource_sum(
        [{"resources": row["resources"]} for row in rows]
    )
    if recomputed_resources != checkpoint["resources"]:
        raise StageDFinalizationIntegrityError(
            "checkpoint resource totals disagree with source attempts"
        )
    scientific_errors = [
        {"unit_digest": row["unit_digest"], "reasons": row["scientific_errors"]}
        for row in rows if row["scientific_errors"]
    ]
    source_failure_rows = [
        {
            "unit_digest": row["unit_digest"],
            "group_selection_order": row["selection_order"],
            "method_index": row["method_index"],
            "state": row["terminal_state"],
            "terminal": row["terminal"],
            "attempt_count": len(row["attempts"]),
            "failure_class": row["failure_class"],
            "failure_code": row["failure_code"],
        }
        for row in rows
    ]
    if source_failure_rows != checkpoint["failure_rows"]:
        raise StageDFinalizationIntegrityError(
            "checkpoint failure table differs from source-derived unit outcomes"
        )

    source_summary = _inventory_summary(source_records)
    source_inventory = _self_document(
        "stage_d_source_inventory",
        root_scope="entire_stage_d_root_excluding_final_subtree",
        files=source_records, file_count=source_summary["file_count"],
        total_bytes=source_summary["total_bytes"],
    )
    expected_attempt_count = sum(len(row["attempts"]) for row in checkpoint["units"])
    expected_provenance_count = sum(
        attempt["state"] in {"succeeded", "scientific_failed"}
        for row in checkpoint["units"] for attempt in row["attempts"]
    )
    expected_counts = {
        **_EXPECTED_COUNTS,
        "attempts": expected_attempt_count,
        "attempt_statuses": expected_attempt_count,
        "attempt_manifests": expected_attempt_count,
        "provenance_records": expected_provenance_count,
        "artifact_records": contract_artifact_records,
        "source_files": len(source_records),
    }
    observed_counts = {
        "registry_rows": len(groups), "image_groups": len(groups),
        "execution_units": len(rows),
        "primary_clean_maps": covered_primary_maps,
        "audit_repeat_clean_maps": covered_audit_maps,
        "map_work_items": (
            covered_primary_maps + covered_audit_maps
        ),
        "fidelity_curves": covered_curve_bundles * 10,
        "fidelity_curve_points": covered_curve_bundles * 10 * 21,
        "logical_work_items": (
            covered_primary_maps + covered_audit_maps
            + covered_curve_bundles * 10 * 21
        ),
        "attempts": sum(len(row["attempts"]) for row in rows),
        "attempt_statuses": observed_statuses,
        "attempt_manifests": observed_manifests,
        "provenance_records": observed_provenance,
        "artifact_records": sum(observed_artifact_kinds.values()),
        "source_files": len(source_records),
    }
    exact_control_count_names = (
        "registry_rows", "image_groups", "execution_units", "attempts",
        "attempt_statuses", "attempt_manifests", "provenance_records",
        "artifact_records", "source_files",
    )
    if any(
        observed_counts[name] != expected_counts[name]
        for name in exact_control_count_names
    ) or (
        contract_primary_maps != observed_counts["primary_clean_maps"]
        or contract_audit_maps != observed_counts["audit_repeat_clean_maps"]
        or contract_curve_bundles * 10 != observed_counts["fidelity_curves"]
    ):
        raise StageDFinalizationIntegrityError(
            "observed controls or accounted scientific artifacts drifted"
        )
    retention = {
        "expected": {
            "L0_unit_coverage": 1000, "L1_unit_coverage": 1000,
            "L2_unit_coverage": 10,
            "L3_file_count": 0,
        },
        "observed": {
            "L0_unit_coverage": len(retention_units["L0"]),
            "L1_unit_coverage": len(retention_units["L1"]),
            "L2_unit_coverage": len(retention_units["L2"]),
            "L3_file_count": retention_level_files["L3"],
            "files_by_level": {
                level: retention_level_files[level]
                for level in ("L0", "L1", "L2", "L3")
            },
            "file_unit_coverage": {
                level: len(retained_file_units[level])
                for level in ("L0", "L1", "L2", "L3")
            },
        },
        "audit_image_count": sum(group.audit for group in plan.groups),
        "audit_unit_count": sum(unit.audit for unit in plan.units),
    }
    if (
        retention["audit_image_count"] != 5
        or retention["audit_unit_count"] != 10
        or any(
            retention_units[level] != expected_retained_units[level]
            for level in ("L0", "L1", "L2")
        )
        or retention_units["L3"]
    ):
        raise StageDFinalizationIntegrityError(
            "retention coverage or audit-only/L3 policy drifted"
        )
    reconciliation = _self_document(
        "stage_d_reconciliation", plan_digest=plan.canonical_digest,
        expected_image_count=500, expected_unit_count=1000,
        image_count=500, unit_count=len(rows),
        expected_counts=expected_counts, observed_counts=observed_counts,
        state_counts=dict(sorted(Counter(row["terminal_state"] for row in rows).items())),
        group_rows=groups, rows=rows, failure_rows=source_failure_rows,
        na_rows=[
            {"unit_digest": row["unit_digest"], "reason": row["failure_code"]}
            for row in rows if row["terminal_state"] == "scientific_failed"
        ],
        resources=recomputed_resources, retention=retention,
        checkpoint_failure_table_reconciled=True,
        checkpoint_resources_reconciled=True, no_rows_dropped=True,
        zero_imputation_performed=False,
    )
    validity = "PASS" if not scientific_errors else "FAIL"
    evidence = _self_document(
        "stage_d_final_evidence_report", package_complete=True,
        scientific_validity=validity, workflow_status="PASS",
        plan_digest=plan.canonical_digest, estimate_digest=estimate.canonical_digest,
        preflight_digest=preflight.canonical_digest,
        execution_context_digest=context["runtime_binding_digest"],
        counts={
            "images": 500, "groups": 500, "units": 1000,
            "attempts": len(observed_attempts),
            "source_files": len(source_records),
            "scientific_failures": len(scientific_errors),
        },
        scientific_failures=scientific_errors,
        authorization={
            "formal": context["formal"],
            "execution_authorization": context["execution_authorization"],
            "authorization_scope": context["authorization_scope"],
            "analysis_authorization": preflight.analysis_authorization,
            "stage_d_execution_started": False,
        },
        analysis_boundary={
            "status": "NOT_PERFORMED", "analysis_performed": False,
            "bootstrap_performed": False, "decision_performed": False,
            "delta_fid_computed": False, "aggregate_effects_computed": False,
            "d6_eligibility": (
                "SYNTHETIC_ONLY" if context["formal"] is False
                else "NOT_AUTHORIZED"
            ),
        },
        analysis_performed=False, bootstrap_performed=False, decision_performed=False,
        reconciliation_digest=reconciliation["canonical_digest"],
        source_inventory_digest=source_inventory["canonical_digest"],
    )
    payload_values = {
        "evidence_report.json": _json_bytes(evidence),
        "reconciliation.json": _json_bytes(reconciliation),
        "source_inventory.json": _json_bytes(source_inventory),
    }
    payload_rows = [
        {"path": name, "bytes": len(payload_values[name]), "sha256": hashlib.sha256(payload_values[name]).hexdigest()}
        for name in FINAL_PAYLOAD_FILES[:-1]
    ]
    payload_manifest = _self_document(
        "stage_d_final_payload_manifest", immutable=True,
        artifact_count=len(payload_rows), artifacts=payload_rows,
    )
    payload_values["payload_manifest.json"] = _json_bytes(payload_manifest)
    sealed_root_inventory = _sealed_inventory(
        _inventory_with_overlay(source_records, payload_values)
    )
    status = _self_document(
        "stage_d_final_status", terminal=True, package_complete=True,
        scientific_validity=validity, workflow_status="PASS",
        plan_digest=plan.canonical_digest,
        payload_manifest_sha256=hashlib.sha256(payload_values["payload_manifest.json"]).hexdigest(),
        sealed_root_inventory=sealed_root_inventory,
    )
    payload_values[FINAL_STATUS_FILENAME] = _json_bytes(status)
    for name, document in (
        ("evidence_report.json", evidence),
        ("reconciliation.json", reconciliation),
        ("source_inventory.json", source_inventory),
        ("payload_manifest.json", payload_manifest),
        ("status.json", status),
    ):
        _validate_final_document(name, document)
    return payload_values, {
        "evidence_report": evidence, "reconciliation": reconciliation,
        "source_inventory": source_inventory, "payload_manifest": payload_manifest,
        "status": status,
    }


@dataclass(frozen=True, slots=True)
class StageDRootValidation(Mapping[str, Any]):
    package_complete: bool
    scientific_validity: str
    workflow_status: str
    plan_digest: str
    final_status_sha256: str
    root_inventory: Mapping[str, Any]
    evidence_report: Mapping[str, Any]
    reconciliation: Mapping[str, Any]
    source_inventory: Mapping[str, Any]
    payload_manifest: Mapping[str, Any]
    whole_root_ordered_inventory: tuple[Mapping[str, Any], ...]
    _data: Mapping[str, Any] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        mapping_fields = (
            "root_inventory", "evidence_report", "reconciliation",
            "source_inventory", "payload_manifest",
        )
        for name in mapping_fields:
            object.__setattr__(self, name, _freeze(_plain(getattr(self, name))))
        frozen_whole = tuple(
            _freeze(_plain(item)) for item in self.whole_root_ordered_inventory
        )
        object.__setattr__(self, "whole_root_ordered_inventory", frozen_whole)
        data = {name: getattr(self, name) for name in _RESULT_KEYS}
        object.__setattr__(self, "_data", MappingProxyType(data))

    def to_dict(self) -> dict[str, Any]:
        return _plain(self._data)

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)


def _result(
    documents: Mapping[str, Any], whole: Sequence[Mapping[str, Any]],
    status_raw: bytes,
) -> StageDRootValidation:
    status = documents["status"]
    whole = _plain(list(whole), "whole-root ordered inventory")
    status_path = f"{FINAL_DIRECTORY_NAME}/{FINAL_STATUS_FILENAME}"
    without_status = [row for row in whole if row["path"] != status_path]
    if (
        len(without_status) + 1 != len(whole)
        or status["sealed_root_inventory"] != _sealed_inventory(without_status)
    ):
        raise StageDFinalizationIntegrityError(
            "terminal status sealed-root inventory differs from verified bytes"
        )
    status_rows = [row for row in whole if row["path"] == status_path]
    if status_rows != [{
        "path": status_path, "bytes": len(status_raw),
        "sha256": hashlib.sha256(status_raw).hexdigest(),
    }]:
        raise StageDFinalizationIntegrityError(
            "terminal status bytes differ from the verified inventory"
        )
    summary = _inventory_summary(whole)
    evidence = documents["evidence_report"]
    if (
        status["package_complete"] is not True
        or status["scientific_validity"] != evidence["scientific_validity"]
        or status["workflow_status"] != evidence["workflow_status"]
        or status["plan_digest"] != evidence["plan_digest"]
    ):
        raise StageDFinalizationIntegrityError(
            "terminal status conclusion differs from the evidence report"
        )
    return StageDRootValidation(
        package_complete=True, scientific_validity=evidence["scientific_validity"],
        workflow_status=evidence["workflow_status"], plan_digest=evidence["plan_digest"],
        final_status_sha256=hashlib.sha256(status_raw).hexdigest(), root_inventory=summary,
        evidence_report=documents["evidence_report"],
        reconciliation=documents["reconciliation"],
        source_inventory=documents["source_inventory"],
        payload_manifest=documents["payload_manifest"],
        whole_root_ordered_inventory=tuple(whole),
    )


def finalize_stage_d_root(output_root: Path | str, plan: Any, estimate: Any, preflight: Any, *, allow_formal: bool = False) -> StageDRootValidation:
    """Seal one complete D4 root without executing or authorizing work."""

    if type(allow_formal) is not bool:
        raise TypeError("allow_formal must be bool")
    typed_plan, typed_estimate, typed_preflight = _validate_controls(plan, estimate, preflight)
    named_root = _safe_root(output_root)
    with _retained_root(named_root) as (root, root_descriptor, root_identity), _root_locks(root):
        _verify_root_binding(named_root, root_descriptor, root_identity)
        final = root / FINAL_DIRECTORY_NAME
        if os.path.lexists(final):
            metadata = os.lstat(final)
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                raise StageDFinalizationIntegrityError(
                    "final path is a forbidden symlink or non-directory"
                )
            raise StageDFinalizationBlockedError(
                "final directory has already been claimed and cannot be reused"
            )
        prechecked = _checkpoint_control(
            root, typed_plan, typed_estimate, typed_preflight,
        )
        _reject_active_checkpoint_control(prechecked)
        (
            source_records, _captured_whole, captured_bytes,
            captured_directories,
        ) = _source_snapshot(
            root, sealed=False, root_descriptor=root_descriptor,
        )
        prechecked = _checkpoint_control(
            root, typed_plan, typed_estimate, typed_preflight,
            raw_bytes=captured_bytes[_runtime.RUNTIME_CHECKPOINT_FILENAME],
        )
        _validate_source_directories(
            source_records, captured_directories, typed_plan, prechecked,
        )
        snapshot_owner, snapshot_root = _materialize_source_snapshot(
            source_records, captured_bytes, captured_directories,
        )
        try:
            payloads, documents = _build_source_documents(
                snapshot_root, typed_plan, typed_estimate, typed_preflight,
                allow_formal=allow_formal, sealed=False,
                source_records=source_records,
                prechecked_checkpoint=prechecked,
                captured_bytes=captured_bytes,
                logical_root=named_root,
            )
        finally:
            snapshot_owner.cleanup()
        staging_name, staging_descriptor, staging_identity = (
            _create_staging_directory(root_descriptor)
        )
        staging = root / staging_name
        published = False
        try:
            for name in FINAL_PAYLOAD_FILES:
                _atomic_create_at(staging_descriptor, name, payloads[name])
            # The terminal status is written last inside the unpublished tree.
            _atomic_create_at(
                staging_descriptor, FINAL_STATUS_FILENAME,
                payloads[FINAL_STATUS_FILENAME],
            )
            _fsync_directory(descriptor=staging_descriptor)
            _verify_root_binding(named_root, root_descriptor, root_identity)
            if _root_identity(staging) != staging_identity:
                raise StageDFinalizationIntegrityError(
                    "D5 staging directory identity changed before publication"
                )
            if os.path.lexists(final):
                metadata = os.lstat(final)
                if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                    raise StageDFinalizationIntegrityError(
                        "final path became a symlink or non-directory before publication"
                    )
                raise StageDFinalizationBlockedError(
                    "final directory was claimed before publication"
                )
            current_source, current_directories = _inventory_and_directories(
                root, exclude_final=True,
                exclude_top_level=frozenset({staging.name}),
                root_descriptor=root_descriptor,
            )
            if (
                current_source != source_records
                or current_directories != captured_directories
            ):
                raise StageDFinalizationIntegrityError(
                    "execution source changed before atomic final publication"
                )
            _publish_directory_no_replace(
                root_descriptor, staging_name, FINAL_DIRECTORY_NAME,
                staging_descriptor,
            )
            published = True
            _fsync_directory(descriptor=root_descriptor)
        finally:
            if not published:
                _cleanup_staging_directory(
                    root, staging_name, staging_identity,
                    root_descriptor=root_descriptor,
                )
            os.close(staging_descriptor)
        _verify_root_binding(named_root, root_descriptor, root_identity)
        _root_layout(root, typed_plan, sealed=True)
        for name in FINAL_FILES:
            actual = _read_regular(final / name, f"published final control {name}")
            if actual != payloads[name]:
                raise StageDFinalizationIntegrityError(
                    f"published final control differs from staged bytes: {name}"
                )
        current_source, current_directories = _inventory_and_directories(
            root, exclude_final=True, root_descriptor=root_descriptor,
        )
        if (
            current_source != source_records
            or current_directories != captured_directories
        ):
            raise StageDFinalizationIntegrityError(
                "execution source changed during final publication"
            )
        whole = _whole_inventory_now(root, root_descriptor=root_descriptor)
        expected_whole = _inventory_with_overlay(
            source_records, {name: payloads[name] for name in FINAL_PAYLOAD_FILES},
        )
        expected_whole.append({
            "path": f"{FINAL_DIRECTORY_NAME}/{FINAL_STATUS_FILENAME}",
            "bytes": len(payloads[FINAL_STATUS_FILENAME]),
            "sha256": hashlib.sha256(
                payloads[FINAL_STATUS_FILENAME]
            ).hexdigest(),
        })
        expected_whole.sort(key=lambda row: row["path"])
        if whole != expected_whole:
            raise StageDFinalizationIntegrityError(
                "published whole-root inventory differs from staged bytes"
            )
        result = _result(
            documents, whole, payloads[FINAL_STATUS_FILENAME],
        )
        _validate_root_validation_document(result.to_dict())
        return result


def validate_stage_d_root(output_root: Path | str, plan: Any, estimate: Any, preflight: Any, *, allow_formal: bool = False) -> StageDRootValidation:
    """Independently recompute and byte-compare an already sealed D5 root."""

    if type(allow_formal) is not bool:
        raise TypeError("allow_formal must be bool")
    typed_plan, typed_estimate, typed_preflight = _validate_controls(plan, estimate, preflight)
    named_root = _safe_root(output_root)
    with _retained_root(named_root) as (root, root_descriptor, root_identity), _root_locks(root):
        _verify_root_binding(named_root, root_descriptor, root_identity)
        prechecked = _checkpoint_control(
            root, typed_plan, typed_estimate, typed_preflight,
        )
        _reject_active_checkpoint_control(prechecked)
        (
            source_records, captured_whole, captured_bytes,
            captured_directories,
        ) = _source_snapshot(
            root, sealed=True, root_descriptor=root_descriptor,
        )
        prechecked = _checkpoint_control(
            root, typed_plan, typed_estimate, typed_preflight,
            raw_bytes=captured_bytes[_runtime.RUNTIME_CHECKPOINT_FILENAME],
        )
        _validate_source_directories(
            source_records, captured_directories, typed_plan, prechecked,
        )
        snapshot_owner, snapshot_root = _materialize_source_snapshot(
            source_records, captured_bytes, captured_directories,
        )
        try:
            expected, documents = _build_source_documents(
                snapshot_root, typed_plan, typed_estimate, typed_preflight,
                allow_formal=allow_formal, sealed=False,
                source_records=source_records,
                prechecked_checkpoint=prechecked,
                captured_bytes=captured_bytes,
                logical_root=named_root,
            )
        finally:
            snapshot_owner.cleanup()
        final = root / FINAL_DIRECTORY_NAME
        document_types = {
            "evidence_report.json": "stage_d_final_evidence_report",
            "reconciliation.json": "stage_d_reconciliation",
            "source_inventory.json": "stage_d_source_inventory",
            "payload_manifest.json": "stage_d_final_payload_manifest",
            "status.json": "stage_d_final_status",
        }
        for name in FINAL_FILES:
            relative = f"{FINAL_DIRECTORY_NAME}/{name}"
            try:
                actual = captured_bytes[relative]
            except KeyError as error:
                raise StageDFinalizationIntegrityError(
                    f"final control is absent from captured root: {name}"
                ) from error
            parsed = _strict_json(actual, f"final control {name}")
            _validate_final_document(name, parsed)
            if actual != expected[name]:
                raise StageDFinalizationIntegrityError(
                    f"final control differs from source recomputation: {name}"
                )
        _verify_root_binding(named_root, root_descriptor, root_identity)
        _root_layout(root, typed_plan, sealed=True)
        current_whole, current_directories = _inventory_and_directories(
            root, exclude_final=False, root_descriptor=root_descriptor,
        )
        if (
            current_whole != captured_whole
            or current_directories != captured_directories
        ):
            raise StageDFinalizationIntegrityError(
                "sealed root changed during independent validation"
            )
        result = _result(
            documents, captured_whole,
            captured_bytes[f"{FINAL_DIRECTORY_NAME}/{FINAL_STATUS_FILENAME}"],
        )
        _validate_root_validation_document(result.to_dict())
        return result


__all__ = [
    "FINAL_DIRECTORY_NAME", "FINAL_FILES", "FINAL_PAYLOAD_FILES",
    "FINAL_STATUS_FILENAME",
    "StageDFinalizationBlockedError", "StageDFinalizationError",
    "StageDFinalizationIntegrityError",
    "StageDFinalizeBlockedError", "StageDFinalizeError",
    "StageDFinalizeIntegrityError", "StageDRootValidation",
    "finalize_stage_d_root", "validate_stage_d_root",
]
