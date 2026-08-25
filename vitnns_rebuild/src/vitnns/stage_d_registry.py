"""Metadata-only Stage D D1 registry construction and validation.

This module deliberately has no image, model, attribution, or outcome imports.
It binds each allowed metadata input from one safely opened byte stream before
strict parsing, then deterministically constructs the tracked D1 package.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import re
import stat
import subprocess
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from .constants import REPO_ROOT


class StageDRegistryError(ValueError):
    """Raised when a D1 source or registry package fails closed."""


D0_COMPLETION_REF = "refs/archive/stage-d-d0-completion"
D0_COMPLETION_COMMIT = "4c76d468e1368858c52ff8ef73e640463e15ac49"
D0_SCOPE_CANONICAL_DIGEST = (
    "7508dcb7c063eaeecfc41f2f63f18e70dd5d660bec06c50e7b4b893d9a48ac5f"
)
D0_SCOPE_FILE_SHA256 = (
    "217ef1fc00c3dbe971b35ff8d0793fb1eb928a7316db4ac0a5bcd34edfb6c06d"
)
D0_SCOPE_SCHEMA_SHA256 = (
    "425f7fd16a1d004817dd2888f95d9b5b1818f7be6063bef522f05d438e6b5dfb"
)
D0_SCOPE_MARKDOWN_SHA256 = (
    "913aaf81f9c56c1bb71d0202384fa29e392af966ef49bd8e27e0ff4df1c8186a"
)

PACKAGE_ID = "STAGE_D_FIDELITY_V1"
PACKAGE_DIR = REPO_ROOT / "manifests/stage_d_fidelity_v1"
PACKAGE_SCHEMA_NAME = "stage_d_registry_package.schema.json"
PACKAGE_FILES = (
    PACKAGE_SCHEMA_NAME,
    "source_binding.json",
    "registry.json",
    "pre_authorization_access_attestation.json",
    "registry_metadata.json",
    "manifest.json",
)
MANIFEST_PAYLOADS = PACKAGE_FILES[:-1]
SCHEMA_REF_PREFIX = f"{PACKAGE_SCHEMA_NAME}#/$defs/"

AUDIT_NAMESPACE = "vitnns-stage-d-fidelity-v1|audit-overlay|registry-order-v1|"
AUDIT_ALGORITHM_ID = (
    "sha256_utf8_namespace_pipe_ordered_sample_ids_digest_pipe_"
    "zero_padded_selection_order_pipe_sample_id_raw_digest_ascending_v1"
)

SOURCE_FILE_SPECS: tuple[tuple[str, str, str, str, str], ...] = (
    (
        "manifests/stage_c_plus_v2/stage_d_v2_reserve.json",
        "1432183a0ce9c701e60cc28acc3596fe16bb253dfc013126bf3168b5af05099a",
        "2e66fa2426fb7c42caefe7907a6ef1ec9a42adb1788528d59af9b5354eb0a99f",
        "SELF_EXCLUDING_CANONICAL_DIGEST",
        "candidate 500-row order and recorded identity/label/image-hash metadata; no live-byte equality",
    ),
    (
        "manifests/stage_c_plus_v2/stage_d_v2_reserve_access_log.json",
        "75a319820db12102dde22308ab2e12196e0e1ca6fcdd870e2943d22a0643a621",
        "41478eee77f708a1158142d700fdfd65503b120aa672400af01fe14f9eb317fa",
        "SELF_EXCLUDING_CANONICAL_DIGEST",
        "candidate producer access-log assertions only; not an absolute historical-access proof",
    ),
    (
        "manifests/stage_c_plus_v2/cplus_registry.json",
        "b69fe4dd8d8e923c483a1b7ddd555098955bd905975fbfd52b14437cf89f0672",
        "21623d6142a9cbf6208a56e1f0ff743336e6c243c9448a05703521614361e536",
        "SELF_EXCLUDING_CANONICAL_DIGEST",
        "candidate first 70 selections used only to verify the declared 570-item ordering",
    ),
    (
        "manifests/stage_c_plus_v2/eligible_manifest.json",
        "bf3bc8066d3083ae032c6c260fda2c56b4f2027e234e3a797ec84cf6b9015f60",
        "7e39467dfaa5c41b9e118b202553256446f8a44320229db5ebdc50ef6087285b",
        "SELF_EXCLUDING_CANONICAL_DIGEST",
        "candidate producer's 50,000-row identity and official-label metadata; live assets not re-read",
    ),
    (
        "manifests/stage_c_plus_v2/prior_access_ledger.json",
        "00bf7ed27a85ad3da5dc16ab933e12d40a592186a7500ad66137126d933946cc",
        "d44322e1e9cc63452c159936f1e55cfd1459f97f82930243b02cd12b1ccfe5d2",
        "SELF_EXCLUDING_CANONICAL_DIGEST",
        "absence from declared closed prior-access records within their stated scope only",
    ),
    (
        "manifests/stage_c_plus_v2/sample_selection.json",
        "0ec3ebde8c24b4a2cee3485fafbd9b06f77abcd8a39b60aabd1a3ebd74aa0cec",
        "76deef80f27f1793a871aad02abd06977dc97fa0d0d796aa3a80be6e00dda7dd",
        "SELF_EXCLUDING_CANONICAL_DIGEST",
        "candidate selection counts, algorithm, embedded registries, and source bindings",
    ),
    (
        "manifests/stage_c_plus_v2/source_archive_identity.json",
        "876111f84c8a72cee06e5c8d89ecb9a109be8120140d146bd01363b779365d1f",
        "018b4e4edc900ed5f5adb23002a575270085b56e3bd6a6bcee4f17853c393645",
        "SELF_EXCLUDING_CANONICAL_DIGEST",
        "candidate producer's recorded official archive identity; archive bytes not read by D1",
    ),
    (
        "manifests/stage_c_plus_v2/source_inventory.json",
        "cc3fcb0d5579d646f5cc56c2bf7c5241cfbc475a9ecd39c578195d072d42ae67",
        "d3dfd4449b7f5c07328dd5edd687c9f8d59ebeb67872708ccb90cd59ea527725",
        "SELF_EXCLUDING_CANONICAL_DIGEST",
        "declared prior-access source inventory, including its completeness attestation",
    ),
    (
        "manifests/stage_c_plus_v2/source_spec.json",
        "4f34f5326b2259a529cfba0f5c9851ce7ed2c979be6ae8e8bd928e566a14ed66",
        "662714ea062fd1edfe14ac34704456faf7411af7387c81fc968b83d56a6e75b4",
        "SELF_EXCLUDING_CANONICAL_DIGEST",
        "user-scoped static historical boundary with explicit non-exhaustiveness limits",
    ),
    (
        "manifests/stage_c_sample_registry.csv",
        "9a6667bec6a27c715d272382a494f06924db881dca991dc4fac4751dbe7f37f0",
        "none",
        "NOT_JSON",
        "the frozen Stage C 20-row identity/path/image-hash set for three-way disjointness",
    ),
    (
        "manifests/stage_c_sample_registry_metadata.json",
        "bedb9df9a77807c86da0b898f8d44767c8276f21ab69821965f2b30d7184a4af",
        "3f1b3611bab17fe0b6786a0c16ba2fbfbeba9a4ce6c112ea48160cf9e742b36b",
        "FULL_DOCUMENT",
        "Stage C registry provenance and official-label metadata only",
    ),
)

SOURCE_JSON_PATHS = tuple(item[0] for item in SOURCE_FILE_SPECS if item[3] != "NOT_JSON")
SOURCE_CSV_PATH = "manifests/stage_c_sample_registry.csv"

STAGE_C_COLUMNS = (
    "schema_version", "sample_id", "image_id", "relative_path",
    "image_sha256", "size_bytes", "width", "height",
    "official_wnid", "imagenet1k_index", "label_source", "role",
    "audit", "selection_order", "source_selection_index",
    "selection_rule",
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_WNID_RE = re.compile(r"^n[0-9]{8}$")
_RELATIVE_IMAGE_RE = re.compile(r"^ILSVRC2012_val_[0-9]{8}\.JPEG$")
_SAMPLE_ID_RE = re.compile(
    r"^ILSVRC2012_val/ILSVRC2012_val_[0-9]{8}\.JPEG$"
)


@dataclass(frozen=True)
class BoundMetadata:
    relative_path: str
    size_bytes: int
    mtime_ns: int
    sha256: str
    payload: bytes
    document: Any | None


@dataclass(frozen=True)
class ValidatedSources:
    bindings: Mapping[str, BoundMetadata]
    documents: Mapping[str, Mapping[str, Any]]
    stage_c_rows: tuple[Mapping[str, str], ...]


def _validate_json_safe(value: Any, context: str = "JSON value") -> None:
    if value is None or type(value) in {str, bool, int}:
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise StageDRegistryError(f"{context} contains a non-finite number")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str) or not key:
                raise StageDRegistryError(f"{context} keys must be non-empty strings")
            _validate_json_safe(child, f"{context}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _validate_json_safe(child, f"{context}[{index}]")
        return
    raise StageDRegistryError(
        f"{context} contains non-JSON-safe {type(value).__name__}"
    )


def canonical_json_bytes(value: Any) -> bytes:
    _validate_json_safe(value)
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def canonical_file_bytes(value: Any) -> bytes:
    return canonical_json_bytes(value) + b"\n"


def semantic_digest(document: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {key: value for key, value in document.items() if key != "canonical_digest"}
    )


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise StageDRegistryError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite(token: str) -> None:
    raise StageDRegistryError(f"non-finite JSON number is forbidden: {token}")


def strict_json_bytes(payload: bytes, context: str) -> Any:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise StageDRegistryError(f"{context} is not UTF-8") from error
    try:
        value = json.loads(
            text, object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except json.JSONDecodeError as error:
        raise StageDRegistryError(f"{context} is invalid JSON: {error}") from error
    _validate_json_safe(value, context)
    return value


def _safe_open_directory(path: Path | str) -> int:
    absolute = Path(os.path.abspath(os.fspath(path)))
    if not absolute.is_absolute():
        raise StageDRegistryError("metadata root must be absolute")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    current = os.open(absolute.anchor, flags)
    try:
        for part in absolute.parts[1:]:
            try:
                next_fd = os.open(part, flags, dir_fd=current)
            except OSError as error:
                raise StageDRegistryError(
                    f"metadata root has a missing, symlink, or unsafe component: {part}"
                ) from error
            os.close(current)
            current = next_fd
        if not stat.S_ISDIR(os.fstat(current).st_mode):
            raise StageDRegistryError("metadata root is not a directory")
        return current
    except BaseException:
        os.close(current)
        raise


def _safe_read_relative(root_fd: int, relative_path: str) -> BoundMetadata:
    pure = PurePosixPath(relative_path)
    if (
        pure.is_absolute() or not pure.parts or ".." in pure.parts
        or "." in pure.parts or any(not part for part in pure.parts)
    ):
        raise StageDRegistryError(f"unsafe metadata path: {relative_path}")
    directory_fd = os.dup(root_fd)
    try:
        for component in pure.parts[:-1]:
            flags = (
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0)
            )
            try:
                next_fd = os.open(component, flags, dir_fd=directory_fd)
            except OSError as error:
                raise StageDRegistryError(
                    f"metadata path has a missing, symlink, or unsafe directory: {relative_path}"
                ) from error
            os.close(directory_fd)
            directory_fd = next_fd
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        try:
            fd = os.open(pure.parts[-1], flags, dir_fd=directory_fd)
        except OSError as error:
            raise StageDRegistryError(
                f"cannot safely open metadata file: {relative_path}"
            ) from error
        try:
            before = os.fstat(fd)
            if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
                raise StageDRegistryError(
                    f"metadata input is not a single-link regular file: {relative_path}"
                )
            chunks: list[bytes] = []
            while True:
                chunk = os.read(fd, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            payload = b"".join(chunks)
            after = os.fstat(fd)
        finally:
            os.close(fd)
    finally:
        os.close(directory_fd)
    identity = lambda value: (
        value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_nlink
    )
    if identity(before) != identity(after) or len(payload) != before.st_size:
        raise StageDRegistryError(f"metadata changed while being read: {relative_path}")
    return BoundMetadata(
        relative_path=relative_path, size_bytes=before.st_size,
        mtime_ns=before.st_mtime_ns, sha256=hashlib.sha256(payload).hexdigest(),
        payload=payload, document=None,
    )


def _bind_source_files(repo_root: Path | str) -> dict[str, BoundMetadata]:
    root_fd = _safe_open_directory(repo_root)
    try:
        raw = {
            path: _safe_read_relative(root_fd, path)
            for path, *_ in SOURCE_FILE_SPECS
        }
    finally:
        os.close(root_fd)
    expected = {path: sha for path, sha, *_ in SOURCE_FILE_SPECS}
    result: dict[str, BoundMetadata] = {}
    for path, bound in raw.items():
        if bound.sha256 != expected[path]:
            raise StageDRegistryError(
                f"source metadata file SHA-256 drifted: {path}; "
                f"expected {expected[path]}, observed {bound.sha256}"
            )
        document = None
        if path in SOURCE_JSON_PATHS:
            document = strict_json_bytes(bound.payload, path)
            if not isinstance(document, Mapping):
                raise StageDRegistryError(f"source JSON must be an object: {path}")
        result[path] = BoundMetadata(
            relative_path=bound.relative_path, size_bytes=bound.size_bytes,
            mtime_ns=bound.mtime_ns, sha256=bound.sha256, payload=bound.payload,
            document=document,
        )
    return result


def _exact(value: Any, keys: Sequence[str], context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise StageDRegistryError(f"{context} must be an object")
    expected = set(keys)
    actual = set(value)
    if actual != expected:
        raise StageDRegistryError(
            f"{context} fields drifted: missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}"
        )
    return value


def _list(value: Any, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise StageDRegistryError(f"{context} must be an array")
    return value


def _integer(value: Any, context: str, minimum: int | None = None, maximum: int | None = None) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise StageDRegistryError(f"{context} must be an integer")
    if minimum is not None and value < minimum:
        raise StageDRegistryError(f"{context} is below its minimum")
    if maximum is not None and value > maximum:
        raise StageDRegistryError(f"{context} is above its maximum")
    return value


def _digest(value: Any, context: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise StageDRegistryError(f"{context} must be a lowercase SHA-256")
    return value


def _check_self_digest(document: Mapping[str, Any], context: str) -> None:
    observed = _digest(document.get("canonical_digest"), f"{context}.canonical_digest")
    expected = semantic_digest(document)
    if observed != expected:
        raise StageDRegistryError(
            f"{context} canonical digest drifted: expected {expected}, observed {observed}"
        )


_SOURCE_ROW_KEYS = (
    "canonical_digest", "height", "image_format", "image_id",
    "image_sha256", "imagenet1k_index", "label_source", "mode",
    "official_ilsvrc2012_class_id", "official_wnid", "relative_path",
    "sample_id", "schema_version", "size_bytes",
    "validation_image_number", "width",
)
_CANDIDATE_ROW_KEYS = _SOURCE_ROW_KEYS + (
    "audit_overlay", "role", "selection_order", "selection_sha256",
)
_REGISTRY_SOURCE_KEYS = (
    "bindings", "canonical_digest", "document_type",
    "metadata_only_after_sealing", "ordered_image_sha256s_sha256",
    "ordered_sample_ids_sha256", "ordered_selection_sha256s_digest",
    "role", "rows", "rows_canonical_digest", "sample_count",
    "schema_version", "sealed", "selection_algorithm", "status",
)
_REGISTRY_BINDING_KEYS = (
    "eligible_manifest_canonical_digest", "eligible_manifest_file_sha256",
    "eligible_manifest_ordered_sample_ids_sha256",
    "prior_access_ledger_canonical_digest", "prior_access_ledger_file_sha256",
    "prior_access_ledger_ordered_sample_ids_sha256",
)


def _validate_source_row(
    row: Any, context: str, *, validate_self_digest: bool = True,
) -> Mapping[str, Any]:
    row = _exact(row, _SOURCE_ROW_KEYS, context)
    if row["schema_version"] != 1:
        raise StageDRegistryError(f"{context}.schema_version drifted")
    if validate_self_digest:
        _check_self_digest(row, context)
    relative = row["relative_path"]
    sample_id = row["sample_id"]
    image_id = row["image_id"]
    if not isinstance(relative, str) or _RELATIVE_IMAGE_RE.fullmatch(relative) is None:
        raise StageDRegistryError(f"{context}.relative_path is unsafe or non-canonical")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or len(pure.parts) != 1 or ".." in pure.parts:
        raise StageDRegistryError(f"{context}.relative_path escapes the flat namespace")
    if not isinstance(sample_id, str) or _SAMPLE_ID_RE.fullmatch(sample_id) is None:
        raise StageDRegistryError(f"{context}.sample_id is invalid")
    if sample_id != f"ILSVRC2012_val/{relative}" or relative != f"{image_id}.JPEG":
        raise StageDRegistryError(f"{context} identity/path binding drifted")
    number = _integer(row["validation_image_number"], f"{context}.validation_image_number", 1, 50000)
    if image_id != f"ILSVRC2012_val_{number:08d}":
        raise StageDRegistryError(f"{context} image number binding drifted")
    _digest(row["image_sha256"], f"{context}.image_sha256")
    _integer(row["size_bytes"], f"{context}.size_bytes", 1)
    _integer(row["width"], f"{context}.width", 1)
    _integer(row["height"], f"{context}.height", 1)
    _integer(row["official_ilsvrc2012_class_id"], f"{context}.official_ilsvrc2012_class_id", 1, 1000)
    _integer(row["imagenet1k_index"], f"{context}.imagenet1k_index", 0, 999)
    if not isinstance(row["official_wnid"], str) or _WNID_RE.fullmatch(row["official_wnid"]) is None:
        raise StageDRegistryError(f"{context}.official_wnid is invalid")
    if row["label_source"] != "official_devkit_val_gt":
        raise StageDRegistryError(f"{context}.label_source drifted")
    if row["image_format"] != "JPEG" or row["mode"] not in {"RGB", "L", "CMYK"}:
        raise StageDRegistryError(f"{context} image metadata is invalid")
    return row


def _validate_candidate_row(row: Any, context: str, role: str) -> Mapping[str, Any]:
    row = _exact(row, _CANDIDATE_ROW_KEYS, context)
    # Candidate rows have a digest over the complete candidate record.  Reuse
    # the base identity/label checks without interpreting that digest as the
    # eligible-manifest row digest, then verify the full record below.
    _validate_source_row(
        {key: row[key] for key in _SOURCE_ROW_KEYS}, context,
        validate_self_digest=False,
    )
    if row["role"] != role or not isinstance(row["audit_overlay"], bool):
        raise StageDRegistryError(f"{context} role/audit metadata drifted")
    _integer(row["selection_order"], f"{context}.selection_order", 0)
    _digest(row["selection_sha256"], f"{context}.selection_sha256")
    _check_self_digest(row, context)
    return row


def _validate_registry_source(
    document: Mapping[str, Any], *, context: str, count: int, role: str,
    document_type: str, status: str, sealed: bool, metadata_only: bool,
) -> list[Mapping[str, Any]]:
    document = _exact(document, _REGISTRY_SOURCE_KEYS, context)
    if (
        document["schema_version"] != 1
        or document["document_type"] != document_type
        or document["status"] != status
        or document["role"] != role
        or document["sealed"] is not sealed
        or document["metadata_only_after_sealing"] is not metadata_only
    ):
        raise StageDRegistryError(f"{context} identity/status drifted")
    _exact(document["bindings"], _REGISTRY_BINDING_KEYS, f"{context}.bindings")
    algorithm = _exact(
        document["selection_algorithm"],
        ("algorithm_id", "namespace", "primary_order", "tie_breaker"),
        f"{context}.selection_algorithm",
    )
    expected_algorithm = (
        "sha256_utf8_vitnns_cplus_v2_20260824_relative_path_pipe_"
        "image_sha256_bytewise_ascending_then_relative_path_utf8_ascending_v1"
    )
    if algorithm != {
        "algorithm_id": expected_algorithm,
        "namespace": "vitnns-cplus-v2|20260824|",
        "primary_order": "selection_sha256 raw bytes ascending",
        "tie_breaker": "relative_path UTF-8 bytes ascending",
    }:
        raise StageDRegistryError(f"{context} selection algorithm drifted")
    rows = _list(document["rows"], f"{context}.rows")
    if document["sample_count"] != count or len(rows) != count:
        raise StageDRegistryError(f"{context} row count drifted")
    validated = [
        _validate_candidate_row(row, f"{context}.rows[{index}]", role)
        for index, row in enumerate(rows)
    ]
    if [row["selection_order"] for row in validated] != list(range(count)):
        raise StageDRegistryError(f"{context} selection_order is not contiguous")
    for field in ("sample_id", "image_id", "relative_path", "image_sha256", "selection_sha256"):
        if len({row[field] for row in validated}) != count:
            raise StageDRegistryError(f"{context}.{field} is not unique")
    for row in validated:
        expected = hashlib.sha256(
            (algorithm["namespace"] + row["relative_path"] + "|" + row["image_sha256"]).encode("utf-8")
        ).hexdigest()
        if row["selection_sha256"] != expected:
            raise StageDRegistryError(f"{context} selection digest drifted")
    ordered = sorted(
        validated, key=lambda row: (bytes.fromhex(row["selection_sha256"]), row["relative_path"].encode("utf-8")),
    )
    if validated != ordered:
        raise StageDRegistryError(f"{context} declared selection order drifted")
    expected_digests = {
        "rows_canonical_digest": canonical_sha256({"rows": validated}),
        "ordered_sample_ids_sha256": canonical_sha256([row["sample_id"] for row in validated]),
        "ordered_image_sha256s_sha256": canonical_sha256({"ordered_image_sha256s": [row["image_sha256"] for row in validated]}),
        "ordered_selection_sha256s_digest": canonical_sha256({"ordered_selection_sha256s": [row["selection_sha256"] for row in validated]}),
    }
    if any(document[key] != value for key, value in expected_digests.items()):
        raise StageDRegistryError(f"{context} ordered/row digest drifted")
    _check_self_digest(document, context)
    return validated


def _validate_archive_identity(document: Mapping[str, Any], context: str) -> None:
    _exact(document, (
        "canonical_digest", "document_type", "filename", "md5",
        "member_count", "member_policy", "ordered_member_files_sha256",
        "ordered_member_image_sha256s_sha256", "ordered_member_names_sha256",
        "schema_version", "sha256", "size_bytes", "source_url",
        "status", "validation_id_bijective", "validation_id_max",
        "validation_id_min",
    ), context)
    if (
        document["schema_version"] != 1
        or document["document_type"] != "ilsvrc2012_official_validation_archive_identity"
        or document["member_count"] != 50000
        or document["validation_id_min"] != 1
        or document["validation_id_max"] != 50000
        or document["validation_id_bijective"] is not True
        or document["status"] != "OFFICIAL_ARCHIVE_VALIDATED"
    ):
        raise StageDRegistryError(f"{context} semantics drifted")
    for key in ("sha256", "ordered_member_files_sha256", "ordered_member_image_sha256s_sha256", "ordered_member_names_sha256"):
        _digest(document[key], f"{context}.{key}")
    _check_self_digest(document, context)


def _validate_eligible_manifest(document: Mapping[str, Any], archive: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    context = "eligible_manifest"
    _exact(document, (
        "canonical_digest", "capacity_sufficient_before_exclusions",
        "data_root", "diagnostic_only", "document_type",
        "duplicate_content_alias_count", "duplicate_content_group_count",
        "frame_scope", "identity_collision_reasons", "mode_counts",
        "official_devkit", "ordered_image_sha256s_sha256",
        "ordered_sample_ids_sha256", "required_disjoint_count", "rows",
        "rows_canonical_digest", "sample_count", "scan_policy",
        "schema_version", "source_archive", "status",
        "unique_image_content_count",
    ), context)
    if (
        document["schema_version"] != 1
        or document["document_type"] != "stage_c_plus_v2_eligible_manifest"
        or document["status"] != "ELIGIBLE_FRAME_VALIDATED"
        or document["sample_count"] != 50000
        or document["required_disjoint_count"] != 570
        or document["diagnostic_only"] is not False
    ):
        raise StageDRegistryError(f"{context} identity/count drifted")
    mode_counts = _list(document["mode_counts"], f"{context}.mode_counts")
    for index, value in enumerate(mode_counts):
        _exact(value, ("count", "mode"), f"{context}.mode_counts[{index}]")
    official = _exact(document["official_devkit"], (
        "canonical_digest", "devkit_sha256", "label_source",
        "leaf_class_count", "meta_mat_sha256", "synset_order_sha256",
        "validation_ground_truth_sha256", "validation_identity_count",
    ), f"{context}.official_devkit")
    if official["label_source"] != "official_devkit_val_gt" or official["leaf_class_count"] != 1000 or official["validation_identity_count"] != 50000:
        raise StageDRegistryError(f"{context} official label chain drifted")
    for key in ("devkit_sha256", "meta_mat_sha256", "synset_order_sha256", "validation_ground_truth_sha256"):
        _digest(official[key], f"{context}.official_devkit.{key}")
    _check_self_digest(official, f"{context}.official_devkit")
    scan = _exact(document["scan_policy"], ("decode", "display_access", "gpu_access", "model_access", "safe_read"), f"{context}.scan_policy")
    if any(scan[key] is not False for key in ("display_access", "gpu_access", "model_access")):
        raise StageDRegistryError(f"{context} source scan policy drifted")
    source_archive = _exact(document["source_archive"], ("bound", "identity", "identity_file_sha256", "required_for_formal_frame"), f"{context}.source_archive")
    if source_archive["identity"] != archive or source_archive["bound"] is not True or source_archive["required_for_formal_frame"] is not True:
        raise StageDRegistryError(f"{context} archive binding drifted")
    rows = _list(document["rows"], f"{context}.rows")
    if len(rows) != 50000:
        raise StageDRegistryError(f"{context} must contain 50,000 rows")
    validated = [_validate_source_row(row, f"{context}.rows[{index}]") for index, row in enumerate(rows)]
    if [row["validation_image_number"] for row in validated] != list(range(1, 50001)):
        raise StageDRegistryError(f"{context} official validation order drifted")
    for field in ("sample_id", "image_id", "relative_path"):
        if len({row[field] for row in validated}) != 50000:
            raise StageDRegistryError(f"{context}.{field} is not unique")
    if len({row["image_sha256"] for row in validated}) != document["unique_image_content_count"]:
        raise StageDRegistryError(f"{context} unique image-content count drifted")
    if document["rows_canonical_digest"] != canonical_sha256({"rows": validated}):
        raise StageDRegistryError(f"{context} row digest drifted")
    if document["ordered_sample_ids_sha256"] != canonical_sha256([row["sample_id"] for row in validated]):
        raise StageDRegistryError(f"{context} ordered sample digest drifted")
    if document["ordered_image_sha256s_sha256"] != canonical_sha256({"ordered_image_sha256s": [row["image_sha256"] for row in validated]}):
        raise StageDRegistryError(f"{context} ordered image digest drifted")
    if Counter(row["mode"] for row in validated) != Counter({item["mode"]: item["count"] for item in mode_counts}):
        raise StageDRegistryError(f"{context} mode counts drifted")
    if document["duplicate_content_alias_count"] != 50000 - document["unique_image_content_count"]:
        raise StageDRegistryError(f"{context} duplicate-content count drifted")
    _check_self_digest(document, context)
    return validated


_EVIDENCE_FILE_KEYS = ("bytes", "path", "sha256")
_ACCESS_RECORD_KEYS = (
    "access_kinds", "canonical_digest", "evidence_member",
    "evidence_member_sha256", "evidence_path", "evidence_ref",
    "evidence_sha256", "identity_resolution", "image_sha256",
    "relative_path", "sample_id", "source_id", "source_image_sha256",
    "validation_image_number",
)


def _validate_evidence_files(values: Any, context: str) -> None:
    for index, value in enumerate(_list(values, context)):
        value = _exact(value, _EVIDENCE_FILE_KEYS, f"{context}[{index}]")
        _integer(value["bytes"], f"{context}[{index}].bytes", 1)
        _digest(value["sha256"], f"{context}[{index}].sha256")
        if not isinstance(value["path"], str) or not value["path"]:
            raise StageDRegistryError(f"{context}[{index}].path is invalid")


def _validate_source_spec(document: Mapping[str, Any]) -> None:
    context = "source_spec"
    _exact(document, (
        "canonical_digest", "classification_policy", "document_type",
        "inventory_complete", "schema_version", "scope",
        "user_scoped_historical_boundary",
    ), context)
    boundary = _exact(document["user_scoped_historical_boundary"], (
        "excluded_legacy_roots", "historical_directory_byte_audit_performed",
        "historical_directory_exhaustiveness_claimed",
        "known_validation_image_numbers", "known_validation_image_numbers_sha256",
        "membership_basis", "scope_attestation", "static_evidence_path",
        "static_source_id",
    ), f"{context}.user_scoped_historical_boundary")
    numbers = _list(boundary["known_validation_image_numbers"], f"{context}.known_validation_image_numbers")
    if len(numbers) != 698 or len(set(numbers)) != 698 or numbers != sorted(numbers):
        raise StageDRegistryError(f"{context} known ID set drifted")
    for index, value in enumerate(numbers):
        _integer(value, f"{context}.known_validation_image_numbers[{index}]", 1, 50000)
    if (
        document["schema_version"] != 1
        or document["document_type"] != "stage_c_plus_v2_prior_access_source_spec"
        or document["inventory_complete"] is not True
        or boundary["historical_directory_byte_audit_performed"] is not False
        or boundary["historical_directory_exhaustiveness_claimed"] is not False
        or boundary["membership_basis"] != "STATIC_CONSERVATIVE_SUPERSET"
    ):
        raise StageDRegistryError(f"{context} scope boundary drifted")
    _digest(boundary["known_validation_image_numbers_sha256"], f"{context}.known_validation_image_numbers_sha256")
    _check_self_digest(document, context)


def _validate_source_inventory(document: Mapping[str, Any], spec: Mapping[str, Any]) -> None:
    context = "source_inventory"
    _exact(document, (
        "builder_algorithm_id", "canonical_digest", "document_type",
        "eligible_manifest_canonical_digest", "eligible_manifest_file_sha256",
        "inventory_complete", "schema_version", "scope",
        "source_spec_binding", "sources", "workspace_root",
    ), context)
    binding = _exact(document["source_spec_binding"], ("canonical_digest", "file_sha256"), f"{context}.source_spec_binding")
    if binding["canonical_digest"] != spec["canonical_digest"] or binding["file_sha256"] != dict((p, s) for p, s, *_ in SOURCE_FILE_SPECS)["manifests/stage_c_plus_v2/source_spec.json"]:
        raise StageDRegistryError(f"{context} source-spec binding drifted")
    sources = _list(document["sources"], f"{context}.sources")
    if len(sources) != 1:
        raise StageDRegistryError(f"{context} source count drifted")
    source = _exact(sources[0], (
        "access_kinds", "evidence_digest_algorithm", "evidence_file_count",
        "evidence_files", "evidence_kind", "evidence_path",
        "evidence_sha256", "parser_mode", "records",
        "resolution_status", "source_id", "source_kind",
        "unresolved_reason",
    ), f"{context}.sources[0]")
    _validate_evidence_files(source["evidence_files"], f"{context}.sources[0].evidence_files")
    records = _list(source["records"], f"{context}.sources[0].records")
    if len(records) != 698:
        raise StageDRegistryError(f"{context} record count drifted")
    for index, record in enumerate(records):
        _exact(record, tuple(key for key in _ACCESS_RECORD_KEYS if key not in {"canonical_digest", "source_id", "source_image_sha256"}), f"{context}.sources[0].records[{index}]")
    expected_numbers = spec["user_scoped_historical_boundary"]["known_validation_image_numbers"]
    if [record["validation_image_number"] for record in records] != expected_numbers:
        raise StageDRegistryError(f"{context} records differ from the source spec")
    if document["inventory_complete"] is not True or source["resolution_status"] != "RESOLVED":
        raise StageDRegistryError(f"{context} is not closed/resolved")
    _check_self_digest(document, context)


def _validate_prior_ledger(
    document: Mapping[str, Any], eligible_rows: Sequence[Mapping[str, Any]],
    inventory: Mapping[str, Any], spec: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    context = "prior_access_ledger"
    _exact(document, (
        "accessed_eligible_count", "bindings", "canonical_digest",
        "closed_contract", "content_equivalent_prior_access_excluded_count",
        "coverage_rows", "coverage_rows_canonical_digest",
        "document_type", "duplicate_alias_excluded_count",
        "duplicate_content_alias_count", "duplicate_content_group_count",
        "duplicate_content_groups", "duplicate_content_policy",
        "eligible_row_count", "external_access_records",
        "known_access_record_count", "ordered_sample_ids_sha256",
        "resolved_access_records", "schema_version",
        "selection_eligible_count", "source_count", "source_evidence",
        "status", "unresolved", "unresolved_access_records",
        "unseen_eligible_count",
    ), context)
    bindings = _exact(document["bindings"], (
        "eligible_manifest_canonical_digest", "eligible_manifest_file_sha256",
        "eligible_ordered_sample_ids_sha256", "historical_stage_c_registry",
        "source_inventory_canonical_digest", "source_inventory_file_sha256",
        "source_spec_binding",
    ), f"{context}.bindings")
    historical = _exact(bindings["historical_stage_c_registry"], (
        "evidence_digest_algorithm", "evidence_file_count", "evidence_files",
        "evidence_kind", "evidence_path", "evidence_sha256",
        "record_count", "status",
    ), f"{context}.bindings.historical_stage_c_registry")
    _validate_evidence_files(historical["evidence_files"], f"{context}.historical.evidence_files")
    _exact(bindings["source_spec_binding"], ("canonical_digest", "file_sha256"), f"{context}.bindings.source_spec_binding")
    closed = _exact(document["closed_contract"], (
        "all_sources_resolved", "eligible_identity_coverage_complete",
        "historical_stage_c_1_0_included", "scope",
        "source_inventory_attested_complete", "unresolved_count",
    ), f"{context}.closed_contract")
    if closed != {
        "all_sources_resolved": True,
        "eligible_identity_coverage_complete": True,
        "historical_stage_c_1_0_included": True,
        "scope": "all_pre_cplus_model_output_attribution_statistic_result_and_display_access",
        "source_inventory_attested_complete": True,
        "unresolved_count": 0,
    }:
        raise StageDRegistryError(f"{context} closed-contract semantics drifted")
    coverage = _list(document["coverage_rows"], f"{context}.coverage_rows")
    if len(coverage) != 50000 or document["eligible_row_count"] != 50000:
        raise StageDRegistryError(f"{context} coverage is incomplete")
    coverage_keys = (
        "access_kinds", "access_record_digests", "canonical_digest",
        "content_canonical_representative_relative_path", "image_sha256",
        "prior_access_status", "relative_path", "sample_id",
        "selection_eligibility", "selection_exclusion_reason", "source_ids",
    )
    for index, row in enumerate(coverage):
        _exact(row, coverage_keys, f"{context}.coverage_rows[{index}]")
        _check_self_digest(row, f"{context}.coverage_rows[{index}]")
        for key in ("access_kinds", "access_record_digests", "source_ids"):
            _list(row[key], f"{context}.coverage_rows[{index}].{key}")
    if document["coverage_rows_canonical_digest"] != canonical_sha256({"rows": coverage}):
        raise StageDRegistryError(f"{context} coverage digest drifted")
    if document["ordered_sample_ids_sha256"] != canonical_sha256([row["sample_id"] for row in coverage]):
        raise StageDRegistryError(f"{context} ordered sample digest drifted")
    for source, row in zip(eligible_rows, coverage):
        if (source["sample_id"], source["relative_path"], source["image_sha256"]) != (row["sample_id"], row["relative_path"], row["image_sha256"]):
            raise StageDRegistryError(f"{context} coverage/eligible identity drifted")
    resolved = _list(document["resolved_access_records"], f"{context}.resolved_access_records")
    if len(resolved) != document["known_access_record_count"]:
        raise StageDRegistryError(f"{context} resolved record count drifted")
    for index, record in enumerate(resolved):
        _exact(record, _ACCESS_RECORD_KEYS, f"{context}.resolved_access_records[{index}]")
        _check_self_digest(record, f"{context}.resolved_access_records[{index}]")
    groups = _list(document["duplicate_content_groups"], f"{context}.duplicate_content_groups")
    for index, group in enumerate(groups):
        _exact(group, ("canonical_representative_relative_path", "image_sha256", "member_sample_ids", "prior_access_excludes_entire_class"), f"{context}.duplicate_content_groups[{index}]")
    _exact(document["duplicate_content_policy"], ("identity_key", "prior_access_propagation", "unseen_representative"), f"{context}.duplicate_content_policy")
    for key in ("external_access_records", "unresolved", "unresolved_access_records"):
        if document[key] != []:
            raise StageDRegistryError(f"{context}.{key} must be empty")
    evidence = _list(document["source_evidence"], f"{context}.source_evidence")
    if len(evidence) != document["source_count"] or not evidence:
        raise StageDRegistryError(f"{context} source-evidence count drifted")
    for index, source in enumerate(evidence):
        _exact(source, (
            "access_kinds", "evidence_digest_algorithm", "evidence_file_count",
            "evidence_files", "evidence_kind", "evidence_path",
            "evidence_sha256", "resolution_status", "source_id",
            "source_kind", "unresolved_reason",
        ), f"{context}.source_evidence[{index}]")
        _validate_evidence_files(source["evidence_files"], f"{context}.source_evidence[{index}].evidence_files")
    if (
        document["schema_version"] != 1
        or document["document_type"] != "stage_c_plus_v2_prior_access_ledger"
        or document["status"] != "PRIOR_ACCESS_LEDGER_CLOSED"
        or len(evidence) != document["source_count"]
        or bindings["source_inventory_canonical_digest"] != inventory["canonical_digest"]
        or bindings["source_spec_binding"]["canonical_digest"] != spec["canonical_digest"]
    ):
        raise StageDRegistryError(f"{context} source/status binding drifted")
    _check_self_digest(document, context)
    result = {row["sample_id"]: row for row in coverage}
    if len(result) != 50000:
        raise StageDRegistryError(f"{context} sample IDs are not unique")
    return result


def _validate_access_log(document: Mapping[str, Any], reserve: Mapping[str, Any]) -> None:
    context = "reserve_access_log"
    _exact(document, (
        "canonical_digest", "document_type", "entries",
        "forbidden_access_detected", "policy", "reserve_binding",
        "schema_version", "status",
    ), context)
    entries = _list(document["entries"], f"{context}.entries")
    for index, entry in enumerate(entries):
        _exact(entry, ("access_class", "actor", "canonical_digest", "operation", "purpose", "sequence"), f"{context}.entries[{index}]")
        _check_self_digest(entry, f"{context}.entries[{index}]")
    policy = _exact(document["policy"], ("allowed_operations", "default", "forbidden_access_classes", "image_bytes_allowed", "result_access_allowed"), f"{context}.policy")
    binding = _exact(document["reserve_binding"], ("canonical_digest", "file_sha256", "ordered_sample_ids_sha256"), f"{context}.reserve_binding")
    if binding != {
        "canonical_digest": reserve["canonical_digest"],
        "file_sha256": dict((p, s) for p, s, *_ in SOURCE_FILE_SPECS)["manifests/stage_c_plus_v2/stage_d_v2_reserve.json"],
        "ordered_sample_ids_sha256": reserve["ordered_sample_ids_sha256"],
    }:
        raise StageDRegistryError(f"{context} reserve binding drifted")
    if (
        document["schema_version"] != 1
        or document["document_type"] != "stage_d_v2_reserve_access_log"
        or document["status"] != "RESERVE_SEALED_METADATA_ONLY"
        or document["forbidden_access_detected"] is not False
        or policy["default"] != "DENY"
        or policy["image_bytes_allowed"] is not False
        or policy["result_access_allowed"] is not False
    ):
        raise StageDRegistryError(f"{context} access boundary drifted")
    _check_self_digest(document, context)


def _validate_sample_selection(
    document: Mapping[str, Any], reserve: Mapping[str, Any],
    cplus: Mapping[str, Any], eligible: Mapping[str, Any],
    ledger: Mapping[str, Any], inventory: Mapping[str, Any], spec: Mapping[str, Any],
) -> None:
    context = "sample_selection"
    _exact(document, (
        "bindings", "canonical_digest", "confirmatory_registry", "counts",
        "document_type", "output_digests", "reason_codes",
        "schema_version", "selection_algorithm_id",
        "stage_d_v2_reserve_manifest", "status", "success",
    ), context)
    bindings = _exact(document["bindings"], (
        "eligible_manifest_canonical_digest", "eligible_manifest_file_sha256",
        "prior_access_ledger_canonical_digest", "prior_access_ledger_file_sha256",
        "source_inventory_canonical_digest", "source_inventory_file_sha256",
        "source_spec_binding",
    ), f"{context}.bindings")
    _exact(bindings["source_spec_binding"], ("canonical_digest", "file_sha256"), f"{context}.bindings.source_spec_binding")
    counts = _exact(document["counts"], (
        "content_equivalent_prior_access_excluded", "duplicate_aliases_excluded",
        "eligible_total", "prior_access_excluded", "required_confirmatory",
        "required_reserve", "required_total_unseen", "unseen_available",
        "unseen_identity_rows",
    ), f"{context}.counts")
    outputs = _exact(document["output_digests"], (
        "confirmatory_ordered_sample_ids_sha256",
        "confirmatory_registry_canonical_digest", "confirmatory_registry_file_sha256",
        "reserve_manifest_canonical_digest", "reserve_manifest_file_sha256",
        "reserve_ordered_sample_ids_sha256",
    ), f"{context}.output_digests")
    expected_algorithm = reserve["selection_algorithm"]["algorithm_id"]
    if (
        document["schema_version"] != 1
        or document["document_type"] != "stage_c_plus_v2_sample_selection"
        or document["status"] != "REGISTRIES_SELECTED"
        or document["success"] is not True
        or document["reason_codes"] != []
        or document["selection_algorithm_id"] != expected_algorithm
        or document["stage_d_v2_reserve_manifest"] != reserve
        or document["confirmatory_registry"] != cplus
        or counts["required_reserve"] != 500
        or counts["required_confirmatory"] != 70
        or counts["required_total_unseen"] != 570
        or bindings["eligible_manifest_canonical_digest"] != eligible["canonical_digest"]
        or bindings["prior_access_ledger_canonical_digest"] != ledger["canonical_digest"]
        or bindings["source_inventory_canonical_digest"] != inventory["canonical_digest"]
        or bindings["source_spec_binding"]["canonical_digest"] != spec["canonical_digest"]
        or outputs["reserve_manifest_canonical_digest"] != reserve["canonical_digest"]
        or outputs["reserve_ordered_sample_ids_sha256"] != reserve["ordered_sample_ids_sha256"]
        or outputs["confirmatory_registry_canonical_digest"] != cplus["canonical_digest"]
    ):
        raise StageDRegistryError(f"{context} semantics/bindings drifted")
    _check_self_digest(document, context)


def _parse_stage_c_csv(payload: bytes) -> tuple[Mapping[str, str], ...]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise StageDRegistryError("Stage C registry CSV is not UTF-8") from error
    try:
        reader = csv.DictReader(io.StringIO(text, newline=""), strict=True)
        if tuple(reader.fieldnames or ()) != STAGE_C_COLUMNS:
            raise StageDRegistryError("Stage C registry CSV header drifted")
        rows = tuple(reader)
    except csv.Error as error:
        raise StageDRegistryError(f"Stage C registry CSV is malformed: {error}") from error
    if len(rows) != 20 or any(None in row for row in rows):
        raise StageDRegistryError("Stage C registry must contain exactly 20 closed rows")
    for index, row in enumerate(rows):
        if set(row) != set(STAGE_C_COLUMNS):
            raise StageDRegistryError(f"Stage C registry row {index} fields drifted")
        if (
            _SAMPLE_ID_RE.fullmatch(row["sample_id"]) is None
            or _RELATIVE_IMAGE_RE.fullmatch(row["relative_path"]) is None
            or _SHA256_RE.fullmatch(row["image_sha256"]) is None
            or row["label_source"] != "official_devkit_val_gt"
        ):
            raise StageDRegistryError(f"Stage C registry row {index} is invalid")
    for field in ("sample_id", "relative_path", "image_sha256"):
        if len({row[field] for row in rows}) != 20:
            raise StageDRegistryError(f"Stage C registry {field} is not unique")
    return rows


def _validate_stage_c_metadata(document: Mapping[str, Any]) -> None:
    context = "stage_c_registry_metadata"
    _exact(document, (
        "schema_version", "document_type", "dataset", "official_labels",
        "registry", "selection", "generation", "asset_findings",
    ), context)
    _exact(document["dataset"], (
        "logical_root", "runtime_root_relative_to_workspace",
        "source_manifest_path", "source_manifest_sha256",
        "source_manifest_metadata_path", "source_manifest_metadata_sha256",
    ), f"{context}.dataset")
    _exact(document["official_labels"], ("devkit_relative_path", "devkit_sha256", "label_source", "synset_order_sha256"), f"{context}.official_labels")
    registry = _exact(document["registry"], ("audit_count", "calibration_count", "holdout_count", "path", "sample_count", "sha256"), f"{context}.registry")
    _exact(document["selection"], (
        "audit_selection_orders_zero_based", "calibration_selection_orders_zero_based",
        "forbidden_selection_information", "legacy_list_path",
        "legacy_list_sha256", "legacy_script_path", "legacy_script_sha256",
        "legacy_source_read_only", "rule_id", "selection_information_used",
        "source_selection_indices_zero_based",
    ), f"{context}.selection")
    _exact(document["generation"], ("base_git_commit", "dependency_lock_sha256", "dirty_reason", "stage_a_command", "tree_dirty_during_generation"), f"{context}.generation")
    findings = _list(document["asset_findings"], f"{context}.asset_findings")
    for index, finding in enumerate(findings):
        if not isinstance(finding, Mapping) or set(finding) not in ({"disposition", "finding", "sample_id"}, {"disposition", "finding", "scope"}):
            raise StageDRegistryError(f"{context}.asset_findings[{index}] fields drifted")
    if (
        document["schema_version"] != 1
        or document["document_type"] != "stage_c_sample_registry_metadata"
        or registry["sample_count"] != 20
        or registry["sha256"] != dict((p, s) for p, s, *_ in SOURCE_FILE_SPECS)[SOURCE_CSV_PATH]
        or document["official_labels"]["label_source"] != "official_devkit_val_gt"
    ):
        raise StageDRegistryError(f"{context} semantics drifted")


def load_and_validate_sources(repo_root: Path | str = REPO_ROOT) -> ValidatedSources:
    verify_d0_completion_ref(repo_root)
    bound = _bind_source_files(repo_root)
    documents = {
        path: value.document for path, value in bound.items()
        if value.document is not None
    }
    expected_canonical = {path: canonical for path, _, canonical, *_ in SOURCE_FILE_SPECS}
    modes = {path: mode for path, _, _, mode, _ in SOURCE_FILE_SPECS}
    for path, document in documents.items():
        if not isinstance(document, Mapping):
            raise StageDRegistryError(f"source JSON must be an object: {path}")
        observed = semantic_digest(document) if modes[path] == "SELF_EXCLUDING_CANONICAL_DIGEST" else canonical_sha256(document)
        if observed != expected_canonical[path]:
            raise StageDRegistryError(
                f"source metadata canonical digest drifted: {path}; "
                f"expected {expected_canonical[path]}, observed {observed}"
            )
    archive = documents["manifests/stage_c_plus_v2/source_archive_identity.json"]
    spec = documents["manifests/stage_c_plus_v2/source_spec.json"]
    inventory = documents["manifests/stage_c_plus_v2/source_inventory.json"]
    eligible = documents["manifests/stage_c_plus_v2/eligible_manifest.json"]
    ledger = documents["manifests/stage_c_plus_v2/prior_access_ledger.json"]
    reserve = documents["manifests/stage_c_plus_v2/stage_d_v2_reserve.json"]
    cplus = documents["manifests/stage_c_plus_v2/cplus_registry.json"]
    access_log = documents["manifests/stage_c_plus_v2/stage_d_v2_reserve_access_log.json"]
    selection = documents["manifests/stage_c_plus_v2/sample_selection.json"]
    stage_c_metadata = documents["manifests/stage_c_sample_registry_metadata.json"]
    _validate_archive_identity(archive, "source_archive_identity")
    _validate_source_spec(spec)
    _validate_source_inventory(inventory, spec)
    eligible_rows = _validate_eligible_manifest(eligible, archive)
    coverage = _validate_prior_ledger(ledger, eligible_rows, inventory, spec)
    cplus_rows = _validate_registry_source(
        cplus, context="cplus_registry", count=70, role="CPLUS_CONFIRMATORY",
        document_type="stage_c_plus_v2_confirmatory_registry", status="FROZEN",
        sealed=False, metadata_only=False,
    )
    reserve_rows = _validate_registry_source(
        reserve, context="stage_d_v2_reserve", count=500,
        role="STAGE_D_V2_RESERVE",
        document_type="stage_d_v2_sealed_reserve_manifest",
        status="RESERVE_SEALED_METADATA_ONLY", sealed=True, metadata_only=True,
    )
    _validate_access_log(access_log, reserve)
    _validate_sample_selection(selection, reserve, cplus, eligible, ledger, inventory, spec)
    _validate_stage_c_metadata(stage_c_metadata)
    stage_c_rows = _parse_stage_c_csv(bound[SOURCE_CSV_PATH].payload)

    eligible_by_id = {row["sample_id"]: row for row in eligible_rows}
    for role, rows in (("CPLUS_CONFIRMATORY", cplus_rows), ("STAGE_D_V2_RESERVE", reserve_rows)):
        for row in rows:
            source = eligible_by_id.get(row["sample_id"])
            if source is None:
                raise StageDRegistryError(f"{role} row is absent from eligible metadata")
            candidate_base = {
                key: value for key, value in row.items()
                if key not in {"audit_overlay", "role", "selection_order", "selection_sha256", "canonical_digest"}
            }
            source_base = {key: value for key, value in source.items() if key != "canonical_digest"}
            if candidate_base != source_base:
                raise StageDRegistryError(f"{role} row differs from eligible metadata")

    class_to_label: dict[int, tuple[str, int]] = {}
    wnid_to_label: dict[str, tuple[int, int]] = {}
    index_to_label: dict[int, tuple[int, str]] = {}
    class_counts: Counter[int] = Counter()
    for row in eligible_rows:
        class_id = row["official_ilsvrc2012_class_id"]
        wnid = row["official_wnid"]
        index = row["imagenet1k_index"]
        if class_id in class_to_label and class_to_label[class_id] != (wnid, index):
            raise StageDRegistryError("official class-ID mapping is inconsistent")
        if wnid in wnid_to_label and wnid_to_label[wnid] != (class_id, index):
            raise StageDRegistryError("official WNID mapping is inconsistent")
        if index in index_to_label and index_to_label[index] != (class_id, wnid):
            raise StageDRegistryError("ImageNet-1k index mapping is inconsistent")
        class_to_label[class_id] = (wnid, index)
        wnid_to_label[wnid] = (class_id, index)
        index_to_label[index] = (class_id, wnid)
        class_counts[class_id] += 1
    if len(class_to_label) != len(wnid_to_label) or len(wnid_to_label) != len(index_to_label) or len(class_to_label) != 1000 or set(class_counts.values()) != {50}:
        raise StageDRegistryError("official 1,000-class metadata chain is incomplete")

    selection_algorithm = reserve["selection_algorithm"]
    selectable: list[tuple[bytes, bytes, str]] = []
    for row in eligible_rows:
        entry = coverage[row["sample_id"]]
        if entry["selection_eligibility"] == "ELIGIBLE_CANONICAL_REPRESENTATIVE":
            score = hashlib.sha256(
                (selection_algorithm["namespace"] + row["relative_path"] + "|" + row["image_sha256"]).encode("utf-8")
            ).digest()
            selectable.append((score, row["relative_path"].encode("utf-8"), row["sample_id"]))
    selectable.sort()
    expected_ids = [item[2] for item in selectable[:570]]
    observed_ids = [row["sample_id"] for row in cplus_rows] + [row["sample_id"] for row in reserve_rows]
    if len(selectable) != ledger["selection_eligible_count"] or expected_ids != observed_ids:
        raise StageDRegistryError("candidate 70+500 selection cannot be deterministically rebuilt")

    for row in reserve_rows:
        entry = coverage[row["sample_id"]]
        if entry["prior_access_status"] != "UNSEEN" or entry["selection_eligibility"] != "ELIGIBLE_CANONICAL_REPRESENTATIVE" or entry["access_record_digests"] != []:
            raise StageDRegistryError("reserve row appears in declared prior-access records")

    for field in ("sample_id", "relative_path", "image_sha256"):
        reserve_values = {row[field] for row in reserve_rows}
        stage_c_values = {row[field] for row in stage_c_rows}
        if not reserve_values.isdisjoint(stage_c_values):
            raise StageDRegistryError(f"reserve overlaps Stage C 20-row registry by {field}")

    return ValidatedSources(
        bindings=bound, documents=documents, stage_c_rows=stage_c_rows,
    )


def _with_digest(document: dict[str, Any]) -> dict[str, Any]:
    if "canonical_digest" in document:
        raise StageDRegistryError("cannot add a second canonical digest")
    result = dict(document)
    result["canonical_digest"] = canonical_sha256(result)
    return result


def _audit_selection(rows: Sequence[Mapping[str, Any]], ordered_sample_ids_sha256: str) -> dict[str, Any]:
    candidates: list[tuple[bytes, int, Mapping[str, Any], str]] = []
    for row in rows:
        message = (
            AUDIT_NAMESPACE + ordered_sample_ids_sha256 + "|"
            + f"{row['selection_order']:03d}" + "|" + row["sample_id"]
        ).encode("utf-8")
        digest = hashlib.sha256(message).hexdigest()
        candidates.append((bytes.fromhex(digest), row["selection_order"], row, digest))
    candidates.sort(key=lambda value: (value[0], value[1]))
    selected = candidates[:5]
    entries = [
        {
            "audit_overlay_id": f"STAGE_D_FIDELITY_V1_AUDIT_{rank:02d}",
            "audit_rank": rank,
            "sample_id": row["sample_id"],
            "selection_order": order,
            "selection_digest": digest,
        }
        for rank, (_, order, row, digest) in enumerate(selected)
    ]
    return {
        "namespace": AUDIT_NAMESPACE,
        "algorithm_id": AUDIT_ALGORITHM_ID,
        "hash_input_format": "namespace + ordered_sample_ids_sha256 + '|' + selection_order_3digit + '|' + sample_id",
        "ordering": "selection_digest raw bytes ascending; selection_order ascending tie-break",
        "input_ordered_sample_ids_sha256": ordered_sample_ids_sha256,
        "audit_image_count": 5,
        "statistical_n_increment": 0,
        "selection_uses_image_bytes_or_outcomes": False,
        "forbidden_selection_information": [
            "image_content", "class_label", "model_prediction",
            "prediction_correctness", "attribution_map", "fidelity_result",
            "human_preference",
        ],
        "selected_sample_ids": [entry["sample_id"] for entry in entries],
        "entries": entries,
    }


def _build_source_binding(sources: ValidatedSources) -> dict[str, Any]:
    records = []
    spec_by_path = {item[0]: item for item in SOURCE_FILE_SPECS}
    for path, *_ in SOURCE_FILE_SPECS:
        _, _, canonical, mode, proof_scope = spec_by_path[path]
        bound = sources.bindings[path]
        records.append({
            "path": path, "bytes": bound.size_bytes,
            "mtime_stability_checked_before_parse": True,
            "file_sha256": bound.sha256, "canonical_digest": canonical,
            "canonical_digest_mode": mode, "proof_scope": proof_scope,
            "parsed_from_same_bound_bytes": True,
            "live_image_or_outcome_bytes_read": False,
        })
    reserve = sources.documents["manifests/stage_c_plus_v2/stage_d_v2_reserve.json"]
    ledger = sources.documents["manifests/stage_c_plus_v2/prior_access_ledger.json"]
    eligible = sources.documents["manifests/stage_c_plus_v2/eligible_manifest.json"]
    return _with_digest({
        "schema_ref": SCHEMA_REF_PREFIX + "source_binding",
        "schema_version": 1,
        "document_type": "stage_d_fidelity_v1_source_binding",
        "package_id": PACKAGE_ID,
        "d0_scope": {
            "completion_ref": D0_COMPLETION_REF,
            "completion_commit": D0_COMPLETION_COMMIT,
            "machine_record_path": "config/stage_d_scope_decision.json",
            "machine_record_file_sha256": D0_SCOPE_FILE_SHA256,
            "machine_record_canonical_digest": D0_SCOPE_CANONICAL_DIGEST,
            "schema_file_sha256": D0_SCOPE_SCHEMA_SHA256,
            "markdown_file_sha256": D0_SCOPE_MARKDOWN_SHA256,
            "preparation_authorization": "APPROVED_TO_PREPARE_STAGE_D",
            "execution_authorization": "NOT_ISSUED",
            "analysis_authorization": "NOT_EVALUATED",
        },
        "source_files": records,
        "candidate_chain": {
            "reserve_file_sha256": sources.bindings["manifests/stage_c_plus_v2/stage_d_v2_reserve.json"].sha256,
            "reserve_canonical_digest": reserve["canonical_digest"],
            "reserve_ordered_sample_ids_sha256": reserve["ordered_sample_ids_sha256"],
            "reserve_row_count": 500,
            "eligible_row_count": 50000,
            "ledger_coverage_row_count": 50000,
            "ledger_selection_eligible_count": ledger["selection_eligible_count"],
            "declared_prior_access_scope": ledger["closed_contract"]["scope"],
            "official_devkit_sha256": eligible["official_devkit"]["devkit_sha256"],
            "official_synset_order_sha256": eligible["official_devkit"]["synset_order_sha256"],
            "official_label_source": "official_devkit_val_gt",
            "selection_algorithm_id": reserve["selection_algorithm"]["algorithm_id"],
        },
        "stage_c_disjointness_source": {
            "registry_path": SOURCE_CSV_PATH,
            "registry_file_sha256": sources.bindings[SOURCE_CSV_PATH].sha256,
            "metadata_path": "manifests/stage_c_sample_registry_metadata.json",
            "metadata_file_sha256": sources.bindings["manifests/stage_c_sample_registry_metadata.json"].sha256,
            "row_count": 20,
            "identity_fields": ["sample_id", "relative_path", "image_sha256"],
        },
        "validation_scope": {
            "metadata_only": True,
            "candidate_recorded_image_hash_format_and_binding_validated": True,
            "live_image_hash_recomputed": False,
            "live_image_equality_deferred_to": "D8.0_AFTER_EXECUTION_AUTHORIZATION",
            "stage_d_outcome_accessed": False,
        },
    })


def _build_registry(sources: ValidatedSources, source_binding: Mapping[str, Any]) -> dict[str, Any]:
    reserve = sources.documents["manifests/stage_c_plus_v2/stage_d_v2_reserve.json"]
    source_rows = reserve["rows"]
    ordered_ids_digest = canonical_sha256([row["sample_id"] for row in source_rows])
    audit = _audit_selection(source_rows, ordered_ids_digest)
    audit_by_id = {entry["sample_id"]: entry for entry in audit["entries"]}
    rows = []
    for candidate in source_rows:
        audit_entry = audit_by_id.get(candidate["sample_id"])
        rows.append(_with_digest({
            "schema_version": 1,
            "selection_order": candidate["selection_order"],
            "sample_id": candidate["sample_id"],
            "image_id": candidate["image_id"],
            "validation_image_number": candidate["validation_image_number"],
            "relative_path": candidate["relative_path"],
            "image_sha256": candidate["image_sha256"],
            "size_bytes": candidate["size_bytes"],
            "width": candidate["width"],
            "height": candidate["height"],
            "image_format": candidate["image_format"],
            "mode": candidate["mode"],
            "official_ilsvrc2012_class_id": candidate["official_ilsvrc2012_class_id"],
            "official_wnid": candidate["official_wnid"],
            "imagenet1k_index": candidate["imagenet1k_index"],
            "label_source": candidate["label_source"],
            "role": "STAGE_D_FIDELITY_PRIMARY_COHORT",
            "candidate_row_canonical_digest": candidate["canonical_digest"],
            "candidate_selection_sha256": candidate["selection_sha256"],
            "image_hash_evidence": "CANDIDATE_RECORDED_NOT_LIVE_RECOMPUTED_D1",
            "audit_overlay": audit_entry is not None,
            "audit_overlay_id": None if audit_entry is None else audit_entry["audit_overlay_id"],
        }))
    return _with_digest({
        "schema_ref": SCHEMA_REF_PREFIX + "registry",
        "schema_version": 1,
        "document_type": "stage_d_fidelity_v1_registry",
        "package_id": PACKAGE_ID,
        "status": "FROZEN_METADATA_ONLY",
        "source_binding_canonical_digest": source_binding["canonical_digest"],
        "candidate_reserve_binding": {
            "path": "manifests/stage_c_plus_v2/stage_d_v2_reserve.json",
            "file_sha256": sources.bindings["manifests/stage_c_plus_v2/stage_d_v2_reserve.json"].sha256,
            "canonical_digest": reserve["canonical_digest"],
            "ordered_sample_ids_sha256": reserve["ordered_sample_ids_sha256"],
        },
        "cohort": {
            "sample_count": 500, "statistical_n": 500,
            "selection_order_min": 0, "selection_order_max": 499,
            "candidate_order_preserved": True, "rows_reselected": False,
            "rows_replaced_or_supplemented": False,
            "selection_used_class_content_prediction_result_or_human_preference": False,
        },
        "image_byte_scope": {
            "candidate_recorded_image_sha256_only": True,
            "live_reserve_jpeg_stat_hash_decode_or_preprocess_performed_d1": False,
            "live_byte_equality_status": "NOT_EVALUATED_D1_DEFERRED_TO_D8_0",
        },
        "ordered_sample_ids_sha256": ordered_ids_digest,
        "ordered_relative_paths_sha256": canonical_sha256([row["relative_path"] for row in rows]),
        "ordered_image_sha256s_sha256": canonical_sha256({"ordered_image_sha256s": [row["image_sha256"] for row in rows]}),
        "rows_canonical_digest": canonical_sha256({"rows": rows}),
        "audit_selection": audit,
        "rows": rows,
    })


def _build_attestation(
    sources: ValidatedSources, source_binding: Mapping[str, Any],
    registry: Mapping[str, Any],
) -> dict[str, Any]:
    ledger = sources.documents["manifests/stage_c_plus_v2/prior_access_ledger.json"]
    spec = sources.documents["manifests/stage_c_plus_v2/source_spec.json"]
    return _with_digest({
        "schema_ref": SCHEMA_REF_PREFIX + "pre_authorization_access_attestation",
        "schema_version": 1,
        "document_type": "stage_d_fidelity_v1_pre_authorization_access_attestation",
        "package_id": PACKAGE_ID,
        "attestation_id": "STAGE_D_D1_METADATA_ONLY_PRE_AUTHORIZATION_20260825",
        "attestation_date": "2026-08-25",
        "source_binding_canonical_digest": source_binding["canonical_digest"],
        "registry_canonical_digest": registry["canonical_digest"],
        "prior_access_ledger": {
            "path": "manifests/stage_c_plus_v2/prior_access_ledger.json",
            "file_sha256": sources.bindings["manifests/stage_c_plus_v2/prior_access_ledger.json"].sha256,
            "canonical_digest": ledger["canonical_digest"],
            "status": ledger["status"],
            "declared_scope": ledger["closed_contract"]["scope"],
            "source_count": ledger["source_count"],
            "known_access_record_count": ledger["known_access_record_count"],
            "coverage_row_count": len(ledger["coverage_rows"]),
            "reserve_rows_checked": 500,
            "reserve_rows_with_prior_access_record": 0,
            "reserve_rows_marked_unseen": 500,
            "historical_directory_byte_audit_performed": spec["user_scoped_historical_boundary"]["historical_directory_byte_audit_performed"],
            "historical_directory_exhaustiveness_claimed": spec["user_scoped_historical_boundary"]["historical_directory_exhaustiveness_claimed"],
            "bounded_claim": "All 500 reserve identities are absent from prior-access records in the declared and closed ledger scope.",
        },
        "current_task_access_facts": {
            "reserve_metadata_contents_accessed": True,
            "reserve_image_paths_statted": False,
            "reserve_image_bytes_accessed": False,
            "reserve_image_hashes_recomputed_from_live_bytes": False,
            "reserve_jpeg_decoded_or_preprocessed": False,
            "reserve_result_accessed": False,
            "stage_d_outcome_accessed": False,
            "model_loaded": False,
            "model_inference_performed": False,
            "model_or_gpu_used": False,
            "gpu_used": False,
            "candidate_access_log_modified": False,
            "c9_or_c10_evidence_modified": False,
            "paper_data_or_figures_generated": False,
        },
        "epistemic_boundary": {
            "absolute_historical_non_access_claim_made": False,
            "ledger_absence_is_scope_bounded": True,
            "schema_and_digests_fix_declared_content_not_independent_behavior_proof": True,
            "candidate_access_log_is_bound_not_rewritten": True,
        },
        "authorization": {
            "preparation_authorization_observed": "APPROVED_TO_PREPARE_STAGE_D",
            "execution_authorization": "NOT_ISSUED",
            "analysis_authorization": "NOT_EVALUATED",
            "stage_d_execution_started": False,
            "authorized_next_task_after_d1_pass": "D2_PROTOCOL_AND_STATISTICS_FREEZE_METADATA_ONLY",
        },
    })


def _build_metadata(
    sources: ValidatedSources, source_binding: Mapping[str, Any],
    registry: Mapping[str, Any], attestation: Mapping[str, Any],
) -> dict[str, Any]:
    rows = registry["rows"]
    return _with_digest({
        "schema_ref": SCHEMA_REF_PREFIX + "registry_metadata",
        "schema_version": 1,
        "document_type": "stage_d_fidelity_v1_registry_metadata",
        "package_id": PACKAGE_ID,
        "status": "D1_PASS_REGISTRY_FROZEN",
        "bindings": {
            "source_binding_canonical_digest": source_binding["canonical_digest"],
            "registry_canonical_digest": registry["canonical_digest"],
            "attestation_canonical_digest": attestation["canonical_digest"],
            "d0_scope_canonical_digest": D0_SCOPE_CANONICAL_DIGEST,
        },
        "counts": {
            "registry_rows": 500,
            "unique_sample_ids": len({row["sample_id"] for row in rows}),
            "unique_relative_paths": len({row["relative_path"] for row in rows}),
            "unique_candidate_recorded_image_sha256s": len({row["image_sha256"] for row in rows}),
            "unique_official_wnids": len({row["official_wnid"] for row in rows}),
            "audit_overlay_rows": sum(row["audit_overlay"] for row in rows),
            "statistical_n": 500,
            "stage_c_registry_rows": len(sources.stage_c_rows),
        },
        "disjointness": {
            "stage_c_registry_file_sha256": sources.bindings[SOURCE_CSV_PATH].sha256,
            "sample_id_overlap_count": 0,
            "relative_path_overlap_count": 0,
            "image_sha256_overlap_count": 0,
            "all_three_identity_dimensions_disjoint": True,
        },
        "audit_overlay": {
            "namespace": AUDIT_NAMESPACE,
            "algorithm_id": AUDIT_ALGORITHM_ID,
            "selected_sample_ids": registry["audit_selection"]["selected_sample_ids"],
            "audit_image_count": 5,
            "statistical_n_increment": 0,
        },
        "label_validation": {
            "label_source": "official_devkit_val_gt",
            "official_mapping_bijective_class_count": 1000,
            "eligible_rows_per_official_class": 50,
            "registry_rows_matched_to_eligible_metadata": 500,
            "class_balance_required": False,
            "repeated_class_labels_allowed": True,
        },
        "rebuild_contract": {
            "builder_module": "vitnns.stage_d_registry",
            "builder_function": "build_stage_d_registry_package",
            "source_file_count": len(SOURCE_FILE_SPECS),
            "candidate_order_preserved_exactly": True,
            "exact_canonical_output_bytes_required": True,
            "live_image_access_required": False,
            "outcome_access_required": False,
        },
        "scientific_boundary": {
            "stage_d_outcome_status": "NOT_EVALUATED",
            "effect_or_validity_conclusion_generated": False,
            "execution_authorization_issued": False,
            "d2_contract_frozen": False,
        },
    })


def build_stage_d_registry_package(
    repo_root: Path | str = REPO_ROOT, *, schema_bytes: bytes | None = None,
) -> dict[str, bytes]:
    root = Path(repo_root)
    sources = load_and_validate_sources(root)
    if schema_bytes is None:
        schema_root_fd = _safe_open_directory(root)
        try:
            schema_bound = _safe_read_relative(
                schema_root_fd, f"manifests/stage_d_fidelity_v1/{PACKAGE_SCHEMA_NAME}"
            )
        finally:
            os.close(schema_root_fd)
        schema_bytes = schema_bound.payload
    schema = strict_json_bytes(schema_bytes, PACKAGE_SCHEMA_NAME)
    if not isinstance(schema, Mapping):
        raise StageDRegistryError("D1 package schema must be an object")
    if schema_bytes != canonical_file_bytes(schema):
        raise StageDRegistryError("D1 package schema bytes are not canonical")
    source_binding = _build_source_binding(sources)
    registry = _build_registry(sources, source_binding)
    attestation = _build_attestation(sources, source_binding, registry)
    metadata = _build_metadata(sources, source_binding, registry, attestation)
    documents = {
        "source_binding.json": source_binding,
        "registry.json": registry,
        "pre_authorization_access_attestation.json": attestation,
        "registry_metadata.json": metadata,
    }
    payloads: dict[str, bytes] = {PACKAGE_SCHEMA_NAME: schema_bytes}
    payloads.update({name: canonical_file_bytes(value) for name, value in documents.items()})
    manifest_records = []
    for name in MANIFEST_PAYLOADS:
        payload = payloads[name]
        value = schema if name == PACKAGE_SCHEMA_NAME else documents[name]
        manifest_records.append({
            "path": name, "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "canonical_digest": canonical_sha256(value) if name == PACKAGE_SCHEMA_NAME else value["canonical_digest"],
            "digest_mode": "FULL_DOCUMENT" if name == PACKAGE_SCHEMA_NAME else "SELF_EXCLUDING_CANONICAL_DIGEST",
        })
    manifest = _with_digest({
        "schema_ref": SCHEMA_REF_PREFIX + "manifest",
        "schema_version": 1,
        "document_type": "stage_d_fidelity_v1_registry_package_manifest",
        "package_id": PACKAGE_ID,
        "status": "COMPLETE_METADATA_ONLY",
        "payload_count": len(manifest_records),
        "payloads": manifest_records,
        "registry_canonical_digest": registry["canonical_digest"],
        "source_binding_canonical_digest": source_binding["canonical_digest"],
        "attestation_canonical_digest": attestation["canonical_digest"],
        "registry_metadata_canonical_digest": metadata["canonical_digest"],
        "execution_authorization": "NOT_ISSUED",
    })
    payloads["manifest.json"] = canonical_file_bytes(manifest)
    return payloads


def write_stage_d_registry_package(
    repo_root: Path | str = REPO_ROOT, package_dir: Path | str = PACKAGE_DIR,
) -> dict[str, bytes]:
    """Write the deterministic D1 package; callers must provide a fresh directory."""

    target = Path(package_dir)
    if not target.is_dir():
        raise StageDRegistryError("D1 package directory must already exist")
    existing = {item.name for item in target.iterdir()}
    if existing != {PACKAGE_SCHEMA_NAME}:
        raise StageDRegistryError(
            "fresh D1 generation requires a directory containing only the schema"
        )
    schema_bytes = (target / PACKAGE_SCHEMA_NAME).read_bytes()
    payloads = build_stage_d_registry_package(repo_root, schema_bytes=schema_bytes)
    for name in PACKAGE_FILES[1:]:
        destination = target / name
        if destination.exists() or destination.is_symlink():
            raise StageDRegistryError(f"refusing to overwrite D1 package file: {name}")
        destination.write_bytes(payloads[name])
    return payloads


def _validate_output_registry(document: Mapping[str, Any]) -> None:
    rows = document["rows"]
    if len(rows) != 500 or [row["selection_order"] for row in rows] != list(range(500)):
        raise StageDRegistryError("D1 registry row count/order drifted")
    for field in ("sample_id", "relative_path", "image_sha256", "canonical_digest"):
        if len({row[field] for row in rows}) != 500:
            raise StageDRegistryError(f"D1 registry {field} is not unique")
    for index, row in enumerate(rows):
        _check_self_digest(row, f"registry.rows[{index}]")
        if row["role"] != "STAGE_D_FIDELITY_PRIMARY_COHORT" or row["image_hash_evidence"] != "CANDIDATE_RECORDED_NOT_LIVE_RECOMPUTED_D1":
            raise StageDRegistryError("D1 registry row role/hash scope drifted")
    if document["rows_canonical_digest"] != canonical_sha256({"rows": rows}):
        raise StageDRegistryError("D1 registry rows digest drifted")
    if document["ordered_sample_ids_sha256"] != canonical_sha256([row["sample_id"] for row in rows]):
        raise StageDRegistryError("D1 ordered sample digest drifted")
    expected_audit = _audit_selection(rows, document["ordered_sample_ids_sha256"])
    if document["audit_selection"] != expected_audit:
        raise StageDRegistryError("D1 audit overlay selection drifted")
    selected = set(expected_audit["selected_sample_ids"])
    if {row["sample_id"] for row in rows if row["audit_overlay"]} != selected:
        raise StageDRegistryError("D1 audit overlay flags drifted")
    if sum(row["audit_overlay"] for row in rows) != 5:
        raise StageDRegistryError("D1 must contain exactly five audit overlays")
    _check_self_digest(document, "registry")


def validate_stage_d_registry_package(
    package_dir: Path | str = PACKAGE_DIR, *, repo_root: Path | str = REPO_ROOT,
    verify_sources: bool = True, verify_rebuild: bool = True,
) -> dict[str, Any]:
    root_fd = _safe_open_directory(package_dir)
    try:
        names = set(os.listdir(root_fd))
        if names != set(PACKAGE_FILES):
            raise StageDRegistryError(
                f"D1 package inventory drifted: missing={sorted(set(PACKAGE_FILES)-names)}, "
                f"extra={sorted(names-set(PACKAGE_FILES))}"
            )
        bound = {name: _safe_read_relative(root_fd, name) for name in PACKAGE_FILES}
    finally:
        os.close(root_fd)
    documents: dict[str, Mapping[str, Any]] = {}
    for name, item in bound.items():
        value = strict_json_bytes(item.payload, name)
        if not isinstance(value, Mapping):
            raise StageDRegistryError(f"D1 package file is not an object: {name}")
        if item.payload != canonical_file_bytes(value):
            raise StageDRegistryError(f"D1 package file bytes are not canonical: {name}")
        documents[name] = value
    try:
        from jsonschema import Draft202012Validator
    except ImportError as error:
        raise StageDRegistryError("jsonschema is required to validate the D1 package") from error
    schema = documents[PACKAGE_SCHEMA_NAME]
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    for name in PACKAGE_FILES[1:]:
        errors = sorted(validator.iter_errors(documents[name]), key=lambda error: list(error.path))
        if errors:
            raise StageDRegistryError(f"D1 package schema validation failed for {name}: {errors[0].message}")
        _check_self_digest(documents[name], name)
    manifest = documents["manifest.json"]
    if manifest["payload_count"] != len(MANIFEST_PAYLOADS) or [item["path"] for item in manifest["payloads"]] != list(MANIFEST_PAYLOADS):
        raise StageDRegistryError("D1 manifest payload inventory/order drifted")
    for record in manifest["payloads"]:
        item = bound[record["path"]]
        value = documents[record["path"]]
        expected_canonical = canonical_sha256(value) if record["path"] == PACKAGE_SCHEMA_NAME else value["canonical_digest"]
        if (record["bytes"], record["sha256"], record["canonical_digest"]) != (item.size_bytes, item.sha256, expected_canonical):
            raise StageDRegistryError(f"D1 manifest binding drifted: {record['path']}")
    registry = documents["registry.json"]
    source_binding = documents["source_binding.json"]
    attestation = documents["pre_authorization_access_attestation.json"]
    metadata = documents["registry_metadata.json"]
    _validate_output_registry(registry)
    if registry["source_binding_canonical_digest"] != source_binding["canonical_digest"]:
        raise StageDRegistryError("D1 registry/source binding drifted")
    if attestation["registry_canonical_digest"] != registry["canonical_digest"] or attestation["source_binding_canonical_digest"] != source_binding["canonical_digest"]:
        raise StageDRegistryError("D1 attestation binding drifted")
    if metadata["bindings"] != {
        "source_binding_canonical_digest": source_binding["canonical_digest"],
        "registry_canonical_digest": registry["canonical_digest"],
        "attestation_canonical_digest": attestation["canonical_digest"],
        "d0_scope_canonical_digest": D0_SCOPE_CANONICAL_DIGEST,
    }:
        raise StageDRegistryError("D1 metadata binding drifted")
    facts = attestation["current_task_access_facts"]
    forbidden_true = (
        "reserve_image_paths_statted", "reserve_image_bytes_accessed",
        "reserve_image_hashes_recomputed_from_live_bytes",
        "reserve_jpeg_decoded_or_preprocessed", "reserve_result_accessed",
        "stage_d_outcome_accessed", "model_loaded",
        "model_inference_performed", "model_or_gpu_used", "gpu_used",
        "candidate_access_log_modified", "c9_or_c10_evidence_modified",
        "paper_data_or_figures_generated",
    )
    if facts["reserve_metadata_contents_accessed"] is not True or any(facts[key] is not False for key in forbidden_true):
        raise StageDRegistryError("D1 pre-authorization access attestation drifted")
    if attestation["authorization"]["execution_authorization"] != "NOT_ISSUED" or attestation["authorization"]["stage_d_execution_started"] is not False:
        raise StageDRegistryError("D1 must not authorize or start execution")
    if verify_sources or verify_rebuild:
        rebuilt = build_stage_d_registry_package(repo_root, schema_bytes=bound[PACKAGE_SCHEMA_NAME].payload)
        if verify_rebuild:
            for name in PACKAGE_FILES:
                if rebuilt[name] != bound[name].payload:
                    raise StageDRegistryError(f"D1 package is not deterministically rebuildable: {name}")
    return {
        "status": "PASS", "package_id": PACKAGE_ID,
        "file_count": len(PACKAGE_FILES), "registry_rows": len(registry["rows"]),
        "audit_overlay_count": len(registry["audit_selection"]["entries"]),
        "audit_sample_ids": registry["audit_selection"]["selected_sample_ids"],
        "file_sha256": {name: bound[name].sha256 for name in PACKAGE_FILES},
        "canonical_digests": {
            name: canonical_sha256(documents[name]) if name == PACKAGE_SCHEMA_NAME else documents[name]["canonical_digest"]
            for name in PACKAGE_FILES
        },
        "reserve_image_bytes_accessed": False,
        "reserve_result_accessed": False,
        "model_or_gpu_used": False,
        "execution_authorization": "NOT_ISSUED",
    }


def verify_d0_completion_ref(repo_root: Path | str = REPO_ROOT) -> str:
    result = subprocess.run(
        ["git", "rev-parse", f"{D0_COMPLETION_REF}^{{commit}}"],
        cwd=repo_root, check=True, capture_output=True, text=True,
    ).stdout.strip()
    if result != D0_COMPLETION_COMMIT:
        raise StageDRegistryError("D0 completion ref drifted")
    return result


__all__ = [
    "AUDIT_ALGORITHM_ID", "AUDIT_NAMESPACE", "D0_COMPLETION_COMMIT",
    "D0_COMPLETION_REF", "D0_SCOPE_CANONICAL_DIGEST", "PACKAGE_DIR",
    "PACKAGE_FILES", "PACKAGE_ID", "PACKAGE_SCHEMA_NAME",
    "SOURCE_FILE_SPECS", "StageDRegistryError",
    "build_stage_d_registry_package", "canonical_file_bytes",
    "canonical_json_bytes", "canonical_sha256",
    "load_and_validate_sources", "semantic_digest", "strict_json_bytes",
    "validate_stage_d_registry_package", "verify_d0_completion_ref",
    "write_stage_d_registry_package",
]
