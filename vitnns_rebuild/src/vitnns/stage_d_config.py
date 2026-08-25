"""Strict, immutable loaders for the frozen Stage D D2 contracts.

The loaders in this module are control-plane only.  They bind JSON inputs from
single descriptor-backed reads, validate the Draft 2020-12 schemas and frozen
instances, and enforce the cross-document rules that JSON Schema cannot state.
They do not import or construct the image/model/attribution execution stack.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import stat
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any

from .constants import REPO_ROOT
from .stage_d_registry import canonical_sha256


class StageDConfigError(ValueError):
    """Raised when a Stage D config or run spec fails closed."""


# A generic alias is convenient for callers validating the pair as one contract.
StageDContractError = StageDConfigError

STAGE_D_CONFIG_PATH = REPO_ROOT / "config/stage_d.json"
STAGE_D_CONFIG_SCHEMA_PATH = REPO_ROOT / "config/stage_d.schema.json"
STAGE_D_RUN_SPEC_PATH = REPO_ROOT / "config/stage_d_run_spec.json"
STAGE_D_RUN_SPEC_SCHEMA_PATH = REPO_ROOT / "config/stage_d_run_spec.schema.json"

# Compatibility-style aliases matching the repository's Stage C loader naming.
DEFAULT_STAGE_D_CONFIG = STAGE_D_CONFIG_PATH
DEFAULT_STAGE_D_SCHEMA = STAGE_D_CONFIG_SCHEMA_PATH
DEFAULT_STAGE_D_RUN_SPEC = STAGE_D_RUN_SPEC_PATH
DEFAULT_STAGE_D_RUN_SPEC_SCHEMA = STAGE_D_RUN_SPEC_SCHEMA_PATH

STAGE_D_CONFIG_FILE_SHA256 = (
    "88fdc9cb2b4a5bd74ff7d8e68c79463cafac6437e29b5564c2115a13f1f1136d"
)
STAGE_D_CONFIG_SCHEMA_FILE_SHA256 = (
    "64fed10aed29a1c97807ebee2d91a29c3dc218993909b4f0c8fab345d8f78c32"
)
STAGE_D_RUN_SPEC_FILE_SHA256 = (
    "43c61f190a9ad5811d878a7516039287f3914548f5444934fc06f41769c97d20"
)
STAGE_D_RUN_SPEC_SCHEMA_FILE_SHA256 = (
    "ea23b61bb0d9f42fd6a18498c1c6d900b82f159da2b9f87ef51f7c0428d87646"
)
STAGE_D_CONFIG_CANONICAL_DIGEST = (
    "21f9dcf1d615f2531f7d7d9a36319b958e06d0606c325890dd31ad89b7c2caae"
)
STAGE_D_RUN_SPEC_CANONICAL_DIGEST = (
    "56a1d946c96ce5f86699e52a359557c1cdfd6ebbfd5aa43646d5e3a2d6428c85"
)

# Short aliases are intentionally exported as well; all values remain frozen.
STAGE_D_CONFIG_SHA256 = STAGE_D_CONFIG_FILE_SHA256
STAGE_D_CONFIG_SCHEMA_SHA256 = STAGE_D_CONFIG_SCHEMA_FILE_SHA256
STAGE_D_RUN_SPEC_SHA256 = STAGE_D_RUN_SPEC_FILE_SHA256
STAGE_D_RUN_SPEC_SCHEMA_SHA256 = STAGE_D_RUN_SPEC_SCHEMA_FILE_SHA256

_DRAFT_2020_12 = "https://json-schema.org/draft/2020-12/schema"
_MAX_CONTRACT_BYTES = 16 * 1024 * 1024
_D1_ORDERED_SAMPLE_IDS_SHA256 = (
    "b187ff5cbacc1b595f06d12e309602b09965e01e7ee1461fd8a694354d4b8f50"
)
_D1_REGISTRY_FILE_SHA256 = (
    "9aace3abf02a2c07d992d88b765c53cfeffa7f83a6ed72ecefb8cecf9eb79b6b"
)
_D1_REGISTRY_CANONICAL_DIGEST = (
    "514ce4dd2e1fedc1744893e5543651ea42c053458b233ff11a45f700bb4bc028"
)
_DOCUMENT_ISSUER = object()


@dataclass(frozen=True, slots=True)
class _BoundFile:
    source_path: Path
    relative_path: str
    payload: bytes = field(repr=False)
    file_sha256: str
    size_bytes: int


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return copy.deepcopy(value)


@dataclass(frozen=True, slots=True)
class _FrozenDocument(Mapping[str, Any]):
    """Immutable exact-document wrapper shared by config and run spec."""

    _data: Mapping[str, Any] = field(repr=False)
    source_path: Path
    file_sha256: str
    size_bytes: int
    schema_path: Path
    schema_file_sha256: str
    _issuer: object = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._issuer is not _DOCUMENT_ISSUER:
            raise StageDConfigError(
                "Stage D document wrappers must be created by the strict loader"
            )
        if not isinstance(self._data, Mapping):
            raise TypeError("Stage D document data must be a mapping")
        object.__setattr__(self, "_data", _freeze(dict(self._data)))
        object.__setattr__(self, "source_path", Path(self.source_path))
        object.__setattr__(self, "schema_path", Path(self.schema_path))

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __getattr__(self, name: str) -> Any:
        # Attribute access is a convenience for top-level JSON names.  Mapping
        # access remains authoritative and handles names such as ``$schema``.
        data = object.__getattribute__(self, "_data")
        try:
            return data[name]
        except KeyError as error:
            raise AttributeError(name) from error

    @property
    def canonical_digest(self) -> str:
        return self._data["canonical_digest"]

    @property
    def schema_ref(self) -> str:
        return self._data["$schema"]

    @property
    def schema_sha256(self) -> str:
        return self.schema_file_sha256

    def to_dict(self) -> dict[str, Any]:
        """Return a detached, exact JSON-shaped copy of the loaded document."""

        return _thaw(self._data)


class StageDConfig(_FrozenDocument):
    """Validated immutable Stage D D2 protocol configuration."""

    __slots__ = ()


class StageDRunSpec(_FrozenDocument):
    """Validated immutable Stage D D2 run specification."""

    __slots__ = ()


@dataclass(frozen=True, slots=True)
class StageDContracts(Mapping[str, _FrozenDocument]):
    """An immutable, cross-validated config/run-spec pair."""

    config: StageDConfig
    run_spec: StageDRunSpec

    def __post_init__(self) -> None:
        if not isinstance(self.config, StageDConfig):
            raise StageDConfigError("contracts.config must be a StageDConfig")
        if not isinstance(self.run_spec, StageDRunSpec):
            raise StageDConfigError("contracts.run_spec must be a StageDRunSpec")
        _validate_cross_bindings(self.config, self.run_spec)

    def __getitem__(self, key: str) -> _FrozenDocument:
        if key == "config":
            return self.config
        if key == "run_spec":
            return self.run_spec
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return iter(("config", "run_spec"))

    def __len__(self) -> int:
        return 2

    @property
    def config_canonical_digest(self) -> str:
        return self.config.canonical_digest

    @property
    def run_spec_canonical_digest(self) -> str:
        return self.run_spec.canonical_digest

    @property
    def config_file_sha256(self) -> str:
        return self.config.file_sha256

    @property
    def run_spec_file_sha256(self) -> str:
        return self.run_spec.file_sha256

    def to_dict(self) -> dict[str, dict[str, Any]]:
        return {
            "config": self.config.to_dict(),
            "run_spec": self.run_spec.to_dict(),
        }


def _absolute_lexical(path: Path | str, context: str) -> Path:
    raw = os.fspath(path)
    if not isinstance(raw, str) or not raw or "\x00" in raw:
        raise StageDConfigError(f"{context} must be a non-empty text path")
    if "\\" in raw:
        raise StageDConfigError(f"{context} must use POSIX path separators")
    pure = PurePosixPath(raw)
    if ".." in pure.parts or "." in pure.parts:
        raise StageDConfigError(f"{context} contains an unsafe path component")
    return Path(os.path.abspath(raw))


def _relative_input_path(
    path: Path | str, repo_root: Path | str, context: str,
) -> tuple[Path, PurePosixPath]:
    root = _absolute_lexical(repo_root, "repository root")
    raw = os.fspath(path)
    if not isinstance(raw, str) or not raw or "\x00" in raw:
        raise StageDConfigError(f"{context} must be a non-empty text path")
    if "\\" in raw:
        raise StageDConfigError(f"{context} must use POSIX path separators")
    pure_input = PurePosixPath(raw)
    if ".." in pure_input.parts or "." in pure_input.parts:
        raise StageDConfigError(f"{context} contains an unsafe path component")
    candidate = Path(os.path.abspath(raw)) if pure_input.is_absolute() else root / raw
    candidate = Path(os.path.abspath(os.fspath(candidate)))
    try:
        relative = candidate.relative_to(root)
    except ValueError as error:
        raise StageDConfigError(f"{context} escapes repository root: {raw}") from error
    pure_relative = PurePosixPath(relative.as_posix())
    if (
        not pure_relative.parts
        or ".." in pure_relative.parts
        or "." in pure_relative.parts
        or any(not part for part in pure_relative.parts)
    ):
        raise StageDConfigError(f"{context} is not a safe repository-relative path")
    return root, pure_relative


def _directory_flags() -> int:
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        raise StageDConfigError(
            "this platform cannot provide descriptor-safe no-follow reads"
        )
    return os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)


def _open_directory_no_follow(path: Path, context: str) -> int:
    flags = _directory_flags()
    try:
        current = os.open(path.anchor, flags)
    except OSError as error:
        raise StageDConfigError(f"cannot safely open {context}: {path}") from error
    try:
        for component in path.parts[1:]:
            try:
                next_fd = os.open(component, flags, dir_fd=current)
            except OSError as error:
                raise StageDConfigError(
                    f"{context} has a missing, symlink, or unsafe component: "
                    f"{component}"
                ) from error
            os.close(current)
            current = next_fd
        if not stat.S_ISDIR(os.fstat(current).st_mode):
            raise StageDConfigError(f"{context} is not a directory: {path}")
        return current
    except BaseException:
        os.close(current)
        raise


def _stat_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev, value.st_ino, value.st_mode, value.st_nlink,
        value.st_uid, value.st_gid, value.st_size, value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _safe_read_relative(
    repo_root: Path | str, path: Path | str, context: str,
) -> _BoundFile:
    root, relative = _relative_input_path(path, repo_root, context)
    root_fd = _open_directory_no_follow(root, "repository root")
    directory_fd = os.dup(root_fd)
    os.close(root_fd)
    try:
        flags = _directory_flags()
        for component in relative.parts[:-1]:
            try:
                next_fd = os.open(component, flags, dir_fd=directory_fd)
            except OSError as error:
                raise StageDConfigError(
                    f"{context} has a missing, symlink, or unsafe directory: "
                    f"{relative.as_posix()}"
                ) from error
            os.close(directory_fd)
            directory_fd = next_fd
        file_flags = (
            os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NONBLOCK", 0)
        )
        filename = relative.parts[-1]
        try:
            entry_before = os.stat(
                filename, dir_fd=directory_fd, follow_symlinks=False,
            )
            fd = os.open(filename, file_flags, dir_fd=directory_fd)
        except OSError as error:
            raise StageDConfigError(
                f"cannot safely open {context}: {relative.as_posix()}"
            ) from error
        try:
            before = os.fstat(fd)
            if (
                not stat.S_ISREG(entry_before.st_mode)
                or not stat.S_ISREG(before.st_mode)
                or entry_before.st_nlink != 1
                or before.st_nlink != 1
                or _stat_identity(entry_before) != _stat_identity(before)
            ):
                raise StageDConfigError(
                    f"{context} is not an ordinary single-link regular file: "
                    f"{relative.as_posix()}"
                )
            if before.st_size > _MAX_CONTRACT_BYTES:
                raise StageDConfigError(
                    f"{context} exceeds the {_MAX_CONTRACT_BYTES}-byte safety limit"
                )
            chunks: list[bytes] = []
            total = 0
            while True:
                chunk = os.read(fd, min(1024 * 1024, _MAX_CONTRACT_BYTES + 1 - total))
                if not chunk:
                    break
                chunks.append(chunk)
                total += len(chunk)
                if total > _MAX_CONTRACT_BYTES:
                    raise StageDConfigError(
                        f"{context} exceeds the {_MAX_CONTRACT_BYTES}-byte safety limit"
                    )
            payload = b"".join(chunks)
            after = os.fstat(fd)
            try:
                entry_after = os.stat(
                    filename, dir_fd=directory_fd, follow_symlinks=False,
                )
            except OSError as error:
                raise StageDConfigError(
                    f"{context} changed while being read"
                ) from error
        finally:
            os.close(fd)
    finally:
        os.close(directory_fd)
    if (
        _stat_identity(before) != _stat_identity(after)
        or _stat_identity(after) != _stat_identity(entry_after)
        or len(payload) != before.st_size
    ):
        raise StageDConfigError(f"{context} changed while being read")
    return _BoundFile(
        source_path=root / Path(*relative.parts),
        relative_path=relative.as_posix(),
        payload=payload,
        file_sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
    )


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise StageDConfigError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_token(token: str) -> None:
    raise StageDConfigError(f"non-finite JSON number is forbidden: {token}")


def _validate_json_value(value: Any, context: str) -> None:
    if value is None or type(value) in {bool, int}:
        return
    if isinstance(value, str):
        try:
            value.encode("utf-8")
        except UnicodeEncodeError as error:
            raise StageDConfigError(
                f"{context} contains an invalid Unicode surrogate"
            ) from error
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise StageDConfigError(f"{context} contains a non-finite number")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise StageDConfigError(f"{context} has a non-string object key")
            _validate_json_value(key, f"{context} object key")
            _validate_json_value(child, f"{context}.{key}")
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            _validate_json_value(child, f"{context}[{index}]")
        return
    raise StageDConfigError(
        f"{context} contains non-JSON value {type(value).__name__}"
    )


def strict_json_bytes(payload: bytes, context: str = "Stage D JSON") -> Any:
    """Parse strict UTF-8 JSON without duplicate keys or non-finite values."""

    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise StageDConfigError(f"{context} must be UTF-8") from error
    try:
        value = json.loads(
            text, object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_token,
        )
    except json.JSONDecodeError as error:
        raise StageDConfigError(f"{context} is invalid JSON: {error}") from error
    _validate_json_value(value, context)
    return value


def _parse_bound_json(bound: _BoundFile, context: str) -> Mapping[str, Any]:
    value = strict_json_bytes(bound.payload, context)
    if not isinstance(value, Mapping):
        raise StageDConfigError(f"{context} root must be a JSON object")
    return value


def _schema_location(error: Any) -> str:
    parts = [str(part) for part in error.absolute_path]
    return "$" + "".join(
        f"[{part}]" if part.isdigit() else f".{part}" for part in parts
    )


def _assert_effectively_closed_schema(rule: Any, context: str = "$") -> None:
    if not isinstance(rule, Mapping):
        return
    if isinstance(rule.get("const"), Mapping):
        # An object-valued const admits no extra or missing keys.
        return
    if rule.get("type") == "object" or "properties" in rule:
        properties = rule.get("properties")
        required = rule.get("required")
        if not isinstance(properties, Mapping):
            raise StageDConfigError(f"{context} object schema lacks properties")
        if rule.get("additionalProperties") is not False:
            raise StageDConfigError(f"{context} object schema is not closed")
        if not isinstance(required, list) or set(required) != set(properties):
            raise StageDConfigError(
                f"{context} object schema does not require every property"
            )
    for keyword in ("properties", "patternProperties", "$defs", "dependentSchemas"):
        children = rule.get(keyword)
        if isinstance(children, Mapping):
            for name, child in children.items():
                _assert_effectively_closed_schema(child, f"{context}/{keyword}/{name}")
    for keyword in ("items", "contains", "not", "if", "then", "else"):
        child = rule.get(keyword)
        if isinstance(child, Mapping):
            _assert_effectively_closed_schema(child, f"{context}/{keyword}")
    for keyword in ("prefixItems", "allOf", "anyOf", "oneOf"):
        children = rule.get(keyword)
        if isinstance(children, list):
            for index, child in enumerate(children):
                _assert_effectively_closed_schema(
                    child, f"{context}/{keyword}/{index}"
                )


def _validate_schema_document(schema: Mapping[str, Any], context: str) -> Any:
    if schema.get("$schema") != _DRAFT_2020_12:
        raise StageDConfigError(f"{context} must declare Draft 2020-12")
    try:
        from jsonschema import Draft202012Validator
        from jsonschema.exceptions import SchemaError
    except ImportError as error:
        raise StageDConfigError(
            "jsonschema is required for Stage D Draft 2020-12 validation"
        ) from error
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as error:
        raise StageDConfigError(f"{context} fails Draft 2020-12 meta-validation: {error.message}") from error
    _assert_effectively_closed_schema(schema)
    return Draft202012Validator(schema)


def _strict_const_match(actual: Any, expected: Any, context: str) -> None:
    """Apply exact JSON types to schema ``const`` values.

    JSON Schema deliberately equates numerically equal integers and numbers.
    The frozen D2 documents additionally preserve their JSON scalar types and,
    crucially, never permit Python's ``True == 1`` shortcut.
    """

    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            raise StageDConfigError(f"{context} must be an object")
        missing = set(expected) - set(actual)
        unknown = set(actual) - set(expected)
        if missing:
            raise StageDConfigError(f"{context} has missing fields: {sorted(missing)}")
        if unknown:
            raise StageDConfigError(f"{context} has unknown fields: {sorted(unknown)}")
        for key in expected:
            _strict_const_match(actual[key], expected[key], f"{context}.{key}")
        return
    if isinstance(expected, list):
        if not isinstance(actual, list):
            raise StageDConfigError(f"{context} must be an array")
        if len(actual) != len(expected):
            raise StageDConfigError(
                f"{context} length drifted: expected {len(expected)}, got {len(actual)}"
            )
        for index, (item, frozen) in enumerate(zip(actual, expected)):
            _strict_const_match(item, frozen, f"{context}[{index}]")
        return
    if type(actual) is bool or type(expected) is bool:
        if type(actual) is not type(expected) or actual != expected:
            raise StageDConfigError(
                f"{context} has wrong JSON type or boolean/value drift from "
                "its frozen schema const: "
                f"expected {expected!r}, got {actual!r}"
            )
        return
    if type(actual) in {int, float} and type(expected) in {int, float}:
        if not math.isfinite(float(actual)) or actual != expected:
            raise StageDConfigError(
                f"{context} differs from its frozen schema const: "
                f"expected {expected!r}, got {actual!r}"
            )
        return
    if type(actual) is not type(expected):
        raise StageDConfigError(
            f"{context} has wrong JSON type: expected {type(expected).__name__}, "
            f"got {type(actual).__name__}"
        )
    if actual != expected:
        raise StageDConfigError(
            f"{context} differs from its frozen schema const: "
            f"expected {expected!r}, got {actual!r}"
        )


def _strict_enum_contains(actual: Any, candidates: Sequence[Any]) -> bool:
    for candidate in candidates:
        try:
            _strict_const_match(actual, candidate, "enum candidate")
        except StageDConfigError:
            continue
        return True
    return False


def _schema_declares_numeric(rule: Mapping[str, Any]) -> bool:
    declared = rule.get("type")
    if declared in {"integer", "number"}:
        return True
    if isinstance(declared, list) and any(item in {"integer", "number"} for item in declared):
        return True
    frozen = rule.get("const", object())
    if type(frozen) in {int, float}:
        return True
    enum = rule.get("enum")
    return isinstance(enum, list) and any(type(item) in {int, float} for item in enum)


def _enforce_schema_scalar_types(value: Any, rule: Any, context: str) -> None:
    if not isinstance(rule, Mapping):
        return
    if type(value) is bool and _schema_declares_numeric(rule):
        raise StageDConfigError(
            f"{context} must be numeric; booleans are not integers or numbers"
        )
    if "const" in rule:
        _strict_const_match(value, rule["const"], context)
    enum = rule.get("enum")
    if isinstance(enum, list) and not _strict_enum_contains(value, enum):
        raise StageDConfigError(f"{context} is outside its type-exact schema enum")
    properties = rule.get("properties")
    if isinstance(value, Mapping) and isinstance(properties, Mapping):
        for key, child in value.items():
            if key in properties:
                _enforce_schema_scalar_types(child, properties[key], f"{context}.{key}")
    items = rule.get("items")
    if isinstance(value, list) and isinstance(items, Mapping):
        for index, child in enumerate(value):
            _enforce_schema_scalar_types(child, items, f"{context}[{index}]")
    prefix_items = rule.get("prefixItems")
    if isinstance(value, list) and isinstance(prefix_items, list):
        for index, child_rule in enumerate(prefix_items[: len(value)]):
            _enforce_schema_scalar_types(value[index], child_rule, f"{context}[{index}]")


def _validate_instance(
    document: Mapping[str, Any], schema: Mapping[str, Any], validator: Any, context: str,
) -> None:
    _enforce_schema_scalar_types(document, schema, context)
    try:
        errors = sorted(
            validator.iter_errors(document),
            key=lambda error: tuple(str(part) for part in error.absolute_path),
        )
    except Exception as error:
        raise StageDConfigError(
            f"{context} schema reference or validation failed: {error}"
        ) from error
    if errors:
        error = errors[0]
        raise StageDConfigError(
            f"{context} fails Draft 2020-12 validation at "
            f"{_schema_location(error)}: {error.message}"
        )


def _self_digest(document: Mapping[str, Any]) -> str:
    try:
        return canonical_sha256(
            {key: value for key, value in document.items() if key != "canonical_digest"}
        )
    except ValueError as error:
        raise StageDConfigError(f"cannot canonicalize Stage D document: {error}") from error


def _validate_self_digest(
    document: Mapping[str, Any], expected_frozen: str, context: str,
) -> None:
    observed = document.get("canonical_digest")
    if not isinstance(observed, str):
        raise StageDConfigError(f"{context}.canonical_digest must be a string")
    computed = _self_digest(document)
    if observed != computed:
        raise StageDConfigError(
            f"{context} canonical self-digest mismatch: declared {observed}, "
            f"computed {computed}"
        )
    if observed != expected_frozen:
        raise StageDConfigError(
            f"{context} canonical digest drifted: expected {expected_frozen}, "
            f"observed {observed}"
        )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise StageDConfigError(message)


def _safe_repo_reference(raw: Any, context: str) -> None:
    if not isinstance(raw, str) or not raw or "\x00" in raw or "\\" in raw:
        raise StageDConfigError(f"{context} must be a non-empty POSIX path")
    path = PurePosixPath(raw)
    if (
        path.is_absolute() or not path.parts or ".." in path.parts
        or "." in path.parts or path.as_posix() != raw
    ):
        raise StageDConfigError(
            f"{context} must be a normalized repository-relative POSIX path"
        )


def _safe_absolute_reference(raw: Any, context: str) -> None:
    if not isinstance(raw, str) or not raw or "\x00" in raw or "\\" in raw:
        raise StageDConfigError(f"{context} must be an absolute POSIX path")
    path = PurePosixPath(raw)
    if not path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise StageDConfigError(f"{context} must be a normalized absolute POSIX path")
    if path.as_posix() != raw:
        raise StageDConfigError(f"{context} must be a normalized absolute POSIX path")


def _unique(values: Sequence[Any], context: str) -> None:
    rendered = [
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        for value in values
    ]
    if len(rendered) != len(set(rendered)):
        raise StageDConfigError(f"{context} must contain unique values")


def _strictly_increasing_numbers(values: Sequence[Any], context: str) -> None:
    if any(type(value) not in {int, float} or not math.isfinite(value) for value in values):
        raise StageDConfigError(f"{context} must contain finite numbers, not booleans")
    if any(right <= left for left, right in zip(values, values[1:])):
        raise StageDConfigError(f"{context} must be unique and strictly increasing")


def _require_disabled(value: Mapping[str, Any], context: str) -> None:
    for key, item in value.items():
        if type(item) is bool and item is False:
            continue
        if isinstance(item, list) and not item:
            continue
        raise StageDConfigError(f"{context}.{key} enables forbidden work")


def _validate_config_semantics(config: Mapping[str, Any]) -> None:
    _require(
        (config["schema_version"], config["document_type"], config["stage"], config["milestone"])
        == (1, "stage_d_fidelity_protocol_config", "D", "D2"),
        "Stage D config identity drifted",
    )
    protocol = config["protocol"]
    for key in ("path", "acceptance_path", "config_contract_path", "run_spec_path", "run_spec_schema_path"):
        _safe_repo_reference(protocol[key], f"config.protocol.{key}")
    _require(protocol["version"] == "1.0.0", "Stage D protocol version drifted")
    _require(protocol["result_status"] == "NOT_EVALUATED", "D2 cannot contain a result")

    authority = config["authority"]
    _require(
        authority["preparation_authorization"] == "APPROVED_TO_PREPARE_STAGE_D",
        "Stage D preparation authority drifted",
    )
    _require(authority["execution_authorization"] == "NOT_ISSUED", "D2 cannot issue execution authority")
    _require(authority["analysis_authorization"] == "NOT_EVALUATED", "D2 cannot issue analysis authority")
    _require(authority["stage_d_execution_started"] is False, "D2 cannot claim execution started")
    _require(authority["does_not_override_c10"] is True, "D2 must preserve C10 authority")
    _require(
        authority["authorized_next_task_after_d2_pass"]
        == "D3_CPU_CONTROL_PLANE_IMPLEMENTATION_ONLY",
        "D2 may authorize only the D3 CPU control plane",
    )

    upstream = config["upstream"]
    for section, keys in (
        (upstream["d0"], ("machine_record_path",)),
        (upstream["d1"], ("package_manifest_path", "registry_path", "attestation_path")),
    ):
        for key in keys:
            _safe_repo_reference(section[key], f"config.upstream.{key}")
    _safe_absolute_reference(upstream["c10"]["formal_root"], "config.upstream.c10.formal_root")
    _require(upstream["d1"]["candidate_recorded_image_hashes_only"] is True, "D2 must retain candidate-recorded-only image hashes")
    _require(
        upstream["d1"]["live_image_equality_status"] == "NOT_EVALUATED_D1_DEFERRED_TO_D8_0",
        "D2 cannot claim live reserve-image equality",
    )
    c10 = upstream["c10"]
    _require(
        (c10["primary_decision"], c10["stage_c_acceptance"], c10["workflow_blocker"], c10["spatial_status"])
        == ("NO_GO_NOVELTY", "BLOCKED", "HOLM_BLOCKED_UNDERSPECIFIED", "FAIL_SPATIAL"),
        "frozen C10 status was rewritten",
    )
    _require(c10["downstream_authorized"] is False and c10["stage_d_started"] is False, "C10 must remain unauthorized and not started")

    access = config["access_boundary"]
    _require(access["reserve_metadata_contents_accessed"] is True, "D2 metadata-access attestation drifted")
    for key in (
        "reserve_image_paths_statted", "reserve_image_bytes_accessed",
        "reserve_image_hashes_recomputed_from_live_bytes",
        "reserve_jpeg_decoded_or_preprocessed", "reserve_result_accessed",
        "stage_d_outcome_accessed", "model_loaded",
        "model_inference_performed", "model_or_gpu_used", "gpu_used",
        "runner_or_execution_chain_implemented", "smoke_or_formal_workload_run",
        "paper_data_or_figures_generated", "candidate_access_log_modified",
        "c9_or_c10_evidence_modified",
    ):
        _require(access[key] is False, f"D2 access boundary forbids config.access_boundary.{key}")

    cohort = config["cohort"]
    _require(cohort["registry_row_count"] == cohort["statistical_n"] == 500, "Stage D cohort denominator must remain 500")
    _require(
        (cohort["selection_order_min"], cohort["selection_order_max"]) == (0, 499)
        and cohort["selection_order_contiguous_and_unique"] is True,
        "Stage D selection order contract drifted",
    )
    _require(cohort["stage_c_overlap_counts_by_sample_id_path_image_hash"] == [0, 0, 0], "Stage C/Stage D overlap is forbidden")
    for key in ("c10_and_stage_d_pooling_allowed", "sample_replacement_allowed", "complete_case_analysis_allowed"):
        _require(cohort[key] is False, f"config.cohort.{key} must remain false")
    _require(cohort["misclassified_samples_retained"] is True, "misclassified samples must remain in the denominator")

    model = config["model_and_preprocessing"]
    for key in ("model_config_path", "requirements_lock_path"):
        _safe_repo_reference(model[key], f"config.model_and_preprocessing.{key}")

    primary = config["primary_scope"]
    methods = primary["methods_in_image_group_order"]
    _unique(methods, "config.primary_scope.methods_in_image_group_order")
    _require(
        (primary["block_index"], primary["head_index"], primary["target_id"], primary["endpoint_id"], primary["statistical_unit"])
        == (0, 9, "fixed_clean_attention_cls_head_direction", "fine_4x4_cell_delta_fid", "image"),
        "Stage D primary scope drifted",
    )
    _require(len(methods) == 2, "Stage D primary scope requires exactly two methods")

    target = config["target_and_map"]
    _require(target["baseline_uses_clean_fixed_attention"] is True and target["attention_direction_detached_and_frozen"] is True, "Stage D target must use frozen clean attention")
    _require(target["rgb_channels_replaced_together"] is True, "replacement must bind all RGB channels")
    _require(target["cls_contribution_retained_separately"] is True, "matched CLS contribution must be retained separately")
    _require(target["native_single_pixel_claim_allowed"] is False, "native-pixel claims are forbidden")
    height = model["center_crop_height"]
    width = model["center_crop_width"]
    _require(height % 4 == width % 4 == 0 and target["fine_cell_count"] == (height // 4) * (width // 4), "fine-cell geometry does not equal the frozen 4x4 partition")

    fidelity = config["fidelity"]
    fractions = fidelity["curve_fractions"]
    repeats = fidelity["tie_repeat_indices"]
    directions = fidelity["directions"]
    _strictly_increasing_numbers(fractions, "config.fidelity.curve_fractions")
    _strictly_increasing_numbers(repeats, "config.fidelity.tie_repeat_indices")
    _require(fractions[0] == 0 and fractions[-1] == 1, "fidelity fractions must span zero through one")
    _require(len(fractions) == fidelity["curve_fraction_count"], "curve fraction count contradicts its array")
    _require(len(repeats) == fidelity["tie_repeat_count"], "tie repeat count contradicts its array")
    _require(directions == ["insertion", "deletion"], "fidelity directions or order drifted")
    _require(fidelity["tie_priority_paired_across_methods_and_directions"] is True, "tie priority must remain paired")
    _require(fidelity["response_clipping_allowed"] is False, "response clipping is forbidden")
    _require(fidelity["within_image_auc_repeat_sd"] == "population_ddof_0_over_complete_five_repeats", "within-image repeat SD must use ddof=0")

    derived = config["registered_derived_metrics"]
    _require(derived["sufficiency_fraction"] == derived["comprehensiveness_fraction"], "registered derived fractions disagree")
    _require(derived["additional_curve_points_or_work_items"] == 0, "registered derived metrics cannot add work")

    audit = config["audit"]
    audit_count = audit["audit_image_count"]
    for key in ("audit_overlay_ids", "audit_sample_ids", "audit_selection_orders"):
        _require(len(audit[key]) == audit_count, f"config.audit.{key} count drifted")
        _unique(audit[key], f"config.audit.{key}")
    _require(all(type(value) is int and 0 <= value < cohort["registry_row_count"] for value in audit["audit_selection_orders"]), "audit selection order is outside the cohort")
    _require(audit["primary_clean_run_included"] is True, "audit repeatability must include the primary run")
    _require(audit["additional_repeat_clean_runs_per_method"] == audit["audit_repeatability_runs"] - 1, "audit repeat count is contradictory")
    _require(audit["statistical_n_increment"] == 0 and audit["audit_or_l2_enters_primary_statistics_or_decision"] is False, "audit/L2 cannot enter primary statistics")

    workload = config["workload"]
    counts = {
        "image_groups": cohort["registry_row_count"],
        "methods_per_image_group": len(methods),
    }
    counts["execution_units"] = counts["image_groups"] * counts["methods_per_image_group"]
    counts["primary_clean_maps"] = counts["execution_units"]
    counts["audit_repeat_clean_maps"] = audit_count * counts["methods_per_image_group"] * audit["additional_repeat_clean_runs_per_method"]
    counts["map_work_items"] = counts["primary_clean_maps"] + counts["audit_repeat_clean_maps"]
    counts["directions_per_unit"] = len(directions)
    counts["tie_repeats_per_direction"] = len(repeats)
    counts["fractions_per_curve"] = len(fractions)
    counts["fidelity_curves"] = counts["execution_units"] * counts["directions_per_unit"] * counts["tie_repeats_per_direction"]
    counts["fidelity_curve_points"] = counts["fidelity_curves"] * counts["fractions_per_curve"]
    counts["logical_work_items"] = counts["map_work_items"] + counts["fidelity_curve_points"]
    for key, expected in counts.items():
        _require(workload[key] == expected, f"config.workload.{key} contradicts independent expansion")
    _require(workload["logical_work_items_are_forward_or_backward_calls"] is False, "logical work items cannot be treated as model call counts")

    validity = config["validity"]
    _require(validity["required_image_coordinates"] == counts["image_groups"] and validity["required_unit_coordinates"] == counts["execution_units"], "validity coordinate denominator drifted")
    for key in ("direction_degeneration_allowed", "provenance_hash_or_contract_drift_allowed", "row_drop_imputation_or_replacement_allowed", "display_or_l2_as_quantitative_input_allowed"):
        _require(validity[key] is False, f"config.validity.{key} must remain false")
    _unique(validity["invalidating_na_reasons"], "config.validity.invalidating_na_reasons")

    statistics = config["statistics"]
    _require(statistics["confirmatory_n"] == cohort["statistical_n"] == statistics["bootstrap_cluster_size"], "statistics denominator drifted")
    _require(statistics["bootstrap_draw_shape"] == [statistics["bootstrap_resamples"], statistics["bootstrap_cluster_size"]], "bootstrap draw shape is contradictory")
    _require(statistics["ordered_sample_ids_sha256"] == upstream["d1"]["ordered_sample_ids_sha256"], "statistics/D1 sample-order digest drifted")
    _require(statistics["tie_repeats_are_independent_samples"] is False, "tie repeats cannot increase statistical n")
    _require(statistics["equivalence_test_performed"] is False, "no equivalence test is registered")
    _require(statistics["descriptive_curve_band"]["enters_primary_decision"] is False, "descriptive bands cannot enter the primary decision")

    rationale = config["sample_size_rationale"]
    gates = config["decision_gates"]
    _require(rationale["positive_count_minimum"] == gates["fidelity_pass_positive_count_minimum_inclusive"] == gates["fidelity_fail_if_positive_count_below"], "400/500 sign-guard thresholds disagree")
    _require(rationale["directional_consistency_fraction"] * cohort["statistical_n"] == rationale["positive_count_minimum"], "sample-size consistency fraction is contradictory")
    _require(rationale["formal_power_claim_made"] is False, "D2 cannot claim a formal power calculation")
    _require(gates["fidelity_evaluation_order"] == ["FAIL_FIDELITY", "PASS_FIDELITY", "INCONCLUSIVE_FIDELITY"], "fidelity decision order drifted")
    _require(gates["workflow_blocked_is_scientific_decision"] is False, "BLOCKED is not a scientific decision")

    secondary = config["secondary_and_spatial"]
    for key in ("run_stability", "spatial_in_primary_decision", "secondary_inference_enabled", "secondary_p_values_emitted", "secondary_winner_selection_performed", "repairs_c10_holm_blocker"):
        _require(secondary[key] is False, f"config.secondary_and_spatial.{key} must remain false")
    _require(secondary["holm_status"] == "NOT_APPLICABLE_NO_SECONDARY_INFERENCE", "Holm status contradicts the no-secondary contract")
    _require_disabled(config["forbidden_workload"], "config.forbidden_workload")

    retry = config["retry_and_recovery"]
    _unique(retry["retryable_infrastructure_failure_codes"], "config.retry_and_recovery.retryable_infrastructure_failure_codes")
    _unique(retry["nonretryable_failure_classes"], "config.retry_and_recovery.nonretryable_failure_classes")
    _require(set(retry["retryable_infrastructure_failure_codes"]).isdisjoint(retry["nonretryable_failure_classes"]), "retryable and nonretryable failure classes overlap")
    _require(retry["maximum_attempts_per_unit"] == retry["maximum_authorized_infrastructure_retries_per_unit"] + 1 == 2, "retry/attempt limits are contradictory")
    _require(retry["terminal_unit_overwrite_or_append_allowed"] is False and retry["cross_root_splicing_allowed"] is False, "immutable attempts and roots are mandatory")

    retention = config["retention"]
    _require(retention["l3_scope"] == "NONE" and retention["display_png_saved"] is False, "L3/display artifacts are forbidden")
    _require(retention["intermediate_result_metrics_in_cli_or_logs_allowed"] is False, "CLI/log result leakage is forbidden")
    provenance = config["provenance"]
    _unique(provenance["required_bindings"], "config.provenance.required_bindings")
    _require(provenance["ordinary_files_only_no_symlink_or_hardlink"] is True and provenance["source_recomputing_validation_required"] is True, "provenance safety requirements drifted")

    sharding = config["sharding"]
    _require(sharding["shard_count_status"] == "DEFERRED_TO_D7_BEFORE_EXECUTION_AUTHORIZATION", "D2 cannot choose a shard count")
    _require(sharding["group_assignment_formula"] == "selection_order_modulo_frozen_shard_count", "shard assignment formula drifted")
    _require(sharding["image_group_not_split_across_shards"] is True, "image groups cannot split across shards")

    cost = config["cost"]
    _require(cost["old_c10_pass_cost_authorizes_stage_d"] is False and cost["d7_independent_budget_required"] is True and cost["d7_may_loosen_caps"] is False, "D2 cost authority drifted")
    _require(cost["launch_safety_caps"]["runtime_breach_is_no_go_cost"] is False, "launch-safety breach cannot be relabeled as scientific cost failure")
    _require(cost["completed_campaign_cost_gate"]["evaluated_only_after_complete_campaign"] is True, "campaign cost cannot be evaluated early")
    _require(cost["d7_projection_formula"]["logical_work_items_used_as_call_count"] is False, "logical work items cannot estimate model calls")
    _require(cost["budget_status_at_d2"] == "NOT_EVALUATED", "D2 cannot contain a budget result")


def _validate_run_spec_semantics(run_spec: Mapping[str, Any]) -> None:
    _require(
        (run_spec["schema_version"], run_spec["document_type"], run_spec["stage"], run_spec["milestone"])
        == (1, "stage_d_fidelity_run_spec", "D", "D2"),
        "Stage D run-spec identity drifted",
    )
    binding = run_spec["config_binding"]
    _safe_repo_reference(binding["config_path"], "run_spec.config_binding.config_path")
    _require(
        binding == {
            "config_path": "config/stage_d.json",
            "config_file_sha256": STAGE_D_CONFIG_FILE_SHA256,
            "config_canonical_digest": STAGE_D_CONFIG_CANONICAL_DIGEST,
        },
        "run spec does not bind the frozen Stage D config",
    )
    registry = run_spec["registry_binding"]
    _safe_repo_reference(registry["registry_path"], "run_spec.registry_binding.registry_path")
    _require(registry["registry_file_sha256"] == _D1_REGISTRY_FILE_SHA256, "run-spec D1 registry file binding drifted")
    _require(registry["registry_canonical_digest"] == _D1_REGISTRY_CANONICAL_DIGEST, "run-spec D1 registry canonical binding drifted")
    _require(registry["ordered_sample_ids_sha256"] == _D1_ORDERED_SAMPLE_IDS_SHA256, "run-spec D1 sample-order binding drifted")

    selection = run_spec["selection"]
    sample_ids = selection["sample_ids"]
    _require(len(sample_ids) == selection["row_count"], "run-spec row count contradicts sample IDs")
    _unique(sample_ids, "run_spec.selection.sample_ids")
    for index, sample_id in enumerate(sample_ids):
        _safe_repo_reference(sample_id, f"run_spec.selection.sample_ids[{index}]")
    _require(
        (selection["selection_order_min"], selection["selection_order_max"])
        == (0, len(sample_ids) - 1),
        "run-spec selection-order range is contradictory",
    )
    _require(canonical_sha256(sample_ids) == registry["ordered_sample_ids_sha256"], "run-spec sample order does not match its canonical D1 digest")

    coordinates = run_spec["execution_coordinates"]
    methods = coordinates["method_ids"]
    _unique(methods, "run_spec.execution_coordinates.method_ids")
    _require(len(methods) == 2 and coordinates["method_order_within_image_group"] == "AS_LISTED", "run spec requires exactly two ordered methods")
    _require((coordinates["block_index"], coordinates["head_index"]) == (0, 9), "run-spec block/head drifted")

    fidelity = run_spec["fidelity"]
    fractions = fidelity["curve_fractions"]
    repeats = fidelity["tie_repeat_indices"]
    directions = fidelity["directions"]
    _strictly_increasing_numbers(fractions, "run_spec.fidelity.curve_fractions")
    _strictly_increasing_numbers(repeats, "run_spec.fidelity.tie_repeat_indices")
    _require(fractions[0] == 0 and fractions[-1] == 1, "run-spec fractions must span zero through one")
    _require(directions == ["insertion", "deletion"], "run-spec directions or order drifted")
    _require(fidelity["spatial_scope"] == "FINE_4X4_CELLS_ONLY" and fidelity["fine_cell_count"] == 3136, "run spec must remain fine-4x4 only")
    _require(fidelity["tie_priority_paired_across_methods_and_directions"] is True, "run-spec tie priority must remain paired")
    _require(fidelity["registered_derived_metrics_additional_curve_points_or_work_items"] == 0, "derived metrics cannot add run-spec work")

    audit = run_spec["audit"]
    audit_count = audit["audit_image_count"]
    for key in ("audit_overlay_ids", "audit_sample_ids", "audit_selection_orders"):
        _require(len(audit[key]) == audit_count, f"run_spec.audit.{key} count drifted")
        _unique(audit[key], f"run_spec.audit.{key}")
    for sample_id, order in zip(audit["audit_sample_ids"], audit["audit_selection_orders"]):
        _require(type(order) is int and 0 <= order < len(sample_ids), "run-spec audit selection order is invalid")
        _require(sample_ids[order] == sample_id, "run-spec audit sample/order binding drifted")
    _require(audit["primary_clean_run_included"] is True, "audit primary clean run must be included")
    _require(audit["additional_repeat_clean_runs_per_method"] == audit["audit_repeatability_runs"] - 1, "run-spec audit repeat counts contradict")
    _require(audit["statistical_n_increment"] == 0, "audit overlays cannot increase statistical n")

    workload = run_spec["workload"]
    expected = {
        "image_groups": len(sample_ids),
        "methods_per_image_group": len(methods),
    }
    expected["execution_units"] = expected["image_groups"] * expected["methods_per_image_group"]
    expected["primary_clean_maps"] = expected["execution_units"]
    expected["audit_repeat_clean_maps"] = audit_count * len(methods) * audit["additional_repeat_clean_runs_per_method"]
    expected["map_work_items"] = expected["primary_clean_maps"] + expected["audit_repeat_clean_maps"]
    expected["fidelity_curves"] = expected["execution_units"] * len(directions) * len(repeats)
    expected["fidelity_curve_points"] = expected["fidelity_curves"] * len(fractions)
    expected["logical_work_items"] = expected["map_work_items"] + expected["fidelity_curve_points"]
    for key, value in expected.items():
        _require(workload[key] == value, f"run_spec.workload.{key} contradicts independent expansion")
    _require(workload["logical_work_items_are_forward_or_backward_calls"] is False, "run-spec logical work items cannot be model call counts")
    _require_disabled(run_spec["forbidden_workload"], "run_spec.forbidden_workload")

    sharding = run_spec["sharding"]
    _require(sharding["shard_count_status"] == "DEFERRED_TO_D7_BEFORE_EXECUTION_AUTHORIZATION", "run spec cannot choose a shard count at D2")
    _require(sharding["group_assignment_formula"] == "selection_order_modulo_frozen_shard_count", "run-spec shard formula drifted")
    _require(sharding["image_group_not_split_across_shards"] is True, "run-spec image groups cannot split across shards")


def _validate_cross_bindings(config: StageDConfig, run_spec: StageDRunSpec) -> None:
    document = config.to_dict()
    request = run_spec.to_dict()
    binding = request["config_binding"]
    _require(binding["config_file_sha256"] == config.file_sha256, "run spec config file SHA-256 does not match loaded config bytes")
    _require(binding["config_canonical_digest"] == config.canonical_digest, "run spec config canonical digest does not match loaded config")
    _require(document["protocol"]["run_spec_path"] == "config/stage_d_run_spec.json" and document["protocol"]["run_spec_schema_path"] == "config/stage_d_run_spec.schema.json", "config protocol run-spec paths drifted")

    d1 = document["upstream"]["d1"]
    expected_registry = {
        "d1_completion_ref": d1["completion_ref"],
        "d1_completion_commit": d1["completion_commit"],
        "package_id": d1["package_id"],
        "registry_path": d1["registry_path"],
        "registry_file_sha256": d1["registry_file_sha256"],
        "registry_canonical_digest": d1["registry_canonical_digest"],
        "ordered_sample_ids_sha256": d1["ordered_sample_ids_sha256"],
    }
    _require(request["registry_binding"] == expected_registry, "config/run-spec D1 registry bindings disagree")

    cohort = document["cohort"]
    selection = request["selection"]
    _require(
        (selection["row_count"], selection["selection_order_min"], selection["selection_order_max"])
        == (cohort["registry_row_count"], cohort["selection_order_min"], cohort["selection_order_max"]),
        "config/run-spec cohort bindings disagree",
    )
    primary = document["primary_scope"]
    coordinates = request["execution_coordinates"]
    for config_key, run_key in (
        ("block_index", "block_index"), ("head_index", "head_index"),
        ("target_id", "target_id"), ("statistical_unit", "statistical_unit"),
    ):
        _require(primary[config_key] == coordinates[run_key], f"config/run-spec {run_key} binding disagrees")
    _require(primary["methods_in_image_group_order"] == coordinates["method_ids"], "config/run-spec method order disagrees")

    configured_fidelity = document["fidelity"]
    requested_fidelity = request["fidelity"]
    _require(primary["endpoint_id"] == requested_fidelity["endpoint_id"], "config/run-spec endpoint disagrees")
    _require(document["target_and_map"]["fine_cell_count"] == requested_fidelity["fine_cell_count"], "config/run-spec fine-cell count disagrees")
    for key in ("directions", "curve_fractions", "tie_repeat_indices", "tie_priority_paired_across_methods_and_directions"):
        _require(configured_fidelity[key] == requested_fidelity[key], f"config/run-spec fidelity.{key} disagrees")
    derived = document["registered_derived_metrics"]
    _require(requested_fidelity["registered_derived_fraction"] == derived["sufficiency_fraction"] == derived["comprehensiveness_fraction"], "config/run-spec derived fraction disagrees")
    _require(requested_fidelity["registered_derived_metrics_additional_curve_points_or_work_items"] == derived["additional_curve_points_or_work_items"], "config/run-spec derived-work count disagrees")

    for key in ("audit_image_count", "audit_overlay_ids", "audit_sample_ids", "audit_selection_orders", "audit_repeatability_runs", "primary_clean_run_included", "additional_repeat_clean_runs_per_method", "repeatability_scope", "statistical_n_increment"):
        _require(document["audit"][key] == request["audit"][key], f"config/run-spec audit.{key} disagrees")
    for key, value in request["workload"].items():
        _require(document["workload"][key] == value, f"config/run-spec workload.{key} disagrees")

    forbidden_pairs = (
        ("run_random_sanity", "random_sanity"),
        ("run_center_sanity", "center_sanity"),
        ("run_patch_scale_fidelity", "patch_scale_fidelity"),
        ("run_spatial_or_stability_endpoints", "spatial_or_stability_endpoints"),
        ("run_secondary_inference", "secondary_inference"),
        ("run_display_pipeline", "display_pipeline"),
        ("other_method_ids", "other_methods"),
        ("other_block_head_coordinates", "other_blocks_or_heads"),
        ("class_logit_method_ids", "class_logit_methods"),
    )
    for run_key, config_key in forbidden_pairs:
        _require(request["forbidden_workload"][run_key] == document["forbidden_workload"][config_key], f"config/run-spec forbidden scope {run_key} disagrees")
    _require(request["forbidden_workload"]["run_stability"] == document["secondary_and_spatial"]["run_stability"], "config/run-spec stability scope disagrees")
    _require(request["sharding"] == document["sharding"], "config/run-spec sharding contract disagrees")


def _load_validated_document(
    *,
    path: Path | str,
    schema_path: Path | str,
    repo_root: Path | str,
    expected_file_sha256: str,
    expected_schema_sha256: str,
    expected_canonical_digest: str,
    context: str,
    semantic_validator: Any,
    document_type: type[_FrozenDocument],
) -> _FrozenDocument:
    bound = _safe_read_relative(repo_root, path, context)
    schema_bound = _safe_read_relative(repo_root, schema_path, f"{context} schema")
    if schema_bound.file_sha256 != expected_schema_sha256:
        raise StageDConfigError(
            f"{context} schema file SHA-256 drifted: expected "
            f"{expected_schema_sha256}, observed {schema_bound.file_sha256}"
        )
    document = _parse_bound_json(bound, context)
    schema = _parse_bound_json(schema_bound, f"{context} schema")
    validator = _validate_schema_document(schema, f"{context} schema")
    _validate_instance(document, schema, validator, context)
    _validate_self_digest(document, expected_canonical_digest, context)
    semantic_validator(document)
    if bound.file_sha256 != expected_file_sha256:
        raise StageDConfigError(
            f"{context} file SHA-256 drifted: expected {expected_file_sha256}, "
            f"observed {bound.file_sha256}"
        )
    return document_type(
        _data=document, source_path=bound.source_path,
        file_sha256=bound.file_sha256, size_bytes=bound.size_bytes,
        schema_path=schema_bound.source_path,
        schema_file_sha256=schema_bound.file_sha256,
        _issuer=_DOCUMENT_ISSUER,
    )


def _rebase_default_path(
    path: Path | str, default: Path, relative: str, repo_root: Path | str,
) -> Path | str:
    """Interpret exported absolute defaults relative to a custom repo root."""

    requested = Path(os.path.abspath(os.fspath(path)))
    frozen_default = Path(os.path.abspath(os.fspath(default)))
    root = Path(os.path.abspath(os.fspath(repo_root)))
    frozen_root = Path(os.path.abspath(os.fspath(REPO_ROOT)))
    if requested == frozen_default and root != frozen_root:
        return relative
    return path


def load_stage_d_config(
    path: Path | str = STAGE_D_CONFIG_PATH,
    *,
    schema_path: Path | str = STAGE_D_CONFIG_SCHEMA_PATH,
    repo_root: Path | str = REPO_ROOT,
) -> StageDConfig:
    """Safely load and validate the exact frozen Stage D config."""

    path = _rebase_default_path(
        path, STAGE_D_CONFIG_PATH, "config/stage_d.json", repo_root,
    )
    schema_path = _rebase_default_path(
        schema_path, STAGE_D_CONFIG_SCHEMA_PATH,
        "config/stage_d.schema.json", repo_root,
    )
    result = _load_validated_document(
        path=path, schema_path=schema_path, repo_root=repo_root,
        expected_file_sha256=STAGE_D_CONFIG_FILE_SHA256,
        expected_schema_sha256=STAGE_D_CONFIG_SCHEMA_FILE_SHA256,
        expected_canonical_digest=STAGE_D_CONFIG_CANONICAL_DIGEST,
        context="Stage D config", semantic_validator=_validate_config_semantics,
        document_type=StageDConfig,
    )
    if not isinstance(result, StageDConfig):  # defensive typing boundary
        raise StageDConfigError("internal Stage D config wrapper error")
    return result


def load_stage_d_run_spec(
    path: Path | str = STAGE_D_RUN_SPEC_PATH,
    *,
    schema_path: Path | str = STAGE_D_RUN_SPEC_SCHEMA_PATH,
    repo_root: Path | str = REPO_ROOT,
    config: StageDConfig | None = None,
) -> StageDRunSpec:
    """Safely load the frozen run spec and optionally bind it to a config."""

    path = _rebase_default_path(
        path, STAGE_D_RUN_SPEC_PATH, "config/stage_d_run_spec.json", repo_root,
    )
    schema_path = _rebase_default_path(
        schema_path, STAGE_D_RUN_SPEC_SCHEMA_PATH,
        "config/stage_d_run_spec.schema.json", repo_root,
    )
    result = _load_validated_document(
        path=path, schema_path=schema_path, repo_root=repo_root,
        expected_file_sha256=STAGE_D_RUN_SPEC_FILE_SHA256,
        expected_schema_sha256=STAGE_D_RUN_SPEC_SCHEMA_FILE_SHA256,
        expected_canonical_digest=STAGE_D_RUN_SPEC_CANONICAL_DIGEST,
        context="Stage D run spec", semantic_validator=_validate_run_spec_semantics,
        document_type=StageDRunSpec,
    )
    if not isinstance(result, StageDRunSpec):  # defensive typing boundary
        raise StageDConfigError("internal Stage D run-spec wrapper error")
    if config is not None:
        if not isinstance(config, StageDConfig):
            raise StageDConfigError("config must be a validated StageDConfig")
        _validate_cross_bindings(config, result)
    return result


def load_stage_d_contracts(
    config_path: Path | str = STAGE_D_CONFIG_PATH,
    run_spec_path: Path | str = STAGE_D_RUN_SPEC_PATH,
    *,
    config_schema_path: Path | str = STAGE_D_CONFIG_SCHEMA_PATH,
    run_spec_schema_path: Path | str = STAGE_D_RUN_SPEC_SCHEMA_PATH,
    repo_root: Path | str = REPO_ROOT,
) -> StageDContracts:
    """Load and cross-validate the frozen Stage D config/run-spec bundle."""

    config = load_stage_d_config(
        config_path, schema_path=config_schema_path, repo_root=repo_root,
    )
    run_spec = load_stage_d_run_spec(
        run_spec_path, schema_path=run_spec_schema_path, repo_root=repo_root,
        config=config,
    )
    return StageDContracts(config=config, run_spec=run_spec)


__all__ = [
    "DEFAULT_STAGE_D_CONFIG", "DEFAULT_STAGE_D_RUN_SPEC",
    "DEFAULT_STAGE_D_RUN_SPEC_SCHEMA", "DEFAULT_STAGE_D_SCHEMA",
    "STAGE_D_CONFIG_CANONICAL_DIGEST", "STAGE_D_CONFIG_FILE_SHA256",
    "STAGE_D_CONFIG_PATH", "STAGE_D_CONFIG_SCHEMA_FILE_SHA256",
    "STAGE_D_CONFIG_SCHEMA_PATH", "STAGE_D_CONFIG_SCHEMA_SHA256",
    "STAGE_D_CONFIG_SHA256", "STAGE_D_RUN_SPEC_CANONICAL_DIGEST",
    "STAGE_D_RUN_SPEC_FILE_SHA256", "STAGE_D_RUN_SPEC_PATH",
    "STAGE_D_RUN_SPEC_SCHEMA_FILE_SHA256",
    "STAGE_D_RUN_SPEC_SCHEMA_PATH", "STAGE_D_RUN_SPEC_SCHEMA_SHA256",
    "STAGE_D_RUN_SPEC_SHA256", "StageDConfig", "StageDConfigError",
    "StageDContractError", "StageDContracts", "StageDRunSpec",
    "load_stage_d_config", "load_stage_d_contracts",
    "load_stage_d_run_spec", "strict_json_bytes",
]
