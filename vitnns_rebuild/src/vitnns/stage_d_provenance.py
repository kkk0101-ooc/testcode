"""Stage-D-specific provenance for one immutable execution attempt.

The document intentionally does not import, wrap, or identify a Stage C
artifact.  It is an acyclic inner document: provenance binds the scientific
payload inventory, the attempt manifest binds the provenance bytes, and the
status commit marker binds the manifest bytes.

This module is metadata-only.  It never opens an image, model, state dict, or
outcome file.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from .stage_d_plan import (
    BLOCK_INDEX,
    CURVE_FRACTIONS,
    HEAD_INDEX,
    METHOD_IDS,
    TARGET_ID,
    DIRECTIONS,
    TIE_REPEAT_INDICES,
    StageDCurvePoint,
    StageDImageGroup,
    StageDMapCoordinate,
    StageDUnit,
)


SCHEMA_VERSION = 1
DOCUMENT_TYPE = "stage_d_execution_unit_provenance"
FORMAL_MODE = "formal"
SYNTHETIC_MODE = "SYNTHETIC_D4_TEST_ONLY"
D7_NOT_ISSUED = "NOT_ISSUED"
D7_AUTHORIZED = "AUTHORIZED_STAGE_D_FIDELITY_SCOPE"
SYNTHETIC_IMAGE_STATUS = "SYNTHETIC_NO_IMAGE_ACCESSED"
FORMAL_IMAGE_STATUS = "VERIFIED_AFTER_D7_AUTHORIZATION"
SUCCESS_STATUS = "succeeded"
SCIENTIFIC_FAILURE_STATUS = "scientific_failure"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_OBJECT_RE = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
_SAFE_ARRAY_KEY_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]*$")
_AUDIT_REPEAT_MAP_KEY = "repeat_clean_metric_ready_map"
_ISSUER = object()

_ROOT_FIELDS = {
    "schema_version", "document_type", "mode", "unit_identity",
    "unit_identity_sha256", "bindings", "canonical_digest",
}
_BINDING_FIELDS = {
    "d7_authorization", "producer", "dependencies", "model",
    "source_bindings", "coordinates", "image_identity",
    "scientific_contract", "execution", "terminal", "failure",
    "resources", "artifact_inventory",
}
_COORDINATE_FIELDS = {
    "plan_digest", "group_digest", "pair_digest", "unit_digest",
    "work_item_digests", "work_item_count",
    "work_item_sequence_digest",
}
_SCIENTIFIC_FIELDS = {
    "method", "target", "block_index", "head_index", "operator",
    "projection", "sign_semantics", "normalization", "baseline",
    "fractions", "seed_contract",
}
_EXECUTION_FIELDS = {"attempt", "retry", "claim", "shard"}
_TERMINAL_FIELDS = {"status", "terminal", "complete"}
_RESOURCE_FIELDS = {
    "elapsed_seconds", "forward_calls", "backward_calls",
    "l0_l1_payload_bytes", "l2_payload_bytes",
    "cuda_peak_reserved_bytes",
}
_INVENTORY_FIELDS = {
    "level", "path", "kind", "format", "dtype", "shape",
    "bytes", "sha256",
}
_PAYLOAD_PATHS = (
    "L0/result.json",
    "L1/metric_ready_map.npy",
    "L1/fidelity_curves.npz",
)
_AUDIT_PATH = "L2/audit_evidence.npz"
FAILURE_PATH = "L0/failure.json"
FAILURE_DOCUMENT_TYPE = "stage_d_scientific_failure_evidence"
SCIENTIFIC_FAILURE_SOURCE = "stage_d_scientific_execution"


class StageDProvenanceError(ValueError):
    """Raised when Stage D provenance is incomplete or inconsistent."""


def _plain(value: Any, context: str = "value") -> Any:
    converter = getattr(value, "to_dict", None)
    if callable(converter) and not isinstance(value, Mapping):
        value = converter()
    if value is None or type(value) in {str, bool, int}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise StageDProvenanceError(
                f"{context} contains a non-finite number"
            )
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, child in value.items():
            if not isinstance(key, str) or not key:
                raise StageDProvenanceError(
                    f"{context} keys must be non-empty strings"
                )
            result[key] = _plain(child, f"{context}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [
            _plain(child, f"{context}[{index}]")
            for index, child in enumerate(value)
        ]
    raise StageDProvenanceError(
        f"{context} is not JSON-compatible: {type(value).__name__}"
    )


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(child) for key, child in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(child) for child in value)
    if isinstance(value, tuple):
        return tuple(_freeze(child) for child in value)
    return copy.deepcopy(value)


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [_thaw(child) for child in value]
    return copy.deepcopy(value)


def canonical_json_bytes(value: Any) -> bytes:
    """Return compact, sorted-key, finite UTF-8 JSON without a newline."""

    plain = _plain(value, "canonical JSON input")
    return json.dumps(
        plain, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_file_bytes(value: Any) -> bytes:
    """Return canonical JSON bytes with the required terminal newline."""

    return canonical_json_bytes(value) + b"\n"


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _exact(value: Any, fields: set[str], context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise StageDProvenanceError(f"{context} must be an object")
    actual = set(value)
    if actual != fields:
        raise StageDProvenanceError(
            f"{context} fields differ: missing={sorted(fields - actual)}, "
            f"unknown={sorted(actual - fields)}"
        )
    return value


def _nonempty_mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise StageDProvenanceError(f"{context} must be a non-empty object")
    return value


def _sha256(value: Any, context: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise StageDProvenanceError(f"{context} must be a lowercase SHA-256")
    return value


def _binding_digest(value: Any, context: str) -> str:
    if isinstance(value, str):
        return _sha256(value, context)
    if not isinstance(value, Mapping):
        candidate = getattr(value, "canonical_digest", None)
        if isinstance(candidate, str):
            return _sha256(candidate, context)
        converter = getattr(value, "to_dict", None)
        if callable(converter):
            value = converter()
    if isinstance(value, Mapping):
        for key in ("canonical_digest", "digest", "sha256"):
            candidate = value.get(key)
            if isinstance(candidate, str):
                return _sha256(candidate, context)
    raise StageDProvenanceError(f"{context} has no content digest")


def _typed_unit(value: Any, context: str = "unit") -> StageDUnit:
    if isinstance(value, StageDUnit):
        return value
    document = _plain(value, context)
    if not isinstance(document, Mapping):
        raise StageDProvenanceError(f"{context} must be a StageDUnit")
    try:
        return StageDUnit.from_dict(document)
    except Exception as error:
        raise StageDProvenanceError(f"{context} is invalid: {error}") from error


def _typed_group(value: Any, context: str = "group") -> StageDImageGroup:
    if isinstance(value, StageDImageGroup):
        return value
    document = _plain(value, context)
    if not isinstance(document, Mapping):
        raise StageDProvenanceError(f"{context} must be a StageDImageGroup")
    try:
        return StageDImageGroup.from_dict(document)
    except Exception as error:
        raise StageDProvenanceError(f"{context} is invalid: {error}") from error


def _unit_work_item_digests(
    unit: StageDUnit, group: StageDImageGroup | None = None,
) -> tuple[str, ...]:
    matches = [StageDMapCoordinate.create(unit, repeat=False).canonical_digest]
    if unit.audit:
        matches.append(StageDMapCoordinate.create(unit, repeat=True).canonical_digest)
    matches.extend(
        StageDCurvePoint.create(
            unit, direction=direction, repeat_index=repeat_index,
            fraction_index=fraction_index,
        ).canonical_digest
        for direction in DIRECTIONS
        for repeat_index in TIE_REPEAT_INDICES
        for fraction_index in range(len(CURVE_FRACTIONS))
    )
    expected_count = 212 if unit.audit else 211
    if len(matches) != expected_count or len(set(matches)) != expected_count:
        raise StageDProvenanceError(
            "unit/group work coordinates are incomplete or duplicated"
        )
    if group is not None and unit.canonical_digest not in {
        item.canonical_digest for item in group.units
    }:
        raise StageDProvenanceError("unit is not a member of the image group")
    return tuple(matches)


def _coerce_work_item_digests(
    value: Sequence[Any] | None, unit: StageDUnit, group: StageDImageGroup,
) -> tuple[str, ...]:
    expected = _unit_work_item_digests(unit, group)
    if value is None:
        return expected
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise StageDProvenanceError("work_items must be a sequence")
    observed = tuple(
        _binding_digest(item, f"work_items[{index}]")
        for index, item in enumerate(value)
    )
    if observed != expected:
        raise StageDProvenanceError(
            "work_items do not exactly cover the unit/group coordinates"
        )
    return observed


def _normalize_named(value: Any, name: str, context: str) -> dict[str, Any]:
    if isinstance(value, str):
        result = {name: value}
    else:
        result = _plain(value, context)
    if not isinstance(result, dict) or not result:
        raise StageDProvenanceError(f"{context} must be non-empty")
    candidate = result.get(name)
    if not isinstance(candidate, str) or not candidate:
        raise StageDProvenanceError(f"{context}.{name} must be non-empty")
    return result


def _reject_forbidden_identity(value: Any, context: str) -> None:
    """Reject accidental cross-stage identity and unregistered scope."""

    forbidden_scope = (
        "random_sanity", "center_sanity", "patch_scale",
        "run_stability", "class_logit_method", "display_pipeline",
    )
    if isinstance(value, Mapping):
        for key, child in value.items():
            folded = key.casefold()
            if "stage_c" in folded or "stage c" in folded:
                raise StageDProvenanceError(
                    f"{context}.{key} contains a forbidden foreign-stage identity"
                )
            if any(token in folded for token in forbidden_scope):
                raise StageDProvenanceError(
                    f"{context}.{key} contains forbidden extra scientific scope"
                )
            _reject_forbidden_identity(child, f"{context}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_forbidden_identity(child, f"{context}[{index}]")
    elif isinstance(value, str):
        folded = value.casefold()
        if "stage_c" in folded or "stage c" in folded:
            raise StageDProvenanceError(
                f"{context} contains a forbidden foreign-stage identity"
            )


def _validate_digest_fields(value: Any, context: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            lowered = key.casefold()
            if lowered.endswith("sha256") or lowered.endswith("digest"):
                if child is not None:
                    _sha256(child, f"{context}.{key}")
            else:
                _validate_digest_fields(child, f"{context}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _validate_digest_fields(child, f"{context}[{index}]")


def _provenance_resources(value: Any) -> dict[str, Any]:
    """Keep post-publication control bytes outside the provenance cycle."""

    result = _plain(value, "resources")
    if not isinstance(result, dict):
        raise StageDProvenanceError("resources must be an object")
    control_bytes = result.pop("control_and_provenance_bytes", None)
    if control_bytes not in {None, 0}:
        raise StageDProvenanceError(
            "control_and_provenance_bytes is disk-derived after publication"
        )
    # Small direct callers may provide a subset.  Production supplies the full
    # resource projection; unknown or self-referential fields remain forbidden.
    if not result or set(result) - _RESOURCE_FIELDS:
        raise StageDProvenanceError(
            "bindings.resources has missing core data or unknown fields"
        )
    for name, raw in result.items():
        if name == "elapsed_seconds":
            if type(raw) not in {int, float} or not math.isfinite(float(raw)) or raw < 0:
                raise StageDProvenanceError(f"bindings.resources.{name} is invalid")
            result[name] = float(raw)
        elif type(raw) is not int or raw < 0:
            raise StageDProvenanceError(f"bindings.resources.{name} is invalid")
    return result


def _attempt_number(value: Mapping[str, Any]) -> int:
    number = value.get("number", value.get("attempt_number"))
    if type(number) is not int or number not in {1, 2}:
        raise StageDProvenanceError(
            "bindings.execution.attempt must bind attempt number 1 or 2"
        )
    return number


def _scientific_paths(*, audit: bool) -> list[str]:
    return list(_PAYLOAD_PATHS) + ([_AUDIT_PATH] if audit else [])


def normalize_stage_d_scientific_failure(
    value: Any, *, audit: bool, produced: Sequence[str],
) -> dict[str, Any]:
    """Normalize one phase-specific failure against produced artifacts.

    ``produced`` names only scientific L0/L1/L2 payloads; the mandatory
    failure evidence and provenance files are completion controls and are not
    listed as scientific products.
    """

    context = "bindings.failure"
    if not isinstance(value, Mapping):
        raise StageDProvenanceError(
            "scientific failure requires a structured failure object"
        )
    allowed = {"code", "phase", "source", "produced", "absent", "details"}
    minimal = {"code", "phase"}
    if not minimal <= set(value) or set(value) - allowed:
        raise StageDProvenanceError(
            f"{context} fields differ: missing={sorted(minimal - set(value))}, "
            f"unknown={sorted(set(value) - allowed)}"
        )
    for name in ("code", "phase"):
        if not isinstance(value[name], str) or not value[name]:
            raise StageDProvenanceError(f"{context}.{name} must be non-empty")
    source = value.get("source", SCIENTIFIC_FAILURE_SOURCE)
    if source != SCIENTIFIC_FAILURE_SOURCE:
        raise StageDProvenanceError(
            f"{context}.source must identify Stage D scientific execution"
        )
    if (
        isinstance(produced, (str, bytes))
        or not isinstance(produced, Sequence)
        or any(not isinstance(path, str) for path in produced)
    ):
        raise StageDProvenanceError(
            f"{context}.produced must be an ordered array"
        )
    expected = _scientific_paths(audit=audit)
    produced_list = list(produced)
    expected_produced = [path for path in expected if path in produced_list]
    expected_absent = [path for path in expected if path not in produced_list]
    if produced_list != expected_produced or len(set(produced_list)) != len(produced_list):
        raise StageDProvenanceError(
            "scientific failure produced artifacts are reordered, duplicated, "
            "or outside the closed scientific layout"
        )
    if "produced" in value and value["produced"] != expected_produced:
        raise StageDProvenanceError(
            "scientific failure produced declaration disagrees with inventory"
        )
    if "absent" in value and value["absent"] != expected_absent:
        raise StageDProvenanceError(
            "scientific failure absent declaration disagrees with inventory"
        )
    result = {
        "code": value["code"], "phase": value["phase"],
        "source": source, "produced": expected_produced,
        "absent": expected_absent,
    }
    if "details" in value:
        result["details"] = _plain(value["details"], f"{context}.details")
    return result


def stage_d_scientific_failure_artifact(
    failure: Mapping[str, Any], *, l0_scalars: Mapping[str, Any], audit: bool,
    produced: Sequence[str],
) -> tuple[dict[str, Any], dict[str, Any], bytes]:
    """Return normalized failure, exact inventory record, and file bytes."""

    normalized = normalize_stage_d_scientific_failure(
        failure, audit=audit, produced=produced
    )
    scalars = _plain(l0_scalars, "scientific failure l0_scalars")
    if not isinstance(scalars, dict):
        raise StageDProvenanceError(
            "scientific failure l0_scalars must be an object"
        )
    document = {
        "schema_version": SCHEMA_VERSION,
        "document_type": FAILURE_DOCUMENT_TYPE,
        "failure": normalized,
        "l0_scalars": scalars,
    }
    raw = canonical_json_file_bytes(document)
    record = {
        "level": "L0", "path": FAILURE_PATH,
        "kind": "scientific_failure", "format": "json",
        "dtype": "json", "shape": None, "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    return normalized, record, raw


def _validate_inventory(
    value: Any, *, audit: bool, terminal_status: str, failure: Any,
) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, (list, tuple)):
        raise StageDProvenanceError(
            "bindings.artifact_inventory must be an ordered array"
        )
    scientific_paths = _scientific_paths(audit=audit)
    if terminal_status == SUCCESS_STATUS:
        if failure is not None:
            raise StageDProvenanceError(
                "successful provenance cannot contain failure evidence"
            )
        expected_paths = scientific_paths
    elif terminal_status == SCIENTIFIC_FAILURE_STATUS:
        failure_record = normalize_stage_d_scientific_failure(
            failure, audit=audit, produced=failure.get("produced", [])
            if isinstance(failure, Mapping) else [],
        )
        if dict(failure) != failure_record:
            raise StageDProvenanceError(
                "scientific failure must persist normalized source/produced/absent fields"
            )
        expected_paths = [*failure_record["produced"], FAILURE_PATH]
    else:  # validated by the caller; defensive fail closed here too.
        raise StageDProvenanceError("terminal status is invalid for inventory")
    records: list[dict[str, Any]] = []
    for index, raw in enumerate(value):
        item = _exact(raw, _INVENTORY_FIELDS, f"artifact_inventory[{index}]")
        record = _plain(item, f"artifact_inventory[{index}]")
        path = record["path"]
        if not isinstance(path, str) or path not in {
            *scientific_paths, FAILURE_PATH,
        }:
            raise StageDProvenanceError(
                f"artifact_inventory[{index}].path is outside the closed layout"
            )
        if record["level"] != path.split("/", 1)[0]:
            raise StageDProvenanceError(
                f"artifact_inventory[{index}].level disagrees with path"
            )
        if type(record["bytes"]) is not int or record["bytes"] < 0:
            raise StageDProvenanceError(
                f"artifact_inventory[{index}].bytes is invalid"
            )
        _sha256(record["sha256"], f"artifact_inventory[{index}].sha256")
        records.append(record)
    if [item["path"] for item in records] != expected_paths:
        raise StageDProvenanceError(
            "artifact inventory does not exactly cover the closed payload layout"
        )
    expected_contracts = {
        "L0/result.json": ("L0", "result", "json", "json", None),
        "L1/metric_ready_map.npy": (
            "L1", "metric_ready_map", "npy", "float32", [56, 56],
        ),
        "L1/fidelity_curves.npz": (
            "L1", "fidelity_curves", "npz", "float64", [2, 5, 21],
        ),
        FAILURE_PATH: ("L0", "scientific_failure", "json", "json", None),
    }
    for record in records:
        expected = expected_contracts.get(record["path"])
        if expected is not None:
            observed = tuple(record[key] for key in (
                "level", "kind", "format", "dtype", "shape",
            ))
            if observed != expected:
                raise StageDProvenanceError(
                    f"{record['path']} inventory contract is invalid"
                )
        elif (
            record["level"] != "L2"
            or record["kind"] != "audit_evidence"
            or record["format"] != "npz"
            or not isinstance(record["dtype"], Mapping)
            or not isinstance(record["shape"], Mapping)
            or set(record["dtype"]) != set(record["shape"])
            or not record["dtype"]
            or any(_SAFE_ARRAY_KEY_RE.fullmatch(key) is None for key in record["dtype"])
        ):
            raise StageDProvenanceError(
                "L2 audit evidence inventory contract is invalid"
            )
        elif (
            _AUDIT_REPEAT_MAP_KEY not in record["dtype"]
            or record["dtype"][_AUDIT_REPEAT_MAP_KEY] != "float32"
            or record["shape"][_AUDIT_REPEAT_MAP_KEY] != [56, 56]
        ):
            raise StageDProvenanceError(
                "L2 audit evidence lacks the required repeat-clean metric-ready map"
            )
    return tuple(records)


@dataclass(frozen=True, slots=True)
class StageDProvenance(Mapping[str, Any]):
    """Deeply immutable validated provenance document."""

    _document: Mapping[str, Any] = field(repr=False)
    _issuer: object = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._issuer is not _ISSUER:
            raise StageDProvenanceError(
                "StageDProvenance must be created by the builder or validator"
            )
        object.__setattr__(self, "_document", _freeze(_plain(self._document)))

    def to_dict(self) -> dict[str, Any]:
        return _thaw(self._document)

    @property
    def canonical_digest(self) -> str:
        return self._document["canonical_digest"]

    @property
    def mode(self) -> str:
        return self._document["mode"]

    @property
    def unit_identity_sha256(self) -> str:
        return self._document["unit_identity_sha256"]

    def __getitem__(self, key: str) -> Any:
        return self._document[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._document)

    def __len__(self) -> int:
        return len(self._document)


def build_stage_d_provenance(
    *,
    mode: str,
    producer: Mapping[str, Any],
    dependencies: Mapping[str, Any],
    model: Mapping[str, Any],
    source_bindings: Mapping[str, Any],
    plan: Any,
    group: StageDImageGroup | Mapping[str, Any],
    unit: StageDUnit | Mapping[str, Any],
    image_identity: Mapping[str, Any],
    method: Mapping[str, Any] | str,
    target: Mapping[str, Any] | str,
    operator: Any,
    projection: Any,
    sign_semantics: Any,
    normalization: Any,
    baseline: Any,
    seed_contract: Mapping[str, Any],
    attempt: Mapping[str, Any] | int,
    retry: Mapping[str, Any],
    claim: Mapping[str, Any],
    shard: Mapping[str, Any],
    terminal: Mapping[str, Any],
    failure: Mapping[str, Any] | None,
    resources: Mapping[str, Any],
    artifact_inventory: Sequence[Mapping[str, Any]],
    failure_l0_scalars: Mapping[str, Any] | None = None,
    d7_authorization: Mapping[str, Any] | None = None,
    pair: Any | None = None,
    work_items: Sequence[Any] | None = None,
    fractions: Sequence[int | float] = CURVE_FRACTIONS,
) -> StageDProvenance:
    """Build and validate provenance from runner-owned metadata records."""

    typed_unit = _typed_unit(unit)
    typed_group = _typed_group(group)
    work_digests = _coerce_work_item_digests(
        work_items, typed_unit, typed_group
    )
    if pair is not None and _binding_digest(pair, "pair") != (
        typed_group.pair_binding.canonical_digest
    ):
        raise StageDProvenanceError("pair binding differs from the image group")
    if isinstance(attempt, int) and not isinstance(attempt, bool):
        attempt_value: Mapping[str, Any] = {"number": attempt}
    else:
        attempt_value = attempt
    authorization = (
        {"status": D7_NOT_ISSUED, "digest": None}
        if d7_authorization is None and mode == SYNTHETIC_MODE
        else _plain(d7_authorization, "d7_authorization")
    )
    method_value = _normalize_named(method, "id", "method")
    target_value = _normalize_named(target, "id", "target")
    fraction_values = list(fractions)
    coordinates = {
        "plan_digest": _binding_digest(plan, "plan"),
        "group_digest": typed_group.canonical_digest,
        "pair_digest": typed_group.pair_binding.canonical_digest,
        "unit_digest": typed_unit.canonical_digest,
        "work_item_digests": list(work_digests),
        "work_item_count": len(work_digests),
        "work_item_sequence_digest": canonical_sha256(
            {"work_item_digests": list(work_digests)}
        ),
    }
    terminal_value = _plain(terminal, "terminal")
    failure_value = _plain(failure, "failure")
    inventory_value = _plain(list(artifact_inventory), "artifact_inventory")
    terminal_status = (
        terminal_value.get("status")
        if isinstance(terminal_value, Mapping) else None
    )
    if terminal_status == SCIENTIFIC_FAILURE_STATUS and failure_l0_scalars is not None:
        scientific_records = [
            item for item in inventory_value
            if isinstance(item, Mapping) and item.get("path") != FAILURE_PATH
        ]
        produced = [item.get("path") for item in scientific_records]
        normalized_failure, failure_record, _raw = (
            stage_d_scientific_failure_artifact(
                failure_value, l0_scalars=failure_l0_scalars,
                audit=typed_unit.audit, produced=produced,
            )
        )
        supplied_failure_records = [
            item for item in inventory_value
            if isinstance(item, Mapping) and item.get("path") == FAILURE_PATH
        ]
        if supplied_failure_records and supplied_failure_records != [failure_record]:
            raise StageDProvenanceError(
                "supplied failure artifact differs from deterministic evidence"
            )
        inventory_value = [*scientific_records, failure_record]
        failure_value = normalized_failure
    elif failure_l0_scalars is not None:
        raise StageDProvenanceError(
            "failure_l0_scalars are legal only for scientific failure"
        )
    bindings = {
        "d7_authorization": authorization,
        "producer": _plain(producer, "producer"),
        "dependencies": _plain(dependencies, "dependencies"),
        "model": _plain(model, "model"),
        "source_bindings": _plain(source_bindings, "source_bindings"),
        "coordinates": coordinates,
        "image_identity": _plain(image_identity, "image_identity"),
        "scientific_contract": {
            "method": method_value,
            "target": target_value,
            "block_index": typed_unit.block_index,
            "head_index": typed_unit.head_index,
            "operator": _plain(operator, "operator"),
            "projection": _plain(projection, "projection"),
            "sign_semantics": _plain(sign_semantics, "sign_semantics"),
            "normalization": _plain(normalization, "normalization"),
            "baseline": _plain(baseline, "baseline"),
            "fractions": _plain(fraction_values, "fractions"),
            "seed_contract": _plain(seed_contract, "seed_contract"),
        },
        "execution": {
            "attempt": _plain(attempt_value, "attempt"),
            "retry": _plain(retry, "retry"),
            "claim": _plain(claim, "claim"),
            "shard": _plain(shard, "shard"),
        },
        "terminal": terminal_value,
        "failure": failure_value,
        "resources": _provenance_resources(resources),
        "artifact_inventory": inventory_value,
    }
    document = {
        "schema_version": SCHEMA_VERSION,
        "document_type": DOCUMENT_TYPE,
        "mode": mode,
        "unit_identity": typed_unit.to_dict(),
        "unit_identity_sha256": typed_unit.canonical_digest,
        "bindings": bindings,
    }
    document["canonical_digest"] = canonical_sha256(document)
    return validate_stage_d_provenance(
        document, expected_unit=typed_unit, expected_group=typed_group
    )


def validate_stage_d_provenance(
    value: StageDProvenance | Mapping[str, Any],
    *,
    expected_unit: StageDUnit | Mapping[str, Any] | None = None,
    expected_group: StageDImageGroup | Mapping[str, Any] | None = None,
    expected_plan_digest: str | None = None,
    expected_mode: str | None = None,
    expected_attempt_number: int | None = None,
    expected_terminal: Mapping[str, Any] | None = None,
    expected_failure: Mapping[str, Any] | None = None,
    expected_inventory: Sequence[Mapping[str, Any]] | None = None,
    expected_bindings: Mapping[str, Any] | None = None,
    observed_git_state: Mapping[str, Any] | None = None,
) -> StageDProvenance:
    """Validate structure, semantic coordinates, and supplied source facts."""

    document = value.to_dict() if isinstance(value, StageDProvenance) else _plain(
        value, "provenance"
    )
    root = _exact(document, _ROOT_FIELDS, "provenance")
    if root["schema_version"] != SCHEMA_VERSION:
        raise StageDProvenanceError("provenance schema_version is invalid")
    if root["document_type"] != DOCUMENT_TYPE:
        raise StageDProvenanceError("provenance document_type is invalid")
    if root["mode"] not in {FORMAL_MODE, SYNTHETIC_MODE}:
        raise StageDProvenanceError("provenance mode is invalid")
    claimed_digest = _sha256(root["canonical_digest"], "canonical_digest")
    semantic = {key: _thaw(child) for key, child in root.items() if key != "canonical_digest"}
    if canonical_sha256(semantic) != claimed_digest:
        raise StageDProvenanceError("provenance canonical_digest mismatch")
    # Historical source evidence may legitimately contain Stage C paths and
    # statuses (especially the required C10 binding).  Foreign-stage rejection
    # is therefore limited to the Stage D identity and scientific contract;
    # independent expected-source checks remain available below.
    _reject_forbidden_identity(
        {
            "document_type": root["document_type"],
            "unit_identity": root["unit_identity"],
        },
        "provenance.identity",
    )

    unit = _typed_unit(root["unit_identity"], "provenance.unit_identity")
    if root["unit_identity_sha256"] != unit.canonical_digest:
        raise StageDProvenanceError(
            "unit_identity_sha256 disagrees with the typed Stage D unit"
        )
    if expected_unit is not None:
        typed_expected_unit = _typed_unit(expected_unit, "expected_unit")
        if typed_expected_unit.to_dict() != unit.to_dict():
            raise StageDProvenanceError("provenance unit does not match expected unit")

    bindings = _exact(root["bindings"], _BINDING_FIELDS, "bindings")
    d7 = _exact(
        bindings["d7_authorization"], {"status", "digest"},
        "bindings.d7_authorization",
    )
    if root["mode"] == SYNTHETIC_MODE:
        if d7 != {"status": D7_NOT_ISSUED, "digest": None}:
            raise StageDProvenanceError(
                "synthetic provenance must explicitly record D7 NOT_ISSUED"
            )
    else:
        if d7.get("status") != D7_AUTHORIZED:
            raise StageDProvenanceError("formal provenance lacks D7 authorization")
        _sha256(d7.get("digest"), "bindings.d7_authorization.digest")

    producer = _exact(
        bindings["producer"], {"commit", "tree", "clean"},
        "bindings.producer",
    )
    for name in ("commit", "tree"):
        if not isinstance(producer[name], str) or _GIT_OBJECT_RE.fullmatch(
            producer[name]
        ) is None:
            raise StageDProvenanceError(
                f"bindings.producer.{name} is not a Git object id"
            )
    if type(producer["clean"]) is not bool:
        raise StageDProvenanceError("bindings.producer.clean must be boolean")
    if root["mode"] == FORMAL_MODE and producer["clean"] is not True:
        raise StageDProvenanceError("formal provenance requires a clean producer")
    if observed_git_state is not None and _plain(observed_git_state) != dict(producer):
        raise StageDProvenanceError("producer does not match observed Git state")

    for name in ("dependencies", "model"):
        _nonempty_mapping(bindings[name], f"bindings.{name}")
        _validate_digest_fields(bindings[name], f"bindings.{name}")
    resources = _provenance_resources(bindings["resources"])
    if resources != dict(bindings["resources"]):
        raise StageDProvenanceError(
            "bindings.resources is not the canonical acyclic projection"
        )
    sources = _exact(
        bindings["source_bindings"], {"d0", "d1", "d2", "c10"},
        "bindings.source_bindings",
    )
    for name, source in sources.items():
        _nonempty_mapping(source, f"bindings.source_bindings.{name}")
        _validate_digest_fields(source, f"bindings.source_bindings.{name}")
        if name != "c10":
            _reject_forbidden_identity(
                source, f"bindings.source_bindings.{name}"
            )
    # C10 is the sole historical foreign-stage source slot.  All other
    # execution/provenance sections must remain Stage-D-specific.
    for name in (
        "d7_authorization", "producer", "dependencies", "model",
        "coordinates", "image_identity", "execution", "terminal",
        "failure", "resources", "artifact_inventory",
    ):
        _reject_forbidden_identity(bindings[name], f"bindings.{name}")

    coordinates = _exact(
        bindings["coordinates"], _COORDINATE_FIELDS, "bindings.coordinates"
    )
    for name in ("plan_digest", "group_digest", "pair_digest", "unit_digest"):
        _sha256(coordinates[name], f"bindings.coordinates.{name}")
    if coordinates["unit_digest"] != unit.canonical_digest:
        raise StageDProvenanceError("coordinate unit digest mismatch")
    if coordinates["pair_digest"] != unit.pair_binding_digest:
        raise StageDProvenanceError("coordinate pair digest mismatch")
    digests = coordinates["work_item_digests"]
    if not isinstance(digests, list) or any(
        not isinstance(item, str) or _SHA256_RE.fullmatch(item) is None
        for item in digests
    ):
        raise StageDProvenanceError("work item digests are invalid")
    expected_count = 212 if unit.audit else 211
    if (
        coordinates["work_item_count"] != len(digests)
        or len(digests) != expected_count
        or len(set(digests)) != len(digests)
    ):
        raise StageDProvenanceError("work item coverage is incomplete or duplicated")
    if coordinates["work_item_sequence_digest"] != canonical_sha256(
        {"work_item_digests": digests}
    ):
        raise StageDProvenanceError("work item sequence digest mismatch")
    if digests != list(_unit_work_item_digests(unit)):
        raise StageDProvenanceError(
            "work item digests do not exactly cover the typed Stage D unit"
        )
    if expected_plan_digest is not None and coordinates["plan_digest"] != _sha256(
        expected_plan_digest, "expected_plan_digest"
    ):
        raise StageDProvenanceError("plan digest does not match expected plan")

    group: StageDImageGroup | None = None
    if expected_group is not None:
        group = _typed_group(expected_group, "expected_group")
        expected_digests = _unit_work_item_digests(unit, group)
        if coordinates["group_digest"] != group.canonical_digest:
            raise StageDProvenanceError("group digest does not match expected group")
        if coordinates["pair_digest"] != group.pair_binding.canonical_digest:
            raise StageDProvenanceError("pair digest does not match expected group")
        if list(expected_digests) != digests:
            raise StageDProvenanceError(
                "work item digests do not cover expected unit/group coordinates"
            )

    image = _nonempty_mapping(bindings["image_identity"], "bindings.image_identity")
    status = image.get("status")
    if not isinstance(status, str) or not status:
        raise StageDProvenanceError("image_identity.status must be non-empty")
    if root["mode"] == SYNTHETIC_MODE and status != SYNTHETIC_IMAGE_STATUS:
        raise StageDProvenanceError(
            "synthetic provenance must state that no image was accessed"
        )
    if root["mode"] == FORMAL_MODE and status != FORMAL_IMAGE_STATUS:
        raise StageDProvenanceError("formal image identity status is invalid")
    if root["mode"] == FORMAL_MODE:
        image_sha256 = image.get("image_sha256", image.get("sha256"))
        _sha256(image_sha256, "bindings.image_identity.image_sha256")
        if group is not None and image_sha256 != group.image_sha256:
            raise StageDProvenanceError(
                "formal image identity does not match the expected group"
            )
    _validate_digest_fields(image, "bindings.image_identity")

    scientific = _exact(
        bindings["scientific_contract"], _SCIENTIFIC_FIELDS,
        "bindings.scientific_contract",
    )
    _reject_forbidden_identity(scientific, "bindings.scientific_contract")
    method = _nonempty_mapping(scientific["method"], "scientific_contract.method")
    target = _nonempty_mapping(scientific["target"], "scientific_contract.target")
    if method.get("id") != unit.method_id or unit.method_id not in METHOD_IDS:
        raise StageDProvenanceError("method does not match the Stage D unit")
    if target.get("id") != unit.target_id or unit.target_id != TARGET_ID:
        raise StageDProvenanceError("target does not match the Stage D unit")
    if (
        scientific["block_index"] != unit.block_index
        or scientific["block_index"] != BLOCK_INDEX
        or scientific["head_index"] != unit.head_index
        or scientific["head_index"] != HEAD_INDEX
    ):
        raise StageDProvenanceError("block/head coordinate is not frozen")
    for name in (
        "operator", "projection", "sign_semantics", "normalization",
        "baseline", "seed_contract",
    ):
        child = scientific[name]
        if child is None or child == "" or child == {}:
            raise StageDProvenanceError(f"scientific_contract.{name} is empty")
    fractions = scientific["fractions"]
    if (
        not isinstance(fractions, list)
        or len(fractions) != len(CURVE_FRACTIONS)
        or any(
            isinstance(left, bool)
            or not isinstance(left, (int, float))
            or float(left) != float(right)
            for left, right in zip(fractions, CURVE_FRACTIONS)
        )
    ):
        raise StageDProvenanceError("fidelity fractions differ from the frozen 21")

    execution = _exact(
        bindings["execution"], _EXECUTION_FIELDS, "bindings.execution"
    )
    for name in _EXECUTION_FIELDS:
        _nonempty_mapping(execution[name], f"bindings.execution.{name}")
        _validate_digest_fields(execution[name], f"bindings.execution.{name}")
    number = _attempt_number(execution["attempt"])
    if expected_attempt_number is not None and number != expected_attempt_number:
        raise StageDProvenanceError("provenance attempt number mismatch")
    shard = execution["shard"]
    shard_count = shard.get("count", shard.get("shard_count"))
    shard_id = shard.get("id", shard.get("shard_id"))
    if (
        type(shard_count) is not int or shard_count <= 0
        or type(shard_id) is not int or not 0 <= shard_id < shard_count
    ):
        raise StageDProvenanceError("execution shard id/count is invalid")
    if shard_id != unit.selection_order % shard_count:
        raise StageDProvenanceError(
            "execution shard violates selection_order modulo shard_count"
        )

    terminal = _exact(
        bindings["terminal"], _TERMINAL_FIELDS, "bindings.terminal"
    )
    if terminal.get("status") not in {SUCCESS_STATUS, SCIENTIFIC_FAILURE_STATUS}:
        raise StageDProvenanceError("terminal status is invalid")
    if terminal.get("terminal") is not True or terminal.get("complete") is not True:
        raise StageDProvenanceError(
            "success and scientific failure must both be terminal and complete"
        )
    failure = bindings["failure"]
    if terminal["status"] == SUCCESS_STATUS:
        if failure is not None:
            raise StageDProvenanceError("successful provenance cannot contain failure")
    elif not isinstance(failure, Mapping) or not failure:
        raise StageDProvenanceError("scientific failure requires structured failure")
    if expected_terminal is not None and dict(terminal) != _plain(expected_terminal):
        raise StageDProvenanceError("terminal binding does not match expected status")
    if expected_failure is not None and _plain(failure) != _plain(expected_failure):
        raise StageDProvenanceError("failure binding does not match expected failure")

    inventory = _validate_inventory(
        bindings["artifact_inventory"], audit=unit.audit,
        terminal_status=terminal["status"], failure=failure,
    )
    if expected_inventory is not None and list(inventory) != _plain(
        list(expected_inventory), "expected_inventory"
    ):
        raise StageDProvenanceError("artifact inventory differs from expected bytes")
    if expected_mode is not None and root["mode"] != expected_mode:
        raise StageDProvenanceError("provenance mode does not match expected mode")
    if expected_bindings is not None:
        expected = _plain(expected_bindings, "expected_bindings")
        for key, child in expected.items():
            if key not in bindings or _plain(bindings[key]) != child:
                raise StageDProvenanceError(
                    f"provenance binding {key!r} does not match expected source"
                )
    return StageDProvenance(_plain(root), _ISSUER)


__all__ = [
    "D7_AUTHORIZED", "D7_NOT_ISSUED", "DOCUMENT_TYPE",
    "FAILURE_DOCUMENT_TYPE", "FAILURE_PATH",
    "FORMAL_IMAGE_STATUS", "FORMAL_MODE", "SCHEMA_VERSION",
    "SCIENTIFIC_FAILURE_STATUS", "SUCCESS_STATUS",
    "SCIENTIFIC_FAILURE_SOURCE",
    "SYNTHETIC_IMAGE_STATUS", "SYNTHETIC_MODE",
    "StageDProvenance", "StageDProvenanceError",
    "build_stage_d_provenance", "canonical_json_bytes",
    "canonical_json_file_bytes", "canonical_sha256",
    "normalize_stage_d_scientific_failure",
    "stage_d_scientific_failure_artifact",
    "validate_stage_d_provenance",
]
