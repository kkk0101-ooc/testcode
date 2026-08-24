"""Deterministic C10 atomic long-table construction from C9 evidence.

This module is deliberately a pure, in-memory tabular boundary.  It accepts
records that have already passed the C9 batch/artifact validator and expands
only their serialized L0 scalar/resource evidence.  It never opens an
artifact, image, numpy array, or display file and it has no model/backend
entry point.  Callers may append separately derived, source-bound L1 rows and
then use the same validation, ordering, CSV, and manifest functions.

Metric identities and source paths use semantic selectors (scale, direction,
ranking, replicate, fraction, perturbation, and view).  A position in a JSON
list is never part of a metric identity.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Any


SCHEMA_VERSION = 1
DOCUMENT_TYPE = "stage_c_c10_atomic_long_table"
REFERENCE_FULL_C9_BASE_ROW_COUNT = 297_844

DIRECT_METHOD_ID = "direct_raw_cls_attention_per_head"
NNS_METHOD_ID = (
    "conditional_fixed_clean_attention_head_direction_value_path_ig"
)
MATCHED_METHOD_ID = (
    "matched_fixed_clean_attention_patch_token_contribution"
)
FULL_MODEL_IG_METHOD_ID = "full_model_class_logit_ig"
IG_METHOD_IDS = frozenset({NNS_METHOD_ID, FULL_MODEL_IG_METHOD_ID})
PERTURBATION_IDS = (
    "translate_right_down_4",
    "center_crop_216_resize_224",
    "gaussian_rgb_std_0_01",
)
FIDELITY_SCALES = ("patch_16x16", "fine_4x4_cell")
CURVE_DIRECTIONS = ("deletion", "insertion")
TARGET_RANKING = "target_oriented_positive_tie_permuted"
RANDOM_RANKING = "global_random"
CENTER_RANKING = "center_distance_then_row_major"

INVALIDATING_NA_REASONS = frozenset({
    "empty_valid_region", "shape_mismatch", "non_finite",
    "degenerate_direction", "missing_artifact",
})
ROW_STATUSES = frozenset({
    "OK", "NA", "INVALID", "UNIT_FAILED", "UNIT_INCOMPLETE",
})
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_KIND_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_MISSING = object()

# Column order is part of the byte-level format.  Nullable scalar cells use an
# empty CSV field; JSON-valued columns always contain canonical JSON text.
CSV_COLUMNS = (
    "schema_version", "row_key_sha256", "row_hash_sha256",
    "plan_index", "sample_id", "sample_role", "audit",
    "comparison_line", "fidelity_eligible",
    "structural_reference_only", "method_id",
    "reference_method_id", "config_id", "block_index",
    "head_index", "target_ref", "metric_family", "metric_id",
    "input_variant", "perturbation_id", "fidelity_scale",
    "curve_direction", "ranking_kind", "replicate_index",
    "fraction_index", "fraction", "statistic", "view",
    "component", "value_type", "value", "status",
    "na_reason", "invalidating", "details_json", "unit_state",
    "clean_top1_correct", "misclassified", "failure_json",
    "source_kind", "source_field_path", "source_attempt_path",
    "source_unit_key_sha256", "source_status_sha256",
    "source_manifest_sha256", "source_provenance_sha256",
    "source_result_sha256", "source_artifact_sha256",
    "source_unit_identity_sha256",
    "secondary_sources_json", "source_binding_sha256",
)

_NULLABLE_STRING_COLUMNS = frozenset({
    "reference_method_id", "config_id", "target_ref",
    "input_variant", "perturbation_id", "fidelity_scale",
    "curve_direction", "ranking_kind", "statistic", "view",
    "component", "na_reason", "source_attempt_path",
    "source_unit_key_sha256", "source_status_sha256",
    "source_manifest_sha256", "source_provenance_sha256",
    "source_result_sha256", "source_artifact_sha256",
    "source_unit_identity_sha256",
})
_NULLABLE_INT_COLUMNS = frozenset({
    "plan_index", "block_index", "head_index", "replicate_index",
    "fraction_index",
})
_NULLABLE_FLOAT_COLUMNS = frozenset({"fraction"})
_NULLABLE_BOOL_COLUMNS = frozenset({"clean_top1_correct", "misclassified"})
_BOOL_COLUMNS = frozenset({
    "audit", "fidelity_eligible", "structural_reference_only",
    "invalidating",
})
_JSON_COLUMNS = frozenset({
    "details_json", "failure_json", "secondary_sources_json",
})

_SEMANTIC_KEY_COLUMNS = (
    "sample_id", "method_id", "reference_method_id", "config_id",
    "metric_family", "metric_id", "input_variant",
    "perturbation_id", "fidelity_scale", "curve_direction",
    "ranking_kind", "replicate_index", "fraction_index",
    "statistic", "view", "component",
)
_PRIMARY_SOURCE_COLUMNS = (
    "source_kind", "source_field_path", "source_attempt_path",
    "source_unit_key_sha256", "source_status_sha256",
    "source_manifest_sha256", "source_provenance_sha256",
    "source_result_sha256", "source_artifact_sha256",
    "source_unit_identity_sha256",
)
_SECONDARY_SOURCE_KEYS = frozenset({
    "method_id", "source_kind", "source_field_path",
    "source_attempt_path", "source_unit_key_sha256",
    "source_status_sha256", "source_manifest_sha256",
    "source_provenance_sha256", "source_result_sha256",
    "source_artifact_sha256", "source_unit_identity_sha256",
})
_HASH_COLUMNS = frozenset({
    "row_key_sha256", "row_hash_sha256", "source_binding_sha256",
    "source_unit_key_sha256", "source_status_sha256",
    "source_manifest_sha256", "source_provenance_sha256",
    "source_result_sha256", "source_artifact_sha256",
    "source_unit_identity_sha256",
})


class StageC10TableError(ValueError):
    """Base error for the C10 atomic table contract."""


class StageC10TableIntegrityError(StageC10TableError):
    """Raised when source evidence or a table violates the contract."""


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise StageC10TableIntegrityError(
            "value is not finite canonical JSON"
        ) from error


def _canonical_json(value: Any) -> str:
    return _canonical_bytes(value).decode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _require_sha256(value: Any, context: str, *, nullable: bool = False) -> str | None:
    if value is None and nullable:
        return None
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise StageC10TableIntegrityError(
            f"{context} must be a lowercase SHA-256"
        )
    return value


def _plain(value: Any, context: str = "value") -> Any:
    method = getattr(value, "to_dict", None)
    if callable(method):
        value = method()
    if value is None or type(value) in {str, bool, int}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise StageC10TableIntegrityError(f"{context} is non-finite")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise StageC10TableIntegrityError(
                    f"{context} has a non-string/empty key"
                )
            result[key] = _plain(item, f"{context}.{key}")
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_plain(item, f"{context}[]") for item in value]
    raise StageC10TableIntegrityError(
        f"{context} contains unsupported type {type(value).__name__}"
    )


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise StageC10TableIntegrityError(f"{context} must be an object")
    return value


def _sequence(value: Any, context: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise StageC10TableIntegrityError(f"{context} must be an array")
    return value


def _selector_value(value: Any) -> str:
    if value is None:
        return "none"
    if type(value) is bool:
        return "true" if value else "false"
    if type(value) is float:
        if not math.isfinite(value):
            raise StageC10TableIntegrityError("selector is non-finite")
        return format(value, ".17g")
    return str(value)


def metric_id_from_selectors(base: str, /, **selectors: Any) -> str:
    """Build a stable metric id from named semantic selectors.

    Keyword insertion order is intentionally ignored.  Selector names are
    sorted, making the identity independent of any source-list position.
    """

    if not isinstance(base, str) or not base or "|" in base:
        raise StageC10TableIntegrityError("metric id base is invalid")
    pieces = [base]
    for name in sorted(selectors):
        if not isinstance(name, str) or not name or "|" in name or "=" in name:
            raise StageC10TableIntegrityError("metric selector name is invalid")
        rendered = _selector_value(selectors[name])
        if any(character in rendered for character in "|\n\r"):
            raise StageC10TableIntegrityError("metric selector value is invalid")
        pieces.append(f"{name}={rendered}")
    return "|".join(pieces)


def _selector_path(collection: str, selectors: Mapping[str, Any], field: str) -> str:
    clauses = ",".join(
        f"{name}={_selector_value(selectors[name])}" for name in sorted(selectors)
    )
    return f"/scalars/{collection}{{{clauses}}}/{field}"


def _forbidden_quantitative_source(source_kind: str, source_field_path: str) -> bool:
    kind = source_kind.lower()
    path = source_field_path.lower().replace("\\", "/")
    return (
        "l3" in kind or "display" in kind
        or path.startswith("/l3/") or "/l3/" in path
        or path.endswith((".png", ".jpg", ".jpeg"))
    )


def schema_document() -> dict[str, Any]:
    """Return the complete, deterministic CSV/schema contract document."""

    integer = {"type": ["integer", "null"]}
    number = {"type": ["number", "null"]}
    boolean = {"type": ["boolean", "null"]}
    properties: dict[str, Any] = {}
    for column in CSV_COLUMNS:
        if column == "schema_version":
            properties[column] = {"type": "integer", "const": 1}
        elif column in _NULLABLE_INT_COLUMNS:
            properties[column] = dict(integer)
        elif column in _NULLABLE_FLOAT_COLUMNS:
            properties[column] = dict(number)
        elif column in _BOOL_COLUMNS:
            properties[column] = {"type": "boolean"}
        elif column in _NULLABLE_BOOL_COLUMNS:
            properties[column] = dict(boolean)
        elif column == "value":
            properties[column] = {
                "type": ["number", "integer", "boolean", "string", "null"]
            }
        elif column in _NULLABLE_STRING_COLUMNS:
            properties[column] = {"type": ["string", "null"]}
        else:
            properties[column] = {"type": "string"}
    properties["status"] = {"enum": sorted(ROW_STATUSES)}
    properties["value_type"] = {
        "enum": ["bool", "float", "int", "null", "string"]
    }
    for column in _HASH_COLUMNS:
        properties[column]["pattern"] = "^[0-9a-f]{64}$"
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "schema_version": SCHEMA_VERSION,
        "document_type": "stage_c_c10_atomic_long_table_schema",
        "title": "Stage C10 atomic per-sample long table",
        "type": "array",
        "items": {
            "type": "object", "additionalProperties": False,
            "required": list(CSV_COLUMNS), "properties": properties,
        },
        "csv": {
            "encoding": "UTF-8", "line_terminator": "\n",
            "columns": list(CSV_COLUMNS),
            "null_encoding": "empty_field",
            "json_columns": sorted(_JSON_COLUMNS),
            "numeric_round_trip": "Python repr / JSON finite-number grammar",
        },
        "identity": {
            "semantic_key_columns": list(_SEMANTIC_KEY_COLUMNS),
            "source_binding_columns": list(_PRIMARY_SOURCE_COLUMNS),
            "secondary_sources_column": "secondary_sources_json",
            "row_key": (
                "sha256(canonical_json({semantic_key,source_binding_sha256}))"
            ),
            "row_hash": "sha256(canonical_json(row_without_row_hash_sha256))",
        },
        "statuses": {
            "OK": "finite/source scalar retained",
            "NA": "structured non-invalidating NA retained without imputation",
            "INVALID": "invalidating structured NA/source state retained",
            "UNIT_FAILED": "expected metric row retained for a failed unit",
            "UNIT_INCOMPLETE": (
                "expected metric row retained for a nonterminal/missing/tampered unit"
            ),
        },
        "source_policy": {
            "base_rows": "verified in-memory C9 L0 records only",
            "caller_appended_rows": "may be source-bound L1 derivations",
            "l3_display_image_quantitative_sources_forbidden": True,
            "filesystem_reads_performed_by_this_module": False,
        },
        "applicability": {
            "fidelity_curve": "eligible units: 44 curves x (AUC + 21 responses)",
            "fidelity_aggregate": (
                "eligible units: 4 target aggregates x "
                "(AUC mean/std + 21 response mean/std pairs)"
            ),
            "top20": "eligible units: 2 scales x 5 repeats x 3 metrics",
            "stability": "eligible units: 3 perturbations x 12 metrics",
            "raw_structural": "structural raw units: 3 perturbations x 7 metrics",
            "nns_residual": "NNS units: 3 perturbations x 3 metrics",
            "resources": "all units: 6 metrics",
            "endpoint_parity": "NNS units: 4 metrics",
            "ig_conservation": "NNS/full-model-IG units: 5 final metrics",
            "matched_contribution": "matched units: 8 token/spatial metrics",
            "repeatability": "audit units: pass and maximum absolute error",
            "prediction": "all units: correctness plus 3 perturbation flips",
            "unit_status": "all units: one status metric",
        },
        "reference_full_success_base_row_count": (
            REFERENCE_FULL_C9_BASE_ROW_COUNT
        ),
        "reference_count_is_validation_rule": False,
    }


def _object_dict(value: Any, context: str) -> dict[str, Any]:
    method = getattr(value, "to_dict", None)
    if callable(method):
        value = method()
    if not isinstance(value, Mapping):
        raise StageC10TableIntegrityError(f"{context} must be an object")
    return _plain(value, context)


def _plan_fields(plan_unit: Any) -> dict[str, Any]:
    unit = _object_dict(plan_unit, "plan unit")
    key = unit.get("key", unit.get("unit_key"))
    sample = unit.get("sample")
    key = _mapping(key, "plan unit.key")
    sample = _mapping(sample, "plan unit.sample")
    required_key = {
        "sample_id", "method_id", "config_id", "block_index",
        "head_index", "target_ref", "canonical_digest",
    }
    if not required_key <= set(key):
        raise StageC10TableIntegrityError("plan unit key is incomplete")
    if not {"role", "audit"} <= set(sample):
        raise StageC10TableIntegrityError("plan unit sample metadata is incomplete")
    if key["sample_id"] != sample.get("sample_id"):
        raise StageC10TableIntegrityError("plan unit sample identity disagrees")
    _require_sha256(key["canonical_digest"], "plan unit canonical digest")
    role = sample["role"]
    if role not in {"calibration", "holdout"}:
        raise StageC10TableIntegrityError("plan unit role is invalid")
    if type(sample["audit"]) is not bool:
        raise StageC10TableIntegrityError("plan unit audit must be boolean")
    comparison_line = unit.get("comparison_line")
    if comparison_line not in {"value_path", "class_logit"}:
        raise StageC10TableIntegrityError("plan unit comparison line is invalid")
    for name in ("fidelity_eligible", "structural_reference_only"):
        if type(unit.get(name)) is not bool:
            raise StageC10TableIntegrityError(f"plan unit {name} must be boolean")
    return {
        "key": dict(key), "sample": dict(sample),
        "comparison_line": comparison_line,
        "fidelity_eligible": unit["fidelity_eligible"],
        "structural_reference_only": unit["structural_reference_only"],
    }


def _record_hashes(record: Mapping[str, Any], *, success: bool) -> dict[str, str | None]:
    hashes = record.get("hashes")
    if not success and not isinstance(hashes, Mapping):
        return {
            "status_sha256": None, "manifest_sha256": None,
            "provenance_sha256": None, "result_sha256": None,
            "unit_identity_sha256": None,
        }
    hashes = _mapping(hashes, "evidence hashes")
    result: dict[str, str | None] = {}
    for name in (
        "status_sha256", "manifest_sha256", "provenance_sha256",
        "result_sha256", "unit_identity_sha256",
    ):
        result[name] = _require_sha256(
            hashes.get(name), f"evidence hashes.{name}", nullable=not success
        )
    selected = hashes.get("selected_status_sha256")
    if selected is not None:
        _require_sha256(selected, "evidence selected status hash")
        if selected != result["status_sha256"]:
            raise StageC10TableIntegrityError(
                "selected/status SHA-256 binding disagrees"
            )
    return result


def base_row(
    evidence_record: Mapping[str, Any], plan_unit: Any, *, plan_index: int | None = None,
) -> dict[str, Any]:
    """Build common row fields from one verified in-memory C9 record.

    No path in the record is opened.  The returned mapping is intended for
    :func:`make_row`, including by callers that append L1-derived metrics.
    """

    record = _mapping(evidence_record, "evidence record")
    unit = _plan_fields(plan_unit)
    key = unit["key"]
    record_key = _mapping(record.get("unit_key"), "evidence record.unit_key")
    if dict(record_key) != key:
        raise StageC10TableIntegrityError(
            "evidence unit key differs from its plan unit"
        )
    state = record.get("state")
    if not isinstance(state, str) or not state:
        raise StageC10TableIntegrityError("evidence unit state is missing")
    success = state == "succeeded"
    if success:
        if record.get("trusted_by_batch_validator", True) is not True:
            raise StageC10TableIntegrityError(
                "successful evidence is not trusted by the C9 batch validator"
            )
        errors = record.get("validation_errors", [])
        if errors not in ([], ()):
            raise StageC10TableIntegrityError(
                "successful evidence contains validation errors"
            )
        _mapping(record.get("scalars"), "successful evidence.scalars")
        _mapping(record.get("resources"), "successful evidence.resources")
    hashes = _record_hashes(record, success=success)
    attempt = record.get("attempt_path")
    if success and (not isinstance(attempt, str) or not attempt):
        raise StageC10TableIntegrityError(
            "successful evidence lacks its selected attempt path"
        )
    if attempt is not None and (not isinstance(attempt, str) or not attempt):
        raise StageC10TableIntegrityError("attempt_path must be non-empty or null")
    if record.get("audit") is not None and record.get("audit") is not unit["sample"]["audit"]:
        raise StageC10TableIntegrityError("evidence audit flag differs from plan")
    record_index = record.get("plan_index")
    if record_index is not None:
        if type(record_index) is not int or record_index < 0:
            raise StageC10TableIntegrityError("evidence plan_index is invalid")
        if plan_index is not None and record_index != plan_index:
            raise StageC10TableIntegrityError(
                "evidence plan_index differs from ordered plan"
            )
        plan_index = record_index
    if plan_index is not None and (type(plan_index) is not int or plan_index < 0):
        raise StageC10TableIntegrityError("plan_index is invalid")
    prediction = record.get("scalars", {}).get("prediction", {})
    correct = prediction.get("clean_top1_correct") if isinstance(prediction, Mapping) else None
    if correct is not None and type(correct) is not bool:
        raise StageC10TableIntegrityError(
            "prediction.clean_top1_correct must be boolean or null"
        )
    failure = record.get("failure")
    return {
        "schema_version": SCHEMA_VERSION, "plan_index": plan_index,
        "sample_id": key["sample_id"],
        "sample_role": unit["sample"]["role"],
        "audit": unit["sample"]["audit"],
        "comparison_line": unit["comparison_line"],
        "fidelity_eligible": unit["fidelity_eligible"],
        "structural_reference_only": unit["structural_reference_only"],
        "method_id": key["method_id"], "reference_method_id": None,
        "config_id": key["config_id"],
        "block_index": key["block_index"],
        "head_index": key["head_index"], "target_ref": key["target_ref"],
        "unit_state": state, "clean_top1_correct": correct,
        "misclassified": None if correct is None else not correct,
        "failure_json": _canonical_json(failure),
        "source_attempt_path": attempt,
        "source_unit_key_sha256": key["canonical_digest"],
        "source_status_sha256": hashes["status_sha256"],
        "source_manifest_sha256": hashes["manifest_sha256"],
        "source_provenance_sha256": hashes["provenance_sha256"],
        "source_result_sha256": hashes["result_sha256"],
        "source_artifact_sha256": hashes["result_sha256"],
        "source_unit_identity_sha256": hashes["unit_identity_sha256"],
    }


def source_binding(
    evidence_record: Mapping[str, Any], *, source_kind: str,
    source_field_path: str, method_id: str | None = None,
    source_artifact_sha256: str | None = None,
) -> dict[str, Any]:
    """Create a canonical secondary-source binding without reading it."""

    record = _mapping(evidence_record, "secondary evidence record")
    key = _mapping(record.get("unit_key"), "secondary record.unit_key")
    success = record.get("state") == "succeeded"
    hashes = _record_hashes(record, success=success)
    result = {
        "method_id": method_id if method_id is not None else key.get("method_id"),
        "source_kind": source_kind, "source_field_path": source_field_path,
        "source_attempt_path": record.get("attempt_path"),
        "source_unit_key_sha256": key.get("canonical_digest"),
        "source_status_sha256": hashes["status_sha256"],
        "source_manifest_sha256": hashes["manifest_sha256"],
        "source_provenance_sha256": hashes["provenance_sha256"],
        "source_result_sha256": hashes["result_sha256"],
        "source_artifact_sha256": (
            hashes["result_sha256"]
            if source_artifact_sha256 is None else source_artifact_sha256
        ),
        "source_unit_identity_sha256": hashes["unit_identity_sha256"],
    }
    return _validate_secondary_source(result, "secondary source")


def _validate_secondary_source(value: Any, context: str) -> dict[str, Any]:
    item = _mapping(value, context)
    if set(item) != _SECONDARY_SOURCE_KEYS:
        raise StageC10TableIntegrityError(
            f"{context} fields differ from the secondary-source schema"
        )
    if not isinstance(item["method_id"], str) or not item["method_id"]:
        raise StageC10TableIntegrityError(f"{context}.method_id is invalid")
    kind = item["source_kind"]
    path = item["source_field_path"]
    if (
        not isinstance(kind, str) or _SOURCE_KIND_RE.fullmatch(kind) is None
        or not isinstance(path, str) or not path.startswith("/")
        or _forbidden_quantitative_source(kind, path)
    ):
        raise StageC10TableIntegrityError(
            f"{context} has a forbidden/invalid quantitative source"
        )
    for name in _SECONDARY_SOURCE_KEYS - {
        "method_id", "source_kind", "source_field_path",
        "source_attempt_path",
    }:
        _require_sha256(item[name], f"{context}.{name}", nullable=True)
    attempt = item["source_attempt_path"]
    if attempt is not None and (not isinstance(attempt, str) or not attempt):
        raise StageC10TableIntegrityError(
            f"{context}.source_attempt_path is invalid"
        )
    return dict(item)


def _normalize_secondary_sources(value: Any) -> str:
    if value is None or value == "":
        parsed: Any = []
    elif isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError) as error:
            raise StageC10TableIntegrityError(
                "secondary_sources_json is not JSON"
            ) from error
    else:
        parsed = value
    sources = [
        _validate_secondary_source(item, f"secondary_sources[{index}]")
        for index, item in enumerate(_sequence(parsed, "secondary_sources"))
    ]
    encoded = [_canonical_json(item) for item in sources]
    if len(encoded) != len(set(encoded)):
        raise StageC10TableIntegrityError("secondary sources contain duplicates")
    sources.sort(key=lambda item: _canonical_bytes(item))
    return _canonical_json(sources)


def _value_type(value: Any) -> str:
    if value is None:
        return "null"
    if type(value) is bool:
        return "bool"
    if type(value) is int:
        return "int"
    if type(value) is float:
        if not math.isfinite(value):
            raise StageC10TableIntegrityError("row value is non-finite")
        return "float"
    if isinstance(value, str):
        return "string"
    raise StageC10TableIntegrityError(
        "atomic row value must be null, boolean, integer, float, or string"
    )


def _row_source_binding_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "primary": {name: row[name] for name in _PRIMARY_SOURCE_COLUMNS},
        "secondary": json.loads(row["secondary_sources_json"]),
    }


def _row_key_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "semantic_key": {name: row[name] for name in _SEMANTIC_KEY_COLUMNS},
        "source_binding_sha256": row["source_binding_sha256"],
    }


def _finalize_row(row: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(row)
    result["secondary_sources_json"] = _normalize_secondary_sources(
        result.get("secondary_sources_json")
    )
    result["source_binding_sha256"] = _sha256(
        _row_source_binding_payload(result)
    )
    result["row_key_sha256"] = _sha256(_row_key_payload(result))
    result["row_hash_sha256"] = _sha256({
        key: result[key] for key in CSV_COLUMNS if key != "row_hash_sha256"
    })
    return result


def make_row(
    base: Mapping[str, Any], *, metric_family: str, metric_id: str,
    source_field_path: str, value: Any = None, status: str = "OK",
    na_reason: str | None = None, invalidating: bool | None = None,
    details: Mapping[str, Any] | None = None, source_kind: str = "c9_l0_scalar",
    source_artifact_sha256: str | None = None,
    reference_method_id: str | None = None,
    secondary_sources_json: Any = None, input_variant: str | None = None,
    perturbation_id: str | None = None, fidelity_scale: str | None = None,
    curve_direction: str | None = None, ranking_kind: str | None = None,
    replicate_index: int | None = None, fraction_index: int | None = None,
    fraction: float | None = None, statistic: str | None = None,
    view: str | None = None, component: str | None = None,
) -> dict[str, Any]:
    """Create one atomic row and bind all primary/secondary sources.

    This public constructor is also the supported boundary for caller-derived
    L1 rows.  Such callers select an explicit ``source_kind`` and supply one
    or more canonical ``secondary_sources_json`` bindings for pair metrics.
    """

    if not isinstance(base, Mapping):
        raise StageC10TableIntegrityError("base row must be an object")
    required_base = set(CSV_COLUMNS) - {
        "row_key_sha256", "row_hash_sha256", "metric_family",
        "metric_id", "input_variant", "perturbation_id",
        "fidelity_scale", "curve_direction", "ranking_kind",
        "replicate_index", "fraction_index", "fraction", "statistic",
        "view", "component", "value_type", "value", "status",
        "na_reason", "invalidating", "details_json", "source_kind",
        "source_field_path", "secondary_sources_json",
        "source_binding_sha256",
    }
    if not required_base <= set(base):
        missing = sorted(required_base - set(base))
        raise StageC10TableIntegrityError(f"base row lacks fields: {missing}")
    if (
        not isinstance(metric_family, str) or not metric_family
        or not isinstance(metric_id, str) or not metric_id
    ):
        raise StageC10TableIntegrityError("metric family/id must be non-empty")
    if not isinstance(source_kind, str) or _SOURCE_KIND_RE.fullmatch(source_kind) is None:
        raise StageC10TableIntegrityError("source_kind is invalid")
    if not isinstance(source_field_path, str) or not source_field_path.startswith("/"):
        raise StageC10TableIntegrityError("source_field_path must be absolute")
    if _forbidden_quantitative_source(source_kind, source_field_path):
        raise StageC10TableIntegrityError(
            "L3/display/PNG/JPEG cannot be a quantitative row source"
        )
    if status not in ROW_STATUSES:
        raise StageC10TableIntegrityError(f"unsupported row status {status!r}")
    value_type = _value_type(value)
    if status == "OK":
        if value is None or na_reason is not None:
            raise StageC10TableIntegrityError(
                "OK row requires a value and no NA reason"
            )
        invalidating = False if invalidating is None else invalidating
    else:
        if value is not None:
            raise StageC10TableIntegrityError(
                "non-OK row value must be null (never zero/imputed)"
            )
        if not isinstance(na_reason, str) or not na_reason:
            raise StageC10TableIntegrityError(
                "non-OK row requires an explicit reason"
            )
        expected_invalidating = (
            status in {"INVALID", "UNIT_FAILED", "UNIT_INCOMPLETE"}
            or na_reason in INVALIDATING_NA_REASONS
        )
        invalidating = expected_invalidating if invalidating is None else invalidating
        if type(invalidating) is not bool or invalidating is not expected_invalidating:
            raise StageC10TableIntegrityError(
                "row invalidating flag contradicts status/reason"
            )
    if type(invalidating) is not bool:
        raise StageC10TableIntegrityError("invalidating must be boolean")
    if reference_method_id is not None and (
        not isinstance(reference_method_id, str) or not reference_method_id
    ):
        raise StageC10TableIntegrityError("reference_method_id is invalid")
    row = dict(base)
    row.update({
        "reference_method_id": reference_method_id,
        "metric_family": metric_family, "metric_id": metric_id,
        "input_variant": input_variant, "perturbation_id": perturbation_id,
        "fidelity_scale": fidelity_scale,
        "curve_direction": curve_direction, "ranking_kind": ranking_kind,
        "replicate_index": replicate_index,
        "fraction_index": fraction_index, "fraction": fraction,
        "statistic": statistic, "view": view, "component": component,
        "value_type": value_type, "value": value, "status": status,
        "na_reason": na_reason, "invalidating": invalidating,
        "details_json": _canonical_json({} if details is None else _plain(details)),
        "source_kind": source_kind,
        "source_field_path": source_field_path,
        "secondary_sources_json": secondary_sources_json,
    })
    if source_artifact_sha256 is not None:
        row["source_artifact_sha256"] = _require_sha256(
            source_artifact_sha256, "source_artifact_sha256"
        )
    return _finalize_row(row)


def _metric_state(value: Any, context: str) -> tuple[Any, str, str | None, bool, dict[str, Any]]:
    if isinstance(value, Mapping) and "status" in value:
        status = value.get("status")
        if status == "OK":
            if "value" not in value:
                raise StageC10TableIntegrityError(f"{context} OK metric lacks value")
            return value["value"], "OK", None, False, _plain(value.get("details", {}))
        if status == "NA":
            if value.get("value") is not None:
                raise StageC10TableIntegrityError(f"{context} NA metric has a value")
            reason = value.get("reason")
            if not isinstance(reason, str) or not reason:
                raise StageC10TableIntegrityError(f"{context} NA lacks a reason")
            return None, ("INVALID" if reason in INVALIDATING_NA_REASONS else "NA"), reason, reason in INVALIDATING_NA_REASONS, _plain(value.get("details", {}))
        raise StageC10TableIntegrityError(f"{context} metric status is invalid")
    if value is None:
        raise StageC10TableIntegrityError(f"{context} unexpectedly contains null")
    _value_type(value)
    return value, "OK", None, False, {}


def _emit(
    rows: list[dict[str, Any]], base: Mapping[str, Any], *, family: str,
    metric: str, path: str, raw: Any, selectors: Mapping[str, Any] | None = None,
    source_kind: str = "c9_l0_scalar", details: Mapping[str, Any] | None = None,
) -> None:
    selectors = {} if selectors is None else dict(selectors)
    value, status, reason, invalidating, embedded_details = _metric_state(raw, path)
    merged_details = dict(embedded_details)
    if details is not None:
        merged_details.update(_plain(details))
    rows.append(make_row(
        base, metric_family=family,
        metric_id=metric_id_from_selectors(metric, **selectors),
        source_field_path=path, source_kind=source_kind, value=value,
        status=status, na_reason=reason, invalidating=invalidating,
        details=merged_details,
        input_variant=selectors.get("input_variant", selectors.get("perturbation")),
        perturbation_id=selectors.get("perturbation"),
        fidelity_scale=selectors.get("scale"),
        curve_direction=selectors.get("direction"),
        ranking_kind=selectors.get("ranking"),
        replicate_index=selectors.get("replicate"),
        fraction_index=selectors.get("fraction_index"),
        fraction=selectors.get("fraction"),
        statistic=selectors.get("statistic"), view=selectors.get("view"),
        component=selectors.get("component"),
    ))


def _unique_by(
    values: Any, *, collection: str, selector_fields: Sequence[str],
    expected: Iterable[tuple[Any, ...]],
) -> dict[tuple[Any, ...], tuple[Mapping[str, Any], str]]:
    items = _sequence(values, f"scalars.{collection}")
    found: dict[tuple[Any, ...], tuple[Mapping[str, Any], str]] = {}
    for index, raw in enumerate(items):
        item = _mapping(raw, f"scalars.{collection}[{index}]")
        key = tuple(item.get(name) for name in selector_fields)
        if key in found:
            raise StageC10TableIntegrityError(
                f"scalars.{collection} has duplicate semantic selector {key!r}"
            )
        selectors = {name: value for name, value in zip(selector_fields, key)}
        found[key] = (item, _selector_path(collection, selectors, "" ).rstrip("/"))
    expected_set = set(expected)
    if set(found) != expected_set:
        missing = sorted(expected_set - set(found), key=repr)
        extra = sorted(set(found) - expected_set, key=repr)
        raise StageC10TableIntegrityError(
            f"scalars.{collection} applicability mismatch; missing={missing}, extra={extra}"
        )
    return found


def _container_field(
    item: Mapping[str, Any], field: str, context: str,
) -> Any:
    status = item.get("status")
    if status == "OK":
        if field not in item:
            raise StageC10TableIntegrityError(f"{context} lacks {field}")
        return item[field]
    if status == "NA":
        return _mapping(item.get("na"), f"{context}.na")
    raise StageC10TableIntegrityError(f"{context} status is invalid")


def _singleton(value: Any, context: str) -> Any:
    items = _sequence(value, context)
    if len(items) != 1:
        raise StageC10TableIntegrityError(f"{context} must contain one scalar")
    return items[0]


def _fidelity_curve_selectors() -> tuple[tuple[str, str, str, int | None], ...]:
    result: list[tuple[str, str, str, int | None]] = []
    for scale in FIDELITY_SCALES:
        for direction in CURVE_DIRECTIONS:
            result.extend(
                (scale, direction, TARGET_RANKING, replicate)
                for replicate in range(5)
            )
            result.extend(
                (scale, direction, RANDOM_RANKING, replicate)
                for replicate in range(5)
            )
            result.append((scale, direction, CENTER_RANKING, None))
    return tuple(result)


def _spec(
    family: str, metric: str, path: str, *, selectors: Mapping[str, Any] | None = None,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    selectors = {} if selectors is None else dict(selectors)
    return {
        "family": family, "metric": metric, "path": path,
        "selectors": selectors, "details": {} if details is None else dict(details),
    }


def _expected_specs(unit: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Enumerate the applicability matrix without looking at observations."""

    method = unit["key"]["method_id"]
    specs: list[dict[str, Any]] = []
    if unit["fidelity_eligible"]:
        for scale, direction, ranking, replicate in _fidelity_curve_selectors():
            identity = {
                "scale": scale, "direction": direction,
                "ranking": ranking, "replicate": replicate,
            }
            root = _selector_path("fidelity_curves", {
                "scale": scale, "direction": direction,
                "ranking_kind": ranking, "replicate": replicate,
            }, "").rstrip("/")
            specs.append(_spec("fidelity_curve", "fidelity_curve_auc", f"{root}/auc", selectors=identity))
            for fraction_index in range(21):
                selectors = {
                    **identity, "fraction_index": fraction_index,
                    "fraction": fraction_index / 20.0,
                }
                specs.append(_spec(
                    "fidelity_curve", "fidelity_curve_response",
                    f"{root}/responses{{fraction_index={fraction_index}}}",
                    selectors=selectors,
                ))
        for scale in FIDELITY_SCALES:
            for direction in CURVE_DIRECTIONS:
                root = _selector_path("fidelity_aggregates", {
                    "scale": scale, "direction": direction,
                    "ranking_kind": TARGET_RANKING,
                }, "").rstrip("/")
                for statistic, field in (("mean", "auc_mean"), ("population_std", "auc_std")):
                    selectors = {
                        "scale": scale, "direction": direction,
                        "ranking": TARGET_RANKING, "statistic": statistic,
                    }
                    specs.append(_spec("fidelity_aggregate", "fidelity_auc", f"{root}/{field}", selectors=selectors))
                for fraction_index in range(21):
                    for statistic, field in (("mean", "response_mean"), ("population_std", "response_std")):
                        selectors = {
                            "scale": scale, "direction": direction,
                            "ranking": TARGET_RANKING,
                            "fraction_index": fraction_index,
                            "fraction": fraction_index / 20.0,
                            "statistic": statistic,
                        }
                        specs.append(_spec(
                            "fidelity_aggregate", "fidelity_response",
                            f"{root}/{field}{{fraction_index={fraction_index}}}",
                            selectors=selectors,
                        ))
        for scale in FIDELITY_SCALES:
            for replicate in range(5):
                selectors0 = {
                    "scale": scale, "ranking": TARGET_RANKING,
                    "replicate": replicate,
                }
                root = _selector_path("top20", {
                    "scale": scale, "ranking_kind": TARGET_RANKING,
                    "replicate": replicate,
                }, "").rstrip("/")
                for field in ("sufficiency", "sufficiency_gap", "comprehensiveness"):
                    specs.append(_spec("top20", field, f"{root}/{field}", selectors=selectors0))
        ordinary = (
            ("pearson", ("signed", "positive", "negative")),
            ("spearman", ("signed", "positive", "negative")),
            ("jensen_shannon", ("positive", "negative")),
            ("top10_jaccard", ("positive", "negative")),
            ("top20_jaccard", ("positive", "negative")),
        )
        for perturbation in PERTURBATION_IDS:
            root = _selector_path("stability", {"perturbation_id": perturbation}, "").rstrip("/")
            for metric, views in ordinary:
                for view in views:
                    selectors = {"perturbation": perturbation, "view": view}
                    specs.append(_spec("stability", metric, f"{root}/fixed_target_primary/metrics/{metric}/{view}", selectors=selectors))
    if unit["structural_reference_only"]:
        raw_metrics = (
            ("pearson", "pearson"), ("spearman", "spearman"),
            ("jensen_shannon_divergence", "jensen_shannon_divergence"),
            ("kl", "kl/left_to_right"), ("kl", "kl/right_to_left"),
            ("hotspot_jaccard_top_10_percent", "hotspot_jaccard_top_10_percent"),
            ("hotspot_jaccard_top_20_percent", "hotspot_jaccard_top_20_percent"),
        )
        for perturbation in PERTURBATION_IDS:
            root = _selector_path("structural_perturbations", {"perturbation_id": perturbation}, "").rstrip("/")
            for metric, suffix in raw_metrics:
                component = suffix.rsplit("/", 1)[1] if "/" in suffix else None
                selectors = {"perturbation": perturbation}
                if component is not None:
                    selectors["component"] = component
                specs.append(_spec("raw_structural_stability", metric, f"{root}/metrics/{suffix}", selectors=selectors))
    if method == NNS_METHOD_ID:
        for perturbation in PERTURBATION_IDS:
            root = _selector_path("residual_stability", {"perturbation_id": perturbation}, "").rstrip("/")
            for metric, suffix, component in (
                ("residual_rho", "clean/rho", "clean"),
                ("residual_rho", "aligned_perturbed/rho", "aligned_perturbed"),
                ("residual_signed_spearman", "signed_spearman", None),
            ):
                selectors = {"perturbation": perturbation}
                if component is not None:
                    selectors["component"] = component
                specs.append(_spec("nns_residual", metric, f"{root}/{suffix}", selectors=selectors))
        for field in (
            "clean_formula_max_abs_error",
            "clean_natural_vs_fixed_replay_max_abs_error",
            "clean_natural_vs_fixed_replay_model_output_max_abs_error",
            "baseline_fixed_replay_vs_clean_A_matmul_natural_V0_max_abs_error",
        ):
            specs.append(_spec("endpoint_parity", field, f"/scalars/provenance_fragment/endpoint_parity/{field}"))
    if method in IG_METHOD_IDS:
        for field in ("abs", "rel", "pass", "delta", "sum"):
            specs.append(_spec("ig_conservation", f"ig_conservation_{field}", f"/scalars/provenance_fragment/ig_result/conservation/{field}"))
    if method == MATCHED_METHOD_ID:
        for field in (
            "token_contribution_count", "token_sum", "cls_contribution",
            "scalar_delta_f_z_minus_f_x0", "conservation_abs_error",
            "conservation_tolerance", "patch_support",
            "positive_patch_evidence_fraction",
        ):
            specs.append(_spec("matched_contribution", field, f"/scalars/provenance_fragment/matched_contribution/{field}"))
    if unit["sample"]["audit"]:
        for field in ("pass", "max_abs_error"):
            root = _selector_path("repeatability", {"repeat_index": 1}, "").rstrip("/")
            specs.append(_spec("repeatability", f"repeatability_{field}", f"{root}/{field}", selectors={"replicate": 1}))
    for metric, path in (
        ("wall_time_seconds", "/resources/wall_time_seconds"),
        ("artifact_bytes", "/resources/artifact_bytes"),
        ("forward_calls", "/resources/calls/forward_calls"),
        ("backward_calls", "/resources/calls/backward_calls"),
        ("cuda_peak_allocated_bytes", "/resources/cuda/peak_allocated_bytes"),
        ("cuda_peak_reserved_bytes", "/resources/cuda/peak_reserved_bytes"),
    ):
        specs.append(_spec("resources", metric, path))
    specs.append(_spec("prediction", "clean_top1_correct", "/scalars/prediction/clean_top1_correct"))
    for perturbation in PERTURBATION_IDS:
        collection = "structural_perturbations" if unit["structural_reference_only"] else "stability"
        root = _selector_path(collection, {"perturbation_id": perturbation}, "").rstrip("/")
        suffix = "prediction_flipped" if collection == "structural_perturbations" else "natural_prediction_descriptive/prediction_flipped"
        specs.append(_spec("prediction", "prediction_flipped", f"{root}/{suffix}", selectors={"perturbation": perturbation}))
    specs.append(_spec("unit_status", "unit_status", "/state"))
    return specs


def _emit_failed_template(
    base: Mapping[str, Any], unit: Mapping[str, Any], state: str,
) -> list[dict[str, Any]]:
    status = "UNIT_FAILED" if state == "failed" else "UNIT_INCOMPLETE"
    reason = "unit_failed" if state == "failed" else f"unit_{state}"
    return [
        make_row(
            base, metric_family=item["family"],
            metric_id=metric_id_from_selectors(item["metric"], **item["selectors"]),
            source_field_path=item["path"], value=None, status=status,
            na_reason=reason, invalidating=True, details=item["details"],
            input_variant=item["selectors"].get("input_variant"),
            perturbation_id=item["selectors"].get("perturbation"),
            fidelity_scale=item["selectors"].get("scale"),
            curve_direction=item["selectors"].get("direction"),
            ranking_kind=item["selectors"].get("ranking"),
            replicate_index=item["selectors"].get("replicate"),
            fraction_index=item["selectors"].get("fraction_index"),
            fraction=item["selectors"].get("fraction"),
            statistic=item["selectors"].get("statistic"),
            view=item["selectors"].get("view"),
            component=item["selectors"].get("component"),
        )
        for item in _expected_specs(unit)
    ]


def _expand_success_record(
    record: Mapping[str, Any], unit: Mapping[str, Any], *, plan_index: int,
) -> list[dict[str, Any]]:
    base = base_row(record, unit, plan_index=plan_index)
    scalars = _mapping(record["scalars"], "record.scalars")
    resources = _mapping(record["resources"], "record.resources")
    key = unit["key"]
    if scalars.get("sample_id") != key["sample_id"] or scalars.get("method_id") != key["method_id"]:
        raise StageC10TableIntegrityError(
            "scalar sample/method identity differs from unit key"
        )
    rows: list[dict[str, Any]] = []

    if unit["fidelity_eligible"]:
        curve_expected = _fidelity_curve_selectors()
        curves = _unique_by(
            scalars.get("fidelity_curves"), collection="fidelity_curves",
            selector_fields=("scale", "direction", "ranking_kind", "replicate"),
            expected=curve_expected,
        )
        for selector_key in curve_expected:
            curve, root = curves[selector_key]
            scale, direction, ranking, replicate = selector_key
            selectors0 = {
                "scale": scale, "direction": direction,
                "ranking": ranking, "replicate": replicate,
            }
            _emit(rows, base, family="fidelity_curve", metric="fidelity_curve_auc", path=f"{root}/auc", raw=_container_field(curve, "auc", root), selectors=selectors0)
            if curve.get("status") == "OK":
                responses = _sequence(curve.get("responses"), f"{root}/responses")
                fractions = _sequence(curve.get("curve_fractions"), f"{root}/curve_fractions")
                if len(responses) != 21 or list(fractions) != [index / 20.0 for index in range(21)]:
                    raise StageC10TableIntegrityError(f"{root} is not a complete 21-point curve")
            else:
                responses = [curve.get("na")] * 21
            for fraction_index, response in enumerate(responses):
                selectors = {
                    **selectors0, "fraction_index": fraction_index,
                    "fraction": fraction_index / 20.0,
                }
                _emit(rows, base, family="fidelity_curve", metric="fidelity_curve_response", path=f"{root}/responses{{fraction_index={fraction_index}}}", raw=response, selectors=selectors)

        aggregate_expected = {
            (scale, direction, TARGET_RANKING)
            for scale in FIDELITY_SCALES for direction in CURVE_DIRECTIONS
        }
        aggregates: dict[tuple[Any, ...], tuple[Mapping[str, Any], str]] = {}
        for index, raw in enumerate(_sequence(scalars.get("fidelity_aggregates"), "scalars.fidelity_aggregates")):
            aggregate = _mapping(raw, f"scalars.fidelity_aggregates[{index}]")
            curves_in_aggregate = _sequence(aggregate.get("curves"), f"aggregate[{index}].curves")
            if len(curves_in_aggregate) != 5:
                raise StageC10TableIntegrityError("fidelity aggregate must bind five curves")
            first = _mapping(curves_in_aggregate[0], f"aggregate[{index}].curves[0]")
            selector_key = (first.get("scale"), first.get("direction"), first.get("ranking_kind"))
            if selector_key in aggregates:
                raise StageC10TableIntegrityError("duplicate fidelity aggregate selectors")
            if any(
                (curve.get("scale"), curve.get("direction"), curve.get("ranking_kind")) != selector_key
                for curve in curves_in_aggregate if isinstance(curve, Mapping)
            ) or any(not isinstance(curve, Mapping) for curve in curves_in_aggregate):
                raise StageC10TableIntegrityError("aggregate embeds mixed curve selectors")
            selector_map = {"scale": selector_key[0], "direction": selector_key[1], "ranking_kind": selector_key[2]}
            aggregates[selector_key] = (aggregate, _selector_path("fidelity_aggregates", selector_map, "").rstrip("/"))
        if set(aggregates) != aggregate_expected:
            raise StageC10TableIntegrityError("fidelity aggregate applicability mismatch")
        for scale in FIDELITY_SCALES:
            for direction in CURVE_DIRECTIONS:
                aggregate, root = aggregates[(scale, direction, TARGET_RANKING)]
                for statistic, field in (("mean", "auc_mean"), ("population_std", "auc_std")):
                    selectors = {"scale": scale, "direction": direction, "ranking": TARGET_RANKING, "statistic": statistic}
                    _emit(rows, base, family="fidelity_aggregate", metric="fidelity_auc", path=f"{root}/{field}", raw=_container_field(aggregate, field, root), selectors=selectors)
                if aggregate.get("status") == "OK":
                    means = _sequence(aggregate.get("response_mean"), f"{root}/response_mean")
                    stds = _sequence(aggregate.get("response_std"), f"{root}/response_std")
                    if len(means) != 21 or len(stds) != 21:
                        raise StageC10TableIntegrityError("aggregate response vectors must have 21 values")
                else:
                    means = stds = [aggregate.get("na")] * 21
                for fraction_index in range(21):
                    for statistic, field, vector in (("mean", "response_mean", means), ("population_std", "response_std", stds)):
                        selectors = {"scale": scale, "direction": direction, "ranking": TARGET_RANKING, "fraction_index": fraction_index, "fraction": fraction_index / 20.0, "statistic": statistic}
                        _emit(rows, base, family="fidelity_aggregate", metric="fidelity_response", path=f"{root}/{field}{{fraction_index={fraction_index}}}", raw=vector[fraction_index], selectors=selectors)

        top_expected = {(scale, TARGET_RANKING, replicate) for scale in FIDELITY_SCALES for replicate in range(5)}
        top = _unique_by(scalars.get("top20"), collection="top20", selector_fields=("scale", "ranking_kind", "replicate"), expected=top_expected)
        for selector_key in sorted(top, key=repr):
            item, root = top[selector_key]
            scale, ranking, replicate = selector_key
            selectors = {"scale": scale, "ranking": ranking, "replicate": replicate}
            for field in ("sufficiency", "sufficiency_gap", "comprehensiveness"):
                _emit(rows, base, family="top20", metric=field, path=f"{root}/{field}", raw=_container_field(item, field, root), selectors=selectors)

        stability = _unique_by(scalars.get("stability"), collection="stability", selector_fields=("perturbation_id",), expected=((item,) for item in PERTURBATION_IDS))
        ordinary = (("pearson", ("signed", "positive", "negative")), ("spearman", ("signed", "positive", "negative")), ("jensen_shannon", ("positive", "negative")), ("top10_jaccard", ("positive", "negative")), ("top20_jaccard", ("positive", "negative")))
        for perturbation in PERTURBATION_IDS:
            item, root = stability[(perturbation,)]
            metrics = _mapping(_mapping(item.get("fixed_target_primary"), f"{root}/fixed_target_primary").get("metrics"), f"{root}/metrics")
            for metric, views in ordinary:
                branches = _mapping(metrics.get(metric), f"{root}/metrics/{metric}")
                for view in views:
                    _emit(rows, base, family="stability", metric=metric, path=f"{root}/fixed_target_primary/metrics/{metric}/{view}", raw=branches.get(view, _MISSING), selectors={"perturbation": perturbation, "view": view})

    if unit["structural_reference_only"]:
        structural = _unique_by(scalars.get("structural_perturbations"), collection="structural_perturbations", selector_fields=("perturbation_id",), expected=((item,) for item in PERTURBATION_IDS))
        for perturbation in PERTURBATION_IDS:
            item, root = structural[(perturbation,)]
            metrics = _mapping(item.get("metrics"), f"{root}/metrics")
            raw_metrics = (("pearson", "pearson", None), ("spearman", "spearman", None), ("jensen_shannon_divergence", "jensen_shannon_divergence", None), ("kl", "kl", "left_to_right"), ("kl", "kl", "right_to_left"), ("hotspot_jaccard_top_10_percent", "hotspot_jaccard_top_10_percent", None), ("hotspot_jaccard_top_20_percent", "hotspot_jaccard_top_20_percent", None))
            for metric, field, component in raw_metrics:
                raw = metrics.get(field, _MISSING)
                path = f"{root}/metrics/{field}"
                selectors = {"perturbation": perturbation}
                if component is not None:
                    raw = _mapping(raw, path).get(component, _MISSING)
                    path += f"/{component}"
                    selectors["component"] = component
                _emit(rows, base, family="raw_structural_stability", metric=metric, path=path, raw=raw, selectors=selectors)

    method = key["method_id"]
    if method == NNS_METHOD_ID:
        residual = _unique_by(scalars.get("residual_stability"), collection="residual_stability", selector_fields=("perturbation_id",), expected=((item,) for item in PERTURBATION_IDS))
        for perturbation in PERTURBATION_IDS:
            item, root = residual[(perturbation,)]
            for metric, suffix, component in (("residual_rho", "clean/rho", "clean"), ("residual_rho", "aligned_perturbed/rho", "aligned_perturbed"), ("residual_signed_spearman", "signed_spearman", None)):
                raw: Any = item
                for part in suffix.split("/"):
                    raw = _mapping(raw, f"{root}/{suffix}").get(part, _MISSING)
                selectors = {"perturbation": perturbation}
                if component is not None:
                    selectors["component"] = component
                _emit(rows, base, family="nns_residual", metric=metric, path=f"{root}/{suffix}", raw=raw, selectors=selectors)
        parity = _mapping(_mapping(scalars.get("provenance_fragment"), "provenance_fragment").get("endpoint_parity"), "endpoint_parity")
        for field in ("clean_formula_max_abs_error", "clean_natural_vs_fixed_replay_max_abs_error", "clean_natural_vs_fixed_replay_model_output_max_abs_error", "baseline_fixed_replay_vs_clean_A_matmul_natural_V0_max_abs_error"):
            _emit(rows, base, family="endpoint_parity", metric=field, path=f"/scalars/provenance_fragment/endpoint_parity/{field}", raw=parity.get(field, _MISSING))

    if method in IG_METHOD_IDS:
        fragment = _mapping(scalars.get("provenance_fragment"), "provenance_fragment")
        conservation = _mapping(_mapping(fragment.get("ig_result"), "ig_result").get("conservation"), "ig_result.conservation")
        for field in ("abs", "rel", "pass", "delta", "sum"):
            raw = _singleton(conservation.get(field), f"ig_result.conservation.{field}")
            _emit(rows, base, family="ig_conservation", metric=f"ig_conservation_{field}", path=f"/scalars/provenance_fragment/ig_result/conservation/{field}", raw=raw)

    if method == MATCHED_METHOD_ID:
        contribution = _mapping(_mapping(scalars.get("provenance_fragment"), "provenance_fragment").get("matched_contribution"), "matched_contribution")
        if contribution.get("status") != "OK":
            raise StageC10TableIntegrityError("matched contribution is not OK")
        for field in ("token_contribution_count", "token_sum", "cls_contribution", "scalar_delta_f_z_minus_f_x0", "conservation_abs_error", "conservation_tolerance", "patch_support", "positive_patch_evidence_fraction"):
            _emit(rows, base, family="matched_contribution", metric=field, path=f"/scalars/provenance_fragment/matched_contribution/{field}", raw=contribution.get(field, _MISSING))

    repeats = scalars.get("repeatability")
    if unit["sample"]["audit"]:
        repeat = _unique_by(repeats, collection="repeatability", selector_fields=("repeat_index",), expected=((1,),))[(1,)]
        item, root = repeat
        for field in ("pass", "max_abs_error"):
            _emit(rows, base, family="repeatability", metric=f"repeatability_{field}", path=f"{root}/{field}", raw=item.get(field, _MISSING), selectors={"replicate": 1})
    elif list(_sequence(repeats, "scalars.repeatability")):
        raise StageC10TableIntegrityError("non-audit unit contains repeatability rows")

    resource_fields = (("wall_time_seconds", resources), ("artifact_bytes", resources), ("forward_calls", _mapping(resources.get("calls"), "resources.calls")), ("backward_calls", _mapping(resources.get("calls"), "resources.calls")), ("cuda_peak_allocated_bytes", _mapping(resources.get("cuda"), "resources.cuda")), ("cuda_peak_reserved_bytes", _mapping(resources.get("cuda"), "resources.cuda")))
    for metric, container in resource_fields:
        field = metric.removeprefix("cuda_")
        path = f"/resources/cuda/{field}" if metric.startswith("cuda_") else f"/resources/calls/{field}" if metric.endswith("_calls") else f"/resources/{field}"
        _emit(rows, base, family="resources", metric=metric, path=path, raw=container.get(field, _MISSING))
    prediction = _mapping(scalars.get("prediction"), "scalars.prediction")
    _emit(rows, base, family="prediction", metric="clean_top1_correct", path="/scalars/prediction/clean_top1_correct", raw=prediction.get("clean_top1_correct", _MISSING))
    collection = "structural_perturbations" if unit["structural_reference_only"] else "stability"
    predictions = _unique_by(scalars.get(collection), collection=collection, selector_fields=("perturbation_id",), expected=((item,) for item in PERTURBATION_IDS))
    for perturbation in PERTURBATION_IDS:
        item, root = predictions[(perturbation,)]
        if collection == "structural_perturbations":
            raw = item.get("prediction_flipped", _MISSING)
            path = f"{root}/prediction_flipped"
        else:
            natural = _mapping(item.get("natural_prediction_descriptive"), f"{root}/natural_prediction_descriptive")
            raw = natural.get("prediction_flipped", _MISSING)
            path = f"{root}/natural_prediction_descriptive/prediction_flipped"
        _emit(rows, base, family="prediction", metric="prediction_flipped", path=path, raw=raw, selectors={"perturbation": perturbation})
    _emit(rows, base, family="unit_status", metric="unit_status", path="/state", raw=record["state"], source_kind="c9_unit_status")

    expected = _expected_specs(unit)
    expected_keys = Counter((item["family"], metric_id_from_selectors(item["metric"], **item["selectors"]), item["path"]) for item in expected)
    actual_keys = Counter((row["metric_family"], row["metric_id"], row["source_field_path"]) for row in rows)
    if actual_keys != expected_keys:
        raise StageC10TableIntegrityError("emitted rows differ from applicability matrix")
    return rows


def build_c10_long_table(
    evidence_records: Sequence[Mapping[str, Any]], plan_units: Sequence[Any], *,
    failed_unit_policy: str = "error",
) -> list[dict[str, Any]]:
    """Expand exact ordered plan units into atomic C10 base rows.

    ``failed_unit_policy='error'`` is the formal fail-closed mode.  ``'emit'``
    retains a complete expected template using ``UNIT_FAILED`` or
    ``UNIT_INCOMPLETE`` rows; it is intended for reconciliation/diagnostics.
    """

    if failed_unit_policy not in {"error", "emit"}:
        raise StageC10TableError(
            "failed_unit_policy must be 'error' or 'emit'"
        )
    if isinstance(evidence_records, (str, bytes, Mapping)) or isinstance(
        plan_units, (str, bytes, Mapping)
    ):
        raise TypeError("evidence_records and plan_units must be sequences")
    units = [_plan_fields(unit) for unit in plan_units]
    unit_digests = [unit["key"]["canonical_digest"] for unit in units]
    if len(unit_digests) != len(set(unit_digests)):
        raise StageC10TableIntegrityError("plan has duplicate execution units")
    by_digest: dict[str, Mapping[str, Any]] = {}
    for index, raw in enumerate(evidence_records):
        record = _mapping(raw, f"evidence_records[{index}]")
        key = _mapping(record.get("unit_key"), f"evidence_records[{index}].unit_key")
        digest = _require_sha256(
            key.get("canonical_digest"),
            f"evidence_records[{index}] unit digest",
        )
        assert digest is not None
        if digest in by_digest:
            raise StageC10TableIntegrityError(
                f"duplicate evidence record for unit {digest}"
            )
        by_digest[digest] = record
    extras = sorted(set(by_digest) - set(unit_digests))
    if extras:
        raise StageC10TableIntegrityError(
            f"evidence contains units outside the plan: {extras}"
        )
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for index, unit in enumerate(units):
        digest = unit["key"]["canonical_digest"]
        record = by_digest.get(digest)
        if record is None:
            synthetic = {
                "plan_index": index, "unit_key": unit["key"],
                "state": "missing", "attempt_path": None,
                "audit": unit["sample"]["audit"], "failure": None,
            }
            if failed_unit_policy == "error":
                failures.append(f"{digest}:missing")
                continue
            base = base_row(synthetic, unit, plan_index=index)
            rows.extend(_emit_failed_template(base, unit, "missing"))
            continue
        state = record.get("state")
        if state != "succeeded":
            if failed_unit_policy == "error":
                failures.append(f"{digest}:{state}")
                continue
            base = base_row(record, unit, plan_index=index)
            rows.extend(_emit_failed_template(base, unit, str(state)))
            continue
        rows.extend(_expand_success_record(record, unit, plan_index=index))
    if failures:
        raise StageC10TableIntegrityError(
            "formal C10 requires every C9 unit to succeed; " + ", ".join(failures)
        )
    return validate_rows(rows, sort=True)


def expected_base_row_count(plan_units: Sequence[Any]) -> int:
    """Derive the L0 row-space size from the exact plan applicability."""

    if isinstance(plan_units, (str, bytes, bytearray, Mapping)):
        raise TypeError("plan_units must be a sequence")
    return sum(
        len(_expected_specs(_plan_fields(unit))) for unit in plan_units
    )


def expected_base_row_coordinates(
    plan_units: Sequence[Any],
) -> Iterator[dict[str, Any]]:
    """Return the exact observation-independent L0 applicability keys."""

    if isinstance(plan_units, (str, bytes, bytearray, Mapping)):
        raise TypeError("plan_units must be a sequence")
    for index, raw_unit in enumerate(plan_units):
        unit = _plan_fields(raw_unit)
        key = unit["key"]
        for item in _expected_specs(unit):
            yield {
                "plan_index": index,
                "sample_id": key["sample_id"],
                "method_id": key["method_id"],
                "reference_method_id": None,
                "config_id": key["config_id"],
                "source_unit_key_sha256": key["canonical_digest"],
                "metric_family": item["family"],
                "metric_id": metric_id_from_selectors(
                    item["metric"], **item["selectors"]
                ),
                "source_kind": (
                    "c9_unit_status"
                    if item["family"] == "unit_status"
                    else "c9_l0_scalar"
                ),
                "source_field_path": item["path"],
            }


def row_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    """Return the frozen deterministic row ordering key."""

    def optional_int(value: Any) -> int:
        return -1 if value is None else int(value)

    return (
        optional_int(row.get("plan_index")), str(row.get("sample_id")),
        str(row.get("method_id")), str(row.get("config_id") or ""),
        str(row.get("metric_family")), str(row.get("metric_id")),
        str(row.get("reference_method_id") or ""),
        str(row.get("source_binding_sha256")),
    )


def _validate_row(row: Any, context: str) -> dict[str, Any]:
    item = _plain(_mapping(row, context), context)
    if set(item) != set(CSV_COLUMNS):
        raise StageC10TableIntegrityError(
            f"{context} columns differ from the frozen schema"
        )
    if item["schema_version"] != SCHEMA_VERSION:
        raise StageC10TableIntegrityError(f"{context} schema version is invalid")
    for name in ("audit", "fidelity_eligible", "structural_reference_only", "invalidating"):
        if type(item[name]) is not bool:
            raise StageC10TableIntegrityError(f"{context}.{name} must be boolean")
    for name in _NULLABLE_BOOL_COLUMNS:
        if item[name] is not None and type(item[name]) is not bool:
            raise StageC10TableIntegrityError(f"{context}.{name} is invalid")
    for name in _NULLABLE_INT_COLUMNS:
        if item[name] is not None and type(item[name]) is not int:
            raise StageC10TableIntegrityError(f"{context}.{name} is invalid")
    for name in _NULLABLE_FLOAT_COLUMNS:
        if item[name] is not None and type(item[name]) is not float:
            raise StageC10TableIntegrityError(f"{context}.{name} is invalid")
    for name in _NULLABLE_STRING_COLUMNS:
        if item[name] == "":
            raise StageC10TableIntegrityError(
                f"{context}.{name} must use null rather than an empty string"
            )
    if item["sample_role"] not in {"calibration", "holdout"}:
        raise StageC10TableIntegrityError(f"{context} sample role is invalid")
    if item["status"] not in ROW_STATUSES:
        raise StageC10TableIntegrityError(f"{context} status is invalid")
    if _value_type(item["value"]) != item["value_type"]:
        raise StageC10TableIntegrityError(f"{context} value type is inconsistent")
    if item["status"] == "OK":
        if item["value"] is None or item["na_reason"] is not None or item["invalidating"]:
            raise StageC10TableIntegrityError(f"{context} OK state is inconsistent")
    elif item["value"] is not None or not item["na_reason"]:
        raise StageC10TableIntegrityError(f"{context} non-OK state is inconsistent")
    for name in _JSON_COLUMNS:
        raw = item[name]
        if not isinstance(raw, str):
            raise StageC10TableIntegrityError(f"{context}.{name} must be canonical JSON text")
        try:
            decoded = json.loads(raw)
        except (TypeError, ValueError) as error:
            raise StageC10TableIntegrityError(f"{context}.{name} is invalid JSON") from error
        if raw != _canonical_json(decoded):
            raise StageC10TableIntegrityError(f"{context}.{name} is not canonical JSON")
    item["secondary_sources_json"] = _normalize_secondary_sources(item["secondary_sources_json"])
    if _forbidden_quantitative_source(item["source_kind"], item["source_field_path"]):
        raise StageC10TableIntegrityError(f"{context} uses forbidden L3/display/image source")
    for name in _HASH_COLUMNS:
        _require_sha256(item[name], f"{context}.{name}", nullable=name.startswith("source_") and name not in {"source_binding_sha256"})
    if item["source_binding_sha256"] != _sha256(_row_source_binding_payload(item)):
        raise StageC10TableIntegrityError(f"{context} source binding hash disagrees")
    if item["row_key_sha256"] != _sha256(_row_key_payload(item)):
        raise StageC10TableIntegrityError(f"{context} row key hash disagrees")
    if item["row_hash_sha256"] != _sha256({key: item[key] for key in CSV_COLUMNS if key != "row_hash_sha256"}):
        raise StageC10TableIntegrityError(f"{context} row hash disagrees")
    return item


def validate_rows(
    rows: Sequence[Mapping[str, Any]], *, require_sorted: bool = False,
    sort: bool = False,
) -> list[dict[str, Any]]:
    """Validate rows, reject duplicates, and optionally sort canonically."""

    if isinstance(rows, (str, bytes, Mapping)):
        raise TypeError("rows must be a sequence of mappings")
    result = [_validate_row(row, f"rows[{index}]") for index, row in enumerate(rows)]
    keys = [row["row_key_sha256"] for row in result]
    if len(keys) != len(set(keys)):
        raise StageC10TableIntegrityError("table contains duplicate row keys")
    ordered = sorted(result, key=row_sort_key)
    if require_sorted and result != ordered:
        raise StageC10TableIntegrityError("rows are not in canonical order")
    return ordered if sort else result


def rows_to_csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    """Validate/sort rows and return canonical UTF-8 RFC-4180 CSV bytes."""

    ordered = validate_rows(rows, sort=True)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer, fieldnames=CSV_COLUMNS, extrasaction="raise", lineterminator="\n"
    )
    writer.writeheader()
    for row in ordered:
        encoded: dict[str, str] = {}
        for name in CSV_COLUMNS:
            value = row[name]
            if value is None:
                encoded[name] = ""
            elif type(value) is bool:
                encoded[name] = "true" if value else "false"
            elif type(value) in {int, float}:
                encoded[name] = repr(value)
            else:
                encoded[name] = value
        writer.writerow(encoded)
    return buffer.getvalue().encode("utf-8")


def _parse_bool(value: str, context: str, *, nullable: bool) -> bool | None:
    if nullable and value == "":
        return None
    if value == "true":
        return True
    if value == "false":
        return False
    raise StageC10TableIntegrityError(f"{context} must be true/false")


def _parse_int(value: str, context: str, *, nullable: bool) -> int | None:
    if nullable and value == "":
        return None
    if re.fullmatch(r"0|-?[1-9][0-9]*", value) is None:
        raise StageC10TableIntegrityError(f"{context} is not canonical integer text")
    return int(value)


def _parse_float(value: str, context: str, *, nullable: bool) -> float | None:
    if nullable and value == "":
        return None
    try:
        result = float(value)
    except ValueError as error:
        raise StageC10TableIntegrityError(f"{context} is not a number") from error
    if not math.isfinite(result) or repr(result) != value:
        raise StageC10TableIntegrityError(f"{context} is not canonical finite float text")
    return result


def read_csv_bytes(data: bytes, *, require_sorted: bool = True) -> list[dict[str, Any]]:
    """Strictly parse canonical table bytes back to typed row dictionaries."""

    if not isinstance(data, bytes):
        raise TypeError("CSV payload must be bytes")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise StageC10TableIntegrityError("CSV is not UTF-8") from error
    if not text.endswith("\n") or "\r" in text:
        raise StageC10TableIntegrityError("CSV must use canonical LF termination")
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if tuple(reader.fieldnames or ()) != CSV_COLUMNS:
        raise StageC10TableIntegrityError("CSV header differs from frozen schema")
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(reader):
        if None in raw or any(value is None for value in raw.values()):
            raise StageC10TableIntegrityError(f"CSV row {index} has missing/extra fields")
        row: dict[str, Any] = {}
        for name, value in raw.items():
            assert value is not None
            context = f"CSV row {index}.{name}"
            if name in _BOOL_COLUMNS:
                row[name] = _parse_bool(value, context, nullable=False)
            elif name in _NULLABLE_BOOL_COLUMNS:
                row[name] = _parse_bool(value, context, nullable=True)
            elif name == "schema_version":
                row[name] = _parse_int(value, context, nullable=False)
            elif name in _NULLABLE_INT_COLUMNS:
                row[name] = _parse_int(value, context, nullable=True)
            elif name in _NULLABLE_FLOAT_COLUMNS:
                row[name] = _parse_float(value, context, nullable=True)
            elif name == "value":
                value_type = raw.get("value_type")
                if value_type == "null":
                    if value != "":
                        raise StageC10TableIntegrityError(f"{context} null encoding is invalid")
                    row[name] = None
                elif value_type == "bool":
                    row[name] = _parse_bool(value, context, nullable=False)
                elif value_type == "int":
                    row[name] = _parse_int(value, context, nullable=False)
                elif value_type == "float":
                    row[name] = _parse_float(value, context, nullable=False)
                elif value_type == "string":
                    row[name] = value
                else:
                    raise StageC10TableIntegrityError(f"{context} has unknown value_type")
            elif name in _NULLABLE_STRING_COLUMNS:
                row[name] = None if value == "" else value
            else:
                row[name] = value
        rows.append(row)
    result = validate_rows(rows, require_sorted=require_sorted)
    if rows_to_csv_bytes(result) != data:
        raise StageC10TableIntegrityError("CSV bytes are not canonical round-trip output")
    return result


def build_row_manifest(
    rows: Sequence[Mapping[str, Any]], *, csv_bytes: bytes | None = None,
) -> dict[str, Any]:
    """Build the row-count/hash/reconciliation manifest."""

    ordered = validate_rows(rows, sort=True)
    payload = rows_to_csv_bytes(ordered) if csv_bytes is None else csv_bytes
    if not isinstance(payload, bytes) or read_csv_bytes(payload) != ordered:
        raise StageC10TableIntegrityError("manifest CSV does not encode supplied rows")
    def counts(name: str) -> dict[str, int]:
        return dict(sorted(Counter(str(row[name]) for row in ordered).items()))
    unit_rows: dict[str, int] = Counter(
        str(row["source_unit_key_sha256"]) for row in ordered
    )
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "document_type": "stage_c_c10_row_manifest",
        "row_count": len(ordered),
        "csv_sha256": hashlib.sha256(payload).hexdigest(),
        "schema_sha256": hashlib.sha256((
            json.dumps(
                schema_document(), indent=2, sort_keys=True,
                ensure_ascii=False, allow_nan=False,
            ) + "\n"
        ).encode("utf-8")).hexdigest(),
        "schema_canonical_digest": _sha256(schema_document()),
        "ordered_row_hashes_sha256": _sha256(
            [row["row_hash_sha256"] for row in ordered]
        ),
        "ordered_row_keys_sha256": _sha256(
            [row["row_key_sha256"] for row in ordered]
        ),
        "status_counts": counts("status"),
        "role_counts": counts("sample_role"),
        "family_counts": counts("metric_family"),
        "source_kind_counts": counts("source_kind"),
        "unit_count": len(unit_rows),
        "unit_row_counts_sha256": _sha256(dict(sorted(unit_rows.items()))),
        "l3_display_image_sources_forbidden": True,
        "l3_display_image_source_count": 0,
        "reference_full_success_base_row_count": REFERENCE_FULL_C9_BASE_ROW_COUNT,
        "reference_count_is_validation_rule": False,
    }
    result["canonical_digest"] = _sha256(result)
    return result


def validate_csv_bytes(data: bytes) -> dict[str, Any]:
    """Read, type-check, canonicalize, and summarize one CSV payload."""

    rows = read_csv_bytes(data, require_sorted=True)
    return build_row_manifest(rows, csv_bytes=data)


# Friendly aliases for integration code.
write_csv_bytes = rows_to_csv_bytes
read_rows_csv = read_csv_bytes
row_manifest = build_row_manifest

__all__ = [
    "CSV_COLUMNS", "DOCUMENT_TYPE", "REFERENCE_FULL_C9_BASE_ROW_COUNT",
    "ROW_STATUSES", "SCHEMA_VERSION", "StageC10TableError",
    "StageC10TableIntegrityError", "base_row", "build_c10_long_table",
    "build_row_manifest", "expected_base_row_coordinates",
    "expected_base_row_count", "make_row",
    "metric_id_from_selectors",
    "read_csv_bytes", "read_rows_csv", "row_manifest", "row_sort_key",
    "rows_to_csv_bytes", "schema_document", "source_binding",
    "validate_csv_bytes", "validate_rows", "write_csv_bytes",
]
