"""Deterministic, model-free Stage D analysis and paper packages.

This module is deliberately a post-processing boundary.  It accepts an
already computed analysis result (or the exact mapping returned by its
``to_dict`` method), and never knows how to open an execution root, L0/L1
evidence, an image, or a model.  ``analysis.json`` is the sole numeric source
for every table and figure recipe in the package.

Publication is fail-closed: a complete private sibling directory is fsynced
and then installed with Linux ``renameat2(RENAME_NOREPLACE)``.  Existing
destinations are never replaced.
"""

from __future__ import annotations

import csv
import ctypes
import errno
import hashlib
import io
import json
import math
import os
import secrets
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from jsonschema import Draft202012Validator

from .stage_d_analysis import StageDAnalysisResult
from .stage_d_provenance import canonical_json_bytes, canonical_sha256

SCHEMA_VERSION = 1
ANALYSIS_DOCUMENT_TYPE = "stage_d_analysis_result"
PACKAGE_SCHEMA_PATH = "config/stage_d_analysis_package.schema.json"
FIGURE_NAMESPACE = "stage-d-figure-bootstrap|20260821|"

PER_SAMPLE_COLUMNS = (
    "selection_order", "sample_id", "status", "validity_status",
    "misclassified", "nns_insertion_auc", "nns_deletion_auc",
    "matched_insertion_auc", "matched_deletion_auc", "delta_fid",
    "nns_insertion_auc_repeat_sd_ddof_0",
    "nns_deletion_auc_repeat_sd_ddof_0",
    "matched_insertion_auc_repeat_sd_ddof_0",
    "matched_deletion_auc_repeat_sd_ddof_0",
)
MAIN_COLUMNS = (
    "cohort", "n", "metric_id", "positive_count", "mean",
    "median", "standard_deviation_sample_ddof_1",
    "bootstrap_ci_lower", "bootstrap_ci_upper", "fidelity_status",
    "overall_decision",
)
COHORT_COLUMNS = (
    "cohort", "n", "historical", "pooled", "validity_status",
    "fidelity_status", "overall_decision", "workflow_status",
    "spatial_status", "holm_status", "median_delta_fid",
    "bootstrap_ci_lower", "bootstrap_ci_upper",
)
QC_COLUMNS = ("field_path", "value_json")
CURVE_COLUMNS = (
    "method_id", "direction", "fraction_index", "fraction", "n",
    "median", "pointwise_ci_lower", "pointwise_ci_upper",
)

PAYLOAD_FILES = (
    "analysis.json",
    "analysis_report.json",
    "decision.json",
    "figure_recipes.json",
    "paper_data/cohort_comparison.csv",
    "paper_data/cohort_comparison.json",
    "paper_data/curve_summary.csv",
    "paper_data/curve_summary.json",
    "paper_data/main_fidelity_table.csv",
    "paper_data/main_fidelity_table.json",
    "paper_data/per_sample_delta.csv",
    "paper_data/per_sample_delta.json",
    "paper_data/resource_and_qc.csv",
    "paper_data/resource_and_qc.json",
    "provenance.json",
)
MANIFEST_FILE = "payload_manifest.json"
STATUS_FILE = "status.json"
ALL_FILES = (*PAYLOAD_FILES, MANIFEST_FILE, STATUS_FILE)
ALL_DIRECTORIES = ("paper_data",)

_ANALYSIS_FIELDS = frozenset({
    "schema_version", "document_type", "validity_status",
    "fidelity_status", "cost_status", "overall_decision",
    "per_sample", "summary", "bootstrap", "curves",
    "c10_reference", "resources_and_qc", "provenance",
    "canonical_digest",
})
_PER_SAMPLE_FIELDS = frozenset(PER_SAMPLE_COLUMNS)
_CURVE_FIELDS = frozenset(CURVE_COLUMNS)
_SHA256 = frozenset("0123456789abcdef")


class StageDPaperError(RuntimeError):
    """Base error for deterministic Stage D paper packages."""


class StageDPaperContractError(StageDPaperError, ValueError):
    """The analysis value or package violates the frozen contract."""


class StageDPaperBlockedError(StageDPaperError):
    """Publication is blocked without changing the destination."""


def _plain(value: Any, context: str = "value") -> Any:
    method = getattr(value, "to_dict", None)
    if callable(method):
        value = method()
    if value is None or type(value) in {str, bool, int}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise StageDPaperContractError(f"{context} is non-finite")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, child in value.items():
            if not isinstance(key, str) or not key:
                raise StageDPaperContractError(f"{context} has an invalid key")
            result[key] = _plain(child, f"{context}.{key}")
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_plain(child, f"{context}[]") for child in value]
    raise StageDPaperContractError(
        f"{context} contains unsupported {type(value).__name__}"
    )


def _json_file_bytes(value: Any) -> bytes:
    return canonical_json_bytes(_plain(value)) + b"\n"


def _self_document(document_type: str, **fields: Any) -> dict[str, Any]:
    result = {
        "schema_version": SCHEMA_VERSION,
        "document_type": document_type,
        **_plain(fields),
    }
    result["canonical_digest"] = canonical_sha256(result)
    return result


def _validate_self_digest(document: Mapping[str, Any], context: str) -> None:
    claimed = document.get("canonical_digest")
    if not isinstance(claimed, str) or len(claimed) != 64 or any(
        character not in _SHA256 for character in claimed
    ):
        raise StageDPaperContractError(f"{context} has an invalid canonical_digest")
    unsigned = {key: value for key, value in document.items() if key != "canonical_digest"}
    if claimed != canonical_sha256(unsigned):
        raise StageDPaperContractError(f"{context} canonical_digest mismatch")


def _strict_json(raw: bytes, context: str) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise StageDPaperContractError(f"{context} has duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8"), object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite token {token}")
            ),
        )
    except StageDPaperContractError:
        raise
    except Exception as error:
        raise StageDPaperContractError(f"invalid JSON in {context}: {error}") from error
    if not isinstance(value, dict) or raw != _json_file_bytes(value):
        raise StageDPaperContractError(f"{context} is not canonical JSON-file bytes")
    return value


def _exact_mapping(value: Any, fields: frozenset[str], context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise StageDPaperContractError(f"{context} must be an object")
    result = _plain(value, context)
    if set(result) != set(fields):
        raise StageDPaperContractError(
            f"{context} fields differ: missing={sorted(fields-set(result))}, "
            f"extra={sorted(set(result)-fields)}"
        )
    return result


def _finite(value: Any, context: str, *, nullable: bool = False) -> float | None:
    if value is None and nullable:
        return None
    if type(value) not in {int, float} or type(value) is bool:
        raise StageDPaperContractError(f"{context} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise StageDPaperContractError(f"{context} must be finite")
    return result


def _load_schema(repo_root: Path | str | None) -> dict[str, Any]:
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
    path = root / PACKAGE_SCHEMA_PATH
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise StageDPaperContractError(f"required analysis schema is unavailable: {path}") from error
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise StageDPaperContractError(
                    f"{PACKAGE_SCHEMA_PATH} has duplicate key {key!r}"
                )
            result[key] = value
        return result

    try:
        schema = json.loads(
            raw.decode("utf-8"), object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite token {token}")
            ),
        )
    except Exception as error:
        raise StageDPaperContractError("analysis package schema is not strict JSON") from error
    if not isinstance(schema, dict):
        raise StageDPaperContractError("analysis package schema must be an object")
    try:
        Draft202012Validator.check_schema(schema)
    except Exception as error:
        raise StageDPaperContractError("analysis package schema is invalid") from error
    return schema


def _validate_document_schema(
    document: Mapping[str, Any], schema: Mapping[str, Any], context: str,
) -> None:
    errors = sorted(
        Draft202012Validator(schema).iter_errors(document),
        key=lambda item: [str(part) for part in item.path],
    )
    if errors:
        location = ".".join(str(item) for item in errors[0].path) or "root"
        raise StageDPaperContractError(
            f"{context} fails {PACKAGE_SCHEMA_PATH} at {location}: "
            f"{errors[0].message}"
        )


def _validate_analysis(value: Any, *, repo_root: Path | str | None) -> dict[str, Any]:
    result = _plain(value, "analysis result")
    result = _exact_mapping(result, _ANALYSIS_FIELDS, "analysis result")
    if result["schema_version"] != SCHEMA_VERSION or result["document_type"] != ANALYSIS_DOCUMENT_TYPE:
        raise StageDPaperContractError("analysis result identity is invalid")
    _validate_self_digest(result, "analysis result")
    schema = _load_schema(repo_root)
    _validate_document_schema(result, schema, "analysis result")

    rows = result["per_sample"]
    if not isinstance(rows, list) or len(rows) != 500:
        raise StageDPaperContractError("analysis.per_sample must contain exactly 500 rows")
    checked_rows: list[dict[str, Any]] = []
    for index, raw in enumerate(rows):
        row = _exact_mapping(raw, _PER_SAMPLE_FIELDS, f"per_sample[{index}]")
        if row["selection_order"] != index or not isinstance(row["sample_id"], str) or not row["sample_id"]:
            raise StageDPaperContractError(f"per_sample[{index}] identity/order is invalid")
        if row["misclassified"] is not None and type(row["misclassified"]) is not bool:
            raise StageDPaperContractError(
                f"per_sample[{index}].misclassified must be bool or null"
            )
        for field in PER_SAMPLE_COLUMNS[5:]:
            _finite(row[field], f"per_sample[{index}].{field}", nullable=True)
        checked_rows.append(row)
    if len({row["sample_id"] for row in checked_rows}) != 500:
        raise StageDPaperContractError("analysis sample identifiers must be unique")

    if result["validity_status"] not in {"PASS", "FAIL"}:
        raise StageDPaperContractError("analysis.validity_status must be PASS or FAIL")
    valid = result["validity_status"] == "PASS"
    if not valid:
        if result["overall_decision"] != "NO_GO_VALIDITY" or result["bootstrap"] is not None:
            raise StageDPaperContractError("invalid analysis must suppress bootstrap and decide NO_GO_VALIDITY")
    elif result["bootstrap"] is None:
        raise StageDPaperContractError("valid analysis requires bootstrap metadata")
    else:
        _validate_primary_bootstrap(result)
    if not isinstance(result["summary"], Mapping):
        raise StageDPaperContractError("analysis.summary must be an object")
    _summary_fields(result)
    if not isinstance(result["curves"], list):
        raise StageDPaperContractError("analysis.curves must be an array")
    for index, raw in enumerate(result["curves"]):
        row = _exact_mapping(raw, _CURVE_FIELDS, f"curves[{index}]")
        if row["method_id"] not in {
            "conditional_fixed_clean_attention_head_direction_value_path_ig",
            "matched_fixed_clean_attention_patch_token_contribution",
        } or row["direction"] not in {"insertion", "deletion"}:
            raise StageDPaperContractError(f"curves[{index}] method/direction is invalid")
        if type(row["fraction_index"]) is not int or not 0 <= row["fraction_index"] <= 20:
            raise StageDPaperContractError(f"curves[{index}].fraction_index is invalid")
        for field in ("fraction", "median", "pointwise_ci_lower", "pointwise_ci_upper"):
            _finite(row[field], f"curves[{index}].{field}")
        if row["n"] != 500:
            raise StageDPaperContractError(f"curves[{index}].n must be 500")
        if not math.isclose(
            row["fraction"], row["fraction_index"] / 20.0,
            abs_tol=0.0, rel_tol=0.0,
        ):
            raise StageDPaperContractError(f"curves[{index}].fraction is not frozen")
    if valid:
        if len(result["curves"]) != 84:
            raise StageDPaperContractError("valid analysis must contain all 84 pointwise curve rows")
        coordinates = [
            (row["method_id"], row["direction"], row["fraction_index"])
            for row in result["curves"]
        ]
        expected_coordinates = [
            (method, direction, index)
            for method in (
                "conditional_fixed_clean_attention_head_direction_value_path_ig",
                "matched_fixed_clean_attention_patch_token_contribution",
            )
            for direction in ("insertion", "deletion")
            for index in range(21)
        ]
        if sorted(coordinates) != sorted(expected_coordinates):
            raise StageDPaperContractError("valid analysis curve coordinates are incomplete or duplicated")
        if any(
            row[field] is None
            for row in checked_rows for field in PER_SAMPLE_COLUMNS[5:]
        ):
            raise StageDPaperContractError("valid analysis cannot contain missing effect values")
    elif result["curves"]:
        raise StageDPaperContractError("invalid analysis must suppress descriptive curves")
    c10 = result["c10_reference"]
    if (
        not isinstance(c10, Mapping)
        or c10.get("n") != 15
        or c10.get("historical") is not True
        or c10.get("pooled") is not False
        or c10.get("overall_decision") != "NO_GO_NOVELTY"
        or c10.get("workflow_status") != "BLOCKED"
        or c10.get("spatial_status") != "FAIL_SPATIAL"
        or c10.get("holm_status") != "HOLM_BLOCKED_UNDERSPECIFIED"
    ):
        raise StageDPaperContractError("c10 historical cohort separation is invalid")
    provenance = result["provenance"]
    if not isinstance(provenance, Mapping):
        raise StageDPaperContractError("analysis provenance must be an object")
    synthetic_only = provenance.get("synthetic_only")
    formal_result = provenance.get("formal_result")
    if (
        type(synthetic_only) is not bool
        or type(formal_result) is not bool
        or synthetic_only == formal_result
        or provenance.get("holm_status") != "NOT_APPLICABLE_NO_SECONDARY_INFERENCE"
        or provenance.get("c10_stage_d_pooling_allowed") is not False
    ):
        raise StageDPaperContractError("analysis provenance has invalid scope controls")
    if synthetic_only:
        required_provenance = {
            "execution_authorization": "NOT_ISSUED",
            "analysis_authorization": "NOT_EVALUATED",
            "stage_d_execution_started": False,
        }
        if any(
            provenance.get(name) != expected
            for name, expected in required_provenance.items()
        ):
            raise StageDPaperContractError(
                "analysis provenance is not synthetic-only D6 scope"
            )
    else:
        required_formal = {
            "d9_analysis_authorization": "D9_AUTHORIZED_D10_FULL_ANALYSIS",
            "d9_allowed_path": "D10_FULL_ANALYSIS_AND_DECISION",
            "d9_sealed_root_read": True,
            "d9_l0_read": True,
            "d9_l1_read": True,
            "d9_l2_read": False,
            "jpeg_read": False,
            "model_loaded": False,
            "cuda_or_gpu_used": False,
            "d8_shards_run": False,
            "retry_attempts_created": False,
            "execution_authorization": "AUTHORIZED_STAGE_D_FIDELITY_SCOPE",
            "analysis_authorization": "D9_AUTHORIZED_D10_FULL_ANALYSIS",
            "stage_d_execution_started": True,
        }
        if any(
            provenance.get(name) != expected
            for name, expected in required_formal.items()
        ):
            raise StageDPaperContractError(
                "analysis provenance is not D9-authorized formal D10 scope"
            )
        for name in ("d9_report_sha256", "d9_report_canonical_digest"):
            value = provenance.get(name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in _SHA256 for character in value)
            ):
                raise StageDPaperContractError(
                    "formal analysis provenance lacks D9 report digests"
                )
    if valid:
        resources_and_qc = result["resources_and_qc"]
        curve_bootstrap = (
            resources_and_qc.get("descriptive_curve_bootstrap")
            if isinstance(resources_and_qc, Mapping) else None
        )
        if (
            not isinstance(curve_bootstrap, Mapping)
            or curve_bootstrap.get("namespace") != FIGURE_NAMESPACE
            or curve_bootstrap.get("simultaneous") is not False
            or curve_bootstrap.get("enters_primary_decision") is not False
            or not isinstance(curve_bootstrap.get("coordinates"), list)
            or len(curve_bootstrap["coordinates"]) != 84
        ):
            raise StageDPaperContractError("descriptive curve bootstrap is incomplete")
    return result


def _csv_bytes(columns: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(columns), extrasaction="raise", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        encoded: dict[str, str] = {}
        if set(row) != set(columns):
            raise StageDPaperContractError("CSV row differs from its frozen columns")
        for column in columns:
            value = row[column]
            if value is None:
                encoded[column] = ""
            elif type(value) is bool:
                encoded[column] = "true" if value else "false"
            elif type(value) in {int, float}:
                if type(value) is float and not math.isfinite(value):
                    raise StageDPaperContractError("CSV contains a non-finite number")
                encoded[column] = repr(value)
            else:
                encoded[column] = str(value)
        writer.writerow(encoded)
    return buffer.getvalue().encode("utf-8")


def _table_document(name: str, columns: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return _self_document(
        "stage_d_paper_table", table_id=name, columns=list(columns),
        row_count=len(rows), rows=list(rows),
    )


def _summary_fields(analysis: Mapping[str, Any]) -> dict[str, Any]:
    summary = analysis["summary"]
    required = {
        "metric_id", "n", "positive_count", "mean", "median",
        "standard_deviation_sample_ddof_1", "bootstrap_ci_lower",
        "bootstrap_ci_upper",
    }
    if set(summary) != required:
        raise StageDPaperContractError("analysis.summary fields differ from the frozen contract")
    if summary["n"] != 500:
        raise StageDPaperContractError("analysis.summary denominator is invalid")
    if analysis["validity_status"] == "PASS":
        if type(summary["positive_count"]) is not int or not 0 <= summary["positive_count"] <= 500:
            raise StageDPaperContractError("analysis.summary positive count is invalid")
    elif summary["positive_count"] is not None:
        raise StageDPaperContractError("invalid analysis must not retain a positive count")
    for field in ("mean", "median", "standard_deviation_sample_ddof_1"):
        _finite(summary[field], f"summary.{field}", nullable=analysis["validity_status"] != "PASS")
    for field in ("bootstrap_ci_lower", "bootstrap_ci_upper"):
        _finite(summary[field], f"summary.{field}", nullable=analysis["validity_status"] != "PASS")
    return dict(summary)


def _validate_primary_bootstrap(analysis: Mapping[str, Any]) -> None:
    bootstrap = analysis["bootstrap"]
    if not isinstance(bootstrap, Mapping):
        raise StageDPaperContractError("valid analysis bootstrap must be an object")
    required = {
        "status", "metric_id", "seed_material", "seed",
        "seed_hash_extraction", "prng", "resamples", "draw_shape",
        "draw_indices_dtype_and_order", "distribution_dtype_and_order",
        "statistic", "even_median_rule", "quantile_probabilities",
        "quantile_method", "confidence_interval", "draw_indices_sha256",
        "distribution_sha256",
    }
    if set(bootstrap) != required or bootstrap.get("status") != "PERFORMED":
        raise StageDPaperContractError("primary bootstrap metadata is not closed")
    frozen = {
        "metric_id": "fine_4x4_cell_delta_fid_median",
        "seed_material": "stage-d-bootstrap|20260821|fine_4x4_cell_delta_fid_median",
        "seed": "18136624947342364318",
        "seed_hash_extraction": "sha256_first_8_bytes_big_endian_unsigned",
        "prng": "numpy.random.Generator(numpy.random.PCG64(seed))",
        "resamples": 10000,
        "draw_indices_dtype_and_order": "little_endian_int64_c_order",
        "distribution_dtype_and_order": "little_endian_float64_c_order",
        "statistic": "paired_median",
        "even_median_rule": "arithmetic_mean_of_two_middle_order_statistics",
        "quantile_method": "linear",
    }
    if any(bootstrap.get(name) != value for name, value in frozen.items()):
        raise StageDPaperContractError("primary bootstrap frozen contract drifted")
    if list(bootstrap.get("draw_shape", ())) != [10000, 500] or list(
        bootstrap.get("quantile_probabilities", ())
    ) != [0.025, 0.975]:
        raise StageDPaperContractError("primary bootstrap shape or quantiles drifted")
    for name in ("draw_indices_sha256", "distribution_sha256"):
        value = bootstrap.get(name)
        if not isinstance(value, str) or len(value) != 64 or any(
            character not in _SHA256 for character in value
        ):
            raise StageDPaperContractError("primary bootstrap digest is invalid")
    interval = bootstrap.get("confidence_interval")
    summary = analysis["summary"]
    if (
        not isinstance(interval, (list, tuple)) or len(interval) != 2
        or list(interval) != [summary["bootstrap_ci_lower"], summary["bootstrap_ci_upper"]]
    ):
        raise StageDPaperContractError("bootstrap interval does not match summary")


def _flatten_qc(value: Any, prefix: str = "") -> list[dict[str, str]]:
    if isinstance(value, Mapping):
        rows: list[dict[str, str]] = []
        for key in sorted(value):
            path = f"{prefix}.{key}" if prefix else key
            rows.extend(_flatten_qc(value[key], path))
        return rows
    if isinstance(value, list):
        rows = []
        for index, child in enumerate(value):
            rows.extend(_flatten_qc(child, f"{prefix}[{index}]"))
        return rows
    return [{"field_path": prefix, "value_json": canonical_json_bytes(value).decode("utf-8")}]


def _derived_payloads(analysis: Mapping[str, Any]) -> dict[str, bytes]:
    valid = analysis["validity_status"] == "PASS"
    provenance_source = analysis["provenance"]
    if not isinstance(provenance_source, Mapping):
        raise StageDPaperContractError("analysis provenance must be an object")
    formal_result = provenance_source.get("formal_result") is True
    synthetic_only = provenance_source.get("synthetic_only") is True
    summary = _summary_fields(analysis)
    per_sample = [dict(row) for row in analysis["per_sample"]]
    main_rows = [] if not valid else [{
        "cohort": "stage_d", "n": 500, "metric_id": summary["metric_id"],
        "positive_count": summary["positive_count"], "mean": summary["mean"],
        "median": summary["median"],
        "standard_deviation_sample_ddof_1": summary["standard_deviation_sample_ddof_1"],
        "bootstrap_ci_lower": summary["bootstrap_ci_lower"],
        "bootstrap_ci_upper": summary["bootstrap_ci_upper"],
        "fidelity_status": analysis["fidelity_status"],
        "overall_decision": analysis["overall_decision"],
    }]
    c10 = analysis["c10_reference"]
    if not isinstance(c10, Mapping) or c10.get("n") != 15:
        raise StageDPaperContractError("c10_reference must retain historical n=15")
    cohort_rows = [
        {
            "cohort": "c10", "n": 15, "historical": True, "pooled": False,
            "validity_status": c10.get("validity_status"),
            "fidelity_status": c10.get("fidelity_status"),
            "overall_decision": c10.get("overall_decision"),
            "workflow_status": c10.get("workflow_status"),
            "spatial_status": c10.get("spatial_status"),
            "holm_status": c10.get("holm_status"),
            "median_delta_fid": c10.get("median"),
            "bootstrap_ci_lower": c10.get("bootstrap_ci_lower"),
            "bootstrap_ci_upper": c10.get("bootstrap_ci_upper"),
        },
        {
            "cohort": "stage_d", "n": 500, "historical": False, "pooled": False,
            "validity_status": analysis["validity_status"],
            "fidelity_status": analysis["fidelity_status"],
            "overall_decision": analysis["overall_decision"],
            "workflow_status": "NOT_A_SCIENTIFIC_DECISION",
            "spatial_status": "OUT_OF_STAGE_D_SCOPE",
            "holm_status": "NOT_APPLICABLE_NO_SECONDARY_INFERENCE",
            "median_delta_fid": summary["median"] if valid else None,
            "bootstrap_ci_lower": summary["bootstrap_ci_lower"] if valid else None,
            "bootstrap_ci_upper": summary["bootstrap_ci_upper"] if valid else None,
        },
    ]
    qc_rows = _flatten_qc(analysis["resources_and_qc"])
    curves = sorted(
        (dict(row) for row in analysis["curves"]),
        key=lambda row: (row["method_id"], row["direction"], row["fraction_index"]),
    ) if valid else []
    placeholder = None if valid else {
        "status": "NOT_GENERATED_DUE_TO_VALIDITY",
        "reason": "primary validity did not pass; complete-case analysis is forbidden",
    }

    tables = {
        "per_sample_delta": (PER_SAMPLE_COLUMNS, per_sample),
        "main_fidelity_table": (MAIN_COLUMNS, main_rows),
        "cohort_comparison": (COHORT_COLUMNS, cohort_rows),
        "resource_and_qc": (QC_COLUMNS, qc_rows),
        "curve_summary": (CURVE_COLUMNS, curves),
    }
    payloads: dict[str, bytes] = {"analysis.json": _json_file_bytes(analysis)}
    for name, (columns, rows) in tables.items():
        document = _table_document(name, columns, rows)
        if placeholder is not None and name in {"main_fidelity_table", "curve_summary"}:
            document["generation"] = placeholder
            document["canonical_digest"] = canonical_sha256(
                {key: value for key, value in document.items() if key != "canonical_digest"}
            )
        payloads[f"paper_data/{name}.json"] = _json_file_bytes(document)
        payloads[f"paper_data/{name}.csv"] = _csv_bytes(columns, rows)

    decision = _self_document(
        "stage_d_paper_decision", validity_status=analysis["validity_status"],
        fidelity_status=analysis["fidelity_status"], cost_status=analysis["cost_status"],
        overall_decision=analysis["overall_decision"], component_flags={
            "validity_status": analysis["validity_status"],
            "fidelity_status": analysis["fidelity_status"],
            "cost_status": analysis["cost_status"],
        }, holm_status="NOT_APPLICABLE_NO_SECONDARY_INFERENCE",
        effect_artifacts_status=("GENERATED" if valid else "NOT_GENERATED_DUE_TO_VALIDITY"),
    )
    recipes = _self_document(
        "stage_d_figure_recipes", renders_png=False, numeric_source="paper_data_only",
        c10_stage_d_pooling_allowed=False, cohort_order=["c10", "stage_d"],
        figures=[
            {"figure_id": "A", "status": "GENERATED" if valid else "NOT_GENERATED_DUE_TO_VALIDITY",
             "data": "paper_data/cohort_comparison.csv", "kind": "separate_cohort_median_ci",
             "x": "cohort", "y": "median_delta_fid", "interval": ["bootstrap_ci_lower", "bootstrap_ci_upper"],
             "x_axis": {"label": "Cohort", "scale": "categorical"},
             "y_axis": {"label": "Median delta FID", "scale": "linear"},
             "colors": {"c10": "#7f7f7f", "stage_d": "#1f77b4"},
             "cohorts_must_not_be_pooled": True, "caption_fields": ["cohort", "n", "historical", "overall_decision", "workflow_status", "spatial_status", "holm_status"]},
            {"figure_id": "B", "status": "GENERATED" if valid else "NOT_GENERATED_DUE_TO_VALIDITY",
             "data": "paper_data/per_sample_delta.csv", "kind": "stage_d_delta_fid_ecdf",
             "x": "delta_fid", "group": None, "missing": "explicit_status_no_complete_case",
             "x_axis": {"label": "Per-image delta FID", "scale": "linear"},
             "y_axis": {"label": "Empirical cumulative proportion", "scale": "linear"},
             "colors": {"valid": "#1f77b4", "invalid": "#7f7f7f"},
             "caption_fields": ["status", "validity_status", "misclassified"]},
            {"figure_id": "C", "status": "GENERATED" if valid else "NOT_GENERATED_DUE_TO_VALIDITY",
             "data": "paper_data/curve_summary.csv", "kind": "median_fidelity_curves_pointwise_band",
             "x": "fraction", "y": "median", "facet": "direction", "color": "method_id",
             "method_order": ["nns", "matched"], "direction_order": ["insertion", "deletion"],
             "within_image_reduction": "mean_of_five_tie_repeats",
             "across_image_statistic": "pointwise_median", "band": {
                 "lower": "pointwise_ci_lower", "upper": "pointwise_ci_upper",
                 "namespace": FIGURE_NAMESPACE, "resamples": 10000, "cluster_size": 500,
                 "prng": "numpy.random.Generator(numpy.random.PCG64(seed))",
                 "quantile_method": "linear", "simultaneous": False,
                 "enters_primary_decision": False},
             "x_axis": {"label": "Intervention fraction", "scale": "linear"},
             "y_axis": {"label": "Normalized target response", "scale": "linear"},
             "colors": {
                 "conditional_fixed_clean_attention_head_direction_value_path_ig": "#1f77b4",
                 "matched_fixed_clean_attention_patch_token_contribution": "#ff7f0e"},
             "missing": "explicit_status_no_complete_case",
             "caption_fields": ["method_id", "direction", "n"]},
        ],
    )
    provenance = _self_document(
        "stage_d_paper_provenance", source_analysis_canonical_digest=analysis["canonical_digest"],
        analysis_provenance=analysis["provenance"], numeric_sources=["analysis.json"],
        execution_root_read=formal_result, l0_read=formal_result,
        l1_read=formal_result, l2_read=False,
        image_read=False, model_loaded=False, png_rendered=False,
        formal_result=formal_result,
    )
    report = _self_document(
        "stage_d_analysis_report", complete=True,
        synthetic_only=synthetic_only, formal_result=formal_result,
        validity_status=analysis["validity_status"], fidelity_status=analysis["fidelity_status"],
        overall_decision=analysis["overall_decision"], expected_coordinate_count=500,
        observed_coordinate_count=len(per_sample), bootstrap_performed=valid,
        effect_outputs_status="GENERATED" if valid else "NOT_GENERATED_DUE_TO_VALIDITY",
        c10_n=15, stage_d_n=500, pooled_n=None, figures_rendered=0,
    )
    payloads.update({
        "decision.json": _json_file_bytes(decision),
        "figure_recipes.json": _json_file_bytes(recipes),
        "provenance.json": _json_file_bytes(provenance),
        "analysis_report.json": _json_file_bytes(report),
    })
    if set(payloads) != set(PAYLOAD_FILES):
        raise AssertionError("internal Stage D paper payload registry drifted")
    return payloads


def _inventory(payloads: Mapping[str, bytes], names: Sequence[str]) -> list[dict[str, Any]]:
    return [
        {"path": name, "bytes": len(payloads[name]),
         "sha256": hashlib.sha256(payloads[name]).hexdigest()}
        for name in sorted(names)
    ]


def _complete_payloads(analysis: Mapping[str, Any]) -> dict[str, bytes]:
    provenance = analysis["provenance"]
    if not isinstance(provenance, Mapping):
        raise StageDPaperContractError("analysis provenance must be an object")
    formal_result = provenance.get("formal_result") is True
    payloads = _derived_payloads(analysis)
    manifest = _self_document(
        "stage_d_analysis_payload_manifest", immutable=True,
        payload_count=len(PAYLOAD_FILES), artifacts=_inventory(payloads, PAYLOAD_FILES),
    )
    payloads[MANIFEST_FILE] = _json_file_bytes(manifest)
    excluding_status = _inventory(payloads, (*PAYLOAD_FILES, MANIFEST_FILE))
    status = _self_document(
        "stage_d_analysis_package_status", complete=True, terminal=True, immutable=True,
        formal_result=formal_result, file_count_excluding_status=len(excluding_status),
        whole_root_excluding_status=excluding_status,
        whole_root_excluding_status_digest=canonical_sha256({"files": excluding_status}),
        payload_manifest_sha256=hashlib.sha256(payloads[MANIFEST_FILE]).hexdigest(),
    )
    payloads[STATUS_FILE] = _json_file_bytes(status)
    return payloads


def _validate_payload_documents(
    payloads: Mapping[str, bytes], *, repo_root: Path | str | None,
) -> None:
    schema = _load_schema(repo_root)
    for name in ALL_FILES:
        if not name.endswith(".json"):
            continue
        document = _strict_json(payloads[name], name)
        _validate_self_digest(document, name)
        _validate_document_schema(document, schema, name)


def _absolute(value: Path | str) -> Path:
    return Path(os.path.abspath(os.fspath(value)))


def _open_directory_chain(path: Path) -> int:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path.anchor, flags)
    try:
        for part in path.parts[1:]:
            child = os.open(part, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise StageDPaperContractError("package root is not a directory")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _read_file_at(directory_fd: int, name: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    descriptor = os.open(name, flags, dir_fd=directory_fd)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise StageDPaperContractError(f"unsafe package file: {name}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        identity = lambda item: (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns, item.st_ctime_ns, item.st_nlink)
        raw = b"".join(chunks)
        if identity(before) != identity(after) or len(raw) != before.st_size:
            raise StageDPaperContractError(f"package file changed while reading: {name}")
        return raw
    finally:
        os.close(descriptor)


def _safe_read_package(root: Path) -> dict[str, bytes]:
    root_fd = _open_directory_chain(root)
    paper_fd: int | None = None
    try:
        if set(os.listdir(root_fd)) != {
            "analysis.json", "analysis_report.json", "decision.json",
            "figure_recipes.json", "paper_data", "provenance.json",
            MANIFEST_FILE, STATUS_FILE,
        }:
            raise StageDPaperContractError("package root has a non-closed layout")
        paper_fd = os.open(
            "paper_data", os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=root_fd,
        )
        expected_paper = {PurePosixPath(name).name for name in ALL_FILES if name.startswith("paper_data/")}
        if set(os.listdir(paper_fd)) != expected_paper:
            raise StageDPaperContractError("paper_data has a non-closed layout")
        result: dict[str, bytes] = {}
        for relative in ALL_FILES:
            path = PurePosixPath(relative)
            result[relative] = _read_file_at(paper_fd if len(path.parts) == 2 else root_fd, path.name)
        return result
    except OSError as error:
        raise StageDPaperContractError(f"cannot safely read package: {error}") from error
    finally:
        if paper_fd is not None:
            os.close(paper_fd)
        os.close(root_fd)


def _fsync_directory(descriptor: int) -> None:
    os.fsync(descriptor)


def _write_at(directory_fd: int, name: str, raw: bytes) -> None:
    descriptor = os.open(
        name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        0o600, dir_fd=directory_fd,
    )
    try:
        view = memoryview(raw)
        while view:
            count = os.write(descriptor, view)
            if count <= 0:
                raise StageDPaperError(f"write made no progress: {name}")
            view = view[count:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_no_replace(parent_fd: int, staging: str, destination: str, staging_fd: int) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise StageDPaperBlockedError("renameat2(RENAME_NOREPLACE) is unavailable")
    renameat2.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
    renameat2.restype = ctypes.c_int
    retained = os.fstat(staging_fd)
    named = os.stat(staging, dir_fd=parent_fd, follow_symlinks=False)
    if not stat.S_ISDIR(retained.st_mode) or (retained.st_dev, retained.st_ino) != (named.st_dev, named.st_ino):
        raise StageDPaperContractError("staging directory identity changed")
    if renameat2(parent_fd, os.fsencode(staging), parent_fd, os.fsencode(destination), 1) == 0:
        _fsync_directory(parent_fd)
        return
    number = ctypes.get_errno()
    if number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise StageDPaperBlockedError("destination already exists; overwrite is forbidden")
    raise StageDPaperError(f"atomic no-replace publication failed with errno {number}")


def _publish(payloads: Mapping[str, bytes], destination: Path | str) -> Path:
    target = _absolute(destination)
    if target.name in {"", ".", ".."}:
        raise StageDPaperBlockedError("destination name is invalid")
    parent_fd = _open_directory_chain(target.parent)
    staging = f".stage-d-paper-staging-{secrets.token_hex(16)}"
    staging_fd: int | None = None
    paper_fd: int | None = None
    published = False
    try:
        try:
            os.mkdir(staging, 0o700, dir_fd=parent_fd)
        except FileExistsError as error:
            raise StageDPaperBlockedError("staging directory collision") from error
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        staging_fd = os.open(staging, flags, dir_fd=parent_fd)
        os.mkdir("paper_data", 0o700, dir_fd=staging_fd)
        paper_fd = os.open("paper_data", flags, dir_fd=staging_fd)
        for name in PAYLOAD_FILES:
            path = PurePosixPath(name)
            _write_at(paper_fd if len(path.parts) == 2 else staging_fd, path.name, payloads[name])
        _write_at(staging_fd, MANIFEST_FILE, payloads[MANIFEST_FILE])
        _fsync_directory(paper_fd)
        _fsync_directory(staging_fd)
        # Terminal status is always the final staged write.
        _write_at(staging_fd, STATUS_FILE, payloads[STATUS_FILE])
        _fsync_directory(staging_fd)
        _publish_no_replace(parent_fd, staging, target.name, staging_fd)
        published = True
        return target
    finally:
        if paper_fd is not None:
            os.close(paper_fd)
        if staging_fd is not None:
            os.close(staging_fd)
        if not published:
            # Delete only the known private staging tree and exact known names.
            try:
                stage_path = target.parent / staging
                for relative in reversed(ALL_FILES):
                    candidate = stage_path / relative
                    try:
                        if candidate.is_file() and not candidate.is_symlink():
                            candidate.unlink()
                    except OSError:
                        pass
                try:
                    (stage_path / "paper_data").rmdir()
                    stage_path.rmdir()
                except OSError:
                    pass
            finally:
                os.close(parent_fd)
        else:
            os.close(parent_fd)


def build_stage_d_analysis_package(
    result: Any, destination: Path | str, *, repo_root: Path | str | None = None,
) -> dict[str, Any]:
    """Build and atomically publish a fresh deterministic analysis package."""

    if not isinstance(result, StageDAnalysisResult):
        raise StageDPaperContractError(
            "analysis package requires an in-process D6-issued aggregation result"
        )
    analysis = _validate_analysis(result, repo_root=repo_root)
    payloads = _complete_payloads(analysis)
    _validate_payload_documents(payloads, repo_root=repo_root)
    root = _publish(payloads, destination)
    return validate_stage_d_analysis_package(root, repo_root=repo_root)


def validate_stage_d_analysis_package(
    root: Path | str, *, repo_root: Path | str | None = None,
) -> dict[str, Any]:
    """Safely validate and deterministically recompute a sealed package."""

    path = _absolute(root)
    payloads = _safe_read_package(path)
    analysis = _validate_analysis(_strict_json(payloads["analysis.json"], "analysis.json"), repo_root=repo_root)
    expected = _complete_payloads(analysis)
    if payloads != expected:
        drift = sorted(name for name in ALL_FILES if payloads.get(name) != expected.get(name))
        raise StageDPaperContractError(f"package bytes are not reproducible from analysis.json: {drift}")
    _validate_payload_documents(payloads, repo_root=repo_root)
    return {
        "root": str(path), "valid": True, "complete": True,
        "formal_result": analysis["provenance"]["formal_result"],
        "analysis_canonical_digest": analysis["canonical_digest"],
        "file_count": len(ALL_FILES),
        "status_sha256": hashlib.sha256(payloads[STATUS_FILE]).hexdigest(),
        "manifest_sha256": hashlib.sha256(payloads[MANIFEST_FILE]).hexdigest(),
    }


def analysis_control_from_package(
    root: Path | str, *, repo_root: Path | str | None = None,
) -> dict[str, Any]:
    """Read analysis.json only after the complete package has validated."""

    validate_stage_d_analysis_package(root, repo_root=repo_root)
    payloads = _safe_read_package(_absolute(root))
    return _validate_analysis(
        _strict_json(payloads["analysis.json"], "analysis.json"),
        repo_root=repo_root,
    )


def build_stage_d_paper_package(
    analysis_root: Path | str, destination: Path | str, *,
    repo_root: Path | str | None = None,
) -> dict[str, Any]:
    """Rebuild a package using only a validated canonical analysis package."""

    validate_stage_d_analysis_package(analysis_root, repo_root=repo_root)
    source = _safe_read_package(_absolute(analysis_root))
    analysis = _strict_json(source["analysis.json"], "analysis.json")
    payloads = _complete_payloads(_validate_analysis(analysis, repo_root=repo_root))
    _validate_payload_documents(payloads, repo_root=repo_root)
    root = _publish(payloads, destination)
    return validate_stage_d_analysis_package(root, repo_root=repo_root)


validate_stage_d_paper_package = validate_stage_d_analysis_package

__all__ = [
    "ALL_FILES", "ANALYSIS_DOCUMENT_TYPE", "FIGURE_NAMESPACE",
    "PACKAGE_SCHEMA_PATH", "PAYLOAD_FILES", "StageDPaperBlockedError",
    "StageDPaperContractError", "StageDPaperError",
    "build_stage_d_analysis_package", "build_stage_d_paper_package",
    "analysis_control_from_package",
    "validate_stage_d_analysis_package", "validate_stage_d_paper_package",
]
