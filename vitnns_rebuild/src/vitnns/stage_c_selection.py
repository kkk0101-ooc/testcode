"""Fail-closed C8 calibration aggregation and frozen selection output.

The selection boundary consumes only validated L0/control evidence.  It never
opens an image or an L3 display artifact, and it deliberately contains no
bootstrap, multiple-testing, or holdout code.
"""

from __future__ import annotations

import collections
import dataclasses
import hashlib
import json
import math
import os
import stat
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .stage_c_fidelity import (
    CURVE_FRACTIONS, MATCHED_METHOD_ID, NNS_METHOD_ID, CurveAggregate,
    CurveDirection, FidelityCurveResult, FidelityScale, RankingKind,
    aggregate_five_curves, value_path_delta_fidelity,
)
from .stage_c_maps import NAReason, StructuredNA
from .stage_c_perturbations import PERTURBATION_IDS
from .stage_c_plan import (
    AUDIT_SAMPLE_IDS, CALIBRATION_SAMPLE_IDS, CLASS_METHOD_IDS,
    VALUE_METHOD_IDS, ExecutionPlan, UnitKey, _validate_selection_document,
)
from .stage_c_provenance import canonical_sha256, validate_stage_c_provenance

SCHEMA_VERSION = 1
C8_EXECUTION_UNIT_COUNT = 2180
C8_UNITS_PER_SAMPLE = 436
GENERATED_BY = "vitnns.stage_c_selection.build_stage_c_selection"
SELECTION_DOCUMENT_TYPE = "stage_c_selection"
EVIDENCE_DOCUMENT_TYPE = "stage_c_c8_calibration_evidence"
REPORT_DOCUMENT_TYPE = "stage_c_c8_selection_report"
STATUS_PASS = "PASS"
STATUS_BLOCKED = "BLOCKED"

DIRECT_METHOD_ID = VALUE_METHOD_IDS[0]
VALUE_METHOD_ORDER = tuple(VALUE_METHOD_IDS)
METHOD_ORDER = (*VALUE_METHOD_IDS, *CLASS_METHOD_IDS)
DEPTH_BANDS = ((0, 1, 2), (3, 4, 5), (6, 7, 8), (9, 10, 11))
_IG_STEPS = (512, 1024, 2048, 4096, 8192)
_ENDPOINT_LIMIT = 1e-6
_IG_ATOL = 1e-4
_IG_RTOL = 0.01
_TOKEN_ATOL = 1e-6
_TOKEN_RTOL = 1e-5
_SHA256 = frozenset("0123456789abcdef")
_INPUT_HASH_KEYS = {
    "code_commit", "code_tree_clean", "model_sha256",
    "dependency_lock_sha256", "registry_sha256",
    "registry_metadata_sha256", "protocol_sha256",
    "acceptance_sha256", "config_sha256", "config_schema_sha256",
    "calibration_artifact_manifest_sha256",
}
_CONTROL_NAMES = (
    "execution_plan.json", "preflight.json", "failure_table.json",
    "batch_status.json", "c8_launch_authorization.json",
)
_BATCH_STATUS_KEYS = {
    "schema_version", "document_type", "plan_digest",
    "expected_unit_count", "orchestration_complete", "successful",
    "failure_table_sha256", "lifecycle_state",
    "active_claim_count", "validation_errors",
}
_RECORD_KEYS = {
    "plan_index", "unit_key", "state",
    "trusted_by_batch_validator", "hashes", "scalars",
    "provenance", "resources", "runner_telemetry",
    "artifact_inventory", "work_item_counts", "audit",
    "validation_errors",
}
_SOURCE_RECORD_KEYS = _RECORD_KEYS | {
    "attempt_path", "l0_l1_cap_accounting_bytes", "level_bytes",
    "total_regular_file_bytes",
}
_HASH_KEYS = {
    "manifest_sha256", "status_sha256", "selected_status_sha256",
    "provenance_sha256", "result_sha256", "unit_identity_sha256",
}
_CURVE_KEYS = {
    "status", "method_id", "comparison_line", "target_ref",
    "target_identity_sha256", "scale", "direction",
    "ranking_kind", "sample_id", "block", "head", "replicate",
    "seed", "tie_priority_sha256", "source_unit_map_sha256",
    "score_sha256", "input_pair_sha256", "ranking_sha256",
    "intervention_sha256", "curve_fractions", "unit_counts",
    "actual_fractions", "responses", "auc", "response_clipped",
    "auc_integration",
}
_AGGREGATE_KEYS = {
    "status", "replicate_role", "replicate_count",
    "standard_deviation", "curves", "response_mean",
    "response_std", "auc_mean", "auc_std",
}


class StageCSelectionError(ValueError):
    """Base error for malformed C8 evidence or selection output."""


class StageCSelectionIntegrityError(StageCSelectionError):
    """Raised when evidence, a report, or a hash binding is inconsistent."""


class StageCSelectionBlockedError(StageCSelectionError):
    """Raised when a caller tries to publish a BLOCKED selection."""

    def __init__(self, message: str, *, report: Mapping[str, Any] | None = None):
        super().__init__(message)
        self.report = None if report is None else _plain(report)


def _plain(value: Any, context: str = "value") -> Any:
    method = getattr(value, "to_dict", None)
    if callable(method):
        value = method()
    if value is None or type(value) in {str, bool, int}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise StageCSelectionIntegrityError(
                f"{context} contains a non-finite number"
            )
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise StageCSelectionIntegrityError(
                    f"{context} keys must be non-empty strings"
                )
            result[key] = _plain(item, f"{context}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [_plain(item, f"{context}[{index}]") for index, item in enumerate(value)]
    raise StageCSelectionIntegrityError(
        f"{context} is not JSON-compatible: {type(value).__name__}"
    )


def _selection_generator_code(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {"commit", "dirty"}:
        raise StageCSelectionIntegrityError(
            "selection generator Git state must contain exactly commit and dirty"
        )
    commit = value.get("commit")
    if (
        not isinstance(commit, str) or len(commit) != 40
        or set(commit) - _SHA256 or value.get("dirty") is not False
    ):
        raise StageCSelectionIntegrityError(
            "selection generator Git state must be a clean 40-hex commit"
        )
    return {"commit": commit, "dirty": False}


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        _plain(value), sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")


def _digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _exact(value: Any, keys: set[str], context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise StageCSelectionIntegrityError(f"{context} must be an object")
    if set(value) != keys:
        missing = sorted(keys - set(value))
        unknown = sorted(set(value) - keys)
        raise StageCSelectionIntegrityError(
            f"{context} fields differ (missing={missing}, unknown={unknown})"
        )
    return value


def _sha(value: Any, context: str) -> str:
    if (
        not isinstance(value, str) or len(value) != 64
        or set(value) - _SHA256
    ):
        raise StageCSelectionIntegrityError(
            f"{context} must be a lowercase SHA-256"
        )
    return value


def _finite(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StageCSelectionIntegrityError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise StageCSelectionIntegrityError(f"{context} must be finite")
    return result


def _strict_json(path: Path) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise StageCSelectionIntegrityError(
                    f"duplicate JSON key in {path}: {key}"
                )
            result[key] = value
        return result

    if path.is_symlink() or not path.is_file():
        raise StageCSelectionIntegrityError(
            f"missing, non-regular, or symlink JSON input: {path}"
        )
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON number {token}")
            ),
        )
    except StageCSelectionIntegrityError:
        raise
    except Exception as error:
        raise StageCSelectionIntegrityError(f"invalid JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise StageCSelectionIntegrityError(f"JSON document is not an object: {path}")
    return value


def _self_digest(document: Mapping[str, Any]) -> str:
    return _digest({key: value for key, value in document.items() if key != "canonical_digest"})


def _records_digest(records: Sequence[Mapping[str, Any]]) -> str:
    return _digest({"unit_evidence_records": list(records)})


def _budget_record(record: Mapping[str, Any]) -> dict[str, Any]:
    inventory = record["artifact_inventory"]
    levels = record.get("level_bytes")
    if not isinstance(levels, Mapping):
        levels = {level: 0 for level in ("L0", "L1", "L2", "L3", "control")}
        for item in inventory:
            if isinstance(item, Mapping) and item.get("level") in levels:
                levels[item["level"]] += int(item["bytes"])
    else:
        levels = dict(levels)
    # Manifest/status byte counts are bound and validated by the source API,
    # but not represented as artifact inventory entries.  Source-built
    # evidence records carry their measured totals explicitly.
    non_l2 = record.get("l0_l1_cap_accounting_bytes")
    total = record.get("total_regular_file_bytes")
    if type(non_l2) is not int:
        non_l2 = levels["L0"] + levels["L1"] + levels["L3"]
    if type(total) is not int:
        total = non_l2 + levels["L2"]
    return {
        "unit_key": record["unit_key"], "state": record["state"],
        "trusted_by_batch_validator": record["trusted_by_batch_validator"],
        "attempt_path": record.get("attempt_path"),
        "resources": record["resources"],
        "runner_telemetry": record["runner_telemetry"],
        "work_item_counts": record["work_item_counts"],
        "l0_l1_cap_accounting_bytes": non_l2, "level_bytes": levels,
        "total_regular_file_bytes": total,
    }


def _post_c8_budget(evidence: Mapping[str, Any]) -> dict[str, Any]:
    from .stage_c_benchmark import FROZEN_CAPS, project_benchmark_budget

    binding = evidence.get("c7_report_binding")
    if not isinstance(binding, Mapping):
        return {
            "status": STATUS_BLOCKED,
            "reason": "missing_frozen_c6_storage_model_binding",
            "exact_unit_count": 2500, "projection_safety_factor": 1.25,
            "projection": None,
        }
    storage_model = binding.get("c6_storage_model")
    if not isinstance(storage_model, Mapping):
        estimate = binding.get("recomputed_estimate")
        storage_model = estimate.get("storage") if isinstance(estimate, Mapping) else None
    if not isinstance(storage_model, Mapping):
        return {
            "status": STATUS_BLOCKED,
            "reason": "missing_frozen_c6_storage_model_binding",
            "exact_unit_count": 2500, "projection_safety_factor": 1.25,
            "projection": None,
        }
    payload = storage_model.get("payload_bytes_before_control_overhead")
    source_bytes = evidence["bindings"]["source_tree_total_bytes"]
    control_hashes = evidence["bindings"]["control_file_sha256"]
    # The control-byte operand is retained explicitly by source-built
    # evidence.  Injectable evidence must provide it in the C7 binding.
    control_bytes = binding.get("source_batch_control_bytes")
    if type(payload) is not int or payload < 0 or type(control_bytes) is not int or control_bytes < 0:
        return {
            "status": STATUS_BLOCKED, "reason": "invalid_storage_model_operands",
            "exact_unit_count": 2500, "projection_safety_factor": 1.25,
            "projection": None, "control_hashes": control_hashes,
        }
    usable = [
        record for record in evidence["records"]
        if record.get("state") == "succeeded"
        and record.get("trusted_by_batch_validator") is True
        and record.get("validation_errors") == []
    ]
    if not usable:
        return {
            "status": STATUS_BLOCKED, "reason": "no_valid_c8_budget_units",
            "exact_unit_count": 2500, "projection_safety_factor": 1.25,
            "projection": None,
        }
    try:
        projection = project_benchmark_budget(
            [_budget_record(record) for record in usable],
            caps=FROZEN_CAPS, c6_storage_payload_bytes=payload,
            source_batch_regular_file_bytes=source_bytes,
            source_unit_count=len(usable),
            c6_storage_model=storage_model,
            source_batch_control_bytes=control_bytes,
        )
    except Exception as error:
        return {
            "status": STATUS_BLOCKED, "reason": f"budget_recomputation_failed:{error}",
            "exact_unit_count": 2500, "projection_safety_factor": 1.25,
            "projection": None,
        }
    return {
        "status": STATUS_PASS if projection.get("all_hard_caps_pass") is True else STATUS_BLOCKED,
        "reason": None if projection.get("all_hard_caps_pass") is True else "post_c8_frozen_cap_failure",
        "exact_unit_count": 2500, "projection_safety_factor": 1.25,
        "projection": projection,
    }


def _candidate_key(block: int, head: int) -> str:
    return f"block_{block:02d}_head_{head:02d}"


def _candidate_ref(block: int, head: int) -> dict[str, Any]:
    return {
        "candidate_key": _candidate_key(block, head),
        "block_index": block, "head_index": head,
        "depth_band": block // 3,
    }


def _expected_unit_order() -> tuple[tuple[str, str, int | None, int | None], ...]:
    result: list[tuple[str, str, int | None, int | None]] = []
    for sample_id in CALIBRATION_SAMPLE_IDS:
        for method_id in METHOD_ORDER:
            if method_id in VALUE_METHOD_IDS:
                result.extend(
                    (sample_id, method_id, block, head)
                    for block in range(12) for head in range(12)
                )
            else:
                result.append((sample_id, method_id, None, None))
    return tuple(result)


_EXPECTED_UNIT_ORDER = _expected_unit_order()


def _validate_exact_plan(plan: Any, config: Any) -> dict[str, Any]:
    if not isinstance(plan, ExecutionPlan):
        raise TypeError("plan must be an ExecutionPlan")
    document = ExecutionPlan.from_dict(plan.to_dict()).to_dict()
    spec = document["run_spec"]
    expected_spec = {
        "schema_version": 1, "document_type": "stage_c_run_spec",
        "sample_ids": list(CALIBRATION_SAMPLE_IDS),
        "method_ids": list(METHOD_ORDER),
        "value_config_source": "calibration_all_144",
        "value_candidates": [],
        "fidelity_pairing_context": {"block_index": 0, "head_index": 0},
        "audit_repeatability_runs": 2,
        "audit_repeatability_scope": "clean_map_only",
        "run_registered_fidelity": True, "run_sanity": True,
        "run_stability": True,
    }
    if spec != expected_spec:
        raise StageCSelectionIntegrityError("C8 RunSpec is not the frozen exact document")
    selection = document.get("selection", {})
    if (
        selection.get("status") != "not_applicable_pre_c8"
        or selection.get("sha256") is not None
        or selection.get("primary") is not None
        or selection.get("depth_band_winners") != []
    ):
        raise StageCSelectionIntegrityError("C8 evidence must not consume a selection")
    units = document["units"]
    observed = tuple(
        (item["key"]["sample_id"], item["key"]["method_id"],
         item["key"]["block_index"], item["key"]["head_index"])
        for item in units
    )
    if observed != _EXPECTED_UNIT_ORDER or len(units) != C8_EXECUTION_UNIT_COUNT:
        raise StageCSelectionIntegrityError(
            "C8 execution plan does not have the exact 2,180-unit order"
        )
    if any(item["sample"].get("role") != "calibration" for item in units):
        raise StageCSelectionIntegrityError("C8 plan contains a holdout unit")
    methods = [item.id for item in getattr(config, "methods", ())]
    if methods != list(METHOD_ORDER):
        raise StageCSelectionIntegrityError("config method order differs from C8")
    return document


def _validate_input_hashes(
    value: Any, calibration_manifest_sha256: str, code: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    hashes = dict(_exact(value, _INPUT_HASH_KEYS, "input_hashes"))
    if (
        not isinstance(hashes["code_commit"], str)
        or len(hashes["code_commit"]) != 40
        or set(hashes["code_commit"]) - _SHA256
    ):
        raise StageCSelectionIntegrityError("input_hashes.code_commit is invalid")
    if hashes["code_tree_clean"] is not True:
        raise StageCSelectionIntegrityError("selection requires a clean code tree")
    for name in _INPUT_HASH_KEYS - {"code_commit", "code_tree_clean"}:
        _sha(hashes[name], f"input_hashes.{name}")
    if hashes["calibration_artifact_manifest_sha256"] != calibration_manifest_sha256:
        raise StageCSelectionIntegrityError(
            "input calibration artifact manifest does not match all 2,180 records"
        )
    if dict(code) != {"commit": hashes["code_commit"], "dirty": False}:
        raise StageCSelectionIntegrityError("selection code binding disagrees with evidence")
    compared = 0
    for index, record in enumerate(records):
        provenance = record.get("provenance")
        bindings = provenance.get("bindings") if isinstance(provenance, Mapping) else None
        if not isinstance(bindings, Mapping):
            if record.get("state") == "succeeded":
                raise StageCSelectionIntegrityError(
                    f"successful record {index} has no provenance bindings"
                )
            continue
        try:
            observed = {
                "code_commit": bindings["code"]["commit"],
                "code_tree_clean": not bindings["code"]["dirty"],
                "model_sha256": bindings["model"]["state_dict_sha256"],
                "dependency_lock_sha256": bindings["dependencies"]["lock_sha256"],
                "registry_sha256": bindings["registry"]["sha256"],
                "registry_metadata_sha256": bindings["registry"]["metadata_sha256"],
                "protocol_sha256": bindings["protocol"]["sha256"],
                "acceptance_sha256": bindings["acceptance"]["sha256"],
                "config_sha256": bindings["configuration"]["sha256"],
                "config_schema_sha256": bindings["configuration"]["schema_sha256"],
            }
        except (KeyError, TypeError) as error:
            raise StageCSelectionIntegrityError(
                f"record {index} provenance lacks a selection input binding"
            ) from error
        for name, observed_value in observed.items():
            if observed_value != hashes[name]:
                raise StageCSelectionIntegrityError(
                    f"record {index} provenance disagrees with input_hashes.{name}"
                )
        compared += 1
    if not compared:
        raise StageCSelectionIntegrityError("no provenance record binds selection inputs")
    return hashes


def _metric_value(value: Any, context: str) -> float | None:
    if not isinstance(value, Mapping):
        raise StageCSelectionIntegrityError(f"{context} must be a metric object")
    status = value.get("status")
    if status == "OK":
        _exact(value, {"status", "value", "details"}, context)
        return _finite(value["value"], f"{context}.value")
    if status == "NA":
        allowed = {"status", "value", "reason"}
        if "detail" in value:
            allowed.add("detail")
        _exact(value, allowed, context)
        if value["value"] is not None:
            raise StageCSelectionIntegrityError(f"{context} NA value must be null")
        try:
            NAReason(value["reason"])
        except (TypeError, ValueError) as error:
            raise StageCSelectionIntegrityError(f"{context} NA reason is invalid") from error
        return None
    raise StageCSelectionIntegrityError(f"{context} status must be OK or NA")


def _structured_na(value: Any, context: str) -> StructuredNA:
    _metric_value(value, context)
    return StructuredNA(NAReason(value["reason"]), value.get("detail"))


def _curve_from_dict(value: Any, context: str) -> FidelityCurveResult:
    if not isinstance(value, Mapping):
        raise StageCSelectionIntegrityError(f"{context} must be an object")
    keys = set(_CURVE_KEYS)
    if value.get("status") == "NA":
        keys.add("na")
    curve = _exact(value, keys, context)
    if curve["curve_fractions"] != list(CURVE_FRACTIONS):
        raise StageCSelectionIntegrityError(f"{context} curve fractions drifted")
    if curve["response_clipped"] is not False or curve["auc_integration"] != (
        "trapezoid_over_actual_k_divided_by_units"
    ):
        raise StageCSelectionIntegrityError(f"{context} curve contract drifted")
    na = None if curve["status"] == "OK" else _structured_na(curve["na"], f"{context}.na")
    try:
        result = FidelityCurveResult(
            method_id=curve["method_id"],
            comparison_line=curve["comparison_line"], target_ref=curve["target_ref"],
            target_identity_sha256=curve["target_identity_sha256"],
            scale=curve["scale"], direction=curve["direction"],
            ranking_kind=curve["ranking_kind"], sample_id=curve["sample_id"],
            block=curve["block"], head=curve["head"],
            replicate=curve["replicate"], seed=curve["seed"],
            tie_priority_sha256=curve["tie_priority_sha256"],
            source_unit_map_sha256=curve["source_unit_map_sha256"],
            score_sha256=curve["score_sha256"],
            input_pair_sha256=curve["input_pair_sha256"],
            ranking_sha256=curve["ranking_sha256"],
            intervention_sha256=tuple(curve["intervention_sha256"]),
            unit_counts=tuple(curve["unit_counts"]),
            actual_fractions=tuple(curve["actual_fractions"]),
            responses=None if curve["responses"] is None else tuple(curve["responses"]),
            auc=curve["auc"], na=na,
        )
    except Exception as error:
        raise StageCSelectionIntegrityError(f"invalid {context}: {error}") from error
    if result.status != curve["status"]:
        raise StageCSelectionIntegrityError(f"{context} status is inconsistent")
    return result


def _aggregate_from_dict(value: Any, context: str) -> CurveAggregate:
    if not isinstance(value, Mapping):
        raise StageCSelectionIntegrityError(f"{context} must be an object")
    keys = set(_AGGREGATE_KEYS)
    if value.get("status") == "NA":
        keys.add("na")
    item = _exact(value, keys, context)
    if (
        item["replicate_role"]
        != "paired_nuisance_resolution_not_independent_samples"
        or item["replicate_count"] != 5
        or item["standard_deviation"]
        != "population_ddof_0_over_complete_five_replicates"
        or not isinstance(item["curves"], list)
    ):
        raise StageCSelectionIntegrityError(f"{context} aggregate contract drifted")
    curves = tuple(
        _curve_from_dict(curve, f"{context}.curves[{index}]")
        for index, curve in enumerate(item["curves"])
    )
    na = None if item["status"] == "OK" else _structured_na(item["na"], f"{context}.na")
    try:
        result = CurveAggregate(
            curves=curves,
            response_mean=None if item["response_mean"] is None else tuple(item["response_mean"]),
            response_std_population=None if item["response_std"] is None else tuple(item["response_std"]),
            auc_mean=item["auc_mean"], auc_std_population=item["auc_std"], na=na,
        )
    except Exception as error:
        raise StageCSelectionIntegrityError(f"invalid {context}: {error}") from error
    if result.status != item["status"]:
        raise StageCSelectionIntegrityError(f"{context} status is inconsistent")
    return result


def _scientific_target_identity(identity: Any) -> dict[str, Any]:
    """Return the cross-unit target identity, excluding process-local state.

    ``runtime_model_binding`` deliberately contains live Python/tensor/storage
    identities.  It is a within-unit mutation guard, not a reproducible part of
    the scientific target shared by independently executed methods.
    """

    if not isinstance(identity, Mapping):
        raise StageCSelectionIntegrityError("target identity is missing")
    normalized = _plain(identity, "target identity")
    frozen = normalized.get("frozen_target")
    if not isinstance(frozen, dict):
        raise StageCSelectionIntegrityError("frozen target identity is missing")
    runtime = frozen.get("runtime_model_binding")
    # Pre-C8 synthetic fixtures predate the serialized runtime guard.  Real
    # C8 records always carry it; if present it must have its exact contract.
    if runtime is None:
        return normalized
    if not isinstance(runtime, dict) or set(runtime) != {
        "kind", "sha256", "is_canonical_model_artifact_hash",
    }:
        raise StageCSelectionIntegrityError(
            "frozen target runtime model binding is invalid"
        )
    if (
        runtime.get("kind")
        != "same_live_model_instance_and_tensor_state_versions"
        or runtime.get("is_canonical_model_artifact_hash") is not False
    ):
        raise StageCSelectionIntegrityError(
            "frozen target runtime model binding is invalid"
        )
    _sha(runtime.get("sha256"), "frozen target runtime model binding")
    del frozen["runtime_model_binding"]
    return normalized


def _validated_aggregates(
    record: Mapping[str, Any], *, scientific_target_identity_sha256: str,
) -> dict[tuple[str, str], CurveAggregate]:
    scalars = record["scalars"]
    raw_curves = scalars.get("fidelity_curves") if isinstance(scalars, Mapping) else None
    raw_aggregates = scalars.get("fidelity_aggregates") if isinstance(scalars, Mapping) else None
    if not isinstance(raw_curves, list) or len(raw_curves) != 44:
        raise StageCSelectionIntegrityError("fidelity unit requires exactly 44 complete curves")
    curves = [
        _curve_from_dict(item, f"fidelity_curves[{index}]")
        for index, item in enumerate(raw_curves)
    ]
    target_identity = scalars.get("target_identity")
    if not isinstance(target_identity, Mapping):
        raise StageCSelectionIntegrityError("fidelity target identity is missing")
    full_target_identity_sha256 = canonical_sha256(target_identity)
    has_runtime_binding = isinstance(
        target_identity.get("frozen_target"), Mapping
    ) and "runtime_model_binding" in target_identity["frozen_target"]
    if has_runtime_binding and any(
        curve.target_identity_sha256 != full_target_identity_sha256 for curve in curves
    ):
        raise StageCSelectionIntegrityError(
            "fidelity curve does not bind its unit's complete target identity"
        )
    _sha(scientific_target_identity_sha256, "scientific target identity")
    if not has_runtime_binding:
        scientific_target_identity_sha256 = curves[0].target_identity_sha256
    expected_groups: set[tuple[str, str, str, int | None]] = set()
    for scale in (FidelityScale.PATCH_16, FidelityScale.FINE_4):
        for direction in (CurveDirection.DELETION, CurveDirection.INSERTION):
            expected_groups.update(
                (scale.value, direction.value, RankingKind.TARGET_ORIENTED_POSITIVE.value, replicate)
                for replicate in range(5)
            )
            expected_groups.update(
                (scale.value, direction.value, RankingKind.GLOBAL_RANDOM.value, replicate)
                for replicate in range(5)
            )
            expected_groups.add(
                (scale.value, direction.value, RankingKind.CENTER_DISTANCE_ROW_MAJOR.value, None)
            )
    groups = [(c.scale.value, c.direction.value, c.ranking_kind.value, c.replicate) for c in curves]
    if len(set(groups)) != 44 or set(groups) != expected_groups:
        raise StageCSelectionIntegrityError("fidelity curves do not cover the frozen 44 groups")
    key = record["unit_key"]
    if any(
        c.method_id != key["method_id"] or c.sample_id != key["sample_id"]
        or c.block != key["block_index"] or c.head != key["head_index"]
        or c.na is not None
        for c in curves
    ):
        raise StageCSelectionIntegrityError("fidelity curves cross unit identity or contain NA")
    if not isinstance(raw_aggregates, list) or len(raw_aggregates) != 4:
        raise StageCSelectionIntegrityError("fidelity unit requires four target aggregates")
    result: dict[tuple[str, str], CurveAggregate] = {}
    serialized_curves = collections.Counter(_canonical_bytes(item) for item in raw_curves)
    for index, raw in enumerate(raw_aggregates):
        aggregate = _aggregate_from_dict(raw, f"fidelity_aggregates[{index}]")
        first = aggregate.curves[0]
        pair = (first.scale.value, first.direction.value)
        if (
            pair in result
            or first.ranking_kind is not RankingKind.TARGET_ORIENTED_POSITIVE
            or first.method_id != key["method_id"]
        ):
            raise StageCSelectionIntegrityError("fidelity aggregate identity is invalid")
        for curve in raw["curves"]:
            encoded = _canonical_bytes(curve)
            if serialized_curves[encoded] != 1:
                raise StageCSelectionIntegrityError(
                    "aggregate curve is absent or duplicated in fidelity_curves"
                )
        result[pair] = dataclasses.replace(
            aggregate,
            curves=tuple(
                dataclasses.replace(
                    curve,
                    target_identity_sha256=scientific_target_identity_sha256,
                )
                for curve in aggregate.curves
            ),
        )
    expected_pairs = {
        (scale.value, direction.value)
        for scale in (FidelityScale.PATCH_16, FidelityScale.FINE_4)
        for direction in (CurveDirection.DELETION, CurveDirection.INSERTION)
    }
    if set(result) != expected_pairs:
        raise StageCSelectionIntegrityError("fidelity aggregates do not cover both scales/directions")
    return result


def _validate_map_coverage(record: Mapping[str, Any], *, ig: bool) -> None:
    maps = record["scalars"].get("map_execution_records")
    if not isinstance(maps, list):
        raise StageCSelectionIntegrityError("map execution records are missing")
    expected = [("clean", 0), *((item, None) for item in PERTURBATION_IDS)]
    if record["audit"]:
        expected.append(("repeat_clean", 1))
    observed = [
        (item.get("input_variant"), item.get("repeat_index"))
        for item in maps if isinstance(item, Mapping)
    ]
    if observed != expected:
        raise StageCSelectionIntegrityError("required clean/perturbation map coverage drifted")
    required_map_keys = {
        "method_id", "input_variant", "repeat_index", "status",
        "ig_result", "attempted_steps_total", "backward_invocations",
        "actual_calls", "exception",
    }
    for index, item in enumerate(maps):
        if not isinstance(item, Mapping) or set(item) != required_map_keys or item.get("status") != "completed":
            raise StageCSelectionIntegrityError(f"map {index} did not complete")
        if item["method_id"] != record["unit_key"]["method_id"] or item["exception"] is not None:
            raise StageCSelectionIntegrityError(f"map {index} identity/status drifted")
        calls = _exact(
            item["actual_calls"],
            {"accounting_unit", "forward_calls", "backward_calls"},
            f"map {index}.actual_calls",
        )
        if calls["accounting_unit"] != "batched_model_invocation":
            raise StageCSelectionIntegrityError(f"map {index} call accounting drifted")
        if any(type(calls[name]) is not int or calls[name] < 0 for name in ("forward_calls", "backward_calls")):
            raise StageCSelectionIntegrityError(f"map {index} call counts are invalid")
        if item["backward_invocations"] != calls["backward_calls"]:
            raise StageCSelectionIntegrityError(f"map {index} backward counts disagree")
        summary = item.get("ig_result")
        if not ig:
            if summary is not None or item["attempted_steps_total"] != 0 or item["backward_invocations"] != 0:
                raise StageCSelectionIntegrityError("non-IG method claims IG evidence")
            continue
        if not isinstance(summary, Mapping) or summary.get("converged") is not True:
            raise StageCSelectionIntegrityError(f"map {index} IG did not converge")
        attempts = summary.get("attempts")
        if not isinstance(attempts, list) or not attempts:
            raise StageCSelectionIntegrityError(f"map {index} IG attempts are missing")
        steps: list[int] = []
        passes: list[bool] = []
        for attempt_index, attempt in enumerate(attempts):
            item_context = f"map {index} IG attempt {attempt_index}"
            attempt = _exact(
                attempt, {"steps", "delta", "sum", "abs", "rel", "pass"},
                item_context,
            )
            if type(attempt["steps"]) is not int:
                raise StageCSelectionIntegrityError(f"{item_context} steps are invalid")
            steps.append(attempt["steps"])
            vectors = {}
            for field in ("delta", "sum", "abs", "rel"):
                raw = attempt[field]
                if not isinstance(raw, list) or len(raw) != 1:
                    raise StageCSelectionIntegrityError(f"{item_context}.{field} is invalid")
                vectors[field] = _finite(raw[0], f"{item_context}.{field}[0]")
            raw_pass = attempt["pass"]
            if not isinstance(raw_pass, list) or len(raw_pass) != 1 or type(raw_pass[0]) is not bool:
                raise StageCSelectionIntegrityError(f"{item_context}.pass is invalid")
            expected_abs = abs(vectors["delta"] - vectors["sum"])
            expected_rel = expected_abs / max(abs(vectors["delta"]), 1.1920928955078125e-7)
            if not math.isclose(vectors["abs"], expected_abs, rel_tol=1e-6, abs_tol=1e-12):
                raise StageCSelectionIntegrityError(f"{item_context} conservation abs is inconsistent")
            if not math.isclose(vectors["rel"], expected_rel, rel_tol=1e-6, abs_tol=1e-12):
                raise StageCSelectionIntegrityError(f"{item_context} conservation rel is inconsistent")
            passed = expected_abs <= _IG_ATOL + _IG_RTOL * abs(vectors["delta"])
            if raw_pass[0] is not passed:
                raise StageCSelectionIntegrityError(f"{item_context} pass is inconsistent")
            passes.append(passed)
        if steps != list(_IG_STEPS[:len(steps)]) or summary.get("selected_steps") != steps[-1]:
            raise StageCSelectionIntegrityError(f"map {index} IG schedule drifted")
        if passes != [False] * (len(passes) - 1) + [True]:
            raise StageCSelectionIntegrityError(f"map {index} IG stop rule drifted")
        metadata = summary.get("metadata")
        if not isinstance(metadata, Mapping) or metadata.get("step_chunk_size") != 8:
            raise StageCSelectionIntegrityError(f"map {index} IG metadata drifted")
        expected_attempted = sum(steps)
        expected_backward = sum(math.ceil(step / 8) for step in steps)
        if item["attempted_steps_total"] != expected_attempted:
            raise StageCSelectionIntegrityError(f"map {index} attempted steps disagree")
        if item["backward_invocations"] != expected_backward:
            raise StageCSelectionIntegrityError(f"map {index} IG backward count disagrees")
        if calls["forward_calls"] != expected_backward + 4:
            raise StageCSelectionIntegrityError(f"map {index} IG forward count disagrees")
    if record["unit_key"]["method_id"] in {DIRECT_METHOD_ID, MATCHED_METHOD_ID}:
        expected_perturbed = 1 if record["unit_key"]["method_id"] == DIRECT_METHOD_ID else 2
        for index, item in enumerate(maps):
            expected_forward = (
                0 if item["input_variant"] in {"clean", "repeat_clean"}
                else expected_perturbed
            )
            if item["actual_calls"]["forward_calls"] != expected_forward:
                raise StageCSelectionIntegrityError(f"map {index} non-IG forward count disagrees")
    counts = record["work_item_counts"]
    expected_counts = {
        "map_runs": len(expected), "clean_map_runs": 1,
        "perturbation_map_runs": 3,
        "repeatability_map_runs": 1 if record["audit"] else 0,
    }
    if any(counts.get(name) != value for name, value in expected_counts.items()):
        raise StageCSelectionIntegrityError("map execution counts disagree with plan")
    actual_forward = sum(item["actual_calls"]["forward_calls"] for item in maps)
    actual_backward = sum(item["actual_calls"]["backward_calls"] for item in maps)
    resources = record["resources"].get("calls")
    if not isinstance(resources, Mapping):
        raise StageCSelectionIntegrityError("unit resource call counts are missing")
    if resources.get("forward_calls", -1) < actual_forward or resources.get("backward_calls") != actual_backward:
        raise StageCSelectionIntegrityError("map and unit resource call counts disagree")


def _validate_direct_fidelity_absence(record: Mapping[str, Any]) -> None:
    scalars = record["scalars"]
    if (
        scalars.get("target_ref") is not None
        or scalars.get("target_identity") is not None
        or scalars.get("fidelity_enabled") is not False
        or scalars.get("fidelity_curves") != []
        or scalars.get("fidelity_aggregates", []) != []
        or scalars.get("top20", []) != []
    ):
        raise StageCSelectionIntegrityError(
            "direct structural Value method fabricated registered fidelity"
        )
    counts = record["work_item_counts"]
    if any(counts.get(name) != 0 for name in (
        "fidelity_curves", "fidelity_curve_points",
        "target_tie_curve_points", "random_sanity_curve_points",
        "center_sanity_curve_points", "logical_scalar_evaluations",
    )):
        raise StageCSelectionIntegrityError("direct method plan contains fidelity work")
    if counts.get("work_items") != counts.get("map_runs"):
        raise StageCSelectionIntegrityError("direct method work-item count drifted")


def _validate_targeted_fidelity_shape(record: Mapping[str, Any]) -> None:
    scalars = record["scalars"]
    if scalars.get("fidelity_enabled") is not True:
        raise StageCSelectionIntegrityError("targeted Value method disabled fidelity")
    if not isinstance(scalars.get("top20"), list) or len(scalars["top20"]) != 10:
        raise StageCSelectionIntegrityError("targeted Value method requires ten top20 rows")
    counts = record["work_item_counts"]
    expected = {
        "fidelity_curves": 44, "fidelity_curve_points": 924,
        "target_tie_curve_points": 420, "random_sanity_curve_points": 420,
        "center_sanity_curve_points": 84, "logical_scalar_evaluations": 924,
    }
    if any(counts.get(name) != value for name, value in expected.items()):
        raise StageCSelectionIntegrityError("targeted fidelity work counts drifted")
    if counts.get("work_items") != counts.get("map_runs") + 924:
        raise StageCSelectionIntegrityError("targeted fidelity total work count drifted")


def _validate_required_l1(record: Mapping[str, Any]) -> None:
    paths = [
        item.get("path") for item in record["artifact_inventory"]
        if isinstance(item, Mapping) and item.get("level") == "L1"
    ]
    expected = {"L1/clean.npy", *(f"L1/{item}.npy" for item in PERTURBATION_IDS)}
    if set(paths) != expected or len(paths) != 4:
        raise StageCSelectionIntegrityError("unit lacks exact clean plus three L1 maps")


def _target_checks(
    nns: Mapping[str, Any], matched: Mapping[str, Any], block: int, head: int,
) -> str:
    identities = []
    for label, record in (("NNS", nns), ("matched", matched)):
        scalars = record["scalars"]
        identity = scalars.get("target_identity")
        if not isinstance(identity, Mapping):
            raise StageCSelectionIntegrityError(f"{label} target identity is missing")
        frozen = identity.get("frozen_target")
        if (
            identity.get("comparison_line") != "value_path"
            or identity.get("target_ref")
            != "fixed_clean_attention_cls_head_direction"
            or not isinstance(frozen, Mapping)
            or frozen.get("candidate")
            != {"block_index": block, "head_index": head}
            or frozen.get("status") != "OK"
        ):
            raise StageCSelectionIntegrityError(f"{label} target/direction is invalid")
        direction = _finite(frozen.get("direction_norm"), f"{label} direction_norm")
        threshold = _finite(frozen.get("direction_threshold"), f"{label} direction_threshold")
        unit = frozen.get("direction_unit")
        if direction <= threshold or not isinstance(unit, Mapping) or unit.get("finite") is not True:
            raise StageCSelectionIntegrityError(f"{label} direction is degenerate")
        identities.append(identity)
    scientific_identities = tuple(
        _scientific_target_identity(identity) for identity in identities
    )
    if _canonical_bytes(scientific_identities[0]) != _canonical_bytes(
        scientific_identities[1]
    ):
        raise StageCSelectionIntegrityError("NNS and matched target identities differ")
    fragment = nns["scalars"].get("provenance_fragment")
    parity = fragment.get("endpoint_parity") if isinstance(fragment, Mapping) else None
    fields = {
        "threshold", "comparison", "clean_formula_max_abs_error",
        "clean_natural_vs_fixed_replay_max_abs_error",
        "clean_natural_vs_fixed_replay_model_output_max_abs_error",
        "baseline_fixed_replay_vs_clean_A_matmul_natural_V0_max_abs_error",
    }
    parity = _exact(parity, fields, "NNS endpoint parity")
    if parity["threshold"] != _ENDPOINT_LIMIT or parity["comparison"] != "strict_less_than":
        raise StageCSelectionIntegrityError("endpoint parity policy drifted")
    for field in fields - {"threshold", "comparison"}:
        if _finite(parity[field], f"endpoint parity.{field}") >= _ENDPOINT_LIMIT:
            raise StageCSelectionIntegrityError("endpoint parity is not strictly below 1e-6")
    return canonical_sha256(scientific_identities[0])


def _validate_token_parity(record: Mapping[str, Any]) -> None:
    fragment = record["scalars"].get("provenance_fragment")
    contribution = fragment.get("matched_contribution") if isinstance(fragment, Mapping) else None
    if not isinstance(contribution, Mapping) or contribution.get("status") != "OK":
        raise StageCSelectionIntegrityError("matched token contribution is unavailable")
    if contribution.get("token_contribution_count") != 197:
        raise StageCSelectionIntegrityError("matched token contribution count is not 197")
    token_sum = _finite(contribution.get("token_sum"), "matched token_sum")
    reference = _finite(
        contribution.get("scalar_delta_f_z_minus_f_x0"),
        "matched scalar_delta_f_z_minus_f_x0",
    )
    error = _finite(contribution.get("conservation_abs_error"), "matched conservation error")
    tolerance = _finite(contribution.get("conservation_tolerance"), "matched conservation tolerance")
    expected_tolerance = _TOKEN_ATOL + _TOKEN_RTOL * abs(reference)
    if (
        not math.isclose(tolerance, expected_tolerance, rel_tol=1e-7, abs_tol=1e-12)
        or not math.isclose(error, abs(token_sum - reference), rel_tol=1e-7, abs_tol=1e-12)
        or error > tolerance
    ):
        raise StageCSelectionIntegrityError("matched token conservation/parity failed")
    for name in ("token_contributions_sha256", "patch_contributions_sha256"):
        _sha(contribution.get(name), f"matched {name}")


def _residual_values(record: Mapping[str, Any]) -> tuple[dict[str, float | None], float | None, bool]:
    raw = record["scalars"].get("residual_stability")
    if not isinstance(raw, list) or len(raw) != len(PERTURBATION_IDS):
        raise StageCSelectionIntegrityError("NNS residual stability must contain three records")
    spearman: dict[str, float | None] = {}
    rho: float | None = None
    clean_identity: tuple[Any, ...] | None = None
    spatial = False
    for index, perturbation_id in enumerate(PERTURBATION_IDS):
        item = raw[index]
        if not isinstance(item, Mapping) or item.get("perturbation_id") != perturbation_id:
            raise StageCSelectionIntegrityError("residual perturbation order drifted")
        clean = item.get("clean")
        aligned = item.get("aligned_perturbed")
        if not isinstance(clean, Mapping) or not isinstance(aligned, Mapping):
            raise StageCSelectionIntegrityError("residual pair is missing")
        if clean.get("perturbation_id") != perturbation_id or aligned.get("perturbation_id") != perturbation_id:
            raise StageCSelectionIntegrityError("residual pair perturbation mismatch")
        if clean.get("perturbation_tensor_sha256") != aligned.get("perturbation_tensor_sha256"):
            raise StageCSelectionIntegrityError("residual pair tensor hash mismatch")
        identity = (
            clean.get("method_id"), clean.get("method_version"),
            clean.get("comparison_line"), clean.get("target_identity"),
        )
        if (
            identity[0] != NNS_METHOD_ID or identity[2] != "value_path"
            or aligned.get("method_id") != identity[0]
            or aligned.get("method_version") != identity[1]
            or aligned.get("comparison_line") != identity[2]
            or aligned.get("target_identity") != identity[3]
        ):
            raise StageCSelectionIntegrityError("residual scientific identity mismatch")
        if clean_identity is None:
            clean_identity = identity
        elif identity != clean_identity:
            raise StageCSelectionIntegrityError("clean residual identity changed across perturbations")
        spearman[perturbation_id] = _metric_value(
            item.get("signed_spearman"), f"{perturbation_id} residual Spearman"
        )
        clean_rho = _metric_value(clean.get("rho"), f"{perturbation_id} clean rho")
        details = clean.get("details")
        if not isinstance(details, Mapping) or details.get("formula") != "R(M)=M-U(P(M));rho=l1(R)/l1(M)":
            raise StageCSelectionIntegrityError("clean residual formula drifted")
        numerator = _finite(details.get("numerator_l1"), "clean residual numerator")
        denominator = _finite(details.get("denominator_l1"), "clean residual denominator")
        if clean_rho is None:
            if denominator != 0.0:
                raise StageCSelectionIntegrityError("clean residual NA contradicts denominator")
        elif denominator <= 0.0 or not math.isclose(clean_rho, numerator / denominator, rel_tol=1e-12, abs_tol=1e-12):
            raise StageCSelectionIntegrityError("clean residual rho formula is inconsistent")
        spatial = spatial or clean.get("status") == "SPATIAL_FAILURE" or aligned.get("status") == "SPATIAL_FAILURE"
        if perturbation_id == "gaussian_rgb_std_0_01":
            rho = clean_rho
    return spearman, rho, spatial


def _record_validity(record: Mapping[str, Any], context: str) -> list[str]:
    errors: list[str] = []
    key = record["unit_key"]
    if record["state"] != "succeeded":
        errors.append(f"{context}:unit_not_succeeded")
    if record["trusted_by_batch_validator"] is not True:
        errors.append(f"{context}:artifact_or_batch_validation_failed")
    if record["validation_errors"] != []:
        errors.append(f"{context}:validation_errors_present")
    hashes = record["hashes"]
    if not isinstance(hashes, Mapping) or set(hashes) != _HASH_KEYS:
        errors.append(f"{context}:hash_chain_incomplete")
    else:
        try:
            for name in _HASH_KEYS:
                _sha(hashes[name], f"{context}.hashes.{name}")
            if hashes["status_sha256"] != hashes["selected_status_sha256"]:
                raise StageCSelectionIntegrityError("selected status hash mismatch")
        except StageCSelectionIntegrityError:
            errors.append(f"{context}:hash_chain_incomplete")
    telemetry = record["runner_telemetry"]
    lifecycle = telemetry.get("lifecycle") if isinstance(telemetry, Mapping) else None
    integrity = lifecycle.get("integrity") if isinstance(lifecycle, Mapping) else None
    restoration = lifecycle.get("restoration") if isinstance(lifecycle, Mapping) else None
    if (
        not isinstance(integrity, Mapping)
        or integrity.get("matched") is not True
        or integrity.get("restored") is not True
        or integrity.get("entry_sha256") != integrity.get("exit_sha256")
        or not isinstance(restoration, Mapping)
        or restoration.get("restored") is not True
        or restoration.get("changes_detected") is not False
    ):
        errors.append(f"{context}:lifecycle_invalid")
    inventory = record["artifact_inventory"]
    if not isinstance(inventory, list) or not inventory:
        errors.append(f"{context}:artifact_inventory_invalid")
    elif any(
        not isinstance(item, Mapping) or item.get("level") not in {"L0", "L1", "L2", "L3"}
        or not isinstance(item.get("path"), str)
        or (item.get("level") == "L3" and (item.get("display_only") is not True or item.get("metric_input") is not False))
        for item in inventory
    ):
        errors.append(f"{context}:artifact_inventory_invalid")
    # The existing batch/artifact validator applies the exact preflight caps.
    # Selection nevertheless requires actual resource fields to be present.
    resources = record["resources"]
    calls = resources.get("calls") if isinstance(resources, Mapping) else None
    cuda = resources.get("cuda") if isinstance(resources, Mapping) else None
    if (
        not isinstance(calls, Mapping) or type(calls.get("forward_calls")) is not int
        or type(calls.get("backward_calls")) is not int
        or not isinstance(cuda, Mapping)
        or resources.get("status") != "completed"
    ):
        errors.append(f"{context}:resource_evidence_invalid")
    else:
        forward = calls.get("forward_calls")
        backward = calls.get("backward_calls")
        reserved = cuda.get("peak_reserved_bytes")
        if forward < 0 or backward < 0 or forward > 30_000 or backward > 30_000:
            errors.append(f"{context}:resource_hard_cap_exceeded")
        if reserved is not None and (
            type(reserved) is not int or reserved < 0 or reserved > 90 * 1024**3
        ):
            errors.append(f"{context}:resource_hard_cap_exceeded")
        try:
            non_l2 = sum(
                item["bytes"] for item in inventory
                if isinstance(item, Mapping) and item.get("level") != "L2"
            )
        except (KeyError, TypeError):
            errors.append(f"{context}:resource_evidence_invalid")
        else:
            if non_l2 > 64 * 1024**2:
                errors.append(f"{context}:resource_hard_cap_exceeded")
    return list(dict.fromkeys(errors))


def validate_c8_calibration_evidence(evidence: Mapping[str, Any]) -> None:
    """Validate the closed injectable 2,180-record C8 evidence document."""

    keys = {
        "schema_version", "document_type", "source_root",
        "holdout_accessed", "selection_performed", "bootstrap_performed",
        "holm_performed", "decision_performed", "expected_unit_count",
        "batch_complete", "batch_successful", "calibration_sample_ids",
        "method_ids", "bindings", "records",
        "calibration_artifact_manifest_sha256", "c7_report_binding",
        "canonical_digest",
    }
    document = _exact(evidence, keys, "C8 evidence")
    if document["schema_version"] != 1 or document["document_type"] != EVIDENCE_DOCUMENT_TYPE:
        raise StageCSelectionIntegrityError("C8 evidence identity fields are invalid")
    for flag in ("holdout_accessed", "selection_performed", "bootstrap_performed", "holm_performed", "decision_performed"):
        if document[flag] is not False:
            raise StageCSelectionIntegrityError(f"C8 evidence {flag} must be false")
    if document["expected_unit_count"] != C8_EXECUTION_UNIT_COUNT:
        raise StageCSelectionIntegrityError("C8 evidence expected unit count is not 2,180")
    if document["calibration_sample_ids"] != list(CALIBRATION_SAMPLE_IDS):
        raise StageCSelectionIntegrityError("C8 evidence does not use the exact five IDs")
    if document["method_ids"] != list(METHOD_ORDER):
        raise StageCSelectionIntegrityError("C8 evidence method order drifted")
    bindings = _exact(
        document["bindings"],
        {"source_root", "execution_plan_sha256", "run_spec_sha256",
         "preflight_sha256", "control_file_sha256", "code",
         "source_tree_total_bytes"},
        "C8 evidence.bindings",
    )
    if bindings["source_root"] != document["source_root"]:
        raise StageCSelectionIntegrityError("C8 source_root bindings disagree")
    for name in ("execution_plan_sha256", "run_spec_sha256", "preflight_sha256"):
        _sha(bindings[name], f"C8 evidence.bindings.{name}")
    controls = _exact(
        bindings["control_file_sha256"], set(_CONTROL_NAMES),
        "C8 evidence control hashes",
    )
    for name in _CONTROL_NAMES:
        _sha(controls[name], f"C8 evidence control {name}")
    code = _exact(bindings["code"], {"commit", "dirty"}, "C8 evidence code")
    if (
        not isinstance(code["commit"], str) or len(code["commit"]) != 40
        or set(code["commit"]) - _SHA256 or code["dirty"] is not False
    ):
        raise StageCSelectionIntegrityError("C8 evidence must bind a clean code state")
    if (
        type(bindings["source_tree_total_bytes"]) is not int
        or bindings["source_tree_total_bytes"] < 0
        or bindings["source_tree_total_bytes"] > 32 * 1024**3
    ):
        raise StageCSelectionIntegrityError("C8 source tree exceeds its hard byte cap")
    records = document["records"]
    if not isinstance(records, list) or len(records) != C8_EXECUTION_UNIT_COUNT:
        raise StageCSelectionIntegrityError("C8 evidence requires exactly 2,180 records")
    observed_order = []
    successful = True
    complete = True
    l2_by_sample: collections.Counter[str] = collections.Counter()
    total_payload_bytes = 0
    for index, raw in enumerate(records):
        if not isinstance(raw, Mapping):
            raise StageCSelectionIntegrityError(f"C8 evidence.records[{index}] must be an object")
        allowed_record_keys = _SOURCE_RECORD_KEYS if set(raw) == _SOURCE_RECORD_KEYS else _RECORD_KEYS
        record = _exact(raw, allowed_record_keys, f"C8 evidence.records[{index}]")
        if record["plan_index"] != index:
            raise StageCSelectionIntegrityError("C8 evidence plan indices are not exact")
        try:
            key = UnitKey.from_dict(record["unit_key"]).to_dict()
        except Exception as error:
            raise StageCSelectionIntegrityError(f"invalid unit key at index {index}: {error}") from error
        observed_order.append((key["sample_id"], key["method_id"], key["block_index"], key["head_index"]))
        if type(record["trusted_by_batch_validator"]) is not bool or type(record["audit"]) is not bool:
            raise StageCSelectionIntegrityError(f"C8 evidence record {index} booleans are invalid")
        if record["audit"] is not (key["sample_id"] in AUDIT_SAMPLE_IDS):
            raise StageCSelectionIntegrityError(f"C8 evidence record {index} audit scope drifted")
        if not isinstance(record["validation_errors"], list) or any(
            not isinstance(item, str) or not item for item in record["validation_errors"]
        ):
            raise StageCSelectionIntegrityError(f"C8 evidence record {index} errors are invalid")
        for field in ("scalars", "provenance", "resources", "runner_telemetry", "work_item_counts"):
            if not isinstance(record[field], Mapping):
                raise StageCSelectionIntegrityError(f"C8 evidence record {index}.{field} is invalid")
        if not isinstance(record["artifact_inventory"], list):
            raise StageCSelectionIntegrityError(f"C8 evidence record {index} inventory is invalid")
        if record["state"] not in {"succeeded", "failed", "missing", "incomplete", "tampered"}:
            raise StageCSelectionIntegrityError(f"C8 evidence record {index} state is invalid")
        hashes = _exact(record["hashes"], _HASH_KEYS, f"C8 evidence record {index}.hashes")
        if record["state"] == "succeeded":
            for name in _HASH_KEYS:
                _sha(hashes[name], f"C8 evidence record {index}.hashes.{name}")
            if hashes["status_sha256"] != hashes["selected_status_sha256"]:
                raise StageCSelectionIntegrityError(f"C8 evidence record {index} selected status mismatch")
            if record["provenance"].get("unit_identity") != key:
                raise StageCSelectionIntegrityError(f"C8 evidence record {index} provenance identity drifted")
            try:
                validate_stage_c_provenance(record["provenance"], expected_unit=key)
            except Exception as error:
                raise StageCSelectionIntegrityError(
                    f"C8 evidence record {index} provenance is invalid: {error}"
                ) from error
            if record["scalars"].get("sample_id") != key["sample_id"]:
                raise StageCSelectionIntegrityError(f"C8 evidence record {index} scalar sample drifted")
            if record["scalars"].get("method_id") != key["method_id"]:
                raise StageCSelectionIntegrityError(f"C8 evidence record {index} scalar method drifted")
        else:
            for name, digest_value in hashes.items():
                if digest_value is not None:
                    _sha(digest_value, f"C8 evidence record {index}.hashes.{name}")
            if record["provenance"]:
                try:
                    validate_stage_c_provenance(record["provenance"], expected_unit=key)
                except Exception as error:
                    raise StageCSelectionIntegrityError(
                        f"C8 evidence record {index} retained provenance is invalid: {error}"
                    ) from error
        record_errors = _record_validity(record, f"record_{index}")
        successful = successful and not record_errors
        complete = complete and record["state"] in {"succeeded", "failed"}
        for artifact in record["artifact_inventory"]:
            if not isinstance(artifact, Mapping) or type(artifact.get("bytes")) is not int or artifact["bytes"] < 0:
                raise StageCSelectionIntegrityError(f"C8 evidence record {index} artifact bytes are invalid")
            total_payload_bytes += artifact["bytes"]
            if artifact.get("level") == "L2":
                l2_by_sample[key["sample_id"]] += artifact["bytes"]
        timings = record["runner_telemetry"].get("phase_timings")
        wall = timings.get("synchronized_total_wall_time_seconds") if isinstance(timings, Mapping) else None
        if record["state"] == "succeeded":
            _finite(wall, f"C8 evidence record {index} wall time")
    if tuple(observed_order) != _EXPECTED_UNIT_ORDER:
        raise StageCSelectionIntegrityError("C8 evidence records are not in exact plan order")
    if type(document["batch_complete"]) is not bool or type(document["batch_successful"]) is not bool:
        raise StageCSelectionIntegrityError("C8 batch flags must be booleans")
    if document["batch_complete"] is not complete:
        raise StageCSelectionIntegrityError("C8 batch_complete contradicts unit records")
    if document["batch_successful"] is not successful:
        raise StageCSelectionIntegrityError("C8 batch_successful contradicts unit records")
    if document["batch_successful"] and not document["batch_complete"]:
        raise StageCSelectionIntegrityError("successful C8 evidence must be complete")
    if any(value > 1024**3 for value in l2_by_sample.values()):
        raise StageCSelectionIntegrityError("C8 audit-sample L2 hard cap exceeded")
    if total_payload_bytes > 32 * 1024**3:
        raise StageCSelectionIntegrityError("C8 artifact hard cap exceeded")
    manifest = _records_digest(records)
    if document["calibration_artifact_manifest_sha256"] != manifest:
        raise StageCSelectionIntegrityError("C8 global artifact manifest digest mismatch")
    if document["canonical_digest"] != _self_digest(document):
        raise StageCSelectionIntegrityError("C8 evidence canonical digest mismatch")


def _safe_selected_attempt(root: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise StageCSelectionIntegrityError("selected attempt path is missing")
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    resolved = path.resolve()
    try:
        resolved.relative_to((root / "units").resolve())
    except ValueError as error:
        raise StageCSelectionIntegrityError("selected attempt escapes C8 units root") from error
    if resolved.is_symlink() or not resolved.is_dir():
        raise StageCSelectionIntegrityError("selected attempt is unsafe")
    return resolved


def _validate_finalized_batch_status(
    status: Any, failure_table: Any, *, plan_digest: str,
    failure_table_sha256: str,
) -> None:
    status = _exact(status, _BATCH_STATUS_KEYS, "C8 batch_status.json")
    table = _exact(
        failure_table,
        {"schema_version", "document_type", "plan_digest", "rows"},
        "C8 failure_table.json",
    )
    rows = table["rows"]
    if not isinstance(rows, list) or len(rows) != C8_EXECUTION_UNIT_COUNT:
        raise StageCSelectionIntegrityError(
            "C8 failure table does not contain exactly 2,180 rows"
        )
    if (
        table["schema_version"] != 1
        or table["document_type"] != "stage_c_failure_table"
        or table["plan_digest"] != plan_digest
        or status["schema_version"] != 1
        or status["document_type"] != "stage_c_batch_status"
        or status["plan_digest"] != plan_digest
        or status["expected_unit_count"] != C8_EXECUTION_UNIT_COUNT
        or status["lifecycle_state"] != "finalized"
        or status["orchestration_complete"] is not True
        or type(status["successful"]) is not bool
        or status["active_claim_count"] != 0
        or status["failure_table_sha256"] != failure_table_sha256
    ):
        raise StageCSelectionIntegrityError(
            "C8 source is not an exact finalized batch status"
        )
    if (
        not isinstance(status["validation_errors"], list)
        or status["validation_errors"]
    ):
        raise StageCSelectionIntegrityError(
            "C8 finalized batch retains validation errors"
        )


def build_c8_calibration_evidence(
    output_root: Path | str, plan: ExecutionPlan, config: Any,
    observed_git_state: Mapping[str, Any], c7_report_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Read and normalize a complete formal C8 root without image/L3 reads.

    The existing batch validator verifies every inventory payload (including
    L1/L2 and display-only L3 hashes).  This reader subsequently opens only
    the two L0 JSON records needed for quantitative aggregation/provenance.
    """

    plan_document = _validate_exact_plan(plan, config)
    code = dict(_exact(observed_git_state, {"commit", "dirty"}, "observed_git_state"))
    if code.get("dirty") is not False:
        raise StageCSelectionIntegrityError("formal C8 evidence requires clean Git state")
    root = Path(output_root).absolute()
    if root.is_symlink() or not root.is_dir():
        raise StageCSelectionIntegrityError("C8 output root is missing or unsafe")
    from .stage_c_benchmark import map_execution_records
    from .stage_c_resources import measure_artifact_tree_bytes

    mapped = map_execution_records(root, plan, observed_git_state=code)
    validation = mapped.get("batch_validation")
    if not isinstance(validation, Mapping):
        raise StageCSelectionIntegrityError("C8 batch validation is unavailable")
    mapped_records = mapped.get("records")
    if not isinstance(mapped_records, list) or len(mapped_records) != C8_EXECUTION_UNIT_COUNT:
        raise StageCSelectionIntegrityError("C8 source did not map exactly 2,180 records")
    control_documents: dict[str, dict[str, Any]] = {}
    control_hashes: dict[str, str] = {}
    for name in _CONTROL_NAMES:
        path = root / name
        control_documents[name] = _strict_json(path)
        control_hashes[name] = _file_sha256(path)
    if control_documents["execution_plan.json"] != plan_document:
        raise StageCSelectionIntegrityError("persisted C8 execution plan drifted")
    preflight = control_documents["preflight.json"]
    if (
        preflight.get("plan_digest") != plan.canonical_digest
        or preflight.get("within_budget") is not True
        or preflight.get("violations") != []
    ):
        raise StageCSelectionIntegrityError("C8 preflight is not complete and passing")
    preflight_digest = preflight.get("canonical_digest")
    _sha(preflight_digest, "C8 preflight canonical digest")
    if preflight_digest != _self_digest(preflight):
        raise StageCSelectionIntegrityError("C8 preflight canonical digest mismatch")
    batch_status = control_documents["batch_status.json"]
    failure_table = control_documents["failure_table.json"]
    _validate_finalized_batch_status(
        batch_status, failure_table, plan_digest=plan.canonical_digest,
        failure_table_sha256=control_hashes["failure_table.json"],
    )
    # A finalized batch with ordinary terminal failures is valid C8 evidence:
    # it must produce an explicit BLOCKED report.  Missing/tampered/incomplete
    # units or any mapping/control error remain integrity failures.
    validation_rows = validation.get("rows")
    validation_errors = validation.get("errors")
    if (
        validation.get("orchestration_complete") is not True
        or not isinstance(validation_rows, list)
        or len(validation_rows) != C8_EXECUTION_UNIT_COUNT
        or not isinstance(validation_errors, list)
        or validation_errors
        or any(
            not isinstance(item, Mapping)
            or item.get("state") not in {"succeeded", "failed"}
            for item in validation_rows
        )
    ):
        raise StageCSelectionIntegrityError(
            "C8 batch integrity validation did not produce an exact terminal batch"
        )
    batch_complete = batch_status.get("orchestration_complete") is True
    batch_successful = batch_status.get("successful") is True
    records: list[dict[str, Any]] = []
    for index, mapped_record in enumerate(mapped_records):
        if not isinstance(mapped_record, Mapping):
            raise StageCSelectionIntegrityError(f"mapped record {index} is not an object")
        identity = plan.units[index].key.to_dict()
        selected_value = mapped_record.get("attempt_path")
        result: Mapping[str, Any] = {}
        provenance: Mapping[str, Any] = {}
        if isinstance(selected_value, str) and selected_value:
            selected = _safe_selected_attempt(root, selected_value)
            result_path = selected / "L0/result.json"
            provenance_path = selected / "L0/provenance.json"
            if result_path.is_file():
                result = _strict_json(result_path)
            if provenance_path.is_file():
                provenance = _strict_json(provenance_path)
        if mapped_record.get("unit_key") != identity:
            raise StageCSelectionIntegrityError(f"mapped identity drift at index {index}")
        if result and result.get("unit_identity") != identity:
            raise StageCSelectionIntegrityError(f"L0 identity drift at index {index}")
        telemetry = mapped_record.get("runner_telemetry")
        record = {
            "plan_index": index, "unit_key": identity,
            "state": mapped_record.get("state"),
            "trusted_by_batch_validator": mapped_record.get("trusted_by_batch_validator"),
            "hashes": _plain({
                name: mapped_record.get("hashes", {}).get(name)
                for name in _HASH_KEYS
            }),
            "scalars": _plain(result.get("scalars", {})),
            "provenance": _plain(provenance),
            "resources": _plain(result.get("resources", {})),
            "runner_telemetry": _plain(telemetry or {}),
            "artifact_inventory": _plain(mapped_record.get("artifact_inventory", [])),
            "work_item_counts": _plain(mapped_record.get("work_item_counts", {})),
            "audit": bool(mapped_record.get("audit")),
            "validation_errors": _plain(
                [*mapped_record.get("validation_errors", []), *mapped_record.get("errors", [])]
            ),
            "attempt_path": mapped_record.get("attempt_path"),
            "l0_l1_cap_accounting_bytes": mapped_record.get("l0_l1_cap_accounting_bytes"),
            "level_bytes": _plain(mapped_record.get("level_bytes", {})),
            "total_regular_file_bytes": mapped_record.get("total_regular_file_bytes"),
        }
        # This marker is added only by the source reader after the existing
        # batch validator has validated the full provenance document.  It is
        # intentionally not part of the injectable evidence schema.
        records.append(record)
    binding = None if c7_report_binding is None else _plain(c7_report_binding)
    if isinstance(binding, Mapping) and "source_batch_control_bytes" not in binding:
        binding = dict(binding)
        binding["source_batch_control_bytes"] = sum(
            (root / name).stat().st_size for name in _CONTROL_NAMES
        )
    document: dict[str, Any] = {
        "schema_version": 1, "document_type": EVIDENCE_DOCUMENT_TYPE,
        "source_root": str(root), "holdout_accessed": False,
        "selection_performed": False, "bootstrap_performed": False,
        "holm_performed": False, "decision_performed": False,
        "expected_unit_count": C8_EXECUTION_UNIT_COUNT,
        "batch_complete": batch_complete, "batch_successful": batch_successful,
        "calibration_sample_ids": list(CALIBRATION_SAMPLE_IDS),
        "method_ids": list(METHOD_ORDER),
        "bindings": {
            "source_root": str(root),
            "execution_plan_sha256": plan.canonical_digest,
            "run_spec_sha256": canonical_sha256(plan.run_spec.to_dict()),
            "preflight_sha256": preflight_digest,
            "control_file_sha256": control_hashes, "code": code,
            "source_tree_total_bytes": measure_artifact_tree_bytes(root).total_bytes,
        },
        "records": records,
        "calibration_artifact_manifest_sha256": _records_digest(records),
        "c7_report_binding": binding,
    }
    document["canonical_digest"] = _self_digest(document)
    validate_c8_calibration_evidence(document)
    return document


def _candidate_unit_records(
    records: Sequence[Mapping[str, Any]], block: int, head: int,
) -> list[Mapping[str, Any]]:
    return [
        record for record in records
        if record["unit_key"]["block_index"] == block
        and record["unit_key"]["head_index"] == head
        and record["unit_key"]["method_id"] in VALUE_METHOD_IDS
    ]


def _sample_candidate_metrics(
    by_method: Mapping[str, Mapping[str, Any]], sample_id: str, block: int, head: int,
) -> tuple[dict[str, Any], list[tuple[str, str]]]:
    failures: list[tuple[str, str]] = []
    for method_id in VALUE_METHOD_IDS:
        context = f"{sample_id}:{method_id}"
        failures.extend((sample_id, item) for item in _record_validity(by_method[method_id], context))
    direct, nns, matched = (by_method[item] for item in VALUE_METHOD_IDS)
    if failures:
        return {
            "sample_id": sample_id, "delta_fid_fine4": None,
            "residual_spearman": {item: None for item in PERTURBATION_IDS},
            "gaussian_full_mask_clean_rho": None,
            "spatial_degenerate": False, "valid": False,
        }, failures
    for label, record, ig in (("direct", direct, False), ("NNS", nns, True), ("matched", matched, False)):
        try:
            _validate_required_l1(record)
            _validate_map_coverage(record, ig=ig)
        except StageCSelectionIntegrityError as error:
            failures.append((sample_id, f"{label}:{error}"))
    try:
        _validate_direct_fidelity_absence(direct)
    except StageCSelectionIntegrityError as error:
        failures.append((sample_id, f"direct:{error}"))
    for label, record in (("NNS", nns), ("matched", matched)):
        try:
            _validate_targeted_fidelity_shape(record)
        except StageCSelectionIntegrityError as error:
            failures.append((sample_id, f"{label}:{error}"))
    scientific_target_identity_sha256: str | None = None
    try:
        scientific_target_identity_sha256 = _target_checks(
            nns, matched, block, head
        )
    except StageCSelectionIntegrityError as error:
        failures.append((sample_id, f"target:{error}"))
    try:
        _validate_token_parity(matched)
    except StageCSelectionIntegrityError as error:
        failures.append((sample_id, f"token_parity:{error}"))
    delta: float | None = None
    try:
        if scientific_target_identity_sha256 is None:
            raise StageCSelectionIntegrityError(
                "scientific target identity is unavailable"
            )
        nns_aggregates = _validated_aggregates(
            nns,
            scientific_target_identity_sha256=scientific_target_identity_sha256,
        )
        matched_aggregates = _validated_aggregates(
            matched,
            scientific_target_identity_sha256=scientific_target_identity_sha256,
        )
        outcome = value_path_delta_fidelity(
            nns_insertion=nns_aggregates[(FidelityScale.FINE_4.value, CurveDirection.INSERTION.value)],
            nns_deletion=nns_aggregates[(FidelityScale.FINE_4.value, CurveDirection.DELETION.value)],
            matched_insertion=matched_aggregates[(FidelityScale.FINE_4.value, CurveDirection.INSERTION.value)],
            matched_deletion=matched_aggregates[(FidelityScale.FINE_4.value, CurveDirection.DELETION.value)],
        )
        if isinstance(outcome, StructuredNA):
            raise StageCSelectionIntegrityError("fine-4 delta fidelity is NA")
        delta = _finite(outcome.value, "delta_fid_fine4")
    except Exception as error:
        failures.append((sample_id, f"fidelity:{error}"))
    residuals: dict[str, float | None] = {item: None for item in PERTURBATION_IDS}
    rho: float | None = None
    spatial = False
    try:
        residuals, rho, spatial = _residual_values(nns)
    except StageCSelectionIntegrityError as error:
        failures.append((sample_id, f"residual:{error}"))
    return {
        "sample_id": sample_id, "delta_fid_fine4": delta,
        "residual_spearman": residuals,
        "gaussian_full_mask_clean_rho": rho,
        "spatial_degenerate": spatial,
        "valid": not failures,
    }, failures


def _build_candidate(
    records: Sequence[Mapping[str, Any]], block: int, head: int,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    candidate_records = _candidate_unit_records(records, block, head)
    if len(candidate_records) != 15:
        raise StageCSelectionIntegrityError(
            f"candidate {_candidate_key(block, head)} does not have exactly 15 records"
        )
    expected = [
        (sample_id, method_id)
        for sample_id in CALIBRATION_SAMPLE_IDS for method_id in VALUE_METHOD_IDS
    ]
    observed = [(item["unit_key"]["sample_id"], item["unit_key"]["method_id"]) for item in candidate_records]
    if observed != expected:
        raise StageCSelectionIntegrityError("candidate records are not sample/method ordered")
    per_sample: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for sample_id in CALIBRATION_SAMPLE_IDS:
        by_method = {
            item["unit_key"]["method_id"]: item
            for item in candidate_records if item["unit_key"]["sample_id"] == sample_id
        }
        metrics, sample_failures = _sample_candidate_metrics(by_method, sample_id, block, head)
        per_sample.append(metrics)
        failures.extend(
            {"candidate_key": _candidate_key(block, head), "sample_id": item_sample, "reason": reason}
            for item_sample, reason in sample_failures
        )
    deltas = [item["delta_fid_fine4"] for item in per_sample]
    complete_delta = all(value is not None for value in deltas)
    n_pos = sum(value > 0 for value in deltas if value is not None) if complete_delta else None
    m_fid = float(statistics.median(deltas)) if complete_delta else None
    perturbation_medians: dict[str, float | None] = {}
    for perturbation_id in PERTURBATION_IDS:
        values = [item["residual_spearman"][perturbation_id] for item in per_sample]
        perturbation_medians[perturbation_id] = (
            None if any(value is None for value in values)
            else float(statistics.median(values))
        )
    m_stab = (
        None if any(value is None for value in perturbation_medians.values())
        else min(perturbation_medians.values())
    )
    rho_values = [item["gaussian_full_mask_clean_rho"] for item in per_sample]
    m_res = (
        None if any(value is None for value in rho_values)
        else float(statistics.median(rho_values))
    )
    validity_failures = list(dict.fromkeys(item["reason"] for item in failures))
    resource_ok = not any(
        reason.endswith("resource_evidence_invalid")
        or reason.endswith("resource_hard_cap_exceeded")
        or reason.endswith("artifact_or_batch_validation_failed")
        for reason in validity_failures
    )
    eligible = not failures and complete_delta
    rank_tuple = None if not eligible else [
        n_pos, round(m_fid, 6),
        -1.0 if m_stab is None else round(m_stab, 6),
        0.0 if m_res is None else round(m_res, 6),
        -block, -head,
    ]
    selection_record = {
        **_candidate_ref(block, head), "calibration_sample_count": 5,
        "eligible": eligible, "validity_failures": validity_failures,
        "count_delta_fid_fine4_gt_zero": n_pos,
        "median_delta_fid_fine4": m_fid,
        "minimum_perturbation_median_residual_spearman": m_stab,
        "median_patch_residual_l1_ratio": m_res,
        "rounded_rank_tuple": rank_tuple,
        "resource_within_hard_budget": resource_ok,
        "artifact_manifest_sha256": _records_digest(candidate_records),
        "rank_within_depth_band": None,
        "selected_depth_band_winner": False, "selected_primary": False,
    }
    audit = {
        **_candidate_ref(block, head),
        "unit_evidence_count": 15,
        "artifact_manifest_sha256": selection_record["artifact_manifest_sha256"],
        "per_sample": per_sample,
        "perturbation_medians": perturbation_medians,
        "spatial_degenerate_sample_count": sum(item["spatial_degenerate"] for item in per_sample),
    }
    return selection_record, audit, failures


def _selection_document(
    candidates: list[dict[str, Any]], failures: list[dict[str, Any]],
    input_hashes: Mapping[str, Any], winners: Sequence[Mapping[str, Any]],
    primary: Mapping[str, Any], generator_commit: str,
) -> dict[str, Any]:
    document = {
        "schema_version": 1, "document_type": SELECTION_DOCUMENT_TYPE,
        "generated_by": f"{GENERATED_BY}@{generator_commit}",
        "holdout_accessed": False,
        "input_hashes": dict(input_hashes),
        "calibration_sample_ids": list(CALIBRATION_SAMPLE_IDS),
        "candidate_records": candidates,
        "depth_band_winners": [dict(item) for item in winners],
        # The frozen selection schema permits only candidate-scoped failures.
        # Global unit failures remain in the richer C8 report and never appear
        # in a PASS selection (they force C8 BLOCKED below).
        "primary": dict(primary),
        "failure_records": [
            item for item in failures if item.get("candidate_key") is not None
        ],
    }
    validate_stage_c_selection_document(document, expected_input_hashes=input_hashes)
    return document


def build_stage_c_selection(
    evidence: Mapping[str, Any], input_hashes: Mapping[str, Any],
    *, generator_git_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a PASS selection or an explicit BLOCKED report.

    A BLOCKED result always has ``selection_document=None`` and is therefore
    impossible to publish as the frozen selection file.
    """

    validate_c8_calibration_evidence(evidence)
    document = _plain(evidence, "evidence")
    generator = _selection_generator_code(
        document["bindings"]["code"]
        if generator_git_state is None else generator_git_state
    )
    budget = _post_c8_budget(document)
    hashes = _validate_input_hashes(
        input_hashes, document["calibration_artifact_manifest_sha256"],
        document["bindings"]["code"], document["records"],
    )
    records = document["records"]
    candidates: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for block in range(12):
        for head in range(12):
            candidate, audit, candidate_failures = _build_candidate(records, block, head)
            candidates.append(candidate)
            audits.append(audit)
            failures.extend(candidate_failures)
    global_failures: list[dict[str, Any]] = []
    for record in records:
        key = record["unit_key"]
        if key["method_id"] not in CLASS_METHOD_IDS:
            continue
        unit_errors = _record_validity(
            record, f"{key['sample_id']}:{key['method_id']}"
        )
        for reason in unit_errors:
            global_failures.append({
                "candidate_key": None,
                "sample_id": key["sample_id"],
                "reason": f"global_unit:{key['method_id']}:{reason}",
                "unit_key": dict(key),
            })
    failures.extend(global_failures)
    winners: list[dict[str, Any]] = []
    blocked_reasons: list[str] = []
    for band in range(4):
        eligible = [item for item in candidates if item["depth_band"] == band and item["eligible"]]
        ordered = sorted(eligible, key=lambda item: tuple(item["rounded_rank_tuple"]), reverse=True)
        for rank, item in enumerate(ordered, 1):
            item["rank_within_depth_band"] = rank
        if not ordered:
            blocked_reasons.append(f"depth_band_{band}_has_no_eligible_candidate")
            continue
        ordered[0]["selected_depth_band_winner"] = True
        winners.append(_candidate_ref(ordered[0]["block_index"], ordered[0]["head_index"]))
    if not document["batch_complete"]:
        blocked_reasons.append("c8_batch_not_complete")
    if global_failures:
        blocked_reasons.append("global_class_unit_failure")
    if budget["status"] != STATUS_PASS:
        blocked_reasons.append(str(budget["reason"]))
    primary: dict[str, Any] | None = None
    if len(winners) == 4:
        by_key = {item["candidate_key"]: item for item in candidates}
        winning_record = max(
            (by_key[item["candidate_key"]] for item in winners),
            key=lambda item: tuple(item["rounded_rank_tuple"]),
        )
        winning_record["selected_primary"] = True
        primary = _candidate_ref(winning_record["block_index"], winning_record["head_index"])
    status = STATUS_PASS if not blocked_reasons else STATUS_BLOCKED
    selection = None
    if status == STATUS_PASS:
        assert primary is not None
        selection = _selection_document(
            candidates, failures, hashes, winners, primary, generator["commit"]
        )
    else:
        for item in candidates:
            item["selected_depth_band_winner"] = False
            item["selected_primary"] = False
    summary = {
        "expected_unit_count": C8_EXECUTION_UNIT_COUNT,
        "observed_unit_count": len(records),
        "succeeded_unit_count": sum(item["state"] == "succeeded" for item in records),
        "trusted_unit_count": sum(item["trusted_by_batch_validator"] is True for item in records),
        "candidate_count": len(candidates),
        "eligible_candidate_count": sum(item["eligible"] for item in candidates),
        "candidate_manifest_unit_count": 15,
    }
    report: dict[str, Any] = {
        "schema_version": 1, "document_type": REPORT_DOCUMENT_TYPE,
        "status": status, "integrity_valid": True,
        "blocked_reasons": list(dict.fromkeys(blocked_reasons)),
        "holdout_accessed": False,
        "selection_performed": status == STATUS_PASS,
        "bootstrap_performed": False, "holm_performed": False,
        "decision_performed": False,
        "bindings": {
            **document["bindings"],
            "c7_report_binding": document["c7_report_binding"],
            "calibration_evidence_sha256": document["canonical_digest"],
            "calibration_artifact_manifest_sha256": document["calibration_artifact_manifest_sha256"],
            "input_hashes": hashes,
            "selection_generator_code": generator,
        },
        "evidence_summary": summary,
        "post_c8_budget": budget,
        "candidate_records": candidates, "candidate_audit_records": audits,
        "depth_band_winners": winners if status == STATUS_PASS else [],
        "primary": primary if status == STATUS_PASS else None,
        "failure_records": failures, "selection_document": selection,
        "selection_document_sha256": None if selection is None else _digest(selection),
    }
    report["canonical_digest"] = _self_digest(report)
    validate_stage_c_selection_report(
        report, generator_git_state=generator
    )
    return report


def validate_stage_c_selection_document(
    document: Mapping[str, Any], *, expected_input_hashes: Mapping[str, Any] | None = None,
) -> None:
    """Public wrapper for the frozen schema's strict semantic validator."""

    try:
        _validate_selection_document(document)
    except Exception as error:
        raise StageCSelectionIntegrityError(f"invalid selection document: {error}") from error
    candidates = document["candidate_records"]
    observed = [(item["block_index"], item["head_index"]) for item in candidates]
    expected = [(block, head) for block in range(12) for head in range(12)]
    if observed != expected:
        raise StageCSelectionIntegrityError("selection candidates must be block-major/head-minor")
    if expected_input_hashes is not None and document["input_hashes"] != dict(expected_input_hashes):
        raise StageCSelectionIntegrityError("selection input hashes differ from expected bindings")
    failure_tuples = [(item["candidate_key"], item["sample_id"], item["reason"]) for item in document["failure_records"]]
    if len(failure_tuples) != len(set(failure_tuples)):
        raise StageCSelectionIntegrityError("selection failure records contain duplicates")


def validate_stage_c_selection_report(
    report: Mapping[str, Any], *, evidence: Mapping[str, Any] | None = None,
    input_hashes: Mapping[str, Any] | None = None,
    generator_git_state: Mapping[str, Any] | None = None,
) -> None:
    """Validate report integrity and optionally recompute it from evidence."""

    keys = {
        "schema_version", "document_type", "status", "integrity_valid",
        "blocked_reasons", "holdout_accessed", "selection_performed",
        "bootstrap_performed", "holm_performed", "decision_performed",
        "bindings", "evidence_summary", "candidate_records",
        "post_c8_budget",
        "candidate_audit_records", "depth_band_winners", "primary",
        "failure_records", "selection_document",
        "selection_document_sha256", "canonical_digest",
    }
    document = _exact(report, keys, "selection report")
    if document["schema_version"] != 1 or document["document_type"] != REPORT_DOCUMENT_TYPE:
        raise StageCSelectionIntegrityError("selection report identity fields are invalid")
    if document["canonical_digest"] != _self_digest(document):
        raise StageCSelectionIntegrityError("selection report canonical digest mismatch")
    if document["integrity_valid"] is not True or document["holdout_accessed"] is not False:
        raise StageCSelectionIntegrityError("selection report integrity/access flags are invalid")
    bindings = document["bindings"]
    if not isinstance(bindings, Mapping):
        raise StageCSelectionIntegrityError("selection report bindings are invalid")
    generator = _selection_generator_code(
        bindings.get("selection_generator_code")
    )
    if generator_git_state is not None:
        expected_generator_value = generator_git_state
    elif evidence is not None:
        expected_generator_value = evidence["bindings"]["code"]
    else:
        expected_generator_value = generator
    expected_generator = _selection_generator_code(expected_generator_value)
    if generator != expected_generator:
        raise StageCSelectionIntegrityError(
            "selection generator code binding differs from expected Git state"
        )
    for flag in ("bootstrap_performed", "holm_performed", "decision_performed"):
        if document[flag] is not False:
            raise StageCSelectionIntegrityError(f"selection report {flag} must be false")
    status = document["status"]
    if status not in {STATUS_PASS, STATUS_BLOCKED}:
        raise StageCSelectionIntegrityError("selection report status is invalid")
    if not isinstance(document["candidate_records"], list) or len(document["candidate_records"]) != 144:
        raise StageCSelectionIntegrityError("selection report requires 144 candidates")
    budget = document["post_c8_budget"]
    if not isinstance(budget, Mapping) or budget.get("exact_unit_count") != 2500 or budget.get("projection_safety_factor") != 1.25:
        raise StageCSelectionIntegrityError("selection report post-C8 budget operands drifted")
    if status == STATUS_PASS and budget.get("status") != STATUS_PASS:
        raise StageCSelectionIntegrityError("PASS selection has a blocked post-C8 budget")
    if status == STATUS_PASS:
        if document["blocked_reasons"] != [] or document["selection_performed"] is not True:
            raise StageCSelectionIntegrityError("PASS report flags are inconsistent")
        selection = document["selection_document"]
        if selection is None or selection["candidate_records"] != document["candidate_records"]:
            raise StageCSelectionIntegrityError("PASS report selection payload disagrees")
        if selection.get("generated_by") != f"{GENERATED_BY}@{generator['commit']}":
            raise StageCSelectionIntegrityError(
                "selection generated_by differs from generator code binding"
            )
        validate_stage_c_selection_document(
            selection, expected_input_hashes=document["bindings"]["input_hashes"]
        )
        if document["selection_document_sha256"] != _digest(selection):
            raise StageCSelectionIntegrityError("selection document digest mismatch")
    else:
        if (
            not isinstance(document["blocked_reasons"], list)
            or not document["blocked_reasons"]
            or document["selection_performed"] is not False
            or document["selection_document"] is not None
            or document["selection_document_sha256"] is not None
            or document["depth_band_winners"] != []
            or document["primary"] is not None
        ):
            raise StageCSelectionIntegrityError("BLOCKED report contains a selection")
    if evidence is not None:
        if input_hashes is None:
            raise TypeError("input_hashes are required for source recomputation")
        rebuilt = build_stage_c_selection(
            evidence, input_hashes, generator_git_state=expected_generator
        )
        if _plain(rebuilt) != _plain(report):
            raise StageCSelectionIntegrityError(
                "selection report differs from source-evidence recomputation"
            )


def validate_stage_c_selection(
    value: Mapping[str, Any], *, evidence: Mapping[str, Any] | None = None,
    input_hashes: Mapping[str, Any] | None = None,
    generator_git_state: Mapping[str, Any] | None = None,
) -> None:
    """Validate either a frozen selection document or its C8 report."""

    if not isinstance(value, Mapping):
        raise TypeError("value must be a mapping")
    if value.get("document_type") == SELECTION_DOCUMENT_TYPE:
        validate_stage_c_selection_document(value, expected_input_hashes=input_hashes)
    elif value.get("document_type") == REPORT_DOCUMENT_TYPE:
        validate_stage_c_selection_report(
            value, evidence=evidence, input_hashes=input_hashes,
            generator_git_state=generator_git_state,
        )
    else:
        raise StageCSelectionIntegrityError("unknown selection document type")


def _pretty_json_bytes(document: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            _plain(document), indent=2, sort_keys=True,
            ensure_ascii=False, allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _exclusive_json(path: Path, document: Mapping[str, Any]) -> str:
    if path.parent.exists() and path.parent.is_symlink():
        raise StageCSelectionIntegrityError("output parent cannot be a symlink")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _pretty_json_bytes(document)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise
    return hashlib.sha256(payload).hexdigest()


def _validate_existing_output(path: Path, expected: bytes, context: str) -> str | None:
    """Return the exact existing digest, or ``None`` when the path is absent."""

    if not os.path.lexists(path):
        return None
    try:
        mode = path.lstat().st_mode
    except OSError as error:
        raise StageCSelectionIntegrityError(
            f"cannot inspect existing {context}: {error}"
        ) from error
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise StageCSelectionIntegrityError(
            f"existing {context} is not a safe regular file"
        )
    try:
        observed = path.read_bytes()
    except OSError as error:
        raise StageCSelectionIntegrityError(
            f"cannot read existing {context}: {error}"
        ) from error
    if observed != expected:
        raise StageCSelectionIntegrityError(
            f"existing {context} differs from source-evidence recomputation"
        )
    return hashlib.sha256(observed).hexdigest()


def _create_or_accept_exact_json(
    path: Path, document: Mapping[str, Any], *, context: str,
) -> str:
    """Idempotently publish one immutable JSON file without accepting drift."""

    expected = _pretty_json_bytes(document)
    existing = _validate_existing_output(path, expected, context)
    if existing is not None:
        return existing
    try:
        return _exclusive_json(path, document)
    except FileExistsError:
        # A concurrent publisher may have won the O_EXCL race.  It is safe to
        # accept only the exact bytes derived from this evidence.
        raced = _validate_existing_output(path, expected, context)
        if raced is None:
            raise
        return raced


def write_stage_c_selection(
    path: Path | str, value: Mapping[str, Any], *,
    evidence: Mapping[str, Any] | None = None,
    input_hashes: Mapping[str, Any] | None = None,
    generator_git_state: Mapping[str, Any] | None = None,
) -> str:
    """Exclusively create the PASS selection file and return its byte hash."""
    if evidence is None or input_hashes is None:
        raise TypeError("formal selection publication requires evidence and input_hashes")

    report = value if value.get("document_type") == REPORT_DOCUMENT_TYPE else None
    if report is not None:
        validate_stage_c_selection_report(
            report, evidence=evidence, input_hashes=input_hashes,
            generator_git_state=generator_git_state,
        )
        if report["status"] != STATUS_PASS:
            raise StageCSelectionBlockedError("BLOCKED C8 has no selection file", report=report)
        document = report["selection_document"]
    else:
        rebuilt = build_stage_c_selection(
            evidence, input_hashes, generator_git_state=generator_git_state
        )
        if rebuilt["status"] != STATUS_PASS or rebuilt["selection_document"] != value:
            raise StageCSelectionIntegrityError(
                "selection document differs from source-evidence recomputation"
            )
        document = value
    return _exclusive_json(Path(path).absolute(), document)


def write_stage_c_selection_report(
    path: Path | str, report: Mapping[str, Any], *,
    evidence: Mapping[str, Any] | None = None,
    input_hashes: Mapping[str, Any] | None = None,
    generator_git_state: Mapping[str, Any] | None = None,
) -> str:
    """Exclusively create a validated PASS or BLOCKED selection report."""
    if evidence is None or input_hashes is None:
        raise TypeError("formal report publication requires evidence and input_hashes")

    validate_stage_c_selection_report(
        report, evidence=evidence, input_hashes=input_hashes,
        generator_git_state=generator_git_state,
    )
    return _exclusive_json(Path(path).absolute(), report)


def write_stage_c_selection_outputs(
    selection_path: Path | str, report_path: Path | str,
    report: Mapping[str, Any], *, evidence: Mapping[str, Any],
    input_hashes: Mapping[str, Any],
    generator_git_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Publish the report and, only for PASS, the frozen selection."""

    validate_stage_c_selection_report(
        report, evidence=evidence, input_hashes=input_hashes,
        generator_git_state=generator_git_state,
    )
    selection_destination = Path(selection_path).absolute()
    report_destination = Path(report_path).absolute()
    if selection_destination.resolve(strict=False) == report_destination.resolve(strict=False):
        raise StageCSelectionIntegrityError(
            "selection and report output paths must be distinct"
        )

    report_payload = _pretty_json_bytes(report)
    report_hash = _validate_existing_output(
        report_destination, report_payload, "selection report"
    )
    selection_hash: str | None = None
    selection_document = report.get("selection_document")
    if report["status"] == STATUS_PASS:
        if not isinstance(selection_document, Mapping):
            raise StageCSelectionIntegrityError(
                "PASS report has no selection document"
            )
        selection_payload = _pretty_json_bytes(selection_document)
        selection_hash = _validate_existing_output(
            selection_destination, selection_payload, "selection document"
        )
    elif os.path.lexists(selection_destination):
        raise StageCSelectionIntegrityError(
            "BLOCKED C8 must not have an existing selection output"
        )

    # Validate every pre-existing side before creating either missing side.
    # Report-first ordering means a failure during the second creation leaves
    # a precisely verifiable partial publication that can be retried.
    if report_hash is None:
        report_hash = _create_or_accept_exact_json(
            report_destination, report, context="selection report"
        )
    if report["status"] == STATUS_PASS and selection_hash is None:
        assert isinstance(selection_document, Mapping)
        selection_hash = _create_or_accept_exact_json(
            selection_destination, selection_document,
            context="selection document",
        )
    return {
        "status": report["status"], "report_sha256": report_hash,
        "selection_sha256": selection_hash,
    }


def validate_stage_c_selection_from_source(
    selection_path: Path | str | None, report_path: Path | str, *,
    output_root: Path | str, plan: ExecutionPlan, config: Any,
    observed_git_state: Mapping[str, Any], input_hashes: Mapping[str, Any],
    c7_report_binding: Mapping[str, Any] | None = None,
    generator_git_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Re-read source evidence and reject manual report/selection edits."""

    evidence = build_c8_calibration_evidence(
        output_root, plan, config, observed_git_state, c7_report_binding
    )
    report = _strict_json(Path(report_path).absolute())
    validate_stage_c_selection_report(
        report, evidence=evidence, input_hashes=input_hashes,
        generator_git_state=generator_git_state,
    )
    if report["status"] == STATUS_BLOCKED:
        if selection_path is not None and Path(selection_path).exists():
            raise StageCSelectionIntegrityError("BLOCKED report must not have a selection file")
        return report
    if selection_path is None:
        raise StageCSelectionIntegrityError("PASS report requires a selection file")
    selection = _strict_json(Path(selection_path).absolute())
    validate_stage_c_selection_document(selection, expected_input_hashes=input_hashes)
    if selection != report["selection_document"]:
        raise StageCSelectionIntegrityError("selection file differs from recomputed report")
    return report


__all__ = [
    "C8_EXECUTION_UNIT_COUNT", "C8_UNITS_PER_SAMPLE", "DEPTH_BANDS",
    "EVIDENCE_DOCUMENT_TYPE", "GENERATED_BY", "REPORT_DOCUMENT_TYPE",
    "STATUS_BLOCKED", "STATUS_PASS", "StageCSelectionBlockedError",
    "StageCSelectionError", "StageCSelectionIntegrityError",
    "build_c8_calibration_evidence", "build_stage_c_selection",
    "validate_c8_calibration_evidence", "validate_stage_c_selection",
    "validate_stage_c_selection_document",
    "validate_stage_c_selection_from_source",
    "validate_stage_c_selection_report", "write_stage_c_selection",
    "write_stage_c_selection_outputs", "write_stage_c_selection_report",
]
