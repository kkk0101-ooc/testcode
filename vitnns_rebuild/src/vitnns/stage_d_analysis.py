"""Pure, synthetic-only D6 aggregation over a D5 validation receipt.

No function in this module opens a root, image, model, map, L2 object, or
display artifact.  The only accepted numerical source is the unforgeable
same-snapshot receipt issued by :mod:`stage_d_analysis_input`.  This keeps
D5's source-recomputing validation ahead of every D6 calculation.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import InitVar, dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np

from .stage_d_analysis_input import (
    StageDValidatedAnalysisInput, StageDValidatedAnalysisUnit,
)
from .stage_d_plan import (
    CURVE_FRACTIONS, MATCHED_METHOD_ID, METHOD_IDS, NNS_METHOD_ID,
    STATISTICAL_N, canonical_sha256, validate_stage_d_plan,
)


SCHEMA_VERSION = 1
DOCUMENT_TYPE = "stage_d_analysis_result"
PRIMARY_NAMESPACE = "stage-d-bootstrap|20260821|"
FIGURE_NAMESPACE = "stage-d-figure-bootstrap|20260821|"
PRIMARY_METRIC_ID = "fine_4x4_cell_delta_fid_median"
BOOTSTRAP_RESAMPLES = 10_000
_SHA256 = frozenset("0123456789abcdef")
_RESOURCE_FIELDS = (
    "elapsed_seconds", "forward_calls", "backward_calls",
    "l0_l1_payload_bytes", "l2_payload_bytes",
    "control_and_provenance_bytes", "cuda_peak_reserved_bytes",
)


class StageDAnalysisError(RuntimeError):
    """Base error for D6 closed-input analysis."""


class StageDAnalysisContractError(StageDAnalysisError, ValueError):
    """The receipt, plan, or frozen aggregation contract drifted."""


_RESULT_ISSUER = object()


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(child) for key, child in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(child) for child in value)
    if isinstance(value, tuple):
        return tuple(_freeze(child) for child in value)
    return value


@dataclass(frozen=True, slots=True, eq=False)
class StageDAnalysisResult(Mapping[str, Any]):
    """In-process-only result issued by the D5-receipt aggregation path."""

    _data: Mapping[str, Any] = field(repr=False)
    _issuer: InitVar[object] = None

    def __post_init__(self, _issuer: object) -> None:
        if _issuer is not _RESULT_ISSUER:
            raise TypeError("StageDAnalysisResult is issued only by D6 aggregation")
        value = _plain(self._data, "analysis result")
        object.__setattr__(self, "_data", _freeze(value))

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def to_dict(self) -> dict[str, Any]:
        return _plain(self._data, "analysis result")


def _plain(value: Any, context: str = "value") -> Any:
    method = getattr(value, "to_dict", None)
    if callable(method):
        value = method()
    if value is None or type(value) in {str, bool, int}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise StageDAnalysisContractError(f"{context} contains non-finite number")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, child in value.items():
            if not isinstance(key, str) or not key:
                raise StageDAnalysisContractError(f"{context} has an invalid key")
            result[key] = _plain(child, f"{context}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [_plain(child, f"{context}[]") for child in value]
    raise StageDAnalysisContractError(
        f"{context} contains unsupported {type(value).__name__}"
    )


def _self_document(**fields: Any) -> dict[str, Any]:
    document = {
        "schema_version": SCHEMA_VERSION, "document_type": DOCUMENT_TYPE,
        **_plain(fields),
    }
    document["canonical_digest"] = canonical_sha256(document)
    return document


def _issue_result(document: dict[str, Any]) -> StageDAnalysisResult:
    return StageDAnalysisResult(document, _issuer=_RESULT_ISSUER)


def _seed(material: str) -> int:
    return int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:8], "big")


def _median(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise StageDAnalysisContractError("median input must be finite non-empty vector")
    ordered = np.sort(values, kind="stable")
    middle = len(ordered) // 2
    return float(ordered[middle]) if len(ordered) % 2 else float(
        (ordered[middle - 1] + ordered[middle]) * 0.5
    )


def _bootstrap(
    values: np.ndarray, material: str, *, metric_id: str = PRIMARY_METRIC_ID,
) -> tuple[dict[str, Any], np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (STATISTICAL_N,) or not np.isfinite(values).all():
        raise StageDAnalysisContractError("bootstrap requires all 500 finite image values")
    seed = _seed(material)
    generator = np.random.Generator(np.random.PCG64(seed))
    draws = generator.integers(0, STATISTICAL_N, size=(BOOTSTRAP_RESAMPLES, STATISTICAL_N), endpoint=False, dtype=np.int64)
    sampled = values[draws]
    # Selection is exactly the frozen even-median rule but avoids sorting the
    # other 498 values of each deterministic 500-image draw.
    selected = np.partition(sampled, (249, 250), axis=1)
    distribution = (selected[:, 249] + selected[:, 250]) * 0.5
    ci = np.quantile(distribution, [0.025, 0.975], method="linear")
    draw_bytes = np.ascontiguousarray(draws, dtype="<i8").tobytes(order="C")
    distribution_bytes = np.ascontiguousarray(distribution, dtype="<f8").tobytes(order="C")
    metadata = {
        "status": "PERFORMED", "metric_id": metric_id,
        "seed_material": material, "seed": str(seed),
        "seed_hash_extraction": "sha256_first_8_bytes_big_endian_unsigned",
        "prng": "numpy.random.Generator(numpy.random.PCG64(seed))",
        "resamples": BOOTSTRAP_RESAMPLES, "draw_shape": [BOOTSTRAP_RESAMPLES, STATISTICAL_N],
        "draw_indices_dtype_and_order": "little_endian_int64_c_order",
        "distribution_dtype_and_order": "little_endian_float64_c_order",
        "statistic": "paired_median",
        "even_median_rule": "arithmetic_mean_of_two_middle_order_statistics",
        "quantile_probabilities": [0.025, 0.975], "quantile_method": "linear",
        "confidence_interval": [float(ci[0]), float(ci[1])],
        "draw_indices_sha256": hashlib.sha256(draw_bytes).hexdigest(),
        "distribution_sha256": hashlib.sha256(distribution_bytes).hexdigest(),
    }
    return metadata, distribution


def _curve_auc(record: Mapping[str, Any]) -> float:
    actual = np.asarray(record.get("actual_fractions"), dtype=np.float64)
    responses = np.asarray(record.get("responses"), dtype=np.float64)
    if actual.shape != (21,) or responses.shape != (21,) or not (
        np.isfinite(actual).all() and np.isfinite(responses).all()
    ):
        raise StageDAnalysisContractError("validated L0 curve record is malformed")
    return float(np.sum(np.diff(actual) * (responses[:-1] + responses[1:]) * 0.5, dtype=np.float64))


def _extract_unit(
    unit: StageDValidatedAnalysisUnit,
) -> tuple[dict[str, float], dict[str, float], np.ndarray, bool | None]:
    if unit.l0 is None or unit.curves is None or unit.numeric_payload_status != "VALIDATED":
        raise StageDAnalysisContractError("valid analysis input withheld or lacks numerical payload")
    l0 = unit.l0
    records = l0.get("fidelity_curves")
    if not isinstance(records, (list, tuple)) or len(records) != 10:
        raise StageDAnalysisContractError("validated L0 lacks the ten frozen curve records")
    aucs: dict[str, float] = {}
    repeat_sd: dict[str, float] = {}
    for direction_index, direction in enumerate(("insertion", "deletion")):
        values: list[float] = []
        for repeat in range(5):
            record = records[direction_index * 5 + repeat]
            if not isinstance(record, Mapping) or record.get("direction") != direction or record.get("replicate") != repeat:
                raise StageDAnalysisContractError("validated L0 curve order drifted")
            value = _curve_auc(record)
            if not math.isclose(value, float(record.get("auc")), abs_tol=1e-12, rel_tol=1e-12):
                raise StageDAnalysisContractError("validated L0 curve AUC does not recompute")
            values.append(value)
        aucs[direction] = float(np.mean(np.asarray(values, dtype=np.float64)))
        repeat_sd[direction] = float(np.std(np.asarray(values, dtype=np.float64), ddof=0))
    curves = np.asarray(unit.curves, dtype=np.float64)
    if curves.shape != (2, 5, 21) or not np.isfinite(curves).all():
        raise StageDAnalysisContractError("validated L1 curves are malformed")
    for direction_index, direction in enumerate(("insertion", "deletion")):
        for repeat in range(5):
            observed = np.asarray(records[direction_index * 5 + repeat]["responses"], dtype=np.float64)
            if not np.array_equal(curves[direction_index, repeat], observed):
                raise StageDAnalysisContractError("validated L0/L1 curve mismatch")
    prediction = l0.get("prediction")
    correct = prediction.get("clean_top1_correct") if isinstance(prediction, Mapping) else None
    if type(correct) is not bool:
        raise StageDAnalysisContractError("validated L0 prediction does not preserve classification status")
    return aucs, repeat_sd, curves, not correct


def _c10_reference() -> dict[str, Any]:
    return {
        "cohort": "c10", "n": 15, "historical": True, "pooled": False,
        "validity_status": "PASS_VALIDITY", "fidelity_status": "PASS_EFFECT",
        "overall_decision": "NO_GO_NOVELTY", "workflow_status": "BLOCKED",
        "spatial_status": "FAIL_SPATIAL",
        "holm_status": "HOLM_BLOCKED_UNDERSPECIFIED",
        "median": 1.7534972022995547, "positive_count": 15,
        "bootstrap_ci_lower": 1.3317595184568727,
        "bootstrap_ci_upper": 3.3813212483296478,
    }


def _campaign_cost(receipt: StageDValidatedAnalysisInput) -> dict[str, Any]:
    controls = receipt.root_validation.reconciliation
    resources = controls["resources"]
    inventory = receipt.root_inventory
    elapsed_hours = float(resources["elapsed_seconds"]) / 3600.0
    by_group: dict[int, float] = {}
    for row in receipt.unit_rows:
        by_group[row.selection_order] = by_group.get(row.selection_order, 0.0) + float(row.resources["elapsed_seconds"]) / 3600.0
    max_group = max(by_group.values(), default=0.0)
    bytes_total = int(inventory["total_bytes"])
    failures: list[str] = []
    if bytes_total > 68_719_476_736:
        failures.append("stage_d_total_retained_regular_file_bytes_max")
    if elapsed_hours > 2000.0:
        failures.append("stage_d_total_h20_gpu_hours_max")
    if max_group > 4.0:
        failures.append("stage_d_h20_gpu_hours_per_image_group_max")
    return {
        "evaluated": True, "actual_cost_includes_all_attempts_and_audit_repeats": True,
        "total_retained_regular_file_bytes": bytes_total,
        "total_h20_gpu_hours": elapsed_hours,
        "h20_gpu_hours_per_image_group_max": max_group,
        "gate_failures": failures, "status": "PASS_COST" if not failures else "FAIL_COST",
    }


def _resources(value: Mapping[str, Any]) -> dict[str, int | float]:
    if set(value) != set(_RESOURCE_FIELDS):
        raise StageDAnalysisContractError("D5 receipt resource fields drifted")
    result: dict[str, int | float] = {}
    for name in _RESOURCE_FIELDS:
        item = value[name]
        if name == "elapsed_seconds":
            if type(item) not in {int, float} or not math.isfinite(float(item)) or item < 0:
                raise StageDAnalysisContractError("D5 receipt elapsed_seconds is invalid")
            result[name] = float(item)
        elif type(item) is not int or item < 0:
            raise StageDAnalysisContractError(f"D5 receipt resource {name} is invalid")
        else:
            result[name] = item
    return result


def _reconciliation_qc(receipt: StageDValidatedAnalysisInput) -> dict[str, Any]:
    reconciliation = receipt.root_validation.reconciliation
    if not isinstance(reconciliation, Mapping):
        raise StageDAnalysisContractError("D5 reconciliation is not a mapping")
    counts = {
        "expected_image_count": reconciliation.get("expected_image_count"),
        "expected_unit_count": reconciliation.get("expected_unit_count"),
        "image_count": reconciliation.get("image_count"),
        "unit_count": reconciliation.get("unit_count"),
    }
    if counts != {
        "expected_image_count": 500, "expected_unit_count": 1000,
        "image_count": 500, "unit_count": 1000,
    }:
        raise StageDAnalysisContractError("D5 reconciliation cardinality drifted")
    resources = reconciliation.get("resources")
    if not isinstance(resources, Mapping):
        raise StageDAnalysisContractError("D5 reconciliation lacks resource totals")
    inventory = _plain(receipt.root_inventory, "D5 root inventory")
    if set(inventory) != {"file_count", "total_bytes", "canonical_digest"}:
        raise StageDAnalysisContractError("D5 root inventory is not closed")
    return {
        "d5_receipt": {
            "package_complete": receipt.package_complete,
            "scientific_validity": receipt.scientific_validity,
            "workflow_status": receipt.workflow_status,
            "plan_digest": receipt.plan_digest,
            "final_status_sha256": receipt.final_status_sha256,
            "root_inventory": inventory,
        },
        "reconciliation": {**counts, "resource_totals": _resources(resources)},
    }


def _decision(summary: Mapping[str, Any], cost: Mapping[str, Any]) -> tuple[str, str, str]:
    if summary.get("validity_status", "PASS") != "PASS":
        return "NOT_EVALUATED_DUE_TO_VALIDITY", "NOT_EVALUATED_DUE_TO_VALIDITY", "NO_GO_VALIDITY"
    if summary["positive_count"] < 400 or summary["bootstrap_ci_upper"] < 0.02:
        return "FAIL_FIDELITY", "NOT_EVALUATED_DUE_TO_FIDELITY", "NO_GO_FIDELITY"
    if (summary["median"] >= 0.02 and summary["positive_count"] >= 400
            and summary["bootstrap_ci_lower"] > 0.0):
        if cost["status"] == "FAIL_COST":
            return "PASS_FIDELITY", "FAIL_COST", "NO_GO_COST"
        return "PASS_FIDELITY", "PASS_COST", "GO_FIDELITY_SCOPE"
    return "INCONCLUSIVE_FIDELITY", "NOT_EVALUATED_DUE_TO_FIDELITY", "INCONCLUSIVE_FIDELITY"


def _invalid_rows(receipt: StageDValidatedAnalysisInput) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for order in range(STATISTICAL_N):
        pair = receipt.unit_rows[order * 2:order * 2 + 2]
        if len(pair) != 2 or tuple(row.method_id for row in pair) != METHOD_IDS:
            raise StageDAnalysisContractError("receipt no longer provides ordered NNS/matched pairs")
        nns, matched = pair
        reasons = sorted({
            *(nns.scientific_errors), *(matched.scientific_errors),
            *([nns.failure_code] if nns.failure_code else []),
            *([matched.failure_code] if matched.failure_code else []),
            *([] if nns.numeric_payload_status == "VALIDATED" else [nns.numeric_payload_status]),
            *([] if matched.numeric_payload_status == "VALIDATED" else [matched.numeric_payload_status]),
        })
        invalid = (
            nns.terminal_state != "succeeded" or matched.terminal_state != "succeeded"
            or bool(reasons)
        )
        status = "INVALID" if invalid else "VALID_BUT_WITHHELD_BY_COHORT_INVALIDITY"
        rows.append({
            "selection_order": order, "sample_id": nns.sample_id, "status": status,
            "validity_status": "FAIL", "misclassified": None,
            "nns_insertion_auc": None, "nns_deletion_auc": None,
            "matched_insertion_auc": None, "matched_deletion_auc": None, "delta_fid": None,
            "nns_insertion_auc_repeat_sd_ddof_0": None,
            "nns_deletion_auc_repeat_sd_ddof_0": None,
            "matched_insertion_auc_repeat_sd_ddof_0": None,
            "matched_deletion_auc_repeat_sd_ddof_0": None,
        })
        if invalid:
            failures.append({"selection_order": order, "sample_id": nns.sample_id, "reasons": reasons})
    return rows, {"invalid_coordinate_count": len(failures), "invalid_coordinates": failures}


def aggregate_stage_d_analysis(
    receipt: StageDValidatedAnalysisInput, plan: Any,
) -> StageDAnalysisResult:
    """Aggregate an already D5-validated synthetic input exactly once.

    The type check intentionally makes a serialized external summary unusable
    as an input.  A caller must first obtain a fresh D5 receipt.
    """

    if not isinstance(receipt, StageDValidatedAnalysisInput):
        raise StageDAnalysisContractError("aggregate requires a D5-issued analysis receipt")
    typed_plan = validate_stage_d_plan(plan)
    if receipt.package_complete is not True or receipt.workflow_status != "PASS":
        raise StageDAnalysisContractError("aggregate requires a complete PASS-workflow D5 receipt")
    if receipt.plan_digest != typed_plan.canonical_digest:
        raise StageDAnalysisContractError("receipt plan digest differs from typed plan")
    if len(receipt.unit_rows) != 1000:
        raise StageDAnalysisContractError("receipt must contain all 1,000 reconciled units")

    common_provenance = {
        "synthetic_only": True, "formal_result": False,
        "d5_receipt_plan_digest": receipt.plan_digest,
        "d5_final_status_sha256": receipt.final_status_sha256,
        "d5_root_inventory": _plain(receipt.root_inventory),
        "execution_authorization": "NOT_ISSUED",
        "analysis_authorization": "NOT_EVALUATED",
        "stage_d_execution_started": False,
        "holm_status": "NOT_APPLICABLE_NO_SECONDARY_INFERENCE",
        "c10_stage_d_pooling_allowed": False,
    }
    reconciliation_qc = _reconciliation_qc(receipt)
    analysis_input_complete = all(
        row.terminal_state == "succeeded"
        and not row.scientific_errors
        and row.numeric_payload_status == "VALIDATED"
        for row in receipt.unit_rows
    )
    if receipt.scientific_validity != "PASS" or not analysis_input_complete:
        per_sample, qc = _invalid_rows(receipt)
        summary = {
            "metric_id": "fine_4x4_cell_delta_fid", "n": STATISTICAL_N,
            "positive_count": None, "mean": None, "median": None,
            "standard_deviation_sample_ddof_1": None,
            "bootstrap_ci_lower": None, "bootstrap_ci_upper": None,
        }
        cost = {"evaluated": False, "status": "NOT_EVALUATED_DUE_TO_VALIDITY", "gate_failures": []}
        return _issue_result(_self_document(
            validity_status="FAIL", fidelity_status="NOT_EVALUATED_DUE_TO_VALIDITY",
            cost_status="NOT_EVALUATED_DUE_TO_VALIDITY", overall_decision="NO_GO_VALIDITY",
            per_sample=per_sample, summary=summary, bootstrap=None, curves=[],
            c10_reference=_c10_reference(), resources_and_qc={
                **reconciliation_qc,
                "validity": qc, "cost": cost,
                "effect_outputs_status": "NOT_GENERATED_DUE_TO_VALIDITY",
            }, provenance=common_provenance,
        ))

    per_sample: list[dict[str, Any]] = []
    deltas = np.empty(STATISTICAL_N, dtype=np.float64)
    curve_values: dict[tuple[str, str], np.ndarray] = {
        (method, direction): np.empty((STATISTICAL_N, 21), dtype=np.float64)
        for method in METHOD_IDS for direction in ("insertion", "deletion")
    }
    for order in range(STATISTICAL_N):
        pair = receipt.unit_rows[order * 2:order * 2 + 2]
        if len(pair) != 2 or tuple(row.method_id for row in pair) != METHOD_IDS:
            raise StageDAnalysisContractError("receipt NNS/matched pair order drifted")
        nns, matched = pair
        if nns.selection_order != order or matched.selection_order != order or nns.sample_id != matched.sample_id:
            raise StageDAnalysisContractError("receipt image coordinate pairing drifted")
        if nns.pair_digest != matched.pair_digest or nns.group_digest != matched.group_digest:
            raise StageDAnalysisContractError("receipt pair/group binding drifted")
        nns_auc, nns_sd, nns_curves, nns_misclassified = _extract_unit(nns)
        matched_auc, matched_sd, matched_curves, matched_misclassified = _extract_unit(matched)
        if nns_misclassified != matched_misclassified:
            raise StageDAnalysisContractError("paired L0 prediction classification drifted")
        delta = 0.5 * (
            nns_auc["insertion"] - matched_auc["insertion"]
            + matched_auc["deletion"] - nns_auc["deletion"]
        )
        deltas[order] = delta
        per_sample.append({
            "selection_order": order, "sample_id": nns.sample_id, "status": "VALID",
            "validity_status": "PASS", "misclassified": nns_misclassified,
            "nns_insertion_auc": nns_auc["insertion"], "nns_deletion_auc": nns_auc["deletion"],
            "matched_insertion_auc": matched_auc["insertion"], "matched_deletion_auc": matched_auc["deletion"],
            "delta_fid": float(delta),
            "nns_insertion_auc_repeat_sd_ddof_0": nns_sd["insertion"],
            "nns_deletion_auc_repeat_sd_ddof_0": nns_sd["deletion"],
            "matched_insertion_auc_repeat_sd_ddof_0": matched_sd["insertion"],
            "matched_deletion_auc_repeat_sd_ddof_0": matched_sd["deletion"],
        })
        for unit, matrix in ((nns, nns_curves), (matched, matched_curves)):
            for direction_index, direction in enumerate(("insertion", "deletion")):
                curve_values[(unit.method_id, direction)][order] = matrix[direction_index].mean(axis=0)

    primary_bootstrap, _ = _bootstrap(deltas, PRIMARY_NAMESPACE + PRIMARY_METRIC_ID)
    summary = {
        "metric_id": "fine_4x4_cell_delta_fid", "n": STATISTICAL_N,
        "positive_count": int(np.count_nonzero(deltas > 0.0)), "mean": float(deltas.mean()),
        "median": _median(deltas), "standard_deviation_sample_ddof_1": float(deltas.std(ddof=1)),
        "bootstrap_ci_lower": primary_bootstrap["confidence_interval"][0],
        "bootstrap_ci_upper": primary_bootstrap["confidence_interval"][1],
    }
    cost = _campaign_cost(receipt)
    fidelity_status, cost_status, decision = _decision(summary, cost)
    curves: list[dict[str, Any]] = []
    figure_bootstrap: list[dict[str, Any]] = []
    for method in METHOD_IDS:
        for direction in ("insertion", "deletion"):
            values = curve_values[(method, direction)]
            for index, fraction in enumerate(CURVE_FRACTIONS):
                material = f"{FIGURE_NAMESPACE}{method}|{direction}|{index}"
                metric_id = f"{method}|{direction}|{index}"
                metadata, _ = _bootstrap(
                    values[:, index], material, metric_id=metric_id,
                )
                figure_bootstrap.append({
                    "method_id": method, "direction": direction,
                    "fraction_index": index, "metadata": metadata,
                })
                curves.append({
                    "method_id": method, "direction": direction, "fraction_index": index,
                    "fraction": float(fraction), "n": STATISTICAL_N,
                    "median": _median(values[:, index]),
                    "pointwise_ci_lower": metadata["confidence_interval"][0],
                    "pointwise_ci_upper": metadata["confidence_interval"][1],
                })
    return _issue_result(_self_document(
        validity_status="PASS", fidelity_status=fidelity_status, cost_status=cost_status,
        overall_decision=decision, per_sample=per_sample, summary=summary,
        bootstrap=primary_bootstrap, curves=curves, c10_reference=_c10_reference(),
        resources_and_qc={
            **reconciliation_qc,
            "validity": {"invalid_coordinate_count": 0, "invalid_coordinates": []},
            "cost": cost, "effect_outputs_status": "GENERATED",
            "descriptive_curve_bootstrap": {
                "namespace": FIGURE_NAMESPACE, "coordinates": figure_bootstrap,
                "simultaneous": False, "enters_primary_decision": False,
            },
        }, provenance=common_provenance,
    ))


def closed_analysis_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    """Return the CLI-safe control projection without any outcomes."""

    value = _plain(result, "analysis result")
    required = {
        "schema_version", "document_type", "validity_status", "fidelity_status",
        "cost_status", "overall_decision", "per_sample", "summary", "bootstrap",
        "curves", "c10_reference", "resources_and_qc", "provenance", "canonical_digest",
    }
    if set(value) != required or value.get("document_type") != DOCUMENT_TYPE:
        raise StageDAnalysisContractError("analysis result does not have a closed D6 shape")
    return {
        "schema_version": 1, "document_type": "stage_d_d6_control_summary",
        "status": "PASS", "synthetic_only": True, "formal_result": False,
        "analysis_canonical_digest": value["canonical_digest"],
        "coordinate_count": len(value["per_sample"]),
        "execution_authorization": "NOT_ISSUED",
        "analysis_authorization": "NOT_EVALUATED",
        "stage_d_execution_started": False,
    }


__all__ = [
    "BOOTSTRAP_RESAMPLES", "DOCUMENT_TYPE", "FIGURE_NAMESPACE",
    "PRIMARY_METRIC_ID", "PRIMARY_NAMESPACE", "StageDAnalysisContractError",
    "StageDAnalysisError", "StageDAnalysisResult", "aggregate_stage_d_analysis",
    "closed_analysis_summary",
]
