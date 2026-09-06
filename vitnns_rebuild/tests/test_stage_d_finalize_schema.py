"""Pure schema tests for the five D5 controls and validation result.

These tests deliberately do not construct a Stage D execution root.  The
representative documents are small, but include every object shape emitted by
``stage_d_finalize`` (including both retry and scientific-failure branches).
"""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import pytest

from vitnns.stage_d_finalize import (
    StageDFinalizationIntegrityError,
    _validate_final_document,
    _validate_root_validation_document,
)
from vitnns.stage_d_plan import (
    MATCHED_METHOD_ID,
    NNS_METHOD_ID,
    canonical_json_bytes,
    canonical_sha256,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _sign(value: dict[str, Any]) -> dict[str, Any]:
    value["canonical_digest"] = canonical_sha256({
        key: child for key, child in value.items()
        if key != "canonical_digest"
    })
    return value


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return canonical_json_bytes(value) + b"\n"


def _inventory_row(path: str, label: str, byte_count: int = 10) -> dict[str, Any]:
    return {
        "path": path, "bytes": byte_count, "sha256": _sha(label),
    }


def _resources(*, elapsed: float = 0.25) -> dict[str, int | float]:
    return {
        "elapsed_seconds": elapsed,
        "forward_calls": 1,
        "backward_calls": 1,
        "l0_l1_payload_bytes": 30,
        "l2_payload_bytes": 40,
        "control_and_provenance_bytes": 50,
        "cuda_peak_reserved_bytes": 0,
    }


def _artifact(
    *, path: str, level: str, kind: str, artifact_format: str,
    dtype: Any, shape: Any, label: str,
) -> dict[str, Any]:
    return {
        "level": level, "path": path, "kind": kind,
        "format": artifact_format, "dtype": dtype, "shape": shape,
        "bytes": 10, "sha256": _sha(label),
    }


def _pair_binding() -> dict[str, Any]:
    return _sign({
        "schema_version": 1,
        "document_type": "stage_d_pair_binding",
        "sample_id": "synthetic-000",
        "selection_order": 0,
        "image_sha256": _sha("image"),
        "model_context_digest": _sha("model-context"),
        "target_context_digest": _sha("target-context"),
        "fidelity_context_digest": _sha("fidelity-context"),
        "plan_context_digest": _sha("plan-context"),
    })


def _retry_authorization(
    unit_digest: str, prior_attempt_digest: str,
) -> dict[str, Any]:
    return _sign({
        "schema_version": 1,
        "document_type": "stage_d_runtime_retry_authorization",
        "status": "COORDINATOR_AUTHORIZED_RETRY",
        "unit_digest": unit_digest,
        "prior_attempt_digest": prior_attempt_digest,
        "failure_code": "SCHEDULER_PREEMPTION",
        "next_attempt_number": 2,
        "execution_context_digest": _sha("execution-context"),
    })


def _execution(
    *, unit_digest: str, attempt_number: int, retry: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "attempt": {
            "number": attempt_number,
            "id": f"attempt-{attempt_number:06d}",
        },
        "retry": dict(retry),
        "claim": {
            "owner_token_digest": _sha(
                f"owner-{unit_digest}-{attempt_number}"
            ),
            "group_owner_token_digest": _sha("group-owner"),
        },
        "shard": {"id": 0, "count": 1},
    }


def _provenance(
    *, unit_digest: str, attempt_number: int, retry: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "sha256": _sha(f"provenance-file-{unit_digest}-{attempt_number}"),
        "canonical_digest": _sha(
            f"provenance-document-{unit_digest}-{attempt_number}"
        ),
        "mode": "SYNTHETIC_D4_TEST_ONLY",
        "unit_identity_sha256": unit_digest,
        "bindings_digest": _sha(
            f"provenance-bindings-{unit_digest}-{attempt_number}"
        ),
        "execution": _execution(
            unit_digest=unit_digest, attempt_number=attempt_number, retry=retry,
        ),
    }


def _attempt(
    *, unit_digest: str, number: int, state: str, failure: Any,
    failure_class: str | None, failure_code: str | None,
    retry_digest: str | None, artifacts: list[dict[str, Any]],
    provenance: dict[str, Any] | None,
) -> dict[str, Any]:
    terminal = state in {"succeeded", "scientific_failed"}
    status = {
        "succeeded": "succeeded",
        "scientific_failed": "scientific_failure",
        "infrastructure_failed": "infrastructure_interruption",
    }[state]
    return {
        "attempt_number": number,
        "attempt_id": f"attempt-{number:06d}",
        "attempt_path": (
            f"units/{unit_digest}/attempts/attempt-{number:06d}"
        ),
        "owner_token_digest": _sha(f"owner-{unit_digest}-{number}"),
        "checkpoint_attempt_digest": _sha(
            f"checkpoint-attempt-{unit_digest}-{number}"
        ),
        "state": state,
        "status": status,
        "terminal": terminal,
        "complete": terminal,
        "status_sha256": _sha(f"status-{unit_digest}-{number}"),
        "manifest_sha256": _sha(f"manifest-{unit_digest}-{number}"),
        "status_manifest_sha256": _sha(
            f"manifest-{unit_digest}-{number}"
        ),
        "manifest_status": status,
        "manifest_terminal": terminal,
        "manifest_complete": terminal,
        "attempt_inventory_digest": _sha(
            f"inventory-{unit_digest}-{number}"
        ),
        "failure_class": failure_class,
        "failure_code": failure_code,
        "failure": failure,
        "retry_authorization_digest": retry_digest,
        "result_manifest_published": terminal,
        "published_l0_l1_payload_count": (
            0 if state != "succeeded" else 3
        ),
        "resources": _resources(),
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "provenance": provenance,
    }


def _sum_resources(*values: Mapping[str, int | float]) -> dict[str, int | float]:
    return {
        key: (
            max(value[key] for value in values)
            if key == "cuda_peak_reserved_bytes"
            else sum(value[key] for value in values)
        )
        for key in _resources()
    }


def _documents() -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    plan_digest = _sha("plan")
    group_digest = _sha("group")
    pair = _pair_binding()
    pair_digest = pair["canonical_digest"]
    retry_unit = _sha("retry-unit")
    failed_unit = _sha("failed-unit")
    infrastructure_failure = {
        "code": "SCHEDULER_PREEMPTION",
        "resources": _resources(),
    }
    first_attempt = _attempt(
        unit_digest=retry_unit, number=1, state="infrastructure_failed",
        failure=infrastructure_failure,
        failure_class="INFRASTRUCTURE_FAILURE",
        failure_code="SCHEDULER_PREEMPTION", retry_digest=None,
        artifacts=[], provenance=None,
    )
    retry = _retry_authorization(
        retry_unit, first_attempt["attempt_inventory_digest"],
    )
    success_artifacts = [
        _artifact(
            path="L0/result.json", level="L0", kind="result",
            artifact_format="json", dtype="json", shape=None,
            label="result",
        ),
        _artifact(
            path="L1/metric_ready_map.npy", level="L1",
            kind="metric_ready_map", artifact_format="npy",
            dtype="float32", shape=[56, 56], label="map",
        ),
        _artifact(
            path="L1/fidelity_curves.npz", level="L1",
            kind="fidelity_curves", artifact_format="npz",
            dtype="float64", shape=[2, 5, 21], label="curves",
        ),
        _artifact(
            path="L2/audit_evidence.npz", level="L2",
            kind="audit_evidence", artifact_format="npz",
            dtype={
                "raw_attribution": "float32",
                "repeat_clean_metric_ready_map": "float32",
            },
            shape={
                "raw_attribution": [1],
                "repeat_clean_metric_ready_map": [56, 56],
            },
            label="audit",
        ),
        _artifact(
            path="L0/provenance.json", level="L0",
            kind="provenance", artifact_format="json", dtype="json",
            shape=None, label="provenance-success",
        ),
    ]
    success_provenance = _provenance(
        unit_digest=retry_unit, attempt_number=2, retry=retry,
    )
    success_artifacts[-1]["sha256"] = success_provenance["sha256"]
    second_attempt = _attempt(
        unit_digest=retry_unit, number=2, state="succeeded", failure=None,
        failure_class=None, failure_code=None,
        retry_digest=retry["canonical_digest"],
        artifacts=success_artifacts, provenance=success_provenance,
    )

    scientific_failure = {
        "code": "degenerate_direction",
        "phase": "attribution",
        "source": "stage_d_scientific_execution",
        "produced": [],
        "absent": [
            "L0/result.json",
            "L1/metric_ready_map.npy",
            "L1/fidelity_curves.npz",
            "L2/audit_evidence.npz",
        ],
        "details_present": True,
        "details_sha256": _sha("scientific-failure-details"),
    }
    failure_artifacts = [
        _artifact(
            path="L0/failure.json", level="L0",
            kind="scientific_failure", artifact_format="json",
            dtype="json", shape=None, label="scientific-failure",
        ),
        _artifact(
            path="L0/provenance.json", level="L0",
            kind="provenance", artifact_format="json", dtype="json",
            shape=None, label="provenance-failure",
        ),
    ]
    failure_provenance = _provenance(
        unit_digest=failed_unit, attempt_number=1,
        retry={"status": "NOT_A_RETRY"},
    )
    failure_artifacts[-1]["sha256"] = failure_provenance["sha256"]
    failed_attempt = _attempt(
        unit_digest=failed_unit, number=1, state="scientific_failed",
        failure=scientific_failure, failure_class="SCIENTIFIC_INVALIDITY",
        failure_code="degenerate_direction", retry_digest=None,
        artifacts=failure_artifacts, provenance=failure_provenance,
    )

    rows = [
        {
            "selection_order": 0, "sample_id": "synthetic-000",
            "group_digest": group_digest, "pair_digest": pair_digest,
            "unit_digest": retry_unit, "method_index": 0,
            "method_id": NNS_METHOD_ID, "audit": True,
            "terminal_state": "succeeded", "terminal": True,
            "failure_class": None, "failure_code": None,
            "retry_authorization_digest": retry["canonical_digest"],
            "resources": _sum_resources(
                first_attempt["resources"], second_attempt["resources"],
            ),
            "attempts": [first_attempt, second_attempt],
            "scientific_errors": [],
        },
        {
            "selection_order": 0, "sample_id": "synthetic-000",
            "group_digest": group_digest, "pair_digest": pair_digest,
            "unit_digest": failed_unit, "method_index": 1,
            "method_id": MATCHED_METHOD_ID, "audit": True,
            "terminal_state": "scientific_failed", "terminal": True,
            "failure_class": "SCIENTIFIC_INVALIDITY",
            "failure_code": "degenerate_direction",
            "retry_authorization_digest": None,
            "resources": _resources(), "attempts": [failed_attempt],
            "scientific_errors": ["degenerate_direction"],
        },
    ]
    group_rows = [{
        "selection_order": 0, "sample_id": "synthetic-000",
        "relative_path": "ILSVRC2012_val_00000001.JPEG",
        "image_sha256": _sha("image"),
        "registry_row_digest": _sha("registry-row"),
        "audit": True, "audit_overlay_id": "audit-000",
        "group_digest": group_digest, "pair_binding": pair,
        "unit_digests": [retry_unit, failed_unit],
    }]

    expected_counts = {
        "registry_rows": 500, "image_groups": 500,
        "execution_units": 1000, "primary_clean_maps": 1000,
        "audit_repeat_clean_maps": 10, "map_work_items": 1010,
        "fidelity_curves": 10000, "fidelity_curve_points": 210000,
        "logical_work_items": 211010, "attempts": 1001,
        "attempt_statuses": 1001, "attempt_manifests": 1001,
        "provenance_records": 1000, "artifact_records": 4007,
        "source_files": 1,
    }
    observed_counts = {
        **expected_counts,
        "primary_clean_maps": 999, "audit_repeat_clean_maps": 9,
        "map_work_items": 1008,
        "fidelity_curves": 9990, "fidelity_curve_points": 209790,
        "logical_work_items": 210798,
    }
    level_counts = {"L0": 2000, "L1": 1998, "L2": 9, "L3": 0}
    level_coverage = {"L0": 1000, "L1": 999, "L2": 9, "L3": 0}
    reconciliation = _sign({
        "schema_version": 1,
        "document_type": "stage_d_reconciliation",
        "plan_digest": plan_digest,
        "expected_image_count": 500, "expected_unit_count": 1000,
        "image_count": 500, "unit_count": 1000,
        "expected_counts": expected_counts,
        "observed_counts": observed_counts,
        "state_counts": {"scientific_failed": 1, "succeeded": 999},
        "group_rows": group_rows, "rows": rows,
        "failure_rows": [
            {
                "unit_digest": retry_unit, "group_selection_order": 0,
                "method_index": 0, "state": "succeeded",
                "terminal": True, "attempt_count": 2,
                "failure_class": None, "failure_code": None,
            },
            {
                "unit_digest": failed_unit, "group_selection_order": 0,
                "method_index": 1, "state": "scientific_failed",
                "terminal": True, "attempt_count": 1,
                "failure_class": "SCIENTIFIC_INVALIDITY",
                "failure_code": "degenerate_direction",
            },
        ],
        "na_rows": [{
            "unit_digest": failed_unit, "reason": "degenerate_direction",
        }],
        "resources": _sum_resources(*(row["resources"] for row in rows)),
        "retention": {
            "expected": {
                "L0_unit_coverage": 1000, "L1_unit_coverage": 1000,
                "L2_unit_coverage": 10, "L3_file_count": 0,
            },
            "observed": {
                "L0_unit_coverage": 1000, "L1_unit_coverage": 999,
                "L2_unit_coverage": 9, "L3_file_count": 0,
                "files_by_level": level_counts,
                "file_unit_coverage": level_coverage,
            },
            "audit_image_count": 5, "audit_unit_count": 10,
        },
        "checkpoint_failure_table_reconciled": True,
        "checkpoint_resources_reconciled": True,
        "no_rows_dropped": True,
        "zero_imputation_performed": False,
    })

    source_row = _inventory_row("runtime_checkpoint.json", "checkpoint")
    source_inventory = _sign({
        "schema_version": 1,
        "document_type": "stage_d_source_inventory",
        "root_scope": "entire_stage_d_root_excluding_final_subtree",
        "files": [source_row], "file_count": 1,
        "total_bytes": source_row["bytes"],
    })
    evidence = _sign({
        "schema_version": 1,
        "document_type": "stage_d_final_evidence_report",
        "package_complete": True, "workflow_status": "PASS",
        "scientific_validity": "FAIL", "analysis_authorized": False,
        "plan_digest": plan_digest,
        "estimate_digest": _sha("estimate"),
        "preflight_digest": _sha("preflight"),
        "execution_context_digest": _sha("execution-context"),
        "counts": {
            "images": 500, "groups": 500, "units": 1000,
            "attempts": 1001, "source_files": 1,
            "scientific_failures": 1,
        },
        "scientific_failures": [{
            "unit_digest": failed_unit,
            "reasons": ["degenerate_direction"],
        }],
        "authorization": {
            "formal": False, "execution_authorization": "NOT_ISSUED",
            "authorization_scope": "SYNTHETIC_D4_TEST_ONLY",
            "analysis_authorization": "NOT_EVALUATED",
            "stage_d_execution_started": False,
        },
        "analysis_boundary": {
            "status": "NOT_PERFORMED", "analysis_performed": False,
            "bootstrap_performed": False, "decision_performed": False,
            "delta_fid_computed": False,
            "aggregate_effects_computed": False,
            "d6_eligibility": "SYNTHETIC_ONLY",
        },
        "actual_cost_audit": {
            "evaluated": True,
            "actual_cost_includes_all_attempts_and_audit_repeats": True,
            "projection_source": (
                "D9_RECOMPUTED_FROM_VALIDATED_ATTEMPT_RESOURCES_AND_ROOT_INVENTORY"
            ),
            "resource_totals": reconciliation["resources"],
            "limits": {
                "stage_d_total_retained_regular_file_bytes_max": 68719476736,
                "stage_d_total_h20_gpu_hours_max": 2000.0,
                "stage_d_h20_gpu_hours_per_image_group_max": 4.0,
            },
            "total_h20_gpu_hours": (
                reconciliation["resources"]["elapsed_seconds"] / 3600.0
            ),
            "h20_gpu_hours_per_image_group_max": 0.0,
            "total_retained_regular_file_bytes": source_row["bytes"],
            "source_file_count": 1,
            "gate_failures": [],
            "status": "PASS_COST",
        },
        "analysis_performed": False, "bootstrap_performed": False,
        "decision_performed": False,
        "reconciliation_digest": reconciliation["canonical_digest"],
        "source_inventory_digest": source_inventory["canonical_digest"],
    })

    first_three = {
        "evidence_report.json": evidence,
        "reconciliation.json": reconciliation,
        "source_inventory.json": source_inventory,
    }
    payload_rows = [
        {
            "path": name, "bytes": len(_json_bytes(document)),
            "sha256": hashlib.sha256(_json_bytes(document)).hexdigest(),
        }
        for name, document in first_three.items()
    ]
    payload_manifest = _sign({
        "schema_version": 1,
        "document_type": "stage_d_final_payload_manifest",
        "immutable": True, "artifact_count": 3,
        "artifacts": payload_rows,
    })
    documents = {
        **first_three, "payload_manifest.json": payload_manifest,
    }

    sealed_files = sorted(
        [source_row, *(
            {
                "path": f"final/{name}",
                "bytes": len(_json_bytes(document)),
                "sha256": hashlib.sha256(_json_bytes(document)).hexdigest(),
            }
            for name, document in documents.items()
        )],
        key=lambda item: item["path"],
    )
    sealed_unsigned = {
        "scope": "whole_root_excluding_status_self",
        "ordering": "path_lexicographic",
        "excluded_paths": ["final/status.json"],
        "files": sealed_files, "file_count": len(sealed_files),
        "total_bytes": sum(item["bytes"] for item in sealed_files),
    }
    sealed_inventory = {
        **sealed_unsigned,
        "canonical_digest": canonical_sha256(sealed_unsigned),
    }
    status = _sign({
        "schema_version": 1,
        "document_type": "stage_d_final_status",
        "terminal": True, "package_complete": True,
        "scientific_validity": "FAIL", "workflow_status": "PASS",
        "plan_digest": plan_digest,
        "payload_manifest_sha256": hashlib.sha256(
            _json_bytes(payload_manifest)
        ).hexdigest(),
        "sealed_root_inventory": sealed_inventory,
    })
    documents["status.json"] = status

    whole_inventory = sorted(
        [
            *sealed_files,
            {
                "path": "final/status.json",
                "bytes": len(_json_bytes(status)),
                "sha256": hashlib.sha256(_json_bytes(status)).hexdigest(),
            },
        ],
        key=lambda item: item["path"],
    )
    root_validation = {
        "package_complete": True, "workflow_status": "PASS",
        "scientific_validity": "FAIL", "analysis_authorized": False,
        "plan_digest": plan_digest,
        "final_status_sha256": hashlib.sha256(
            _json_bytes(status)
        ).hexdigest(),
        "root_inventory": {
            "file_count": len(whole_inventory),
            "total_bytes": sum(item["bytes"] for item in whole_inventory),
            "canonical_digest": canonical_sha256({
                "files": whole_inventory,
            }),
        },
        "actual_cost_audit": evidence["actual_cost_audit"],
        "evidence_report": evidence,
        "reconciliation": reconciliation,
        "source_inventory": source_inventory,
        "payload_manifest": payload_manifest,
        "whole_root_ordered_inventory": whole_inventory,
    }
    return documents, root_validation


@pytest.fixture(scope="module")
def schema_values() -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    return _documents()


def _at(value: Any, path: tuple[str | int, ...]) -> Any:
    current = value
    for part in path:
        current = current[part]
    return current


def _refresh_document_digest(name: str, document: dict[str, Any]) -> None:
    if name == "status.json":
        _sign(document["sealed_root_inventory"])
    _sign(document)


@dataclass(frozen=True)
class Family:
    label: str
    document: str
    path: tuple[str | int, ...]
    field: str
    wrong_type: Any
    invalid_value: Any


FAMILIES = (
    Family("evidence-root", "evidence_report.json", (), "package_complete", 1, False),
    Family("evidence-counts", "evidence_report.json", ("counts",), "images", True, -1),
    Family("scientific-failure-row", "evidence_report.json", ("scientific_failures", 0), "unit_digest", 7, "g" * 64),
    Family("scientific-failure-reasons", "evidence_report.json", ("scientific_failures", 0), "reasons", "degenerate_direction", [""]),
    Family("authorization", "evidence_report.json", ("authorization",), "execution_authorization", False, "AUTHORIZED_STAGE_D_FIDELITY_SCOPE"),
    Family("analysis-boundary", "evidence_report.json", ("analysis_boundary",), "status", 0, "PERFORMED"),
    Family("reconciliation-root", "reconciliation.json", (), "expected_image_count", True, -1),
    Family("expected-counts", "reconciliation.json", ("expected_counts",), "registry_rows", True, -1),
    Family("observed-counts", "reconciliation.json", ("observed_counts",), "registry_rows", True, -1),
    Family("state-counts", "reconciliation.json", ("state_counts",), "scientific_failed", True, -1),
    Family("group-row", "reconciliation.json", ("group_rows", 0), "selection_order", True, -1),
    Family("pair-binding", "reconciliation.json", ("group_rows", 0, "pair_binding"), "document_type", 7, "foreign_pair_binding"),
    Family("unit-digest-list", "reconciliation.json", ("group_rows", 0), "unit_digests", "digest", [_sha("only-one-unit")]),
    Family("unit-row", "reconciliation.json", ("rows", 0), "method_id", 7, "foreign_method"),
    Family("scientific-errors", "reconciliation.json", ("rows", 1), "scientific_errors", "degenerate_direction", [""]),
    Family("resources", "reconciliation.json", ("rows", 0, "resources"), "elapsed_seconds", "0.5", -0.5),
    Family("attempt", "reconciliation.json", ("rows", 0, "attempts", 1), "attempt_number", True, 3),
    Family("infrastructure-failure", "reconciliation.json", ("rows", 0, "attempts", 0, "failure"), "code", 7, "UNKNOWN_FAILURE"),
    Family("scientific-failure", "reconciliation.json", ("rows", 1, "attempts", 0, "failure"), "phase", 7, ""),
    Family("scientific-failure-paths", "reconciliation.json", ("rows", 1, "attempts", 0, "failure"), "produced", "L0/result.json", ["L3/forbidden.dat"]),
    Family("scientific-failure-details-presence", "reconciliation.json", ("rows", 1, "attempts", 0, "failure"), "details_present", 1, False),
    Family("scientific-failure-details-digest", "reconciliation.json", ("rows", 1, "attempts", 0, "failure"), "details_sha256", 7, "g" * 64),
    Family("artifact", "reconciliation.json", ("rows", 0, "attempts", 1, "artifacts", 0), "level", 0, "L3"),
    Family("audit-artifact-dtype", "reconciliation.json", ("rows", 0, "attempts", 1, "artifacts", 3, "dtype"), "repeat_clean_metric_ready_map", 7, "float64"),
    Family("audit-artifact-shape", "reconciliation.json", ("rows", 0, "attempts", 1, "artifacts", 3, "shape"), "repeat_clean_metric_ready_map", 7, [14, 14]),
    Family("attempt-resources", "reconciliation.json", ("rows", 0, "attempts", 1, "resources"), "forward_calls", True, -1),
    Family("provenance-projection", "reconciliation.json", ("rows", 0, "attempts", 1, "provenance"), "mode", 0, "unknown-formal"),
    Family("provenance-execution", "reconciliation.json", ("rows", 0, "attempts", 1, "provenance", "execution"), "attempt", [], {"number": 3, "id": "attempt-000003"}),
    Family("attempt-binding", "reconciliation.json", ("rows", 0, "attempts", 1, "provenance", "execution", "attempt"), "number", True, 3),
    Family("retry-authorization", "reconciliation.json", ("rows", 0, "attempts", 1, "provenance", "execution", "retry"), "status", 7, "RETRY_PENDING"),
    Family("not-a-retry", "reconciliation.json", ("rows", 1, "attempts", 0, "provenance", "execution", "retry"), "status", 7, "RETRY_PENDING"),
    Family("claim-binding", "reconciliation.json", ("rows", 0, "attempts", 1, "provenance", "execution", "claim"), "owner_token_digest", 7, "g" * 64),
    Family("shard-binding", "reconciliation.json", ("rows", 0, "attempts", 1, "provenance", "execution", "shard"), "count", True, 0),
    Family("failure-row", "reconciliation.json", ("failure_rows", 1), "state", 7, "pending"),
    Family("na-row", "reconciliation.json", ("na_rows", 0), "reason", 7, "UNKNOWN_FAILURE"),
    Family("campaign-resources", "reconciliation.json", ("resources",), "forward_calls", True, -1),
    Family("retention", "reconciliation.json", ("retention",), "audit_image_count", True, -1),
    Family("retention-expected", "reconciliation.json", ("retention", "expected"), "L0_unit_coverage", True, -1),
    Family("retention-observed", "reconciliation.json", ("retention", "observed"), "L0_unit_coverage", True, -1),
    Family("retained-files-by-level", "reconciliation.json", ("retention", "observed", "files_by_level"), "L0", True, -1),
    Family("retained-unit-coverage", "reconciliation.json", ("retention", "observed", "file_unit_coverage"), "L0", True, -1),
    Family("source-inventory-root", "source_inventory.json", (), "root_scope", 7, "wrong_scope"),
    Family("source-inventory-row", "source_inventory.json", ("files", 0), "bytes", True, -1),
    Family("payload-manifest-root", "payload_manifest.json", (), "immutable", 1, False),
    Family("payload-artifact", "payload_manifest.json", ("artifacts", 0), "path", 7, "../evidence_report.json"),
    Family("status-root", "status.json", (), "terminal", 1, False),
    Family("sealed-inventory", "status.json", ("sealed_root_inventory",), "scope", 7, "wrong_scope"),
    Family("sealed-inventory-row", "status.json", ("sealed_root_inventory", "files", 0), "bytes", True, -1),
)


MUTATIONS: dict[str, Callable[[dict[str, Any], Family], None]] = {
    "unknown": lambda target, _family: target.__setitem__("unexpected", True),
    "missing": lambda target, family: target.pop(family.field),
    "wrong-type": lambda target, family: target.__setitem__(
        family.field, copy.deepcopy(family.wrong_type)
    ),
    "invalid-value": lambda target, family: target.__setitem__(
        family.field, copy.deepcopy(family.invalid_value)
    ),
}


def test_all_five_representative_final_documents_are_valid(
    schema_values: tuple[dict[str, dict[str, Any]], dict[str, Any]],
) -> None:
    documents, _root_validation = schema_values
    assert set(documents) == {
        "evidence_report.json", "reconciliation.json",
        "source_inventory.json", "payload_manifest.json", "status.json",
    }
    for name, document in documents.items():
        _validate_final_document(name, copy.deepcopy(document))


@pytest.mark.parametrize("mutation_name", MUTATIONS, ids=MUTATIONS)
@pytest.mark.parametrize("family", FAMILIES, ids=lambda item: item.label)
def test_final_documents_reject_self_resigned_nested_schema_mutations(
    schema_values: tuple[dict[str, dict[str, Any]], dict[str, Any]],
    family: Family, mutation_name: str,
) -> None:
    documents, _root_validation = schema_values
    document = copy.deepcopy(documents[family.document])
    target = _at(document, family.path)
    assert isinstance(target, dict)
    if family.label == "state-counts" and mutation_name == "missing":
        pytest.skip("terminal state keys are data-dependent, not required fields")
    MUTATIONS[mutation_name](target, family)
    if "canonical_digest" in target:
        _sign(target)
    _refresh_document_digest(family.document, document)

    with pytest.raises(StageDFinalizationIntegrityError):
        _validate_final_document(family.document, document)


ROOT_FAMILIES = (
    Family("root-validation", "", (), "package_complete", 1, False),
    Family("root-inventory-summary", "", ("root_inventory",), "file_count", True, -1),
    Family("whole-root-inventory-row", "", ("whole_root_ordered_inventory", 0), "bytes", True, -1),
)


def test_representative_root_validation_document_is_valid(
    schema_values: tuple[dict[str, dict[str, Any]], dict[str, Any]],
) -> None:
    _documents_by_name, root_validation = schema_values
    _validate_root_validation_document(copy.deepcopy(root_validation))


@pytest.mark.parametrize("mutation_name", MUTATIONS, ids=MUTATIONS)
@pytest.mark.parametrize("family", ROOT_FAMILIES, ids=lambda item: item.label)
def test_root_validation_rejects_nested_schema_mutations(
    schema_values: tuple[dict[str, dict[str, Any]], dict[str, Any]],
    family: Family, mutation_name: str,
) -> None:
    _documents_by_name, pristine = schema_values
    root_validation = copy.deepcopy(pristine)
    target = _at(root_validation, family.path)
    assert isinstance(target, dict)
    MUTATIONS[mutation_name](target, family)

    with pytest.raises(StageDFinalizationIntegrityError):
        _validate_root_validation_document(root_validation)


def test_root_validation_recursively_validates_embedded_final_documents(
    schema_values: tuple[dict[str, dict[str, Any]], dict[str, Any]],
) -> None:
    _documents_by_name, pristine = schema_values
    root_validation = copy.deepcopy(pristine)
    root_validation["evidence_report"]["authorization"]["unexpected"] = True
    _sign(root_validation["evidence_report"])

    with pytest.raises(StageDFinalizationIntegrityError):
        _validate_root_validation_document(root_validation)


@pytest.mark.parametrize(
    "name, expected_type",
    [
        ("unknown.json", "stage_d_final_evidence_report"),
        ("evidence_report.json", "stage_d_reconciliation"),
    ],
)
def test_final_document_rejects_unknown_name_or_mismatched_identity(
    schema_values: tuple[dict[str, dict[str, Any]], dict[str, Any]],
    name: str, expected_type: str,
) -> None:
    documents, _root_validation = schema_values
    document = copy.deepcopy(documents["evidence_report.json"])
    document["document_type"] = expected_type
    _sign(document)

    with pytest.raises(StageDFinalizationIntegrityError):
        _validate_final_document(name, document)
