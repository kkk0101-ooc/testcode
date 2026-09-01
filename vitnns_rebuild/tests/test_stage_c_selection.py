from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

import vitnns.stage_c_selection as selection
from vitnns.stage_c_fidelity import (
    MATCHED_METHOD_ID, NNS_METHOD_ID, CurveDirection, FidelityCurveResult,
    FidelityScale, RankingKind, aggregate_five_curves,
    actual_curve_fractions, curve_unit_counts, fidelity_seed,
    tie_priority_sha256, trapezoid_auc,
)
from vitnns.stage_c_plan import CALIBRATION_SAMPLE_IDS, CLASS_METHOD_IDS, VALUE_METHOD_IDS


HASH = "a" * 64


@pytest.fixture(autouse=True)
def _accept_synthetic_provenance(monkeypatch):
    monkeypatch.setattr(selection, "validate_stage_c_provenance", lambda *_a, **_k: None)
    monkeypatch.setattr(
        selection, "_validate_input_hashes",
        lambda value, manifest, code, records: dict(value),
    )


def _curve(method: str, sample: str, block: int, head: int, scale: FidelityScale,
           direction: CurveDirection, replicate: int, auc_hint: float) -> FidelityCurveResult:
    count = scale.unit_count
    seed = fidelity_seed(sample, block, head, replicate)
    values = tuple(
        float(index / 20 if direction is CurveDirection.INSERTION else 1 - index / 20)
        for index in range(21)
    )
    # A small interior offset changes AUC while preserving mathematical endpoints.
    values = tuple(
        value if index in {0, 20} else value + auc_hint
        for index, value in enumerate(values)
    )
    fractions = actual_curve_fractions(count)
    ranking_hash = hashlib.sha256(f"{method}:{scale.value}:{replicate}".encode()).hexdigest()
    return FidelityCurveResult(
        method_id=method, comparison_line="value_path",
        target_ref="fixed_clean_attention_cls_head_direction",
        target_identity_sha256="1" * 64, scale=scale, direction=direction,
        ranking_kind=RankingKind.TARGET_ORIENTED_POSITIVE, sample_id=sample,
        block=block, head=head, replicate=replicate, seed=seed,
        tie_priority_sha256=tie_priority_sha256(count, seed),
        source_unit_map_sha256="2" * 64, score_sha256="3" * 64,
        input_pair_sha256="4" * 64, ranking_sha256=ranking_hash,
        intervention_sha256=tuple(f"{index:064x}" for index in range(21)),
        unit_counts=curve_unit_counts(count), actual_fractions=fractions,
        responses=values, auc=trapezoid_auc(fractions, values),
    )


def _fidelity_payload(method: str, sample: str, block: int, head: int,
                      *, advantage: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    curves: list[FidelityCurveResult] = []
    aggregates = []
    for scale in (FidelityScale.PATCH_16, FidelityScale.FINE_4):
        for direction in (CurveDirection.DELETION, CurveDirection.INSERTION):
            offset = 0.0
            if scale is FidelityScale.FINE_4:
                if method == NNS_METHOD_ID and direction is CurveDirection.INSERTION:
                    offset = advantage
                if method == NNS_METHOD_ID and direction is CurveDirection.DELETION:
                    offset = -advantage
            group = tuple(
                _curve(method, sample, block, head, scale, direction, repeat, offset)
                for repeat in range(5)
            )
            curves.extend(group)
            aggregates.append(aggregate_five_curves(group).to_dict())
            for repeat in range(5):
                seeded = group[repeat]
                curves.append(FidelityCurveResult(
                    **{**seeded.__dict__,
                       "ranking_kind": RankingKind.GLOBAL_RANDOM,
                       "source_unit_map_sha256": None, "score_sha256": None,
                       "ranking_sha256": hashlib.sha256(f"random:{scale.value}:{repeat}".encode()).hexdigest()}
                ))
            center = group[0]
            curves.append(FidelityCurveResult(
                **{**center.__dict__, "ranking_kind": RankingKind.CENTER_DISTANCE_ROW_MAJOR,
                   "replicate": None, "seed": None, "tie_priority_sha256": None,
                   "source_unit_map_sha256": None, "score_sha256": None,
                   "ranking_sha256": hashlib.sha256(f"center:{scale.value}".encode()).hexdigest()}
            ))
    return [item.to_dict() for item in curves], aggregates


def _metric(value: float | None) -> dict[str, Any]:
    if value is None:
        return {"status": "NA", "value": None, "reason": "constant_left"}
    return {"status": "OK", "value": value, "details": {}}


def _target(block: int, head: int) -> dict[str, Any]:
    return {
        "comparison_line": "value_path",
        "target_ref": "fixed_clean_attention_cls_head_direction",
        "clean_input_sha256": "5" * 64,
        "frozen_target": {
            "target_kind": "fixed_clean_attention_cls_head_direction",
            "candidate": {"block_index": block, "head_index": head},
            "status": "OK", "direction_norm": 1.0,
            "direction_threshold": 1e-6,
            "direction_unit": {"finite": True, "sha256": "6" * 64},
            "clean_input_sha256": "5" * 64,
        },
    }


def _maps(method: str, *, audit: bool, ig: bool) -> list[dict[str, Any]]:
    result = []
    variants = [("clean", 0), *((item, None) for item in selection.PERTURBATION_IDS)]
    if audit:
        variants.append(("repeat_clean", 1))
    for variant, repeat in variants:
        ig_result = None
        if ig:
            ig_result = {
                "selected_steps": 512, "converged": True,
                "attempts": [{"steps": 512, "delta": [1.0], "sum": [1.0],
                              "abs": [0.0], "rel": [0.0], "pass": [True]}],
                "metadata": {"step_chunk_size": 8},
            }
        result.append({
            "method_id": method, "input_variant": variant,
            "repeat_index": repeat, "status": "completed",
            "ig_result": ig_result, "attempted_steps_total": 512 if ig else 0,
            "backward_invocations": 64 if ig else 0,
            "actual_calls": {"accounting_unit": "batched_model_invocation",
                             "forward_calls": (68 if ig else (0 if variant in {"clean", "repeat_clean"} else (1 if method == VALUE_METHOD_IDS[0] else 2))),
                             "backward_calls": 64 if ig else 0},
            "exception": None,
        })
    return result


def _inventory() -> list[dict[str, Any]]:
    return [
        {"level": "L1", "path": f"L1/{name}.npy", "sha256": HASH,
         "bytes": 1, "kind": "metric_ready", "dtype": "float32",
         "shape": [14, 14], "format": "npy"}
        for name in ("clean", *selection.PERTURBATION_IDS)
    ]


def _telemetry() -> dict[str, Any]:
    phases = {name: {"wall_time_seconds": 0.0, "invocations": 1} for name in (
        "attribution_or_map_compute", "fidelity_compute",
        "integrity_checking", "artifact_finalize",
    )}
    return {
        "schema_version": 1,
        "phase_timings": {
            "clock": "perf_counter",
            "synchronization": "device_before_and_after_each_phase",
            "phases": phases, "synchronized_total_wall_time_seconds": 1.0,
            "total_boundary": "unit_start_through_first_complete_validation_excluding_telemetry_publication",
            "phases_are_mutually_exclusive": False,
        },
        "lifecycle": {
            "integrity": {
                "schema_version": 1, "algorithm": "stage_c_strong_model_sha256_v1",
                "checked": ["model"], "entry_sha256": HASH, "exit_sha256": HASH,
                "matched": True, "restored": True, "differences": [],
                "signature_calls": 2, "verification_error_type": None,
                "verification_error_message": None, "body_exception_type": None,
            },
            "restoration": {"schema_version": 1, "checked": ["model"],
                            "changes_detected": False, "restored": True,
                            "differences": [], "errors": []},
        },
        "measurement_environment": {"device_type": "cuda", "device": "cuda:0",
                                    "device_name": "NVIDIA H20",
                                    "cuda_runtime_version": "12.1", "cudnn_version": 8902},
    }


def _record(index: int, sample: str, method: str, block: int | None, head: int | None, *, full: bool) -> dict[str, Any]:
    key = {
        "schema_version": 1, "sample_id": sample, "method_id": method,
        "config_id": f"block_{block:02d}_head_{head:02d}" if block is not None else f"global_{method}",
        "block_index": block, "head_index": head,
        "target_ref": None if method == VALUE_METHOD_IDS[0] else (
            "fixed_clean_attention_cls_head_direction" if method in VALUE_METHOD_IDS else "clean_predicted_class_raw_logit"
        ),
    }
    key["canonical_digest"] = hashlib.sha256(
        json.dumps(key, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    audit = sample in selection.AUDIT_SAMPLE_IDS
    scalars: dict[str, Any] = {"sample_id": sample, "method_id": method}
    map_count = 5 if audit else 4
    targeted = method in {NNS_METHOD_ID, MATCHED_METHOD_ID}
    work_counts = {
        "map_runs": map_count, "clean_map_runs": 1,
        "perturbation_map_runs": 3,
        "repeatability_map_runs": int(audit),
        "fidelity_curves": 44 if targeted else 0,
        "fidelity_curve_points": 924 if targeted else 0,
        "target_tie_curve_points": 420 if targeted else 0,
        "random_sanity_curve_points": 420 if targeted else 0,
        "center_sanity_curve_points": 84 if targeted else 0,
        "logical_scalar_evaluations": 924 if targeted else 0,
    }
    work_counts["work_items"] = map_count + work_counts["fidelity_curve_points"]
    if method in VALUE_METHOD_IDS:
        scalars["map_execution_records"] = _maps(method, audit=audit, ig=method == NNS_METHOD_ID)
        scalars["fidelity_enabled"] = targeted
        scalars["fidelity_curves"] = []
        scalars["target_identity"] = None
        if method != VALUE_METHOD_IDS[0] and full:
            curves, aggregates = _fidelity_payload(method, sample, block, head, advantage=(0.001 * (12 - block) + 0.00001 * (12 - head)))
            scalars.update({"fidelity_curves": curves, "fidelity_aggregates": aggregates,
                            "top20": [{} for _ in range(10)],
                            "target_identity": _target(block, head)})
        if method == NNS_METHOD_ID and full:
            parity = {
                "threshold": 1e-6, "comparison": "strict_less_than",
                "clean_formula_max_abs_error": 0.0,
                "clean_natural_vs_fixed_replay_max_abs_error": 0.0,
                "clean_natural_vs_fixed_replay_model_output_max_abs_error": 0.0,
                "baseline_fixed_replay_vs_clean_A_matmul_natural_V0_max_abs_error": 0.0,
            }
            scalars["provenance_fragment"] = {"endpoint_parity": parity}
            scalars["residual_stability"] = [
                {"perturbation_id": perturbation,
                 "clean": {"status": "OK", "rho": _metric(0.2),
                            "method_id": NNS_METHOD_ID, "method_version": "1.0.0",
                            "comparison_line": "value_path", "target_identity": _target(block, head),
                            "details": {"formula": "R(M)=M-U(P(M));rho=l1(R)/l1(M)",
                                        "numerator_l1": 0.2, "denominator_l1": 1.0},
                            "perturbation_id": perturbation, "perturbation_tensor_sha256": "7" * 64},
                 "aligned_perturbed": {"status": "OK",
                                        "method_id": NNS_METHOD_ID, "method_version": "1.0.0",
                                        "comparison_line": "value_path", "target_identity": _target(block, head),
                                        "perturbation_id": perturbation, "perturbation_tensor_sha256": "7" * 64},
                 "signed_spearman": _metric(0.6)}
                for perturbation in selection.PERTURBATION_IDS
            ]
        elif method == MATCHED_METHOD_ID and full:
            scalars["provenance_fragment"] = {"matched_contribution": {
                "status": "OK", "token_contribution_count": 197,
                "token_contributions_sha256": "8" * 64,
                "patch_contributions_sha256": "9" * 64,
                "token_sum": 1.0, "scalar_delta_f_z_minus_f_x0": 1.0,
                "conservation_abs_error": 0.0,
                "conservation_tolerance": 1.1e-5,
            }}
    hashes = {name: hashlib.sha256(f"{index}:{name}".encode()).hexdigest() for name in selection._HASH_KEYS}
    hashes["selected_status_sha256"] = hashes["status_sha256"]
    return {
        "plan_index": index, "unit_key": key, "state": "succeeded",
        "trusted_by_batch_validator": True, "hashes": hashes,
        "scalars": scalars, "provenance": {"unit_identity": key},
        "resources": {"status": "completed", "calls": {"accounting_unit": "batched_model_invocation", "forward_calls": 10000, "backward_calls": 64 * map_count if method == NNS_METHOD_ID else 0}, "cuda": {"status": "measured", "peak_reserved_bytes": 1}},
        "runner_telemetry": _telemetry(),
        "artifact_inventory": _inventory(), "work_item_counts": work_counts,
        "audit": audit, "validation_errors": [],
    }


def _budget_binding() -> dict[str, Any]:
    storage = {
        "estimate_model": {
            "l0_base_bytes_per_unit": 32768,
            "l0_bytes_per_work_item": 192,
            "l1_float32_patch_map_bytes": 784,
            "l2_shared_attention_bytes_per_audit_sample": 22_353_984,
        },
        "L0": {"estimated_bytes": 1},
        "L1": {"estimated_bytes": 9592 * 784},
        "L2": {"estimated_bytes": 3_000_000_000},
        "L3": {"estimated_bytes": 0},
        "control": {
            "estimated_bytes": 100_000, "file_count": 4364,
            "unit_manifest_files": 2180, "unit_status_files": 2180,
            "batch_control_files": 4,
        },
        "payload_bytes_before_control_overhead": 3_008_000_000,
    }
    return {"c6_storage_model": storage, "source_batch_control_bytes": 100_000}


@pytest.fixture(scope="module")
def evidence() -> dict[str, Any]:
    records = []
    index = 0
    for sample in CALIBRATION_SAMPLE_IDS:
        for method in (*VALUE_METHOD_IDS, *CLASS_METHOD_IDS):
            if method in VALUE_METHOD_IDS:
                for block in range(12):
                    for head in range(12):
                        records.append(_record(index, sample, method, block, head, full=(block, head) in {(0, 0), (0, 1), (3, 0), (6, 0), (9, 0)})); index += 1
            else:
                records.append(_record(index, sample, method, None, None, full=False)); index += 1
    document = {
        "schema_version": 1, "document_type": selection.EVIDENCE_DOCUMENT_TYPE,
        "source_root": "/synthetic/c8", "holdout_accessed": False,
        "selection_performed": False, "bootstrap_performed": False,
        "holm_performed": False, "decision_performed": False,
        "expected_unit_count": 2180, "batch_complete": True,
        "batch_successful": True, "calibration_sample_ids": list(CALIBRATION_SAMPLE_IDS),
        "method_ids": [*VALUE_METHOD_IDS, *CLASS_METHOD_IDS],
        "bindings": {"source_root": "/synthetic/c8", "execution_plan_sha256": HASH,
                     "run_spec_sha256": HASH, "preflight_sha256": HASH,
                     "control_file_sha256": {name: HASH for name in selection._CONTROL_NAMES},
                     "code": {"commit": "b" * 40, "dirty": False},
                     "source_tree_total_bytes": 2180 * 4},
        "records": records, "c7_report_binding": _budget_binding(),
        "calibration_artifact_manifest_sha256": selection._records_digest(records),
    }
    document["canonical_digest"] = selection._self_digest(document)
    return document


def _input_hashes(evidence: dict[str, Any]) -> dict[str, Any]:
    result = {name: HASH for name in selection._INPUT_HASH_KEYS}
    result.update({"code_commit": "b" * 40, "code_tree_clean": True,
                   "calibration_artifact_manifest_sha256": evidence["calibration_artifact_manifest_sha256"]})
    return result


def _resign(value: dict[str, Any]) -> None:
    value["calibration_artifact_manifest_sha256"] = selection._records_digest(value["records"])
    value["canonical_digest"] = selection._self_digest(value)


def test_build_selection_exact_candidates_ranking_and_schema(evidence):
    report = selection.build_stage_c_selection(evidence, _input_hashes(evidence))
    assert report["status"] == "PASS"
    result = report["selection_document"]
    assert len(result["candidate_records"]) == 144
    assert [(x["block_index"], x["head_index"]) for x in result["candidate_records"]] == [
        (block, head) for block in range(12) for head in range(12)
    ]
    assert [item["candidate_key"] for item in result["depth_band_winners"]] == [
        "block_00_head_00", "block_03_head_00", "block_06_head_00", "block_09_head_00"
    ]
    assert result["primary"]["candidate_key"] == "block_00_head_00"
    candidate = result["candidate_records"][0]
    assert candidate["count_delta_fid_fine4_gt_zero"] == 5
    assert candidate["rounded_rank_tuple"][-2:] == [0, 0]
    assert report["evidence_summary"]["candidate_manifest_unit_count"] == 15
    selection.validate_stage_c_selection_document(result, expected_input_hashes=_input_hashes(evidence))
    selection.validate_stage_c_selection_report(report, evidence=evidence, input_hashes=_input_hashes(evidence))


def test_recovery_context_requires_outer_c8_authority(evidence):
    with pytest.raises(
        selection.StageCSelectionIntegrityError,
        match="requires its outer launch authority",
    ):
        selection.validate_c8_calibration_evidence(
            evidence, trusted_recovery_context={
                "recovery_binding": object(),
                "recovery_launch_record": object(),
            },
        )


def test_recovery_input_model_hash_must_match_authority(
    evidence, monkeypatch,
):
    recovery_sha = "4" * 64
    recovery = {
        "interpretation": selection._RECOVERY_INTERPRETATION,
        "model": {
            "identity": "RECOVERY_V2_NEW_ARTIFACT_NOT_HISTORICAL_2E718",
            "state_dict_sha256": recovery_sha,
            "canonical_tensor_digest": "5" * 64,
        },
        "recovery_binding": {
            "path": "config/recovery.json", "file_sha256": "6" * 64,
            "canonical_digest": "7" * 64,
        },
        "c7_launch": {
            "path": "evidence/c7/launch.json",
            "file_sha256": "8" * 64, "canonical_digest": "9" * 64,
            "launch_commit": "1" * 40, "producer_commit": "2" * 40,
            "producer_tree": "3" * 40,
        },
        "c7_report": {
            "path": "evidence/c7/report.json",
            "file_sha256": "a" * 64, "canonical_digest": "b" * 64,
            "report_commit": "4" * 40, "status": "PASS",
            "source_commit": "1" * 40,
        },
        "c8_launch": {
            "file_sha256": "c" * 64, "canonical_digest": "d" * 64,
            "launch_commit": "b" * 40,
        },
        "c8_execution_source": {"commit": "b" * 40, "dirty": False},
    }
    authority = object()
    context = {"recovery_binding": object(), "recovery_launch_record": object()}
    monkeypatch.setattr(
        selection, "_validated_recovery_inputs",
        lambda supplied, supplied_context: (authority, context, recovery),
    )
    monkeypatch.setattr(
        selection, "validate_c8_calibration_evidence", lambda *_a, **_k: None,
    )
    hashes = _input_hashes(evidence)
    assert hashes["model_sha256"] != recovery_sha
    with pytest.raises(
        selection.StageCSelectionIntegrityError,
        match="model SHA-256 differs from recovery C8 authority",
    ):
        selection.build_stage_c_selection(
            evidence, hashes, trusted_recovery_authority=authority,
            trusted_recovery_context=context,
        )


def test_recovery_binding_validator_rejects_execution_source_drift():
    value = {
        "interpretation": selection._RECOVERY_INTERPRETATION,
        "model": {
            "identity": "RECOVERY_V2_NEW_ARTIFACT_NOT_HISTORICAL_2E718",
            "state_dict_sha256": "4" * 64,
            "canonical_tensor_digest": "5" * 64,
        },
        "recovery_binding": {
            "path": "config/recovery.json", "file_sha256": "6" * 64,
            "canonical_digest": "7" * 64,
        },
        "c7_launch": {
            "path": "evidence/c7/launch.json",
            "file_sha256": "8" * 64, "canonical_digest": "9" * 64,
            "launch_commit": "1" * 40, "producer_commit": "2" * 40,
            "producer_tree": "3" * 40,
        },
        "c7_report": {
            "path": "evidence/c7/report.json",
            "file_sha256": "a" * 64, "canonical_digest": "b" * 64,
            "report_commit": "4" * 40, "status": "PASS",
            "source_commit": "1" * 40,
        },
        "c8_launch": {
            "file_sha256": "c" * 64, "canonical_digest": "d" * 64,
            "launch_commit": "b" * 40,
        },
        "c8_execution_source": {"commit": "e" * 40, "dirty": False},
    }
    with pytest.raises(
        selection.StageCSelectionIntegrityError, match="execution source differs",
    ):
        selection._validate_recovery_authority_binding(value)


def test_na_propagates_to_aggregate_but_not_eligibility(evidence):
    changed = copy.deepcopy(evidence)
    record = next(item for item in changed["records"] if item["unit_key"]["method_id"] == NNS_METHOD_ID)
    record["scalars"]["residual_stability"][0]["signed_spearman"] = _metric(None)
    record["scalars"]["residual_stability"][2]["clean"]["rho"] = _metric(None)
    record["scalars"]["residual_stability"][2]["clean"]["details"].update(
        {"numerator_l1": 0.0, "denominator_l1": 0.0}
    )
    _resign(changed)
    report = selection.build_stage_c_selection(changed, _input_hashes(changed))
    candidate = report["candidate_records"][0]
    assert candidate["eligible"] is True
    assert candidate["minimum_perturbation_median_residual_spearman"] is None
    assert candidate["median_patch_residual_l1_ratio"] is None
    assert candidate["rounded_rank_tuple"][2:4] == [-1.0, 0.0]


def test_spatial_degeneration_is_retained_not_filtered(evidence):
    changed = copy.deepcopy(evidence)
    record = next(item for item in changed["records"] if item["unit_key"]["method_id"] == NNS_METHOD_ID)
    record["scalars"]["residual_stability"][0]["clean"]["status"] = "SPATIAL_FAILURE"
    _resign(changed)
    report = selection.build_stage_c_selection(changed, _input_hashes(changed))
    assert report["candidate_records"][0]["eligible"] is True
    assert report["candidate_audit_records"][0]["spatial_degenerate_sample_count"] == 1


def test_failure_empties_band_and_blocks_without_selection(evidence, tmp_path, monkeypatch):
    changed = copy.deepcopy(evidence)
    for record in changed["records"]:
        key = record["unit_key"]
        if key["block_index"] in {0, 1, 2} and key["method_id"] == NNS_METHOD_ID:
            record["state"] = "failed"
            record["trusted_by_batch_validator"] = False
            record["validation_errors"] = ["synthetic failure"]
            record["scalars"] = {}
            record["provenance"] = {}
            record["runner_telemetry"] = {}
            record["hashes"] = {name: None for name in selection._HASH_KEYS}
    changed["batch_successful"] = False
    _resign(changed)
    monkeypatch.setattr(selection, "validate_stage_c_provenance", lambda *_a, **_k: None)
    report = selection.build_stage_c_selection(changed, _input_hashes(changed))
    assert report["status"] == "BLOCKED"
    assert "depth_band_0_has_no_eligible_candidate" in report["blocked_reasons"]
    assert report["selection_document"] is None
    paths = selection.write_stage_c_selection_outputs(
        tmp_path / "selection.json", tmp_path / "report.json", report,
        evidence=changed, input_hashes=_input_hashes(changed),
    )
    assert paths["selection_sha256"] is None
    assert not (tmp_path / "selection.json").exists()
    assert (tmp_path / "report.json").is_file()


def test_exact_order_global_success_and_manifest_tamper_fail_closed(evidence):
    changed = copy.deepcopy(evidence)
    changed["records"][0], changed["records"][1] = changed["records"][1], changed["records"][0]
    for index, item in enumerate(changed["records"]):
        item["plan_index"] = index
    _resign(changed)
    with pytest.raises(selection.StageCSelectionIntegrityError, match="exact plan order"):
        selection.build_stage_c_selection(changed, _input_hashes(changed))

    changed = copy.deepcopy(evidence)
    changed["batch_successful"] = False
    _resign(changed)
    with pytest.raises(selection.StageCSelectionIntegrityError, match="contradicts"):
        selection.build_stage_c_selection(changed, _input_hashes(changed))

    changed = copy.deepcopy(evidence)
    changed["records"][0]["scalars"]["manual_edit"] = True
    changed["canonical_digest"] = selection._self_digest(changed)
    with pytest.raises(selection.StageCSelectionIntegrityError, match="manifest digest"):
        selection.build_stage_c_selection(changed, _input_hashes(changed))


def test_report_source_recomputation_and_exclusive_writers(evidence, tmp_path, monkeypatch):
    monkeypatch.setattr(selection, "validate_stage_c_provenance", lambda *_a, **_k: None)
    hashes = _input_hashes(evidence)
    report = selection.build_stage_c_selection(evidence, hashes)
    tampered = copy.deepcopy(report)
    tampered["candidate_records"][0]["median_delta_fid_fine4"] += 1
    tampered["selection_document"]["candidate_records"][0]["median_delta_fid_fine4"] += 1
    tampered["selection_document_sha256"] = selection._digest(tampered["selection_document"])
    tampered["canonical_digest"] = selection._self_digest(tampered)
    with pytest.raises(selection.StageCSelectionIntegrityError):
        selection.validate_stage_c_selection_report(tampered, evidence=evidence, input_hashes=hashes)

    result = selection.write_stage_c_selection_outputs(
        tmp_path / "selection.json", tmp_path / "report.json", report,
        evidence=evidence, input_hashes=hashes,
    )
    assert result["status"] == "PASS" and result["selection_sha256"]
    retry = selection.write_stage_c_selection_outputs(
        tmp_path / "selection.json", tmp_path / "report.json", report,
        evidence=evidence, input_hashes=hashes,
    )
    assert retry == result


@pytest.mark.parametrize(
    "damage, expected",
    [
        (lambda direct, _nns, _matched: direct.update({"fidelity_enabled": True}),
         "fabricated registered fidelity"),
        (lambda _direct, nns, _matched: nns.update({"fidelity_curves": nns["fidelity_curves"][:-1]}),
         "44 complete curves"),
        (lambda _direct, _nns, matched: matched["provenance_fragment"]["matched_contribution"].pop("scalar_delta_f_z_minus_f_x0"),
         "scalar_delta_f_z_minus_f_x0"),
        (lambda _direct, nns, _matched: nns["residual_stability"][1]["clean"].update({"method_version": "2.0.0"}),
         "scientific identity mismatch"),
        (lambda _direct, nns, _matched: nns["map_execution_records"][0].update({"attempted_steps_total": 1024}),
         "attempted steps disagree"),
        (lambda _direct, nns, _matched: nns["residual_stability"][0]["clean"]["rho"].update({"value": 0.3}),
         "rho formula is inconsistent"),
    ],
)
def test_scientific_audit_gaps_fail_candidate_closed(evidence, damage, expected):
    changed = copy.deepcopy(evidence)
    records = [item for item in changed["records"] if (
        item["unit_key"]["sample_id"] == CALIBRATION_SAMPLE_IDS[0]
        and item["unit_key"]["block_index"] == 0
        and item["unit_key"]["head_index"] == 0
    )]
    by_method = {item["unit_key"]["method_id"]: item["scalars"] for item in records}
    damage(by_method[VALUE_METHOD_IDS[0]], by_method[NNS_METHOD_ID], by_method[MATCHED_METHOD_ID])
    _resign(changed)
    report = selection.build_stage_c_selection(changed, _input_hashes(changed))
    candidate = report["candidate_records"][0]
    assert candidate["eligible"] is False
    assert any(expected in item for item in candidate["validity_failures"])


def test_post_c8_budget_uses_frozen_estimator_and_blocks_over_cap(evidence):
    report = selection.build_stage_c_selection(evidence, _input_hashes(evidence))
    budget = report["post_c8_budget"]
    assert budget["exact_unit_count"] == 2500
    assert budget["projection_safety_factor"] == 1.25
    assert budget["projection"]["exact_unit_counts"]["C8_C9"] == 2500
    assert budget["projection"]["time"]["C8_C9_h20_gpu_hours"] == pytest.approx(2500 * 1.25 / 3600)

    changed = copy.deepcopy(evidence)
    for item in changed["records"]:
        item["runner_telemetry"]["phase_timings"]["synchronized_total_wall_time_seconds"] = 2000.0
        for phase in item["runner_telemetry"]["phase_timings"]["phases"].values():
            phase["wall_time_seconds"] = 0.0
    _resign(changed)
    blocked = selection.build_stage_c_selection(changed, _input_hashes(changed))
    assert blocked["status"] == "BLOCKED"
    assert "post_c8_frozen_cap_failure" in blocked["blocked_reasons"]
    assert blocked["selection_document"] is None


def test_selection_never_opens_jpeg_or_l3(monkeypatch, evidence):
    original = Path.open
    opened: list[str] = []

    def guarded(path: Path, *args, **kwargs):
        text = str(path)
        opened.append(text)
        if text.endswith(".JPEG") or "/L3/" in text:
            raise AssertionError(f"forbidden quantitative read: {text}")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)
    report = selection.build_stage_c_selection(evidence, _input_hashes(evidence))
    assert report["status"] == "PASS"
    assert not any(path.endswith(".JPEG") or "/L3/" in path for path in opened)


def test_one_candidate_failure_is_local_and_batch_may_still_select(evidence):
    changed = copy.deepcopy(evidence)
    failed = next(item for item in changed["records"] if (
        item["unit_key"]["sample_id"] == CALIBRATION_SAMPLE_IDS[0]
        and item["unit_key"]["method_id"] == NNS_METHOD_ID
        and item["unit_key"]["block_index"] == 0
        and item["unit_key"]["head_index"] == 0
    ))
    failed["state"] = "failed"
    failed["trusted_by_batch_validator"] = False
    failed["validation_errors"] = ["synthetic terminal failure"]
    failed["hashes"] = {name: None for name in selection._HASH_KEYS}
    failed["scalars"] = {}
    failed["provenance"] = {}
    failed["runner_telemetry"] = {}
    changed["batch_successful"] = False
    _resign(changed)
    report = selection.build_stage_c_selection(changed, _input_hashes(changed))
    assert report["status"] == "PASS"
    assert report["primary"]["candidate_key"] == "block_00_head_01"
    assert any(item["candidate_key"] == "block_00_head_00" for item in report["failure_records"])


def test_global_class_failure_blocks_and_is_reported(evidence):
    changed = copy.deepcopy(evidence)
    failed = next(item for item in changed["records"] if (
        item["unit_key"]["method_id"] in CLASS_METHOD_IDS
    ))
    failed["state"] = "failed"
    failed["trusted_by_batch_validator"] = False
    failed["validation_errors"] = ["synthetic global failure"]
    failed["hashes"] = {name: None for name in selection._HASH_KEYS}
    failed["scalars"] = {}
    failed["provenance"] = {}
    failed["runner_telemetry"] = {}
    changed["batch_successful"] = False
    _resign(changed)
    report = selection.build_stage_c_selection(changed, _input_hashes(changed))
    assert report["status"] == "BLOCKED"
    assert "global_class_unit_failure" in report["blocked_reasons"]
    failures = [item for item in report["failure_records"] if item.get("unit_key") == failed["unit_key"]]
    assert failures and failures[0]["candidate_key"] is None
    assert report["selection_document"] is None


def test_public_writers_require_source_recomputation(evidence, tmp_path):
    hashes = _input_hashes(evidence)
    report = selection.build_stage_c_selection(evidence, hashes)
    with pytest.raises(TypeError, match="requires evidence"):
        selection.write_stage_c_selection_report(tmp_path / "report.json", report)
    with pytest.raises(TypeError, match="requires evidence"):
        selection.write_stage_c_selection(tmp_path / "selection.json", report)
    tampered = copy.deepcopy(report)
    tampered["candidate_records"][0]["median_delta_fid_fine4"] += 1.0
    tampered["canonical_digest"] = selection._self_digest(tampered)
    with pytest.raises(selection.StageCSelectionIntegrityError):
        selection.write_stage_c_selection_report(
            tmp_path / "tampered.json", tampered, evidence=evidence, input_hashes=hashes
        )


@pytest.mark.parametrize(
    "change",
    [
        lambda status: status.update({"lifecycle_state": "running", "orchestration_complete": False}),
        lambda status: status.update({"lifecycle_state": "finalizing", "orchestration_complete": False}),
        lambda status: status.update({"expected_unit_count": 2179}),
        lambda status: status.update({"failure_table_sha256": "f" * 64}),
        lambda status: status.update({"unknown": True}),
    ],
)
def test_source_reader_requires_closed_finalized_batch_status(change):
    plan_digest = "1" * 64
    table = {
        "schema_version": 1, "document_type": "stage_c_failure_table",
        "plan_digest": plan_digest, "rows": [{} for _ in range(2180)],
    }
    table_hash = selection._digest(table)
    status = {
        "schema_version": 1, "document_type": "stage_c_batch_status",
        "plan_digest": plan_digest, "expected_unit_count": 2180,
        "orchestration_complete": True, "successful": False,
        "failure_table_sha256": table_hash, "lifecycle_state": "finalized",
        "active_claim_count": 0, "validation_errors": [],
    }
    change(status)
    with pytest.raises(selection.StageCSelectionIntegrityError):
        selection._validate_finalized_batch_status(
            status, table, plan_digest=plan_digest,
            failure_table_sha256=table_hash,
        )


def test_finalized_batch_status_allows_terminal_failures():
    plan_digest = "1" * 64
    table = {
        "schema_version": 1, "document_type": "stage_c_failure_table",
        "plan_digest": plan_digest, "rows": [{} for _ in range(2180)],
    }
    table_hash = selection._digest(table)
    status = {
        "schema_version": 1, "document_type": "stage_c_batch_status",
        "plan_digest": plan_digest, "expected_unit_count": 2180,
        "orchestration_complete": True, "successful": False,
        "failure_table_sha256": table_hash, "lifecycle_state": "finalized",
        "active_claim_count": 0, "validation_errors": [],
    }
    selection._validate_finalized_batch_status(
        status, table, plan_digest=plan_digest, failure_table_sha256=table_hash
    )


def test_publish_rejects_same_selection_and_report_path(evidence, tmp_path):
    hashes = _input_hashes(evidence)
    report = selection.build_stage_c_selection(evidence, hashes)
    same = tmp_path / "same.json"
    with pytest.raises(
        selection.StageCSelectionIntegrityError, match="must be distinct"
    ):
        selection.write_stage_c_selection_outputs(
            same, same, report, evidence=evidence, input_hashes=hashes
        )
    assert not same.exists()


def test_partial_publish_exact_retry_completes_missing_second_file(
    evidence, tmp_path, monkeypatch,
):
    hashes = _input_hashes(evidence)
    report = selection.build_stage_c_selection(evidence, hashes)
    report_path = tmp_path / "report.json"
    selection_path = tmp_path / "selection.json"
    original = selection._create_or_accept_exact_json
    injected = False

    def fail_second(path, document, *, context):
        nonlocal injected
        if context == "selection document" and not injected:
            injected = True
            raise OSError("injected second-file publication failure")
        return original(path, document, context=context)

    monkeypatch.setattr(selection, "_create_or_accept_exact_json", fail_second)
    with pytest.raises(OSError, match="injected second-file"):
        selection.write_stage_c_selection_outputs(
            selection_path, report_path, report,
            evidence=evidence, input_hashes=hashes,
        )
    assert report_path.read_bytes() == selection._pretty_json_bytes(report)
    assert not selection_path.exists()

    monkeypatch.setattr(selection, "_create_or_accept_exact_json", original)
    result = selection.write_stage_c_selection_outputs(
        selection_path, report_path, report,
        evidence=evidence, input_hashes=hashes,
    )
    assert result["status"] == "PASS"
    assert report_path.read_bytes() == selection._pretty_json_bytes(report)
    assert selection_path.read_bytes() == selection._pretty_json_bytes(
        report["selection_document"]
    )


@pytest.mark.parametrize("tampered_side", ["report", "selection"])
def test_publish_rejects_tampered_existing_output_before_creating_other(
    evidence, tmp_path, tampered_side,
):
    hashes = _input_hashes(evidence)
    report = selection.build_stage_c_selection(evidence, hashes)
    report_path = tmp_path / "report.json"
    selection_path = tmp_path / "selection.json"
    existing = report_path if tampered_side == "report" else selection_path
    missing = selection_path if tampered_side == "report" else report_path
    existing.write_bytes(b"{\"tampered\":true}\n")
    with pytest.raises(
        selection.StageCSelectionIntegrityError, match="differs from source-evidence"
    ):
        selection.write_stage_c_selection_outputs(
            selection_path, report_path, report,
            evidence=evidence, input_hashes=hashes,
        )
    assert not missing.exists()
