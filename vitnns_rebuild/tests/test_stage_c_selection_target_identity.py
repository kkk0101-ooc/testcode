from __future__ import annotations

import copy
import dataclasses
import hashlib

import pytest

import vitnns.stage_c_selection as selection
from vitnns.stage_c_fidelity import (
    NNS_METHOD_ID, CurveDirection, FidelityCurveResult, FidelityScale,
    RankingKind, actual_curve_fractions, aggregate_five_curves,
    curve_unit_counts, fidelity_seed, tie_priority_sha256, trapezoid_auc,
)


def _identity(runtime: str, *, direction: str = "d" * 64) -> dict:
    return {
        "comparison_line": "value_path",
        "target_ref": "fixed_clean_attention_cls_head_direction",
        "clean_input_sha256": "a" * 64,
        "frozen_target": {
            "target_kind": "fixed_clean_attention_cls_head_direction",
            "candidate": {"block_index": 0, "head_index": 0},
            "status": "OK",
            "direction_norm": 1.0,
            "direction_threshold": 1e-6,
            "direction_unit": {"finite": True, "sha256": direction},
            "runtime_model_binding": {
                "kind": "same_live_model_instance_and_tensor_state_versions",
                "sha256": runtime,
                "is_canonical_model_artifact_hash": False,
            },
        },
    }


def _record(identity: dict) -> dict:
    return {
        "scalars": {
            "target_identity": identity,
            "provenance_fragment": {
                "endpoint_parity": {
                    "threshold": 1e-6,
                    "comparison": "strict_less_than",
                    "clean_formula_max_abs_error": 0.0,
                    "clean_natural_vs_fixed_replay_max_abs_error": 0.0,
                    "clean_natural_vs_fixed_replay_model_output_max_abs_error": 0.0,
                    "baseline_fixed_replay_vs_clean_A_matmul_natural_V0_max_abs_error": 0.0,
                }
            },
        }
    }


def _curve(scale, direction, ranking, replicate, identity_sha256):
    count = scale.unit_count
    seed = None if replicate is None else fidelity_seed("sample", 0, 0, replicate)
    values = tuple(
        float(i / 20 if direction is CurveDirection.INSERTION else 1 - i / 20)
        for i in range(21)
    )
    return FidelityCurveResult(
        method_id=NNS_METHOD_ID, comparison_line="value_path",
        target_ref="fixed_clean_attention_cls_head_direction",
        target_identity_sha256=identity_sha256, scale=scale, direction=direction,
        ranking_kind=ranking, sample_id="sample", block=0, head=0,
        replicate=replicate, seed=seed,
        tie_priority_sha256=(None if seed is None else tie_priority_sha256(count, seed)),
        source_unit_map_sha256=("b" * 64 if ranking is RankingKind.TARGET_ORIENTED_POSITIVE else None),
        score_sha256=("c" * 64 if ranking is RankingKind.TARGET_ORIENTED_POSITIVE else None),
        input_pair_sha256="e" * 64,
        ranking_sha256=hashlib.sha256(f"{scale}:{direction}:{ranking}:{replicate}".encode()).hexdigest(),
        intervention_sha256=tuple(f"{i:064x}" for i in range(21)),
        unit_counts=curve_unit_counts(count), actual_fractions=actual_curve_fractions(count),
        responses=values, auc=trapezoid_auc(actual_curve_fractions(count), values),
    )


def _fidelity_record(identity: dict) -> dict:
    digest = selection.canonical_sha256(identity)
    curves = []
    aggregates = []
    for scale in (FidelityScale.PATCH_16, FidelityScale.FINE_4):
        for direction in (CurveDirection.DELETION, CurveDirection.INSERTION):
            target = tuple(
                _curve(scale, direction, RankingKind.TARGET_ORIENTED_POSITIVE, i, digest)
                for i in range(5)
            )
            curves.extend(target)
            aggregates.append(aggregate_five_curves(target).to_dict())
            curves.extend(
                _curve(scale, direction, RankingKind.GLOBAL_RANDOM, i, digest)
                for i in range(5)
            )
            curves.append(_curve(
                scale, direction, RankingKind.CENTER_DISTANCE_ROW_MAJOR, None, digest
            ))
    return {
        "unit_key": {"method_id": NNS_METHOD_ID, "sample_id": "sample",
                     "block_index": 0, "head_index": 0},
        "scalars": {"target_identity": identity,
                    "fidelity_curves": [curve.to_dict() for curve in curves],
                    "fidelity_aggregates": aggregates},
    }


def test_runtime_binding_is_not_part_of_cross_unit_scientific_identity():
    nns = _record(_identity("1" * 64))
    matched = _record(_identity("2" * 64))
    digest = selection._target_checks(nns, matched, 0, 0)
    assert digest == selection.canonical_sha256(
        selection._scientific_target_identity(nns["scalars"]["target_identity"])
    )


def test_scientific_target_difference_still_fails_closed():
    with pytest.raises(selection.StageCSelectionIntegrityError, match="identities differ"):
        selection._target_checks(
            _record(_identity("1" * 64)),
            _record(_identity("2" * 64, direction="f" * 64)), 0, 0,
        )


def test_curve_must_bind_its_units_complete_target_before_normalization():
    identity = _identity("1" * 64)
    record = _fidelity_record(identity)
    scientific_digest = selection.canonical_sha256(
        selection._scientific_target_identity(identity)
    )
    aggregates = selection._validated_aggregates(
        record, scientific_target_identity_sha256=scientific_digest
    )
    assert {
        curve.target_identity_sha256
        for aggregate in aggregates.values() for curve in aggregate.curves
    } == {scientific_digest}

    record["scalars"]["fidelity_curves"][0]["target_identity_sha256"] = "0" * 64
    with pytest.raises(selection.StageCSelectionIntegrityError, match="complete target identity"):
        selection._validated_aggregates(
            record, scientific_target_identity_sha256=scientific_digest
        )


def test_generator_git_state_rejects_dirty_or_tampered_values():
    commit = "1" * 40
    assert selection._selection_generator_code(
        {"commit": commit, "dirty": False}
    ) == {"commit": commit, "dirty": False}
    for invalid in (
        {"commit": commit, "dirty": True},
        {"commit": "g" * 40, "dirty": False},
        {"commit": commit[:-1], "dirty": False},
        {"commit": commit, "dirty": False, "extra": True},
    ):
        with pytest.raises(
            selection.StageCSelectionIntegrityError, match="generator Git state"
        ):
            selection._selection_generator_code(invalid)


def test_selection_generated_by_binds_generator_commit(monkeypatch):
    monkeypatch.setattr(
        selection, "validate_stage_c_selection_document", lambda *_a, **_k: None
    )
    commit = "2" * 40
    document = selection._selection_document(
        [], [], {}, [], {}, commit
    )
    assert document["generated_by"] == f"{selection.GENERATED_BY}@{commit}"


def test_report_validation_rejects_explicit_generator_mismatch(monkeypatch):
    source = {"commit": "1" * 40, "dirty": False}
    generator = {"commit": "2" * 40, "dirty": False}
    evidence = {"bindings": {"code": source}}
    report = {
        "schema_version": 1,
        "document_type": selection.REPORT_DOCUMENT_TYPE,
        "status": selection.STATUS_BLOCKED,
        "integrity_valid": True,
        "blocked_reasons": ["test"],
        "holdout_accessed": False,
        "selection_performed": False,
        "bootstrap_performed": False,
        "holm_performed": False,
        "decision_performed": False,
        "bindings": {"selection_generator_code": generator},
        "evidence_summary": {},
        "post_c8_budget": {
            "status": selection.STATUS_BLOCKED,
            "reason": "test",
            "exact_unit_count": 2500,
            "projection_safety_factor": 1.25,
            "projection": None,
        },
        "candidate_records": [{} for _ in range(144)],
        "candidate_audit_records": [],
        "depth_band_winners": [],
        "primary": None,
        "failure_records": [],
        "selection_document": None,
        "selection_document_sha256": None,
    }
    report["canonical_digest"] = selection._self_digest(report)
    selection.validate_stage_c_selection_report(
        report, generator_git_state=generator
    )
    with pytest.raises(
        selection.StageCSelectionIntegrityError,
        match="generator code binding differs",
    ):
        selection.validate_stage_c_selection_report(
            report,
            generator_git_state={"commit": "3" * 40, "dirty": False},
        )


def test_report_binds_generator_and_rejects_tampering(monkeypatch):
    source = {"commit": "1" * 40, "dirty": False}
    generator = {"commit": "2" * 40, "dirty": False}
    evidence = {
        "bindings": {"code": source},
        "calibration_artifact_manifest_sha256": "a" * 64,
        "records": [], "batch_complete": True,
        "c7_report_binding": {}, "canonical_digest": "b" * 64,
    }
    monkeypatch.setattr(selection, "validate_c8_calibration_evidence", lambda *_: None)
    monkeypatch.setattr(selection, "_validate_input_hashes", lambda *_: {})
    monkeypatch.setattr(
        selection, "_post_c8_budget",
        lambda *_: {"status": selection.STATUS_PASS, "reason": None,
                   "exact_unit_count": 2500, "projection_safety_factor": 1.25},
    )
    monkeypatch.setattr(
        selection, "_build_candidate",
        lambda _records, block, head: ({
            "candidate_key": f"block_{block:02d}_head_{head:02d}",
            "block_index": block, "head_index": head,
            "depth_band": block // 3, "eligible": False,
            "rounded_rank_tuple": None, "selected_depth_band_winner": False,
            "selected_primary": False,
        }, {}, []),
    )
    report = selection.build_stage_c_selection(
        evidence, {}, generator_git_state=generator
    )
    assert report["bindings"]["selection_generator_code"] == generator

    tampered = copy.deepcopy(report)
    tampered["bindings"]["selection_generator_code"]["commit"] = "3" * 40
    tampered["canonical_digest"] = selection._self_digest(tampered)
    with pytest.raises(selection.StageCSelectionIntegrityError, match="expected Git state"):
        selection.validate_stage_c_selection_report(
            tampered, evidence=evidence, input_hashes={},
            generator_git_state=generator,
        )

    with pytest.raises(selection.StageCSelectionIntegrityError, match="clean 40-hex"):
        selection.build_stage_c_selection(
            evidence, {}, generator_git_state={**generator, "dirty": True}
        )
