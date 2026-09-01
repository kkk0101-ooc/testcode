"""Model- and image-free tests for Stage D scientific source validation."""

from __future__ import annotations

import hashlib
import copy
import json
import math
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from vitnns.stage_d_scientific import (
    MATCHED_METHOD_ID,
    NNS_METHOD_ID,
    PAIR_BINDING_KEYS,
    PRODUCTION_L0_KEYS,
    reconcile_stage_d_pair,
    validate_stage_d_pair_bindings,
    validate_stage_d_scientific_evidence,
)


ENDPOINT_LIMIT = 1e-6
REPEATABILITY_ATOL = 1e-6
REPEATABILITY_RTOL = 1e-5
SEED_NAMESPACE = "stage-d-fidelity|20260821|"
TARGET_ID = "fixed_clean_attention_cls_head_direction"


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    source = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(source.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(
        json.dumps(list(source.shape), separators=(",", ":")).encode("ascii")
    )
    digest.update(b"\0")
    digest.update(source.tobytes(order="C"))
    return digest.hexdigest()


def _identity_sha256(value: dict[str, Any]) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def _canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _resign_attempt_provenance(value: dict[str, Any]) -> None:
    unsigned = dict(value)
    unsigned.pop("canonical_digest", None)
    value["canonical_digest"] = _canonical_sha256(unsigned)


def _seed(sample_id: str, repeat: int) -> int:
    raw = f"{SEED_NAMESPACE}{sample_id}|0|9|{repeat}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


def _tie_priority_sha256(seed: int) -> str:
    random_order = np.random.default_rng(seed).permutation(3136)
    priority = np.empty(3136, dtype=np.int64)
    priority[random_order] = np.arange(3136, dtype=np.int64)
    return _array_sha256(priority)


def _ranking_sha256(metric: np.ndarray, seed: int) -> str:
    scores = np.maximum(
        metric, np.float32(0.0),
    ).astype(np.float32, copy=False).reshape(-1)
    random_order = np.random.default_rng(seed).permutation(scores.size)
    priority = np.empty(scores.size, dtype=np.int64)
    priority[random_order] = np.arange(scores.size, dtype=np.int64)
    ranking = np.lexsort(
        (priority, -scores.astype(np.float64)),
    ).astype(np.int64)
    return _array_sha256(ranking)


def _tensor_sha256(value: np.ndarray) -> str:
    source = np.ascontiguousarray(value, dtype=np.float32)
    digest = hashlib.sha256()
    digest.update(b"torch.float32")
    digest.update(b"|")
    digest.update(
        ",".join(str(item) for item in source.shape).encode("ascii")
    )
    digest.update(b"|")
    digest.update(source.tobytes(order="C"))
    return digest.hexdigest()


def _input_pair_sha256(clean_sha256: str) -> str:
    baseline = np.zeros((1, 3, 224, 224), dtype=np.float32)
    return hashlib.sha256(
        f"{clean_sha256}|{_tensor_sha256(baseline)}".encode("ascii")
    ).hexdigest()


def _auc(actual_fractions: list[float], responses: list[float]) -> float:
    x = np.asarray(actual_fractions, dtype=np.float64)
    y = np.asarray(responses, dtype=np.float64)
    return float(
        np.sum(np.diff(x) * (y[:-1] + y[1:]) * 0.5, dtype=np.float64)
    )


def _unit(method_id: str, *, audit: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        sample_id="synthetic-sample-000", method_id=method_id,
        target_id=TARGET_ID, block_index=0, head_index=9, audit=audit,
        canonical_digest=_sha(f"unit-{method_id}-audit-{audit}"),
        pair_binding_digest=_sha("shared-pair"),
    )


def _attempt_provenance(unit: SimpleNamespace) -> dict[str, Any]:
    fractions = [index / 20.0 for index in range(21)]
    bindings = {
        "d7_authorization": {"status": "NOT_ISSUED", "digest": None},
        "producer": {"commit": "a" * 40, "tree": "b" * 40, "clean": True},
        "dependencies": {"lock_sha256": _sha("dependencies")},
        "model": {
            "config_sha256": _sha("model-config"),
            "state_dict_sha256": _sha("model-state"),
        },
        "source_bindings": {
            name: {"canonical_digest": _sha(f"source-{name}")}
            for name in ("d0", "d1", "d2", "c10")
        },
        "coordinates": {
            "plan_digest": _sha("plan"),
            "group_digest": _sha("group"),
            "pair_digest": unit.pair_binding_digest,
            "unit_digest": unit.canonical_digest,
            "work_item_digests": [_sha(f"work-{unit.method_id}")],
            "work_item_count": 1,
            "work_item_sequence_digest": _sha(f"work-sequence-{unit.method_id}"),
        },
        "image_identity": {
            "status": "SYNTHETIC_NO_IMAGE_ACCESSED",
            "fixture_id": unit.sample_id,
        },
        "scientific_contract": {
            "method": {"id": unit.method_id, "version": "1.0.0"},
            "target": {"id": unit.target_id, "version": "1.0.0"},
            "block_index": unit.block_index,
            "head_index": unit.head_index,
            "operator": {"id": "fine_4x4_cell_projection", "version": "1.0.0"},
            "projection": {"id": "signed_56x56_fine_cell_mass"},
            "sign_semantics": {"id": "target_oriented_signed"},
            "normalization": {"id": "signed_endpoint_delta_zero_to_one"},
            "baseline": {"id": "normalized_zero_canonical_rgb_0_5"},
            "fractions": fractions,
            "seed_contract": {"namespace": SEED_NAMESPACE, "base_seed": 20260821},
        },
        "execution": {
            "attempt": {"number": 1, "id": "attempt-000001"},
            "retry": {"status": "NOT_A_RETRY"},
            "claim": {"owner_token_digest": _sha("owner")},
            "shard": {"id": 0, "count": 1},
        },
        "terminal": {"status": "succeeded", "terminal": True, "complete": True},
        "failure": None,
        "resources": {"forward_calls": 1, "backward_calls": 1},
        "artifact_inventory": [],
    }
    document = {
        "schema_version": 1,
        "document_type": "stage_d_execution_unit_provenance",
        "mode": "SYNTHETIC_D4_TEST_ONLY",
        "unit_identity": {
            "sample_id": unit.sample_id, "method_id": unit.method_id,
            "target_id": unit.target_id, "block_index": unit.block_index,
            "head_index": unit.head_index,
            "canonical_digest": unit.canonical_digest,
        },
        "unit_identity_sha256": unit.canonical_digest,
        "bindings": bindings,
    }
    document["canonical_digest"] = _canonical_sha256(document)
    return document


def _target() -> dict[str, Any]:
    return {
        "candidate": {"block_index": 0, "head_index": 9},
        "clean_value_max_abs_error": 0.0,
        "clean_replay_max_abs_error": 0.0,
        "clean_model_output_max_abs_error": 0.0,
        "baseline_replay_max_abs_error": 0.0,
        "direction_norm": 1.0,
        "direction_threshold": 0.1,
    }


def _ig_result() -> dict[str, Any]:
    attempt = {
        "steps": 512,
        "delta": [1.0],
        "sum": [1.0],
        "abs": [0.0],
        "rel": [0.0],
        "pass": [True],
    }
    return {
        "attempts": [attempt],
        "selected_steps": 512,
        "converged": True,
        "metadata": {
            "method": "integrated_gradients",
            "quadrature": "gauss_legendre",
            "integration_dtype": "float32",
            "input_shape": [1, 3, 224, 224],
            "step_schedule": [512, 1024, 2048, 4096, 8192],
            "step_chunk_size": 8,
            "atol": 1e-4, "rtol": 0.01,
            "scope": "fixed_clean_attention_head_direction_value_path",
            "target": {
                "kind": TARGET_ID, "block_index": 0, "head_index": 9,
                "query_token": 0,
            },
            "includes_residual_paths": False,
            "is_complete_attention_attribution": False,
            "fixed_clean_attention": True, "frozen_direction": True,
            "intervention_calls": 66, "pre_projection_calls": 66,
        },
        "conservation": {
            name: attempt[name]
            for name in ("delta", "sum", "abs", "rel", "pass")
        },
    }


def _provenance(method_id: str, target: dict[str, Any]) -> dict[str, Any]:
    if method_id == NNS_METHOD_ID:
        return {
            "target": target,
            "endpoint_parity": {
                "threshold": ENDPOINT_LIMIT,
                "comparison": "strict_less_than",
                "clean_formula_max_abs_error": 0.0,
                "clean_natural_vs_fixed_replay_max_abs_error": 0.0,
                "clean_natural_vs_fixed_replay_model_output_max_abs_error": 0.0,
                "baseline_fixed_replay_vs_clean_A_matmul_natural_V0_max_abs_error": 0.0,
            },
            "ig_result": _ig_result(),
        }
    return {
        "target": target,
        "matched_contribution": {
            "status": "OK",
            "token_contribution_count": 197,
            "cls_contribution": 0.0,
            "token_sum": 1.0,
            "scalar_delta_f_z_minus_f_x0": 1.0,
            "direction_norm": 1.0,
            "conservation_abs_error": 0.0,
            "conservation_tolerance": 1e-6 + 1e-5,
        },
    }


def _curve_records(
    unit: SimpleNamespace, metric: np.ndarray, target_identity: dict[str, Any],
) -> tuple[list[dict[str, Any]], np.ndarray]:
    fractions = [index / 20.0 for index in range(21)]
    unit_counts = [0] + [
        math.ceil(fraction * 3136) for fraction in fractions[1:]
    ]
    actual_fractions = [count / 3136 for count in unit_counts]
    metric_hash = _array_sha256(metric)
    score_hash = _array_sha256(
        np.maximum(metric, np.float32(0.0)).astype(np.float32, copy=False)
    )
    target_digest = _identity_sha256(target_identity)
    clean_sha256 = target_identity["clean_input_sha256"]
    baseline_sha256 = _tensor_sha256(
        np.zeros((1, 3, 224, 224), dtype=np.float32)
    )
    input_pair_sha256 = _input_pair_sha256(
        clean_sha256
    )
    records: list[dict[str, Any]] = []
    persisted = np.empty((2, 5, 21), dtype=np.float64)
    for direction_index, direction in enumerate(("insertion", "deletion")):
        for repeat in range(5):
            seed = _seed(unit.sample_id, repeat)
            responses = (
                list(actual_fractions)
                if direction == "insertion"
                else [1.0 - value for value in actual_fractions]
            )
            persisted[direction_index, repeat] = responses
            records.append({
                "status": "OK",
                "method_id": unit.method_id,
                "comparison_line": "value_path",
                "target_ref": unit.target_id,
                "target_identity_sha256": target_digest,
                "scale": "fine_4x4_cell",
                "direction": direction,
                "ranking_kind": "target_oriented_positive_tie_permuted",
                "sample_id": unit.sample_id,
                "block": unit.block_index,
                "head": unit.head_index,
                "replicate": repeat,
                "seed": seed,
                "tie_priority_sha256": _tie_priority_sha256(seed),
                "source_unit_map_sha256": metric_hash,
                "score_sha256": score_hash,
                "input_pair_sha256": input_pair_sha256,
                "ranking_sha256": _ranking_sha256(metric, seed),
                "intervention_sha256": [
                    (
                        baseline_sha256
                        if (direction == "insertion" and index == 0)
                        or (direction == "deletion" and index == 20)
                        else clean_sha256
                        if (direction == "insertion" and index == 20)
                        or (direction == "deletion" and index == 0)
                        else _sha(
                            f"intervention-{direction}-{repeat}-{index}"
                        )
                    )
                    for index in range(21)
                ],
                "curve_fractions": fractions,
                "unit_counts": unit_counts,
                "actual_fractions": actual_fractions,
                "responses": responses,
                "auc": _auc(actual_fractions, responses),
                "response_clipped": False,
                "auc_integration": (
                    "trapezoid_over_actual_k_divided_by_units"
                ),
                "seed_namespace": SEED_NAMESPACE,
            })
    return records, persisted


def _aggregates(
    records: list[dict[str, Any]], persisted: np.ndarray,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for direction_index in range(2):
        curve_records = records[
            direction_index * 5:(direction_index + 1) * 5
        ]
        matrix = persisted[direction_index]
        aucs = np.asarray([
            _auc(record["actual_fractions"], record["responses"])
            for record in curve_records
        ], dtype=np.float64)
        result.append({
            "status": "OK",
            "replicate_role": (
                "paired_nuisance_resolution_not_independent_samples"
            ),
            "replicate_count": 5,
            "standard_deviation": (
                "population_ddof_0_over_complete_five_replicates"
            ),
            "curves": curve_records,
            "response_mean": matrix.mean(axis=0).tolist(),
            "response_std": matrix.std(axis=0, ddof=0).tolist(),
            "auc_mean": float(aucs.mean()),
            "auc_std": float(aucs.std(ddof=0)),
        })
    return result


def _top20(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    paired_fields = (
        "method_id", "comparison_line", "target_ref",
        "target_identity_sha256", "ranking_kind", "sample_id",
        "block", "head", "replicate", "seed",
        "tie_priority_sha256", "input_pair_sha256", "ranking_sha256",
        "scale",
    )
    result: list[dict[str, Any]] = []
    for repeat in range(5):
        insertion = records[repeat]
        deletion = records[5 + repeat]
        row = {name: insertion[name] for name in paired_fields}
        row.update({
            "status": "OK",
            "fraction_requested": 0.2,
            "k": 628,
            "actual_fraction": 628 / 3136,
            "sufficiency": insertion["responses"][4],
            "sufficiency_gap": 1.0 - insertion["responses"][4],
            "comprehensiveness": 1.0 - deletion["responses"][4],
            "decision_gate": False,
        })
        result.append(row)
    return result


def _map_execution_records(
    unit: SimpleNamespace, provenance: dict[str, Any],
) -> list[dict[str, Any]]:
    nns = unit.method_id == NNS_METHOD_ID

    def record(input_variant: str, repeat_index: int | None) -> dict[str, Any]:
        backward_calls = 64 if nns else 0
        retained_ig = provenance.get("ig_result") if nns else None
        observed_ig = (
            {name: retained_ig[name] for name in (
                "selected_steps", "converged", "attempts", "metadata",
            )}
            if retained_ig is not None else None
        )
        return {
            "method_id": unit.method_id,
            "input_variant": input_variant,
            "repeat_index": repeat_index,
            "status": "completed",
            "ig_result": observed_ig,
            "attempted_steps_total": 512 if nns else 0,
            "backward_invocations": backward_calls,
            "actual_calls": {
                "accounting_unit": "batched_model_invocation",
                "forward_calls": 68 if nns else 0,
                "backward_calls": backward_calls,
            },
            "exception": None,
        }

    records = [record("clean", 0)]
    if unit.audit:
        records.append(record("repeat_clean", 1))
    return records


def _production_fixture(
    method_id: str, *, audit_unit: bool = False,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, dict[str, np.ndarray] | None, SimpleNamespace, dict[str, Any]]:
    unit = _unit(method_id, audit=audit_unit)
    metric = np.zeros((56, 56), dtype=np.float32)
    metric[0, 0] = np.float32(1.0)
    target = _target()
    target_identity = {
        "comparison_line": "value_path",
        "target_ref": unit.target_id,
        "clean_input_sha256": _sha("normalized-clean-input"),
        "frozen_target": target,
    }
    provenance = _provenance(method_id, target)
    records, persisted = _curve_records(unit, metric, target_identity)
    audit = (
        {"repeat_clean_metric_ready_map": metric.copy()}
        if audit_unit else None
    )
    repeatability = (
        [{
            "repeat_index": 1,
            "atol": REPEATABILITY_ATOL,
            "rtol": REPEATABILITY_RTOL,
            "max_abs_error": 0.0,
            "pass": True,
        }]
        if audit_unit else []
    )
    l0 = {
        "method_id": unit.method_id,
        "method_version": "1.0.0",
        "comparison_line": "value_path",
        "target_ref": unit.target_id,
        "status": "OK",
        "na": None,
        "projection": (
            "pixel_mass_to_patch_sum"
            if method_id == NNS_METHOD_ID
            else "patch_mass_to_pixel_uniform"
        ),
        "sign_semantics": "signed",
        "provenance_fragment": provenance,
        "sample_id": unit.sample_id,
        "prediction": {
            "clean_top1_index": 17, "official_truth_index": 17,
            "clean_top1_correct": True,
        },
        "stability": [],
        "map_execution_records": _map_execution_records(unit, provenance),
        "lifecycle": {
            "integrity": {
                "schema_version": 1,
                "algorithm": "synthetic-fixture",
                "checked": [],
                "entry_sha256": _sha("lifecycle"),
                "exit_sha256": _sha("lifecycle"),
                "matched": True,
                "restored": True,
                "differences": [],
                "signature_calls": 2,
                "verification_error_type": None,
                "verification_error_message": None,
                "body_exception_type": None,
            },
            "restoration": None,
        },
        "repeatability": repeatability,
        "fidelity_aggregates": _aggregates(records, persisted),
        "top20": _top20(records),
        "fidelity_curves": records,
        "perturbations": [],
        "target_identity": target_identity,
        "clean_tensor_hashes": {
            "canonical_rgb_sha256": _sha("canonical-rgb"),
            "normalized_tensor_sha256": target_identity[
                "clean_input_sha256"
            ],
        },
        "fidelity_enabled": True,
        "stability_enabled": False,
    }
    assert set(l0) == PRODUCTION_L0_KEYS
    return l0, metric, persisted, audit, unit, _attempt_provenance(unit)


def _validate(
    fixture: tuple[
        dict[str, Any], np.ndarray, np.ndarray,
        dict[str, np.ndarray] | None, SimpleNamespace, dict[str, Any],
    ],
):
    l0, metric, curves, audit, unit, attempt_provenance = fixture
    return validate_stage_d_scientific_evidence(
        l0, metric, curves, audit, unit,
        endpoint_parity_limit=ENDPOINT_LIMIT,
        repeatability_atol=REPEATABILITY_ATOL,
        repeatability_rtol=REPEATABILITY_RTOL,
        fidelity_seed_namespace=SEED_NAMESPACE,
        provenance=attempt_provenance,
    )


def _refresh_target_digest(l0: dict[str, Any]) -> None:
    digest = _identity_sha256(l0["target_identity"])
    for record in l0["fidelity_curves"]:
        record["target_identity_sha256"] = digest
    for aggregate in l0["fidelity_aggregates"]:
        for record in aggregate["curves"]:
            record["target_identity_sha256"] = digest
    for record in l0["top20"]:
        record["target_identity_sha256"] = digest


def _refresh_metric_digests(l0: dict[str, Any], metric: np.ndarray) -> None:
    metric_digest = _array_sha256(metric)
    score_digest = _array_sha256(
        np.maximum(metric, np.float32(0.0)).astype(np.float32, copy=False)
    )
    for record in l0["fidelity_curves"]:
        record["source_unit_map_sha256"] = metric_digest
        record["score_sha256"] = score_digest
        record["ranking_sha256"] = _ranking_sha256(
            metric, record["seed"]
        )
    for aggregate in l0["fidelity_aggregates"]:
        for record in aggregate["curves"]:
            record["source_unit_map_sha256"] = metric_digest
            record["score_sha256"] = score_digest
            record["ranking_sha256"] = _ranking_sha256(
                metric, record["seed"]
            )
    for record in l0["top20"]:
        record["ranking_sha256"] = _ranking_sha256(
            metric, record["seed"]
        )


def test_synthetic_envelope_is_exact_and_accepts_only_strictly_below_limit(
) -> None:
    unit = _unit(NNS_METHOD_ID)
    metric = np.zeros((56, 56), dtype=np.float32)
    curves = np.zeros((2, 5, 21), dtype=np.float64)

    accepted = validate_stage_d_scientific_evidence(
        {"endpoint_parity": math.nextafter(ENDPOINT_LIMIT, 0.0)},
        metric, curves, None, unit, endpoint_parity_limit=ENDPOINT_LIMIT,
    )
    extra_field = validate_stage_d_scientific_evidence(
        {"endpoint_parity": 0.0, "producer_claim": "PASS"},
        metric, curves, None, unit, endpoint_parity_limit=ENDPOINT_LIMIT,
    )

    assert accepted.valid and accepted.errors == ()
    assert accepted.pair_binding is None
    assert extra_field.errors == ("production_l0_envelope_invalid",)


@pytest.mark.parametrize("endpoint", [ENDPOINT_LIMIT, -ENDPOINT_LIMIT])
def test_synthetic_endpoint_boundary_is_strict(endpoint: float) -> None:
    verdict = validate_stage_d_scientific_evidence(
        {"endpoint_parity": endpoint},
        np.zeros((56, 56), dtype=np.float32),
        np.zeros((2, 5, 21), dtype=np.float64),
        None, _unit(NNS_METHOD_ID), endpoint_parity_limit=ENDPOINT_LIMIT,
    )

    assert verdict.errors == ("synthetic_endpoint_parity_failed",)


def test_complete_production_envelopes_validate_and_pair_without_values(
) -> None:
    nns = _validate(_production_fixture(NNS_METHOD_ID))
    matched = _validate(_production_fixture(MATCHED_METHOD_ID))

    assert nns.valid and nns.errors == ()
    assert matched.valid and matched.errors == ()
    assert set(nns.pair_binding or {}) == PAIR_BINDING_KEYS
    assert "responses" not in (nns.pair_binding or {})
    assert reconcile_stage_d_pair(nns, matched) == ()


@pytest.mark.parametrize("derived_mode", ["fast", "full"])
def test_legacy_l0_remains_valid_when_unit_has_derived_trace_mode(
    derived_mode: str,
) -> None:
    fixture = _production_fixture(NNS_METHOD_ID)
    fixture[4].trace_mode = derived_mode

    verdict = _validate(fixture)

    assert verdict.valid
    assert "trace_mode_missing" not in verdict.errors


def test_new_payload_trace_mode_must_match_unit_policy() -> None:
    fixture = _production_fixture(NNS_METHOD_ID)
    fixture[4].trace_mode = "full"
    fixture[0]["trace_mode"] = "fast"
    fixture[0].pop("provenance_fragment")
    fixture[0].pop("map_execution_records")

    assert "trace_mode_unit_mismatch" in _validate(fixture).errors


def _with_trace_mode(fixture, mode: str):
    l0, _metric, _curves, audit, unit, _provenance = fixture
    unit.trace_mode = mode
    l0["trace_mode"] = mode
    if mode == "fast":
        l0.pop("provenance_fragment")
        l0.pop("map_execution_records")
    else:
        contract_names = (
            "method_id", "method_version", "comparison_line",
            "target_ref", "projection", "sign_semantics",
        )
        l0["nns_contract"] = {name: l0[name] for name in contract_names}
        l0["trace"] = {
            "provenance_fragment": copy.deepcopy(l0["provenance_fragment"]),
            "map_execution_records": copy.deepcopy(l0["map_execution_records"]),
        }
        assert audit is not None
        audit["nns_trace"] = np.ones((3, 4, 4), dtype=np.float32)
    return fixture


def test_fast_nns_accepts_only_the_explicitly_omitted_trace_fields() -> None:
    fixture = _with_trace_mode(_production_fixture(NNS_METHOD_ID), "fast")
    assert _validate(fixture).valid

    fixture[0].pop("lifecycle")
    assert "production_l0_envelope_invalid" in _validate(fixture).errors


def test_full_nns_requires_and_validates_retained_trace() -> None:
    fixture = _with_trace_mode(
        _production_fixture(NNS_METHOD_ID, audit_unit=True), "full"
    )
    assert _validate(fixture).valid

    fixture[0].pop("trace")
    assert "production_l0_envelope_invalid" in _validate(fixture).errors

    fixture = _with_trace_mode(
        _production_fixture(NNS_METHOD_ID, audit_unit=True), "full"
    )
    assert fixture[3] is not None
    fixture[3].pop("nns_trace")
    assert "nns_l2_trace_missing_or_invalid" in _validate(fixture).errors


def test_production_endpoint_boundary_is_recomputed_not_trusted() -> None:
    fixture = _production_fixture(NNS_METHOD_ID)
    l0 = fixture[0]
    target = l0["provenance_fragment"]["target"]
    target["clean_value_max_abs_error"] = ENDPOINT_LIMIT
    l0["target_identity"]["frozen_target"] = target
    l0["provenance_fragment"]["endpoint_parity"][
        "clean_formula_max_abs_error"
    ] = ENDPOINT_LIMIT
    _refresh_target_digest(l0)

    assert _validate(fixture).errors == ("endpoint_parity_failed",)


def test_nns_recomputes_ig_attempt_and_retained_map_conservation() -> None:
    attempt_fixture = _production_fixture(NNS_METHOD_ID)
    attempt_l0 = attempt_fixture[0]
    ig_result = attempt_l0["provenance_fragment"]["ig_result"]
    ig_result["attempts"][0]["abs"] = [0.25]
    ig_result["conservation"]["abs"] = [0.25]
    assert _validate(attempt_fixture).errors == (
        "ig_attempt_conservation_invalid",
    )

    map_fixture = _production_fixture(NNS_METHOD_ID)
    map_l0, metric = map_fixture[0], map_fixture[1]
    metric[0, 1] = np.float32(0.25)
    _refresh_metric_digests(map_l0, metric)
    assert _validate(map_fixture).errors == ("ig_l1_conservation_invalid",)


def test_matched_recomputes_token_and_cls_l1_conservation() -> None:
    token_fixture = _production_fixture(MATCHED_METHOD_ID)
    token_record = token_fixture[0]["provenance_fragment"][
        "matched_contribution"
    ]
    token_record["cls_contribution"] = 0.0001
    token_record["token_sum"] = 1.0 + token_record["cls_contribution"]
    token_record["conservation_abs_error"] = abs(
        token_record["token_sum"]
        - token_record["scalar_delta_f_z_minus_f_x0"]
    )
    assert _validate(token_fixture).errors == (
        "matched_token_conservation_failed",
    )

    cls_fixture = _production_fixture(MATCHED_METHOD_ID)
    cls_fixture[0]["provenance_fragment"]["matched_contribution"][
        "cls_contribution"
    ] = 0.25
    assert _validate(cls_fixture).errors == (
        "matched_l1_conservation_invalid",
    )


@pytest.mark.parametrize(
    "bad_curves",
    [
        np.zeros((2, 5, 21), dtype=np.float32),
        np.zeros((2, 5, 20), dtype=np.float64),
    ],
    ids=("wrong-npz-dtype", "wrong-npz-shape"),
)
def test_decoded_curve_npz_requires_exact_dtype_and_shape(
    bad_curves: np.ndarray,
) -> None:
    fixture = list(_production_fixture(NNS_METHOD_ID))
    fixture[2] = bad_curves

    assert _validate(tuple(fixture)).errors == (
        "fidelity_curve_array_invalid",
    )


def test_curve_auc_and_npz_values_are_source_recomputed() -> None:
    auc_fixture = _production_fixture(NNS_METHOD_ID)
    auc_fixture[0]["fidelity_curves"][0]["auc"] += 0.01
    assert _validate(auc_fixture).errors == (
        "fidelity_curve_auc_invalid:0",
    )

    npz_fixture = _production_fixture(NNS_METHOD_ID)
    npz_fixture[2][0, 0, 10] += 0.125
    errors = _validate(npz_fixture).errors
    assert "fidelity_curve_l1_mismatch:0" in errors
    assert "fidelity_aggregate_response_mean_invalid:0" in errors
    assert "fidelity_aggregate_response_std_invalid:0" in errors


def test_curve_ranking_hash_is_recomputed_from_retained_map_and_seed() -> None:
    fixture = _production_fixture(NNS_METHOD_ID)
    replacement = _sha("self-consistent-but-false-ranking")
    for index in (0, 5):
        fixture[0]["fidelity_curves"][index]["ranking_sha256"] = (
            replacement
        )
    fixture[0]["top20"][0]["ranking_sha256"] = replacement

    assert _validate(fixture).errors == (
        "fidelity_curve_ranking_invalid:0",
        "fidelity_curve_ranking_invalid:5",
    )


def test_opaque_intervention_hashes_require_reversed_shared_endpoints() -> None:
    fixture = _production_fixture(NNS_METHOD_ID)
    replacement = _sha("self-consistent-but-unpaired-intervention")
    fixture[0]["fidelity_curves"][5]["intervention_sha256"][0] = (
        replacement
    )

    assert _validate(fixture).errors == (
        "fidelity_curve_intervention_endpoint_binding_invalid:5",
    )


def test_opaque_intervention_interiors_are_not_claimed_as_recomputed() -> None:
    fixture = _production_fixture(NNS_METHOD_ID)
    fixture[0]["fidelity_curves"][0]["intervention_sha256"][10] = (
        _sha("different-opaque-interior")
    )

    assert _validate(fixture).valid


def test_audit_repeat_map_is_recomputed_at_source_tolerances() -> None:
    valid_fixture = _production_fixture(NNS_METHOD_ID, audit_unit=True)
    assert _validate(valid_fixture).valid

    tampered_fixture = _production_fixture(NNS_METHOD_ID, audit_unit=True)
    l0, metric, _curves, audit, _unit_value, _provenance_value = (
        tampered_fixture
    )
    assert audit is not None
    audit["repeat_clean_metric_ready_map"][0, 1] = np.float32(2e-6)
    difference = np.abs(
        metric.astype(np.float64)
        - audit["repeat_clean_metric_ready_map"].astype(np.float64)
    )
    l0["repeatability"][0]["max_abs_error"] = float(difference.max())
    l0["repeatability"][0]["pass"] = False

    assert _validate(tampered_fixture).errors == (
        "audit_repeatability_failed",
    )


def test_input_pair_hash_is_recomputed_from_retained_clean_binding() -> None:
    fixture = _production_fixture(MATCHED_METHOD_ID)
    replacement = _sha("substituted-input-pair")
    for record in fixture[0]["fidelity_curves"]:
        record["input_pair_sha256"] = replacement
    for record in fixture[0]["top20"]:
        record["input_pair_sha256"] = replacement

    assert _validate(fixture).errors == tuple(
        f"fidelity_curve_input_pair_binding_invalid:{index}"
        for index in range(10)
    )


def test_attempt_provenance_is_distinct_from_scientific_fragment() -> None:
    fixture = _production_fixture(NNS_METHOD_ID)
    fixture[-1]["bindings"]["scientific_contract"]["baseline"] = {
        "id": "tampered-baseline"
    }
    _resign_attempt_provenance(fixture[-1])

    assert _validate(fixture).errors == (
        "attempt_provenance_scientific_contract_invalid",
    )


def test_invalid_member_cannot_reconcile_as_a_valid_pair() -> None:
    nns = _validate(_production_fixture(NNS_METHOD_ID))
    matched_fixture = _production_fixture(MATCHED_METHOD_ID)
    matched_fixture[0]["method_version"] = "tampered"
    matched = _validate(matched_fixture)

    assert not matched.valid
    assert reconcile_stage_d_pair(nns, matched) == (
        "pair_member_scientifically_invalid",
    )


def test_prediction_requires_the_closed_production_shape() -> None:
    fixture = _production_fixture(NNS_METHOD_ID)
    fixture[0]["prediction"]["unregistered"] = True

    assert _validate(fixture).errors == ("l0_scope_or_identity_invalid",)


def test_nns_map_execution_rejects_old_forward_overhead() -> None:
    fixture = _production_fixture(NNS_METHOD_ID)
    fixture[0]["map_execution_records"][0]["actual_calls"][
        "forward_calls"
    ] = 66

    assert _validate(fixture).errors == (
        "map_execution_ig_accounting_invalid",
    )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda metadata: metadata.pop("intervention_calls"),
        lambda metadata: metadata.update({"pre_projection_calls": 65}),
    ],
    ids=("missing-intervention-calls", "wrong-pre-projection-calls"),
)
def test_nns_map_execution_requires_closed_success_call_metadata(
    mutate,
) -> None:
    fixture = _production_fixture(NNS_METHOD_ID)
    metadata = fixture[0]["map_execution_records"][0]["ig_result"][
        "metadata"
    ]
    mutate(metadata)

    assert _validate(fixture).errors == (
        "map_execution_ig_accounting_invalid",
    )


def _drop_forward_calls(record: dict[str, Any]) -> None:
    record["actual_calls"].pop("forward_calls")


def _string_backward_calls(record: dict[str, Any]) -> None:
    record["actual_calls"]["backward_calls"] = "64"
    record["backward_invocations"] = "64"


def _non_mapping_calls(record: dict[str, Any]) -> None:
    record["actual_calls"] = []


def _string_ig_step(record: dict[str, Any]) -> None:
    record["ig_result"]["attempts"][0]["steps"] = "512"


def _non_list_ig_attempts(record: dict[str, Any]) -> None:
    record["ig_result"]["attempts"] = {"steps": 512}


def _non_mapping_ig_metadata(record: dict[str, Any]) -> None:
    record["ig_result"]["metadata"] = []


@pytest.mark.parametrize(
    ("mutate", "expected_errors"),
    [
        (_drop_forward_calls, (
            "map_execution_call_accounting_invalid",
            "map_execution_ig_accounting_invalid",
        )),
        (_string_backward_calls, (
            "map_execution_call_accounting_invalid",
            "map_execution_ig_accounting_invalid",
        )),
        (_non_mapping_calls, (
            "map_execution_call_accounting_invalid",
            "map_execution_ig_accounting_invalid",
        )),
        (_string_ig_step, (
            "ig_attempt_record_invalid",
            "map_execution_ig_evidence_missing",
        )),
        (_non_list_ig_attempts, (
            "map_execution_ig_evidence_missing",
        )),
        (_non_mapping_ig_metadata, ("map_execution_ig_evidence_missing",)),
    ],
    ids=(
        "calls-missing-field", "calls-string-count", "calls-not-mapping",
        "steps-string", "attempts-not-list", "metadata-not-mapping",
    ),
)
def test_malformed_execution_evidence_returns_deterministic_errors(
    mutate, expected_errors: tuple[str, ...],
) -> None:
    fixture = _production_fixture(NNS_METHOD_ID)
    mutate(fixture[0]["map_execution_records"][0])

    assert _validate(fixture).errors == expected_errors
