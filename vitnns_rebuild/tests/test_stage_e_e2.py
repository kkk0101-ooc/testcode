"""E2 frozen scope, operator, execution, source validation, and false-PASS tests."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np
import pytest
from jsonschema import Draft202012Validator

import vitnns.stage_e_e2 as e2
from vitnns.stage_d_provenance import canonical_sha256

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_frozen_config_schema_scope_budget_and_stage_boundary() -> None:
    config = e2._validate_config(REPO_ROOT)
    assert config["canonical_digest"] == e2.EXPECTED_CONFIG_DIGEST
    assert tuple(row["variant_id"] for row in config["scope"]["variants"]) == e2.VARIANTS
    assert tuple(tuple(row) for row in config["scientific_contract"]["pairwise_order"]) == e2.PAIR_ORDER
    assert config["scope"]["expected_new_units"] == 1000
    assert config["scope"]["expected_variant_rows"] == 2000
    assert config["scope"]["expected_pairwise_rows"] == 3000
    assert config["authorization"] == {
        "requires": "E1_PASS_E2_ONLY", "current_stage": "E2_ONLY",
        "on_pass": "E3_ONLY", "e3_started": False,
        "e4_started": False, "stage_f_started": False,
    }
    assert config["budget"] == {"gpu_hours_max": 2.5, "cpu_hours_max": 8.0, "retained_bytes_max": 4294967296}


def test_e1_terminal_archive_status_inventory_and_source_validation() -> None:
    completion = json.loads((REPO_ROOT / e2.E1_COMPLETION_PATH).read_text())
    status = json.loads((REPO_ROOT / e2.E1_STATUS_PATH).read_text())
    inventory = json.loads((REPO_ROOT / e2.E1_INVENTORY_PATH).read_text())
    assert completion["canonical_digest"] == e2.E1_COMPLETION_DIGEST
    assert completion["next_authorization"] == "E2_ONLY"
    assert status["canonical_digest"] == e2.E1_STATUS_DIGEST
    assert inventory["canonical_digest"] == e2.E1_INVENTORY_DIGEST
    assert canonical_sha256({"files": inventory["files"]}) == e2.E1_FILE_PROJECTION_DIGEST
    assert e2._git(REPO_ROOT, "rev-parse", "refs/archive/stage-e-e1-completion^{commit}") == e2.E1_COMMIT


def test_typed_plan_registry_and_exact_compute_policy() -> None:
    plan = e2._build_plan(REPO_ROOT, e2._validate_config(REPO_ROOT))
    assert len(plan["groups"]) == 500 and len(plan["units"]) == 1000
    assert [row["selection_order"] for row in plan["groups"]] == list(range(500))
    assert len({row["sample_id"] for row in plan["groups"]}) == 500
    assert {row["variant_id"] for row in plan["units"]} == set(e2.NEW_VARIANTS)
    assert all(row["scale"] == "fine_4x4_cell" for row in plan["units"])
    assert e2.VARIANTS[:2] == ("uniform_mass_divide_by_16", "mass_preserving_nearest")


def test_operator_shape_modes_digest_basis_conservation_and_equivalence() -> None:
    result = e2.operator_golden()
    assert result["status"] == "PASS" and result["basis_count"] == 196
    for variant in e2.VARIANTS:
        matrix = e2.operator(variant)
        assert matrix.shape == (3136, 196) and matrix.dtype == np.float64
        assert np.isfinite(matrix).all()
        assert np.max(np.abs(matrix.sum(axis=0) - 1.0)) <= 2e-12
        assert result["operators"][variant]["operator_sha256"] == e2.OPERATOR_DIGESTS[variant]
    assert np.array_equal(e2.operator(e2.VARIANTS[0]), e2.operator(e2.VARIANTS[1]))
    assert e2.operator("bicubic").min() < 0.0


def test_all_500_native_recoveries_hash_conservation_maps_and_equivalence() -> None:
    plan = e2._build_plan(REPO_ROOT, e2._validate_config(REPO_ROOT))
    for group in plan["groups"]:
        unit = {key: group[key] for key in ("source_map_path", "source_map_sha256", "source_result_path", "source_result_sha256")}
        result, source_map, native = e2._load_source(REPO_ROOT, unit)
        assert native.shape == (14, 14) and np.isfinite(native).all()
        uniform = e2.transformed_map(native, e2.VARIANTS[0])
        nearest = e2.transformed_map(native, e2.VARIANTS[1])
        assert np.array_equal(uniform, nearest) and np.array_equal(uniform, source_map.astype(np.float64))
        ranked = e2.ranking(uniform, seed=e2.d9_seed(group["sample_id"], 0))
        assert ranked.shape == (3136,) and set(ranked.tolist()) == set(range(3136))
        for variant in e2.VARIANTS[2:]:
            transformed = e2.transformed_map(native, variant)
            assert abs(float(transformed.sum()) - float(native.sum())) <= 1e-12 + 1e-12 * abs(float(native.sum()))


def test_d9_curves_seed_tie_ranking_and_uniform_reuse_are_source_validated() -> None:
    plan = e2._build_plan(REPO_ROOT, e2._validate_config(REPO_ROOT))
    for group in plan["groups"][:5]:
        unit = {key: group[key] for key in ("source_map_path", "source_map_sha256", "source_result_path", "source_result_sha256")}
        result, source_map, _ = e2._load_source(REPO_ROOT, unit)
        curves = e2._validate_d9_curves(group["sample_id"], result, source_map)
        assert len(curves) == 10
        for curve in curves:
            assert curve["seed"] == e2.d9_seed(group["sample_id"], curve["replicate"])
            assert curve["seed_namespace"] == e2.SEED_NAMESPACE


def test_signed_transform_positive_part_tie_and_fine_cell_geometry() -> None:
    signed = np.zeros((56, 56), dtype=np.float64); signed[0, 0] = -100.0; signed[0, 1] = 1.0
    ranked = e2.ranking(signed, seed=7)
    assert ranked[0] == 1 and signed[0, 0] == -100.0
    assert e2.UNIT_COUNTS == (0, 157, 314, 471, 628, 784, 941, 1098, 1255, 1412, 1568, 1725, 1882, 2039, 2196, 2352, 2509, 2666, 2823, 2980, 3136)
    selected = np.zeros(3136, dtype=np.bool_); selected[ranked[:157]] = True
    mask = np.repeat(np.repeat(selected.reshape(56, 56), 4, axis=0), 4, axis=1)
    assert mask.shape == (224, 224) and int(mask.sum()) == 157 * 16


def test_auc_and_six_pairwise_delta_synthetic_golden() -> None:
    assert e2.trapezoid_auc(e2.ACTUAL_FRACTIONS) == pytest.approx(0.5)
    rows = {variant: {"insertion_auc_mean": 0.8 - index * 0.1, "deletion_auc_mean": 0.2 + index * 0.1} for index, variant in enumerate(e2.VARIANTS)}
    values = [e2.delta_fid(rows[a], rows[b]) for a, b in e2.PAIR_ORDER]
    assert values == pytest.approx([0.1, 0.2, 0.3, 0.1, 0.2, 0.1])


@pytest.mark.parametrize("mutation", [
    lambda value: value["scope"]["variants"].pop(),
    lambda value: value["scope"].update({"expected_new_units": 999}),
    lambda value: value["scientific_contract"].update({"unit_count": 196}),
    lambda value: value["scientific_contract"].update({"tie_seed_namespace": "variant-specific"}),
    lambda value: value["scientific_contract"].update({"p_values_emitted": True}),
    lambda value: value["authorization"].update({"on_pass": "E4_ONLY"}),
    lambda value: value["budget"].update({"gpu_hours_max": 3.0}),
])
def test_scope_seed_decision_or_budget_drift_cannot_validate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation) -> None:
    source = json.loads((REPO_ROOT / e2.CONFIG_PATH).read_text()); mutation(source)
    source["canonical_digest"] = canonical_sha256({key: value for key, value in source.items() if key != "canonical_digest"})
    path = tmp_path / "stage_e_e2.json"; path.write_text(json.dumps(source) + "\n")
    monkeypatch.setattr(e2, "CONFIG_PATH", str(path))
    with pytest.raises(e2.StageEE2Error): e2._validate_config(REPO_ROOT)


def test_uniform_and_nearest_have_no_execution_units_or_gpu_attempt_surface() -> None:
    plan = e2._build_plan(REPO_ROOT, e2._validate_config(REPO_ROOT))
    assert all(row["variant_id"] in e2.NEW_VARIANTS for row in plan["units"])
    with pytest.raises(e2.e0.StageEE0Error): e2.operator("lanczos")


def test_attempt_placeholder_missing_curve_or_wrong_variant_cannot_pass(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(e2, "ROOT_PATH", "formal")
    unit = {"canonical_digest": "a" * 64, "variant_id": e2.VARIANTS[0], "sample_id": "x"}
    root = tmp_path / "formal/units" / ("a" * 64) / "attempts/attempt-000001"; root.mkdir(parents=True)
    result = e2._signed({"unit_digest": "a" * 64, "variant_id": e2.VARIANTS[0], "sample_id": "x", "validity_status": "VALID", "curves": []})
    raw = e2.canonical_json_bytes(result) + b"\n"; (root / "result.json").write_bytes(raw)
    status = e2._signed({"terminal": True, "attempt_id": "attempt-000001"}); sraw = e2.canonical_json_bytes(status) + b"\n"; (root / "status.json").write_bytes(sraw)
    manifest = e2._signed({"result_sha256": e2._sha(raw), "status_sha256": e2._sha(sraw)}); (root / "manifest.json").write_bytes(e2.canonical_json_bytes(manifest) + b"\n")
    with pytest.raises(e2.StageEE2Error): e2._validate_attempt(tmp_path, unit)
    with pytest.raises(e2.StageEE2Error): e2.trapezoid_auc([math.nan] * 21)


def test_budget_and_false_pass_guards_reject_every_attack() -> None:
    assert e2.budget_reasons({"gpu_hours": 2.500001, "cpu_seconds": 0}, 0) == ("gpu_hours",)
    assert e2.budget_reasons({"gpu_hours": 0, "cpu_seconds": 8 * 3600 + 1}, 0) == ("cpu_hours",)
    assert e2.budget_reasons({"gpu_hours": 0, "cpu_seconds": 0}, 4294967297) == ("retained_bytes",)
    valid = {name: True for name in e2.FALSE_PASS_ATTACKS}
    assert e2.false_pass_reasons(valid) == ()
    for name in e2.FALSE_PASS_ATTACKS:
        mutated = dict(valid); mutated[name] = False
        assert e2.false_pass_reasons(mutated) == (name,)


def test_schemas_render_boundary_and_no_later_stage_execution_surface() -> None:
    for name in (e2.CONFIG_SCHEMA_PATH, e2.OUTPUT_SCHEMA_PATH, "config/stage_e_stage_completion.schema.json"):
        schema = json.loads((REPO_ROOT / name).read_text()); Draft202012Validator.check_schema(schema)
    config = e2._validate_config(REPO_ROOT)
    assert all(config["execution"][key] is False for key in ("renders_png", "renders_pdf", "renders_svg"))
    assert config["scientific_contract"]["winner_selection_allowed"] is False
    assert config["scientific_contract"]["new_go_no_go_allowed"] is False
    public = {name for name in dir(e2) if name.startswith(("run_", "start_", "execute_"))}
    assert public == {"run_shard"}


def test_completion_if_present_is_atomic_and_authorizes_only_e3() -> None:
    original = REPO_ROOT / e2.COMPLETION_PATH
    superseding = REPO_ROOT / e2.SUPERSEDING_COMPLETION_PATH
    assert not (REPO_ROOT / e2.COMPLETION_PENDING_PATH).exists()
    assert not (REPO_ROOT / e2.SUPERSEDING_COMPLETION_PENDING_PATH).exists()
    if superseding.exists():
        value = json.loads(superseding.read_text())
        assert value["status"] == "PASS" and value["next_authorization"] == "E3_ONLY"
    elif original.exists():
        value = json.loads(original.read_text())
        assert value["status"] == "BLOCKED" and value["next_authorization"] == "NONE"
    else:
        assert not (REPO_ROOT / e2.ROOT_PATH).exists()


def test_budget_waiver_does_not_erase_actual_cost_or_scientific_gates() -> None:
    waiver_path = REPO_ROOT / e2.BUDGET_WAIVER_PATH
    if not waiver_path.exists():
        pytest.skip("owner budget waiver not yet recorded")
    waiver = json.loads(waiver_path.read_text())
    assert waiver["waiver_scope"] == "E2_GPU_AND_CPU_HARD_BUDGET_ACCEPTANCE_ONLY"
    assert waiver["waived_failed_limits"] == ["gpu_hours", "cpu_hours"]
    assert waiver["actual_cost_must_remain_recorded_as_fail_cost"] is True
    assert waiver["scientific_or_integrity_gate_waived"] is False
    assert waiver["terminal_attempt_recompute_authorized"] is False
    assert waiver["e3_execution_authorized_in_this_task"] is False
    assert waiver["supersedes_blocker_only"]["completion_canonical_digest"] == e2.BLOCKED_COMPLETION_DIGEST


def test_original_blocked_receipt_is_immutable_after_waiver() -> None:
    completion = REPO_ROOT / e2.COMPLETION_PATH
    if not completion.exists():
        pytest.skip("original BLOCKED completion not yet available")
    raw = completion.read_bytes(); value = json.loads(raw)
    assert e2._sha(raw) == e2.BLOCKED_COMPLETION_SHA256
    assert value["canonical_digest"] == e2.BLOCKED_COMPLETION_DIGEST
    assert value["status"] == "BLOCKED"
    assert value["cost_status"] == "FAIL_COST"
    assert value["next_authorization"] == "NONE"


def test_superseding_completion_if_present_requires_waiver_and_starts_no_e3() -> None:
    completion = REPO_ROOT / e2.SUPERSEDING_COMPLETION_PATH
    if not completion.exists():
        pytest.skip("superseding completion not yet available")
    value = e2.validate_superseding_completion(REPO_ROOT)
    assert value["status"] == "PASS"
    assert value["completion_basis"] == "PASS_WITH_EXPLICIT_OWNER_BUDGET_WAIVER"
    assert value["next_authorization"] == "E3_ONLY"
    assert value["e3_started"] is False
