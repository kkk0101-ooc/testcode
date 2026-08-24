from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from vitnns.constants import REPO_ROOT, WORKSPACE_ROOT
from vitnns.io_utils import sha256_file
from vitnns.stage_c_config import load_stage_c_config
from vitnns.stage_c_plan import (
    BudgetExceededError,
    ExecutionPlan,
    RegistryContractError,
    RunSpec,
    RunSpecError,
    SelectionContractError,
    SelectionState,
    StageCEstimate,
    UnitKey,
    ValueCandidate,
    build_execution_plan,
    enforce_stage_c_budget,
    estimate_stage_c_run,
    load_run_spec,
    load_stage_c_registry,
    load_stage_c_selection,
    make_run_spec,
    unit_key_digest,
)


CONFIG = load_stage_c_config()
REGISTRY_PATH = REPO_ROOT / CONFIG.registry.path
METADATA_PATH = REPO_ROOT / CONFIG.registry.metadata_path
REGISTRY_SCHEMA = REPO_ROOT / "config/stage_c_registry.schema.json"
SELECTION_SCHEMA = REPO_ROOT / "config/stage_c_selection.schema.json"
CLASS_METHODS = (
    "direct_raw_cls_attention_last_block_head_mean",
    "attention_rollout_mean_head_residual_all_blocks",
    "vit_grad_cam_block11_norm1",
    "full_model_class_logit_ig",
)
VALUE_METHODS = (
    "direct_raw_cls_attention_per_head",
    "conditional_fixed_clean_attention_head_direction_value_path_ig",
    "matched_fixed_clean_attention_patch_token_contribution",
)
ALL_METHODS = VALUE_METHODS + CLASS_METHODS
PAIR = {"block_index": 0, "head_index": 0}


@pytest.fixture(scope="module")
def registry():
    return load_stage_c_registry(CONFIG)


def _spec(
    registry, *, sample_ids=None, method_ids=ALL_METHODS,
    value_config_source="explicit", value_candidates=(PAIR,),
    fidelity=True, sanity=True, stability=True, repeatability=2,
):
    return make_run_spec(
        sample_ids=sample_ids or [registry.rows[0].sample_id],
        method_ids=method_ids, value_config_source=value_config_source,
        value_candidates=value_candidates,
        fidelity_pairing_context=PAIR if fidelity or sanity else None,
        audit_repeatability_runs=repeatability,
        audit_repeatability_scope="clean_map_only",
        run_registered_fidelity=fidelity, run_sanity=sanity,
        run_stability=stability,
    )


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _selection_document() -> dict:
    zero = "0" * 64
    candidates = []
    for block in range(12):
        for head in range(12):
            key = f"block_{block:02d}_head_{head:02d}"
            candidates.append({
                "candidate_key": key, "block_index": block,
                "head_index": head, "depth_band": block // 3,
                "calibration_sample_count": 5, "eligible": True,
                "validity_failures": [],
                "count_delta_fid_fine4_gt_zero": 5,
                "median_delta_fid_fine4": 1.0 - block / 100 - head / 10000,
                "minimum_perturbation_median_residual_spearman": 0.6,
                "median_patch_residual_l1_ratio": 0.1,
                "rounded_rank_tuple": [
                    5, round(1.0 - block / 100 - head / 10000, 6),
                    0.6, 0.1, -block, -head,
                ],
                "resource_within_hard_budget": True,
                "artifact_manifest_sha256": zero,
                "rank_within_depth_band": head + 1 + (block % 3) * 12,
                "selected_depth_band_winner": head == 0 and block % 3 == 0,
                "selected_primary": block == 0 and head == 0,
            })
    refs = [
        {"candidate_key": f"block_{block:02d}_head_00",
         "block_index": block, "head_index": 0, "depth_band": block // 3}
        for block in (0, 3, 6, 9)
    ]
    protocol = REPO_ROOT / CONFIG.protocol.path
    return {
        "schema_version": 1, "document_type": "stage_c_selection",
        "generated_by": "test", "holdout_accessed": False,
        "input_hashes": {
            "code_commit": "0" * 40, "code_tree_clean": True,
            "model_sha256": __import__("vitnns.constants", fromlist=["MODEL_SHA256"]).MODEL_SHA256,
            "dependency_lock_sha256": sha256_file(REPO_ROOT / "requirements.lock.txt"),
            "registry_sha256": CONFIG.registry.sha256,
            "registry_metadata_sha256": CONFIG.registry.metadata_sha256,
            "protocol_sha256": sha256_file(protocol),
            "acceptance_sha256": sha256_file(REPO_ROOT / "docs/stage_c_acceptance.md"),
            "config_sha256": sha256_file(REPO_ROOT / "config/stage_c.json"),
            "config_schema_sha256": sha256_file(REPO_ROOT / "config/stage_c.schema.json"),
            "calibration_artifact_manifest_sha256": zero,
        },
        "calibration_sample_ids": [
            row.sample_id for row in load_stage_c_registry(CONFIG).rows
            if row.role == "calibration"
        ],
        "candidate_records": candidates, "depth_band_winners": refs,
        "primary": refs[0], "failure_records": [],
    }


def test_registry_strict_load_hash_types_and_optional_live_assets(registry):
    assert len(registry.rows) == 20
    assert registry.live_assets_verified is False
    assert sum(row.role == "calibration" for row in registry.rows) == 5
    assert sum(row.role == "holdout" for row in registry.rows) == 15
    assert sum(row.audit for row in registry.rows) == 2
    assert all(type(row.size_bytes) is int and type(row.audit) is bool for row in registry.rows)
    live = load_stage_c_registry(CONFIG, verify_live_assets=True)
    assert live.live_assets_verified is True
    assert Path(live.data_root).resolve() == (WORKSPACE_ROOT / "49_vit /datas/ILSVRC2012_val").resolve()
    assert type(live.to_dict()["rows"][0]["audit"]) is bool


def test_registry_rejects_header_lexical_integer_and_config_hash(tmp_path: Path):
    text = REGISTRY_PATH.read_text(encoding="utf-8")
    bad_header = tmp_path / "bad_header.csv"
    bad_header.write_text(text.replace("schema_version,sample_id", "sample_id,schema_version", 1), encoding="utf-8")
    with pytest.raises(RegistryContractError, match="header"):
        load_stage_c_registry(CONFIG, registry_path=bad_header)

    bad_integer = tmp_path / "bad_integer.csv"
    bad_integer.write_text(text.replace("1,ILSVRC", "01,ILSVRC", 1), encoding="utf-8")
    with pytest.raises(RegistryContractError, match="canonical unsigned integer"):
        load_stage_c_registry(CONFIG, registry_path=bad_integer)

    copied = tmp_path / "copy.csv"
    copied.write_text(text, encoding="utf-8")
    assert sha256_file(copied) == CONFIG.registry.sha256
    tampered = text.replace("calibration,true", "calibration,false", 1)
    copied.write_text(tampered, encoding="utf-8")
    with pytest.raises(RegistryContractError, match="SHA-256"):
        load_stage_c_registry(CONFIG, registry_path=copied)

    schema = tmp_path / "registry.schema.json"
    schema.write_text(REGISTRY_SCHEMA.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(RegistryContractError, match="schema SHA-256"):
        load_stage_c_registry(CONFIG, registry_schema_path=schema)


def test_registry_live_asset_tamper_is_optional_and_detected(tmp_path: Path):
    # Point at a complete-looking but invalid tree.  Loading the registry alone
    # stays a cheap metadata operation; requested live validation fails closed.
    snapshot = load_stage_c_registry(CONFIG, data_root=tmp_path, verify_live_assets=False)
    assert snapshot.live_assets_verified is False
    with pytest.raises(RegistryContractError, match="live image"):
        load_stage_c_registry(CONFIG, data_root=tmp_path, verify_live_assets=True)


def test_selection_missing_is_explicit_and_required_fails(tmp_path: Path):
    missing_path = tmp_path / "nonexistent-selection.json"
    state = load_stage_c_selection(CONFIG, selection_path=missing_path)
    assert state.to_dict() == {
        "schema_version": 1, "status": "not_applicable_pre_c8",
        "path": "selection/stage_c_selection.json",
        "reason": (
            "C8 selection has not been produced; only calibration "
            "explicit/all-144 scope is legal"
        ),
        "sha256": None, "primary": None, "depth_band_winners": [],
    }
    assert SelectionState.from_dict(state.to_dict()) == state
    with pytest.raises(SelectionContractError, match="required.*missing"):
        load_stage_c_selection(
            CONFIG, selection_path=missing_path, required=True
        )


def test_default_selection_loads_frozen_c8_machine_file():
    state = load_stage_c_selection(CONFIG, required=True)
    winners = tuple(
        ValueCandidate(block, head)
        for block, head in ((0, 9), (4, 2), (6, 10), (11, 2))
    )
    assert state.status == "present" and state.reason is None
    assert state.path == "selection/stage_c_selection.json"
    assert state.sha256 == (
        "5dff9b5aec77cab22b157b9bf2b5d63b2252b7e50ebe728250206d655a142254"
    )
    assert state.primary == ValueCandidate(0, 9)
    assert state.depth_band_winners == winners
    assert state.primary in state.depth_band_winners
    assert SelectionState.from_dict(state.to_dict()) == state


def test_selection_strict_schema_semantics_hashes_and_round_trip(tmp_path: Path):
    document = _selection_document()
    path = tmp_path / "selection.json"
    _write_json(path, document)
    state = load_stage_c_selection(CONFIG, selection_path=path)
    assert state.status == "present" and state.reason is None
    assert state.primary == ValueCandidate(0, 0)
    assert state.depth_band_winners == tuple(ValueCandidate(block, 0) for block in (0, 3, 6, 9))
    assert SelectionState.from_dict(state.to_dict()) == state

    duplicate = json.loads(json.dumps(document))
    duplicate["candidate_records"][1]["block_index"] = 0
    duplicate["candidate_records"][1]["head_index"] = 0
    duplicate["candidate_records"][1]["candidate_key"] = "block_00_head_00"
    duplicate["candidate_records"][1]["depth_band"] = 0
    duplicate["candidate_records"][1]["rounded_rank_tuple"][-2:] = [0, 0]
    _write_json(path, duplicate)
    with pytest.raises(SelectionContractError, match="exactly once"):
        load_stage_c_selection(CONFIG, selection_path=path)

    bad_hash = json.loads(json.dumps(document))
    bad_hash["input_hashes"]["registry_sha256"] = "f" * 64
    _write_json(path, bad_hash)
    with pytest.raises(SelectionContractError, match="hash binding mismatch"):
        load_stage_c_selection(CONFIG, selection_path=path)

    _write_json(path, document)
    schema_copy = tmp_path / "selection.schema.json"
    schema_copy.write_text(
        SELECTION_SCHEMA.read_text(encoding="utf-8") + " ", encoding="utf-8"
    )
    with pytest.raises(SelectionContractError, match="schema SHA-256"):
        load_stage_c_selection(CONFIG, selection_path=path, schema_path=schema_copy)

    with pytest.raises(SelectionContractError, match="cannot override static"):
        load_stage_c_selection(
            CONFIG, selection_path=path,
            expected_input_hashes={"registry_sha256": "f" * 64},
        )


def test_selection_rank_winner_and_failure_semantics_fail_closed(tmp_path: Path):
    path = tmp_path / "selection.json"
    document = _selection_document()
    document["candidate_records"][0]["rank_within_depth_band"] = 2
    _write_json(path, document)
    with pytest.raises(SelectionContractError, match="ranks"):
        load_stage_c_selection(CONFIG, selection_path=path)

    document = _selection_document()
    document["primary"] = document["depth_band_winners"][1]
    document["candidate_records"][0]["selected_primary"] = False
    document["candidate_records"][36]["selected_primary"] = True
    _write_json(path, document)
    with pytest.raises(SelectionContractError, match="primary"):
        load_stage_c_selection(CONFIG, selection_path=path)

    document = _selection_document()
    document["failure_records"] = [{
        "candidate_key": "block_00_head_00",
        "sample_id": "ILSVRC2012_val/ILSVRC2012_val_00002607.JPEG",
        "reason": "not calibration",
    }]
    _write_json(path, document)
    with pytest.raises(SelectionContractError, match="non-calibration"):
        load_stage_c_selection(CONFIG, selection_path=path)


def test_run_spec_is_closed_explicit_and_round_trips(registry, tmp_path: Path):
    spec = _spec(registry)
    assert RunSpec.from_dict(spec.to_dict()) == spec
    path = tmp_path / "run.json"
    _write_json(path, spec.to_dict())
    assert load_run_spec(path) == spec

    unknown = spec.to_dict()
    unknown["default_head"] = 0
    _write_json(path, unknown)
    with pytest.raises(RunSpecError, match="unknown=.*default_head"):
        load_run_spec(path)

    with pytest.raises(RunSpecError, match="pairing context"):
        make_run_spec(
            sample_ids=[registry.rows[0].sample_id], method_ids=CLASS_METHODS,
            value_config_source="none", value_candidates=[],
            fidelity_pairing_context=None, audit_repeatability_runs=2,
            audit_repeatability_scope="clean_map_only",
            run_registered_fidelity=True, run_sanity=True, run_stability=True,
        )


def test_unit_key_global_has_no_fake_config_and_digest_is_stable(registry):
    spec = _spec(
        registry, method_ids=CLASS_METHODS, value_config_source="none",
        value_candidates=(),
    )
    plan = build_execution_plan(CONFIG, registry, spec)
    assert len(plan.units) == 4
    for unit in plan.units:
        key = unit.key
        assert key.block_index is None and key.head_index is None
        assert key.config_id == f"global_{key.method_id}"
        assert key.canonical_digest == unit_key_digest(
            sample_id=key.sample_id, method_id=key.method_id,
            config_id=key.config_id, block_index=None, head_index=None,
            target_ref=key.target_ref,
        )
        assert UnitKey.from_dict(key.to_dict()) == key
    assert ExecutionPlan.from_dict(plan.to_dict()).to_dict() == plan.to_dict()


def test_plan_expands_exact_one_candidate_counts_and_structural_raw(registry):
    plan = build_execution_plan(CONFIG, registry, _spec(registry))
    assert len(plan.units) == 7
    by_method = {unit.key.method_id: unit for unit in plan.units}
    raw = by_method[VALUE_METHODS[0]]
    assert raw.work_item_counts == {
        "work_items": 5, "map_runs": 5, "clean_map_runs": 1,
        "perturbation_map_runs": 3, "repeatability_map_runs": 1,
        "fidelity_curves": 0, "fidelity_curve_points": 0,
        "target_tie_curve_points": 0, "random_sanity_curve_points": 0,
        "center_sanity_curve_points": 0, "logical_scalar_evaluations": 0,
    }
    assert raw.key.target_ref is None and raw.structural_reference_only
    for method_id in ALL_METHODS[1:]:
        counts = by_method[method_id].work_item_counts
        assert counts["map_runs"] == 5
        assert counts["fidelity_curves"] == 44
        assert counts["fidelity_curve_points"] == 924
        assert counts["target_tie_curve_points"] == 420
        assert counts["random_sanity_curve_points"] == 420
        assert counts["center_sanity_curve_points"] == 84


def test_pre_c8_all_144_exact_counts_and_holdout_fails_closed(registry):
    calibration_ids = [row.sample_id for row in registry.rows if row.role == "calibration"]
    spec = _spec(
        registry, sample_ids=calibration_ids, value_config_source="calibration_all_144",
        value_candidates=(),
    )
    plan = build_execution_plan(CONFIG, registry, spec)
    estimate = estimate_stage_c_run(CONFIG, plan)
    counts = estimate.to_dict()["counts"]
    assert counts["execution_units"] == 2180
    assert counts["fidelity_eligible_units"] == 1460
    assert counts["structural_reference_units"] == 720
    assert counts["fidelity_curves"] == 64240
    assert counts["fidelity_curve_points"] == 1_349_040
    assert counts["map_runs"] == 9592
    assert counts["work_items"] == 1_358_632

    holdout = next(row for row in registry.rows if row.role == "holdout")
    holdout_spec = _spec(
        registry, sample_ids=[holdout.sample_id], value_config_source="explicit",
        value_candidates=(PAIR,),
    )
    with pytest.raises(SelectionContractError, match="holdout scope"):
        build_execution_plan(CONFIG, registry, holdout_spec)


def test_selected_four_config_cross_line_exact_counts(registry, tmp_path: Path):
    document = _selection_document()
    path = tmp_path / "selection.json"
    _write_json(path, document)
    selection = load_stage_c_selection(CONFIG, selection_path=path)
    spec = _spec(
        registry, sample_ids=[row.sample_id for row in registry.rows],
        value_config_source="selection_band_winners", value_candidates=(),
    )
    plan = build_execution_plan(CONFIG, registry, spec, selection=selection)
    counts = estimate_stage_c_run(CONFIG, plan).to_dict()["counts"]
    assert counts["execution_units"] == 320
    assert counts["fidelity_eligible_units"] == 240
    assert counts["structural_reference_units"] == 80
    assert counts["fidelity_curves"] == 10_560
    assert counts["fidelity_curve_points"] == 221_760
    assert counts["map_runs"] == 1312
    assert counts["work_items"] == 223_072
    # Four global class units per sample, never 4 configs times 4 methods.
    assert sum(unit.comparison_line == "class_logit" for unit in plan.units) == 80


def test_checked_in_c9_run_spec_expands_exact_frozen_plan(registry):
    selection = load_stage_c_selection(CONFIG, required=True)
    spec = load_run_spec(REPO_ROOT / "config/stage_c_c9_run_spec.json")
    plan = build_execution_plan(CONFIG, registry, spec, selection=selection)
    counts = estimate_stage_c_run(CONFIG, plan).to_dict()["counts"]

    assert tuple(spec.sample_ids) == tuple(row.sample_id for row in registry.rows)
    assert spec.method_ids == ALL_METHODS
    assert spec.value_config_source == "selection_band_winners"
    assert spec.value_candidates == ()
    assert spec.fidelity_pairing_context == ValueCandidate(0, 9)
    assert selection.primary == ValueCandidate(0, 9)
    assert selection.depth_band_winners == (
        ValueCandidate(0, 9), ValueCandidate(4, 2),
        ValueCandidate(6, 10), ValueCandidate(11, 2),
    )
    assert {key: counts[key] for key in (
        "execution_units", "fidelity_eligible_units",
        "structural_reference_units", "fidelity_curves",
        "fidelity_curve_points", "map_runs", "work_items",
    )} == {
        "execution_units": 320,
        "fidelity_eligible_units": 240,
        "structural_reference_units": 80,
        "fidelity_curves": 10_560,
        "fidelity_curve_points": 221_760,
        "map_runs": 1_312,
        "work_items": 223_072,
    }
    assert sum(unit.comparison_line == "class_logit" for unit in plan.units) == 80
    assert all(
        (item.pairing_block_index, item.pairing_head_index) == (0, 9)
        for unit in plan.units if unit.comparison_line == "class_logit"
        for item in unit.work_items if item.kind == "fidelity_curve_point"
    )


def test_estimate_breaks_out_logic_calls_files_and_round_trips(registry):
    plan = build_execution_plan(CONFIG, registry, _spec(registry))
    estimate = estimate_stage_c_run(CONFIG, plan)
    value = estimate.to_dict()
    assert StageCEstimate.from_dict(value).to_dict() == value
    assert value["within_budget"] is True
    assert value["counts"]["logical_scalar_evaluations"] == 6 * 924
    assert value["compute"]["accounting_unit"] == "batched_model_invocation"
    assert value["compute"]["estimated_forward_calls_total"] > value["counts"]["logical_scalar_evaluations"]
    assert set(value["storage"]) >= {"L0", "L1", "L2", "L3", "control"}
    assert value["storage"]["ordinary_files_included"] is True
    assert value["caps"]["config_mutated"] is False


def test_cap_overrides_are_test_only_tighten_and_fail_before_write(registry, tmp_path: Path):
    plan = build_execution_plan(CONFIG, registry, _spec(registry))
    frozen_before = CONFIG.budget.to_dict()
    estimate = estimate_stage_c_run(
        CONFIG, plan,
        cap_overrides={"forward_calls_per_execution_unit_max": 1},
    )
    assert estimate.within_budget is False
    assert "forward_calls_per_execution_unit_max" in estimate.violations
    with pytest.raises(BudgetExceededError, match="forward_calls"):
        enforce_stage_c_budget(estimate)
    assert not tmp_path.iterdir().__next__ if False else list(tmp_path.iterdir()) == []
    assert CONFIG.budget.to_dict() == frozen_before

    with pytest.raises(RunSpecError, match="cannot loosen"):
        estimate_stage_c_run(
            CONFIG, plan, cap_overrides={
                "forward_calls_per_execution_unit_max":
                    CONFIG.budget["forward_calls_per_execution_unit_max"] + 1
            },
        )
    with pytest.raises(RunSpecError, match="only hard-cap fields"):
        estimate_stage_c_run(
            CONFIG, plan, cap_overrides={"storage_control_overhead_factor": 1.0}
        )
    with pytest.raises(RunSpecError, match="only hard-cap fields"):
        estimate_stage_c_run(
            CONFIG, plan, cap_overrides={"projection_safety_factor": 1.0}
        )


def test_budget_equality_passes_and_one_over_fails(registry):
    plan = build_execution_plan(CONFIG, registry, _spec(registry))
    baseline = estimate_stage_c_run(CONFIG, plan)
    maximum = baseline.compute["maximum_forward_calls_per_execution_unit"]
    equal = estimate_stage_c_run(
        CONFIG, plan, cap_overrides={"forward_calls_per_execution_unit_max": maximum}
    )
    assert "forward_calls_per_execution_unit_max" not in equal.violations
    if maximum > 1:
        over = estimate_stage_c_run(
            CONFIG, plan, cap_overrides={"forward_calls_per_execution_unit_max": maximum - 1}
        )
        assert "forward_calls_per_execution_unit_max" in over.violations


def test_repeatability_scope_is_clean_only_and_options_never_guessed(registry):
    no_extras = _spec(
        registry, method_ids=(VALUE_METHODS[0],), fidelity=False, sanity=False,
        stability=False, repeatability=3,
    )
    plan = build_execution_plan(CONFIG, registry, no_extras)
    items = plan.units[0].work_items
    assert [item.input_variant for item in items] == [
        "clean", "repeat_clean", "repeat_clean"
    ]
    assert [item.repeat_index for item in items] == [0, 1, 2]
    with pytest.raises(TypeError):
        # Every unfrozen execution policy is keyword-only and required.
        make_run_spec(  # type: ignore[call-arg]
            sample_ids=[registry.rows[0].sample_id], method_ids=[VALUE_METHODS[0]],
            value_candidates=[PAIR], value_config_source="explicit",
        )


def test_work_item_curve_and_plan_from_dict_fail_closed(registry):
    plan = build_execution_plan(CONFIG, registry, _spec(registry))
    document = plan.to_dict()
    curve_index = next(
        index for index, item in enumerate(document["units"][1]["work_items"])
        if item["kind"] == "fidelity_curve_point"
    )
    del document["units"][1]["work_items"][curve_index]
    with pytest.raises(RunSpecError, match="incomplete fidelity curve"):
        ExecutionPlan.from_dict(document)

    document = plan.to_dict()
    document["units"][0]["work_items"][0]["input_variant"] = "mystery"
    with pytest.raises(RunSpecError, match="unregistered"):
        ExecutionPlan.from_dict(document)


def test_execution_unit_requires_one_shared_fidelity_pairing_context(registry):
    plan = build_execution_plan(CONFIG, registry, _spec(registry))

    # A global class unit may use the RunSpec's explicit pair, but every point
    # in that unit must use that same pair.
    class_unit = next(
        unit for unit in plan.units if unit.comparison_line == "class_logit"
    )
    class_document = class_unit.to_dict()
    class_curve = next(
        item for item in class_document["work_items"]
        if item["kind"] == "fidelity_curve_point"
    )
    assert (
        class_curve["pairing_block_index"], class_curve["pairing_head_index"]
    ) == (0, 0)
    class_curve["pairing_head_index"] = 1
    with pytest.raises(RunSpecError, match="share one pairing context"):
        type(class_unit).from_dict(class_document)

    # A per-head Value-path unit is stricter: its sole pairing must equal its
    # own UnitKey block/head even when all points are changed consistently.
    value_unit = next(
        unit for unit in plan.units
        if unit.key.method_id == VALUE_METHODS[1]
    )
    assert {
        (item.pairing_block_index, item.pairing_head_index)
        for item in value_unit.work_items
        if item.kind == "fidelity_curve_point"
    } == {(value_unit.key.block_index, value_unit.key.head_index)}
    value_document = value_unit.to_dict()
    for item in value_document["work_items"]:
        if item["kind"] == "fidelity_curve_point":
            item["pairing_head_index"] = 1
    with pytest.raises(RunSpecError, match="must equal the unit block/head"):
        type(value_unit).from_dict(value_document)


def test_registry_snapshot_from_dict_rejects_collection_drift(registry):
    document = registry.to_dict()
    document["rows"] = document["rows"][:-1]
    with pytest.raises(RegistryContractError, match="20 rows|ordered"):
        type(registry).from_dict(document)

    document = registry.to_dict()
    document["rows"][1]["sample_id"] = document["rows"][0]["sample_id"]
    with pytest.raises(RegistryContractError, match="sample_id"):
        type(registry).from_dict(document)


def test_estimate_digest_and_violations_cannot_be_forged(registry):
    plan = build_execution_plan(CONFIG, registry, _spec(registry))
    estimate = estimate_stage_c_run(CONFIG, plan)
    document = estimate.to_dict()
    document["compute"]["maximum_forward_calls_per_execution_unit"] = 10**9
    with pytest.raises(RunSpecError, match="canonical_digest"):
        StageCEstimate.from_dict(document)

    document = estimate.to_dict()
    document["within_budget"] = False
    with pytest.raises(RunSpecError, match="not violations"):
        StageCEstimate.from_dict(document)

    # Even a self-consistent serialization is descriptive only: execution
    # authorization requires an estimate freshly issued by the estimator.
    restored = StageCEstimate.from_dict(estimate.to_dict())
    with pytest.raises(RunSpecError, match="freshly computed"):
        enforce_stage_c_budget(restored)

    incomplete = estimate.to_dict()
    del incomplete["compute"]["maximum_forward_calls_per_execution_unit"]
    incomplete["canonical_digest"] = __import__(
        "hashlib"
    ).sha256(
        json.dumps(
            {key: value for key, value in incomplete.items() if key != "canonical_digest"},
            sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    with pytest.raises(Exception, match="compute.*fields differ"):
        StageCEstimate.from_dict(incomplete)


def test_storage_file_counts_follow_logical_writer_layout(registry):
    plan = build_execution_plan(CONFIG, registry, _spec(registry))
    estimate = estimate_stage_c_run(CONFIG, plan).to_dict()
    counts, storage = estimate["counts"], estimate["storage"]
    assert storage["L0"]["file_count"] == 2 * counts["execution_units"]
    assert storage["L1"]["file_count"] == counts["map_runs"]
    assert storage["control"]["unit_manifest_files"] == counts["execution_units"]
    assert storage["control"]["unit_status_files"] == counts["execution_units"]


def test_compute_estimate_has_explicit_setup_endpoint_and_stability_calls(registry):
    plan = build_execution_plan(CONFIG, registry, _spec(registry))
    compute = estimate_stage_c_run(CONFIG, plan).to_dict()["compute"]
    by_method = compute["by_method"]
    raw = by_method[VALUE_METHODS[0]]
    assert raw["setup_forward_calls"] == 1
    assert raw["stability_prediction_forward_calls"] == 4
    assert raw["curve_points"] == 0
    nns = by_method[VALUE_METHODS[1]]
    assert nns["setup_forward_calls"] == 6
    assert nns["fidelity_endpoint_forward_calls"] == 2
    assert nns["stability_prediction_forward_calls"] == 4
    class_ig = by_method[CLASS_METHODS[-1]]
    assert class_ig["setup_forward_calls"] == 2
    assert class_ig["fidelity_endpoint_forward_calls"] == 2
    assert class_ig["stability_prediction_forward_calls"] == 4
