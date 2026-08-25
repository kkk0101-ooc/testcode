"""Synthetic/metadata-only tests for the independent Stage D planner.

These tests intentionally never open, stat, hash, decode, or preprocess a
reserve JPEG and never import the model/runner stack.
"""

from __future__ import annotations

import copy
import json
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import pytest

from vitnns.stage_d_plan import (
    CURVE_FRACTIONS, DIRECTIONS, FIDELITY_CURVE_POINT_COUNT,
    FIDELITY_SCALE, LOGICAL_WORK_ITEM_COUNT, MAP_WORK_ITEM_COUNT, METHOD_IDS,
    PROFILE_SOURCE_STATUS, PROFILE_UNIT_KINDS, StageDBudgetError,
    StageDCurvePoint, StageDEstimate,
    StageDExecutionPlan, StageDMapCoordinate, StageDPlanError, StageDPreflight,
    StageDResourceProfile, build_stage_d_plan, enforce_stage_d_preflight,
    estimate_stage_d_run, preflight_stage_d, stage_d_shard_id,
    validate_stage_d_shard_count, validate_stage_d_shard_mapping,
    UNVALIDATED_PROFILE_STATUS,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return json.loads((REPO_ROOT / name).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def documents() -> tuple[dict, dict, dict]:
    return (
        _load("config/stage_d.json"),
        _load("config/stage_d_run_spec.json"),
        _load("manifests/stage_d_fidelity_v1/registry.json"),
    )


@pytest.fixture(scope="module")
def plan(documents: tuple[dict, dict, dict]) -> StageDExecutionPlan:
    return build_stage_d_plan(*documents)


def _resign(document: dict) -> None:
    from vitnns.stage_d_plan import canonical_sha256

    document["canonical_digest"] = canonical_sha256({
        key: value for key, value in document.items()
        if key != "canonical_digest"
    })


def _profiles(
    *, source_status: str = PROFILE_SOURCE_STATUS,
    **overrides: int | float,
) -> list[StageDResourceProfile]:
    result = []
    for method in METHOD_IDS:
        for unit_kind in PROFILE_UNIT_KINDS:
            values = {
                "seconds": 10.0,
                "forward_calls": 20,
                "backward_calls": 10 if method == METHOD_IDS[0] else 0,
                "l0_l1_payload_bytes": 1024,
                "l2_payload_bytes": 2048 if unit_kind == "audit_repeat_map" else 0,
                "control_and_provenance_bytes": 512,
                "cuda_peak_reserved_bytes": 4096,
            }
            values.update(overrides)
            result.append(StageDResourceProfile(
                method_id=method, unit_kind=unit_kind,
                source_status=source_status, **values,
            ))
    return result


def test_exact_typed_plan_is_compact_ordered_and_content_addressed(
    plan: StageDExecutionPlan,
) -> None:
    assert plan.counts == {
        "image_groups": 500, "execution_units": 1000,
        "primary_clean_maps": 1000, "audit_repeat_clean_maps": 10,
        "map_work_items": 1010, "fidelity_curves": 10000,
        "fidelity_curve_points": 210000, "logical_work_items": 211010,
        "statistical_n": 500, "statistical_n_increment": 0,
    }
    assert len(plan.groups) == 500
    assert len(plan.units) == 1000
    assert len(plan.map_coordinates) == MAP_WORK_ITEM_COUNT
    assert len(plan.curve_points) == FIDELITY_CURVE_POINT_COUNT
    assert len(plan.coordinates) == LOGICAL_WORK_ITEM_COUNT
    assert all(tuple(unit.method_id for unit in group.units) == METHOD_IDS for group in plan.groups)
    assert all(
        group.units[0].pair_binding_digest == group.units[1].pair_binding_digest
        for group in plan.groups
    )
    assert isinstance(plan.coordinates[0], StageDMapCoordinate)
    assert isinstance(plan.coordinates[MAP_WORK_ITEM_COUNT], StageDCurvePoint)
    assert isinstance(plan.coordinates[-1], StageDCurvePoint)
    last = plan.coordinates[-1]
    assert last.method_id == METHOD_IDS[1]
    assert last.direction == DIRECTIONS[-1]
    assert last.repeat_index == 4
    assert last.fraction == CURVE_FRACTIONS[-1]
    assert last.fidelity_scale == FIDELITY_SCALE
    assert last.intervention_count == 3136
    # Compact wire form retains groups/dimensions, not 210k point dictionaries.
    encoded = json.dumps(plan.to_dict(), separators=(",", ":"))
    assert len(encoded) < 2_000_000
    assert "curve_points" not in plan.to_dict()


def test_mapping_or_to_dict_inputs_and_immutable_round_trip(
    documents: tuple[dict, dict, dict], plan: StageDExecutionPlan,
) -> None:
    wrapped = [SimpleNamespace(to_dict=lambda value=value: copy.deepcopy(value)) for value in documents]
    rebuilt = build_stage_d_plan(*wrapped)
    assert rebuilt.canonical_digest == plan.canonical_digest
    assert StageDExecutionPlan.from_dict(plan.to_dict()).to_dict() == plan.to_dict()
    with pytest.raises(FrozenInstanceError):
        plan.groups[0].sample_id = "tamper"  # type: ignore[misc]
    with pytest.raises(TypeError):
        plan.counts["image_groups"] = 1  # type: ignore[index]


def test_validated_loader_and_manifest_objects_are_accepted() -> None:
    from vitnns.stage_d_config import load_stage_d_contracts
    from vitnns.stage_d_manifest import verify_stage_d_manifest

    contracts = load_stage_d_contracts()
    manifest = verify_stage_d_manifest(contracts.config, verify_c10=False)
    plan = build_stage_d_plan(
        contracts.config, contracts.run_spec, manifest,
    )
    assert plan.config_digest == contracts.config.canonical_digest
    assert plan.registry_digest == manifest.registry_document["canonical_digest"]


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("primary_scope", "head_index", 8),
        ("primary_scope", "methods_in_image_group_order", list(reversed(METHOD_IDS))),
        ("fidelity", "directions", ["deletion", "insertion"]),
        ("fidelity", "curve_fractions", [float(x) for x in CURVE_FRACTIONS[:-1]]),
        ("fidelity", "tie_repeat_indices", [0, 1, 2, 3]),
        ("forbidden_workload", "patch_scale_fidelity", True),
    ],
)
def test_config_scientific_or_forbidden_drift_fails_even_if_resigned(
    documents: tuple[dict, dict, dict], section: str, key: str, value: object,
) -> None:
    config, run_spec, registry = copy.deepcopy(documents)
    config[section][key] = value
    _resign(config)
    run_spec["config_binding"]["config_canonical_digest"] = config["canonical_digest"]
    _resign(run_spec)
    with pytest.raises(StageDPlanError):
        build_stage_d_plan(config, run_spec, registry)


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("execution_coordinates", "head_index", 8),
        ("execution_coordinates", "method_ids", list(reversed(METHOD_IDS))),
        ("fidelity", "spatial_scope", "PATCH_16X16"),
        ("fidelity", "directions", ["insertion"]),
        ("fidelity", "tie_repeat_indices", [0, 1, 2, 3, 4, 5]),
        ("audit", "statistical_n_increment", 5),
        ("forbidden_workload", "run_random_sanity", True),
    ],
)
def test_run_spec_coordinate_and_scope_drift_fails_even_if_resigned(
    documents: tuple[dict, dict, dict], section: str, key: str, value: object,
) -> None:
    config, run_spec, registry = copy.deepcopy(documents)
    run_spec[section][key] = value
    _resign(run_spec)
    with pytest.raises(StageDPlanError):
        build_stage_d_plan(config, run_spec, registry)


def test_registry_order_count_audit_and_non_audit_repeat_fail_closed(
    documents: tuple[dict, dict, dict], plan: StageDExecutionPlan,
) -> None:
    config, run_spec, registry = copy.deepcopy(documents)
    registry["rows"] = registry["rows"][:-1]
    _resign(registry)
    with pytest.raises(StageDPlanError):
        build_stage_d_plan(config, run_spec, registry)

    document = plan.to_dict()
    document["groups"][0]["audit"] = True
    document["groups"][0]["audit_overlay_id"] = "invented"
    with pytest.raises(StageDPlanError):
        StageDExecutionPlan.from_dict(document)

    non_audit = plan.groups[0].units[0]
    with pytest.raises(StageDPlanError):
        StageDMapCoordinate(
            unit_digest=non_audit.canonical_digest,
            pair_binding_digest=non_audit.pair_binding_digest,
            sample_id=non_audit.sample_id, selection_order=0,
            method_id=non_audit.method_id, block_index=0, head_index=9,
            target_id=non_audit.target_id, map_kind="audit_repeat_clean_map",
            clean_run_index=1, audit=False, audit_overlay_id=None,
            statistical_n_increment=0, canonical_digest="0" * 64,
        )


@pytest.mark.parametrize(
    "path",
    [
        ("counts", "logical_work_items"),
        ("dimensions", "fidelity_scale"),
        ("groups", 0, "units", 0, "method_id"),
        ("groups", 0, "pair_binding", "plan_context_digest"),
        ("coordinate_digest",),
        ("canonical_digest",),
    ],
)
def test_plan_from_dict_rejects_nested_or_digest_tamper(
    plan: StageDExecutionPlan, path: tuple[object, ...],
) -> None:
    document = plan.to_dict()
    cursor = document
    for part in path[:-1]:
        cursor = cursor[part]  # type: ignore[index]
    leaf = path[-1]
    current = cursor[leaf]  # type: ignore[index]
    cursor[leaf] = (
        current + 1 if type(current) is int
        else "0" * 64 if isinstance(current, str) and len(current) == 64
        else "tampered"
    )  # type: ignore[index]
    with pytest.raises(StageDPlanError):
        StageDExecutionPlan.from_dict(document)


def test_sharding_is_future_positive_modulo_only(plan: StageDExecutionPlan) -> None:
    assert "shard_count" not in plan.to_dict()
    assert validate_stage_d_shard_count(7) == 7
    assert stage_d_shard_id(499, 7) == 499 % 7
    mapping = validate_stage_d_shard_mapping(plan, 7)
    assert mapping[0] == 0 and mapping[499] == 499 % 7
    with pytest.raises(StageDPlanError):
        validate_stage_d_shard_count(True)
    with pytest.raises(StageDPlanError):
        validate_stage_d_shard_count(0)
    bad = dict(mapping)
    bad[1] = bad[1] + 1
    with pytest.raises(StageDPlanError):
        validate_stage_d_shard_mapping(plan, 7, bad)


def test_metadata_only_estimate_keeps_d7_not_evaluated(
    documents: tuple[dict, dict, dict], plan: StageDExecutionPlan,
) -> None:
    estimate = estimate_stage_d_run(documents[0], plan)
    assert estimate.profile_status == "D7_PROFILE_REQUIRED"
    assert estimate.budget_status == "NOT_EVALUATED"
    assert estimate.profiles == ()
    assert estimate.projections["logical_work_items_used_as_call_count"] is False
    assert estimate.projections["forward_calls_total"] is None
    assert StageDEstimate.from_dict(estimate.to_dict()).to_dict() == estimate.to_dict()
    preflight = preflight_stage_d(estimate)
    assert preflight.status == "PROFILE_REQUIRED"
    assert preflight.execution_authorization == "NOT_ISSUED"
    assert preflight.analysis_authorization == "NOT_EVALUATED"
    assert preflight.d7_final is False
    assert StageDPreflight.from_dict(preflight.to_dict()).to_dict() == preflight.to_dict()
    with pytest.raises(StageDBudgetError):
        enforce_stage_d_preflight(preflight)


def test_profiles_sum_by_typed_method_and_audit_not_logical_items(
    documents: tuple[dict, dict, dict], plan: StageDExecutionPlan,
) -> None:
    profiles = _profiles()
    estimate = estimate_stage_d_run(documents[0], plan, profiles)
    projection = estimate.projections
    assert projection["forward_calls_total"] == 20 * 1000 + 20 * 10
    assert projection["backward_calls_total"] == 10 * 500 + 10 * 5
    assert projection["forward_calls_total"] != LOGICAL_WORK_ITEM_COUNT
    assert projection["logical_work_items_used_as_call_count"] is False
    # Synthetic/pre-D7 profiles never become a final budget pass.
    assert estimate.profile_status == UNVALIDATED_PROFILE_STATUS
    assert estimate.budget_status == "NOT_EVALUATED"
    assert preflight_stage_d(estimate).status == "PROFILE_REQUIRED"


def test_launch_caps_and_campaign_gates_remain_distinct(
    documents: tuple[dict, dict, dict], plan: StageDExecutionPlan,
) -> None:
    launch = estimate_stage_d_run(
        documents[0], plan, _profiles(forward_calls=30_001)
    )
    assert launch.budget_status == "BLOCKED_NO_LAUNCH"
    assert "forward_calls_per_execution_unit_max" in launch.launch_violations
    assert launch.completed_gate_violations == ()

    campaign = estimate_stage_d_run(
        documents[0], plan, _profiles(seconds=10_000.0)
    )
    assert campaign.budget_status == "BLOCKED_NO_LAUNCH"
    assert campaign.launch_violations == ()
    assert "stage_d_total_h20_gpu_hours_max" in campaign.completed_gate_violations
    assert preflight_stage_d(campaign).status == "BLOCKED_NO_LAUNCH"


def test_preflight_convenience_form_accepts_config_plan_profiles(
    documents: tuple[dict, dict, dict], plan: StageDExecutionPlan,
) -> None:
    preflight = preflight_stage_d(documents[0], plan, _profiles())
    assert preflight.status == "PROFILE_REQUIRED"
    assert preflight.execution_authorization == "NOT_ISSUED"


def test_estimate_and_preflight_tamper_and_profile_status_fail_closed(
    documents: tuple[dict, dict, dict], plan: StageDExecutionPlan,
) -> None:
    with pytest.raises(StageDPlanError):
        _profiles(source_status="D7_VALIDATED")
    estimate = estimate_stage_d_run(documents[0], plan).to_dict()
    estimate["projections"]["forward_calls_total"] = LOGICAL_WORK_ITEM_COUNT
    _resign(estimate)
    with pytest.raises(StageDPlanError):
        StageDEstimate.from_dict(estimate)
    preflight = preflight_stage_d(estimate_stage_d_run(documents[0], plan)).to_dict()
    preflight["execution_authorization"] = "AUTHORIZED_STAGE_D_FIDELITY_SCOPE"
    _resign(preflight)
    with pytest.raises(StageDPlanError):
        StageDPreflight.from_dict(preflight)
