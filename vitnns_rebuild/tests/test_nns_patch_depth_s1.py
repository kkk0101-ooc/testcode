from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from vitnns import nns_patch_depth_s1 as s1


ROOT = Path(__file__).resolve().parents[1]


def test_run_spec_is_closed_and_pilot_only() -> None:
    value = json.loads((ROOT / "config/nns_vit_patch_depth/s1_r1_run_spec.json").read_text())
    assert value["stage_id"] == "S1-R1"
    assert value["stage_token"] == "S1_PILOT_ONLY"
    assert value["data_access"]["allowed_outcome_split"] == "pilot"
    assert value["data_access"]["forbidden_outcome_splits"] == ["probe_train", "probe_validation", "confirmatory_holdout"]
    assert value["p1"]["permutations"] == 64
    assert value["p2"]["patch_budgets"] == [4, 8, 16]
    assert value["d1"]["shallow_blocks"] == [0, 1, 2, 3]
    assert value["d1"]["deep_blocks"] == [8, 9, 10, 11]
    assert value["scientific_decisions_allowed"] is False
    assert value["amendment"]["id"] == "S1-R1-A1"
    assert value["amendment"]["smoke_state_at_freeze"] == "6_OF_6_VALID_UNITS_AND_3_OF_3_PASS_SHARDS"
    assert value["escalation"]["engineering_stop_conditions"]["minimum_control_matching_fraction"] == 0.90
    assert value["parallel"]["minimum_workers"] == 1 and value["parallel"]["maximum_workers"] == 8
    assert value["parallel"]["existing_compute_pid_is_disqualifying"] is False
    assert value["parallel"]["fewer_than_eight_workers_is_failure_or_blocked"] is False


def test_a1_producer_and_validator_read_control_gate_from_spec() -> None:
    producer = (ROOT / "scripts/run_nns_vit_patch_depth_s1.py").read_text()
    validator = (ROOT / "scripts/validate_nns_vit_patch_depth_s1.py").read_text()
    assert "p2_matched_coordinate_fraction\"] >= 0.98" not in producer
    assert "p2_matched_coordinate_fraction\"] >= 0.98" not in validator
    assert "minimum_control_matching_fraction_required" in producer
    assert "required_control_fraction" in validator


def test_a1_gpu_eligibility_is_memory_and_utilization_based() -> None:
    producer = (ROOT / "scripts/run_nns_vit_patch_depth_s1.py").read_text()
    assert 'int(free) >= minimum_free and int(utilization) <= maximum_utilization' in producer
    assert 'uuid not in active' not in producer
    assert 'existing_compute_processes' in producer and 'existing_compute_used_memory_mib' in producer


def test_a1_amendment_is_prospective_and_smoke_bound() -> None:
    text = (ROOT / "docs/nns_vit_patch_depth_s1_r1_a1_amendment_20260914.md").read_text()
    assert "6/6 unit terminals" in text and "3/3 shard terminals" in text
    assert "Only new intermediate indices 6–23" in text
    assert "must not be deleted, overwritten, recomputed, or replaced" in text
    assert "must not issue `SUPPORTED` or `NOT_SUPPORTED`" in text


def test_a2_contract_is_fixed_gpu_dynamic_claim() -> None:
    value = json.loads((ROOT / "config/nns_vit_patch_depth/s1_r1_run_spec.json").read_text())["execution_amendment_a2"]
    assert value["id"] == "S1-R1-A2"
    assert value["fixed_gpu_set_per_launch"] and value["one_long_lived_worker_per_gpu"]
    assert value["queue_order"] == "ascending_missing_schedule_index"
    assert value["claim_creation"] == "O_CREAT_O_EXCL_then_fsync_file_and_directory"
    assert value["unit_seed_inputs"] == ["schedule_index", "source_image_id"]
    assert "physical_gpu" in value["unit_seed_forbidden_inputs"]


def test_a2_atomic_claim_skips_terminal_and_is_single_winner(tmp_path: Path) -> None:
    from vitnns import nns_patch_depth_s1_queue as queue
    from concurrent.futures import ThreadPoolExecutor
    stage = tmp_path / "stage"; launch = stage / "dynamic_launches/launch-001"; existing = stage / "smoke/shards/shard-00/units/unit-0000/completion/unit.json"; existing.parent.mkdir(parents=True)
    existing.write_text(json.dumps({"document_type": "s1_unit_terminal", "status": "VALID", "payload": {"identity": {"schedule_index": 0}}}))
    schedule = {i: {"source_image_id": f"source-{i}", "image_sha256": f"{i:064x}"} for i in range(3)}; worker = {"worker_id": "worker-00", "worker_pid": 123, "physical_gpu_index": 2, "gpu_uuid": "GPU-x"}
    workers = [{**worker, "worker_id": f"worker-{index}", "worker_pid": 123 + index} for index in range(4)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        claims = list(pool.map(lambda args: queue.claim_next(launch, [0, 1, 2], stage, args[1], schedule, now_ns=10 + args[0]), enumerate(workers)))
    won = sorted(item["schedule_index"] for item in claims if item is not None)
    assert won == [1, 2] and len(list((launch / "claims").glob("*.claim.json"))) == 2
    assert queue.claim_next(launch, [0, 1, 2], stage, worker, schedule) is None
    assert queue.missing_indices(stage, [0, 1, 2]) == [1, 2]


def test_a2_fresh_launch_ignores_stale_claim_and_resource_is_unit_based(tmp_path: Path) -> None:
    from vitnns import nns_patch_depth_s1_queue as queue
    stage = tmp_path / "stage"; old = stage / "dynamic_launches/old"; new = stage / "dynamic_launches/new"; schedule = {5: {"source_image_id": "source-5", "image_sha256": "5" * 64}}; worker = {"worker_id": "w", "worker_pid": 1, "physical_gpu_index": 0, "gpu_uuid": "GPU"}
    assert queue.claim_next(old, [5], stage, worker, schedule, now_ns=1)["schedule_index"] == 5
    assert queue.claim_next(new, [5], stage, worker, schedule, now_ns=2)["schedule_index"] == 5
    assert queue.aggregate_unit_resources(stage, [5])["static_shard_inference_used"] is False


def test_a4_disables_static_full_and_includes_dynamic_terminals() -> None:
    runner = (ROOT / "scripts/run_nns_vit_patch_depth_s1.py").read_text()
    validator = (ROOT / "scripts/validate_nns_vit_patch_depth_s1.py").read_text()
    assert "static full_pilot launch is disabled by S1-R1-A2" in runner
    pattern = "dynamic_launches/*/dynamic_units/unit-*/attempts/*/completion/unit.json"
    assert pattern in runner and pattern in validator
    assert '"dynamic_claim"' in validator


def test_geometry_preserves_four_semantics_and_patch_occupancy() -> None:
    labels = np.zeros((224, 224), dtype=np.int32)
    labels[:16, :16] = 7
    labels[:16, 16:32] = 8
    result = s1.patch_occupancy(labels, 7)
    assert set(result) == {"background", "other_category", "void", "target"}
    assert result["target"][0] == 1.0 and result["other_category"][1] == 1.0
    np.testing.assert_array_equal(sum(result.values()), np.ones(196))


def test_geometry_typed_na_paths() -> None:
    with pytest.raises(s1.S1TypedNA, match="integer") as mismatch:
        s1.patch_occupancy(np.zeros((223, 224), dtype=np.int32), 1)
    assert mismatch.value.code == "NA_GEOMETRY_SIZE_MISMATCH"
    with pytest.raises(s1.S1TypedNA) as empty:
        s1.patch_occupancy(np.zeros((224, 224), dtype=np.int32), 1)
    assert empty.value.code == "NA_TARGET_DISAPPEARED_AFTER_CROP"
    value = np.zeros((224, 224), dtype=np.int32); value[0, 0] = 1; value[1, 1] = 1000
    with pytest.raises(s1.S1TypedNA) as void:
        s1.patch_occupancy(value, 1)
    assert void.value.code == "NA_VOID_MASK"


def test_matrix_plus_cls_source_score_and_signed_bookkeeping() -> None:
    matrix = np.zeros((196, 196)); matrix[:, 0] = np.arange(196) - 98
    cls = np.linspace(-1, 1, 196)
    result = s1.source_patch_scores(matrix, cls)
    np.testing.assert_allclose(result["signed"], matrix.sum(1) + cls)
    assert result["negative_mass"] <= 0 <= result["positive_mass"]
    assert result["absolute_mass"] >= abs(result["signed_mass"])
    assert result["matrix_signed_mass"] + result["cls_signed_mass"] == pytest.approx(result["signed_mass"])


def test_p1_synthetic_centered_constant_position_empty_and_void() -> None:
    target = np.zeros(196); center = np.flatnonzero(s1.POSITION_STRATA == 0); target[center] = 1.0
    semantic = np.zeros(196); semantic[center] = 1.0
    aligned = s1.p1_alignment(semantic, target, "centered", 0, 64)
    assert aligned["corrected_alignment"] == pytest.approx(0.0)  # one stratum: no semantic claim beyond position
    constant = s1.p1_alignment(np.ones(196), target, "constant", 0, 64)
    assert constant["corrected_alignment"] == pytest.approx(0.0, abs=1e-15)
    position = s1.p1_alignment((7 - s1.POSITION_STRATA).astype(float), target, "position", 0, 64)
    assert position["corrected_alignment"] == pytest.approx(0.0, abs=1e-15)
    with pytest.raises(s1.S1TypedNA) as empty:
        s1.p1_alignment(np.zeros(196), target, "empty", 0, 64)
    assert empty.value.code == "NA_ZERO_POSITIVE_MASS"


def test_p1_detects_within_stratum_semantic_alignment() -> None:
    target = np.zeros(196); ring = np.flatnonzero(s1.POSITION_STRATA == 3); target[ring[:10]] = 1.0
    score = np.zeros(196); score[ring[:10]] = 1.0
    assert s1.p1_alignment(score, target, "semantic", 2, 64)["corrected_alignment"] > 0.5


def test_p2_matched_controls_and_failure_are_typed() -> None:
    score = np.linspace(1, 0, 196)
    result = s1.topk_and_controls(score, "fixture", 3, 8, 4, 512, 0.25)
    selected_counts = np.bincount(s1.POSITION_STRATA[result["selected"]], minlength=7)
    for control in result["controls"]:
        np.testing.assert_array_equal(np.bincount(s1.POSITION_STRATA[control["indices"]], minlength=7), selected_counts)
        assert len(control["indices"]) == 8
    with pytest.raises(s1.S1TypedNA) as failed:
        s1.matched_controls(list(range(196)), "fixture", 0, 196, 1, 1, 0.0)
    assert failed.value.code == "NA_MATCHED_CONTROL_UNAVAILABLE"


def test_d1_requires_complete_trajectory() -> None:
    rows = [{"block": block, "status": "VALID", "corrected_alignment": float(block)} for block in range(12)]
    value = s1.d1_contrast(rows)
    assert value["deep_minus_shallow"] == 8.0 and value["scientific_decision"] is None
    with pytest.raises(s1.S1TypedNA) as incomplete:
        s1.d1_contrast(rows[:-1])
    assert incomplete.value.code == "NA_INCOMPLETE_DEPTH_TRAJECTORY"


def test_probe_train_only_normalization_constant_and_shuffle() -> None:
    rng = np.random.Generator(np.random.PCG64(7))
    train = rng.normal(size=(80, 8)); validation = rng.normal(size=(40, 8)); truth = train[:, 0] * 0.4 - train[:, 1] * 0.2
    model = s1.fit_ridge(train, truth, 1.0)
    assert not np.allclose(model.mean, np.vstack((train, validation)).mean(0))
    assert np.mean((s1.predict_ridge(model, train) - truth) ** 2) < 0.01
    constant = np.ones((20, 5)); constant_model = s1.fit_ridge(constant, np.linspace(0, 1, 20), 1.0)
    assert np.isfinite(s1.predict_ridge(constant_model, constant)).all()
    shuffled = rng.permutation(truth); shuffled_model = s1.fit_ridge(train, shuffled, 1.0)
    assert np.mean((s1.predict_ridge(shuffled_model, validation) - (validation[:, 0] * 0.4 - validation[:, 1] * 0.2)) ** 2) > 0.01
    assert s1.serialize_ridge(model) == s1.serialize_ridge(s1.fit_ridge(train, truth, 1.0))


def test_probe_split_is_image_level_disjoint_and_detects_wrong_class_count() -> None:
    records = [
        {"annotation_wnid": "a", "source_image_id": "a1"},
        {"annotation_wnid": "a", "source_image_id": "a2"},
        {"annotation_wnid": "b", "source_image_id": "b1"},
        {"annotation_wnid": "b", "source_image_id": "b2"},
    ]
    train, validation = s1.probe_split(records)
    assert len(train) == len(validation) == 2
    assert not ({row["source_image_id"] for row in train} & {row["source_image_id"] for row in validation})
    with pytest.raises(s1.S1Error):
        s1.probe_split(records[:-1])


def test_worker_json_normalizes_numpy_scalars() -> None:
    from vitnns import nns_patch_depth_s1_worker as worker
    value = worker.signed("fixture", "PASS", {"index": np.int64(3), "array": np.asarray([1, 2], dtype=np.int64)})
    json.dumps(value, allow_nan=False)
    assert value["payload"] == {"index": 3, "array": [1, 2]}
