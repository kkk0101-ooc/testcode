from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import numpy as np
import pytest

from vitnns import nns_patch_depth_s4 as s4


ROOT = Path(__file__).resolve().parents[1]


def fixture_schedule(count: int = 3) -> list[dict]:
    return [
        {
            "schedule_index": index, "source_image_id": f"source-{index}",
            "image_sha256": f"{index:064x}", "annotation_sha256": f"{index + 10:064x}",
        }
        for index in range(count)
    ]


def publish_fixture_terminal(root: Path, unit_type: str, index: int, status: str) -> None:
    path = s4.terminal_path(root, unit_type, index)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(s4.signed("s4_unit_terminal", status, {"unit_type": unit_type, "schedule_index": index})))


def test_position_strata_and_p1_hash_permutations() -> None:
    assert sorted(np.unique(s4.POSITION_STRATA).tolist()) == list(range(7))
    positive = np.ones((12, 196), dtype=np.float32)
    occupancy = np.linspace(0.0, 1.0, 196, dtype=np.float32)
    result = s4.p1_raw(positive, occupancy, "fixture")
    assert result["permutation_indices"].shape == (12, 64, 196)
    assert result["numerators"].shape == (12, 65)
    assert result["denominators"].tolist() == [196.0] * 12
    for block in range(12):
        for repeat in range(64):
            permutation = result["permutation_indices"][block, repeat]
            for stratum in range(7):
                indices = np.flatnonzero(s4.POSITION_STRATA == stratum)
                assert sorted(permutation[indices].tolist()) == indices.tolist()
    again = s4.p1_raw(positive, occupancy, "fixture")
    for key in result:
        np.testing.assert_array_equal(result[key], again[key])


def test_p1_zero_mass_is_explicit_raw_state() -> None:
    result = s4.p1_raw(np.zeros((12, 196)), np.ones(196), "zero")
    assert result["valid_blocks"].tolist() == [0] * 12
    assert result["denominators"].tolist() == [0.0] * 12
    assert np.count_nonzero(result["numerators"]) == 0


def test_p2_plan_exact_strata_boundary_and_determinism() -> None:
    rng = np.random.Generator(np.random.PCG64(7))
    positive = rng.random((12, 196))
    left = s4.build_p2_plan(positive, "pilot-only-fixture")
    right = s4.build_p2_plan(positive, "pilot-only-fixture")
    assert left == right and len(left) == 36
    for coordinate in left:
        assert coordinate["status"] == "MATCHED" and coordinate["budget"] in (4, 8, 16) and len(coordinate["controls"]) == 4
        selected_counts = np.bincount(s4.POSITION_STRATA[coordinate["selected"]], minlength=7)
        tolerance = max(4, int(np.ceil(0.25 * coordinate["selected_boundary_complexity"])))
        for control in coordinate["controls"]:
            np.testing.assert_array_equal(np.bincount(s4.POSITION_STRATA[control["indices"]], minlength=7), selected_counts)
            assert control["boundary_absolute_error"] <= tolerance


def test_dynamic_claim_is_single_winner_dependency_aware_and_nonbarrier(tmp_path: Path) -> None:
    root = tmp_path / "stage"
    launch = root / "launches/launch-001"
    schedule = fixture_schedule()
    workers = [
        {"worker_id": f"worker-{index}", "worker_pid": 100 + index, "physical_gpu_index": index, "gpu_uuid": f"GPU-{index}"}
        for index in range(8)
    ]
    with ThreadPoolExecutor(max_workers=8) as pool:
        claims = list(pool.map(lambda worker: s4.claim_next(launch, root, schedule, worker), workers))
    winners = [claim for claim in claims if claim is not None]
    assert sorted((item["unit_type"], item["schedule_index"]) for item in winners) == [("NNS_IMAGE", 0), ("NNS_IMAGE", 1), ("NNS_IMAGE", 2)]
    assert len(list((launch / "claims").glob("**/*.claim.json"))) == 3
    publish_fixture_terminal(root, "NNS_IMAGE", 0, "VALID_NNS")
    claim = s4.claim_next(launch, root, schedule, workers[0])
    assert claim is not None and claim["unit_type"] == "OCCLUSION_PRIMARY" and claim["schedule_index"] == 0


def test_fresh_launch_ignores_stale_claim_and_global_terminal_skips(tmp_path: Path) -> None:
    root = tmp_path / "stage"
    schedule = fixture_schedule(1)
    worker = {"worker_id": "worker", "worker_pid": 1, "physical_gpu_index": 0, "gpu_uuid": "GPU"}
    old = root / "launches/old"
    new = root / "launches/new"
    assert s4.claim_next(old, root, schedule, worker)["unit_type"] == "NNS_IMAGE"
    assert s4.claim_next(new, root, schedule, worker)["unit_type"] == "NNS_IMAGE"
    publish_fixture_terminal(root, "NNS_IMAGE", 0, "VALID_NNS")
    assert s4.claim_next(root / "launches/recovery", root, schedule, worker)["unit_type"] == "OCCLUSION_PRIMARY"


def test_atomic_terminal_hardlink_no_overwrite(tmp_path: Path) -> None:
    terminal = tmp_path / "unit.json"
    document = s4.signed("s4_unit_terminal", "VALID_NNS", {"unit_type": "NNS_IMAGE", "schedule_index": 0})
    s4.publish_json_no_overwrite(tmp_path / "first.tmp", terminal, document)
    s4.publish_json_no_overwrite(tmp_path / "same.tmp", terminal, document)
    with pytest.raises(s4.S4Error, match="conflicting immutable"):
        s4.publish_json_no_overwrite(tmp_path / "bad.tmp", terminal, s4.signed("s4_unit_terminal", "FAILED_NNS", {"unit_type": "NNS_IMAGE", "schedule_index": 0}))


def test_no_static_shard_or_forbidden_seed_inputs_in_s4_source() -> None:
    source = (ROOT / "src/vitnns/nns_patch_depth_s4.py").read_text()
    worker = (ROOT / "src/vitnns/nns_patch_depth_s4_worker.py").read_text()
    assert "index % K" not in source + worker
    assert "schedule_index %" not in source + worker
    for forbidden in ("feature_prediction", "control_prediction", "feature_mse", "control_mse"):
        assert forbidden not in worker
    assert "CUBLAS_WORKSPACE_CONFIG" in worker and "@ right" in worker and "backward()" in worker


def test_s1_pilot_only_nns_and_occlusion_evidence_is_bound() -> None:
    terminal = json.loads((ROOT / "evidence/nns_vit_patch_depth/s1_r1_pilot_20260914/terminal_receipt.json").read_text())
    repeat = json.loads((ROOT / "evidence/nns_vit_patch_depth/s1_r1_pilot_20260914/deterministic_repeatability_report.json").read_text())
    nns = json.loads((ROOT / "evidence/nns_vit_patch_depth/s1_r1_pilot_20260914/nns_reconstruction_parity_report.json").read_text())
    p1p2 = json.loads((ROOT / "evidence/nns_vit_patch_depth/s1_r1_pilot_20260914/p1_p2_control_validation_report.json").read_text())
    assert terminal["status"] == "PASS" and terminal["payload"]["scientific_claim_decision"] is None
    assert repeat["status"] == "PASS" and repeat["payload"]["all_byte_identical"]
    assert nns["status"] == "PASS" and nns["payload"]["all_valid_closure_pass"]
    assert p1p2["status"] == "PASS" and p1p2["payload"]["p2_operator_selection_used_logit_outcome"] is False
