from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from vitnns import stage_y_y2 as y2

REPO = Path(__file__).resolve().parents[1]


def test_schedule_and_run_spec_are_frozen_to_one_through_eight_workers() -> None:
    schedule = json.loads((REPO / "config/stage_y/y2_parallel_schedule.json").read_text())
    spec = json.loads((REPO / "config/stage_y/y2_run_spec.json").read_text())
    schema = json.loads((REPO / "config/stage_y/y2_run_spec.schema.json").read_text())
    Draft202012Validator.check_schema(schema); Draft202012Validator(schema).validate(spec)
    assert schedule["worker_contract"]["minimum_workers"] == spec["execution"]["minimum_device_count"] == 1
    assert schedule["worker_contract"]["maximum_workers"] == spec["execution"]["maximum_device_count"] == 8
    assert schedule["y2_sharding"]["shard_counts_by_worker_count"]["4"] == [25] * 4
    assert schedule["y3_sharding"]["shard_counts_by_worker_count"]["4"] == [40] * 4
    assert schedule["y3_sharding"]["launch_authorized_in_y2_p0_r1"] is False
    assert schedule["accounting_rule"].endswith("never divided by worker count")


def test_y2_and_y3_modulo_sharding_is_complete_and_nonadaptive() -> None:
    for workers in range(1, 9):
        assert [len(y2.expected_y2_orders(shard, workers)) for shard in y2.shard_ids(workers)] == y2.shard_counts(100, workers)
        assert sorted(order for shard in y2.shard_ids(workers) for order in y2.expected_y2_orders(shard, workers)) == list(range(100))
        for order in range(100): assert order in y2.expected_y2_orders(y2.y2_shard_id(order, workers), workers)
        y3 = [[(image, transform) for image in range(20) for transform in range(8) if y2.y3_shard_id(image, transform, workers) == shard] for shard in y2.shard_ids(workers)]
        assert [len(items) for items in y3] == y2.shard_counts(160, workers)
    with pytest.raises(y2.StageYY2ScheduleError): y2.y2_shard_id(100)
    with pytest.raises(y2.StageYY2ScheduleError): y2.y3_shard_id(20, 0)


def test_resource_projection_recomputes_fixed_cost_per_worker_and_aggregate_gpu_hours() -> None:
    ledger = y2.read_json(REPO / "evidence/stage_y/y1_r1_resource_telemetry_waiver_20260911/formal/conservative_budget_receipt.json")
    result = y2.schedule_from_source_ledger(ledger)
    assert result["y2"]["per_shard"][0]["cap_h20_gpu_hours"] == 8.4
    assert result["y2"]["per_shard"][4]["cap_h20_gpu_hours"] == 7.7
    assert result["y2"]["critical_path_worker_wall_hours"] == 9.0
    assert result["y2"]["overall_stage_wall_hours"] == 12.0
    assert result["y3"]["per_shard"][0]["cap_h20_gpu_hours"] == 12.8
    assert result["y3"]["critical_path_worker_wall_hours"] == 13.0
    assert result["y3"]["overall_stage_wall_hours"] == 18.0
    assert result["y2"]["fleet_concurrent_reserved_gpu_gib"] == result["y3"]["fleet_concurrent_reserved_gpu_gib"] == 168.0
    assert result["y2"]["safety_projected"]["h20_gpu_hours"] == pytest.approx(63.97206770833333)
    assert result["y3"]["safety_projected"]["h20_gpu_hours"] == pytest.approx(102.35530833333334)
    four = y2.schedule_from_source_ledger(ledger, 4)
    assert four["y2"]["shard_counts"] == [25] * 4
    assert four["y2"]["critical_path_worker_wall_hours"] == 16.0
    assert four["y2"]["overall_stage_wall_hours"] == 19.0
    assert four["y2"]["fleet_concurrent_reserved_gpu_gib"] == 84.0


def _cohort() -> dict:
    records = [{"selection_order": order, "source_object_id": f"synthetic:{order:03d}", "file_sha256": f"{order:064x}"} for order in range(100)]
    return y2.signed("stage_y_y2_cohort_registry", "PASS", {"records": records})


def test_manifest_builder_writes_only_eight_isolated_roots(tmp_path: Path) -> None:
    manifests = y2.build_shard_manifests(_cohort(), tmp_path)
    assert [row["payload"]["unit_count"] for row in manifests] == [13,13,13,13,12,12,12,12]
    roots = [row["payload"]["root"] for row in manifests]
    assert len(set(roots)) == 8
    for shard, document in enumerate(manifests):
        assert document["payload"]["selection_orders"] == list(range(shard, 100, 8))
        assert document["payload"]["dynamic_rebalancing"] is False


def _populate_synthetic_shards(root: Path, worker_count: int = 8) -> None:
    manifests = y2.build_shard_manifests(_cohort(), root, worker_count)
    for shard, manifest in enumerate(manifests):
        shard_root = root / "shards" / f"shard-{shard:02d}"
        binding = {
            "physical_index": shard, "cuda_visible_devices": str(shard),
            "gpu_uuid": f"GPU-SYNTHETIC-{shard}", "pci_bus_id": f"0000:{shard:02x}:00.0",
            "device_name": "NVIDIA H20", "hostname": "synthetic-host",
            "pid": 1000 + shard, "environment_digest": "e" * 64,
        }
        y2.write_json_new(shard_root / "worker_identity.json", y2.signed("stage_y_y2_worker_identity", "PASS", binding))
        for row in manifest["payload"]["records"]:
            order = row["selection_order"]; unit_root = shard_root / "units" / f"unit-{order:03d}"
            y2.begin_unit(unit_root, shard_id=shard, selection_order=order, source_object_id_value=row["source_object_id"], worker_count=worker_count)
            y2.append_attempt(unit_root, {"attempt_id": "synthetic-0", "steps": 2, "all_contracts_pass": True})
            attempts = [y2.file_record(path) for path in sorted((unit_root / "attempts").glob("attempt-*.json"))]
            scientific_path = unit_root / "scientific_result.json"
            y2.write_json_new(scientific_path, y2.signed("stage_y_y2_unit_scientific_result", "VALID", {"synthetic": True}))
            payload = {
                "unit_digest": y2.canonical_digest({"selection_order": order, "source_object_id": row["source_object_id"]}),
                "attempt_id": "synthetic-0", "shard_id": shard, "selection_order": order,
                "source_object_id": row["source_object_id"], "condition_id": "clean",
                "failure_code": None, "target_identity": {"class_id": 0},
                "source_binding": {"cohort_digest": manifest["payload"]["cohort_canonical_digest"], "file_sha256": row["file_sha256"]},
                "checkpoint_sha256": "c" * 64, "environment_digest": "e" * 64,
                "numeric_contract_digest": "n" * 64, "quadrature_attempts": [{"steps": 2, "all_contracts_pass": True}],
                "resource_telemetry": {"gpu_active_seconds": 36.0, "cpu_seconds": 72.0, "wall_seconds": 36.0, "peak_allocated_gpu_bytes": 1, "peak_reserved_gpu_bytes": 2, "retained_bytes": 3, "calls": {}},
                "scientific_record": y2.file_record(scientific_path), "array_records": [], "attempt_records": attempts,
            }
            y2.finish_unit(unit_root, "VALID", payload, worker_count=worker_count)
        pending = shard_root / "completion/shard.pending.json"
        y2.write_json_new(pending, y2.signed("stage_y_y2_shard_pending", "PENDING", {"shard_id": shard, "shard_count": worker_count, "terminal_published": False}))
        fixed = {"gpu_active_seconds": 0.0, "cpu_seconds": 0.0, "wall_seconds": 0.0, "peak_reserved_gpu_bytes": 0, "retained_bytes": 0}
        terminal = y2.signed("stage_y_y2_shard_terminal", "VALID", {"shard_id": shard, "shard_count": worker_count, "unit_count": len(manifest["payload"]["records"]), "terminal_counts": {"VALID": len(manifest["payload"]["records"]), "FAILED_NUMERICAL": 0, "BLOCKED_INFRASTRUCTURE": 0, "BLOCKED_BUDGET": 0}, "worker_identity": binding, "fixed_initialization_resource": fixed, "resource_caps": manifest["payload"]["resource_caps"], "shard_elapsed_seconds": 0.0, "completed_unix_ns": 1})
        y2.publish_terminal(pending, shard_root / "completion/shard.json", terminal)


def test_unit_receipt_is_atomic_and_reconciler_sums_all_shards_cpu_only(tmp_path: Path) -> None:
    _populate_synthetic_shards(tmp_path)
    assert not list(tmp_path.glob("shards/shard-*/units/unit-*/completion/*.pending.json"))
    result = y2.reconcile_shards(tmp_path)
    assert result["status"] == "PASS"
    assert result["payload"]["valid_count"] == 100
    assert result["payload"]["attempt_record_count"] == 100
    assert result["payload"]["resource_totals"]["h20_gpu_hours"] == pytest.approx(1.0)
    assert result["payload"]["gpu_hours_accounting"] == "SUM_ACROSS_ALL_SHARDS_NOT_DIVIDED_BY_WORKER_COUNT"
    assert result["payload"]["bootstrap_executed"] is result["payload"]["holm_executed"] is result["payload"]["claim_decision_executed"] is False


def test_reconciler_rejects_missing_shard_duplicate_mapping_and_environment_drift(tmp_path: Path) -> None:
    _populate_synthetic_shards(tmp_path)
    missing = tmp_path / "shards/shard-07"; moved = tmp_path / "shards/shard-missing"; missing.rename(moved)
    with pytest.raises(y2.StageYY2Error): y2.reconcile_shards(tmp_path)
    moved.rename(missing)
    identity_path = tmp_path / "shards/shard-07/worker_identity.json"
    identity = y2.read_json(identity_path); identity["payload"]["gpu_uuid"] = "GPU-SYNTHETIC-0"
    unsigned = dict(identity); unsigned.pop("canonical_digest"); identity["canonical_digest"] = y2.canonical_digest(unsigned)
    identity_path.write_bytes(y2.canonical_bytes(identity) + b"\n")
    with pytest.raises(y2.StageYY2Error): y2.reconcile_shards(tmp_path)


def test_worker_identity_and_unit_terminal_payloads_fail_closed(tmp_path: Path) -> None:
    valid = {"physical_index": 0, "cuda_visible_devices": "0", "gpu_uuid": "GPU-X", "pci_bus_id": "0000:00:00.0", "device_name": "NVIDIA H20", "hostname": "h", "pid": 1, "environment_digest": "e" * 64}
    assert y2.worker_binding(valid) == valid
    with pytest.raises(y2.StageYY2ScheduleError): y2.worker_binding({**valid, "gpu_uuid": ""})
    root = tmp_path / "unit"; y2.begin_unit(root, shard_id=0, selection_order=0, source_object_id_value="synthetic:0")
    y2.append_attempt(root, {"attempt_id": "a"})
    with pytest.raises(y2.StageYY2ProvenanceError): y2.finish_unit(root, "VALID", {"selection_order": 0})


def test_cli_surface_and_cpu_only_p0_import_boundary() -> None:
    launcher = (REPO / "scripts/run_stage_y_y2.py").read_text(); ast.parse(launcher)
    for command in ("seal-cohort","build-shards","validate-startup","run-shard","run-all","reconcile","seal","validate","validate-publication"):
        assert command in launcher
    core = (REPO / "src/vitnns/stage_y_y2.py").read_text(); ast.parse(core)
    assert "import torch" not in core and "from PIL" not in core and "import timm" not in core
    worker = (REPO / "src/vitnns/stage_y_y2_worker.py").read_text(); ast.parse(worker)
    assert "unit_gpu_end.record()" in worker and "time.process_time() - unit_cpu_started" in worker
    assert '"gpu_active_seconds": 0.0' not in worker
    validator = (REPO / "scripts/validate_stage_y_y2_p0.py").read_text(); ast.parse(validator)
    assert "nvidia-smi" not in validator


def test_idle_device_selection_uses_four_free_cards_without_gpu_access() -> None:
    rows = [{"physical_index": i, "device_name": "NVIDIA H20", "gpu_uuid": f"GPU-{i}", "pci_bus_id": f"0000:{i:02x}:00.0", "total_memory_mib": 98304, "free_memory_mib": 90000, "driver": "synthetic"} for i in range(8)]
    selected = y2.select_idle_h20_devices(rows, {"GPU-1","GPU-3","GPU-5","GPU-7"})
    assert [row["physical_index"] for row in selected] == [0,2,4,6]
    assert y2.shard_counts(100, len(selected)) == [25] * 4
    with pytest.raises(y2.StageYY2ScheduleError, match="no eligible"):
        y2.select_idle_h20_devices(rows, {f"GPU-{i}" for i in range(8)})


def test_formal_y2_root_is_absent_during_p0() -> None:
    assert not (REPO / y2.ROOT).exists()
    assert not (REPO / y2.REPORT_ROOT).exists()
