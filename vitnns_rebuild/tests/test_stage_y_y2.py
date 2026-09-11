from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from vitnns import stage_y_y2 as y2

REPO = Path(__file__).resolve().parents[1]
LEDGER = REPO / "evidence/stage_y/y1_r1_resource_telemetry_waiver_20260911/formal/conservative_budget_receipt.json"


def _load_launcher():
    spec = importlib.util.spec_from_file_location("run_stage_y_y2_test", REPO / "scripts/run_stage_y_y2.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def test_device_probe_source_is_valid_python() -> None:
    source = (REPO / "scripts/run_stage_y_y2.py").read_text()
    tree = ast.parse(source)
    function = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "_device_probe_script")
    ast.parse(ast.literal_eval(function.body[0].value))


def test_all_closed_schemas_are_valid_and_run_spec_validates() -> None:
    names = ("y2_artifact.schema.json","y2_run_spec.schema.json","y2_unit_terminal.schema.json","y2_completion.schema.json","y2_startup_gate.schema.json","y2_shard_manifest.schema.json")
    # The generic artifact schema validates the signed envelope. Every
    # operational payload with a dedicated schema is recursively closed.
    for name in names:
        Draft202012Validator.check_schema(json.loads((REPO / "config/stage_y" / name).read_text()))
    schema = json.loads((REPO / "config/stage_y/y2_run_spec.schema.json").read_text())
    Draft202012Validator(schema).validate(json.loads((REPO / "config/stage_y/y2_run_spec.json").read_text()))
    for name in names[1:]:
        document = json.loads((REPO / "config/stage_y" / name).read_text())
        assert not any(
            isinstance(node, dict) and node.get("type") == "object"
            and "additionalProperties" not in node and "const" not in node
            for node in _walk_json(document)
        ), f"open object schema: {name}"


def _walk_json(value):
    yield value
    if isinstance(value, dict):
        for child in value.values(): yield from _walk_json(child)
    elif isinstance(value, list):
        for child in value: yield from _walk_json(child)


def test_all_worker_counts_cover_y2_and_y3_exactly_once() -> None:
    for workers in range(1, 9):
        y2_orders = [order for shard in y2.shard_ids(workers) for order in y2.expected_y2_orders(shard, workers)]
        assert sorted(y2_orders) == list(range(100)) and len(set(y2_orders)) == 100
        y3_units = [(image, transform) for shard in y2.shard_ids(workers) for image in range(20) for transform in range(8) if y2.y3_shard_id(image, transform, workers) == shard]
        assert sorted(y3_units) == [(image, transform) for image in range(20) for transform in range(8)]


def test_four_idle_devices_lock_four_25_unit_shards() -> None:
    rows = [{"physical_index": i,"device_name": "NVIDIA H20","gpu_uuid": f"GPU-{i}","pci_bus_id": f"0000:{i:02x}:00.0","total_memory_mib": 98304,"free_memory_mib": 90000,"driver": "synthetic"} for i in range(8)]
    selected = y2.select_idle_h20_devices(rows, {"GPU-1","GPU-3","GPU-5","GPU-7"})
    assert [row["physical_index"] for row in selected] == [0,2,4,6]
    assert y2.shard_counts(100, len(selected)) == [25,25,25,25]


def test_zero_idle_devices_and_more_than_eight_workers_fail_closed() -> None:
    row = {"physical_index": 0,"device_name": "NVIDIA H20","gpu_uuid": "GPU-0","pci_bus_id": "0000:00:00.0","total_memory_mib": 98304,"free_memory_mib": 90000,"driver": "synthetic"}
    with pytest.raises(y2.StageYY2ScheduleError): y2.select_idle_h20_devices([row], {"GPU-0"})
    with pytest.raises(y2.StageYY2ScheduleError): y2.shard_ids(9)
    with pytest.raises(y2.StageYY2ScheduleError): y2.shard_ids(4.5)


def test_budget_reprojection_changes_critical_path_not_aggregate_accounting() -> None:
    ledger = y2.read_json(LEDGER)
    one, four, eight = (y2.schedule_from_source_ledger(ledger, count) for count in (1,4,8))
    assert one["y2"]["critical_path_worker_wall_hours"] == 64.0
    assert four["y2"]["critical_path_worker_wall_hours"] == 16.0
    assert eight["y2"]["critical_path_worker_wall_hours"] == 9.0
    for result in (one, four, eight):
        assert result["y2"]["safety_projected"]["h20_gpu_hours"] == pytest.approx(63.97206770833333)
        assert result["y2"]["gpu_accounting"] == "SUM_ACROSS_WORKERS_NEVER_DIVIDE_BY_DEVICE_COUNT"


def test_worker_requires_launcher_authorization_and_persists_scientific_result() -> None:
    worker = (REPO / "src/vitnns/stage_y_y2_worker.py").read_text()
    assert "STAGE_Y2_LAUNCH_AUTHORIZATION" in worker
    assert "scientific_result.json" in worker and '"scientific_record"' in worker
    assert "failure_calls = error.calls" in worker
    module = ast.parse(worker)
    top_level_imports = [
        alias.name for node in module.body if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    ]
    assert not any(name in {"torch", "torchvision", "timm", "stage_y_y1", "stage_y_y1_worker"} or name.endswith("stage_y_y1") or name.endswith("stage_y_y1_worker") for name in top_level_imports)
    execute = next(node for node in ast.walk(module) if isinstance(node, ast.FunctionDef) and node.name == "execute")
    imported = [node.lineno for node in ast.walk(execute) if isinstance(node, (ast.Import, ast.ImportFrom)) and any(alias.name in {"torch","torchvision","timm","modeling"} or alias.name.endswith("modeling") for alias in node.names)]
    authorization = next(node.lineno for node in ast.walk(execute) if isinstance(node, ast.Compare) and "STAGE_Y2_LAUNCH_AUTHORIZATION" in ast.unparse(node))
    assert imported and min(imported) > authorization
    pending_gate = next(node.lineno for node in ast.walk(execute) if isinstance(node, ast.Raise) and "launcher-owned shard PENDING" in ast.unparse(node))
    assert min(imported) > pending_gate
    launcher = (REPO / "scripts/run_stage_y_y2.py").read_text()
    assert "worker fleet launch failed before the locked roster was fully started" in launcher
    assert "process.terminate()" in launcher
    assert launcher.count('env["PYTHONDONTWRITEBYTECODE"] = "1"') >= 2


def test_launcher_failure_path_atomically_seals_blocked_stage(tmp_path: Path) -> None:
    launcher = _load_launcher()
    root = tmp_path / y2.ROOT; report = tmp_path / y2.REPORT_ROOT
    (root / "completion").mkdir(parents=True)
    y2.write_json_new(root / "completion/y2.pending.json", y2.signed("stage_y_y2_pending_completion", "PENDING", {"terminal_published": False}))
    result = launcher.seal_control_failure(tmp_path, y2.StageYY2ScheduleError("no idle H20"), "validate-startup")
    terminal = y2.read_json(root / "completion/y2.json")
    assert result["status"] == terminal["status"] == "BLOCKED_RESOURCE_SCHEDULE"
    assert terminal["payload"]["next_authorization"] == "NONE"
    assert terminal["payload"]["formal_outcome_access"] is False
    assert not (root / "completion/y2.pending.json").exists()
    schema = json.loads((REPO / "config/stage_y/y2_completion.schema.json").read_text())
    Draft202012Validator(schema).validate(terminal)


def test_run_all_launches_locked_roster_concurrently_without_gpu_probe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    launcher = _load_launcher()
    root = tmp_path / y2.ROOT
    devices = [{"physical_index": index, "gpu_uuid": f"GPU-{index}"} for index in range(4)]
    gate = y2.signed("stage_y_y2_multi_gpu_startup_gate", "PASS", {"locked_worker_count": 4, "devices": devices})
    y2.write_json_new(root / "multi_gpu_startup_gate.json", gate)
    projection = y2.schedule_from_source_ledger(y2.read_json(LEDGER), 4)["y2"]
    for shard in range(4):
        shard_root = root / "shards" / f"shard-{shard:02d}"
        y2.write_json_new(shard_root / "shard_manifest.json", y2.signed("stage_y_y2_shard_manifest", "PASS", {"shard_id": shard, "unit_count": 25, "resource_caps": projection["per_shard"][shard]}))

    created = []
    class FakeProcess:
        def __init__(self, command, **kwargs):
            self.command = command; self.pid = 5000 + len(created); self.returncode = 0; created.append(self)
        def communicate(self):
            return (json.dumps({"status": "VALID"}), "")

    monkeypatch.setattr(launcher.subprocess, "Popen", FakeProcess)
    result = launcher.run_all(tmp_path)
    assert result == {"status": "PASS", "locked_worker_count": 4, "next_action": "reconcile"}
    assert len(created) == 4
    assert all((root / "shards" / f"shard-{shard:02d}/completion/shard.pending.json").is_file() for shard in range(4))
    roster = y2.read_json(root / "worker_launch_roster.json")
    assert roster["payload"]["concurrent_launch"] is True
    assert [row["shard_id"] for row in roster["payload"]["commands"]] == [0, 1, 2, 3]


def test_formal_validator_recomputes_inventory_and_supports_blocked_terminal() -> None:
    source = (REPO / "scripts/validate_stage_y_y2.py").read_text(); ast.parse(source)
    assert "whole_root_inventory" in source and "validate-blocked-publication" in source
    assert "reconcile_shards" in source and "next_authorization" in source
    launcher = (REPO / "scripts/run_stage_y_y2.py").read_text()
    assert "module.validate(repo, publication=publication, allow_blocked=False)" in launcher


def test_completion_schema_rejects_pass_without_reconciliation_and_blocked_without_failure() -> None:
    schema = json.loads((REPO / "config/stage_y/y2_completion.schema.json").read_text())
    validator = Draft202012Validator(schema)
    file_record = {"path": "synthetic.json", "bytes": 1, "sha256": "a" * 64}
    payload = {
        "terminal": True, "reconciliation": file_record, "failure": None,
        "whole_root_inventory": file_record, "terminal_access_summary": file_record,
        "report": file_record, "valid_clean_count": 90,
        "formal_outcome_access": True, "conditions": ["clean"],
        "bootstrap_executed": False, "holm_executed": False,
        "claim_decisions": [],
        "next_authorization": "STAGE_Y3_PERTURBATION_MECHANISM_PRODUCTION_ONLY",
    }
    valid = y2.signed("stage_y_y2_terminal_completion", "PASS", payload)
    assert not list(validator.iter_errors(valid))
    invalid_pass = y2.signed("stage_y_y2_terminal_completion", "PASS", {**payload, "reconciliation": None})
    assert list(validator.iter_errors(invalid_pass))
    invalid_blocked = y2.signed("stage_y_y2_terminal_completion", "BLOCKED_RESOURCE_SCHEDULE", {**payload, "reconciliation": None, "next_authorization": "NONE", "valid_clean_count": 0, "formal_outcome_access": False})
    assert list(validator.iter_errors(invalid_blocked))


def test_p0_r1_does_not_materialize_formal_y2() -> None:
    assert not (REPO / y2.ROOT).exists()
    assert not (REPO / y2.REPORT_ROOT).exists()
