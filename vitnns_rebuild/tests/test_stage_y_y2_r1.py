from __future__ import annotations

import ast
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from vitnns import stage_y_y2 as y2

REPO = Path(__file__).resolve().parents[1]
ORIGINAL = REPO / y2.ORIGINAL_ROOT


def _launcher():
    spec = importlib.util.spec_from_file_location("run_stage_y_y2_r1_test", REPO / "scripts/run_stage_y_y2.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    module.y2.activate_recovery_contract()
    return module


def test_run_spec_is_closed_and_preserves_scientific_contract() -> None:
    spec = json.loads((REPO / "config/stage_y/y2_r1_run_spec.json").read_text())
    schema = json.loads((REPO / "config/stage_y/y2_r1_run_spec.schema.json").read_text())
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(spec)
    old = json.loads((REPO / "config/stage_y/y2_run_spec.json").read_text())
    assert spec["execution"] == {key: old["execution"][key] for key in spec["execution"]}
    assert spec["coverage"] == old["coverage"]


def test_launcher_injects_exact_environment_before_every_child() -> None:
    source = (REPO / "scripts/run_stage_y_y2.py").read_text()
    tree = ast.parse(source)
    for name in ("_probe_devices", "run_shard", "run_all"):
        function = next(node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name)
        assignments = [node.lineno for node in ast.walk(function) if isinstance(node, ast.Assign) and "CUBLAS_WORKSPACE_CONFIG" in ast.unparse(node)]
        launches = [node.lineno for node in ast.walk(function) if isinstance(node, ast.Call) and (ast.unparse(node.func).endswith("subprocess.run") or ast.unparse(node.func).endswith("subprocess.Popen"))]
        assert assignments and launches and min(assignments) < max(launches)
    assert '"CUBLAS_WORKSPACE_CONFIG"] = y2.REQUIRED_CUBLAS_WORKSPACE_CONFIG' in source


@pytest.mark.parametrize("value", [None, ":16:8", "4096:8", ""])
def test_worker_wrong_or_missing_environment_fails_before_torch_import_and_jpeg(value: str | None) -> None:
    env = dict(os.environ); env["PYTHONPATH"] = str(REPO / "src"); env["STAGE_Y2_R1_RECOVERY"] = "1"
    if value is None: env.pop("CUBLAS_WORKSPACE_CONFIG", None)
    else: env["CUBLAS_WORKSPACE_CONFIG"] = value
    code = "import sys; from vitnns.stage_y_y2_worker import execute; print('torch_loaded_before_execute', 'torch' in sys.modules); execute('/nonexistent',0)"
    result = subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env, text=True, capture_output=True)
    assert result.returncode != 0
    assert "CUBLAS_WORKSPACE_CONFIG must equal :4096:8 before Torch initialization" in result.stderr
    assert "torch_loaded_before_execute False" in result.stdout


def test_startup_probe_runs_cublas_float32_float64_backward_and_exact_repeat() -> None:
    source = _launcher()._device_probe_script()
    ast.parse(source)
    assert "torch.matmul" in source and "torch.float32" in source and "torch.float64" in source
    assert "loss.backward()" in source and "torch.equal" in source
    assert "torch.use_deterministic_algorithms(True)" in source
    assert "warn_only=True" not in source


def test_environment_identity_is_closed_and_includes_executable_and_cublas() -> None:
    assert y2.ENVIRONMENT_IDENTITY_FIELDS == {
        "python_executable_realpath","python_version","torch_version","torchvision_version","timm_version","cuda_version","driver","device_name","cublas_workspace_config","deterministic_algorithms_enabled","deterministic_warn_only_enabled","cudnn_benchmark","cudnn_deterministic","cuda_matmul_allow_tf32","cudnn_allow_tf32","checkpoint_sha256","numeric_contract_digest"
    }


def test_original_hundred_failures_are_homogeneous_and_science_empty() -> None:
    paths = sorted(ORIGINAL.glob("shards/shard-*/units/unit-*/completion/unit.json"))
    terminals = [json.loads(path.read_text()) for path in paths]
    assert len(terminals) == 100 and {item["status"] for item in terminals} == {"BLOCKED_INFRASTRUCTURE"}
    codes = {item["payload"]["failure_code"] for item in terminals}
    assert len(codes) == 1 and "uses CuBLAS" in next(iter(codes))
    assert all(item["payload"]["target_identity"] == {} and item["payload"]["quadrature_attempts"] == [] for item in terminals)
    assert all(item["payload"]["array_records"] == [] and item["payload"]["scientific_record"] is None for item in terminals)


def test_original_completion_and_cohort_subset_identities_are_frozen() -> None:
    completion = json.loads((ORIGINAL / "completion/y2.json").read_text())
    cohort = json.loads((ORIGINAL / "cohort_registry.json").read_text())
    audit = json.loads((ORIGINAL / "audit_subset_registry.json").read_text())
    perturbation = json.loads((ORIGINAL / "perturbation_subset_registry.json").read_text())
    assert completion["status"] == "BLOCKED_INFRASTRUCTURE" and completion["payload"]["valid_clean_count"] == 0 and completion["payload"]["next_authorization"] == "NONE"
    assert cohort["canonical_digest"] == "156eab4bd86ff9979abc6acf8b06af3aa5a90e90cd02edef7713539b147ce277"
    assert cohort["payload"]["selection_digest"] == "8b3faa6d437bda38e963985ac12f14f3b440a3e4021915081f1d5e1b1c80d2cf"
    assert audit["payload"]["count"] == 10 and perturbation["payload"]["count"] == 20


def test_retry_identity_binds_original_terminal_and_attempt() -> None:
    worker = (REPO / "src/vitnns/stage_y_y2_worker.py").read_text()
    for token in ("supersedes_original_y2_unit","original_y2_unit_path","original_attempt_digest","original_terminal_digest","original_failure_code","CUBLAS_WORKSPACE_CONFIG_MISSING_BEFORE_TORCH_INITIALIZATION","retry_index","scientific_scope_changed","target_metric_tolerance_changed"):
        assert token in worker


def test_cumulative_resources_and_nonadaptive_sharding_are_enforced() -> None:
    core = (REPO / "src/vitnns/stage_y_y2.py").read_text()
    assert "original_y2_resource_totals" in core and "cumulative_resource_totals" in core
    assert "resource_accounting_includes_original_y2" in core
    for workers in range(1, 9):
        assert sorted(order for shard in y2.shard_ids(workers) for order in y2.expected_y2_orders(shard, workers)) == list(range(100))


def test_original_root_and_report_match_blocked_archive_commit() -> None:
    result = subprocess.run(["git","diff","--exit-code","1a621724b32b208036fb5f2a3993a48697e933c8","--",y2.ORIGINAL_ROOT.as_posix(),y2.ORIGINAL_REPORT_ROOT.as_posix()], cwd=REPO)
    assert result.returncode == 0


def test_false_pass_audit_rejects_all_sixteen_attacks() -> None:
    path = REPO / "scripts/validate_stage_y_y2_r1.py"
    spec = importlib.util.spec_from_file_location("validate_y2_r1_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    result = module.false_pass_audit(REPO)
    assert result["status"] == "PASS"
    assert result["attack_count"] == result["rejected_count"] == 16


def test_recovery_never_authorizes_y3_before_pass() -> None:
    source = (REPO / "scripts/run_stage_y_y2.py").read_text()
    assert "STAGE_Y3_PERTURBATION_MECHANISM_PRODUCTION_ONLY" in source
    assert '"bootstrap_executed": False' in source
    assert '"holm_executed": False' in source
    assert '"claim_decisions": []' in source
