from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vitnns import stage_c_cli
from vitnns.stage_c_provenance import build_unit_identity
from vitnns.stage_c_runner import StageCBatchValidationError


def _stdout_json(capsys) -> dict:
    captured = capsys.readouterr()
    assert captured.err == ""
    return json.loads(captured.out)


def _write_explicit_inputs(tmp_path: Path) -> tuple[Path, Path]:
    config = tmp_path / "stage_c.json"
    run_spec = tmp_path / "run_spec.json"
    config.write_text("{}\n", encoding="utf-8")
    run_spec.write_text("{}\n", encoding="utf-8")
    return config, run_spec


def _install_plan_pipeline(monkeypatch, *, within_budget: bool = True):
    config = object()
    registry = object()
    selection = object()
    run_spec = SimpleNamespace(sample_ids=("sample-a",), to_dict=lambda: {})
    plan = SimpleNamespace(units=(object(),), digest="a" * 64, to_dict=lambda: {})
    estimate = SimpleNamespace(
        within_budget=within_budget,
        violations=() if within_budget else ("forward_calls_per_execution_unit_max",),
        to_dict=lambda: {
            "within_budget": within_budget,
            "violations": [] if within_budget else ["forward_calls_per_execution_unit_max"],
        },
    )
    observed: list[tuple[str, object]] = []

    def record(name, value):
        observed.append((name, value))
        return value

    monkeypatch.setattr(stage_c_cli, "load_stage_c_config", lambda path: record("config", config))
    monkeypatch.setattr(stage_c_cli, "load_stage_c_registry", lambda loaded, **kwargs: record("registry", registry))
    monkeypatch.setattr(stage_c_cli, "load_stage_c_selection", lambda loaded, **kwargs: record("selection", selection))
    monkeypatch.setattr(stage_c_cli, "load_run_spec", lambda path: record("run_spec", run_spec))
    monkeypatch.setattr(stage_c_cli, "build_execution_plan", lambda *args, **kwargs: record("plan", plan))
    monkeypatch.setattr(stage_c_cli, "estimate_stage_c_run", lambda *args, **kwargs: record("estimate", estimate))
    monkeypatch.setattr(
        stage_c_cli, "validate_c7_execution_plan",
        lambda *args, **kwargs: {"valid": False},
    )
    monkeypatch.setattr(
        stage_c_cli, "validate_c7_diagnostic_plan",
        lambda *args, **kwargs: {"valid": False},
    )
    monkeypatch.setattr(
        stage_c_cli, "require_fresh_c7_output_root",
        lambda *args, **kwargs: None,
    )
    return SimpleNamespace(
        config=config, registry=registry, selection=selection, run_spec=run_spec,
        plan=plan, estimate=estimate, observed=observed,
    )


def test_estimate_requires_explicit_config(capsys, tmp_path):
    assert stage_c_cli.main(["estimate", "--run-spec", str(tmp_path / "spec.json")]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    error = json.loads(captured.err)["error"]
    assert error["type"] == "ArgumentError"
    assert "--config" in error["message"]


@pytest.mark.parametrize(
    ("command", "extra"),
    [
        ("estimate", []),
        ("run-sample", ["--output-root", "out", "--sample-id", "sample-a"]),
        ("run-batch", ["--output-root", "out"]),
        ("benchmark", ["--output-root", "out", "--c7-diagnostic-gate", "gate.json"]),
        ("validate", ["--output-root", "out"]),
        ("aggregate", ["--output-root", "out"]),
        ("c7-diagnostic-gate", ["--output-root", "out"]),
        (
            "benchmark-report",
            ["--output-root", "out", "--c7-diagnostic-gate", "gate.json"],
        ),
    ],
)
def test_every_command_requires_an_explicit_run_spec(capsys, tmp_path, command, extra):
    argv = [command, "--config", str(tmp_path / "config.json"), *extra]

    assert stage_c_cli.main(argv) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    error = json.loads(captured.err)["error"]
    assert error["type"] == "ArgumentError"
    assert "--run-spec" in error["message"]


def test_estimate_prints_one_stable_json_document(monkeypatch, capsys, tmp_path):
    config_path, spec_path = _write_explicit_inputs(tmp_path)
    pipeline = _install_plan_pipeline(monkeypatch)

    assert stage_c_cli.main([
        "estimate", "--config", str(config_path), "--run-spec", str(spec_path)
    ]) == 0

    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == (
        json.dumps(
            {"violations": [], "within_budget": True},
            indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False,
        )
        + "\n"
    )
    payload = json.loads(captured.out)
    assert payload == {"violations": [], "within_budget": True}
    assert [name for name, _value in pipeline.observed] == [
        "config", "registry", "selection", "run_spec", "plan", "estimate"
    ]


def test_estimate_over_budget_has_json_and_exit_two(monkeypatch, capsys, tmp_path):
    config_path, spec_path = _write_explicit_inputs(tmp_path)
    _install_plan_pipeline(monkeypatch, within_budget=False)

    code = stage_c_cli.main([
        "estimate", "--config", str(config_path), "--run-spec", str(spec_path)
    ])

    assert code == 2
    assert _stdout_json(capsys)["within_budget"] is False


def test_run_preflight_failure_does_not_construct_backend_or_create_output(
    monkeypatch, capsys, tmp_path
):
    config_path, spec_path = _write_explicit_inputs(tmp_path)
    output_root = tmp_path / "must-not-exist"
    _install_plan_pipeline(monkeypatch, within_budget=False)
    backend_called = False

    def backend_factory(**_kwargs):
        nonlocal backend_called
        backend_called = True
        raise AssertionError("backend factory ran before the preflight gate")

    code = stage_c_cli.main(
        [
            "run-batch", "--config", str(config_path),
            "--run-spec", str(spec_path), "--output-root", str(output_root),
        ],
        backend_factory=backend_factory,
    )

    assert code == 2
    assert backend_called is False
    assert not output_root.exists()
    assert _stdout_json(capsys)["within_budget"] is False


@pytest.mark.parametrize(
    ("command", "extra", "runner_name", "expected_resume"),
    [
        (
            "run-sample", ["--sample-id", "sample-a"],
            "run_stage_c_sample", True,
        ),
        ("run-batch", ["--no-resume"], "run_stage_c_batch", False),
    ],
)
def test_run_commands_dispatch_explicit_plan_and_emit_json(
    monkeypatch, capsys, tmp_path, command, extra, runner_name, expected_resume
):
    config_path, spec_path = _write_explicit_inputs(tmp_path)
    output_root = tmp_path / "output"
    pipeline = _install_plan_pipeline(monkeypatch)
    backend = object()
    observed = {}

    def fake_run(*args, **kwargs):
        observed["args"] = args
        observed["kwargs"] = kwargs
        return SimpleNamespace(
            successful=True, orchestration_complete=True,
            to_dict=lambda: {
                "successful": True, "orchestration_complete": True,
                "unit_results": [],
            },
        )

    monkeypatch.setattr(stage_c_cli, runner_name, fake_run)
    argv = [
        command, "--config", str(config_path), "--run-spec", str(spec_path),
        "--output-root", str(output_root), "--device", "cpu", *extra,
    ]

    assert stage_c_cli.main(argv, backend_factory=lambda **_kwargs: backend) == 0
    assert observed["args"][0] is pipeline.plan
    assert observed["kwargs"]["backend"] is backend
    assert observed["kwargs"]["output_root"] == output_root
    assert observed["kwargs"]["estimate"] is pipeline.estimate
    assert observed["kwargs"]["resume"] is expected_resume
    if command == "run-sample":
        assert observed["args"][1] == "sample-a"
    assert _stdout_json(capsys)["successful"] is True


def test_run_batch_terminal_failure_has_exit_three(monkeypatch, capsys, tmp_path):
    config_path, spec_path = _write_explicit_inputs(tmp_path)
    output_root = tmp_path / "output"
    _install_plan_pipeline(monkeypatch)
    monkeypatch.setattr(
        stage_c_cli,
        "run_stage_c_batch",
        lambda *args, **kwargs: SimpleNamespace(
            successful=False, orchestration_complete=True,
            to_dict=lambda: {"successful": False, "orchestration_complete": True},
        ),
    )

    code = stage_c_cli.main(
        [
            "run-batch", "--config", str(config_path),
            "--run-spec", str(spec_path), "--output-root", str(output_root),
        ],
        backend_factory=lambda **_kwargs: object(),
    )

    assert code == 3
    assert _stdout_json(capsys)["successful"] is False


def test_validate_integrity_failure_has_exit_four(monkeypatch, capsys, tmp_path):
    config_path, spec_path = _write_explicit_inputs(tmp_path)
    output_root = tmp_path / "output"
    _install_plan_pipeline(monkeypatch)
    monkeypatch.setattr(
        stage_c_cli,
        "validate_stage_c_batch",
        lambda *args, **kwargs: SimpleNamespace(
            valid=False, orchestration_complete=False, successful=False,
            to_dict=lambda: {
                "valid": False, "orchestration_complete": False,
                "successful": False, "errors": ["artifact hash mismatch"],
            },
        ),
    )

    code = stage_c_cli.main([
        "validate", "--config", str(config_path), "--run-spec", str(spec_path),
        "--output-root", str(output_root),
    ])

    assert code == 4
    assert _stdout_json(capsys)["errors"] == ["artifact hash mismatch"]


def test_aggregate_dispatches_without_constructing_backend(monkeypatch, capsys, tmp_path):
    config_path, spec_path = _write_explicit_inputs(tmp_path)
    output_root = tmp_path / "output"
    _install_plan_pipeline(monkeypatch)
    backend_called = False

    def backend_factory(**_kwargs):
        nonlocal backend_called
        backend_called = True
        raise AssertionError("aggregate must not construct the compute backend")

    monkeypatch.setattr(
        stage_c_cli,
        "aggregate_stage_c_batch",
        lambda root, plan, **kwargs: {
            "document_type": "stage_c_l0_long_table", "row_count": 1,
            "source_levels": ["L0"],
        },
    )

    assert stage_c_cli.main(
        [
            "aggregate", "--config", str(config_path),
            "--run-spec", str(spec_path), "--output-root", str(output_root),
        ],
        backend_factory=backend_factory,
    ) == 0
    assert backend_called is False
    assert _stdout_json(capsys)["source_levels"] == ["L0"]


def test_aggregate_integrity_failure_has_exit_four(monkeypatch, capsys, tmp_path):
    config_path, spec_path = _write_explicit_inputs(tmp_path)
    output_root = tmp_path / "output"
    _install_plan_pipeline(monkeypatch)
    monkeypatch.setattr(
        stage_c_cli,
        "aggregate_stage_c_batch",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            StageCBatchValidationError("artifact hash mismatch")
        ),
    )

    code = stage_c_cli.main([
        "aggregate", "--config", str(config_path),
        "--run-spec", str(spec_path), "--output-root", str(output_root),
    ])

    assert code == 4
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err)["error"] == {
        "message": "artifact hash mismatch",
        "type": "StageCBatchValidationError",
    }


@pytest.mark.parametrize(
    ("command", "writer_name", "suffix"),
    [
        (
            "c7-diagnostic-gate", "write_c7_diagnostic_gate",
            ".c7-diagnostic-gate.json",
        ),
        ("benchmark-report", "write_benchmark_report", ".benchmark-report.json"),
    ],
)
def test_c7_reports_dispatch_without_constructing_backend(
    monkeypatch, capsys, tmp_path, command, writer_name, suffix
):
    config_path, spec_path = _write_explicit_inputs(tmp_path)
    output_root = tmp_path / "source-batch"
    pipeline = _install_plan_pipeline(monkeypatch)
    git_state = {"commit": "a" * 40, "dirty": False}
    observed = {}
    monkeypatch.setattr(stage_c_cli, "capture_git_state", lambda _root: git_state)
    monkeypatch.setattr(stage_c_cli, "_configure_c7_execution", lambda _device: None)
    monkeypatch.setattr(
        stage_c_cli, "capture_benchmark_environment",
        lambda device: {"device": device, "device_name": "NVIDIA H20"},
    )
    monkeypatch.setattr(stage_c_cli, "sha256_file", lambda _path: "d" * 64)

    def fake_writer(destination, root, plan, **kwargs):
        observed.update({
            "destination": destination, "root": root, "plan": plan,
            "kwargs": kwargs,
        })
        return {
            "document_type": f"stage_c_{command.replace('-', '_')}",
            "status": "PASS", "exit_class": "success",
        }

    monkeypatch.setattr(stage_c_cli, writer_name, fake_writer)
    code = stage_c_cli.main([
        command, "--config", str(config_path), "--run-spec", str(spec_path),
        "--output-root", str(output_root), "--device", "cuda:0",
        *(
            ["--c7-diagnostic-gate", str(tmp_path / "gate.json")]
            if command == "benchmark-report" else []
        ),
    ], backend_factory=lambda **_kwargs: pytest.fail("backend constructed"))

    assert code == 0
    assert observed == {
        "destination": output_root.with_name(output_root.name + suffix),
        "root": output_root, "plan": pipeline.plan,
        "kwargs": {
            "config": pipeline.config,
            "environment": {"device": "cuda:0", "device_name": "NVIDIA H20"},
            "run_spec_path": spec_path,
            "run_spec_file_sha256": "d" * 64,
            "observed_git_state": git_state,
            **({"recomputed_estimate": pipeline.estimate}
               if command == "benchmark-report" else {}),
            **({"c7_diagnostic_gate": tmp_path / "gate.json"}
               if command == "benchmark-report" else {}),
        },
    }
    assert _stdout_json(capsys)["status"] == "PASS"


@pytest.mark.parametrize(
    ("command", "writer_name"),
    [
        ("c7-diagnostic-gate", "write_c7_diagnostic_gate"),
        ("benchmark-report", "write_benchmark_report"),
    ],
)
def test_c7_valid_blocker_has_exit_two(
    monkeypatch, capsys, tmp_path, command, writer_name
):
    config_path, spec_path = _write_explicit_inputs(tmp_path)
    _install_plan_pipeline(monkeypatch)
    monkeypatch.setattr(
        stage_c_cli, "capture_git_state",
        lambda _root: {"commit": "a" * 40, "dirty": False},
    )
    monkeypatch.setattr(
        stage_c_cli, "capture_benchmark_environment", lambda _device: {}
    )
    monkeypatch.setattr(stage_c_cli, "_configure_c7_execution", lambda _device: None)
    monkeypatch.setattr(stage_c_cli, "sha256_file", lambda _path: "d" * 64)
    monkeypatch.setattr(
        stage_c_cli, writer_name,
        lambda *_args, **_kwargs: {
            "status": "BLOCKED", "exit_class": "preflight",
            "blocked_reasons": ["hard cap exceeded"],
        },
    )

    code = stage_c_cli.main([
        command, "--config", str(config_path), "--run-spec", str(spec_path),
        "--output-root", str(tmp_path / "batch"),
        *(
            ["--c7-diagnostic-gate", str(tmp_path / "gate.json")]
            if command == "benchmark-report" else []
        ),
    ])

    assert code == 2
    assert _stdout_json(capsys)["status"] == "BLOCKED"


def test_benchmark_report_terminal_failure_has_exit_three(
    monkeypatch, capsys, tmp_path
):
    config_path, spec_path = _write_explicit_inputs(tmp_path)
    _install_plan_pipeline(monkeypatch)
    monkeypatch.setattr(
        stage_c_cli, "capture_git_state",
        lambda _root: {"commit": "a" * 40, "dirty": False},
    )
    monkeypatch.setattr(
        stage_c_cli, "capture_benchmark_environment", lambda _device: {}
    )
    monkeypatch.setattr(stage_c_cli, "_configure_c7_execution", lambda _device: None)
    monkeypatch.setattr(stage_c_cli, "sha256_file", lambda _path: "d" * 64)
    monkeypatch.setattr(
        stage_c_cli, "write_benchmark_report",
        lambda *_args, **_kwargs: {
            "status": "BLOCKED", "exit_class": "terminal",
            "blocked_reasons": ["one unit failed"],
        },
    )

    code = stage_c_cli.main([
        "benchmark-report", "--config", str(config_path),
        "--run-spec", str(spec_path),
        "--output-root", str(tmp_path / "batch"),
        "--c7-diagnostic-gate", str(tmp_path / "gate.json"),
    ])

    assert code == 3
    assert _stdout_json(capsys)["exit_class"] == "terminal"


def test_benchmark_integrity_error_has_exit_four(
    monkeypatch, capsys, tmp_path
):
    config_path, spec_path = _write_explicit_inputs(tmp_path)
    _install_plan_pipeline(monkeypatch)
    monkeypatch.setattr(
        stage_c_cli, "capture_git_state",
        lambda _root: {"commit": "a" * 40, "dirty": False},
    )
    monkeypatch.setattr(
        stage_c_cli, "capture_benchmark_environment", lambda _device: {}
    )
    monkeypatch.setattr(stage_c_cli, "_configure_c7_execution", lambda _device: None)
    monkeypatch.setattr(stage_c_cli, "sha256_file", lambda _path: "d" * 64)
    monkeypatch.setattr(
        stage_c_cli, "write_benchmark_report",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            stage_c_cli.StageCBenchmarkIntegrityError("source hash mismatch")
        ),
    )

    code = stage_c_cli.main([
        "benchmark-report", "--config", str(config_path),
        "--run-spec", str(spec_path),
        "--output-root", str(tmp_path / "batch"),
        "--c7-diagnostic-gate", str(tmp_path / "gate.json"),
    ])

    assert code == 4
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err)["error"] == {
        "message": "source hash mismatch",
        "type": "StageCBenchmarkIntegrityError",
    }


def test_exact_c7_run_requires_gate_before_backend_or_output(
    monkeypatch, capsys, tmp_path
):
    config_path, spec_path = _write_explicit_inputs(tmp_path)
    _install_plan_pipeline(monkeypatch)
    monkeypatch.setattr(stage_c_cli, "_is_exact_c7_plan", lambda *_args: True)
    output_root = tmp_path / "must-stay-absent"

    code = stage_c_cli.main([
        "run-sample", "--config", str(config_path), "--run-spec", str(spec_path),
        "--output-root", str(output_root), "--sample-id", "sample-a",
    ], backend_factory=lambda **_kwargs: pytest.fail("backend constructed"))

    assert code == 2
    assert not output_root.exists()
    assert "--c7-diagnostic-gate" in _stdout_json(capsys)["blocked_reasons"][0]


def test_exact_c7_gate_is_checked_before_backend_and_output(
    monkeypatch, capsys, tmp_path
):
    config_path, spec_path = _write_explicit_inputs(tmp_path)
    output_root = tmp_path / "must-stay-absent"
    pipeline = _install_plan_pipeline(monkeypatch)
    monkeypatch.setattr(stage_c_cli, "_is_exact_c7_plan", lambda *_args: True)
    monkeypatch.setattr(
        stage_c_cli, "capture_git_state",
        lambda _root: {"commit": "a" * 40, "dirty": False},
    )
    observed = {}

    def block(gate, plan, **kwargs):
        observed.update({"gate": gate, "plan": plan, **kwargs})
        raise stage_c_cli.StageCBenchmarkBlockedError(
            "diagnostic wall time is 720 seconds",
            reasons=("diagnostic wall time must be below 720 seconds",),
        )

    monkeypatch.setattr(stage_c_cli, "require_c7_launch_gate", block)
    gate = tmp_path / "gate.json"
    code = stage_c_cli.main([
        "run-sample", "--config", str(config_path), "--run-spec", str(spec_path),
        "--output-root", str(output_root), "--sample-id", "sample-a",
        "--c7-diagnostic-gate", str(gate),
    ], backend_factory=lambda **_kwargs: pytest.fail("backend constructed"))

    assert code == 2
    assert observed["gate"] == gate
    assert observed["plan"] is pipeline.plan
    assert observed["output_root"] == output_root
    assert not output_root.exists()
    assert "below 720" in _stdout_json(capsys)["blocked_reasons"][0]


def test_benchmark_command_requires_exact_scope(monkeypatch, capsys, tmp_path):
    config_path, spec_path = _write_explicit_inputs(tmp_path)
    _install_plan_pipeline(monkeypatch)
    monkeypatch.setattr(stage_c_cli, "_is_exact_c7_plan", lambda *_args: False)

    code = stage_c_cli.main([
        "benchmark", "--config", str(config_path), "--run-spec", str(spec_path),
        "--output-root", str(tmp_path / "out"),
        "--c7-diagnostic-gate", str(tmp_path / "gate.json"),
    ], backend_factory=lambda **_kwargs: pytest.fail("backend constructed"))

    assert code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "exact checked-in C7" in json.loads(captured.err)["error"]["message"]


def test_benchmark_command_checks_gate_then_dispatches_batch(
    monkeypatch, capsys, tmp_path
):
    config_path, spec_path = _write_explicit_inputs(tmp_path)
    pipeline = _install_plan_pipeline(monkeypatch)
    monkeypatch.setattr(stage_c_cli, "_is_exact_c7_plan", lambda *_args: True)
    monkeypatch.setattr(
        stage_c_cli, "capture_git_state",
        lambda _root: {"commit": "a" * 40, "dirty": False},
    )
    events = []
    monkeypatch.setattr(
        stage_c_cli, "require_c7_launch_gate",
        lambda *_args, **_kwargs: events.append("gate") or {"status": "PASS"},
    )
    monkeypatch.setattr(
        stage_c_cli, "write_c7_launch_authorization",
        lambda *_args, **_kwargs: events.append("authorization"),
    )
    monkeypatch.setattr(
        stage_c_cli, "_configure_c7_execution",
        lambda device: events.append(f"configure:{device}"),
    )
    monkeypatch.setattr(
        stage_c_cli, "run_stage_c_batch",
        lambda *_args, **_kwargs: (
            events.append("run")
            or SimpleNamespace(
                successful=True, orchestration_complete=True,
                to_dict=lambda: {
                    "successful": True, "orchestration_complete": True,
                },
            )
        ),
    )
    backend = object()

    code = stage_c_cli.main([
        "benchmark", "--config", str(config_path), "--run-spec", str(spec_path),
        "--output-root", str(tmp_path / "fresh"),
        "--c7-diagnostic-gate", str(tmp_path / "gate.json"),
    ], backend_factory=lambda **_kwargs: events.append("backend") or backend)

    assert code == 0
    assert events == ["gate", "configure:cpu", "authorization", "backend", "run"]
    assert pipeline.plan is not None
    assert _stdout_json(capsys)["successful"] is True


def test_c7_configuration_sets_frozen_determinism(monkeypatch):
    seeded = []
    deterministic = []
    monkeypatch.setattr(stage_c_cli, "seed_everything", seeded.append)
    monkeypatch.setattr(
        torch, "use_deterministic_algorithms", deterministic.append
    )
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", True)

    stage_c_cli._configure_c7_execution("cpu")

    assert seeded == [20260821]
    assert deterministic == [True]
    assert torch.backends.cudnn.benchmark is False


@pytest.mark.parametrize("workspace", [None, "", "4096:8", ":1:1"])
def test_c7_cuda_configuration_rejects_bad_workspace_before_seeding(
    monkeypatch, workspace
):
    if workspace is None:
        monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    else:
        monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", workspace)
    seeded = []
    monkeypatch.setattr(stage_c_cli, "seed_everything", seeded.append)

    with pytest.raises(RuntimeError, match="CUBLAS_WORKSPACE_CONFIG"):
        stage_c_cli._configure_c7_execution("cuda:0")

    assert seeded == []


def test_c7_cuda_configuration_disables_tf32(monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setattr(stage_c_cli, "seed_everything", lambda _seed: None)
    monkeypatch.setattr(torch, "use_deterministic_algorithms", lambda _flag: None)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", True)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", True)

    stage_c_cli._configure_c7_execution("cuda:0")

    assert torch.backends.cuda.matmul.allow_tf32 is False
    assert torch.backends.cudnn.allow_tf32 is False


def test_exact_diagnostic_run_configures_c7_before_backend(
    monkeypatch, capsys, tmp_path
):
    config_path, spec_path = _write_explicit_inputs(tmp_path)
    _install_plan_pipeline(monkeypatch)
    monkeypatch.setattr(
        stage_c_cli, "_is_exact_c7_diagnostic_plan", lambda *_args: True
    )
    events = []
    monkeypatch.setattr(
        stage_c_cli, "_configure_c7_execution",
        lambda device: events.append(f"configure:{device}"),
    )
    monkeypatch.setattr(
        stage_c_cli, "run_stage_c_sample",
        lambda *_args, **_kwargs: (
            events.append("run")
            or SimpleNamespace(
                successful=True, orchestration_complete=True,
                to_dict=lambda: {
                    "successful": True, "orchestration_complete": True,
                },
            )
        ),
    )

    code = stage_c_cli.main([
        "run-sample", "--config", str(config_path), "--run-spec", str(spec_path),
        "--output-root", str(tmp_path / "diagnostic"),
        "--sample-id", "sample-a", "--device", "cpu", "--no-resume",
    ], backend_factory=lambda **_kwargs: events.append("backend") or object())

    assert code == 0
    assert events == ["configure:cpu", "backend", "run"]
    assert _stdout_json(capsys)["successful"] is True


def test_exact_diagnostic_run_requires_fresh_root_before_backend(
    monkeypatch, capsys, tmp_path
):
    config_path, spec_path = _write_explicit_inputs(tmp_path)
    _install_plan_pipeline(monkeypatch)
    monkeypatch.setattr(
        stage_c_cli, "_is_exact_c7_diagnostic_plan", lambda *_args: True
    )
    monkeypatch.setattr(
        stage_c_cli, "require_fresh_c7_output_root",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            stage_c_cli.StageCBenchmarkBlockedError(
                "diagnostic output root is not fresh",
                reasons=("diagnostic output root is not fresh and empty",),
            )
        ),
    )

    code = stage_c_cli.main([
        "run-sample", "--config", str(config_path), "--run-spec", str(spec_path),
        "--output-root", str(tmp_path / "old-diagnostic"),
        "--sample-id", "sample-a", "--no-resume",
    ], backend_factory=lambda **_kwargs: pytest.fail("backend constructed"))

    assert code == 2
    assert "not fresh" in _stdout_json(capsys)["blocked_reasons"][0]


def test_default_backend_factory_binds_provenance_and_loader_receives_model(
    monkeypatch, tmp_path
):
    sample_id = "ILSVRC2012_val/ILSVRC2012_val_00000079.JPEG"
    image_path = tmp_path / "ILSVRC2012_val_00000079.JPEG"
    image_path.write_bytes(b"fixture-image")
    row = SimpleNamespace(
        sample_id=sample_id, relative_path=image_path.name, image_sha256="a" * 64
    )
    registry = SimpleNamespace(
        data_root=str(tmp_path), by_sample_id=lambda: {sample_id: row}
    )
    config, plan = object(), object()
    args = SimpleNamespace(
        state_dict=tmp_path / "model.pth", config=tmp_path / "stage_c.json"
    )
    backend = stage_c_cli._default_backend_factory(
        config=config, registry=registry, plan=plan, args=args
    )
    assert backend.config is config
    assert backend.registry is registry
    assert backend.plan is plan
    assert callable(backend.build_provenance)

    model = object()
    seen: dict[str, object] = {}
    specification = SimpleNamespace(
        input_size=(3, 224, 224), resize_shorter_side=248,
        center_crop=(224, 224), interpolation="bicubic",
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    )

    class FakeImage:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def convert(self, mode):
            assert mode == "RGB"
            return self

    from PIL import Image
    from torchvision.transforms import functional as transform_functional
    monkeypatch.setattr(stage_c_cli, "sha256_file", lambda _path: row.image_sha256)
    monkeypatch.setattr(Image, "open", lambda _path: FakeImage())
    # The closure must receive the exact model selected by the production
    # backend; patching internals beneath it keeps this test model-free.
    monkeypatch.setattr(backend, "_model", lambda _device: model)
    original_loader = backend.sample_loader

    def model_observing_loader(sample, received_model):
        seen["model"] = received_model
        return original_loader(sample, received_model)

    backend.sample_loader = model_observing_loader
    from vitnns import modeling
    monkeypatch.setattr(
        modeling, "canonical_preprocess_spec", lambda received: specification
    )
    monkeypatch.setattr(transform_functional, "resize", lambda image, *args, **kwargs: image)
    monkeypatch.setattr(transform_functional, "center_crop", lambda image, *args, **kwargs: image)
    monkeypatch.setattr(
        transform_functional, "pil_to_tensor",
        lambda _image: torch.zeros((3, 224, 224), dtype=torch.uint8),
    )
    unit = SimpleNamespace(
        key=build_unit_identity(
            sample_id=sample_id, method_id="vit_grad_cam_block11_norm1",
            config_id="global_vit_grad_cam_block11_norm1",
            block_index=None, head_index=None,
            target_ref="clean_predicted_class_raw_logit",
        ),
        sample=SimpleNamespace(sample_id=sample_id),
    )
    loaded = backend._sample_context(unit, "cpu", model)

    assert seen["model"] is model
    assert loaded["registry_row"] is row
    assert loaded["canonical_rgb"].shape == (1, 3, 224, 224)
    assert loaded["normalized_input"].shape == (1, 3, 224, 224)


def test_input_error_is_json_on_stderr_and_exit_one(monkeypatch, capsys, tmp_path):
    config_path, spec_path = _write_explicit_inputs(tmp_path)
    monkeypatch.setattr(
        stage_c_cli, "load_stage_c_config",
        lambda _path: (_ for _ in ()).throw(ValueError("bad explicit run input")),
    )

    code = stage_c_cli.main([
        "estimate", "--config", str(config_path), "--run-spec", str(spec_path)
    ])

    assert code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err) == {
        "error": {"message": "bad explicit run input", "type": "ValueError"}
    }
