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
    registry_kwargs: dict[str, object] = {}
    selection_kwargs: dict[str, object] = {}

    def record(name, value):
        observed.append((name, value))
        return value

    monkeypatch.setattr(stage_c_cli, "load_stage_c_config", lambda path: record("config", config))
    def load_registry(loaded, **kwargs):
        registry_kwargs.clear()
        registry_kwargs.update(kwargs)
        return record("registry", registry)

    monkeypatch.setattr(stage_c_cli, "load_stage_c_registry", load_registry)
    def load_selection(loaded, **kwargs):
        selection_kwargs.clear()
        selection_kwargs.update(kwargs)
        return record("selection", selection)

    monkeypatch.setattr(stage_c_cli, "load_stage_c_selection", load_selection)
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
        registry_kwargs=registry_kwargs, selection_kwargs=selection_kwargs,
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
        ("c8-init", ["--output-root", "out", "--c7-report", "c7.json"]),
        ("c8-run-shard", [
            "--output-root", "out",
            "--shard-index", "0", "--shard-count", "1",
        ]),
        ("c8-finalize", ["--output-root", "out"]),
        ("c8-select", [
            "--output-root", "out", "--report-output", "report.json",
        ]),
        ("c9-init", [
            "--output-root", "out", "--c8-root", "c8",
            "--c8-report", "c8-report.json",
            "--c8-recovery-selection", "recovery.json",
        ]),
        ("c9-run-shard", [
            "--output-root", "out",
            "--shard-index", "0", "--shard-count", "1",
        ]),
        ("c9-finalize", ["--output-root", "out"]),
        ("c9-validate", ["--output-root", "out"]),
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
    git_state = {"commit": "c" * 40, "dirty": False}
    monkeypatch.setattr(
        stage_c_cli, "capture_git_state", lambda _root: git_state
    )
    backend = stage_c_cli._default_backend_factory(
        config=config, registry=registry, plan=plan, args=args
    )
    assert backend.config is config
    assert backend.registry is registry
    assert backend.plan is plan
    assert backend.code_state == git_state
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


def _install_c8_pipeline(monkeypatch):
    pipeline = _install_plan_pipeline(monkeypatch)
    units = tuple(
        SimpleNamespace(
            key=SimpleNamespace(canonical_digest=digest),
            sample=SimpleNamespace(sample_id=f"calibration-{index}"),
        )
        for index, digest in enumerate(("1" * 64, "2" * 64))
    )
    pipeline.plan.units = units
    pipeline.plan.canonical_digest = "a" * 64
    pipeline.estimate.canonical_digest = "b" * 64
    monkeypatch.setattr(
        stage_c_cli, "validate_c8_execution_plan",
        lambda plan, config: {"valid": True, "violations": []},
    )
    return pipeline


def test_c8_init_gates_and_initializes_without_backend(
    monkeypatch, capsys, tmp_path
):
    config_path, spec_path = _write_explicit_inputs(tmp_path)
    pipeline = _install_c8_pipeline(monkeypatch)
    root = tmp_path / "c8"
    report = tmp_path / "c7.json"
    git_state = {"commit": "c" * 40, "dirty": False}
    authorization = {"canonical_digest": "d" * 64}
    measured = SimpleNamespace(canonical_digest="e" * 64)
    events = []
    monkeypatch.setattr(stage_c_cli, "capture_git_state", lambda _root: git_state)

    def authorize(plan, config, **kwargs):
        events.append(("authorize", kwargs))
        assert plan is pipeline.plan and config is pipeline.config
        return authorization, measured

    def initialize(output_root, plan, estimate, auth, **kwargs):
        events.append(("initialize", output_root, kwargs))
        assert (plan, estimate, auth) == (pipeline.plan, measured, authorization)
        return Path(output_root).absolute()

    monkeypatch.setattr(stage_c_cli, "build_c8_launch_authorization", authorize)
    monkeypatch.setattr(stage_c_cli, "initialize_c8_output_root", initialize)
    code = stage_c_cli.main([
        "c8-init", "--config", str(config_path), "--run-spec", str(spec_path),
        "--output-root", str(root), "--c7-report", str(report),
    ], backend_factory=lambda **_kwargs: pytest.fail("backend constructed"))

    assert code == stage_c_cli.EXIT_SUCCESS
    assert [event[0] for event in events] == ["authorize", "initialize"]
    assert events[0][1]["observed_git_state"] == git_state
    assert events[1][2] == {
        "config": pipeline.config,
        "observed_git_state": git_state,
        "repo_root": stage_c_cli.REPO_ROOT,
    }
    assert _stdout_json(capsys)["expected_unit_count"] == 2
    assert pipeline.registry_kwargs == {"verify_live_assets": False}
    assert "selection" not in [name for name, _value in pipeline.observed]


def test_c8_commands_reject_nonexact_plan_before_backend(
    monkeypatch, capsys, tmp_path
):
    config_path, spec_path = _write_explicit_inputs(tmp_path)
    _install_plan_pipeline(monkeypatch)
    monkeypatch.setattr(
        stage_c_cli, "validate_c8_execution_plan",
        lambda *_args: {"valid": False, "violations": ["scope drifted"]},
    )
    code = stage_c_cli.main([
        "c8-run-shard", "--config", str(config_path),
        "--run-spec", str(spec_path), "--output-root", str(tmp_path / "c8"),
        "--shard-index", "0", "--shard-count", "1",
    ], backend_factory=lambda **_kwargs: pytest.fail("backend constructed"))

    assert code == stage_c_cli.EXIT_INPUT_ERROR
    captured = capsys.readouterr()
    assert "exact checked-in" in json.loads(captured.err)["error"]["message"]


def test_c8_init_rejects_dirty_git_before_authorization_or_backend(
    monkeypatch, capsys, tmp_path
):
    config_path, spec_path = _write_explicit_inputs(tmp_path)
    _install_c8_pipeline(monkeypatch)
    monkeypatch.setattr(
        stage_c_cli, "capture_git_state",
        lambda _root: {"commit": "c" * 40, "dirty": True},
    )
    monkeypatch.setattr(
        stage_c_cli, "build_c8_launch_authorization",
        lambda *_args, **_kwargs: pytest.fail("authorization attempted"),
    )
    code = stage_c_cli.main([
        "c8-init", "--config", str(config_path),
        "--run-spec", str(spec_path), "--output-root", str(tmp_path / "c8"),
        "--c7-report", str(tmp_path / "c7.json"),
    ], backend_factory=lambda **_kwargs: pytest.fail("backend constructed"))

    assert code == stage_c_cli.EXIT_PREFLIGHT_BLOCKED
    assert "clean Git" in _stdout_json(capsys)["blocked_reasons"][0]


def test_c8_plan_order_shards_are_balanced_complete_and_disjoint():
    plan = SimpleNamespace(units=tuple(
        SimpleNamespace(key=SimpleNamespace(canonical_digest=f"{index:064x}"))
        for index in range(11)
    ))
    shards = [
        stage_c_cli._c8_shard_unit_digests(
            plan, shard_index=index, shard_count=3
        )
        for index in range(3)
    ]

    assert [len(shard) for shard in shards] == [4, 4, 3]
    expected = tuple(unit.key.canonical_digest for unit in plan.units)
    reconstructed = tuple(
        shards[index % len(shards)][index // len(shards)]
        for index in range(len(expected))
    )
    assert reconstructed == expected
    assert len(set().union(*(set(shard) for shard in shards))) == 11
    assert shards == [expected[0::3], expected[1::3], expected[2::3]]


@pytest.mark.parametrize(
    ("index", "count"),
    [(-1, 2), (2, 2), (0, 0), (0, 3)],
)
def test_c8_shard_rejects_invalid_index_or_empty_partition_before_backend(
    monkeypatch, capsys, tmp_path, index, count
):
    config_path, spec_path = _write_explicit_inputs(tmp_path)
    pipeline = _install_c8_pipeline(monkeypatch)
    authorization = {
        "c7_report_path": str((tmp_path / "c7.json").absolute()),
        "code": {"commit": "c" * 40, "dirty": False},
    }
    code = stage_c_cli.main([
        "c8-run-shard", "--config", str(config_path),
        "--run-spec", str(spec_path), "--output-root", str(tmp_path / "c8"),
        "--shard-index", str(index), "--shard-count", str(count),
    ], backend_factory=lambda **_kwargs: pytest.fail("backend constructed"))

    assert code == stage_c_cli.EXIT_INPUT_ERROR
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err)["error"]["type"] == "ArgumentError"


def test_c8_shard_rejects_dirty_or_changed_git_before_backend(
    monkeypatch, capsys, tmp_path
):
    config_path, spec_path = _write_explicit_inputs(tmp_path)
    pipeline = _install_c8_pipeline(monkeypatch)
    authorization = {
        "c7_report_path": str((tmp_path / "c7.json").absolute()),
        "code": {"commit": "c" * 40, "dirty": False},
    }
    monkeypatch.setattr(
        stage_c_cli, "_load_c8_initialized_estimate",
        lambda *_args: (pipeline.estimate, authorization),
    )
    monkeypatch.setattr(
        stage_c_cli, "validate_fixed_c7_report", lambda *_args, **_kwargs: {}
    )
    monkeypatch.setattr(
        stage_c_cli, "build_c8_measured_preflight",
        lambda *_args: pipeline.estimate,
    )
    monkeypatch.setattr(
        stage_c_cli, "_validate_c8_shard_checkpoint_locked", lambda *_args: None
    )
    monkeypatch.setattr(
        stage_c_cli, "capture_git_state",
        lambda _root: {"commit": "c" * 40, "dirty": True},
    )
    code = stage_c_cli.main([
        "c8-run-shard", "--config", str(config_path),
        "--run-spec", str(spec_path), "--output-root", str(tmp_path / "c8"),
        "--shard-index", "0", "--shard-count", "1",
    ], backend_factory=lambda **_kwargs: pytest.fail("backend constructed"))

    assert code == stage_c_cli.EXIT_INTEGRITY_FAILURE
    captured = capsys.readouterr()
    assert "clean Git state" in json.loads(captured.err)["error"]["message"]


def test_c8_shard_validates_controls_then_constructs_backend_and_runs(
    monkeypatch, capsys, tmp_path
):
    config_path, spec_path = _write_explicit_inputs(tmp_path)
    pipeline = _install_c8_pipeline(monkeypatch)
    git_state = {"commit": "c" * 40, "dirty": False}
    authorization = {
        "c7_report_path": str((tmp_path / "c7.json").absolute()),
        "code": git_state,
    }
    events = []
    monkeypatch.setattr(
        stage_c_cli, "_load_c8_initialized_estimate",
        lambda *_args: events.append("controls") or (pipeline.estimate, authorization),
    )
    monkeypatch.setattr(
        stage_c_cli, "validate_fixed_c7_report",
        lambda *_args, **_kwargs: events.append("report") or {},
    )
    monkeypatch.setattr(
        stage_c_cli, "build_c8_measured_preflight",
        lambda *_args: events.append("preflight") or pipeline.estimate,
    )
    monkeypatch.setattr(
        stage_c_cli, "_validate_c8_shard_checkpoint_locked",
        lambda *_args: events.append("checkpoint"),
    )
    monkeypatch.setattr(stage_c_cli, "capture_git_state", lambda _root: git_state)
    monkeypatch.setattr(
        stage_c_cli, "_configure_c8_execution",
        lambda device: events.append(f"configure:{device}"),
    )
    observed = {}

    def run_shard(*args, **kwargs):
        events.append("run")
        observed.update(kwargs)
        return ({"state": "succeeded"},)

    monkeypatch.setattr(stage_c_cli, "run_c8_shard", run_shard)
    code = stage_c_cli.main([
        "c8-run-shard", "--config", str(config_path),
        "--run-spec", str(spec_path), "--output-root", str(tmp_path / "c8"),
        "--shard-index", "0", "--shard-count", "2",
    ], backend_factory=lambda **_kwargs: events.append("backend") or object())

    assert code == stage_c_cli.EXIT_SUCCESS
    assert events == [
        "controls", "report", "preflight", "checkpoint",
        "configure:cpu", "backend", "run"
    ]
    assert observed["unit_digests"] == ("1" * 64,)
    assert observed["estimate"] is pipeline.estimate
    assert observed["observed_git_state"] == git_state
    result = _stdout_json(capsys)
    assert result["unit_count"] == 1
    assert (result["shard_index"], result["shard_count"]) == (0, 2)


def test_c8_shard_rejects_fresh_preflight_drift_before_backend(
    monkeypatch, capsys, tmp_path
):
    config_path, spec_path = _write_explicit_inputs(tmp_path)
    pipeline = _install_c8_pipeline(monkeypatch)
    authorization = {
        "c7_report_path": str((tmp_path / "c7.json").absolute()),
        "code": {"commit": "c" * 40, "dirty": False},
    }
    fresh = SimpleNamespace(
        to_dict=lambda: {"canonical_digest": "f" * 64}
    )
    monkeypatch.setattr(
        stage_c_cli, "_load_c8_initialized_estimate",
        lambda *_args: (pipeline.estimate, authorization),
    )
    monkeypatch.setattr(
        stage_c_cli, "validate_fixed_c7_report", lambda *_args, **_kwargs: {}
    )
    monkeypatch.setattr(
        stage_c_cli, "build_c8_measured_preflight", lambda *_args: fresh
    )

    code = stage_c_cli.main([
        "c8-run-shard", "--config", str(config_path),
        "--run-spec", str(spec_path), "--output-root", str(tmp_path / "c8"),
        "--shard-index", "0", "--shard-count", "2",
    ], backend_factory=lambda **_kwargs: pytest.fail("backend constructed"))

    assert code == stage_c_cli.EXIT_INTEGRITY_FAILURE
    captured = capsys.readouterr()
    assert "freshly measured" in json.loads(captured.err)["error"]["message"]


def test_c8_shard_rejects_legacy_unit_digest_option(capsys, tmp_path):
    config_path, spec_path = _write_explicit_inputs(tmp_path)
    code = stage_c_cli.main([
        "c8-run-shard", "--config", str(config_path),
        "--run-spec", str(spec_path), "--output-root", str(tmp_path / "c8"),
        "--shard-index", "0", "--shard-count", "1",
        "--unit-digest", "1" * 64,
    ])

    assert code == stage_c_cli.EXIT_INPUT_ERROR
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "unrecognized arguments" in json.loads(captured.err)["error"]["message"]


@pytest.mark.parametrize(
    ("state", "expected_code"),
    [
        ("failed", stage_c_cli.EXIT_TERMINAL_FAILURE),
        ("incomplete", stage_c_cli.EXIT_INTEGRITY_FAILURE),
    ],
)
def test_c8_shard_result_state_controls_exit(
    monkeypatch, capsys, tmp_path, state, expected_code
):
    config_path, spec_path = _write_explicit_inputs(tmp_path)
    pipeline = _install_c8_pipeline(monkeypatch)
    git_state = {"commit": "c" * 40, "dirty": False}
    authorization = {
        "c7_report_path": str((tmp_path / "c7.json").absolute()),
        "code": git_state,
    }
    monkeypatch.setattr(
        stage_c_cli, "_load_c8_initialized_estimate",
        lambda *_args: (pipeline.estimate, authorization),
    )
    monkeypatch.setattr(
        stage_c_cli, "validate_fixed_c7_report", lambda *_args, **_kwargs: {}
    )
    monkeypatch.setattr(
        stage_c_cli, "build_c8_measured_preflight",
        lambda *_args: pipeline.estimate,
    )
    monkeypatch.setattr(
        stage_c_cli, "_validate_c8_shard_checkpoint_locked", lambda *_args: None
    )
    monkeypatch.setattr(stage_c_cli, "capture_git_state", lambda _root: git_state)
    monkeypatch.setattr(stage_c_cli, "_configure_c8_execution", lambda _device: None)
    monkeypatch.setattr(
        stage_c_cli, "run_c8_shard", lambda *_args, **_kwargs: ({"state": state},),
    )

    code = stage_c_cli.main([
        "c8-run-shard", "--config", str(config_path),
        "--run-spec", str(spec_path), "--output-root", str(tmp_path / "c8"),
        "--shard-index", "0", "--shard-count", "2",
    ], backend_factory=lambda **_kwargs: object())

    assert code == expected_code
    assert _stdout_json(capsys)["status"] == "BLOCKED"


def test_c8_shard_rejects_already_claimed_row_before_backend(
    monkeypatch, capsys, tmp_path
):
    config_path, spec_path = _write_explicit_inputs(tmp_path)
    pipeline = _install_c8_pipeline(monkeypatch)
    root = tmp_path / "c8"
    authorization = {
        "c7_report_path": str((tmp_path / "c7.json").absolute()),
        "code": {"commit": "c" * 40, "dirty": False},
    }
    monkeypatch.setattr(
        stage_c_cli, "_load_c8_initialized_estimate",
        lambda *_args: (pipeline.estimate, authorization),
    )
    monkeypatch.setattr(
        stage_c_cli, "validate_fixed_c7_report", lambda *_args, **_kwargs: {}
    )
    monkeypatch.setattr(
        stage_c_cli, "build_c8_measured_preflight",
        lambda *_args: pipeline.estimate,
    )
    monkeypatch.setattr(
        stage_c_cli, "capture_git_state", lambda _root: authorization["code"]
    )
    monkeypatch.setattr(
        stage_c_cli, "_validate_c8_shard_checkpoint_locked",
        lambda *_args: (_ for _ in ()).throw(
            stage_c_cli.ArgumentError("C8 shard overlaps units that are not pending")
        ),
    )

    code = stage_c_cli.main([
        "c8-run-shard", "--config", str(config_path),
        "--run-spec", str(spec_path), "--output-root", str(root),
        "--shard-index", "0", "--shard-count", "2",
    ], backend_factory=lambda **_kwargs: pytest.fail("backend constructed"))

    assert code == stage_c_cli.EXIT_INPUT_ERROR
    captured = capsys.readouterr()
    assert "not pending" in json.loads(captured.err)["error"]["message"]


def test_c8_finalize_and_select_do_not_construct_backend(
    monkeypatch, capsys, tmp_path
):
    config_path, spec_path = _write_explicit_inputs(tmp_path)
    pipeline = _install_c8_pipeline(monkeypatch)
    root = tmp_path / "c8"
    git_state = {"commit": "c" * 40, "dirty": False}
    monkeypatch.setattr(stage_c_cli, "capture_git_state", lambda _root: git_state)
    monkeypatch.setattr(
        stage_c_cli, "finalize_c8_output_root",
        lambda *_args, **_kwargs: {
            "status": "PASS", "orchestration_complete": True,
            "successful": True, "errors": [], "rows": [],
        },
    )
    assert stage_c_cli.main([
        "c8-finalize", "--config", str(config_path),
        "--run-spec", str(spec_path), "--output-root", str(root),
    ], backend_factory=lambda **_kwargs: pytest.fail("backend constructed")) == 0
    assert _stdout_json(capsys)["status"] == "PASS"


def test_c8_select_validate_recomputes_source_without_backend(
    monkeypatch, capsys, tmp_path
):
    config_path, spec_path = _write_explicit_inputs(tmp_path)
    pipeline = _install_c8_pipeline(monkeypatch)
    root = tmp_path / "c8"
    git_state = {"commit": "c" * 40, "dirty": False}
    authorization = {
        "code": git_state, "c7_report_path": "/c7.json",
        "c7_report_file_sha256": "1" * 64,
        "c7_report_canonical_digest": "2" * 64,
        "c7_code_commit": "3" * 40, "c7_archive_ref": "archive",
        "c7_archive_tree": "4" * 40, "c7_slowest_unit_seconds": 1.0,
        "c7_peak_reserved_bytes": 1, "c7_projected_c8_h20_gpu_hours": 1.0,
        "c7_projected_c8_artifact_bytes": 1,
        "c7_projected_c8_l2_bytes_per_audit": 1,
    }
    evidence = {"calibration_artifact_manifest_sha256": "5" * 64}
    events = []
    monkeypatch.setattr(stage_c_cli, "capture_git_state", lambda _root: git_state)
    monkeypatch.setattr(
        stage_c_cli, "_load_c8_initialized_estimate",
        lambda *_args: events.append("controls") or (pipeline.estimate, authorization),
    )
    monkeypatch.setattr(
        stage_c_cli, "validate_fixed_c7_report",
        lambda path, **kwargs: events.append(("report", path, kwargs)) or {
            "fresh": True,
            "bindings": {"recomputed_estimate": {"storage": {"x": 1}}},
            "measurement_boundary": {"source_batch_control_bytes": 17},
        },
    )
    monkeypatch.setattr(
        stage_c_cli, "build_c8_measured_preflight",
        lambda plan, config, report: (
            events.append(("preflight", plan, config, report))
            or pipeline.estimate
        ),
    )
    monkeypatch.setattr(
        stage_c_cli, "build_c8_calibration_evidence",
        lambda *_args, **_kwargs: events.append(("evidence", _args, _kwargs)) or evidence,
    )
    monkeypatch.setattr(
        stage_c_cli, "_c8_input_hashes", lambda **_kwargs: {"x": "y"}
    )
    monkeypatch.setattr(
        stage_c_cli, "validate_stage_c_selection_from_source",
        lambda *args, **kwargs: events.append((args, kwargs)) or {
            "status": "PASS", "holdout_accessed": False,
        },
    )
    report_path = tmp_path / "report.json"
    selection_path = tmp_path / "selection.json"
    code = stage_c_cli.main([
        "c8-select", "--mode", "validate",
        "--config", str(config_path), "--run-spec", str(spec_path),
        "--output-root", str(root), "--report-output", str(report_path),
        "--selection-output", str(selection_path),
    ], backend_factory=lambda **_kwargs: pytest.fail("backend constructed"))

    assert code == stage_c_cli.EXIT_SUCCESS
    assert events[0] == "controls"
    assert events[1] == (
        "report", "/c7.json", {"repo_root": stage_c_cli.REPO_ROOT}
    )
    assert events[2][0] == "preflight"
    assert events[2][3]["fresh"] is True
    assert events[3][0] == "evidence"
    assert events[3][1][4]["recomputed_estimate"] == {
        "storage": {"x": 1}
    }
    assert events[3][1][4]["source_batch_control_bytes"] == 17
    assert events[4][0][:2] == (selection_path, report_path)
    assert events[4][1]["c7_report_binding"] == events[3][1][4]
    assert _stdout_json(capsys)["holdout_accessed"] is False


def test_c8_select_build_publishes_without_backend(
    monkeypatch, capsys, tmp_path
):
    config_path, spec_path = _write_explicit_inputs(tmp_path)
    pipeline = _install_c8_pipeline(monkeypatch)
    root = tmp_path / "c8"
    git_state = {"commit": "c" * 40, "dirty": False}
    events = []
    monkeypatch.setattr(
        stage_c_cli, "capture_git_state",
        lambda _root: events.append("git") or git_state,
    )

    authorization = {
        "code": git_state, "c7_report_path": "/c7.json",
        "c7_report_file_sha256": "1" * 64,
        "c7_report_canonical_digest": "2" * 64,
        "c7_code_commit": "3" * 40, "c7_archive_ref": "archive",
        "c7_archive_tree": "4" * 40, "c7_slowest_unit_seconds": 1.0,
        "c7_peak_reserved_bytes": 1, "c7_projected_c8_h20_gpu_hours": 1.0,
        "c7_projected_c8_artifact_bytes": 1,
        "c7_projected_c8_l2_bytes_per_audit": 1,
    }
    evidence = {"calibration_artifact_manifest_sha256": "5" * 64}
    monkeypatch.setattr(
        stage_c_cli, "_load_c8_initialized_estimate",
        lambda *_args: events.append("controls") or (pipeline.estimate, authorization),
    )
    monkeypatch.setattr(stage_c_cli, "validate_fixed_c7_report", lambda *_args, **_kwargs: (
        events.append("report") or {
            "bindings": {"recomputed_estimate": {"storage": {"x": 1}}},
            "measurement_boundary": {"source_batch_control_bytes": 17},
        }
    ))
    monkeypatch.setattr(
        stage_c_cli, "build_c8_measured_preflight",
        lambda *_args: events.append("preflight") or pipeline.estimate,
    )
    monkeypatch.setattr(
        stage_c_cli, "build_c8_calibration_evidence",
        lambda *_args, **_kwargs: events.append("evidence") or evidence,
    )
    monkeypatch.setattr(
        stage_c_cli, "_c8_input_hashes", lambda **_kwargs: {"x": "y"}
    )
    def build_selection(*_args, generator_git_state):
        assert generator_git_state == git_state
        events.append("selection")
        return {"status": "PASS"}

    monkeypatch.setattr(
        stage_c_cli, "build_stage_c_selection", build_selection,
    )
    monkeypatch.setattr(
        stage_c_cli, "write_stage_c_selection_outputs",
        lambda *_args, **_kwargs: events.append("write") or {
            "status": "PASS", "report_sha256": "6" * 64,
            "selection_sha256": "7" * 64,
        },
    )
    assert stage_c_cli.main([
        "c8-select", "--config", str(config_path),
        "--run-spec", str(spec_path), "--output-root", str(root),
        "--report-output", str(tmp_path / "report.json"),
        "--selection-output", str(tmp_path / "selection.json"),
    ], backend_factory=lambda **_kwargs: pytest.fail("backend constructed")) == 0
    assert events == [
        "controls", "report", "preflight", "git",
        "evidence", "selection", "write",
    ]
    assert _stdout_json(capsys)["status"] == "PASS"


@pytest.mark.parametrize("mode", ["build", "validate"])
def test_c8_select_rejects_fresh_preflight_drift_before_evidence_or_backend(
    monkeypatch, capsys, tmp_path, mode
):
    config_path, spec_path = _write_explicit_inputs(tmp_path)
    pipeline = _install_c8_pipeline(monkeypatch)
    authorization = {
        "code": {"commit": "c" * 40, "dirty": False},
        "c7_report_path": "/c7.json",
    }
    fresh = SimpleNamespace(to_dict=lambda: {"canonical_digest": "f" * 64})
    monkeypatch.setattr(
        stage_c_cli, "_load_c8_initialized_estimate",
        lambda *_args: (pipeline.estimate, authorization),
    )
    monkeypatch.setattr(
        stage_c_cli, "validate_fixed_c7_report", lambda *_args, **_kwargs: {}
    )
    monkeypatch.setattr(
        stage_c_cli, "build_c8_measured_preflight", lambda *_args: fresh
    )
    monkeypatch.setattr(
        stage_c_cli, "build_c8_calibration_evidence",
        lambda *_args, **_kwargs: pytest.fail("evidence read before preflight gate"),
    )
    argv = [
        "c8-select", "--mode", mode, "--config", str(config_path),
        "--run-spec", str(spec_path), "--output-root", str(tmp_path / "c8"),
        "--report-output", str(tmp_path / "report.json"),
    ]
    if mode == "build":
        argv.extend(("--selection-output", str(tmp_path / "selection.json")))

    code = stage_c_cli.main(
        argv, backend_factory=lambda **_kwargs: pytest.fail("backend constructed")
    )

    assert code == stage_c_cli.EXIT_INTEGRITY_FAILURE
    captured = capsys.readouterr()
    assert "freshly measured" in json.loads(captured.err)["error"]["message"]


def test_c8_select_accepts_clean_generator_commit_after_source_execution(
    monkeypatch, capsys, tmp_path
):
    config_path, spec_path = _write_explicit_inputs(tmp_path)
    pipeline = _install_c8_pipeline(monkeypatch)
    source_git = {"commit": "c" * 40, "dirty": False}
    generator_git = {"commit": "d" * 40, "dirty": False}
    authorization = {
        "code": source_git, "c7_report_path": "/c7.json",
        "c7_report_file_sha256": "1" * 64,
        "c7_report_canonical_digest": "2" * 64,
        "c7_code_commit": "3" * 40, "c7_archive_ref": "archive",
        "c7_archive_tree": "4" * 40, "c7_slowest_unit_seconds": 1.0,
        "c7_peak_reserved_bytes": 1,
        "c7_projected_c8_h20_gpu_hours": 1.0,
        "c7_projected_c8_artifact_bytes": 1,
        "c7_projected_c8_l2_bytes_per_audit": 1,
    }
    monkeypatch.setattr(
        stage_c_cli, "_load_c8_initialized_estimate",
        lambda *_args: (pipeline.estimate, authorization),
    )
    monkeypatch.setattr(
        stage_c_cli, "validate_fixed_c7_report", lambda *_args, **_kwargs: {}
    )
    monkeypatch.setattr(
        stage_c_cli, "build_c8_measured_preflight",
        lambda *_args: pipeline.estimate,
    )
    monkeypatch.setattr(
        stage_c_cli, "_c8_report_binding",
        lambda *_args, **_kwargs: {"fresh_c7": True},
    )
    monkeypatch.setattr(
        stage_c_cli, "capture_git_state",
        lambda _root: generator_git,
    )
    observed = {}

    def build_evidence(_root, _plan, _config, git_state, _c7_binding):
        observed["source"] = git_state
        return {"calibration_artifact_manifest_sha256": "5" * 64}

    monkeypatch.setattr(
        stage_c_cli, "build_c8_calibration_evidence",
        build_evidence,
    )
    monkeypatch.setattr(
        stage_c_cli, "_c8_input_hashes", lambda **_kwargs: {"x": "y"},
    )

    def validate_source(*_args, **kwargs):
        observed["validate_source"] = kwargs["observed_git_state"]
        observed["generator"] = kwargs["generator_git_state"]
        return {"status": "PASS", "holdout_accessed": False}

    monkeypatch.setattr(
        stage_c_cli, "validate_stage_c_selection_from_source", validate_source,
    )

    code = stage_c_cli.main([
        "c8-select", "--mode", "validate",
        "--config", str(config_path), "--run-spec", str(spec_path),
        "--output-root", str(tmp_path / "c8"),
        "--report-output", str(tmp_path / "report.json"),
    ], backend_factory=lambda **_kwargs: pytest.fail("backend constructed"))

    assert code == stage_c_cli.EXIT_SUCCESS
    assert observed == {
        "source": source_git,
        "validate_source": source_git,
        "generator": generator_git,
    }
    assert _stdout_json(capsys)["status"] == "PASS"


def _install_c9_pipeline(monkeypatch, *, unit_count: int = 11):
    pipeline = _install_plan_pipeline(monkeypatch)
    pipeline.plan.units = tuple(
        SimpleNamespace(
            key=SimpleNamespace(canonical_digest=f"{index:064x}"),
            sample=SimpleNamespace(sample_id=f"sample-{index}"),
        )
        for index in range(unit_count)
    )
    pipeline.plan.canonical_digest = "a" * 64
    pipeline.estimate.canonical_digest = "b" * 64
    monkeypatch.setattr(
        stage_c_cli, "validate_c9_execution_plan",
        lambda plan, config: {"valid": True, "violations": []},
    )
    return pipeline


def _c9_argv(
    command: str, config_path: Path, spec_path: Path, root: Path,
) -> list[str]:
    return [
        command, "--config", str(config_path), "--run-spec",
        str(spec_path), "--output-root", str(root),
    ]


def test_c9_pipeline_requires_tracked_selection_and_never_verifies_live_assets(
    monkeypatch, capsys, tmp_path
):
    config_path, spec_path = _write_explicit_inputs(tmp_path)
    pipeline = _install_c9_pipeline(monkeypatch)
    git_state = {"commit": "c" * 40, "dirty": False}
    monkeypatch.setattr(stage_c_cli, "capture_git_state", lambda _root: git_state)
    monkeypatch.setattr(
        stage_c_cli, "validate_c9_output_root",
        lambda *_args, **_kwargs: {
            "status": "PASS", "integrity_valid": True,
            "orchestration_complete": True, "successful": True,
        },
    )

    code = stage_c_cli.main(
        _c9_argv(
            "c9-validate", config_path, spec_path, tmp_path / "c9"
        ),
        backend_factory=lambda **_kwargs: pytest.fail("backend constructed"),
    )

    assert code == stage_c_cli.EXIT_SUCCESS
    assert pipeline.registry_kwargs == {"verify_live_assets": False}
    assert pipeline.selection_kwargs == {
        "selection_path": stage_c_cli._TRACKED_STAGE_C_SELECTION,
        "required": True, "config_path": config_path,
    }
    assert _stdout_json(capsys)["status"] == "PASS"


def test_c9_missing_required_selection_fails_before_plan_root_and_backend(
    monkeypatch, capsys, tmp_path
):
    config_path, spec_path = _write_explicit_inputs(tmp_path)
    output_root = tmp_path / "must-not-exist"
    _install_plan_pipeline(monkeypatch)
    monkeypatch.setattr(
        stage_c_cli, "load_stage_c_selection",
        lambda _config, **kwargs: (
            pytest.fail("C9 selection was not required")
            if kwargs.get("required") is not True
            else (_ for _ in ()).throw(ValueError("tracked selection missing"))
        ),
    )
    monkeypatch.setattr(
        stage_c_cli, "build_execution_plan",
        lambda *_args, **_kwargs: pytest.fail("plan built without selection"),
    )
    argv = [
        *_c9_argv("c9-init", config_path, spec_path, output_root),
        "--c8-root", str(tmp_path / "c8"),
        "--c8-report", str(tmp_path / "report.json"),
        "--c8-recovery-selection", str(tmp_path / "recovery.json"),
    ]

    code = stage_c_cli.main(
        argv, backend_factory=lambda **_kwargs: pytest.fail("backend constructed")
    )

    assert code == stage_c_cli.EXIT_INPUT_ERROR
    assert not output_root.exists()
    assert "tracked selection missing" in json.loads(
        capsys.readouterr().err
    )["error"]["message"]


def test_c9_nonexact_plan_fails_before_git_root_and_backend(
    monkeypatch, capsys, tmp_path
):
    config_path, spec_path = _write_explicit_inputs(tmp_path)
    output_root = tmp_path / "must-not-exist"
    _install_c9_pipeline(monkeypatch)
    monkeypatch.setattr(
        stage_c_cli, "validate_c9_execution_plan",
        lambda *_args: {"valid": False, "violations": ["scope drifted"]},
    )
    monkeypatch.setattr(
        stage_c_cli, "capture_git_state",
        lambda _root: pytest.fail("git captured after invalid plan"),
    )
    argv = [
        *_c9_argv("c9-init", config_path, spec_path, output_root),
        "--c8-root", str(tmp_path / "c8"),
        "--c8-report", str(tmp_path / "report.json"),
        "--c8-recovery-selection", str(tmp_path / "recovery.json"),
    ]

    code = stage_c_cli.main(
        argv, backend_factory=lambda **_kwargs: pytest.fail("backend constructed")
    )

    assert code == stage_c_cli.EXIT_INPUT_ERROR
    assert not output_root.exists()
    assert "scope drifted" in json.loads(
        capsys.readouterr().err
    )["error"]["message"]


@pytest.mark.parametrize(
    "escape",
    [
        ("--selection", "alternate.json"),
        ("--registry", "alternate.csv"),
        ("--data-root", "alternate-images"),
        ("--state-dict", "alternate.pth"),
        ("--no-resume",),
        ("--unit-digest", "1" * 64),
    ],
)
def test_c9_shard_rejects_caller_selection_resume_and_unit_filters(
    capsys, tmp_path, escape
):
    config_path, spec_path = _write_explicit_inputs(tmp_path)
    argv = _c9_argv(
        "c9-run-shard", config_path, spec_path, tmp_path / "c9"
    )
    argv.extend(("--shard-index", "0", "--shard-count", "1", *escape))

    assert stage_c_cli.main(argv) == stage_c_cli.EXIT_INPUT_ERROR
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "unrecognized arguments" in json.loads(captured.err)["error"]["message"]


def test_c9_init_completes_all_gates_before_creating_root_or_backend(
    monkeypatch, capsys, tmp_path
):
    config_path, spec_path = _write_explicit_inputs(tmp_path)
    pipeline = _install_c9_pipeline(monkeypatch)
    root = tmp_path / "c9"
    c8_root = tmp_path / "c8"
    c8_report = tmp_path / "c8-report.json"
    recovery = tmp_path / "recovery.json"
    git_state = {"commit": "c" * 40, "dirty": False}
    authorization = {"canonical_digest": "d" * 64}
    measured = SimpleNamespace(canonical_digest="e" * 64)
    events = []
    monkeypatch.setattr(
        stage_c_cli, "capture_git_state",
        lambda _root: events.append("git") or git_state,
    )

    def authorize(plan, config, **kwargs):
        events.append("handoff+preflight")
        assert (plan, config) == (pipeline.plan, pipeline.config)
        assert kwargs == {
            "c8_root": c8_root, "c8_report_path": c8_report,
            "recovery_selection_path": recovery, "output_root": root,
            "observed_git_state": git_state,
            "repo_root": stage_c_cli.REPO_ROOT,
        }
        assert not root.exists()
        return authorization, measured

    def initialize(output_root, plan, estimate, auth, **kwargs):
        events.append("initialize")
        assert (output_root, plan, estimate, auth) == (
            root, pipeline.plan, measured, authorization
        )
        assert kwargs == {
            "config": pipeline.config, "observed_git_state": git_state,
            "repo_root": stage_c_cli.REPO_ROOT,
        }
        return Path(output_root).absolute()

    monkeypatch.setattr(stage_c_cli, "build_c9_launch_authorization", authorize)
    monkeypatch.setattr(stage_c_cli, "initialize_c9_output_root", initialize)
    argv = [
        *_c9_argv("c9-init", config_path, spec_path, root),
        "--c8-root", str(c8_root), "--c8-report", str(c8_report),
        "--c8-recovery-selection", str(recovery),
    ]

    code = stage_c_cli.main(
        argv, backend_factory=lambda **_kwargs: pytest.fail("backend constructed")
    )

    assert code == stage_c_cli.EXIT_SUCCESS
    assert events == ["git", "handoff+preflight", "initialize"]
    payload = _stdout_json(capsys)
    assert payload["expected_unit_count"] == 11
    assert payload["holdout_accessed"] is False
    assert payload["selection_performed"] is False
    assert payload["decision_performed"] is False


def test_c9_init_handoff_failure_precedes_root_and_backend(
    monkeypatch, capsys, tmp_path
):
    config_path, spec_path = _write_explicit_inputs(tmp_path)
    _install_c9_pipeline(monkeypatch)
    root = tmp_path / "must-not-exist"
    git_state = {"commit": "c" * 40, "dirty": False}
    monkeypatch.setattr(stage_c_cli, "capture_git_state", lambda _root: git_state)
    monkeypatch.setattr(
        stage_c_cli, "build_c9_launch_authorization",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            stage_c_cli.StageCBenchmarkIntegrityError("C8 handoff drifted")
        ),
    )
    monkeypatch.setattr(
        stage_c_cli, "initialize_c9_output_root",
        lambda *_args, **_kwargs: pytest.fail("root initialized"),
    )
    argv = [
        *_c9_argv("c9-init", config_path, spec_path, root),
        "--c8-root", str(tmp_path / "c8"),
        "--c8-report", str(tmp_path / "report.json"),
        "--c8-recovery-selection", str(tmp_path / "recovery.json"),
    ]

    code = stage_c_cli.main(
        argv, backend_factory=lambda **_kwargs: pytest.fail("backend constructed")
    )

    assert code == stage_c_cli.EXIT_INTEGRITY_FAILURE
    assert not root.exists()
    assert "handoff drifted" in json.loads(capsys.readouterr().err)["error"]["message"]


def test_c9_shard_validates_root_before_determinism_backend_and_core_partition(
    monkeypatch, capsys, tmp_path
):
    config_path, spec_path = _write_explicit_inputs(tmp_path)
    pipeline = _install_c9_pipeline(monkeypatch, unit_count=11)
    root = tmp_path / "c9"
    git_state = {"commit": "c" * 40, "dirty": False}
    events = []
    monkeypatch.setattr(
        stage_c_cli, "capture_git_state",
        lambda _root: events.append("git") or git_state,
    )
    monkeypatch.setattr(
        stage_c_cli, "load_c9_initialized_state",
        lambda *args, **kwargs: (
            events.append(("controls", args, kwargs))
            or (pipeline.estimate, {"code": git_state})
        ),
    )
    monkeypatch.setattr(
        stage_c_cli, "_configure_c9_execution",
        lambda device: events.append(("configure", device)),
    )
    observed = {}

    def backend_factory(**kwargs):
        events.append("backend")
        observed["backend"] = kwargs
        return object()

    def run_shard(plan, **kwargs):
        events.append("run")
        observed["plan"] = plan
        observed["run"] = kwargs
        return ({"state": "succeeded"},)

    monkeypatch.setattr(stage_c_cli, "run_c9_shard", run_shard)
    argv = [
        *_c9_argv("c9-run-shard", config_path, spec_path, root),
        "--shard-index", "1", "--shard-count", "3",
    ]

    code = stage_c_cli.main(argv, backend_factory=backend_factory)

    assert code == stage_c_cli.EXIT_SUCCESS
    assert [item[0] if isinstance(item, tuple) else item for item in events] == [
        "git", "controls", "configure", "backend", "run",
    ]
    controls = events[1]
    assert controls[1] == (root, pipeline.plan)
    assert controls[2] == {
        "observed_git_state": git_state, "repo_root": stage_c_cli.REPO_ROOT,
        "require_checkpoint": True,
    }
    assert observed["plan"] is pipeline.plan
    assert len(observed["plan"].units) == 11
    assert observed["run"]["shard_index"] == 1
    assert observed["run"]["shard_count"] == 3
    assert "unit_digests" not in observed["run"]
    assert "resume" not in observed["run"]
    assert _stdout_json(capsys)["unit_count"] == 1


@pytest.mark.parametrize(
    ("index", "count"), [(-1, 2), (2, 2), (0, 0), (0, 12)],
)
def test_c9_shard_rejects_invalid_coordinates_before_root_or_backend(
    monkeypatch, capsys, tmp_path, index, count
):
    config_path, spec_path = _write_explicit_inputs(tmp_path)
    _install_c9_pipeline(monkeypatch, unit_count=11)
    monkeypatch.setattr(
        stage_c_cli, "load_c9_initialized_state",
        lambda *_args, **_kwargs: pytest.fail("root controls read"),
    )
    argv = [
        *_c9_argv("c9-run-shard", config_path, spec_path, tmp_path / "c9"),
        "--shard-index", str(index), "--shard-count", str(count),
    ]

    code = stage_c_cli.main(
        argv, backend_factory=lambda **_kwargs: pytest.fail("backend constructed")
    )

    assert code == stage_c_cli.EXIT_INPUT_ERROR
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err)["error"]["type"] == "ArgumentError"


def test_c9_root_gate_failure_precedes_determinism_and_backend(
    monkeypatch, capsys, tmp_path
):
    config_path, spec_path = _write_explicit_inputs(tmp_path)
    _install_c9_pipeline(monkeypatch)
    git_state = {"commit": "c" * 40, "dirty": False}
    monkeypatch.setattr(stage_c_cli, "capture_git_state", lambda _root: git_state)
    monkeypatch.setattr(
        stage_c_cli, "load_c9_initialized_state",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            stage_c_cli.StageCBenchmarkIntegrityError("authorization drifted")
        ),
    )
    monkeypatch.setattr(
        stage_c_cli, "_configure_c9_execution",
        lambda _device: pytest.fail("deterministic/GPU configuration attempted"),
    )
    argv = [
        *_c9_argv(
            "c9-run-shard", config_path, spec_path, tmp_path / "c9"
        ),
        "--shard-index", "0", "--shard-count", "1",
    ]

    code = stage_c_cli.main(
        argv, backend_factory=lambda **_kwargs: pytest.fail("backend constructed")
    )

    assert code == stage_c_cli.EXIT_INTEGRITY_FAILURE
    assert "authorization drifted" in json.loads(
        capsys.readouterr().err
    )["error"]["message"]


def test_c9_configuration_sets_frozen_determinism(monkeypatch):
    seeded = []
    deterministic = []
    monkeypatch.setattr(stage_c_cli, "seed_everything", seeded.append)
    monkeypatch.setattr(
        torch, "use_deterministic_algorithms", deterministic.append
    )
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", True)

    stage_c_cli._configure_c9_execution("cpu")

    assert seeded == [stage_c_cli.STAGE_C_SEED]
    assert deterministic == [True]
    assert torch.backends.cudnn.benchmark is False


def test_c9_cuda_configuration_rejects_missing_workspace_before_seeding(
    monkeypatch,
):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    seeded = []
    monkeypatch.setattr(stage_c_cli, "seed_everything", seeded.append)

    with pytest.raises(RuntimeError, match="CUBLAS_WORKSPACE_CONFIG"):
        stage_c_cli._configure_c9_execution("cuda:0")

    assert seeded == []


@pytest.mark.parametrize(
    ("command", "function_name"),
    [
        ("c9-finalize", "finalize_c9_output_root"),
        ("c9-validate", "validate_c9_output_root"),
    ],
)
def test_c9_finalize_and_validate_use_dedicated_core_without_backend(
    monkeypatch, capsys, tmp_path, command, function_name
):
    config_path, spec_path = _write_explicit_inputs(tmp_path)
    pipeline = _install_c9_pipeline(monkeypatch)
    root = tmp_path / "c9"
    git_state = {"commit": "c" * 40, "dirty": False}
    calls = []
    monkeypatch.setattr(stage_c_cli, "capture_git_state", lambda _root: git_state)
    monkeypatch.setattr(
        stage_c_cli, "validate_stage_c_batch",
        lambda *_args, **_kwargs: pytest.fail("generic validator called"),
    )
    monkeypatch.setattr(
        stage_c_cli, function_name,
        lambda *args, **kwargs: calls.append((args, kwargs)) or {
            "status": "PASS", "integrity_valid": True,
            "orchestration_complete": True, "successful": True,
        },
    )

    code = stage_c_cli.main(
        _c9_argv(command, config_path, spec_path, root),
        backend_factory=lambda **_kwargs: pytest.fail("backend constructed"),
    )

    assert code == stage_c_cli.EXIT_SUCCESS
    assert calls == [((root, pipeline.plan), {"observed_git_state": git_state})]
    assert _stdout_json(capsys)["status"] == "PASS"


@pytest.mark.parametrize(
    ("report", "expected_code"),
    [
        ({"status": "BLOCKED", "integrity_valid": True,
          "orchestration_complete": True, "successful": False},
         stage_c_cli.EXIT_TERMINAL_FAILURE),
        ({"status": "BLOCKED", "integrity_valid": False,
          "orchestration_complete": True, "successful": False},
         stage_c_cli.EXIT_INTEGRITY_FAILURE),
        ({"status": "BLOCKED", "integrity_valid": True,
          "orchestration_complete": False, "successful": False},
         stage_c_cli.EXIT_INTEGRITY_FAILURE),
    ],
)
def test_c9_evidence_report_controls_exit_code(
    monkeypatch, capsys, tmp_path, report, expected_code
):
    config_path, spec_path = _write_explicit_inputs(tmp_path)
    _install_c9_pipeline(monkeypatch)
    monkeypatch.setattr(
        stage_c_cli, "capture_git_state",
        lambda _root: {"commit": "c" * 40, "dirty": False},
    )
    monkeypatch.setattr(
        stage_c_cli, "validate_c9_output_root",
        lambda *_args, **_kwargs: report,
    )

    code = stage_c_cli.main(
        _c9_argv(
            "c9-validate", config_path, spec_path, tmp_path / "c9"
        )
    )

    assert code == expected_code
    assert _stdout_json(capsys) == report


def test_c9_blocked_error_uses_c9_document_type(
    monkeypatch, capsys, tmp_path
):
    config_path, spec_path = _write_explicit_inputs(tmp_path)
    _install_c9_pipeline(monkeypatch)
    monkeypatch.setattr(
        stage_c_cli, "capture_git_state",
        lambda _root: {"commit": "c" * 40, "dirty": True},
    )
    argv = [
        *_c9_argv("c9-init", config_path, spec_path, tmp_path / "c9"),
        "--c8-root", str(tmp_path / "c8"),
        "--c8-report", str(tmp_path / "report.json"),
        "--c8-recovery-selection", str(tmp_path / "recovery.json"),
    ]

    assert stage_c_cli.main(argv) == stage_c_cli.EXIT_PREFLIGHT_BLOCKED
    payload = _stdout_json(capsys)
    assert payload["document_type"] == "stage_c_c9_launch_authorization"
    assert "clean Git" in payload["blocked_reasons"][0]


@pytest.mark.parametrize(
    "command", ["c9-run-shard", "c9-finalize", "c9-validate"]
)
def test_c9_post_launch_blocked_error_is_integrity_not_authorization(
    monkeypatch, capsys, tmp_path, command
):
    config_path, spec_path = _write_explicit_inputs(tmp_path)
    pipeline = _install_c9_pipeline(monkeypatch)
    git_state = {"commit": "c" * 40, "dirty": False}
    monkeypatch.setattr(
        stage_c_cli, "capture_git_state", lambda _root: git_state
    )
    error = stage_c_cli.StageCBenchmarkBlockedError(
        "formal root is incomplete", reasons=("unit: incomplete",)
    )
    if command == "c9-run-shard":
        monkeypatch.setattr(
            stage_c_cli, "load_c9_initialized_state",
            lambda *_args, **_kwargs: (
                pipeline.estimate, {"code": git_state}
            ),
        )
        monkeypatch.setattr(
            stage_c_cli, "_configure_c9_execution", lambda _device: None
        )
        monkeypatch.setattr(
            stage_c_cli, "run_c9_shard",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(error),
        )
        argv = [
            *_c9_argv(command, config_path, spec_path, tmp_path / "c9"),
            "--shard-index", "0", "--shard-count", "1",
        ]
    else:
        monkeypatch.setattr(
            stage_c_cli,
            ("finalize_c9_output_root" if command == "c9-finalize"
             else "validate_c9_output_root"),
            lambda *_args, **_kwargs: (_ for _ in ()).throw(error),
        )
        argv = _c9_argv(
            command, config_path, spec_path, tmp_path / "c9"
        )

    code = stage_c_cli.main(
        argv, backend_factory=lambda **_kwargs: object()
    )

    assert code == stage_c_cli.EXIT_INTEGRITY_FAILURE
    captured = capsys.readouterr()
    assert captured.out == ""
    payload = json.loads(captured.err)
    assert payload == {
        "error": {
            "message": "formal root is incomplete",
            "type": "StageCBenchmarkBlockedError",
            "reasons": ["unit: incomplete"],
        }
    }
