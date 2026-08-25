"""Focused metadata/fake-runtime tests for the Stage D D4 CLI."""

from __future__ import annotations

import argparse
import ast
import copy
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import vitnns.stage_d_cli as cli


class _Document:
    def __init__(self, payload: dict) -> None:
        self._payload = copy.deepcopy(payload)

    def __getattr__(self, name: str):
        try:
            return self._payload[name]
        except KeyError as error:
            raise AttributeError(name) from error

    def to_dict(self) -> dict:
        return copy.deepcopy(self._payload)


def _pipeline(*, plan: _Document | None = None) -> cli.StageDCLIPipeline:
    config = _Document({
        "canonical_digest": "a" * 64,
        "file_sha256": "b" * 64,
        "schema_file_sha256": "c" * 64,
    })
    run_spec = _Document({
        "canonical_digest": "d" * 64,
        "file_sha256": "e" * 64,
        "schema_file_sha256": "f" * 64,
    })
    contracts = SimpleNamespace(config=config, run_spec=run_spec)
    manifest = SimpleNamespace(
        canonical_digest="1" * 64,
        bindings={
            stage: {"verification_status": "PASS"}
            for stage in ("d0", "d1", "d2", "c10")
        },
        registry_rows=tuple({"selection_order": index} for index in range(500)),
    )
    return cli.StageDCLIPipeline(
        contracts, manifest, plan or _Document({
            "document_type": "stage_d_execution_plan",
            "canonical_digest": "2" * 64,
        }),
    )


def _estimate(status: str = "NOT_EVALUATED") -> _Document:
    return _Document({
        "document_type": "stage_d_preflight_estimate",
        "profile_status": "D7_PROFILE_REQUIRED",
        "budget_status": status,
        "canonical_digest": "3" * 64,
    })


def _preflight(status: str = "PROFILE_REQUIRED") -> _Document:
    return _Document({
        "document_type": "stage_d_metadata_preflight",
        "status": status,
        "execution_authorization": "NOT_ISSUED",
        "analysis_authorization": "NOT_EVALUATED",
        "d7_final": False,
        "canonical_digest": "4" * 64,
    })


def _invoke(
    argv: list[str], **kwargs,
) -> tuple[int, dict | None, dict | None]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    code = cli.main(argv, stdout=stdout, stderr=stderr, **kwargs)
    return (
        code, json.loads(stdout.getvalue()) if stdout.getvalue() else None,
        json.loads(stderr.getvalue()) if stderr.getvalue() else None,
    )


def _runtime_argv(
    command: str, authorization: Path, output_root: Path,
    *, retry_authorizations: tuple[Path, ...] = (),
) -> list[str]:
    argv = [
        command, "--authorization", str(authorization),
        "--output-root", str(output_root),
    ]
    if command == "run-shard":
        argv.extend(["--shard-index", "2", "--device", "fake-device"])
        for retry_authorization in retry_authorizations:
            argv.extend(["--retry-authorization", str(retry_authorization)])
    return argv


def _write_authorization(path: Path, payload: dict | None = None) -> dict:
    document = {
        "document_type": "fake_d7_execution_authorization",
        "canonical_digest": "7" * 64,
    } if payload is None else payload
    path.write_text(json.dumps(document), encoding="utf-8")
    return document


def _runtime_summary(**changes) -> _Document:
    document = {
        "schema_version": 1,
        "document_type": "stage_d_shard_summary",
        "shard_id": 2,
        "shard_count": 4,
        "state": "completed",
        "counts": {
            "expected_groups": 4, "visited_groups": 4,
            "claimed_groups": 3, "terminal_groups": 4,
            "expected_units": 8, "attempted_units": 6,
            "terminal_units": 8, "succeeded_units": 7,
            "scientific_failed_units": 1, "retry_pending_units": 0,
            "blocked_units": 0, "skipped_terminal_units": 2,
        },
        "resources": {
            "elapsed_seconds": 2.5, "forward_calls": 12,
            "backward_calls": 6, "l0_l1_payload_bytes": 1024,
            "l2_payload_bytes": 0,
            "control_and_provenance_bytes": 512,
            "cuda_peak_reserved_bytes": 4096,
        },
        "checkpoint_generation": 9,
        "checkpoint_digest": "a" * 64,
        "execution_context_digest": "b" * 64,
    }
    document.update(changes)
    return _Document(document)


def _install_estimate_pipeline(
    monkeypatch: pytest.MonkeyPatch, *, blocked: bool = False,
    calls: list | None = None,
) -> tuple[_Document, _Document]:
    recorded = [] if calls is None else calls
    estimate = _estimate("BLOCKED_NO_LAUNCH" if blocked else "NOT_EVALUATED")
    preflight = _preflight(
        "BLOCKED_NO_LAUNCH" if blocked else "PROFILE_REQUIRED"
    )

    def make_estimate(config, plan, profiles):
        recorded.append(("estimate", config, plan, profiles))
        return estimate

    def check_estimate(value):
        recorded.append(("validate-estimate", value))
        return value

    def make_preflight(value):
        recorded.append(("preflight", value))
        return preflight

    def check_preflight(value):
        recorded.append(("validate-preflight", value))
        return value

    monkeypatch.setattr(cli, "estimate_stage_d_run", make_estimate)
    monkeypatch.setattr(cli, "validate_stage_d_estimate", check_estimate)
    monkeypatch.setattr(cli, "preflight_stage_d", make_preflight)
    monkeypatch.setattr(cli, "validate_stage_d_preflight", check_preflight)
    return estimate, preflight


def test_parser_exposes_exact_d4_command_surface() -> None:
    parser = cli.build_parser()
    subparsers = [
        action for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    ]
    assert len(subparsers) == 1
    assert set(subparsers[0].choices) == {
        "validate-contracts", "plan", "estimate",
        "prepare-authorization", "init", "run-shard",
    }
    forbidden = {
        "finalize", "aggregate", "export", "synthetic",
    }
    assert forbidden.isdisjoint(subparsers[0].choices)


def test_help_is_success_and_does_not_load_pipeline(capsys) -> None:
    def unexpected(_args):  # pragma: no cover - failure assertion
        raise AssertionError("help must not load Stage D inputs")

    assert cli.main(["--help"], pipeline_provider=unexpected) == cli.EXIT_SUCCESS
    captured = capsys.readouterr()
    assert "validate-contracts" in captured.out
    assert "prepare-authorization" in captured.out
    assert captured.err == ""


def test_argument_errors_are_structured_json_on_stderr() -> None:
    code, output, error = _invoke(["not-a-command"])
    assert code == cli.EXIT_INPUT_ERROR
    assert output is None
    assert error["error"]["code"] == "STAGE_D_INPUT_ERROR"
    assert error["error"]["type"] == "ArgumentError"
    assert "invalid choice" in error["error"]["message"]


def test_validate_contracts_fully_verifies_sources_without_building_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = _pipeline()
    calls: list[tuple] = []

    def load(**kwargs):
        calls.append(("load", kwargs))
        return pipeline.contracts

    def verify(config, **kwargs):
        calls.append(("verify", config, kwargs))
        return pipeline.manifest

    monkeypatch.setattr(cli, "load_stage_d_contracts", load)
    monkeypatch.setattr(cli, "verify_stage_d_manifest", verify)
    monkeypatch.setattr(
        cli, "build_stage_d_plan",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("validate-contracts must not build a plan"),
        ),
    )

    code, output, error = _invoke(["validate-contracts"])
    assert code == cli.EXIT_SUCCESS
    assert error is None
    assert output["document_type"] == "stage_d_contract_validation"
    assert output["status"] == "PASS"
    assert output["registry_row_count"] == 500
    assert output["source_verification"] == {
        "d0": "PASS", "d1": "PASS",
        "d2": "PASS", "c10": "PASS",
    }
    assert [item[0] for item in calls] == ["load", "verify"]
    assert calls[1][2]["verify_c10"] is True


def test_plan_loads_verifies_builds_then_validates_and_emits_full_document(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = _pipeline()
    plan = _Document({
        "document_type": "stage_d_execution_plan",
        "counts": {"image_groups": 500, "logical_work_items": 211010},
        "groups": [{"selection_order": 0}],
        "canonical_digest": "5" * 64,
    })
    calls: list[str] = []
    monkeypatch.setattr(
        cli, "load_stage_d_contracts",
        lambda **_kwargs: calls.append("load") or pipeline.contracts,
    )
    monkeypatch.setattr(
        cli, "verify_stage_d_manifest",
        lambda *_args, **kwargs: (
            calls.append(f"verify:{kwargs['verify_c10']}") or pipeline.manifest
        ),
    )
    monkeypatch.setattr(
        cli, "build_stage_d_plan",
        lambda *_args: calls.append("build-plan") or plan,
    )
    monkeypatch.setattr(
        cli, "validate_stage_d_plan",
        lambda value: calls.append("validate-plan") or value,
    )

    code, output, error = _invoke(["plan"])
    assert code == cli.EXIT_SUCCESS
    assert error is None
    assert output == plan.to_dict()
    assert calls == ["load", "verify:True", "build-plan", "validate-plan"]


def test_estimate_without_profiles_is_a_successful_not_evaluated_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple] = []
    estimate, preflight = _install_estimate_pipeline(monkeypatch, calls=calls)
    pipeline = _pipeline()
    code, output, error = _invoke(
        ["estimate"], pipeline_provider=lambda _args: pipeline,
    )
    assert code == cli.EXIT_SUCCESS
    assert error is None
    assert output == {
        "schema_version": 1,
        "document_type": "stage_d_estimate_report",
        "status": "PASS",
        "estimate": estimate.to_dict(),
        "preflight": preflight.to_dict(),
    }
    assert calls[0][0] == "estimate"
    assert calls[0][3] is None
    assert [item[0] for item in calls] == [
        "estimate", "validate-estimate", "preflight",
        "validate-preflight",
    ]
    assert output["estimate"]["profile_status"] == "D7_PROFILE_REQUIRED"
    assert output["estimate"]["budget_status"] == "NOT_EVALUATED"
    assert output["preflight"]["execution_authorization"] == "NOT_ISSUED"


def test_blocked_estimate_is_json_stdout_with_stable_exit_two(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_estimate_pipeline(monkeypatch, blocked=True)
    code, output, error = _invoke(
        ["estimate"], pipeline_provider=lambda _args: _pipeline(),
    )
    assert code == cli.EXIT_PREFLIGHT_BLOCKED
    assert error is None
    assert output["status"] == "BLOCKED"
    assert output["preflight"]["status"] == "BLOCKED_NO_LAUNCH"


def test_profile_file_uses_strict_json_and_reports_duplicate_keys_as_input_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    profile_path = tmp_path / "profiles.json"
    profile_path.write_text('{"profile":1,"profile":2}', encoding="utf-8")
    monkeypatch.setattr(
        cli, "estimate_stage_d_run",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("malformed profiles must fail before estimation"),
        ),
    )
    code, output, error = _invoke(
        ["estimate", "--profiles", str(profile_path)],
        pipeline_provider=lambda _args: _pipeline(),
    )
    assert code == cli.EXIT_INPUT_ERROR
    assert output is None
    assert error["error"]["code"] == "STAGE_D_INPUT_ERROR"
    assert "duplicate JSON key" in error["error"]["message"]


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
def test_profile_file_rejects_nonfinite_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, token: str,
) -> None:
    profile_path = tmp_path / "profiles.json"
    profile_path.write_text(f"[{{\"seconds\":{token}}}]", encoding="utf-8")
    code, output, error = _invoke(
        ["estimate", "--profiles", str(profile_path)],
        pipeline_provider=lambda _args: _pipeline(),
    )
    assert code == cli.EXIT_INPUT_ERROR
    assert output is None
    assert "non-finite" in error["error"]["message"]


def test_valid_profile_json_is_passed_without_normalization(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    profile_document = {"method": {"unit": {"seconds": 1}}}
    profile_path = tmp_path / "profiles.json"
    profile_path.write_text(json.dumps(profile_document), encoding="utf-8")
    calls: list[tuple] = []
    _install_estimate_pipeline(monkeypatch, calls=calls)
    code, _output, error = _invoke(
        ["estimate", "--profiles", str(profile_path)],
        pipeline_provider=lambda _args: _pipeline(),
    )
    assert code == cli.EXIT_SUCCESS
    assert error is None
    assert calls[0][3] == profile_document


def test_semantically_invalid_profiles_are_structured_input_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    profile_path = tmp_path / "profiles.json"
    profile_path.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(
        cli, "estimate_stage_d_run",
        lambda *_args: (_ for _ in ()).throw(
            cli.StageDPlanError("profiles must provide all four combinations"),
        ),
    )
    code, output, error = _invoke(
        ["estimate", "--profiles", str(profile_path)],
        pipeline_provider=lambda _args: _pipeline(),
    )
    assert code == cli.EXIT_INPUT_ERROR
    assert output is None
    assert error["error"]["code"] == "STAGE_D_INPUT_ERROR"
    assert "all four combinations" in error["error"]["message"]


def test_prepare_authorization_observes_stable_clean_git_and_revalidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list = []
    estimate, preflight = _install_estimate_pipeline(monkeypatch, calls=calls)
    pipeline = _pipeline()
    commit = "a" * 40
    preparation_inputs = SimpleNamespace(
        code={"commit": commit, "dirty": False}
    )
    authorization = _Document({
        "document_type": "stage_d_preparation_authorization",
        "status": "PASS",
        "preparation_authorization": "APPROVED_TO_PREPARE_STAGE_D",
        "execution_authorization": "NOT_ISSUED",
        "analysis_authorization": "NOT_EVALUATED",
        "stage_d_execution_started": False,
        "scientific_result": "NOT_EVALUATED",
        "canonical_digest": "6" * 64,
    })

    def input_provider(*, repo_root):
        calls.append(("preparation-inputs", repo_root))
        return preparation_inputs

    def build(**kwargs):
        calls.append(("build-authorization", kwargs))
        return authorization

    def validate(value, **kwargs):
        calls.append(("validate-authorization", value, kwargs))
        return value

    monkeypatch.setattr(cli, "build_stage_d_preparation_authorization", build)
    monkeypatch.setattr(
        cli, "validate_stage_d_preparation_authorization", validate,
    )
    code, output, error = _invoke(
        ["prepare-authorization"],
        pipeline_provider=lambda _args: pipeline,
        preparation_input_provider=input_provider,
    )
    assert code == cli.EXIT_SUCCESS
    assert error is None
    assert output == authorization.to_dict()
    assert output["execution_authorization"] == "NOT_ISSUED"
    assert "AUTHORIZED_STAGE_D_FIDELITY_SCOPE" not in json.dumps(output)
    names = [item[0] for item in calls]
    assert names == [
        "estimate", "validate-estimate", "preflight",
        "validate-preflight", "preparation-inputs",
        "build-authorization", "validate-authorization",
    ]
    build_kwargs = calls[-2][1]
    assert build_kwargs == {
        "contracts": pipeline.contracts, "manifest": pipeline.manifest,
        "plan": pipeline.plan, "budget": estimate,
        "preflight": preflight, "preparation_inputs": preparation_inputs,
    }
    assert calls[-1][2] == build_kwargs
    assert not ({"code", "dependencies", "model"} & set(build_kwargs))


def test_prepare_authorization_propagates_dirty_git_refusal_before_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_estimate_pipeline(monkeypatch)
    monkeypatch.setattr(
        cli, "build_stage_d_preparation_authorization",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("dirty Git must not reach authorization builder"),
        ),
    )

    def refuse_dirty(**_kwargs):
        raise cli.StageDAuthorizationError(
            "Stage D preparation refuses a dirty producer code state"
        )

    code, output, error = _invoke(
        ["prepare-authorization"],
        pipeline_provider=lambda _args: _pipeline(),
        preparation_input_provider=refuse_dirty,
    )
    assert code == cli.EXIT_INTEGRITY_FAILURE
    assert output is None
    assert error["error"]["code"] == "STAGE_D_INTEGRITY_ERROR"
    assert "dirty producer" in error["error"]["message"]


def test_prepare_authorization_propagates_git_change_across_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_estimate_pipeline(monkeypatch)
    monkeypatch.setattr(
        cli, "build_stage_d_preparation_authorization",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("drifted Git must not reach authorization builder"),
        ),
    )

    def refuse_drift(**_kwargs):
        raise cli.StageDAuthorizationError(
            "Git state changed during preparation-input capture"
        )

    code, output, error = _invoke(
        ["prepare-authorization"],
        pipeline_provider=lambda _args: _pipeline(),
        preparation_input_provider=refuse_drift,
    )
    assert code == cli.EXIT_INTEGRITY_FAILURE
    assert output is None
    assert "changed during" in error["error"]["message"]


def test_authorization_validation_error_is_structured_integrity_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_estimate_pipeline(monkeypatch)
    preparation_inputs = SimpleNamespace(
        code={"commit": "a" * 40, "dirty": False}
    )
    monkeypatch.setattr(
        cli, "build_stage_d_preparation_authorization",
        lambda **_kwargs: (_ for _ in ()).throw(
            cli.StageDAuthorizationError("source binding drifted"),
        ),
    )
    code, output, error = _invoke(
        ["prepare-authorization"],
        pipeline_provider=lambda _args: _pipeline(),
        preparation_input_provider=lambda **_kwargs: preparation_inputs,
    )
    assert code == cli.EXIT_INTEGRITY_FAILURE
    assert output is None
    assert error["error"] == {
        "code": "STAGE_D_INTEGRITY_ERROR",
        "message": "source binding drifted",
        "type": "StageDAuthorizationError",
    }


def test_prepare_authorization_capture_boundary_has_no_git_provider_parameter(
) -> None:
    parameters = cli.main.__signature__.parameters if hasattr(
        cli.main, "__signature__"
    ) else None
    if parameters is None:
        import inspect
        parameters = inspect.signature(cli.main).parameters
    assert "git_state_provider" not in parameters
    assert "git_command_runner" not in parameters


def test_cli_source_has_no_model_or_runtime_imports() -> None:
    source_path = Path(cli.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    forbidden = {
        "PIL", "cv2", "timm", "torch", "torchvision",
        "attribution", "dataset", "extractor", "modeling", "runner",
        "stage_c_cli", "stage_c_provenance", "stage_c_runner",
        "stage_d_runner",
    }
    assert not any(
        name == blocked or name.startswith(blocked + ".")
        for name in imported for blocked in forbidden
    )


def test_runtime_parsers_require_formal_inputs_and_expose_no_bypass() -> None:
    parser = cli.build_parser()
    subparsers = next(
        action for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    for command in ("init", "run-shard"):
        command_parser = subparsers.choices[command]
        options = {
            option
            for action in command_parser._actions
            for option in action.option_strings
        }
        assert {"--output-root", "--authorization"} <= options
        assert {
            "--synthetic", "--skip-authorization", "--shard-count",
            "--finalize", "--aggregate", "--result",
        }.isdisjoint(options)
    run_options = {
        option
        for action in subparsers.choices["run-shard"]._actions
        for option in action.option_strings
    }
    assert {
        "--shard-index", "--device", "--retry-authorization",
    } <= run_options
    init_options = {
        option
        for action in subparsers.choices["init"]._actions
        for option in action.option_strings
    }
    assert "--retry-authorization" not in init_options


def test_init_uses_strict_authorization_and_injected_runtime_gate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    estimate, preflight = _install_estimate_pipeline(monkeypatch)
    authorization_path = tmp_path / "authorization.json"
    authorization = _write_authorization(authorization_path)
    output_root = tmp_path / "formal-root"
    pipeline = _pipeline()
    events: list[str] = []

    class Gate:
        def validate_execution_context(self, context, **kwargs):
            events.append("gate")
            assert context == authorization
            assert kwargs == {
                "operation": "init", "output_root": output_root,
                "plan": pipeline.plan, "estimate": estimate,
                "preflight": preflight,
            }
            return SimpleNamespace(canonical_digest="8" * 64)

    def gate_factory():
        events.append("gate-factory")
        return Gate()

    def initializer(
        root, plan, observed_estimate, observed_preflight, *,
        execution_context, execution_gate,
    ):
        events.append("initializer")
        assert (root, plan, observed_estimate, observed_preflight) == (
            output_root, pipeline.plan, estimate, preflight,
        )
        execution_gate.validate_execution_context(
            execution_context, operation="init", output_root=root, plan=plan,
            estimate=observed_estimate, preflight=observed_preflight,
        )
        events.append("runtime-metadata")
        return root

    code, output, error = _invoke(
        _runtime_argv("init", authorization_path, output_root),
        pipeline_provider=lambda _args: events.append("pipeline") or pipeline,
        runtime_initializer=initializer,
        execution_gate_factory=gate_factory,
        backend_factory=lambda **_kwargs: pytest.fail(
            "init must not construct a backend"
        ),
    )
    assert code == cli.EXIT_SUCCESS
    assert error is None
    assert events == [
        "pipeline", "gate-factory", "initializer", "gate",
        "runtime-metadata",
    ]
    assert output == {
        "schema_version": 1,
        "document_type": "stage_d_initialization_control",
        "status": "PASS",
        "control": {
            "operation": "init",
            "output_root": str(output_root),
            "external_execution_gate_validated": True,
            "analysis_authorization": "NOT_EVALUATED",
        },
        "digests": {"plan_digest": "2" * 64},
        "counts": {},
    }
    assert not output_root.exists()


@pytest.mark.parametrize(
    "result",
    [
        {
            "document_type": "stage_d_initialization_control",
            "control": {"score": 0.9},
        },
        {
            "document_type": "stage_d_initialization_control",
            "progress": {"value": 0.8},
        },
    ],
)
def test_init_rejects_runtime_documents_instead_of_forwarding_unknown_fields(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, result: dict,
) -> None:
    _install_estimate_pipeline(monkeypatch)
    authorization_path = tmp_path / "authorization.json"
    _write_authorization(authorization_path)
    code, output, error = _invoke(
        _runtime_argv("init", authorization_path, tmp_path / "formal-root"),
        pipeline_provider=lambda _args: _pipeline(),
        runtime_initializer=lambda *_args, **_kwargs: result,
        execution_gate_factory=lambda: (lambda *_args, **_kwargs: object()),
    )
    assert code == cli.EXIT_INTEGRITY_FAILURE
    assert output is None
    assert error["error"]["code"] == "STAGE_D_RUNTIME_INTEGRITY_ERROR"
    serialized = json.dumps(error)
    assert "0.9" not in serialized
    assert "0.8" not in serialized


def test_run_shard_orders_gate_and_runtime_metadata_before_backend(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    estimate, preflight = _install_estimate_pipeline(monkeypatch)
    authorization_path = tmp_path / "authorization.json"
    authorization = _write_authorization(authorization_path)
    output_root = tmp_path / "initialized-root"
    pipeline = _pipeline()
    events: list[str] = []
    backend = object()

    class Gate:
        def validate_execution_context(self, context, **kwargs):
            events.append("authorization")
            assert context == authorization
            assert kwargs["operation"] == "run-shard"
            assert kwargs["output_root"] == output_root
            assert kwargs["plan"] is pipeline.plan
            assert kwargs["estimate"] is estimate
            assert kwargs["preflight"] is preflight
            return SimpleNamespace(canonical_digest="8" * 64)

    def backend_factory(**kwargs):
        events.append("backend")
        assert kwargs["device"] == "fake-device"
        return backend

    def shard_runner(
        plan, *, output_root, estimate, preflight, shard_id,
        execution_context, execution_gate, backend_factory, device,
        retry_authorizations,
    ):
        events.append("runner")
        execution_gate.validate_execution_context(
            execution_context, operation="run-shard",
            output_root=output_root, plan=plan, estimate=estimate,
            preflight=preflight,
        )
        events.append("runtime-metadata")
        assert shard_id == 2
        assert retry_authorizations == ()
        assert backend_factory(device=device) is backend
        events.append("run")
        return _runtime_summary()

    code, output, error = _invoke(
        _runtime_argv("run-shard", authorization_path, output_root),
        pipeline_provider=lambda _args: events.append("pipeline") or pipeline,
        shard_runner=shard_runner, execution_gate_factory=Gate,
        backend_factory=backend_factory,
    )
    assert code == cli.EXIT_SUCCESS
    assert error is None
    assert events == [
        "pipeline", "runner", "authorization", "runtime-metadata",
        "backend", "run",
    ]
    assert output["control"]["shard_index"] == 2
    assert output["counts"]["expected_units"] == 8
    assert output["resources"]["l0_l1_payload_bytes"] == 1024


def test_runtime_shard_summary_is_projected_to_closed_output_categories(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    _install_estimate_pipeline(monkeypatch)
    authorization_path = tmp_path / "authorization.json"
    _write_authorization(authorization_path)
    counts = {
        "expected_groups": 4, "visited_groups": 4,
        "claimed_groups": 3, "terminal_groups": 4,
        "expected_units": 8, "attempted_units": 6,
        "terminal_units": 8, "succeeded_units": 7,
        "scientific_failed_units": 1, "retry_pending_units": 0,
        "blocked_units": 0, "skipped_terminal_units": 2,
    }
    resources = {
        "elapsed_seconds": 2.5, "forward_calls": 12,
        "backward_calls": 6, "l0_l1_payload_bytes": 1024,
        "l2_payload_bytes": 0, "control_and_provenance_bytes": 512,
        "cuda_peak_reserved_bytes": 4096,
    }
    runtime_summary = _runtime_summary(counts=counts, resources=resources)
    code, output, error = _invoke(
        _runtime_argv(
            "run-shard", authorization_path, tmp_path / "formal-root",
        ),
        pipeline_provider=lambda _args: _pipeline(),
        shard_runner=lambda *_args, **_kwargs: runtime_summary,
        execution_gate_factory=lambda: (lambda *_args, **_kwargs: object()),
        backend_factory=lambda **_kwargs: object(),
    )
    assert code == cli.EXIT_SUCCESS
    assert error is None
    assert set(output) == {
        "schema_version", "document_type", "status", "control",
        "terminal", "resources", "digests", "counts",
    }
    assert output["document_type"] == "stage_d_shard_control"
    assert output["status"] == "PASS"
    assert output["control"] == {
        "shard_index": 2, "shard_count": 4,
        "checkpoint_generation": 9,
    }
    assert output["terminal"] == {"state": "completed"}
    assert output["counts"] == counts
    assert output["resources"] == resources
    assert output["digests"] == {
        "checkpoint_digest": "a" * 64,
        "execution_context_digest": "b" * 64,
    }


def test_retry_authorizations_are_strictly_loaded_and_passed_in_cli_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    _install_estimate_pipeline(monkeypatch)
    authorization_path = tmp_path / "authorization.json"
    _write_authorization(authorization_path)
    first_path = tmp_path / "retry-one.json"
    second_path = tmp_path / "retry-two.json"
    first = _write_authorization(
        first_path, {"unit_digest": "1" * 64, "retry": 1},
    )
    second = _write_authorization(
        second_path, {"unit_digest": "2" * 64, "retry": 1},
    )
    observed: list[tuple] = []

    def shard_runner(*_args, retry_authorizations, **_kwargs):
        observed.append(retry_authorizations)
        return _runtime_summary()

    code, output, error = _invoke(
        _runtime_argv(
            "run-shard", authorization_path, tmp_path / "formal-root",
            retry_authorizations=(first_path, second_path),
        ),
        pipeline_provider=lambda _args: _pipeline(),
        shard_runner=shard_runner,
        execution_gate_factory=lambda: (lambda *_args, **_kwargs: object()),
        backend_factory=lambda **_kwargs: object(),
    )
    assert code == cli.EXIT_SUCCESS
    assert error is None
    assert output["document_type"] == "stage_d_shard_control"
    assert observed == [(first, second)]


def test_malformed_retry_authorization_fails_before_gate_runner_and_backend(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    _install_estimate_pipeline(monkeypatch)
    authorization_path = tmp_path / "authorization.json"
    _write_authorization(authorization_path)
    retry_path = tmp_path / "retry.json"
    retry_path.write_text('{"unit":1,"unit":2}', encoding="utf-8")
    events: list[str] = []
    code, output, error = _invoke(
        _runtime_argv(
            "run-shard", authorization_path, tmp_path / "formal-root",
            retry_authorizations=(retry_path,),
        ),
        pipeline_provider=lambda _args: _pipeline(),
        shard_runner=lambda *_args, **_kwargs: events.append("runner"),
        execution_gate_factory=lambda: events.append("gate") or object(),
        backend_factory=lambda **_kwargs: events.append("backend"),
    )
    assert code == cli.EXIT_INPUT_ERROR
    assert output is None
    assert events == []
    assert error["error"]["code"] == "STAGE_D_INPUT_ERROR"
    assert "duplicate JSON key" in error["error"]["message"]


@pytest.mark.parametrize("payload", [[], "retry", 1, None])
def test_retry_authorization_requires_one_strict_json_object_per_option(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, payload,
) -> None:
    _install_estimate_pipeline(monkeypatch)
    authorization_path = tmp_path / "authorization.json"
    _write_authorization(authorization_path)
    retry_path = tmp_path / "retry.json"
    retry_path.write_text(json.dumps(payload), encoding="utf-8")
    events: list[str] = []
    code, output, error = _invoke(
        _runtime_argv(
            "run-shard", authorization_path, tmp_path / "formal-root",
            retry_authorizations=(retry_path,),
        ),
        pipeline_provider=lambda _args: _pipeline(),
        shard_runner=lambda *_args, **_kwargs: events.append("runner"),
        execution_gate_factory=lambda: events.append("gate") or object(),
        backend_factory=lambda **_kwargs: events.append("backend"),
    )
    assert code == cli.EXIT_INPUT_ERROR
    assert output is None
    assert events == []
    assert error["error"]["code"] == "STAGE_D_INPUT_ERROR"
    assert "must be a JSON object" in error["error"]["message"]


def test_retry_authorization_rejects_symlink_and_oversized_file_before_gate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    _install_estimate_pipeline(monkeypatch)
    authorization_path = tmp_path / "authorization.json"
    _write_authorization(authorization_path)
    target = tmp_path / "retry-target.json"
    target.write_text("{}", encoding="utf-8")
    symlink = tmp_path / "retry-link.json"
    symlink.symlink_to(target)
    oversized = tmp_path / "retry-oversized.json"
    oversized.write_text('{"x":1}', encoding="utf-8")
    monkeypatch.setattr(cli, "_RETRY_AUTHORIZATION_FILE_LIMIT", 2)

    for retry_path in (symlink, oversized):
        events: list[str] = []
        code, output, error = _invoke(
            _runtime_argv(
                "run-shard", authorization_path, tmp_path / "formal-root",
                retry_authorizations=(retry_path,),
            ),
            pipeline_provider=lambda _args: _pipeline(),
            shard_runner=lambda *_args, **_kwargs: events.append("runner"),
            execution_gate_factory=lambda: events.append("gate") or object(),
            backend_factory=lambda **_kwargs: events.append("backend"),
        )
        assert code == cli.EXIT_INPUT_ERROR
        assert output is None
        assert events == []
        assert error["error"]["code"] == "STAGE_D_INPUT_ERROR"


def test_authorization_error_prevents_backend_and_hides_runtime_message(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    _install_estimate_pipeline(monkeypatch)
    authorization_path = tmp_path / "authorization.json"
    _write_authorization(authorization_path)
    events: list[str] = []

    class StageDExecutionAuthorizationRequired(Exception):
        pass

    class RefusingGate:
        def validate_execution_context(self, *_args, **_kwargs):
            events.append("authorization")
            raise StageDExecutionAuthorizationRequired(
                "secret prediction=0.91 and delta_fid=0.42"
            )

    def shard_runner(
        plan, *, output_root, estimate, preflight, execution_context,
        execution_gate, backend_factory, **_kwargs,
    ):
        events.append("runner")
        execution_gate.validate_execution_context(
            execution_context, operation="run-shard",
            output_root=output_root, plan=plan, estimate=estimate,
            preflight=preflight,
        )
        backend_factory()  # pragma: no cover - the gate must have raised

    code, output, error = _invoke(
        _runtime_argv(
            "run-shard", authorization_path, tmp_path / "formal-root",
        ),
        pipeline_provider=lambda _args: _pipeline(),
        shard_runner=shard_runner, execution_gate_factory=RefusingGate,
        backend_factory=lambda **_kwargs: events.append("backend"),
    )
    assert code == cli.EXIT_PREFLIGHT_BLOCKED
    assert output is None
    assert events == ["runner", "authorization"]
    assert error["error"] == {
        "code": "STAGE_D_EXECUTION_BLOCKED",
        "message": (
            "Stage D execution is blocked because its D7/runtime gate "
            "did not authorize a safe operation"
        ),
        "type": "StageDCLIRuntimeBlockedError",
    }
    serialized = json.dumps(error).casefold()
    assert "prediction" not in serialized
    assert "delta_fid" not in serialized


def test_default_unavailable_gate_fails_closed_without_running_default_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    _install_estimate_pipeline(monkeypatch)
    authorization_path = tmp_path / "authorization.json"
    _write_authorization(authorization_path)
    output_root = tmp_path / "formal-root"
    events: list[str] = []

    def initializer(
        root, plan, estimate, preflight, *, execution_context, execution_gate,
    ):
        events.append("initializer")
        return execution_gate.validate_execution_context(
            execution_context, operation="init", output_root=root, plan=plan,
            estimate=estimate, preflight=preflight,
        )

    code, output, error = _invoke(
        _runtime_argv("init", authorization_path, output_root),
        pipeline_provider=lambda _args: _pipeline(),
        runtime_initializer=initializer,
        backend_factory=lambda **_kwargs: events.append("backend"),
    )
    assert code == cli.EXIT_PREFLIGHT_BLOCKED
    assert output is None
    assert events == ["initializer"]
    assert error["error"]["code"] == "STAGE_D_EXECUTION_BLOCKED"
    assert "D7/runtime gate" in error["error"]["message"]
    assert not output_root.exists()


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.update({"AUC": 0.8}),
        lambda d: d.update({"progress": {"value": 0.7}}),
        lambda d: d["counts"].update({"value": 5}),
        lambda d: d["resources"].update({"score": 0.6}),
        lambda d: d.update({"control": {"score": 0.9}}),
        lambda d: d.update({"terminal": {"ranking": ["sample"]}}),
        lambda d: d.update({"delta_fid": 0.4}),
        lambda d: d.update({"summary": "hidden"}),
        lambda d: d.update({"prediction": 0.9}),
    ],
)
def test_runtime_stdout_rejects_all_unknown_and_scientific_fields(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mutate,
) -> None:
    _install_estimate_pipeline(monkeypatch)
    authorization_path = tmp_path / "authorization.json"
    _write_authorization(authorization_path)

    def shard_runner(*_args, **_kwargs):
        document = _runtime_summary().to_dict()
        mutate(document)
        return document

    code, output, error = _invoke(
        _runtime_argv(
            "run-shard", authorization_path, tmp_path / "formal-root",
        ),
        pipeline_provider=lambda _args: _pipeline(),
        shard_runner=shard_runner,
        execution_gate_factory=lambda: (lambda *_args, **_kwargs: object()),
        backend_factory=lambda **_kwargs: object(),
    )
    assert code == cli.EXIT_INTEGRITY_FAILURE
    assert output is None
    assert error["error"]["code"] == "STAGE_D_RUNTIME_INTEGRITY_ERROR"
    serialized = json.dumps(error).casefold()
    assert "0.8" not in serialized
    assert "0.9" not in serialized
    assert "0.4" not in serialized
    assert "0.7" not in serialized
    assert "0.6" not in serialized


def test_malformed_authorization_fails_before_gate_runner_and_backend(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    _install_estimate_pipeline(monkeypatch)
    authorization_path = tmp_path / "authorization.json"
    authorization_path.write_text('{"digest":1,"digest":2}', encoding="utf-8")
    events: list[str] = []
    code, output, error = _invoke(
        _runtime_argv(
            "run-shard", authorization_path, tmp_path / "formal-root",
        ),
        pipeline_provider=lambda _args: _pipeline(),
        shard_runner=lambda *_args, **_kwargs: events.append("runner"),
        execution_gate_factory=lambda: events.append("gate") or object(),
        backend_factory=lambda **_kwargs: events.append("backend"),
    )
    assert code == cli.EXIT_INPUT_ERROR
    assert output is None
    assert events == []
    assert error["error"]["code"] == "STAGE_D_INPUT_ERROR"
    assert "duplicate JSON key" in error["error"]["message"]
