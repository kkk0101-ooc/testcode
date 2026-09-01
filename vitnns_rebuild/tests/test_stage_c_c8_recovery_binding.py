from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from vitnns import stage_c_cli
from vitnns.recovery_c8_launch import RecoveryC8LaunchRecord


SOURCE_GIT = {"commit": "a" * 40, "dirty": False}
GENERATOR_GIT = {"commit": "b" * 40, "dirty": False}


def _typed_c8_authority(tmp_path) -> RecoveryC8LaunchRecord:
    config_path = tmp_path / "config.json"
    run_spec_path = tmp_path / "run-spec.json"
    formal_root = tmp_path / "c8"
    reports_root = tmp_path / "reports"
    selection_root = tmp_path / "selection"
    binding = SimpleNamespace(
        model_path=tmp_path / "recovery-model.pth",
        data_root=tmp_path / "images",
        stage_c_input={"registry_path": "manifests/registry.csv"},
    )
    c7_launch = SimpleNamespace(binding=binding)
    document = {
        "canonical_digest": "8" * 64,
        "c8_contracts": {
            "stage_c_config": {"path": str(config_path)},
            "run_spec": {"path": str(run_spec_path)},
        },
        "c7_report": {"path": str(tmp_path / "c7.json")},
    }
    return RecoveryC8LaunchRecord(
        path=tmp_path / "recovery_c8_launch_record.json",
        schema_path=tmp_path / "recovery_c8_launch_record.schema.json",
        document=document, binding=binding, c7_launch_record=c7_launch,
        model_path=binding.model_path, data_root=binding.data_root,
        devkit_path=tmp_path / "devkit.tar.gz", formal_root=formal_root,
        reports_root=reports_root, selection_root=selection_root,
        report_path=reports_root / "stage_c_c8_selection_report.json",
        selection_path=selection_root / "stage_c_selection.json",
        model_identity={"state_dict_sha256": "9" * 64},
        file_sha256="a" * 64, launch_commit=GENERATOR_GIT["commit"],
        repo_root=tmp_path,
    )


def _install_c8_select_controls(
    monkeypatch, tmp_path, *, authorization_code=SOURCE_GIT,
):
    config = object()
    plan = object()
    estimate = SimpleNamespace(to_dict=lambda: {"fresh": True})
    authority = _typed_c8_authority(tmp_path)
    context = {
        "recovery_binding": authority.binding,
        "recovery_launch_record": authority.c7_launch_record,
    }
    authorization = {
        "code": dict(authorization_code),
        "c7_report_path": "/c7.json",
    }
    monkeypatch.setattr(
        stage_c_cli, "_load_pipeline",
        lambda _args, *, c8, recovery_c8_record: (
            config, object(), None, object(), plan, estimate
        ),
    )
    monkeypatch.setattr(
        stage_c_cli, "load_recovery_c8_launch_record",
        lambda *_args, **_kwargs: authority,
    )
    monkeypatch.setattr(
        stage_c_cli, "capture_authorized_c8_source_state",
        lambda record, phase: dict(GENERATOR_GIT),
    )
    monkeypatch.setattr(
        stage_c_cli, "recovery_c8_provenance_context",
        lambda record: context,
    )
    monkeypatch.setattr(
        stage_c_cli, "_require_exact_c8_plan", lambda *_args: {"valid": True},
    )
    monkeypatch.setattr(
        stage_c_cli, "_load_c8_initialized_estimate",
        lambda *_args, **_kwargs: (estimate, authorization),
    )
    monkeypatch.setattr(
        stage_c_cli, "validate_fixed_c7_report", lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        stage_c_cli, "build_c8_measured_preflight",
        lambda *_args, **_kwargs: estimate,
    )
    monkeypatch.setattr(
        stage_c_cli, "_c8_report_binding",
        lambda *_args, **_kwargs: {"fresh_c7": True},
    )
    return SimpleNamespace(
        config=config, plan=plan, estimate=estimate, authorization=authorization,
        authority=authority, context=context,
    )


def _argv(tmp_path, mode):
    authority = _typed_c8_authority(tmp_path)
    values = [
        "c8-select", "--mode", mode,
        "--config", str(tmp_path / "config.json"),
        "--run-spec", str(tmp_path / "run-spec.json"),
        "--output-root", str(authority.formal_root),
        "--report-output", str(authority.report_path),
        "--selection-output", str(authority.selection_path),
        "--recovery-c8-launch-record", str(authority.path),
    ]
    return values


@pytest.mark.parametrize("mode", ["build", "validate"])
def test_c8_select_recovery_keeps_source_binding_with_new_clean_generator(
    monkeypatch, capsys, tmp_path, mode,
):
    controls = _install_c8_select_controls(monkeypatch, tmp_path)
    observed = {}
    evidence = {"calibration_artifact_manifest_sha256": "c" * 64}
    input_hashes = {"code_commit": SOURCE_GIT["commit"]}
    def build_evidence(
        root, plan, config, source_git_state, c7_binding, **kwargs
    ):
        observed["evidence"] = (
            root, plan, config, dict(source_git_state), c7_binding,
        )
        observed["evidence_trust"] = kwargs
        return evidence

    def build_hashes(**kwargs):
        observed["hashes_source"] = dict(kwargs["source_git_state"])
        return input_hashes

    monkeypatch.setattr(stage_c_cli, "build_c8_calibration_evidence", build_evidence)
    monkeypatch.setattr(stage_c_cli, "_c8_input_hashes", build_hashes)

    if mode == "build":
        def build_selection(
            received_evidence, received_hashes, *, generator_git_state,
            **kwargs,
        ):
            observed["build"] = (
                received_evidence, received_hashes, dict(generator_git_state),
            )
            observed["build_trust"] = kwargs
            return {"status": "PASS"}

        def write_outputs(
            _selection, _report, _built, *, evidence, input_hashes,
            generator_git_state, **kwargs,
        ):
            observed["write"] = (
                evidence, input_hashes, dict(generator_git_state),
            )
            observed["write_trust"] = kwargs
            return {
                "status": "PASS", "report_sha256": "d" * 64,
                "selection_sha256": "e" * 64,
            }

        monkeypatch.setattr(stage_c_cli, "build_stage_c_selection", build_selection)
        monkeypatch.setattr(
            stage_c_cli, "write_stage_c_selection_outputs", write_outputs,
        )
    else:
        def validate_source(
            _selection, _report, *, output_root, plan, config,
            observed_git_state, input_hashes, c7_report_binding,
            generator_git_state, **kwargs,
        ):
            observed["validate"] = {
                "output_root": output_root, "plan": plan, "config": config,
                "source": dict(observed_git_state),
                "input_hashes": input_hashes,
                "c7": c7_report_binding,
                "generator": dict(generator_git_state),
            }
            observed["validate_trust"] = kwargs
            return {"status": "PASS", "holdout_accessed": False}

        monkeypatch.setattr(
            stage_c_cli, "validate_stage_c_selection_from_source", validate_source,
        )

    assert stage_c_cli.main(_argv(tmp_path, mode)) == stage_c_cli.EXIT_SUCCESS
    assert json.loads(capsys.readouterr().out)["status"] == "PASS"
    assert observed["evidence"][1:4] == (
        controls.plan, controls.config, SOURCE_GIT,
    )
    assert observed["hashes_source"] == SOURCE_GIT
    assert observed["evidence_trust"] == {
        "trusted_recovery_authority": controls.authority,
        "trusted_recovery_context": controls.context,
    }
    if mode == "build":
        assert observed["build"] == (evidence, input_hashes, GENERATOR_GIT)
        assert observed["write"] == (evidence, input_hashes, GENERATOR_GIT)
        assert observed["build_trust"] == {
            "trusted_recovery_authority": controls.authority,
            "trusted_recovery_context": controls.context,
        }
        assert observed["write_trust"] == observed["build_trust"]
    else:
        assert observed["validate"]["source"] == SOURCE_GIT
        assert observed["validate"]["generator"] == GENERATOR_GIT
        assert observed["validate_trust"] == {
            "trusted_recovery_authority": controls.authority,
            "trusted_recovery_context": controls.context,
        }


def test_c8_select_recovery_rejects_dirty_generator_before_evidence(
    monkeypatch, capsys, tmp_path,
):
    _install_c8_select_controls(monkeypatch, tmp_path)
    monkeypatch.setattr(
        stage_c_cli, "capture_authorized_c8_source_state",
        lambda *_args: (_ for _ in ()).throw(
            stage_c_cli.RecoveryC8LaunchError("dirty generator")
        ),
    )
    monkeypatch.setattr(
        stage_c_cli, "build_c8_calibration_evidence",
        lambda *_args, **_kwargs: pytest.fail("dirty generator read evidence"),
    )

    assert stage_c_cli.main(_argv(tmp_path, "build")) == (
        stage_c_cli.EXIT_INTEGRITY_FAILURE
    )
    payload = json.loads(capsys.readouterr().err)
    assert "dirty generator" in payload["error"]["message"]


def test_c8_select_recovery_rejects_source_authorization_drift(
    monkeypatch, capsys, tmp_path,
):
    drifted_source = {"commit": "f" * 40, "dirty": False}
    _install_c8_select_controls(
        monkeypatch, tmp_path, authorization_code=drifted_source,
    )
    def reject_drift(
        _root, _plan, _config, source_git_state, _c7_binding, **_kwargs
    ):
        assert source_git_state == drifted_source
        raise stage_c_cli.StageCBenchmarkIntegrityError(
            "source authorization code differs from execution artifacts"
        )

    monkeypatch.setattr(
        stage_c_cli, "build_c8_calibration_evidence", reject_drift,
    )

    assert stage_c_cli.main(_argv(tmp_path, "build")) == (
        stage_c_cli.EXIT_INTEGRITY_FAILURE
    )
    payload = json.loads(capsys.readouterr().err)
    assert "source authorization code differs" in payload["error"]["message"]


def test_c8_input_hashes_bind_source_commit_not_generator(monkeypatch, tmp_path):
    config = SimpleNamespace(
        protocol=SimpleNamespace(path="docs/protocol.md"),
        registry=SimpleNamespace(sha256="1" * 64, metadata_sha256="2" * 64),
    )
    monkeypatch.setattr(stage_c_cli, "sha256_file", lambda _path: "3" * 64)

    hashes = stage_c_cli._c8_input_hashes(
        config=config, config_path=tmp_path / "config.json",
        evidence={"calibration_artifact_manifest_sha256": "4" * 64},
        source_git_state=SOURCE_GIT,
    )

    assert hashes["code_commit"] == SOURCE_GIT["commit"]
    assert hashes["code_commit"] != GENERATOR_GIT["commit"]
    assert hashes["code_tree_clean"] is True
