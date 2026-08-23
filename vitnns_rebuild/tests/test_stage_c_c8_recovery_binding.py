from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from vitnns import stage_c_cli


SOURCE_GIT = {"commit": "a" * 40, "dirty": False}
GENERATOR_GIT = {"commit": "b" * 40, "dirty": False}


def _install_c8_select_controls(monkeypatch, *, authorization_code=SOURCE_GIT):
    config = object()
    plan = object()
    estimate = SimpleNamespace(to_dict=lambda: {"fresh": True})
    authorization = {
        "code": dict(authorization_code),
        "c7_report_path": "/c7.json",
    }
    monkeypatch.setattr(
        stage_c_cli, "_load_pipeline",
        lambda _args, *, c8: (config, object(), None, object(), plan, estimate),
    )
    monkeypatch.setattr(
        stage_c_cli, "_require_exact_c8_plan", lambda *_args: {"valid": True},
    )
    monkeypatch.setattr(
        stage_c_cli, "_load_c8_initialized_estimate",
        lambda *_args: (estimate, authorization),
    )
    monkeypatch.setattr(
        stage_c_cli, "validate_fixed_c7_report", lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        stage_c_cli, "build_c8_measured_preflight",
        lambda *_args: estimate,
    )
    monkeypatch.setattr(
        stage_c_cli, "_c8_report_binding",
        lambda *_args, **_kwargs: {"fresh_c7": True},
    )
    return SimpleNamespace(
        config=config, plan=plan, estimate=estimate, authorization=authorization
    )


def _argv(tmp_path, mode):
    values = [
        "c8-select", "--mode", mode,
        "--config", str(tmp_path / "config.json"),
        "--run-spec", str(tmp_path / "run-spec.json"),
        "--output-root", str(tmp_path / "c8"),
        "--report-output", str(tmp_path / "report.json"),
    ]
    if mode == "build":
        values.extend(("--selection-output", str(tmp_path / "selection.json")))
    return values


@pytest.mark.parametrize("mode", ["build", "validate"])
def test_c8_select_recovery_keeps_source_binding_with_new_clean_generator(
    monkeypatch, capsys, tmp_path, mode,
):
    controls = _install_c8_select_controls(monkeypatch)
    observed = {}
    evidence = {"calibration_artifact_manifest_sha256": "c" * 64}
    input_hashes = {"code_commit": SOURCE_GIT["commit"]}
    monkeypatch.setattr(
        stage_c_cli, "capture_git_state", lambda _root: dict(GENERATOR_GIT),
    )

    def build_evidence(root, plan, config, source_git_state, c7_binding):
        observed["evidence"] = (
            root, plan, config, dict(source_git_state), c7_binding,
        )
        return evidence

    def build_hashes(**kwargs):
        observed["hashes_source"] = dict(kwargs["source_git_state"])
        return input_hashes

    monkeypatch.setattr(stage_c_cli, "build_c8_calibration_evidence", build_evidence)
    monkeypatch.setattr(stage_c_cli, "_c8_input_hashes", build_hashes)

    if mode == "build":
        def build_selection(received_evidence, received_hashes, *, generator_git_state):
            observed["build"] = (
                received_evidence, received_hashes, dict(generator_git_state),
            )
            return {"status": "PASS"}

        def write_outputs(
            _selection, _report, _built, *, evidence, input_hashes,
            generator_git_state,
        ):
            observed["write"] = (
                evidence, input_hashes, dict(generator_git_state),
            )
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
            generator_git_state,
        ):
            observed["validate"] = {
                "output_root": output_root, "plan": plan, "config": config,
                "source": dict(observed_git_state),
                "input_hashes": input_hashes,
                "c7": c7_report_binding,
                "generator": dict(generator_git_state),
            }
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
    if mode == "build":
        assert observed["build"] == (evidence, input_hashes, GENERATOR_GIT)
        assert observed["write"] == (evidence, input_hashes, GENERATOR_GIT)
    else:
        assert observed["validate"]["source"] == SOURCE_GIT
        assert observed["validate"]["generator"] == GENERATOR_GIT


def test_c8_select_recovery_rejects_dirty_generator_before_evidence(
    monkeypatch, capsys, tmp_path,
):
    _install_c8_select_controls(monkeypatch)
    monkeypatch.setattr(
        stage_c_cli, "capture_git_state",
        lambda _root: {"commit": GENERATOR_GIT["commit"], "dirty": True},
    )
    monkeypatch.setattr(
        stage_c_cli, "build_c8_calibration_evidence",
        lambda *_args, **_kwargs: pytest.fail("dirty generator read evidence"),
    )

    assert stage_c_cli.main(_argv(tmp_path, "build")) == (
        stage_c_cli.EXIT_PREFLIGHT_BLOCKED
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "BLOCKED"
    assert "clean Git worktree" in payload["blocked_reasons"][0]


def test_c8_select_recovery_rejects_source_authorization_drift(
    monkeypatch, capsys, tmp_path,
):
    drifted_source = {"commit": "f" * 40, "dirty": False}
    _install_c8_select_controls(
        monkeypatch, authorization_code=drifted_source,
    )
    monkeypatch.setattr(
        stage_c_cli, "capture_git_state", lambda _root: dict(GENERATOR_GIT),
    )

    def reject_drift(_root, _plan, _config, source_git_state, _c7_binding):
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
