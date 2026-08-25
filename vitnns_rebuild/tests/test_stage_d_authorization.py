"""Focused metadata-only tests for Stage D preparation authorization."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil

import pytest

import vitnns.stage_d_authorization as authorization_module
import vitnns.stage_d_manifest as manifest_module
from vitnns.stage_d_authorization import (
    MODEL_CONFIG_BYTES,
    REQUIREMENTS_LOCK_BYTES,
    StageDAuthorizationError,
    StageDPreparationAuthorization,
    StageDPreparationInputs,
    build_stage_d_preparation_authorization,
    canonical_sha256,
    capture_stage_d_preparation_inputs,
    preparation_authorization_digest,
    validate_stage_d_preparation_authorization,
)
from vitnns.stage_d_config import load_stage_d_contracts
from vitnns.stage_d_manifest import verify_stage_d_manifest
from vitnns.stage_d_plan import (
    build_stage_d_plan,
    estimate_stage_d_run,
    preflight_stage_d,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
HEAD = "a" * 40
TREE = "b" * 40


def _load(relative: str) -> dict:
    return json.loads((REPO_ROOT / relative).read_text(encoding="utf-8"))


def _fake_c10_result(binding: dict) -> dict:
    file_fields = {
        "status.json": "status_file_sha256",
        "stage_c_report.json": "report_file_sha256",
        "decision.json": "decision_file_sha256",
        "holm.json": "holm_file_sha256",
        "provenance.json": "provenance_file_sha256",
        "manifest.json": "manifest_file_sha256",
    }
    canonical_fields = {
        "status.json": "status_canonical_digest",
        "stage_c_report.json": "report_canonical_digest",
        "decision.json": "decision_canonical_digest",
        "holm.json": "holm_canonical_digest",
        "provenance.json": "provenance_canonical_digest",
        "manifest.json": "manifest_canonical_digest",
    }
    return {
        "status": "PASS", "root": binding["formal_root"],
        "file_count": binding["whole_root_file_count"],
        "total_bytes": binding["whole_root_total_bytes"],
        "whole_root_inventory_digest": binding["whole_root_inventory_digest"],
        "file_sha256": {name: binding[field] for name, field in file_fields.items()},
        "canonical_digests": {
            name: binding[field] for name, field in canonical_fields.items()
        },
        "archive_ref": binding["producer_ref"],
        "producer_commit": binding["producer_commit"],
        "producer_tree": binding["producer_tree"],
        "source_tree": binding["producer_source_tree"],
        "scientific_results_recomputed": False,
    }


def _git_runner(_root: Path, arguments: tuple[str, ...]) -> str:
    if arguments == ("rev-parse", "--show-toplevel"):
        return str(REPO_ROOT.parent)
    if arguments == ("rev-parse", "--show-prefix"):
        return REPO_ROOT.name + "/"
    if arguments == ("rev-parse", "HEAD"):
        return HEAD
    if arguments == ("rev-parse", "HEAD^{tree}"):
        return TREE
    if arguments == ("status", "--porcelain=v1", "--untracked-files=all"):
        return ""
    raise AssertionError(arguments)


def _resign(document: dict) -> dict:
    document["canonical_digest"] = canonical_sha256({
        key: value for key, value in document.items()
        if key != "canonical_digest"
    })
    return document


@pytest.fixture(scope="module")
def issued_inputs() -> dict:
    contracts = load_stage_d_contracts()
    binding = contracts.config["upstream"]["c10"]
    original = manifest_module.verify_c10_sealed_evidence
    manifest_module.verify_c10_sealed_evidence = (
        lambda root, *, repo_root: _fake_c10_result(dict(binding))
    )
    try:
        manifest = verify_stage_d_manifest(contracts.config, repo_root=REPO_ROOT)
    finally:
        manifest_module.verify_c10_sealed_evidence = original
    plan = build_stage_d_plan(contracts.config, contracts.run_spec, manifest)
    budget = estimate_stage_d_run(contracts.config, plan)
    preflight = preflight_stage_d(budget)
    original_git = authorization_module._run_git
    authorization_module._run_git = _git_runner
    try:
        preparation_inputs = capture_stage_d_preparation_inputs(
            repo_root=REPO_ROOT,
        )
    finally:
        authorization_module._run_git = original_git
    return {
        "contracts": contracts, "manifest": manifest, "plan": plan,
        "budget": budget, "preflight": preflight,
        "preparation_inputs": preparation_inputs,
    }


def test_capture_issues_immutable_clean_static_metadata_proof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(authorization_module, "_run_git", _git_runner)
    captured = capture_stage_d_preparation_inputs(repo_root=REPO_ROOT)
    assert isinstance(captured, StageDPreparationInputs)
    assert captured.code == {"commit": HEAD, "tree": TREE, "dirty": False}
    assert captured.dependencies["requirements_lock_bytes"] == REQUIREMENTS_LOCK_BYTES
    assert captured.model["model_config_bytes"] == MODEL_CONFIG_BYTES
    assert captured.model["state_dict_file_accessed"] is False
    assert captured.canonical_digest == canonical_sha256(captured._payload())
    with pytest.raises(TypeError):
        captured.code["dirty"] = True
    with pytest.raises(TypeError, match="can only be issued"):
        StageDPreparationInputs()
    with pytest.raises(StageDAuthorizationError, match="issuer"):
        StageDPreparationInputs._issue(
            issuer=object(), code=captured.code,
            dependencies=captured.dependencies, model=captured.model,
        )

def test_capture_refuses_dirty_git_before_reading_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def dirty(_root: Path, arguments: tuple[str, ...]) -> str:
        if arguments == ("rev-parse", "--show-toplevel"):
            return str(REPO_ROOT.parent)
        if arguments == ("rev-parse", "--show-prefix"):
            return REPO_ROOT.name + "/"
        if arguments[0] == "status":
            return " M src/vitnns/stage_d_authorization.py"
        return HEAD if arguments[-1] == "HEAD" else TREE

    monkeypatch.setattr(authorization_module, "_run_git", dirty)
    monkeypatch.setattr(
        authorization_module, "_safe_read_static_metadata",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("dirty code must fail before metadata reads")
        ),
    )
    with pytest.raises(StageDAuthorizationError, match="dirty producer"):
        capture_stage_d_preparation_inputs(
            repo_root=REPO_ROOT,
        )


def test_capture_reads_only_model_config_and_dependency_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    original = authorization_module._safe_read_static_metadata

    def record(repo_root, relative_path, *, expected_sha256):
        calls.append(relative_path)
        return original(
            repo_root, relative_path, expected_sha256=expected_sha256,
        )

    monkeypatch.setattr(authorization_module, "_run_git", _git_runner)
    monkeypatch.setattr(authorization_module, "_safe_read_static_metadata", record)
    captured = capture_stage_d_preparation_inputs(repo_root=REPO_ROOT)
    assert calls == ["config/model.json", "requirements.lock.txt"]
    assert captured.model["state_dict_file_accessed"] is False


def test_capture_rejects_model_semantic_drift_without_state_dict_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    original = authorization_module._safe_read_static_metadata

    def changed(repo_root, relative_path, *, expected_sha256):
        calls.append(relative_path)
        payload, digest = original(
            repo_root, relative_path, expected_sha256=expected_sha256,
        )
        if relative_path == "config/model.json":
            old = b"vit_base_patch16_224.augreg2_in21k_ft_in1k"
            payload = payload.replace(old, b"x" * len(old), 1)
        return payload, digest

    monkeypatch.setattr(authorization_module, "_run_git", _git_runner)
    monkeypatch.setattr(authorization_module, "_safe_read_static_metadata", changed)
    with pytest.raises(StageDAuthorizationError, match="semantic identity drifted"):
        capture_stage_d_preparation_inputs(repo_root=REPO_ROOT)
    assert calls == ["config/model.json", "requirements.lock.txt"]


def test_capture_rejects_repository_root_or_git_snapshot_drift(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        authorization_module, "_run_git",
        lambda _root, arguments: (
            str(tmp_path)
            if arguments == ("rev-parse", "--show-toplevel")
            else "" if arguments == ("rev-parse", "--show-prefix")
            else HEAD
        ),
    )
    with pytest.raises(StageDAuthorizationError, match="top-level/prefix"):
        capture_stage_d_preparation_inputs(repo_root=REPO_ROOT)

    snapshots = iter((HEAD, HEAD, "c" * 40, "c" * 40))

    def changes(_root: Path, arguments: tuple[str, ...]) -> str:
        if arguments == ("rev-parse", "--show-toplevel"):
            return str(REPO_ROOT.parent)
        if arguments == ("rev-parse", "--show-prefix"):
            return REPO_ROOT.name + "/"
        if arguments == ("status", "--porcelain=v1", "--untracked-files=all"):
            return ""
        return next(snapshots)

    monkeypatch.setattr(authorization_module, "_run_git", changes)
    with pytest.raises(StageDAuthorizationError, match="Git state changed"):
        capture_stage_d_preparation_inputs(repo_root=REPO_ROOT)


@pytest.mark.parametrize(
    "relative_path, expected_hash, expected",
    [
        ("config/model.json", "0" * 64, "SHA-256 drifted"),
        ("requirements.lock.txt", "0" * 64, "SHA-256 drifted"),
    ],
)
def test_safe_static_metadata_reader_rejects_hash_drift(
    relative_path: str, expected_hash: str, expected: str,
) -> None:
    with pytest.raises(StageDAuthorizationError, match=expected):
        authorization_module._safe_read_static_metadata(
            REPO_ROOT, relative_path, expected_sha256=expected_hash,
        )


def test_safe_static_metadata_reader_rejects_symlink_and_hardlink(
    tmp_path: Path,
) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    outside = tmp_path / "model.source.json"
    shutil.copyfile(REPO_ROOT / "config/model.json", outside)
    (config_dir / "model.json").symlink_to(outside)
    with pytest.raises(StageDAuthorizationError, match="safely open"):
        authorization_module._safe_read_static_metadata(
            tmp_path, "config/model.json",
            expected_sha256=authorization_module.MODEL_CONFIG_SHA256,
        )
    (config_dir / "model.json").unlink()
    os.link(outside, config_dir / "model.json")
    with pytest.raises(StageDAuthorizationError, match="single-link regular"):
        authorization_module._safe_read_static_metadata(
            tmp_path, "config/model.json",
            expected_sha256=authorization_module.MODEL_CONFIG_SHA256,
        )


def test_builder_requires_all_three_issued_trust_inputs(issued_inputs: dict) -> None:
    artifact = build_stage_d_preparation_authorization(**issued_inputs)
    for name in ("contracts", "manifest", "preparation_inputs"):
        changed = dict(issued_inputs)
        changed[name] = issued_inputs[name].to_dict()
        with pytest.raises(StageDAuthorizationError, match="must be issued"):
            build_stage_d_preparation_authorization(**changed)
    assert artifact.preparation_inputs_digest == (
        issued_inputs["preparation_inputs"].canonical_digest
    )
    restored = StageDPreparationAuthorization.from_dict(artifact.to_dict())
    assert validate_stage_d_preparation_authorization(
        restored, **issued_inputs,
    ) is restored


def test_plan_budget_and_preflight_accept_validated_serialized_forms(
    issued_inputs: dict,
) -> None:
    changed = dict(issued_inputs)
    for name in ("plan", "budget", "preflight"):
        changed[name] = issued_inputs[name].to_dict()
    artifact = build_stage_d_preparation_authorization(**changed)
    assert artifact.plan_binding["canonical_digest"] == (
        issued_inputs["plan"].canonical_digest
    )


def test_builds_immutable_canonical_preparation_only_artifact(
    issued_inputs: dict,
) -> None:
    artifact = build_stage_d_preparation_authorization(**issued_inputs)
    assert isinstance(artifact, StageDPreparationAuthorization)
    assert artifact.preparation_authorization == "APPROVED_TO_PREPARE_STAGE_D"
    assert artifact.execution_authorization == "NOT_ISSUED"
    assert artifact.analysis_authorization == "NOT_EVALUATED"
    assert artifact.stage_d_execution_started is False
    assert artifact.scientific_result == "NOT_EVALUATED"
    assert artifact.is_execution_authorized is False
    assert artifact.c10_historical_status == {
        "primary_decision": "NO_GO_NOVELTY",
        "spatial_status": "FAIL_SPATIAL",
        "stage_c_acceptance": "BLOCKED",
        "workflow_blocker": "HOLM_BLOCKED_UNDERSPECIFIED",
        "downstream_authorized": False, "stage_d_started": False,
    }
    assert artifact.budget_binding["profile_status"] == "D7_PROFILE_REQUIRED"
    assert artifact.budget_binding["budget_status"] == "NOT_EVALUATED"
    assert artifact.budget_binding["preflight_status"] == "PROFILE_REQUIRED"
    assert artifact.canonical_digest == preparation_authorization_digest(artifact)
    assert "AUTHORIZED_STAGE_D_FIDELITY_SCOPE" not in json.dumps(
        artifact.to_dict(), sort_keys=True,
    )
    with pytest.raises(TypeError):
        artifact.source_bindings["d0"]["completion_commit"] = "0" * 40


@pytest.mark.parametrize(
    "mutate, expected",
    [
        (lambda d: d.update({"execution_authorization": "ISSUED"}), "authority/result"),
        (lambda d: d.update({"analysis_authorization": "AUTHORIZED"}), "authority/result"),
        (lambda d: d.update({"stage_d_execution_started": True}), "authority/result"),
        (lambda d: d.update({"scientific_result": "PASS_FIDELITY"}), "authority/result"),
        (lambda d: d["c10_historical_status"].update({"primary_decision": "GO"}), "C10 historical"),
        (lambda d: d["budget_binding"].update({"d7_final": True}), "budget/preflight"),
    ],
)
def test_intrinsic_validation_rejects_resigned_authority_and_c10_tamper(
    issued_inputs: dict, mutate, expected: str,
) -> None:
    document = build_stage_d_preparation_authorization(**issued_inputs).to_dict()
    mutate(document)
    _resign(document)
    with pytest.raises(StageDAuthorizationError, match=expected):
        StageDPreparationAuthorization.from_dict(document)


def test_validator_rejects_self_resigned_artifact_source_drift(
    issued_inputs: dict,
) -> None:
    artifact = build_stage_d_preparation_authorization(**issued_inputs)
    changed = artifact.to_dict()
    changed["code"]["commit"] = "c" * 40
    changed["preparation_inputs_digest"] = canonical_sha256({
        "schema_version": 1, "document_type": "stage_d_preparation_inputs",
        "status": "PASS", "code": changed["code"],
        "dependencies": changed["dependencies"], "model": changed["model"],
    })
    _resign(changed)
    with pytest.raises(StageDAuthorizationError, match="source binding drifted"):
        validate_stage_d_preparation_authorization(changed, **issued_inputs)


def test_plain_fabricated_pass_inputs_are_rejected_before_plan_work() -> None:
    fake_contracts = {"config": {}, "run_spec": {}}
    with pytest.raises(StageDAuthorizationError, match="contracts must be issued"):
        build_stage_d_preparation_authorization(
            contracts=fake_contracts, manifest={}, plan={}, budget={},
            preparation_inputs={},
        )
