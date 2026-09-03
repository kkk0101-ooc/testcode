"""Recovery-v2 D0-D6 binding and terminal-receipt tests."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil

import pytest

from vitnns.stage_d_recovery import (
    BINDING_PATH, PASS_SCOPES, STAGES, StageDRecoveryError,
    canonical_file_bytes, create_stage_bundle, load_recovery_binding,
    semantic_digest, strict_json_bytes, validate_stage_bundle,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PRODUCER = "f1dc44d3994cd6dd6942659a6623aaf6d90dbe4b"
SOURCE_TREE = "b963edb24f3a4a007b6cae66775bf916039c2328"


def _copy_sources(destination: Path) -> None:
    for relative in (
        "config", "docs", "manifests", "reports", "src",
        "requirements.lock.txt",
    ):
        source = REPO_ROOT / relative
        target = destination / relative
        if source.is_dir():
            shutil.copytree(source, target)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)


def test_recovery_binding_is_closed_semantic_overlay() -> None:
    document, file_sha256, digest = load_recovery_binding(REPO_ROOT)
    assert file_sha256 == "03165a4d3b733fed5ffa01ec07a643c445b8d3727b13b9408aa009b3e37bc2fb"
    assert digest == "51a6cec2a62162cc4ba49b545ce5b689118440d1dc1ca6a0fcd5cacb4fe5b8e3"
    assert document["recovery_chain"]["c10"]["report_canonical_digest"] == "efc4e880793166fdf2e7869396cc0d0df9699e27c53071aa83c83ddce472e849"
    assert document["model"]["file_sha256"] == "489bfa2f1aa3b6781e268a87eb629816b89eebba439420978a794b8e8151565d"
    assert document["historical_supersession"]["historical_bindings_are_current"] is False
    assert set(document["authorization"].values()) == {False}
    assert set(document["reserve_access_boundary"].values()) == {False}
    schema = json.loads(
        (REPO_ROOT / "config/stage_d_recovery_v2_binding.schema.json").read_text()
    )
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"])


def test_binding_rejects_self_resigned_legacy_or_authority_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _copy_sources(tmp_path)
    path = tmp_path / BINDING_PATH
    document = json.loads(path.read_text())
    document["model"]["file_sha256"] = "2e718037429f18ec77aa629471beb2e99993d7383375188bab777aba04d5e4d0"
    document["authorization"]["execution_authorization"] = True
    document["canonical_digest"] = semantic_digest(document)
    path.write_bytes(canonical_file_bytes(document))
    with pytest.raises(StageDRecoveryError):
        load_recovery_binding(tmp_path)


def test_strict_json_rejects_duplicate_and_nonfinite() -> None:
    with pytest.raises(StageDRecoveryError, match="duplicate"):
        strict_json_bytes(b'{"x":1,"x":2}', "duplicate")
    with pytest.raises(StageDRecoveryError, match="non-finite"):
        strict_json_bytes(b'{"x":NaN}', "nonfinite")


def test_all_stage_scopes_are_preparation_or_readiness_only() -> None:
    assert tuple(PASS_SCOPES) == STAGES
    assert PASS_SCOPES["D0"] == "FORMAL_PREPARATION_PASS"
    assert PASS_SCOPES["D5"] == "VALIDATION_IMPLEMENTATION_READINESS_PASS"
    assert PASS_SCOPES["D6"] == "ANALYSIS_IMPLEMENTATION_READINESS_PASS"


def test_stage_bundle_is_pending_before_atomic_terminal_and_rejects_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _copy_sources(tmp_path)
    monkeypatch.setattr(
        "vitnns.stage_d_recovery._run_git",
        lambda _root, *args: (
            PRODUCER if args == ("rev-parse", f"{PRODUCER}^{{commit}}")
            else SOURCE_TREE if args == ("rev-parse", f"{PRODUCER}:src/vitnns")
            else "" if args == ("rev-parse", "--show-prefix")
            else (_ for _ in ()).throw(AssertionError(args))
        ),
    )
    result = create_stage_bundle(
        "D0", producer_commit=PRODUCER, producer_source_tree=SOURCE_TREE,
        validation_command="synthetic focused command", repo_root=tmp_path,
    )
    assert result["status"] == "PASS"
    root = tmp_path / "reports/stage_d_recovery_v2_20260904/d0"
    assert not (root / "status.pending.json").exists()
    assert validate_stage_bundle("D0", repo_root=tmp_path, verify_git_ref=False)["status"] == "PASS"
    status = json.loads((root / "status.json").read_text())
    status["blocking_reasons"] = ["hidden failure"]
    status["canonical_digest"] = semantic_digest(status)
    (root / "status.json").write_bytes(canonical_file_bytes(status))
    with pytest.raises(StageDRecoveryError, match="terminal status"):
        validate_stage_bundle("D0", repo_root=tmp_path, verify_git_ref=False)


def test_bundle_rejects_missing_duplicate_and_wrong_direct_upstream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _copy_sources(tmp_path)
    monkeypatch.setattr(
        "vitnns.stage_d_recovery._run_git",
        lambda _root, *args: (
            PRODUCER if args == ("rev-parse", f"{PRODUCER}^{{commit}}")
            else SOURCE_TREE if args == ("rev-parse", f"{PRODUCER}:src/vitnns")
            else "" if args == ("rev-parse", "--show-prefix")
            else (_ for _ in ()).throw(AssertionError(args))
        ),
    )
    create_stage_bundle(
        "D0", producer_commit=PRODUCER, producer_source_tree=SOURCE_TREE,
        validation_command="focused", repo_root=tmp_path,
    )
    completion_path = tmp_path / "reports/stage_d_recovery_v2_20260904/d0/completion.json"
    completion = json.loads(completion_path.read_text())
    completion["source_files"].append(copy.deepcopy(completion["source_files"][0]))
    completion["canonical_digest"] = semantic_digest(completion)
    completion_path.write_bytes(canonical_file_bytes(completion))
    with pytest.raises(StageDRecoveryError, match="duplicate source"):
        validate_stage_bundle("D0", repo_root=tmp_path, verify_git_ref=False)


def test_production_tree_has_no_d7_authority_or_reserve_path_operation() -> None:
    source = (REPO_ROOT / "src/vitnns/stage_d_recovery.py").read_text()
    assert "AUTHORIZED_STAGE_D_FIDELITY_SCOPE" not in source
    assert "reserve_image_exists_checked" in source
    assert "reserve_image_statted" in source
    assert "stage_d_outcome_accessed" in source
